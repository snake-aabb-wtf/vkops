# -*- coding: utf-8 -*-
"""server.py — OpenAI-compatible chat completions API over the vkops Vulkan engine.

Endpoints:
  GET  /v1/models              list the model (Bearer auth)
  POST /v1/chat/completions    OpenAI chat completions (stream + non-stream)
  GET  /health                 liveness probe (no auth)

Auth: Authorization: Bearer <token>  (token printed at startup / api_token.txt)
Bind : 127.0.0.1:8000  (expose via tailscale funnel / tunnel)
"""
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import chat
import ops

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = r"C:\vkops\qwen3-4b"
MODEL_ID = "qwen3-4b-vkops"
NUM_LAYERS, HIDDEN, INTER = 36, 2560, 9728
MAX_CTX = 1024
HOST, PORT = "127.0.0.1", 8000

TOKEN = os.environ.get("VKOPS_API_TOKEN", "") or os.urandom(8).hex()
GPU_LOCK = threading.Lock()
GENERATED_AT = int(time.time())


def build_prompt_ids(messages):
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = "".join(p.get("text", "") for p in content
                              if isinstance(p, dict) and p.get("type") == "text")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return tok.encode("".join(parts)).ids


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet default access log
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self):
        return self.headers.get("Authorization", "") == f"Bearer {TOKEN}"

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": MODEL_ID})
        elif self.path == "/v1/models" and self._authed():
            self._send_json(200, {"object": "list", "data": [{
                "id": MODEL_ID, "object": "model", "created": GENERATED_AT,
                "owned_by": "vkops"}]})
        elif self.path.startswith("/v1/"):
            self._send_json(401, {"error": {"message": "invalid api key",
                                            "type": "auth_error"}})
        else:
            self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found"}})
            return
        if not self._authed():
            self._send_json(401, {"error": {"message": "invalid api key",
                                            "type": "auth_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send_json(400, {"error": {"message": "bad json"}})
            return

        messages = body.get("messages") or []
        if not messages:
            self._send_json(400, {"error": {"message": "messages required"}})
            return
        stream = bool(body.get("stream", False))
        temperature = float(body.get("temperature", 0.7))
        top_p = float(body.get("top_p", 0.8))
        max_tokens = int(body.get("max_tokens", 256))

        prompt_ids = build_prompt_ids(messages)
        budget = MAX_CTX - len(prompt_ids) - 4
        if budget <= 0:
            self._send_json(400, {"error": {
                "message": f"prompt too long ({len(prompt_ids)} tokens, ctx {MAX_CTX})"}})
            return
        max_tokens = max(1, min(max_tokens, budget))

        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        with GPU_LOCK:  # one engine, serialize generation
            if stream:
                self._stream_response(cid, created, prompt_ids,
                                      temperature, top_p, max_tokens)
            else:
                self._complete_response(cid, created, prompt_ids,
                                        temperature, top_p, max_tokens)

    def _complete_response(self, cid, created, prompt_ids, temperature, top_p, max_tokens):
        t0 = time.perf_counter()
        rng = np.random.default_rng()
        gen = eng.generate(prompt_ids, max_new=max_tokens, temperature=temperature,
                           top_p=top_p, rng=rng, eos=chat.IM_END)
        text = tok.decode(gen)
        dt = time.perf_counter() - t0
        print(f"[chat] prompt={len(prompt_ids)} completion={len(gen)} "
              f"{len(gen)/dt:.1f} tok/s", flush=True)
        self._send_json(200, {
            "id": cid, "object": "chat.completion", "created": created,
            "model": MODEL_ID, "system_fingerprint": "vkops-vulkan",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "logprobs": None,
                         "finish_reason": "stop" if len(gen) < max_tokens else "length"}],
            "usage": {"prompt_tokens": len(prompt_ids),
                      "completion_tokens": len(gen),
                      "total_tokens": len(prompt_ids) + len(gen)}})

    def _stream_response(self, cid, created, prompt_ids, temperature, top_p, max_tokens):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(obj):
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        def chunk(delta, finish=None):
            emit({"id": cid, "object": "chat.completion.chunk", "created": created,
                  "model": MODEL_ID,
                  "choices": [{"index": 0, "delta": delta, "logprobs": None,
                               "finish_reason": finish}]})

        chunk({"role": "assistant"})
        rng = np.random.default_rng()
        t0 = time.perf_counter()
        gen = eng.generate(prompt_ids, max_new=max_tokens, temperature=temperature,
                           top_p=top_p, rng=rng, eos=chat.IM_END,
                           callback=lambda t: chunk({"content": tok.decode([t])}))
        dt = time.perf_counter() - t0
        finish = "stop" if len(gen) < max_tokens else "length"
        chunk({})
        emit("data: [DONE]")
        print(f"[chat-stream] prompt={len(prompt_ids)} completion={len(gen)} "
              f"{len(gen)/dt:.1f} tok/s", flush=True)


print(f"[vkops-server] api token: {TOKEN}", flush=True)
with open(r"C:\vkops\api_token.txt", "w") as f:
    f.write(TOKEN)
print("[vkops-server] loading model ...", flush=True)
gpu = ops.GPU()
bufs = chat.stream_upload(gpu, MODEL_DIR, NUM_LAYERS)
eng = chat.Qwen3Engine(gpu, bufs, NUM_LAYERS, HIDDEN, INTER, max_ctx=MAX_CTX)
import tokenizers
tok = tokenizers.Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
print(f"[vkops-server] ready on http://{HOST}:{PORT}  (model: {MODEL_ID})", flush=True)

server = ThreadingHTTPServer((HOST, PORT), Handler)
server.serve_forever()
