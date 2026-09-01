# -*- coding: utf-8 -*-
"""gen_gibberish.py — a 4-layer Qwen3-0.6B "model" (layers 1,5,15,26) running on
vkops Vulkan operators, generating text. Expect glorious gibberish: the layers
are non-contiguous, so the residual stream skips 24 of 28 layers.

GPU (packed FP16): all projections, FFN, final norm, lm_head.
CPU (host-mapped buffers, microsecond reads): per-head q/k-norm, RoPE,
GQA attention, sampling. No KV cache — full recompute per step (P <= 64).

All activation buffers are preallocated once (stable handles keep the
descriptor-set cache valid); zero temp allocations inside the loop.

Usage:
  python gen_gibberish.py selftest   # pipeline vs numpy reference (real weights)
  python gen_gibberish.py generate   # generate text
"""
import json
import math
import struct
import sys
import time

import numpy as np

import ops

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MODEL = [r"C:\vkops\qwen3_gen.safetensors",       # embed + layers 1,5,15,26 + final norm
         r"C:\vkops\qwen3_gen_more.safetensors"]  # layers 23,24,27
LAYERS = [1, 5, 15, 23, 24, 26, 27]
HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128
HIDDEN, INTER = 1024, 3072
ROPE_THETA = 1e6
EPS = 1e-6
MAX_P = 64
N_GEN = 48


# ---------------------------------------------------------------- loading ----
def load_tensors(paths, wanted):
    """Read only `wanted` tensors, searching across several safetensors files.
    data_offsets are relative to each file's tensor-data region (8 + header_len)."""
    headers = {}
    for path in paths:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        headers[path] = (h, 8 + n)
    out = {}
    for name in wanted:
        for path in paths:
            h, data_start = headers[path]
            if name in h:
                v = h[name]
                s, e = v["data_offsets"]
                with open(path, "rb") as f:
                    f.seek(data_start + s)
                    raw = f.read(e - s)
                if v["dtype"] == "BF16":
                    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                    arr = (u << 16).view(np.float32).reshape(v["shape"])
                elif v["dtype"] == "F16":
                    arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(v["shape"])
                else:
                    arr = np.frombuffer(raw, dtype=np.float32).reshape(v["shape"])
                out[name] = np.ascontiguousarray(arr)
                break
        else:
            raise KeyError(f"tensor not found in any file: {name}")
    return out


# ------------------------------------------------------------- cpu pieces ----
def head_rmsnorm(x, w, eps=EPS):
    P = x.shape[0]
    xr = x.reshape(P, -1, HEAD_DIM)
    inv = 1.0 / np.sqrt((xr * xr).mean(-1, keepdims=True) + eps)
    return (xr * inv * w).reshape(P, -1)


def rope(x, pos, heads):
    P = x.shape[0]
    xr = x.reshape(P, heads, HEAD_DIM).astype(np.float64)
    x1, x2 = xr[..., : HEAD_DIM // 2], xr[..., HEAD_DIM // 2:]
    inv = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM // 2, dtype=np.float64) / (HEAD_DIM // 2)))
    ang = pos[:, None].astype(np.float64) * inv[None, :]
    cos = np.cos(ang)[:, None, :]
    sin = np.sin(ang)[:, None, :]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return np.concatenate([o1, o2], -1).astype(np.float32).reshape(P, heads * HEAD_DIM)


def attention(q, k, v, P):
    qh = q.reshape(P, HEADS, HEAD_DIM)
    kh = k.reshape(P, KV_HEADS, HEAD_DIM)
    vh = v.reshape(P, KV_HEADS, HEAD_DIM)
    out = np.empty((P, HEADS, HEAD_DIM), np.float32)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    mask = np.triu(np.full((P, P), -1e30, np.float32), 1)
    for hq in range(HEADS):
        kvh = hq // (HEADS // KV_HEADS)
        s = qh[:, hq, :] @ kh[:, kvh, :].T * scale + mask
        s -= s.max(-1, keepdims=True)
        wgt = np.exp(s)
        wgt /= wgt.sum(-1, keepdims=True)
        out[:, hq, :] = wgt @ vh[:, kvh, :]
    return out.reshape(P, HEADS * HEAD_DIM)


def rmsnorm_np(x, w, eps=EPS):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


# ----------------------------------------------------------- engine (GPU) ----
class Engine:
    def __init__(self, gpu, w, layers):
        self.gpu = gpu
        self.embed = w["model.embed_tokens.weight"]            # [V, H] f32 (CPU)
        self.V, self.H = self.embed.shape
        self.embed16 = gpu.tensor_f16(self.embed.astype(np.float16))  # lm_head
        self.final_norm = gpu.tensor(w["model.norm.weight"])
        self.layers = []
        for L in layers:
            p = f"model.layers.{L}."
            d = {
                "ln1": gpu.tensor(w[p + "input_layernorm.weight"]),
                "ln2": gpu.tensor(w[p + "post_attention_layernorm.weight"]),
                "q": gpu.tensor_f16(w[p + "self_attn.q_proj.weight"].astype(np.float16)),
                "k": gpu.tensor_f16(w[p + "self_attn.k_proj.weight"].astype(np.float16)),
                "v": gpu.tensor_f16(w[p + "self_attn.v_proj.weight"].astype(np.float16)),
                "o": gpu.tensor_f16(w[p + "self_attn.o_proj.weight"].astype(np.float16)),
                "qn": w[p + "self_attn.q_norm.weight"],
                "kn": w[p + "self_attn.k_norm.weight"],
                "g": gpu.tensor_f16(w[p + "mlp.gate_proj.weight"].astype(np.float16)),
                "u": gpu.tensor_f16(w[p + "mlp.up_proj.weight"].astype(np.float16)),
                "dn": gpu.tensor_f16(w[p + "mlp.down_proj.weight"].astype(np.float16)),
            }
            self.layers.append(d)
        self._alloc(maxp=MAX_P)

    def _alloc(self, maxp):
        gpu = self.gpu
        f32h, f16h = maxp * self.H * 4, maxp * self.H * 2
        f32i, f16i = maxp * INTER * 4, maxp * INTER * 2
        qn, kvn = maxp * HEADS * HEAD_DIM * 4, maxp * KV_HEADS * HEAD_DIM * 4
        self.b = {
            "h": gpu.empty(f32h), "x16": gpu.empty(f16h), "x216": gpu.empty(f16h),
            "q": gpu.empty(qn), "k": gpu.empty(kvn), "v": gpu.empty(kvn),
            "att": gpu.empty(qn), "att16": gpu.empty(f16h), "o": gpu.empty(f32h),
            "res1": gpu.empty(f32h), "g": gpu.empty(f32i), "u": gpu.empty(f32i),
            "a": gpu.empty(f32i), "a16": gpu.empty(f16i), "dn": gpu.empty(f32h),
            "res2": gpu.empty(f32h), "xn": gpu.empty(f32h), "xn16": gpu.empty(f16h),
            "last16": gpu.empty(f16h), "logits": gpu.empty(self.V * 4),
        }

    def forward(self, ids):
        gpu, b = self.gpu, self.b
        P, H = len(ids), self.H
        pos = np.arange(P)
        h = b["h"]
        h.upload(self.embed[ids].astype(np.float32))

        for d in self.layers:
            # ---- attention block ----
            x = gpu.rmsnorm(h, d["ln1"], P, H, out=b["xn"])
            x16 = gpu.to_f16(x, P * H, dst=b["x16"])
            q = gpu.gemm_fp16(x16, d["q"], P, HEADS * HEAD_DIM, H, bT=True, out=b["q"])
            k = gpu.gemm_fp16(x16, d["k"], P, KV_HEADS * HEAD_DIM, H, bT=True, out=b["k"])
            v = gpu.gemm_fp16(x16, d["v"], P, KV_HEADS * HEAD_DIM, H, bT=True, out=b["v"])
            q_np = q.np(np.float32, count=P * HEADS * HEAD_DIM)[:P * HEADS * HEAD_DIM].reshape(P, -1).copy()
            k_np = k.np(np.float32, count=P * KV_HEADS * HEAD_DIM).reshape(P, -1).copy()
            v_np = v.np(np.float32, count=P * KV_HEADS * HEAD_DIM).reshape(P, -1).copy()
            q_np = rope(head_rmsnorm(q_np, d["qn"]), pos, HEADS)
            k_np = rope(head_rmsnorm(k_np, d["kn"]), pos, KV_HEADS)
            att = attention(q_np, k_np, v_np, P)
            b["att"].upload(att)
            att16 = gpu.to_f16(b["att"], P * HEADS * HEAD_DIM, dst=b["att16"])
            o = gpu.gemm_fp16(att16, d["o"], P, H, HEADS * HEAD_DIM, bT=True, out=b["o"])
            h = gpu.add(h, o, P * H, out=b["res1"])

            # ---- mlp block ----
            x2 = gpu.rmsnorm(h, d["ln2"], P, H, out=b["xn"])
            x216 = gpu.to_f16(x2, P * H, dst=b["x216"])
            g = gpu.gemm_fp16(x216, d["g"], P, INTER, H, bT=True, out=b["g"])
            u = gpu.gemm_fp16(x216, d["u"], P, INTER, H, bT=True, out=b["u"])
            a = gpu.silu_mul(g, u, P * INTER, out=b["a"])
            a16 = gpu.to_f16(a, P * INTER, dst=b["a16"])
            dn = gpu.gemm_fp16(a16, d["dn"], P, H, INTER, bT=True, out=b["dn"])
            h = gpu.add(h, dn, P * H, out=b["res2"])

        # ---- final norm + lm_head on last position ----
        x = gpu.rmsnorm(h, self.final_norm, P, H, out=b["xn"])
        gpu.to_f16(x, P * H, dst=b["xn16"])
        last_f16 = b["xn16"].np(np.float16, count=P * H).reshape(P, H)[P - 1].copy()
        b["last16"].np(np.float16, count=H)[:] = last_f16
        logits = gpu.gemm_fp16(b["last16"], self.embed16, 1, self.V, H, bT=True, out=b["logits"])
        return logits.np(np.float32, count=self.V).copy()


# ---------------------------------------------------- numpy reference fwd ----
def np_forward(w, layers, ids):
    """Mirror of the GPU pipeline with identical f16-quantized weights."""
    wq = {k: (v.astype(np.float16).astype(np.float32) if ("proj" in k or "embed" in k) else v)
          for k, v in w.items()}
    h = wq["model.embed_tokens.weight"][ids].astype(np.float32)
    pos = np.arange(len(ids))
    for L in layers:
        p = f"model.layers.{L}."
        x = rmsnorm_np(h, w[p + "input_layernorm.weight"])
        q = x @ wq[p + "self_attn.q_proj.weight"].T
        k = x @ wq[p + "self_attn.k_proj.weight"].T
        v = x @ wq[p + "self_attn.v_proj.weight"].T
        q = rope(head_rmsnorm(q, w[p + "self_attn.q_norm.weight"]), pos, HEADS)
        k = rope(head_rmsnorm(k, w[p + "self_attn.k_norm.weight"]), pos, KV_HEADS)
        att = attention(q, k, v, len(ids))
        h = h + att @ wq[p + "self_attn.o_proj.weight"].T
        x2 = rmsnorm_np(h, w[p + "post_attention_layernorm.weight"])
        g = x2 @ wq[p + "mlp.gate_proj.weight"].T
        u = x2 @ wq[p + "mlp.up_proj.weight"].T
        a = (g / (1 + np.exp(-g))) * u
        h = h + a @ wq[p + "mlp.down_proj.weight"].T
    x = rmsnorm_np(h, w["model.norm.weight"])
    return x[-1] @ wq["model.embed_tokens.weight"].T


# ------------------------------------------------------------------ main -----
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "generate"
    gpu = ops.GPU()
    print(f"[vk] {gpu.info()}", flush=True)

    print("[load] reading model tensors ...", flush=True)
    t0 = time.perf_counter()
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    for L in LAYERS:
        p = f"model.layers.{L}."
        names += [p + s for s in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")]
    w = load_tensors(MODEL, names)
    print(f"[load] {len(w)} tensors from {len(MODEL)} files in {time.perf_counter()-t0:.1f}s", flush=True)

    eng = Engine(gpu, w, LAYERS)
    ids0 = [7, 13, 29, 101, 512, 900]
    logits = eng.forward(ids0)
    ref = np_forward(w, LAYERS, ids0)
    d = np.max(np.abs(logits - ref)) / (np.max(np.abs(ref)) + 1e-9)
    top_ref = np.argsort(-ref)[:5]
    top_gpu = np.argsort(-logits)[:5]
    print(f"[selftest] rel_err={d:.3e}  top5 ref={list(top_ref)} gpu={list(top_gpu)}  "
          f"overlap={len(set(top_ref) & set(top_gpu))}/5  "
          f"[{'PASS' if d < 2e-2 else 'FAIL'}]", flush=True)
    if mode == "selftest":
        return 0 if d < 2e-2 else 1

    # ---- tokenizer (decode only) ----
    tk = json.load(open("tokenizer.json", encoding="utf-8"))
    vocab = tk["model"]["vocab"]
    id2tok = {v: k for k, v in vocab.items()}

    def decode(ids):
        s = []
        for i in ids:
            t = id2tok.get(i, f"<{i}>")
            s.append(t.replace("Ġ", " ").replace("Ċ", "\n"))
        return "".join(s)

    # prompt from real vocab pieces
    want = ["The", " little", " robot", " said", " hello", " world"]
    prompt = [vocab[t] for t in want if t in vocab]
    print(f"[prompt] ids={prompt} -> {decode(prompt)!r}", flush=True)

    rng = np.random.default_rng(42)
    ids = list(prompt)
    t0 = time.perf_counter()
    for step in range(N_GEN):
        logits = eng.forward(ids)
        l = logits.astype(np.float64)
        l -= l.max()
        p = np.exp(l / 0.9)
        p /= p.sum()
        top = np.argsort(-p)[:50]
        tw = p[top] / p[top].sum()
        nxt = int(rng.choice(top, p=tw))
        ids.append(nxt)
        print(decode([nxt]), end="", flush=True)
    dt = time.perf_counter() - t0
    print()
    print(f"\n[gen] {N_GEN} tokens in {dt:.1f}s ({N_GEN/dt:.1f} tok/s incl. full recompute)")
    print("[gen] full text:\n" + "-" * 60)
    print(decode(ids))
    print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
