# -*- coding: utf-8 -*-
"""gen_chat.py — real chat with a dense Qwen3-4B-Instruct-2507 (abliterated)
running on vkops Vulkan operators (packed FP16) + CPU attention with KV cache.

Usage:
  python gen_chat.py selftest            # cache-path equivalence + numpy reference
  python gen_chat.py "你的问题"           # one-shot chat
  python gen_chat.py                      # interactive loop
"""
import json
import os
import sys
import time

import numpy as np

import chat
import ops

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = r"C:\vkops\qwen3-4b"
NUM_LAYERS, HIDDEN, INTER = 36, 2560, 9728
MAX_CTX = 512

if len(sys.argv) > 2 and sys.argv[1] == "selftest" and os.path.isdir(sys.argv[2]):
    MODEL_DIR = sys.argv[2]  # e.g. synthetic weight set for early pipeline validation


def apply_template(msg):
    return f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"


def load_engine(gpu):
    print("[load] streaming model weights (bf16 -> f16) ...", flush=True)
    t0 = time.perf_counter()
    bufs = chat.stream_upload(gpu, MODEL_DIR, NUM_LAYERS)
    print(f"[load] done in {time.perf_counter()-t0:.1f}s", flush=True)
    return chat.Qwen3Engine(gpu, bufs, NUM_LAYERS, HIDDEN, INTER, max_ctx=MAX_CTX)


def load_tokenizer():
    import tokenizers
    tok_dir = r"C:\vkops\qwen3-4b" if os.path.isdir(r"C:\vkops\qwen3-4b") else MODEL_DIR
    tok = tokenizers.Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
    return tok


def selftest(gpu, eng, tok):
    print("== selftest 1: KV-cache consistency (prefill vs prefill+decode) ==", flush=True)
    ids = tok.encode("The capital of France is").ids
    ids = ids[:10]
    full = eng.forward(ids, pos_start=0)
    eng.clear_cache()
    part = eng.forward(ids[:-1], pos_start=0)
    last = eng.forward([ids[-1]], pos_start=len(ids) - 1)
    d = np.max(np.abs(full - last)) / (np.max(np.abs(full)) + 1e-9)
    same_top = int(np.argmax(full) == np.argmax(last))
    print(f"  rel_err={d:.3e}  argmax_match={same_top}  "
          f"[{'PASS' if d < 1e-2 and same_top else 'FAIL'}]", flush=True)
    ok = d < 1e-2 and same_top

    print("== selftest 2: GPU prefill vs numpy streaming reference (fp16-quantized) ==",
          flush=True)
    P = 6
    ids2 = tok.encode("你好，请介绍一下你自己。").ids[:P]
    t0 = time.perf_counter()
    got = eng.forward(ids2, pos_start=0)
    print(f"  GPU prefill done in {time.perf_counter()-t0:.1f}s", flush=True)
    eng.clear_cache()

    headers = chat.scan_headers(MODEL_DIR)
    h = chat.read_tensor(MODEL_DIR, "model.embed_tokens.weight", headers)[ids2]
    h = h.astype(np.float16).astype(np.float32)
    pos = np.arange(len(ids2))
    for li in range(NUM_LAYERS):
        p = f"model.layers.{li}."
        q = lambda nm: chat.read_tensor(MODEL_DIR, p + nm, headers) \
            .astype(np.float16).astype(np.float32)
        f32 = lambda nm: chat.read_tensor(MODEL_DIR, p + nm, headers)
        x = chat.rmsnorm_np(h, f32("input_layernorm.weight"))
        qq = chat.rope(chat.head_rmsnorm(x @ q("self_attn.q_proj.weight").T,
                                         f32("self_attn.q_norm.weight")), pos, chat.HEADS)
        kk = chat.rope(chat.head_rmsnorm(x @ q("self_attn.k_proj.weight").T,
                                         f32("self_attn.k_norm.weight")), pos, chat.KV_HEADS)
        vv = x @ q("self_attn.v_proj.weight").T
        att = chat.attention_np(qq, kk, vv, len(ids2), len(ids2), True)
        h = h + att @ q("self_attn.o_proj.weight").T
        x2 = chat.rmsnorm_np(h, f32("post_attention_layernorm.weight"))
        g = x2 @ q("mlp.gate_proj.weight").T
        u = x2 @ q("mlp.up_proj.weight").T
        a = (g / (1 + np.exp(-g))) * u
        h = h + a @ q("mlp.down_proj.weight").T
        if li % 9 == 0:
            print(f"  numpy layer {li+1}/36 ...", flush=True)
    x = chat.rmsnorm_np(h, chat.read_tensor(MODEL_DIR, "model.norm.weight", headers))
    ref = x[-1] @ chat.read_tensor(MODEL_DIR, "model.embed_tokens.weight", headers) \
        .astype(np.float16).astype(np.float32).T
    d2 = np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-9)
    top_ref = np.argsort(-ref)[:10]
    top_gpu = np.argsort(-got)[:10]
    ov = len(set(top_ref) & set(top_gpu))
    print(f"  rel_err={d2:.3e}  top10 overlap={ov}/10  "
          f"[{'PASS' if d2 < 5e-2 and ov >= 8 else 'FAIL'}]", flush=True)
    return ok and d2 < 5e-2 and ov >= 8


def chat_once(gpu, eng, tok, user_msg, max_new=200):
    prompt_ids = tok.encode(apply_template(user_msg)).ids
    print(f"[prompt] {len(prompt_ids)} tokens", flush=True)
    pieces = []

    def cb(nxt):
        t = tok.decode([nxt])
        pieces.append(t)
        print(t, end="", flush=True)

    t0 = time.perf_counter()
    gen = eng.generate(prompt_ids, max_new=max_new, temperature=0.7, top_p=0.8,
                       rng=np.random.default_rng(int(time.time())), eos=chat.IM_END,
                       callback=cb)
    dt = time.perf_counter() - t0
    print()
    print(f"[gen] {len(gen)} tokens in {dt:.1f}s -> {len(gen)/dt:.1f} tok/s", flush=True)
    return tok.decode(gen)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    gpu = ops.GPU()
    print(f"[vk] {gpu.info()}", flush=True)
    eng = load_engine(gpu)
    tok = load_tokenizer()

    if mode == "selftest":
        ok = selftest(gpu, eng, tok)
        print("SELFTEST", "PASSED" if ok else "FAILED", flush=True)
        return 0 if ok else 1

    if mode == "cmp_paths":
        ids2 = tok.encode("你好，请介绍一下你自己。").ids[:6]
        old_logits, old_states = eng.forward_perop(ids2, 0, collect_states=True)
        eng.clear_cache()
        new_logits = eng.forward(ids2, 0)
        eng.clear_cache()
        eng.forward(ids2, 0, collect_states=True)
        new_states = eng.layer_states
        d_log = np.max(np.abs(old_logits - new_logits)) / (np.max(np.abs(old_logits)) + 1e-9)
        print(f"logits old-vs-new: rel_err={d_log:.3e}", flush=True)
        for li, (a, b) in enumerate(zip(old_states, new_states)):
            e = np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-9)
            if e > 1e-4 or li < 3:
                print(f"state {li}: old-vs-new rel_err={e:.3e}", flush=True)
        return 0

    if mode == "diverge":
        ids2 = tok.encode("你好，请介绍一下你自己。").ids[:6]
        eng.forward(ids2, pos_start=0, collect_states=True)
        states = eng.layer_states
        headers = chat.scan_headers(MODEL_DIR)

        def wt(nm):
            return chat.read_tensor(MODEL_DIR, nm, headers) \
                .astype(np.float16).astype(np.float32)

        def wf(nm):
            return chat.read_tensor(MODEL_DIR, nm, headers)

        h = chat.read_tensor(MODEL_DIR, "model.embed_tokens.weight", headers)[ids2] \
            .astype(np.float16).astype(np.float32)
        states_np = [h.copy()]
        pos = np.arange(len(ids2))
        for li in range(NUM_LAYERS):
            p = f"model.layers.{li}."
            x = chat.rmsnorm_np(h, wf(p + "input_layernorm.weight"))
            qq = chat.rope(chat.head_rmsnorm(x @ wt(p + "self_attn.q_proj.weight").T,
                                             wf(p + "self_attn.q_norm.weight")), pos, chat.HEADS)
            kk = chat.rope(chat.head_rmsnorm(x @ wt(p + "self_attn.k_proj.weight").T,
                                             wf(p + "self_attn.k_norm.weight")), pos, chat.KV_HEADS)
            vv = x @ wt(p + "self_attn.v_proj.weight").T
            att = chat.attention_np(qq, kk, vv, len(ids2), len(ids2), True)
            h = h + att @ wt(p + "self_attn.o_proj.weight").T
            states_np.append(h.copy())
            x2 = chat.rmsnorm_np(h, wf(p + "post_attention_layernorm.weight"))
            g = x2 @ wt(p + "mlp.gate_proj.weight").T
            u = x2 @ wt(p + "mlp.up_proj.weight").T
            a = (g / (1 + np.exp(-g))) * u
            h = h + a @ wt(p + "mlp.down_proj.weight").T
            states_np.append(h.copy())
        first = None
        for li, (g, r) in enumerate(zip(states, states_np)):
            e = np.max(np.abs(g - r)) / (np.max(np.abs(r)) + 1e-9)
            tag = ""
            if e > 0.05 and first is None:
                first = li
                tag = "   <<< FIRST DIVERGENCE"
                print("   gpu:", g[0][:8], flush=True)
                print("   ref:", r[0][:8], flush=True)
            print(f"state {li}: rel_err={e:.3e}{tag}", flush=True)
        print("first divergence at state", first, flush=True)
        return 0

    if mode:
        chat_once(gpu, eng, tok, " ".join(sys.argv[1:]))
        return 0

    print("interactive mode (empty line to quit)", flush=True)
    while True:
        try:
            msg = input("you> ").strip()
        except EOFError:
            break
        if not msg:
            break
        chat_once(gpu, eng, tok, msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
