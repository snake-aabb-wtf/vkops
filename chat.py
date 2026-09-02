# -*- coding: utf-8 -*-
"""chat.py — dense Qwen3 chat engine on vkops Vulkan operators (packed FP16).

Streaming loader: reads safetensors shards tensor-by-tensor (bf16 -> f32 -> f16),
uploads, frees — the model never materializes in RAM as f32.

KV cache: prefill runs all prompt rows once (M=P) and fills per-layer CPU-side
K/V caches; each generated token runs M=1 through the layers. CPU-side numpy
does per-head q/k-norm, RoPE (rotate-half), GQA attention (32:8) and sampling;
host-mapped buffers make GPU->CPU reads microsecond-cheap.
"""
import json
import math
import os
import struct
import sys
import time

import numpy as np

import ops

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
ROPE_THETA = 5e6
EPS = 1e-6
IM_END = 151645
VOCAB = 151936


# ---------------------------------------------------------------- loading ----
def read_tensor(model_dir, name, headers=None):
    """Read one tensor (bf16/f16/f32 -> f32) from the sharded safetensors files."""
    if headers is None:
        headers = scan_headers(model_dir)
    shard, v = headers[name]
    data_start = v["data_start"]
    s, e = v["data_offsets"]
    with open(os.path.join(model_dir, shard), "rb") as f:
        f.seek(data_start + s)
        raw = f.read(e - s)
    if v["dtype"] == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        arr = (u << 16).view(np.float32).reshape(v["shape"])
    elif v["dtype"] == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(v["shape"])
    else:
        arr = np.frombuffer(raw, dtype=np.float32).reshape(v["shape"])
    return np.ascontiguousarray(arr)


def scan_headers(model_dir):
    """Return {tensor_name: (shard_filename, {dtype, shape, data_offsets, data_start})}."""
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx_path, "rb") as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted(set(weight_map.values()))
    headers = {}
    for shard in shards:
        with open(os.path.join(model_dir, shard), "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(hl))
        h.pop("__metadata__", None)
        for name, v in h.items():
            v["data_start"] = 8 + hl
            headers[name] = (shard, v)
    return headers


def stream_upload(gpu, model_dir, num_layers, f32_keys=("norm",)):
    """Upload all model tensors; big ones as f16, norms as f32."""
    headers = scan_headers(model_dir)
    bufs = {}
    wanted = [n for n in headers
              if n.startswith("model.") and "vision" not in n and "mtp" not in n]
    n = 0
    t0 = time.time()
    for name in sorted(wanted):
        arr = read_tensor(model_dir, name, headers)
        if any(k in name for k in f32_keys):
            bufs[name] = gpu.tensor(arr)
        else:
            bufs[name] = gpu.tensor_f16(arr.astype(np.float16))
        n += 1
        if n % 50 == 0:
            print(f"  uploaded {n}/{len(wanted)} tensors ({time.time()-t0:.0f}s)", flush=True)
        del arr
    print(f"[load] {n} tensors uploaded in {time.time()-t0:.0f}s", flush=True)
    return bufs


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


def rmsnorm_np(x, w, eps=EPS):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def attention_np(q, k, v, M, T, causal):
    """q [M, Hq*d] roped/normalized; k, v full caches [T, Hkv*d]; returns [M, Hq*d]."""
    qh = q.reshape(M, HEADS, HEAD_DIM)
    out = np.empty((M, HEADS * HEAD_DIM), np.float32)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    for hq in range(HEADS):
        kvh = hq // (HEADS // KV_HEADS)
        s = qh[:, hq, :] @ k[:T, kvh * HEAD_DIM:(kvh + 1) * HEAD_DIM].T * scale
        if causal:
            s += np.triu(np.full((M, T), -1e30, np.float32), T - M + 1)
        s -= s.max(-1, keepdims=True)
        wgt = np.exp(s)
        wgt /= wgt.sum(-1, keepdims=True)
        out[:, hq * HEAD_DIM:(hq + 1) * HEAD_DIM] = \
            wgt @ v[:T, kvh * HEAD_DIM:(kvh + 1) * HEAD_DIM]
    return out.reshape(M, HEADS * HEAD_DIM)


# ----------------------------------------------------------------- engine ----
class Qwen3Engine:
    def __init__(self, gpu, bufs, num_layers, hidden, inter, max_ctx=512):
        self.gpu = gpu
        self.b = bufs
        self.num_layers, self.H, self.inter = num_layers, hidden, inter
        self.V = VOCAB
        self.max_ctx = max_ctx
        self._embed_view = None
        self.max_ctx = max_ctx
        self.embed16 = bufs["model.embed_tokens.weight"]
        self.final_norm = bufs["model.norm.weight"]
        self._layers = []
        for li in range(num_layers):
            p = f"model.layers.{li}."
            self._layers.append({
                "ln1": bufs[p + "input_layernorm.weight"],
                "ln2": bufs[p + "post_attention_layernorm.weight"],
                "q": bufs[p + "self_attn.q_proj.weight"],
                "k": bufs[p + "self_attn.k_proj.weight"],
                "v": bufs[p + "self_attn.v_proj.weight"],
                "o": bufs[p + "self_attn.o_proj.weight"],
                # q/k norms are applied on the CPU side -> keep numpy copies
                "qn": bufs[p + "self_attn.q_norm.weight"].np(np.float32, count=HEAD_DIM).copy(),
                "kn": bufs[p + "self_attn.k_norm.weight"].np(np.float32, count=HEAD_DIM).copy(),
                "g": bufs[p + "mlp.gate_proj.weight"],
                "u": bufs[p + "mlp.up_proj.weight"],
                "dn": bufs[p + "mlp.down_proj.weight"],
            })
        self._alloc(max_ctx)

    def _alloc(self, maxp):
        gpu, H, I = self.gpu, self.H, self.inter
        f32h, f16h = maxp * H * 4, maxp * H * 2
        f32i, f16i = maxp * I * 4, maxp * I * 2
        qn, kvn = maxp * HEADS * HEAD_DIM * 4, maxp * KV_HEADS * HEAD_DIM * 4
        self.s = {
            "xn": gpu.empty(f32h), "x16": gpu.empty(f16h), "x216": gpu.empty(f16h),
            "q": gpu.empty(qn), "k": gpu.empty(kvn), "v": gpu.empty(kvn),
            "att": gpu.empty(qn), "att16": gpu.empty(f16h), "o": gpu.empty(f32h),
            "ra": gpu.empty(f32h), "rb": gpu.empty(f32h),
            "g": gpu.empty(f32i), "u": gpu.empty(f32i),
            "a": gpu.empty(f32i), "a16": gpu.empty(f16i), "dn": gpu.empty(f32h),
        }
        # per-layer f16 input buffers so descriptor sets stay per-layer stable
        self.x16_layer = [gpu.empty(f16h) for _ in range(self.num_layers)]
        self.kc = [np.zeros((maxp, KV_HEADS * HEAD_DIM), np.float32)
                   for _ in range(self.num_layers)]
        self.vc = [np.zeros((maxp, KV_HEADS * HEAD_DIM), np.float32)
                   for _ in range(self.num_layers)]
        # lm_head runs on all M rows (B matrix read once regardless of M)
        self.logits_buf = gpu.empty(maxp * VOCAB * 4)

    def _embed_f16(self):
        if self._embed_view is None:
            self._embed_view = self.embed16.np(np.float16, count=self.V * self.H) \
                .reshape(self.V, self.H)
        return self._embed_view

    def clear_cache(self):
        for kc, vc in zip(self.kc, self.vc):
            kc[:] = 0
            vc[:] = 0

    def forward_perop(self, ids, pos_start=0, collect_states=False):
        """Reference path: one submit per op (the original implementation).
        Used to A/B against the batched forward on the same device."""
        gpu, s = self.gpu, self.s
        M, H = len(ids), self.H
        T = pos_start + M
        h = s["ra"]
        h.upload(self._embed_f16()[np.asarray(ids)].astype(np.float32))
        causal = (pos_start == 0 and M > 1)
        states = None
        if collect_states:
            states = [h.np(np.float32, count=M * H).reshape(M, H).copy()]
        for li, d in enumerate(self._layers):
            x = gpu.rmsnorm(h, d["ln1"], M, H, out=s["xn"])
            x16 = gpu.to_f16(x, M * H, dst=self.x16_layer[li])
            q = gpu.gemm_fp16(x16, d["q"], M, HEADS * HEAD_DIM, H, bT=True, out=s["q"])
            k = gpu.gemm_fp16(x16, d["k"], M, KV_HEADS * HEAD_DIM, H, bT=True, out=s["k"])
            v = gpu.gemm_fp16(x16, d["v"], M, KV_HEADS * HEAD_DIM, H, bT=True, out=s["v"])
            q_np = q.np(np.float32, count=M * HEADS * HEAD_DIM).reshape(M, -1).copy()
            k_np = k.np(np.float32, count=M * KV_HEADS * HEAD_DIM).reshape(M, -1).copy()
            v_np = v.np(np.float32, count=M * KV_HEADS * HEAD_DIM).reshape(M, -1).copy()
            q_np = rope(head_rmsnorm(q_np, d["qn"]), pos_start + np.arange(M), HEADS)
            k_np = rope(head_rmsnorm(k_np, d["kn"]), pos_start + np.arange(M), KV_HEADS)
            self.kc[li][pos_start:T] = k_np
            self.vc[li][pos_start:T] = v_np
            att = attention_np(q_np, self.kc[li], self.vc[li], M, T, causal)
            s["att"].upload(att)
            att16 = gpu.to_f16(s["att"], M * HEADS * HEAD_DIM, dst=s["att16"])
            o = gpu.gemm_fp16(att16, d["o"], M, H, HEADS * HEAD_DIM, bT=True, out=s["o"])
            h = gpu.add(h, o, M * H, out=s["rb"])
            if collect_states:
                states.append(h.np(np.float32, count=M * H).reshape(M, H).copy())
            x2 = gpu.rmsnorm(h, d["ln2"], M, H, out=s["xn"])
            x216 = gpu.to_f16(x2, M * H, dst=s["x216"])
            g = gpu.gemm_fp16(x216, d["g"], M, self.inter, H, bT=True, out=s["g"])
            u = gpu.gemm_fp16(x216, d["u"], M, self.inter, H, bT=True, out=s["u"])
            a = gpu.silu_mul(g, u, M * self.inter, out=s["a"])
            a16 = gpu.to_f16(a, M * self.inter, dst=s["a16"])
            dn = gpu.gemm_fp16(a16, d["dn"], M, H, self.inter, bT=True, out=s["dn"])
            h = gpu.add(h, dn, M * H, out=s["ra"])
            if collect_states:
                states.append(h.np(np.float32, count=M * H).reshape(M, H).copy())
        x = gpu.rmsnorm(h, self.final_norm, M, H, out=s["xn"])
        gpu.to_f16(x, M * H, dst=s["x216"])
        logits = gpu.gemm_fp16(s["x216"], self.embed16, M, VOCAB, H,
                               bT=True, out=self.logits_buf)
        if collect_states:
            return logits.np(np.float32, count=M * VOCAB) \
                .reshape(M, VOCAB)[M - 1].copy(), states
        return logits.np(np.float32, count=M * VOCAB).reshape(M, VOCAB)[M - 1].copy()

    def forward(self, ids, pos_start=0, collect_states=False, stop_after=None):
        """Run ids (list) at positions pos_start..; cache K/V; return last logits.

        Batched submission: all GPU dispatches between two CPU-dependency points
        (attention reads) are recorded into ONE command buffer. Per token this is
        ~37 submits instead of one per op (~500)."""
        gpu, s = self.gpu, self.s
        M, H = len(ids), self.H
        T = pos_start + M
        h = s["ra"]
        h.upload(self._embed_f16()[np.asarray(ids)].astype(np.float32))
        causal = (pos_start == 0 and M > 1)
        jobs = []
        states = None
        if collect_states:
            self.layer_states = [h.np(np.float32, count=M * H).reshape(M, H).copy()]
            states = self.layer_states
        layers = self._layers if stop_after is None else self._layers[:stop_after]
        for li, d in enumerate(layers):
            # ---- attention segment: fused-norm + q/k/v, then CPU attention ----
            j, x16 = gpu.job_rmsnorm_f16(h, d["ln1"], M, H, out=self.x16_layer[li])
            jobs.append(j)
            jobs.append(gpu.job_gemm_fp16(x16, d["q"], M, HEADS * HEAD_DIM, H,
                                          bT=True, out=s["q"])[0])
            jobs.append(gpu.job_gemm_fp16(x16, d["k"], M, KV_HEADS * HEAD_DIM, H,
                                          bT=True, out=s["k"])[0])
            jobs.append(gpu.job_gemm_fp16(x16, d["v"], M, KV_HEADS * HEAD_DIM, H,
                                          bT=True, out=s["v"])[0])
            gpu.dev.submit_jobs(jobs)
            jobs = []

            q_np = s["q"].np(np.float32, count=M * HEADS * HEAD_DIM).reshape(M, -1).copy()
            k_np = s["k"].np(np.float32, count=M * KV_HEADS * HEAD_DIM).reshape(M, -1).copy()
            v_np = s["v"].np(np.float32, count=M * KV_HEADS * HEAD_DIM).reshape(M, -1).copy()
            q_np = rope(head_rmsnorm(q_np, d["qn"]), pos_start + np.arange(M), HEADS)
            k_np = rope(head_rmsnorm(k_np, d["kn"]), pos_start + np.arange(M), KV_HEADS)
            self.kc[li][pos_start:T] = k_np
            self.vc[li][pos_start:T] = v_np
            att = attention_np(q_np, self.kc[li], self.vc[li], M, T, causal)
            s["att16"].upload(att.astype(np.float16))  # CPU converts, no cvt kernel

            # ---- B1: o-projection + residual add ----
            jobs.append(gpu.job_gemm_fp16(s["att16"], d["o"], M, H, HEADS * HEAD_DIM,
                                          bT=True, out=s["o"])[0])
            jobs.append(gpu.job_add(h, s["o"], M * H, out=s["rb"])[0])
            h = s["rb"]
            if collect_states:
                gpu.dev.submit_jobs(jobs)
                jobs = []
                states.append(h.np(np.float32, count=M * H).reshape(M, H).copy())

            # ---- B2: mlp + residual add ----
            j, xn16 = gpu.job_rmsnorm_f16(h, d["ln2"], M, H, out=s["x216"])
            jobs.append(j)
            jobs.append(gpu.job_gemm_fp16(xn16, d["g"], M, self.inter, H,
                                          bT=True, out=s["g"])[0])
            jobs.append(gpu.job_gemm_fp16(xn16, d["u"], M, self.inter, H,
                                          bT=True, out=s["u"])[0])
            jobs.append(gpu.job_silu_mul_f16(s["g"], s["u"], M * self.inter,
                                             out=s["a16"])[0])
            jobs.append(gpu.job_gemm_fp16(s["a16"], d["dn"], M, H, self.inter,
                                          bT=True, out=s["dn"])[0])
            h = s["ra"]
            jobs.append(gpu.job_add(s["rb"], s["dn"], M * H, out=h)[0])
            if collect_states:
                gpu.dev.submit_jobs(jobs)
                jobs = []
                states.append(h.np(np.float32, count=M * H).reshape(M, H).copy())

        # ---- final norm + lm_head on all M rows (read last row of logits) ----
        if stop_after is not None:
            return None  # per-layer states collected in self.layer_states
        j, x16 = gpu.job_rmsnorm_f16(h, self.final_norm, M, H, out=s["x216"])
        jobs.append(j)
        jobs.append(gpu.job_gemm_fp16(x16, self.embed16, M, VOCAB, H,
                                      bT=True, out=self.logits_buf)[0])
        if os.environ.get("VKOPS_SUBMIT_EACH"):
            for jb in jobs:
                gpu.dev.submit_jobs([jb])
        else:
            gpu.dev.submit_jobs(jobs)
        return self.logits_buf.np(np.float32, count=M * VOCAB) \
            .reshape(M, VOCAB)[M - 1].copy()

    def generate(self, prompt_ids, max_new=128, temperature=0.7, top_p=0.8,
                 rng=None, eos=IM_END, callback=None):
        rng = rng or np.random.default_rng()
        self.clear_cache()
        ids = list(prompt_ids)
        logits = self.forward(ids, pos_start=0)
        generated = []
        for _ in range(max_new):
            nxt = sample(logits, temperature, top_p, rng)
            if nxt == eos:
                break
            generated.append(nxt)
            if callback:
                callback(nxt)
            logits = self.forward([nxt], pos_start=len(ids))
            ids.append(nxt)
        return generated


def sample(logits, temperature=0.7, top_p=0.8, rng=None):
    rng = rng or np.random.default_rng()
    l = logits.astype(np.float64)
    l -= l.max()
    p = np.exp(l / temperature)
    p /= p.sum()
    order = np.argsort(-p)
    csum = np.cumsum(p[order])
    cutoff = int(np.searchsorted(csum, top_p) + 1)
    idx = order[:max(cutoff, 1)]
    w = p[idx] / p[idx].sum()
    return int(rng.choice(idx, p=w))
