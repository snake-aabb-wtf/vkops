# -*- coding: utf-8 -*-
"""test_ops.py — verify Vulkan operators against numpy and benchmark them.

Run:  python -u test_ops.py
"""
import os
import statistics
import time

import numpy as np

import ops
import st
from ops import ACT_NONE, ACT_RELU, ACT_SILU


def rel_err(a, b):
    return np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12)


def test_matmul(gpu):
    print("== matmul correctness ==")
    ok = True
    cases = [
        # (M, K, N, bias, act, bT)
        (37, 65, 129, False, ACT_NONE, False),
        (64, 256, 128, True, ACT_RELU, False),
        (511, 1023, 257, False, ACT_NONE, False),
        (129, 257, 511, False, ACT_SILU, False),
        (128, 512, 256, False, ACT_NONE, True),   # B stored [N,K]
    ]
    for i, (M, K, N, bias, act, bT) in enumerate(cases):
        rng = np.random.default_rng(i)
        A = rng.standard_normal((M, K)).astype(np.float32)
        Bshape = (N, K) if bT else (K, N)
        B = (rng.standard_normal(Bshape) * 0.1).astype(np.float32)
        C = rng.standard_normal(N).astype(np.float32) if bias else None
        Ab, Bb, Cb = gpu.tensor(A), gpu.tensor(B), (gpu.tensor(C) if bias else None)
        Db = gpu.gemm(Ab, Bb, M, N, K, C=Cb, act=act, bT=bT)
        D = ops.GPU.to_np(Db, (M, N))
        ref = A @ (B.T if bT else B)
        if bias:
            ref = ref + C
        if act == 1:
            ref = np.maximum(ref, 0)
        elif act == 3:
            ref = ref / (1.0 + np.exp(-ref))
        e = rel_err(D, ref)
        status = "PASS" if e < 1e-3 else "FAIL"
        ok &= e < 1e-3
        print(f"  case {i}: {M}x{K}@{K}x{N} bias={bias} act={act} bT={bT} -> rel_err={e:.2e} [{status}]")
    return ok


def test_elementwise(gpu):
    print("== elementwise/norm correctness ==")
    ok = True
    rng = np.random.default_rng(7)
    n = 100003  # non-multiple of workgroup size on purpose
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    ab, bb = gpu.tensor(a), gpu.tensor(b)
    out = ops.GPU.to_np(gpu.silu_mul(ab, bb), (n,))
    ref = (a / (1 + np.exp(-a))) * b
    e = rel_err(out, ref); ok &= e < 1e-5
    print(f"  silu_mul n={n}: rel_err={e:.2e} [{'PASS' if e < 1e-5 else 'FAIL'}]")

    out = ops.GPU.to_np(gpu.add(ab, bb), (n,))
    e = rel_err(out, a + b); ok &= e < 1e-6
    print(f"  add      n={n}: rel_err={e:.2e} [{'PASS' if e < 1e-6 else 'FAIL'}]")

    rows, width = 33, 1024
    x = rng.standard_normal((rows, width)).astype(np.float32)
    w = (rng.standard_normal(width) * 0.2 + 1).astype(np.float32)
    xb, wb = gpu.tensor(x), gpu.tensor(w)
    out = ops.GPU.to_np(gpu.rmsnorm(xb, wb, rows, width), (rows, width))
    ref = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w
    e = rel_err(out, ref); ok &= e < 1e-5
    print(f"  rmsnorm  {rows}x{width}: rel_err={e:.2e} [{'PASS' if e < 1e-5 else 'FAIL'}]")
    return ok


def test_safetensors_ffn(gpu):
    print("== safetensors -> FFN block (5 dispatches, one submit) ==")
    d_model, hidden, M = 512, 1376, 64
    wpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_weights.safetensors")
    ops.make_ffn_weights(d_model, hidden, seed=42, path=wpath)
    w = st.load_file(wpath)  # <-- the actual safetensors reader being tested
    print(f"  loaded {wpath}: {len(w)} tensors, dtypes:"
          f" {sorted({str(v.dtype) for v in w.values()})}")
    rng = np.random.default_rng(9)
    x = rng.standard_normal((M, d_model)).astype(np.float32)
    xb = gpu.tensor(x)
    wnb = gpu.tensor(w["norm.weight"])
    wgb = gpu.tensor(w["ffn.gate_proj.weight"])   # [hidden, d_model] -> bT path
    wub = gpu.tensor(w["ffn.up_proj.weight"])
    wdb = gpu.tensor(w["ffn.down_proj.weight"])   # [d_model, hidden] -> bT path
    outb = gpu.ffn_block(xb, wnb, wgb, wub, wdb, M, d_model, hidden)
    got = ops.GPU.to_np(outb, (M, d_model))
    ref = ops.ffn_numpy(x, w)
    e = rel_err(got, ref)
    ok = e < 5e-3
    print(f"  FFN {M}x{d_model}->{hidden}->{d_model}: rel_err={e:.2e} [{'PASS' if ok else 'FAIL'}]")
    return ok


def bench_matmul(gpu, M=1024, K=4096, N=4096, reps=15):
    print(f"== matmul benchmark {M}x{K} @ {K}x{N} (FP32) ==")
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    Ab, Bb = gpu.tensor(A), gpu.tensor(B)
    Db = gpu.gemm(Ab, Bb, M, N, K)  # warmup + compile
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        gpu.dev.submit_jobs([{
            "pipeline": gpu._pipeline("gemm", 4),
            "buffers": [Ab, Bb, gpu._dummy, Db],
            "push": __import__("struct").pack("<4I4f", M, N, K, 0, 0.0, 0.0, 0.0, 0.0),
            "gx": (N + 15) // 16, "gy": (M + 15) // 16, "gz": 1,
        }])
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    gflops = 2 * M * N * K / med / 1e9
    print(f"  median {med*1000:.2f} ms -> {gflops:.1f} GFLOPS (max {max(times)*1000:.1f} ms)")
    return gflops


def bench_ffn(gpu, M=1024, d_model=4096, hidden=11008, reps=10):
    print(f"== FFN block benchmark M={M} d={d_model} h={hidden} ==")
    x = np.zeros((M, d_model), np.float32)
    w = ops.make_ffn_weights(d_model, hidden, seed=1)
    xb = gpu.tensor(x)
    bufs = [gpu.tensor(w[k]) for k in
            ["norm.weight", "ffn.gate_proj.weight", "ffn.up_proj.weight", "ffn.down_proj.weight"]]
    gpu.ffn_block(xb, *bufs, M, d_model, hidden)  # warmup
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        gpu.ffn_block(xb, *bufs, M, d_model, hidden)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    flops = 3 * 2 * M * d_model * hidden
    print(f"  median {med*1000:.2f} ms -> {flops/med/1e9:.1f} GFLOPS, {M/med:.0f} tokens/s")
    return med


def main():
    gpu = ops.GPU()
    print(f"[vk] device: {gpu.info()}\n")
    ok = True
    ok &= test_matmul(gpu)
    ok &= test_elementwise(gpu)
    ok &= test_safetensors_ffn(gpu)
    print()
    bench_matmul(gpu)
    bench_ffn(gpu)
    print("\n====", "ALL TESTS PASSED" if ok else "SOME TESTS FAILED", "====")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
