# -*- coding: utf-8 -*-
"""bench_fp16.py — packed-FP16 GEMM/FFN: correctness vs f64 reference + perf vs FP32 v2."""
import statistics
import sys
import time

import numpy as np

import ops

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def rel_err(a, b):
    return np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12)


def main():
    gpu = ops.GPU()
    print(f"[vk] {gpu.info()}  fp16_enabled={gpu.dev.fp16_enabled}\n")
    if not gpu.dev.fp16_enabled:
        print("device does not support shaderFloat16/16bit storage")
        return 1

    # ---- correctness (ref computed in float64 from the same f16 inputs) ----
    print("== gemm_fp16 correctness (ref: f64 on identical f16 inputs) ==")
    ok = True
    for (M, K, N, bT) in [(256, 384, 512, True), (128, 512, 256, False),
                          (96, 320, 206, True), (37, 512, 132, False),
                          (1024, 4096, 4096, True)]:
        rng = np.random.default_rng(0)
        A16 = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
        B16 = (rng.standard_normal((N, K) if bT else (K, N)) * 0.1).astype(np.float16)
        Ab, Bb = gpu.tensor_f16(A16), gpu.tensor_f16(B16)
        Db = gpu.gemm_fp16(Ab, Bb, M, N, K, bT=bT)
        got = ops.GPU.to_np(Db, (M, N))
        ref = A16.astype(np.float64) @ (B16.astype(np.float64).T if bT else B16.astype(np.float64))
        e = rel_err(got.astype(np.float64), ref)
        ok &= e < 1e-2
        print(f"  {M}x{K}@{K}x{N} bT={bT}: rel_err={e:.2e} [{'PASS' if e < 1e-2 else 'FAIL'}]")
    if not ok:
        print("correctness failed, skip benchmark")
        return 1

    # ---- GEMM benchmark: f32 v2 vs fp16 ----
    M, K, N = 1024, 4096, 4096
    rng = np.random.default_rng(0)
    A32 = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B32 = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    A16, B16 = A32.astype(np.float16), B32.astype(np.float16)
    Ab32, Bb32 = gpu.tensor(A32), gpu.tensor(B32)
    Ab16, Bb16 = gpu.tensor_f16(A16), gpu.tensor_f16(B16)
    Db = gpu.empty(M * N * 4)
    struct = __import__("struct")

    print(f"\n== GEMM {M}x{K} @ {K}x{N}, weights: f32 {A32.nbytes+B32.nbytes>>20}MB"
          f" -> f16 {(A16.nbytes+B16.nbytes)>>20}MB ==")
    variants = [
        ("v1 f32 (16x16)", lambda: gpu.gemm(Ab32, Bb32, M, N, K, bT=True, out=Db)),
        ("fp16 packed", lambda: gpu.gemm_fp16(Ab16, Bb16, M, N, K, bT=True, out=Db)),
    ]
    for name, fn in variants:
        fn()
        times = []
        for _ in range(15):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        print(f"  {name:12s}: {med*1000:8.2f} ms  {2*M*N*K/med/1e9:7.1f} GFLOPS")

    # ---- FFN end-to-end ----
    print("\n== FFN block, FP16 weights ==")
    d_model, hidden = 4096, 11008
    w = ops.make_ffn_weights(d_model, hidden, seed=1)
    w16 = {k: v.astype(np.float16) for k, v in w.items()}
    Ms = 64
    x = np.random.default_rng(9).standard_normal((Ms, d_model)).astype(np.float32)
    outb = gpu.ffn_block_fp16(
        gpu.tensor(x), gpu.tensor(w["norm.weight"]),
        gpu.tensor_f16(w16["ffn.gate_proj.weight"]), gpu.tensor_f16(w16["ffn.up_proj.weight"]),
        gpu.tensor_f16(w16["ffn.down_proj.weight"]), Ms, d_model, hidden)
    got = ops.GPU.to_np(outb, (Ms, d_model))
    w16r = {k: v.astype(np.float32) for k, v in w16.items()}  # quantized weights in f32
    ref_quant = ops.ffn_numpy(x, w16r)      # isolates kernel accumulation error
    ref_f32 = ops.ffn_numpy(x, w)           # includes weight quantization effect
    e_k = rel_err(got, ref_quant)
    e_q = rel_err(got, ref_f32)
    print(f"  numeric vs f32-numpy on same f16 weights: rel_err={e_k:.2e} "
          f"[{'PASS' if e_k < 1e-2 else 'FAIL'}]")
    print(f"  total vs pure-f32 (incl. weight quantization): rel_err={e_q:.2e}")

    M2 = 1024
    x2 = np.zeros((M2, d_model), np.float32)
    xb2 = gpu.tensor(x2)
    wnb = gpu.tensor(w["norm.weight"])
    wgb = gpu.tensor_f16(w16["ffn.gate_proj.weight"])
    wub = gpu.tensor_f16(w16["ffn.up_proj.weight"])
    wdb = gpu.tensor_f16(w16["ffn.down_proj.weight"])
    gpu.ffn_block_fp16(xb2, wnb, wgb, wub, wdb, M2, d_model, hidden)  # warmup
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        gpu.ffn_block_fp16(xb2, wnb, wgb, wub, wdb, M2, d_model, hidden)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    flops = 3 * 2 * M2 * d_model * hidden
    print(f"  FFN fp16 M={M2}: {med*1000:.2f} ms  {flops/med/1e9:.1f} GFLOPS  {M2/med:.0f} tokens/s")
    print("  (v2 f32 baseline: 558.71 ms / 495.8 GFLOPS / 1833 tokens/s)")
    print(f"  FFN weights: f32 {sum(v.nbytes for v in w.values())>>20} MB ->"
          f" f16 {sum(v.nbytes for v in w16.values())>>20} MB")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
