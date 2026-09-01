# -*- coding: utf-8 -*-
"""bench_gemm_v2.py — verify + benchmark the register-blocked GEMM (v2) vs v1."""
import statistics
import sys
import time

import numpy as np

import ops
from ops import _load_shader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_gemm_v2(gpu, Ab, Bb, Db, M, N, K, bT):
    push = __import__("struct").pack("<4I4f", M, N, K, 0, 0.0, 1.0 if bT else 0.0, 0.0, 0.0)
    gpu.dev.submit_jobs([{
        "pipeline": gpu._pipeline("gemm_v2", 3),
        "buffers": [Ab, Bb, Db],
        "push": push,
        "gx": (N + 63) // 64, "gy": (M + 63) // 64, "gz": 1,
    }])


def rel_err(a, b):
    return np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12)


def main():
    gpu = ops.GPU()
    print(f"[vk] {gpu.info()}\n")

    # correctness (v2 is bit-compatible GEMM, no bias/act in probe shader)
    print("== gemm_v2 correctness ==")
    ok = True
    for (M, K, N, bT) in [(256, 384, 512, True), (128, 512, 256, False),
                          (100, 300, 200, True), (1024, 4096, 4096, True)]:
        rng = np.random.default_rng(0)
        A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        B = (rng.standard_normal((N, K) if bT else (K, N)) * 0.1).astype(np.float32)
        Ab, Bb = gpu.tensor(A), gpu.tensor(B)
        Db = gpu.empty(M * N * 4)
        run_gemm_v2(gpu, Ab, Bb, Db, M, N, K, bT)
        got = ops.GPU.to_np(Db, (M, N))
        ref = A @ (B.T if bT else B)
        e = rel_err(got, ref)
        ok &= e < 1e-3
        print(f"  {M}x{K}@{K}x{N} bT={bT}: rel_err={e:.2e} [{'PASS' if e < 1e-3 else 'FAIL'}")
    if not ok:
        print("correctness failed, skip benchmark")
        return 1

    # benchmark: v1 vs v2
    M, K, N = 1024, 4096, 4096
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)   # torch layout
    Ab, Bb = gpu.tensor(A), gpu.tensor(B)
    Db = gpu.empty(M * N * 4)

    print(f"\n== benchmark {M}x{K} @ {K}x{N} (bT, FP32) ==")
    for name, runner in [
        ("v1 16x16 tile", lambda: gpu.gemm(Ab, Bb, M, N, K, bT=True, out=Db)),
        ("v2 4x4 regblock", lambda: run_gemm_v2(gpu, Ab, Bb, Db, M, N, K, True)),
    ]:
        runner()  # warmup
        times = []
        for _ in range(15):
            t0 = time.perf_counter()
            runner()
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        gflops = 2 * M * N * K / med / 1e9
        print(f"  {name:18s}: {med*1000:8.2f} ms  {gflops:7.1f} GFLOPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
