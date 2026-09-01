# -*- coding: utf-8 -*-
"""bench_ffn_v2.py — FFN block with gemm_v2 (register-blocked) dispatches, one submit."""
import statistics
import struct
import sys
import time

import numpy as np

import ops
from ops import _ceil_div, _push

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    gpu = ops.GPU()
    print(f"[vk] {gpu.info()}\n")

    M, d_model, hidden = 1024, 4096, 11008
    x = np.zeros((M, d_model), np.float32)
    w = ops.make_ffn_weights(d_model, hidden, seed=1)
    xb = gpu.tensor(x)
    wnb = gpu.tensor(w["norm.weight"])
    wgb = gpu.tensor(w["ffn.gate_proj.weight"])
    wub = gpu.tensor(w["ffn.up_proj.weight"])
    wdb = gpu.tensor(w["ffn.down_proj.weight"])

    h = gpu.empty(M * d_model * 4)
    g = gpu.empty(M * hidden * 4)
    u = gpu.empty(M * hidden * 4)
    a = gpu.empty(M * hidden * 4)
    out = gpu.empty(M * d_model * 4)

    pl_rms = gpu._pipeline("rmsnorm", 3)
    pl_g2 = gpu._pipeline("gemm_v2", 3)
    pl_sm = gpu._pipeline("silu_mul", 3)

    def ffn_v2():
        gpu.dev.submit_jobs([
            {"pipeline": pl_rms, "buffers": [xb, wnb, h],
             "push": _push(M, d_model, f0=1e-6), "gx": M},
            {"pipeline": pl_g2, "buffers": [h, wgb, g],
             "push": _push(M, hidden, d_model, 0, 0.0, 1.0),
             "gx": _ceil_div(hidden, 64), "gy": _ceil_div(M, 64)},
            {"pipeline": pl_g2, "buffers": [h, wub, u],
             "push": _push(M, hidden, d_model, 0, 0.0, 1.0),
             "gx": _ceil_div(hidden, 64), "gy": _ceil_div(M, 64)},
            {"pipeline": pl_sm, "buffers": [g, u, a],
             "push": _push(M * hidden), "gx": _ceil_div(M * hidden, 256)},
            {"pipeline": pl_g2, "buffers": [a, wdb, out],
             "push": _push(M, d_model, hidden, 0, 0.0, 1.0),
             "gx": _ceil_div(d_model, 64), "gy": _ceil_div(M, 64)},
        ])

    # numeric sanity: compare small case vs numpy
    Ms, ds, hs = 64, 512, 1376
    w2 = ops.make_ffn_weights(ds, hs, seed=42)
    xs = np.random.default_rng(9).standard_normal((Ms, ds)).astype(np.float32)
    xsb = gpu.tensor(xs)
    h2 = gpu.empty(Ms * ds * 4); g2 = gpu.empty(Ms * hs * 4)
    u2 = gpu.empty(Ms * hs * 4); a2 = gpu.empty(Ms * hs * 4)
    o2 = gpu.empty(Ms * ds * 4)
    gpu.dev.submit_jobs([
        {"pipeline": pl_rms, "buffers": [xsb, gpu.tensor(w2["norm.weight"]), h2],
         "push": _push(Ms, ds, f0=1e-6), "gx": Ms},
        {"pipeline": pl_g2, "buffers": [h2, gpu.tensor(w2["ffn.gate_proj.weight"]), g2],
         "push": _push(Ms, hs, ds, 0, 0.0, 1.0), "gx": _ceil_div(hs, 64), "gy": _ceil_div(Ms, 64)},
        {"pipeline": pl_g2, "buffers": [h2, gpu.tensor(w2["ffn.up_proj.weight"]), u2],
         "push": _push(Ms, hs, ds, 0, 0.0, 1.0), "gx": _ceil_div(hs, 64), "gy": _ceil_div(Ms, 64)},
        {"pipeline": pl_sm, "buffers": [g2, u2, a2],
         "push": _push(Ms * hs), "gx": _ceil_div(Ms * hs, 256)},
        {"pipeline": pl_g2, "buffers": [a2, gpu.tensor(w2["ffn.down_proj.weight"]), o2],
         "push": _push(Ms, ds, hs, 0, 0.0, 1.0), "gx": _ceil_div(ds, 64), "gy": _ceil_div(Ms, 64)},
    ])
    got = ops.GPU.to_np(o2, (Ms, ds))
    ref = ops.ffn_numpy(xs, w2)
    e = np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-12)
    print(f"FFN(v2) numeric: rel_err={e:.2e} [{'PASS' if e < 5e-3 else 'FAIL'}]\n")

    ffn_v2()  # warmup
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        ffn_v2()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    flops = 3 * 2 * M * d_model * hidden
    print(f"FFN v2  M={M} d={d_model} h={hidden}: {med*1000:.2f} ms  "
          f"{flops/med/1e9:.1f} GFLOPS  {M/med:.0f} tokens/s")
    print("(v1 baseline was 2293.20 ms / 120.8 GFLOPS / 447 tokens/s)")


if __name__ == "__main__":
    raise SystemExit(main())
