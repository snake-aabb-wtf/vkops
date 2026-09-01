# -*- coding: utf-8 -*-
"""real_model_test.py — run a real Qwen3-0.6B safetensors file through vkops.

Selectively reads only the tensors needed for one decoder layer (embeddings +
layer-0 norm + FFN weights), so the 1.4 GB file never has to be fully loaded.

Checks the GPU FFN block (packed FP16) against a numpy f32 reference computed
on the same bf16->f16-quantized weights, then benchmarks tokens/s.
"""
import json
import statistics
import struct
import sys
import time

import numpy as np

import ops

MODEL = r"C:\vkops\qwen3_layer0.safetensors"   # 21MB subset (bf16 bits preserved)
LAYER = 0


def load_tensors(path, wanted):
    """Read only `wanted` tensors from a safetensors file; BF16 -> F32.
    Note: data_offsets are relative to the tensor-data region (8 + header_len)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    data_start = 8 + n
    out = {}
    with open(path, "rb") as f:
        for name in wanted:
            v = header[name]
            s, e = v["data_offsets"]
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
    return out


def main():
    gpu = ops.GPU()
    print(f"[vk] {gpu.info()}")
    if not gpu.dev.fp16_enabled:
        print("fp16 not supported on this device, aborting")
        return 1

    keys = [f"model.layers.{LAYER}.mlp.{p}_proj.weight" for p in ("gate", "up", "down")] + \
           [f"model.layers.{LAYER}.input_layernorm.weight", "embed.sample"]
    t0 = time.perf_counter()
    w = load_tensors(MODEL, keys)
    dt = time.perf_counter() - t0
    mb = sum(v.nbytes for v in w.values()) >> 20
    print(f"[load] selective read: {len(keys)} tensors, {mb} MB in {dt:.2f}s")

    w_gate = w[f"model.layers.{LAYER}.mlp.gate_proj.weight"]
    hidden, inter = w_gate.shape[1], w_gate.shape[0]
    print(f"[model] hidden={hidden}, inter={inter}")

    # --- real inputs: bf16 embedding rows sampled from the real table ---
    M = 1024
    x = w["embed.sample"][:M].copy()

    wref = {
        "norm.weight": w[f"model.layers.{LAYER}.input_layernorm.weight"],
        "ffn.gate_proj.weight": w_gate,
        "ffn.up_proj.weight": w[f"model.layers.{LAYER}.mlp.up_proj.weight"],
        "ffn.down_proj.weight": w[f"model.layers.{LAYER}.mlp.down_proj.weight"],
    }

    # --- GPU fp16 run on real weights ---
    wnb = gpu.tensor(wref["norm.weight"])
    wgb = gpu.tensor_f16(wref["ffn.gate_proj.weight"].astype(np.float16))
    wub = gpu.tensor_f16(wref["ffn.up_proj.weight"].astype(np.float16))
    wdb = gpu.tensor_f16(wref["ffn.down_proj.weight"].astype(np.float16))
    xb = gpu.tensor(x)
    outb = gpu.ffn_block_fp16(xb, wnb, wgb, wub, wdb, M, hidden, inter)
    got = ops.GPU.to_np(outb, (M, hidden))

    # --- numpy reference on the same bf16->f16-quantized weights ---
    wq = {k: v.astype(np.float16).astype(np.float32) for k, v in wref.items()}
    ref = ops.ffn_numpy(x, wq)
    e = np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-12)
    ok = e < 1e-2
    print(f"[numeric] layer {LAYER} FFN (real weights, M={M}): rel_err={e:.2e} "
          f"[{'PASS' if ok else 'FAIL'}]")

    # --- benchmark fp16 ---
    gpu.ffn_block_fp16(xb, wnb, wgb, wub, wdb, M, hidden, inter)  # warmup
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        gpu.ffn_block_fp16(xb, wnb, wgb, wub, wdb, M, hidden, inter)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    flops = 3 * 2 * M * hidden * inter
    print(f"[bench] FFN fp16 M={M} d={hidden} h={inter}: {med*1000:.2f} ms  "
          f"{flops/med/1e9:.1f} GFLOPS  {M/med:.0f} tokens/s")
    print(f"[scale] whole model, FFN only, 28 layers: {28*med*1000:.0f} ms/forward "
          f"-> {M/(28*med):.0f} tok/s")

    # --- f32 comparison on the same real weights ---
    gf32 = [gpu.tensor(wref["ffn.gate_proj.weight"]), gpu.tensor(wref["ffn.up_proj.weight"]),
            gpu.tensor(wref["ffn.down_proj.weight"])]
    xb2 = gpu.tensor(x)
    gpu.ffn_block(xb2, wnb, *gf32, M, hidden, inter)  # warmup
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        gpu.ffn_block(xb2, wnb, *gf32, M, hidden, inter)
        times.append(time.perf_counter() - t0)
    med32 = statistics.median(times)
    print(f"[bench] FFN f32  M={M}: {med32*1000:.2f} ms  {flops/med32/1e9:.1f} GFLOPS  "
          f"{M/med32:.0f} tokens/s  (fp16 speedup: {med32/med:.2f}x)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
