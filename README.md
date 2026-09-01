# vkops — Vulkan Compute Operator Library

A minimal, dependency-light (numpy only) Vulkan compute operator library that
loads **safetensors** weight files and runs compute operators on any Vulkan
GPU — developed and validated on an AMD Ryzen 7 5700G (Vega 8 iGPU, Windows,
no CUDA needed).

## Operators

| op | shader | notes |
|---|---|---|
| `gemm` (v1) | `shaders/gemm.comp` | 16x16 tile, f32, bias + fused relu/gelu/silu |
| `gemm` (v2) | `shaders/gemm_v2.comp` | 64x64 tile, 4x4/thread register blocking |
| `gemm_fp16` | `shaders/gemm_fp16.comp` | packed FP16 (`V_PK_FMA_F16`), f16vec4-blocked LDS, f32 cross-slice accumulation |
| `linear` | — | nn.Linear-style wrapper (torch `[out,in]` weights, no CPU transpose) |
| `silu_mul` | `shaders/silu_mul.comp` | SwiGLU-style fused activation |
| `add`, `rmsnorm` | `shaders/add.comp`, `shaders/rmsnorm.comp` | |
| `ffn_block` / `ffn_block_fp16` | — | llama-style FFN, whole block recorded into ONE submit |
| `cvt_f32_f16` | `shaders/cvt_f32_f16.comp` | GPU-side f32 -> f16 conversion |

`safetensors` reader/writer lives in `st.py` (F32/F16/BF16, auto-upcast to F32
for the f32 pipelines; F16 weights feed `gemm_fp16` directly).

## Performance (Ryzen 7 5700G, Vega 8 iGPU, measured)

GEMM 1024x4096x4096 / FFN d=4096 h=11008 M=1024:

| kernel | GEMM GFLOPS | FFN tokens/s |
|---|---|---|
| v1 f32 (16x16 tile) | 218 | 447 |
| v2 f32 (register blocked) | 516 | 1833 |
| v3 fp16 packed | **1083** | **3720** |

Device rooflines measured on this iGPU: ~2.99 TFLOPS FP32 FMA peak,
38.4 GB/s memory bandwidth; fp16 GEMM runs at ~88% of streaming bandwidth.
FP16 numerics: rel err ~1e-3 vs f64 (slice-local f16 accumulation, f32
cross-slice).

## Layout

```
vk.py            ctypes bindings to vulkan-1.dll (device/pipeline/dispatch core)
ops.py           tensor-level operator API + FFN blocks
st.py            safetensors reader/writer (pure numpy)
shaders/*.comp   GLSL compute shaders (compiled at runtime via glslc)
test_ops.py      numeric verification vs numpy + benchmarks
bench_*.py       per-optimization benchmarks (v2, ffn, fp16)
probe_perf.py    device roofline probes (features / bandwidth / FMA peak)
debug_pool.py    descriptor pool isolation test
deploy.py        SFTP deployment to the target machine
```

## Usage sketch

```python
from ops import GPU, make_ffn_weights
import st

gpu = GPU()
w = st.load_file("weights.safetensors")          # real safetensors file
x = gpu.tensor(batch_tokens)                      # f32 activations
out = gpu.ffn_block_fp16(x,
    gpu.tensor(w["norm.weight"]),
    gpu.tensor_f16(w["ffn.gate_proj.weight"].astype("float16")),
    gpu.tensor_f16(w["ffn.up_proj.weight"].astype("float16")),
    gpu.tensor_f16(w["ffn.down_proj.weight"].astype("float16")),
    M, d_model, hidden)
```

## Requirements

- Windows (or any OS with adjustments) + Vulkan 1.3 device + Vulkan SDK (`glslc` on PATH)
- Python 3.10+, numpy
- FP16 path: `shaderFloat16` + `storageBuffer16BitAccess` (auto-detected)
- Shape constraints: `gemm_fp16` needs `K % 32 == 0` (and `N % 4 == 0` when
  weights are stored `[K,N]`); f32 `gemm`/`gemm_v2` handle arbitrary shapes.

## Deployment / test on the target machine

Credentials come from the environment (not committed):

```powershell
$env:VKOPS_SSH_PASS = '...'
python deploy.py                    # push sources to C:/vkops on the target
python ..\ssh_exec.py administrator <pass> "cd /d C:\vkops && python -u test_ops.py"
```

`ref/` (the `vulkan_core.h` snapshot used to verify enum values) is
intentionally not committed — pull it from any Vulkan SDK 1.4.x install.

## Roadmap

- LDS double buffering (hide global latency, last ~12% of bandwidth)
- Workgroup swizzle for L2 tile reuse
- Split-K GEMV for M=1 decode (~70 tok/s bandwidth ceiling on this machine)
- GPU timestamp queries for per-dispatch timing; RGP profiling
