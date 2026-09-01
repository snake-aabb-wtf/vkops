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
38.4 GB/s memory bandwidth; the fp16 GEMM runs at ~88% of streaming
bandwidth. FP16 numerics: rel err ~1e-3 vs f64 (slice-local f16
accumulation, f32 cross-slice).

## Real-model demo: a 7-layer Qwen3-0.6B

`real_model_test.py` runs one decoder-layer FFN with real Qwen3-0.6B weights
(BF16 -> FP16, verified against numpy, rel err 3.3e-3).

`gen_gibberish.py` assembles a working transformer out of **layers
1, 5, 15, 23, 24, 26, 27** (7 of 28 — the rest simply don't exist in the
weight file) and generates text end to end:

```
The
éŀŃä¼¦çŀŃllum przecRARY bundæĪĺåĽ½#errorãģ°LtdØ¹Ø§Ùħç¼ĵåĨ²Sc Pandora
<void="#">oyumen chaireduyartaamansey-č="#aåģ» am then am simulation
ceriesæľĭåıĭä»¬ figuresencing else initiativesä½įç½®ResponseBody
ÑģÐ»Ð¸å¯®uguay kidding_MASTERCKET terynam
```

Glorious gibberish — and provably *correct* gibberish: the GPU pipeline
matches a numpy reference with identical top-5 logits at every step. The
nonsense comes from the 21 missing layers, not from math bugs. GPU does all
projections/FFN/lm-head (packed FP16); CPU does per-head q/k-norm, RoPE
(rotate-half, theta 1e6), GQA attention (16:8) and top-k sampling over
host-mapped buffers (microsecond round-trips).

To try it you need the weights: grab `model.safetensors` from
[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (or the
hf-mirror), plus `tokenizer.json` from the same repo. The scripts read only
a small subset of tensors, so a subset file works too.

## Layout

```
vk.py               ctypes bindings to vulkan-1.dll (device/pipeline/dispatch core)
ops.py              tensor-level operator API + FFN blocks
st.py               safetensors reader/writer (pure numpy)
shaders/*.comp      GLSL compute shaders (compiled at runtime via glslc)
test_ops.py         numeric verification vs numpy + benchmarks
bench_*.py          per-optimization benchmarks (v2, ffn, fp16)
probe_perf.py       device roofline probes (features / bandwidth / FMA peak)
debug_pool.py       descriptor pool isolation test
gen_gibberish.py    7-layer Qwen3-0.6B text generation demo
real_model_test.py  single-layer FFN on real Qwen3 weights
deploy.py           SFTP deployment to a target machine (env-var credentials)
AGENTS.md           notes for AI coding agents working on this repo
```

## Usage sketch

```python
from ops import GPU
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

- Windows + a Vulkan 1.2+ device (1.3 recommended) + Vulkan SDK (`glslc` on
  PATH; the compile target-env is selected automatically from the device's
  supported API version)
- Python 3.10+, numpy
- FP16 path: `shaderFloat16` + `storageBuffer16BitAccess` (auto-detected,
  requires Vulkan >= 1.2); otherwise clean fallback to f32
- Shape constraints: `gemm_fp16` needs `K % 32 == 0` (and `N % 4 == 0` when
  weights are stored `[K,N]`); f32 `gemm`/`gemm_v2` handle arbitrary shapes.

## Deployment / test on a target machine

Credentials come from the environment (never committed):

```powershell
$env:VKOPS_SSH_HOST = '...'   # e.g. the target's Tailscale IP
$env:VKOPS_SSH_USER = '...'
$env:VKOPS_SSH_PASS = '...'
python deploy.py                    # push sources to C:/vkops on the target
# then run remotely: python -u C:\vkops\test_ops.py
```

`ref/` (the `vulkan_core.h` snapshot used to verify enum values) is
intentionally not committed — pull it from any Vulkan SDK 1.4.x install.

## Gotchas worth remembering

- safetensors `data_offsets` are relative to the tensor-data region
  (`8 + header_len`), **not** absolute file offsets.
- `VkPipelineShaderStageCreateInfo.stage` takes the *bit* value
  (`VK_SHADER_STAGE_COMPUTE_BIT = 0x20`), not an enum index.
- `vkAllocateDescriptorSets` wants the descriptor **set** layout, not the
  pipeline layout (mismatch surfaces as a confusing OUT_OF_POOL_MEMORY).
- On APUs prefer cached system-heap memory: the DEVICE_LOCAL heap is a tiny
  carveout the display also eats.
- If activation buffers are freed/reallocated while their descriptor sets
  are cached, recycled handles can silently alias old bindings — preallocate.

## License

MIT — see [LICENSE](LICENSE).
