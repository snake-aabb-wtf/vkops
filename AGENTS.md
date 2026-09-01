# AGENTS.md — notes for AI coding agents

This file orients AI coding agents (Claude Code, Codex, Cursor, ...) working
on this repository. Read it before changing anything.

## What this project is

`vkops` is a from-scratch Vulkan compute stack in Python: ctypes bindings to
`vulkan-1.dll` (`vk.py`), GLSL compute shaders compiled at runtime with
`glslc` (`shaders/*.comp`), a tensor-level operator API (`ops.py`), and a
safetensors reader/writer (`st.py`). It was built to run transformer math on
a Windows AMD APU (Ryzen 7 5700G, Vega 8 iGPU) that has no CUDA. The whole
thing has zero dependencies beyond numpy.

The demo endpoint is `gen_gibberish.py`: a 7-of-28-layer Qwen3-0.6B that
provably computes correctly (top-5 logits identical to a numpy reference)
and emits glorious gibberish, because 21 layers are missing.

## Environment

- Development happens on a Windows workstation; execution/validation happens
  on a remote Windows target machine with the Vega 8 iGPU.
- Target connection details (host/user/password) come **only** from
  environment variables `VKOPS_SSH_HOST` / `VKOPS_SSH_USER` /
  `VKOPS_SSH_PASS`. Never hardcode them, never commit them. This repo is
  public.
- Deployment: `python deploy.py` pushes tracked sources to the target
  (`C:\vkops`). For files larger than a few MB use chunked parallel SFTP
  (16-24 concurrent connections); a single stream over Tailscale DERP can be
  as slow as ~50 KB/s while 24 streams reach ~0.5-1 MB/s. Reassemble parts
  remotely and verify MD5 before use.
- The target has a full Vulkan SDK at `C:\VulkanSDK\<ver>\Bin` (glslc). A
  dev machine without the SDK can run with a pre-populated SPIR-V cache
  (`%TEMP%\vkops_spv`, key = sha1(target-env + "|" + source)) or with
  `glslc.exe` + `shaderc_shared.dll` copied from the SDK Bin directory.
- Chinese-locale Windows consoles default to GBK: scripts that may print
  model text must call `sys.stdout.reconfigure(encoding="utf-8",
  errors="replace")` first, and set `PYTHONIOENCODING=utf-8` when spawning.

## Verification workflow (mandatory)

1. Make changes locally in this repo.
2. Push to the target (deploy.py or chunked push) and run the numeric
   regression there: `python -u C:\vkops\test_ops.py` — it must end with
   `ALL TESTS PASSED`.
3. Any kernel-level optimization also gets a benchmark run (bench_*.py) and
   the measured number goes into the commit message.
4. Numeric checks compare against numpy with f16-quantized weights when
   validating the FP16 path (isolates accumulation error from quantization).
5. When touching Vulkan plumbing, enable the validation layer
   (`VKOPS_VALIDATE=1` env var) — it turns silent corruption into explicit
   VUID messages (this caught two real bugs already).

## Hard-won gotchas

- safetensors `data_offsets` are relative to the tensor-data region
  (`8 + header_len`), not absolute file offsets. Symptom of getting it
  wrong: weights "contain" values beyond f16 range and cast to inf.
- `VkPipelineShaderStageCreateInfo.stage` wants the *bit* value
  `VK_SHADER_STAGE_COMPUTE_BIT = 0x20`, not an enum index (5).
- `vkAllocateDescriptorSets` takes the descriptor **set** layout, not the
  pipeline layout. Wrong one returns `VK_ERROR_OUT_OF_POOL_MEMORY
  (-1000069000)` with no hint at the real cause.
- Structure type enums differ from memory: pull the values from a real
  `vulkan_core.h` (any SDK 1.4.x), don't trust recall.
- Device creation pNext chains for FP16 features are core since Vulkan 1.2;
  on 1.1 devices they would need extension names, so `fp16_enabled` requires
  API >= 1.2 plus the features.
- glslc `--target-env` must match the device API (SPIR-V 1.6 needs a 1.3
  driver); the SPIR-V cache key includes the target-env for this reason.
- Descriptor sets cache on (pipeline, buffer-handle) tuples. Never free and
  reallocate activation buffers inside a loop with a warm cache — recycled
  handles alias stale bindings. Preallocate per-role buffers and use the
  `out=` parameter on every op.
- Old/iGPU-specific: prefer cached system-heap memory over the tiny
  DEVICE_LOCAL carveout; big benchmarks on old discrete cards (e.g. GT 730)
  can trip Windows TDR — keep validation cases small.

## FP16 path constraints

`gemm_fp16` requires `K % 32 == 0` and (only when `bT=False`) `N % 4 == 0`.
Accumulation is f16 within a 32-wide k-slice and f32 across slices; expect
~1e-3 rel err vs f64, which is fine for inference. Reference numerics must
quantize weights identically (f32 -> f16 -> f32) before comparing.

## Performance baselines (5700G, Vega 8)

Measured device rooflines: 2.99 TFLOPS FP32 FMA (probe_perf.py), 38.4 GB/s
stream bandwidth. GEMM 1024x4096x4096: v1 f32 218 GFLOPS -> v2 f32 516 ->
v3 fp16 1083 (88% of streaming bandwidth — memory-bound now). FFN
d=4096/h=11008: 3720 tok/s fp16. If your change regresses these, the commit
message should say why it's still worth it.

## Conventions

- Every op takes an `out=` buffer parameter; long-running sequences
  preallocate everything up front.
- Shaders live in `shaders/*.comp`, registered in `ops.SHADER_FILES`.
- Keep the library dependency-light: numpy only. No torch, no external
  bindings.
- Branch/commit directly to `main`; perf numbers in commit messages.
