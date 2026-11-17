# vkops 5700G decode optimization, 2026-11-17

## Applied result

The verified implementation is installed in `C:\vkops` on the existing Ryzen 7 5700G / Vega 8 target. It uses the complete 36-layer Qwen3-4B model, with all 398 tensors loaded. Original model files, experimental files, drivers, BIOS, network settings and API/server configuration were not changed. No service was started. The verified source was subsequently brought to the development PC for Git versioning, with model files, credentials and experiment artifacts excluded.

Same 23-token prompt and 24 fixed decode tokens, warmed model:

| Measurement | Decode tokens/s | Median prefill seconds |
|---|---:|---:|
| Fresh original-path baseline | 1.9202 | 0.6529 |
| Optimized isolated copy | 4.3810 | 0.6192 |
| Optimized original project after deployment | **4.3471** | **0.6248** |

Deployment speedup is **2.264x**, about **126% more tokens per second**. Earlier recovered experiments measured 4.43 tokens/s; normal measurement variation is why the deployed result above is the final number. These are short-context single-stream decode measurements, not a guarantee for all prompt lengths or concurrent users. Model loading remains about 22 seconds and is excluded from decode timing.

## Implementation

- `ops.py` dispatches transposed-weight, single-row FP16 operations to the new `shaders\gemv_fp16.comp`. The kernel uses FP16 storage and FP32 accumulation; existing multi-row GEMM/prefill and non-transposed fallbacks remain.
- `chat.py` groups the query heads sharing each KV head to reduce CPU attention overhead.
- `chat.py` corrects the attention FP16 buffer allocation to use query-head width, which is larger than hidden width in this model.
- Added `test_decode.py`, `test_attention.py`, `test_chat_buffers.py`, and `bench_decode.py` for reproducibility.

Only two existing runtime source files were replaced. The new shader and test/benchmark scripts were added. SHA256 hashes were checked before backup, against reviewed candidates, and after copying.

## Verification

- `test_ops.py`: all tests passed in the isolated copy with Vulkan validation enabled.
- Dedicated GEMV: odd output widths, short K, FP64 references on identical FP16 inputs, elementwise accumulation bounds, output canaries, changed inputs through warm descriptors, and original GEMM fallback all passed. Repeated on deployed files with Vulkan validation; no validation errors observed.
- `bench_fp16.py`: correctness passed; FP16 GEMM **1088 GFLOPS** and FFN **3779 tokens/s**, consistent with the existing 1083 / 3720 baselines.
- Canonical attention tests: independent original function versus the actual integrated implementation; causal/noncausal cases, poisoned unused cache, genuinely noncontiguous arrays, and 512/1024-token prefills passed.
- Complete-model selftest: KV-cache relative error **0.002027**, argmax matched; GPU prefill versus FP16-quantized NumPy reference relative error **0.001744**, top 10 overlap **10/10**.
- Fixed-token original-versus-deployed logits at prefill and decode steps 0, 7, 23: relative maximum errors **0.00087–0.00154**, argmax matched, top 10 overlap **10/10** at every sample. All 12 greedy output tokens were identical. Floating-point logits are not bitwise identical, and this limited sample does not prove identical output on every prompt.
- Full existing server-context boundary: 1024-token optimized prefill versus 1023-token prefill plus cached decode passed; relative error **0.001190**, argmax matched, top 10 overlap **10/10**. Full prefill took **19.30 seconds**. This tests the optimized engine at capacity, not old-versus-new long-context throughput or rejection of inputs beyond capacity.
- Independent read-only review found no remaining production blocker after test corrections.

An old `attention.log` failure is retained as historical evidence. Large-amplitude FP32 inputs exposed differing summation order, not incorrect head/cache indexing. Both paths were compared to independent FP64 calculations. The canonical test distinguishes mathematical dot/softmax perturbation checks from the assumption-dependent post-softmax numerical tolerance. The exploratory candidate script now requires an explicit original-source reference, avoiding accidental self-comparison.

## Run and reproduce

Existing usage is unchanged:

```powershell
Set-Location C:\vkops
python gen_chat.py
python test_decode.py
python test_chat_buffers.py
python -c "import test_attention; test_attention.test_candidates()"
python gen_chat.py selftest C:\vkops\qwen3-4b
python bench_decode.py --label my_new_benchmark --steps 24
```

Use a new benchmark label to avoid overwriting earlier artifacts. Full logs, benchmark JSON and fixed-token logits remain in `C:\vkops-opt-20261117`, including `recovery_*.log`, `deployed_benchmark.log`, `deployed.json`, and `deployed_compare.log`. Recovered original experiments remain there as well.

## Rollback

Original code is recoverable from Git commit `7b941f2ab774a2e030a6ddf94947aa0efea2703b`. The separate target backup directory was removed at the owner's request. Its deployment manifest is retained with the experiment artifacts as `deployment_manifest.json`.

With vkops processes stopped, restore the original runtime from Git:

```powershell
git restore --source=7b941f2ab774a2e030a6ddf94947aa0efea2703b --worktree -- chat.py ops.py
```

The added shader/tests can stay; the restored original runtime does not use the new shader. For kernel-only A/B testing without rollback, set `$env:VKOPS_DISABLE_GEMV='1'` in the shell before launching; remove that environment variable to restore optimized routing. This toggle does not disable grouped CPU attention.

## Further optimization candidates

Profile longer contexts and representative requests before more changes. Potential next steps are dimension-specific GEMV selection, moving attention/RoPE onto the GPU, and reducing CPU/GPU synchronization. Large GEMV microbenchmarks already approach roughly 38 GB/s, the measured memory bandwidth, so another large FP16 speedup is not assured. Quantization or speculative decoding would be separate quality/architecture tradeoffs, not part of this implementation.
