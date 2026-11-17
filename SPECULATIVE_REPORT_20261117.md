# Vega 8 cached-block verification and optional n-gram decoding

Measured 2026-11-17 UTC on Ryzen 7 5700G / AMD Radeon integrated graphics,
using the existing full 36-layer Qwen3-4B-Instruct-2507 abliterated model.
Weights, target RoPE theta (5e6), precision and architecture are unchanged.

## What changed

- Cached multi-token forward now applies a shifted causal mask, including at
  nonzero positions. The old path exposed later draft tokens to earlier rows.
- `all_logits=True` returns `[M,V]`; row i predicts the token after input i.
- Explicit valid KV length and checked truncation support rejection/overwrite.
  Invalid IDs, gaps and capacity overflow fail before GPU work. Failed forwards
  invalidate the overwritten suffix rather than treating partially written KV
  as valid.
- A dedicated 2/4/8-row FP16-storage kernel reuses each weight across input rows,
  with FP32 FMA/reduction matching the single-row kernel. The old large GEMM and
  non-transposed paths remain available.
- Optional greedy n-gram proposals are checked by the target, with contiguous
  prefix acceptance, rollback, EOS/budget handling and rejection backoff.
- `temperature=0` now means argmax rather than the old division-by-zero failure.
  N-gram is explicitly opt-in in Python and the API and requires temperature 0.
  Ordinary nonzero-temperature sampling defaults are unchanged.
- Optional selected full-layer feature snapshots prepare a separate external
  drafter experiment. Capture is disabled by default and has no extra GPU
  submissions/copies when disabled.

## Known-token verifier ceiling, not actual speculative speed

23-token prompt, 24 teacher-forced tokens, median of three warm runs. These
figures exclude any drafter and assume every supplied token can be retained.

| Input block | Previous kernel, token/s | Small-row kernel, token/s |
|---|---:|---:|
| 1 | 4.450 | 4.446 |
| 2 | 3.909 | 8.509 |
| 4 | 7.747 | 14.924 |
| 8 | 15.201 | 17.932 |

All 24 per-position argmax values matched the serial path. Maximum relative
logit error was 0 for block 2, 0.000768 for block 4, 0.000639 for block 8.
The new kernel also applies to ordinary 2–8-token prefills, whose FP32
accumulation can differ numerically from the previous FP16-slice algorithm.

## Actual n-gram generation

Two A/B repeats per public test prompt, alternating run order, with M1/M2/M4
pipelines explicitly warmed. Decode includes draft lookup, verification and
rollback. Model loading is excluded. The final columns include prompt prefill.
All baseline and n-gram token IDs matched in these tests.

| Workload | Tokens | Serial token/s | N-gram token/s | Ratio | Serial total s | N-gram total s |
|---|---:|---:|---:|---:|---:|---:|
| Short English explanation | 55 | 4.409 | 4.407 | 1.000x | 13.081 | 13.088 |
| Python Fibonacci code | 64 | 4.393 | 4.912 | 1.118x | 15.174 | 13.633 |
| Chinese explanation | 64 | 4.407 | 4.490 | 1.019x | 15.117 | 14.845 |
| Repeated sentence | 64 | 4.377 | 12.687 | 2.898x | 15.247 | 5.671 |

This is workload dependent, not a general 2.9x chat acceleration. The small
Chinese difference is close enough to run-to-run variation that it should not
be treated as a robust universal gain. N-gram remains off by default.

## Correctness and regression evidence

- Core CPU unit suites: 18 tests, including both API response paths, all-row
  logits, partial acceptance/rejection, backoff, EOS, cache poisoning/rollback,
  failure recovery, feature selection and invalid inputs.
- Vulkan `test_ops.py`, `test_decode.py`, `test_small_gemm.py`, buffer-capacity
  and CPU attention regressions passed. Small-row tests cover M2/3/4/5/7/8,
  non-multiple output tails, repeated buffers, canaries, independent FP64
  error bounds, and bit-identical individual GEMV reductions.
- Full 36-layer selftest: short-prefix cached versus prefill logits identical;
  NumPy FP16-weight reference relative error 0.000748, top-10 overlap 10/10.
- 1024-token full-model boundary: cached final block versus full prefill
  relative error 0.001532, same argmax; future-token independence, exact rollback
  replay, overflow rejection and full-layer feature taps passed.
- Existing large GEMM: 1098.3 GFLOPS; FFN: 3796 token/s, with numerical checks
  passing. No measured regression from the prior approximately 1083 GFLOPS /
  3720 token/s roofline tests.
- Independent code review covered shader barriers/tails, causal alignment,
  accepted-prefix state, feature hooks and API/default compatibility.

Floating-point batching is not a proof of bitwise identity for every possible
prompt, especially near tied logits. The algorithm verifies the target's block
argmax; these measured workloads additionally matched serial token IDs.

## Reproduction and rollback

See `README.md` for opt-in usage and test commands. Benchmarks use existing
local model files and write JSON/log artifacts; those artifacts and model
weights are not part of this source change.

`VKOPS_DISABLE_SMALL_GEMM=1` restores the previous small-matrix dispatch.
`VKOPS_DISABLE_GEMV=1` independently restores the previous single-row dispatch.
Neither switch reintroduces the old causal/cache bug. To restore the complete
previous runtime, stop users of the engine and restore `chat.py`, `ops.py` and
`server.py` from Git commit `87473097392ec0be0f13bfcb453a7e0d29e27e06`.
Additional shader/helper files are unused by that restored version.

No driver, BIOS, network or credential configuration was changed. No target
model was uploaded. External DFlash/DSpark/MTP integration is separate from
the validated opt-in n-gram path and is not claimed as an achieved speedup here.
