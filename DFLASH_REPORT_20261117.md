# DFlash1 on the 5700G: useful for one code workload, not a default chat speedup

Measured 2026-11-17 UTC with the unchanged full 36-layer local
Qwen3-4B-Instruct-2507 abliterated target and the Vulkan verifier from `ae3c9c5`.

## Checkpoint and compatibility

The experimental drafter is
[`z-lab/Qwen3-4B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16),
whose model card specifies MIT. The single safetensors file is 1,074,860,568
bytes, 58 BF16 tensors / 537,427,200 parameters, SHA256:

`a21d0ffe5970c84e9a0710ccb12087d1a4bcbedf979e623476c92c9d005a9cd4`

The implementation reads tensors directly; it never executes checkpoint Python
or `auto_map` code. Matrices use the existing FP16 storage / FP32-accumulation
path, norms stay FP32. The five-layer drafter shares the target embedding/head.

**This is cross-variant:** the public drafter targets original Qwen3-4B, not the
local Instruct-2507/abliterated variant. Drafter RoPE remains 1e6; target RoPE
remains 5e6. Matching dimensions do not establish learned compatibility.

## Implementation

- Capture complete target decoder outputs at 0-based layers 1/9/17/25/33.
- Project and normalize these features; draft queries attend noncausally to
  cached real context and the complete noise block.
- Persist only target-context K/V, never previous noise-token K/V.
- Verify proposed tokens with the target; keep only the contiguous accepted
  prefix and its target features, then use the target correction token.
- `output_rows` avoids computing unused LM-head rows, without changing the
  full noise block or its attention. Default draft API still returns B-1 rows.
- EOS, output budget, target/draft capacity, and failed computations are checked.
- No Torch/Transformers dependency, remote-code execution, training, weight
  modification, driver change or target precision change.

## Measurements

Early 32-token explanation screening showed why large blocks are not selected:
serial 4.39 token/s; B16/verify8 2.99; B8/verify8 3.03; B4/verify4 4.51. These
are different inference block shapes on a checkpoint trained with B16.

The final implementation was retested with B4/verify4, two alternating A/B
repeats per prompt, up to 64 tokens, after warming both generation paths.
Decode timing includes drafter work, target feature capture/verification and
rollback, but excludes model loading and prompt prefill. All generated token
IDs matched the serial target in every run.

| Public test workload | Tokens | Serial token/s | DFlash token/s | Ratio | Accepted proposals |
|---|---:|---:|---:|---:|---:|
| English sky explanation | 55 | 4.408 | 4.254 | 0.965x | 17 / 114 |
| Python Fibonacci function | 64 | 4.384 | 8.487 | 1.936x | 42 / 65 |
| Chinese explanation | 64 | 4.403 | 4.272 | 0.970x | 20 / 129 |

The code example is a real, reproducible win, **not proof that all code tasks
double in speed**. Ordinary explanation/Chinese samples were slower. A separate
block-size sweep found B2/verify2 about 4.59 / 5.64 / 4.35 token/s on these
three prompts; no setting dominates every workload. B4 is the explicit API
experiment, not an automatic or universal acceleration policy.

## Correctness evidence

- Real-checkpoint Vulkan versus CPU FP16-input reference: relative logit error
  0.000393; all three checked draft argmax positions matched.
- Real incremental draft cache versus no-cache computation: zero logit error.
- CPU suites: 37 tests covering the target verifier, n-gram, DFlash, safetensors
  validation and both server response routes. DFlash tests use an independent
  per-head FP64-attention reference and cover rejection, features, EOS, context
  tails, output-head prefixes, buffer boundaries and failed cache transactions.
- Independent review plus randomized state-machine/API checks found no blocker.
- Configuring DFlash together with either debug disable-kernel flag is rejected
  before draft-weight upload/server startup, rather than breaking an active stream.

Finite-precision batch shapes can differ near tied logits. These test outputs
match exactly, but there is no claim of bitwise identity for all possible
prompts. Only greedy decoding is implemented; exact stochastic rejection
sampling is not implemented.

## Use and limits

See `README.md` for explicit local-model configuration and the
`speculative_dflash` API field. Without configuration, no draft weights are
loaded; without the request flag, ordinary decoding remains unchanged.
The drafter adds roughly 1.1 GB of weights plus activations/cache. Block
verification can increase first-token/stream-chunk latency even where total
generation throughput improves. Real DFlash performance was measured on short
prompts; the target-only 1024-context validation is not a DFlash long-context
performance result.

Reproduce with `bench_dflash.py --draft-model <local-directory>`. Model files,
transfer/download helpers, raw benchmark JSON and experimental logs are not
committed. No private target model was uploaded.

The next high-value direction is a drafter trained for the exact 2507 target
variant, followed by broader prompt/long-context measurements. DFlash2, DSpark
and native MTP are not silently substituted or advertised as tested speedups.
