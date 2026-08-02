# Attempt 0001: RWKV-7 reference contract

## Status

- Classification: progress checkpoint
- Product branch: `feature/kernel`
- FlashRWKV branch: `main`
- Parent attempt: none
- Nearest retained elite: none
- Baseline FlashRWKV revision: `54c863de7645a0172c5f5a8606d52c0d9ab9abae`
- OpenSpec planning revision: `89b2590038eb87985c47d5823b9d01780c5cd6cf`

This attempt freezes the mathematical, correctness, and measurement boundary
before the first RWKV implementation replaces the disabled KDA data plane. It is
not CUDA performance evidence.

## Provenance

| Role | Revision or source | Files used |
| --- | --- | --- |
| Training semantics and backward | `rwkv-lm@46691027da83376414147f878f2232e1a3d2f406` | `cuda/wkv7_cuda.cu`, `cuda/wkv7_cuda_fp32.cu`, `cuda/wkv7_op.cpp`, `cuda/wkv7_op_fp32.cpp` |
| Chunk and autograd baseline | `fla-rwkv@3adcb3c50a9e78c6ef6d173543305b1d5ef8fa4c` | `fla/ops/rwkv7/chunk.py`, `fla/ops/rwkv7/fused_recurrent.py`, `tests/ops/test_rwkv7.py` |
| Packed-varlen serving baseline | `vllm-rwkv@6d683f9e49a2997e405c47edc147872c8609513b` | `csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp16_v2.{cpp,cu}`, `csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp32_v2.{cpp,cu}`, `tests/kernels/test_rwkv7_wkv_canonical.py`, `benchmarks/kernels/benchmark_rwkv7_wkv.py` |
| SM90 pipeline ideas | imported FlashKDA tree at `FlashRWKV@54c863d` | legacy architecture-placeholder CUDA tree, `docs/20260420-flashkda-v1-deep-dive.md` |

Albatross/vLLM-derived files are Apache-2.0 material and must retain SPDX and
source attribution if adapted. FlashKDA and FLA material is MIT. No KDA
recurrence, normalization, gate parameterization, matrix inverse, or workspace
contract is part of the RWKV operator.

## Operator contract

The core accepts post-transform
`r, log_decay, k, v, a, b` and canonical state `S[K,V]`:

```text
S_t = diag(exp(log_decay_t)) S_(t-1)
    + b_t (a_t^T S_(t-1))
    + k_t v_t^T
o_t = scale * r_t^T S_t
```

The training adapter alone converts RWKV-LM decay logits:

```text
log_decay = -exp(-0.5) * sigmoid(decay_logits)
```

FLA already consumes log-decay. Albatross/vLLM and RWKV-LM store the transpose
`[V,K]`; any reuse must cross an explicit adapter and a non-symmetric directional
fixture. The core never accepts vLLM's fused `w0` or elapsed-token phase.

## Correctness boundary

- Oracle: pure PyTorch recurrence with FP32 math and no extension import.
- Signed head shape: `K = V = 64`; the reference remains dimension-generic so
  small directional fixtures can expose a state transpose.
- Layouts: fixed `[B,T,H,D]`; packed `[1,total_tokens,H,D]` with strictly
  increasing `cu_seqlens`.
- State: fixed `[B,H,K,V]`, packed `[N,H,K,V]`, or explicit state pool plus
  unique in-range `state_indices`.
- Required fixtures: `T={1,15,16,17,63,64,65}`, zero and nonzero state,
  packed-versus-separate, scrambled slot mapping, untouched pool rows, and a
  non-symmetric `K != V` orientation case at the reference layer.
- Backward uses one shared upstream gradient for Torch, FlashRWKV, and FLA;
  output-only and output-plus-final-state losses are distinct cases.
- `fp32io16` means FP32 state and accumulation with FP16/BF16 token I/O.
  `fp16` drift is measured separately and cannot weaken the correctness mode.

## Measurement contract

- Canonical GPU: `rwkv-sha-pro6000x8`, GPU 0, through `helicopter-dev`.
- Runners: package correctness runner, package JSON operator benchmark, then
  Nsight on the exact already-built binary and fixture.
- Inputs and correctness gates are created before timing. Compilation, autotune,
  logging, serialization, and fixture cloning are outside the timed interval.
- Each result records leaf/parent revisions, binary provenance, CUDA/PyTorch,
  GPU, dtype, mode, shape, complete `seq_lens`, kernel family/config, warmup,
  synchronization policy, raw latency samples, error summary, and baseline
  revision.
- Functional measurement starts from identical initial state for every sample.
  Stateful serving measurement intentionally evolves one state pool in place.
  They are separate result kinds and are never compared as the same latency.
- Profiles retained from the serving baseline: decode batches
  `1,16,32,64,128,320`; equal chunk 16 at batch 320; ragged 1 through 16 at
  batch 320; long ragged at batch 32; and `128 + 31*1` skew.
- No speedup is valid unless output and final state pass for that exact case.

## Bottleneck and behavior cell

- Primary bottleneck: algorithmic structure plus parallelism coordination.
  A token recurrence leaves one CTA per sequence/head serial over long context.
- Secondary bottleneck for decode: state data movement and launch overhead.
- First structural child after the independent recurrent baseline:
  token/chunk-parallel transform build plus sequence/head state scan.
- Memory access pattern: canonical `[K,V]` state with explicit serving transpose
  bridge; recurrent state staged once per sequence/head when capacity permits.
- Tunable space is deferred until the two-stage algorithm is correct. Its first
  complete sweep is chunk size `{16,32,64}` crossed with explicit warp, stage,
  and state-tile candidates, keyed by architecture, dtype, mode, layout, and
  sequence-length bucket.
- The KDA six-workspace layout (7424 bytes per chunk/head) is not a starting
  point. Materialized transforms and DPLR-factor/recompute are separate behavior
  cells with independent memory and profiler evidence.

## First implementation contract

1. Replace the disabled `flash_kda` Python surface with `flash_rwkv`.
2. Land the dimension-generic FP32 Torch oracle, differentiable decay adapter,
   fixed/packed validation, and versioned tolerances.
3. Pass CPU tests for recurrence, gradients, boundaries, state orientation,
   metadata rejection, and slot isolation.
4. Only then add an independent CUDA recurrent family; do not call the future
   chunk implementation from the baseline.

## Commands and result

Planned first correctness command:

```bash
./bin/helicopter-dev run feature/kernel -- \
  env PYTHONPATH=src/kernel/flash-rwkv \
  uv run --no-sync python -m pytest \
  src/kernel/flash-rwkv/tests/test_reference.py -q
```

Result on 2026-07-30: `13 passed in 0.73s`. The first sandboxed
`prepare feature/kernel --local` could not open the configured proxy tunnel;
the identical controlled prepare succeeded with network permission and built
the checkout-local environments. This result covers only the PyTorch oracle,
validation, and adapter contract; it is not CUDA correctness or performance
evidence.

After adding the independent dependency group and editable source, the
installation contract was rerun as:

```bash
./bin/helicopter-dev prepare feature/kernel --local \
  --components flash-rwkv,dev
./bin/helicopter-dev run feature/kernel -- \
  uv run --no-sync python -m pytest \
  src/kernel/flash-rwkv/tests/test_reference.py -q
```

The prepare log built and installed
`flash-rwkv==0.1.0` from this checkout's
`src/kernel/flash-rwkv`, and the installed-package test passed
`13 passed in 0.71s` without a `PYTHONPATH` override.

## CUDA recurrent checkpoint

The independent FP32-state recurrent family was then built and tested on
`rwkv-sha-pro6000x8`, GPU 0:

```bash
./bin/helicopter-dev prepare feature/kernel --remote \
  --components flash-rwkv,dev
./bin/helicopter-dev run feature/kernel --gpu 0 -- \
  uv run --no-sync python -m pytest \
  src/kernel/flash-rwkv/tests/test_recurrent_cuda.py -q
```

The first remote prepare stopped before compilation because the existing
remote `.venv` had no Python executable. `env status` confirmed no environment
process, an absent `.venv/bin/python`, 8 RTX PRO 6000 GPUs, and
`/usr/local/cuda/bin/nvcc`. `repair feature/kernel` rebuilt the managed venv;
the original prepare command then succeeded unchanged.

The source-fingerprinted editable build loaded `flash_rwkv._C`. GPU correctness
passed `8 passed in 1.78s` for fixed `T={1,15,16,17,65}`, nonzero FP32 state,
packed ragged sequences, scrambled slot mapping, untouched pool rows, and the
explicit in-place stateful API. Output and final state were gated by the
versioned `fp32io16` relative-RMSE threshold. This is correctness evidence, not
latency evidence.

Decision: retain this attempt as the FP32-state recurrent progress checkpoint
and correctness baseline for the separate FP16-state behavior cell.
