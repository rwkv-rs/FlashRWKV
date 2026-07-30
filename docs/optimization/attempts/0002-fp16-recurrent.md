# Attempt 0002: FP16-state recurrent and short-sequence pipeline

## Status

- Classification: retained local optimum for recurrent decode/short-varlen
- Parent attempt: `0001-reference-contract`
- Nearest retained elite: this attempt
- Behavior cell: FP16 state, packed one-CTA-per-sequence/head recurrence,
  vectorized state movement, pipelined token input

## Hypothesis

For decode and short ragged sequences the recurrence is not long enough for a
chunk transform to repay its setup cost. The primary bottleneck changes from
long-sequence algorithmic serialization to state data movement and launch/
pipeline overhead. A distinct FP16-state family can halve state traffic and use
packed half operations while retaining the same public `log_decay` and
canonical `[K,V]` contract.

The implementation will adapt only the generic half2, swizzled/coalesced state
movement, and asynchronous token-buffer ideas from
`vllm-rwkv@6d683f9e49a2997e405c47edc147872c8609513b`
`rwkv7_wkv_fp16_v2.cu`. It will not import vLLM's `w0`, elapsed-token stochastic
rounding, `[V,K]` state interpretation, or model-fused decay transform.

## Implementation contract

1. Add a separate `recurrent_fp16` CUDA entry point and dispatch it only for
   `mode="fp16"`; never alias the FP32-state kernel.
2. Keep canonical state `[slots,H,K,V]` but store it as FP16. Load/store across
   the V dimension coalescently and vectorize where alignment permits.
3. Stage two token buffers so the next token's `r,log_decay,k,v,a,b` loads can
   overlap the current recurrence without changing sequence boundaries.
4. Preserve fixed, packed, functional slot-pool, and explicit in-place
   stateful surfaces.
5. Compare the same cases against the FP32 oracle. Report FP16 output/state
   drift separately; do not weaken the `fp32io16` correctness gate.

## Measurement boundary

- Hardware: `rwkv-sha-pro6000x8`, GPU 0.
- Profiles: decode batches `1,16,32,64,128,320`; equal chunk 16 at batch 320;
  ragged `1..16` at batch 320; long ragged at batch 32; and `128 + 31*1`
  skew.
- Timing kinds: functional identical-input and stateful steady-state are
  separate.
- Warmup, CUDA events, explicit synchronization, raw samples, source
  fingerprint, mode, dtype, shape, and full sequence lengths are mandatory.
- This attempt can become an elite only after both correctness and the
  recurrent measurement contract pass. Compilation alone is invalid.

## Correctness and tolerance evidence

The first correctness-only pass used the complete profile set with hidden size
4096 (64 heads) and source-set SHA-256
`28f1e3718d0c84582632669a23cd1fc51443d47e911493eb19ead8d5e8688843`.
Its largest observed FP16 relative-RMSE was:

- output: `0.0011854387121275067`;
- selected final-state rows: `0.0015057249693199992`.

Both maxima came from or were bounded by `equal_chunk16_b320`. The versioned
fixture therefore gates FP16 output and state at `0.003`, about twice the worst
observation, while the independent FP32-state gate remains `0.002`. The
tightened CUDA suite passed `16/16` cases on the canonical GPU.

Correctness artifact:
`artifacts/feature-kernel-12a9715266/20260730T213632Z-2491522/`.
The earlier report-only calibration artifact is
`artifacts/feature-kernel-12a9715266/20260730T213436Z-2487725/`.

## Measurement result

The formal run used 20 warmups, 30 raw samples, and 20 launches per stateful
sample. All 20 mode/profile cases passed their exact output/final-state gate,
preserved untouched slots, remained finite, and produced valid measurements.
P50 operator latency in milliseconds was:

| Profile | FP32 functional | FP32 stateful | FP16 functional | FP16 stateful |
|---|---:|---:|---:|---:|
| `decode_b1` | 0.011328 | 0.004245 | 0.011024 | 0.004108 |
| `decode_b16` | 0.011312 | 0.008376 | 0.010880 | 0.006274 |
| `decode_b32` | 0.013312 | 0.012470 | 0.011440 | 0.008409 |
| `decode_b64` | 0.022528 | 0.020674 | 0.015360 | 0.013370 |
| `decode_b128` | 0.098304 | 0.152320 | 0.028672 | 0.023753 |
| `decode_b320` | 0.378896 | 0.470809 | 0.149504 | 0.239312 |
| `equal_chunk16_b320` | 0.598016 | 0.676452 | 0.391168 | 0.463525 |
| `ragged_chunk16_b320` | 0.499712 | 0.562568 | 0.278528 | 0.334472 |
| `ragged_long_b32` | 0.172032 | 0.143826 | 0.104784 | 0.094154 |
| `ragged_skew_b32` | 0.071696 | 0.069872 | 0.054528 | 0.051422 |

Formal artifact:
`artifacts/feature-kernel-12a9715266/20260730T213752Z-2497166/`.
Its JSON retains every sequence length, error, and raw sample; the extension
SHA-256 is
`106dde13c2c03cc3270c54714d725a883fe1b38487cfc80b0356f8cd602d27a5`.
These are device-local recurrent baselines, not cross-hardware optimality
claims. Functional and stateful numbers have deliberately different cache and
state histories and must not be compared as if they were one timing kind.

## Acceptance commands

```bash
./bin/helicopter-dev prepare feature/kernel --remote \
  --components flash-rwkv,dev
./bin/helicopter-dev run feature/kernel --gpu 0 -- \
  uv run --no-sync python -m pytest \
  src/kernel/flash-rwkv/tests/test_recurrent_cuda.py -q
./bin/helicopter-dev run feature/kernel --gpu 0 -- \
  uv run --no-sync python \
  src/kernel/flash-rwkv/benchmarks/benchmark_recurrent.py
```
