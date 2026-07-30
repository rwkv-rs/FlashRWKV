# Attempt 0003: Materialized affine chunk forward

## Status

- Classification: correctness checkpoint; performance unmeasured
- Parent attempt: `0002-fp16-recurrent`
- Nearest retained elite: `0002` recurrent decode/short-varlen
- Behavior cell: FP32 materialized chunk transforms, boundary scan, output
  replay

## Hypothesis

The recurrent family exposes only sequence/head parallelism and serializes all
tokens assigned to one CTA. For long fixed or packed sequences, a chunk family
can expose chunk/head parallelism by representing each chunk as an affine state
transform, scanning only chunk boundaries, and replaying chunks in parallel
from those boundaries.

For one token:

```text
M_t = diag(exp(log_decay_t)) + b_t a_t^T
S_t = M_t S_(t-1) + k_t v_t^T
```

For a chunk, materialize:

```text
A_new = M_t A_old
B_new = M_t B_old + k_t v_t^T
S_chunk_end = A_chunk S_chunk_start + B_chunk
```

This is intentionally the high-memory baseline. It establishes the two-stage
semantics and a measurable behavior cell before DPLR factorization,
recomputation, Tensor Core tiling, or autotuning is introduced.

## Implementation contract

1. Build FP32 `A_chunk` and `B_chunk` for chunk sizes 16, 32, and 64, with
   actual token start/end metadata so the tail is masked by construction.
2. Scan chunk transforms per sequence/head, writing the FP32 state at every
   chunk boundary and the functional final state.
3. Replay each chunk from its scanned boundary state to emit token outputs;
   chunks and heads are independent grid work.
4. Preserve canonical `[K,V]` state, explicit `log_decay`, fixed layout,
   packed-varlen layout, nonzero initial state, and slot mapping.
5. Keep allocations and Python validation outside the low-level three-kernel
   operator boundary. The first implementation is forward-only and
   `fp32io16`; unsupported gradient or FP16-state requests fail explicitly.
6. Compare every retained shape against the independent Torch recurrence and
   compare fixed and packed cases against
   `fla.ops.rwkv7.chunk_rwkv7` using the same inputs and chunk size.

## Expected bottleneck

Materializing three `[num_chunks,H,64,64]` FP32 tensors (`A`, `B`, and boundary
state) produces substantial global traffic and memory consumption. The
transform-build work is also quadratic in key size for every token. This cell
is retained only if it is correct; task 3.3 will compare it against a
DPLR-factor/recompute cell and may reject it as the canonical implementation.

## Correctness result

The controlled remote build resolved both editable packages to the current
`flash-rwkv` and `fla-rwkv` submodules, rebuilt the native extension, imported
it, and passed `uv pip check`.

The CUDA suite then passed `18/18` cases:

- fixed `T={1,15,16,17,31,32,33,65}` with chunk size 16;
- chunk sizes `{16,32,64}` at `T=65`, including masked tails;
- packed lengths `{1,16,17,33}` with scrambled slot mapping and untouched pool
  rows;
- FLA fixed comparisons for chunk sizes `{16,32,64}`;
- FLA packed-varlen comparison for lengths `{3,16,17,35}`;
- explicit rejection of FP16-state chunk, invalid chunk size, and gradient
  requests before backward exists.

Artifact:
`artifacts/feature-kernel-12a9715266/20260730T215428Z-2550158/`.
The 179.80-second first run includes FLA Triton compilation and is not a
performance measurement. The only warnings were FLA/Torch JIT deprecations.
The combined reference, recurrent, and chunk regression then passed `47/47`
in 6.12 seconds with the warmed FLA cache:
`artifacts/feature-kernel-12a9715266/20260730T220137Z-2558144/`.

## Acceptance commands

```bash
./bin/helicopter-dev prepare feature/kernel --remote \
  --components flash-rwkv,fla-rwkv,dev
./bin/helicopter-dev run feature/kernel --gpu 0 -- \
  uv run --no-sync python -m pytest \
  src/kernel/flash-rwkv/tests/test_chunk_cuda.py -q
```

No memory, benchmark, or profiler result has been recorded yet, so this
correctness checkpoint is not an optimization elite.
