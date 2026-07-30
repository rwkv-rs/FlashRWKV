# Attempt 0004: Materialized chunk dispatch and autotune

## Status

- Classification: retained parameter local optimum within the materialized
  chunk behavior cell
- Family-dispatch result: superseded by `0005`; explicit materialized chunk is
  retained, but `algorithm="auto"` no longer selects it without a measured
  crossover
- Parent attempt: `0003-materialized-chunk`
- Nearest retained elite: `0002-fp16-recurrent` for decode/short-varlen;
  `0004` only selects materialized chunk parameters
- Canonical evidence:
  `artifacts/feature-kernel-12a9715266/20260730T230302Z-2627268/`

## Hypothesis

Family selection and kernel parameter selection solve different problems.
Decode, short sequences, and FP16-state inference should stay on the recurrent
family. Longer `fp32io16` sequences may use chunk, but chunk size, build
pipeline, and state-scan tile must be selected independently for each signed
dispatch key rather than compiled into one global default.

The complete parameter space is:

```text
chunk_size   = {16, 32, 64}
build        = {(2 warps, 1 stage), (4 warps, 1 stage), (4 warps, 2 stages)}
state_tile   = {16, 32, 64}
candidate count = 27
```

The cache key is:

```text
(SM, input dtype, state mode, fixed/packed, sequence-length bucket)
```

An absent key uses the correctness-verified
`c16-w2-s1-t64` fallback. It does not borrow a result from another
architecture or numerical mode.

## Implementation

- At this checkpoint, `algorithm="auto"` chose recurrent for `fp16` or a
  maximum sequence length of at most 16, and materialized chunk for longer
  `fp32io16` inputs. Attempt `0005` subsequently disproved that family
  crossover on the signed SM120 profiles and restored recurrent-only auto
  dispatch.
- `ChunkConfig` owns the explicit candidate contract and rejects values outside
  the enumerated space.
- The native transform builder has real 2-warp, 4-warp, and double-buffered
  4-warp variants. The state scan has real 16-, 32-, and 64-row tile variants.
- `benchmarks/autotune_chunk.py` checks each candidate against the validated
  low-level FP32-state recurrent operator before timing it. Failed candidates
  cannot enter winner selection.
- Timing covers the preallocated low-level three-kernel operator only.
  Metadata construction and allocation are outside the interval, and state
  reset is ordered before the CUDA start event.
- `flash_rwkv/chunk-tuning-v1.json` packages the 12 SM120 winners together
  with the source hash and canonical artifact pointer.

## Canonical sweep

Hardware was one `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`, GPU 0,
compute capability 12.0, Torch `2.11.0+cu130`, and CUDA runtime 13.0.
The sweep used hidden size 4096, deterministic seed `20260730`, 5 warmups, and
15 retained CUDA-event samples for every candidate.

Profiles:

- fixed: lengths 64, 256, and 1024;
- packed medium: lengths 17, 32, and 64;
- packed long: lengths 65, 128, and 256;
- packed very-long: lengths 257, 512, and 1024;
- both FP16 and BF16 token I/O with FP32 state.

All 324 candidate/profile combinations passed correctness and retained all 15
samples. The maximum observed output and final-state relative RMSE were
`2.9027780e-5` and `2.4351291e-7`, respectively, below the signed
`0.007` and `0.008` gates.

Selected configurations and median operator latency:

| dtype | layout | bucket | config | p50 ms |
| --- | --- | --- | --- | ---: |
| FP16 | fixed | medium | `c16-w4-s1-t64` | 0.089728 |
| FP16 | fixed | long | `c64-w4-s1-t32` | 0.207392 |
| FP16 | fixed | very-long | `c64-w4-s1-t32` | 0.603776 |
| FP16 | packed | medium | `c32-w4-s1-t64` | 0.110144 |
| FP16 | packed | long | `c32-w4-s1-t32` | 0.292960 |
| FP16 | packed | very-long | `c64-w4-s1-t32` | 0.989664 |
| BF16 | fixed | medium | `c16-w4-s1-t32` | 0.089664 |
| BF16 | fixed | long | `c64-w4-s1-t32` | 0.208064 |
| BF16 | fixed | very-long | `c64-w4-s1-t64` | 0.603616 |
| BF16 | packed | medium | `c32-w4-s1-t64` | 0.110336 |
| BF16 | packed | long | `c32-w4-s1-t64` | 0.293312 |
| BF16 | packed | very-long | `c64-w4-s1-t64` | 0.988704 |

The one-stage 4-warp builder won every measured key. This is a result for the
materialized algorithm on SM120, not evidence that double buffering or these
tiles are globally inferior.

## Verification

A reduced smoke for the initial materialized implementation exercised all 27
candidates with two raw samples:

```text
artifacts/feature-kernel-12a9715266/20260730T222031Z-2578143/
```

The initial exact 12-entry cache passed config and CUDA chunk regression
`70/70`:

```text
artifacts/feature-kernel-12a9715266/20260730T222406Z-2582411/
```

The initial leaf-wide reference, recurrent, chunk, dispatch, and package
regression passed `99/99`:

```text
artifacts/feature-kernel-12a9715266/20260730T222619Z-2585168/
```

After extracting the shared replay kernel and adding the factor/recompute
behavior cell, the complete 324-case sweep was rerun. The packaged cache
entries compare byte-for-byte equal to that canonical autotuner payload, and
its recorded native source-set SHA-256 is
`b7e82395c285aa3943b9d43201245437cffa4a4675e8891853b53238af451858`.
The final leaf-wide regression, including the refreshed cache and dispatch
fixtures, passed `105/105`:

```text
artifacts/feature-kernel-12a9715266/20260730T230618Z-2631184/
```

## Next decision

This attempt closes parameter selection for the current materialized behavior
cell. It does not establish materialized transforms as the canonical long
sequence algorithm. Attempt 0005 must compare its three FP32
`[num_chunks,H,64,64]` workspaces against a DPLR factor/recompute cell using
memory, profiler, and identical-boundary latency evidence.
