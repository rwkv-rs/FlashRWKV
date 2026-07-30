# Attempt 0005: Factor/recompute chunk crossover

## Status

- Classification: retained behavior cell, rejected as an automatic dispatch
  winner
- Parent attempt: `0004-materialized-chunk-autotune`
- Canonical family: `0002-fp16-recurrent`
- Strategy benchmark:
  `artifacts/feature-kernel-12a9715266/20260730T230058Z-2624869/`
- Nsight Systems evidence:
  `artifacts/feature-kernel-12a9715266/20260730T225222Z-2615101/` and
  `artifacts/feature-kernel-12a9715266/20260730T225438Z-2617642/`

## Hypothesis

The materialized chunk cell stores three FP32
`[num_chunks, num_heads, 64, 64]` workspaces and constructs a full affine
transform for every chunk. A factor/recompute cell can instead preserve the
token DPLR factors `diag(decay) + b a^T`, serially recompute only the chunk
boundary states, and replay each chunk in parallel from those boundaries.

This should reduce workspace and remove the expensive transform build. The
decision criterion is not whether factor/recompute beats materialized alone:
it must also beat the existing recurrent kernel at an identical output and
final-state boundary before becoming an automatic dispatch target.

## Implementation

- `recompute_chunk_fp32` stores one FP32 boundary-state workspace, scans the
  DPLR factors to construct those boundaries and the final state, then calls
  the shared output replay kernel.
- Materialized and factor/recompute share the same replay implementation, so
  the behavior-cell comparison isolates boundary construction and workspace
  policy.
- The benchmark preallocates every workspace, resets state before the CUDA
  start event, randomizes the interleaved strategy order, and retains every
  raw sample.
- Every chunk size in `{16, 32, 64}` must pass the FP32-state recurrent
  correctness gate before it can participate in factor/recompute selection.
- Both chunk cells remain available through explicit low-level calls. Public
  `algorithm="auto"` stays on recurrent until a measured crossover exists.

## Correctness and latency

The signed benchmark covered:

- fixed lengths 64, 256, and 1024;
- packed medium lengths 17, 32, and 64;
- packed long lengths 65, 128, and 256;
- packed very-long lengths 257, 512, and 1024;
- FP16 and BF16 token I/O, FP32 state, hidden size 4096;
- 30 interleaved retained samples per strategy.

All 36 factor/recompute candidates passed correctness and retained all seven
tuning samples. All 36 selected strategy results passed correctness and
retained all 30 timing samples. The maximum output and final-state relative
RMSE were `2.8632632e-5` and `2.3907549e-7`, below the signed `0.007` and
`0.008` limits.

Median preallocated operator latency and workspace:

| dtype | profile | materialized config | materialized ms / MiB | factor config | factor ms / MiB | recurrent ms |
| --- | --- | --- | ---: | --- | ---: | ---: |
| FP16 | fixed medium | `c16-w4-s1-t32` | 0.089808 / 12 | `c16` | 0.046384 / 4 | 0.037872 |
| BF16 | fixed medium | `c16-w4-s1-t32` | 0.089920 / 12 | `c16` | 0.046480 / 4 | 0.037616 |
| FP16 | fixed long | `c64-w4-s1-t32` | 0.208528 / 12 | `c32` | 0.140912 / 8 | 0.128208 |
| BF16 | fixed long | `c64-w4-s1-t32` | 0.208320 / 12 | `c32` | 0.140928 / 8 | 0.127840 |
| FP16 | fixed very-long | `c64-w4-s1-t64` | 0.608240 / 48 | `c64` | 0.514256 / 16 | 0.493312 |
| BF16 | fixed very-long | `c64-w4-s1-t64` | 0.607856 / 48 | `c64` | 0.514272 / 16 | 0.490784 |
| FP16 | packed medium | `c32-w4-s1-t32` | 0.110352 / 12 | `c16` | 0.050464 / 8 | 0.037872 |
| BF16 | packed medium | `c32-w4-s1-t64` | 0.110336 / 12 | `c16` | 0.049440 / 8 | 0.037632 |
| FP16 | packed long | `c32-w4-s1-t64` | 0.296800 / 45 | `c16` | 0.152608 / 29 | 0.129872 |
| BF16 | packed long | `c32-w4-s1-t64` | 0.295152 / 45 | `c16` | 0.151824 / 29 | 0.130080 |
| FP16 | packed very-long | `c64-w4-s1-t32` | 0.989072 / 87 | `c64` | 0.599616 / 29 | 0.559344 |
| BF16 | packed very-long | `c64-w4-s1-t64` | 0.992384 / 87 | `c64` | 0.648080 / 29 | 0.541808 |

Factor/recompute beat materialized and used less workspace in all 12 cells,
but direct recurrent still beat factor/recompute in all 12. The benchmark
SHA-256 is
`b16c3fac5603ae2f31692f14384f1ab682786f2c6b5fe35cb167f913db8027f4`;
the recorded source-set SHA-256 is
`52cf99cdcf8231018188206e4b09f287b4257ff32fc4b82fbe6702d2389ae68f`.

## Profiler evidence

The representative profiler workload is packed very-long FP16 with lengths
257, 512, and 1024, hidden size 4096, on one RTX PRO 6000 Blackwell GPU.
Each report captures five iterations inside a CUDA Profiler API range.

Materialized chunk:

| kernel | launches | total ms | average ms | share |
| --- | ---: | ---: | ---: | ---: |
| `build_transforms_kernel` | 5 | 3.364117 | 0.672823 | 68.6% |
| `emit_outputs_kernel` | 5 | 0.792118 | 0.158424 | 16.2% |
| `scan_boundaries_kernel` | 5 | 0.748214 | 0.149643 | 15.3% |

The transform builder launches 1856 blocks of 128 threads, uses 92 registers
per thread and 34,048 bytes of static shared memory per block.

Factor/recompute:

| kernel | launches | total ms | average ms | share |
| --- | ---: | ---: | ---: | ---: |
| `scan_factor_boundaries_kernel` | 5 | 2.518018 | 0.503604 | 79.2% |
| `emit_outputs_kernel` | 5 | 0.659542 | 0.131908 | 20.8% |

The factor scan removes the materialized transform builder, but it is itself a
serial recurrence over each sequence/head and is followed by a second output
replay. This explains why the cell improves materialized chunk yet still loses
to direct recurrent.

Nsight Compute was attempted on the dominant materialized kernel, but the
remote driver rejected hardware counters with `ERR_NVGPUCTRPERM`. No NCU
report is treated as evidence. The two valid Nsight Systems SQLite reports and
their raw kernel records are the profiler source of truth.

The final leaf-wide reference, recurrent, chunk, recompute, cache, dispatch,
and package regression passed `105/105`:

```text
artifacts/feature-kernel-12a9715266/20260730T230618Z-2631184/
```

## Decision

Retain factor/recompute as a correct, lower-workspace comparison cell and local
optimum over materialized chunk. Reject both current chunk cells as automatic
dispatch winners. `algorithm="auto"` therefore selects recurrent for all
currently supported sequence lengths and state modes; explicit
`algorithm="chunk"` remains available for experimentation and future
structural work.
