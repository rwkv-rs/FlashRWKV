# Attempt 0007: Unified benchmark and profiler closure

## Status

- Classification: retained canonical measurement baseline
- Parent attempt: `0006-fixed-length-backward`
- Leaf revision:
  `5db6568f6a42f2fd6c5c6f703a21f3bdbe4da81b`
- FLA comparison revision:
  `3adcb3c50a9e78c6ef6d173543305b1d5ef8fa4c`
- Device: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, SM120, GPU 0
- Runtime: PyTorch 2.11.0+cu130, CUDA 13.0

The benchmark and profiler runs use the public package boundary. Every timed
case first passes its exact correctness gate. Functional measurements preserve
identical inputs and initial state; in-place state-pool measurements are
reported separately and are not compared as the same operator boundary.

## Evidence

- Current SM120 autotune:
  `artifacts/feature-kernel-12a9715266/20260731T002456Z-2724186/`
- Complete fixed-length training benchmark:
  `artifacts/feature-kernel-12a9715266/20260731T002811Z-2727925/`
- Fixed-length training Nsight Systems report and SQLite:
  `artifacts/feature-kernel-12a9715266/20260731T002920Z-2729270/`
- Packed recurrent Nsight Systems report and SQLite:
  `artifacts/feature-kernel-12a9715266/20260731T003023Z-2730524/`
- Complete unified forward benchmark:
  `artifacts/feature-kernel-12a9715266/20260731T003336Z-2733786/`
- Complete package regression:
  `artifacts/feature-kernel-12a9715266/20260731T003522Z-2735698/`
- Nsight Compute permission diagnostic:
  `artifacts/feature-kernel-12a9715266/20260730T225544Z-2619041/`

The autotune run tested 27 candidates for every one of 12
architecture/dtype/layout/length keys. All 324 candidate cases passed
correctness. The committed cache is byte-for-byte identical to the artifact's
entries and records native source-set SHA-256
`1640caf1c2eb2c3fa234bb907875cd1e6c3ee2e979b7f757bd8d57c763cbc08a`.
The profiled extension SHA-256 is
`902fc47aa05caf42a83f8f8659b5f35e32f76fcb5e096c26a40e954ecc49d93b`.
These identities exclude a stale binary or stale tuning cache as an
explanation for the results.

## Unified forward result

The complete run retained 30 raw samples for each functional provider and 30
ten-iteration samples for the separate stateful boundary. All 60 functional
provider cases and all 20 stateful cases passed. FP16 p50 latency in
milliseconds is:

| Profile | Flash recurrent | Flash chunk | FLA chunk | Flash stateful |
| --- | ---: | ---: | ---: | ---: |
| `decode_b1` | 0.104080 | 0.166432 | 0.351808 | 0.066552 |
| `decode_b16` | 0.113872 | 0.189120 | 0.352144 | 0.072822 |
| `decode_b32` | 0.119856 | 0.265504 | 0.355232 | 0.079872 |
| `decode_b64` | 0.159872 | 0.623824 | 0.408368 | 0.091432 |
| `decode_b128` | 0.422048 | 1.202160 | 0.566688 | 0.190118 |
| `decode_b320` | 1.019792 | 2.812896 | 1.076656 | 0.537429 |
| `equal_chunk16_b320` | 1.185360 | 4.991296 | 1.785824 | 0.720496 |
| `ragged_1_16_b320` | 1.158672 | 3.828352 | 1.416576 | 0.630507 |
| `ragged_long_b32` | 0.287216 | 1.108976 | 0.486624 | 0.187325 |
| `ragged_skewed_b32` | 0.180400 | 0.355984 | 0.389408 | 0.106394 |

BF16 has the same ordering. Flash recurrent beats Flash chunk and FLA chunk
in all 20 functional cases. No measured recurrent/chunk crossover exists, so
`algorithm="auto"` remains recurrent. The stateful numbers are lower because
they intentionally omit the functional state copy and evolve the state pool;
they are not labelled identical-input latency.

## Fixed-length training result

The complete run retained 20 raw forward-plus-backward samples per provider.
All 12 provider cases passed output, final-state, and seven-gradient gates.

| Profile | Dtype | Flash chunk p50 (ms) | FLA chunk p50 (ms) | Flash / FLA |
| --- | --- | ---: | ---: | ---: |
| `tail17_b1` | FP16 | 0.668224 | 1.905264 | 0.351 |
| `tail17_b1` | BF16 | 0.647824 | 1.399808 | 0.463 |
| `multi_chunk64_b4` | FP16 | 1.532160 | 1.925776 | 0.796 |
| `multi_chunk64_b4` | BF16 | 1.515632 | 1.559488 | 0.972 |
| `tail257_b1` | FP16 | 3.291920 | 2.220496 | 1.483 |
| `tail257_b1` | BF16 | 3.329680 | 2.568352 | 1.296 |

Flash is faster for the short case and the four-sequence medium case. FLA is
faster for the single long sequence. This is a measured training crossover,
not a universal backend ranking.

The long case also exposed an invalid reuse of the forward-only tuning cache:
a 64-token backward checkpoint interval produced a maximum gradient relative
RMSE of 5.57 while FLA passed at 0.00309. Fixing the training checkpoint
interval to the RWKV-LM value of 16 reduced Flash's maximum error to
`1.72e-4`. The public API still accepts larger forward chunks, but autograd
does not weaken this stability policy.

## Nsight Systems and occupancy bounds

The fixed training profile captures five `tail257_b1` FP16 iterations. GPU
kernel time is:

| Kernel stage | Average (us) | GPU kernel time | Grid | Block | Registers/thread | Static shared memory |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| backward reverse scan | 2726.632 | 90.1% | `(64,1)` | 64 | 206 | 35,072 B |
| boundary scan | 155.621 | 5.1% | `(64,1)` | 64 | 128 | 24,576 B |
| transform build | 100.722 | 3.3% | `(1088,1)` | 128 | 92 | 34,048 B |
| output replay | 18.240 | 0.6% | `(1088,1)` | 64 | 128 | 1,280 B |

The SQLite device record reports 188 SMs, 65,536 registers/SM, 102,400 bytes
shared memory/SM, and 48 warps/SM. These values give a static resource upper
bound, not a measured active-occupancy counter:

- backward is limited to at most two 64-thread blocks per SM by shared memory,
  or four resident warps out of 48 (8.33%);
- the B=1 grid contains only 64 blocks, so at most 64 of 188 SMs receive a
  block in a wave; the grid-wide warp-slot upper bound is
  `64*2/(188*48) = 1.42%`;
- boundary scan has the same 64-block device-underfill;
- build and replay have 1088 blocks and are not grid-starved.

The five profiled iterations move only 1.049 MB device-to-device per
iteration, averaging 0.000909 ms. Training is therefore dominated by the
monolithic reverse scan, not allocation copies or transfer traffic. The
stable 16-token checkpoint policy moved the remaining long-sequence problem
from numerical correctness to backward parallelism. A future attempt should
split key/value tiles or otherwise increase the B=1 backward grid; this
attempt does not claim that unimplemented split.

The packed `ragged_long_b32` profile launches one recurrent kernel per
iteration with grid `(64,32)`, 64 threads, 146 registers/thread, and 1,292
bytes static shared memory. It averages 0.163263 ms. Registers allow at most
seven blocks/SM, or 14 of 48 resident warps (29.17% static upper bound), and
the 2048-block grid is large enough to populate the device. The functional
benchmark also copies 33.554 MB of FP32 state per iteration, averaging
0.017843 ms; the separate stateful boundary intentionally removes that copy.

Nsight Compute 2025.3 is installed, but the canonical host rejects hardware
counter collection with `ERR_NVGPUCTRPERM`. No achieved-occupancy, cache-hit,
or DRAM-throughput counter is invented. The report retains Nsight Systems
timelines, launch geometry, resource usage, CUDA API totals, and transfer
records, and states the counter limitation explicitly.

## Decision

- Retain recurrent as the automatic fixed/packed inference family.
- Retain stateful recurrent as a separate in-place serving boundary.
- Retain chunk autograd with fixed 16-token checkpoints.
- Retain the measured training crossover instead of claiming one provider is
  universally faster.
- Treat the B=1 reverse scan's low parallelism as the next backward
  optimization target.

