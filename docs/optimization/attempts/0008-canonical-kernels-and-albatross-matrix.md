# Attempt 0008: Canonical kernels and Albatross B/T matrix

## Status

- Classification: active contract and implementation attempt
- Parent attempt: `0007-unified-benchmark-and-profiler`
- Primary bottleneck class: API/measurement attribution first, algorithmic
  structure second
- Retained elite: vllm-rwkv-derived recurrent inference from attempt 0007
- Target device: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, SM120

Attempt 0007 established that the existing recurrent inference kernel is the
retained inference baseline, while the monolithic training backward is the
largest measured training bottleneck. The immediate blocker for further
comparison is nevertheless attribution: the public table mixes wrappers,
algorithms, layouts, numerical modes, and providers, and the benchmark does
not use Albatross's 21-shape B/T matrix.

## Hypothesis

1. A strict registry can make every measured row traceable to one native or
   external operator without creating aliases for reference/stateful wrappers.
2. A RWKV-LM-derived recurrent forward can produce the exact boundary and
   token auxiliaries expected by the retained reverse-scan backward.
3. A KDA-derived RWKV path can preserve independent `K1 prepare` and
   `K2 recurrence` launches while remaining mathematically equivalent to the
   canonical RWKV-7 recurrence.
4. Reusing Albatross's B/T matrix and percentile/throughput definitions while
   narrowing timing to one logical operator will expose real kernel scaling
   without model or scheduler noise.

## Planned change

- Add canonical kernel registry names and provider/maturity/layout metadata.
- Pair a new RWKV-LM recurrent forward with the existing derived backward.
- Add BF16 KDA-derived fixed and packed chunk paths with independent K1/K2.
- Measure the exact 21 Albatross B/T cases with
  `tok_s_p50 = B*T*1000/p50_ms`.

The KDA logical operator timing includes both launches. Input construction,
state reset, correctness checks, compilation, metadata, and serialization
remain outside CUDA event boundaries.

## Acceptance

- Registry contract tests pass and reject name/metadata mismatches.
- Training output, final state, and all requested gradients pass the FP32
  oracle gate.
- KDA fixed/packed output and final state pass their BF16-mode oracle gate,
  including tails and sequence isolation.
- Every available registry entry produces exactly 21 measured rows with raw
  samples and the fixed statistics schema.

## Decision

Pending implementation and GPU evidence. A failed correctness gate invalidates
the corresponding performance row; no provider ranking is retained from an
incorrect case.
