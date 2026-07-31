# Attempt 0006: Fixed-length chunk backward

## Status

- Classification: canonical fixed-length training backward
- Parent attempt: `0005-factor-recompute-cross-over`
- Training source:
  `BlinkDL/RWKV-LM@952102498e9ed367ea0a59ee64106916d474d30f`
- Independent comparison source:
  `fla-org/flash-linear-attention@3adcb3c50a9e78c6ef6d173543305b1d5ef8fa4c`
- Targeted correctness evidence:
  `artifacts/feature-kernel-12a9715266/20260731T002238Z-2721314/`
- Long-sequence benchmark evidence:
  `artifacts/feature-kernel-12a9715266/20260731T002330Z-2722425/`
- Leaf-wide regression evidence:
  `artifacts/feature-kernel-12a9715266/20260731T003522Z-2735698/`

## Contract

The training path keeps the public canonical recurrence and FP32
`state[K,V]` layout from `0001-reference-contract`. It accepts fixed
`[B,T,H,64]` FP16, BF16, or FP32 token tensors and an optional FP32 initial
state. Packed inputs remain forward-only until a packed backward has its own
correctness and measurement contract.

The backward returns gradients for the six token inputs
`r,log_decay,k,v,a,b` and, when requested, the initial state. The
`rwkv7_from_decay_logits` adapter remains outside the custom autograd boundary,
so PyTorch applies the exact chain rule from canonical `dlog_decay` to the
RWKV-LM decay logits.

The reverse reconstruction divides by `exp(log_decay)`, matching the
RWKV-LM training kernel's checkpoint/backstepping method. The intended
training domain is therefore the RWKV-LM adapter
`log_decay = -exp(-0.5) * sigmoid(decay_logits)`, whose decay is finite and
strictly nonzero in the supported floating-point formats.

Training always checkpoints every 16 tokens, independently of the
forward-only tuning cache or an explicitly requested larger forward chunk.
The build-warp, pipeline-stage, and state-tile choices remain unchanged.
This follows the RWKV-LM reconstruction interval and bounds numerical
amplification while backstepping through `exp(log_decay)`.

## Checkpoint policy

The forward allocates materialized transform and bias workspaces only for the
forward launch. The autograd context persists:

- the original six token tensors required by the recurrence;
- one FP32 state at each chunk boundary;
- the per-token FP32 contraction `a_t^T S_(t-1)`;
- the FP32 final state and compact chunk metadata.

It does not persist the full transform or bias workspaces. The reverse kernel
launches one CTA per `(batch, head)`, walks chunks and tokens in reverse, and
reconstructs the previous state from the next state, the DPLR update factors,
and the saved contraction. Masked tail chunks use their exact token endpoints.

PyTorch's `needs_input_grad` controls allocation of each output gradient.
Unrequested token gradients and `dS0` are passed to the native kernel as null
pointers rather than backed by placeholder tensors. Upstream gradients for the
output and final state are independently optional.

## Correctness

The targeted suite compares the same upstream gradients against both the
independent FP32 Torch recurrence and FLA `chunk_rwkv7`. It covers:

- BF16, length 17, output-only loss and a masked tail chunk;
- FP16, length 33, output plus final-state loss and two tail boundaries;
- FP16, length 257, output plus final-state loss, with an explicitly requested
  64-token forward chunk reduced to 16-token training checkpoints;
- nonzero FP32 initial state with `dS0`;
- batch size 2 with only `r`, `v`, and `b` requesting gradients;
- all six canonical token gradients, final state, and output.

Every FlashRWKV and FLA result passed the signed relative-RMSE gates in
`tests/fixtures/tolerances-v1.json`. The targeted config/autograd run passed
`29/29`, including the length-257 regression; the subsequent package
regression passed `110/110`.

## Decision

Retain this path as the canonical fixed-length training implementation for
`algorithm="chunk"`. This attempt establishes numerical correctness and
checkpoint behavior only. It makes no training-speed claim; fixed-length
forward/backward timing and profiler evidence belong to the unified benchmark
and profiler work in the next attempt.
