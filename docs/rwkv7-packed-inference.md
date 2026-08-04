# RWKV7 packed inference contract

This document defines the external packed state contract used by vLLM for the
RWKV7 TimeMix and ChannelMix shift-state operators. The fixed-length public
operators are unchanged:

```python
infer_tmix_mix6_fp16(x[B, T, C], shift_state[B, C], mixes[6])
infer_tmix_kk_a_gate_fp16(key[B, T, C], key_scale[C], gate_bias[C], gate_delta[B, T, C], key_gate_scale[C])
infer_tmix_lnx_rkvres_xg_fp16(recurrent_output[B, T, C], receptance[B, T, C], key[B, T, C], value[B, T, C], residual_scale[C], norm_weight[C], norm_bias[C], gate[B, T, C])
infer_tmix_vres_gate_fp16(value[B, T, C], first_value[B, T, C], gate_bias[C], gate_delta[B, T, C])
infer_cmix_mix_fp16(x[B, T, C], shift_state[B, C], mix[C])
```

The two new packed operators are:

```python
infer_tmix_mix6_fp16_varlen(
    x[total_tokens, C],
    state_pool[state_pool_size, C],
    state_indices[B],
    cu_seqlens[B + 1],
    mixes[6],
    *,
    token_batch_indices=None,
)
infer_cmix_mix_fp16_varlen(
    x[total_tokens, C],
    state_pool[state_pool_size, C],
    state_indices[B],
    cu_seqlens[B + 1],
    mix[C],
    *,
    token_batch_indices=None,
)
```

All primary and metadata tensors are CUDA tensors. The primary and state
tensors are contiguous `float16`; index tensors are contiguous `int32`; `C`
is positive and even. `x` is packed in sequence order and
`cu_seqlens[0] == 0`, `cu_seqlens[-1] == total_tokens`, with strictly positive
sequence lengths. `state_indices[b]` selects the state row belonging to the
sequence represented by `[cu_seqlens[b], cu_seqlens[b + 1])`. A single call
must not contain duplicate state rows.

For each sequence, the first token reads its predecessor from
`state_pool[state_indices[b]]`. Every later token reads the previous packed
token in that same sequence. The final input row of every sequence is copied
back to the selected state row. The output is a new tensor; `state_pool` is
updated in place. The native implementation uses a compute launch followed by
a state-writeback launch, so state writeback cannot race with first-token
reads from another CTA.

`token_batch_indices[total_tokens]` is optional. If it is omitted, the native
kernel locates a token's sequence by binary-searching `cu_seqlens`. If the
caller already has a request id for every packed token, it may pass that
`int32` map; vLLM's existing `req_id` array is the intended producer. The map
must satisfy `cu_seqlens[b] <= token < cu_seqlens[b + 1]` whenever its value is
`b`. It is an execution hint, not a new state or slot API.

The scheduler owns semantic metadata validation (bounds, monotonicity,
non-empty sequences, unique state rows, and request-id consistency) before
dispatch. `fla-rwkv` exposes the corresponding recurrent metadata ticket for
the WKV call; the same validated `cu_seqlens/state_indices` pair must be used
for these two shift-state calls. The packed public API deliberately does not
expose a `slot`-named function or a slot-specific implementation.

The provider is registered as:

| public identity | native operator |
| --- | --- |
| `flash_rwkv/infer_tmix_mix6_fp16_varlen_forward` | `rwkv7_fast_ops_fp16::tmix_mix6_varlen` |
| `flash_rwkv/infer_cmix_mix_fp16_varlen_forward` | `rwkv7_fast_ops_fp16::cmix_mix_varlen` |

`fla-rwkv` pins this repository by the exact Git revision in
`fla/ops/rwkv7/backends/flash_rwkv.py`. vLLM should import the public wrappers
through `fla.ops.rwkv7.flash_rwkv`, pass its packed hidden rows directly, map
`query_start_loc` to `cu_seqlens`, map its state-pool row selection to
`state_indices`, and pass its existing per-token `req_id` as
`token_batch_indices`. It should not add another local CUDA/C++ implementation
or invent a slot-specific public API.
