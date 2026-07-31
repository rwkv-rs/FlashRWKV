# FlashRWKV

FlashRWKV provides source-attributed RWKV-7 CUDA kernels behind one canonical
recurrence contract. Kernel identity is the pair `(provider, name)`; origin,
maturity, layout, autograd support, state behavior, and internal stages are
recorded separately instead of being overloaded into an `algorithm` label.

The implementation targets head size 64 and canonical state layout
`[N,H,K,V]`.

## Kernel registry

Canonical names follow:

```text
{pretrain|infer}_{recurrent|chunk}_{numerical-mode}_{forward|backward}[_varlen]
```

| Provider | Canonical name | Status | Layout | Autograd | State | Stages |
| --- | --- | --- | --- | --- | --- | --- |
| `rwkv-lm` | `pretrain_recurrent_fp32io16_forward` | stable | fixed | yes | functional FP32 | forward recurrence |
| `rwkv-lm` | `pretrain_recurrent_fp32io16_backward` | stable | fixed | yes | functional FP32 | backward recurrence |
| `vllm-rwkv` | `infer_recurrent_fp32io16_forward_varlen` | stable | packed | no | functional FP32 | forward recurrence |
| `vllm-rwkv` | `infer_recurrent_fp16_forward_varlen` | stable | packed | no | functional FP16 | forward recurrence |
| `flashkda-derived` | `infer_chunk_bf16_forward` | experimental | fixed | no | functional BF16 | K1 prepare → K2 recurrence |
| `flashkda-derived` | `infer_chunk_bf16_forward_varlen` | experimental | packed | no | functional BF16 | K1 prepare → K2 recurrence |
| `fla` | `pretrain_chunk_fp32io16_forward` | external | fixed | yes | functional FP32 | FLA chunk forward |
| `fla` | `pretrain_chunk_fp32io16_backward` | external | fixed | yes | functional FP32 | FLA chunk backward |
| `fla` | `infer_recurrent_fp32io16_forward_varlen` | external | packed | no | functional FP32 | FLA fused recurrent forward |

`flash_rwkv.kernel_specs()` returns this immutable registry.
`flash_rwkv.get_kernel_spec(name, provider=...)` resolves one identity and
rejects ambiguous names. The FLA and vllm-rwkv providers intentionally share
`infer_recurrent_fp32io16_forward_varlen`; the provider is therefore required
for that lookup.

The FP32 oracle `rwkv7_reference` is a correctness implementation, not a
kernel. `rwkv7_recurrent_stateful` is an in-place state-pool wrapper over the
vllm-rwkv recurrent kernels, not an additional kernel identity. The older
`rwkv7(..., algorithm=...)` entrypoint remains as a compatibility layer.

## Recurrence and layouts

For every token, all providers evaluate:

```text
S_t = diag(exp(log_decay_t)) S_(t-1)
    + b_t (a_t^T S_(t-1))
    + k_t v_t^T
y_t = scale * r_t^T S_t
```

Fixed inputs use `[B,T,H,64]`. Packed-varlen inputs use
`[1,total_tokens,H,64]` and strictly increasing `cu_seqlens[B+1]`.

### Stable pretraining

The public autograd operation couples the two separately named native
forward/backward kernels:

```python
import torch
from flash_rwkv import pretrain_recurrent_fp32io16_forward

B, T, H, D = 2, 128, 32, 64
shape = (B, T, H, D)
r, log_decay, k, v, a, b = (
    torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for _ in range(6)
)
log_decay = (-0.5 * torch.sigmoid(log_decay)).requires_grad_(True)
initial_state = torch.zeros(
    B, H, D, D, device="cuda", dtype=torch.float32, requires_grad=True
)

output, final_state = pretrain_recurrent_fp32io16_forward(
    r,
    log_decay,
    k,
    v,
    a,
    b,
    initial_state=initial_state,
    output_final_state=True,
)
```

The forward stores FP32 state at each 16-token checkpoint and the token
auxiliary values needed by the reverse recurrence. The backward consumes that
context and returns gradients for all six token inputs and the initial state.

### Stable packed inference

```python
from flash_rwkv import infer_recurrent_fp32io16_forward_varlen

output, final_state = infer_recurrent_fp32io16_forward_varlen(
    r_packed,
    log_decay_packed,
    k_packed,
    v_packed,
    a_packed,
    b_packed,
    initial_state=initial_state,
    cu_seqlens=cu_seqlens,
)
```

Use `infer_recurrent_fp16_forward_varlen` only when both token tensors and
global state are FP16.

### Experimental KDA-derived two-stage inference

`infer_chunk_bf16_forward` and
`infer_chunk_bf16_forward_varlen` are complete logical operators composed of
two visible native launches:

1. K1 prepare builds per-chunk `P/Q` transforms and per-token output terms.
2. K2 recurrence propagates the boundary state across chunks and produces
   output/final state.

Inputs, outputs, and global state are BF16. K1 workspaces and K2 local
accumulators are FP32. The path implements RWKV-7 recurrence, not the original
KDA attention operator.

### External FLA provider

The adapters in `flash_rwkv.providers.fla` map the same tensor/state/scale
contract to FLA's `chunk_rwkv7` and `fused_recurrent_rwkv7`. FLA source is not
copied into this repository. The fixed training adapter uses chunk size 16
with `safe_gate=True`; benchmark inputs keep `log_decay` in FLA's documented
safe range.

## Installation

A CUDA-enabled PyTorch environment, matching CUDA toolkit, Ninja, and a C++
toolchain are required.

```bash
git clone https://github.com/rwkv-rs/FlashRWKV.git
cd FlashRWKV
python -m pip install -v --no-build-isolation .
```

The Helicopter product checkout installs FlashRWKV and FLA through their
dedicated dependency groups.

## Correctness and benchmark

Run the regression suite:

```bash
pytest -q
```

Run all nine provider/name identities:

```bash
python benchmarks/benchmark_rwkv7.py
```

The canonical runner uses the 21-case Albatross B/T matrix:

```text
1x1 1x2 1x4 1x8 1x16 1x32 1x64 1x128 1x256
2x1 4x1 8x1 16x1 32x1 64x1 128x1 256x1
2x2 4x4 8x8 16x16
```

Every per-kernel CSV has exactly:

```text
label,B,T,iters,p10_ms,p50_ms,p90_ms,tok_s_p50
```

and:

```text
tok_s_p50 = B * T * 1000 / p50_ms
```

Each shape passes output/final-state correctness first; backward entries also
pass all-input gradient correctness. Invalid cases are retained in JSON but
do not produce a performance row.

CUDA events enclose one named logical operator. Input generation,
compilation/autotune, correctness, packed metadata construction, state
clone/reset, logging, and serialization are outside the interval. The
KDA-derived interval deliberately includes consecutive K1 and K2 launches.
Raw samples, errors, precision, workspace, configuration, source revision,
runtime, and hardware are retained in JSON; a Markdown report and one exact
eight-field CSV per identity are written alongside it.

The default benchmark uses hidden size 4096, BF16 token I/O for `fp32io16`,
five warmups, and 30 retained samples. Numerical modes are reported
separately and are not presented as like-for-like performance rankings.

Additional development tools remain available:

- `benchmarks/benchmark_training.py`: the same canonical runner restricted to
  the four pretraining identities;
- `benchmarks/autotune_chunk.py`: legacy materialized-chunk configuration
  exploration;
- `benchmarks/compare_chunk_strategies.py`: legacy development comparison;
- `benchmarks/benchmark_recurrent.py`: detailed recurrent/stateful
  characterization.

## Provenance and licenses

The RWKV-LM pretraining forward/backward pair is adapted from
[BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) at
`952102498e9ed367ea0a59ee64106916d474d30f`.

The stable inference kernels are adapted from
[rwkv-rs/vllm-rwkv](https://github.com/rwkv-rs/vllm-rwkv) at
`6d683f9e49a2997e405c47edc147872c8609513b`; its FP16 lineage includes
[BlinkDL/Albatross](https://github.com/BlinkDL/Albatross)
`faster3a_2607` at `63c53f4abf2cd891dd3a18c8f44f5b2cccc8c64b`.

The explicit K1/K2 separation follows
[MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) at
`1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`, while the implemented algebra is
the canonical RWKV-7 recurrence.

[FLA](https://github.com/fla-org/flash-linear-attention) at
`3adcb3c50a9e78c6ef6d173543305b1d5ef8fa4c` remains an external dependency.

See [NOTICE](NOTICE), [LICENSE](LICENSE), and
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) for file-level boundaries.
