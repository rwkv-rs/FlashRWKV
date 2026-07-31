# FlashRWKV

FlashRWKV provides a numerical reference and CUDA kernels for the canonical
RWKV-7 recurrence. It exposes one operator contract for fixed-length training
and packed-varlen serving instead of embedding model-specific time-mix
parameter generation in the kernel.

The current implementation targets head size 64 and canonical FP32
`state[K,V]`. It has been built and correctness-tested on an NVIDIA RTX PRO
6000 Blackwell GPU with PyTorch 2.11 and CUDA 13.0.

## Capabilities

| Algorithm | Layout | Numerical mode | Autograd | State behavior |
| --- | --- | --- | --- | --- |
| `reference` | fixed and packed | FP32 oracle | yes | functional |
| `recurrent` | fixed and packed | `fp32io16`, `fp16` | no | functional |
| `chunk` | fixed | `fp32io16` | yes | functional |
| `chunk` | packed | `fp32io16` | no | functional |
| `rwkv7_recurrent_stateful` | packed | `fp32io16`, `fp16` | no | in-place state pool |

`algorithm="auto"` currently selects recurrent. Correctness-gated SM120
measurements found no sequence-length crossover where the current
materialized or factor/recompute chunk implementation beats recurrent. Chunk
remains explicitly selectable for training and continued optimization.

## Operator

For every token, FlashRWKV evaluates

```text
S_t = diag(exp(log_decay_t)) S_(t-1)
    + b_t (a_t^T S_(t-1))
    + k_t v_t^T
y_t = scale * r_t^T S_t
```

Inputs use `[B,T,H,D]`, with `D=64`. The canonical state uses
`[N,H,K,V]`, not the transposed `[N,H,V,K]` layout used by some RWKV serving
kernels.

```python
import torch
from flash_rwkv import rwkv7

B, T, H, D = 2, 128, 32, 64
shape = (B, T, H, D)
r, log_decay, k, v, a, b = (
    torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    for _ in range(6)
)
log_decay = -0.5 * torch.sigmoid(log_decay)
initial_state = torch.zeros(
    B, H, D, D, device="cuda", dtype=torch.float32
)

output, final_state = rwkv7(
    r,
    log_decay,
    k,
    v,
    a,
    b,
    initial_state=initial_state,
    output_final_state=True,
    algorithm="chunk",
)
```

RWKV-LM training code can keep its raw decay logits outside the core
operator:

```python
from flash_rwkv import rwkv7_from_decay_logits

output, final_state = rwkv7_from_decay_logits(
    r,
    decay_logits,
    k,
    v,
    a,
    b,
    initial_state=initial_state,
    output_final_state=True,
    algorithm="chunk",
)
```

The adapter applies
`log_decay = -exp(-0.5) * sigmoid(decay_logits)` with a differentiable
PyTorch operation, so the CUDA backward returns the canonical
`dlog_decay` and PyTorch applies the remaining chain rule.

Fixed-length autograd uses a 16-token checkpoint interval for stable reverse
state reconstruction. Forward-only calls may still use autotuned 16-, 32-, or
64-token chunks; requesting a larger chunk does not weaken the training
checkpoint policy.

Packed inputs use `B=1`, concatenate tokens along `T`, and provide strictly
increasing `cu_seqlens[N+1]`. For in-place serving, also provide unique
`state_indices[N]` to `rwkv7_recurrent_stateful`; only those state-pool rows
are updated.

## Installation

A CUDA-enabled PyTorch environment, a matching CUDA toolkit, Ninja, and a C++
toolchain are required.

```bash
git clone https://github.com/rwkv-rs/FlashRWKV.git
cd FlashRWKV
python -m pip install -v --no-build-isolation .
```

The Helicopter checkout installs this repository through its dedicated
`flash-rwkv` dependency group and control-plane preparation workflow.

## Correctness and measurement

Run the package regression:

```bash
pytest -q
```

Run the unified correctness-gated JSON benchmark:

```bash
python benchmarks/benchmark_rwkv7.py
```

The runner reuses the Albatross/vllm-rwkv decode, equal-chunk, 1..16 ragged,
long-ragged, and skewed-ragged profiles. It compares public functional calls
for FlashRWKV recurrent, FlashRWKV chunk, and FLA `chunk_rwkv7` with identical
inputs and one CUDA event/synchronization boundary. In-place recurrent
state-pool measurements are reported separately and are not labelled
identical-input latency.

Additional development runners are:

- `benchmarks/benchmark_training.py`: fixed-length FlashRWKV/FLA
  forward-backward comparison using the same output and final-state upstream
  gradients;
- `benchmarks/autotune_chunk.py`: exhaustive chunk configuration sweep and
  versioned cache generation;
- `benchmarks/compare_chunk_strategies.py`: materialized,
  factor/recompute, and recurrent low-level behavior-cell comparison;
- `benchmarks/benchmark_recurrent.py`: detailed recurrent numerical-mode and
  stateful-boundary characterization.

Every timing runner gates the exact case on output and final-state
correctness, retains raw samples, and records source/binary revision,
hardware, runtime, shape, sequence lengths, configuration, and numerical
mode in JSON. Optimization decisions and profiler evidence are archived in
`docs/optimization/attempts/`.

## Provenance and licenses

FlashRWKV began as a code-only copy of
[MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA). Its KDA
operator, package, kernels, tests, and benchmark data-plane have been removed.
The remaining project uses FlashKDA's MIT-licensed repository history and
general separation of token-parallel preparation from recurrent state work as
design context; no KDA mathematical API remains.

The recurrent kernels and serving benchmark profiles are adapted from
[rwkv-rs/vllm-rwkv](https://github.com/rwkv-rs/vllm-rwkv) at
`6d683f9e49a2997e405c47edc147872c8609513b`, whose dense-kernel lineage
includes [BlinkDL/Albatross](https://github.com/BlinkDL/Albatross). The
fixed-length backward checkpoint/backstepping method is adapted from
[BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) at
`952102498e9ed367ea0a59ee64106916d474d30f`. Those derived files are
Apache-2.0 and retain SPDX/source headers.

[FLA](https://github.com/fla-org/flash-linear-attention) `chunk_rwkv7` at
`3adcb3c50a9e78c6ef6d173543305b1d5ef8fa4c` is an external correctness and
performance comparison dependency; its implementation is not bundled here.

See [NOTICE](NOTICE), [LICENSE](LICENSE), and
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) for the file-level boundary.
