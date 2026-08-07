# FlashRWKV2

FlashRWKV2 is a high-performance CUDA operator library for RWKV-7. It provides
composable inference and training operators; complete models, schedulers, and
training frameworks remain the responsibility of downstream projects.

## Requirements

- Python 3.10 or newer
- An NVIDIA CUDA build environment and a supported CUDA device
- `uv`, using the repository-local `./.venv`
- SM120 or newer for the current native build

The runtime and reproducible source-build contracts currently pin PyTorch
2.11.0. When the build cannot detect a local GPU, set
`TORCH_CUDA_ARCH_LIST=12.0` explicitly.

## Installation

```bash
git clone https://github.com/rwkv-rs/FlashRWKV2.git
cd FlashRWKV2
uv sync
TORCH_CUDA_ARCH_LIST=12.0 \
  ./.venv/bin/python -m pip install -v --no-build-isolation -e .
```

## Kernel API

See the [Kernel API reference](docs/kernel_api.md) for the complete public
operator surface and tensor contracts.

## Tests

```bash
./.venv/bin/python -m pytest -q
```

CUDA tests require a successfully built `flashrwkv2._C` extension and a
supported GPU.

## Benchmarks

Operator benchmarks live in [`benchmarks/`](benchmarks/). For example, run the
WKV7 recurrent correctness benchmark with:

```bash
./.venv/bin/python -m benchmarks.tmix.wkv7.bench \
  --shapes h32d64 \
  --dtype bfloat16 \
  --correctness-only \
  --output /tmp/flashrwkv2-wkv7-correctness.json
```

These benchmarks measure individual operators. They do not report or infer
model-level latency.

For Albatross-compatible TMix low-rank inference, `varlen` means that the
operator consumes packed token rows; it does not mean that one fused kernel is
used for every row count.  The public composite callers automatically use the
canonical fused rank-in window at `M<=7`, the fused rank-out/value-residual
window at `M<=4`, and the canonical large-row linear dispatcher otherwise.
Callers may provide both the original checkpoint layout and a runtime layout.
Both layouts must be prepared outside the timed forward region and retained for
the lifetime of the inference weights; FlashRWKV2 never transposes or copies a
missing layout during dispatch.

The existing low-rank operator benchmark reports correctness and latency as
JSON without adding a model-level benchmark:

```bash
./.venv/bin/python benchmarks/tmix/linear/bench.py \
  --operator projection-group --layout both --channels 4096
```

## License

FlashRWKV2 is distributed under the [MIT License](LICENSE).
