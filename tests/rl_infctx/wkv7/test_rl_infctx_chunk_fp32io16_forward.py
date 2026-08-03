# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional

from flash_rwkv import (
    ChunkConfig,
    _extension,
    enumerate_chunk_configs,
    rl_infctx_chunk_fp32io16_factor_recompute,
    rwkv7,
)
from flash_rwkv.reference import rwkv7_decay_logits_reference

HEAD_SIZE = 64
TOLERANCE = json.loads(
    (Path(__file__).parents[2] / "fixtures/tolerances-v1.json").read_text(
        encoding="utf-8"
    )
)["fp32io16_chunk"]


@pytest.fixture(scope="module", autouse=True)
def require_flash_rwkv_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    try:
        from flash_rwkv import _C  # noqa: F401
    except ImportError as error:
        pytest.fail(f"FlashRWKV CUDA extension is unavailable: {error!r}")


def _inputs(
    *,
    batch_size: int,
    sequence_length: int,
    num_heads: int = 2,
    dtype: torch.dtype = torch.float16,
    seed: int = 42,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch_size, sequence_length, num_heads, HEAD_SIZE)

    def normal(scale: float) -> torch.Tensor:
        return scale * torch.randn(
            shape,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )

    direction = functional.normalize(normal(1.0), dim=-1)
    strength = 0.1 * torch.rand(
        shape,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    decay_logits = torch.randn(
        shape,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    tensors = (
        normal(0.05),
        decay_logits,
        normal(0.05),
        normal(0.05),
        -direction,
        direction * strength,
    )
    return tuple(tensor.to(dtype).contiguous() for tensor in tensors)


def _assert_relative_rmse(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    threshold: float,
) -> None:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    assert float(error / baseline) < threshold


def _fla_raw_chunk_operator() -> object:
    module = pytest.importorskip("fla.ops.rwkv7")
    operator = module.chunk_rwkv7
    parameters = tuple(inspect.signature(operator).parameters)
    if len(parameters) < 2 or parameters[1] != "decay_logits":
        pytest.fail(
            "installed FLA chunk_rwkv7 does not expose the required raw "
            "decay_logits ABI"
        )
    return operator


def _assert_chunk_matches_reference(
    inputs: tuple[torch.Tensor, ...],
    initial_state: torch.Tensor,
    *,
    chunk_size: int,
    scale: float = 1.0,
    cu_seqlens: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_output, expected_state = rwkv7_decay_logits_reference(
        *inputs,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
    )
    actual_output, actual_state = rwkv7(
        *inputs,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        algorithm="chunk",
        chunk_size=chunk_size,
    )
    torch.cuda.synchronize()
    assert actual_state is not None
    _assert_relative_rmse(
        actual_output,
        expected_output,
        threshold=TOLERANCE["output_relative_rmse"],
    )
    _assert_relative_rmse(
        actual_state,
        expected_state,
        threshold=TOLERANCE["state_relative_rmse"],
    )
    return actual_output, actual_state


def _chunk_metadata(
    sequence_lengths: tuple[int, ...],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_chunk_offsets = [0]
    chunk_token_starts: list[int] = []
    chunk_token_ends: list[int] = []
    token_start = 0
    for sequence_length in sequence_lengths:
        token_end = token_start + sequence_length
        for chunk_start in range(token_start, token_end, chunk_size):
            chunk_token_starts.append(chunk_start)
            chunk_token_ends.append(min(chunk_start + chunk_size, token_end))
        sequence_chunk_offsets.append(len(chunk_token_starts))
        token_start = token_end
    return (
        torch.tensor(
            sequence_chunk_offsets,
            device="cuda",
            dtype=torch.int32,
        ),
        torch.tensor(
            chunk_token_starts,
            device="cuda",
            dtype=torch.int32,
        ),
        torch.tensor(
            chunk_token_ends,
            device="cuda",
            dtype=torch.int32,
        ),
    )


def _run_recompute_chunk(
    inputs: tuple[torch.Tensor, ...],
    initial_state: torch.Tensor,
    *,
    sequence_lengths: tuple[int, ...],
    chunk_size: int,
    state_indices: torch.Tensor | None = None,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
    ) = _chunk_metadata(sequence_lengths, chunk_size)
    flattened_inputs = tuple(
        tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
        for tensor in inputs
    )
    if state_indices is None:
        state_indices = torch.arange(
            len(sequence_lengths),
            device="cuda",
            dtype=torch.int32,
        )
    state = initial_state.clone()
    output = torch.empty_like(flattened_inputs[3])
    boundary = torch.empty(
        (
            chunk_token_starts.numel(),
            flattened_inputs[0].shape[1],
            HEAD_SIZE,
            HEAD_SIZE,
        ),
        device="cuda",
        dtype=torch.float32,
    )
    _extension.recompute_chunk_fp32_from_decay_logits(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        state_indices,
        state,
        *flattened_inputs,
        output,
        boundary,
        scale,
        decay_bias=decay_bias,
    )
    return output.reshape(inputs[3].shape), state, boundary


@pytest.mark.parametrize("sequence_length", [1, 15, 16, 17, 31, 32, 33, 65])
def test_fixed_chunk_matches_reference(sequence_length: int) -> None:
    inputs = _inputs(
        batch_size=2,
        sequence_length=sequence_length,
        seed=sequence_length,
    )
    initial_state = 0.02 * torch.randn(
        2,
        2,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.float32,
    )
    _assert_chunk_matches_reference(
        inputs,
        initial_state,
        chunk_size=16,
    )


@pytest.mark.parametrize(
    ("dtype", "sequence_length", "chunk_size"),
    [
        (torch.float16, 17, 16),
        (torch.bfloat16, 65, 32),
        (torch.float16, 65, 64),
    ],
)
def test_recompute_chunk_matches_reference(
    dtype: torch.dtype,
    sequence_length: int,
    chunk_size: int,
) -> None:
    inputs = _inputs(
        batch_size=2,
        sequence_length=sequence_length,
        dtype=dtype,
        seed=700 + sequence_length + chunk_size,
    )
    initial_state = 0.02 * torch.randn(
        2,
        2,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.float32,
    )
    expected_output, expected_state = rwkv7_decay_logits_reference(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
    )
    actual_output, actual_state, boundary = _run_recompute_chunk(
        inputs,
        initial_state,
        sequence_lengths=(sequence_length,) * 2,
        chunk_size=chunk_size,
    )
    torch.cuda.synchronize()

    _assert_relative_rmse(
        actual_output,
        expected_output,
        threshold=TOLERANCE["output_relative_rmse"],
    )
    _assert_relative_rmse(
        actual_state,
        expected_state,
        threshold=TOLERANCE["state_relative_rmse"],
    )
    expected_chunks = 2 * ((sequence_length + chunk_size - 1) // chunk_size)
    assert boundary.shape == (expected_chunks, 2, HEAD_SIZE, HEAD_SIZE)


def test_public_recompute_fixed_with_decay_bias_matches_reference() -> None:
    inputs = _inputs(
        batch_size=2,
        sequence_length=17,
        dtype=torch.float16,
        seed=799,
    )
    decay_bias = torch.linspace(
        -0.25,
        0.25,
        2 * HEAD_SIZE,
        device="cuda",
        dtype=torch.float16,
    ).reshape(2, HEAD_SIZE)
    initial_state = 0.02 * torch.randn(
        2,
        2,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.float32,
    )
    expected_output, expected_state = rwkv7_decay_logits_reference(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        decay_bias=decay_bias,
    )
    actual_output, actual_state = rl_infctx_chunk_fp32io16_factor_recompute(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=16,
        decay_bias=decay_bias,
    )
    torch.cuda.synchronize()

    assert actual_state is not None
    _assert_relative_rmse(
        actual_output,
        expected_output,
        threshold=TOLERANCE["output_relative_rmse"],
    )
    _assert_relative_rmse(
        actual_state,
        expected_state,
        threshold=TOLERANCE["state_relative_rmse"],
    )


def test_recompute_packed_slot_mapping_matches_reference() -> None:
    sequence_lengths = (3, 16, 17, 35)
    inputs = _inputs(
        batch_size=1,
        sequence_length=sum(sequence_lengths),
        dtype=torch.bfloat16,
        seed=811,
    )
    cu_seqlens = torch.tensor(
        [0, 3, 19, 36, 71],
        device="cuda",
        dtype=torch.int64,
    )
    state_indices = torch.tensor(
        [5, 1, 4, 2],
        device="cuda",
        dtype=torch.int32,
    )
    initial_state = 0.02 * torch.randn(
        7,
        2,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.float32,
    )
    decay_bias = torch.linspace(
        0.15,
        -0.15,
        2 * HEAD_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected_output, expected_state = rwkv7_decay_logits_reference(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
    )
    actual_output, actual_state = rl_infctx_chunk_fp32io16_factor_recompute(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        chunk_size=16,
        decay_bias=decay_bias,
    )
    torch.cuda.synchronize()

    assert actual_state is not None
    _assert_relative_rmse(
        actual_output,
        expected_output,
        threshold=TOLERANCE["output_relative_rmse"],
    )
    _assert_relative_rmse(
        actual_state,
        expected_state,
        threshold=TOLERANCE["state_relative_rmse"],
    )
    untouched = torch.tensor([0, 3, 6], device="cuda")
    assert torch.equal(
        actual_state.index_select(0, untouched),
        initial_state.index_select(0, untouched),
    )


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_chunk_size_and_masked_tail_match_reference(
    chunk_size: int,
) -> None:
    inputs = _inputs(
        batch_size=1,
        sequence_length=65,
        dtype=torch.bfloat16,
        seed=100 + chunk_size,
    )
    initial_state = 0.02 * torch.randn(
        1,
        2,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.float32,
    )
    _assert_chunk_matches_reference(
        inputs,
        initial_state,
        chunk_size=chunk_size,
        scale=0.125,
    )


def test_packed_chunk_slot_mapping_matches_reference() -> None:
    sequence_lengths = (1, 16, 17, 33)
    inputs = _inputs(
        batch_size=1,
        sequence_length=sum(sequence_lengths),
        seed=71,
    )
    cu_seqlens = torch.tensor(
        [0, 1, 17, 34, 67],
        device="cuda",
        dtype=torch.int64,
    )
    state_indices = torch.tensor(
        [5, 1, 7, 3],
        device="cuda",
        dtype=torch.int32,
    )
    state_pool = 0.02 * torch.randn(
        9,
        2,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.float32,
    )
    _, actual_pool = _assert_chunk_matches_reference(
        inputs,
        state_pool,
        chunk_size=16,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
    )
    untouched = torch.tensor([0, 2, 4, 6, 8], device="cuda")
    assert torch.equal(
        actual_pool.index_select(0, untouched),
        state_pool.index_select(0, untouched),
    )


@pytest.mark.parametrize(
    "config",
    enumerate_chunk_configs(),
    ids=lambda config: config.identifier,
)
def test_all_materialized_config_variants_match_reference(
    config: ChunkConfig,
) -> None:
    inputs = _inputs(
        batch_size=1,
        sequence_length=65,
        num_heads=1,
        seed=401,
    )
    initial_state = 0.02 * torch.randn(
        1,
        1,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.float32,
    )
    expected_output, expected_state = rwkv7_decay_logits_reference(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
    )
    actual_output, actual_state = rwkv7(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        algorithm="chunk",
        chunk_config=config,
    )
    torch.cuda.synchronize()
    _assert_relative_rmse(
        actual_output,
        expected_output,
        threshold=TOLERANCE["output_relative_rmse"],
    )
    _assert_relative_rmse(
        actual_state,
        expected_state,
        threshold=TOLERANCE["state_relative_rmse"],
    )


@pytest.mark.parametrize(
    ("sequence_length", "explicit_algorithm"),
    [(16, "recurrent"), (17, "recurrent"), (65, "recurrent")],
)
def test_auto_family_dispatch_matches_explicit_algorithm(
    sequence_length: int,
    explicit_algorithm: str,
) -> None:
    inputs = _inputs(
        batch_size=1,
        sequence_length=sequence_length,
        num_heads=1,
        seed=500 + sequence_length,
    )
    initial_state = 0.02 * torch.randn(
        1,
        1,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.float32,
    )
    auto_output, auto_state = rwkv7(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        algorithm="auto",
    )
    explicit_output, explicit_state = rwkv7(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        algorithm=explicit_algorithm,
    )
    torch.cuda.synchronize()
    assert torch.equal(auto_output, explicit_output)
    assert torch.equal(auto_state, explicit_state)


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_fixed_raw_chunk_matches_fla(chunk_size: int) -> None:
    chunk_rwkv7 = _fla_raw_chunk_operator()
    inputs = _inputs(
        batch_size=1,
        sequence_length=65,
        dtype=torch.bfloat16,
        seed=200 + chunk_size,
    )
    initial_state = 0.02 * torch.randn(
        1,
        2,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.float32,
    )
    actual_output, actual_state = _assert_chunk_matches_reference(
        inputs,
        initial_state,
        chunk_size=chunk_size,
        scale=0.125,
    )
    with torch.no_grad():
        fla_output, fla_state = chunk_rwkv7(
            *inputs,
            scale=0.125,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=chunk_size,
        )
    torch.cuda.synchronize()
    _assert_relative_rmse(
        actual_output,
        fla_output,
        threshold=TOLERANCE["output_relative_rmse"],
    )
    _assert_relative_rmse(
        actual_state,
        fla_state,
        threshold=TOLERANCE["state_relative_rmse"],
    )


def test_packed_raw_chunk_matches_fla() -> None:
    chunk_rwkv7 = _fla_raw_chunk_operator()
    sequence_lengths = (3, 16, 17, 35)
    inputs = _inputs(
        batch_size=1,
        sequence_length=sum(sequence_lengths),
        dtype=torch.bfloat16,
        seed=311,
    )
    cu_seqlens = torch.tensor(
        [0, 3, 19, 36, 71],
        device="cuda",
        dtype=torch.int64,
    )
    cu_seqlens_cpu = cu_seqlens.cpu()
    initial_state = 0.02 * torch.randn(
        4,
        2,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.float32,
    )
    actual_output, actual_state = _assert_chunk_matches_reference(
        inputs,
        initial_state,
        chunk_size=16,
        cu_seqlens=cu_seqlens,
    )
    with torch.no_grad():
        fla_output, fla_state = chunk_rwkv7(
            *inputs,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_size=16,
        )
    torch.cuda.synchronize()
    _assert_relative_rmse(
        actual_output,
        fla_output,
        threshold=TOLERANCE["output_relative_rmse"],
    )
    _assert_relative_rmse(
        actual_state,
        fla_state,
        threshold=TOLERANCE["state_relative_rmse"],
    )


def test_chunk_rejects_unsupported_mode_and_size() -> None:
    inputs = _inputs(batch_size=1, sequence_length=17)
    with pytest.raises(ValueError, match="only mode='fp32io16'"):
        rwkv7(*inputs, algorithm="chunk", mode="fp16")
    with pytest.raises(ValueError, match="one of 16, 32, or 64"):
        rwkv7(*inputs, algorithm="chunk", chunk_size=8)
    with pytest.raises(ValueError, match="either chunk_size or config"):
        rwkv7(
            *inputs,
            algorithm="chunk",
            chunk_size=16,
            chunk_config=ChunkConfig(16, 2, 1, 64),
        )


def test_chunk_autograd_fails_closed_without_dispatching_recurrent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched_recurrent = False

    def recurrent_autograd_sentinel(*_args: object, **_kwargs: object) -> None:
        nonlocal dispatched_recurrent
        dispatched_recurrent = True
        raise AssertionError("chunk must not dispatch the recurrent autograd op")

    monkeypatch.setattr(
        "flash_rwkv.ops.pretrain_recurrent_fp32io16_from_decay_logits_autograd",
        recurrent_autograd_sentinel,
    )
    inputs = list(_inputs(batch_size=1, sequence_length=17))
    inputs[0].requires_grad_(True)
    with pytest.raises(RuntimeError, match="chunk.*autograd is unsupported"):
        rwkv7(*inputs, algorithm="chunk")
    assert not dispatched_recurrent
