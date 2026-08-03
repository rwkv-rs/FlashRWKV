# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from flash_rwkv import (
    _extension,
    infer_recurrent_fp16_forward_varlen,
    infer_recurrent_fp32io16_forward_varlen,
    prepare_recurrent_metadata,
    rwkv7,
    rwkv7_recurrent_stateful,
    validate_packed_metadata_strict,
)
from flash_rwkv.reference import rwkv7_decay_logits_reference

HEAD_SIZE = 64
TOLERANCES = json.loads(
    (Path(__file__).parents[2] / "fixtures/tolerances-v1.json").read_text(
        encoding="utf-8"
    )
)
RECURRENT_TOLERANCES = {
    "fp32io16": TOLERANCES["fp32io16_recurrent"],
    "fp16": TOLERANCES["fp16"],
}
HOSTILE_METADATA_CASES = (
    ("malformed-start", (1, 2, 3), (0, 1)),
    ("malformed-end", (0, 1, 2), (0, 1)),
    ("nonmonotonic-overlap", (0, 2, 1, 3), (0, 1, 2)),
    ("negative-slot", (0, 1, 3), (-1, 1)),
    ("out-of-range-slot", (0, 1, 3), (0, 5)),
    ("duplicate-slot", (0, 1, 3), (2, 2)),
)


@pytest.fixture(scope="module", autouse=True)
def require_flash_rwkv_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    try:
        from flash_rwkv import _C  # noqa: F401
    except ImportError as error:
        pytest.skip(f"FlashRWKV CUDA extension is unavailable: {error!r}")


def _inputs(
    *,
    batch_size: int,
    sequence_length: int,
    num_heads: int = 1,
    head_size: int = HEAD_SIZE,
    seed: int = 42,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch_size, sequence_length, num_heads, head_size)
    r = 0.1 * torch.randn(shape, generator=generator, device="cuda")
    decay_logits = torch.randn(shape, generator=generator, device="cuda")
    k = 0.1 * torch.randn(shape, generator=generator, device="cuda")
    v = 0.1 * torch.randn(shape, generator=generator, device="cuda")
    a = 0.1 * torch.randn(shape, generator=generator, device="cuda")
    b = 0.1 * torch.randn(shape, generator=generator, device="cuda")
    return tuple(
        tensor.to(torch.float16)
        for tensor in (r, decay_logits, k, v, a, b)
    )


def _assert_relative_rmse_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    threshold: float,
) -> None:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    assert float(error / baseline) < threshold


@pytest.mark.parametrize(
    "mode",
    ["fp32io16", "fp16"],
)
@pytest.mark.parametrize("sequence_length", [1, 15, 16, 17, 65])
def test_fixed_recurrent_matches_fp32_reference(
    sequence_length: int,
    mode: str,
) -> None:
    inputs = _inputs(batch_size=2, sequence_length=sequence_length, seed=sequence_length)
    state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
    initial_state = 0.01 * torch.randn(
        2, 1, HEAD_SIZE, HEAD_SIZE, device="cuda", dtype=state_dtype
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
        algorithm="recurrent",
        mode=mode,
    )
    torch.cuda.synchronize()

    tolerance = RECURRENT_TOLERANCES[mode]
    _assert_relative_rmse_close(
        actual_output,
        expected_output,
        threshold=tolerance["output_relative_rmse"],
    )
    _assert_relative_rmse_close(
        actual_state,
        expected_state,
        threshold=tolerance["state_relative_rmse"],
    )


@pytest.mark.parametrize(
    "mode",
    ["fp32io16", "fp16"],
)
def test_packed_slot_mapping_matches_reference_and_preserves_pool(
    mode: str,
) -> None:
    sequence_lengths = (1, 4, 2)
    inputs = _inputs(
        batch_size=1,
        sequence_length=sum(sequence_lengths),
        num_heads=2,
        seed=9,
    )
    cu_seqlens = torch.tensor([0, 1, 5, 7], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([4, 1, 5], device="cuda", dtype=torch.int32)
    state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
    state_pool = 0.01 * torch.randn(
        7, 2, HEAD_SIZE, HEAD_SIZE, device="cuda", dtype=state_dtype
    )

    expected_output, expected_pool = rwkv7_decay_logits_reference(
        *inputs,
        initial_state=state_pool,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
    )
    actual_output, actual_pool = rwkv7(
        *inputs,
        initial_state=state_pool,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        algorithm="recurrent",
        mode=mode,
    )
    torch.cuda.synchronize()

    tolerance = RECURRENT_TOLERANCES[mode]
    _assert_relative_rmse_close(
        actual_output,
        expected_output,
        threshold=tolerance["output_relative_rmse"],
    )
    _assert_relative_rmse_close(
        actual_pool,
        expected_pool,
        threshold=tolerance["state_relative_rmse"],
    )
    untouched = torch.tensor([0, 2, 3, 6], device="cuda")
    assert torch.equal(
        actual_pool.index_select(0, untouched),
        state_pool.index_select(0, untouched),
    )


@pytest.mark.parametrize(
    "mode",
    ["fp32io16", "fp16"],
)
def test_stateful_recurrent_updates_only_selected_rows(
    mode: str,
) -> None:
    inputs = _inputs(batch_size=1, sequence_length=5, seed=19)
    cu_seqlens = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
    state_pool = 0.01 * torch.randn(
        5, 1, HEAD_SIZE, HEAD_SIZE, device="cuda", dtype=state_dtype
    )
    initial_state = state_pool.clone()
    cu_seqlens_data_ptr = cu_seqlens.data_ptr()
    state_indices_data_ptr = state_indices.data_ptr()

    expected_output, expected_pool = rwkv7_decay_logits_reference(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
    )
    actual_output = rwkv7_recurrent_stateful(
        *inputs,
        state_pool=state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        mode=mode,
    )
    torch.cuda.synchronize()

    assert cu_seqlens.data_ptr() == cu_seqlens_data_ptr
    assert state_indices.data_ptr() == state_indices_data_ptr

    tolerance = RECURRENT_TOLERANCES[mode]
    _assert_relative_rmse_close(
        actual_output,
        expected_output,
        threshold=tolerance["output_relative_rmse"],
    )
    _assert_relative_rmse_close(
        state_pool,
        expected_pool,
        threshold=tolerance["state_relative_rmse"],
    )
    untouched = torch.tensor([0, 2, 4], device="cuda")
    assert torch.equal(
        state_pool.index_select(0, untouched),
        initial_state.index_select(0, untouched),
    )


@pytest.mark.parametrize("head_size", [128, 256])
@pytest.mark.parametrize("mode", ["fp32io16", "fp16"])
def test_large_head_stateful_recurrent_matches_fp32_reference(
    head_size: int,
    mode: str,
) -> None:
    inputs = _inputs(
        batch_size=1,
        sequence_length=3,
        head_size=head_size,
        seed=head_size,
    )
    cu_seqlens = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
    state_pool = 0.01 * torch.randn(
        4,
        1,
        head_size,
        head_size,
        device="cuda",
        dtype=state_dtype,
    )
    initial_state = state_pool.clone()
    expected_output, expected_pool = rwkv7_decay_logits_reference(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
    )

    actual_output = rwkv7_recurrent_stateful(
        *inputs,
        state_pool=state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        mode=mode,
    )
    torch.cuda.synchronize()

    tolerance = RECURRENT_TOLERANCES[mode]
    _assert_relative_rmse_close(
        actual_output,
        expected_output,
        threshold=tolerance["output_relative_rmse"],
    )
    _assert_relative_rmse_close(
        state_pool,
        expected_pool,
        threshold=tolerance["state_relative_rmse"],
    )
    assert torch.equal(state_pool[[1, 3]], initial_state[[1, 3]])


@pytest.mark.parametrize(
    ("mode", "named_operator"),
    [
        ("fp32io16", infer_recurrent_fp32io16_forward_varlen),
        ("fp16", infer_recurrent_fp16_forward_varlen),
    ],
)
def test_named_varlen_operator_matches_legacy_dispatch(
    mode: str,
    named_operator: object,
) -> None:
    inputs = _inputs(batch_size=1, sequence_length=7, seed=29)
    cu_seqlens = torch.tensor([0, 2, 7], device="cuda", dtype=torch.int32)
    state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
    initial_state = 0.01 * torch.randn(
        2, 1, HEAD_SIZE, HEAD_SIZE, device="cuda", dtype=state_dtype
    )
    expected = rwkv7(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        algorithm="recurrent",
        mode=mode,
    )
    actual = named_operator(  # type: ignore[operator]
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_strict_debug_validation_rejects_duplicate_state_indices() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        validate_packed_metadata_strict(
            torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32),
            torch.tensor([0, 0], device="cuda", dtype=torch.int32),
            total_tokens=2,
            state_pool_size=2,
        )


@pytest.mark.parametrize("mode", ["fp32io16", "fp16"])
@pytest.mark.parametrize(
    ("case", "offsets", "slots"),
    HOSTILE_METADATA_CASES,
    ids=tuple(case[0] for case in HOSTILE_METADATA_CASES),
)
def test_raw_recurrent_native_op_fails_closed_for_hostile_metadata(
    mode: str,
    case: str,
    offsets: tuple[int, ...],
    slots: tuple[int, ...],
) -> None:
    from flash_rwkv import _C

    del case
    inputs = _inputs(batch_size=1, sequence_length=3, seed=59)
    flattened = tuple(tensor.reshape(3, 1, HEAD_SIZE) for tensor in inputs)
    state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
    state = torch.randn(
        5,
        1,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cuda",
        dtype=state_dtype,
    )
    state_before = state.clone()
    output = torch.ones_like(flattened[3])
    query_start_loc = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    state_indices = torch.tensor(slots, device="cuda", dtype=torch.int32)
    operator = (
        _C.recurrent_fp32_from_decay_logits
        if mode == "fp32io16"
        else _C.recurrent_fp16_from_decay_logits
    )

    operator(
        query_start_loc,
        state_indices,
        state,
        *flattened,
        output,
        1.0,
    )
    torch.cuda.synchronize()

    assert torch.equal(state, state_before)
    assert torch.isnan(output).all()


@pytest.mark.parametrize("mode", ["fp32io16", "fp16"])
@pytest.mark.parametrize(
    ("case", "offsets", "slots"),
    HOSTILE_METADATA_CASES,
    ids=tuple(case[0] for case in HOSTILE_METADATA_CASES),
)
def test_public_stateful_recurrent_fails_closed_for_hostile_metadata(
    mode: str,
    case: str,
    offsets: tuple[int, ...],
    slots: tuple[int, ...],
) -> None:
    del case
    inputs = _inputs(batch_size=1, sequence_length=3, seed=37)
    state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
    state_pool = torch.randn(
        5,
        1,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cuda",
        dtype=state_dtype,
    )
    state_before = state_pool.clone()
    output = rwkv7_recurrent_stateful(
        *inputs,
        state_pool=state_pool,
        cu_seqlens=torch.tensor(offsets, device="cuda", dtype=torch.int32),
        state_indices=torch.tensor(slots, device="cuda", dtype=torch.int32),
        mode=mode,
    )
    torch.cuda.synchronize()

    assert torch.equal(state_pool, state_before)
    assert torch.isnan(output).all()


def test_stateful_recurrent_passes_device_metadata_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(batch_size=1, sequence_length=3, seed=39)
    cu_seqlens = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([4, 1], device="cuda", dtype=torch.int32)
    state_pool = torch.zeros(
        5, 1, HEAD_SIZE, HEAD_SIZE, device="cuda", dtype=torch.float32
    )
    observed: dict[str, torch.Tensor] = {}

    def recurrent_fp32_from_decay_logits(
        query_start_loc: torch.Tensor,
        cuda_state_indices: torch.Tensor,
        _state_pool: torch.Tensor,
        *_arguments: object,
        **_keywords: object,
    ) -> None:
        observed["cu_seqlens"] = query_start_loc
        observed["state_indices"] = cuda_state_indices
        output = _arguments[-2]
        assert isinstance(output, torch.Tensor)
        output.zero_()

    monkeypatch.setattr(
        _extension,
        "recurrent_fp32_from_decay_logits",
        recurrent_fp32_from_decay_logits,
    )
    rwkv7_recurrent_stateful(
        *inputs,
        state_pool=state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
    )

    assert observed["cu_seqlens"] is cu_seqlens
    assert observed["state_indices"] is state_indices


@pytest.mark.parametrize("mode", ["fp32io16", "fp16"])
def test_stateful_recurrent_packed_path_is_cuda_graph_capturable(mode: str) -> None:
    inputs = _inputs(batch_size=1, sequence_length=3, seed=49)
    cu_seqlens = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([4, 1], device="cuda", dtype=torch.int32)
    state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
    state_pool = torch.zeros(
        5, 1, HEAD_SIZE, HEAD_SIZE, device="cuda", dtype=state_dtype
    )
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        validated_metadata = prepare_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=3,
            state_pool_size=5,
        )
        for _ in range(3):
            rwkv7_recurrent_stateful(
                *inputs,
                state_pool=state_pool,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                mode=mode,
                validated_metadata=validated_metadata,
            )
    torch.cuda.current_stream().wait_stream(capture_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output = rwkv7_recurrent_stateful(
            *inputs,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode=mode,
            validated_metadata=validated_metadata,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert captured_output.shape == inputs[3].shape
    assert torch.isfinite(captured_output).all()


def _raw_native_ticket_call(
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    validated_metadata: object,
    *,
    total_tokens: int = 3,
    state_pool_size: int = 5,
) -> None:
    from flash_rwkv import _C

    inputs = _inputs(
        batch_size=1,
        sequence_length=total_tokens,
        seed=113 + total_tokens + state_pool_size,
    )
    flattened = tuple(
        tensor.reshape(total_tokens, 1, HEAD_SIZE) for tensor in inputs
    )
    state_pool = torch.zeros(
        state_pool_size,
        1,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.float32,
    )
    output = torch.empty_like(flattened[3])
    _C.recurrent_fp32_from_decay_logits(
        query_start_loc,
        state_indices,
        state_pool,
        *flattened,
        output,
        1.0,
        None,
        None,
        validated_metadata,
    )


def test_prepared_metadata_reuses_snapshot_across_two_stateful_calls() -> None:
    inputs = _inputs(batch_size=1, sequence_length=3, seed=71)
    cu_seqlens = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    state_pool = 0.01 * torch.randn(
        5, 1, HEAD_SIZE, HEAD_SIZE, device="cuda", dtype=torch.float32
    )
    expected_pool = state_pool.clone()
    original_id = id(state_pool)
    original_data_ptr = state_pool.data_ptr()
    validated_metadata = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=3,
        state_pool_size=5,
    )

    for _ in range(2):
        expected_output, expected_pool = rwkv7_decay_logits_reference(
            *inputs,
            initial_state=expected_pool,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
        )
        actual_output = rwkv7_recurrent_stateful(
            *inputs,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=validated_metadata,
        )
        torch.cuda.synchronize()
        _assert_relative_rmse_close(
            actual_output,
            expected_output,
            threshold=RECURRENT_TOLERANCES["fp32io16"][
                "output_relative_rmse"
            ],
        )
        _assert_relative_rmse_close(
            state_pool,
            expected_pool,
            threshold=RECURRENT_TOLERANCES["fp32io16"][
                "state_relative_rmse"
            ],
        )

    assert id(state_pool) == original_id
    assert state_pool.data_ptr() == original_data_ptr


def test_prepared_metadata_rejects_wrong_tensor_identity() -> None:
    cu_seqlens = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    validated_metadata = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=3,
        state_pool_size=5,
    )

    with pytest.raises(RuntimeError, match="query_start_loc identity mismatch"):
        _raw_native_ticket_call(
            cu_seqlens.clone(),
            state_indices,
            validated_metadata,
        )
    with pytest.raises(RuntimeError, match="state_indices identity mismatch"):
        _raw_native_ticket_call(
            cu_seqlens,
            state_indices.clone(),
            validated_metadata,
        )


def test_prepared_metadata_rejects_shape_or_storage_change() -> None:
    cu_seqlens = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    validated_metadata = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=3,
        state_pool_size=5,
    )
    cu_seqlens.resize_(4).copy_(
        torch.tensor([0, 1, 2, 4], device="cuda", dtype=torch.int32)
    )
    state_indices.resize_(3).copy_(
        torch.tensor([3, 1, 2], device="cuda", dtype=torch.int32)
    )

    with pytest.raises(RuntimeError, match="validated_metadata .*mismatch"):
        _raw_native_ticket_call(
            cu_seqlens,
            state_indices,
            validated_metadata,
            total_tokens=4,
        )


@pytest.mark.parametrize(
    ("total_tokens", "state_pool_size", "message"),
    [
        (4, 5, "total_tokens mismatch"),
        (3, 6, "state_pool_size mismatch"),
    ],
)
def test_prepared_metadata_rejects_bound_dimension_change(
    total_tokens: int,
    state_pool_size: int,
    message: str,
) -> None:
    cu_seqlens = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    validated_metadata = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=3,
        state_pool_size=5,
    )

    with pytest.raises(RuntimeError, match=message):
        _raw_native_ticket_call(
            cu_seqlens,
            state_indices,
            validated_metadata,
            total_tokens=total_tokens,
            state_pool_size=state_pool_size,
        )


@pytest.mark.parametrize("source", ["query_start_loc", "state_indices"])
def test_prepared_metadata_rejects_versioned_source_mutation(source: str) -> None:
    cu_seqlens = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    validated_metadata = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=3,
        state_pool_size=5,
    )
    mutated = cu_seqlens if source == "query_start_loc" else state_indices
    mutated.add_(0)

    with pytest.raises(RuntimeError, match=f"{source} version mismatch"):
        _raw_native_ticket_call(
            cu_seqlens,
            state_indices,
            validated_metadata,
        )


def test_prepared_metadata_rejects_cross_stream_consumer() -> None:
    cu_seqlens = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    validated_metadata = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=3,
        state_pool_size=5,
    )
    other_stream = torch.cuda.Stream()

    with torch.cuda.stream(other_stream), pytest.raises(
        RuntimeError, match="stream mismatch"
    ):
        _raw_native_ticket_call(
            cu_seqlens,
            state_indices,
            validated_metadata,
        )
    other_stream.synchronize()


def test_inference_metadata_mutation_after_prepare_uses_ticket_snapshot() -> None:
    inputs = _inputs(batch_size=1, sequence_length=3, seed=83)
    expected_cu_seqlens = torch.tensor(
        [0, 1, 3], device="cuda", dtype=torch.int32
    )
    expected_state_indices = torch.tensor(
        [3, 1], device="cuda", dtype=torch.int32
    )
    initial_state = 0.01 * torch.randn(
        5, 1, HEAD_SIZE, HEAD_SIZE, device="cuda", dtype=torch.float32
    )
    expected_output, expected_pool = rwkv7_decay_logits_reference(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=expected_cu_seqlens,
        state_indices=expected_state_indices,
    )
    state_pool = initial_state.clone()

    with torch.inference_mode():
        cu_seqlens = torch.tensor(
            [0, 1, 3], device="cuda", dtype=torch.int32
        )
        state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
        validated_metadata = prepare_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=3,
            state_pool_size=5,
        )
        cu_seqlens.copy_(
            torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
        )
        state_indices.copy_(
            torch.tensor([0, 0], device="cuda", dtype=torch.int32)
        )
        actual_output = rwkv7_recurrent_stateful(
            *inputs,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=validated_metadata,
        )
    torch.cuda.synchronize()

    _assert_relative_rmse_close(
        actual_output,
        expected_output,
        threshold=RECURRENT_TOLERANCES["fp32io16"]["output_relative_rmse"],
    )
    _assert_relative_rmse_close(
        state_pool,
        expected_pool,
        threshold=RECURRENT_TOLERANCES["fp32io16"]["state_relative_rmse"],
    )


def test_direct_metadata_path_snapshots_before_async_source_mutation() -> None:
    from flash_rwkv import _C

    inputs = _inputs(batch_size=1, sequence_length=3, seed=89)
    flattened = tuple(tensor.reshape(3, 1, HEAD_SIZE) for tensor in inputs)
    cu_seqlens = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    initial_state = 0.01 * torch.randn(
        5, 1, HEAD_SIZE, HEAD_SIZE, device="cuda", dtype=torch.float32
    )
    expected_output, expected_pool = rwkv7_decay_logits_reference(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
    )
    state_pool = initial_state.clone()
    output = torch.empty_like(flattened[3])

    _C.recurrent_fp32_from_decay_logits(
        cu_seqlens,
        state_indices,
        state_pool,
        *flattened,
        output,
        1.0,
    )
    cu_seqlens.copy_(
        torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
    )
    state_indices.copy_(
        torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    )
    torch.cuda.synchronize()

    _assert_relative_rmse_close(
        output.reshape_as(expected_output),
        expected_output,
        threshold=RECURRENT_TOLERANCES["fp32io16"]["output_relative_rmse"],
    )
    _assert_relative_rmse_close(
        state_pool,
        expected_pool,
        threshold=RECURRENT_TOLERANCES["fp32io16"]["state_relative_rmse"],
    )


def test_ticket_reuse_launches_only_recurrent_kernels() -> None:
    from torch.profiler import ProfilerActivity, profile

    inputs = _inputs(batch_size=1, sequence_length=3, seed=97)
    cu_seqlens = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    state_pool = torch.zeros(
        5, 1, HEAD_SIZE, HEAD_SIZE, device="cuda", dtype=torch.float32
    )
    validated_metadata = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=3,
        state_pool_size=5,
    )
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    ) as trace:
        for _ in range(2):
            rwkv7_recurrent_stateful(
                *inputs,
                state_pool=state_pool,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                validated_metadata=validated_metadata,
            )
        torch.cuda.synchronize()

    cuda_kernel_names = [
        event.name
        for event in trace.events()
        if "cuda" in str(event.device_type).lower()
    ]
    recurrent_kernel_names = [
        name for name in cuda_kernel_names if "recurrent_fp32_kernel" in name
    ]
    assert len(recurrent_kernel_names) == 2, cuda_kernel_names
    assert len(cuda_kernel_names) == 2, cuda_kernel_names
    assert not any(
        "validate_recurrent_metadata_kernel" in name
        for name in cuda_kernel_names
    )


@pytest.mark.parametrize("mode", ["fp32io16", "fp16"])
def test_stateful_recurrent_fuses_decay_bias(mode: str) -> None:
    inputs = _inputs(
        batch_size=1,
        sequence_length=4,
        num_heads=2,
        seed=101,
    )
    cu_seqlens = torch.tensor([0, 1, 4], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([4, 1], device="cuda", dtype=torch.int32)
    decay_bias = torch.linspace(
        -0.25,
        0.25,
        2 * HEAD_SIZE,
        device="cuda",
        dtype=torch.float16,
    ).reshape(2, HEAD_SIZE)
    state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
    initial_state = 0.01 * torch.randn(
        5, 2, HEAD_SIZE, HEAD_SIZE, device="cuda", dtype=state_dtype
    )
    expected_output, expected_pool = rwkv7_decay_logits_reference(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
    )
    state_pool = initial_state.clone()

    actual_output = rwkv7_recurrent_stateful(
        *inputs,
        state_pool=state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        mode=mode,
        decay_bias=decay_bias,
    )
    torch.cuda.synchronize()

    tolerance = RECURRENT_TOLERANCES[mode]
    _assert_relative_rmse_close(
        actual_output,
        expected_output,
        threshold=tolerance["output_relative_rmse"],
    )
    _assert_relative_rmse_close(
        state_pool,
        expected_pool,
        threshold=tolerance["state_relative_rmse"],
    )


def _dithered_log_decay(
    decay_logits: torch.Tensor,
    decay_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    elapsed_t: torch.Tensor,
) -> torch.Tensor:
    retention = torch.exp(
        -0.6065306597126334 * torch.sigmoid(decay_logits.float() + decay_bias)
    )
    phase = torch.empty_like(decay_logits, dtype=torch.int64)
    offsets = tuple(int(value) for value in cu_seqlens.cpu().tolist())
    slots = tuple(int(value) for value in state_indices.cpu().tolist())
    elapsed = tuple(int(value) for value in elapsed_t.cpu().tolist())
    _, _, num_heads, head_size = decay_logits.shape
    element_phase = torch.arange(
        num_heads * head_size,
        device=decay_logits.device,
        dtype=torch.int64,
    ).reshape(num_heads, head_size)
    for sequence_index, state_slot in enumerate(slots):
        token_start = offsets[sequence_index]
        token_end = offsets[sequence_index + 1]
        token_phase = torch.arange(
            token_end - token_start,
            device=decay_logits.device,
            dtype=torch.int64,
        ).reshape(-1, 1, 1)
        phase[0, token_start:token_end] = (
            elapsed[state_slot] + token_phase + element_phase
        )
    bits = (2654435769 * phase).bitwise_and(0xFFFFFFFF)
    signed_bits = torch.where(bits >= 0x80000000, bits - 0x100000000, bits)
    dither = (2.0**-41) * signed_bits.float()
    return torch.log(retention + dither).to(dtype=decay_logits.dtype).contiguous()


def test_fp16_elapsed_t_matches_independent_dithered_recurrence() -> None:
    from flash_rwkv.reference import rwkv7_reference

    inputs = _inputs(batch_size=1, sequence_length=4, seed=107)
    cu_seqlens = torch.tensor([0, 1, 4], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([4, 1], device="cuda", dtype=torch.int32)
    decay_bias = torch.linspace(
        -0.1,
        0.1,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.float16,
    ).reshape(1, HEAD_SIZE)
    elapsed_t = torch.tensor(
        [5, 17, 29, 41, 53], device="cuda", dtype=torch.int32
    )
    elapsed_before = elapsed_t.clone()
    initial_state = 0.01 * torch.randn(
        5, 1, HEAD_SIZE, HEAD_SIZE, device="cuda", dtype=torch.float16
    )
    dithered_log_decay = _dithered_log_decay(
        inputs[1],
        decay_bias,
        cu_seqlens,
        state_indices,
        elapsed_t,
    )
    expected_output, expected_pool = rwkv7_reference(
        inputs[0],
        dithered_log_decay,
        *inputs[2:],
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
    )
    state_pool = initial_state.clone()

    actual_output = rwkv7_recurrent_stateful(
        *inputs,
        state_pool=state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        mode="fp16",
        decay_bias=decay_bias,
        elapsed_t=elapsed_t,
    )
    torch.cuda.synchronize()

    assert torch.equal(elapsed_t, elapsed_before)
    _assert_relative_rmse_close(
        actual_output,
        expected_output,
        threshold=RECURRENT_TOLERANCES["fp16"]["output_relative_rmse"],
    )
    _assert_relative_rmse_close(
        state_pool,
        expected_pool,
        threshold=RECURRENT_TOLERANCES["fp16"]["state_relative_rmse"],
    )


def test_fp32io16_rejects_elapsed_t_before_launch() -> None:
    inputs = _inputs(batch_size=1, sequence_length=2, seed=109)
    cu_seqlens = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    state_pool = torch.zeros(
        2, 1, HEAD_SIZE, HEAD_SIZE, device="cuda", dtype=torch.float32
    )
    elapsed_t = torch.zeros(2, device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="valid only for mode='fp16'"):
        rwkv7_recurrent_stateful(
            *inputs,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode="fp32io16",
            elapsed_t=elapsed_t,
        )


def test_fp16_mode_rejects_non_fp16_tokens_before_cuda_launch() -> None:
    inputs = tuple(
        tensor.float()
        for tensor in _inputs(batch_size=1, sequence_length=1)
    )
    with pytest.raises(TypeError, match="requires fp16 token tensors"):
        rwkv7(*inputs, algorithm="recurrent", mode="fp16")
