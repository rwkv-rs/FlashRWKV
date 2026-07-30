# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from flash_rwkv import (
    decay_logits_to_log_decay,
    rwkv7,
    rwkv7_from_decay_logits,
    rwkv7_reference,
)


def _inputs(
    *,
    batch_size: int,
    sequence_length: int,
    num_heads: int = 2,
    key_size: int = 4,
    value_size: int = 3,
    seed: int = 42,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    key_shape = (batch_size, sequence_length, num_heads, key_size)
    value_shape = (batch_size, sequence_length, num_heads, value_size)
    r = torch.randn(key_shape, generator=generator)
    log_decay = -torch.rand(key_shape, generator=generator)
    k = torch.randn(key_shape, generator=generator)
    v = torch.randn(value_shape, generator=generator)
    a = torch.randn(key_shape, generator=generator)
    b = torch.randn(key_shape, generator=generator)
    return r, log_decay, k, v, a, b


def _manual_recurrence(
    inputs: tuple[torch.Tensor, ...],
    initial_state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r, log_decay, k, v, a, b = inputs
    state = initial_state.clone()
    outputs = []
    for token_index in range(r.shape[1]):
        previous = state
        a_state = torch.einsum("bhk,bhkv->bhv", a[:, token_index], previous)
        state = (
            log_decay[:, token_index].exp().unsqueeze(-1) * previous
            + b[:, token_index].unsqueeze(-1) * a_state.unsqueeze(-2)
            + k[:, token_index].unsqueeze(-1) * v[:, token_index].unsqueeze(-2)
        )
        outputs.append(
            scale * torch.einsum("bhk,bhkv->bhv", r[:, token_index], state)
        )
    return torch.stack(outputs, dim=1), state


def test_reference_uses_canonical_key_value_state_orientation() -> None:
    r = torch.tensor([[[[2.0, -1.0]]]])
    log_decay = torch.log(torch.tensor([[[[0.5, 0.25]]]]))
    k = torch.tensor([[[[-1.0, 2.0]]]])
    v = torch.tensor([[[[0.5, -1.0, 2.0]]]])
    a = torch.tensor([[[[1.0, -2.0]]]])
    b = torch.tensor([[[[3.0, 4.0]]]])
    inputs = (r, log_decay, k, v, a, b)
    initial_state = torch.tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 7.0]]]]
    )

    expected_output, expected_state = _manual_recurrence(
        inputs, initial_state, scale=0.75
    )
    output, final_state = rwkv7_reference(
        *inputs,
        scale=0.75,
        initial_state=initial_state,
        output_final_state=True,
    )

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(final_state, expected_state)


def test_packed_sequences_match_independent_fixed_calls_with_tail_lengths() -> None:
    sequence_lengths = (1, 15, 16, 17)
    inputs = _inputs(batch_size=1, sequence_length=sum(sequence_lengths))
    initial_state = torch.randn(len(sequence_lengths), 2, 4, 3)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()], dtype=torch.int32
    )

    packed_output, packed_state = rwkv7_reference(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    separate_outputs = []
    separate_states = []
    start = 0
    for sequence_index, sequence_length in enumerate(sequence_lengths):
        end = start + sequence_length
        sliced_inputs = tuple(tensor[:, start:end] for tensor in inputs)
        output, state = rwkv7_reference(
            *sliced_inputs,
            initial_state=initial_state[sequence_index : sequence_index + 1],
            output_final_state=True,
        )
        separate_outputs.append(output)
        separate_states.append(state)
        start = end

    torch.testing.assert_close(packed_output, torch.cat(separate_outputs, dim=1))
    torch.testing.assert_close(packed_state, torch.cat(separate_states, dim=0))


def test_slot_mapped_state_pool_updates_only_selected_rows() -> None:
    sequence_lengths = (2, 3, 1)
    inputs = _inputs(batch_size=1, sequence_length=sum(sequence_lengths), seed=7)
    cu_seqlens = torch.tensor([0, 2, 5, 6], dtype=torch.int64)
    state_pool = torch.randn(6, 2, 4, 3)
    state_indices = torch.tensor([4, 1, 5], dtype=torch.int32)

    output, updated_pool = rwkv7_reference(
        *inputs,
        initial_state=state_pool,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
    )
    expected_output, expected_states = rwkv7_reference(
        *inputs,
        initial_state=state_pool.index_select(0, state_indices.to(torch.long)),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    expected_pool = state_pool.index_copy(
        0, state_indices.to(torch.long), expected_states
    )

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(updated_pool, expected_pool)
    untouched = torch.tensor([0, 2, 3])
    assert torch.equal(
        updated_pool.index_select(0, untouched),
        state_pool.index_select(0, untouched),
    )


@pytest.mark.parametrize(
    ("cu_seqlens", "state_indices", "state_rows", "match"),
    [
        ([1, 4], None, 1, "start at 0"),
        ([0, 2, 2, 4], None, 3, "strictly increasing"),
        ([0, 3], None, 1, "packed token count"),
        ([0, 2, 4], [0, 0], 2, "must be unique"),
        ([0, 2, 4], [0, 2], 2, "within the state pool"),
    ],
)
def test_invalid_packed_metadata_fails_before_execution(
    cu_seqlens: list[int],
    state_indices: list[int] | None,
    state_rows: int,
    match: str,
) -> None:
    inputs = _inputs(batch_size=1, sequence_length=4)
    initial_state = torch.randn(state_rows, 2, 4, 3)

    with pytest.raises(ValueError, match=match):
        rwkv7_reference(
            *inputs,
            initial_state=initial_state,
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
            state_indices=(
                None
                if state_indices is None
                else torch.tensor(state_indices, dtype=torch.int64)
            ),
        )


def test_packed_layout_rejects_batch_greater_than_one() -> None:
    inputs = _inputs(batch_size=2, sequence_length=2)
    with pytest.raises(ValueError, match="B must be 1"):
        rwkv7_reference(
            *inputs,
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        )


def test_decay_logits_adapter_preserves_output_state_and_gradient() -> None:
    r, _, k, v, a, b = _inputs(
        batch_size=1,
        sequence_length=3,
        num_heads=1,
        key_size=3,
        value_size=2,
    )
    logits_adapter = torch.randn(1, 3, 1, 3, requires_grad=True)
    logits_explicit = logits_adapter.detach().clone().requires_grad_(True)
    initial_adapter = torch.randn(1, 1, 3, 2, requires_grad=True)
    initial_explicit = initial_adapter.detach().clone().requires_grad_(True)

    output_adapter, state_adapter = rwkv7_from_decay_logits(
        r,
        logits_adapter,
        k,
        v,
        a,
        b,
        initial_state=initial_adapter,
        output_final_state=True,
    )
    explicit_log_decay = -math.exp(-0.5) * torch.sigmoid(logits_explicit)
    output_explicit, state_explicit = rwkv7(
        r,
        explicit_log_decay,
        k,
        v,
        a,
        b,
        initial_state=initial_explicit,
        output_final_state=True,
    )

    output_weight = torch.randn_like(output_adapter)
    state_weight = torch.randn_like(state_adapter)
    loss_adapter = (
        (output_adapter * output_weight).sum()
        + (state_adapter * state_weight).sum()
    )
    loss_explicit = (
        (output_explicit * output_weight).sum()
        + (state_explicit * state_weight).sum()
    )
    adapter_gradients = torch.autograd.grad(
        loss_adapter, (logits_adapter, initial_adapter)
    )
    explicit_gradients = torch.autograd.grad(
        loss_explicit, (logits_explicit, initial_explicit)
    )

    torch.testing.assert_close(output_adapter, output_explicit)
    torch.testing.assert_close(state_adapter, state_explicit)
    for actual, expected in zip(adapter_gradients, explicit_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


def test_output_is_unchanged_when_final_state_is_not_requested() -> None:
    inputs = _inputs(batch_size=2, sequence_length=3)
    initial_state = torch.randn(2, 2, 4, 3)
    output_with_state, final_state = rwkv7_reference(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
    )
    output_only, absent_state = rwkv7_reference(
        *inputs,
        initial_state=initial_state,
        output_final_state=False,
    )

    torch.testing.assert_close(output_only, output_with_state)
    assert final_state is not None
    assert absent_state is None


def test_tolerance_fixture_is_versioned() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "tolerances-v1.json"
    fixture = json.loads(fixture_path.read_text())

    assert fixture["schema_version"] == 1
    assert fixture["fp32io16_recurrent"]["output_relative_rmse"] == 0.002
    assert fixture["fp16"]["report_only_until_task"] == "2.2"


def test_decay_logit_transform_matches_the_training_formula() -> None:
    logits = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
    actual = decay_logits_to_log_decay(logits)
    expected = -math.exp(-0.5) * torch.sigmoid(logits)

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected_gradient = (
        -math.exp(-0.5)
        * torch.sigmoid(logits.detach())
        * (1.0 - torch.sigmoid(logits.detach()))
    )
    torch.testing.assert_close(logits.grad, expected_gradient)
