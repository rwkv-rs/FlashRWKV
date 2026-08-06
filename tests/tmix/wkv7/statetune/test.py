# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flash_rwkv.tmix.wkv7.statetune import statetune_recurrent_fp32io16


def _case(requires_grad: bool = False) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260805)
    device = torch.device("cuda")
    batch, heads, head_size, total_tokens = 2, 1, 64, 5
    state = torch.randn(
        batch, heads, head_size, head_size, device=device
    ).mul_(0.01)
    values = [
        torch.randn(
            total_tokens, heads, head_size, device=device, dtype=torch.float16
        ).mul_(0.01)
        for _ in range(6)
    ]
    if requires_grad:
        state.requires_grad_()
        values = [value.requires_grad_() for value in values]
    sequence_chunk_offsets = torch.tensor([0, 2, 3], device=device, dtype=torch.int32)
    chunk_token_starts = torch.tensor([0, 1, 3], device=device, dtype=torch.int32)
    chunk_token_ends = torch.tensor([1, 3, 5], device=device, dtype=torch.int32)
    return (
        state,
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        *values,
    )


def _reference(
    case: tuple[torch.Tensor, ...], scale: float
) -> tuple[torch.Tensor, ...]:
    state, sequence, starts, ends, r, decay_logits, k, v, a, b = case
    batch, heads, head_size, _ = state.shape
    device = state.device
    retention = lambda logits: torch.exp2(
        torch.tensor(-0.8750387749145276, device=device)
        / (
            1.0
            + torch.exp2(
                torch.tensor(-1.4426950408889634, device=device) * logits
            )
        )
    )
    states: list[torch.Tensor] = []
    rows: list[torch.Tensor] = []
    boundaries: list[torch.Tensor] = []
    state_dot_a: list[torch.Tensor] = []
    for sequence_index in range(batch):
        current = state[sequence_index, 0].float()
        for chunk_index in range(
            int(sequence[sequence_index]), int(sequence[sequence_index + 1])
        ):
            boundaries.append(current.clone())
            for token_index in range(
                int(starts[chunk_index]), int(ends[chunk_index])
            ):
                rr = r[token_index, 0].float()
                decay = retention(decay_logits[token_index, 0].float())
                kk = k[token_index, 0].float()
                vv = v[token_index, 0].float()
                aa = a[token_index, 0].float()
                bb = b[token_index, 0].float()
                dot = (aa[:, None] * current).sum(dim=0)
                state_dot_a.append(dot)
                current = (
                    decay[:, None] * current
                    + bb[:, None] * dot[None, :]
                    + kk[:, None] * vv[None, :]
                )
                rows.append((float(scale) * (rr[:, None] * current).sum(dim=0)).to(v.dtype))
        states.append(current)
    return (
        torch.stack(rows).view(r.shape[0], heads, head_size),
        torch.stack(states).view(batch, heads, head_size, head_size),
        torch.stack(boundaries).view(len(boundaries), heads, head_size, head_size),
        torch.stack(state_dot_a).view(r.shape[0], heads, head_size),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statetune_matches_train_temp_recurrence_and_initial_gradient() -> None:
    case = _case(requires_grad=True)
    state, sequence, starts, ends, r, decay_logits, k, v, a, b = case
    initial_state = state.detach().clone()
    actual = statetune_recurrent_fp32io16(
        state, sequence, starts, ends, r, decay_logits, k, v, a, b, scale=0.75
    )

    reference_case = tuple(
        value.detach().clone().requires_grad_()
        if value.is_floating_point()
        else value.detach().clone()
        for value in case
    )
    expected = _reference(reference_case, 0.75)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert torch.allclose(actual_value, expected_value, atol=3e-4, rtol=3e-4)

    (actual[0].float().square().sum() + actual[1].square().sum()).backward()
    actual_gradients = [
        state.grad.detach().clone(),
        *[value.grad.detach().clone() for value in (r, decay_logits, k, v, a, b)],
    ]

    (expected[0].float().square().sum() + expected[1].square().sum()).backward()
    expected_gradients = [
        reference_case[0].grad,
        *[value.grad for value in reference_case[4:]],
    ]
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.allclose(actual_gradient, expected_gradient, atol=3e-4, rtol=3e-3)
        assert torch.isfinite(actual_gradient).all()

    # StateTune must not mutate the caller-owned initial state during forward.
    assert torch.equal(state.detach(), initial_state)
