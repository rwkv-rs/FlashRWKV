# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

import flash_rwkv
from flash_rwkv.tmix.wkv7 import pretrain_recurrent_fp32io16


ROOT = Path(__file__).resolve().parents[4]
AVAILABLE = (
    torch.cuda.is_available()
    and flash_rwkv._C is not None
    and hasattr(flash_rwkv._C, "pretrain_recurrent_fp32io16_forward")
    and hasattr(flash_rwkv._C, "pretrain_recurrent_fp32io16_backward")
)


def _require_cuda() -> None:
    if not AVAILABLE:
        pytest.skip("CUDA extension with train_temp recurrent bindings is required")


def _case(requires_grad: bool = False) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260805)
    device = torch.device("cuda")
    B, H, D, total = 2, 1, 64, 5
    state = torch.randn(B, H, D, D, device=device).mul_(0.01)
    values = [torch.randn(total, H, D, device=device, dtype=torch.float16).mul_(0.01) for _ in range(6)]
    if requires_grad:
        state.requires_grad_()
        values = [value.requires_grad_() for value in values]
    sequence_chunk_offsets = torch.tensor([0, 2, 3], device=device, dtype=torch.int32)
    chunk_token_starts = torch.tensor([0, 1, 3], device=device, dtype=torch.int32)
    chunk_token_ends = torch.tensor([1, 3, 5], device=device, dtype=torch.int32)
    return (state, sequence_chunk_offsets, chunk_token_starts, chunk_token_ends, *values)


def _reference(case: tuple[torch.Tensor, ...], scale: float) -> tuple[torch.Tensor, ...]:
    state, sequence, starts, ends, r, decay_logits, k, v, a, b = case
    B, H, D, _ = state.shape
    device = state.device
    retention = lambda logits: torch.exp2(
        torch.tensor(-0.8750387749145276, device=device)
        / (1.0 + torch.exp2(torch.tensor(-1.4426950408889634, device=device) * logits))
    )
    states: list[torch.Tensor] = []
    rows: list[torch.Tensor] = []
    boundaries: list[torch.Tensor] = []
    state_dot_a: list[torch.Tensor] = []
    for sequence_index in range(B):
        current = state[sequence_index, 0].float()
        for chunk_index in range(int(sequence[sequence_index]), int(sequence[sequence_index + 1])):
            boundaries.append(current.clone())
            for token_index in range(int(starts[chunk_index]), int(ends[chunk_index])):
                rr = r[token_index, 0].float()
                dd = retention(decay_logits[token_index, 0].float())
                kk = k[token_index, 0].float()
                vv = v[token_index, 0].float()
                aa = a[token_index, 0].float()
                bb = b[token_index, 0].float()
                dot = (aa[:, None] * current).sum(dim=0)
                state_dot_a.append(dot)
                current = dd[:, None] * current + bb[:, None] * dot[None, :] + kk[:, None] * vv[None, :]
                rows.append((float(scale) * (rr[:, None] * current).sum(dim=0)).to(v.dtype))
        states.append(current)
    return (
        torch.stack(rows).view(total := r.shape[0], H, D),
        torch.stack(states).view(B, H, D, D),
        torch.stack(boundaries).view(len(boundaries), H, D, D),
        torch.stack(state_dot_a).view(r.shape[0], H, D),
    )


def test_training_public_contract_is_raw_decay_logits_only() -> None:
    signature = inspect.signature(pretrain_recurrent_fp32io16)
    assert "decay_logits" in signature.parameters
    assert "initial_state" in signature.parameters
    assert "log_decay" not in signature.parameters
    assert "state_pool" not in signature.parameters
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "csrc/sm90/tmix/wkv7/pretrain_recurrent_fp32io16_forward.cpp",
            ROOT / "csrc/sm90/tmix/wkv7/pretrain_recurrent_fp32io16_forward.cu",
            ROOT / "csrc/sm90/tmix/wkv7/pretrain_recurrent_fp32io16_backward.cpp",
            ROOT / "csrc/sm90/tmix/wkv7/pretrain_recurrent_fp32io16_backward.cu",
        )
    )
    assert "log_decay" not in source
    assert "pretrain_common" not in source


def test_forward_matches_train_temp_recurrence_and_checkpoints() -> None:
    _require_cuda()
    case = _case()
    state, sequence, starts, ends, r, decay_logits, k, v, a, b = case
    actual = pretrain_recurrent_fp32io16(
        state, sequence, starts, ends, r, decay_logits, k, v, a, b, scale=0.75
    )
    expected = _reference(case, 0.75)
    assert torch.allclose(actual[0], expected[0], atol=3e-4, rtol=3e-4)
    assert torch.allclose(actual[1], expected[1], atol=3e-4, rtol=3e-4)
    assert torch.allclose(actual[2], expected[2], atol=3e-4, rtol=3e-4)
    assert torch.allclose(actual[3], expected[3], atol=3e-4, rtol=3e-4)


def test_backward_matches_reference_and_returns_initial_state_gradient() -> None:
    _require_cuda()
    case = _case(requires_grad=True)
    actual = pretrain_recurrent_fp32io16(*case, scale=0.75)
    (actual[0].float().square().sum() + actual[1].square().sum()).backward()
    actual_gradients = [case[0].grad.detach().clone(), *[value.grad.detach().clone() for value in case[4:]]]

    reference_case = tuple(
        value.detach().clone().requires_grad_()
        if value.is_floating_point()
        else value.detach().clone()
        for value in case
    )
    expected = _reference(reference_case, 0.75)
    (expected[0].float().square().sum() + expected[1].square().sum()).backward()
    expected_gradients = [reference_case[0].grad, *[value.grad for value in reference_case[4:]]]
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.allclose(actual_gradient, expected_gradient, atol=3e-4, rtol=3e-3)
        assert torch.isfinite(actual_gradient).all()


@pytest.mark.parametrize("bad_index", [0, 1, 2])
def test_invalid_chunk_metadata_is_rejected(bad_index: int) -> None:
    _require_cuda()
    case = list(_case())
    if bad_index == 0:
        case[1] = torch.tensor([0, 2, 2], device="cuda", dtype=torch.int32)
    elif bad_index == 1:
        case[2] = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
    else:
        case[3] = torch.tensor([1, 1, 5], device="cuda", dtype=torch.int32)
    with pytest.raises((ValueError, RuntimeError)):
        pretrain_recurrent_fp32io16(*case)
