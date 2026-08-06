# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest
import torch

import flash_rwkv
from flash_rwkv.tmix.wkv7 import pretrain_recurrent_bf16


ROOT = Path(__file__).resolve().parents[4]
W_SCALE = -0.6065306597
UPSTREAM_BODY_HASHES = {
    "pretrain_recurrent_bf16_forward.cpp": "a3def05547be8bf79d81d76c4d4795958c712e42ef94fb5d4161a66b00b0cb74",
    "pretrain_recurrent_bf16_forward.cu": "2f36786a261582198f3ec2deca7b2af241a364f2fae3045b19731a0149109083",
}
AVAILABLE = (
    torch.cuda.is_available()
    and flash_rwkv._C is not None
    and hasattr(torch.ops.rwkv7_clampw_v3, "forward")
    and hasattr(torch.ops.rwkv7_clampw_v3, "backward")
)


def _require_cuda() -> None:
    if not AVAILABLE:
        pytest.skip("CUDA extension with canonical clampw v3 bindings is required")


def _case(
    batch: int,
    tokens: int,
    heads: int,
    *,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260806 + batch + tokens + heads)
    shape = (batch, tokens, heads * 64)
    values = tuple(
        (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.05)
        .contiguous()
        .requires_grad_(requires_grad)
        for _ in range(6)
    )
    return values


def _reference(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    r, w, k, v, a, b = values
    batch, tokens, channels = r.shape
    heads = channels // 64
    r4, w4, k4, v4, a4, b4 = (
        tensor.view(batch, tokens, heads, 64).float()
        for tensor in (r, w, k, v, a, b)
    )
    state = torch.zeros((batch, heads, 64, 64), device=r.device)
    outputs = []
    for token in range(tokens):
        retention = torch.exp(W_SCALE / (1.0 + torch.exp(-w4[:, token])))
        state_dot_a = torch.einsum("bhkv,bhk->bhv", state, a4[:, token])
        state = (
            state * retention.unsqueeze(-1)
            + b4[:, token].unsqueeze(-1) * state_dot_a.unsqueeze(-2)
            + k4[:, token].unsqueeze(-1) * v4[:, token].unsqueeze(-2)
        )
        outputs.append(
            torch.einsum("bhkv,bhk->bhv", state, r4[:, token])
        )
    return torch.stack(outputs, dim=1).reshape(batch, tokens, channels)


def test_native_bodies_match_pinned_train_temp_sources() -> None:
    native_root = ROOT / "csrc/sm90/tmix/wkv7"
    for filename, expected_hash in UPSTREAM_BODY_HASHES.items():
        lines = (native_root / filename).read_bytes().splitlines(keepends=True)
        body = b"".join(lines[6:])
        assert hashlib.sha256(body).hexdigest() == expected_hash


def test_public_contract_matches_clampw_v3() -> None:
    signature = inspect.signature(pretrain_recurrent_bf16)
    assert tuple(signature.parameters) == ("r", "w", "k", "v", "a", "b")
    assert flash_rwkv.pretrain_recurrent_bf16 is pretrain_recurrent_bf16
    assert "pretrain_recurrent_bf16" in flash_rwkv.__all__
    assert "pretrain_recurrent_fp32io16" not in flash_rwkv.__all__
    assert not hasattr(flash_rwkv, "pretrain_recurrent_fp32io16")


@pytest.mark.parametrize("batch,tokens,heads", [(1, 16, 1), (2, 32, 2)])
def test_forward_matches_clampw_recurrence(
    batch: int, tokens: int, heads: int
) -> None:
    _require_cuda()
    values = _case(batch, tokens, heads)
    actual = pretrain_recurrent_bf16(*values)
    expected = _reference(values)
    assert actual.shape == values[0].shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual.float(), expected, atol=0.02, rtol=0.02)
    assert torch.equal(actual, pretrain_recurrent_bf16(*values))


def test_backward_matches_clampw_recurrence() -> None:
    _require_cuda()
    values = _case(1, 16, 1, requires_grad=True)
    actual = pretrain_recurrent_bf16(*values)
    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    actual_gradients = [value.grad.detach().float() for value in values]

    reference_values = tuple(
        value.detach().clone().requires_grad_() for value in values
    )
    expected = _reference(reference_values)
    expected.backward(grad_output.float())
    expected_gradients = [value.grad.detach().float() for value in reference_values]
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients
    ):
        assert torch.isfinite(actual_gradient).all()
        assert torch.allclose(
            actual_gradient, expected_gradient, atol=0.03, rtol=0.05
        )


def test_invalid_input_contract_is_rejected() -> None:
    _require_cuda()
    valid = list(_case(1, 16, 1))

    with pytest.raises(TypeError, match="bfloat16"):
        pretrain_recurrent_bf16(valid[0].float(), *valid[1:])
    with pytest.raises(ValueError, match="T must be divisible"):
        short = [tensor[:, :15].contiguous() for tensor in valid]
        pretrain_recurrent_bf16(*short)
    with pytest.raises(ValueError, match="C must be divisible"):
        narrow = [tensor[:, :, :63].contiguous() for tensor in valid]
        pretrain_recurrent_bf16(*narrow)
    with pytest.raises(ValueError, match="must match r shape"):
        pretrain_recurrent_bf16(
            valid[0], valid[1][:, :, :32].contiguous(), *valid[2:]
        )
    with pytest.raises(ValueError, match="contiguous CUDA"):
        noncontiguous = valid[0].transpose(1, 2)
        pretrain_recurrent_bf16(noncontiguous, *valid[1:])
    with pytest.raises(ValueError, match="contiguous CUDA"):
        cpu = tuple(tensor.cpu() for tensor in valid)
        pretrain_recurrent_bf16(*cpu)
