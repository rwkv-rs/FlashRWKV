# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flash_rwkv import (
    pretrain_tmix_a_gate_bf16,
    pretrain_tmix_mix6_bf16,
    pretrain_tmix_vres_gate_bf16,
)


def _inputs(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    channels: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(912)
    return (
        torch.randn(channels, device=device, dtype=dtype).mul_(0.25),
        torch.randn(2, 3, channels, device=device, dtype=dtype).mul_(0.25),
    )


def test_time_mix_a_gate_rejects_non_cuda_inputs_before_loading_extension() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        pretrain_tmix_a_gate_bf16(*_inputs(device="cpu"))


def test_time_mix_a_gate_rejects_non_bf16_inputs() -> None:
    with pytest.raises(TypeError, match="torch.bfloat16"):
        pretrain_tmix_a_gate_bf16(*_inputs(device="cpu", dtype=torch.float32))


def test_time_mix_a_gate_rejects_incompatible_shapes() -> None:
    a0, a12 = _inputs(device="cpu")
    with pytest.raises(ValueError, match="a0 must have shape"):
        pretrain_tmix_a_gate_bf16(a0[:-1], a12)


@pytest.mark.parametrize("channels", [7, 8])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_time_mix_a_gate_forward_and_gradients_match_torch_reference(
    channels: int,
) -> None:
    native_inputs = tuple(
        tensor.requires_grad_(True)
        for tensor in _inputs(device="cuda", channels=channels)
    )
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in native_inputs
    )
    grad_output = torch.randn_like(native_inputs[1])

    output = pretrain_tmix_a_gate_bf16(*native_inputs)
    reference = torch.sigmoid(reference_inputs[0] + reference_inputs[1])
    output.backward(grad_output)
    reference.backward(grad_output)

    torch.testing.assert_close(output, reference, atol=0.003, rtol=0.02)
    for gradient, reference_gradient in zip(
        (tensor.grad for tensor in native_inputs),
        (tensor.grad for tensor in reference_inputs),
        strict=True,
    ):
        torch.testing.assert_close(
            gradient,
            reference_gradient,
            atol=0.008,
            rtol=0.05,
        )


def _vres_inputs(
    *,
    device: torch.device | str,
    channels: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(913)
    return (
        torch.randn(2, 3, channels, device=device, dtype=torch.bfloat16).mul_(0.25),
        torch.randn(2, 3, channels, device=device, dtype=torch.bfloat16).mul_(0.25),
        torch.randn(channels, device=device, dtype=torch.bfloat16).mul_(0.25),
        torch.randn(2, 3, channels, device=device, dtype=torch.bfloat16).mul_(0.25),
    )


def test_time_mix_vres_gate_rejects_non_cuda_inputs_before_extension() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        pretrain_tmix_vres_gate_bf16(*_vres_inputs(device="cpu", channels=8))


def test_time_mix_vres_gate_rejects_shape_mismatch() -> None:
    value, first_value, v0, v12 = _vres_inputs(device="cpu", channels=8)
    with pytest.raises(ValueError, match="first_value"):
        pretrain_tmix_vres_gate_bf16(value, first_value[:, :-1], v0, v12)


@pytest.mark.parametrize("channels", [7, 8])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_time_mix_vres_gate_forward_and_gradients_match_torch_reference(
    channels: int,
) -> None:
    native_inputs = tuple(
        tensor.requires_grad_(True)
        for tensor in _vres_inputs(device="cuda", channels=channels)
    )
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in native_inputs
    )
    grad_output = torch.randn_like(native_inputs[0])

    output = pretrain_tmix_vres_gate_bf16(*native_inputs)
    value, first_value, v0, v12 = reference_inputs
    gate = torch.sigmoid(v0 + v12)
    reference = value + (first_value - value) * gate
    output.backward(grad_output)
    reference.backward(grad_output)

    torch.testing.assert_close(output, reference, atol=0.004, rtol=0.02)
    for gradient, reference_gradient in zip(
        (tensor.grad for tensor in native_inputs),
        (tensor.grad for tensor in reference_inputs),
        strict=True,
    ):
        torch.testing.assert_close(
            gradient,
            reference_gradient,
            atol=0.01,
            rtol=0.05,
        )


def _mix6_inputs(
    *,
    device: torch.device | str,
    channels: int = 8,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(914)
    x = torch.randn(2, 3, channels, device=device, dtype=torch.bfloat16).mul_(0.25)
    mixes = tuple(
        torch.rand(channels, device=device, dtype=torch.bfloat16) for _ in range(6)
    )
    return x, *mixes


def test_time_mix_mix6_rejects_non_cuda_inputs_before_extension() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        pretrain_tmix_mix6_bf16(*_mix6_inputs(device="cpu"))


def test_time_mix_mix6_rejects_odd_channels() -> None:
    with pytest.raises(ValueError, match="even channel"):
        pretrain_tmix_mix6_bf16(*_mix6_inputs(device="cpu", channels=7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_time_mix_mix6_outputs_and_gradients_match_torch_reference() -> None:
    native_inputs = tuple(
        tensor.requires_grad_(True) for tensor in _mix6_inputs(device="cuda")
    )
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in native_inputs
    )
    grad_outputs = tuple(torch.randn_like(native_inputs[0]) for _ in range(6))

    outputs = pretrain_tmix_mix6_bf16(*native_inputs)
    x, *mixes = reference_inputs
    previous = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
    delta = previous - x
    references = tuple(x + delta * mix for mix in mixes)
    torch.autograd.backward(outputs, grad_outputs)
    torch.autograd.backward(references, grad_outputs)

    for output, reference in zip(outputs, references, strict=True):
        torch.testing.assert_close(output, reference, atol=0.003, rtol=0.02)
    for gradient, reference_gradient in zip(
        (tensor.grad for tensor in native_inputs),
        (tensor.grad for tensor in reference_inputs),
        strict=True,
    ):
        torch.testing.assert_close(
            gradient,
            reference_gradient,
            atol=0.012,
            rtol=0.06,
        )
