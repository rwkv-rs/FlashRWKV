# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flash_rwkv import (
    pretrain_tmix_a_gate_bf16,
    pretrain_tmix_kk_pre_bf16,
    pretrain_tmix_lnx_rkvres_xg_bf16,
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


def _kk_pre_inputs(
    *,
    device: torch.device | str,
    channels: int = 64,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(915)
    return (
        torch.randn(1, 3, channels, device=device, dtype=torch.bfloat16).mul_(0.2),
        torch.rand(channels, device=device, dtype=torch.bfloat16),
        torch.sigmoid(torch.randn(1, 3, channels, device=device, dtype=torch.bfloat16)),
        torch.rand(channels, device=device, dtype=torch.bfloat16),
    )


def test_time_mix_kk_pre_rejects_non_cuda_inputs_before_extension() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        pretrain_tmix_kk_pre_bf16(*_kk_pre_inputs(device="cpu"))


def test_time_mix_kk_pre_rejects_non_64_head_geometry() -> None:
    with pytest.raises(ValueError, match="divisible by 64"):
        pretrain_tmix_kk_pre_bf16(*_kk_pre_inputs(device="cpu", channels=96))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_time_mix_kk_pre_outputs_and_gradients_match_torch_reference() -> None:
    native_inputs = tuple(
        tensor.requires_grad_(True) for tensor in _kk_pre_inputs(device="cuda")
    )
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in native_inputs
    )
    grad_outputs = tuple(torch.randn_like(native_inputs[0]) for _ in range(3))

    outputs = pretrain_tmix_kk_pre_bf16(*native_inputs)
    key, key_scale, learning_rate, learning_rate_scale = reference_inputs
    direction = key * key_scale
    direction = (
        torch.nn.functional.normalize(
            direction.view(1, 3, 1, 64).float(),
            dim=-1,
            eps=1e-12,
        )
        .view_as(key)
        .to(torch.bfloat16)
    )
    references = (
        key * (1 + (learning_rate - 1) * learning_rate_scale),
        -direction,
        direction * learning_rate,
    )
    torch.autograd.backward(outputs, grad_outputs)
    torch.autograd.backward(references, grad_outputs)

    for output, reference in zip(outputs, references, strict=True):
        torch.testing.assert_close(output, reference, atol=0.004, rtol=0.03)
    for gradient, reference_gradient in zip(
        (tensor.grad for tensor in native_inputs),
        (tensor.grad for tensor in reference_inputs),
        strict=True,
    ):
        torch.testing.assert_close(
            gradient,
            reference_gradient,
            atol=0.015,
            rtol=0.08,
        )


def _lnx_inputs(
    *,
    device: torch.device | str,
    channels: int = 64,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(916)
    features = tuple(
        torch.randn(1, 3, channels, device=device, dtype=torch.bfloat16).mul_(0.2)
        for _ in range(4)
    )
    residual_scale = torch.randn(
        channels // 64,
        64,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.1)
    norm_weight = torch.rand(channels, device=device, dtype=torch.bfloat16)
    norm_bias = torch.randn(channels, device=device, dtype=torch.bfloat16).mul_(0.1)
    gate = torch.sigmoid(
        torch.randn(1, 3, channels, device=device, dtype=torch.bfloat16)
    )
    return *features, residual_scale, norm_weight, norm_bias, gate


def test_time_mix_lnx_rejects_non_cuda_inputs_before_extension() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        pretrain_tmix_lnx_rkvres_xg_bf16(*_lnx_inputs(device="cpu"))


def test_time_mix_lnx_rejects_non_64_head_geometry() -> None:
    with pytest.raises(ValueError, match="divisible by 64"):
        pretrain_tmix_lnx_rkvres_xg_bf16(*_lnx_inputs(device="cpu", channels=96))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_time_mix_lnx_output_and_gradients_match_torch_reference() -> None:
    native_inputs = tuple(
        tensor.requires_grad_(True) for tensor in _lnx_inputs(device="cuda")
    )
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in native_inputs
    )
    grad_output = torch.randn_like(native_inputs[0])

    output = pretrain_tmix_lnx_rkvres_xg_bf16(*native_inputs)
    recurrent_output, receptance, key, value, residual_scale, weight, bias, gate = (
        reference_inputs
    )
    normalized = torch.nn.functional.group_norm(
        recurrent_output.float().reshape(-1, 64),
        num_groups=1,
        weight=weight.float(),
        bias=bias.float(),
        eps=64e-5,
    ).reshape_as(recurrent_output)
    residual = (
        receptance.float().view(1, 3, 1, 64)
        * key.float().view(1, 3, 1, 64)
        * residual_scale.float()
    ).sum(dim=-1, keepdim=True) * value.float().view(1, 3, 1, 64)
    reference = ((normalized + residual.view_as(normalized)) * gate.float()).to(
        torch.bfloat16
    )
    output.backward(grad_output)
    reference.backward(grad_output)

    torch.testing.assert_close(output, reference, atol=0.006, rtol=0.04)
    for gradient, reference_gradient in zip(
        (tensor.grad for tensor in native_inputs),
        (tensor.grad for tensor in reference_inputs),
        strict=True,
    ):
        torch.testing.assert_close(
            gradient,
            reference_gradient,
            atol=0.02,
            rtol=0.1,
        )
