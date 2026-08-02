# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flash_rwkv import pretrain_cmix_bf16


def _inputs(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    channels: int = 8,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1234)
    return (
        torch.randn(2, 3, channels, device=device, dtype=dtype).mul_(0.25),
        torch.rand(channels, device=device, dtype=dtype),
        torch.randn(4 * channels, channels, device=device, dtype=dtype).mul_(0.1),
        torch.randn(channels, 4 * channels, device=device, dtype=dtype).mul_(0.1),
    )


def _reference(
    x: torch.Tensor,
    x_k: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
) -> torch.Tensor:
    previous = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
    mixed = x + (previous - x) * x_k
    activation = torch.relu(mixed @ key_weight.T).square()
    return activation @ value_weight.T


def test_channel_mix_rejects_non_cuda_inputs_before_loading_extension() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        pretrain_cmix_bf16(*_inputs(device="cpu"))


def test_channel_mix_rejects_non_bf16_inputs() -> None:
    with pytest.raises(TypeError, match="torch.bfloat16"):
        pretrain_cmix_bf16(*_inputs(device="cpu", dtype=torch.float32))


def test_channel_mix_rejects_incompatible_weight_shapes() -> None:
    x, x_k, key_weight, value_weight = _inputs(device="cpu")
    with pytest.raises(ValueError, match="key_weight"):
        pretrain_cmix_bf16(x, x_k, key_weight[:-1], value_weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_channel_mix_forward_and_gradients_match_torch_reference() -> None:
    native_inputs = tuple(
        tensor.requires_grad_(True) for tensor in _inputs(device="cuda")
    )
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in native_inputs
    )
    grad_output = torch.randn_like(native_inputs[0])

    output = pretrain_cmix_bf16(*native_inputs)
    reference = _reference(*reference_inputs)
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
            atol=0.006,
            rtol=0.05,
        )
