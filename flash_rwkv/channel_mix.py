# SPDX-License-Identifier: MIT

"""RWKV-7 ChannelMix training operator backed by the imported CUDA family."""

from __future__ import annotations

import torch

from . import _extension


def pretrain_cmix_bf16(
    x: torch.Tensor,
    x_k: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply the fixed-length BF16 RWKV-7 ChannelMix training operator.

    Shapes are ``x[B,T,C]``, ``x_k[C]``, ``key_weight[4C,C]``, and
    ``value_weight[C,4C]``. The native forward retains the mixed input and
    squared-ReLU activation needed by its paired backward implementation.
    """

    _validate_cmix_inputs(x, x_k, key_weight, value_weight)
    return _PretrainCmixBf16Function.apply(x, x_k, key_weight, value_weight)


def _validate_cmix_inputs(
    x: torch.Tensor,
    x_k: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
) -> None:
    tensors = {
        "x": x,
        "x_k": x_k,
        "key_weight": key_weight,
        "value_weight": value_weight,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if x.ndim != 3 or any(dimension <= 0 for dimension in x.shape):
        raise ValueError("x must have non-empty shape [B, T, C]")
    channels = x.shape[2]
    if channels % 2:
        raise ValueError("ChannelMix BF16 requires an even channel count")
    if x_k.shape != (channels,):
        raise ValueError(f"x_k must have shape [{channels}]")
    if key_weight.shape != (4 * channels, channels):
        raise ValueError(f"key_weight must have shape [{4 * channels}, {channels}]")
    if value_weight.shape != (channels, 4 * channels):
        raise ValueError(f"value_weight must have shape [{channels}, {4 * channels}]")
    if any(tensor.device != x.device for tensor in tensors.values()):
        raise ValueError("all ChannelMix tensors must be on the same device")
    if not x.is_cuda:
        raise ValueError("pretrain_cmix_bf16 requires CUDA tensors")


class _PretrainCmixBf16Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        x_k: torch.Tensor,
        key_weight: torch.Tensor,
        value_weight: torch.Tensor,
    ) -> torch.Tensor:
        output, mixed, activation = _extension.pretrain_cmix_bf16_forward(
            x,
            x_k,
            key_weight,
            value_weight,
        )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            x,
            x_k,
            key_weight,
            value_weight,
            mixed,
            activation,
        )
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if grad_output is None:
            return None, None, None, None
        x, x_k, key_weight, value_weight, mixed, activation = ctx.saved_tensors
        gradients = _extension.pretrain_cmix_bf16_backward(
            grad_output.contiguous(),
            x,
            x_k,
            key_weight,
            value_weight,
            mixed,
            activation,
        )
        return tuple(
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad, strict=True)
        )
