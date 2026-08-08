# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ..wkv7 import _extension


def _check_inputs(key, key_scale, learning_rate, learning_rate_scale, head_size) -> None:
    if head_size not in {64, 128, 256}:
        raise ValueError("head_size must be one of 64, 128, or 256")
    tensors = {
        "key": key,
        "key_scale": key_scale,
        "learning_rate": learning_rate,
        "learning_rate_scale": learning_rate_scale,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA bfloat16")
    if key.ndim != 3 or key.numel() == 0 or key.shape[-1] % head_size:
        raise ValueError("key must have non-empty shape [B,T,C], C divisible by head_size")
    if learning_rate.shape != key.shape:
        raise ValueError("learning_rate must match key")
    if key_scale.shape != (key.shape[-1],) or learning_rate_scale.shape != key_scale.shape:
        raise ValueError("kk-pre scale vectors must have shape [C]")
    if any(tensor.device != key.device for tensor in tensors.values()):
        raise ValueError("kk-pre tensors must share a device")


class _KKPre(torch.autograd.Function):
    @staticmethod
    def forward(ctx, key, key_scale, learning_rate, learning_rate_scale, head_size):
        outputs = _extension().pretrain_tmix_kk_pre_forward(
            key, key_scale, learning_rate, learning_rate_scale, head_size
        )
        ctx.save_for_backward(key, key_scale, learning_rate, learning_rate_scale, outputs[3])
        ctx.head_size = head_size
        return outputs[0], outputs[1], outputs[2]

    @staticmethod
    def backward(ctx, grad_new_key, grad_negative_direction, grad_scaled_direction):
        key, key_scale, learning_rate, learning_rate_scale, inverse_norm = ctx.saved_tensors
        zeros = torch.zeros_like(key)
        gradients = _extension().pretrain_tmix_kk_pre_backward(
            zeros if grad_new_key is None else grad_new_key.contiguous(),
            zeros if grad_negative_direction is None else grad_negative_direction.contiguous(),
            zeros if grad_scaled_direction is None else grad_scaled_direction.contiguous(),
            key,
            key_scale,
            learning_rate,
            learning_rate_scale,
            inverse_norm,
            ctx.head_size,
        )
        return tuple(
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad[:4], strict=True)
        ) + (None,)


def pretrain_tmix_kk_pre_bf16(
    key, key_scale, learning_rate, learning_rate_scale, *, head_size=64
):
    """Train-temp per-head key normalization and direction preparation."""

    _check_inputs(key, key_scale, learning_rate, learning_rate_scale, head_size)
    return _KKPre.apply(key, key_scale, learning_rate, learning_rate_scale, head_size)


__all__ = ["pretrain_tmix_kk_pre_bf16"]
