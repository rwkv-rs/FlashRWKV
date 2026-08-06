# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ..wkv7 import _extension


def _check(a0: torch.Tensor, a12: torch.Tensor) -> None:
    for name, tensor in (("a0", a0), ("a12", a12)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA bfloat16")
    if a0.ndim != 1 or a12.ndim != 3 or a0.shape != (a12.shape[-1],):
        raise ValueError("a0 must be [C] and a12 must be [B,T,C]")
    if any(size <= 0 for size in a12.shape) or a0.device != a12.device:
        raise ValueError("a-gate inputs must be non-empty and share a device")


class _AGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a0: torch.Tensor, a12: torch.Tensor) -> torch.Tensor:
        output = _extension().pretrain_tmix_a_gate_forward(a0, a12)
        ctx.save_for_backward(a0, a12)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor | None):
        if grad_output is None:
            return None, None
        a0, a12 = ctx.saved_tensors
        gradients = _extension().pretrain_tmix_a_gate_backward(
            grad_output.contiguous(), a0, a12
        )
        return gradients[0], gradients[1]


def pretrain_tmix_a_gate_bf16(a0: torch.Tensor, a12: torch.Tensor) -> torch.Tensor:
    """Train-temp BF16 sigmoid a-gate with native broadcast reduction."""

    _check(a0, a12)
    return _AGate.apply(a0, a12)


__all__ = ["pretrain_tmix_a_gate_bf16"]
