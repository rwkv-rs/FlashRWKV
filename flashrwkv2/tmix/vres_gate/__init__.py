# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ..wkv7 import _extension


def infer_tmix_vres_gate_forward_varlen(
    v: torch.Tensor,
    v_first: torch.Tensor,
    v0: torch.Tensor,
    v12: torch.Tensor,
) -> torch.Tensor:
    """Run the Albatross TMix value-residual gate on packed rows."""

    tensors = {"v": v, "v_first": v_first, "v0": v0, "v12": v12}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float16:
            raise TypeError(f"{name} must have dtype torch.float16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be CUDA and contiguous")
        if tensor.device != v.device:
            raise ValueError(f"{name} must share v's device")
    if v.ndim != 2 or v.shape[0] <= 0 or v.shape[1] <= 0:
        raise ValueError("v must have packed shape [total_tokens,C]")
    if v_first.shape != v.shape or v12.shape != v.shape or v0.shape != (v.shape[1],):
        raise ValueError("v_first and v12 must match v; v0 must have shape [C]")
    return _extension().tmix_vres_gate_forward_varlen(v, v_first, v0, v12)


def _check(value: torch.Tensor, first_value: torch.Tensor, v0: torch.Tensor, v12: torch.Tensor) -> None:
    tensors = {"value": value, "first_value": first_value, "v0": v0, "v12": v12}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA bfloat16")
        if tensor.device != value.device:
            raise ValueError(f"{name} must share value's device")
    if value.ndim != 3 or value.numel() == 0:
        raise ValueError("value must have shape [B,T,C]")
    if first_value.shape != value.shape or v12.shape != value.shape or v0.shape != (value.shape[-1],):
        raise ValueError("invalid v-residual gate shapes")


class _VResGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, first_value, v0, v12):
        output = _extension().pretrain_tmix_vres_gate_forward(value, first_value, v0, v12)
        ctx.save_for_backward(value, first_value, v0, v12)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        gradients = _extension().pretrain_tmix_vres_gate_backward(
            grad_output.contiguous(), *ctx.saved_tensors
        )
        return tuple(gradients)


def pretrain_tmix_vres_gate_bf16(value, first_value, v0, v12):
    """Train-temp BF16 value-residual gate with native FP32 reduction."""

    _check(value, first_value, v0, v12)
    return _VResGate.apply(value, first_value, v0, v12)


__all__ = [
    "infer_tmix_vres_gate_forward_varlen",
    "pretrain_tmix_vres_gate_bf16",
]
