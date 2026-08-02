# SPDX-License-Identifier: MIT

"""RWKV-7 TimeMix training primitives backed by imported CUDA operators."""

from __future__ import annotations

import torch

from . import _extension


def pretrain_tmix_a_gate_bf16(
    a0: torch.Tensor,
    a12: torch.Tensor,
) -> torch.Tensor:
    """Apply ``sigmoid(a0 + a12)`` for the RWKV-7 TimeMix gate.

    ``a0`` has shape ``[C]`` and ``a12`` has shape ``[B, T, C]``. The paired
    CUDA backward reduces the broadcast ``a0`` gradient in FP32 before writing
    its BF16 result.
    """

    _validate_a_gate_inputs(a0, a12)
    return _PretrainTmixAGateBf16Function.apply(a0, a12)


def pretrain_tmix_vres_gate_bf16(
    value: torch.Tensor,
    first_value: torch.Tensor,
    v0: torch.Tensor,
    v12: torch.Tensor,
) -> torch.Tensor:
    """Blend current and first-layer values through the RWKV-7 TimeMix gate."""

    _validate_vres_gate_inputs(value, first_value, v0, v12)
    return _PretrainTmixVresGateBf16Function.apply(value, first_value, v0, v12)


def _validate_a_gate_inputs(a0: torch.Tensor, a12: torch.Tensor) -> None:
    for name, tensor in {"a0": a0, "a12": a12}.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if a0.ndim != 1 or a0.shape[0] <= 0:
        raise ValueError("a0 must have non-empty shape [C]")
    if a12.ndim != 3 or any(dimension <= 0 for dimension in a12.shape):
        raise ValueError("a12 must have non-empty shape [B, T, C]")
    if a0.shape != (a12.shape[2],):
        raise ValueError(f"a0 must have shape [{a12.shape[2]}]")
    if a0.device != a12.device:
        raise ValueError("a0 and a12 must be on the same device")
    if not a12.is_cuda:
        raise ValueError("pretrain_tmix_a_gate_bf16 requires CUDA tensors")


def _validate_vres_gate_inputs(
    value: torch.Tensor,
    first_value: torch.Tensor,
    v0: torch.Tensor,
    v12: torch.Tensor,
) -> None:
    tensors = {
        "value": value,
        "first_value": first_value,
        "v0": v0,
        "v12": v12,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if value.ndim != 3 or any(dimension <= 0 for dimension in value.shape):
        raise ValueError("value must have non-empty shape [B, T, C]")
    if first_value.shape != value.shape:
        raise ValueError("first_value must have the same shape as value")
    if v12.shape != value.shape:
        raise ValueError("v12 must have the same shape as value")
    if v0.shape != (value.shape[2],):
        raise ValueError(f"v0 must have shape [{value.shape[2]}]")
    if any(tensor.device != value.device for tensor in tensors.values()):
        raise ValueError("all TimeMix value-residual tensors must share a device")
    if not value.is_cuda:
        raise ValueError("pretrain_tmix_vres_gate_bf16 requires CUDA tensors")


class _PretrainTmixAGateBf16Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        a0: torch.Tensor,
        a12: torch.Tensor,
    ) -> torch.Tensor:
        output = _extension.pretrain_tmix_a_gate_bf16_forward(a0, a12)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(a0, a12)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if grad_output is None:
            return None, None
        a0, a12 = ctx.saved_tensors
        gradients = _extension.pretrain_tmix_a_gate_bf16_backward(
            grad_output.contiguous(),
            a0,
            a12,
        )
        return tuple(
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad, strict=True)
        )


class _PretrainTmixVresGateBf16Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        value: torch.Tensor,
        first_value: torch.Tensor,
        v0: torch.Tensor,
        v12: torch.Tensor,
    ) -> torch.Tensor:
        output = _extension.pretrain_tmix_vres_gate_bf16_forward(
            value,
            first_value,
            v0,
            v12,
        )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(value, first_value, v0, v12)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if grad_output is None:
            return None, None, None, None
        value, first_value, v0, v12 = ctx.saved_tensors
        gradients = _extension.pretrain_tmix_vres_gate_bf16_backward(
            grad_output.contiguous(),
            value,
            first_value,
            v0,
            v12,
        )
        return tuple(
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad, strict=True)
        )
