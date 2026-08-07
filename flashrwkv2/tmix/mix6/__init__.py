# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import torch

from ..wkv7 import (
    _check_metadata_inputs,
    _extension,
    _resolve_max_seqlen,
    prepare_recurrent_metadata,
)


def infer_tmix_mix6_forward_varlen(
    x: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    *,
    shift_state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    max_seqlen: int | None = None,
    validated_metadata: object | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run the Albatross TMix mix6 family on packed token rows."""

    tensors = (x, x_r, x_w, x_k, x_v, x_a, x_g)
    names = ("x", "x_r", "x_w", "x_k", "x_v", "x_a", "x_g")
    for name, tensor in zip(names, tensors, strict=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float16:
            raise TypeError(f"{name} must have dtype torch.float16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be CUDA and contiguous")
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError("x must have packed shape [total_tokens,C]")
    if any(tensor.shape != (x.shape[1],) for tensor in tensors[1:]):
        raise ValueError("all mix6 coefficients must have shape [C]")
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all mix6 tensors must share a CUDA device")
    if not isinstance(shift_state_pool, torch.Tensor):
        raise TypeError("shift_state_pool must be a torch.Tensor")
    if (
        shift_state_pool.dtype != torch.float16
        or not shift_state_pool.is_cuda
        or not shift_state_pool.is_contiguous()
        or shift_state_pool.ndim != 2
        or shift_state_pool.shape[1] != x.shape[1]
    ):
        raise ValueError("shift_state_pool must be contiguous CUDA float16 [slots,C]")
    if shift_state_pool.device != x.device:
        raise ValueError("shift_state_pool must share x's device")
    _check_metadata_inputs(cu_seqlens, state_indices)
    if validated_metadata is None:
        launch_max_seqlen = _resolve_max_seqlen(cu_seqlens, max_seqlen)
        ticket = prepare_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=x.shape[0],
            state_pool_size=shift_state_pool.shape[0],
            max_seqlen=launch_max_seqlen,
        )
    else:
        # The native ticket owns max_seqlen inference and snapshot validation.
        # Do not synchronously inspect CUDA offsets on each scheduler launch.
        if max_seqlen is None:
            launch_max_seqlen = -1
        elif (
            not isinstance(max_seqlen, int)
            or isinstance(max_seqlen, bool)
            or max_seqlen <= 0
        ):
            raise ValueError("max_seqlen must be a positive integer")
        else:
            launch_max_seqlen = int(max_seqlen)
        ticket = validated_metadata
    return tuple(
        _extension().tmix_mix6_forward_varlen(
            x,
            shift_state_pool,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            cu_seqlens,
            state_indices,
            launch_max_seqlen,
            ticket,
        )
    )


def infer_tmix_mix6_add_layer_norm_forward_varlen(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    *,
    shift_state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    max_seqlen: int | None = None,
    eps: float = 1.0e-5,
    validated_metadata: object | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run Albatross's fused B==1,T==1 TMix add-layer-norm path."""

    tensors = (x, residual, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g)
    names = ("x", "residual", "weight", "bias", "x_r", "x_w", "x_k", "x_v", "x_a", "x_g")
    for name, tensor in zip(names, tensors, strict=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float16:
            raise TypeError(f"{name} must have dtype torch.float16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be CUDA and contiguous")
        if tensor.device != x.device:
            raise ValueError(f"{name} must share x's device")
    if x.ndim != 2 or x.shape != (1, 4096):
        raise ValueError("canonical Albatross fused TMix requires [1,4096]")
    if residual.shape != x.shape:
        raise ValueError("residual must have the same shape as x")
    if any(tensor.shape != (x.shape[1],) for tensor in tensors[2:]):
        raise ValueError("normalization and mix parameters must have shape [4096]")
    if not isinstance(shift_state_pool, torch.Tensor):
        raise TypeError("shift_state_pool must be a torch.Tensor")
    if (
        shift_state_pool.dtype != torch.float16
        or not shift_state_pool.is_cuda
        or not shift_state_pool.is_contiguous()
        or shift_state_pool.ndim != 2
        or shift_state_pool.shape[1] != x.shape[1]
        or shift_state_pool.device != x.device
    ):
        raise ValueError("shift_state_pool must be contiguous CUDA float16 [slots,4096]")
    if not isinstance(eps, (float, int)) or isinstance(eps, bool) or not math.isfinite(float(eps)):
        raise ValueError("eps must be finite")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")
    _check_metadata_inputs(cu_seqlens, state_indices)
    if state_indices.ndim != 1 or state_indices.numel() != 1:
        raise ValueError("canonical Albatross fused TMix requires B=1")
    if validated_metadata is None:
        launch_max_seqlen = _resolve_max_seqlen(cu_seqlens, max_seqlen)
        if launch_max_seqlen != 1:
            raise ValueError("canonical Albatross fused TMix requires max_seqlen=1")
        ticket = prepare_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=1,
            state_pool_size=shift_state_pool.shape[0],
            max_seqlen=launch_max_seqlen,
        )
    else:
        if max_seqlen is None:
            launch_max_seqlen = -1
        elif (
            not isinstance(max_seqlen, int)
            or isinstance(max_seqlen, bool)
            or max_seqlen <= 0
        ):
            raise ValueError("max_seqlen must be a positive integer")
        else:
            launch_max_seqlen = int(max_seqlen)
            if launch_max_seqlen != 1:
                raise ValueError("canonical Albatross fused TMix requires max_seqlen=1")
        ticket = validated_metadata
    return tuple(
        _extension().tmix_mix6_add_layer_norm_forward_varlen(
            x,
            residual,
            shift_state_pool,
            weight,
            bias,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            cu_seqlens,
            state_indices,
            launch_max_seqlen,
            float(eps),
            ticket,
        )
    )


class _PretrainMix6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_r, x_w, x_k, x_v, x_a, x_g):
        output = _extension().pretrain_tmix_mix6_forward(
            x, x_r, x_w, x_k, x_v, x_a, x_g
        )
        ctx.save_for_backward(x, x_r, x_w, x_k, x_v, x_a, x_g)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if any(gradient is None for gradient in grad_outputs):
            return (None,) * 7
        return tuple(
            _extension().pretrain_tmix_mix6_backward(
                *(gradient.contiguous() for gradient in grad_outputs),
                *ctx.saved_tensors,
            )
        )


def pretrain_tmix_mix6_bf16(
    x: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Train-temp BF16 six-way shifted TimeMix preparation."""

    tensors = (x, x_r, x_w, x_k, x_v, x_a, x_g)
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be contiguous CUDA bfloat16 [B,T,C]")
    if x.ndim != 3 or x.numel() == 0:
        raise ValueError("x must have shape [B,T,C]")
    for name, tensor in zip(("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"), tensors[1:], strict=True):
        if tensor.dtype != torch.bfloat16 or not tensor.is_cuda or not tensor.is_contiguous() or tensor.shape != (x.shape[-1],) or tensor.device != x.device:
            raise ValueError(f"{name} must be contiguous CUDA bfloat16 [C]")
    return _PretrainMix6.apply(*tensors)


__all__ = [
    "infer_tmix_mix6_forward_varlen",
    "infer_tmix_mix6_add_layer_norm_forward_varlen",
    "pretrain_tmix_mix6_bf16",
]
