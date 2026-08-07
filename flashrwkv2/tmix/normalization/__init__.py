# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ..wkv7 import _extension


def _check_rows(tensor: torch.Tensor, name: str, reference: torch.Tensor | None = None) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float16:
        raise TypeError(f"{name} must have dtype torch.float16")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be CUDA and contiguous")
    if tensor.ndim != 2 or tensor.shape[0] <= 0:
        raise ValueError(f"{name} must have packed shape [total_tokens,C]")
    if reference is not None and (tensor.shape != reference.shape or tensor.device != reference.device):
        raise ValueError(f"{name} must match the packed input")


def _check_affine(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> None:
    for name, tensor in (("weight", weight), ("bias", bias)):
        if tensor.dtype != torch.float16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA float16")
        if tensor.device != x.device or tensor.shape != (x.shape[1],):
            raise ValueError(f"{name} must have shape [C]")


def infer_tmix_layer_norm_forward_varlen(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, eps: float = 1.0e-5
) -> torch.Tensor:
    _check_rows(x, "x")
    _check_affine(x, weight, bias)
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("eps must be positive")
    return _extension().tmix_layer_norm_forward_varlen(x, weight, bias, float(eps))


def infer_tmix_add_layer_norm_forward_varlen(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1.0e-5,
    batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_rows(x, "x")
    _check_rows(residual, "residual", x)
    _check_affine(x, weight, bias)
    if batch_size is not None and (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
        or batch_size > x.shape[0]
    ):
        raise ValueError("batch_size must be a positive integer no larger than total_tokens")
    return tuple(
        _extension().tmix_add_layer_norm_forward_varlen(
            x,
            residual,
            weight,
            bias,
            float(eps),
            -1 if batch_size is None else int(batch_size),
        )
    )


def infer_tmix_add_last_layer_norm_forward_varlen(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    _check_rows(x, "x")
    _check_rows(residual, "residual", x)
    _check_affine(x, weight, bias)
    return _extension().tmix_add_last_layer_norm_forward_varlen(
        x, residual, weight, bias, float(eps)
    )


def infer_tmix_add_forward_varlen(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    _check_rows(x, "x")
    _check_rows(residual, "residual", x)
    return _extension().tmix_add_forward_varlen(x, residual)


__all__ = [
    "infer_tmix_layer_norm_forward_varlen",
    "infer_tmix_add_layer_norm_forward_varlen",
    "infer_tmix_add_last_layer_norm_forward_varlen",
    "infer_tmix_add_forward_varlen",
]
