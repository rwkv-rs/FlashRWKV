# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ...tmix.wkv7 import (
    _check_metadata_inputs,
    _extension,
    _resolve_max_seqlen,
    prepare_recurrent_metadata,
)

_MAX_GRID_DIM_YZ = 65535


def _check_sparse_grid_rows(rows: int, operator: str, grid_dimension: str) -> None:
    if rows > _MAX_GRID_DIM_YZ:
        raise ValueError(
            f"{operator} supports at most {_MAX_GRID_DIM_YZ} packed rows because "
            f"rows map to CUDA {grid_dimension}; got rows={rows}"
        )


def _check_half(tensor: torch.Tensor, name: str, reference: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float16 or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous CUDA float16")
    if tensor.device != reference.device:
        raise ValueError(f"{name} must share the input device")


def infer_cmix_sparse_forward_varlen(
    x: torch.Tensor,
    x_k: torch.Tensor,
    key_fc: torch.Tensor,
    value_fc: torch.Tensor,
    *,
    shift_state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    max_seqlen: int | None = None,
    validated_metadata: object | None = None,
) -> torch.Tensor:
    """Run the canonical no-fc CMix sparse path on packed requests."""

    _check_half(x, "x", x)
    for name, tensor in (("x_k", x_k), ("key_fc", key_fc), ("value_fc", value_fc)):
        _check_half(tensor, name, x)
    _check_half(shift_state_pool, "shift_state_pool", x)
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError("x must have packed shape [total_tokens,C]")
    if x_k.shape != (x.shape[1],):
        raise ValueError("x_k must have shape [C]")
    if key_fc.ndim != 2 or key_fc.shape[1] != x.shape[1]:
        raise ValueError("key_fc must have shape [F,C]")
    if value_fc.ndim != 2 or value_fc.shape != (key_fc.shape[0], x.shape[1]):
        raise ValueError("value_fc must have shape [F,C]")
    if shift_state_pool.ndim != 2 or shift_state_pool.shape[1] != x.shape[1]:
        raise ValueError("shift_state_pool must have shape [slots,C]")
    _check_sparse_grid_rows(x.shape[0], "cmix sparse combined", "grid.y/grid.z")
    _check_metadata_inputs(cu_seqlens, state_indices)
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
    if ticket is None:
        ticket = prepare_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=x.shape[0],
            state_pool_size=shift_state_pool.shape[0],
            max_seqlen=(
                _resolve_max_seqlen(cu_seqlens, max_seqlen)
                if max_seqlen is None
                else launch_max_seqlen
            ),
        )
    return _extension().cmix_sparse_forward_varlen(
        x,
        shift_state_pool,
        x_k,
        key_fc,
        value_fc,
        cu_seqlens,
        state_indices,
        launch_max_seqlen,
        ticket,
    )


def infer_cmix_sparse_up_forward_varlen(
    x: torch.Tensor,
    x_k: torch.Tensor,
    key_fc: torch.Tensor,
    *,
    shift_state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    max_seqlen: int | None = None,
    validated_metadata: object | None = None,
) -> torch.Tensor:
    """Return the packed sparse up projection before the ReLU-square/down step."""

    _check_half(x, "x", x)
    for name, tensor in (("x_k", x_k), ("key_fc", key_fc), ("shift_state_pool", shift_state_pool)):
        _check_half(tensor, name, x)
    if x.ndim != 2 or x_k.shape != (x.shape[1],) or key_fc.ndim != 2 or key_fc.shape[1] != x.shape[1]:
        raise ValueError("invalid CMix sparse up shapes")
    if shift_state_pool.ndim != 2 or shift_state_pool.shape[1] != x.shape[1]:
        raise ValueError("shift_state_pool must have shape [slots,C]")
    _check_sparse_grid_rows(x.shape[0], "cmix sparse up", "grid.y")
    _check_metadata_inputs(cu_seqlens, state_indices)
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
    if ticket is None:
        ticket = prepare_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=x.shape[0],
            state_pool_size=shift_state_pool.shape[0],
            max_seqlen=(
                _resolve_max_seqlen(cu_seqlens, max_seqlen)
                if max_seqlen is None
                else launch_max_seqlen
            ),
        )
    return _extension().cmix_sparse_up_forward_varlen(
        x,
        shift_state_pool,
        x_k,
        key_fc,
        cu_seqlens,
        state_indices,
        launch_max_seqlen,
        ticket,
    )


def infer_cmix_sparse_down_relu_forward_varlen(
    preact: torch.Tensor,
    value_fc: torch.Tensor,
    *,
    batch_size: int | None = None,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    """Run the Albatross sparse ReLU-square/down family with caller dispatch."""

    _check_half(preact, "preact", preact)
    _check_half(value_fc, "value_fc", preact)
    if preact.ndim != 2 or value_fc.ndim != 2 or value_fc.shape[0] != preact.shape[1]:
        raise ValueError("preact must be [rows,F] and value_fc must be [F,C]")
    if value_fc.shape[1] <= 0 or value_fc.shape[1] % 2:
        raise ValueError("value_fc must have an even C in shape [F,C]")
    _check_sparse_grid_rows(preact.shape[0], "cmix sparse down", "grid.y/grid.z")
    if (batch_size is None) != (max_seqlen is None):
        raise ValueError("batch_size and max_seqlen must be provided together")
    if batch_size is None:
        launch_batch = -1
        launch_max_seqlen = -1
    else:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not isinstance(max_seqlen, int)
            or isinstance(max_seqlen, bool)
            or batch_size <= 0
            or max_seqlen <= 0
            or batch_size * max_seqlen < preact.shape[0]
        ):
            raise ValueError("batch_size and max_seqlen must be positive and cover packed rows")
        launch_batch = batch_size
        launch_max_seqlen = max_seqlen
    return _extension().cmix_sparse_down_relu_forward_varlen(
        preact, value_fc, launch_batch, launch_max_seqlen
    )


__all__ = [
    "infer_cmix_sparse_forward_varlen",
    "infer_cmix_sparse_up_forward_varlen",
    "infer_cmix_sparse_down_relu_forward_varlen",
]
