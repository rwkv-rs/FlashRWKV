# SPDX-License-Identifier: MIT

from __future__ import annotations

from functools import cache
from types import ModuleType

import torch


@cache
def _load_extension() -> ModuleType:
    try:
        from . import _C
    except ImportError as error:
        raise RuntimeError(
            "FlashRWKV CUDA extension is not built for this source and environment; "
            "prepare the flash-rwkv dependency group through helicopter-dev"
        ) from error
    return _C


def recurrent_fp32(
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    scale: float,
) -> None:
    _load_extension().recurrent_fp32(
        query_start_loc,
        state_indices,
        state,
        r,
        log_decay,
        k,
        v,
        a,
        b,
        output,
        scale,
    )


def recurrent_fp16(
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    scale: float,
) -> None:
    _load_extension().recurrent_fp16(
        query_start_loc,
        state_indices,
        state,
        r,
        log_decay,
        k,
        v,
        a,
        b,
        output,
        scale,
    )


def materialized_chunk_fp32(
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    transform: torch.Tensor,
    bias: torch.Tensor,
    boundary: torch.Tensor,
    scale: float,
) -> None:
    _load_extension().materialized_chunk_fp32(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        state_indices,
        state,
        r,
        log_decay,
        k,
        v,
        a,
        b,
        output,
        transform,
        bias,
        boundary,
        scale,
    )
