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
            "install the package with its CUDA build dependencies before using "
            "an accelerated algorithm"
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


def pretrain_recurrent_fp32io16_forward(
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    state: torch.Tensor,
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    boundary: torch.Tensor,
    state_dot_a: torch.Tensor,
    scale: float,
) -> None:
    _load_extension().pretrain_recurrent_fp32io16_forward(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        state,
        r,
        log_decay,
        k,
        v,
        a,
        b,
        output,
        boundary,
        state_dot_a,
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
    build_warps: int,
    stages: int,
    state_tile: int,
    scale: float,
    state_dot_a: torch.Tensor | None = None,
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
        build_warps,
        stages,
        state_tile,
        scale,
        state_dot_a,
    )


def pretrain_recurrent_fp32io16_backward(
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    final_state: torch.Tensor,
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state_dot_a: torch.Tensor,
    grad_output: torch.Tensor | None,
    grad_final_state: torch.Tensor | None,
    boundary: torch.Tensor,
    grad_r: torch.Tensor | None,
    grad_log_decay: torch.Tensor | None,
    grad_k: torch.Tensor | None,
    grad_v: torch.Tensor | None,
    grad_a: torch.Tensor | None,
    grad_b: torch.Tensor | None,
    grad_initial_state: torch.Tensor | None,
    scale: float,
) -> None:
    _load_extension().pretrain_recurrent_fp32io16_backward(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        final_state,
        r,
        log_decay,
        k,
        v,
        a,
        b,
        state_dot_a,
        grad_output,
        grad_final_state,
        boundary,
        grad_r,
        grad_log_decay,
        grad_k,
        grad_v,
        grad_a,
        grad_b,
        grad_initial_state,
        scale,
    )


def infer_chunk_bf16_forward_k1_prepare(
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    chunk_transform: torch.Tensor,
    chunk_bias: torch.Tensor,
    token_transform: torch.Tensor,
    token_bias: torch.Tensor,
    scale: float,
) -> None:
    _load_extension().infer_chunk_bf16_forward_k1_prepare(
        chunk_token_starts,
        chunk_token_ends,
        r,
        log_decay,
        k,
        v,
        a,
        b,
        chunk_transform,
        chunk_bias,
        token_transform,
        token_bias,
        scale,
    )


def infer_chunk_bf16_forward_k2_recurrence(
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    state: torch.Tensor,
    output: torch.Tensor,
    chunk_transform: torch.Tensor,
    chunk_bias: torch.Tensor,
    token_transform: torch.Tensor,
    token_bias: torch.Tensor,
) -> None:
    _load_extension().infer_chunk_bf16_forward_k2_recurrence(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        state,
        output,
        chunk_transform,
        chunk_bias,
        token_transform,
        token_bias,
    )


def recompute_chunk_fp32(
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
    boundary: torch.Tensor,
    scale: float,
) -> None:
    _load_extension().recompute_chunk_fp32(
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
        boundary,
        scale,
    )


def pretrain_cmix_bf16_forward(
    x: torch.Tensor,
    x_k: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the RWKV-LM ChannelMix forward operator."""

    _load_extension()
    output = torch.ops.rwkv7_cmix_bf16_v5.forward(
        x,
        x_k,
        key_weight,
        value_weight,
    )
    return output[0], output[1], output[2]


def pretrain_cmix_bf16_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    x_k: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    mixed: torch.Tensor,
    activation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the RWKV-LM ChannelMix backward operator."""

    _load_extension()
    gradients = torch.ops.rwkv7_cmix_bf16_v5.backward(
        grad_output,
        x,
        x_k,
        key_weight,
        value_weight,
        mixed,
        activation,
    )
    return gradients[0], gradients[1], gradients[2], gradients[3]


def pretrain_l2wrap_ce_bf16_forward(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the RWKV-LM cross-entropy + L2Wrap forward operator."""

    _load_extension()
    result = torch.ops.rwkv7_l2wrap_ce_bf16_v2.forward(logits, targets)
    return result[0], result[1], result[2], result[3]


def pretrain_l2wrap_ce_bf16_backward(
    grad_loss: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    logsumexp: torch.Tensor,
    max_values: torch.Tensor,
    argmax: torch.Tensor,
) -> torch.Tensor:
    """Call the paired RWKV-LM cross-entropy + L2Wrap backward operator."""

    _load_extension()
    return torch.ops.rwkv7_l2wrap_ce_bf16_v2.backward(
        grad_loss,
        logits,
        targets,
        logsumexp,
        max_values,
        argmax,
    )
