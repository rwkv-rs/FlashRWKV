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
    validated_metadata: object | None = None,
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
        validated_metadata,
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
    validated_metadata: object | None = None,
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
        validated_metadata,
    )


def prepare_recurrent_metadata(
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    total_tokens: int,
    state_pool_size: int,
) -> object:
    return _load_extension().prepare_recurrent_metadata(
        query_start_loc,
        state_indices,
        total_tokens,
        state_pool_size,
    )


def recurrent_fp32_from_decay_logits(
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    decay_bias: torch.Tensor | None = None,
    elapsed_t: torch.Tensor | None = None,
    validated_metadata: object | None = None,
) -> None:
    _load_extension().recurrent_fp32_from_decay_logits(
        query_start_loc,
        state_indices,
        state,
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        output,
        scale,
        decay_bias,
        elapsed_t,
        validated_metadata,
    )


def recurrent_fp16_from_decay_logits(
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    decay_bias: torch.Tensor | None = None,
    elapsed_t: torch.Tensor | None = None,
    validated_metadata: object | None = None,
) -> None:
    _load_extension().recurrent_fp16_from_decay_logits(
        query_start_loc,
        state_indices,
        state,
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        output,
        scale,
        decay_bias,
        elapsed_t,
        validated_metadata,
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


def pretrain_recurrent_fp32io16_from_decay_logits_forward(
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    state: torch.Tensor,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    boundary: torch.Tensor,
    state_dot_a: torch.Tensor,
    scale: float,
) -> None:
    _load_extension().pretrain_recurrent_fp32io16_from_decay_logits_forward(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        state,
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        output,
        boundary,
        state_dot_a,
        scale,
    )


def statetune_recurrent_fp32io16_forward(
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
    _load_extension().statetune_recurrent_fp32io16_forward(
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


def statetune_recurrent_fp32io16_from_decay_logits_forward(
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    state: torch.Tensor,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    boundary: torch.Tensor,
    state_dot_a: torch.Tensor,
    scale: float,
) -> None:
    _load_extension().statetune_recurrent_fp32io16_from_decay_logits_forward(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        state,
        r,
        decay_logits,
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


def materialized_chunk_fp32_from_decay_logits(
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
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
    decay_bias: torch.Tensor | None = None,
) -> None:
    _load_extension().materialized_chunk_fp32_from_decay_logits(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        state_indices,
        state,
        r,
        decay_logits,
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
        decay_bias,
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


def pretrain_recurrent_fp32io16_from_decay_logits_backward(
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    final_state: torch.Tensor,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state_dot_a: torch.Tensor,
    grad_output: torch.Tensor | None,
    grad_final_state: torch.Tensor | None,
    boundary: torch.Tensor,
    grad_r: torch.Tensor | None,
    grad_decay_logits: torch.Tensor | None,
    grad_k: torch.Tensor | None,
    grad_v: torch.Tensor | None,
    grad_a: torch.Tensor | None,
    grad_b: torch.Tensor | None,
    grad_initial_state: torch.Tensor | None,
    scale: float,
) -> None:
    _load_extension().pretrain_recurrent_fp32io16_from_decay_logits_backward(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        final_state,
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        state_dot_a,
        grad_output,
        grad_final_state,
        boundary,
        grad_r,
        grad_decay_logits,
        grad_k,
        grad_v,
        grad_a,
        grad_b,
        grad_initial_state,
        scale,
    )


def statetune_recurrent_fp32io16_backward(
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
    _load_extension().statetune_recurrent_fp32io16_backward(
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


def statetune_recurrent_fp32io16_from_decay_logits_backward(
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    final_state: torch.Tensor,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state_dot_a: torch.Tensor,
    grad_output: torch.Tensor | None,
    grad_final_state: torch.Tensor | None,
    boundary: torch.Tensor,
    grad_r: torch.Tensor | None,
    grad_decay_logits: torch.Tensor | None,
    grad_k: torch.Tensor | None,
    grad_v: torch.Tensor | None,
    grad_a: torch.Tensor | None,
    grad_b: torch.Tensor | None,
    grad_initial_state: torch.Tensor | None,
    scale: float,
) -> None:
    _load_extension().statetune_recurrent_fp32io16_from_decay_logits_backward(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        final_state,
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        state_dot_a,
        grad_output,
        grad_final_state,
        boundary,
        grad_r,
        grad_decay_logits,
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


def infer_chunk_bf16_forward_k1_prepare_from_decay_logits(
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    chunk_transform: torch.Tensor,
    chunk_bias: torch.Tensor,
    token_transform: torch.Tensor,
    token_bias: torch.Tensor,
    scale: float,
    decay_bias: torch.Tensor | None = None,
) -> None:
    _load_extension().infer_chunk_bf16_forward_k1_prepare_from_decay_logits(
        chunk_token_starts,
        chunk_token_ends,
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        chunk_transform,
        chunk_bias,
        token_transform,
        token_bias,
        scale,
        decay_bias,
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


def recompute_chunk_fp32_from_decay_logits(
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    boundary: torch.Tensor,
    scale: float,
    decay_bias: torch.Tensor | None = None,
) -> None:
    _load_extension().recompute_chunk_fp32_from_decay_logits(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        state_indices,
        state,
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        output,
        boundary,
        scale,
        decay_bias,
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


def pretrain_tmix_a_gate_bf16_forward(
    a0: torch.Tensor,
    a12: torch.Tensor,
) -> torch.Tensor:
    """Call the RWKV-LM TimeMix a-gate forward operator."""

    _load_extension()
    return torch.ops.rwkv7_tmix_a_gate_bf16.forward(a0, a12)


def pretrain_tmix_a_gate_bf16_backward(
    grad_output: torch.Tensor,
    a0: torch.Tensor,
    a12: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the paired RWKV-LM TimeMix a-gate backward operator."""

    _load_extension()
    gradients = torch.ops.rwkv7_tmix_a_gate_bf16.backward(
        grad_output,
        a0,
        a12,
    )
    return gradients[0], gradients[1]


def pretrain_tmix_vres_gate_bf16_forward(
    value: torch.Tensor,
    first_value: torch.Tensor,
    v0: torch.Tensor,
    v12: torch.Tensor,
) -> torch.Tensor:
    """Call the RWKV-LM TimeMix value-residual gate forward operator."""

    _load_extension()
    return torch.ops.rwkv7_tmix_vres_gate_bf16_v3.forward(
        value,
        first_value,
        v0,
        v12,
    )


def pretrain_tmix_vres_gate_bf16_backward(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    first_value: torch.Tensor,
    v0: torch.Tensor,
    v12: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the paired RWKV-LM TimeMix value-residual gate backward operator."""

    _load_extension()
    gradients = torch.ops.rwkv7_tmix_vres_gate_bf16_v3.backward(
        grad_output,
        value,
        first_value,
        v0,
        v12,
    )
    return gradients[0], gradients[1], gradients[2], gradients[3]


def pretrain_head_l2wrap_ce_bf16_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the RWKV-LM fused output-head loss operator."""

    _load_extension()
    result = torch.ops.rwkv7_head_l2wrap_ce_bf16_v4.forward(
        hidden,
        weight,
        targets,
        chunk_rows,
    )
    return result[0], result[1], result[2]


def pretrain_tmix_mix6_bf16_forward(
    x: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Call the RWKV-LM fused six-way TimeMix forward operator."""

    _load_extension()
    return tuple(
        torch.ops.rwkv7_tmix_mix6_bf16_v5.forward(
            x,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
        )
    )


def pretrain_tmix_mix6_bf16_backward(
    grad_r: torch.Tensor,
    grad_w: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    grad_a: torch.Tensor,
    grad_g: torch.Tensor,
    x: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Call the paired RWKV-LM fused six-way TimeMix backward operator."""

    _load_extension()
    return tuple(
        torch.ops.rwkv7_tmix_mix6_bf16_v5.backward(
            grad_r,
            grad_w,
            grad_k,
            grad_v,
            grad_a,
            grad_g,
            x,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
        )
    )


def pretrain_tmix_kk_pre_bf16_forward(
    key: torch.Tensor,
    key_scale: torch.Tensor,
    learning_rate: torch.Tensor,
    learning_rate_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the RWKV-LM TimeMix key-preparation forward operator."""

    _load_extension()
    result = torch.ops.rwkv7_tmix_kk_pre_bf16_v5.forward(
        key,
        key_scale,
        learning_rate,
        learning_rate_scale,
        64,
    )
    return result[0], result[1], result[2], result[3]


def pretrain_tmix_kk_pre_bf16_backward(
    grad_new_key: torch.Tensor,
    grad_negative_direction: torch.Tensor,
    grad_scaled_direction: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    learning_rate: torch.Tensor,
    learning_rate_scale: torch.Tensor,
    inverse_norm: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Call the paired RWKV-LM TimeMix key-preparation backward operator."""

    _load_extension()
    return tuple(
        torch.ops.rwkv7_tmix_kk_pre_bf16_v5.backward(
            grad_new_key,
            grad_negative_direction,
            grad_scaled_direction,
            key,
            key_scale,
            learning_rate,
            learning_rate_scale,
            inverse_norm,
            64,
        )
    )


def pretrain_tmix_lnx_rkvres_xg_bf16_forward(
    recurrent_output: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the RWKV-LM fused TimeMix output forward operator."""

    _load_extension()
    result = torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.forward(
        recurrent_output,
        receptance,
        key,
        value,
        residual_scale,
        norm_weight,
        norm_bias,
        gate,
    )
    return result[0], result[1], result[2]


def pretrain_tmix_lnx_rkvres_xg_bf16_backward(
    grad_output: torch.Tensor,
    recurrent_output: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    gate: torch.Tensor,
    mean: torch.Tensor,
    reciprocal_std: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Call the paired RWKV-LM fused TimeMix output backward operator."""

    _load_extension()
    return tuple(
        torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.backward(
            grad_output,
            recurrent_output,
            receptance,
            key,
            value,
            residual_scale,
            norm_weight,
            norm_bias,
            gate,
            mean,
            reciprocal_std,
        )
    )
