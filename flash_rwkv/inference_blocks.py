# SPDX-License-Identifier: MIT

"""Safe public wrappers for the Albatross FP16 fused inference operators."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ._extension import _load_extension

_HEAD_SIZE = 64


def infer_tmix_mix6_fp16(
    x: torch.Tensor,
    shift_state: torch.Tensor,
    mixes: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Create six TimeMix projections and advance the request shift state."""

    batch_size, token_count, channels = _validate_shift_inputs(
        x, shift_state, state_name="shift_state"
    )
    if len(mixes) != 6:
        raise ValueError("mixes must contain x_r, x_w, x_k, x_v, x_a, and x_g")
    for name, mix in zip(
        ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"), mixes, strict=True
    ):
        _validate_vector(mix, channels, x, name)
    _load_extension()
    return tuple(
        torch.ops.rwkv7_fast_ops_fp16.tmix_mix6(
            batch_size,
            token_count,
            channels,
            x,
            shift_state,
            *mixes,
        )
    )


def infer_tmix_kk_a_gate_fp16(
    key: torch.Tensor,
    key_scale: torch.Tensor,
    gate_bias: torch.Tensor,
    gate_delta: torch.Tensor,
    key_gate_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare gated key, negative normalized direction, and gated direction."""

    batch_size, token_count, channels = _validate_sequence(key, "key")
    if channels % _HEAD_SIZE:
        raise ValueError("key channels must be divisible by head size 64")
    if gate_delta.shape != key.shape:
        raise ValueError("gate_delta must have the same shape as key")
    _validate_tensor(gate_delta, key, "gate_delta")
    for name, vector in {
        "key_scale": key_scale,
        "gate_bias": gate_bias,
        "key_gate_scale": key_gate_scale,
    }.items():
        _validate_vector(vector, channels, key, name)
    _load_extension()
    result = torch.ops.rwkv7_fast_ops_fp16.tmix_kk_a_gate(
        batch_size,
        token_count,
        channels,
        channels // _HEAD_SIZE,
        key,
        key_scale,
        gate_bias,
        gate_delta,
        key_gate_scale,
    )
    return result[0], result[1], result[2]


def infer_tmix_lnx_rkvres_xg_fp16(
    recurrent_output: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Apply fused per-head normalization, residual, and TimeMix output gate."""

    batch_size, token_count, channels = _validate_sequence(
        recurrent_output, "recurrent_output"
    )
    if channels % _HEAD_SIZE:
        raise ValueError("recurrent_output channels must be divisible by head size 64")
    for name, tensor in {
        "receptance": receptance,
        "key": key,
        "value": value,
        "gate": gate,
    }.items():
        _validate_tensor(tensor, recurrent_output, name)
        if tensor.shape != recurrent_output.shape:
            raise ValueError(f"{name} must have the same shape as recurrent_output")
    for name, vector in {
        "residual_scale": residual_scale,
        "norm_weight": norm_weight,
        "norm_bias": norm_bias,
    }.items():
        _validate_vector(vector, channels, recurrent_output, name)
    _load_extension()
    return torch.ops.rwkv7_fast_ops_fp16.tmix_lnx_rkvres_xg(
        batch_size,
        token_count,
        channels,
        channels // _HEAD_SIZE,
        recurrent_output,
        receptance,
        key,
        value,
        residual_scale,
        norm_weight,
        norm_bias,
        gate,
    )


def infer_tmix_vres_gate_fp16(
    value: torch.Tensor,
    first_value: torch.Tensor,
    gate_bias: torch.Tensor,
    gate_delta: torch.Tensor,
) -> torch.Tensor:
    """Blend the current value with the request's immutable first-layer value."""

    batch_size, token_count, channels = _validate_sequence(value, "value")
    for name, tensor in {
        "first_value": first_value,
        "gate_delta": gate_delta,
    }.items():
        _validate_tensor(tensor, value, name)
        if tensor.shape != value.shape:
            raise ValueError(f"{name} must have the same shape as value")
    _validate_vector(gate_bias, channels, value, "gate_bias")
    _load_extension()
    return torch.ops.rwkv7_fast_ops_fp16.tmix_vres_gate(
        batch_size,
        token_count,
        channels,
        value,
        first_value,
        gate_bias,
        gate_delta,
    )


def infer_cmix_mix_fp16(
    x: torch.Tensor,
    shift_state: torch.Tensor,
    mix: torch.Tensor,
) -> torch.Tensor:
    """Create the ChannelMix input and advance the request shift state."""

    batch_size, token_count, channels = _validate_shift_inputs(
        x, shift_state, state_name="shift_state"
    )
    _validate_vector(mix, channels, x, "mix")
    _load_extension()
    return torch.ops.rwkv7_fast_ops_fp16.cmix_mix(
        batch_size,
        token_count,
        channels,
        x,
        shift_state,
        mix,
    )


def _validate_shift_inputs(
    x: torch.Tensor,
    shift_state: torch.Tensor,
    *,
    state_name: str,
) -> tuple[int, int, int]:
    batch_size, token_count, channels = _validate_sequence(x, "x")
    _validate_tensor(shift_state, x, state_name)
    if shift_state.shape != (batch_size, channels):
        raise ValueError(f"{state_name} must have shape [{batch_size}, {channels}]")
    if channels % 2:
        raise ValueError("FP16 fused mix operators require an even channel count")
    if _storage_ranges_overlap(x, shift_state):
        raise ValueError(f"{state_name} must not alias x")
    return batch_size, token_count, channels


def _validate_sequence(tensor: torch.Tensor, name: str) -> tuple[int, int, int]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 3 or any(dimension <= 0 for dimension in tensor.shape):
        raise ValueError(f"{name} must have non-empty shape [B, T, C]")
    _validate_tensor(tensor, tensor, name)
    return tensor.shape


def _validate_vector(
    tensor: torch.Tensor,
    channels: int,
    reference: torch.Tensor,
    name: str,
) -> None:
    _validate_tensor(tensor, reference, name)
    if tensor.shape != (channels,):
        raise ValueError(f"{name} must have shape [{channels}]")


def _validate_tensor(
    tensor: torch.Tensor,
    reference: torch.Tensor,
    name: str,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float16:
        raise TypeError(f"{name} must have dtype torch.float16")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.device != reference.device:
        raise ValueError(f"{name} must be on the same CUDA device as the primary input")


def _storage_ranges_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_start = left.data_ptr()
    right_start = right.data_ptr()
    left_end = left_start + left.numel() * left.element_size()
    right_end = right_start + right.numel() * right.element_size()
    return left_start < right_end and right_start < left_end
