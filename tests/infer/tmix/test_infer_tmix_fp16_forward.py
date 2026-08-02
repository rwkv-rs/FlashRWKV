# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flash_rwkv import (
    infer_tmix_kk_a_gate_fp16,
    infer_tmix_lnx_rkvres_xg_fp16,
    infer_tmix_mix6_fp16,
    infer_tmix_vres_gate_fp16,
)
from flash_rwkv._extension import _load_extension


def test_tmix_inference_rejects_cpu_before_native_dispatch() -> None:
    x = torch.zeros(1, 1, 64, dtype=torch.float16)
    shift = torch.zeros(1, 64, dtype=torch.float16)
    mixes = tuple(torch.zeros(64, dtype=torch.float16) for _ in range(6))
    with pytest.raises(ValueError, match="CUDA"):
        infer_tmix_mix6_fp16(x, shift, mixes)


def test_tmix_inference_rejects_state_alias_before_native_dispatch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    storage = torch.zeros(2, 64, device="cuda", dtype=torch.float16)
    x = storage.view(1, 2, 64)
    shift = storage[:1]
    mixes = tuple(torch.zeros(64, device="cuda", dtype=torch.float16) for _ in range(6))
    with pytest.raises(ValueError, match="must not alias"):
        infer_tmix_mix6_fp16(x, shift, mixes)


def test_raw_tmix_inference_rejects_empty_launch_metadata() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    _load_extension()
    x = torch.empty(0, 1, 64, device="cuda", dtype=torch.float16)
    shift = torch.empty(0, 64, device="cuda", dtype=torch.float16)
    mixes = tuple(torch.empty(64, device="cuda", dtype=torch.float16) for _ in range(6))
    with pytest.raises(RuntimeError, match="B, T, and C must be positive"):
        torch.ops.rwkv7_fast_ops_fp16.tmix_mix6(0, 1, 64, x, shift, *mixes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tmix_mix6_matches_reference_and_advances_state() -> None:
    torch.manual_seed(2607)
    x = torch.randn(2, 3, 128, device="cuda", dtype=torch.float16).mul_(0.2)
    shift_state = torch.randn(2, 128, device="cuda", dtype=torch.float16).mul_(0.2)
    initial_state = shift_state.clone()
    mixes = tuple(
        torch.randn(128, device="cuda", dtype=torch.float16).mul_(0.2) for _ in range(6)
    )

    outputs = infer_tmix_mix6_fp16(x, shift_state, mixes)

    previous = torch.cat((initial_state[:, None], x[:, :-1]), dim=1)
    delta = previous.float() - x.float()
    for output, mix in zip(outputs, mixes, strict=True):
        reference = (x.float() + delta * mix.float()).half()
        torch.testing.assert_close(output, reference, atol=0.002, rtol=0.002)
    torch.testing.assert_close(shift_state, x[:, -1], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tmix_key_and_value_gates_match_reference() -> None:
    torch.manual_seed(2608)
    shape = (2, 3, 128)
    key = torch.randn(*shape, device="cuda", dtype=torch.float16).mul_(0.2)
    key_scale = torch.randn(128, device="cuda", dtype=torch.float16).mul_(0.2)
    gate_bias = torch.randn(128, device="cuda", dtype=torch.float16).mul_(0.2)
    gate_delta = torch.randn(*shape, device="cuda", dtype=torch.float16).mul_(0.2)
    key_gate_scale = torch.randn(128, device="cuda", dtype=torch.float16).mul_(0.2)

    new_key, negative_direction, scaled_direction = infer_tmix_kk_a_gate_fp16(
        key,
        key_scale,
        gate_bias,
        gate_delta,
        key_gate_scale,
    )
    gate = torch.sigmoid(gate_bias.float() + gate_delta.float())
    direction_input = (key.float() * key_scale.float()).view(2, 3, 2, 64)
    direction = direction_input / direction_input.square().sum(
        -1, keepdim=True
    ).sqrt().clamp_min(1e-12)
    direction = direction.view_as(key)
    key_reference = (key.float() * (1.0 + key_gate_scale.float() * (gate - 1.0))).half()
    torch.testing.assert_close(new_key, key_reference, atol=0.002, rtol=0.003)
    torch.testing.assert_close(
        negative_direction, -direction.half(), atol=0.002, rtol=0.003
    )
    torch.testing.assert_close(
        scaled_direction, (direction * gate).half(), atol=0.002, rtol=0.003
    )

    value = torch.randn(*shape, device="cuda", dtype=torch.float16).mul_(0.2)
    first_value = torch.randn(*shape, device="cuda", dtype=torch.float16).mul_(0.2)
    blended = infer_tmix_vres_gate_fp16(
        value,
        first_value,
        gate_bias,
        gate_delta,
    )
    reference = (value.float() + (first_value.float() - value.float()) * gate).half()
    torch.testing.assert_close(blended, reference, atol=0.002, rtol=0.003)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tmix_output_transform_matches_reference() -> None:
    torch.manual_seed(2609)
    shape = (2, 3, 128)
    tensors = tuple(
        torch.randn(*shape, device="cuda", dtype=torch.float16).mul_(0.2)
        for _ in range(5)
    )
    recurrent_output, receptance, key, value, gate = tensors
    residual_scale = torch.randn(128, device="cuda", dtype=torch.float16).mul_(0.2)
    norm_weight = torch.randn(128, device="cuda", dtype=torch.float16).mul_(0.2)
    norm_bias = torch.randn(128, device="cuda", dtype=torch.float16).mul_(0.2)

    output = infer_tmix_lnx_rkvres_xg_fp16(
        recurrent_output,
        receptance,
        key,
        value,
        residual_scale,
        norm_weight,
        norm_bias,
        gate,
    )
    grouped_output = recurrent_output.float().view(2, 3, 2, 64)
    mean = grouped_output.mean(dim=-1, keepdim=True)
    reciprocal_std = (
        grouped_output.var(dim=-1, correction=0, keepdim=True) + 64e-5
    ).rsqrt()
    normalized = (grouped_output - mean) * reciprocal_std
    normalized = normalized * norm_weight.float().view(1, 1, 2, 64)
    normalized = normalized + norm_bias.float().view(1, 1, 2, 64)
    residual = (
        receptance.float().view(2, 3, 2, 64)
        * key.float().view(2, 3, 2, 64)
        * residual_scale.float().view(1, 1, 2, 64)
    ).sum(dim=-1, keepdim=True) * value.float().view(2, 3, 2, 64)
    reference = ((normalized + residual) * gate.float().view(2, 3, 2, 64)).half()
    torch.testing.assert_close(output, reference.view_as(output), atol=0.004, rtol=0.02)
