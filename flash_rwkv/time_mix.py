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


def pretrain_tmix_mix6_bf16(
    x: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Produce the six shifted RWKV-7 TimeMix inputs in one CUDA operator."""

    mixes = (x_r, x_w, x_k, x_v, x_a, x_g)
    _validate_mix6_inputs(x, mixes)
    return _PretrainTmixMix6Bf16Function.apply(x, *mixes)


def pretrain_tmix_kk_pre_bf16(
    key: torch.Tensor,
    key_scale: torch.Tensor,
    learning_rate: torch.Tensor,
    learning_rate_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize the RWKV-7 key direction and form recurrent key operands."""

    _validate_kk_pre_inputs(key, key_scale, learning_rate, learning_rate_scale)
    return _PretrainTmixKkPreBf16Function.apply(
        key,
        key_scale,
        learning_rate,
        learning_rate_scale,
    )


def pretrain_tmix_lnx_rkvres_xg_bf16(
    recurrent_output: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Fuse per-head normalization, receptance-key residual, and output gate."""

    inputs = (
        recurrent_output,
        receptance,
        key,
        value,
        residual_scale,
        norm_weight,
        norm_bias,
        gate,
    )
    _validate_lnx_rkvres_xg_inputs(*inputs)
    return _PretrainTmixLnxRkvresXgBf16Function.apply(*inputs)


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


def _validate_mix6_inputs(
    x: torch.Tensor,
    mixes: tuple[torch.Tensor, ...],
) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.dtype != torch.bfloat16:
        raise TypeError("x must have dtype torch.bfloat16")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.ndim != 3 or any(dimension <= 0 for dimension in x.shape):
        raise ValueError("x must have non-empty shape [B, T, C]")
    if x.shape[2] % 2:
        raise ValueError("pretrain_tmix_mix6_bf16 requires an even channel count")
    for name, mix in zip(
        ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"), mixes, strict=True
    ):
        if not isinstance(mix, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if mix.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
        if not mix.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if mix.shape != (x.shape[2],):
            raise ValueError(f"{name} must have shape [{x.shape[2]}]")
        if mix.device != x.device:
            raise ValueError(f"{name} must be on the same device as x")
    if not x.is_cuda:
        raise ValueError("pretrain_tmix_mix6_bf16 requires CUDA tensors")


def _validate_kk_pre_inputs(
    key: torch.Tensor,
    key_scale: torch.Tensor,
    learning_rate: torch.Tensor,
    learning_rate_scale: torch.Tensor,
) -> None:
    tensors = {
        "key": key,
        "key_scale": key_scale,
        "learning_rate": learning_rate,
        "learning_rate_scale": learning_rate_scale,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if key.ndim != 3 or any(dimension <= 0 for dimension in key.shape):
        raise ValueError("key must have non-empty shape [B, T, C]")
    if key.shape[2] % 64:
        raise ValueError("pretrain_tmix_kk_pre_bf16 requires C divisible by 64")
    if learning_rate.shape != key.shape:
        raise ValueError("learning_rate must have the same shape as key")
    for name, vector in {
        "key_scale": key_scale,
        "learning_rate_scale": learning_rate_scale,
    }.items():
        if vector.shape != (key.shape[2],):
            raise ValueError(f"{name} must have shape [{key.shape[2]}]")
    if any(tensor.device != key.device for tensor in tensors.values()):
        raise ValueError("all TimeMix key-preparation tensors must share a device")
    if not key.is_cuda:
        raise ValueError("pretrain_tmix_kk_pre_bf16 requires CUDA tensors")


def _validate_lnx_rkvres_xg_inputs(
    recurrent_output: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    gate: torch.Tensor,
) -> None:
    tensors = {
        "recurrent_output": recurrent_output,
        "receptance": receptance,
        "key": key,
        "value": value,
        "residual_scale": residual_scale,
        "norm_weight": norm_weight,
        "norm_bias": norm_bias,
        "gate": gate,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if recurrent_output.ndim != 3 or any(
        dimension <= 0 for dimension in recurrent_output.shape
    ):
        raise ValueError("recurrent_output must have non-empty shape [B, T, C]")
    channels = recurrent_output.shape[2]
    if channels % 64:
        raise ValueError("pretrain_tmix_lnx_rkvres_xg_bf16 requires C divisible by 64")
    for name, tensor in {
        "receptance": receptance,
        "key": key,
        "value": value,
        "gate": gate,
    }.items():
        if tensor.shape != recurrent_output.shape:
            raise ValueError(f"{name} must have the same shape as recurrent_output")
    if residual_scale.shape != (channels // 64, 64):
        raise ValueError(f"residual_scale must have shape [{channels // 64}, 64]")
    for name, vector in {"norm_weight": norm_weight, "norm_bias": norm_bias}.items():
        if vector.shape != (channels,):
            raise ValueError(f"{name} must have shape [{channels}]")
    if any(tensor.device != recurrent_output.device for tensor in tensors.values()):
        raise ValueError("all fused TimeMix output tensors must share a device")
    if not recurrent_output.is_cuda:
        raise ValueError("pretrain_tmix_lnx_rkvres_xg_bf16 requires CUDA tensors")


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


class _PretrainTmixMix6Bf16Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        x_r: torch.Tensor,
        x_w: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        x_a: torch.Tensor,
        x_g: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        inputs = (x, x_r, x_w, x_k, x_v, x_a, x_g)
        outputs = _extension.pretrain_tmix_mix6_bf16_forward(*inputs)
        ctx.save_for_backward(*inputs)
        return outputs

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        gradients = _extension.pretrain_tmix_mix6_bf16_backward(
            *(gradient.contiguous() for gradient in grad_outputs),
            *ctx.saved_tensors,
        )
        return tuple(
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad, strict=True)
        )


class _PretrainTmixKkPreBf16Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        key: torch.Tensor,
        key_scale: torch.Tensor,
        learning_rate: torch.Tensor,
        learning_rate_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_key, negative_direction, scaled_direction, inverse_norm = (
            _extension.pretrain_tmix_kk_pre_bf16_forward(
                key,
                key_scale,
                learning_rate,
                learning_rate_scale,
            )
        )
        ctx.save_for_backward(
            key,
            key_scale,
            learning_rate,
            learning_rate_scale,
            inverse_norm,
        )
        return new_key, negative_direction, scaled_direction

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_new_key: torch.Tensor,
        grad_negative_direction: torch.Tensor,
        grad_scaled_direction: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        gradients = _extension.pretrain_tmix_kk_pre_bf16_backward(
            grad_new_key.contiguous(),
            grad_negative_direction.contiguous(),
            grad_scaled_direction.contiguous(),
            *ctx.saved_tensors,
        )
        return tuple(
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad, strict=True)
        )


class _PretrainTmixLnxRkvresXgBf16Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        recurrent_output: torch.Tensor,
        receptance: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        residual_scale: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        output, mean, reciprocal_std = (
            _extension.pretrain_tmix_lnx_rkvres_xg_bf16_forward(
                recurrent_output,
                receptance,
                key,
                value,
                residual_scale,
                norm_weight,
                norm_bias,
                gate,
            )
        )
        ctx.save_for_backward(
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
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        gradients = _extension.pretrain_tmix_lnx_rkvres_xg_bf16_backward(
            grad_output.contiguous(),
            *ctx.saved_tensors,
        )
        return tuple(
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad, strict=True)
        )
