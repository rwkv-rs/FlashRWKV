# SPDX-License-Identifier: MIT

"""Build and runtime architecture contracts for the native extension."""

from __future__ import annotations

import re

MINIMUM_WHEEL_ARCHITECTURE = "sm90"
COMPILE_ONLY_ARCHITECTURES = ("sm90", "sm120")
RUNTIME_VALIDATED_ARCHITECTURES = ("sm120",)
BUILD_TARGET_MATRIX = (
    ("sm80", "fail_closed"),
    ("sm90", "compile_only"),
    ("sm120", "runtime_validated"),
)

TRANSLATION_UNIT_ARCHITECTURES = {
    "csrc/validation/recurrent_metadata.cu": "common",
    "csrc/common/wkv7/recurrent_common_fp32io16_backward.cu": "common",
    "csrc/common/wkv7/recurrent_common_fp32io16_forward.cu": "common",
    "csrc/infer/wkv7/infer_common_chunk_bf16_forward_k1_prepare.cu": "common",
    "csrc/infer/wkv7/infer_common_chunk_bf16_forward_k2_recurrence.cu": "common",
    "csrc/infer/tmix/infer_common_tmix_fp16_forward.cu": "common",
    "csrc/infer/cmix/infer_common_cmix_fp16_forward.cu": "common",
    "csrc/infer/elementwise/infer_common_elementwise_fp16_forward.cu": "common",
    "csrc/infer/wkv7/infer_common_recurrent_fp32io16_forward_varlen.cu": "common",
    "csrc/infer/wkv7/infer_sm80_recurrent_fp16_forward_varlen.cu": "sm80",
    "csrc/pretrain/wkv7/pretrain_common_head_l2wrap_ce_bf16_forward.cu": "common",
    "csrc/pretrain/wkv7/pretrain_common_l2wrap_ce_bf16_forward_backward.cu": "common",
    "csrc/pretrain/wkv7/pretrain_common_tmix_a_gate_bf16_forward_backward.cu": "common",
    "csrc/pretrain/wkv7/pretrain_common_tmix_lnx_rkvres_xg_bf16_forward_backward.cu": "common",
    "csrc/pretrain/wkv7/pretrain_common_tmix_vres_gate_bf16_forward_backward.cu": "common",
    "csrc/pretrain/wkv7/pretrain_sm90_cmix_bf16_forward_backward.cu": "sm90",
    "csrc/pretrain/wkv7/pretrain_sm90_tmix_kk_pre_bf16_forward_backward.cu": "sm90",
    "csrc/pretrain/wkv7/pretrain_sm90_tmix_mix6_bf16_forward_backward.cu": "sm90",
    "csrc/rl_infctx/wkv7/rl_infctx_common_chunk_fp32io16_backward_replay.cu": "common",
    "csrc/rl_infctx/wkv7/rl_infctx_common_chunk_fp32io16_forward_materialized.cu": "common",
    "csrc/rl_infctx/wkv7/rl_infctx_common_chunk_fp32io16_forward_recompute.cu": "common",
}

_ARCHITECTURE_PATTERN = re.compile(
    r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)(?:a)?(?:\+PTX)?",
    re.IGNORECASE,
)


def parse_torch_cuda_arch_list(value: str) -> tuple[tuple[int, int], ...]:
    """Parse the numeric subset of ``TORCH_CUDA_ARCH_LIST`` fail closed."""

    tokens = tuple(token for token in re.split(r"[;,\s]+", value.strip()) if token)
    if not tokens:
        raise ValueError("TORCH_CUDA_ARCH_LIST must name at least one architecture")
    capabilities: list[tuple[int, int]] = []
    for token in tokens:
        match = _ARCHITECTURE_PATTERN.fullmatch(token)
        if match is None:
            raise ValueError(
                "FlashRWKV requires numeric CUDA architecture tokens such as "
                f"9.0 or 12.0+PTX; got {token!r}"
            )
        capabilities.append((int(match["major"]), int(match["minor"])))
    return tuple(capabilities)


def validate_wheel_architectures(
    requested: str | None,
    *,
    detected: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Reject full-extension builds below the SM90 package minimum."""

    if requested is not None:
        capabilities = parse_torch_cuda_arch_list(requested)
    elif detected is not None:
        capabilities = (detected,)
    else:
        raise ValueError(
            "FlashRWKV native builds require TORCH_CUDA_ARCH_LIST with targets "
            ">= 9.0 (SM90 minimum) when no CUDA device is available"
        )
    unsupported = tuple(capability for capability in capabilities if capability < (9, 0))
    if unsupported:
        rendered = ", ".join(f"{major}.{minor}" for major, minor in unsupported)
        raise RuntimeError(
            "FlashRWKV's complete CUDA extension requires compute capability "
            f">= 9.0 because its SM90 translation units use float2 atomicAdd; "
            f"unsupported target(s): {rendered}"
        )
    return capabilities
