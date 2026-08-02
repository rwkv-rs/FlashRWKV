# SPDX-License-Identifier: MIT

"""Machine-readable provenance for imported native source families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImportedSourceFamily:
    """One fixed upstream revision and the native files derived from it."""

    name: str
    repository: str
    revision: str
    license: str
    paths: tuple[str, ...]


IMPORTED_SOURCE_FAMILIES: tuple[ImportedSourceFamily, ...] = (
    ImportedSourceFamily(
        name="flashkda-chunk",
        repository="https://github.com/MoonshotAI/FlashKDA",
        revision="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b",
        license="MIT",
        paths=(
            "csrc/infer/wkv7/infer_common_chunk_bf16_bindings.cpp",
            "csrc/infer/wkv7/infer_common_chunk_bf16_forward_k1_prepare.cu",
            "csrc/infer/wkv7/infer_common_chunk_bf16_forward_k2_recurrence.cu",
        ),
    ),
    ImportedSourceFamily(
        name="vllm-recurrent",
        repository="https://github.com/rwkv-rs/vllm-rwkv",
        revision="6d683f9e49a2997e405c47edc147872c8609513b",
        license="Apache-2.0",
        paths=(
            "csrc/infer/wkv7/infer_common_recurrent_varlen_bindings.cpp",
            "csrc/infer/wkv7/infer_sm80_recurrent_fp16_forward_varlen.cu",
            "csrc/infer/wkv7/infer_common_recurrent_fp32io16_forward_varlen.cu",
        ),
    ),
    ImportedSourceFamily(
        name="albatross-fused-infer",
        repository="https://github.com/BlinkDL/Albatross",
        revision="ee3308f6922e59f2166c7fac3c5a192340a2b48e",
        license="Apache-2.0",
        paths=(
            "csrc/infer/wkv7/infer_common_fused_fp16_forward_registration.cpp",
            "csrc/infer/wkv7/infer_common_fused_fp16_forward.cu",
        ),
    ),
    ImportedSourceFamily(
        name="rwkv-lm-pretrain",
        repository="https://github.com/BlinkDL/RWKV-LM",
        revision="952102498e9ed367ea0a59ee64106916d474d30f",
        license="Apache-2.0",
        paths=(
            "csrc/pretrain/wkv7/pretrain_common_recurrent_fp32io16_bindings.cpp",
            "csrc/pretrain/wkv7/pretrain_common_recurrent_fp32io16_forward.cu",
            "csrc/pretrain/wkv7/pretrain_common_recurrent_fp32io16_backward.cu",
            "csrc/pretrain/wkv7/pretrain_common_head_l2wrap_ce_bf16_forward_registration.cpp",
            "csrc/pretrain/wkv7/pretrain_common_head_l2wrap_ce_bf16_forward.cu",
            "csrc/pretrain/wkv7/pretrain_common_cmix_bf16_forward_backward_registration.cpp",
            "csrc/pretrain/wkv7/pretrain_sm90_cmix_bf16_forward_backward.cu",
            "csrc/pretrain/wkv7/pretrain_common_l2wrap_ce_bf16_forward_backward_registration.cpp",
            "csrc/pretrain/wkv7/pretrain_common_l2wrap_ce_bf16_forward_backward.cu",
            "csrc/pretrain/wkv7/pretrain_common_tmix_a_gate_bf16_forward_backward_registration.cpp",
            "csrc/pretrain/wkv7/pretrain_common_tmix_a_gate_bf16_forward_backward.cu",
            "csrc/pretrain/wkv7/pretrain_common_tmix_kk_pre_bf16_forward_backward_registration.cpp",
            "csrc/pretrain/wkv7/pretrain_sm90_tmix_kk_pre_bf16_forward_backward.cu",
            "csrc/pretrain/wkv7/pretrain_common_tmix_lnx_rkvres_xg_bf16_forward_backward_registration.cpp",
            "csrc/pretrain/wkv7/pretrain_common_tmix_lnx_rkvres_xg_bf16_forward_backward.cu",
            "csrc/pretrain/wkv7/pretrain_common_tmix_mix6_bf16_forward_backward_registration.cpp",
            "csrc/pretrain/wkv7/pretrain_sm90_tmix_mix6_bf16_forward_backward.cu",
            "csrc/pretrain/wkv7/pretrain_common_tmix_vres_gate_bf16_forward_backward_registration.cpp",
            "csrc/pretrain/wkv7/pretrain_common_tmix_vres_gate_bf16_forward_backward.cu",
        ),
    ),
)


_BY_NAME = {family.name: family for family in IMPORTED_SOURCE_FAMILIES}


def imported_source_family(name: str) -> ImportedSourceFamily:
    """Resolve one fixed source family by its stable internal name."""

    try:
        return _BY_NAME[name]
    except KeyError as error:
        raise KeyError(f"unknown imported source family: {name!r}") from error
