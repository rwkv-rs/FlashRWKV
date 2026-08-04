# SPDX-License-Identifier: MIT

from __future__ import annotations

from flash_rwkv.registry import INFERENCE_OPERATOR_SPECS, inference_operator_specs


def test_inference_registry_binds_fixed_and_packed_public_ops() -> None:
    assert inference_operator_specs() == INFERENCE_OPERATOR_SPECS
    assert {spec.name for spec in INFERENCE_OPERATOR_SPECS} == {
        "infer_tmix_mix6_fp16_forward",
        "infer_tmix_kk_a_gate_fp16_forward",
        "infer_tmix_lnx_rkvres_xg_fp16_forward",
        "infer_tmix_vres_gate_fp16_forward",
        "infer_cmix_mix_fp16_forward",
        "infer_tmix_mix6_fp16_varlen_forward",
        "infer_cmix_mix_fp16_varlen_forward",
    }
    assert {
        spec.source_revision for spec in INFERENCE_OPERATOR_SPECS
    } == {
        "ee3308f6922e59f2166c7fac3c5a192340a2b48e",
        "55b6894079e8a9966ceed1adac54dd0dfd084018",
    }
