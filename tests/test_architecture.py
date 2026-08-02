# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from flash_rwkv.architecture import (
    BUILD_TARGET_MATRIX,
    COMPILE_ONLY_ARCHITECTURES,
    MINIMUM_WHEEL_ARCHITECTURE,
    RUNTIME_VALIDATED_ARCHITECTURES,
    TRANSLATION_UNIT_ARCHITECTURES,
    parse_torch_cuda_arch_list,
    validate_wheel_architectures,
)
from flash_rwkv.registry import (
    INFERENCE_OPERATOR_SPECS,
    KERNEL_SPECS,
    TRAINING_OPERATOR_SPECS,
)

ROOT = Path(__file__).parents[1]


def _extension_cuda_sources() -> set[str]:
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg != "sources":
            continue
        assert isinstance(node.value, ast.List)
        return {
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant)
            and isinstance(element.value, str)
            and element.value.endswith(".cu")
        }
    raise AssertionError("CUDAExtension sources list not found")


def test_architecture_contract_separates_build_and_runtime_evidence() -> None:
    assert MINIMUM_WHEEL_ARCHITECTURE == "sm90"
    assert COMPILE_ONLY_ARCHITECTURES == ("sm90", "sm120")
    assert RUNTIME_VALIDATED_ARCHITECTURES == ("sm120",)
    assert BUILD_TARGET_MATRIX == (
        ("sm80", "fail_closed"),
        ("sm90", "compile_only"),
        ("sm120", "runtime_validated"),
    )


def test_every_cuda_translation_unit_has_exact_minimum_architecture_metadata() -> None:
    assert set(TRANSLATION_UNIT_ARCHITECTURES) == _extension_cuda_sources()
    assert {
        path
        for path, architecture in TRANSLATION_UNIT_ARCHITECTURES.items()
        if architecture == "sm80"
    } == {
        "csrc/infer/wkv7/infer_sm80_recurrent_fp16_forward_varlen.cu",
    }
    assert {
        path
        for path, architecture in TRANSLATION_UNIT_ARCHITECTURES.items()
        if architecture == "sm90"
    } == {
        "csrc/pretrain/wkv7/pretrain_sm90_cmix_bf16_forward_backward.cu",
        "csrc/pretrain/wkv7/pretrain_sm90_tmix_kk_pre_bf16_forward_backward.cu",
        "csrc/pretrain/wkv7/pretrain_sm90_tmix_mix6_bf16_forward_backward.cu",
    }


def test_architecture_specific_translation_units_retain_their_isa_evidence() -> None:
    sm80 = (
        ROOT / "csrc/infer/wkv7/infer_sm80_recurrent_fp16_forward_varlen.cu"
    ).read_text(encoding="utf-8")
    assert "cp.async.cg.shared.global" in sm80
    assert "cp.async.wait_group" in sm80

    for path, architecture in TRANSLATION_UNIT_ARCHITECTURES.items():
        if architecture != "sm90":
            continue
        source = (ROOT / path).read_text(encoding="utf-8")
        assert "atomicAdd(reinterpret_cast<float2*>" in source


def test_complete_wheel_rejects_targets_below_sm90() -> None:
    with pytest.raises(RuntimeError, match="requires compute capability >= 9.0"):
        validate_wheel_architectures("8.0")
    assert validate_wheel_architectures("9.0;12.0+PTX") == ((9, 0), (12, 0))
    assert validate_wheel_architectures(None, detected=(12, 0)) == ((12, 0),)


def test_setup_metadata_works_without_cuda_architecture_input() -> None:
    environment = os.environ.copy()
    environment.pop("TORCH_CUDA_ARCH_LIST", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        (sys.executable, "setup.py", "--name"),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "flash-rwkv"


def test_setup_build_ext_without_architecture_or_device_fails_closed() -> None:
    environment = os.environ.copy()
    environment.pop("TORCH_CUDA_ARCH_LIST", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        (sys.executable, "setup.py", "build_ext"),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "require TORCH_CUDA_ARCH_LIST" in result.stderr


@pytest.mark.parametrize("value", ["", "Ampere", "9", "sm90", "9.0+sass"])
def test_architecture_parser_rejects_ambiguous_build_targets(value: str) -> None:
    with pytest.raises(ValueError):
        parse_torch_cuda_arch_list(value)


def test_registry_does_not_turn_compile_targets_into_runtime_claims() -> None:
    internal_kernels = tuple(spec for spec in KERNEL_SPECS if spec.maturity != "external")
    external_kernels = tuple(spec for spec in KERNEL_SPECS if spec.maturity == "external")
    assert all(
        spec.validated_architectures == ("sm120",) for spec in internal_kernels
    )
    assert all(
        spec.package_minimum_architecture == "sm90" for spec in internal_kernels
    )
    assert all(
        spec.translation_unit_architecture is None
        and spec.package_minimum_architecture is None
        and not spec.validated_architectures
        for spec in external_kernels
    )
    assert all(
        spec.validated_architectures == ("sm120",)
        for spec in (*TRAINING_OPERATOR_SPECS, *INFERENCE_OPERATOR_SPECS)
    )
    assert all(
        spec.package_minimum_architecture == "sm90"
        for spec in (*TRAINING_OPERATOR_SPECS, *INFERENCE_OPERATOR_SPECS)
    )
    sm90_operators = {
        spec.name
        for spec in TRAINING_OPERATOR_SPECS
        if spec.translation_unit_architecture == "sm90"
    }
    assert sm90_operators == {
        "pretrain_cmix_bf16",
        "pretrain_tmix_kk_pre_bf16",
        "pretrain_tmix_mix6_bf16",
    }
