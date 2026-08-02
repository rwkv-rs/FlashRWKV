# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tarfile
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
    assert MINIMUM_WHEEL_ARCHITECTURE == "sm60"
    assert COMPILE_ONLY_ARCHITECTURES == ("sm90", "sm120")
    assert RUNTIME_VALIDATED_ARCHITECTURES == ("sm61", "sm120")
    assert BUILD_TARGET_MATRIX == (
        ("sm61", "runtime_validated"),
        ("sm80", "compile_only"),
        ("sm90", "compile_only"),
        ("sm120", "runtime_validated"),
    )


def test_every_cuda_translation_unit_has_exact_minimum_architecture_metadata() -> None:
    assert set(TRANSLATION_UNIT_ARCHITECTURES) == _extension_cuda_sources()
    assert {
        path
        for path, architecture in TRANSLATION_UNIT_ARCHITECTURES.items()
        if architecture == "sm80"
    } == set()
    assert {
        path
        for path, architecture in TRANSLATION_UNIT_ARCHITECTURES.items()
        if architecture == "sm90"
    } == set()


def test_architecture_specific_translation_units_retain_their_isa_evidence() -> None:
    sm80 = (
        ROOT / "csrc/infer/wkv7/infer_sm80_recurrent_fp16_forward_varlen.cu"
    ).read_text(encoding="utf-8")
    assert "cp.async.cg.shared.global" in sm80
    assert "cp.async.wait_group" in sm80
    assert "#if __CUDA_ARCH__ >= 800" in sm80

    compat = (ROOT / "csrc/bf16_compat.cuh").read_text(encoding="utf-8")
    assert "__CUDA_ARCH__ >= 800" in compat
    assert "__bfloat162float(value.x)" in compat


def test_complete_wheel_rejects_targets_below_sm60() -> None:
    with pytest.raises(RuntimeError, match="requires compute capability >= 6.0"):
        validate_wheel_architectures("5.2")
    assert validate_wheel_architectures("6.1;12.0+PTX") == ((6, 1), (12, 0))
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


@pytest.mark.parametrize(
    "command",
    (
        "build",
        "build_ext",
        "build_py",
        "bdist",
        "bdist_egg",
        "bdist_rpm",
        "bdist_wheel",
        "develop",
        "editable_wheel",
        "install",
        "install_lib",
    ),
)
def test_setup_native_commands_without_architecture_or_device_fail_closed(
    command: str,
) -> None:
    environment = os.environ.copy()
    environment.pop("TORCH_CUDA_ARCH_LIST", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        (sys.executable, "setup.py", command),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "require TORCH_CUDA_ARCH_LIST" in result.stderr
    assert ">= 6.0 (SM60 minimum)" in result.stderr


def test_sdist_contains_complete_native_sources_and_preserves_build_contract(
    tmp_path: Path,
) -> None:
    source_tree = tmp_path / "source"
    shutil.copytree(
        ROOT,
        source_tree,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            ".cache",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "artifacts",
            "build",
            "dist",
            "*.egg-info",
            "*.so",
        ),
    )
    environment = os.environ.copy()
    environment.pop("TORCH_CUDA_ARCH_LIST", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    dist_dir = tmp_path / "dist"
    result = subprocess.run(
        (sys.executable, "setup.py", "sdist", "--dist-dir", str(dist_dir)),
        cwd=source_tree,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    archives = tuple(dist_dir.glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as archive:
        member_names = tuple(archive.getnames())
        top_levels = {name.split("/", 1)[0] for name in member_names}
        assert len(top_levels) == 1
        prefix = f"{top_levels.pop()}/"
        relative_members = {
            name.removeprefix(prefix)
            for name in member_names
            if name.startswith(prefix)
        }
        extract_root = tmp_path / "extracted"
        archive.extractall(extract_root, filter="data")

    native_sources = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "csrc").rglob("*")
        if path.is_file()
    }
    assert native_sources <= relative_members
    assert {
        "MANIFEST.in",
        "LICENSE",
        "LICENSES/Apache-2.0.txt",
        "NOTICE",
        "README.md",
        "flash_rwkv/architecture.py",
        "flash_rwkv/provenance.py",
        "flash_rwkv/registry.py",
        "pyproject.toml",
        "setup.py",
    } <= relative_members
    placeholder = "sm" + "xx"
    assert not [name for name in relative_members if placeholder in name.lower()]

    extracted_source = next(extract_root.iterdir())
    extracted_build = subprocess.run(
        (sys.executable, "setup.py", "build"),
        cwd=extracted_source,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert extracted_build.returncode != 0
    assert "require TORCH_CUDA_ARCH_LIST" in extracted_build.stderr
    assert ">= 6.0 (SM60 minimum)" in extracted_build.stderr


@pytest.mark.parametrize("value", ["", "Ampere", "9", "sm90", "9.0+sass"])
def test_architecture_parser_rejects_ambiguous_build_targets(value: str) -> None:
    with pytest.raises(ValueError):
        parse_torch_cuda_arch_list(value)


def test_registry_does_not_turn_compile_targets_into_runtime_claims() -> None:
    internal_kernels = tuple(spec for spec in KERNEL_SPECS if spec.maturity != "external")
    external_kernels = tuple(spec for spec in KERNEL_SPECS if spec.maturity == "external")
    assert all(
        spec.validated_architectures == ("sm61", "sm120") for spec in internal_kernels
    )
    assert all(
        spec.package_minimum_architecture == "sm60" for spec in internal_kernels
    )
    assert all(
        spec.translation_unit_architecture is None
        and spec.package_minimum_architecture is None
        and not spec.validated_architectures
        for spec in external_kernels
    )
    assert all(
        spec.validated_architectures == ("sm61", "sm120")
        for spec in (*TRAINING_OPERATOR_SPECS, *INFERENCE_OPERATOR_SPECS)
    )
    assert all(
        spec.package_minimum_architecture == "sm60"
        for spec in (*TRAINING_OPERATOR_SPECS, *INFERENCE_OPERATOR_SPECS)
    )
    architecture_specific_operators = {
        spec.name
        for spec in TRAINING_OPERATOR_SPECS
        if spec.translation_unit_architecture != "common"
    }
    assert architecture_specific_operators == set()
