"""Static checks for the active module-local native source contract."""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOTS = (ROOT / "csrc" / "sm90", ROOT / "csrc" / "sm120")
GLOBAL_NATIVE = {
    Path("csrc/bindings.cpp"),
    Path("csrc/registration.cpp"),
    Path("csrc/validation.cpp"),
    Path("csrc/validation/recurrent_metadata.cu"),
}
MIRRORED_MODULES = (
    "cmix/mix",
    "cmix/sparse",
    "embedding",
    "head/l2wrap_ce",
    "head/linear",
    "loss/l2wrap_ce",
    "rl_infctx/wkv7",
    "tmix/a_gate",
    "tmix/kk_a_gate",
    "tmix/kk_pre",
    "tmix/linear",
    "tmix/lnx_rkvres_xg",
    "tmix/mix6",
    "tmix/normalization",
    "tmix/vres_gate",
    "tmix/wkv7",
)
PUBLIC_INFERENCE_MODULES = (
    "flash_rwkv.embedding",
    "flash_rwkv.tmix.mix6",
    "flash_rwkv.tmix.kk_a_gate",
    "flash_rwkv.tmix.linear",
    "flash_rwkv.tmix.lnx_rkvres_xg",
    "flash_rwkv.tmix.normalization",
    "flash_rwkv.tmix.vres_gate",
    "flash_rwkv.tmix.wkv7",
    "flash_rwkv.cmix.mix",
    "flash_rwkv.cmix.sparse",
    "flash_rwkv.head.linear",
)
PUBLIC_TRAINING_MODULES = (
    "flash_rwkv.tmix.wkv7",
    "flash_rwkv.tmix.wkv7.statetune",
    "flash_rwkv.tmix.a_gate",
    "flash_rwkv.tmix.vres_gate",
    "flash_rwkv.tmix.mix6",
    "flash_rwkv.tmix.kk_pre",
    "flash_rwkv.tmix.lnx_rkvres_xg",
    "flash_rwkv.cmix.mix",
    "flash_rwkv.loss.l2wrap_ce",
    "flash_rwkv.head.l2wrap_ce",
    "flash_rwkv.rl_infctx.wkv7",
)
PUBLIC_TRAINING_PREFIXES = ("pretrain_", "statetune_", "rl_infctx_")


def _active_native_files() -> set[Path]:
    return {
        path.relative_to(ROOT)
        for native_root in NATIVE_ROOTS
        for path in native_root.rglob("*")
        if path.suffix in {".cpp", ".cu"}
    }


def _setup_sources() -> set[Path]:
    tree = ast.parse((ROOT / "setup.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "CUDAExtension":
            continue
        sources = next(
            keyword.value for keyword in node.keywords if keyword.arg == "sources"
        )
        assert isinstance(sources, ast.List)
        values = set()
        for item in sources.elts:
            assert isinstance(item, ast.Constant) and isinstance(item.value, str)
            values.add(Path(item.value))
        return values
    raise AssertionError("setup.py does not define a CUDAExtension source list")


def test_active_native_sources_are_paired_and_listed() -> None:
    native_files = _active_native_files()
    module_files = native_files - GLOBAL_NATIVE
    assert module_files
    assert module_files <= _setup_sources()
    assert _setup_sources() == module_files | GLOBAL_NATIVE

    for relative in module_files:
        if relative.suffix == ".cpp":
            assert relative.with_suffix(".cu") in module_files, relative
        if relative.suffix == ".cu":
            assert relative.with_suffix(".cpp") in module_files, relative


def test_module_paths_are_mirrored() -> None:
    for module in MIRRORED_MODULES:
        assert (ROOT / "flash_rwkv" / module).exists(), module
        assert (ROOT / "tests" / module).exists(), module
        assert (ROOT / "benchmarks" / module).exists(), module
        assert any(
            (native_root / module).exists() for native_root in NATIVE_ROOTS
        ), module


def test_forbidden_global_and_legacy_paths_are_absent_from_active_tree() -> None:
    assert not (ROOT / "csrc" / "common").exists()
    assert not (ROOT / "flash_rwkv" / "elementwise").exists()
    assert not (ROOT / "csrc" / "elementwise").exists()

    active_paths = {
        path.relative_to(ROOT)
        for root in (
            ROOT / "flash_rwkv",
            ROOT / "csrc" / "sm90",
            ROOT / "csrc" / "sm120",
        )
        for path in root.rglob("*")
        if path.is_file()
    }
    rendered = "\n".join(str(path) for path in sorted(active_paths))
    for forbidden in (
        "infer_common_",
        "pretrain_common_",
        "_registration.cpp",
        "rwkv7_fast_ops_fp16",
        "rwkv7_wkv_fp16_v2",
    ):
        assert forbidden not in rendered


def test_module_cuda_files_have_provenance_headers() -> None:
    for relative in sorted(_active_native_files() - GLOBAL_NATIVE):
        if relative.suffix != ".cu":
            continue
        text = (ROOT / relative).read_text()
        assert "SPDX-License-Identifier:" in text, relative
        assert "revision" in text.lower(), relative


def test_python_surface_stays_operator_only() -> None:
    """FlashRWKV exposes operators; model classes and model forward APIs stay external."""

    for path in sorted((ROOT / "flash_rwkv").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert not re.search(r"(?:rwkv|transformer|model)", node.name, re.I), path
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward":
                argument_names = {argument.arg for argument in node.args.args}
                assert not argument_names.intersection(
                    {"input_ids", "attention_mask", "position_ids"}
                ), path


def test_root_exports_all_public_inference_operators() -> None:
    import flash_rwkv

    expected = {}
    for module_name in PUBLIC_INFERENCE_MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            if not name.startswith("infer_"):
                continue
            assert name not in expected, (
                f"public inference operator {name} is exported by both "
                f"{expected[name].__module__} and {module_name}"
            )
            expected[name] = getattr(module, name)

    assert len(expected) == 44
    root_inference_names = {
        name for name in flash_rwkv.__all__ if name.startswith("infer_")
    }
    assert root_inference_names == set(expected)
    for name, operator in expected.items():
        assert getattr(flash_rwkv, name) is operator


def test_root_exports_all_public_training_operators() -> None:
    import flash_rwkv

    expected = {}
    for module_name in PUBLIC_TRAINING_MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            if not name.startswith(PUBLIC_TRAINING_PREFIXES):
                continue
            assert name not in expected, (
                f"public training operator {name} is exported by both "
                f"{expected[name].__module__} and {module_name}"
            )
            expected[name] = getattr(module, name)

    assert len(expected) == 12
    root_training_names = {
        name
        for name in flash_rwkv.__all__
        if name.startswith(PUBLIC_TRAINING_PREFIXES)
    }
    assert root_training_names == set(expected)
    for name, operator in expected.items():
        assert getattr(flash_rwkv, name) is operator


def test_fp16_elapsed_advance_stays_in_the_wkv7_owner() -> None:
    cuda_source = (
        ROOT
        / "csrc/sm120/tmix/wkv7/infer_recurrent_fp16_forward_varlen.cu"
    ).read_text()
    cpp_source = (
        ROOT
        / "csrc/sm120/tmix/wkv7/infer_recurrent_fp16_forward_varlen.cpp"
    ).read_text()
    python_source = (ROOT / "flash_rwkv/tmix/wkv7/__init__.py").read_text()

    assert "advance_i32_varlen_kernel" in cuda_source
    assert "recurrent_fp16_advance_i32_varlen" in cpp_source
    assert "infer_recurrent_fp16_advance_i32_varlen" in python_source
    assert "elementwise" not in python_source
