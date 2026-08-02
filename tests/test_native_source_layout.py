# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
CSRC = ROOT / "csrc"
EXPECTED_NATIVE_OPS = {
    "infer_chunk_bf16_forward_k1_prepare",
    "infer_chunk_bf16_forward_k2_recurrence",
    "materialized_chunk_fp32",
    "pretrain_recurrent_fp32io16_backward",
    "pretrain_recurrent_fp32io16_forward",
    "recompute_chunk_fp32",
    "recurrent_fp16",
    "recurrent_fp32",
}
EXPECTED_ARGUMENTS = {
    "recurrent_fp32": (
        "query_start_loc", "state_indices", "state", "r", "log_decay", "k",
        "v", "a", "b", "output", "scale",
    ),
    "recurrent_fp16": (
        "query_start_loc", "state_indices", "state", "r", "log_decay", "k",
        "v", "a", "b", "output", "scale",
    ),
    "pretrain_recurrent_fp32io16_forward": (
        "sequence_chunk_offsets", "chunk_token_starts", "chunk_token_ends",
        "state", "r", "log_decay", "k", "v", "a", "b", "output",
        "boundary", "state_dot_a", "scale",
    ),
    "materialized_chunk_fp32": (
        "sequence_chunk_offsets", "chunk_token_starts", "chunk_token_ends",
        "state_indices", "state", "r", "log_decay", "k", "v", "a", "b",
        "output", "transform", "bias", "boundary", "build_warps", "stages",
        "state_tile", "scale", "state_dot_a",
    ),
    "pretrain_recurrent_fp32io16_backward": (
        "sequence_chunk_offsets", "chunk_token_starts", "chunk_token_ends",
        "final_state", "r", "log_decay", "k", "v", "a", "b",
        "state_dot_a", "grad_output", "grad_final_state", "boundary",
        "grad_r", "grad_log_decay", "grad_k", "grad_v", "grad_a", "grad_b",
        "grad_initial_state", "scale",
    ),
    "infer_chunk_bf16_forward_k1_prepare": (
        "chunk_token_starts", "chunk_token_ends", "r", "log_decay", "k", "v",
        "a", "b", "chunk_transform", "chunk_bias", "token_transform",
        "token_bias", "scale",
    ),
    "infer_chunk_bf16_forward_k2_recurrence": (
        "sequence_chunk_offsets", "chunk_token_starts", "chunk_token_ends",
        "state", "output", "chunk_transform", "chunk_bias", "token_transform",
        "token_bias",
    ),
    "recompute_chunk_fp32": (
        "sequence_chunk_offsets", "chunk_token_starts", "chunk_token_ends",
        "state_indices", "state", "r", "log_decay", "k", "v", "a", "b",
        "output", "boundary", "scale",
    ),
}


def _extension_sources() -> set[str]:
    tree = ast.parse((ROOT / "setup.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg != "sources":
            continue
        assert isinstance(node.value, ast.List)
        return {
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant)
            and isinstance(element.value, str)
        }
    raise AssertionError("CUDAExtension sources list not found")


def test_binding_responsibilities_have_distinct_translation_units() -> None:
    sources = _extension_sources()
    assert {
        "csrc/bindings.cpp",
        "csrc/registration.cpp",
        "csrc/validation.cpp",
    } <= sources

    bindings = (CSRC / "bindings.cpp").read_text()
    validation = (CSRC / "validation.cpp").read_text()
    registration = (CSRC / "registration.cpp").read_text()
    assert "PYBIND11_MODULE" not in bindings
    assert "RecurrentDimensions check_recurrent_layout(" not in bindings
    assert "PYBIND11_MODULE" in registration
    assert "check_recurrent_layout(" in validation


def test_cuda_extension_sources_have_unique_ninja_object_stems() -> None:
    sources_by_stem: dict[str, list[str]] = {}
    for source in sorted(_extension_sources()):
        sources_by_stem.setdefault(Path(source).stem, []).append(source)

    collisions = {
        stem: sources
        for stem, sources in sources_by_stem.items()
        if len(sources) > 1
    }
    assert collisions == {}


def test_gpu_workflow_owns_its_exact_fla_reference_environment() -> None:
    workflow = (ROOT / ".github/workflows/pro6000-gpu.yml").read_text()
    fla_revision = "fcd4502ab957513a3d97dbd8aa64851e5e4dba11"
    fla_requirement = (
        "flash-linear-attention @ git+https://github.com/"
        f"rwkv-rs/fla-rwkv.git@{fla_revision}"
    )

    assert "--no-deps --no-build-isolation -e ." in workflow
    requirement_position = workflow.index(fla_requirement)
    assert "--no-deps" in workflow[requirement_position - 100:requirement_position]
    assert workflow.index("--no-deps --no-build-isolation -e .") < workflow.index(
        fla_requirement
    )
    assert f'revision = "{fla_revision}"' in workflow
    assert 'distribution("flash-linear-attention")' in workflow
    assert 'direct_url["url"] != "https://github.com/rwkv-rs/fla-rwkv.git"' in workflow
    assert 'for name in ("fla", "fla.ops.rwkv7")' in workflow
    assert "module_path.is_relative_to(package_root)" in workflow


def test_channel_mix_sources_are_built_from_the_pretrain_family() -> None:
    sources = _extension_sources()
    expected = {
        "csrc/pretrain/wkv7/pretrain_smxx_cmix_bf16_forward_backward_registration.cpp",
        "csrc/pretrain/wkv7/pretrain_smxx_cmix_bf16_forward_backward.cu",
    }
    assert expected <= sources
    for source in expected:
        contents = (ROOT / source).read_text()
        assert "SPDX-License-Identifier: Apache-2.0" in contents
        assert "952102498e9ed367ea0a59ee64106916d474d30f" in contents


def test_l2wrap_sources_are_built_from_the_pretrain_family() -> None:
    sources = _extension_sources()
    expected = {
        "csrc/pretrain/wkv7/pretrain_smxx_l2wrap_ce_bf16_forward_backward_registration.cpp",
        "csrc/pretrain/wkv7/pretrain_smxx_l2wrap_ce_bf16_forward_backward.cu",
    }
    assert expected <= sources
    for source in expected:
        contents = (ROOT / source).read_text()
        assert "SPDX-License-Identifier: Apache-2.0" in contents
        assert "952102498e9ed367ea0a59ee64106916d474d30f" in contents


def test_native_sources_are_owned_by_workload_wkv7_trees() -> None:
    sources = _extension_sources()
    workload_sources = {
        source
        for source in sources
        if source.endswith((".cpp", ".cu"))
        and source not in {
            "csrc/bindings.cpp",
            "csrc/registration.cpp",
            "csrc/validation.cpp",
        }
    }
    assert workload_sources
    assert all((ROOT / source).is_file() for source in sources)
    assert all(
        source.startswith(
            (
                "csrc/pretrain/wkv7/",
                "csrc/rl_infctx/wkv7/",
                "csrc/statetune/wkv7/",
                "csrc/infer/wkv7/",
            )
        )
        for source in workload_sources
    )
    assert not any(
        path.is_file()
        for legacy in ("chunk", "kda", "recurrent")
        for path in (CSRC / legacy).rglob("*")
    )


def test_statetune_registration_points_to_shared_recurrence_sources() -> None:
    source = (
        CSRC
        / "statetune/wkv7/statetune_smxx_recurrent_fp32io16_registration.cpp"
    )
    contents = source.read_text()
    assert source.relative_to(ROOT).as_posix() in _extension_sources()
    assert (
        "csrc/pretrain/wkv7/pretrain_smxx_recurrent_fp32io16_forward.cu"
        in contents
    )
    assert (
        "csrc/pretrain/wkv7/pretrain_smxx_recurrent_fp32io16_backward.cu"
        in contents
    )
    assert "_statetune_recurrent_source_manifest" in contents


def test_all_workloads_own_csrc_tests_and_benchmarks() -> None:
    for workload in ("pretrain", "rl_infctx", "statetune", "infer"):
        csrc = ROOT / "csrc" / workload / "wkv7"
        tests = ROOT / "tests" / workload / "wkv7"
        benchmarks = ROOT / "benchmarks" / workload / "wkv7"
        assert any(path.is_file() for path in csrc.iterdir()), workload
        assert any(path.name.startswith("test_") for path in tests.glob("*.py")), workload
        assert any(
            path.name.startswith(("benchmark_", "autotune_"))
            for path in benchmarks.glob("*.py")
        ), workload


def test_registration_preserves_exact_native_operator_surface() -> None:
    registration = (CSRC / "registration.cpp").read_text()
    blocks = re.findall(r"module\.def\((.*?)\);", registration, re.DOTALL)
    registered = {}
    for block in blocks:
        name = re.search(r'^\s*"([^"]+)"', block)
        assert name is not None
        registered[name.group(1)] = tuple(re.findall(r'py::arg\("([^"]+)"\)', block))
    assert set(registered) == EXPECTED_NATIVE_OPS
    assert registered == EXPECTED_ARGUMENTS
