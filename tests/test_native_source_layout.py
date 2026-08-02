# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import re
import subprocess
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


def test_tracked_paths_and_text_do_not_use_architecture_placeholders() -> None:
    placeholder = "sm" + "xx"
    tracked = subprocess.run(
        ("git", "ls-files"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert not [path for path in tracked if placeholder in path.lower()]
    text_paths = [
        path
        for path in tracked
        if Path(path).suffix
        in {
            ".cpp",
            ".cu",
            ".cuh",
            ".json",
            ".md",
            ".py",
            ".sh",
            ".toml",
            ".yml",
            ".yaml",
        }
        or path == "NOTICE"
    ]
    offenders = [
        path
        for path in text_paths
        if placeholder in (ROOT / path).read_text(encoding="utf-8").lower()
    ]
    assert offenders == []
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


def _function_source(relative_path: str, name: str) -> str:
    source = (ROOT / relative_path).read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            assert node.end_lineno is not None
            return "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])
    raise AssertionError(f"function {name!r} not found in {relative_path}")


def test_binding_responsibilities_have_distinct_translation_units() -> None:
    sources = _extension_sources()
    assert {
        "csrc/bindings.cpp",
        "csrc/registration.cpp",
        "csrc/validation.cpp",
        "csrc/validation/recurrent_metadata.cu",
    } <= sources

    bindings = (CSRC / "bindings.cpp").read_text()
    validation = (CSRC / "validation.cpp").read_text()
    registration = (CSRC / "registration.cpp").read_text()
    assert "PYBIND11_MODULE" not in bindings
    assert "module.def" not in bindings
    assert "torch::Tensor" not in bindings
    assert "register_infer_recurrent_bindings(module)" in bindings
    assert "register_pretrain_recurrent_bindings(module)" in bindings
    assert "register_rl_infctx_experimental_bindings(module)" in bindings
    assert "register_infer_experimental_bindings(module)" in bindings
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


def test_gpu_workflow_binds_dispatch_evidence_to_the_exact_pull_head() -> None:
    workflow = (ROOT / ".github/workflows/pro6000-gpu.yml").read_text()
    quick_workflow = (ROOT / ".github/workflows/quick-contract.yml").read_text()

    assert "pr_number:" in workflow
    assert "source_revision:" in workflow
    assert "context.sha !== sourceRevision" in workflow
    assert "pull.head.sha !== sourceRevision" in workflow
    assert "pull.head.sha !== process.env.HEAD_SHA" in workflow
    assert "refusing to write evidence" in workflow
    assert "ref: ${{ steps.target.outputs.head_sha }}" in workflow
    assert "name: flash-rwkv-pro6000-${{ steps.target.outputs.head_sha }}" in workflow
    assert "runtime_semantic_revision" in workflow
    assert "listJobsForWorkflowRunAttempt" in workflow
    assert "ARTIFACT_ID: ${{ steps.evidence.outputs.artifact-id }}" in workflow
    assert "ARTIFACT_DIGEST: ${{ steps.evidence.outputs.artifact-digest }}" in workflow
    assert "pull_number: prNumber" in workflow
    assert "body: `${prefix}${prefix ? '\\n\\n' : ''}${body}`" in workflow
    assert "decode_b2048" in workflow
    assert "B=2048 packed validator evidence is incomplete" in workflow
    assert "invalid StateTune RESULT identity" in workflow
    assert (
        "tests/infer/wkv7/"
        "test_infer_recurrent_fp16_fp32io16_forward_varlen.py"
        in quick_workflow
    )


def test_packed_hot_path_preserves_scheduler_owned_device_metadata() -> None:
    validation = _function_source(
        "flash_rwkv/validation.py",
        "validate_rwkv7_inputs",
    )
    metadata = _function_source("flash_rwkv/ops.py", "_cuda_metadata")
    benchmark_revision = _function_source(
        "benchmarks/infer/wkv7/"
        "benchmark_infer_recurrent_fp16_fp32io16_forward_varlen.py",
        "_revision_metadata",
    )
    benchmark_case = _function_source(
        "benchmarks/infer/wkv7/"
        "benchmark_infer_recurrent_fp16_fp32io16_forward_varlen.py",
        "_run_case",
    )
    workflow = (ROOT / ".github/workflows/pro6000-gpu.yml").read_text()
    gitignore = (ROOT / ".gitignore").read_text().splitlines()

    assert ".cpu(" not in validation
    assert ".tolist(" not in validation
    assert "query_start_loc = cu_seqlens" in metadata
    assert "else state_indices" in metadata
    native_validation = (CSRC / "validation/recurrent_metadata.cu").read_text()
    assert ".cpu(" not in native_validation
    assert ".item(" not in native_validation
    assert "validate_recurrent_metadata_kernel" in native_validation
    assert "kDuplicateStateSlot" in native_validation
    assert "atomicCAS(&slot_claims[state_slot], 0, 1)" in native_validation
    assert "for (int other" not in native_validation
    assert '"metadata_validation_complexity": "O(sequence_count + state_pool_size)"' in (
        benchmark_case
    )
    assert '"metadata_validation_strategy": "device_slot_claim_bitmap"' in benchmark_case
    assert '"metadata_host_round_trip": False' in benchmark_case
    assert '"kernel_launches_per_operator": 2' in benchmark_case
    assert '"decode_b2048": (1,) * 2048' in (
        ROOT
        / "benchmarks/infer/wkv7/"
        "benchmark_infer_recurrent_fp16_fp32io16_forward_varlen.py"
    ).read_text()
    assert "command -v compute-sanitizer" in workflow
    assert "packed-recurrent-benchmark.json" in workflow
    assert "Verify clean tracked source checkout" in workflow
    assert 'git("status", "--short", "--untracked-files=no")' in workflow
    assert 'prebuild["tracked_source_dirty"]' in workflow
    assert 'packed["source"]["tracked_source_dirty"] is not False' in workflow
    assert 'non_ignored_untracked_paths = [' in workflow
    assert 'ignored_generated_paths = [' in workflow
    assert 'unexpected_ignored_paths = [' in workflow
    assert 'or unexpected_ignored_paths' in workflow
    assert 'Path("artifacts/postbuild-provenance.json")' in workflow
    assert '"--untracked-files=no"' in benchmark_revision
    assert "artifacts/**" in gitignore
    assert 'TORCH_CUDA_ARCH_LIST: "12.0"' in workflow
    assert "TORCH_CUDA_ARCH_LIST=8.0" in workflow
    assert 'for target in 9.0 12.0' in workflow
    assert "architecture-matrix.json" in workflow
    assert '"wheel_minimum_architecture": "sm90"' in workflow
    assert '"runtime_validated_architectures": ["sm120"]' in workflow
    assert "torch.cuda.get_device_capability(0)" in workflow
    assert "benchmark_rl_infctx_chunk_fp32io16_forward.py" in workflow
    assert "rl-infctx-benchmark.json" in workflow


def test_channel_mix_sources_are_built_from_the_pretrain_family() -> None:
    sources = _extension_sources()
    expected = {
        "csrc/pretrain/wkv7/pretrain_common_cmix_bf16_forward_backward_registration.cpp",
        "csrc/pretrain/wkv7/pretrain_sm90_cmix_bf16_forward_backward.cu",
    }
    assert expected <= sources
    for source in expected:
        contents = (ROOT / source).read_text()
        assert "SPDX-License-Identifier: Apache-2.0" in contents
        assert "952102498e9ed367ea0a59ee64106916d474d30f" in contents


def test_l2wrap_sources_are_built_from_the_pretrain_family() -> None:
    sources = _extension_sources()
    expected = {
        "csrc/pretrain/wkv7/pretrain_common_l2wrap_ce_bf16_forward_backward_registration.cpp",
        "csrc/pretrain/wkv7/pretrain_common_l2wrap_ce_bf16_forward_backward.cu",
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
            "csrc/validation/recurrent_metadata.cu",
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
        / "statetune/wkv7/statetune_common_recurrent_fp32io16_registration.cpp"
    )
    contents = source.read_text()
    assert source.relative_to(ROOT).as_posix() in _extension_sources()
    assert (
        "csrc/pretrain/wkv7/pretrain_common_recurrent_fp32io16_forward.cu"
        in contents
    )
    assert (
        "csrc/pretrain/wkv7/pretrain_common_recurrent_fp32io16_backward.cu"
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
    binding_sources = {
        "csrc/infer/wkv7/infer_common_recurrent_varlen_bindings.cpp",
        "csrc/infer/wkv7/infer_common_chunk_bf16_bindings.cpp",
        "csrc/pretrain/wkv7/pretrain_common_recurrent_fp32io16_bindings.cpp",
        "csrc/rl_infctx/wkv7/rl_infctx_common_chunk_fp32io16_bindings.cpp",
    }
    assert binding_sources <= _extension_sources()
    blocks = []
    for relative_path in sorted(binding_sources):
        source = (ROOT / relative_path).read_text()
        blocks.extend(re.findall(r"module\.def\((.*?)\);", source, re.DOTALL))
    registered = {}
    for block in blocks:
        name = re.search(r'^\s*"([^"]+)"', block)
        assert name is not None
        registered[name.group(1)] = tuple(re.findall(r'py::arg\("([^"]+)"\)', block))
    assert set(registered) == EXPECTED_NATIVE_OPS
    assert registered == EXPECTED_ARGUMENTS
