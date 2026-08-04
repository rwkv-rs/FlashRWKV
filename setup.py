# SPDX-License-Identifier: MIT

import importlib.util
import os
import sys
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _architecture_contract():
    path = Path(__file__).parent / "flash_rwkv" / "architecture.py"
    spec = importlib.util.spec_from_file_location(
        "flash_rwkv_build_architecture_contract",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load architecture contract from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


architecture = _architecture_contract()
requested_architectures = os.environ.get("TORCH_CUDA_ARCH_LIST")
detected_architecture = (
    tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
)
build_commands = {
    "bdist",
    "bdist_egg",
    "bdist_rpm",
    "bdist_wheel",
    "build",
    "build_clib",
    "build_ext",
    "build_py",
    "build_scripts",
    "develop",
    "editable_wheel",
    "install",
    "install_lib",
}
native_build = bool(build_commands.intersection(sys.argv[1:]))
if native_build:
    architecture.validate_wheel_architectures(
        requested_architectures,
        detected=detected_architecture,
    )

ext_modules = (
    [
        CUDAExtension(
            name="flash_rwkv._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/registration.cpp",
                "csrc/validation.cpp",
                "csrc/validation/recurrent_metadata.cu",
                "csrc/infer/wkv7/infer_common_recurrent_varlen_bindings.cpp",
                "csrc/infer/wkv7/infer_common_chunk_bf16_bindings.cpp",
                "csrc/common/wkv7/recurrent_common_fp32io16.cpp",
                "csrc/pretrain/wkv7/pretrain_common_recurrent_fp32io16_bindings.cpp",
                "csrc/rl_infctx/wkv7/rl_infctx_common_chunk_fp32io16_bindings.cpp",
                "csrc/rl_infctx/wkv7/rl_infctx_common_chunk_fp32io16_forward_materialized.cu",
                "csrc/rl_infctx/wkv7/rl_infctx_common_chunk_fp32io16_forward_recompute.cu",
                "csrc/rl_infctx/wkv7/rl_infctx_common_chunk_fp32io16_backward_replay.cu",
                "csrc/statetune/wkv7/statetune_common_recurrent_fp32io16_bindings.cpp",
                "csrc/infer/wkv7/infer_common_chunk_bf16_forward_k1_prepare.cu",
                "csrc/infer/wkv7/infer_common_chunk_bf16_forward_k2_recurrence.cu",
                "csrc/infer/tmix/registration/infer_common_tmix_fp16_forward.cpp",
                "csrc/infer/tmix/infer_common_tmix_fp16_forward.cu",
                "csrc/infer/cmix/registration/infer_common_cmix_fp16_forward.cpp",
                "csrc/infer/cmix/infer_common_cmix_fp16_forward.cu",
                "csrc/infer/elementwise/infer_common_elementwise_fp16_forward_registration.cpp",
                "csrc/infer/elementwise/infer_common_elementwise_fp16_forward.cu",
                "csrc/common/wkv7/recurrent_common_fp32io16_backward.cu",
                "csrc/common/wkv7/recurrent_common_fp32io16_forward.cu",
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
                "csrc/infer/wkv7/infer_sm80_recurrent_fp16_forward_varlen.cu",
                "csrc/infer/wkv7/infer_common_recurrent_fp32io16_forward_varlen.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-Wno-psabi"],
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-lineinfo",
                ],
            },
        )
    ]
    if native_build
    else []
)

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if native_build else {},
    zip_safe=False,
)
