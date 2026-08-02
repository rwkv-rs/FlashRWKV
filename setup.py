# SPDX-License-Identifier: MIT

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            name="flash_rwkv._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/registration.cpp",
                "csrc/validation.cpp",
                "csrc/rl_infctx/wkv7/rl_infctx_smxx_chunk_fp32io16_forward_materialized.cu",
                "csrc/rl_infctx/wkv7/rl_infctx_smxx_chunk_fp32io16_forward_recompute.cu",
                "csrc/rl_infctx/wkv7/rl_infctx_smxx_chunk_fp32io16_backward_replay.cu",
                "csrc/statetune/wkv7/statetune_smxx_recurrent_fp32io16_registration.cpp",
                "csrc/infer/wkv7/infer_smxx_chunk_bf16_forward_k1_prepare.cu",
                "csrc/infer/wkv7/infer_smxx_chunk_bf16_forward_k2_recurrence.cu",
                "csrc/infer/wkv7/infer_smxx_fused_fp16_forward_registration.cpp",
                "csrc/infer/wkv7/infer_smxx_fused_fp16_forward.cu",
                "csrc/pretrain/wkv7/pretrain_smxx_recurrent_fp32io16_backward.cu",
                "csrc/pretrain/wkv7/pretrain_smxx_recurrent_fp32io16_forward.cu",
                "csrc/pretrain/wkv7/pretrain_smxx_head_l2wrap_ce_bf16_forward.cpp",
                "csrc/pretrain/wkv7/pretrain_smxx_head_l2wrap_ce_bf16_forward.cu",
                "csrc/pretrain/wkv7/pretrain_smxx_cmix_bf16_forward_backward.cpp",
                "csrc/pretrain/wkv7/pretrain_smxx_cmix_bf16_forward_backward.cu",
                "csrc/pretrain/wkv7/pretrain_smxx_l2wrap_ce_bf16_forward_backward.cpp",
                "csrc/pretrain/wkv7/pretrain_smxx_l2wrap_ce_bf16_forward_backward.cu",
                "csrc/pretrain/wkv7/pretrain_smxx_tmix_a_gate_bf16_forward_backward.cpp",
                "csrc/pretrain/wkv7/pretrain_smxx_tmix_a_gate_bf16_forward_backward.cu",
                "csrc/pretrain/wkv7/pretrain_smxx_tmix_kk_pre_bf16_forward_backward.cpp",
                "csrc/pretrain/wkv7/pretrain_smxx_tmix_kk_pre_bf16_forward_backward.cu",
                "csrc/pretrain/wkv7/pretrain_smxx_tmix_lnx_rkvres_xg_bf16_forward_backward.cpp",
                "csrc/pretrain/wkv7/pretrain_smxx_tmix_lnx_rkvres_xg_bf16_forward_backward.cu",
                "csrc/pretrain/wkv7/pretrain_smxx_tmix_mix6_bf16_forward_backward.cpp",
                "csrc/pretrain/wkv7/pretrain_smxx_tmix_mix6_bf16_forward_backward.cu",
                "csrc/pretrain/wkv7/pretrain_smxx_tmix_vres_gate_bf16_forward_backward.cpp",
                "csrc/pretrain/wkv7/pretrain_smxx_tmix_vres_gate_bf16_forward_backward.cu",
                "csrc/infer/wkv7/infer_smxx_recurrent_fp16_forward_varlen.cu",
                "csrc/infer/wkv7/infer_smxx_recurrent_fp32io16_forward_varlen.cu",
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
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
