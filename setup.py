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
                "csrc/chunk/materialized_fp32.cu",
                "csrc/chunk/recompute_fp32.cu",
                "csrc/chunk/replay_fp32.cu",
                "csrc/kda/infer_chunk_bf16_forward_k1_prepare.cu",
                "csrc/kda/infer_chunk_bf16_forward_k2_recurrence.cu",
                "csrc/pretrain/recurrent_fp32io16_backward.cu",
                "csrc/pretrain/recurrent_fp32io16_forward.cu",
                "csrc/pretrain/head_l2wrap_ce/rwkv7_head_l2wrap_ce_bf16_v4_registration.cpp",
                "csrc/pretrain/head_l2wrap_ce/rwkv7_head_l2wrap_ce_bf16_v4.cu",
                "csrc/pretrain/cmix/rwkv7_cmix_bf16_v5_registration.cpp",
                "csrc/pretrain/cmix/rwkv7_cmix_bf16_v5.cu",
                "csrc/pretrain/l2wrap_ce/rwkv7_l2wrap_ce_bf16_v2_registration.cpp",
                "csrc/pretrain/l2wrap_ce/rwkv7_l2wrap_ce_bf16_v2.cu",
                "csrc/pretrain/tmix_a_gate/rwkv7_tmix_a_gate_bf16_registration.cpp",
                "csrc/pretrain/tmix_a_gate/rwkv7_tmix_a_gate_bf16.cu",
                "csrc/pretrain/tmix_vres_gate/rwkv7_tmix_vres_gate_bf16_v3_registration.cpp",
                "csrc/pretrain/tmix_vres_gate/rwkv7_tmix_vres_gate_bf16_v3.cu",
                "csrc/recurrent/recurrent_fp16.cu",
                "csrc/recurrent/recurrent_fp32.cu",
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
