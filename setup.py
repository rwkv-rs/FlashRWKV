# SPDX-License-Identifier: MIT

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    ext_modules=[
        CUDAExtension(
            name="flash_rwkv._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/chunk/materialized_fp32.cu",
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
