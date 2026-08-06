// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Upstream repository: https://github.com/BlinkDL/Albatross
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e
// Original path: faster3a_2607/cuda/rwkv7_v3a_ops.cu
// Mechanical migration boundary: exact upstream BF16 embedding initial-layer-norm
// kernel and launch are retained. The only local adaptation is the packed-row
// caller binding; this operator is tokenwise and takes no sequence metadata.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <climits>
#include <vector>

using dtype = at::Half;
namespace wmma = nvcuda::wmma;


namespace {

constexpr int LN_THREADS = 256;
constexpr int LN_SMALL_THREADS = 1024;
constexpr int LN_SMALL512_THREADS = 512;
constexpr int LN_SMALL_C = 4096;

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

inline void check_cublas(cublasStatus_t status, const char* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with cublas status ", static_cast<int>(status));
}

inline void check_cublaslt(cublasStatus_t status, const char* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with cublasLt status ", static_cast<int>(status));
}

template <int Act>
__device__ __forceinline__ float apply_act(float x) {
  if constexpr (Act == 1) {
    return tanhf(x);
  } else {
    return 1.0f / (1.0f + expf(-x));
  }
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

__device__ __forceinline__ float bf16_bits_to_float_dev(uint16_t bits) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = static_cast<uint32_t>(bits) << 16;
  return v.f;
}

template <int Threads>
__device__ __forceinline__ float block_sum_t(float x) {
  __shared__ float partial[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  x = warp_sum(x);
  if (lane == 0) {
    partial[warp] = x;
  }
  __syncthreads();
  x = (threadIdx.x < (Threads / 32)) ? partial[lane] : 0.0f;
  if (warp == 0) {
    x = warp_sum(x);
  }
  if (threadIdx.x == 0) {
    partial[0] = x;
  }
  __syncthreads();
  return partial[0];
}

__global__ void emb_ln0_bf16_to_f16_kernel(
    int V,
    int C,
    const uint16_t* __restrict__ emb,
    const uint16_t* __restrict__ weight,
    const uint16_t* __restrict__ bias,
    dtype* __restrict__ out,
    float eps) {
  // Precision path: bf16 inputs -> fp32 two-pass stats/affine -> fp16 output.
  const int tok = blockIdx.x;
  const int tid = threadIdx.x;
  if (tok >= V) {
    return;
  }
  const uint16_t* er = emb + static_cast<int64_t>(tok) * C;
  float sum = 0.0f;
  for (int c = tid; c < C; c += blockDim.x) {
    sum += bf16_bits_to_float_dev(er[c]);
  }
  const float mean = block_sum_t<256>(sum) / static_cast<float>(C);
  float var = 0.0f;
  for (int c = tid; c < C; c += blockDim.x) {
    const float d = bf16_bits_to_float_dev(er[c]) - mean;
    var += d * d;
  }
  const float rstd = rsqrtf(block_sum_t<256>(var) / static_cast<float>(C) + eps);
  dtype* yr = out + static_cast<int64_t>(tok) * C;
  for (int c = tid; c < C; c += blockDim.x) {
    const float x = bf16_bits_to_float_dev(er[c]);
    const float w = bf16_bits_to_float_dev(weight[c]);
    const float b = bf16_bits_to_float_dev(bias[c]);
    yr[c] = static_cast<dtype>((x - mean) * rstd * w + b);
  }
}

} // namespace

at::Tensor emb_ln0_bf16_to_f16_cuda(at::Tensor emb, at::Tensor weight, at::Tensor bias, double eps) {
  auto out = at::empty(emb.sizes(), emb.options().dtype(at::kHalf));
  const int64_t v64 = emb.size(0);
  const int64_t c64 = emb.size(1);
  TORCH_CHECK(v64 <= INT_MAX && c64 <= INT_MAX, "emb shape too large");
  const int V = static_cast<int>(v64);
  const int C = static_cast<int>(c64);
  auto stream = at::cuda::getCurrentCUDAStream();
  emb_ln0_bf16_to_f16_kernel<<<V, 256, 0, stream>>>(
      V, C,
      reinterpret_cast<const uint16_t*>(emb.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint16_t*>(weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint16_t*>(bias.data_ptr<at::BFloat16>()),
      out.data_ptr<dtype>(),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


at::Tensor embedding_ln0_forward_varlen_cuda(
    at::Tensor embedding,
    at::Tensor weight,
    at::Tensor bias,
    double eps) {
  return emb_ln0_bf16_to_f16_cuda(embedding, weight, bias, eps);
}
