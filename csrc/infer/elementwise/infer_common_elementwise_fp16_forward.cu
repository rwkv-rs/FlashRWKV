// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.
// Modified by contributors to the FlashRWKV project.
#include <assert.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#include <vector>

using dtype = at::Half;

namespace {

constexpr int HEAD_SIZE = 64;
constexpr int WARPS_PER_BLOCK = 4;
constexpr float KK_NORMALIZE_EPS = 1.0e-12f;
constexpr float TMIX_LN_X_EPS = 64.0e-5f;
constexpr int FFN_SPMV_THREADS = 128;
constexpr int FFN_TILE = 128;

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

__device__ inline __half2 load_h2(const dtype* ptr) {
  return *reinterpret_cast<const __half2*>(ptr);
}

__device__ inline float load_h1(const dtype* ptr) {
  return __half2float(*reinterpret_cast<const __half*>(ptr));
}

__device__ inline void store_h1(dtype* ptr, float value) {
  *reinterpret_cast<__half*>(ptr) = __float2half_rn(value);
}

__device__ inline void store_h2(dtype* ptr, float x0, float x1) {
  *reinterpret_cast<__half2*>(ptr) = __floats2half2_rn(x0, x1);
}

__device__ inline float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ inline float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__global__ void relu_square_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 v = __half22float2(load_h2(x + idx));
  const float x0 = fmaxf(v.x, 0.0f);
  const float x1 = fmaxf(v.y, 0.0f);
  store_h2(out + idx, x0 * x0, x1 * x1);
}

__global__ void act_tanh_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 v = __half22float2(load_h2(x + idx));
  store_h2(out + idx, tanhf(v.x), tanhf(v.y));
}

__global__ void act_sigmoid_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 v = __half22float2(load_h2(x + idx));
  store_h2(out + idx, sigmoid_fast(v.x), sigmoid_fast(v.y));
}

__global__ void add_vec_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ vec,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int c = static_cast<int>((pair_idx % (C >> 1)) << 1);
  const int64_t idx = pair_idx * 2;
  const float2 xv = __half22float2(load_h2(x + idx));
  const float2 vv = __half22float2(load_h2(vec + c));
  store_h2(out + idx, xv.x + vv.x, xv.y + vv.y);
}

__global__ void add_vec_2d_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ vec,
    dtype* __restrict__ out) {
  const int c_pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int c_pairs = C >> 1;
  if (c_pair >= c_pairs) {
    return;
  }
  // The host guard limits rows to grid.y and requires complete C-wide rows.
  // Keeping row in blockIdx.y removes the 64-bit modulo from every half2 owner.
  const int64_t row = blockIdx.y;
  const int c = c_pair << 1;
  const int64_t idx = row * C + c;
  const float2 xv = __half22float2(load_h2(x + idx));
  const float2 vv = __half22float2(load_h2(vec + c));
  store_h2(out + idx, xv.x + vv.x, xv.y + vv.y);
}

}  // namespace

at::Tensor relu_square_cuda(at::Tensor x) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  relu_square_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      x.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor act_tanh_cuda(at::Tensor x) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  act_tanh_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      x.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor act_sigmoid_cuda(at::Tensor x) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  act_sigmoid_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      x.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor add_vec_cuda(int C, at::Tensor x, at::Tensor vec) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  add_vec_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      C,
      x.data_ptr<dtype>(),
      vec.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor add_vec_2d_cuda(int C, at::Tensor x, at::Tensor vec) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int rows = static_cast<int>(x.numel() / C);
  auto stream = at::cuda::getCurrentCUDAStream();
  add_vec_2d_kernel<<<dim3(static_cast<unsigned int>(ceil_div(C / 2, threads)),
                            static_cast<unsigned int>(rows), 1),
                      threads, 0, stream>>>(
      C,
      x.data_ptr<dtype>(),
      vec.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void add_vec_cfg_out_cuda(
    int C,
    at::Tensor x,
    at::Tensor vec,
    at::Tensor out,
    bool grid2d) {
  constexpr int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (grid2d) {
    const int rows = static_cast<int>(x.numel() / C);
    add_vec_2d_kernel<<<dim3(static_cast<unsigned int>(ceil_div(C / 2, threads)),
                              static_cast<unsigned int>(rows), 1),
                        threads, 0, stream>>>(
        C, x.data_ptr<dtype>(), vec.data_ptr<dtype>(), out.data_ptr<dtype>());
  } else {
    const int64_t total_pairs = x.numel() / 2;
    add_vec_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        C, x.data_ptr<dtype>(), vec.data_ptr<dtype>(), out.data_ptr<dtype>(), total_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
