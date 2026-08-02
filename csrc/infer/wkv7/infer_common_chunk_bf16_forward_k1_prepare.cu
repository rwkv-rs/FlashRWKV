// SPDX-License-Identifier: MIT
// K1/K2 launch separation follows MoonshotAI/FlashKDA at commit
// 1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b. This file implements canonical
// RWKV-7 chunk algebra, not the KDA attention operator.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadSize = 64;

template <typename io_t>
__device__ __forceinline__ float to_float(io_t value) {
  return static_cast<float>(value);
}

struct K1Shared {
  float transform[kHeadSize][kHeadSize];
  float bias[kHeadSize][kHeadSize];
  float r[kHeadSize];
  float decay[kHeadSize];
  float k[kHeadSize];
  float v[kHeadSize];
  float a[kHeadSize];
  float b[kHeadSize];
};

template <typename io_t>
__global__ __launch_bounds__(kHeadSize, 1)
void infer_chunk_bf16_forward_k1_prepare_kernel(
    int num_heads,
    const int* __restrict__ chunk_token_starts,
    const int* __restrict__ chunk_token_ends,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ log_decay_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    float* __restrict__ chunk_transform_ptr,
    float* __restrict__ chunk_bias_ptr,
    float* __restrict__ token_transform_ptr,
    float* __restrict__ token_bias_ptr,
    float scale) {
  const int linear_block = static_cast<int>(blockIdx.x);
  const int chunk_index = linear_block / num_heads;
  const int head_index = linear_block % num_heads;
  const int column = static_cast<int>(threadIdx.x);
  __shared__ K1Shared shared;

#pragma unroll
  for (int row = 0; row < kHeadSize; ++row) {
    shared.transform[row][column] = row == column ? 1.0f : 0.0f;
    shared.bias[row][column] = 0.0f;
  }
  __syncthreads();

  const int token_start = chunk_token_starts[chunk_index];
  const int token_end = chunk_token_ends[chunk_index];
  for (int token_index = token_start;
       token_index < token_end;
       ++token_index) {
    const int64_t input_index =
        (static_cast<int64_t>(token_index) * num_heads + head_index) *
            kHeadSize +
        column;
    shared.r[column] = to_float(r_ptr[input_index]);
    shared.decay[column] =
        expf(to_float(log_decay_ptr[input_index]));
    shared.k[column] = to_float(k_ptr[input_index]);
    shared.v[column] = to_float(v_ptr[input_index]);
    shared.a[column] = to_float(a_ptr[input_index]);
    shared.b[column] = to_float(b_ptr[input_index]);
    __syncthreads();

    float transform_dot_a = 0.0f;
    float bias_dot_a = 0.0f;
#pragma unroll
    for (int row = 0; row < kHeadSize; ++row) {
      transform_dot_a = fmaf(
          shared.a[row],
          shared.transform[row][column],
          transform_dot_a);
      bias_dot_a = fmaf(
          shared.a[row],
          shared.bias[row][column],
          bias_dot_a);
    }
    const float value = shared.v[column];
#pragma unroll
    for (int row = 0; row < kHeadSize; ++row) {
      shared.transform[row][column] = fmaf(
          shared.b[row],
          transform_dot_a,
          shared.decay[row] * shared.transform[row][column]);
      shared.bias[row][column] = fmaf(
          shared.k[row],
          value,
          fmaf(
              shared.b[row],
              bias_dot_a,
              shared.decay[row] * shared.bias[row][column]));
    }
    __syncthreads();

    float output_transform = 0.0f;
    float output_bias = 0.0f;
#pragma unroll
    for (int row = 0; row < kHeadSize; ++row) {
      output_transform = fmaf(
          shared.r[row],
          shared.transform[row][column],
          output_transform);
      output_bias = fmaf(
          shared.r[row],
          shared.bias[row][column],
          output_bias);
    }
    token_transform_ptr[input_index] = scale * output_transform;
    token_bias_ptr[input_index] = scale * output_bias;
    __syncthreads();
  }

  const int64_t workspace_base =
      (static_cast<int64_t>(chunk_index) * num_heads + head_index) *
      kHeadSize * kHeadSize;
#pragma unroll
  for (int row = 0; row < kHeadSize; ++row) {
    const int64_t workspace_index =
        workspace_base + row * kHeadSize + column;
    chunk_transform_ptr[workspace_index] =
        shared.transform[row][column];
    chunk_bias_ptr[workspace_index] = shared.bias[row][column];
  }
}

}  // namespace

void infer_chunk_bf16_forward_k1_prepare_cuda(
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor chunk_transform,
    torch::Tensor chunk_bias,
    torch::Tensor token_transform,
    torch::Tensor token_bias,
    double scale) {
  const c10::cuda::CUDAGuard device_guard(r.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int num_chunks = static_cast<int>(chunk_token_starts.numel());
  const int num_heads = static_cast<int>(r.size(1));
  using io_t = at::BFloat16;

  infer_chunk_bf16_forward_k1_prepare_kernel<io_t>
      <<<num_chunks * num_heads, kHeadSize, 0, stream>>>(
          num_heads,
          chunk_token_starts.data_ptr<int>(),
          chunk_token_ends.data_ptr<int>(),
          r.data_ptr<io_t>(),
          log_decay.data_ptr<io_t>(),
          k.data_ptr<io_t>(),
          v.data_ptr<io_t>(),
          a.data_ptr<io_t>(),
          b.data_ptr<io_t>(),
          chunk_transform.data_ptr<float>(),
          chunk_bias.data_ptr<float>(),
          token_transform.data_ptr<float>(),
          token_bias.data_ptr<float>(),
          static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
