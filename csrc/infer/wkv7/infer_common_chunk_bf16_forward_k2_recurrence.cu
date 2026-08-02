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

template <typename io_t>
__device__ __forceinline__ io_t from_float(float value) {
  return static_cast<io_t>(value);
}

struct K2Shared {
  float token_transform[kHeadSize];
  float chunk_transform[kHeadSize][kHeadSize];
  float next_state[kHeadSize][kHeadSize];
};

template <typename io_t>
__global__ __launch_bounds__(kHeadSize, 1)
void infer_chunk_bf16_forward_k2_recurrence_kernel(
    int num_heads,
    const int* __restrict__ sequence_chunk_offsets,
    const int* __restrict__ chunk_token_starts,
    const int* __restrict__ chunk_token_ends,
    io_t* __restrict__ state_ptr,
    io_t* __restrict__ output_ptr,
    const float* __restrict__ chunk_transform_ptr,
    const float* __restrict__ chunk_bias_ptr,
    const float* __restrict__ token_transform_ptr,
    const float* __restrict__ token_bias_ptr) {
  const int linear_block = static_cast<int>(blockIdx.x);
  const int sequence_index = linear_block / num_heads;
  const int head_index = linear_block % num_heads;
  const int value_index = static_cast<int>(threadIdx.x);
  __shared__ K2Shared shared;

  const int64_t state_base =
      (static_cast<int64_t>(sequence_index) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  float state[kHeadSize];
#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state[key_index] = to_float(
        state_ptr[state_base + key_index * kHeadSize + value_index]);
  }

  const int chunk_start = sequence_chunk_offsets[sequence_index];
  const int chunk_end = sequence_chunk_offsets[sequence_index + 1];
  for (int chunk_index = chunk_start;
       chunk_index < chunk_end;
       ++chunk_index) {
    const int token_start = chunk_token_starts[chunk_index];
    const int token_end = chunk_token_ends[chunk_index];
    for (int token_index = token_start;
         token_index < token_end;
         ++token_index) {
      const int64_t token_base =
          (static_cast<int64_t>(token_index) * num_heads + head_index) *
          kHeadSize;
      shared.token_transform[value_index] =
          token_transform_ptr[token_base + value_index];
      __syncthreads();

      float output = token_bias_ptr[token_base + value_index];
#pragma unroll
      for (int key_index = 0; key_index < kHeadSize; ++key_index) {
        output = fmaf(
            shared.token_transform[key_index],
            state[key_index],
            output);
      }
      output_ptr[token_base + value_index] = from_float<io_t>(output);
      __syncthreads();
    }

    const int64_t workspace_base =
        (static_cast<int64_t>(chunk_index) * num_heads + head_index) *
        kHeadSize * kHeadSize;
#pragma unroll
    for (int row = 0; row < kHeadSize; ++row) {
      shared.chunk_transform[row][value_index] =
          chunk_transform_ptr[
              workspace_base + row * kHeadSize + value_index];
    }
    __syncthreads();

#pragma unroll
    for (int row = 0; row < kHeadSize; ++row) {
      float updated =
          chunk_bias_ptr[
              workspace_base + row * kHeadSize + value_index];
#pragma unroll
      for (int input_key = 0; input_key < kHeadSize; ++input_key) {
        updated = fmaf(
            shared.chunk_transform[row][input_key],
            state[input_key],
            updated);
      }
      shared.next_state[row][value_index] = updated;
    }
    __syncthreads();

#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      state[key_index] = shared.next_state[key_index][value_index];
    }
    __syncthreads();
  }

#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state_ptr[state_base + key_index * kHeadSize + value_index] =
        from_float<io_t>(state[key_index]);
  }
}

}  // namespace

void infer_chunk_bf16_forward_k2_recurrence_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state,
    torch::Tensor output,
    torch::Tensor chunk_transform,
    torch::Tensor chunk_bias,
    torch::Tensor token_transform,
    torch::Tensor token_bias) {
  const c10::cuda::CUDAGuard device_guard(state.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int num_sequences =
      static_cast<int>(sequence_chunk_offsets.numel() - 1);
  const int num_heads = static_cast<int>(state.size(1));
  using io_t = at::BFloat16;

  infer_chunk_bf16_forward_k2_recurrence_kernel<io_t>
      <<<num_sequences * num_heads, kHeadSize, 0, stream>>>(
          num_heads,
          sequence_chunk_offsets.data_ptr<int>(),
          chunk_token_starts.data_ptr<int>(),
          chunk_token_ends.data_ptr<int>(),
          state.data_ptr<io_t>(),
          output.data_ptr<io_t>(),
          chunk_transform.data_ptr<float>(),
          chunk_bias.data_ptr<float>(),
          token_transform.data_ptr<float>(),
          token_bias.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
