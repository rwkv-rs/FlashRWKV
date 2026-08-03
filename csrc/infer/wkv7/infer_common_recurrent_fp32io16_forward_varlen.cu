// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vllm-rwkv rwkv7_wkv_fp32_v2 at commit
// 6d683f9e49a2997e405c47edc147872c8609513b. The state load/store and update
// are transposed to FlashRWKV's canonical [K,V] layout. Product entry points
// consume raw decay logits and fuse the retention transform; a private
// canonical-log-decay specialization remains for independent A/B checks.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "../../common/wkv7/recurrent_decay.cuh"

namespace {

template <typename io_t>
__device__ __forceinline__ float to_float(io_t value) {
  return static_cast<float>(value);
}

template <typename io_t>
__device__ __forceinline__ io_t from_float(float value) {
  return static_cast<io_t>(value);
}

using flash_rwkv::wkv7::RecurrentDecayInput;
using flash_rwkv::wkv7::recurrent_retention;

template <int HeadSize, typename io_t, RecurrentDecayInput DecayInput>
__global__ __launch_bounds__(HeadSize, HeadSize == 64 ? 2 : 1)
void recurrent_fp32_kernel(
    int num_heads,
    int64_t output_elements,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ decay_ptr,
    const io_t* __restrict__ decay_bias_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ output_ptr,
    float scale) {
  const int head_index = static_cast<int>(blockIdx.x);
  const int sequence_index = static_cast<int>(blockIdx.y);
  const int value_index = static_cast<int>(threadIdx.x);

  if (metadata_status[0] != 0) {
    const int64_t block_index =
        static_cast<int64_t>(sequence_index) * num_heads + head_index;
    const int64_t block_count =
        static_cast<int64_t>(gridDim.x) * gridDim.y;
    for (int64_t output_index = block_index * blockDim.x + value_index;
         output_index < output_elements;
         output_index += block_count * blockDim.x) {
      output_ptr[output_index] =
          from_float<io_t>(__int_as_float(0x7fffffff));
    }
    return;
  }

  __shared__ int token_start;
  __shared__ int token_end;
  __shared__ int state_slot;
  if (value_index == 0) {
    token_start = query_start_loc[sequence_index];
    token_end = query_start_loc[sequence_index + 1];
    state_slot = state_indices[sequence_index];
  }
  __syncthreads();

  float* state_base =
      state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          HeadSize * HeadSize;
  float state[HeadSize];
#pragma unroll
  for (int key_index = 0; key_index < HeadSize; ++key_index) {
    state[key_index] = state_base[key_index * HeadSize + value_index];
  }

  __shared__ float r[HeadSize];
  __shared__ float decay[HeadSize];
  __shared__ float k[HeadSize];
  __shared__ float a[HeadSize];
  __shared__ float b[HeadSize];

  for (int token_index = token_start; token_index < token_end; ++token_index) {
    const int64_t input_index =
        (static_cast<int64_t>(token_index) * num_heads + head_index) *
            HeadSize +
        value_index;
    r[value_index] = to_float(r_ptr[input_index]);
    float decay_input = to_float(decay_ptr[input_index]);
    if constexpr (DecayInput == RecurrentDecayInput::kDecayLogits) {
      if (decay_bias_ptr != nullptr) {
        decay_input += to_float(
            decay_bias_ptr[head_index * HeadSize + value_index]);
      }
    }
    decay[value_index] = recurrent_retention<DecayInput>(decay_input);
    k[value_index] = to_float(k_ptr[input_index]);
    a[value_index] = to_float(a_ptr[input_index]);
    b[value_index] = to_float(b_ptr[input_index]);
    __syncthreads();

    float a_state = 0.0f;
#pragma unroll
    for (int key_index = 0; key_index < HeadSize; ++key_index) {
      a_state += a[key_index] * state[key_index];
    }

    const float value = to_float(v_ptr[input_index]);
    float output = 0.0f;
#pragma unroll
    for (int key_index = 0; key_index < HeadSize; ++key_index) {
      const float updated =
          decay[key_index] * state[key_index] +
          b[key_index] * a_state +
          k[key_index] * value;
      state[key_index] = updated;
      output += r[key_index] * updated;
    }
    output_ptr[input_index] = from_float<io_t>(scale * output);
    __syncthreads();
  }

#pragma unroll
  for (int key_index = 0; key_index < HeadSize; ++key_index) {
    state_base[key_index * HeadSize + value_index] = state[key_index];
  }
}

template <int HeadSize, typename io_t, RecurrentDecayInput DecayInput>
void launch_recurrent_fp32(
    int num_sequences,
    int num_heads,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& state_indices,
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    const torch::Tensor& metadata_status,
    float scale,
    cudaStream_t stream) {
  recurrent_fp32_kernel<HeadSize, io_t, DecayInput>
      <<<dim3(num_heads, num_sequences), dim3(HeadSize), 0, stream>>>(
          num_heads,
          output.numel(),
          query_start_loc.data_ptr<int>(),
          state_indices.data_ptr<int>(),
          metadata_status.data_ptr<int>(),
          state.data_ptr<float>(),
          r.data_ptr<io_t>(),
          decay.data_ptr<io_t>(),
          decay_bias.defined() ? decay_bias.data_ptr<io_t>() : nullptr,
          k.data_ptr<io_t>(),
          v.data_ptr<io_t>(),
          a.data_ptr<io_t>(),
          b.data_ptr<io_t>(),
          output.data_ptr<io_t>(),
          scale);
}

}  // namespace

template <flash_rwkv::wkv7::RecurrentDecayInput DecayInput>
void recurrent_fp32_cuda_impl(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay,
    torch::Tensor decay_bias,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor metadata_status,
    double scale) {
  const c10::cuda::CUDAGuard device_guard(state.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int num_sequences = static_cast<int>(state_indices.numel());
  const int num_heads = static_cast<int>(state.size(1));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      r.scalar_type(),
      "flash_rwkv_recurrent_fp32",
      [&] {
        switch (state.size(2)) {
          case 64:
            launch_recurrent_fp32<64, scalar_t, DecayInput>(
                num_sequences, num_heads, query_start_loc, state_indices,
                state, r, decay, decay_bias, k, v, a, b, output,
                metadata_status,
                static_cast<float>(scale), stream);
            break;
          case 128:
            launch_recurrent_fp32<128, scalar_t, DecayInput>(
                num_sequences, num_heads, query_start_loc, state_indices,
                state, r, decay, decay_bias, k, v, a, b, output,
                metadata_status,
                static_cast<float>(scale), stream);
            break;
          case 256:
            launch_recurrent_fp32<256, scalar_t, DecayInput>(
                num_sequences, num_heads, query_start_loc, state_indices,
                state, r, decay, decay_bias, k, v, a, b, output,
                metadata_status,
                static_cast<float>(scale), stream);
            break;
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void recurrent_fp32_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor metadata_status,
    double scale) {
  recurrent_fp32_cuda_impl<
      flash_rwkv::wkv7::RecurrentDecayInput::kLogDecay>(
      query_start_loc, state_indices, state, r, log_decay, torch::Tensor(),
      k, v, a, b, output, metadata_status, scale);
}

void recurrent_fp32_from_decay_logits_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor decay_bias,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor metadata_status,
    double scale) {
  recurrent_fp32_cuda_impl<
      flash_rwkv::wkv7::RecurrentDecayInput::kDecayLogits>(
      query_start_loc, state_indices, state, r, decay_logits, decay_bias, k,
      v, a, b, output, metadata_status, scale);
}
