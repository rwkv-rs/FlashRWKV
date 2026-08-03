// SPDX-License-Identifier: MIT
// Shared output replay for FlashRWKV FP32-state chunk strategies.

#include "rl_infctx_common_chunk_fp32io16_backward_replay.cuh"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include "../../common/wkv7/recurrent_decay.cuh"

namespace {

constexpr int kHeadSize = 64;

using flash_rwkv::wkv7::RecurrentDecayInput;
using flash_rwkv::wkv7::recurrent_retention;

template <typename io_t>
__device__ __forceinline__ float to_float(io_t value) {
  return static_cast<float>(value);
}

template <typename io_t>
__device__ __forceinline__ io_t from_float(float value) {
  return static_cast<io_t>(value);
}

struct OutputShared {
  float r[kHeadSize];
  float decay[kHeadSize];
  float k[kHeadSize];
  float a[kHeadSize];
  float b[kHeadSize];
};

template <typename io_t, RecurrentDecayInput DecayInput>
__global__ __launch_bounds__(kHeadSize, 2)
void emit_outputs_kernel(
    int num_heads,
    const int* __restrict__ chunk_token_starts,
    const int* __restrict__ chunk_token_ends,
    const float* __restrict__ boundary_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ decay_ptr,
    const io_t* __restrict__ decay_bias_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ output_ptr,
    float* __restrict__ state_dot_a_ptr,
    float scale) {
  const int linear_block = static_cast<int>(blockIdx.x);
  const int chunk_index = linear_block / num_heads;
  const int head_index = linear_block % num_heads;
  const int value_index = static_cast<int>(threadIdx.x);
  __shared__ OutputShared shared;

  const int64_t boundary_base =
      (static_cast<int64_t>(chunk_index) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  float state[kHeadSize];
#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state[key_index] =
        boundary_ptr[
            boundary_base + key_index * kHeadSize + value_index];
  }

  const int token_start = chunk_token_starts[chunk_index];
  const int token_end = chunk_token_ends[chunk_index];
  for (int token_index = token_start;
       token_index < token_end;
       ++token_index) {
    const int64_t input_index =
        (static_cast<int64_t>(token_index) * num_heads + head_index) *
            kHeadSize +
        value_index;
    shared.r[value_index] = to_float(r_ptr[input_index]);
    float decay_input = to_float(decay_ptr[input_index]);
    if constexpr (DecayInput == RecurrentDecayInput::kDecayLogits) {
      if (decay_bias_ptr != nullptr) {
        decay_input += to_float(
            decay_bias_ptr[head_index * kHeadSize + value_index]);
      }
    }
    shared.decay[value_index] = recurrent_retention<DecayInput>(decay_input);
    shared.k[value_index] = to_float(k_ptr[input_index]);
    shared.a[value_index] = to_float(a_ptr[input_index]);
    shared.b[value_index] = to_float(b_ptr[input_index]);
    const float value = to_float(v_ptr[input_index]);
    __syncthreads();

    float state_dot_a = 0.0f;
#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      state_dot_a =
          fmaf(shared.a[key_index], state[key_index], state_dot_a);
    }
    if (state_dot_a_ptr != nullptr) {
      state_dot_a_ptr[input_index] = state_dot_a;
    }

    float output = 0.0f;
#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      const float updated = fmaf(
          shared.k[key_index],
          value,
          fmaf(
              shared.b[key_index],
              state_dot_a,
              shared.decay[key_index] * state[key_index]));
      state[key_index] = updated;
      output = fmaf(shared.r[key_index], updated, output);
    }
    output_ptr[input_index] = from_float<io_t>(scale * output);
    __syncthreads();
  }
}

template <typename io_t, RecurrentDecayInput DecayInput>
void launch_replay(
    int num_chunks,
    int num_heads,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& boundary,
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    torch::Tensor* state_dot_a,
    float scale,
    cudaStream_t stream) {
  emit_outputs_kernel<io_t, DecayInput>
      <<<num_chunks * num_heads, kHeadSize, 0, stream>>>(
          num_heads,
          chunk_token_starts.data_ptr<int>(),
          chunk_token_ends.data_ptr<int>(),
          boundary.data_ptr<float>(),
          r.data_ptr<io_t>(),
          decay.data_ptr<io_t>(),
          decay_bias.defined() ? decay_bias.data_ptr<io_t>() : nullptr,
          k.data_ptr<io_t>(),
          v.data_ptr<io_t>(),
          a.data_ptr<io_t>(),
          b.data_ptr<io_t>(),
          output.data_ptr<io_t>(),
          state_dot_a == nullptr
              ? nullptr
              : state_dot_a->data_ptr<float>(),
          scale);
}

}  // namespace

template <flash_rwkv::wkv7::RecurrentDecayInput DecayInput>
void launch_chunk_replay_fp32_impl(
    int num_chunks,
    int num_heads,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& boundary,
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    torch::Tensor* state_dot_a,
    float scale,
    cudaStream_t stream) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      r.scalar_type(),
      "flash_rwkv_chunk_replay_fp32",
      [&] {
        launch_replay<scalar_t, DecayInput>(
            num_chunks,
            num_heads,
            chunk_token_starts,
            chunk_token_ends,
            boundary,
            r,
            decay,
            decay_bias,
            k,
            v,
            a,
            b,
            output,
            state_dot_a,
            scale,
            stream);
      });
}

void launch_chunk_replay_fp32(
    int num_chunks,
    int num_heads,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& boundary,
    const torch::Tensor& r,
    const torch::Tensor& log_decay,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    torch::Tensor* state_dot_a,
    float scale,
    cudaStream_t stream) {
  launch_chunk_replay_fp32_impl<
      flash_rwkv::wkv7::RecurrentDecayInput::kLogDecay>(
      num_chunks, num_heads, chunk_token_starts, chunk_token_ends, boundary,
      r, log_decay, torch::Tensor(), k, v, a, b, output, state_dot_a, scale,
      stream);
}

void launch_chunk_replay_fp32_from_decay_logits(
    int num_chunks,
    int num_heads,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& boundary,
    const torch::Tensor& r,
    const torch::Tensor& decay_logits,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    torch::Tensor* state_dot_a,
    float scale,
    cudaStream_t stream) {
  launch_chunk_replay_fp32_impl<
      flash_rwkv::wkv7::RecurrentDecayInput::kDecayLogits>(
      num_chunks, num_heads, chunk_token_starts, chunk_token_ends, boundary,
      r, decay_logits, decay_bias, k, v, a, b, output, state_dot_a, scale,
      stream);
}
