// SPDX-License-Identifier: MIT
// DPLR-factor recompute chunk forward for the FlashRWKV core contract.

#include "rl_infctx_smxx_chunk_fp32io16_backward_replay.cuh"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
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

struct FactorShared {
  float decay[kHeadSize];
  float k[kHeadSize];
  float a[kHeadSize];
  float b[kHeadSize];
};

template <typename io_t>
__global__ __launch_bounds__(kHeadSize, 2)
void scan_factor_boundaries_kernel(
    int num_heads,
    const int* __restrict__ sequence_chunk_offsets,
    const int* __restrict__ chunk_token_starts,
    const int* __restrict__ chunk_token_ends,
    const int* __restrict__ state_indices,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ log_decay_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    float* __restrict__ boundary_ptr) {
  const int head_index = static_cast<int>(blockIdx.x);
  const int sequence_index = static_cast<int>(blockIdx.y);
  const int value_index = static_cast<int>(threadIdx.x);
  const int state_slot = state_indices[sequence_index];
  __shared__ FactorShared shared;

  const int64_t state_base =
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  float state[kHeadSize];
#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state[key_index] =
        state_ptr[state_base + key_index * kHeadSize + value_index];
  }

  const int chunk_start = sequence_chunk_offsets[sequence_index];
  const int chunk_end = sequence_chunk_offsets[sequence_index + 1];
  for (int chunk_index = chunk_start;
       chunk_index < chunk_end;
       ++chunk_index) {
    const int64_t boundary_base =
        (static_cast<int64_t>(chunk_index) * num_heads + head_index) *
        kHeadSize * kHeadSize;
#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      boundary_ptr[
          boundary_base + key_index * kHeadSize + value_index] =
          state[key_index];
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
      shared.decay[value_index] =
          expf(to_float(log_decay_ptr[input_index]));
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
#pragma unroll
      for (int key_index = 0; key_index < kHeadSize; ++key_index) {
        state[key_index] = fmaf(
            shared.k[key_index],
            value,
            fmaf(
                shared.b[key_index],
                state_dot_a,
                shared.decay[key_index] * state[key_index]));
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state_ptr[state_base + key_index * kHeadSize + value_index] =
        state[key_index];
  }
}

template <typename io_t>
void launch_factor_scan(
    int num_sequences,
    int num_heads,
    const torch::Tensor& sequence_chunk_offsets,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& state_indices,
    torch::Tensor& state,
    const torch::Tensor& log_decay,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& boundary,
    cudaStream_t stream) {
  scan_factor_boundaries_kernel<io_t>
      <<<dim3(num_heads, num_sequences), kHeadSize, 0, stream>>>(
          num_heads,
          sequence_chunk_offsets.data_ptr<int>(),
          chunk_token_starts.data_ptr<int>(),
          chunk_token_ends.data_ptr<int>(),
          state_indices.data_ptr<int>(),
          state.data_ptr<float>(),
          log_decay.data_ptr<io_t>(),
          k.data_ptr<io_t>(),
          v.data_ptr<io_t>(),
          a.data_ptr<io_t>(),
          b.data_ptr<io_t>(),
          boundary.data_ptr<float>());
}

template <typename io_t>
void launch_recompute_chunk(
    int num_sequences,
    int num_chunks,
    int num_heads,
    const torch::Tensor& sequence_chunk_offsets,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& state_indices,
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& log_decay,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    torch::Tensor& boundary,
    float scale,
    cudaStream_t stream) {
  launch_factor_scan<io_t>(
      num_sequences,
      num_heads,
      sequence_chunk_offsets,
      chunk_token_starts,
      chunk_token_ends,
      state_indices,
      state,
      log_decay,
      k,
      v,
      a,
      b,
      boundary,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_chunk_replay_fp32(
      num_chunks,
      num_heads,
      chunk_token_starts,
      chunk_token_ends,
      boundary,
      r,
      log_decay,
      k,
      v,
      a,
      b,
      output,
      nullptr,
      scale,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void recompute_chunk_fp32_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor boundary,
    double scale) {
  const c10::cuda::CUDAGuard device_guard(state.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int num_sequences = static_cast<int>(state_indices.numel());
  const int num_chunks = static_cast<int>(chunk_token_starts.numel());
  const int num_heads = static_cast<int>(state.size(1));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      r.scalar_type(),
      "flash_rwkv_recompute_chunk_fp32",
      [&] {
        launch_recompute_chunk<scalar_t>(
            num_sequences,
            num_chunks,
            num_heads,
            sequence_chunk_offsets,
            chunk_token_starts,
            chunk_token_ends,
            state_indices,
            state,
            r,
            log_decay,
            k,
            v,
            a,
            b,
            output,
            boundary,
            static_cast<float>(scale),
            stream);
      });
}
