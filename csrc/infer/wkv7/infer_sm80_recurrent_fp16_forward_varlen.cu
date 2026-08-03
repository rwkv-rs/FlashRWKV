// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// The half2 arithmetic and cp.async token pipeline are adapted from
// vllm-rwkv rwkv7_wkv_fp16_v2 at commit
// 6d683f9e49a2997e405c47edc147872c8609513b and BlinkDL/Albatross
// faster3a_2607 at commit 63c53f4abf2cd891dd3a18c8f44f5b2cccc8c64b.
// FlashRWKV stores canonical [K,V] state. The product path consumes raw decay
// logits and fuses their retention transform; an explicit log-decay path remains
// available for compatibility and independent-oracle checks.

#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "../../common/wkv7/recurrent_decay.cuh"

namespace {

constexpr int kHeadSize = 64;
constexpr int kHalf2HeadSize = kHeadSize / 2;
constexpr int kHalfPerVector = sizeof(int4) / sizeof(half);
constexpr float kTwoNegative41 = 4.547473508864641e-13f;
constexpr uint32_t kDitherRotation = 2654435769u;

using flash_rwkv::wkv7::RecurrentDecayInput;
using flash_rwkv::wkv7::recurrent_retention;

__device__ __forceinline__ float decay_dither(int phase) {
  const uint32_t bits =
      kDitherRotation * static_cast<uint32_t>(phase);
  return kTwoNegative41 * static_cast<float>(static_cast<int32_t>(bits));
}

template <int Bytes>
__device__ __forceinline__ void cp_async(
    void* shared,
    const void* global,
    bool predicate) {
  static_assert(Bytes == 16 || Bytes == 8 || Bytes == 4);
  const int copied_bytes = predicate ? Bytes : 0;
  const unsigned int shared_address = __cvta_generic_to_shared(shared);
  if constexpr (Bytes == 16) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2, %3;"
        :
        : "r"(shared_address),
          "l"(global),
          "n"(Bytes),
          "r"(copied_bytes));
  } else {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2, %3;"
        :
        : "r"(shared_address),
          "l"(global),
          "n"(Bytes),
          "r"(copied_bytes));
  }
}

__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int NumPending>
__device__ __forceinline__ void cp_wait() {
  if constexpr (NumPending == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(NumPending));
  }
}

__device__ __forceinline__ void prefetch_token(
    int thread,
    int lane,
    int64_t token_base,
    half2* r,
    half2* decay,
    half2* k,
    half2* a,
    half2* b,
    half2* b_dummy,
    const half* r_ptr,
    const half* decay_ptr,
    const half* k_ptr,
    const half* a_ptr,
    const half* b_ptr) {
  cp_async<4>(
      (thread < 32 ? decay : a) + lane,
      reinterpret_cast<const half2*>(
          thread < 32 ? decay_ptr + token_base : a_ptr + token_base) +
          lane,
      true);
  cp_commit();
  cp_async<4>(
      (thread < 32 ? r : k) + lane,
      reinterpret_cast<const half2*>(
          thread < 32 ? r_ptr + token_base : k_ptr + token_base) +
          lane,
      true);
  cp_async<4>(
      (thread < 32 ? b : b_dummy) + lane,
      reinterpret_cast<const half2*>(b_ptr + token_base) + lane,
      thread < 32);
  cp_commit();
}

template <RecurrentDecayInput DecayInput, bool UseDither>
__global__ __launch_bounds__(kHeadSize, 2) void recurrent_fp16_kernel(
    int num_heads,
    int64_t output_elements,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status,
    half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr,
    const half* __restrict__ decay_ptr,
    const half* __restrict__ decay_bias_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    const int* __restrict__ elapsed_t_ptr,
    half* __restrict__ output_ptr,
    float scale) {
  const int head_index = static_cast<int>(blockIdx.x);
  const int sequence_index = static_cast<int>(blockIdx.y);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;

  if (metadata_status[0] != 0) {
    const int64_t block_index =
        static_cast<int64_t>(sequence_index) * num_heads + head_index;
    const int64_t block_count =
        static_cast<int64_t>(gridDim.x) * gridDim.y;
    for (int64_t output_index = block_index * blockDim.x + thread;
         output_index < output_elements;
         output_index += block_count * blockDim.x) {
      output_ptr[output_index] =
          __float2half(__int_as_float(0x7fffffff));
    }
    return;
  }

  int token_start = 0;
  int token_end = 0;
  int state_slot = 0;
  if (lane == 0) {
    token_start = query_start_loc[sequence_index];
    token_end = query_start_loc[sequence_index + 1];
    state_slot = state_indices[sequence_index];
  }
  token_start = __shfl_sync(0xffffffffu, token_start, 0);
  token_end = __shfl_sync(0xffffffffu, token_end, 0);
  state_slot = __shfl_sync(0xffffffffu, state_slot, 0);
  const int token_count = token_end - token_start;
  if (token_count <= 0) {
    return;
  }

  half* state_base =
      state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          kHeadSize * kHeadSize;
  __shared__ __align__(256) half state_shared[kHeadSize][kHeadSize];

#pragma unroll
  for (int vector_iteration = 0;
       vector_iteration < kHeadSize / kHalfPerVector;
       ++vector_iteration) {
    const int vector_index = vector_iteration * kHeadSize + thread;
    const int key_index = vector_index / (kHeadSize / kHalfPerVector);
    const int value_base =
        (vector_index % (kHeadSize / kHalfPerVector)) * kHalfPerVector;
    *reinterpret_cast<int4*>(&state_shared[key_index][value_base]) =
        reinterpret_cast<const int4*>(state_base)[vector_index];
  }
  __syncthreads();

  half2 state[kHalf2HeadSize];
#pragma unroll
  for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
    state[pair_index] = __halves2half2(
        state_shared[pair_index * 2][thread],
        state_shared[pair_index * 2 + 1][thread]);
  }

  __shared__ __align__(128) half2
      r[2][kHalf2HeadSize],
      decay[2][kHalf2HeadSize],
      k[2][kHalf2HeadSize],
      a[2][kHalf2HeadSize],
      b[2][kHalf2HeadSize],
      b_dummy[kHalf2HeadSize];

  int64_t token_base =
      (static_cast<int64_t>(token_start) * num_heads + head_index) * kHeadSize;
  prefetch_token(
      thread,
      lane,
      token_base,
      r[0],
      decay[0],
      k[0],
      a[0],
      b[0],
      b_dummy,
      r_ptr,
      decay_ptr,
      k_ptr,
      a_ptr,
      b_ptr);

  for (int token_offset = 0; token_offset < token_count; ++token_offset) {
    const int current = token_offset & 1;
    const half value = v_ptr[token_base + thread];
    const half2 value2 = __halves2half2(value, value);

    cp_wait<1>();
    __syncthreads();

    half2 state_dot_a2 = __float2half2_rn(0.0f);
#pragma unroll
    for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
      state_dot_a2 =
          __hfma2(a[current][pair_index], state[pair_index], state_dot_a2);
    }
    const half state_dot_a = __hadd(state_dot_a2.x, state_dot_a2.y);
    const half2 state_dot_a_pair =
        __halves2half2(state_dot_a, state_dot_a);
    float decay_input = __half2float(
        reinterpret_cast<half*>(decay[current])[thread]);
    if constexpr (DecayInput == RecurrentDecayInput::kDecayLogits) {
      if (decay_bias_ptr != nullptr) {
        decay_input += __half2float(
            decay_bias_ptr[head_index * kHeadSize + thread]);
      }
    }
    float decay_factor = recurrent_retention<DecayInput>(decay_input);
    if constexpr (UseDither) {
      const int phase = elapsed_t_ptr[state_slot] +
          head_index * kHeadSize + thread + token_offset;
      decay_factor = decay_factor - 1.0f + decay_dither(phase);
    }
    reinterpret_cast<half*>(decay[current])[thread] =
        __float2half_rn(decay_factor);

    cp_wait<0>();
    __syncthreads();

    if (token_offset + 1 < token_count) {
      const int64_t next_token_base =
          token_base + static_cast<int64_t>(num_heads) * kHeadSize;
      prefetch_token(
          thread,
          lane,
          next_token_base,
          r[current ^ 1],
          decay[current ^ 1],
          k[current ^ 1],
          a[current ^ 1],
          b[current ^ 1],
          b_dummy,
          r_ptr,
          decay_ptr,
          k_ptr,
          a_ptr,
          b_ptr);
    }

    half2 output2 = __float2half2_rn(0.0f);
#pragma unroll
    for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
      half2 updated;
      if constexpr (UseDither) {
        updated = __hfma2(
            state_dot_a_pair,
            b[current][pair_index],
            state[pair_index]);
        updated = __hfma2(k[current][pair_index], value2, updated);
        updated = __hfma2(
            state[pair_index], decay[current][pair_index], updated);
      } else {
        updated = __hmul2(
            state[pair_index], decay[current][pair_index]);
        updated = __hfma2(
            state_dot_a_pair, b[current][pair_index], updated);
        updated = __hfma2(k[current][pair_index], value2, updated);
      }
      state[pair_index] = updated;
      output2 = __hfma2(updated, r[current][pair_index], output2);
    }
    const float output =
        (__half2float(output2.x) + __half2float(output2.y)) * scale;
    output_ptr[token_base + thread] = __float2half_rn(output);
    token_base += static_cast<int64_t>(num_heads) * kHeadSize;
  }

#pragma unroll
  for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
    state_shared[pair_index * 2][thread] = state[pair_index].x;
    state_shared[pair_index * 2 + 1][thread] = state[pair_index].y;
  }
  __syncthreads();
#pragma unroll
  for (int vector_iteration = 0;
       vector_iteration < kHeadSize / kHalfPerVector;
       ++vector_iteration) {
    const int vector_index = vector_iteration * kHeadSize + thread;
    const int key_index = vector_index / (kHeadSize / kHalfPerVector);
    const int value_base =
        (vector_index % (kHeadSize / kHalfPerVector)) * kHalfPerVector;
    reinterpret_cast<int4*>(state_base)[vector_index] =
        *reinterpret_cast<const int4*>(&state_shared[key_index][value_base]);
  }
}

template <
    int HeadSize,
    RecurrentDecayInput DecayInput,
    bool UseDither>
__global__ __launch_bounds__(HeadSize, 1)
void recurrent_fp16_generic_kernel(
    int num_heads,
    int64_t output_elements,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status,
    half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr,
    const half* __restrict__ decay_ptr,
    const half* __restrict__ decay_bias_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    const int* __restrict__ elapsed_t_ptr,
    half* __restrict__ output_ptr,
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
          __float2half(__int_as_float(0x7fffffff));
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

  half* state_base =
      state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          HeadSize * HeadSize;
  half state[HeadSize];
#pragma unroll
  for (int key_index = 0; key_index < HeadSize; ++key_index) {
    state[key_index] = state_base[key_index * HeadSize + value_index];
  }

  __shared__ half r[HeadSize];
  __shared__ half decay[HeadSize];
  __shared__ half k[HeadSize];
  __shared__ half a[HeadSize];
  __shared__ half b[HeadSize];

  for (int token_index = token_start; token_index < token_end; ++token_index) {
    const int64_t input_index =
        (static_cast<int64_t>(token_index) * num_heads + head_index) *
            HeadSize +
        value_index;
    r[value_index] = r_ptr[input_index];
    float decay_input = __half2float(decay_ptr[input_index]);
    if constexpr (DecayInput == RecurrentDecayInput::kDecayLogits) {
      if (decay_bias_ptr != nullptr) {
        decay_input += __half2float(
            decay_bias_ptr[head_index * HeadSize + value_index]);
      }
    }
    float decay_factor = recurrent_retention<DecayInput>(decay_input);
    if constexpr (UseDither) {
      const int phase = elapsed_t_ptr[state_slot] +
          head_index * HeadSize + value_index + (token_index - token_start);
      decay_factor = decay_factor - 1.0f + decay_dither(phase);
    }
    decay[value_index] = __float2half_rn(decay_factor);
    k[value_index] = k_ptr[input_index];
    a[value_index] = a_ptr[input_index];
    b[value_index] = b_ptr[input_index];
    __syncthreads();

    float state_dot_a = 0.0f;
#pragma unroll
    for (int key_index = 0; key_index < HeadSize; ++key_index) {
      state_dot_a = fmaf(
          __half2float(a[key_index]),
          __half2float(state[key_index]),
          state_dot_a);
    }

    const float value = __half2float(v_ptr[input_index]);
    float output = 0.0f;
#pragma unroll
    for (int key_index = 0; key_index < HeadSize; ++key_index) {
      float updated = __half2float(b[key_index]) * state_dot_a +
          __half2float(k[key_index]) * value;
      if constexpr (UseDither) {
        updated += __half2float(state[key_index]) *
            (1.0f + __half2float(decay[key_index]));
      } else {
        updated += __half2float(decay[key_index]) *
            __half2float(state[key_index]);
      }
      state[key_index] = __float2half_rn(updated);
      output = fmaf(
          __half2float(r[key_index]),
          __half2float(state[key_index]),
          output);
    }
    output_ptr[input_index] = __float2half_rn(scale * output);
    __syncthreads();
  }

#pragma unroll
  for (int key_index = 0; key_index < HeadSize; ++key_index) {
    state_base[key_index * HeadSize + value_index] = state[key_index];
  }
}

}  // namespace

template <
    flash_rwkv::wkv7::RecurrentDecayInput DecayInput,
    bool UseDither>
void recurrent_fp16_cuda_impl(
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
    torch::Tensor elapsed_t,
    torch::Tensor output,
    torch::Tensor metadata_status,
    double scale) {
  const c10::cuda::CUDAGuard device_guard(state.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(
      static_cast<int>(state.size(1)),
      static_cast<int>(state_indices.numel()));
  switch (state.size(2)) {
    case 64:
      recurrent_fp16_kernel<DecayInput, UseDither>
          <<<grid, kHeadSize, 0, stream>>>(
          static_cast<int>(state.size(1)),
          output.numel(),
          query_start_loc.data_ptr<int>(),
          state_indices.data_ptr<int>(),
          metadata_status.data_ptr<int>(),
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(decay.data_ptr()),
          decay_bias.defined()
              ? reinterpret_cast<const half*>(decay_bias.data_ptr())
              : nullptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          elapsed_t.defined() ? elapsed_t.data_ptr<int>() : nullptr,
          reinterpret_cast<half*>(output.data_ptr()),
          static_cast<float>(scale));
      break;
    case 128:
      recurrent_fp16_generic_kernel<128, DecayInput, UseDither>
          <<<grid, 128, 0, stream>>>(
          static_cast<int>(state.size(1)),
          output.numel(),
          query_start_loc.data_ptr<int>(),
          state_indices.data_ptr<int>(),
          metadata_status.data_ptr<int>(),
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(decay.data_ptr()),
          decay_bias.defined()
              ? reinterpret_cast<const half*>(decay_bias.data_ptr())
              : nullptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          elapsed_t.defined() ? elapsed_t.data_ptr<int>() : nullptr,
          reinterpret_cast<half*>(output.data_ptr()),
          static_cast<float>(scale));
      break;
    case 256:
      recurrent_fp16_generic_kernel<256, DecayInput, UseDither>
          <<<grid, 256, 0, stream>>>(
          static_cast<int>(state.size(1)),
          output.numel(),
          query_start_loc.data_ptr<int>(),
          state_indices.data_ptr<int>(),
          metadata_status.data_ptr<int>(),
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(decay.data_ptr()),
          decay_bias.defined()
              ? reinterpret_cast<const half*>(decay_bias.data_ptr())
              : nullptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          elapsed_t.defined() ? elapsed_t.data_ptr<int>() : nullptr,
          reinterpret_cast<half*>(output.data_ptr()),
          static_cast<float>(scale));
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void recurrent_fp16_cuda(
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
  recurrent_fp16_cuda_impl<
      flash_rwkv::wkv7::RecurrentDecayInput::kLogDecay, false>(
      query_start_loc, state_indices, state, r, log_decay, torch::Tensor(),
      k, v, a, b, torch::Tensor(), output, metadata_status, scale);
}

void recurrent_fp16_from_decay_logits_cuda(
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
    torch::Tensor elapsed_t,
    torch::Tensor output,
    torch::Tensor metadata_status,
    double scale) {
  if (elapsed_t.defined()) {
    recurrent_fp16_cuda_impl<
        flash_rwkv::wkv7::RecurrentDecayInput::kDecayLogits, true>(
        query_start_loc, state_indices, state, r, decay_logits, decay_bias,
        k, v, a, b, elapsed_t, output, metadata_status, scale);
  } else {
    recurrent_fp16_cuda_impl<
        flash_rwkv::wkv7::RecurrentDecayInput::kDecayLogits, false>(
        query_start_loc, state_indices, state, r, decay_logits, decay_bias,
        k, v, a, b, elapsed_t, output, metadata_status, scale);
  }
}
