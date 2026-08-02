// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// The half2 arithmetic and cp.async token pipeline are adapted from
// vllm-rwkv rwkv7_wkv_fp16_v2 at commit
// 6d683f9e49a2997e405c47edc147872c8609513b and BlinkDL/Albatross
// faster3a_2607 at commit 63c53f4abf2cd891dd3a18c8f44f5b2cccc8c64b.
// FlashRWKV consumes explicit log-decay and stores canonical [K,V] state.

#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadSize = 64;
constexpr int kHalf2HeadSize = kHeadSize / 2;
constexpr int kHalfPerVector = sizeof(int4) / sizeof(half);

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
    half2* log_decay,
    half2* k,
    half2* a,
    half2* b,
    half2* b_dummy,
    const half* r_ptr,
    const half* log_decay_ptr,
    const half* k_ptr,
    const half* a_ptr,
    const half* b_ptr) {
  cp_async<4>(
      (thread < 32 ? log_decay : a) + lane,
      reinterpret_cast<const half2*>(
          thread < 32 ? log_decay_ptr + token_base : a_ptr + token_base) +
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

__global__ __launch_bounds__(kHeadSize, 2) void recurrent_fp16_kernel(
    int num_heads,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr,
    const half* __restrict__ log_decay_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    half* __restrict__ output_ptr,
    float scale) {
  const int head_index = static_cast<int>(blockIdx.x);
  const int sequence_index = static_cast<int>(blockIdx.y);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;

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
      log_decay_ptr,
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
    reinterpret_cast<half*>(decay[current])[thread] =
        __float2half_rn(expf(__half2float(
            reinterpret_cast<half*>(decay[current])[thread])));

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
          log_decay_ptr,
          k_ptr,
          a_ptr,
          b_ptr);
    }

    half2 output2 = __float2half2_rn(0.0f);
#pragma unroll
    for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
      half2 updated = __hmul2(state[pair_index], decay[current][pair_index]);
      updated = __hfma2(
          state_dot_a_pair, b[current][pair_index], updated);
      updated = __hfma2(k[current][pair_index], value2, updated);
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

}  // namespace

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
    double scale) {
  const c10::cuda::CUDAGuard device_guard(state.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  recurrent_fp16_kernel<<<
      dim3(static_cast<int>(state.size(1)), static_cast<int>(state_indices.numel())),
      dim3(kHeadSize),
      0,
      stream>>>(
      static_cast<int>(state.size(1)),
      query_start_loc.data_ptr<int>(),
      state_indices.data_ptr<int>(),
      reinterpret_cast<half*>(state.data_ptr()),
      reinterpret_cast<const half*>(r.data_ptr()),
      reinterpret_cast<const half*>(log_decay.data_ptr()),
      reinterpret_cast<const half*>(k.data_ptr()),
      reinterpret_cast<const half*>(v.data_ptr()),
      reinterpret_cast<const half*>(a.data_ptr()),
      reinterpret_cast<const half*>(b.data_ptr()),
      reinterpret_cast<half*>(output.data_ptr()),
      static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
