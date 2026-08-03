// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV project

#include "../validation.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <utility>

namespace flash_rwkv::validation {
namespace {

enum MetadataError : int {
  kInvalidEndpoint = 1 << 0,
  kInvalidSequenceRange = 1 << 1,
  kInvalidStateSlot = 1 << 2,
  kDuplicateStateSlot = 1 << 3,
};

__global__ void validate_recurrent_metadata_kernel(
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    int num_sequences,
    int total_tokens,
    int state_pool_size,
    int* __restrict__ status,
    int* __restrict__ query_start_loc_snapshot,
    int* __restrict__ state_indices_snapshot) {
  __shared__ int shared_status;
  if (threadIdx.x == 0) {
    shared_status = 0;
  }
  __syncthreads();

  if (query_start_loc_snapshot != nullptr) {
    for (int sequence_index = static_cast<int>(threadIdx.x);
         sequence_index < num_sequences;
         sequence_index += static_cast<int>(blockDim.x)) {
      query_start_loc_snapshot[sequence_index] =
          query_start_loc[sequence_index];
      state_indices_snapshot[sequence_index] = state_indices[sequence_index];
    }
    if (threadIdx.x == 0) {
      query_start_loc_snapshot[num_sequences] =
          query_start_loc[num_sequences];
    }
  }
  __syncthreads();

  const int* validated_query_start_loc = query_start_loc_snapshot != nullptr
      ? query_start_loc_snapshot
      : query_start_loc;
  const int* validated_state_indices = state_indices_snapshot != nullptr
      ? state_indices_snapshot
      : state_indices;

  for (int sequence_index = static_cast<int>(threadIdx.x);
       sequence_index < num_sequences;
       sequence_index += static_cast<int>(blockDim.x)) {
    int error = 0;
    if (sequence_index == 0 &&
        (validated_query_start_loc[0] != 0 ||
         validated_query_start_loc[num_sequences] != total_tokens)) {
      error |= kInvalidEndpoint;
    }

    const int token_start = validated_query_start_loc[sequence_index];
    const int token_end = validated_query_start_loc[sequence_index + 1];
    if (token_start < 0 || token_end <= token_start ||
        token_end > total_tokens) {
      error |= kInvalidSequenceRange;
    }

    const int state_slot = validated_state_indices[sequence_index];
    if (state_slot < 0 || state_slot >= state_pool_size) {
      error |= kInvalidStateSlot;
    } else {
      for (int earlier = 0; earlier < sequence_index; ++earlier) {
        if (validated_state_indices[earlier] == state_slot) {
          error |= kDuplicateStateSlot;
          break;
        }
      }
    }

    if (error != 0) {
      atomicOr(&shared_status, error);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    status[0] = shared_status;
  }
}

}  // namespace

PreparedRecurrentMetadata launch_recurrent_metadata_validation(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    int64_t total_tokens,
    int64_t state_pool_size,
    bool snapshot) {
  const c10::cuda::CUDAGuard device_guard(query_start_loc.device());
  auto status = torch::empty({1}, query_start_loc.options());
  auto query_start_loc_snapshot = snapshot
      ? torch::empty_like(query_start_loc)
      : torch::Tensor();
  auto state_indices_snapshot = snapshot
      ? torch::empty_like(state_indices)
      : torch::Tensor();
  const int num_sequences = static_cast<int>(state_indices.numel());
  constexpr int threads = 256;
  validate_recurrent_metadata_kernel<<<
      1,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      query_start_loc.data_ptr<int>(),
      state_indices.data_ptr<int>(),
      num_sequences,
      static_cast<int>(total_tokens),
      static_cast<int>(state_pool_size),
      status.data_ptr<int>(),
      snapshot ? query_start_loc_snapshot.data_ptr<int>() : nullptr,
      snapshot ? state_indices_snapshot.data_ptr<int>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return PreparedRecurrentMetadata{
      std::move(query_start_loc_snapshot),
      std::move(state_indices_snapshot),
      std::move(status)};
}

torch::Tensor validate_recurrent_metadata_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    int64_t total_tokens,
    int64_t state_pool_size) {
  return launch_recurrent_metadata_validation(
             std::move(query_start_loc), std::move(state_indices),
             total_tokens, state_pool_size, true)
      .status;
}

PreparedRecurrentMetadata prepare_recurrent_metadata_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    int64_t total_tokens,
    int64_t state_pool_size) {
  return launch_recurrent_metadata_validation(
      std::move(query_start_loc), std::move(state_indices),
      total_tokens, state_pool_size, true);
}

}  // namespace flash_rwkv::validation
