// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV project

#include "../validation.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

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
    int* __restrict__ slot_claims,
    int* __restrict__ status) {
  const int sequence_index =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (sequence_index >= num_sequences) {
    return;
  }

  int error = 0;
  if (sequence_index == 0 &&
      (query_start_loc[0] != 0 ||
       query_start_loc[num_sequences] != total_tokens)) {
    error |= kInvalidEndpoint;
  }

  const int token_start = query_start_loc[sequence_index];
  const int token_end = query_start_loc[sequence_index + 1];
  if (token_start < 0 || token_end <= token_start ||
      token_end > total_tokens) {
    error |= kInvalidSequenceRange;
  }

  const int state_slot = state_indices[sequence_index];
  if (state_slot < 0 || state_slot >= state_pool_size) {
    error |= kInvalidStateSlot;
  } else if (atomicCAS(&slot_claims[state_slot], 0, 1) != 0) {
    error |= kDuplicateStateSlot;
  }

  if (error != 0) {
    atomicOr(status, error);
  }
}

}  // namespace

torch::Tensor validate_recurrent_metadata_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    int64_t total_tokens,
    int64_t state_pool_size) {
  const c10::cuda::CUDAGuard device_guard(query_start_loc.device());
  auto workspace = torch::zeros(
      {state_pool_size + 1}, query_start_loc.options());
  auto status = workspace.narrow(0, state_pool_size, 1);
  const int num_sequences = static_cast<int>(state_indices.numel());
  constexpr int threads = 256;
  const int blocks = (num_sequences + threads - 1) / threads;
  validate_recurrent_metadata_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      query_start_loc.data_ptr<int>(),
      state_indices.data_ptr<int>(),
      num_sequences,
      static_cast<int>(total_tokens),
      static_cast<int>(state_pool_size),
      workspace.data_ptr<int>(),
      status.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return status;
}

}  // namespace flash_rwkv::validation
