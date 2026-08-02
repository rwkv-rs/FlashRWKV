// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV project

#pragma once

#include <torch/extension.h>

#include <optional>

namespace flash_rwkv::validation {

constexpr int64_t kHeadSize = 64;

struct RecurrentDimensions {
  int64_t num_sequences;
  int64_t num_heads;
  int64_t head_size;
};

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name);
void check_same_device(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name);
void check_optional_like(
    const std::optional<torch::Tensor>& tensor,
    const torch::Tensor& reference,
    const char* name);
RecurrentDimensions check_recurrent_layout(
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
    double scale,
    int64_t required_head_size = 0);
int64_t check_chunk_metadata(
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    const torch::Tensor& state,
    const RecurrentDimensions& dimensions);
torch::Tensor validate_recurrent_metadata_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    int64_t total_tokens,
    int64_t state_pool_size);

}  // namespace flash_rwkv::validation
