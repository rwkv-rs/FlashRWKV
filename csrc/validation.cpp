// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV project

#include "validation.h"

#include <cmath>
#include <limits>
#include <utility>

namespace flash_rwkv::validation {

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(
      tensor.device() == reference.device(),
      name,
      " must be on the same device as state");
}

void check_optional_like(
    const std::optional<torch::Tensor>& tensor,
    const torch::Tensor& reference,
    const char* name) {
  if (!tensor.has_value()) {
    return;
  }
  check_cuda_contiguous(*tensor, name);
  check_same_device(reference, *tensor, name);
  TORCH_CHECK(
      tensor->sizes() == reference.sizes(),
      name,
      " must match the reference shape");
  TORCH_CHECK(
      tensor->scalar_type() == reference.scalar_type(),
      name,
      " must match the reference dtype");
}

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
    int64_t required_head_size) {
  check_cuda_contiguous(query_start_loc, "query_start_loc");
  check_cuda_contiguous(state_indices, "state_indices");
  check_cuda_contiguous(state, "state");
  check_cuda_contiguous(r, "r");
  check_cuda_contiguous(log_decay, "log_decay");
  check_cuda_contiguous(k, "k");
  check_cuda_contiguous(v, "v");
  check_cuda_contiguous(a, "a");
  check_cuda_contiguous(b, "b");
  check_cuda_contiguous(output, "output");

  TORCH_CHECK(
      query_start_loc.scalar_type() == torch::kInt32,
      "query_start_loc must be int32");
  TORCH_CHECK(
      state_indices.scalar_type() == torch::kInt32,
      "state_indices must be int32");
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");

  const int64_t num_sequences = state_indices.numel();
  TORCH_CHECK(
      num_sequences > 0 && num_sequences <= 65535,
      "state_indices must contain 1..65535 sequences");
  TORCH_CHECK(state_indices.dim() == 1, "state_indices must have shape [N]");
  TORCH_CHECK(
      query_start_loc.dim() == 1 &&
          query_start_loc.size(0) == num_sequences + 1,
      "query_start_loc must have shape [N+1]");
  TORCH_CHECK(
      state.dim() == 4 && state.size(0) > 0 && state.size(1) > 0 &&
          state.size(2) == state.size(3),
      "state must have square shape [slots,H,D,D]");
  const int64_t head_size = state.size(2);
  if (required_head_size > 0) {
    TORCH_CHECK(
        head_size == required_head_size,
        "this operator requires head size ",
        required_head_size,
        ", got ",
        head_size);
  } else {
    TORCH_CHECK(
        head_size == 64 || head_size == 128 || head_size == 256,
        "recurrent head size must be 64, 128, or 256, got ",
        head_size);
  }
  TORCH_CHECK(
      state.size(0) <= std::numeric_limits<int>::max(),
      "state slot count must fit in int32");

  const int64_t num_heads = state.size(1);
  TORCH_CHECK(
      num_heads <= std::numeric_limits<int>::max(),
      "head count must fit in int32");
  TORCH_CHECK(
      r.dim() == 3 && r.size(0) > 0 && r.size(1) == num_heads &&
          r.size(2) == head_size,
      "r must have shape [total_tokens,H,D] matching state head size");
  TORCH_CHECK(
      r.size(0) <= std::numeric_limits<int>::max(),
      "token count must fit in int32");
  TORCH_CHECK(
      r.sizes() == log_decay.sizes() && r.sizes() == k.sizes() &&
          r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
          r.sizes() == b.sizes() && r.sizes() == output.sizes(),
      "r,log_decay,k,v,a,b,output shape mismatch");
  TORCH_CHECK(
      r.scalar_type() == log_decay.scalar_type() &&
          r.scalar_type() == k.scalar_type() &&
          r.scalar_type() == v.scalar_type() &&
          r.scalar_type() == a.scalar_type() &&
          r.scalar_type() == b.scalar_type() &&
          r.scalar_type() == output.scalar_type(),
      "r,log_decay,k,v,a,b,output dtype mismatch");

  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &query_start_loc, "query_start_loc"},
           {&state_indices, "state_indices"},
           {&r, "r"},
           {&log_decay, "log_decay"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&output, "output"},
       }) {
    check_same_device(state, *item.first, item.second);
  }

  return RecurrentDimensions{num_sequences, num_heads, head_size};
}

int64_t check_chunk_metadata(
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    const torch::Tensor& state,
    const RecurrentDimensions& dimensions) {
  check_cuda_contiguous(chunk_token_starts, "chunk_token_starts");
  check_cuda_contiguous(chunk_token_ends, "chunk_token_ends");
  TORCH_CHECK(
      chunk_token_starts.scalar_type() == torch::kInt32 &&
          chunk_token_ends.scalar_type() == torch::kInt32,
      "chunk token metadata must be int32");
  TORCH_CHECK(
      chunk_token_starts.dim() == 1 && chunk_token_starts.numel() > 0 &&
          chunk_token_starts.sizes() == chunk_token_ends.sizes(),
      "chunk_token_starts and chunk_token_ends must have shape [C]");
  check_same_device(state, chunk_token_starts, "chunk_token_starts");
  check_same_device(state, chunk_token_ends, "chunk_token_ends");

  const int64_t num_chunks = chunk_token_starts.numel();
  TORCH_CHECK(
      num_chunks * dimensions.num_heads <=
          std::numeric_limits<int>::max(),
      "chunk/head grid must fit in int32");
  TORCH_CHECK(
      dimensions.num_sequences * dimensions.num_heads <=
          std::numeric_limits<int>::max(),
      "sequence/head grid must fit in int32");
  return num_chunks;
}

}  // namespace flash_rwkv::validation
