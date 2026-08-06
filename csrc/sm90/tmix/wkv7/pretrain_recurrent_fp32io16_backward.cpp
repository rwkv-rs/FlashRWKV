// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the RWKV-LM project
// Adapted from RWKV-LM train_temp at revision
// 952102498e9ed367ea0a59ee64106916d474d30f.

#include "../../../validation.h"

#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

void pretrain_recurrent_fp32io16_backward_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor final_state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor state_dot_a,
    torch::Tensor grad_output,
    torch::Tensor grad_final_state,
    torch::Tensor boundary,
    torch::Tensor grad_r,
    torch::Tensor grad_decay_logits,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_b,
    torch::Tensor grad_initial_state,
    double scale);

using flash_rwkv::validation::check_cuda_contiguous;
using flash_rwkv::validation::check_same_device;

namespace {

int64_t validate_training_metadata(
    const torch::Tensor& sequence_chunk_offsets,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    int64_t sequences,
    int64_t total_tokens,
    const torch::Tensor& reference) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &sequence_chunk_offsets, "sequence_chunk_offsets"},
           {&chunk_token_starts, "chunk_token_starts"},
           {&chunk_token_ends, "chunk_token_ends"},
       }) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(reference, *item.first, item.second);
    TORCH_CHECK(item.first->scalar_type() == torch::kInt32,
                item.second, " must be int32");
  }
  TORCH_CHECK(sequence_chunk_offsets.dim() == 1 &&
                  sequence_chunk_offsets.numel() == sequences + 1,
              "sequence_chunk_offsets must have shape [B+1]");
  TORCH_CHECK(chunk_token_starts.dim() == 1 &&
                  chunk_token_starts.numel() > 0 &&
                  chunk_token_starts.sizes() == chunk_token_ends.sizes(),
              "chunk token metadata must have matching shape [C]");
  auto sequence_cpu = sequence_chunk_offsets.to(torch::kCPU).contiguous();
  auto starts_cpu = chunk_token_starts.to(torch::kCPU).contiguous();
  auto ends_cpu = chunk_token_ends.to(torch::kCPU).contiguous();
  const auto* sequence = sequence_cpu.data_ptr<int32_t>();
  const auto* starts = starts_cpu.data_ptr<int32_t>();
  const auto* ends = ends_cpu.data_ptr<int32_t>();
  const int64_t chunks = chunk_token_starts.numel();
  TORCH_CHECK(sequence[0] == 0 && sequence[sequences] == chunks,
              "sequence_chunk_offsets must cover all chunks");
  int32_t previous_token_end = 0;
  for (int64_t sequence_index = 0; sequence_index < sequences;
       ++sequence_index) {
    const int32_t chunk_start = sequence[sequence_index];
    const int32_t chunk_end = sequence[sequence_index + 1];
    TORCH_CHECK(chunk_start >= 0 && chunk_end > chunk_start &&
                    chunk_end <= chunks,
                "each sequence must own at least one ordered chunk");
    TORCH_CHECK(starts[chunk_start] == previous_token_end,
                "chunks must cover packed tokens without gaps");
    for (int32_t chunk = chunk_start; chunk < chunk_end; ++chunk) {
      TORCH_CHECK(starts[chunk] >= 0 && ends[chunk] > starts[chunk] &&
                      ends[chunk] <= total_tokens &&
                      (chunk == chunk_start ||
                       starts[chunk] == ends[chunk - 1]),
                  "chunk token ranges must be contiguous and non-empty");
    }
    previous_token_end = ends[chunk_end - 1];
  }
  TORCH_CHECK(previous_token_end == total_tokens,
              "chunks must cover exactly all packed tokens");
  return chunks;
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
  TORCH_CHECK(tensor->sizes() == reference.sizes(), name,
              " must match the reference shape");
  TORCH_CHECK(tensor->scalar_type() == reference.scalar_type(), name,
              " must match the reference dtype");
}

}  // namespace

void pretrain_recurrent_fp32io16_backward(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor final_state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor state_dot_a,
    std::optional<torch::Tensor> grad_output,
    std::optional<torch::Tensor> grad_final_state,
    torch::Tensor boundary,
    std::optional<torch::Tensor> grad_r,
    std::optional<torch::Tensor> grad_decay_logits,
    std::optional<torch::Tensor> grad_k,
    std::optional<torch::Tensor> grad_v,
    std::optional<torch::Tensor> grad_a,
    std::optional<torch::Tensor> grad_b,
    std::optional<torch::Tensor> grad_initial_state,
    double scale) {
  check_cuda_contiguous(final_state, "final_state");
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");
  TORCH_CHECK(final_state.dim() == 4 && final_state.size(0) > 0 &&
                  final_state.size(1) > 0 &&
                  final_state.size(2) == final_state.size(3) &&
                  (final_state.size(2) == 64 || final_state.size(2) == 128 ||
                   final_state.size(2) == 256) &&
                  final_state.scalar_type() == torch::kFloat32,
              "final_state must be float32 [B,H,D,D] with D in {64,128,256}");
  const int64_t chunks = validate_training_metadata(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
      final_state.size(0), r.size(0), final_state);
  TORCH_CHECK(r.dim() == 3 && r.size(0) > 0 && r.size(1) == final_state.size(1) &&
                  r.size(2) == final_state.size(2),
              "token tensors must have shape [total_tokens,H,D]");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&r, "r"},
           {&decay_logits, "decay_logits"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&state_dot_a, "state_dot_a"},
       }) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(final_state, *item.first, item.second);
    TORCH_CHECK(item.first->sizes() == r.sizes(), item.second,
                " must match token tensor shape");
  }
  TORCH_CHECK(r.scalar_type() == torch::kFloat16 ||
                  r.scalar_type() == torch::kBFloat16,
              "training token tensors must be float16 or bfloat16");
  for (const auto& item : {&decay_logits, &k, &v, &a, &b}) {
    TORCH_CHECK(item->scalar_type() == r.scalar_type(),
                "all token tensors must have the same dtype");
  }
  TORCH_CHECK(state_dot_a.scalar_type() == torch::kFloat32,
              "state_dot_a must be float32");
  check_cuda_contiguous(boundary, "boundary");
  check_same_device(final_state, boundary, "boundary");
  const auto expected_boundary_shape = std::vector<int64_t>{
      chunks, final_state.size(1), final_state.size(2), final_state.size(2)};
  TORCH_CHECK(boundary.scalar_type() == torch::kFloat32 &&
                  boundary.sizes().vec() == expected_boundary_shape,
              "boundary must be float32 with shape [C,H,D,D]");
  TORCH_CHECK(grad_output.has_value() || grad_final_state.has_value(),
              "at least one upstream gradient is required");
  check_optional_like(grad_output, r, "grad_output");
  check_optional_like(grad_final_state, final_state, "grad_final_state");
  check_optional_like(grad_r, r, "grad_r");
  check_optional_like(grad_decay_logits, r, "grad_decay_logits");
  check_optional_like(grad_k, r, "grad_k");
  check_optional_like(grad_v, r, "grad_v");
  check_optional_like(grad_a, r, "grad_a");
  check_optional_like(grad_b, r, "grad_b");
  check_optional_like(grad_initial_state, final_state, "grad_initial_state");

  pretrain_recurrent_fp32io16_backward_cuda(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
      final_state, r, decay_logits, k, v, a, b, state_dot_a,
      grad_output.value_or(torch::Tensor()),
      grad_final_state.value_or(torch::Tensor()), boundary,
      grad_r.value_or(torch::Tensor()),
      grad_decay_logits.value_or(torch::Tensor()), grad_k.value_or(torch::Tensor()),
      grad_v.value_or(torch::Tensor()), grad_a.value_or(torch::Tensor()),
      grad_b.value_or(torch::Tensor()),
      grad_initial_state.value_or(torch::Tensor()), scale);
}
