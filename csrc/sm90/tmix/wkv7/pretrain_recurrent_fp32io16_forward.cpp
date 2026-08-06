// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the RWKV-LM project
// Adapted from RWKV-LM train_temp at revision
// 952102498e9ed367ea0a59ee64106916d474d30f.
// This binding exposes only the raw decay-logit training contract.

#include "../../../validation.h"

#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

void pretrain_recurrent_fp32io16_forward_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor boundary,
    torch::Tensor state_dot_a,
    double scale);

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

  // Chunk metadata is a small scheduler object rather than model data.  The
  // values are checked once at the binding boundary; the CUDA kernels then
  // consume the validated device tensors without hidden padding or copies.
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

void check_training_forward_inputs(
    const torch::Tensor& sequence_chunk_offsets,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& decay_logits,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& output,
    const torch::Tensor& boundary,
    const torch::Tensor& state_dot_a,
    double scale,
    int64_t* chunks_out) {
  check_cuda_contiguous(state, "state");
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");
  TORCH_CHECK(state.dim() == 4 && state.size(0) > 0 && state.size(1) > 0 &&
                  state.size(2) == state.size(3) &&
                  (state.size(2) == 64 || state.size(2) == 128 ||
                   state.size(2) == 256),
              "state must have shape [B,H,D,D] with D in {64,128,256}");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32,
              "training recurrent state must be float32");
  const int64_t chunks = validate_training_metadata(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
      state.size(0), r.size(0), state);
  TORCH_CHECK(r.dim() == 3 && r.size(0) > 0 && r.size(1) == state.size(1) &&
                  r.size(2) == state.size(2),
              "token tensors must have shape [total_tokens,H,D]");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&r, "r"},
           {&decay_logits, "decay_logits"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&output, "output"},
           {&state_dot_a, "state_dot_a"},
       }) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(state, *item.first, item.second);
    TORCH_CHECK(item.first->sizes() == r.sizes() ||
                    item.first == &state_dot_a,
                item.second, " must match token tensor shape");
  }
  TORCH_CHECK(r.scalar_type() == torch::kFloat16 ||
                  r.scalar_type() == torch::kBFloat16,
              "training token tensors must be float16 or bfloat16");
  for (const auto& item : {&decay_logits, &k, &v, &a, &b, &output}) {
    TORCH_CHECK(item->scalar_type() == r.scalar_type(),
                "all token tensors must have the same dtype");
  }
  TORCH_CHECK(state_dot_a.scalar_type() == torch::kFloat32,
              "state_dot_a must be float32");
  const auto expected_boundary_shape =
      std::vector<int64_t>{chunks, state.size(1), state.size(2), state.size(2)};
  TORCH_CHECK(boundary.is_cuda() && boundary.is_contiguous() &&
                  boundary.device() == state.device() &&
                  boundary.scalar_type() == torch::kFloat32 &&
                  boundary.sizes().vec() == expected_boundary_shape,
              "boundary must be float32 with shape [C,H,D,D]");
  *chunks_out = chunks;
}

}  // namespace

void pretrain_recurrent_fp32io16_forward(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor boundary,
    torch::Tensor state_dot_a,
    double scale) {
  int64_t chunks = 0;
  check_training_forward_inputs(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends, state, r,
      decay_logits, k, v, a, b, output, boundary, state_dot_a, scale,
      &chunks);
  (void)chunks;
  pretrain_recurrent_fp32io16_forward_cuda(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends, state, r,
      decay_logits, k, v, a, b, output, boundary, state_dot_a, scale);
}

void register_pretrain_recurrent_bindings(py::module_& module) {
  module.def(
      "pretrain_recurrent_fp32io16_forward",
      &pretrain_recurrent_fp32io16_forward,
      "RWKV-LM train_temp recurrent forward with raw decay logits",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("state"), py::arg("r"),
      py::arg("decay_logits"), py::arg("k"), py::arg("v"), py::arg("a"),
      py::arg("b"), py::arg("output"), py::arg("boundary"),
      py::arg("state_dot_a"), py::arg("scale") = 1.0);
  module.def(
      "pretrain_recurrent_fp32io16_backward",
      &pretrain_recurrent_fp32io16_backward,
      "RWKV-LM train_temp recurrent backward with initial-state gradients",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("final_state"), py::arg("r"),
      py::arg("decay_logits"), py::arg("k"), py::arg("v"), py::arg("a"),
      py::arg("b"), py::arg("state_dot_a"), py::arg("grad_output"),
      py::arg("grad_final_state"), py::arg("boundary"), py::arg("grad_r"),
      py::arg("grad_decay_logits"), py::arg("grad_k"), py::arg("grad_v"),
      py::arg("grad_a"), py::arg("grad_b"),
      py::arg("grad_initial_state"), py::arg("scale") = 1.0);
}
