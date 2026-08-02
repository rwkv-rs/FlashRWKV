// SPDX-License-Identifier: MIT
// Adapted from FlashKDA at commit
// 1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b.

#include "../../bindings.h"
#include "../../validation.h"

#include <cmath>
#include <utility>
#include <vector>

void infer_chunk_bf16_forward_k1_prepare_cuda(
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor chunk_transform,
    torch::Tensor chunk_bias,
    torch::Tensor token_transform,
    torch::Tensor token_bias,
    double scale);
void infer_chunk_bf16_forward_k2_recurrence_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state,
    torch::Tensor output,
    torch::Tensor chunk_transform,
    torch::Tensor chunk_bias,
    torch::Tensor token_transform,
    torch::Tensor token_bias);

using flash_rwkv::validation::check_cuda_contiguous;
using flash_rwkv::validation::check_same_device;
using flash_rwkv::validation::kHeadSize;

void infer_chunk_bf16_forward_k1_prepare(
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor chunk_transform,
    torch::Tensor chunk_bias,
    torch::Tensor token_transform,
    torch::Tensor token_bias,
    double scale) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &chunk_token_starts, "chunk_token_starts"},
           {&chunk_token_ends, "chunk_token_ends"},
           {&r, "r"},
           {&log_decay, "log_decay"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&chunk_transform, "chunk_transform"},
           {&chunk_bias, "chunk_bias"},
           {&token_transform, "token_transform"},
           {&token_bias, "token_bias"},
       }) {
    check_cuda_contiguous(*item.first, item.second);
  }
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");
  TORCH_CHECK(
      chunk_token_starts.scalar_type() == torch::kInt32 &&
          chunk_token_ends.scalar_type() == torch::kInt32,
      "K1 chunk metadata must be int32");
  TORCH_CHECK(
      chunk_token_starts.dim() == 1 &&
          chunk_token_starts.numel() > 0 &&
          chunk_token_starts.sizes() == chunk_token_ends.sizes(),
      "K1 chunk metadata must have shape [C]");
  TORCH_CHECK(
      r.dim() == 3 && r.size(0) > 0 && r.size(1) > 0 &&
          r.size(2) == kHeadSize,
      "K1 token tensors must have shape [total_tokens,H,64]");
  TORCH_CHECK(
      r.sizes() == log_decay.sizes() &&
          r.sizes() == k.sizes() &&
          r.sizes() == v.sizes() &&
          r.sizes() == a.sizes() &&
          r.sizes() == b.sizes(),
      "K1 token tensor shape mismatch");
  TORCH_CHECK(
      r.scalar_type() == torch::kBFloat16 &&
          log_decay.scalar_type() == torch::kBFloat16 &&
          k.scalar_type() == torch::kBFloat16 &&
          v.scalar_type() == torch::kBFloat16 &&
          a.scalar_type() == torch::kBFloat16 &&
          b.scalar_type() == torch::kBFloat16,
      "K1 token tensors must be bf16");

  const int64_t num_chunks = chunk_token_starts.numel();
  const int64_t num_heads = r.size(1);
  const std::vector<int64_t> chunk_shape{
      num_chunks,
      num_heads,
      kHeadSize,
      kHeadSize,
  };
  TORCH_CHECK(
      chunk_transform.sizes().vec() == chunk_shape &&
          chunk_bias.sizes().vec() == chunk_shape &&
          chunk_transform.scalar_type() == torch::kFloat32 &&
          chunk_bias.scalar_type() == torch::kFloat32,
      "K1 chunk workspaces must be fp32 [C,H,64,64]");
  TORCH_CHECK(
      token_transform.sizes() == r.sizes() &&
          token_bias.sizes() == r.sizes() &&
          token_transform.scalar_type() == torch::kFloat32 &&
          token_bias.scalar_type() == torch::kFloat32,
      "K1 token workspaces must be fp32 [total_tokens,H,64]");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &chunk_token_starts, "chunk_token_starts"},
           {&chunk_token_ends, "chunk_token_ends"},
           {&log_decay, "log_decay"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&chunk_transform, "chunk_transform"},
           {&chunk_bias, "chunk_bias"},
           {&token_transform, "token_transform"},
           {&token_bias, "token_bias"},
       }) {
    check_same_device(r, *item.first, item.second);
  }

  infer_chunk_bf16_forward_k1_prepare_cuda(
      chunk_token_starts,
      chunk_token_ends,
      r,
      log_decay,
      k,
      v,
      a,
      b,
      chunk_transform,
      chunk_bias,
      token_transform,
      token_bias,
      scale);
}

void infer_chunk_bf16_forward_k2_recurrence(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state,
    torch::Tensor output,
    torch::Tensor chunk_transform,
    torch::Tensor chunk_bias,
    torch::Tensor token_transform,
    torch::Tensor token_bias) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &sequence_chunk_offsets, "sequence_chunk_offsets"},
           {&chunk_token_starts, "chunk_token_starts"},
           {&chunk_token_ends, "chunk_token_ends"},
           {&state, "state"},
           {&output, "output"},
           {&chunk_transform, "chunk_transform"},
           {&chunk_bias, "chunk_bias"},
           {&token_transform, "token_transform"},
           {&token_bias, "token_bias"},
       }) {
    check_cuda_contiguous(*item.first, item.second);
  }
  TORCH_CHECK(
      sequence_chunk_offsets.scalar_type() == torch::kInt32 &&
          chunk_token_starts.scalar_type() == torch::kInt32 &&
          chunk_token_ends.scalar_type() == torch::kInt32,
      "K2 chunk metadata must be int32");
  TORCH_CHECK(
      state.dim() == 4 && state.size(0) > 0 &&
          state.size(1) > 0 &&
          state.size(2) == kHeadSize &&
          state.size(3) == kHeadSize &&
          state.scalar_type() == torch::kBFloat16,
      "K2 state must be bf16 [N,H,64,64]");
  const int64_t num_sequences = state.size(0);
  const int64_t num_heads = state.size(1);
  TORCH_CHECK(
      sequence_chunk_offsets.dim() == 1 &&
          sequence_chunk_offsets.numel() == num_sequences + 1,
      "K2 sequence_chunk_offsets must have shape [N+1]");
  TORCH_CHECK(
      chunk_token_starts.dim() == 1 &&
          chunk_token_starts.numel() > 0 &&
          chunk_token_starts.sizes() == chunk_token_ends.sizes(),
      "K2 chunk metadata must have shape [C]");
  TORCH_CHECK(
      output.dim() == 3 && output.size(0) > 0 &&
          output.size(1) == num_heads &&
          output.size(2) == kHeadSize &&
          output.scalar_type() == torch::kBFloat16,
      "K2 output must be bf16 [total_tokens,H,64]");

  const int64_t num_chunks = chunk_token_starts.numel();
  const std::vector<int64_t> chunk_shape{
      num_chunks,
      num_heads,
      kHeadSize,
      kHeadSize,
  };
  TORCH_CHECK(
      chunk_transform.sizes().vec() == chunk_shape &&
          chunk_bias.sizes().vec() == chunk_shape &&
          chunk_transform.scalar_type() == torch::kFloat32 &&
          chunk_bias.scalar_type() == torch::kFloat32,
      "K2 chunk workspaces must be fp32 [C,H,64,64]");
  TORCH_CHECK(
      token_transform.sizes() == output.sizes() &&
          token_bias.sizes() == output.sizes() &&
          token_transform.scalar_type() == torch::kFloat32 &&
          token_bias.scalar_type() == torch::kFloat32,
      "K2 token workspaces must be fp32 [total_tokens,H,64]");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &sequence_chunk_offsets, "sequence_chunk_offsets"},
           {&chunk_token_starts, "chunk_token_starts"},
           {&chunk_token_ends, "chunk_token_ends"},
           {&output, "output"},
           {&chunk_transform, "chunk_transform"},
           {&chunk_bias, "chunk_bias"},
           {&token_transform, "token_transform"},
           {&token_bias, "token_bias"},
       }) {
    check_same_device(state, *item.first, item.second);
  }

  infer_chunk_bf16_forward_k2_recurrence_cuda(
      sequence_chunk_offsets,
      chunk_token_starts,
      chunk_token_ends,
      state,
      output,
      chunk_transform,
      chunk_bias,
      token_transform,
      token_bias);
}

void register_infer_experimental_bindings(py::module_& module) {
  module.def(
      "infer_chunk_bf16_forward_k1_prepare",
      &infer_chunk_bf16_forward_k1_prepare,
      "KDA-derived K1 chunk preparation for BF16 inference",
      py::arg("chunk_token_starts"), py::arg("chunk_token_ends"), py::arg("r"),
      py::arg("log_decay"), py::arg("k"), py::arg("v"), py::arg("a"),
      py::arg("b"), py::arg("chunk_transform"), py::arg("chunk_bias"),
      py::arg("token_transform"), py::arg("token_bias"), py::arg("scale"));
  module.def(
      "infer_chunk_bf16_forward_k2_recurrence",
      &infer_chunk_bf16_forward_k2_recurrence,
      "KDA-derived K2 boundary recurrence for BF16 inference",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("state"), py::arg("output"),
      py::arg("chunk_transform"), py::arg("chunk_bias"),
      py::arg("token_transform"), py::arg("token_bias"));
}
