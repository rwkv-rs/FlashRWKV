// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "../../../validation.h"

#include <torch/extension.h>

#include <cstdint>
#include <utility>
#include <vector>

void tmix_kk_a_gate_forward_varlen_cuda(
    int batch_size,
    int max_seqlen,
    int total_tokens,
    int channels,
    int heads,
    int head_size,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor new_k,
    torch::Tensor neg_kk,
    torch::Tensor kka);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;

namespace {

void check_half(const torch::Tensor& tensor, const torch::Tensor& ref, const char* name) {
  check_cuda_contiguous(tensor, name);
  check_same_device(ref, tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
}

}  // namespace

std::vector<torch::Tensor> tmix_kk_a_gate_forward_varlen(
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    int64_t head_size,
    int64_t batch_size,
    int64_t max_seqlen) {
  check_half(k, k, "k");
  TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
              "head_size must be one of 64, 128, or 256");
  TORCH_CHECK(k.dim() == 2 && k.size(0) > 0 && k.size(1) > 0 &&
                  k.size(1) % head_size == 0,
              "k must have packed shape [total_tokens,H*head_size]");
  const int64_t total_tokens = k.size(0);
  const int64_t channels = k.size(1);
  const int heads = static_cast<int>(channels / head_size);
  TORCH_CHECK(batch_size > 0, "batch_size must be positive");
  TORCH_CHECK(max_seqlen > 0, "max_seqlen must be positive");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&k_k, "k_k"},
           {&a0, "a0"}, {&k_a, "k_a"},
       }) {
    check_half(*item.first, k, item.second);
    TORCH_CHECK(item.first->dim() == 1 && item.first->size(0) == channels,
                item.second, " must have shape [C]");
  }
  check_half(a12, k, "a12");
  TORCH_CHECK(a12.sizes() == k.sizes(), "a12 must match k's packed shape");
  auto new_k = torch::empty_like(k);
  auto neg_kk = torch::empty_like(k);
  auto kka = torch::empty_like(k);
  tmix_kk_a_gate_forward_varlen_cuda(
      static_cast<int>(batch_size), static_cast<int>(max_seqlen),
      static_cast<int>(total_tokens), static_cast<int>(channels), heads,
      static_cast<int>(head_size), k, k_k,
      a0, a12, k_a, new_k, neg_kk, kka);
  return {new_k, neg_kk, kka};
}

void register_tmix_kk_a_gate_bindings(py::module_& module) {
  module.def(
      "tmix_kk_a_gate_forward_varlen", &tmix_kk_a_gate_forward_varlen,
      "Packed Albatross TMix key/key-a gate",
      py::arg("k"), py::arg("k_k"), py::arg("a0"), py::arg("a12"),
      py::arg("k_a"), py::arg("head_size") = 64,
      py::arg("batch_size") = 1, py::arg("max_seqlen") = 1);
}
