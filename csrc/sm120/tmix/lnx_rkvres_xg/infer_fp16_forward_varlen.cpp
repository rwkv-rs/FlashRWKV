// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "../../../validation.h"

#include <torch/extension.h>

#include <cstdint>
#include <utility>

void tmix_lnx_rkvres_xg_forward_varlen_cuda(
    int batch_size,
    int max_seqlen,
    int total_tokens,
    int channels,
    int heads,
    int head_size,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g,
    torch::Tensor output);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;

namespace {

void check_half(const torch::Tensor& tensor, const torch::Tensor& reference, const char* name) {
  check_cuda_contiguous(tensor, name);
  check_same_device(reference, tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
}

}  // namespace

torch::Tensor tmix_lnx_rkvres_xg_forward_varlen(
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g,
    int64_t head_size,
    int64_t batch_size,
    int64_t max_seqlen) {
  check_half(x, x, "x");
  TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
              "head_size must be one of 64, 128, or 256");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0 &&
                  x.size(1) % head_size == 0,
              "x must have packed shape [total_tokens,H*head_size]");
  const int64_t total_tokens = x.size(0);
  const int64_t channels = x.size(1);
  const int heads = static_cast<int>(channels / head_size);
  TORCH_CHECK(batch_size > 0, "batch_size must be positive");
  TORCH_CHECK(max_seqlen > 0, "max_seqlen must be positive");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&r, "r"},
           {&k, "k"}, {&v, "v"}, {&g, "g"},
       }) {
    check_half(*item.first, x, item.second);
    TORCH_CHECK(item.first->sizes() == x.sizes(), item.second,
                " must match x's packed shape");
  }
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&r_k, "r_k"},
           {&weight, "weight"}, {&bias, "bias"},
       }) {
    check_half(*item.first, x, item.second);
    TORCH_CHECK(item.first->dim() == 1 && item.first->size(0) == channels,
                item.second, " must have shape [C]");
  }
  auto output = torch::empty_like(x);
  tmix_lnx_rkvres_xg_forward_varlen_cuda(
      static_cast<int>(batch_size), static_cast<int>(max_seqlen),
      static_cast<int>(total_tokens), static_cast<int>(channels), heads,
      static_cast<int>(head_size),
      x, r, k, v, r_k, weight, bias, g, output);
  return output;
}

void register_tmix_lnx_rkvres_xg_bindings(py::module_& module) {
  module.def(
      "tmix_lnx_rkvres_xg_forward_varlen", &tmix_lnx_rkvres_xg_forward_varlen,
      "Packed Albatross TMix lnx/rkv-residual/gate",
      py::arg("x"), py::arg("r"), py::arg("k"), py::arg("v"),
      py::arg("r_k"), py::arg("weight"), py::arg("bias"), py::arg("g"),
      py::arg("head_size") = 64, py::arg("batch_size") = 1,
      py::arg("max_seqlen") = 1);
}
