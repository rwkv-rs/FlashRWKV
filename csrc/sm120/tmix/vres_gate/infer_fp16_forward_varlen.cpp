// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "../../../validation.h"

#include <torch/extension.h>

#include <cstdint>
#include <utility>

void tmix_vres_gate_forward_varlen_cuda(
    int total_tokens,
    int channels,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12,
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

torch::Tensor tmix_vres_gate_forward_varlen(
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12) {
  check_half(v, v, "v");
  TORCH_CHECK(v.dim() == 2 && v.size(0) > 0 && v.size(1) > 0,
              "v must have packed shape [total_tokens,C]");
  const int64_t total_tokens = v.size(0);
  const int64_t channels = v.size(1);
  check_half(v_first, v, "v_first");
  check_half(v12, v, "v12");
  TORCH_CHECK(v_first.sizes() == v.sizes() && v12.sizes() == v.sizes(),
              "v_first and v12 must match v's packed shape");
  check_half(v0, v, "v0");
  TORCH_CHECK(v0.dim() == 1 && v0.size(0) == channels, "v0 must have shape [C]");
  auto output = torch::empty_like(v);
  tmix_vres_gate_forward_varlen_cuda(
      static_cast<int>(total_tokens), static_cast<int>(channels), v, v_first, v0,
      v12, output);
  return output;
}

void register_tmix_vres_gate_bindings(py::module_& module) {
  module.def(
      "tmix_vres_gate_forward_varlen", &tmix_vres_gate_forward_varlen,
      "Packed Albatross TMix value-residual gate");
}
