// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.
// This file owns the TMix caller's packed normalization and residual family.

#include <torch/extension.h>

#include <cmath>
#include <vector>

torch::Tensor tmix_layer_norm_forward_varlen_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);
std::vector<torch::Tensor> tmix_add_layer_norm_forward_varlen_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps,
    int64_t batch_size);
torch::Tensor tmix_add_last_layer_norm_forward_varlen_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps);
torch::Tensor tmix_add_forward_varlen_cuda(torch::Tensor x, torch::Tensor residual);

namespace {

void check_half(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
}

void check_rows(const torch::Tensor& tensor, const torch::Tensor& reference, const char* name) {
  check_half(tensor, name);
  TORCH_CHECK(tensor.device() == reference.device(), name, " must share the input device");
  TORCH_CHECK(tensor.sizes() == reference.sizes(), name, " shape mismatch");
}

void check_affine(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias) {
  check_half(weight, "weight");
  check_half(bias, "bias");
  TORCH_CHECK(weight.device() == x.device() && bias.device() == x.device(),
              "normalization parameters must share x's device");
  TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 &&
                  weight.size(0) == x.size(-1) && bias.size(0) == x.size(-1),
              "weight and bias must have shape [C]");
}

void check_eps(double eps) {
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
}

}  // namespace

torch::Tensor tmix_layer_norm_forward_varlen(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
  check_half(x, "x");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,C]");
  check_affine(x, weight, bias);
  check_eps(eps);
  return tmix_layer_norm_forward_varlen_cuda(x, weight, bias, eps);
}

std::vector<torch::Tensor> tmix_add_layer_norm_forward_varlen(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps,
    int64_t batch_size) {
  check_half(x, "x");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,C]");
  check_rows(residual, x, "residual");
  check_affine(x, weight, bias);
  check_eps(eps);
  TORCH_CHECK(
      batch_size == -1 || (batch_size > 0 && batch_size <= x.size(0)),
      "batch_size must be -1 or a positive value no larger than total_tokens");
  return tmix_add_layer_norm_forward_varlen_cuda(
      x, residual, weight, bias, eps, batch_size);
}

torch::Tensor tmix_add_last_layer_norm_forward_varlen(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {
  check_half(x, "x");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,C]");
  check_rows(residual, x, "residual");
  check_affine(x, weight, bias);
  check_eps(eps);
  return tmix_add_last_layer_norm_forward_varlen_cuda(x, residual, weight, bias, eps);
}

torch::Tensor tmix_add_forward_varlen(torch::Tensor x, torch::Tensor residual) {
  check_half(x, "x");
  check_rows(residual, x, "residual");
  return tmix_add_forward_varlen_cuda(x, residual);
}

void register_tmix_normalization_bindings(py::module_& module) {
  module.def("tmix_layer_norm_forward_varlen", &tmix_layer_norm_forward_varlen,
             py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1.0e-5);
  module.def("tmix_add_layer_norm_forward_varlen", &tmix_add_layer_norm_forward_varlen,
             py::arg("x"), py::arg("residual"), py::arg("weight"), py::arg("bias"),
             py::arg("eps") = 1.0e-5, py::arg("batch_size") = -1);
  module.def("tmix_add_last_layer_norm_forward_varlen",
             &tmix_add_last_layer_norm_forward_varlen,
             py::arg("x"), py::arg("residual"), py::arg("weight"), py::arg("bias"),
             py::arg("eps") = 1.0e-5);
  module.def("tmix_add_forward_varlen", &tmix_add_forward_varlen,
             py::arg("x"), py::arg("residual"));
}
