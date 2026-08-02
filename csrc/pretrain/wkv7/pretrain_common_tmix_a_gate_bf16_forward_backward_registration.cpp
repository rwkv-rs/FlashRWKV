// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from BlinkDL/RWKV-LM commit 952102498e9ed367ea0a59ee64106916d474d30f.
// Modified by contributors to the FlashRWKV project.
#include <torch/extension.h>

#include <vector>

torch::Tensor tmix_a_gate_forward_cuda(
    torch::Tensor a0,
    torch::Tensor a12);

std::vector<torch::Tensor> tmix_a_gate_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor a0,
    torch::Tensor a12);

namespace {

void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must be bf16");
}

torch::Tensor forward(torch::Tensor a0, torch::Tensor a12) {
    check_bf16_cuda(a0, "a0");
    check_bf16_cuda(a12, "a12");
    TORCH_CHECK(a0.dim() == 1 && a0.size(0) > 0, "a0 must have non-empty shape [C]");
    TORCH_CHECK(
        a12.dim() == 3 && a12.size(0) > 0 && a12.size(1) > 0 && a12.size(2) > 0,
        "a12 must have non-empty shape [B, T, C]");
    TORCH_CHECK(a0.size(0) == a12.size(2), "a0 must have shape [C]");
    TORCH_CHECK(a0.device() == a12.device(), "a0 and a12 must be on the same device");
    return tmix_a_gate_forward_cuda(a0, a12);
}

std::vector<torch::Tensor> backward(
    torch::Tensor grad_out,
    torch::Tensor a0,
    torch::Tensor a12) {
    check_bf16_cuda(grad_out, "grad_out");
    check_bf16_cuda(a0, "a0");
    check_bf16_cuda(a12, "a12");
    TORCH_CHECK(a12.dim() == 3, "a12 must have shape [B, T, C]");
    TORCH_CHECK(a0.dim() == 1 && a0.size(0) == a12.size(2), "a0 must have shape [C]");
    TORCH_CHECK(grad_out.sizes() == a12.sizes(), "grad_out shape mismatch");
    TORCH_CHECK(a0.device() == a12.device(), "a0 and a12 must be on the same device");
    TORCH_CHECK(grad_out.device() == a12.device(), "grad_out must be on the same device as a12");
    return tmix_a_gate_backward_cuda(grad_out, a0, a12);
}

} // namespace

TORCH_LIBRARY(rwkv7_tmix_a_gate_bf16, m) {
    m.def("forward", &forward);
    m.def("backward", &backward);
}
