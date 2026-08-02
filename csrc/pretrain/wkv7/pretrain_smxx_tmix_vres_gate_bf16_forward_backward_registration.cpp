// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from BlinkDL/RWKV-LM commit 952102498e9ed367ea0a59ee64106916d474d30f.
// Modified by contributors to the FlashRWKV project.
#include <torch/extension.h>

#include <vector>

torch::Tensor tmix_vres_gate_v3_forward_cuda(
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12);

std::vector<torch::Tensor> tmix_vres_gate_v3_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12);

namespace {

void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must be bf16");
}

void check_inputs(
    const torch::Tensor& v,
    const torch::Tensor& v_first,
    const torch::Tensor& v0,
    const torch::Tensor& v12) {
    check_bf16_cuda(v, "v");
    check_bf16_cuda(v_first, "v_first");
    check_bf16_cuda(v0, "v0");
    check_bf16_cuda(v12, "v12");
    TORCH_CHECK(
        v.dim() == 3 && v.size(0) > 0 && v.size(1) > 0 && v.size(2) > 0,
        "v must have non-empty shape [B, T, C]");
    TORCH_CHECK(v_first.sizes() == v.sizes(), "v_first shape mismatch");
    TORCH_CHECK(v12.sizes() == v.sizes(), "v12 shape mismatch");
    TORCH_CHECK(v0.dim() == 1 && v0.size(0) == v.size(2), "v0 must have shape [C]");
    TORCH_CHECK(v_first.device() == v.device(), "v_first must be on the same device as v");
    TORCH_CHECK(v0.device() == v.device(), "v0 must be on the same device as v");
    TORCH_CHECK(v12.device() == v.device(), "v12 must be on the same device as v");
}

torch::Tensor forward(
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12) {
    check_inputs(v, v_first, v0, v12);
    return tmix_vres_gate_v3_forward_cuda(v, v_first, v0, v12);
}

std::vector<torch::Tensor> backward(
    torch::Tensor grad_out,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12) {
    check_bf16_cuda(grad_out, "grad_out");
    check_inputs(v, v_first, v0, v12);
    TORCH_CHECK(grad_out.sizes() == v.sizes(), "grad_out shape mismatch");
    TORCH_CHECK(grad_out.device() == v.device(), "grad_out must be on the same device as v");
    return tmix_vres_gate_v3_backward_cuda(grad_out, v, v_first, v0, v12);
}

} // namespace

TORCH_LIBRARY(rwkv7_tmix_vres_gate_bf16_v3, m) {
    m.def("forward", &forward);
    m.def("backward", &backward);
}
