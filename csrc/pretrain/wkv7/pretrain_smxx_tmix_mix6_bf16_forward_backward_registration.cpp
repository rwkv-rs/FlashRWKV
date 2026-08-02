// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from BlinkDL/RWKV-LM commit 952102498e9ed367ea0a59ee64106916d474d30f.
// Modified by contributors to the FlashRWKV project.
#include <torch/extension.h>

#include <utility>
#include <vector>

std::vector<torch::Tensor> tmix_mix6_forward_v5_cuda(
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g);

std::vector<torch::Tensor> tmix_mix6_backward_v5_cuda(
    torch::Tensor grad_r,
    torch::Tensor grad_w,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_g,
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g);

namespace {

void check_bf16_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, name, " must be bf16");
}

void check_vec(const torch::Tensor& x, int64_t c, const char* name) {
    TORCH_CHECK(x.dim() == 1, name, " must have shape [C]");
    TORCH_CHECK(x.size(0) == c, name, " shape mismatch");
}

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& x_r,
    const torch::Tensor& x_w,
    const torch::Tensor& x_k,
    const torch::Tensor& x_v,
    const torch::Tensor& x_a,
    const torch::Tensor& x_g) {
    check_bf16_cuda(x, "x");
    check_bf16_cuda(x_r, "x_r");
    check_bf16_cuda(x_w, "x_w");
    check_bf16_cuda(x_k, "x_k");
    check_bf16_cuda(x_v, "x_v");
    check_bf16_cuda(x_a, "x_a");
    check_bf16_cuda(x_g, "x_g");
    TORCH_CHECK(
        x.dim() == 3 && x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0,
        "x must have non-empty shape [B, T, C]");
    const int64_t c = x.size(2);
    TORCH_CHECK((c % 2) == 0, "tmix_mix6_v5 currently requires even C");
    const std::vector<std::pair<const torch::Tensor*, const char*>> mixes = {
        {&x_r, "x_r"}, {&x_w, "x_w"}, {&x_k, "x_k"},
        {&x_v, "x_v"}, {&x_a, "x_a"}, {&x_g, "x_g"}};
    for (const auto& [mix, name] : mixes) {
        check_vec(*mix, c, name);
        TORCH_CHECK(mix->device() == x.device(), name, " must be on the same device as x");
    }
}

} // namespace

std::vector<torch::Tensor> forward(
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    check_inputs(x, x_r, x_w, x_k, x_v, x_a, x_g);
    return tmix_mix6_forward_v5_cuda(x, x_r, x_w, x_k, x_v, x_a, x_g);
}

std::vector<torch::Tensor> backward(
    torch::Tensor grad_r,
    torch::Tensor grad_w,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_g,
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    check_bf16_cuda(grad_r, "grad_r");
    check_bf16_cuda(grad_w, "grad_w");
    check_bf16_cuda(grad_k, "grad_k");
    check_bf16_cuda(grad_v, "grad_v");
    check_bf16_cuda(grad_a, "grad_a");
    check_bf16_cuda(grad_g, "grad_g");
    check_inputs(x, x_r, x_w, x_k, x_v, x_a, x_g);
    TORCH_CHECK(grad_r.sizes() == x.sizes(), "grad_r shape mismatch");
    TORCH_CHECK(grad_w.sizes() == x.sizes(), "grad_w shape mismatch");
    TORCH_CHECK(grad_k.sizes() == x.sizes(), "grad_k shape mismatch");
    TORCH_CHECK(grad_v.sizes() == x.sizes(), "grad_v shape mismatch");
    TORCH_CHECK(grad_a.sizes() == x.sizes(), "grad_a shape mismatch");
    TORCH_CHECK(grad_g.sizes() == x.sizes(), "grad_g shape mismatch");
    TORCH_CHECK(grad_r.device() == x.device(), "grad_r must be on the same device as x");
    TORCH_CHECK(grad_w.device() == x.device(), "grad_w must be on the same device as x");
    TORCH_CHECK(grad_k.device() == x.device(), "grad_k must be on the same device as x");
    TORCH_CHECK(grad_v.device() == x.device(), "grad_v must be on the same device as x");
    TORCH_CHECK(grad_a.device() == x.device(), "grad_a must be on the same device as x");
    TORCH_CHECK(grad_g.device() == x.device(), "grad_g must be on the same device as x");
    return tmix_mix6_backward_v5_cuda(grad_r, grad_w, grad_k, grad_v, grad_a, grad_g, x, x_r, x_w, x_k, x_v, x_a, x_g);
}

TORCH_LIBRARY(rwkv7_tmix_mix6_bf16_v5, m) {
    m.def("forward(Tensor x, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]");
    m.def("backward(Tensor grad_r, Tensor grad_w, Tensor grad_k, Tensor grad_v, Tensor grad_a, Tensor grad_g, Tensor x, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rwkv7_tmix_mix6_bf16_v5, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
