// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from BlinkDL/RWKV-LM commit 952102498e9ed367ea0a59ee64106916d474d30f.
// Modified by contributors to the FlashRWKV project.
#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> cmix_layer_forward_v5_cuda(
    torch::Tensor x,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight);
std::vector<torch::Tensor> cmix_layer_backward_v5_cuda(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight,
    torch::Tensor mixed,
    torch::Tensor act);
torch::Tensor cmix_mix_forward_v5_cuda(torch::Tensor x, torch::Tensor x_k);
std::vector<torch::Tensor> cmix_mix_backward_v5_cuda(torch::Tensor grad_out, torch::Tensor x, torch::Tensor x_k);

namespace {

void check_bf16_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, name, " must be bf16");
}

void check_same_device(
    const torch::Tensor& expected,
    const torch::Tensor& actual,
    const char* name) {
    TORCH_CHECK(actual.device() == expected.device(), name, " must be on the same device as x");
}

} // namespace

std::vector<torch::Tensor> cmix_layer_forward_v5(
    torch::Tensor x,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight) {
    check_bf16_cuda(x, "x");
    check_bf16_cuda(x_k, "x_k");
    check_bf16_cuda(key_weight, "key_weight");
    check_bf16_cuda(value_weight, "value_weight");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, T, C]");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0, "x dimensions must be positive");
    TORCH_CHECK(x_k.dim() == 1, "x_k must have shape [C]");
    TORCH_CHECK(key_weight.dim() == 2, "key_weight must have shape [4C, C]");
    TORCH_CHECK(value_weight.dim() == 2, "value_weight must have shape [C, 4C]");
    TORCH_CHECK((x.size(2) % 2) == 0, "cmix v5 currently requires even C");
    TORCH_CHECK(x.size(2) == x_k.size(0), "channel size mismatch for x_k");
    TORCH_CHECK(
        key_weight.size(0) == 4 * x.size(2) && key_weight.size(1) == x.size(2),
        "key_weight must have shape [4C, C]");
    TORCH_CHECK(
        value_weight.size(0) == x.size(2) && value_weight.size(1) == 4 * x.size(2),
        "value_weight must have shape [C, 4C]");
    check_same_device(x, x_k, "x_k");
    check_same_device(x, key_weight, "key_weight");
    check_same_device(x, value_weight, "value_weight");
    return cmix_layer_forward_v5_cuda(x, x_k, key_weight, value_weight);
}

std::vector<torch::Tensor> cmix_layer_backward_v5(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight,
    torch::Tensor mixed,
    torch::Tensor act) {
    check_bf16_cuda(grad_out, "grad_out");
    check_bf16_cuda(x, "x");
    check_bf16_cuda(x_k, "x_k");
    check_bf16_cuda(key_weight, "key_weight");
    check_bf16_cuda(value_weight, "value_weight");
    check_bf16_cuda(mixed, "mixed");
    check_bf16_cuda(act, "act");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, T, C]");
    TORCH_CHECK((x.size(2) % 2) == 0, "cmix v5 currently requires even C");
    TORCH_CHECK(grad_out.sizes() == x.sizes(), "grad_out shape mismatch");
    TORCH_CHECK(mixed.sizes() == x.sizes(), "mixed shape mismatch");
    TORCH_CHECK(
        act.dim() == 2 && act.size(0) == x.size(0) * x.size(1) && act.size(1) == 4 * x.size(2),
        "act must have shape [B*T, 4C]");
    TORCH_CHECK(x_k.dim() == 1 && x_k.size(0) == x.size(2), "x_k must have shape [C]");
    TORCH_CHECK(
        key_weight.dim() == 2 && key_weight.size(0) == 4 * x.size(2) && key_weight.size(1) == x.size(2),
        "key_weight must have shape [4C, C]");
    TORCH_CHECK(
        value_weight.dim() == 2 && value_weight.size(0) == x.size(2) && value_weight.size(1) == 4 * x.size(2),
        "value_weight must have shape [C, 4C]");
    check_same_device(x, grad_out, "grad_out");
    check_same_device(x, x_k, "x_k");
    check_same_device(x, key_weight, "key_weight");
    check_same_device(x, value_weight, "value_weight");
    check_same_device(x, mixed, "mixed");
    check_same_device(x, act, "act");
    return cmix_layer_backward_v5_cuda(grad_out, x, x_k, key_weight, value_weight, mixed, act);
}

TORCH_LIBRARY(rwkv7_cmix_bf16_v5, m) {
    m.def("forward", cmix_layer_forward_v5);
    m.def("backward", cmix_layer_backward_v5);
}
