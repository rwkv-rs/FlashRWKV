// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from BlinkDL/RWKV-LM commit 952102498e9ed367ea0a59ee64106916d474d30f.
// Modified by contributors to the FlashRWKV project.
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <cstdint>
#include <vector>

void head_l2wrap_ce_row_chunk_loss_and_grad_v4_cuda(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor loss_rows,
    int64_t row_start,
    int64_t total_rows);
void head_l2wrap_ce_reduce_loss_v4_cuda(torch::Tensor loss_rows, torch::Tensor loss);

namespace {

constexpr int64_t HEAD_L2WRAP_CE_VOCAB = 65536;
#ifndef HEAD_CE_CHUNK
#define HEAD_CE_CHUNK 4096
#endif
static_assert(HEAD_CE_CHUNK > 0, "HEAD_CE_CHUNK must be positive");

void check_cuda_contiguous(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

void check_inputs(const torch::Tensor& hidden, const torch::Tensor& weight, const torch::Tensor& targets) {
    check_cuda_contiguous(hidden, "hidden");
    check_cuda_contiguous(weight, "weight");
    check_cuda_contiguous(targets, "targets");
    TORCH_CHECK(
        hidden.dim() == 3 && hidden.size(0) > 0 && hidden.size(1) > 0 && hidden.size(2) > 0,
        "hidden must have non-empty shape [B,T,C]");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [V,C]");
    TORCH_CHECK(weight.size(0) == HEAD_L2WRAP_CE_VOCAB, "head_l2wrap_ce_v4 currently expects vocab=65536");
    TORCH_CHECK(hidden.size(2) == weight.size(1), "hidden / weight shape mismatch");
    TORCH_CHECK(targets.scalar_type() == torch::kLong, "targets must be int64");
    TORCH_CHECK(targets.numel() == hidden.size(0) * hidden.size(1), "targets shape mismatch");
    TORCH_CHECK(weight.device() == hidden.device(), "weight must be on the same device as hidden");
    TORCH_CHECK(targets.device() == hidden.device(), "targets must be on the same device as hidden");
    TORCH_CHECK(hidden.scalar_type() == torch::kBFloat16, "hidden must be bf16");
    TORCH_CHECK(hidden.scalar_type() == weight.scalar_type(), "hidden and weight dtype must match");
}

int64_t choose_chunk_rows(int64_t rows, int64_t channels, int64_t requested) {
    if (requested > 0) {
        return std::min(requested, rows);
    }
    (void)channels;
    return std::min<int64_t>(HEAD_CE_CHUNK, rows);
}

void mm_chunk_out(torch::Tensor& out, const torch::Tensor& h_chunk, const torch::Tensor& weight) {
    at::mm_out(out, h_chunk, weight.transpose(0, 1));
}

} // namespace

std::vector<torch::Tensor> forward(torch::Tensor hidden, torch::Tensor weight, torch::Tensor targets, int64_t chunk_rows_arg) {
    check_inputs(hidden, weight, targets);
    TORCH_CHECK(chunk_rows_arg >= 0, "chunk_rows must be non-negative");
    const int64_t rows = hidden.size(0) * hidden.size(1);
    const int64_t channels = hidden.size(2);
    const int64_t chunk_rows = choose_chunk_rows(rows, channels, chunk_rows_arg);

    auto h2d = hidden.view({rows, channels});
    auto flat_targets = targets.view({rows});
    auto meta_opts = torch::TensorOptions().device(hidden.device()).dtype(torch::kFloat32);
    auto loss_rows = torch::empty({rows}, meta_opts);
    auto loss = torch::empty({}, meta_opts);
    auto grad_hidden = torch::empty_like(hidden);
    auto grad_h2d = grad_hidden.view({rows, channels});
    auto grad_weight = torch::zeros_like(weight);
    auto logits = torch::empty({chunk_rows, HEAD_L2WRAP_CE_VOCAB}, hidden.options());

    for (int64_t start = 0; start < rows; start += chunk_rows) {
        const int64_t len = std::min(chunk_rows, rows - start);
        auto h_chunk = h2d.narrow(0, start, len);
        auto logits_use = logits.narrow(0, 0, len);
        mm_chunk_out(logits_use, h_chunk, weight);
        head_l2wrap_ce_row_chunk_loss_and_grad_v4_cuda(logits_use, flat_targets, loss_rows, start, rows);
        auto grad_h_chunk = grad_h2d.narrow(0, start, len);
        at::mm_out(grad_h_chunk, logits_use, weight);
        grad_weight.addmm_(logits_use.transpose(0, 1), h_chunk);
    }

    head_l2wrap_ce_reduce_loss_v4_cuda(loss_rows, loss);
    return {loss, grad_hidden, grad_weight};
}

TORCH_LIBRARY(rwkv7_head_l2wrap_ce_bf16_v4, m) {
    m.def("forward", &forward);
}
