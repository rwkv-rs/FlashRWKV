// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV project

#pragma once

#include <torch/extension.h>

#include <optional>

void recurrent_fp32(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, double);
void recurrent_fp16(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, double);
void pretrain_recurrent_fp32io16_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, double);
void materialized_chunk_fp32(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t, double, std::optional<torch::Tensor>);
void pretrain_recurrent_fp32io16_backward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
    torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
    std::optional<torch::Tensor>, std::optional<torch::Tensor>,
    std::optional<torch::Tensor>, std::optional<torch::Tensor>,
    std::optional<torch::Tensor>, double);
void infer_chunk_bf16_forward_k1_prepare(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, double);
void infer_chunk_bf16_forward_k2_recurrence(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void recompute_chunk_fp32(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, double);
