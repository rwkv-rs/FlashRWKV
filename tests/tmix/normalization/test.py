# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashrwkv2.tmix.normalization import (
    infer_tmix_add_forward_varlen,
    infer_tmix_add_last_layer_norm_forward_varlen,
    infer_tmix_add_layer_norm_forward_varlen,
    infer_tmix_layer_norm_forward_varlen,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tmix_normalization_and_add_families() -> None:
    torch.manual_seed(53)
    device = torch.device("cuda")
    x = torch.randn(5, 8, device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(8, device=device, dtype=torch.float16)
    bias = torch.randn(8, device=device, dtype=torch.float16)
    expected = F.layer_norm(x.float(), (8,), weight.float(), bias.float(), 1.0e-5)
    output = infer_tmix_layer_norm_forward_varlen(x, weight, bias)
    assert torch.allclose(output.float(), expected, atol=0.04, rtol=0.04)
    added = infer_tmix_add_forward_varlen(x, residual)
    assert torch.equal(added, x + residual)
    sum_output, norm_output = infer_tmix_add_layer_norm_forward_varlen(x, residual, weight, bias)
    expected_sum = (x + residual).float()
    assert torch.equal(sum_output, x + residual)
    assert torch.allclose(norm_output.float(), F.layer_norm(expected_sum, (8,), weight.float(), bias.float(), 1.0e-5), atol=0.04, rtol=0.04)
    last = infer_tmix_add_last_layer_norm_forward_varlen(x, residual, weight, bias)
    assert torch.allclose(last.float(), F.layer_norm(expected_sum, (8,), weight.float(), bias.float(), 1.0e-5), atol=0.04, rtol=0.04)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tmix_add_layer_norm_uses_canonical_stats_dispatch() -> None:
    torch.manual_seed(54)
    device = torch.device("cuda")
    channels = 4096
    weight = torch.randn(channels, device=device, dtype=torch.float16)
    bias = torch.randn(channels, device=device, dtype=torch.float16)

    # With a real B>=2 metadata value, rows<192 selects the exact direct
    # Welford body.  Omitting batch_size is deliberately a separate, valid
    # unannotated tokenwise path and must not guess a request boundary.
    x_small = torch.randn(4, channels, device=device, dtype=torch.float16)
    residual_small = torch.randn_like(x_small)
    sum_small, norm_small = infer_tmix_add_layer_norm_forward_varlen(
        x_small,
        residual_small,
        weight,
        bias,
        batch_size=2,
    )
    expected_small = x_small + residual_small
    assert torch.equal(sum_small, expected_small)
    assert torch.allclose(
        norm_small.float(),
        F.layer_norm(expected_small.float(), (channels,), weight.float(), bias.float(), 1.0e-5),
        atol=0.08,
        rtol=0.08,
    )

    # rows>=192 selects the cache-rounded Welford body.  Its exact upstream
    # contract rounds the residual sum into the returned sum tensor first.
    x_large = torch.randn(192, channels, device=device, dtype=torch.float16)
    residual_large = torch.randn_like(x_large)
    sum_large, norm_large = infer_tmix_add_layer_norm_forward_varlen(
        x_large,
        residual_large,
        weight,
        bias,
        batch_size=192,
    )
    expected_large = x_large + residual_large
    assert torch.equal(sum_large, expected_large)
    assert torch.allclose(
        norm_large.float(),
        F.layer_norm(expected_large.float(), (channels,), weight.float(), bias.float(), 1.0e-5),
        atol=0.08,
        rtol=0.08,
    )
