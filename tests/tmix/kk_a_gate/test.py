# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.tmix.kk_a_gate import infer_tmix_kk_a_gate_forward_varlen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_kk_a_gate_packed_and_grid2d() -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")
    rows, heads = 3, 2
    channels = heads * 64
    k = torch.randn(rows, channels, device=device, dtype=torch.float16)
    k_k = torch.randn(channels, device=device, dtype=torch.float16)
    a0 = torch.randn(channels, device=device, dtype=torch.float16)
    a12 = torch.randn(rows, channels, device=device, dtype=torch.float16)
    k_a = torch.randn(channels, device=device, dtype=torch.float16)
    expected_scaled = k.float().reshape(rows, heads, 64) * k_k.float().reshape(1, heads, 64)
    expected_norm = expected_scaled / expected_scaled.square().sum(-1, keepdim=True).sqrt().clamp_min(1.0e-12)
    gate = torch.sigmoid(a0.float().reshape(1, heads, 64) + a12.float().reshape(rows, heads, 64))
    k_a_f = k_a.float().reshape(1, heads, 64)
    expected_new = k.float().reshape(rows, heads, 64) * (gate * k_a_f + 1.0 - k_a_f)
    expected = (expected_new, -expected_norm, expected_norm * gate)
    outputs = infer_tmix_kk_a_gate_forward_varlen(
        k, k_k, a0, a12, k_a, batch_size=1, max_seqlen=rows
    )
    for output, reference in zip(outputs, expected, strict=True):
        assert torch.allclose(
            output.float().reshape_as(reference), reference, atol=0.04, rtol=0.04
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_kk_a_gate_rejects_unaligned_channels() -> None:
    device = torch.device("cuda")
    packed = torch.zeros(2, 65, device=device, dtype=torch.float16)
    vector = torch.zeros(65, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="packed shape"):
        infer_tmix_kk_a_gate_forward_varlen(packed, vector, vector, packed, vector)
