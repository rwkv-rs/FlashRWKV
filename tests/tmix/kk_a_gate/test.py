# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect

import pytest
import torch

from flashrwkv2.tmix.kk_a_gate import infer_tmix_kk_a_gate_forward_varlen


def test_kk_a_gate_head_size_api() -> None:
    signature = inspect.signature(infer_tmix_kk_a_gate_forward_varlen)
    assert signature.parameters["head_size"].default == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("rows", "batch_size", "max_seqlen"),
    ((65535, 255, 257), (65536, 256, 256)),
)
def test_kk_a_gate_cuda_grid_y_boundaries(
    rows: int, batch_size: int, max_seqlen: int
) -> None:
    device = torch.device("cuda")
    channels = 4096
    k = torch.ones(rows, channels, device=device, dtype=torch.float16)
    k_k = torch.ones(channels, device=device, dtype=torch.float16)
    a0 = torch.zeros(channels, device=device, dtype=torch.float16)
    a12 = torch.zeros(rows, channels, device=device, dtype=torch.float16)
    k_a = torch.ones(channels, device=device, dtype=torch.float16)

    new_k, neg_kk, kka = infer_tmix_kk_a_gate_forward_varlen(
        k,
        k_k,
        a0,
        a12,
        k_a,
        batch_size=batch_size,
        max_seqlen=max_seqlen,
    )
    for output in (new_k, neg_kk, kka):
        assert torch.isfinite(output).all()
    assert torch.allclose(new_k[[0, -1]], torch.full_like(new_k[[0, -1]], 0.5))
    assert torch.allclose(neg_kk[[0, -1]], torch.full_like(neg_kk[[0, -1]], -0.125))
    assert torch.allclose(kka[[0, -1]], torch.full_like(kka[[0, -1]], 0.0625))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("head_size", (64, 128, 256))
def test_kk_a_gate_packed_and_grid2d(head_size: int) -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")
    rows, heads = 3, 2
    channels = heads * head_size
    k = torch.randn(rows, channels, device=device, dtype=torch.float16)
    k_k = torch.randn(channels, device=device, dtype=torch.float16)
    a0 = torch.randn(channels, device=device, dtype=torch.float16)
    a12 = torch.randn(rows, channels, device=device, dtype=torch.float16)
    k_a = torch.randn(channels, device=device, dtype=torch.float16)
    expected_scaled = k.float().reshape(rows, heads, head_size) * k_k.float().reshape(1, heads, head_size)
    expected_norm = expected_scaled / expected_scaled.square().sum(-1, keepdim=True).sqrt().clamp_min(1.0e-12)
    gate = torch.sigmoid(a0.float().reshape(1, heads, head_size) + a12.float().reshape(rows, heads, head_size))
    k_a_f = k_a.float().reshape(1, heads, head_size)
    expected_new = k.float().reshape(rows, heads, head_size) * (gate * k_a_f + 1.0 - k_a_f)
    expected = (expected_new, -expected_norm, expected_norm * gate)
    outputs = infer_tmix_kk_a_gate_forward_varlen(
        k, k_k, a0, a12, k_a, head_size=head_size, batch_size=1, max_seqlen=rows
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


def test_kk_a_gate_rejects_unsupported_head_size() -> None:
    with pytest.raises(ValueError, match="head_size"):
        infer_tmix_kk_a_gate_forward_varlen(None, None, None, None, None, head_size=32)
