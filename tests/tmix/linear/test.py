# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flash_rwkv.tmix.linear import (
    infer_tmix_linear_act_sigmoid_forward_varlen,
    infer_tmix_linear_act_tanh_forward_varlen,
    infer_tmix_linear_attention_c2c_forward_varlen,
    infer_tmix_linear_ffn_key_forward_varlen,
    infer_tmix_linear_forward_varlen,
    infer_tmix_linear_t_forward_varlen,
    infer_tmix_linear_t_sigmoid_forward_varlen,
    infer_tmix_linear_t_tanh_forward_varlen,
    infer_tmix_linear_t_vres_forward_varlen,
    infer_tmix_linear_rank_in_forward_varlen,
    infer_tmix_linear_rank_out_forward_varlen,
    infer_tmix_linear_rank_out_sigmoid_forward_varlen,
    infer_tmix_linear_rank_out_tanh_forward_varlen,
    infer_tmix_lowrank_in_forward_varlen,
    infer_tmix_lowrank_wagv_in_forward_varlen,
    infer_tmix_lowrank_out_forward_varlen,
    infer_tmix_lowrank_vres_forward_varlen,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tmix_linear_and_lowrank_families() -> None:
    torch.manual_seed(51)
    device = torch.device("cuda")
    # The canonical Albatross rank kernels intentionally fail closed outside
    # their tuned shape domain: K >= 1024, output C >= 1024, and M <= 4/8.
    rows, channels, rank, output_channels = 4, 1024, 8, 1024
    x = torch.randn(rows, channels, device=device, dtype=torch.float16)
    weight = torch.randn(output_channels, channels, device=device, dtype=torch.float16)
    assert torch.allclose(
        infer_tmix_linear_forward_varlen(x, weight).float(), x.float() @ weight.float().t(), atol=0.04, rtol=0.04
    )
    transposed = weight.t().contiguous()
    assert torch.allclose(
        infer_tmix_linear_forward_varlen(x, transposed, weight_is_transposed=True).float(), x.float() @ transposed.float(), atol=0.04, rtol=0.04
    )
    attention_linear = infer_tmix_linear_attention_c2c_forward_varlen(x, weight)
    ffn_key_linear = infer_tmix_linear_ffn_key_forward_varlen(x, weight)
    expected_linear = x.float() @ weight.float().t()
    assert torch.allclose(attention_linear.float(), expected_linear, atol=0.04, rtol=0.04)
    assert torch.allclose(ffn_key_linear.float(), expected_linear, atol=0.04, rtol=0.04)

    rank_rows, rank_k, rank_n = 4, 8, 1024
    rank_x = torch.randn(rank_rows, rank_k, device=device, dtype=torch.float16)
    rank_weight_t = torch.randn(rank_n, rank_k, device=device, dtype=torch.float16)
    rank_weight_t = rank_weight_t.contiguous()
    rank_linear = infer_tmix_linear_t_forward_varlen(rank_x, rank_weight_t)
    assert torch.allclose(
        rank_linear.float(), rank_x.float() @ rank_weight_t.float().t(), atol=0.04, rtol=0.04
    )
    rank_tanh = infer_tmix_linear_t_tanh_forward_varlen(rank_x, rank_weight_t)
    assert torch.allclose(
        rank_tanh.float(), torch.tanh(rank_x.float()) @ rank_weight_t.float().t(), atol=0.04, rtol=0.04
    )
    rank_sigmoid = infer_tmix_linear_t_sigmoid_forward_varlen(rank_x, rank_weight_t)
    assert torch.allclose(
        rank_sigmoid.float(), torch.sigmoid(rank_x.float()) @ rank_weight_t.float().t(), atol=0.04, rtol=0.04
    )
    rank_tanh_input = infer_tmix_linear_act_tanh_forward_varlen(rank_x)
    rank_sigmoid_input = infer_tmix_linear_act_sigmoid_forward_varlen(rank_x)
    assert torch.allclose(
        rank_tanh_input.float(), torch.tanh(rank_x.float()), atol=0.002, rtol=0.002
    )
    assert torch.allclose(
        rank_sigmoid_input.float(), torch.sigmoid(rank_x.float()), atol=0.002, rtol=0.002
    )
    rank_v = torch.randn(rank_rows, rank_n, device=device, dtype=torch.float16)
    rank_v_first = torch.randn_like(rank_v)
    rank_v0 = torch.randn(rank_n, device=device, dtype=torch.float16)
    rank_vres = infer_tmix_linear_t_vres_forward_varlen(
        rank_x, rank_weight_t, rank_v, rank_v_first, rank_v0
    )
    gate = torch.sigmoid(rank_x.float() @ rank_weight_t.float().t() + rank_v0.float())
    assert torch.allclose(
        rank_vres.float(), rank_v.float() + (rank_v_first.float() - rank_v.float()) * gate,
        atol=0.05,
        rtol=0.05,
    )

    x_w, x_a, x_g = [torch.randn(rows, channels, device=device, dtype=torch.float16) for _ in range(3)]
    x_v = torch.randn(rows, channels, device=device, dtype=torch.float16)
    w1, a1, g1 = [torch.randn(rank, channels, device=device, dtype=torch.float16) for _ in range(3)]
    v1_t = torch.randn(rank, channels, device=device, dtype=torch.float16)
    lowrank_in = infer_tmix_lowrank_in_forward_varlen(x_w, x_a, x_g, w1, a1, g1)
    for output, source, projection in zip(lowrank_in, (x_w, x_a, x_g), (w1, a1, g1), strict=True):
        assert torch.allclose(output.float(), source.float() @ projection.float().t(), atol=0.04, rtol=0.04)

    lowrank_wagv_in = infer_tmix_lowrank_wagv_in_forward_varlen(
        x_w, x_a, x_g, x_v, w1, a1, g1, v1_t
    )
    for output, source, projection in zip(
        lowrank_wagv_in,
        (x_w, x_a, x_g, x_v),
        (w1, a1, g1, v1_t),
        strict=True,
    ):
        assert torch.allclose(
            output.float(), source.float() @ projection.float().t(), atol=0.04, rtol=0.04
        )

    w2, a2, g2 = [torch.randn(output_channels, rank, device=device, dtype=torch.float16) for _ in range(3)]
    lowrank_out = infer_tmix_lowrank_out_forward_varlen(*lowrank_in, w2, a2, g2)
    assert torch.allclose(lowrank_out[0].float(), torch.tanh(lowrank_in[0].float()) @ w2.float().t(), atol=0.04, rtol=0.04)
    assert torch.allclose(lowrank_out[1].float(), lowrank_in[1].float() @ a2.float().t(), atol=0.04, rtol=0.04)
    assert torch.allclose(lowrank_out[2].float(), torch.sigmoid(lowrank_in[2].float()) @ g2.float().t(), atol=0.04, rtol=0.04)

    v1 = torch.randn(rows, rank, device=device, dtype=torch.float16)
    v2 = torch.randn(output_channels, rank, device=device, dtype=torch.float16)
    v = torch.randn(rows, output_channels, device=device, dtype=torch.float16)
    v_first = torch.randn_like(v)
    v0 = torch.randn(output_channels, device=device, dtype=torch.float16)
    lowrank_vres = infer_tmix_lowrank_vres_forward_varlen(
        *lowrank_in, v1, w2, a2, g2, v2, v, v_first, v0
    )
    expected_v = v.float() + (v_first.float() - v.float()) * torch.sigmoid(v1.float() @ v2.float().t() + v0.float())
    assert torch.allclose(lowrank_vres[3].float(), expected_v, atol=0.05, rtol=0.05)

    # These rows are outside the upstream fused linear_t windows.  The
    # caller must therefore select the exact Albatross large-rank linear
    # body, including the standalone activation helpers for rank-out gates.
    large_in = torch.randn(9, channels, device=device, dtype=torch.float16)
    large_in_weight_t = torch.randn(rank_n, channels, device=device, dtype=torch.float16)
    large_in_output = infer_tmix_linear_rank_in_forward_varlen(
        large_in, weight_t=large_in_weight_t
    )
    assert torch.allclose(
        large_in_output.float(), large_in.float() @ large_in_weight_t.float().t(),
        atol=0.04, rtol=0.04
    )

    small_in = torch.randn(7, channels, device=device, dtype=torch.float16)
    small_in_output = infer_tmix_linear_rank_in_forward_varlen(
        small_in, weight_t=large_in_weight_t
    )
    assert torch.allclose(
        small_in_output.float(), small_in.float() @ large_in_weight_t.float().t(),
        atol=0.04, rtol=0.04
    )

    large_out = torch.randn(5, rank_k, device=device, dtype=torch.float16)
    large_out_weight_t = torch.randn(output_channels, rank_k, device=device, dtype=torch.float16)
    large_out_output = infer_tmix_linear_rank_out_forward_varlen(
        large_out, weight_t=large_out_weight_t
    )
    assert torch.allclose(
        large_out_output.float(), large_out.float() @ large_out_weight_t.float().t(),
        atol=0.04, rtol=0.04
    )
    large_out_tanh = infer_tmix_linear_rank_out_tanh_forward_varlen(
        large_out, weight_t=large_out_weight_t
    )
    assert torch.allclose(
        large_out_tanh.float(),
        torch.tanh(large_out.float()) @ large_out_weight_t.float().t(),
        atol=0.04, rtol=0.04
    )
    large_out_sigmoid = infer_tmix_linear_rank_out_sigmoid_forward_varlen(
        large_out, weight_t=large_out_weight_t
    )
    assert torch.allclose(
        large_out_sigmoid.float(),
        torch.sigmoid(large_out.float()) @ large_out_weight_t.float().t(),
        atol=0.04, rtol=0.04
    )

    small_out = torch.randn(4, rank_k, device=device, dtype=torch.float16)
    small_out_output = infer_tmix_linear_rank_out_forward_varlen(
        small_out, weight_t=large_out_weight_t
    )
    assert torch.allclose(
        small_out_output.float(), small_out.float() @ large_out_weight_t.float().t(),
        atol=0.04, rtol=0.04
    )
    small_out_tanh = infer_tmix_linear_rank_out_tanh_forward_varlen(
        small_out, weight_t=large_out_weight_t
    )
    assert torch.allclose(
        small_out_tanh.float(),
        torch.tanh(small_out.float()) @ large_out_weight_t.float().t(),
        atol=0.04, rtol=0.04
    )
    small_out_sigmoid = infer_tmix_linear_rank_out_sigmoid_forward_varlen(
        small_out, weight_t=large_out_weight_t
    )
    assert torch.allclose(
        small_out_sigmoid.float(),
        torch.sigmoid(small_out.float()) @ large_out_weight_t.float().t(),
        atol=0.04, rtol=0.04
    )

    # Both layouts are supplied for the canonical C=4096 tuned table.  The
    # input table selects the original-layout Lt entry for (rank=128, rows=8),
    # while the output table selects the runtime-layout Lt entry for the same
    # rank/row pair.  This checks the automatic caller dispatch rather than a
    # forced algorithm binding.
    table_in = torch.randn(8, 4096, device=device, dtype=torch.float16)
    table_in_weight = torch.randn(4096, 128, device=device, dtype=torch.float16)
    table_in_weight_t = table_in_weight.t().contiguous()
    table_in_output = infer_tmix_linear_rank_in_forward_varlen(
        table_in, weight=table_in_weight, weight_t=table_in_weight_t
    )
    assert torch.allclose(
        table_in_output.float(), table_in.float() @ table_in_weight.float(),
        atol=0.08, rtol=0.08
    )

    table_out = torch.randn(8, 128, device=device, dtype=torch.float16)
    table_out_weight = torch.randn(128, 4096, device=device, dtype=torch.float16)
    table_out_weight_t = table_out_weight.t().contiguous()
    table_out_output = infer_tmix_linear_rank_out_forward_varlen(
        table_out, weight=table_out_weight, weight_t=table_out_weight_t
    )
    assert torch.allclose(
        table_out_output.float(), table_out.float() @ table_out_weight.float(),
        atol=0.08, rtol=0.08
    )
