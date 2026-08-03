# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.infer.wkv7 import benchmark_fused_decay_recurrent
from benchmarks.infer.wkv7.benchmark_fused_decay_recurrent import CASES
from flash_rwkv.benchmark_contract import (
    ALBATROSS_BT_MATRIX,
    ALBATROSS_ROW_FIELDS,
    format_result,
    percentile,
    summarize_samples,
)


def test_albatross_bt_matrix_is_exact_and_unique() -> None:
    assert ALBATROSS_BT_MATRIX == (
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (1, 16),
        (1, 32),
        (1, 64),
        (1, 128),
        (1, 256),
        (2, 1),
        (4, 1),
        (8, 1),
        (16, 1),
        (32, 1),
        (64, 1),
        (128, 1),
        (256, 1),
        (2, 2),
        (4, 4),
        (8, 8),
        (16, 16),
    )
    assert len(ALBATROSS_BT_MATRIX) == 21
    assert len(set(ALBATROSS_BT_MATRIX)) == 21


def test_summary_has_exact_fields_and_throughput_definition() -> None:
    row = summarize_samples(
        label="B2T4",
        batch_size=2,
        token_count=4,
        samples_ms=[1.0, 2.0, 3.0],
    )

    assert tuple(row.as_dict()) == ALBATROSS_ROW_FIELDS
    assert row.iters == 3
    assert row.p10_ms == pytest.approx(1.2)
    assert row.p50_ms == pytest.approx(2.0)
    assert row.p90_ms == pytest.approx(2.8)
    assert row.tok_s_p50 == pytest.approx(2 * 4 * 1000.0 / 2.0)
    assert format_result(row.as_dict()) == (
        "RESULT B=2 T=4 iters=3 p10_ms=1.2 p50_ms=2.0 "
        "p90_ms=2.8 tok_s_p50=4000.0 label=B2T4"
    )


def test_result_formatter_rejects_incomplete_rows() -> None:
    with pytest.raises(ValueError, match="missing RESULT fields"):
        format_result({"B": 1, "T": 1})


def test_fused_decay_recurrent_benchmark_covers_required_product_cases() -> None:
    assert CASES["b1_t1"] == (1,)
    assert CASES["b320_t1"] == (1,) * 320
    assert CASES["b1_t128"] == (128,)
    assert CASES["packed_b320_t16"] == (16,) * 320
    assert len(CASES["ragged_b320_t1_to_t16"]) == 320
    assert len(set(CASES["ragged_b320_t1_to_t16"])) > 1


def test_fused_decay_recurrent_benchmark_keeps_e2e_and_wkv_only_separate() -> None:
    source = benchmark_fused_decay_recurrent.__file__
    assert source is not None
    contents = Path(source).read_text(encoding="utf-8")
    assert '"A": "unfused_correct_product"' in contents
    assert '"B": "fused_raw_product"' in contents
    assert "inputs.decay_logits + inputs.decay_bias" in contents
    assert "recurrent_fp32_from_decay_logits" in contents
    assert "precomputed_log_decay_is_diagnostic_only" in contents
    assert "fp32io16" in contents
    assert "fp16" in contents
    assert "validate_recurrent_metadata_kernel" in contents
    assert "cuda_kernel_count" in contents


def test_statetune_benchmark_compares_unfused_and_fused_training_paths() -> None:
    path = (
        Path(__file__).parents[1]
        / "benchmarks/statetune/wkv7/benchmark_statetune_recurrent_fp32io16_backward.py"
    )
    contents = path.read_text(encoding="utf-8")
    assert '"A": "unfused_correct_product"' in contents
    assert '"B": "public_raw_fused_recurrent"' in contents
    assert "_unfused_correct_statetune" in contents
    assert "statetune_recurrent_fp32io16_forward" in contents
    assert "fused_speedup_over_unfused" in contents
    assert "timed_transform_materialization_bytes" in contents
    assert "cuda_kernel_count" in contents


@pytest.mark.parametrize(
    ("samples", "quantile"),
    [
        ([], 0.5),
        ([0.0], 0.5),
        ([float("inf")], 0.5),
        ([1.0], -0.1),
        ([1.0], 1.1),
    ],
)
def test_percentile_rejects_invalid_measurements(
    samples: list[float],
    quantile: float,
) -> None:
    with pytest.raises(ValueError):
        percentile(samples, quantile)
