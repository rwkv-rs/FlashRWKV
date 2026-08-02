# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

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
