# SPDX-License-Identifier: MIT

"""Stable row contract for the Albatross-shaped kernel benchmark."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Final

ALBATROSS_BT_MATRIX: Final[tuple[tuple[int, int], ...]] = (
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

ALBATROSS_ROW_FIELDS: Final[tuple[str, ...]] = (
    "label",
    "B",
    "T",
    "iters",
    "p10_ms",
    "p50_ms",
    "p90_ms",
    "tok_s_p50",
)


@dataclass(frozen=True, slots=True)
class AlbatrossBenchmarkRow:
    """One measured B/T row with the exact public field order."""

    label: str
    B: int
    T: int
    iters: int
    p10_ms: float
    p50_ms: float
    p90_ms: float
    tok_s_p50: float

    def as_dict(self) -> dict[str, str | int | float]:
        """Return a serialization mapping with the stable field order."""

        return asdict(self)


def percentile(samples: tuple[float, ...] | list[float], quantile: float) -> float:
    """Return NumPy-compatible linearly interpolated sample percentile."""

    if not samples:
        raise ValueError("at least one latency sample is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if any(not isfinite(sample) or sample <= 0.0 for sample in samples):
        raise ValueError("latency samples must be finite and positive")

    ordered = sorted(float(sample) for sample in samples)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_samples(
    *,
    label: str,
    batch_size: int,
    token_count: int,
    samples_ms: tuple[float, ...] | list[float],
) -> AlbatrossBenchmarkRow:
    """Summarize raw single-operator samples using the fixed contract."""

    if not label:
        raise ValueError("label must be non-empty")
    if batch_size <= 0 or token_count <= 0:
        raise ValueError("B and T must be positive")
    p50_ms = percentile(samples_ms, 0.50)
    return AlbatrossBenchmarkRow(
        label=label,
        B=batch_size,
        T=token_count,
        iters=len(samples_ms),
        p10_ms=percentile(samples_ms, 0.10),
        p50_ms=p50_ms,
        p90_ms=percentile(samples_ms, 0.90),
        tok_s_p50=batch_size * token_count * 1000.0 / p50_ms,
    )


def format_result(row: Mapping[str, object]) -> str:
    """Format one benchmark row as the stable human RESULT line."""

    missing = tuple(field for field in ALBATROSS_ROW_FIELDS if field not in row)
    if missing:
        raise ValueError(f"benchmark row is missing RESULT fields: {missing}")

    def metric(field: str) -> str:
        return str(round(float(row[field]), 6))

    return (
        f"RESULT B={row['B']} T={row['T']} iters={row['iters']} "
        f"p10_ms={metric('p10_ms')} p50_ms={metric('p50_ms')} "
        f"p90_ms={metric('p90_ms')} tok_s_p50={metric('tok_s_p50')} "
        f"label={row['label']}"
    )
