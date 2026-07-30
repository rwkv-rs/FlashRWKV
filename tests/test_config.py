# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from flash_rwkv.config import (
    ChunkConfig,
    ChunkTuningKey,
    enumerate_chunk_configs,
    select_algorithm,
    select_chunk_config,
    sequence_length_bucket,
)


def test_candidate_space_is_complete_and_unique() -> None:
    candidates = enumerate_chunk_configs()

    assert len(candidates) == 27
    assert len({candidate.identifier for candidate in candidates}) == 27
    assert {candidate.chunk_size for candidate in candidates} == {16, 32, 64}
    assert {
        (candidate.build_warps, candidate.stages)
        for candidate in candidates
    } == {(2, 1), (4, 1), (4, 2)}
    assert {candidate.state_tile for candidate in candidates} == {16, 32, 64}


@pytest.mark.parametrize(
    ("length", "bucket"),
    [
        (1, "decode"),
        (2, "short"),
        (16, "short"),
        (17, "medium"),
        (64, "medium"),
        (65, "long"),
        (256, "long"),
        (257, "very_long"),
    ],
)
def test_sequence_length_buckets(length: int, bucket: str) -> None:
    assert sequence_length_bucket(length) == bucket


def test_missing_cache_key_uses_verified_conservative_fallback() -> None:
    key = ChunkTuningKey(
        architecture="sm00",
        dtype="float16",
        mode="fp32io16",
        layout="packed",
        length_bucket="long",
    )
    selection = select_chunk_config(key)

    assert selection.source == "conservative_fallback"
    assert selection.config == ChunkConfig(16, 2, 1, 64)


@pytest.mark.parametrize(
    ("dtype", "layout", "bucket", "expected"),
    [
        ("float16", "fixed", "medium", ChunkConfig(16, 4, 1, 32)),
        ("bfloat16", "fixed", "long", ChunkConfig(64, 4, 1, 32)),
        ("float16", "packed", "medium", ChunkConfig(32, 4, 1, 32)),
        ("bfloat16", "packed", "very_long", ChunkConfig(64, 4, 1, 64)),
    ],
)
def test_canonical_sm120_keys_use_versioned_cache(
    dtype: str,
    layout: str,
    bucket: str,
    expected: ChunkConfig,
) -> None:
    key = ChunkTuningKey(
        architecture="sm120",
        dtype=dtype,
        mode="fp32io16",
        layout=layout,
        length_bucket=bucket,
    )
    selection = select_chunk_config(key)

    assert selection.source == "versioned_cache"
    assert selection.config == expected


@pytest.mark.parametrize(
    ("requested", "mode", "length", "expected"),
    [
        ("recurrent", "fp32io16", 4096, "recurrent"),
        ("chunk", "fp32io16", 1, "chunk"),
        ("auto", "fp16", 4096, "recurrent"),
        ("auto", "fp32io16", 16, "recurrent"),
        ("auto", "fp32io16", 17, "chunk"),
    ],
)
def test_family_dispatch_is_independent_from_chunk_config(
    requested: str,
    mode: str,
    length: int,
    expected: str,
) -> None:
    assert (
        select_algorithm(
            requested,
            mode=mode,
            max_sequence_length=length,
        )
        == expected
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (8, 2, 1, 64),
        (16, 3, 1, 64),
        (16, 2, 2, 64),
        (16, 2, 1, 8),
    ],
)
def test_invalid_chunk_configs_fail_closed(
    arguments: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError):
        ChunkConfig(*arguments)
