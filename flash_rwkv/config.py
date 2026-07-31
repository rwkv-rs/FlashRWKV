# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cache
import json
from pathlib import Path
from typing import Any

import torch


_CACHE_PATH = Path(__file__).with_name("chunk-tuning-v1.json")
_CHUNK_SIZES = (16, 32, 64)
_BUILD_PIPELINES = ((2, 1), (4, 1), (4, 2))
_STATE_TILES = (16, 32, 64)
TRAINING_CHECKPOINT_CHUNK_SIZE = 16


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int
    build_warps: int
    stages: int
    state_tile: int

    def __post_init__(self) -> None:
        if isinstance(self.chunk_size, bool) or self.chunk_size not in _CHUNK_SIZES:
            raise ValueError("chunk_size must be one of 16, 32, or 64")
        if (self.build_warps, self.stages) not in _BUILD_PIPELINES:
            raise ValueError(
                "(build_warps, stages) must be one of "
                "(2, 1), (4, 1), or (4, 2)"
            )
        if self.state_tile not in _STATE_TILES:
            raise ValueError("state_tile must be one of 16, 32, or 64")

    @property
    def identifier(self) -> str:
        return (
            f"c{self.chunk_size}-w{self.build_warps}-"
            f"s{self.stages}-t{self.state_tile}"
        )


@dataclass(frozen=True)
class ChunkTuningKey:
    architecture: str
    dtype: str
    mode: str
    layout: str
    length_bucket: str

    @property
    def identifier(self) -> str:
        return "|".join(
            (
                self.architecture,
                self.dtype,
                self.mode,
                self.layout,
                self.length_bucket,
            )
        )


@dataclass(frozen=True)
class SelectedChunkConfig:
    config: ChunkConfig
    key: ChunkTuningKey
    source: str


def enumerate_chunk_configs() -> tuple[ChunkConfig, ...]:
    return tuple(
        ChunkConfig(
            chunk_size=chunk_size,
            build_warps=build_warps,
            stages=stages,
            state_tile=state_tile,
        )
        for chunk_size in _CHUNK_SIZES
        for build_warps, stages in _BUILD_PIPELINES
        for state_tile in _STATE_TILES
    )


def training_chunk_config(config: ChunkConfig) -> ChunkConfig:
    """Apply the stable RWKV-LM reverse-reconstruction checkpoint policy."""

    return ChunkConfig(
        chunk_size=TRAINING_CHECKPOINT_CHUNK_SIZE,
        build_warps=config.build_warps,
        stages=config.stages,
        state_tile=config.state_tile,
    )


def sequence_length_bucket(max_sequence_length: int) -> str:
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if max_sequence_length == 1:
        return "decode"
    if max_sequence_length <= 16:
        return "short"
    if max_sequence_length <= 64:
        return "medium"
    if max_sequence_length <= 256:
        return "long"
    return "very_long"


def chunk_tuning_key(
    tensor: torch.Tensor,
    *,
    mode: str,
    packed: bool,
    max_sequence_length: int,
) -> ChunkTuningKey:
    if not tensor.is_cuda:
        raise ValueError("chunk tuning keys require a CUDA tensor")
    major, minor = torch.cuda.get_device_capability(tensor.device)
    return ChunkTuningKey(
        architecture=f"sm{major}{minor}",
        dtype=str(tensor.dtype).removeprefix("torch."),
        mode=mode,
        layout="packed" if packed else "fixed",
        length_bucket=sequence_length_bucket(max_sequence_length),
    )


@cache
def _load_tuning_cache() -> dict[str, Any]:
    payload = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("unsupported chunk tuning cache schema")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise RuntimeError("chunk tuning cache entries must be an object")
    return entries


def select_chunk_config(
    key: ChunkTuningKey,
    *,
    chunk_size: int | None = None,
    config: ChunkConfig | None = None,
) -> SelectedChunkConfig:
    if config is not None and chunk_size is not None:
        raise ValueError("pass either chunk_size or config, not both")
    if config is not None:
        return SelectedChunkConfig(config=config, key=key, source="explicit")
    if chunk_size is not None:
        return SelectedChunkConfig(
            config=ChunkConfig(
                chunk_size=chunk_size,
                build_warps=2,
                stages=1,
                state_tile=64,
            ),
            key=key,
            source="explicit_chunk_size",
        )

    entry = _load_tuning_cache().get(key.identifier)
    if entry is not None:
        if not isinstance(entry, dict) or not isinstance(
            entry.get("config"), dict
        ):
            raise RuntimeError(
                f"invalid chunk tuning entry for {key.identifier}"
            )
        return SelectedChunkConfig(
            config=ChunkConfig(**entry["config"]),
            key=key,
            source="versioned_cache",
        )

    return SelectedChunkConfig(
        config=ChunkConfig(
            chunk_size=16,
            build_warps=2,
            stages=1,
            state_tile=64,
        ),
        key=key,
        source="conservative_fallback",
    )


def select_algorithm(
    requested: str,
    *,
    mode: str,
    max_sequence_length: int,
) -> str:
    if requested != "auto":
        return requested
    # The current materialized and factor/recompute chunk behavior cells are
    # both slower than the recurrent kernel on every correctness-gated SM120
    # profile measured so far. Keep them available for explicit experiments,
    # but do not silently dispatch production calls to a disproven crossover.
    del mode, max_sequence_length
    return "recurrent"


def config_as_dict(config: ChunkConfig) -> dict[str, int]:
    return asdict(config)
