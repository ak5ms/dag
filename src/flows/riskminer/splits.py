from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SourceSplit:
    name: str
    start: int
    stop: int
    sources: dict[str, Any]

    @property
    def rows(self) -> int:
        return self.stop - self.start


def _rows(value: Any) -> int:
    if isinstance(value, (str, Path)):
        array = np.load(value, mmap_mode="r", allow_pickle=False)
        rows = int(array.shape[0])
        del array
        return rows
    shape = tuple(getattr(value, "shape", ()))
    if not shape:
        raise ValueError(f"source has no row dimension: {type(value).__name__}")
    return int(shape[0])


def _slice(value: Any, start: int, stop: int) -> Any:
    if isinstance(value, (str, Path)):
        # The cpp_stream compiler accepts ndarray/memmap sources. Keep the
        # slice as a read-only mmap view instead of copying it.
        return np.load(value, mmap_mode="r", allow_pickle=False)[start:stop]
    return value[start:stop]


def split_sources_contiguous(
    sources: Mapping[str, Any],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[SourceSplit, SourceSplit, SourceSplit]:
    """Chronologically split every source without overlap or look-ahead."""

    if not sources:
        raise ValueError("sources cannot be empty")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train+validation fractions must be < 1")
    row_counts = {_rows(value) for value in sources.values()}
    if len(row_counts) != 1:
        raise ValueError(f"source row counts disagree: {sorted(row_counts)}")
    rows = row_counts.pop()
    train_stop = int(rows * train_fraction)
    validation_stop = train_stop + int(rows * validation_fraction)
    if train_stop <= 0 or validation_stop <= train_stop or validation_stop >= rows:
        raise ValueError("split fractions leave an empty partition")

    def make(name: str, start: int, stop: int) -> SourceSplit:
        return SourceSplit(
            name,
            start,
            stop,
            {key: _slice(value, start, stop) for key, value in sources.items()},
        )

    return (
        make("train", 0, train_stop),
        make("validation", train_stop, validation_stop),
        make("test", validation_stop, rows),
    )


__all__ = ["SourceSplit", "split_sources_contiguous"]
