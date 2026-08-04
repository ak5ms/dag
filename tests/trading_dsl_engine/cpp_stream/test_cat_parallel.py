from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from trading_dsl_engine.cpp_stream import compile_formula


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _read(path: Path, rows: int, width: int) -> np.ndarray:
    return np.asarray(
        np.memmap(path, mode="r", dtype=np.float64, shape=(rows, width))
    ).copy()


@pytest.mark.skipif(_available_cpus() < 2, reason="requires at least two CPUs")
def test_cat_root_automatically_row_parallelizes_and_matches_serial(
    tmp_path: Path,
) -> None:
    rows, n = 100_000, 9
    rng = np.random.default_rng(20260802)
    data = {
        "x1": rng.normal(size=(rows, n)),
        "x2": rng.normal(size=(rows, n)),
        "x3": rng.normal(size=(rows, n)),
    }
    runtime = compile_formula("cat(x1, x2, x3)", data, n_instruments=n)

    assert runtime.parallel_plan.mode == "rows"
    assert runtime.parallel_plan.auto_multicore
    assert "all rows are independent" in runtime.parallel_plan.reason

    serial_path = tmp_path / "cat_serial.bin"
    automatic_path = tmp_path / "cat_automatic.bin"
    serial = runtime.run(out_path=serial_path, threads=1)
    automatic = runtime.run(
        out_path=automatic_path,
        threads=0,
        pin_threads=True,
    )

    expected = np.stack((data["x1"], data["x2"], data["x3"]), axis=-1)
    np.testing.assert_array_equal(
        _read(serial_path, rows, n * 3).reshape(rows, n, 3), expected
    )
    np.testing.assert_array_equal(
        _read(automatic_path, rows, n * 3).reshape(rows, n, 3), expected
    )
    assert serial.threads == 1
    assert automatic.available_cpus >= 2
    assert automatic.threads >= 2
    assert automatic.parallel_mode == "rows"


def test_cat_inside_temporal_plan_uses_plan_lane_sharding_not_nested_tasks() -> None:
    rows, n = 32, 9
    rng = np.random.default_rng(17)
    data = {
        "x1": rng.normal(size=(rows, n)),
        "x2": rng.normal(size=(rows, n)),
        "x3": rng.normal(size=(rows, n)),
        "y": rng.normal(size=(rows, n)),
    }
    runtime = compile_formula(
        "get_preds(InstrumentBasisMean(cat(x1, x2, x3), y, 1.0, 64))",
        data,
        n_instruments=n,
    )

    assert runtime.parallel_plan.mode == "lanes"
    generated = runtime.generated_cpp.read_text()
    assert "stackdsl::CatNode" not in generated
    assert "stackdsl::FeatureList" in generated
    assert "std::thread" in generated
