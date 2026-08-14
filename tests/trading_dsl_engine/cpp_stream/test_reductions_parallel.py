from __future__ import annotations

from pathlib import Path

import numpy as np

from trading_dsl_engine.base.dsl import cat, cumsum, ewm, var
from trading_dsl_engine.cpp_stream import compile_formula


def _read(result) -> np.ndarray:
    values = np.fromfile(result.output_path, dtype=np.float64)
    return values.reshape(result.output_shape or ())


def test_temporal_reduction_merges_row_worker_state(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20_000, 8))
    runtime = compile_formula(var("x").sum(axis=0), {"x": x}, n_instruments=8)
    result = runtime.run(out_path=tmp_path / "temporal_sum.bin", threads=4, pin_threads=True)
    np.testing.assert_allclose(_read(result), np.sum(x, axis=0), rtol=1e-12)
    assert runtime.parallel_plan.mode == "rows"
    assert result.parallel_mode == "rows"
    assert result.threads == 4
    assert result.output_mode == "final"
    assert result.output_rows == 1
    assert result.output_path.stat().st_size == 8 * 8


def test_row_reduction_uses_row_sharding(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100_000, 8))
    expression = var("x").sum(axis=1) + 1.0
    runtime = compile_formula(expression, {"x": x}, n_instruments=8)
    serial = runtime.run(out_path=tmp_path / "row_serial.bin", threads=1)
    parallel = runtime.run(out_path=tmp_path / "row_parallel.bin", threads=4, pin_threads=True)
    np.testing.assert_array_equal(_read(serial), _read(parallel))
    np.testing.assert_allclose(_read(parallel), np.sum(x, axis=1) + 1.0)
    assert runtime.parallel_plan.mode == "rows"
    assert parallel.threads == 4
    assert parallel.output_mode == "rows"


def test_lane_local_feature_reduction_remains_lane_parallel(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(50_000, 8))
    y = rng.normal(size=(50_000, 8))
    expression = cat(ewm(var("x"), 8), ewm(var("y"), 8)).sum(axis=2)
    runtime = compile_formula(expression, {"x": x, "y": y}, n_instruments=8)
    serial = runtime.run(out_path=tmp_path / "lane_serial.bin", threads=1)
    parallel = runtime.run(out_path=tmp_path / "lane_parallel.bin", threads=4, pin_threads=True)
    np.testing.assert_array_equal(_read(serial), _read(parallel))
    assert runtime.parallel_plan.mode == "lanes"
    assert parallel.threads == 4


def test_emit_last_merges_lane_owned_state(tmp_path: Path) -> None:
    x = np.arange(80_000, dtype=np.float64).reshape(10_000, 8)
    runtime = compile_formula(cumsum(var("x")).emit("last"), {"x": x}, n_instruments=8)
    result = runtime.run(out_path=tmp_path / "last.bin", threads=4, pin_threads=True)
    np.testing.assert_allclose(_read(result), np.cumsum(x, axis=0)[-1])
    assert runtime.parallel_plan.mode == "lanes"
    assert result.threads == 4
    assert result.output_mode == "final"
