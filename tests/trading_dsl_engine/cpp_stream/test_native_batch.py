from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trading_dsl_engine.base.dsl import var
from trading_dsl_engine.cpp_stream import compile_formula, run_many


def _as_tuple(value):
    return value if isinstance(value, tuple) else (value,)


def test_native_batch_runs_independent_multi_output_dags(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "TRADING_DSL_ENGINE_CPP_STREAM_CACHE",
        str(tmp_path / "cache"),
    )
    monkeypatch.setenv("TRADING_DSL_ENGINE_CPP_PCH", "0")
    monkeypatch.setenv("TRADING_DSL_ENGINE_CPP_LTO", "0")

    rows, instruments = 512, 5
    rng = np.random.default_rng(42)
    data = {"x": rng.normal(size=(rows, instruments))}
    x = var("x")
    runtimes = (
        compile_formula(
            [x + 1.0, (x + 1.0).mean(axis=0)],
            data,
            n_instruments=instruments,
        ),
        compile_formula(
            [x * 2.0, (x * 2.0).std(axis=0)],
            data,
            n_instruments=instruments,
        ),
    )

    serial = tuple(
        runtime.run(out_path=tmp_path / f"serial_{index}.npy", threads=1)
        for index, runtime in enumerate(runtimes)
    )
    batch = run_many(
        runtimes,
        out_paths=[
            tmp_path / "native_0.npy",
            tmp_path / "native_1.npy",
        ],
        workers=2,
        threads_per_runtime=1,
    )

    assert len(batch.results) == 2
    assert 1 <= batch.workers <= 2
    assert batch.wall_seconds >= 0.0
    assert batch.native_seconds_sum >= 0.0
    assert batch.effective_concurrency >= 0.0
    for expected_result, actual_result in zip(serial, batch.results):
        expected = _as_tuple(expected_result.load(mmap_mode=None))
        actual = _as_tuple(actual_result.load(mmap_mode=None))
        assert len(expected) == len(actual) == 2
        for expected_value, actual_value in zip(expected, actual):
            np.testing.assert_allclose(
                actual_value,
                expected_value,
                rtol=1e-13,
                atol=1e-13,
                equal_nan=True,
            )


def test_native_batch_validates_output_path_count(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "TRADING_DSL_ENGINE_CPP_STREAM_CACHE",
        str(tmp_path / "cache"),
    )
    monkeypatch.setenv("TRADING_DSL_ENGINE_CPP_PCH", "0")
    monkeypatch.setenv("TRADING_DSL_ENGINE_CPP_LTO", "0")
    data = {"x": np.arange(16, dtype=np.float64).reshape(8, 2)}
    runtime = compile_formula(var("x") + 1.0, data, n_instruments=2)
    with pytest.raises(ValueError, match="out_paths length"):
        run_many((runtime,), out_paths=())
