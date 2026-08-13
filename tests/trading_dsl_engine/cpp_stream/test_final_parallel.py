from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from flows.riskminer.cpp_stream_eval import build_candidate_score_formula
from trading_dsl_engine.base.dsl import cumsum, emit, ewm, reduction, var
from trading_dsl_engine.cpp_stream import compile_formula


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _read(result) -> np.ndarray:
    values = np.fromfile(result.output_path, dtype=np.float64)
    return values.reshape(result.output_shape or ())


@pytest.mark.skipif(_available_cpus() < 2, reason="requires at least two CPUs")
def test_terminal_sum_merges_row_shards(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260813)
    x = rng.normal(size=(120_000, 9))
    x[::997, 3] = np.nan
    runtime = compile_formula(var("x").sum(axis=0), {"x": x}, n_instruments=9)
    serial = runtime.run(out_path=tmp_path / "sum_serial.bin", threads=1)
    parallel = runtime.run(out_path=tmp_path / "sum_parallel.bin", threads=4, pin_threads=True)
    assert runtime.parallel_plan.mode == "rows"
    assert "worker states are mergeable" in runtime.parallel_plan.reason
    assert serial.threads == 1
    assert parallel.threads >= 2
    np.testing.assert_allclose(_read(parallel), _read(serial), rtol=3e-13, atol=3e-13, equal_nan=True)
    np.testing.assert_allclose(_read(parallel), np.nansum(x, axis=0), rtol=3e-13, atol=3e-13, equal_nan=True)


@pytest.mark.skipif(_available_cpus() < 2, reason="requires at least two CPUs")
def test_terminal_sharpe_merges_mean_and_std_states(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(150_000, 9))
    y = rng.normal(size=(150_000, 9))
    pnl = (var("x") * var("y")).sum(axis=1)
    score = emit(pnl.mean(axis=0) / pnl.std(axis=0, ddof=0), mode="last")
    runtime = compile_formula(score, {"x": x, "y": y}, n_instruments=9)
    serial = runtime.run(out_path=tmp_path / "sharpe_serial.bin", threads=1)
    parallel = runtime.run(out_path=tmp_path / "sharpe_parallel.bin", threads=4, pin_threads=True)
    expected_pnl = np.sum(x * y, axis=1)
    expected = np.mean(expected_pnl) / np.std(expected_pnl, ddof=0)
    assert runtime.parallel_plan.mode == "rows"
    assert parallel.threads >= 2
    np.testing.assert_allclose(_read(serial), expected, rtol=4e-13, atol=4e-13)
    np.testing.assert_allclose(_read(parallel), expected, rtol=4e-13, atol=4e-13)


@pytest.mark.skipif(_available_cpus() < 2, reason="requires at least two CPUs")
def test_parallel_final_suffix_reads_the_global_last_input_row(tmp_path: Path) -> None:
    rows, n = 80_003, 9
    rng = np.random.default_rng(8)
    x = rng.normal(size=(rows, n))
    formula = emit(var("x").sum(axis=0) + var("x"), mode="last")
    runtime = compile_formula(formula, {"x": x}, n_instruments=n)
    result = runtime.run(out_path=tmp_path / "last_input.bin", threads=4, pin_threads=True)
    assert runtime.parallel_plan.mode == "rows"
    assert result.threads >= 2
    np.testing.assert_allclose(_read(result), np.sum(x, axis=0) + x[-1], rtol=4e-13, atol=4e-13)


@pytest.mark.skipif(_available_cpus() < 2, reason="requires at least two CPUs")
def test_lane_local_temporal_sharpe_merges_instrument_slices(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(100_000, 9))
    smoothed = ewm(var("x"), 21)
    score = emit(smoothed.mean(axis=0) / smoothed.std(axis=0, ddof=0), mode="last")
    runtime = compile_formula(score, {"x": x}, n_instruments=9)
    serial = runtime.run(out_path=tmp_path / "lane_serial.bin", threads=1)
    parallel = runtime.run(out_path=tmp_path / "lane_parallel.bin", threads=4, pin_threads=True)
    assert runtime.parallel_plan.mode == "lanes"
    assert "terminal worker state is mergeable" in runtime.parallel_plan.reason
    assert parallel.threads >= 2
    np.testing.assert_allclose(_read(parallel), _read(serial), rtol=5e-13, atol=5e-13, equal_nan=True)


@pytest.mark.skipif(_available_cpus() < 2, reason="requires at least two CPUs")
def test_lane_merged_final_suffix_may_reduce_to_scalar(tmp_path: Path) -> None:
    rows, n = 75_000, 9
    rng = np.random.default_rng(91)
    x = rng.normal(size=(rows, n))
    lane_mean = reduction("mean", ewm(var("x"), 21), axis=0)
    formula = emit(reduction("mean", lane_mean, axis=1), mode="last")
    runtime = compile_formula(formula, {"x": x}, n_instruments=n)
    serial = runtime.run(out_path=tmp_path / "lane_scalar_serial.bin", threads=1)
    parallel = runtime.run(out_path=tmp_path / "lane_scalar_parallel.bin", threads=4, pin_threads=True)
    assert runtime.parallel_plan.mode == "lanes"
    assert runtime.plan.output_shape == ()
    assert parallel.threads >= 2
    np.testing.assert_allclose(_read(parallel), _read(serial), rtol=5e-13, atol=5e-13, equal_nan=True)


@pytest.mark.skipif(_available_cpus() < 2, reason="requires at least two CPUs")
def test_emit_last_merges_lane_owned_temporal_values(tmp_path: Path) -> None:
    rows, n = 60_000, 9
    rng = np.random.default_rng(10)
    x = rng.normal(size=(rows, n))
    runtime = compile_formula(cumsum(var("x")).emit("last"), {"x": x}, n_instruments=n)
    result = runtime.run(out_path=tmp_path / "emit_last.bin", threads=4, pin_threads=True)
    assert runtime.parallel_plan.mode == "lanes"
    assert result.threads >= 2
    np.testing.assert_allclose(_read(result), np.cumsum(x, axis=0)[-1], rtol=5e-13, atol=5e-13)


def test_stateful_cross_sectional_candidate_sharpe_stays_serial(tmp_path: Path) -> None:
    rows, n = 4096, 9
    rng = np.random.default_rng(11)
    alpha = rng.normal(size=(rows, n))
    roll_rets = rng.normal(scale=0.01, size=(rows, n))
    formula = build_candidate_score_formula([var("alpha")])
    runtime = compile_formula(formula, {"alpha": alpha, "roll_rets": roll_rets}, n_instruments=n)
    result = runtime.run(out_path=tmp_path / "candidate_score.bin", threads=4)
    assert runtime.parallel_plan.mode == "serial"
    assert result.threads == 1
    assert "(shift)" in runtime.parallel_plan.reason
    # The row sum has already collapsed the instrument axis, so no lane-owned
    # terminal state remains for the final mean/std pair to merge.
    assert "no terminal state can be merged by instrument lane" in runtime.parallel_plan.reason
    shifted = np.empty_like(alpha)
    shifted[0] = np.nan
    shifted[1:] = alpha[:-1]
    pnl = np.sum(np.nan_to_num(shifted, nan=0.0) * roll_rets, axis=1)
    expected = np.mean(pnl) / np.std(pnl, ddof=0)
    np.testing.assert_allclose(_read(result), expected, rtol=5e-13, atol=5e-13)


def test_fused_gp_expression_contributes_to_automatic_work_score() -> None:
    rows, n = 8, 9
    x = np.ones((rows, n), dtype=np.float64)
    y = np.full((rows, n), 2.0, dtype=np.float64)
    simple = compile_formula("x + y", {"x": x, "y": y}, n_instruments=n)
    deep = compile_formula("((x + y) * (x - y) + (x * y + 3.0)) / (abs(x) + 1.0)", {"x": x, "y": y}, n_instruments=n)
    assert deep.parallel_plan.mode == "rows"
    assert deep.parallel_plan.work_score > simple.parallel_plan.work_score
