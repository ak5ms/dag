from pathlib import Path
import math
import shutil
from types import SimpleNamespace
import sys
import threading
import time

import numpy as np
import pytest

from flows.riskminer import CppStreamCandidateEvaluator, CppStreamPoolEvaluator, build_ridge_pool_score_formula, halflife_to_span
from trading_dsl_engine.base.dsl import div, sub, var


def _require_native_compiler():
    if sys.platform == "win32" or shutil.which("g++") is None:
        pytest.skip("cpp_stream native test requires POSIX and g++")


def _available_cpus() -> int:
    try:
        return len(__import__("os").sched_getaffinity(0))
    except AttributeError:
        return __import__("os").cpu_count() or 1


def test_cpp_stream_batch_scores_match_shift_row_sum_mean_over_std(tmp_path: Path):
    _require_native_compiler()
    rows, instruments = 96, 4
    rng = np.random.default_rng(42)
    soft_side = rng.uniform(-1.0, 1.0, (rows, instruments))
    open_ = rng.lognormal(4.0, 0.02, (rows, instruments))
    close = open_ * (1.0 + rng.normal(0.0, 0.002, (rows, instruments)))
    returns = 0.0008 * np.vstack([np.zeros((1, instruments)), soft_side[:-1]]) + rng.normal(0.0, 0.0002, (rows, instruments))
    candidates = (var("soft_side_wavg"), div(sub(var("close"), var("open")), var("open")))
    evaluator = CppStreamCandidateEvaluator(
        {"soft_side_wavg": soft_side, "open": open_, "close": close, "roll_rets": returns},
        n_instruments=instruments, work_dir=tmp_path, batch_size=8,
    )
    actual = np.array(list(evaluator.evaluate(candidates).values()), dtype=np.float64)
    signal = np.stack((soft_side, (close - open_) / open_), axis=-1)
    shifted = np.full_like(signal, np.nan)
    shifted[1:] = signal[:-1]
    pnl = np.nansum(shifted * returns[..., None], axis=1)
    expected = pnl.mean(axis=0) / pnl.std(axis=0, ddof=0)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
    batch = evaluator.summary.batches[0]
    assert evaluator.summary.finite == 2
    assert "cpp_stream" in batch.runtime_type
    assert batch.output_shape in {(2,), (1, 2)}
    assert batch.requested_threads == 0
    assert batch.actual_threads == 1


@pytest.mark.skipif(_available_cpus() < 2, reason="requires at least two CPUs")
def test_independent_candidate_batches_use_bounded_parallel_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import trading_dsl_engine.cpp_stream as cpp_stream_module
    active = 0
    peak = 0
    lock = threading.Lock()

    class FakeRuntime:
        library_path = tmp_path / "fake-runtime.so"
        def run(self, *, out_path, threads):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.05)
                np.asarray([1.0], dtype=np.float64).tofile(out_path)
            finally:
                with lock:
                    active -= 1
            return SimpleNamespace(output_path=Path(out_path), output_shape=(1,), seconds=0.05, threads=1)

    monkeypatch.setattr(cpp_stream_module, "compile_formula", lambda *args, **kwargs: FakeRuntime())
    rows, instruments = 8, 3
    x = np.ones((rows, instruments), dtype=np.float64)
    returns = np.full((rows, instruments), 0.01, dtype=np.float64)
    candidates = (var("x"), var("x") + 1.0, var("x") * 2.0, var("x") - 3.0)
    evaluator = CppStreamCandidateEvaluator(
        {"x": x, "roll_rets": returns}, n_instruments=instruments,
        work_dir=tmp_path, batch_size=1, workers=2,
    )
    scores = evaluator.evaluate(candidates)
    assert len(scores) == 4
    assert all(value == 1.0 for value in scores.values())
    assert peak >= 2
    assert evaluator.summary.peak_batch_workers == 2
    assert len(evaluator.summary.batches) == 4
    assert all(batch.requested_threads == 1 for batch in evaluator.summary.batches)
    assert all(batch.actual_threads == 1 for batch in evaluator.summary.batches)


def test_halflife_conversion_reconstructs_requested_decay():
    halflife = 1440.0 * 5.0
    span = halflife_to_span(halflife)
    alpha = 2.0 / (span + 1.0)
    assert math.isclose((1.0 - alpha) ** halflife, 0.5, rel_tol=2e-12, abs_tol=2e-12)


def test_pool_formula_contains_ridge_beta_and_final_sharpe():
    formula = build_ridge_pool_score_formula((var("soft_side_wavg"), div(sub(var("close"), var("open")), var("open"))))
    text = repr(formula)
    assert all(name in text for name in ("Ridge", "get_beta", "einsum", "mean", "std"))


def test_small_native_ridge_pool_executes(tmp_path: Path):
    _require_native_compiler()
    rows, instruments = 384, 3
    rng = np.random.default_rng(9)
    side = rng.uniform(-1.0, 1.0, (rows, instruments))
    open_ = rng.lognormal(4.0, 0.02, (rows, instruments))
    close = open_ * (1.0 + rng.normal(0.0, 0.001, (rows, instruments)))
    lagged = np.vstack([np.zeros((1, instruments)), side[:-1]])
    returns = 0.0004 * lagged + rng.normal(0.0, 0.0003, (rows, instruments))
    sources = {
        "soft_side_wavg": side, "open": open_, "close": close, "roll_rets": returns,
        "hs": np.full((rows, instruments), 0.0005),
        "vol": np.full((rows, instruments), 0.01),
        "is_tradable": np.ones((rows, instruments)),
    }
    evaluator = CppStreamPoolEvaluator(sources, n_instruments=instruments, work_dir=tmp_path)
    result = evaluator.evaluate(
        (var("soft_side_wavg"), div(sub(var("close"), var("open")), var("open"))),
        ridge_halflife=64.0, risk_halflife=64.0,
    )
    assert isinstance(result.score, float)
    assert "cpp_stream" in result.runtime_type
    assert result.alpha_count == 2
    assert np.isfinite(result.score)
