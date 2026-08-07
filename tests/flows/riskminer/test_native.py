from pathlib import Path
import math
import shutil
import sys

import numpy as np
import pytest

from flows.riskminer import (
    CppStreamCandidateEvaluator,
    CppStreamPoolEvaluator,
    build_ridge_pool_score_formula,
    halflife_to_span,
)
from trading_dsl_engine.base.dsl import div, sub, var


def _require_native_compiler():
    if sys.platform == "win32" or shutil.which("g++") is None:
        pytest.skip("cpp_stream native test requires POSIX and g++")


def test_cpp_stream_batch_scores_match_shift_row_sum_mean_over_std(
    tmp_path: Path,
):
    _require_native_compiler()
    rows, instruments = 96, 4
    rng = np.random.default_rng(42)
    soft_side = rng.uniform(-1.0, 1.0, (rows, instruments))
    open_ = rng.lognormal(4.0, 0.02, (rows, instruments))
    close = open_ * (1.0 + rng.normal(0.0, 0.002, (rows, instruments)))
    returns = (
        0.0008 * np.vstack([np.zeros((1, instruments)), soft_side[:-1]])
        + rng.normal(0.0, 0.0002, (rows, instruments))
    )

    candidates = (
        var("soft_side_wavg"),
        div(sub(var("close"), var("open")), var("open")),
    )
    evaluator = CppStreamCandidateEvaluator(
        {
            "soft_side_wavg": soft_side,
            "open": open_,
            "close": close,
            "roll_rets": returns,
        },
        n_instruments=instruments,
        work_dir=tmp_path,
        batch_size=8,
    )
    scores = evaluator.evaluate(candidates)
    actual = np.array(list(scores.values()), dtype=np.float64)

    signal = np.stack(
        (soft_side, (close - open_) / open_),
        axis=-1,
    )
    shifted = np.full_like(signal, np.nan)
    shifted[1:] = signal[:-1]
    pnl = np.nansum(shifted * returns[..., None], axis=1)
    expected = pnl.mean(axis=0) / pnl.std(axis=0, ddof=0)

    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
    assert evaluator.summary.finite == 2
    assert evaluator.summary.batches
    batch = evaluator.summary.batches[0]
    assert "cpp_stream" in batch.runtime_type
    assert batch.output_shape in {(2,), (1, 2)}


def test_halflife_conversion_reconstructs_requested_decay():
    halflife = 1440.0 * 5.0
    span = halflife_to_span(halflife)
    alpha = 2.0 / (span + 1.0)
    assert math.isclose(
        (1.0 - alpha) ** halflife,
        0.5,
        rel_tol=2e-12,
        abs_tol=2e-12,
    )


def test_pool_formula_contains_ridge_beta_and_final_sharpe():
    formula = build_ridge_pool_score_formula(
        (
            var("soft_side_wavg"),
            div(sub(var("close"), var("open")), var("open")),
        )
    )
    text = repr(formula)
    assert "Ridge" in text
    assert "get_beta" in text
    assert "einsum" in text
    assert "mean" in text
    assert "std" in text


def test_small_native_ridge_pool_executes(tmp_path: Path):
    _require_native_compiler()
    rows, instruments = 384, 3
    rng = np.random.default_rng(9)
    side = rng.uniform(-1.0, 1.0, (rows, instruments))
    open_ = rng.lognormal(4.0, 0.02, (rows, instruments))
    close = open_ * (1.0 + rng.normal(0.0, 0.001, (rows, instruments)))
    lagged = np.vstack([np.zeros((1, instruments)), side[:-1]])
    returns = 0.0004 * lagged + rng.normal(
        0.0,
        0.0003,
        (rows, instruments),
    )
    sources = {
        "soft_side_wavg": side,
        "open": open_,
        "close": close,
        "roll_rets": returns,
        "hs": np.full((rows, instruments), 0.0005),
        "vol": np.full((rows, instruments), 0.01),
        "is_tradable": np.ones((rows, instruments)),
    }
    evaluator = CppStreamPoolEvaluator(
        sources,
        n_instruments=instruments,
        work_dir=tmp_path,
    )
    result = evaluator.evaluate(
        (
            var("soft_side_wavg"),
            div(sub(var("close"), var("open")), var("open")),
        ),
        ridge_halflife=64.0,
        risk_halflife=64.0,
    )
    assert isinstance(result.score, float)
    assert "cpp_stream" in result.runtime_type
    assert result.alpha_count == 2
    assert np.isfinite(result.score)
