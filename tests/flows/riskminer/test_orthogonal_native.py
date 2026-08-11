from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from flows.riskminer import build_cross_sectional_orthogonal_alpha
from trading_dsl_engine.base.dsl import var
from trading_dsl_engine.cpp_stream import compile_formula


def _require_native_compiler() -> None:
    if sys.platform == "win32" or shutil.which("g++") is None:
        pytest.skip("cpp_stream native test requires POSIX and g++")


def _native_residuals(tmp_path: Path, pool: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    data = {f"x{i}": value for i, value in enumerate(pool)}
    data["y"] = y
    formula = build_cross_sectional_orthogonal_alpha(
        var("y"), tuple(var(f"x{i}") for i in range(len(pool)))
    )
    text = repr(formula)
    assert "Ridge" in text and "get_residuals" in text
    assert "vol" not in text and "hs" not in text
    runtime = compile_formula(formula, data, n_instruments=y.shape[1])
    output = tmp_path / "orthogonal_residuals.bin"
    result = runtime.run(out_path=output)
    path = Path(getattr(result, "output_path", output))
    return np.fromfile(path, dtype=np.float64).reshape(y.shape)


def _pinv_reference(pool: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    expected = np.empty_like(y)
    for row in range(y.shape[0]):
        design = np.column_stack([value[row] for value in pool])
        expected[row] = y[row] - design @ np.linalg.pinv(design) @ y[row]
    return expected


def test_orthogonalization_uses_pinv_when_pool_exceeds_instruments(tmp_path: Path):
    _require_native_compiler()
    rng = np.random.default_rng(903)
    rows, instruments, features = 7, 3, 5
    pool = [rng.normal(size=(rows, instruments)) for _ in range(features)]
    y = rng.normal(size=(rows, instruments))
    actual = _native_residuals(tmp_path, pool, y)
    expected = _pinv_reference(pool, y)
    assert features > instruments
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(actual, 0.0, rtol=0.0, atol=2e-8)


def test_orthogonalization_matches_pinv_for_rank_deficient_pool(tmp_path: Path):
    _require_native_compiler()
    rng = np.random.default_rng(904)
    rows, instruments = 9, 4
    factor = rng.normal(size=(rows, instruments))
    second = rng.normal(size=(rows, instruments))
    pool = [factor, 2.0 * factor, -factor, second, factor + second]
    y = 0.3 * factor - 0.2 * second + rng.normal(size=(rows, instruments))
    actual = _native_residuals(tmp_path, pool, y)
    expected = _pinv_reference(pool, y)
    np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=3e-8)
    assert np.linalg.norm(actual) > 1e-6


def test_complete_reward_dense_native_episode_updates_pool(tmp_path: Path):
    from flows.riskminer import (
        CppStreamOrthogonalEvaluator,
        CppStreamPoolEvaluator,
        RewardDensePoolModel,
        RewardDenseRiskMCTS,
        RidgeAlphaPool,
        RiskMinerConfig,
        SearchShape,
        SemanticInfo,
        TypedRPNEnvironment,
        build_vocabulary,
    )

    _require_native_compiler()
    rows, instruments = 192, 3
    rng = np.random.default_rng(17)
    signal = rng.normal(size=(rows, instruments))
    lagged = np.vstack([np.zeros((1, instruments)), signal[:-1]])
    returns = 0.001 * lagged + rng.normal(scale=0.0002, size=(rows, instruments))
    sources = {
        "x": signal,
        "roll_rets": returns,
        "vol": np.full((rows, instruments), 0.01),
        "hs": np.full((rows, instruments), 0.0005),
        "is_tradable": np.ones((rows, instruments)),
    }
    config = RiskMinerConfig(
        max_depth=1,
        min_formula_depth=1,
        max_tokens=2,
        max_stack=1,
        simulations=1,
        rollouts_per_expansion=1,
        evaluation_batch_size=1,
        archive_size=8,
        seed=2,
    )
    vocabulary = build_vocabulary(
        terminals={
            "x": SemanticInfo(
                frozenset({"numeric", "dimensionless"}), SearchShape.ROW
            )
        },
        literals=(),
        operators=(),
    )
    environment = TypedRPNEnvironment(config=config, vocabulary=vocabulary)
    intermediate = CppStreamOrthogonalEvaluator(
        sources,
        n_instruments=instruments,
        work_dir=tmp_path / "intermediate",
        batch_size=1,
    )
    pool_evaluator = CppStreamPoolEvaluator(
        sources,
        n_instruments=instruments,
        work_dir=tmp_path / "pool",
    )
    pool = RidgeAlphaPool(
        pool_evaluator,
        capacity=2,
        min_improvement=-1e9,
        formula_kwargs={"ridge_halflife": 16.0, "risk_halflife": 16.0},
    )
    result = RewardDenseRiskMCTS(
        environment,
        RewardDensePoolModel(intermediate, pool),
        config=config,
    ).search()
    assert result.trajectories
    assert result.metrics.pool_updates == 1
    assert len(pool.entries) == 1
    assert np.isfinite(pool.score)
    assert result.trajectories[0].step_rewards[0] != 0.0


def test_native_pool_emits_one_importance_per_alpha(tmp_path: Path):
    from flows.riskminer import CppStreamPoolEvaluator

    _require_native_compiler()
    rows, instruments = 128, 3
    rng = np.random.default_rng(18)
    alphas = [rng.normal(size=(rows, instruments)) for _ in range(3)]
    returns = 0.0005 * np.vstack(
        [np.zeros((1, instruments)), alphas[0][:-1]]
    ) + rng.normal(scale=0.0002, size=(rows, instruments))
    sources = {
        **{f"x{i}": value for i, value in enumerate(alphas)},
        "roll_rets": returns,
        "vol": np.full((rows, instruments), 0.01),
        "hs": np.full((rows, instruments), 0.0005),
        "is_tradable": np.ones((rows, instruments)),
    }
    evaluator = CppStreamPoolEvaluator(
        sources, n_instruments=instruments, work_dir=tmp_path
    )
    result = evaluator.evaluate(
        tuple(var(f"x{i}") for i in range(3)),
        include_importance=True,
        ridge_halflife=16.0,
        risk_halflife=16.0,
    )
    assert len(result.coefficient_importance) == 3
    assert all(np.isfinite(value) for value in result.coefficient_importance)


def test_complete_native_pipeline_trains_policy_updates_pool_and_tests(tmp_path: Path):
    from flows.riskminer import (
        RiskMinerConfig,
        SearchShape,
        SemanticInfo,
        train_cpp_stream_riskminer,
    )

    _require_native_compiler()
    rows, instruments = 144, 3
    rng = np.random.default_rng(101)
    signal = rng.normal(size=(rows, instruments))
    lagged = np.vstack([np.zeros((1, instruments)), signal[:-1]])
    returns = 0.0008 * lagged + rng.normal(
        scale=0.00015, size=(rows, instruments)
    )
    vol = np.full((rows, instruments), 0.01)
    hs = np.full((rows, instruments), 0.0005)
    tradable = np.ones((rows, instruments))

    def sources(start: int, stop: int) -> dict[str, np.ndarray]:
        return {
            "x": signal[start:stop],
            "roll_rets": returns[start:stop],
            "vol": vol[start:stop],
            "hs": hs[start:stop],
            "is_tradable": tradable[start:stop],
        }

    terminals = {
        "x": SemanticInfo(
            frozenset({"numeric", "dimensionless"}), SearchShape.ROW
        )
    }
    config = RiskMinerConfig(
        max_depth=1,
        min_formula_depth=1,
        max_tokens=2,
        max_stack=1,
        simulations=1,
        rollouts_per_expansion=1,
        evaluation_batch_size=1,
        archive_size=8,
        replay_capacity=8,
        policy_batch_size=1,
        policy_train_epochs=1,
        pool_capacity=2,
        pool_min_improvement=-1e9,
        seed=11,
    )
    result = train_cpp_stream_riskminer(
        sources(0, 72),
        sources(72, 108),
        test_sources=sources(108, 144),
        n_instruments=instruments,
        work_dir=tmp_path,
        config=config,
        iterations=1,
        terminals=terminals,
        pool_formula_kwargs={
            "ridge_halflife": 8.0,
            "risk_halflife": 8.0,
        },
    )
    iteration = result.training.iterations[0]
    assert iteration.search.metrics.trajectories == 1
    assert iteration.policy_losses
    assert Path(iteration.policy_checkpoint).is_file()
    assert len(result.pool.entries) == 1
    assert np.isfinite(result.pool.score)
    assert result.test_evaluation is not None
    assert np.isfinite(result.test_evaluation.score)
