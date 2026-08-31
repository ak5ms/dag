from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import cvxpy as cp
import numpy as np

from examples import cpp_stream_mpo_one_pass as example
from trading_dsl_engine.base.dsl import var
from trading_dsl_engine.cpp_stream import compile_formula


def _shape(shape):
    return SimpleNamespace(shape=shape)


def _mpo_problem(n_assets: int = 3):
    n_horizons = len(example.HORIZONS)
    common = dict(
        expected_returns=_shape((n_horizons, n_assets)),
        half_spread=_shape((n_assets,)),
        current_weights=_shape((n_assets,)),
        risk_factor_0=_shape((n_assets, n_assets)),
        risk_factor_1=_shape((n_assets, n_assets)),
        risk_factor_2=_shape((n_assets, n_assets)),
        risk_factor_3=_shape((n_assets, n_assets)),
        risk_factor_4=_shape((n_assets, n_assets)),
        risk_factor_5=_shape((n_assets, n_assets)),
        risk_factor_6=_shape((n_assets, n_assets)),
        risk_factor_7=_shape((n_assets, n_assets)),
    )
    try:
        return example.MPO.factory(
            **common,
            trade_allowed=_shape((n_horizons, n_assets)),
        )
    except TypeError:
        return example.MPO.factory(
            **common,
            is_tradable=_shape((n_assets,)),
        )


def test_mpo_uses_abs_and_future_trade_mask() -> None:
    problem = _mpo_problem()
    params = {p.name(): p for p in problem.parameters()}
    variable_names = {v.name() for v in problem.variables()}

    assert "turnover" not in variable_names
    assert "trade_allowed" in params
    dpp_parts = {
        "objective": problem.objective.expr.is_dcp(dpp=True),
        "constraints": [constraint.is_dcp(dpp=True) for constraint in problem.constraints],
    }
    assert problem.is_dpp(), dpp_parts

    n_horizons = len(example.HORIZONS)
    n_assets = 3
    expected = np.empty((n_horizons, n_assets))
    for h in range(n_horizons):
        expected[h] = [1.0, -1.0, 0.0] if h % 2 == 0 else [-1.0, 1.0, 0.0]
    params["expected_returns"].value = expected
    params["half_spread"].value = np.full(n_assets, 1e-5)
    params["current_weights"].value = np.zeros(n_assets)
    for h in range(n_horizons):
        params[f"risk_factor_{h}"].value = np.eye(n_assets)
    params["risk_radius"].value = 1.0

    allowed = np.ones((n_horizons, n_assets))
    allowed[3] = 0.0
    params["trade_allowed"].value = allowed

    problem.solve(solver=cp.CLARABEL)
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    weights = next(v for v in problem.variables() if v.name() == "weights").value
    np.testing.assert_allclose(weights[3], weights[2], rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(weights.sum(axis=1), 0.0, rtol=0.0, atol=1e-8)


def test_planned_trade_mask_uses_current_and_next_sessions(tmp_path: Path) -> None:
    assert hasattr(example, "_planned_trade_allowed")

    minute = 60_000_000.0
    ts = 1_800_000_000_000_000.0
    data = {
        "_ev_ts": np.array([[ts]]),
        "is_tradable_out0": np.array([[1.0]]),
        "session_start0": np.array([[ts - minute]]),
        "session_end0": np.array([[ts + 1.5 * minute]]),
        "next_session_start0": np.array([[ts + 3.0 * minute]]),
        "next_session_end0": np.array([[ts + 70.0 * minute]]),
    }
    runtime = compile_formula(
        example._planned_trade_allowed(var("is_tradable_out0")),
        data,
        n_instruments=1,
    )
    result = runtime.run(out_path=tmp_path / "mask.npy")
    actual = np.asarray(result.load()).reshape(-1)
    np.testing.assert_array_equal(actual, np.array([1, 1, 0, 1, 1, 1, 1, 1], dtype=float))


def test_mpo_uses_total_block_returns_without_elapsed_normalization() -> None:
    source = inspect.getsource(example._formula)
    assert "block_return / block_elapsed" not in source
    assert "fillna(yhat, 0.0) * width" not in source
    assert "risk_block / risk_elapsed**0.5" not in source
