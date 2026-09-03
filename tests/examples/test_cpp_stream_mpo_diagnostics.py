from __future__ import annotations

import inspect

import cvxpy as cp
import numpy as np

import examples.cpp_stream_mpo_one_pass as example
from trading_dsl_engine.base.dsl import var


def test_diagnostic_pnls_delegate_to_alpha_search_ic_and_ic1(monkeypatch):
    calls = []

    def fake_ic(signal, **kwargs):
        calls.append(("ic", signal, kwargs))
        return var("ic_out")

    def fake_ic1(signal, **kwargs):
        calls.append(("ic1", signal, kwargs))
        return var("ic1_out")

    monkeypatch.setattr(example, "ic", fake_ic, raising=False)
    monkeypatch.setattr(example, "ic1", fake_ic1, raising=False)
    signal = var("signal")
    rets = var("returns")
    tradable = var("tradable")
    weights = var("weights")

    actual = example._diagnostic_pnls(
        signal,
        roll_rets=rets,
        is_tradable=tradable,
        w=weights,
        lag=2,
        hz=4,
    )

    assert set(actual) == {"ic", "ic1"}
    assert [name for name, _, _ in calls] == ["ic", "ic1"]
    for _, called_signal, kwargs in calls:
        assert called_signal is signal
        assert kwargs["roll_rets"] is rets
        assert kwargs["is_tradable"] is tradable
        assert kwargs["w"] is weights
        assert kwargs["lag"] == 2
        assert kwargs["hz"] == 4
        assert kwargs["hl"] == example.IC_VOL_SPAN


def test_diagnostic_pnls_scalarize_xs_broadcast(monkeypatch):
    monkeypatch.setattr(example, "ic", lambda *args, **kwargs: var("ic_out"))
    monkeypatch.setattr(example, "ic1", lambda *args, **kwargs: var("ic1_out"))

    actual = example._diagnostic_pnls(
        var("signal"),
        roll_rets=var("returns"),
        is_tradable=var("tradable"),
        w=var("weights"),
        lag=1,
        hz=2,
    )

    assert actual["ic"].fn == "mean"
    assert actual["ic1"].fn == "mean"
    assert "axis" in repr(actual["ic"])
    assert "axis" in repr(actual["ic1"])


def test_formula_uses_only_future_tradeable_horizons_and_negative_zscores():
    formula = example._formula(var("returns"))

    assert example.HORIZONS == (2, 4, 8, 16, 32, 64, 128)
    assert example.TRADE_STARTS == (1, 2, 4, 8, 16, 32, 64)
    assert "0_1" not in formula["alpha_pnl"]
    assert set(formula) >= {
        "returns",
        "features",
        "weights",
        "status",
        "mpo_objective",
        "mpo_gross_pnl",
        "risk",
        "alpha_pnl",
        "yhat_pnl",
    }
    assert "mpo_spread_cost" not in formula
    assert "objective" in repr(formula["mpo_objective"])
    assert len(formula["alpha_pnl"]) == len(example.HORIZONS)
    assert len(formula["yhat_pnl"]) == len(example.HORIZONS)

    features = formula["features"]
    assert features.fn == "cat"
    assert len(features.args) == len(example.FEATURE_HLS)
    for feature in features.args:
        assert feature.fn == "sub"
        assert feature.args[0].value == 0.0

    for horizon in formula["alpha_pnl"].values():
        assert set(horizon) == {"ic", "ic1"}
        assert len(horizon["ic"]) == len(example.FEATURE_HLS)
        assert len(horizon["ic1"]) == len(example.FEATURE_HLS)
    for horizon in formula["yhat_pnl"].values():
        assert set(horizon) == {"ic", "ic1"}


def test_mpo_prices_spread_directly_in_objective_without_spread_constraint():
    n_horizons = len(example.HORIZONS)
    n_assets = 3
    zeros = np.zeros((n_horizons, n_assets))
    factors = [np.eye(n_assets) for _ in range(n_horizons)]
    problem = example.MPO.factory(
        zeros,
        np.full(n_assets, 1e-4),
        np.zeros(n_assets),
        *factors,
        np.ones((n_horizons, n_assets)),
        example.RISK_RADIUS,
    )
    assert problem.is_dpp()
    assert len(problem.constraints) == 3 + n_horizons

    rng = np.random.default_rng(51)
    parameter_values = {
        "expected_returns": rng.normal(scale=2e-4, size=(n_horizons, n_assets)),
        "half_spread": np.array([4e-5, 6e-5, 8e-5]),
        "current_weights": np.array([0.01, -0.02, 0.01]),
        "trade_allowed": np.ones((n_horizons, n_assets)),
        "risk_radius": example.RISK_RADIUS,
        **{f"risk_factor_{h}": np.eye(n_assets) for h in range(n_horizons)},
    }
    for parameter in problem.parameters():
        parameter.value = parameter_values[parameter.name()]

    problem.solve(
        solver=cp.CLARABEL,
        presolve_enable=False,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
    )
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}

    variables = {variable.name(): variable for variable in problem.variables()}
    assert set(variables) == {"weights", "previous_weights"}
    weights = np.asarray(variables["weights"].value)
    current = parameter_values["current_weights"]
    delta = weights - np.vstack([current, weights[:-1]])
    direct_cost = np.sum(parameter_values["half_spread"] * np.abs(delta))
    expected_objective = (
        -np.sum(parameter_values["expected_returns"] * weights) + direct_cost
    )
    np.testing.assert_allclose(
        float(problem.value),
        expected_objective,
        rtol=2e-7,
        atol=2e-10,
    )


def test_plot_diagnostics_shows_every_figure_and_cumsums_objective():
    source = inspect.getsource(example._plot_diagnostics)
    lines = source.splitlines()
    tight_layout_lines = [
        index for index, line in enumerate(lines) if "fig.tight_layout()" in line
    ]
    assert tight_layout_lines
    for index in tight_layout_lines:
        assert lines[index + 1].strip() == "plt.show()"

    assert '_cum(values["mpo_objective"])' in source
    assert '"mpo_objective.png"' in source
