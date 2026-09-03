"""InputData -> gap-aware Ridge forecasts -> sequential Clarabel MPO, in one loop."""

from __future__ import annotations

import os
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flows.alpha_search import ic, ic1
from flows.load import InputData
from flows.pov import RollRets
from flows.riskmodel import risk_covariance
from flows.utils import streak, ts_zscore
from trading_dsl_engine.base.dsl import (
    Ridge,
    cat,
    einsum,
    ffill,
    fillna,
    get_beta,
    isnan,
    psd_factor,
    purify,
    reduce_max,
    rolling_sum,
    shift,
    var,
    where,
)
from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.cpp_stream.optimizer import (
    ClarabelNativePaths,
    build_current_clarabel,
    cvxpy_program,
    get_field,
    previous_solution,
)

HORIZONS = (2, 4, 8, 16, 32, 64, 128)
TRADE_STARTS = (1,) + HORIZONS[:-1]
FEATURE_HLS = (4, 16, 64, 256)
IC_VOL_SPAN = 1440 * 21
RIDGE_HL = 1440 * 21
RISK_SPAN = 1440 * 21
RISK_MIN_PERIODS = 64
RISK_RADIUS = 0.08
TRADE_BIG_M = 1e3
MINUTE_US = 60_000_000.0
ROWS = int(os.environ.get("MPO_EXAMPLE_ROWS", "20000"))
CACHE = Path(".generated/cpp_stream_mpo_one_pass")


def _clarabel() -> ClarabelNativePaths:
    include = os.environ.get("CLARABEL_INCLUDE_DIR")
    library = os.environ.get("CLARABEL_STATIC_LIBRARY")
    if include and library:
        return ClarabelNativePaths(Path(include), Path(library))
    return build_current_clarabel()


def _feature_span(hl: float) -> float:
    return 2 / (1 - 0.5 ** (1 / hl)) - 1


def _diagnostic_pnls(
    signal,
    *,
    roll_rets,
    is_tradable,
    w,
    lag: int,
    hz: int,
):
    kwargs = dict(
        roll_rets=roll_rets,
        is_tradable=is_tradable,
        hl=IC_VOL_SPAN,
        lag=lag,
        hz=hz,
        w=w,
    )
    return {
        "ic": ic(signal, **kwargs).mean(axis=[1]),
        "ic1": ic1(signal, **kwargs).mean(axis=[1]),
    }


def _planned_trade_allowed(tradable):
    """Scheduled availability for each future trade start."""
    raw_ts = var("_ev_ts")
    ts = ffill(raw_ts) + streak(isnan(raw_ts)) * MINUTE_US
    session_start = ffill(var("session_start0"))
    session_end = ffill(var("session_end0"))
    next_session_start = ffill(var("next_session_start0"))
    next_session_end = ffill(var("next_session_end0"))

    allowed = []
    for start in TRADE_STARTS:
        trade_ts = ts + start * MINUTE_US
        in_session = (trade_ts >= session_start) & (trade_ts < session_end)
        in_next_session = (trade_ts >= next_session_start) & (trade_ts < next_session_end)
        allowed.append(
            fillna(where(in_session | in_next_session, 1.0, 0.0), 0.0)
        )
    return cat(*allowed)


@cvxpy_program(
    cache_dir=CACHE / "clarabel",
    clarabel=_clarabel,
    sequential=None,
    solver_settings={"iterative_refinement_enable": False},
)
def MPO(
    expected_returns,
    half_spread,
    current_weights,
    risk_factor_0,
    risk_factor_1,
    risk_factor_2,
    risk_factor_3,
    risk_factor_4,
    risk_factor_5,
    risk_factor_6,
    trade_allowed,
    risk_radius=RISK_RADIUS,
):
    n_horizons, n_assets = expected_returns.shape
    expected_returns = cp.Parameter(expected_returns.shape, name="expected_returns")
    half_spread = cp.Parameter(half_spread.shape, name="half_spread", nonneg=True)
    current_weights = cp.Parameter((n_assets,), name="current_weights")
    risk_factors = tuple(
        cp.Parameter(arg.shape, name=f"risk_factor_{h}")
        for h, arg in enumerate(
            (
                risk_factor_0,
                risk_factor_1,
                risk_factor_2,
                risk_factor_3,
                risk_factor_4,
                risk_factor_5,
                risk_factor_6,
            )
        )
    )
    trade_allowed = cp.Parameter(
        trade_allowed.shape, name="trade_allowed", nonneg=True
    )
    risk_radius = cp.Parameter(name="risk_radius", nonneg=True)

    weights = cp.Variable((n_horizons, n_assets), name="weights")
    previous_weights = cp.Variable((n_assets,), name="previous_weights")
    delta = weights - cp.vstack([previous_weights, weights[:-1]])
    abs_delta = cp.abs(delta)
    spread_cost = cp.sum(cp.multiply(half_spread, abs_delta))
    constraints = [
        previous_weights == current_weights,
        cp.sum(delta, axis=1) == 0,
        abs_delta <= TRADE_BIG_M * trade_allowed,
    ]
    for h, risk_factor in enumerate(risk_factors):
        risk = cp.SOC(risk_radius, risk_factor @ weights[h])
        risk.set_label(f"risk_{h}")
        constraints.append(risk)
    return cp.Problem(
        cp.Minimize(
            -cp.sum(cp.multiply(expected_returns, weights))
            + spread_cost
        ),
        constraints,
    )


def _formula(returns=None):
    returns = RollRets().roll_rets() if returns is None else returns
    tradable = fillna(var("is_tradable_out0"), 0.0)
    hs = var("vw_halfspread_out0")
    fit_weights = purify(1 / hs**2)
    feature_list = tuple(
        -ts_zscore(
            returns,
            _feature_span(hl),
            min_periods=max(2, round(_feature_span(hl))),
        )
        for hl in FEATURE_HLS
    )
    features = cat(*feature_list)

    # Closed rows contribute zero; RollRets puts the close-to-open gap move on
    # the first tradable row after reopening.
    clean_returns = where(tradable != 0, fillna(returns, 0.0), 0.0)

    forecasts, factors = [], []
    alpha_pnl, yhat_pnl = {}, {}
    for start, end in zip(TRADE_STARTS, HORIZONS):
        width = end - start
        block_return = rolling_sum(clean_returns, width, min_periods=width)
        block_observed = rolling_sum(tradable, width, min_periods=width)

        # Ridge predicts the total return of block (start, end] directly.
        # ic1 alignment implies feature time t-end for a target ending at t.
        target = where(block_observed > 0, block_return, float("nan"))
        fit_x = cat(
            *(
                where(
                    shift(tradable, end) != 0,
                    shift(feature, end),
                    float("nan"),
                )
                for feature in feature_list
            )
        )
        beta = get_beta(
            Ridge(
                fit_x,
                y=target,
                weights=fit_weights,
                hl=RIDGE_HL,
                lambda_=0.1,
            )
        )
        yhat = einsum(features, beta, "if,f->i")
        forecasts.append(fillna(yhat, 0.0))

        horizon_key = f"{start}_{end}"
        alpha_diagnostics = [
            _diagnostic_pnls(
                feature,
                roll_rets=returns,
                is_tradable=tradable,
                w=fit_weights,
                lag=start,
                hz=width,
            )
            for feature in feature_list
        ]
        alpha_pnl[horizon_key] = {
            "ic": {
                f"hl_{hl}": diagnostic["ic"]
                for hl, diagnostic in zip(FEATURE_HLS, alpha_diagnostics)
            },
            "ic1": {
                f"hl_{hl}": diagnostic["ic1"]
                for hl, diagnostic in zip(FEATURE_HLS, alpha_diagnostics)
            },
        }
        yhat_pnl[horizon_key] = _diagnostic_pnls(
            yhat,
            roll_rets=returns,
            is_tradable=tradable,
            w=fit_weights,
            lag=start,
            hz=width,
        )

        # Risk uses the total return of the same disjoint block, with no
        # elapsed-time normalization.
        risk_block = shift(block_return, start)
        risk_observed = shift(block_observed, start)
        risk_sample = where(risk_observed > 0, risk_block, float("nan"))
        covariance = risk_covariance(
            risk_sample,
            span=RISK_SPAN,
            min_periods=RISK_MIN_PERIODS,
            ignore_na=True,
            adjust=False,
        )
        factors.append(
            psd_factor(fillna(covariance, 0.0), eigenvalue_floor=1e-8)
        )

    forecast_matrix = cat(*forecasts)
    mpo = MPO(
        expected_returns=forecast_matrix,
        half_spread=fillna(purify(hs), 0.0),
        current_weights=previous_solution("weights[0]", initial=0.0),
        risk_factor_0=factors[0],
        risk_factor_1=factors[1],
        risk_factor_2=factors[2],
        risk_factor_3=factors[3],
        risk_factor_4=factors[4],
        risk_factor_5=factors[5],
        risk_factor_6=factors[6],
        trade_allowed=_planned_trade_allowed(tradable),
        risk_radius=RISK_RADIUS,
    )
    session_open = reduce_max(tradable, axis=[1]) != 0.0
    weights = where(
        session_open,
        get_field(mpo, "weights[0]"),
        float("nan"),
    )
    status = where(
        session_open,
        get_field(mpo, "status"),
        float("nan"),
    )
    mpo_objective = where(
        session_open,
        get_field(mpo, "objective"),
        float("nan"),
    )
    risk_values = {
        f"{start}_{end}": where(
            session_open,
            get_field(mpo, f"risk_{h}.value"),
            float("nan"),
        )
        for h, (start, end) in enumerate(zip(TRADE_STARTS, HORIZONS))
    }

    mpo_gross_pnl = (
        fillna(shift(ffill(weights), 1), 0.0) * clean_returns
    ).sum(axis=[1])

    return {
        "returns": returns,
        "features": features,
        "weights": weights,
        "status": status,
        "mpo_objective": mpo_objective,
        "mpo_gross_pnl": mpo_gross_pnl,
        "risk": risk_values,
        "alpha_pnl": alpha_pnl,
        "yhat_pnl": yhat_pnl,
    }


def _cum(x):
    return np.cumsum(np.where(np.isfinite(x), x, 0.0))


def _plot_diagnostics(data, values, *, plot_dir):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    ts = np.asarray(data["_ev_ts"])
    ts = ts[:, 0] if ts.ndim > 1 else ts
    index = pd.to_datetime(pd.Series(ts).interpolate(), unit="us")
    paths = []

    for start, end in zip(TRADE_STARTS, HORIZONS):
        horizon_key = f"{start}_{end}"
        diagnostics = values["alpha_pnl"][horizon_key]
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, pnl in diagnostics["ic"].items():
            ax.plot(index, _cum(pnl), label=f"{label} ic")
        for label, pnl in diagnostics["ic1"].items():
            ax.plot(index, _cum(pnl), "--", label=f"{label} ic1")
        ax.set_title(f"Alpha PnL: horizon ({start}, {end}] — ic vs ic1")
        ax.set_ylabel("Cumulative PnL")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        plt.show()
        path = plot_dir / f"alphas_horizon_{start}_{end}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

        yhat = values["yhat_pnl"][horizon_key]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(index, _cum(yhat["ic"]), label="ic")
        ax.plot(index, _cum(yhat["ic1"]), "--", label="ic1")
        ax.set_title(f"Aggregated Ridge yhat PnL: horizon ({start}, {end}]")
        ax.set_ylabel("Cumulative PnL")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        plt.show()
        path = plot_dir / f"yhat_horizon_{start}_{end}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(index, _cum(values["mpo_gross_pnl"]), label="gross")
    ax.set_title("Implemented MPO portfolio gross PnL")
    ax.set_ylabel("Cumulative realized PnL")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()
    path = plot_dir / "portfolio_pnl.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(index, _cum(values["mpo_objective"]), label="objective")
    ax.set_title("MPO objective")
    ax.set_ylabel("Cumulative minimized objective")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()
    path = plot_dir / "mpo_objective.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    for start, end in zip(TRADE_STARTS, HORIZONS):
        value = np.asarray(values["risk"][f"{start}_{end}"], dtype=float)
        ax.plot(index, np.linalg.norm(value[:, 1:], axis=1), label=f"({start}, {end}]")
    ax.axhline(RISK_RADIUS, linestyle="--", label="constraint")
    ax.set_title("MPO risk constraint")
    ax.set_ylabel("sqrt(w' S w)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()
    path = plot_dir / "risk_constraint.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def _run(data, *, returns=None, output_dir=CACHE):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_assets = data["is_tradable_out0"].shape[1]
    runtime = compile_formula(
        _formula(returns),
        data,
        n_instruments=n_assets,
    )

    generated = runtime.generated_cpp.read_text()
    row_loop = "for (std::size_t t = row_begin; t < row_end; ++t)"
    assert generated.count(row_loop) == 1
    assert generated.count("stackdsl::ClarabelNode<") == 1

    result = runtime.run(out_path=output_dir / "result.npy")
    values = result.load()
    paths = _plot_diagnostics(data, values, plot_dir=output_dir / "plots")
    return result, paths


def main() -> None:
    data = InputData(nrows=ROWS, idx=None).get_data()
    result, paths = _run(data)
    print(f"rows={result.rows:,} seconds={result.seconds:.3f}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
