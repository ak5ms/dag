"""InputData -> gap-aware Ridge forecasts -> sequential Clarabel MPO, in one loop."""

from __future__ import annotations

import os
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flows.alpha_search import _ic1_terms, ic, ic1
from flows.load import InputData
from flows.pov import RollRets
from flows.riskmodel import risk_covariance
from flows.utils import ewm_std, streak, ts_zscore
from trading_dsl_engine.base.dsl import (
    Ridge,
    abs,
    isfinite,
    xs_rank,
    cat,
    einsum,
    ffill,
    fillna,
    get_beta,
    isnan,
    psd_factor,
    purify,
    rolling_sum,
    shift,
    var,
    where,
)
from trading_dsl_engine.cpp_stream import compile_formula, xs_gauss
from trading_dsl_engine.cpp_stream.optimizer import (
    ClarabelNativePaths,
    build_current_clarabel,
    cvxpy_program,
    get_field,
    previous_solution,
)

HORIZONS = (2, 4, 8, 16, 32, 64, 128)
TRADE_STARTS = (1,) + HORIZONS[:-1]
FEATURE_SPANS = (4, 16, 64, 256)
IC_VOL_SPAN = 1440 * 21
RIDGE_HL = 1440 * 21
RISK_SPAN = 1440 * 21
RISK_MIN_PERIODS = 64
RISK_RADIUS = 0.08
TRADE_BIG_M = 1e3
MINUTE_US = 60_000_000.0
ROWS = int(os.environ.get("MPO_EXAMPLE_ROWS", "200000"))
CACHE = Path(".generated/cpp_stream_mpo_one_pass")


def _clarabel() -> ClarabelNativePaths:
    include = os.environ.get("CLARABEL_INCLUDE_DIR")
    library = os.environ.get("CLARABEL_STATIC_LIBRARY")
    if include and library:
        return ClarabelNativePaths(Path(include), Path(library))
    return build_current_clarabel()


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
    # xs_sum is broadcast over instrument lanes. Match scratch_10.py's
    # np.nansum(..., axis=1), not a mean that changes PnL by n_assets.
    return {
        "ic": ic(signal, **kwargs).fillna(0.0).sum(axis=[1]),
        "ic1": ic1(signal, **kwargs).fillna(0.0).sum(axis=[1]),
    }


def _session_clock():
    raw_ts = var("_ev_ts")
    return (
        ffill(raw_ts) + streak(isnan(raw_ts)) * MINUTE_US,
        *(ffill(var(name)) for name in (
            "session_start0", "session_end0", "next_session_start0", "next_session_end0",
        )),
    )


def _planned_trade_allowed(tradable):
    """Future execution anchors, distinct from future return observations."""
    ts, session_start, session_end, next_start, next_end = _session_clock()
    allowed = []
    for h, start in enumerate(TRADE_STARTS):
        trade_ts = ts + start * MINUTE_US
        available = fillna(
            ((trade_ts >= session_start) & (trade_ts < session_end))
            | ((trade_ts >= next_start) & (trade_ts < next_end)), 0.0,
        )
        if h == 0:
            available = available & (tradable != 0)
        allowed.append(where(available, 1.0, 0.0))
    return cat(*allowed)


def _planned_block_has_returns(start, end):
    """Closed anchors may still own the FULL first reopening return."""
    ts, session_start, session_end, next_start, next_end = _session_clock()
    first, last = ts + (start + 1) * MINUTE_US, ts + end * MINUTE_US
    return fillna(
        ((last >= session_start) & (first < session_end))
        | ((last >= next_start) & (first < next_end)), 0.0,
    )


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
    held_weights,
    execution_allowed,
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
    held_weights = cp.Parameter((n_assets,), name="held_weights")
    execution_allowed = cp.Parameter((n_assets,), name="execution_allowed", nonneg=True)
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
    queued = cp.Variable((n_assets,), name="queued")
    held = cp.Variable((n_assets,), name="held")
    delta = weights - cp.vstack([previous_weights, weights[:-1]])
    abs_delta = cp.abs(delta)
    spread_cost = cp.sum(cp.multiply(half_spread, abs_delta))
    constraints = [
        queued == current_weights,
        held == held_weights,
        # Parameter x variable, not parameter x parameter: preserve DPP.
        previous_weights == held + cp.multiply(execution_allowed, queued - held),
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


def _formula(returns=None, *, beta_override=None, fit_weights=None):
    returns = RollRets().roll_rets() if returns is None else returns
    tradable = fillna(var("is_tradable_out0"), 0.0)
    hs = var("vw_halfspread_out0")
    fit_weights = purify(1 / hs**2) if fit_weights is None else fit_weights
    # Match scratch_10.py exactly: these are EWM spans, not half-lives.
    cleaned = where((abs(returns) <= 0.05) & (returns != 0), returns, float("nan"))
    return_vol = ewm_std(cleaned, span=IC_VOL_SPAN)
    alphas = tuple(xs_gauss(xs_rank(-ts_zscore(cleaned, span))) for span in FEATURE_SPANS)
    feature_list = tuple(alpha * return_vol for alpha in alphas)
    features = cat(*feature_list)

    # Feature cleaning must NOT erase real portfolio losses or gap risk.
    realized_returns = where(isfinite(returns), returns, 0.0)
    forecasts, factors = [], []
    risk_samples, risk_factors = {}, {}
    alpha_pnl, yhat_pnl, forecast_diagnostics = {}, {}, {}
    for start, end in zip(TRADE_STARTS, HORIZONS):
        width = end - start
        # Reuse ic1's matured samples, not shift(feature, end). A closed
        # execution anchor holds BOTH the last executable predictor and weight.
        terms = [
            _ic1_terms(feature, roll_rets=cleaned, is_tradable=tradable,
                       lag=start, hz=width, w=fit_weights)
            for feature in feature_list
        ]
        target, origin_weight = terms[0][1:]
        observed = rolling_sum(where(isfinite(cleaned), 1.0, 0.0), width,
                               min_periods=width) > 0
        # Ridge uses pairwise statistics: masking only y leaves XX updating.
        fit_x = cat(*(where(observed, term[0], float("nan")) for term in terms))
        target = where(observed, target, float("nan"))
        sample_weight = where(observed, origin_weight, 0.0)
        beta = get_beta(Ridge(fit_x, y=target, weights=sample_weight,
                              hl=RIDGE_HL, lambda_=0.1))
        yhat = (einsum(features, beta, "if,f->i") if beta_override is None
                else sum(feature * beta_override for feature in feature_list))
        # ic1 targets a mean. MPO consumes the disjoint block TOTAL, once.
        yhat_total = width * yhat
        # There cannot be marked returns in a known entirely closed block.
        # A partly closed block that contains reopening is NOT scaled by the
        # open fraction: its target already includes the full gap / width.
        forecasts.append(where(_planned_block_has_returns(start, end),
                               fillna(yhat_total, 0.0), 0.0))

        horizon_key = f"{start}_{end}"
        alpha_diagnostics = [
            _diagnostic_pnls(
                feature,
                roll_rets=cleaned,
                is_tradable=tradable,
                w=fit_weights,
                lag=start,
                hz=width,
            )
            for feature in alphas
        ]
        alpha_pnl[horizon_key] = {
            "ic": {
                f"span_{hl}": diagnostic["ic"]
                for hl, diagnostic in zip(FEATURE_SPANS, alpha_diagnostics)
            },
            "ic1": {
                f"span_{hl}": diagnostic["ic1"]
                for hl, diagnostic in zip(FEATURE_SPANS, alpha_diagnostics)
            },
        }
        forecast_diagnostics[horizon_key] = {
            "fit_x": fit_x, "target": target, "sample_weight": sample_weight,
            "beta_fitted": beta, "yhat_rate": yhat, "yhat_total": yhat_total,
        }
        # yhat is in return units. Convert back to alpha units before ic's
        # own inverse-vol normalization. beta=1 then exactly recovers alpha.
        yhat_pnl[horizon_key] = _diagnostic_pnls(
            yhat / return_vol,
            roll_rets=cleaned,
            is_tradable=tradable,
            w=fit_weights,
            lag=start,
            hz=width,
        )

        # Keep full close-to-open returns, without dividing by elapsed gap
        # length. Risk observes the latest mature width-row total (not a
        # second, arbitrarily shifted window or the clipped feature series).
        risk_block = rolling_sum(realized_returns, width, min_periods=width)
        risk_observed = rolling_sum(
            where(isfinite(returns) & ((tradable != 0) | (returns != 0)), 1.0, 0.0),
            width, min_periods=width,
        )
        risk_sample = where(risk_observed > 0, risk_block, float("nan"))
        covariance = risk_covariance(risk_sample, span=RISK_SPAN,
                                     min_periods=RISK_MIN_PERIODS,
                                     ignore_na=True, adjust=False)
        factor = psd_factor(fillna(covariance, 0.0), eigenvalue_floor=1e-8)
        factors.append(factor)
        risk_samples[horizon_key] = risk_sample
        risk_factors[horizon_key] = factor

    # The binding exposes row-major (asset, horizon) buffers as CVXPY's
    # column-major (horizon, asset); do not transpose these a second time.
    # Likewise psd_factor emits L, S=L L', and CVXPY sees L' in the SOC.
    forecast_matrix = cat(*forecasts)
    execution_allowed = where((tradable != 0) & isfinite(hs) & (hs >= 0), 1.0, 0.0)
    trade_allowed = _planned_trade_allowed(execution_allowed)
    effective_spread = fillna(ffill(where(isfinite(hs) & (hs >= 0), hs, float("nan"))), 0.0)
    mpo = MPO(
        expected_returns=forecast_matrix,
        half_spread=effective_spread,
        current_weights=previous_solution("weights[0]", initial=0.0),
        held_weights=previous_solution("previous_weights", initial=0.0),
        execution_allowed=execution_allowed,
        risk_factor_0=factors[0],
        risk_factor_1=factors[1],
        risk_factor_2=factors[2],
        risk_factor_3=factors[3],
        risk_factor_4=factors[4],
        risk_factor_5=factors[5],
        risk_factor_6=factors[6],
        trade_allowed=trade_allowed,
        risk_radius=RISK_RADIUS,
    )
    # Signal at t is an order for t+1's VWAP, earning r[t+2] onward.
    # previous_weights is today's actually executed portfolio, not today's
    # newly optimized target. Closed/missing-quote lanes keep their old holding.
    planned_weights = get_field(mpo, "weights[0]")
    weights = get_field(mpo, "previous_weights")
    status = get_field(mpo, "status")
    mpo_objective = get_field(mpo, "objective")
    risk_values = {
        f"{start}_{end}": get_field(mpo, f"risk_{h}.value")
        for h, (start, end) in enumerate(zip(TRADE_STARTS, HORIZONS))
    }
    held_on_return = fillna(shift(weights, 1), 0.0)
    mpo_gross_pnl = (held_on_return * realized_returns).sum(axis=[1])
    realized_cost = (abs(weights - held_on_return) * effective_spread).sum(axis=[1])

    return {
        "returns": returns,
        "features": features,
        "alphas": cat(*alphas),
        "return_vol": return_vol,
        "clean_returns": cleaned,
        "weights": weights,
        "planned_weights": planned_weights,
        "planned_path": get_field(mpo, "weights"),
        "expected_returns": forecast_matrix,
        "trade_allowed": trade_allowed,
        "status": status,
        "mpo_objective": mpo_objective,
        "mpo_gross_pnl": mpo_gross_pnl,
        "mpo_realized_cost": realized_cost,
        "mpo_net_pnl": mpo_gross_pnl - realized_cost,
        "risk": risk_values,
        "risk_samples": risk_samples,
        "risk_factors": risk_factors,
        "alpha_pnl": alpha_pnl,
        "yhat_pnl": yhat_pnl,
        "forecast_diagnostics": forecast_diagnostics,
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
    if "mpo_net_pnl" in values:
        ax.plot(index, _cum(values["mpo_net_pnl"]), label="net")
    ax.set_title("Executed MPO portfolio PnL")
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
    n_rows, n_assets = data["is_tradable_out0"].shape
    if n_rows < IC_VOL_SPAN:
        raise ValueError(f"{n_rows} rows cannot warm up IC_VOL_SPAN={IC_VOL_SPAN}; "
                         "provide more history rather than silently changing the signal")
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
    if not np.isfinite(values["return_vol"]).any():
        raise ValueError("No valid return volatility after cleaning/session gaps; "
                         "increase the available history")
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
