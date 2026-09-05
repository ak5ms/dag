"""InputData -> gap-aware Ridge forecasts -> sequential Clarabel MPO, in one loop."""

from __future__ import annotations

import os
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flows.alpha_search import _ic_terms, ic, ic1
from flows.load import InputData
from flows.pov import RollRets
from flows.riskmodel import risk_covariance
from flows.utils import streak, ts_zscore, replace, ewm_std
from trading_dsl_engine.base.dsl import (
    Ridge,
    abs as dsl_abs,
    xs_rank,
    cat,
    ceil,
    maximum,
    minimum,
    cumsum,
    ewm,
    isfinite,
    rolling_mean,
    einsum,
    ffill,
    fillna,
    get_beta,
    isnan,
    psd_factor,
    purify,
    reduce_max,
    reduce_min,
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
RIDGE_MIN_PERIODS = 64
RIDGE_LAMBDA = 0.1
RISK_SPAN = 1440 * 21
RISK_MIN_PERIODS = 64
GAP_RISK_SPAN = 21
LONG_GAP_MINUTES = 1440
SHORT_GAP_VOL_FLOOR = 0.005
LONG_GAP_VOL_FLOOR = 0.02
RISK_RADIUS = 0.08
# A controllable plan reserves 10% for next-row risk-estimate/calendar revisions.
# Forced carry retains the full hard radius; infeasible solves must fail closed.
RISK_HEADROOM = 0.90
TRADE_BIG_M = 1e3
MINUTE_US = 60_000_000.0
ROWS = int(os.environ.get("MPO_EXAMPLE_ROWS", "90000"))
CACHE = Path(".generated/cpp_stream_mpo_one_pass")


def _clarabel() -> ClarabelNativePaths:
    include = os.environ.get("CLARABEL_INCLUDE_DIR")
    library = os.environ.get("CLARABEL_STATIC_LIBRARY")
    if include and library:
        return ClarabelNativePaths(Path(include), Path(library))
    return build_current_clarabel()


def _signal_returns(returns):
    """Exactly scratch_10's zero/outlier policy; never use this for actual PnL."""
    return where(dsl_abs(returns) <= 0.05,
                 replace(returns, 0.0, float("nan")), float("nan"))


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
        # scratch_10 uses np.nansum(v, axis=1). xs_sum currently broadcasts its
        # result to all lanes, so preserve that display scale deliberately.
        "ic": ic(signal, **kwargs).fillna(0.).sum(axis=[1]),
        "ic1": ic1(signal, **kwargs).fillna(0.).sum(axis=[1]),
    }


def _session_plan(tradable, half_spread):
    """Calendar-known future availability; realized masks are used only at t."""
    raw_ts = purify(var("_ev_ts"))
    ts = ffill(raw_ts) + streak(isnan(raw_ts)) * MINUTE_US
    session_start = ffill(var("session_start0"))
    session_end = ffill(var("session_end0"))
    next_start = ffill(var("next_session_start0"))
    next_end = ffill(var("next_session_end0"))

    def open_at(offset):
        date = ts + offset * MINUTE_US
        current = fillna((date >= session_start) & (date < session_end), 0.0) != 0
        following = fillna((date >= next_start) & (date < next_end), 0.0) != 0
        return current | following

    opened = {offset: open_at(offset) for offset in {0, *TRADE_STARTS, *HORIZONS}}
    quote_ok = fillna(isfinite(half_spread) & (half_spread > 0), 0.) != 0
    held_cost = ffill(where(quote_ok, half_spread, float("nan")))
    execute = opened[0] & (tradable != 0) & quote_ok
    # Future masks are unknown. Plan from the published calendar and the last
    # valid quote; today's realized mask controls today's fills only. Reoptimize
    # and re-check execution at every row rather than predicting a 128-row halt.
    can_plan = isfinite(held_cost)
    allowed = tuple(where(opened[start] & can_plan, 1., 0.) for start in TRADE_STARTS)
    def count_regular(left, right, start, end):
        # Integer return endpoints j in (start,end], with both j-1 and j open.
        lower = maximum(start + 1., ceil((left - ts) / MINUTE_US) + 1.)
        upper = minimum(float(end), ceil((right - ts) / MINUTE_US) - 1.)
        valid = fillna(isfinite(left) & isfinite(right) & (right > left), 0.) != 0
        return where(valid, maximum(upper - lower + 1., 0.), 0.)

    joined = fillna((next_start <= session_end) & (next_end >= session_start), 0.) != 0
    regular_counts = tuple(
        where(joined,
              count_regular(minimum(session_start, next_start), maximum(session_end, next_end), start, end),
              count_regular(session_start, session_end, start, end)
              + count_regular(next_start, next_end, start, end))
        for start, end in zip(TRADE_STARTS, HORIZONS)
    )
    return {"timestamp": ts, "session_start": session_start, "session_end": session_end,
            "next_start": next_start, "next_end": next_end, "opened": opened,
            "allowed": allowed, "trade_allowed": cat(*allowed), "execution_allowed": where(execute, 1., 0.),
            "regular_counts": regular_counts, "half_spread": fillna(held_cost, 0.)}


def _gap_risk(returns, tradable, session):
    """Event-time total-gap second moments, separate from ordinary minute risk.

    Daily/short and weekend/long closures have separate clocks. No 5% cutoff,
    calendar-width division, or ordinary-minute annualization is applied to a
    realized gap. Floors are explicit conservative bootstrap assumptions.
    """
    ts = session["timestamp"]
    observed = (tradable != 0) & isfinite(returns)
    last_mark = ffill(where(observed, ts, float("nan")))
    gap_minutes = (ts - shift(last_mark)) / MINUTE_US
    gap = fillna(observed & (gap_minutes > 1.), 0.) != 0
    long_gap = fillna(gap_minutes > LONG_GAP_MINUTES, 0.) != 0
    short_sample = where(gap & ~long_gap, returns, float("nan"))
    long_sample = where(gap & long_gap, returns, float("nan"))
    def event_sigma(sample, floor):
        second = ewm(sample ** 2, GAP_RISK_SPAN, min_periods=1,
                     ignore_na=True, adjust=True)
        return maximum(fillna(second, 0.), floor ** 2) ** .5
    short_sigma = event_sigma(short_sample, SHORT_GAP_VOL_FLOOR)
    long_sigma = event_sigma(long_sample, LONG_GAP_VOL_FLOOR)

    before_current_open = fillna(session["session_start"] > ts, 0.) != 0
    next_gap_known = fillna((session["next_start"] > ts)
                           & (session["next_start"] > session["session_end"]), 0.) != 0
    reopen = where(before_current_open, session["session_start"],
                   where(next_gap_known, session["next_start"], float("nan")))
    duration = where(before_current_open, (reopen - last_mark) / MINUTE_US,
                     (reopen - session["session_end"]) / MINUTE_US + 1.)
    expected_sigma = where(fillna(duration > LONG_GAP_MINUTES, 0.) != 0, long_sigma, short_sigma)
    gap_factors = []
    for h, (start, end) in enumerate(zip(TRADE_STARTS, HORIZONS)):
        exposed = fillna((reopen > ts + start * MINUTE_US)
                         & (reopen <= ts + end * MINUTE_US), 0.) != 0
        if h == len(HORIZONS) - 1:
            # Last available pre-close position survives a weekend even when
            # reopening lies beyond the finite optimization horizon.
            tail = (~session["opened"][end]
                    & (fillna(reopen > ts + end * MINUTE_US, 0.) != 0))
            exposed = exposed | tail
        gap_factors.append(where(exposed, expected_sigma, 0.))
    return {"event": gap, "minutes": gap_minutes,
            "sigma": where(long_gap, long_sigma, short_sigma),
            "short_sigma": short_sigma, "long_sigma": long_sigma,
            "planned_sigma": cat(*gap_factors), "planned_factors": gap_factors,
            "ordinary": where(observed & ~gap, returns, float("nan"))}


@cvxpy_program(
    cache_dir=CACHE / "clarabel",
    clarabel=_clarabel,
    sequential=None,
    solver_settings={"iterative_refinement_enable": True},
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
    last_weights,
    execution_allowed,
    gap_volatility,
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
    last_weights = cp.Parameter((n_assets,), name="last_weights")
    execution_allowed = cp.Parameter((n_assets,), name="execution_allowed", nonneg=True)
    gap_volatility = cp.Parameter(expected_returns.shape, name="gap_volatility", nonneg=True)

    weights = cp.Variable((n_horizons, n_assets), name="weights")
    previous_weights = cp.Variable((n_assets,), name="previous_weights")
    previous_plan = cp.Variable((n_assets,), name="previous_plan")
    previous_actual = cp.Variable((n_assets,), name="previous_actual")
    gap_exposure = cp.Variable((n_horizons, n_assets), name="gap_exposure")
    delta = weights - cp.vstack([previous_weights, weights[:-1]])
    abs_delta = cp.abs(delta)
    spread_cost = cp.sum(cp.multiply(half_spread, abs_delta))
    constraints = [
        previous_plan == current_weights,
        previous_actual == last_weights,
        # Variable copies avoid parameter*parameter products and retain DPP.
        previous_weights == (cp.multiply(execution_allowed, previous_plan)
                             + cp.multiply(1 - execution_allowed, previous_actual)),
        cp.sum(weights, axis=1) == 0,
        abs_delta <= TRADE_BIG_M * trade_allowed,
        gap_exposure >= cp.multiply(gap_volatility, weights),
        gap_exposure >= -cp.multiply(gap_volatility, weights),
    ]
    for h, risk_factor in enumerate(risk_factors):
        # Sparse/asynchronous gap samples cannot justify overnight hedge credit:
        # sum(gap_sigma*abs(w)) is the worst-correlation event-risk bound.
        risk = cp.SOC(risk_radius, cp.hstack([risk_factor @ weights[h],
                                             cp.sum(gap_exposure[h])]))
        risk.set_label(f"risk_{h}")
        constraints.append(risk)
    return cp.Problem(
        cp.Minimize(
            -cp.sum(cp.multiply(expected_returns, weights))
            + spread_cost
        ),
        constraints,
    )


def _forecast(feature_list, *, sigma, returns, tradable, w, lag, hz,
              target_valid, beta_override=None):
    """Fit a per-open-bar return rate using the exact ic1 observation state.

    ``ic1`` aligns held positions at t-hz with the trailing hz-return mean.
    Here its position is alpha*sigma, NOT alpha/sigma. All sufficient
    statistics use the same eligible lane mask (Ridge's XX does not inspect Y).
    """
    scaled = tuple(feature * sigma for feature in feature_list)
    positions, readiness = [], []
    for value in scaled:
        position, clean, weight = _ic_terms(
            value, roll_rets=returns, is_tradable=tradable, w=w, lag=lag,
        )
        ready, _, _ = _ic_terms(
            where(isfinite(value), 1.0, float("nan")),
            roll_rets=returns, is_tradable=tradable, w=w, lag=lag,
        )
        positions.append(shift(position, hz))
        readiness.append(shift(ready, hz) > 0.0)
    canonical_x = cat(*positions)
    canonical_y = rolling_mean(clean, hz, min_periods=hz)
    canonical_weight = shift(weight, hz)
    eligible = (rolling_sum(target_valid, hz, min_periods=hz) == hz)
    eligible = eligible & isfinite(canonical_weight) & (canonical_weight > 0.0)
    for ready in readiness:
        eligible = eligible & ready
    eligible = fillna(eligible, 0.) != 0
    fit_x = cat(*(where(eligible, position, float("nan")) for position in positions))
    fit_y = where(eligible, canonical_y, float("nan"))
    fit_weight = where(eligible, canonical_weight, float("nan"))
    beta = get_beta(Ridge(fit_x, y=fit_y, weights=fit_weight, hl=RIDGE_HL,
                          lambda_=RIDGE_LAMBDA))
    fit_count = cumsum(where(reduce_max(eligible, axis=[1]), 1.0, 0.0))
    if beta_override is None:
        yhat = where(fit_count >= RIDGE_MIN_PERIODS,
                     einsum(cat(*scaled), beta, "if,f->i"), float("nan"))
    else:
        coefficients = np.broadcast_to(np.asarray(beta_override, dtype=float), (len(scaled),))
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("beta_override must contain finite coefficients")
        # Zero coefficients must not turn an excluded NaN feature into NaN.
        terms = [float(b) * x for b, x in zip(coefficients, scaled) if b != 0]
        yhat = sum(terms) if terms else where(isfinite(sigma), 0.0, float("nan"))
    return {
        "x": fit_x, "y": fit_y, "weights": fit_weight, "eligible": eligible,
        "canonical_x": canonical_x, "canonical_y": canonical_y,
        "canonical_weights": canonical_weight, "count": fit_count, "beta": beta,
        "yhat": yhat, "alpha_hat": yhat / sigma,
    }


def _formula(returns=None, *, features=None, fit_weights=None, volatility=None,
             beta_override=None):
    returns = RollRets().roll_rets() if returns is None else returns
    tradable = fillna(purify(var("is_tradable_out0")), 0.0)
    hs = var("vw_halfspread_out0")
    fit_weights = (where(fillna(isfinite(hs) & (hs > 0), 0.) != 0, 1 / hs**2, 0.)
                   if fit_weights is None else fit_weights)
    session = _session_plan(tradable, hs)
    signal_returns = _signal_returns(returns)
    feature_list = tuple(
        xs_gauss(xs_rank(-ts_zscore(signal_returns, span)))
        for span in FEATURE_SPANS
    ) if features is None else tuple(features)
    if not feature_list:
        raise ValueError("at least one feature is required")
    feature_names = (tuple(f"span_{span}" for span in FEATURE_SPANS)
                     if features is None else tuple(f"feature_{i}" for i in range(len(feature_list))))
    features = cat(*feature_list)
    sigma = ewm_std(signal_returns, span=IC_VOL_SPAN) if volatility is None else volatility
    gap_risk = _gap_risk(returns, tradable, session)
    ordinary_valid = isfinite(gap_risk["ordinary"])
    calibration_valid = fillna(ordinary_valid & (dsl_abs(returns) <= 0.05), 0.) != 0

    forecasts, factors, gap_factors = [], [], []
    adjustable = 0.0
    alpha_pnl, yhat_pnl, fits, yhats = {}, {}, {}, {}
    for h, (start, end) in enumerate(zip(TRADE_STARTS, HORIZONS)):
        width = end - start
        block_return = rolling_sum(fillna(gap_risk["ordinary"], 0.), width, min_periods=width)
        block_observed = rolling_sum(where(ordinary_valid, 1., 0.), width, min_periods=width)

        # Learn an ordinary-bar return rate; gap events have a separate risk model.
        fit = _forecast(feature_list, sigma=sigma, returns=signal_returns,
                        tradable=tradable, w=fit_weights, lag=start, hz=width,
                        target_valid=where(calibration_valid, 1.0, 0.0),
                        beta_override=beta_override)
        yhat = fit["yhat"]
        forecasts.append(fillna(yhat * session["regular_counts"][h], 0.0))
        horizon_key = f"{start}_{end}"
        fits[horizon_key] = fit
        yhats[horizon_key] = yhat
        alpha_diagnostics = [
            _diagnostic_pnls(
                feature,
                roll_rets=signal_returns,
                is_tradable=tradable,
                w=fit_weights,
                lag=start,
                hz=width,
            )
            for feature in feature_list
        ]
        alpha_pnl[horizon_key] = {
            "ic": {
                name: diagnostic["ic"]
                for name, diagnostic in zip(feature_names, alpha_diagnostics)
            },
            "ic1": {
                name: diagnostic["ic1"]
                for name, diagnostic in zip(feature_names, alpha_diagnostics)
            },
        }
        yhat_pnl[horizon_key] = _diagnostic_pnls(
            fit["alpha_hat"],
            roll_rets=signal_returns,
            is_tradable=tradable,
            w=fit_weights,
            lag=start,
            hz=width,
        )

        # Latest fully observed ordinary block, not a delayed target that keeps
        # arriving while trading is closed. Zero/missing/closure are distinct.
        risk_sample = where(fillna(block_observed == width, 0.) != 0,
                            block_return, float("nan"))
        covariance = risk_covariance(risk_sample, span=RISK_SPAN,
                                     min_periods=RISK_MIN_PERIODS,
                                     ignore_na=True, adjust=False)
        full_factor = psd_factor(fillna(covariance, 0.0), eigenvalue_floor=1e-8)
        # Partial-session scaling is an explicit homogeneous ordinary-risk
        # approximation. A whole reopening event is added separately, unscaled.
        scale = (session["regular_counts"][h] / width) ** .5
        adjustable = maximum(adjustable, session["allowed"][h])
        reserve = where(reduce_min(adjustable, axis=[1]) != 0,
                        RISK_HEADROOM, 1.0)
        factors.append(einsum(full_factor, scale, "ij,j->ij") / reserve)
        gap_factors.append(gap_risk["planned_factors"][h] / reserve)

    forecast_matrix = cat(*forecasts)
    mpo = MPO(
        expected_returns=forecast_matrix,
        half_spread=session["half_spread"],
        current_weights=previous_solution("weights[0]", initial=0.0),
        last_weights=previous_solution("previous_weights", initial=0.0),
        execution_allowed=session["execution_allowed"],
        gap_volatility=cat(*gap_factors),
        risk_factor_0=factors[0],
        risk_factor_1=factors[1],
        risk_factor_2=factors[2],
        risk_factor_3=factors[3],
        risk_factor_4=factors[4],
        risk_factor_5=factors[5],
        risk_factor_6=factors[6],
        trade_allowed=session["trade_allowed"],
        risk_radius=RISK_RADIUS,
    )
    # The solve at t plans a first fill at t+1. Its previous_weights variable
    # reconstructs the actual fill of yesterday's plan (or holds a halted lane).
    weights = get_field(mpo, "previous_weights")
    planned_weights = get_field(mpo, "weights[0]")
    status = get_field(mpo, "status")
    mpo_objective = get_field(mpo, "objective")
    risk_values = {f"{start}_{end}": get_field(mpo, f"risk_{h}.value")
                   for h, (start, end) in enumerate(zip(TRADE_STARTS, HORIZONS))}
    # Every finite economic return is marked, even when no new trade is possible.
    realized_returns = where(isfinite(returns), returns, 0.)
    mpo_gross_pnl = (fillna(shift(weights, 1), 0.) * realized_returns).sum(axis=[1])
    turnover = where(session["execution_allowed"] != 0,
                     dsl_abs(weights - get_field(mpo, "previous_actual")), 0.)
    trading_cost = (turnover * session["half_spread"]).sum(axis=[1])

    return {
        "returns": returns,
        "features": features,
        "signal_returns": signal_returns,
        "ordinary_returns": gap_risk["ordinary"],
        "gap_event": gap_risk["event"],
        "gap_minutes": gap_risk["minutes"],
        "gap_sigma": gap_risk["sigma"],
        "planned_gap_sigma": gap_risk["planned_sigma"],
        "volatility": sigma,
        "scaled_features": cat(*(feature * sigma for feature in feature_list)),
        "fit": fits,
        "yhat": yhats,
        "expected_returns": forecast_matrix,
        "weights": weights,
        "planned_weights": planned_weights,
        "trade_allowed": session["trade_allowed"],
        "execution_allowed": session["execution_allowed"],
        "regular_counts": cat(*session["regular_counts"]),
        "status": status,
        "mpo_objective": mpo_objective,
        "mpo_gross_pnl": mpo_gross_pnl,
        "mpo_trading_cost": trading_cost,
        "mpo_net_pnl": mpo_gross_pnl - trading_cost,
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
    ax.plot(index, _cum(values["mpo_net_pnl"]), label="net")
    ax.set_title("Implemented MPO portfolio PnL")
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
    ax.set_ylabel("Ordinary + gap risk, including planning reserve")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()
    path = plot_dir / "risk_constraint.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def _run(data, *, returns=None, output_dir=CACHE, **forecast_options):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_assets = data["is_tradable_out0"].shape[1]
    runtime = compile_formula(
        _formula(returns, **forecast_options),
        data,
        n_instruments=n_assets,
    )

    generated = runtime.generated_cpp.read_text()
    row_loop = "for (std::size_t t = row_begin; t < row_end; ++t)"
    assert generated.count(row_loop) == 1
    assert generated.count("stackdsl::ClarabelNode<") == 1

    result = runtime.run(out_path=output_dir / "result.npy")
    values = result.load()
    status = np.asarray(values["status"])
    failed = np.flatnonzero(~np.isin(status, (1., 4.)))
    if failed.size:
        row = int(failed[0])
        raise RuntimeError(f"MPO failed at row {row} with status {status[row]}; "
                           "no valid portfolio trajectory exists after this row")
    if not any(np.any(np.isfinite(v)) for v in values["yhat"].values()):
        raise RuntimeError("No fitted forecasts: provide more warmup rows or explicitly "
                           "configure shorter volatility/fit spans")
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
