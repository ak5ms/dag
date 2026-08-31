"""InputData -> gap-aware Ridge forecasts -> sequential Clarabel MPO, in one loop."""

from __future__ import annotations

import os
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flows.load import InputData
from flows.pov import RollRets
from flows.riskmodel import risk_covariance
from flows.utils import streak, ts_zscore
from trading_dsl_engine.base.dsl import (
    Ridge,
    cat,
    einsum,
    fillna,
    get_beta,
    psd_factor,
    purify,
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

HORIZONS = (1, 2, 4, 8, 16, 32, 64, 128)
FEATURE_HLS = (4, 16, 64, 256)
RIDGE_HL = 1440 * 21
RISK_SPAN = 1440 * 21
RISK_RADIUS = 0.08
TRADE_BIG_M = 1e3
ROWS = int(os.environ.get("MPO_EXAMPLE_ROWS", "20000"))
CACHE = Path(".generated/cpp_stream_mpo_one_pass")


def _clarabel() -> ClarabelNativePaths:
    include = os.environ.get("CLARABEL_INCLUDE_DIR")
    library = os.environ.get("CLARABEL_STATIC_LIBRARY")
    if include and library:
        return ClarabelNativePaths(Path(include), Path(library))
    return build_current_clarabel()


@cvxpy_program(cache_dir=CACHE / "clarabel", clarabel=_clarabel, sequential=None)
def MPO(
    expected_returns,
    half_spread,
    current_weights,
    risk_factors,
    is_tradable,
    risk_radius=RISK_RADIUS,
):
    n_horizons, n_assets = expected_returns.shape
    expected_returns = cp.Parameter(expected_returns.shape, name="expected_returns")
    half_spread = cp.Parameter(half_spread.shape, name="half_spread", nonneg=True)
    current_weights = cp.Parameter((n_assets,), name="current_weights")
    risk_factors = cp.Parameter(risk_factors.shape, name="risk_factors")
    is_tradable = cp.Parameter((n_assets,), name="is_tradable", nonneg=True)
    risk_radius = cp.Parameter(name="risk_radius", nonneg=True)

    weights = cp.Variable((n_horizons, n_assets), name="weights")
    turnover = cp.Variable((n_horizons, n_assets), name="turnover")
    delta = weights - cp.vstack([current_weights, weights[:-1]])
    constraints = [
        turnover >= delta,
        turnover >= -delta,
        weights[0] - current_weights <= TRADE_BIG_M * is_tradable,
        weights[0] - current_weights >= -TRADE_BIG_M * is_tradable,
    ]
    for h in range(n_horizons):
        risk = cp.SOC(
            risk_radius,
            risk_factors[h * n_assets : (h + 1) * n_assets] @ weights[h],
        )
        risk.set_label(f"risk_{h}")
        constraints.append(risk)
    return cp.Problem(
        cp.Minimize(
            -cp.sum(cp.multiply(expected_returns, weights))
            + cp.sum(cp.multiply(half_spread, turnover))
        ),
        constraints,
    )


def _formula():
    returns = RollRets().roll_rets()
    tradable = fillna(var("is_tradable_out0"), 0.0)
    hs = var("vw_halfspread_out0")
    fit_weights = purify(1 / hs**2)
    features = cat(
        *(
            ts_zscore(returns, 2 / (1 - 0.5 ** (1 / hl)) - 1)
            for hl in FEATURE_HLS
        )
    )

    # A return after k closed rows spans k+1 bars of elapsed risk time.
    elapsed = where(
        tradable != 0,
        fillna(shift(streak(tradable == 0)), 0.0) + 1.0,
        0.0,
    )
    clean_returns = where(tradable != 0, fillna(returns, 0.0), 0.0)

    forecasts, factors = [], []
    for start, end in zip((0,) + HORIZONS[:-1], HORIZONS):
        width = end - start
        block_return = rolling_sum(clean_returns, width, min_periods=width)
        block_elapsed = rolling_sum(elapsed, width, min_periods=width)

        # ic1 alignment for block (start, end]: lag=start, hz=width,
        # hence the fit feature is shifted by lag+hz=end.
        target = where(
            block_elapsed > 0,
            block_return / block_elapsed,
            float("nan"),
        )
        fit_x = where(
            shift(tradable, end) != 0,
            shift(features, end),
            float("nan"),
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
        forecasts.append(einsum(features, beta, "if,f->i") * width)

        # Disjoint historical risk blocks: (0,1], (1,2], (2,4], ... .
        risk_block = shift(block_return, start)
        risk_elapsed = shift(block_elapsed, start)
        risk_sample = where(
            risk_elapsed > 0,
            risk_block / risk_elapsed**0.5,
            float("nan"),
        )
        factors.append(
            psd_factor(
                risk_covariance(
                    risk_sample,
                    span=RISK_SPAN,
                    min_periods=64,
                    ignore_na=True,
                    adjust=False,
                ),
                eigenvalue_floor=1e-8,
            )
        )

    mpo = MPO(
        expected_returns=cat(*forecasts),
        half_spread=fillna(purify(hs), 0.0),
        current_weights=previous_solution("weights[0]", initial=0.0),
        # cat(NxN, ...) is logical (N,H*N); CVXPY sees (H*N,N).
        risk_factors=cat(*factors),
        is_tradable=tradable,
        risk_radius=RISK_RADIUS,
    )
    weights = tuple(get_field(mpo, f"weights[{h}]") for h in range(len(HORIZONS)))
    risks = tuple(get_field(mpo, f"risk_{h}.value") for h in range(len(HORIZONS)))
    return (returns, features, *weights, *risks)


def _purified_inverse_square(hs):
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / np.asarray(hs, dtype=float) ** 2
    return np.where(np.isfinite(w), w, np.nan)


def _normalize_rows(w, valid=None):
    w = np.asarray(w, dtype=float)
    if valid is not None:
        w = np.where(valid, w, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    total = w.sum(axis=1, keepdims=True)
    return np.divide(w, total, out=np.zeros_like(w), where=total != 0.0)


def _ic1_pnl(returns, signal, w, tradable, hs, *, lag, hz):
    """Exact ic1 gross PnL plus aligned half-spread turnover cost."""
    r = np.asarray(returns, dtype=float)
    s = np.asarray(signal, dtype=float)
    w = np.asarray(w, dtype=float)
    if w.ndim == 0:
        w = np.broadcast_to(w, r.shape).copy()
    mask = pd.DataFrame(np.asarray(tradable, dtype=float)).fillna(0.0).ne(0.0)
    position = pd.DataFrame(s).shift(lag).where(mask).ffill().fillna(0.0)

    valid = np.isfinite(r)
    weighted_return = np.where(valid, r, 0.0) * _normalize_rows(w, valid)
    gross = (
        pd.DataFrame(weighted_return)
        .rolling(hz, min_periods=hz)
        .mean()
        .mul(position.shift(hz))
        .sum(axis=1, min_count=1)
        .to_numpy()
    )

    # ic1 reports average return over hz bars. Charge exact half-spread turnover
    # at the position timestamp, then align it to the horizon endpoint / hz.
    held_w = pd.DataFrame(w).where(mask).ffill().fillna(0.0).to_numpy()
    holdings = position.to_numpy() * _normalize_rows(held_w)
    previous = np.vstack([np.zeros((1, holdings.shape[1])), holdings[:-1]])
    cost = np.sum(
        np.abs(holdings - previous) * np.nan_to_num(hs, nan=0.0), axis=1
    )
    net = gross - pd.Series(cost).shift(hz).to_numpy() / hz
    return gross, net


def _cum(x):
    return np.cumsum(np.where(np.isfinite(x), x, 0.0))


def _plot_diagnostics(data, returns, features, mpo_weights, risk_values):
    plot_dir = CACHE / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ts = np.asarray(data["_ev_ts"])
    ts = ts[:, 0] if ts.ndim > 1 else ts
    index = pd.to_datetime(pd.Series(ts).interpolate(), unit="us")
    hs = np.asarray(data["vw_halfspread_out0"], dtype=float)
    tradable = np.asarray(data["is_tradable_out0"], dtype=float)
    alpha_w = _purified_inverse_square(hs)
    paths = []

    for h, (start, end) in enumerate(zip((0,) + HORIZONS[:-1], HORIZONS)):
        hz = end - start

        fig, ax = plt.subplots(figsize=(10, 5))
        for j, hl in enumerate(FEATURE_HLS):
            gross, net = _ic1_pnl(
                returns,
                features[:, :, j],
                alpha_w,
                tradable,
                hs,
                lag=start,
                hz=hz,
            )
            ax.plot(index, _cum(gross), label=f"HL {hl} gross")
            ax.plot(index, _cum(net), "--", label=f"HL {hl} net")
        ax.set_title(f"Alpha PnL: horizon ({start}, {end}]")
        ax.set_ylabel("Cumulative ic1 PnL")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = plot_dir / f"alphas_horizon_{start}_{end}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(10, 5))
        gross, net = _ic1_pnl(
            returns,
            mpo_weights[h],
            1.0,
            tradable,
            hs,
            lag=start,
            hz=hz,
        )
        ax.plot(index, _cum(gross), label="gross")
        ax.plot(index, _cum(net), "--", label="net")
        ax.set_title(f"MPO PnL: horizon ({start}, {end}]")
        ax.set_ylabel("Cumulative ic1 PnL")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = plot_dir / f"mpo_horizon_{start}_{end}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    for (start, end), value in zip(
        zip((0,) + HORIZONS[:-1], HORIZONS), risk_values
    ):
        value = np.asarray(value, dtype=float)
        ax.plot(index, np.linalg.norm(value[:, 1:], axis=1), label=f"({start}, {end}]")
    ax.axhline(RISK_RADIUS, linestyle="--", label="constraint")
    ax.set_title("MPO risk constraint")
    ax.set_ylabel("sqrt(w' S w)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = plot_dir / "risk_constraint.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    data = InputData(nrows=ROWS, idx=None).get_data()
    n_assets = data["is_tradable_out0"].shape[1]
    runtime = compile_formula(list(_formula()), data, n_instruments=n_assets)

    generated = runtime.generated_cpp.read_text()
    row_loop = "for (std::size_t t = row_begin; t < row_end; ++t)"
    assert generated.count(row_loop) == 1
    assert generated.count("stackdsl::ClarabelNode<") == 1

    result = runtime.run(out_path=CACHE / "result.npy")
    values = result.load()
    returns, features = values[:2]
    mpo_weights = values[2 : 2 + len(HORIZONS)]
    risk_values = values[2 + len(HORIZONS) :]
    paths = _plot_diagnostics(data, returns, features, mpo_weights, risk_values)

    print(f"rows={result.rows:,} seconds={result.seconds:.3f}")
    print(f"plots={CACHE / 'plots'}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
