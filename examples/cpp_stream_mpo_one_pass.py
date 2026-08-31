"""InputData -> gap-aware Ridge forecasts -> sequential Clarabel MPO, in one loop."""

from __future__ import annotations

import os
from pathlib import Path

import cvxpy as cp

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
FEATURE_SPANS = (8, 32, 128, 512)
RIDGE_HL = 1440 * 21
RISK_SPAN = 1440 * 21
RISK_RADIUS = 0.08
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
    half_spread_bps,
    current_weights,
    risk_factors,
    is_tradable,
    risk_radius=RISK_RADIUS,
):
    n_horizons, n_assets = expected_returns.shape
    expected_returns = cp.Parameter(expected_returns.shape, name="expected_returns")
    half_spread_bps = cp.Parameter(
        half_spread_bps.shape, name="half_spread_bps", nonneg=True
    )
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
        cp.multiply(1 - is_tradable, weights[0] - current_weights) == 0,
    ]
    constraints += [
        cp.SOC(
            risk_radius,
            risk_factors[h * n_assets : (h + 1) * n_assets] @ weights[h],
        )
        for h in range(n_horizons)
    ]
    return cp.Problem(
        cp.Minimize(
            -cp.sum(cp.multiply(expected_returns, weights))
            + cp.sum(cp.multiply(half_spread_bps * 1e-4, turnover))
        ),
        constraints,
    )


def _formula():
    returns = RollRets().roll_rets()
    tradable = fillna(var("is_tradable_out0"), 0.0)
    hs = var("vw_halfspread_out0")
    fit_weights = purify(1 / hs**2)
    features = cat(*(ts_zscore(returns, span) for span in FEATURE_SPANS))

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
        # Divide by sqrt(elapsed) so a post-gap return is not treated as 1 minute.
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
        half_spread_bps=fillna(purify(hs), 0.0),
        current_weights=previous_solution("weights[0]", initial=0.0),
        # cat(NxN, ...) is logical (N,H*N); CVXPY sees (H*N,N).
        risk_factors=cat(*factors),
        is_tradable=tradable,
        risk_radius=RISK_RADIUS,
    )
    weights = get_field(mpo, "weights[0]")
    return shift(weights) * returns, weights, get_field(mpo, "objective")


def main() -> None:
    data = InputData(nrows=ROWS, idx=None).get_data()
    n_assets = data["is_tradable_out0"].shape[1]
    runtime = compile_formula(list(_formula()), data, n_instruments=n_assets)

    generated = runtime.generated_cpp.read_text()
    row_loop = "for (std::size_t t = row_begin; t < row_end; ++t)"
    assert generated.count(row_loop) == 1
    assert generated.count("stackdsl::ClarabelNode<") == 1

    result = runtime.run(out_path=CACHE / "result.npy")
    pnl, weights, objective = result.load()
    print(f"rows={result.rows:,} seconds={result.seconds:.3f}")
    print(f"pnl={pnl.shape} weights={weights.shape} objective={objective.shape}")


if __name__ == "__main__":
    main()
