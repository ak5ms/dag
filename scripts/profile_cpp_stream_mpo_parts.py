from __future__ import annotations

import os
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

from examples import cpp_stream_mpo_one_pass as example
from flows.riskmodel import risk_covariance
from flows.utils import ewm_std, ts_zscore
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
from trading_dsl_engine.cpp_stream.optimizer import get_field, previous_solution

ROWS = int(os.environ.get("MPO_PROFILE_ROWS", "5000"))
RUNS = int(os.environ.get("MPO_PROFILE_RUNS", "5"))
N_ASSETS = 3
OUT = Path(os.environ.get("MPO_PROFILE_OUTPUT_DIR", "/dev/shm/mpo_parts_profile"))
MINUTE_US = 60_000_000.0
SESSION_ROWS = 1000
OPEN_ROWS = 980


def _fake_data(rows: int = ROWS, n_assets: int = N_ASSETS) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    base = 1_800_000_000_000_000.0
    row = np.arange(rows)
    ts1 = base + row * MINUTE_US
    session_id = row // SESSION_ROWS
    session_start1 = base + session_id * SESSION_ROWS * MINUTE_US
    session_end1 = session_start1 + OPEN_ROWS * MINUTE_US
    next_session_start1 = session_start1 + SESSION_ROWS * MINUTE_US
    next_session_end1 = next_session_start1 + OPEN_ROWS * MINUTE_US
    tradable1 = (ts1 >= session_start1) & (ts1 < session_end1)

    returns = rng.normal(scale=2e-4, size=(rows, n_assets))
    tradable = np.broadcast_to(tradable1[:, None], (rows, n_assets)).astype(float).copy()
    returns[tradable == 0.0] = 0.0
    for reopen in range(SESSION_ROWS, rows, SESSION_ROWS):
        returns[reopen] *= np.sqrt(SESSION_ROWS - OPEN_ROWS + 1.0)

    def lanes(x):
        return np.broadcast_to(x[:, None], (rows, n_assets)).astype(float).copy()

    return {
        "returns": returns,
        "is_tradable_out0": tradable,
        "vw_halfspread_out0": rng.uniform(3e-5, 8e-5, size=(rows, n_assets)),
        "_ev_ts": lanes(ts1),
        "session_start0": lanes(session_start1),
        "session_end0": lanes(session_end1),
        "next_session_start0": lanes(next_session_start1),
        "next_session_end0": lanes(next_session_end1),
    }


def _parts(returns):
    tradable = fillna(var("is_tradable_out0"), 0.0)
    hs = var("vw_halfspread_out0")
    fit_weights = purify(1 / hs**2)
    feature_list = tuple(
        ts_zscore(
            returns,
            example._feature_span(hl),
            min_periods=max(2, round(example._feature_span(hl))),
        )
        for hl in example.FEATURE_HLS
    )
    features = cat(*feature_list)
    clean_returns = where(tradable != 0, fillna(returns, 0.0), 0.0)

    block_returns = []
    forecasts = []
    yhat_signals = []
    covariances = []
    factors = []
    for start, end in zip(example.TRADE_STARTS, example.HORIZONS):
        width = end - start
        block_return = rolling_sum(clean_returns, width, min_periods=width)
        block_observed = rolling_sum(tradable, width, min_periods=width)
        block_returns.append(block_return)

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
                hl=example.RIDGE_HL,
                lambda_=0.1,
            )
        )
        yhat = einsum(features, beta, "if,f->i")
        forecasts.append(fillna(yhat, 0.0))
        yhat_signals.append(
            purify(
                yhat
                / ewm_std(
                    yhat,
                    example.YHAT_VOL_SPAN,
                    min_periods=example.YHAT_VOL_MIN_PERIODS,
                )
            )
        )

        risk_block = shift(block_return, start)
        risk_observed = shift(block_observed, start)
        risk_sample = where(risk_observed > 0, risk_block, float("nan"))
        covariance = risk_covariance(
            risk_sample,
            span=example.RISK_SPAN,
            min_periods=example.RISK_MIN_PERIODS,
            ignore_na=True,
            adjust=False,
        )
        covariances.append(covariance)
        factors.append(
            psd_factor(fillna(covariance, 0.0), eigenvalue_floor=1e-8)
        )

    return {
        "tradable": tradable,
        "hs": hs,
        "feature_list": feature_list,
        "features": features,
        "block_returns": tuple(block_returns),
        "forecasts": tuple(forecasts),
        "yhat_signals": tuple(yhat_signals),
        "covariances": tuple(covariances),
        "factors": tuple(factors),
        "trade_allowed": example._planned_trade_allowed(tradable),
    }


def _bench(name: str, expressions, data, *, collect=False):
    stage_dir = OUT / name
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    runtime = compile_formula(list(expressions), data, n_instruments=N_ASSETS)
    compile_s = time.perf_counter() - t0

    warmup = runtime.run(out_path=stage_dir / "warmup.npy")
    wall_times = []
    native_times = []
    result = warmup
    for i in range(RUNS):
        t0 = time.perf_counter()
        result = runtime.run(out_path=stage_dir / f"run_{i}.npy")
        wall_times.append(time.perf_counter() - t0)
        native_times.append(result.seconds)

    mean_wall = statistics.mean(wall_times)
    mean_native = statistics.mean(native_times)
    median_wall = statistics.median(wall_times)
    median_native = statistics.median(native_times)
    print(
        "PROFILE "
        f"name={name} compile_s={compile_s:.6f} warmup_native_s={warmup.seconds:.6f} "
        f"mean_wall_s={mean_wall:.6f} median_wall_s={median_wall:.6f} "
        f"mean_native_s={mean_native:.6f} median_native_s={median_native:.6f} "
        f"native_rows_per_s={ROWS / mean_native:.3f}"
    )
    return result.load() if collect else None


def _optimizer_expressions():
    mpo = example.MPO(
        expected_returns=var("expected_returns"),
        half_spread=var("half_spread"),
        current_weights=previous_solution("weights[0]", initial=0.0),
        risk_factor_0=var("risk_factor_0"),
        risk_factor_1=var("risk_factor_1"),
        risk_factor_2=var("risk_factor_2"),
        risk_factor_3=var("risk_factor_3"),
        risk_factor_4=var("risk_factor_4"),
        risk_factor_5=var("risk_factor_5"),
        risk_factor_6=var("risk_factor_6"),
        risk_factor_7=var("risk_factor_7"),
        trade_allowed=var("trade_allowed"),
        risk_radius=example.RISK_RADIUS,
    )
    return [
        get_field(mpo, "weights[0]"),
        *(get_field(mpo, f"risk_{h}.value") for h in range(len(example.HORIZONS))),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = _fake_data()
    parts = _parts(var("returns"))

    _bench("input_baseline", [var("returns")], data)
    _bench("features_4x_zscore", parts["feature_list"], data)
    _bench("block_returns_8x", parts["block_returns"], data)
    _bench("forecasts_8x_ridge_yhat", [cat(*parts["forecasts"])], data)
    _bench("forecast_diagnostics_8x_ewmstd", [cat(*parts["yhat_signals"])], data)
    _bench("risk_covariance_8x", parts["covariances"], data)
    _bench("risk_psd_factors_8x", parts["factors"], data)
    _bench("session_trade_mask", [parts["trade_allowed"]], data)

    upstream_values = _bench(
        "upstream_parameters",
        [
            cat(*parts["forecasts"]),
            fillna(purify(parts["hs"]), 0.0),
            *parts["factors"],
            parts["trade_allowed"],
        ],
        data,
        collect=True,
    )
    expected_returns = np.ascontiguousarray(upstream_values[0])
    half_spread = np.ascontiguousarray(upstream_values[1])
    risk_factors = tuple(np.ascontiguousarray(x) for x in upstream_values[2:10])
    trade_allowed = np.ascontiguousarray(upstream_values[10])
    optimizer_data = {
        "expected_returns": expected_returns,
        "half_spread": half_spread,
        **{f"risk_factor_{h}": risk_factors[h] for h in range(8)},
        "trade_allowed": trade_allowed,
    }
    _bench("mpo_only_exact_params", _optimizer_expressions(), optimizer_data)

    _bench("full_current_formula", example._formula(var("returns")), data)


if __name__ == "__main__":
    main()
