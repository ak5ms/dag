from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src/trading_dsl_engine/cpp_stream/optimizer/templates/direct_clarabel_instance.hpp.j2"
OUT = Path(os.environ.get("MPO_SPEED_OUT", "/dev/shm/mpo_clarabel_speed"))
ROWS = int(os.environ.get("MPO_SPEED_ROWS", "5000"))
RUNS = int(os.environ.get("MPO_SPEED_RUNS", "5"))
N_ASSETS = 3
MINUTE_US = 60_000_000.0
SESSION_ROWS = 1000
OPEN_ROWS = 980

VARIANTS = (
    "baseline_diag",
    "baseline_nodiag",
    "no_refine_diag",
    "no_equil_diag",
    "refine1_diag",
    "equil1_diag",
    "no_refine_no_equil_diag",
    "tol1e7_diag",
    "tol1e6_diag",
    "tol1e5_diag",
    "no_refine_tol1e6_diag",
    "no_refine_no_equil_tol1e6_diag",
    "qdldl_diag",
    "presolve_diag",
    "rebuild_diag",
    "no_refine_nodiag",
    "no_refine_no_equil_nodiag",
    "tol1e6_nodiag",
    "no_refine_tol1e6_nodiag",
)


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
        "returns": np.ascontiguousarray(returns),
        "is_tradable_out0": np.ascontiguousarray(tradable),
        "vw_halfspread_out0": np.ascontiguousarray(rng.uniform(3e-5, 8e-5, size=(rows, n_assets))),
        "_ev_ts": lanes(ts1),
        "session_start0": lanes(session_start1),
        "session_end0": lanes(session_end1),
        "next_session_start0": lanes(next_session_start1),
        "next_session_end0": lanes(next_session_end1),
    }


def _upstream_expressions():
    from examples import cpp_stream_mpo_one_pass as example
    from flows.riskmodel import risk_covariance
    from flows.utils import ts_zscore
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

    returns = var("returns")
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

    forecasts, factors = [], []
    for start, end in zip(example.TRADE_STARTS, example.HORIZONS):
        width = end - start
        block_return = rolling_sum(clean_returns, width, min_periods=width)
        block_observed = rolling_sum(tradable, width, min_periods=width)
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
            Ridge(fit_x, y=target, weights=fit_weights, hl=example.RIDGE_HL, lambda_=0.1)
        )
        yhat = einsum(features, beta, "if,f->i")
        forecasts.append(fillna(yhat, 0.0))

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
        factors.append(psd_factor(fillna(covariance, 0.0), eigenvalue_floor=1e-8))

    return [
        cat(*forecasts),
        fillna(purify(hs), 0.0),
        *factors,
        example._planned_trade_allowed(tradable),
    ]


def _materialize_upstream(path: Path) -> None:
    from trading_dsl_engine.cpp_stream import compile_formula

    data = _fake_data()
    runtime = compile_formula(_upstream_expressions(), data, n_instruments=N_ASSETS)
    result = runtime.run(out_path=OUT / "upstream_result.npy")
    values = result.load()
    payload = {
        "expected_returns": np.ascontiguousarray(values[0]),
        "half_spread": np.ascontiguousarray(values[1]),
        "trade_allowed": np.ascontiguousarray(values[10]),
    }
    for h in range(8):
        payload[f"risk_factor_{h}"] = np.ascontiguousarray(values[2 + h])
    np.savez(path, **payload)
    print(f"UPSTREAM native_s={result.seconds:.6f}", flush=True)


def _patch_template(original: str, variant: str) -> str:
    base = variant.removesuffix("_diag").removesuffix("_nodiag")
    text = original
    anchor = "    settings_.presolve_enable = false;\n"
    extra: list[str] = []
    if "no_refine" in base:
        extra.append("    settings_.iterative_refinement_enable = false;")
    if "no_equil" in base:
        extra.append("    settings_.equilibrate_enable = false;")
    if base == "refine1":
        extra.append("    settings_.iterative_refinement_max_iter = 1;")
    if base == "equil1":
        extra.append("    settings_.equilibrate_max_iter = 1;")
    if "tol1e7" in base:
        extra += [
            "    settings_.tol_gap_abs = 1e-7;",
            "    settings_.tol_gap_rel = 1e-7;",
            "    settings_.tol_feas = 1e-7;",
        ]
    if "tol1e6" in base:
        extra += [
            "    settings_.tol_gap_abs = 1e-6;",
            "    settings_.tol_gap_rel = 1e-6;",
            "    settings_.tol_feas = 1e-6;",
        ]
    if "tol1e5" in base:
        extra += [
            "    settings_.tol_gap_abs = 1e-5;",
            "    settings_.tol_gap_rel = 1e-5;",
            "    settings_.tol_feas = 1e-5;",
        ]
    if base == "qdldl":
        extra.append("    settings_.direct_solve_method = QDLDL;")
    if base == "presolve":
        text = text.replace(anchor, "    settings_.presolve_enable = true;\n", 1)
        anchor = "    settings_.presolve_enable = true;\n"
    if extra:
        text = text.replace(anchor, anchor + "\n".join(extra) + "\n", 1)
    if base == "rebuild":
        text = text.replace(
            "    if (solver_settings_dirty_) reset_solver();\n",
            "    if (solver_ != nullptr) reset_solver();\n",
            1,
        )
    return text


def _optimizer_expressions(request_risk_values: bool):
    from examples import cpp_stream_mpo_one_pass as example
    from trading_dsl_engine.base.dsl import var
    from trading_dsl_engine.cpp_stream.optimizer import get_field, previous_solution

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
    expressions = [get_field(mpo, f"weights[{h}]") for h in range(8)]
    expressions += [
        get_field(mpo, "objective"),
        get_field(mpo, "iterations"),
        get_field(mpo, "status"),
        get_field(mpo, "primal_residual"),
        get_field(mpo, "dual_residual"),
    ]
    if request_risk_values:
        expressions += [get_field(mpo, f"risk_{h}.value") for h in range(8)]
    return expressions


def _child() -> None:
    from trading_dsl_engine.cpp_stream import compile_formula

    variant = os.environ["MPO_CHILD_VARIANT"]
    request_risk = variant.endswith("_diag")
    source = np.load(OUT / "upstream.npz")
    data = {name: np.ascontiguousarray(source[name]) for name in source.files}

    cache = ROOT / ".generated/cpp_stream_mpo_one_pass/clarabel"
    shutil.rmtree(cache, ignore_errors=True)
    t0 = time.perf_counter()
    runtime = compile_formula(_optimizer_expressions(request_risk), data, n_instruments=N_ASSETS)
    compile_s = time.perf_counter() - t0

    warmup = runtime.run(out_path=OUT / f"{variant}_warmup.npy")
    walls: list[float] = []
    natives: list[float] = []
    result = warmup
    for i in range(RUNS):
        t0 = time.perf_counter()
        result = runtime.run(out_path=OUT / f"{variant}_{i}.npy")
        walls.append(time.perf_counter() - t0)
        natives.append(result.seconds)

    values = result.load()
    weights = np.stack([np.asarray(values[h]) for h in range(8)], axis=1)
    np.savez(
        OUT / f"{variant}.npz",
        weights=weights,
        objective=np.asarray(values[8]),
        iterations=np.asarray(values[9]),
        status=np.asarray(values[10]),
        primal_residual=np.asarray(values[11]),
        dual_residual=np.asarray(values[12]),
    )
    metrics = {
        "variant": variant,
        "compile_s": compile_s,
        "warmup_native_s": warmup.seconds,
        "mean_wall_s": statistics.mean(walls),
        "median_wall_s": statistics.median(walls),
        "mean_native_s": statistics.mean(natives),
        "median_native_s": statistics.median(natives),
        "rows_per_s": ROWS / statistics.mean(natives),
        "mean_iterations": float(np.nanmean(np.asarray(values[9], dtype=float))),
        "max_iterations": float(np.nanmax(np.asarray(values[9], dtype=float))),
        "max_primal_residual": float(np.nanmax(np.asarray(values[11], dtype=float))),
        "max_dual_residual": float(np.nanmax(np.asarray(values[12], dtype=float))),
    }
    (OUT / f"{variant}.json").write_text(json.dumps(metrics, sort_keys=True))
    print("VARIANT " + json.dumps(metrics, sort_keys=True), flush=True)


def _feasibility(candidate: dict[str, np.ndarray], upstream) -> dict[str, float]:
    from examples import cpp_stream_mpo_one_pass as example

    w = np.asarray(candidate["weights"], dtype=float)
    mu = np.asarray(upstream["expected_returns"], dtype=float).transpose(0, 2, 1)
    hs = np.asarray(upstream["half_spread"], dtype=float)
    allowed = np.asarray(upstream["trade_allowed"], dtype=float).transpose(0, 2, 1)
    previous = np.vstack([np.zeros((1, N_ASSETS)), w[:-1, 0]])
    delta = np.empty_like(w)
    delta[:, 0] = w[:, 0] - previous
    delta[:, 1:] = w[:, 1:] - w[:, :-1]

    nmv = float(np.nanmax(np.abs(delta.sum(axis=2))))
    trade = float(np.nanmax(np.maximum(np.abs(delta) - example.TRADE_BIG_M * allowed, 0.0)))
    risk = 0.0
    for h in range(8):
        factor = np.asarray(upstream[f"risk_factor_{h}"], dtype=float)
        realized = np.einsum("tij,tj->ti", factor, w[:, h])
        risk = max(risk, float(np.nanmax(np.maximum(np.linalg.norm(realized, axis=1) - example.RISK_RADIUS, 0.0))))
    objective = -np.sum(mu * w, axis=(1, 2)) + np.sum(hs[:, None, :] * np.abs(delta), axis=(1, 2))
    solver_objective = np.asarray(candidate["objective"], dtype=float).reshape(-1)
    objective_gap = float(np.nanmax(np.abs(objective - solver_objective)))
    return {
        "nmv_violation": nmv,
        "trade_violation": trade,
        "risk_violation": risk,
        "objective_recompute_gap": objective_gap,
    }


def _parent() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from examples import cpp_stream_mpo_one_pass as example

    # Build the pinned native Clarabel library once; all child variants reuse it.
    t0 = time.perf_counter()
    example._clarabel()
    print(f"CLARABEL_SETUP seconds={time.perf_counter() - t0:.6f}", flush=True)
    _materialize_upstream(OUT / "upstream.npz")

    original = TEMPLATE.read_text()
    try:
        for variant in VARIANTS:
            TEMPLATE.write_text(_patch_template(original, variant))
            env = os.environ.copy()
            env["MPO_CHILD_VARIANT"] = variant
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve())],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            print(completed.stdout, end="", flush=True)
            if completed.returncode != 0:
                print(f"VARIANT_FAILED name={variant} returncode={completed.returncode}", flush=True)
    finally:
        TEMPLATE.write_text(original)

    baseline = dict(np.load(OUT / "baseline_diag.npz"))
    upstream = np.load(OUT / "upstream.npz")
    base_metrics = json.loads((OUT / "baseline_diag.json").read_text())
    for variant in VARIANTS:
        metrics_path = OUT / f"{variant}.json"
        values_path = OUT / f"{variant}.npz"
        if not metrics_path.exists() or not values_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        candidate = dict(np.load(values_path))
        max_weight_abs = float(np.nanmax(np.abs(candidate["weights"] - baseline["weights"])))
        max_obj_abs = float(np.nanmax(np.abs(candidate["objective"] - baseline["objective"])))
        denom = np.maximum(np.abs(np.asarray(baseline["objective"], dtype=float)), 1e-12)
        max_obj_rel = float(np.nanmax(np.abs(candidate["objective"] - baseline["objective"]) / denom))
        status_equal = bool(np.array_equal(candidate["status"], baseline["status"]))
        feasibility = _feasibility(candidate, upstream)
        speedup = base_metrics["mean_native_s"] / metrics["mean_native_s"]
        strict_ok = (
            status_equal
            and max_weight_abs <= 1e-5
            and max_obj_abs <= 1e-7
            and feasibility["nmv_violation"] <= 1e-7
            and feasibility["trade_violation"] <= 1e-7
            and feasibility["risk_violation"] <= 1e-7
            and feasibility["objective_recompute_gap"] <= 1e-7
        )
        summary = {
            **metrics,
            **feasibility,
            "speedup_vs_baseline": speedup,
            "max_weight_abs_vs_baseline": max_weight_abs,
            "max_objective_abs_vs_baseline": max_obj_abs,
            "max_objective_rel_vs_baseline": max_obj_rel,
            "status_equal": status_equal,
            "strict_ok": strict_ok,
        }
        print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    if os.environ.get("MPO_CHILD_VARIANT"):
        _child()
    else:
        _parent()
