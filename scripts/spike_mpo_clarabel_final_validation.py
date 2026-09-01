from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import cvxpy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src/trading_dsl_engine/cpp_stream/optimizer/templates/direct_clarabel_instance.hpp.j2"
OUT = Path(os.environ.get("MPO_FINAL_OUT", "/dev/shm/mpo_clarabel_final"))
ROWS = int(os.environ.get("MPO_FINAL_ROWS", "5000"))
RUNS = int(os.environ.get("MPO_FINAL_RUNS", "10"))
N_ASSETS = 3
MINUTE_US = 60_000_000.0
SESSION_ROWS = 1000
OPEN_ROWS = 980

VARIANTS = (
    "baseline_nodiag",
    "refine1_nodiag",
    "no_refine_nodiag",
    "no_refine_tol1e6_nodiag",
    "baseline_diag",
    "no_refine_diag",
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
    print("SHAPES " + json.dumps({k: list(v.shape) for k, v in payload.items()}), flush=True)
    print(f"UPSTREAM native_s={result.seconds:.6f}", flush=True)


def _patch_template(original: str, variant: str) -> str:
    base = variant.removesuffix("_diag").removesuffix("_nodiag")
    text = original
    anchor = "    settings_.presolve_enable = false;\n"
    extra: list[str] = []
    if base == "refine1":
        extra.append("    settings_.iterative_refinement_max_iter = 1;")
    if "no_refine" in base:
        extra.append("    settings_.iterative_refinement_enable = false;")
    if "tol1e6" in base:
        extra += [
            "    settings_.tol_gap_abs = 1e-6;",
            "    settings_.tol_gap_rel = 1e-6;",
            "    settings_.tol_feas = 1e-6;",
        ]
    if extra:
        text = text.replace(anchor, anchor + "\n".join(extra) + "\n", 1)
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
    expressions += [get_field(mpo, "objective"), get_field(mpo, "iterations"), get_field(mpo, "status")]
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
    walls, natives = [], []
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
        "solved_fraction": float(np.mean(np.asarray(values[10]).reshape(-1) == 1)),
        "almost_solved_fraction": float(np.mean(np.asarray(values[10]).reshape(-1) == 4)),
    }
    (OUT / f"{variant}.json").write_text(json.dumps(metrics, sort_keys=True))
    print("VARIANT " + json.dumps(metrics, sort_keys=True), flush=True)


def _shape(shape):
    return SimpleNamespace(shape=shape)


def _problem():
    from examples import cpp_stream_mpo_one_pass as example

    h = len(example.HORIZONS)
    n = N_ASSETS
    return example.MPO.factory(
        expected_returns=_shape((h, n)),
        half_spread=_shape((n,)),
        current_weights=_shape((n,)),
        risk_factor_0=_shape((n, n)),
        risk_factor_1=_shape((n, n)),
        risk_factor_2=_shape((n, n)),
        risk_factor_3=_shape((n, n)),
        risk_factor_4=_shape((n, n)),
        risk_factor_5=_shape((n, n)),
        risk_factor_6=_shape((n, n)),
        risk_factor_7=_shape((n, n)),
        trade_allowed=_shape((h, n)),
    )


def _cp_row(name: str, row: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(row, dtype=float)
    if len(shape) <= 1:
        if value.shape != shape:
            raise AssertionError((name, value.shape, shape))
        return value
    # Generated optimizer parameters expose the reverse of CVXPY's shape as
    # their DSL logical shape.  Transpose every rank-2 row back to CVXPY order,
    # including square risk factors where shape alone cannot reveal this.
    value = value.T
    if value.shape != shape:
        raise AssertionError((name, value.shape, shape))
    return value


def _selected_rows() -> np.ndarray:
    grid = np.linspace(0, ROWS - 1, 56, dtype=int)
    edges = []
    for session in range(1, ROWS // SESSION_ROWS + 1):
        reopen = session * SESSION_ROWS
        close = reopen - (SESSION_ROWS - OPEN_ROWS)
        for value in (close - 1, close, close + 1, reopen - 1, reopen, reopen + 1, reopen + 8, reopen + 64):
            if 0 <= value < ROWS:
                edges.append(value)
    fixed = [0, 1, 2, 63, 127, 128, ROWS - 1]
    return np.unique(np.concatenate([grid, np.asarray(edges + fixed, dtype=int)]))


def _validate_candidate(variant: str, upstream) -> dict[str, float | int]:
    candidate = np.load(OUT / f"{variant}.npz")
    weights = np.asarray(candidate["weights"], dtype=float)
    statuses = np.asarray(candidate["status"], dtype=float).reshape(-1)
    problem = _problem()
    params = {p.name(): p for p in problem.parameters()}
    variables = {v.name(): v for v in problem.variables()}
    wvar = variables["weights"]
    pvar = variables["previous_weights"]

    max_violation = 0.0
    max_obj_gap = 0.0
    median_gaps: list[float] = []
    max_first_stage_abs = 0.0
    rms_first_stage: list[float] = []
    ref_failures = 0
    candidate_bad_status = int(np.sum(~np.isin(statuses, [1.0, 4.0])))

    for t in _selected_rows():
        current = np.zeros(N_ASSETS) if t == 0 else weights[t - 1, 0]
        for name, p in params.items():
            if name == "current_weights":
                p.value = current
            elif name == "risk_radius":
                p.value = 0.08
            else:
                p.value = _cp_row(name, upstream[name][t], tuple(int(x) for x in p.shape))

        # Evaluate the native candidate directly in the original, unaugmented
        # CVXPY problem at the exact current state that candidate used.
        wvar.value = weights[t]
        pvar.value = current
        candidate_obj = float(problem.objective.value)
        for constraint in problem.constraints:
            violation = np.asarray(constraint.violation(), dtype=float)
            if violation.size:
                max_violation = max(max_violation, float(np.nanmax(np.abs(violation))))

        try:
            ref_obj = problem.solve(
                solver=cp.CLARABEL,
                verbose=False,
                presolve_enable=False,
                tol_gap_abs=1e-9,
                tol_gap_rel=1e-9,
                tol_feas=1e-9,
            )
        except Exception:
            ref_failures += 1
            continue
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or ref_obj is None:
            ref_failures += 1
            continue
        gap = abs(candidate_obj - float(ref_obj))
        max_obj_gap = max(max_obj_gap, gap)
        median_gaps.append(gap)
        diff = weights[t, 0] - np.asarray(wvar.value[0], dtype=float)
        max_first_stage_abs = max(max_first_stage_abs, float(np.max(np.abs(diff))))
        rms_first_stage.append(float(np.sqrt(np.mean(diff**2))))

    result = {
        "validated_rows": int(len(_selected_rows()) - ref_failures),
        "reference_failures": int(ref_failures),
        "candidate_bad_status_rows": candidate_bad_status,
        "max_constraint_violation": max_violation,
        "max_objective_gap_to_1e9_reference": max_obj_gap,
        "median_objective_gap_to_1e9_reference": float(np.median(median_gaps)) if median_gaps else float("nan"),
        "max_first_stage_weight_abs_to_reference": max_first_stage_abs,
        "median_first_stage_weight_rms_to_reference": float(np.median(rms_first_stage)) if rms_first_stage else float("nan"),
    }
    print("VALIDATION " + variant + " " + json.dumps(result, sort_keys=True), flush=True)
    return result


def _full_formula_benchmark(original: str) -> None:
    from examples import cpp_stream_mpo_one_pass as example
    from trading_dsl_engine.base.dsl import var
    from trading_dsl_engine.cpp_stream import compile_formula

    TEMPLATE.write_text(_patch_template(original, "no_refine_diag"))
    shutil.rmtree(ROOT / ".generated/cpp_stream_mpo_one_pass/clarabel", ignore_errors=True)
    data = _fake_data()
    t0 = time.perf_counter()
    runtime = compile_formula(list(example._formula(var("returns"))), data, n_instruments=N_ASSETS)
    compile_s = time.perf_counter() - t0
    generated = runtime.generated_cpp.read_text()
    row_loop = "for (std::size_t t = row_begin; t < row_end; ++t)"
    assert generated.count(row_loop) == 1
    assert generated.count("stackdsl::ClarabelNode<") == 1
    warmup = runtime.run(out_path=OUT / "full_warmup.npy")
    times = []
    for i in range(RUNS):
        result = runtime.run(out_path=OUT / f"full_{i}.npy")
        times.append(result.seconds)
    payload = {
        "compile_s": compile_s,
        "warmup_native_s": warmup.seconds,
        "mean_native_s": statistics.mean(times),
        "median_native_s": statistics.median(times),
        "rows_per_s": ROWS / statistics.mean(times),
    }
    print("FULL_NO_REFINE " + json.dumps(payload, sort_keys=True), flush=True)


def _parent() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from examples import cpp_stream_mpo_one_pass as example

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
                raise RuntimeError(f"variant {variant} failed with {completed.returncode}")

        upstream = np.load(OUT / "upstream.npz")
        for variant in ("baseline_nodiag", "refine1_nodiag", "no_refine_nodiag", "no_refine_tol1e6_nodiag"):
            _validate_candidate(variant, upstream)
        _full_formula_benchmark(original)
    finally:
        TEMPLATE.write_text(original)


if __name__ == "__main__":
    if os.environ.get("MPO_CHILD_VARIANT"):
        _child()
    else:
        _parent()
