from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import cvxpy as cp
import numpy as np

from trading_dsl_engine.base.dsl import cat, einsum, var
from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.cpp_stream.optimizer import (
    build_current_clarabel,
    cvxpy_program,
    get_field,
    previous_solution,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "src/trading_dsl_engine/cpp_stream/optimizer/templates/direct_clarabel_instance.hpp.j2"
)
OUT = Path(os.environ.get("MPO_CONSTRAINT_OUT", "/dev/shm/mpo_constraint_output"))
ROWS = int(os.environ.get("MPO_CONSTRAINT_ROWS", "5000"))
RUNS = int(os.environ.get("MPO_CONSTRAINT_RUNS", "10"))
N_ASSETS = 3
N_HORIZONS = 8
RISK_RADIUS = 0.08
TRADE_BIG_M = 1e3
GAP_START = 980
GAP_END = 1000


def _data() -> dict[str, np.ndarray]:
    """Fixed-size optimizer inputs with one deliberately infeasible gap row."""

    rng = np.random.default_rng(417)
    row = np.arange(ROWS, dtype=float)
    alpha = np.array([1.0, -0.35, -0.65])
    expected_returns = np.empty((ROWS, N_ASSETS, N_HORIZONS), dtype=float)
    for horizon in range(N_HORIZONS):
        expected_returns[:, :, horizon] = (
            (2.0e-3 / np.sqrt(horizon + 1.0)) * alpha
            + 4.0e-5
            * rng.normal(size=(ROWS, N_ASSETS))
        )

    half_spread = rng.uniform(3e-5, 8e-5, size=(ROWS, N_ASSETS))
    trade_allowed = np.ones((ROWS, N_ASSETS, N_HORIZONS), dtype=float)
    trade_allowed[GAP_START:GAP_END, :, 0] = 0.0

    base = np.array(
        [
            [1.00, 0.22, -0.08],
            [0.04, 1.12, 0.16],
            [0.11, -0.06, 0.91],
        ],
        dtype=float,
    )
    data: dict[str, np.ndarray] = {
        "expected_returns": np.ascontiguousarray(expected_returns),
        "half_spread": np.ascontiguousarray(half_spread),
        "initial_weights": np.zeros((ROWS, N_ASSETS), dtype=float),
        "trade_allowed": np.ascontiguousarray(trade_allowed),
    }
    for horizon in range(N_HORIZONS):
        scale = 1.0 + 0.035 * np.sin(row / (37.0 + 3.0 * horizon))
        factor = scale[:, None, None] * base[None, :, :]
        factor = factor.copy()
        factor[:, 0, 1] += 0.015 * np.cos(row / (53.0 + horizon))
        if horizon == 0:
            # Stage zero is frozen at the first closed row. Tightening its risk
            # map makes the carried open-session portfolio mathematically
            # infeasible, allowing the raw failure-path output to be inspected.
            factor[GAP_START] *= 20.0
        data[f"risk_factor_{horizon}"] = np.ascontiguousarray(factor)
    return data


def _make_program(setting: str, mode: str):
    cache = OUT / f"program-{setting}-{mode}"
    shutil.rmtree(cache, ignore_errors=True)

    @cvxpy_program(
        cache_dir=cache,
        clarabel=build_current_clarabel,
        sequential=None,
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
        risk_factor_7,
        trade_allowed,
        risk_radius=RISK_RADIUS,
    ):
        n_horizons, n_assets = expected_returns.shape
        expected_returns = cp.Parameter(
            expected_returns.shape, name="expected_returns"
        )
        half_spread = cp.Parameter(
            half_spread.shape, name="half_spread", nonneg=True
        )
        current_weights = cp.Parameter((n_assets,), name="current_weights")
        risk_factors = tuple(
            cp.Parameter(argument.shape, name=f"risk_factor_{index}")
            for index, argument in enumerate(
                (
                    risk_factor_0,
                    risk_factor_1,
                    risk_factor_2,
                    risk_factor_3,
                    risk_factor_4,
                    risk_factor_5,
                    risk_factor_6,
                    risk_factor_7,
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
        constraints = [
            previous_weights == current_weights,
            cp.sum(delta, axis=1) == 0,
            abs_delta <= TRADE_BIG_M * trade_allowed,
        ]
        for horizon, risk_factor in enumerate(risk_factors):
            risk = cp.SOC(risk_radius, risk_factor @ weights[horizon])
            risk.set_label(f"risk_{horizon}")
            constraints.append(risk)
        return cp.Problem(
            cp.Minimize(
                -cp.sum(cp.multiply(expected_returns, weights))
                + cp.sum(cp.multiply(half_spread, abs_delta))
            ),
            constraints,
        )

    return MPO


def _bound_program(setting: str, mode: str):
    program = _make_program(setting, mode)
    mpo = program(
        expected_returns=var("expected_returns"),
        half_spread=var("half_spread"),
        current_weights=previous_solution(
            "weights[0]", initial=var("initial_weights")
        ),
        risk_factor_0=var("risk_factor_0"),
        risk_factor_1=var("risk_factor_1"),
        risk_factor_2=var("risk_factor_2"),
        risk_factor_3=var("risk_factor_3"),
        risk_factor_4=var("risk_factor_4"),
        risk_factor_5=var("risk_factor_5"),
        risk_factor_6=var("risk_factor_6"),
        risk_factor_7=var("risk_factor_7"),
        trade_allowed=var("trade_allowed"),
        risk_radius=RISK_RADIUS,
    )
    weights = tuple(
        get_field(mpo, f"weights[{horizon}]")
        for horizon in range(N_HORIZONS)
    )
    common = [
        weights[0],
        get_field(mpo, "objective"),
        get_field(mpo, "iterations"),
        get_field(mpo, "status"),
    ]
    if mode == "none":
        return common
    if mode == "augmented":
        return common + [
            get_field(mpo, f"risk_{horizon}.value")
            for horizon in range(N_HORIZONS)
        ]
    if mode == "post":
        # A rank-2 CVXPY parameter is column-major relative to the DSL's
        # logical row. Thus CVXPY sees factor.T, and factor.T @ w is ij,i->j.
        return common + [
            cat(
                RISK_RADIUS,
                einsum(
                    var(f"risk_factor_{horizon}"),
                    weights[horizon],
                    "ij,i->j",
                ),
            )
            for horizon in range(N_HORIZONS)
        ]
    if mode == "verify":
        augmented = [
            get_field(mpo, f"risk_{horizon}.value")
            for horizon in range(N_HORIZONS)
        ]
        manual = [
            cat(
                RISK_RADIUS,
                einsum(
                    var(f"risk_factor_{horizon}"),
                    weights[horizon],
                    "ij,i->j",
                ),
            )
            for horizon in range(N_HORIZONS)
        ]
        return [get_field(mpo, "status"), *augmented, *manual]
    raise ValueError(mode)


def _checksum(values) -> float:
    total = 0.0
    for value in values:
        array = np.asarray(value, dtype=float)
        total += float(
            np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).sum()
        )
    return total


def _child() -> None:
    setting = os.environ["MPO_CONSTRAINT_SETTING"]
    mode = os.environ["MPO_CONSTRAINT_MODE"]
    data = _data()
    cpp_cache = OUT / f"cpp-{setting}-{mode}"
    shutil.rmtree(cpp_cache, ignore_errors=True)
    os.environ["TRADING_DSL_ENGINE_CPP_STREAM_CACHE"] = str(cpp_cache)

    expressions = _bound_program(setting, mode)
    started = time.perf_counter()
    runtime = compile_formula(expressions, data, n_instruments=N_ASSETS)
    compile_seconds = time.perf_counter() - started

    warmup_path = OUT / f"{setting}-{mode}-warmup.npy"
    warmup = runtime.run(out_path=warmup_path)
    warmup_values = warmup.load(mmap_mode=None)
    checksum = _checksum(warmup_values)

    native_times: list[float] = []
    run_wall_times: list[float] = []
    load_times: list[float] = []
    total_times: list[float] = []
    last_result = warmup
    last_values = warmup_values
    for run in range(RUNS):
        path = OUT / f"{setting}-{mode}-{run}.npy"
        started = time.perf_counter()
        last_result = runtime.run(out_path=path)
        after_run = time.perf_counter()
        last_values = last_result.load(mmap_mode=None)
        after_load = time.perf_counter()
        checksum += _checksum(last_values)
        native_times.append(float(last_result.seconds))
        run_wall_times.append(after_run - started)
        load_times.append(after_load - after_run)
        total_times.append(after_load - started)

    if mode == "verify":
        statuses = np.asarray(last_values[0]).reshape(-1)
        good = np.isin(statuses, [1.0, 4.0])
        maximum = 0.0
        for horizon in range(N_HORIZONS):
            augmented = np.asarray(last_values[1 + horizon], dtype=float)
            manual = np.asarray(
                last_values[1 + N_HORIZONS + horizon], dtype=float
            )
            if np.any(good):
                maximum = max(
                    maximum,
                    float(np.max(np.abs(augmented[good] - manual[good]))),
                )
        print(
            "VERIFY "
            + json.dumps(
                {
                    "good_rows": int(np.sum(good)),
                    "max_abs_same_solution": maximum,
                    "checksum": checksum,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    output_path = OUT / f"{setting}-{mode}-{RUNS - 1}.npy"
    metrics = {
        "setting": setting,
        "mode": mode,
        "rows": ROWS,
        "runs": RUNS,
        "compile_s": compile_seconds,
        "warmup_native_s": float(warmup.seconds),
        "mean_native_s": statistics.mean(native_times),
        "median_native_s": statistics.median(native_times),
        "mean_run_wall_s": statistics.mean(run_wall_times),
        "median_run_wall_s": statistics.median(run_wall_times),
        "mean_load_s": statistics.mean(load_times),
        "median_load_s": statistics.median(load_times),
        "mean_run_plus_load_s": statistics.mean(total_times),
        "median_run_plus_load_s": statistics.median(total_times),
        "output_bytes": output_path.stat().st_size,
        "checksum": checksum,
    }
    (OUT / f"{setting}-{mode}.json").write_text(
        json.dumps(metrics, sort_keys=True) + "\n"
    )

    arrays = {
        "weights0": np.asarray(last_values[0]),
        "objective": np.asarray(last_values[1]),
        "iterations": np.asarray(last_values[2]),
        "status": np.asarray(last_values[3]),
    }
    if mode != "none":
        for horizon in range(N_HORIZONS):
            arrays[f"risk_{horizon}"] = np.asarray(last_values[4 + horizon])
    np.savez(OUT / f"{setting}-{mode}.npz", **arrays)
    print("RESULT " + json.dumps(metrics, sort_keys=True), flush=True)


def _patched_template(original: str, setting: str) -> str:
    if setting == "default":
        return original
    if setting != "no_refine":
        raise ValueError(setting)
    anchor = "    settings_.presolve_enable = false;\n"
    replacement = (
        anchor + "    settings_.iterative_refinement_enable = false;\n"
    )
    if anchor not in original:
        raise RuntimeError("Clarabel settings template anchor changed")
    return original.replace(anchor, replacement, 1)


def _run_child(setting: str, mode: str) -> None:
    environment = os.environ.copy()
    environment["MPO_CONSTRAINT_SETTING"] = setting
    environment["MPO_CONSTRAINT_MODE"] = mode
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="", flush=True)
    if completed.returncode:
        raise RuntimeError(
            f"constraint-output child failed: setting={setting}, mode={mode}, "
            f"returncode={completed.returncode}"
        )


def _print_comparison(setting: str) -> None:
    metrics = {
        mode: json.loads((OUT / f"{setting}-{mode}.json").read_text())
        for mode in ("none", "augmented", "post")
    }
    augmented = metrics["augmented"]
    post = metrics["post"]
    none = metrics["none"]
    comparison = {
        "setting": setting,
        "augmented_over_post_native": (
            augmented["mean_native_s"] / post["mean_native_s"]
        ),
        "augmented_over_post_run_plus_load": (
            augmented["mean_run_plus_load_s"]
            / post["mean_run_plus_load_s"]
        ),
        "post_over_none_native": (
            post["mean_native_s"] / none["mean_native_s"]
        ),
        "augmented_over_none_native": (
            augmented["mean_native_s"] / none["mean_native_s"]
        ),
        "same_output_bytes": (
            augmented["output_bytes"] == post["output_bytes"]
        ),
    }
    print("COMPARISON " + json.dumps(comparison, sort_keys=True), flush=True)


def _print_gap_rows() -> None:
    values = np.load(OUT / "default-augmented.npz")
    weights = np.asarray(values["weights0"], dtype=float)
    statuses = np.asarray(values["status"], dtype=float).reshape(-1)
    objectives = np.asarray(values["objective"], dtype=float).reshape(-1)
    risk = np.asarray(values["risk_0"], dtype=float)
    for row in (
        GAP_START - 1,
        GAP_START,
        GAP_START + 1,
        GAP_END - 1,
        GAP_END,
    ):
        previous = np.zeros(N_ASSETS) if row == 0 else weights[row - 1]
        record = {
            "row": row,
            "status": float(statuses[row]),
            "weights": weights[row].tolist(),
            "all_finite": bool(np.isfinite(weights[row]).all()),
            "max_abs_change_from_previous": float(
                np.max(np.abs(weights[row] - previous))
            ),
            "objective": float(objectives[row]),
            "risk_radius_component": float(risk[row, 0]),
            "risk_vector_norm": float(np.linalg.norm(risk[row, 1:])),
        }
        print("GAP " + json.dumps(record, sort_keys=True), flush=True)


def _parent() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    native = build_current_clarabel()
    print(
        "CLARABEL "
        + json.dumps(
            {
                "include": str(native.include_dir),
                "library": str(native.static_library),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    original = TEMPLATE.read_text()
    try:
        for setting in ("default", "no_refine"):
            TEMPLATE.write_text(_patched_template(original, setting))
            for mode in ("none", "augmented", "post"):
                _run_child(setting, mode)
            _print_comparison(setting)
        TEMPLATE.write_text(original)
        _run_child("default", "verify")
        _print_gap_rows()
    finally:
        TEMPLATE.write_text(original)


if __name__ == "__main__":
    if os.environ.get("MPO_CONSTRAINT_MODE"):
        _child()
    else:
        _parent()
