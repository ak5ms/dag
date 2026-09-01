from __future__ import annotations

import importlib.util
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
TEMPLATE = (
    ROOT
    / "src/trading_dsl_engine/cpp_stream/optimizer/templates/"
    / "direct_clarabel_instance.hpp.j2"
)
OUT = Path(os.environ.get("MPO_CONSTRAINT_OUT", "/dev/shm/mpo_constraint_output"))
ROWS = int(os.environ.get("MPO_FINAL_ROWS", "5000"))
RUNS = int(os.environ.get("MPO_CONSTRAINT_RUNS", "10"))
HORIZONS = 8
ASSETS = 3
RISK_RADIUS = 0.08


def _load_base_module():
    path = ROOT / "scripts/spike_mpo_clarabel_final_validation.py"
    spec = importlib.util.spec_from_file_location("mpo_final_validation_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = OUT
    return module


BASE = _load_base_module()


def _program_expressions(mode: str):
    from examples import cpp_stream_mpo_one_pass as example
    from trading_dsl_engine.base.dsl import einsum, var
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
        risk_radius=RISK_RADIUS,
    )
    weights = [get_field(mpo, f"weights[{h}]") for h in range(HORIZONS)]
    roots = weights + [
        get_field(mpo, "objective"),
        get_field(mpo, "status"),
        get_field(mpo, "iterations"),
    ]
    if mode == "none":
        return roots
    if mode == "augmented":
        return roots + [
            get_field(mpo, f"risk_{h}.value") for h in range(HORIZONS)
        ]
    if mode == "post":
        # Rank-2 DSL values are transposed when presented through CVXPY's
        # column-major parameter ABI. Thus CVXPY's risk_factor @ weights is
        # the same numerical map as DSL risk_factor.T @ weights.
        projected = [
            einsum(
                var(f"risk_factor_{h}"),
                weights[h],
                "ij,i->j",
            )
            for h in range(HORIZONS)
        ]
        # A raw Python scalar cannot be a top-level cpp_stream root. Derive a
        # row scalar from finite input data; CSE computes this once and eight
        # roots return it. The payload remains exactly equal to augmented mode:
        # eight three-vectors plus eight scalar radii = 32 doubles per row.
        radius = var("half_spread").sum(axis=1) * 0.0 + RISK_RADIUS
        return roots + projected + [radius] * HORIZONS
    raise ValueError(mode)


def _child() -> None:
    from trading_dsl_engine.cpp_stream import compile_formula

    setting = os.environ["MPO_CONSTRAINT_SETTING"]
    mode = os.environ["MPO_CONSTRAINT_MODE"]
    source = np.load(OUT / "upstream.npz")
    data = {name: np.ascontiguousarray(source[name]) for name in source.files}

    shutil.rmtree(
        ROOT / ".generated/cpp_stream_mpo_one_pass/clarabel",
        ignore_errors=True,
    )
    os.environ["TRADING_DSL_ENGINE_CPP_STREAM_CACHE"] = str(
        OUT / f"native_cache_{setting}_{mode}"
    )
    t0 = time.perf_counter()
    runtime = compile_formula(
        _program_expressions(mode),
        data,
        n_instruments=ASSETS,
    )
    compile_s = time.perf_counter() - t0

    warm_path = OUT / f"{setting}_{mode}_warm.npy"
    warmup = runtime.run(out_path=warm_path)
    native_times: list[float] = []
    wall_times: list[float] = []
    result = warmup
    result_path = warm_path
    for run in range(RUNS):
        result_path = OUT / f"{setting}_{mode}_{run}.npy"
        t0 = time.perf_counter()
        result = runtime.run(out_path=result_path)
        wall_times.append(time.perf_counter() - t0)
        native_times.append(result.seconds)

    t0 = time.perf_counter()
    values = result.load(mmap_mode=None)
    load_s = time.perf_counter() - t0
    if not isinstance(values, tuple):
        values = (values,)
    weights = np.stack(
        [np.asarray(values[h], dtype=float) for h in range(HORIZONS)],
        axis=1,
    )
    objective = np.asarray(values[HORIZONS], dtype=float).reshape(-1)
    status = np.asarray(values[HORIZONS + 1], dtype=float).reshape(-1)
    iterations = np.asarray(values[HORIZONS + 2], dtype=float).reshape(-1)

    risk_values = np.empty((ROWS, 0, ASSETS + 1), dtype=float)
    if mode == "augmented":
        risk_values = np.stack(
            [
                np.asarray(values[HORIZONS + 3 + h], dtype=float)
                for h in range(HORIZONS)
            ],
            axis=1,
        )
    elif mode == "post":
        projected_offset = HORIZONS + 3
        radius_offset = projected_offset + HORIZONS
        projected = [
            np.asarray(values[projected_offset + h], dtype=float)
            for h in range(HORIZONS)
        ]
        radii = [
            np.asarray(values[radius_offset + h], dtype=float).reshape(ROWS, -1)
            for h in range(HORIZONS)
        ]
        risk_values = np.stack(
            [np.concatenate([radii[h], projected[h]], axis=1) for h in range(HORIZONS)],
            axis=1,
        )

    np.savez(
        OUT / f"{setting}_{mode}.npz",
        weights=weights,
        objective=objective,
        status=status,
        iterations=iterations,
        risk_values=risk_values,
    )
    payload_bytes = int(sum(np.asarray(value).nbytes for value in values))
    metrics = {
        "setting": setting,
        "mode": mode,
        "compile_s": compile_s,
        "warmup_native_s": warmup.seconds,
        "mean_native_s": statistics.mean(native_times),
        "median_native_s": statistics.median(native_times),
        "mean_wall_s": statistics.mean(wall_times),
        "median_wall_s": statistics.median(wall_times),
        "rows_per_s": ROWS / statistics.mean(native_times),
        "load_s": load_s,
        "payload_bytes": payload_bytes,
        "output_file_bytes": result_path.stat().st_size,
        "output_values_per_row": payload_bytes // (ROWS * 8),
        "solved": int(np.sum(status == 1)),
        "primal_infeasible": int(np.sum(status == 2)),
        "almost_solved": int(np.sum(status == 4)),
        "mean_iterations": float(np.mean(iterations)),
    }
    print("RESULT " + json.dumps(metrics, sort_keys=True), flush=True)


def _expected_risk(upstream, weights: np.ndarray) -> np.ndarray:
    expected = np.empty((ROWS, HORIZONS, ASSETS + 1), dtype=float)
    expected[:, :, 0] = RISK_RADIUS
    for h in range(HORIZONS):
        expected[:, h, 1:] = np.einsum(
            "tij,ti->tj",
            np.asarray(upstream[f"risk_factor_{h}"], dtype=float),
            weights[:, h],
        )
    return expected


def _finite_max_abs(left: np.ndarray, right: np.ndarray, rows: np.ndarray) -> float:
    difference = np.abs(left[rows] - right[rows])
    finite = np.isfinite(difference)
    return float(np.max(difference[finite])) if np.any(finite) else float("nan")


def _compare(setting: str, upstream) -> None:
    augmented = np.load(OUT / f"{setting}_augmented.npz")
    post = np.load(OUT / f"{setting}_post.npz")
    good = np.isin(augmented["status"], [1.0, 4.0]) & np.isin(
        post["status"], [1.0, 4.0]
    )
    expected_augmented = _expected_risk(upstream, augmented["weights"])
    expected_post = _expected_risk(upstream, post["weights"])
    print(
        "COMPARE "
        + json.dumps(
            {
                "setting": setting,
                "good_rows": int(np.sum(good)),
                "max_weight_abs_augmented_vs_post": _finite_max_abs(
                    augmented["weights"], post["weights"], good
                ),
                "max_objective_abs_augmented_vs_post": _finite_max_abs(
                    augmented["objective"], post["objective"], good
                ),
                "max_augmented_value_error_from_own_solution": _finite_max_abs(
                    augmented["risk_values"], expected_augmented, good
                ),
                "max_post_value_error_from_own_solution": _finite_max_abs(
                    post["risk_values"], expected_post, good
                ),
                "max_constraint_value_abs_augmented_vs_post": _finite_max_abs(
                    augmented["risk_values"], post["risk_values"], good
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # The production interface currently uses augmented mode. Inspect the
    # close boundary, the infeasible first closed row, later gap rows, reopen,
    # and the row after reopen.
    for row in (979, 980, 981, 999, 1000, 1001):
        weights = augmented["weights"]
        previous = np.zeros(ASSETS) if row == 0 else weights[row - 1, 0]
        current = weights[row, 0]
        risk_value = augmented["risk_values"][row, 0]
        record = {
            "setting": setting,
            "row": row,
            "status": int(augmented["status"][row]),
            "weights": current.tolist(),
            "previous_weights": previous.tolist(),
            "all_finite": bool(np.isfinite(current).all()),
            "max_abs_change_from_previous": float(
                np.max(np.abs(current - previous))
            ),
            "risk_norm": float(np.linalg.norm(risk_value[1:])),
            "risk_violation": float(
                max(0.0, np.linalg.norm(risk_value[1:]) - risk_value[0])
            ),
            "objective": float(augmented["objective"][row]),
        }
        print("GAP " + json.dumps(record, sort_keys=True), flush=True)


def _run_child(setting: str, mode: str) -> None:
    env = os.environ.copy()
    env["MPO_CONSTRAINT_SETTING"] = setting
    env["MPO_CONSTRAINT_MODE"] = mode
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="", flush=True)
    if completed.returncode:
        raise RuntimeError(
            f"constraint-output child failed: setting={setting}, mode={mode}, "
            f"exit={completed.returncode}"
        )


def _parent() -> None:
    from examples import cpp_stream_mpo_one_pass as example

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    setup_start = time.perf_counter()
    example._clarabel()
    print(f"CLARABEL_SETUP seconds={time.perf_counter() - setup_start:.6f}", flush=True)
    BASE._materialize_upstream(OUT / "upstream.npz")
    upstream_file = np.load(OUT / "upstream.npz")
    upstream = {name: np.asarray(upstream_file[name]) for name in upstream_file.files}

    original = TEMPLATE.read_text()
    anchor = "    settings_.presolve_enable = false;\n"
    try:
        for setting in ("default", "no_refine"):
            if setting == "default":
                TEMPLATE.write_text(original)
            else:
                if anchor not in original:
                    raise RuntimeError("Clarabel settings template anchor changed")
                TEMPLATE.write_text(
                    original.replace(
                        anchor,
                        anchor
                        + "    settings_.iterative_refinement_enable = false;\n",
                        1,
                    )
                )
            for mode in ("none", "augmented", "post"):
                _run_child(setting, mode)
            _compare(setting, upstream)
    finally:
        TEMPLATE.write_text(original)


if __name__ == "__main__":
    if os.environ.get("MPO_CONSTRAINT_MODE"):
        _child()
    else:
        _parent()
