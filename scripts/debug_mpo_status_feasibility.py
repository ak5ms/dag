from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import cvxpy as cp
import numpy as np

from scripts import spike_mpo_clarabel_final_validation as f

ROOT = Path(__file__).resolve().parents[1]


def run_variant(name: str, template: str) -> None:
    f.TEMPLATE.write_text(f._patch_template(template, name))
    env = os.environ.copy()
    env["MPO_CHILD_VARIANT"] = name
    env["MPO_FINAL_RUNS"] = "1"
    p = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "spike_mpo_clarabel_final_validation.py")],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(p.stdout, end="", flush=True)
    if p.returncode:
        raise RuntimeError((name, p.returncode))


def inspect(name: str, upstream) -> None:
    candidate = np.load(f.OUT / f"{name}.npz")
    weights = np.asarray(candidate["weights"], dtype=float)
    status = np.asarray(candidate["status"], dtype=int).reshape(-1)
    values, counts = np.unique(status, return_counts=True)
    print(f"STATUS name={name} counts={dict(zip(values.tolist(), counts.tolist()))}", flush=True)
    bad = np.flatnonzero(~np.isin(status, [1, 4]))
    print(f"BAD_ROWS name={name} rows={bad.tolist()} statuses={status[bad].tolist()}", flush=True)

    problem = f._problem()
    params = {p.name(): p for p in problem.parameters()}
    variables = {v.name(): v for v in problem.variables()}
    wvar = variables["weights"]
    pvar = variables["previous_weights"]
    max_by_constraint = np.zeros(len(problem.constraints), dtype=float)
    argmax_by_constraint = np.full(len(problem.constraints), -1, dtype=int)

    for t in np.flatnonzero(np.isin(status, [1, 4])):
        current = np.zeros(f.N_ASSETS) if t == 0 else weights[t - 1, 0]
        for pname, p in params.items():
            if pname == "current_weights":
                p.value = current
            elif pname == "risk_radius":
                p.value = 0.08
            else:
                p.value = f._cp_row(pname, upstream[pname][t], tuple(int(x) for x in p.shape))
        wvar.value = weights[t]
        pvar.value = current
        for i, constraint in enumerate(problem.constraints):
            v = np.asarray(constraint.violation(), dtype=float)
            value = float(np.nanmax(np.abs(v))) if v.size else 0.0
            if value > max_by_constraint[i]:
                max_by_constraint[i] = value
                argmax_by_constraint[i] = int(t)

    for i, constraint in enumerate(problem.constraints):
        print(
            f"CONSTRAINT name={name} idx={i} type={type(constraint).__name__} "
            f"label={getattr(constraint, 'label', None)} max_violation={max_by_constraint[i]:.12g} "
            f"row={argmax_by_constraint[i]}",
            flush=True,
        )

    for t in bad:
        current = np.zeros(f.N_ASSETS) if t == 0 else weights[t - 1, 0]
        for pname, p in params.items():
            if pname == "current_weights":
                p.value = current
            elif pname == "risk_radius":
                p.value = 0.08
            else:
                p.value = f._cp_row(pname, upstream[pname][t], tuple(int(x) for x in p.shape))
        try:
            value = problem.solve(
                solver=cp.CLARABEL,
                verbose=False,
                presolve_enable=False,
                tol_gap_abs=1e-9,
                tol_gap_rel=1e-9,
                tol_feas=1e-9,
            )
            print(f"BAD_REFERENCE name={name} row={t} status={problem.status} objective={value}", flush=True)
        except Exception as exc:
            print(f"BAD_REFERENCE name={name} row={t} exception={type(exc).__name__}:{exc}", flush=True)


def main() -> None:
    f.OUT.mkdir(parents=True, exist_ok=True)
    from examples import cpp_stream_mpo_one_pass as example

    example._clarabel()
    f._materialize_upstream(f.OUT / "upstream.npz")
    upstream = np.load(f.OUT / "upstream.npz")
    original = f.TEMPLATE.read_text()
    try:
        for name in ("baseline_nodiag", "no_refine_nodiag"):
            run_variant(name, original)
            inspect(name, upstream)
    finally:
        f.TEMPLATE.write_text(original)


if __name__ == "__main__":
    main()
