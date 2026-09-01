from __future__ import annotations

import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import cvxpy as cp
import numpy as np

from examples import cpp_stream_mpo_one_pass as example
from flows.riskmodel import risk_covariance
from flows.utils import ts_zscore
from trading_dsl_engine.base.dsl import (
    Ridge, cat, einsum, fillna, get_beta, psd_factor, purify,
    rolling_sum, shift, var, where,
)
from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.cpp_stream.optimizer import cvxpy_program, get_field, previous_solution

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src/trading_dsl_engine/cpp_stream/optimizer/templates/direct_clarabel_instance.hpp.j2"
OUT = Path("/dev/shm/mpo_formulation_spike")
ROWS = int(os.environ.get("MPO_FORM_ROWS", "5000"))
RUNS = int(os.environ.get("MPO_FORM_RUNS", "5"))
N = 3
MINUTE_US = 60_000_000.0


def fake_data():
    rng = np.random.default_rng(7)
    base = 1_800_000_000_000_000.0
    row = np.arange(ROWS)
    sr, op = 1000, 980
    ts = base + row * MINUTE_US
    sid = row // sr
    s0 = base + sid * sr * MINUTE_US
    e0 = s0 + op * MINUTE_US
    ns = s0 + sr * MINUTE_US
    ne = ns + op * MINUTE_US
    is_open = (ts >= s0) & (ts < e0)
    tradable = np.broadcast_to(is_open[:, None], (ROWS, N)).astype(float).copy()
    returns = rng.normal(scale=2e-4, size=(ROWS, N))
    returns[tradable == 0] = 0
    for reopen in range(sr, ROWS, sr):
        returns[reopen] *= np.sqrt(sr - op + 1.0)
    def lanes(x): return np.broadcast_to(x[:, None], (ROWS, N)).astype(float).copy()
    return {
        "returns": np.ascontiguousarray(returns),
        "is_tradable_out0": tradable,
        "vw_halfspread_out0": rng.uniform(3e-5, 8e-5, size=(ROWS, N)),
        "_ev_ts": lanes(ts), "session_start0": lanes(s0), "session_end0": lanes(e0),
        "next_session_start0": lanes(ns), "next_session_end0": lanes(ne),
    }


def upstream_exprs():
    returns = var("returns")
    tradable = fillna(var("is_tradable_out0"), 0.0)
    hs = var("vw_halfspread_out0")
    fit_weights = purify(1 / hs**2)
    fs = tuple(ts_zscore(returns, example._feature_span(hl), min_periods=max(2, round(example._feature_span(hl)))) for hl in example.FEATURE_HLS)
    features = cat(*fs)
    clean = where(tradable != 0, fillna(returns, 0.0), 0.0)
    forecasts, factors = [], []
    for start, end in zip(example.TRADE_STARTS, example.HORIZONS):
        width = end - start
        br = rolling_sum(clean, width, min_periods=width)
        bo = rolling_sum(tradable, width, min_periods=width)
        target = where(bo > 0, br, float("nan"))
        fit_x = cat(*(where(shift(tradable, end) != 0, shift(f, end), float("nan")) for f in fs))
        beta = get_beta(Ridge(fit_x, y=target, weights=fit_weights, hl=example.RIDGE_HL, lambda_=0.1))
        forecasts.append(fillna(einsum(features, beta, "if,f->i"), 0.0))
        rb = shift(br, start); ro = shift(bo, start)
        sample = where(ro > 0, rb, float("nan"))
        cov = risk_covariance(sample, span=example.RISK_SPAN, min_periods=example.RISK_MIN_PERIODS, ignore_na=True, adjust=False)
        factors.append(psd_factor(fillna(cov, 0.0), eigenvalue_floor=1e-8))
    return [cat(*forecasts), fillna(purify(hs), 0.0), *factors, example._planned_trade_allowed(tradable)]


def common_parameters(expected_returns, half_spread, current_weights, risks, trade_allowed, risk_radius):
    er = cp.Parameter(expected_returns.shape, name="expected_returns")
    hs = cp.Parameter(half_spread.shape, name="half_spread", nonneg=True)
    cw = cp.Parameter((expected_returns.shape[1],), name="current_weights")
    rf = tuple(cp.Parameter(r.shape, name=f"risk_factor_{i}") for i, r in enumerate(risks))
    ta = cp.Parameter(trade_allowed.shape, name="trade_allowed", nonneg=True)
    rr = cp.Parameter(name="risk_radius", nonneg=True)
    return er, hs, cw, rf, ta, rr


def build_problem(expected_returns, half_spread, current_weights, risk_factor_0, risk_factor_1, risk_factor_2, risk_factor_3, risk_factor_4, risk_factor_5, risk_factor_6, risk_factor_7, trade_allowed, risk_radius, mode):
    H, n = expected_returns.shape
    er, hs, cw, rf, ta, rr = common_parameters(expected_returns, half_spread, current_weights, (risk_factor_0,risk_factor_1,risk_factor_2,risk_factor_3,risk_factor_4,risk_factor_5,risk_factor_6,risk_factor_7), trade_allowed, risk_radius)
    w = cp.Variable((H,n), name="weights")
    prev = cp.Variable((n,), name="previous_weights")
    delta = w - cp.vstack([prev, w[:-1]])
    constraints = [prev == cw, cp.sum(delta, axis=1) == 0]
    if mode == "current_abs":
        a = cp.abs(delta)
        constraints.append(a <= example.TRADE_BIG_M * ta)
        tcost = cp.sum(cp.multiply(hs, a))
    elif mode == "split_abs_objective":
        constraints += [delta <= example.TRADE_BIG_M * ta, -delta <= example.TRADE_BIG_M * ta]
        tcost = cp.sum(cp.multiply(hs, cp.abs(delta)))
    elif mode == "explicit_epigraph":
        u = cp.Variable((H,n), name="turnover")
        constraints += [u >= delta, u >= -delta, u <= example.TRADE_BIG_M * ta]
        tcost = cp.sum(cp.multiply(hs, u))
    else:
        raise ValueError(mode)
    for h, r in enumerate(rf):
        constraints.append(cp.SOC(rr, r @ w[h]))
    return cp.Problem(cp.Minimize(-cp.sum(cp.multiply(er,w)) + tcost), constraints)


def make_program(mode):
    @cvxpy_program(cache_dir=OUT / f"cache_{mode}", clarabel=example._clarabel, sequential=None)
    def program(expected_returns, half_spread, current_weights, risk_factor_0, risk_factor_1, risk_factor_2, risk_factor_3, risk_factor_4, risk_factor_5, risk_factor_6, risk_factor_7, trade_allowed, risk_radius=example.RISK_RADIUS):
        return build_problem(expected_returns, half_spread, current_weights, risk_factor_0, risk_factor_1, risk_factor_2, risk_factor_3, risk_factor_4, risk_factor_5, risk_factor_6, risk_factor_7, trade_allowed, risk_radius, mode)
    return program


def materialize():
    data = fake_data()
    r = compile_formula(upstream_exprs(), data, n_instruments=N).run(out_path=OUT / "upstream.npy")
    v = r.load()
    payload = {"expected_returns": np.ascontiguousarray(v[0]), "half_spread": np.ascontiguousarray(v[1]), "trade_allowed": np.ascontiguousarray(v[10])}
    for h in range(8): payload[f"risk_factor_{h}"] = np.ascontiguousarray(v[2+h])
    np.savez(OUT / "upstream.npz", **payload)


def expressions(program):
    p = program(expected_returns=var("expected_returns"), half_spread=var("half_spread"), current_weights=previous_solution("weights[0]", initial=0.0), risk_factor_0=var("risk_factor_0"), risk_factor_1=var("risk_factor_1"), risk_factor_2=var("risk_factor_2"), risk_factor_3=var("risk_factor_3"), risk_factor_4=var("risk_factor_4"), risk_factor_5=var("risk_factor_5"), risk_factor_6=var("risk_factor_6"), risk_factor_7=var("risk_factor_7"), trade_allowed=var("trade_allowed"), risk_radius=example.RISK_RADIUS)
    return [get_field(p, f"weights[{h}]") for h in range(8)] + [get_field(p,"objective"), get_field(p,"status"), get_field(p,"iterations")]


def child():
    mode = os.environ["FORM_CHILD"]
    data0 = np.load(OUT / "upstream.npz")
    data = {k: np.ascontiguousarray(data0[k]) for k in data0.files}
    program = make_program(mode)
    t0=time.perf_counter(); runtime=compile_formula(expressions(program), data, n_instruments=N); compile_s=time.perf_counter()-t0
    warm=runtime.run(out_path=OUT/f"{mode}_warm.npy")
    times=[]; result=warm
    for i in range(RUNS):
        result=runtime.run(out_path=OUT/f"{mode}_{i}.npy"); times.append(result.seconds)
    vals=result.load(); weights=np.stack([np.asarray(vals[h]) for h in range(8)],axis=1)
    np.savez(OUT/f"{mode}.npz", weights=weights, objective=np.asarray(vals[8]), status=np.asarray(vals[9]), iterations=np.asarray(vals[10]))
    print(f"RESULT mode={mode} compile_s={compile_s:.6f} mean_s={statistics.mean(times):.6f} median_s={statistics.median(times):.6f} rows_per_s={ROWS/statistics.mean(times):.3f} mean_iter={np.mean(np.asarray(vals[10],dtype=float)):.4f}", flush=True)


def parent():
    OUT.mkdir(parents=True,exist_ok=True); example._clarabel(); materialize(); original=TEMPLATE.read_text()
    # Best solver policy from prior screen so cone-form differences are easier to see.
    anchor="    settings_.presolve_enable = false;\n"
    TEMPLATE.write_text(original.replace(anchor,anchor+"    settings_.iterative_refinement_enable = false;\n",1))
    try:
        for mode in ("current_abs","split_abs_objective","explicit_epigraph"):
            env=os.environ.copy(); env["FORM_CHILD"]=mode
            p=subprocess.run([sys.executable,str(Path(__file__).resolve())],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            print(p.stdout,end="",flush=True)
            if p.returncode: raise RuntimeError((mode,p.returncode))
        base=np.load(OUT/"current_abs.npz")
        for mode in ("split_abs_objective","explicit_epigraph"):
            x=np.load(OUT/f"{mode}.npz")
            good=np.isin(np.asarray(base["status"]).reshape(-1),[1,4]) & np.isin(np.asarray(x["status"]).reshape(-1),[1,4])
            d=x["weights"][good]-base["weights"][good]
            print(f"DIFF mode={mode} good_rows={good.sum()} max_abs={np.max(np.abs(d)):.12g} rms={np.sqrt(np.mean(d*d)):.12g} max_obj_abs={np.max(np.abs(x['objective'][good]-base['objective'][good])):.12g}",flush=True)
    finally:
        TEMPLATE.write_text(original)


if __name__=="__main__": child() if os.environ.get("FORM_CHILD") else parent()
