from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

import numpy as np

from flows.utils import ewm_std, replace
from trading_dsl_engine.base.dsl import (
    Ridge,
    abs as dsl_abs,
    cat,
    ffill,
    get_beta,
    get_residuals,
    purify,
    shift,
    var,
    where,
)
from trading_dsl_engine.cpp_stream import compile_formula


ROWS = int(os.environ.get("GP_BRAINSTORM_ROWS", "250000"))
N = int(os.environ.get("GP_BRAINSTORM_INSTRUMENTS", "9"))
CANDIDATES = int(os.environ.get("GP_BRAINSTORM_CANDIDATES", "8"))
POOL = int(os.environ.get("GP_BRAINSTORM_POOL", "8"))
RUNS = int(os.environ.get("GP_BRAINSTORM_RUNS", "3"))
WARMUPS = int(os.environ.get("GP_BRAINSTORM_WARMUPS", "1"))
ALPHA_PNL_HL = int(os.environ.get("GP_BRAINSTORM_ALPHA_PNL_HL", str(1440 * 21)))
RIDGE_HL = int(os.environ.get("GP_BRAINSTORM_RIDGE_HL", str(1440 * 5)))
RIDGE_LAMBDA = float(os.environ.get("GP_BRAINSTORM_RIDGE_LAMBDA", "1e-3"))
THREADS = int(os.environ.get("GP_BRAINSTORM_THREADS", "1"))
ROOT = Path(os.environ.get("GP_BRAINSTORM_ROOT", "/tmp/gp-brainstorm"))
CACHE_ROOT = ROOT / "cache"
DATA_ROOT = ROOT / "data"
OUT_ROOT = ROOT / "out"
RESULT_PATH = ROOT / "benchmark.json"


def l1_norm(x):
    return purify(x / dsl_abs(x).sum(axis=-1))


def generate_data() -> dict[str, Path]:
    shutil.rmtree(ROOT, ignore_errors=True)
    DATA_ROOT.mkdir(parents=True)
    OUT_ROOT.mkdir(parents=True)
    CACHE_ROOT.mkdir(parents=True)
    rng = np.random.default_rng(42)
    names = [*(f"alpha{i}" for i in range(CANDIDATES)), *(f"pool{i}" for i in range(POOL)), "roll_rets", "hs", "is_tradable"]
    paths = {name: DATA_ROOT / f"{name}.npy" for name in names}
    arrays = {
        name: np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(ROWS, N))
        for name, path in paths.items()
    }
    latent = np.zeros(N, dtype=np.float64)
    chunk = 32768
    for start in range(0, ROWS, chunk):
        stop = min(start + chunk, ROWS)
        m = stop - start
        common = rng.normal(size=(m, 1))
        base = 0.35 * common + rng.normal(size=(m, N))
        for i in range(POOL):
            arrays[f"pool{i}"][start:stop] = base + 0.20 * rng.normal(size=(m, N)) + i * 0.001
        for i in range(CANDIDATES):
            # Candidates deliberately share pool structure so orthogonalization is meaningful.
            arrays[f"alpha{i}"][start:stop] = (
                0.55 * base
                + 0.25 * arrays[f"pool{i % POOL}"][start:stop]
                + 0.35 * rng.normal(size=(m, N))
            )
        signal = arrays["alpha0"][start:stop]
        lagged = np.vstack((latent[None, :], signal[:-1]))
        arrays["roll_rets"][start:stop] = 2.0e-5 * lagged + rng.normal(0.0, 4.0e-4, size=(m, N))
        arrays["hs"][start:stop] = np.clip(rng.lognormal(-9.0, 0.25, size=(m, N)), 2e-5, 8e-4)
        row = np.arange(start, stop)[:, None]
        arrays["is_tradable"][start:stop] = np.broadcast_to(((row % 1440) < 1200).astype(np.float64), (m, N))
        latent = signal[-1].copy()
    for value in arrays.values():
        value.flush()
    arrays.clear()
    return paths


def cleaned_rets():
    r = var("roll_rets")
    return where(dsl_abs(r) <= 0.05, replace(r, 0, float("nan")), float("nan"))


def volatility():
    return ewm_std(cleaned_rets(), span=ALPHA_PNL_HL)


def pnl_contrib(alpha):
    w = alpha / volatility()
    held = shift(ffill(where(var("is_tradable"), w, float("nan"))), 1, 1)
    return held * var("roll_rets")


def fitness_cat_then_sharpe():
    pnls = [pnl_contrib(l1_norm(var(f"alpha{i}"))) for i in range(CANDIDATES)]
    pnl = cat(*pnls).sum(axis=1)
    return pnl.mean(axis=0) / pnl.std(axis=0)


def fitness_sharpe_then_cat():
    scores = []
    for i in range(CANDIDATES):
        pnl = pnl_contrib(l1_norm(var(f"alpha{i}"))).sum(axis=1)
        scores.append(pnl.mean(axis=0) / pnl.std(axis=0))
    return cat(*scores)


def current_pool_ridge():
    vol = volatility()
    hs = var("hs")
    weights = purify(1.0 / (hs * hs))
    alphas = [l1_norm(var(f"pool{i}")) for i in range(POOL)] + [
        l1_norm(var(f"alpha{i}")) for i in range(CANDIDATES)
    ]
    features = [shift(alpha, 1, 1) * vol for alpha in alphas]
    model = Ridge(
        *features,
        y=cleaned_rets(),
        weights=weights,
        hl=float(RIDGE_HL),
        lambda_=RIDGE_LAMBDA,
        nonneg=True,
        recompute_every=1,
    )
    return dsl_abs(get_beta(model)).mean(axis=0)


def xs_orthogonalized_candidates():
    pool = [l1_norm(var(f"pool{i}")) for i in range(POOL)]
    residuals = []
    for i in range(CANDIDATES):
        model = Ridge(
            *pool,
            y=l1_norm(var(f"alpha{i}")),
            weights=1.0,
            hl=0.0,
            lambda_=RIDGE_LAMBDA,
            nonneg=False,
            recompute_every=1,
        )
        residuals.append(purify(get_residuals(model)))
    return residuals


def xs_orthogonalization_only():
    return cat(*xs_orthogonalized_candidates())


def xs_orthogonal_then_univariate_ridge():
    vol = volatility()
    hs = var("hs")
    weights = purify(1.0 / (hs * hs))
    scores = []
    for residual in xs_orthogonalized_candidates():
        feature = shift(l1_norm(residual), 1, 1) * vol
        model = Ridge(
            feature,
            y=cleaned_rets(),
            weights=weights,
            hl=float(RIDGE_HL),
            lambda_=RIDGE_LAMBDA,
            nonneg=True,
            recompute_every=1,
        )
        scores.append(dsl_abs(get_beta(model)).mean(axis=0))
    return cat(*scores)


def bench_formula(name: str, formula, paths: dict[str, Path]) -> dict[str, object]:
    cache = CACHE_ROOT / name
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True)
    os.environ["TRADING_DSL_ENGINE_CPP_STREAM_CACHE"] = str(cache)
    started = time.perf_counter()
    runtime = compile_formula(formula, paths, n_instruments=N, prefetch_rows=16)
    compile_seconds = time.perf_counter() - started
    output = OUT_ROOT / f"{name}.npy"
    for _ in range(WARMUPS):
        runtime.run(out_path=output, threads=THREADS)
    runs = []
    native = []
    cpu = []
    busy = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        result = runtime.run(out_path=output, threads=THREADS)
        runs.append(time.perf_counter() - t0)
        native.append(float(result.seconds))
        cpu.append(float(result.cpu_seconds))
        busy.append(float(result.average_busy_cores))
    values = np.asarray(np.load(output, mmap_mode="r"))
    checksum = float(np.nansum(values))
    explain = runtime.explain()
    return {
        "name": name,
        "compile_seconds": compile_seconds,
        "run_wall_seconds": runs,
        "median_wall_seconds": statistics.median(runs),
        "native_seconds": native,
        "median_native_seconds": statistics.median(native),
        "cpu_seconds": cpu,
        "average_busy_cores": busy,
        "parallel_plan_mode": str(runtime.parallel_plan.mode),
        "parallel_plan_reason": str(runtime.parallel_plan.reason),
        "work_score": int(runtime.parallel_plan.work_score),
        "output_shape": list(runtime.plan.output_shape),
        "checksum": checksum,
        "generated_cpp": str(runtime.generated_cpp),
        "generated_cpp_bytes": Path(runtime.generated_cpp).stat().st_size,
        "explain": explain,
        "extrapolated_5m_native_seconds": statistics.median(native) * 5_000_000 / ROWS,
    }


def hat_reuse_ceiling() -> dict[str, object]:
    # A standalone C++ ceiling for the exact algebraic opportunity: same X and
    # regularized X'X, multiple candidate RHS y. The repeated variant refactors
    # X'X once per y; the cached variant factorizes once and solves all RHS.
    # This is intentionally not reported as cpp_stream performance.
    source = ROOT / "hat_reuse.cpp"
    binary = ROOT / "hat_reuse"
    cpp = r'''
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
constexpr int N=9;
constexpr int P=8;
constexpr int M=8;
constexpr int ROWS=200000;
using Mat=std::array<double,P*P>;
using Vec=std::array<double,P>;
static inline bool chol(Mat a, Mat& L){
  L.fill(0.0);
  for(int i=0;i<P;i++) for(int j=0;j<=i;j++){
    double s=a[i*P+j];
    for(int k=0;k<j;k++) s-=L[i*P+k]*L[j*P+k];
    if(i==j){ if(s<=1e-14) return false; L[i*P+j]=std::sqrt(s); }
    else L[i*P+j]=s/L[j*P+j];
  }
  return true;
}
static inline void solve(const Mat& L,const Vec& b,Vec& x){
  Vec z{};
  for(int i=0;i<P;i++){ double s=b[i]; for(int k=0;k<i;k++) s-=L[i*P+k]*z[k]; z[i]=s/L[i*P+i]; }
  for(int ii=0;ii<P;ii++){ int i=P-1-ii; double s=z[i]; for(int k=i+1;k<P;k++) s-=L[k*P+i]*x[k]; x[i]=s/L[i*P+i]; }
}
int main(){
  std::mt19937_64 g(42); std::normal_distribution<double>d(0,1); double sink=0;
  auto run=[&](bool cached){
    auto t0=std::chrono::steady_clock::now();
    for(int r=0;r<ROWS;r++){
      double X[N][P], Y[N][M];
      for(int n=0;n<N;n++){ for(int p=0;p<P;p++)X[n][p]=d(g); for(int m=0;m<M;m++)Y[n][m]=d(g); }
      Mat A{}; for(int i=0;i<P;i++)for(int j=0;j<P;j++){double s=0;for(int n=0;n<N;n++)s+=X[n][i]*X[n][j];A[i*P+j]=s+(i==j?1e-3:0);}
      Mat shared{}; if(cached) chol(A,shared);
      for(int m=0;m<M;m++){
        Vec b{},x{}; for(int p=0;p<P;p++){double s=0;for(int n=0;n<N;n++)s+=X[n][p]*Y[n][m];b[p]=s;}
        Mat L{}; if(!cached) chol(A,L); else L=shared; solve(L,b,x); sink+=x[0]*1e-300;
      }
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  };
  double repeated=run(false), cached=run(true);
  std::printf("{\"rows\":%d,\"repeated_seconds\":%.9f,\"cached_seconds\":%.9f,\"speedup\":%.9f,\"sink\":%.17g}\n",ROWS,repeated,cached,repeated/cached,sink);
}
'''
    source.write_text(cpp)
    subprocess.run(["g++", "-O3", "-march=native", "-std=c++20", str(source), "-o", str(binary)], check=True)
    payload = json.loads(subprocess.check_output([str(binary)], text=True))
    payload["note"] = "standalone C++ algebraic ceiling, not cpp_stream performance"
    return payload


def main() -> None:
    paths = generate_data()
    cases = [
        ("fitness_cat_then_sharpe", fitness_cat_then_sharpe()),
        ("fitness_sharpe_then_cat", fitness_sharpe_then_cat()),
        ("current_pool_ridge", current_pool_ridge()),
        ("xs_orthogonalization_only", xs_orthogonalization_only()),
        ("xs_orthogonal_then_univariate_ridge", xs_orthogonal_then_univariate_ridge()),
    ]
    results = []
    for name, formula in cases:
        print(f"BENCH_START {name}", flush=True)
        result = bench_formula(name, formula, paths)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    ceiling = hat_reuse_ceiling()
    payload = {
        "configuration": {
            "rows": ROWS,
            "instruments": N,
            "candidates": CANDIDATES,
            "pool": POOL,
            "runs": RUNS,
            "warmups": WARMUPS,
            "threads": THREADS,
        },
        "results": results,
        "hat_reuse_ceiling": ceiling,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("FINAL_JSON=" + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
