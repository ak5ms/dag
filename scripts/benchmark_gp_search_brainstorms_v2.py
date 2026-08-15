from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import numpy as np

from flows.utils import ewm_std, replace
from trading_dsl_engine.base.dsl import Ridge, abs as dsl_abs, cat, ffill, get_beta, get_residuals, purify, shift, var, where
from trading_dsl_engine.cpp_stream import compile_formula

ROWS = int(os.environ.get("GP_BRAINSTORM_ROWS", "250000"))
N = int(os.environ.get("GP_BRAINSTORM_INSTRUMENTS", "9"))
CANDIDATES = int(os.environ.get("GP_BRAINSTORM_CANDIDATES", "8"))
POOL = int(os.environ.get("GP_BRAINSTORM_POOL", "8"))
RUNS = int(os.environ.get("GP_BRAINSTORM_RUNS", "3"))
WARMUPS = int(os.environ.get("GP_BRAINSTORM_WARMUPS", "1"))
ALPHA_PNL_HL = 1440 * 21
RIDGE_HL = 1440 * 5
RIDGE_LAMBDA = 1e-3
THREADS = int(os.environ.get("GP_BRAINSTORM_THREADS", "1"))
ROOT = Path(os.environ.get("GP_BRAINSTORM_ROOT", "/tmp/gp-brainstorm"))
CACHE_ROOT, DATA_ROOT, OUT_ROOT = ROOT / "cache", ROOT / "data", ROOT / "out"
RESULT_PATH = ROOT / "benchmark.json"


def l1_norm(x):
    return purify(x / dsl_abs(x).sum(axis=-1))


def generate_data():
    shutil.rmtree(ROOT, ignore_errors=True)
    for p in (CACHE_ROOT, DATA_ROOT, OUT_ROOT): p.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    names = [*(f"alpha{i}" for i in range(CANDIDATES)), *(f"pool{i}" for i in range(POOL)), "roll_rets", "hs", "is_tradable"]
    paths = {n: DATA_ROOT / f"{n}.npy" for n in names}
    a = {n: np.lib.format.open_memmap(p, mode="w+", dtype=np.float64, shape=(ROWS, N)) for n, p in paths.items()}
    previous = np.zeros(N)
    for start in range(0, ROWS, 32768):
        stop = min(start + 32768, ROWS); m = stop - start
        base = 0.35 * rng.normal(size=(m, 1)) + rng.normal(size=(m, N))
        for i in range(POOL): a[f"pool{i}"][start:stop] = base + .20 * rng.normal(size=(m, N)) + i * .001
        for i in range(CANDIDATES):
            a[f"alpha{i}"][start:stop] = .55 * base + .25 * a[f"pool{i % POOL}"][start:stop] + .35 * rng.normal(size=(m, N))
        sig = a["alpha0"][start:stop]
        lag = np.vstack((previous[None, :], sig[:-1]))
        a["roll_rets"][start:stop] = 2e-5 * lag + rng.normal(0, 4e-4, size=(m, N))
        a["hs"][start:stop] = np.clip(rng.lognormal(-9, .25, size=(m, N)), 2e-5, 8e-4)
        row = np.arange(start, stop)[:, None]
        a["is_tradable"][start:stop] = np.broadcast_to(((row % 1440) < 1200).astype(float), (m, N))
        previous = sig[-1].copy()
    for x in a.values(): x.flush()
    a.clear(); return paths


def cleaned_rets():
    r = var("roll_rets")
    return where(dsl_abs(r) <= .05, replace(r, 0, float("nan")), float("nan"))


def volatility(): return ewm_std(cleaned_rets(), span=ALPHA_PNL_HL)


def pnl_contrib(alpha):
    w = alpha / volatility()
    return shift(ffill(where(var("is_tradable"), w, float("nan"))), 1, 1) * var("roll_rets")


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
    vol, hs = volatility(), var("hs")
    weights = purify(1.0 / (hs * hs))
    alphas = [l1_norm(var(f"pool{i}")) for i in range(POOL)] + [l1_norm(var(f"alpha{i}")) for i in range(CANDIDATES)]
    model = Ridge(*[shift(x, 1, 1) * vol for x in alphas], y=cleaned_rets(), weights=weights,
                  hl=float(RIDGE_HL), lambda_=RIDGE_LAMBDA, nonneg=True, recompute_every=1)
    return dsl_abs(get_beta(model)).mean(axis=0)


def orthogonal_residual(i):
    pool = [l1_norm(var(f"pool{j}")) for j in range(POOL)]
    model = Ridge(*pool, y=l1_norm(var(f"alpha{i}")), weights=1.0, hl=0.0,
                  lambda_=RIDGE_LAMBDA, nonneg=False, recompute_every=1)
    return purify(get_residuals(model))


def xs_orthogonalization_only(): return cat(*(orthogonal_residual(i) for i in range(CANDIDATES)))


def orthogonal_univariate_score(i):
    # Orthogonalize before volatility scaling, then use a K=1 nonnegative temporal Ridge.
    residual = orthogonal_residual(i)
    feature = shift(l1_norm(residual), 1, 1) * volatility()
    hs = var("hs"); weights = purify(1.0 / (hs * hs))
    model = Ridge(feature, y=cleaned_rets(), weights=weights, hl=float(RIDGE_HL),
                  lambda_=RIDGE_LAMBDA, nonneg=True, recompute_every=1)
    return dsl_abs(get_beta(model)).mean(axis=0)


def bench_formula(name, formula, paths):
    cache = CACHE_ROOT / name; shutil.rmtree(cache, ignore_errors=True); cache.mkdir(parents=True)
    os.environ["TRADING_DSL_ENGINE_CPP_STREAM_CACHE"] = str(cache)
    t0 = time.perf_counter(); rt = compile_formula(formula, paths, n_instruments=N, prefetch_rows=16); compile_s = time.perf_counter() - t0
    out = OUT_ROOT / f"{name}.npy"
    for _ in range(WARMUPS): rt.run(out_path=out, threads=THREADS)
    wall=[]; native=[]
    for _ in range(RUNS):
        t0=time.perf_counter(); r=rt.run(out_path=out, threads=THREADS); wall.append(time.perf_counter()-t0); native.append(float(r.seconds))
    vals=np.asarray(np.load(out, mmap_mode="r"))
    return {"name":name,"compile_seconds":compile_s,"median_wall_seconds":statistics.median(wall),
            "median_native_seconds":statistics.median(native),"native_seconds":native,
            "parallel_plan_mode":str(rt.parallel_plan.mode),"parallel_plan_reason":str(rt.parallel_plan.reason),
            "output_shape":list(rt.plan.output_shape),"checksum":float(np.nansum(vals)),
            "generated_cpp_bytes":Path(rt.generated_cpp).stat().st_size,
            "extrapolated_5m_native_seconds":statistics.median(native)*5_000_000/ROWS}


def bench_orthogonal_univariate(paths):
    parts=[]
    for i in range(CANDIDATES):
        print(f"BENCH_START orthogonal_univariate_{i}", flush=True)
        x=bench_formula(f"orthogonal_univariate_{i}", orthogonal_univariate_score(i), paths); parts.append(x); print(json.dumps(x), flush=True)
    med_native=sum(x["median_native_seconds"] for x in parts)
    compile_s=sum(x["compile_seconds"] for x in parts)
    return {"name":"xs_orthogonal_then_univariate_ridge_sequential","compile_seconds":compile_s,
            "median_native_seconds":med_native,"extrapolated_5m_native_seconds":med_native*5_000_000/ROWS,
            "candidate_parts":parts,"note":"sum of 8 separate K=1 programs; these are independent and can be scheduled concurrently on real cores"}


def hat_reuse_ceiling():
    source,binary=ROOT/"hat_reuse.cpp",ROOT/"hat_reuse"
    source.write_text(r'''
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
constexpr int N=9,P=8,M=8,R=200000; using A=std::array<double,P*P>; using V=std::array<double,P>;
inline bool chol(const A&a,A&L){L.fill(0);for(int i=0;i<P;i++)for(int j=0;j<=i;j++){double s=a[i*P+j];for(int k=0;k<j;k++)s-=L[i*P+k]*L[j*P+k];if(i==j){if(s<=1e-14)return false;L[i*P+j]=sqrt(s);}else L[i*P+j]=s/L[j*P+j];}return true;}
inline void solve(const A&L,const V&b,V&x){V z{};for(int i=0;i<P;i++){double s=b[i];for(int k=0;k<i;k++)s-=L[i*P+k]*z[k];z[i]=s/L[i*P+i];}for(int ii=0;ii<P;ii++){int i=P-1-ii;double s=z[i];for(int k=i+1;k<P;k++)s-=L[k*P+i]*x[k];x[i]=s/L[i*P+i];}}
double run(bool cached){std::mt19937_64 g(42);std::normal_distribution<double>d;double sink=0;auto t=std::chrono::steady_clock::now();for(int r=0;r<R;r++){double X[N][P],Y[N][M];for(int n=0;n<N;n++){for(int p=0;p<P;p++)X[n][p]=d(g);for(int m=0;m<M;m++)Y[n][m]=d(g);}A q{};for(int i=0;i<P;i++)for(int j=0;j<P;j++){double s=0;for(int n=0;n<N;n++)s+=X[n][i]*X[n][j];q[i*P+j]=s+(i==j?1e-3:0);}A shared{};if(cached)chol(q,shared);for(int m=0;m<M;m++){V b{},x{};for(int p=0;p<P;p++){double s=0;for(int n=0;n<N;n++)s+=X[n][p]*Y[n][m];b[p]=s;}A L{};if(cached)L=shared;else chol(q,L);solve(L,b,x);sink+=x[0]*1e-300;}}auto s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();if(sink==123)printf("x");return s;}
int main(){double a=run(false),b=run(true);printf("{\"rows\":%d,\"repeated_seconds\":%.9f,\"cached_seconds\":%.9f,\"speedup\":%.9f}\n",R,a,b,a/b);}
''')
    subprocess.run(["g++","-O3","-march=native","-std=c++20",str(source),"-o",str(binary)],check=True)
    x=json.loads(subprocess.check_output([str(binary)],text=True));x["note"]="standalone C++ algebraic ceiling, not cpp_stream performance";return x


def main():
    paths=generate_data(); results=[]
    for name,formula in [("fitness_cat_then_sharpe",fitness_cat_then_sharpe()),
                         ("fitness_sharpe_then_cat",fitness_sharpe_then_cat()),
                         ("current_pool_ridge",current_pool_ridge()),
                         ("xs_orthogonalization_only",xs_orthogonalization_only())]:
        print(f"BENCH_START {name}",flush=True);x=bench_formula(name,formula,paths);results.append(x);print(json.dumps(x),flush=True)
    results.append(bench_orthogonal_univariate(paths)); ceiling=hat_reuse_ceiling()
    payload={"configuration":{"rows":ROWS,"instruments":N,"candidates":CANDIDATES,"pool":POOL,"runs":RUNS,"warmups":WARMUPS,"threads":THREADS},"results":results,"hat_reuse_ceiling":ceiling}
    RESULT_PATH.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print("FINAL_JSON="+json.dumps(payload,sort_keys=True),flush=True)

if __name__ == "__main__": main()
