"""Benchmark three CVXPYgen/Clarabel C-interface lifecycles on a changing-covariance MPO.

Option 1: stock CVXPYgen-generated Clarabel backend (bundled Clarabel 0.6.0,
          one clarabel_DefaultSolver_new() per cpg_solve()).
Option 2: identical generated C lifecycle, but link against Clarabel 0.11.1.
Option 3: Clarabel 0.11.1 plus a minimal generated-C patch that constructs the
          solver once and uses fixed-sparsity A/q/b updates on later cpg_solve() calls.

The benchmark intentionally changes the covariance factor on every solve, so A,
q, and b are all dirty.  It times the public CPG interface, not the Python wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import textwrap
import time

import cvxpy as cp
from cvxpygen import cpg

CLARABEL_CPP_COMMIT = "0de6259a3edfd5cc041ec42b2148599ce63e73cb"
CLARABEL_RS_TAG = "v0.11.1"

DRIVER = r'''\
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <time.h>
#include "cpg_solve.h"
#include "cpg_workspace.h"
#ifndef N_ASSETS
#error N_ASSETS required
#endif
#ifndef N_HORIZONS
#error N_HORIZONS required
#endif
#ifndef WARMUPS
#define WARMUPS 2
#endif
#ifndef RUNS
#define RUNS 10
#endif
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
static long rss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; long rss = -1;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "VmRSS: %ld kB", &rss) == 1) break;
    }
    fclose(f);
    return rss;
}
static void set_problem(int k) {
    const int N = N_ASSETS, H = N_HORIZONS;
    const double phase = 0.013 * (double)(k + 1);
    for (int t=0;t<H;t++) for (int i=0;i<N;i++) {
        const int idx = t*N+i;
        const double er = 2e-4 * sin(0.17*(i+1) + 0.31*(t+1) + phase);
        const double hs = 5e-5 + 2e-5 * (0.5 + 0.5*cos(0.11*(i+1) + phase));
        cpg_update_expected_returns(idx, er);
        cpg_update_half_spread(idx, hs);
    }
    for (int i=0;i<N;i++)
        cpg_update_current_weights(i, 0.002 * sin(0.23*(i+1) + phase));
    for (int t=0;t<H;t++)
        cpg_update_risk_radius(t, 0.08 + 0.002 * sin(0.19*(t+1) + phase));
    const double scale = 1.0 + 0.002 * sin(phase);
    for (int j=0;j<N;j++) for (int i=0;i<N;i++) {
        double v = (i == j)
            ? 1.0 + 0.002 * i / (double)(N > 1 ? N-1 : 1)
            : 0.003 * cos(0.07*(i+1)*(j+1));
        cpg_update_risk_factor(i + j*N, scale * v); // Fortran-order matrix parameter
    }
}
int main(void) {
    cpg_set_solver_default_settings();
    cpg_set_solver_verbose(0);
    cpg_set_solver_presolve_enable(0);
    cpg_set_solver_max_iter(200);
    cpg_set_solver_tol_gap_abs(1e-8);
    cpg_set_solver_tol_gap_rel(1e-8);
    cpg_set_solver_tol_feas(1e-8);
    for (int k=0;k<WARMUPS;k++) { set_problem(k); cpg_solve(); }
    const long rss0 = rss_kb();
    double samples[RUNS];
    volatile double checksum = 0.0;
    for (int k=0;k<RUNS;k++) {
        set_problem(WARMUPS+k);
        const double t0 = now_s();
        cpg_solve();
        samples[k] = (now_s()-t0)*1e3;
        checksum += CPG_Result.prim->weights[0] + 1e-9 * CPG_Result.info->iter;
    }
    const long rss1 = rss_kb();
    double sum=0,min=samples[0],max=samples[0],sorted[RUNS];
    for (int k=0;k<RUNS;k++) { sorted[k]=samples[k]; sum+=samples[k]; if(samples[k]<min)min=samples[k]; if(samples[k]>max)max=samples[k]; }
    for(int i=0;i<RUNS;i++) for(int j=i+1;j<RUNS;j++) if(sorted[j]<sorted[i]) {double x=sorted[i]; sorted[i]=sorted[j]; sorted[j]=x;}
    const double median = (RUNS%2) ? sorted[RUNS/2] : 0.5*(sorted[RUNS/2-1]+sorted[RUNS/2]);
    int p90i = (int)ceil(0.90*RUNS)-1; if(p90i<0)p90i=0; if(p90i>=RUNS)p90i=RUNS-1;
    printf("{\"assets\":%d,\"horizons\":%d,\"runs\":%d,\"mean_ms\":%.9f,\"median_ms\":%.9f,\"p90_ms\":%.9f,\"min_ms\":%.9f,\"max_ms\":%.9f,\"rss_delta_kb\":%ld,\"checksum\":%.12g}\\n",
        N_ASSETS,N_HORIZONS,RUNS,sum/RUNS,median,sorted[p90i],min,max,(rss0>=0&&rss1>=0)?rss1-rss0:-1,(double)checksum);
    return 0;
}
'''


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    kwargs = {"cwd": cwd, "check": True, "text": True}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
    proc = subprocess.run(cmd, **kwargs)
    return proc.stdout.strip() if capture else ""


def build_problem(n_assets: int, n_horizons: int) -> cp.Problem:
    w = cp.Variable((n_horizons, n_assets), name="weights")
    u = cp.Variable((n_horizons, n_assets), name="turnover")
    er = cp.Parameter((n_horizons, n_assets), name="expected_returns")
    hs = cp.Parameter((n_horizons, n_assets), nonneg=True, name="half_spread")
    current = cp.Parameter(n_assets, name="current_weights")
    radius = cp.Parameter(n_horizons, nonneg=True, name="risk_radius")
    factor = cp.Parameter((n_assets, n_assets), name="risk_factor")
    previous = cp.vstack([current, w[:-1]])
    delta = w - previous
    constraints: list[cp.Constraint] = [u >= delta, u >= -delta]
    constraints.extend(cp.SOC(radius[t], factor @ w[t]) for t in range(n_horizons))
    problem = cp.Problem(
        cp.Minimize(-cp.sum(cp.multiply(er, w)) + cp.sum(cp.multiply(hs, u))),
        constraints,
    )
    if not problem.is_dcp(dpp=True):
        raise RuntimeError("benchmark MPO must be DPP-compliant")
    return problem


def generate(problem: cp.Problem, out: Path) -> float:
    if out.exists():
        shutil.rmtree(out)
    t0 = time.perf_counter()
    cpg.generate_code(
        problem,
        code_dir=str(out),
        solver="CLARABEL",
        wrapper=False,
        enable_settings=["verbose", "max_iter", "tol_gap_abs", "tol_gap_rel", "tol_feas"],
    )
    return time.perf_counter() - t0


def build_current_clarabel(root: Path) -> tuple[Path, Path]:
    cpp = root / "Clarabel.cpp"
    rs = root / "Clarabel.rs"
    if not cpp.exists():
        run(["git", "clone", "https://github.com/oxfordcontrol/Clarabel.cpp.git", str(cpp)])
        run(["git", "checkout", CLARABEL_CPP_COMMIT], cwd=cpp)
    if not rs.exists():
        run(["git", "clone", "--depth", "1", "--branch", CLARABEL_RS_TAG,
             "https://github.com/oxfordcontrol/Clarabel.rs.git", str(rs)])
    target_rs = cpp / "Clarabel.rs"
    if target_rs.exists():
        shutil.rmtree(target_rs)
    shutil.copytree(rs, target_rs, ignore=shutil.ignore_patterns(".git"))
    run(["cargo", "build", "--release", "--manifest-path", str(cpp / "rust_wrapper/Cargo.toml")])
    return cpp / "include", cpp / "rust_wrapper/target/release/libclarabel_c.a"


def build_stock_clarabel(generated: Path) -> tuple[Path, Path]:
    solver = generated / "c/solver_code"
    run(["cargo", "build", "--release", "--manifest-path", str(solver / "rust_wrapper/Cargo.toml")])
    return solver / "include", solver / "rust_wrapper/target/release/libclarabel_c.a"


def patch_persistent(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    path = dst / "c/src/cpg_solve.c"
    text = path.read_text()
    start = text.index("void cpg_solve(){")
    end = text.index("// Update solver settings", start)
    old = text[start:end]
    p_init = re.search(r"clarabel_CscMatrix_init\(&P,.*?;\n", old, flags=re.S).group(0).strip()
    a_init = re.search(r"clarabel_CscMatrix_init\(&A,.*?;\n", old, flags=re.S).group(0).strip()
    new_solver = re.search(r"solver = clarabel_DefaultSolver_new\(.*?\);", old, flags=re.S).group(0).strip()
    q_n = int(re.search(r"void cpg_canonicalize_q\(\)\{\s*for\(i=0; i<(\d+);", text).group(1))
    a_n = int(re.search(r"void cpg_canonicalize_A\(\)\{\s*for\(i=0; i<(\d+);", text).group(1))
    b_n = int(re.search(r"void cpg_canonicalize_b\(\)\{\s*for\(i=0; i<(\d+);", text).group(1))
    settings_start = old.index("settings.max_iter")
    marker = "settings.presolve_enable = Canon_Settings.presolve_enable;"
    settings_end = old.index(marker) + len(marker)
    settings_assign = old[settings_start:settings_end]
    replacement = f'''void cpg_solve(){{
  if (Canon_Outdated.q) cpg_canonicalize_q();
  if (Canon_Outdated.A) cpg_canonicalize_A();
  if (Canon_Outdated.b) cpg_canonicalize_b();
  if (!solver) {{
    cpg_copy_all();
    {p_init}
    {a_init}
    settings = clarabel_DefaultSettings_default();
    {settings_assign}
    {new_solver}
  }} else {{
    if (Canon_Outdated.A) {{ cpg_copy_A(); clarabel_DefaultSolver_update_A(solver, Canon_Params_conditioning.A->x, {a_n}); }}
    if (Canon_Outdated.q) {{ cpg_copy_q(); clarabel_DefaultSolver_update_q(solver, Canon_Params_conditioning.q, {q_n}); }}
    if (Canon_Outdated.b) {{ cpg_copy_b(); clarabel_DefaultSolver_update_b(solver, Canon_Params_conditioning.b, {b_n}); }}
  }}
  clarabel_DefaultSolver_solve(solver);
  solution = clarabel_DefaultSolver_solution(solver);
  cpg_retrieve_prim();
  cpg_retrieve_dual();
  cpg_retrieve_info();
  Canon_Outdated.q = 0;
  Canon_Outdated.A = 0;
  Canon_Outdated.b = 0;
}}

'''
    path.write_text(text[:start] + replacement + text[end:])


def compile_binary(
    generated: Path,
    out: Path,
    driver: Path,
    solver_include: Path,
    solver_lib: Path,
    n_assets: int,
    n_horizons: int,
    runs: int,
    warmups: int,
    *,
    current_headers: bool,
) -> None:
    compat = generated / "current_clarabel_compat"
    include_args = [f"-I{generated / 'c/include'}"]
    if current_headers:
        compat.mkdir(exist_ok=True)
        (compat / "Clarabel").write_text("#include <clarabel.h>\n")
        include_args += [f"-I{compat}", f"-I{solver_include}"]
    else:
        include_args += [f"-I{solver_include}"]
    cmd = [
        os.environ.get("CC", "gcc"), "-O3",
        f"-DN_ASSETS={n_assets}", f"-DN_HORIZONS={n_horizons}",
        f"-DRUNS={runs}", f"-DWARMUPS={warmups}",
        *include_args,
        str(generated / "c/src/cpg_workspace.c"),
        str(generated / "c/src/cpg_solve.c"),
        str(driver), str(solver_lib), "-o", str(out),
        "-lm", "-ldl", "-lpthread",
    ]
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="9,24,50")
    parser.add_argument("--horizons", type=int, default=8)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--work-dir", type=Path, default=Path(".cvxpygen_clarabel_bench"))
    parser.add_argument("--json-out", type=Path, default=Path("cvxpygen_clarabel_options.json"))
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    driver = args.work_dir / "bench_driver.c"
    driver.write_text(DRIVER)
    current_inc, current_lib = build_current_clarabel(args.work_dir / "current_clarabel")
    generated: dict[int, Path] = {}
    codegen_s: dict[int, float] = {}
    for n in sizes:
        generated[n] = args.work_dir / f"gen_{n}x{args.horizons}"
        codegen_s[n] = generate(build_problem(n, args.horizons), generated[n])
    stock_inc, stock_lib = build_stock_clarabel(generated[sizes[0]])

    binaries: dict[tuple[int, int], Path] = {}
    for n in sizes:
        for option in (1, 2, 3):
            source = generated[n]
            if option == 3:
                source = args.work_dir / f"gen_{n}x{args.horizons}_persistent"
                patch_persistent(generated[n], source)
            out = args.work_dir / f"bench_{n}x{args.horizons}_option{option}"
            compile_binary(
                source, out, driver,
                stock_inc if option == 1 else current_inc,
                stock_lib if option == 1 else current_lib,
                n, args.horizons, args.runs, args.warmups,
                current_headers=option != 1,
            )
            binaries[n, option] = out

    results: list[dict[str, object]] = []
    for rep in range(args.repetitions):
        for n in sizes:
            for option in (1, 2, 3):
                payload = json.loads(run([str(binaries[n, option])], capture=True))
                payload.update(option=option, repetition=rep + 1)
                results.append(payload)
    output = {
        "options": {
            "1": "stock CVXPYgen Clarabel 0.6.0; new solver each cpg_solve",
            "2": "Clarabel 0.11.1; same new-solver-each-cpg_solve lifecycle",
            "3": "Clarabel 0.11.1; persistent solver with fixed-sparsity A/q/b updates",
        },
        "clarabel_cpp_commit": CLARABEL_CPP_COMMIT,
        "clarabel_rs_tag": CLARABEL_RS_TAG,
        "codegen_seconds": codegen_s,
        "results": results,
    }
    args.json_out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
