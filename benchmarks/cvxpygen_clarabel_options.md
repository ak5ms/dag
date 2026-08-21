# CVXPYgen Clarabel lifecycle benchmark

This benchmark isolates the three integration options discussed for using CVXPYgen's generated C interface inside `cpp_stream`.

## Reproduce

```bash
python scripts/benchmark_cvxpygen_clarabel_options.py --sizes 9,24,50 --horizons 8 --runs 10 --warmups 2 --repetitions 5
```

The script requires CVXPYgen/CVXPY plus `git`, Rust/Cargo, and a C compiler; it builds current Clarabel from pinned sources.

## Options

1. **Stock CVXPYgen:** CVXPYgen 1.0.0 generated C using its bundled Clarabel 0.6.0 backend. `cpg_solve()` calls `clarabel_DefaultSolver_new()` on every solve.
2. **Bump Clarabel only:** exactly the same generated C lifecycle, but compile/link it against Clarabel.rs 0.11.1 / current Clarabel.cpp C API. The solver is still reconstructed on every `cpg_solve()`.
3. **Current Clarabel + persistence:** same CVXPYgen parameter maps and result extraction, but change the generated `cpg_solve()` so the first call constructs Clarabel and later calls update `A`, `q`, and `b` through the Clarabel C API before solving.

The MPO is an 8-horizon SOCP. The shared dense covariance factor changes on every solve, so `A`, `q`, and `b` are all dirty. Presolve is disabled in all three variants and tolerances are `1e-8`.

Each reported runtime is the **median of the per-process medians** from five independent processes, each with 2 warmups + 10 timed solves, pinned to one CPU. RSS growth is the mean resident-set increase during the 10 measured calls.

## Results

| assets × horizons | Option 1: stock | Option 2: only 0.11.1 bump | Option 3: 0.11.1 + persistent | Option 3 vs stock |
|---:|---:|---:|---:|---:|
| 9 × 8 | 1.229 ms | 1.364 ms | **1.166 ms** | **5.4% faster** |
| 24 × 8 | 7.860 ms | 8.242 ms | **7.666 ms** | **2.5% faster** |
| 50 × 8 | 42.174 ms | 43.456 ms | **40.460 ms** | **4.2% faster** |

Correctness matched to numerical precision on a 24×8 instance: all three variants took 15 IPM iterations, objective values differed by less than `9e-18`, and the maximum absolute weight difference was `5.4e-14`.

The Clarabel version bump **by itself did not improve runtime** on this workload. It was 4-11% slower than the bundled 0.6 backend in the robust median measurements. That means the earlier performance gap cannot be explained by "CVXPYgen just uses an old Clarabel".

Persistence helps, but it is not a huge numerical-speed win when covariance changes every row: the KKT numerical factorization still has to reflect the new `A`. The measured improvement over stock was about 2.5-5.4% for these sizes.

## Much larger issue: generated solver lifetime / memory

| assets × horizons | Option 1 RSS growth / 10 solves | Option 2 RSS growth / 10 solves | Option 3 RSS growth / 10 solves |
|---:|---:|---:|---:|
| 9 × 8 | 2.68 MB | 2.71 MB | **0.22 MB** |
| 24 × 8 | 9.45 MB | 9.47 MB | **0.24 MB** |
| 50 × 8 | 28.29 MB | 29.75 MB | **0.21 MB** |

Stock CVXPYgen's generated Clarabel path overwrites the global `solver` pointer with a newly allocated solver on each `cpg_solve()` and does not call `clarabel_DefaultSolver_free()` in that path. The leak scales with problem size. Option 3 retains exactly one solver, so RSS is effectively flat after allocator warmup.

This memory result is the stronger reason to change the lifecycle. Even if persistence were runtime-neutral, the stock repeated-solve lifecycle is unsuitable for a long cpp_stream hot loop.

## Exact code difference

CVXPYgen's public interface remains the same in all cases:

```c
cpg_update_expected_returns(...);
cpg_update_risk_factor(...);
cpg_update_current_weights(...);
cpg_update_risk_radius(...);
cpg_solve();
const double *w = CPG_Result.prim->weights;
```

### Option 1 / Option 2 internals

```c
void cpg_solve(void) {
    cpg_canonicalize_q();
    cpg_canonicalize_A();
    cpg_canonicalize_b();
    cpg_copy_all();

    solver = clarabel_DefaultSolver_new(
        &P, q, &A, b, n_cones, cones, &settings);
    clarabel_DefaultSolver_solve(solver);
    solution = clarabel_DefaultSolver_solution(solver);

    cpg_retrieve_prim();
    cpg_retrieve_dual();
    cpg_retrieve_info();
}
```

Option 2 changes the linked Clarabel implementation only; the generated lifecycle is still the above.

### Option 3 internals

```c
void cpg_solve(void) {
    cpg_canonicalize_dirty_parameters();

    if (!solver) {
        cpg_copy_all();
        solver = clarabel_DefaultSolver_new(
            &P, q, &A, b, n_cones, cones, &settings);
    } else {
        if (A_changed) {
            cpg_copy_A();
            clarabel_DefaultSolver_update_A(solver, A.x, nnzA);
        }
        if (q_changed) {
            cpg_copy_q();
            clarabel_DefaultSolver_update_q(solver, q, n);
        }
        if (b_changed) {
            cpg_copy_b();
            clarabel_DefaultSolver_update_b(solver, b, m);
        }
    }

    clarabel_DefaultSolver_solve(solver);
    solution = clarabel_DefaultSolver_solution(solver);
    cpg_retrieve_prim();
    cpg_retrieve_dual();
    cpg_retrieve_info();
}
```

The **CVXPYgen boundary does not move** in Option 3. CVXPYgen still owns:

- DPP / canonical problem generation
- parameter → canonical `A/q/b` maps
- dirty-parameter tracking
- cone layout
- primal/dual inverse maps
- public `cpg_update_*`, `cpg_solve`, and `CPG_Result` API

The only changed responsibility is Clarabel object lifetime inside the generated backend.

## Conclusion

For `cpp_stream`, the lowest-maintenance design is still to call the **CVXPYgen C API directly**. There is no performance evidence here for replacing CVXPYgen's canonicalization with custom DAG code.

However, I would make a very small CVXPYgen Clarabel-backend patch before using it in a long-running hot loop:

1. Keep one `ClarabelDefaultSolver*` alive across `cpg_solve()` calls.
2. Route dirty canonical blocks to `clarabel_DefaultSolver_update_A/q/b`.
3. Add an explicit cleanup/destructor API.
4. Separately make the generated workspace instance-owned if multiple optimizer instances must execute concurrently. The benchmark above isolates solver version/lifetime; it does not address CVXPYgen's current global generated workspace.

Simply replacing Clarabel 0.6 with 0.11.1 is not sufficient and, on this benchmark, was not faster.
