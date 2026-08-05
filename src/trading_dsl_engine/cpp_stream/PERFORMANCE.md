# cpp_stream performance notes

Benchmarks measure generated native execution only. Input generation, Python/IR
lowering, C++ compilation, mmap setup, and warmup are excluded unless stated.
Absolute throughput varies across hosts; architectural comparisons should use the
same process or workflow job.

## Standard commands

```bash
python scripts/benchmark_cpp_stream.py
python scripts/benchmark_cpp_stream_codegen_fusion.py
CPP_STREAM_RIDGE_CASE=all python scripts/benchmark_cpp_stream_ridge.py
python scripts/benchmark_cpp_stream_einsum.py
python scripts/benchmark_cpp_stream_roll_rets.py
```

The full einsum and roll-rets benchmarks default to 5,000,000 rows, 9 instruments,
one warmup, and ten measured executions.

## Automatic fusion and operator audit: August 5, 2026

The automatic-codegen work was compared with the exact
`26ff5ed1a66a7e0957c4c27fa377018f28f7691d` baseline on one pinned Intel Xeon
Platinum 8370C core using GCC 13.3, `-O3 -march=native -mtune=native -flto`, one
warmup, and ten measured executions. Each pair used the same generated input and
produced the same checksum. Absolute rates on this host are lower than the earlier
8573C audit, so only same-host ratios are compared.

The optimizer is structural rather than formula-specific. It now:

- canonicalizes NaN literals and small literal powers in the neutral IR;
- keeps scalar and fixed-width tensor arithmetic lazy across stage boundaries;
- bundles compatible sibling EWM states, reductions, and Ridge projections;
- fuses a compatible physical producer and output epilogue into one runner stage;
- reuses the canonical `EwmState` update policy for every bundle, preserving
  `span`, `min_periods`, `ignore_na`, and `adjust` semantics.

There is no co-skewness, co-kurtosis, vector-skewness, or multi-projection Ridge
pattern in the optimizer. These cases exercise the same dependency, type, shape,
and state-compatibility rules available to arbitrary formulas.

### EWM higher moments: 5M x 9

| Formula and argument regime | Baseline | Automatic codegen | Speedup |
| --- | ---: | ---: | ---: |
| Co-skewness, finite/default | 0.836 M rows/s | 1.717 M rows/s | 2.05x |
| Co-kurtosis, finite/default | 0.547 M rows/s | 1.520 M rows/s | 2.78x |
| Co-skewness, missing, `span=32`, `min_periods=20`, `ignore_na=False`, `adjust=True` | 0.737 M rows/s | 1.419 M rows/s | 1.93x |
| Co-kurtosis, same pandas-style arguments | 0.486 M rows/s | 1.284 M rows/s | 2.64x |

The finite/default generated source shrank from 77 stages/13 EWM states to one
six-state bundle plus its output epilogue for co-skewness, and from 93 stages/15
EWM states to one eight-state bundle plus its epilogue for co-kurtosis. Generated
source is 16,021 and 19,671 bytes, respectively, and contains packed multiply/FMA
instructions with no external `pow` call.

Earlier on the faster 8573C host, handwritten fused controls established an
approximately 7.92x co-skewness and 10.09x co-kurtosis ceiling over the old plans.
That remains a theoretical, formula-aware ceiling rather than a directly comparable
absolute result. The general optimizer recovers a material fraction without adding
operator-specific state machines; the remaining gap is principally generic state
layout and output/intermediate traffic, not a failure by GCC to vectorize ordinary
arithmetic.

Finite/default automatic-codegen measured seconds:

```text
co-skewness
2.894538, 2.876010, 3.039368, 2.926378, 2.839209,
3.017042, 2.933845, 2.834629, 2.898917, 3.163144

co-kurtosis
3.047023, 3.066419, 3.104320, 3.267215, 3.350727,
3.398217, 3.311926, 3.443382, 4.680847, 3.257409
```

### Other generic bundles

Vector moments used 200,000 x 9 rows with tensor width 16. The Ridge case emitted
only the last row to isolate shared state/solve work from its unusually wide output.

| Case | Baseline | Automatic codegen | Speedup |
| --- | ---: | ---: | ---: |
| Vector skewness | 0.139 M rows/s | 0.481 M rows/s | 3.46x |
| Vector kurtosis | 0.084 M rows/s | 0.457 M rows/s | 5.44x |
| Six projections of one Ridge state | 0.623 M rows/s | 1.873 M rows/s | 3.01x |

### Rolling structures: 200k x 9, 256 periods

| Case | Baseline | Updated | Speedup |
| --- | ---: | ---: | ---: |
| Median | 0.038 M rows/s | 0.102 M rows/s | 2.68x |
| Percentile rank | 0.359 M rows/s | 0.408 M rows/s | 1.14x |
| Entropy | 0.109 M rows/s | 0.107 M rows/s | 0.98x |
| Backfill / first valid | 0.304 M rows/s | 2.289 M rows/s | 7.53x |
| Previous different value, constant-data worst case | 0.308 M rows/s | 2.421 M rows/s | 7.86x |

Median switches from a scan to a fixed-capacity order-statistic tree at 32 periods.
Rank and entropy retain vectorized scans through 2,048 and 1,024 periods because
matched benchmarks showed the tree's maintenance cost is higher below those
crossovers. Backfill now uses a valid-observation deque, and previous-different uses
a run-compressed deque; both are allocation-free and amortized O(1).

### Theil-Sen threshold and selector

These small-row diagnostics use one lane and report median wall seconds because
the expensive full-window count changes with the lookback.

| Periods | Baseline | Updated | Speedup |
| ---: | ---: | ---: | ---: |
| 257 | 1.387 s | 0.183 s | 7.57x |
| 512 | 1.108 s | 0.162 s | 6.84x |
| 513 | 1.179 s | 0.274 s | 4.30x |

The exact selector now remains active through 512 periods. Above that threshold,
the bounded-memory selector assigns candidate ranks once per pass instead of doing
a binary search per point. The updated 257-period measured seconds were `0.183360,
0.167231, 0.191502`; the baseline was `1.573886, 1.387453, 1.385834`.

## Generic einsum: 5M x 9

Successful GitHub-hosted Ubuntu workflow run on August 1, 2026:

```text
rows              5,000,000
instruments       9
input dtype       float64
input format      zero-copy mmap .npy / lazy FeatureList
warmup            1
measured runs     10
compiler          GCC C++20
flags             -O3 -march=native -mtune=native -flto
output            reused /dev/shm mmap
```

| Case | Estimated operations/row | Largest scratch width | Median | Mean | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nf,nf->n`, six lazy features | 54 | 1 | 12.076401 | 12.071388 | 12.097915 M rows/s |
| `...f,...f->...`, same work | 54 | 1 | 12.097036 | 12.091035 | 12.113350 M rows/s |
| `n,n->` scalar reduction | 9 | 1 | 85.027352 | 84.742551 | 85.062145 M rows/s |
| `ij,kj,kl->il`, optimize false | 324 | 9 | 5.532522 | 5.517750 | 5.549309 M rows/s |
| same, greedy | 72 | 2 | 11.604658 | 11.603833 | 11.627009 M rows/s |
| same, optimal | 72 | 2 | 11.624376 | 11.621512 | 11.654887 M rows/s |

Measured distributions:

```text
row_dot
12.070698, 12.056928, 12.097915, 12.071733, 12.075735,
12.092778, 12.077068, 12.071557, 12.061435, 12.037038

ellipsis_dot
12.099192, 12.113350, 12.081671, 12.094155, 12.074098,
12.103474, 12.096291, 12.103080, 12.098407, 12.046629

scalar_reduce
84.879211, 85.062145, 85.029431, 85.035897, 85.025273,
85.057815, 85.051755, 85.052004, 85.025172, 83.207066

nary_none
5.519907, 5.545252, 5.527155, 5.543305, 5.520648,
5.539696, 5.530866, 5.549309, 5.544008, 5.357350

nary_greedy
11.614170, 11.627009, 11.595284, 11.607848, 11.601468,
11.610031, 11.607155, 11.601907, 11.600899, 11.572559

nary_optimal
11.628397, 11.617523, 11.654887, 11.634023, 11.620284,
11.598129, 11.607501, 11.626283, 11.607036, 11.620056
```

All sampled tails were finite. Optimized and unoptimized n-ary cases produced the
same checksum, `2795330.14075`. Greedy reduced estimated work by 77.8% and improved
median throughput by 2.10x. Optimal selected the same-cost path and was 2.10x faster
than left-to-right evaluation.

### Native implementation

Subscripts are parsed once in the backend-neutral IR. The planner expands ellipses,
validates dimensions, canonicalizes labels to integer axis maps, and emits unary or
binary contraction stages. Generated C++ has no runtime subscript parser, dynamic
shape dispatch, or path search.

Optimization modes:

- `False` / `"none"`: left-to-right, matching NumPy's default;
- `True` / `"greedy"`: local work/intermediate minimization;
- `"optimal"`: exhaustive path search through eight operands, then greedy fallback.

Contiguous reductions use fixed-size bulk loads and FMA loops. General mapped loops
handle arbitrary rank, permutation, repeated-label diagonals, scalar operands,
ellipsis broadcasting, and arbitrary reduction axes. Raw tensors remain mmap-backed;
Cat and RBF matrices remain lazy; only selected path intermediates use fixed tensor
scratch.

A header-only external tensor library was evaluated but not adopted. Einsums' generic
optimized API operates on copies rather than views, while Eigen Tensor/TBLIS would
add heavier dependency and materialization requirements. Static generated loops fit
the backend's small compile-time row tensors and lazy-source architecture directly.

## `flows.riskmodel.roll_rets`: 5M x 9 after generic einsum

The exact `flows.riskmodel.roll_rets` expression was rerun in the same full workflow:

```text
median           0.865675 M rows/s
mean             0.865757 M rows/s
best             0.866333 M rows/s
runtime median   5.776 seconds
checksum        -0.790555667227
tail finite      100%
```

Ten runs:

```text
0.865751, 0.865643, 0.865157, 0.864431, 0.864792,
0.866333, 0.866177, 0.866128, 0.865896, 0.866262 M rows/s
```

The prior baseline was 0.855752 M rows/s, so generic einsum did not regress the flow;
the new median is 1.16% higher. The plan remains 50 scalar/vector scratch slots and
one six-wide coefficient-matrix scratch slot. RBF and future-RBF features remain
lazy and feed the contiguous reduction kernel.

## Correctness coverage

The focused suite passed 32 tests before the full benchmarks. Coverage includes:

- implicit and explicit output;
- arbitrary case-sensitive labels;
- scalar operands and scalar reductions;
- ellipsis expansion and broadcasting;
- rejection of ordinary-label broadcasting without ellipsis;
- repeated-label diagonals;
- transposes and outer products;
- raw rank-2 and rank-4 mmap operands;
- nested arbitrary tensor scratch;
- n-ary greedy and optimal paths;
- the exact `roll_rets` graph compared against JAX-flat with finite-output checks,
  `rtol=2e-9`, `atol=2e-9`, and equal-NaN semantics.

## Timestamp-derived grouping

The earlier apparent groupby regression was caused by representing `_ev_ts -> minute`
as nine float64 lanes and hashing it, not by the generic grouped execution scope.
The production `.npy` path preserves native `int64` arithmetic, row-scalar width,
and bounded dense routing. Recent workflow runs measured approximately 21.8 M
rows/s for typed scalar/dense routing versus approximately 4.4 M rows/s for the old
vector/hash representation.

## Cat and Ridge reference

Representative 5M×9 medians from the same class of hosted runner:

| Case | Median throughput |
| --- | ---: |
| Cat root, output width 27 | 11.307 M rows/s |
| Stateful K=3 Ridge predictions | 6.377 M rows/s |
| Stateless K=3 Ridge beta | 9.169 M rows/s |
| One-group grouped stateful predictions | 6.176 M rows/s |
| Three-group grouped stateful predictions | 2.787 M rows/s |

## Regression methodology

1. Keep semantics and workload identical.
2. Run one warmup and at least ten measured executions for full benchmarks.
3. Exclude source generation and native compilation.
4. Report every run and a checksum, not only the best.
5. Verify finite/nontrivial output before accepting throughput.
6. Compare against an independent backend when available.
7. Record contraction work and largest intermediate alongside wall throughput.
8. Prefer generic compile-time policy and layout changes over pattern-specific code.

## Eigen/NNQP and source-pass audit

Ridge preserves the allocation-free fixed-array Cholesky, pivoted Gaussian,
and Jacobi pseudoinverse chain for unconstrained solves. Fixed-size Eigen is
compiled with `EIGEN_DONT_PARALLELIZE` and used by stateless NNQP. Stateless
nonnegative Ridge uses fixed-size active-set NNQP; the stateful path preserves
its exact warm-started coordinate solver. No `Eigen::Dynamic` or Eigen Tensor object is
used in `on_data`.

Use `scripts/benchmark_cpp_stream_io.py` to compare the identical repeated-field Cat
formula over `.npy` and raw mappings. The script also asserts that generated C++ has
one outer row loop and one current-row pointer binding per input. The benchmark is
an architectural I/O check rather than a claim that distinct consumers perform only
one CPU load: repeated within-row reads can occur, but there is no second file scan.
