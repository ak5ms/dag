# cpp_stream performance notes

Benchmarks here measure generated native execution only. Formula compilation, C++
compilation, input generation, mmap setup, and warmup are excluded unless stated.
Absolute throughput varies across hosted CPUs; architectural comparisons should run
in the same process or workflow job.

## Standard commands

```bash
python scripts/benchmark_cpp_stream.py

CPP_STREAM_RIDGE_CASE=all \
python scripts/benchmark_cpp_stream_ridge.py
```

Both scripts default to 5,000,000 rows, 9 instruments, one warmup, and ten measured
executions.

## Timestamp-derived grouping

The apparent drop from roughly 20 M rows/s to roughly 4-5 M rows/s was not caused by
the generic groupby execution scope. It came from representing `_ev_ts -> minute`
as nine float64 lanes and hashing the result.

Same-host ablation:

| Key representation | Throughput |
| --- | ---: |
| Precomputed dense minute | 21.71 M rows/s |
| Precomputed minute + hash | 11.72 M rows/s |
| Vector float calendar + hash | 4.74 M rows/s |
| Row-scalar float calendar + dense | 12.73 M rows/s |
| Row-scalar integer calendar + dense | 20.32 M rows/s |

The production typed `.npy` path preserves `int64` timestamp arithmetic, propagates
row-scalar width, and uses `Key(num_keys=60, ...)` for dense routing. Hosted same-run
comparisons have measured approximately 20-25 M rows/s for native scalar/dense
routing versus approximately 4.2 M rows/s for vector/hash routing.

## Small-width rank

For N=9, exact upper-rank counting is faster than sorting while preserving tie and
NaN semantics:

| Kernel | Sort | Rank count |
| --- | ---: | ---: |
| `xs_rank` | 6.1-6.3 M rows/s | 15.8-16.7 M rows/s |
| grouped `xs_rank` | 5.9-6.4 M rows/s | 12.1-12.9 M rows/s |
| `div -> EWM -> xs_rank` | 5.6-5.7 M rows/s | 10.9-11.6 M rows/s |

The single `XsRankNode` uses compile-time N to choose the algorithm. There is no
grouped rank implementation.

## Cat and Ridge: 5M x 9 baseline

Workflow configuration:

```text
rows              5,000,000
instruments       9
features          3
input dtype       float64
input format      mmap .npy
warmup            1
measured runs     10
compiler          GCC C++20
flags             -O3 -march=native -mtune=native -flto
output            reused /dev/shm mmap
```

Results from the successful full workflow run on August 1, 2026:

| Case | Median | Mean | Best |
| --- | ---: | ---: | ---: |
| Cat root, output width 27 | 11.307 | 11.143 | 11.329 M rows/s |
| Stateful K=3 predictions, `cat` syntax | 6.377 | 6.370 | 6.382 M rows/s |
| Stateful K=3 predictions, separate args | 6.367 | 6.371 | 6.392 M rows/s |
| Stateless K=3 beta | 9.169 | 9.168 | 9.175 M rows/s |
| One-group grouped stateful predictions | 6.176 | 6.177 | 6.182 M rows/s |
| Three-group grouped stateful predictions | 2.787 | 2.786 | 2.788 M rows/s |

Ten-run distributions:

```text
cat_root
11.305, 10.209, 11.034, 11.310, 11.317,
11.273, 11.323, 11.329, 11.327, 11.007

stateful_cat
6.370, 6.382, 6.377, 6.379, 6.310,
6.371, 6.373, 6.378, 6.377, 6.381

stateful_args
6.391, 6.392, 6.373, 6.364, 6.364,
6.366, 6.364, 6.369, 6.353, 6.374

stateless_beta
9.169, 9.175, 9.170, 9.166, 9.164,
9.165, 9.171, 9.168, 9.164, 9.172

one_group
6.172, 6.174, 6.175, 6.180, 6.177,
6.182, 6.173, 6.180, 6.180, 6.176

three_groups
2.786, 2.786, 2.786, 2.786, 2.784,
2.788, 2.787, 2.787, 2.788, 2.787
```

### Interpretation

`Ridge(cat(x1,x2,x3), ...)` and `Ridge(x1,x2,x3, ...)` produced the same generated
C++ cache key and identical checksums. Cat therefore adds no Ridge execution cost;
physical lowering flattens both into the same compile-time `FeatureList`.

The one-group grouped form is 3.2% below direct execution:

```text
1 - 6.176 / 6.377 = 3.15%
```

That is the measured cost of grouped context/resolution for one static group. The
three-group workload performs three independent moment updates and three K=3 solves
per row. Its lower throughput is expected additional work, not an operator-specific
groupby dispatch path.

At 6.377 M rows/s with nine instruments, stateful K=3 Ridge processes about
57.4 million instrument observations and 6.377 million complete K=3 solves per
second in one native row loop.

Cat root writes 27 float64 values per row. Its median corresponds to approximately
2.44 GB/s of output payload before input traffic:

```text
11.307e6 * 27 * 8 = 2.442e9 bytes/s
```

## Generic optimizations used by Ridge

Performance remains one `RidgeNode` for direct and grouped execution. Optimizations
are structural and apply to arbitrary compile-time K:

- compile-time feature width and fixed `std::array` state;
- zero-copy `FeatureList` flattening for nested cat;
- precomputed decay coefficient embedded in the generated type;
- Cholesky-first solve with pivoted and pseudoinverse fallbacks;
- exact `Execution::cross_group` state addressing;
- exact compile-time static partition count;
- finite-panel synchronized moment updates;
- full pairwise-missing fallback when any row value is nonfinite;
- no heap allocation in `on_data`.

There are no K=3-specific kernels, no `GroupedRidgeNode`, and no codegen branch that
selects a grouped Ridge implementation.

## Regression methodology

1. Keep semantics and workload identical.
2. Run one warmup and at least ten measured executions.
3. Exclude source generation and native compilation.
4. Report every run, not only the best.
5. Compare direct and one-group controls before blaming groupby.
6. Use checksums/correctness tests to reject output-changing transformations.
7. Prefer generic compile-time policy and data-layout changes over formula-specific
   branches.

Hosted thresholds should be supplied through environment variables rather than
hard-coded globally because runner CPUs and contention vary.
