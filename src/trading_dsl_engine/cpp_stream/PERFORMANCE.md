# cpp_stream performance notes

Benchmarks measure generated native execution only. Input generation, Python/IR
lowering, C++ compilation, mmap setup, and warmup are excluded unless stated.
Absolute throughput varies across hosts; architectural comparisons should use the
same process or workflow job.

## Standard commands

```bash
python scripts/benchmark_cpp_stream.py
CPP_STREAM_RIDGE_CASE=all python scripts/benchmark_cpp_stream_ridge.py
python scripts/benchmark_cpp_stream_einsum.py
python scripts/benchmark_cpp_stream_roll_rets.py
```

The full einsum and roll-rets benchmarks default to 5,000,000 rows, 9 instruments,
one warmup, and ten measured executions.

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
