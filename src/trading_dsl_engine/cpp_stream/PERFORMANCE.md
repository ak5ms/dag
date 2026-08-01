# cpp_stream performance notes

Benchmarks measure generated native execution only. Input generation, Python/IR
lowering, C++ compilation, mmap setup, and warmup are excluded unless stated.
Absolute throughput varies across hosts; architectural comparisons should use the
same process or workflow job.

## Standard commands

```bash
python scripts/benchmark_cpp_stream.py
CPP_STREAM_RIDGE_CASE=all python scripts/benchmark_cpp_stream_ridge.py
python scripts/benchmark_cpp_stream_roll_rets.py
```

The full benchmarks default to 5,000,000 rows, 9 instruments, one warmup, and ten
measured executions.

## Timestamp-derived grouping

The earlier apparent groupby regression was caused by representing `_ev_ts -> minute`
as nine float64 lanes and hashing it, not by the generic grouped execution scope.

Same-host ablation:

| Key representation | Throughput |
| --- | ---: |
| Precomputed dense minute | 21.71 M rows/s |
| Precomputed minute + hash | 11.72 M rows/s |
| Vector float calendar + hash | 4.74 M rows/s |
| Row-scalar float calendar + dense | 12.73 M rows/s |
| Row-scalar integer calendar + dense | 20.32 M rows/s |

The production `.npy` path preserves native `int64` timestamp arithmetic,
row-scalar width, and bounded dense routing. Recent focused workflow runs measured
approximately 21.8 M rows/s for typed row-scalar dense routing versus 4.38 M rows/s
for the old vector/hash representation.

## Cat and Ridge: 5M x 9

Configuration:

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

| Case | Median | Mean | Best |
| --- | ---: | ---: | ---: |
| Cat root, output width 27 | 11.307 | 11.143 | 11.329 M rows/s |
| Stateful K=3 predictions, `cat` syntax | 6.377 | 6.370 | 6.382 M rows/s |
| Stateful K=3 predictions, separate args | 6.367 | 6.371 | 6.392 M rows/s |
| Stateless K=3 beta | 9.169 | 9.168 | 9.175 M rows/s |
| One-group grouped stateful predictions | 6.176 | 6.177 | 6.182 M rows/s |
| Three-group grouped stateful predictions | 2.787 | 2.786 | 2.788 M rows/s |

`Ridge(cat(x1,x2,x3), ...)` and separate feature arguments generated the same native
source and checksum. One-group grouped execution was 3.15% below direct execution.
The three-group workload performs three independent moment updates and solves per
row, so its lower throughput reflects real work rather than a grouped Ridge class.

## `flows.riskmodel.roll_rets`: 5M x 9

This benchmark imports the actual expression object from `flows.riskmodel`; it is not
a copied or reduced formula. The native plan includes:

- session-grouped cumulative volume state;
- RBF basis and future RBF mass;
- per-instrument `InstrumentBasisMean` coefficients;
- `einsum("nf,nf->n")`;
- named POV stateless policies;
- forward fill and shift;
- comparisons, boolean logic, where/fillna, and arithmetic.

Plan shape:

```text
scalar/vector scratch slots   50
matrix scratch slots           1
matrix scratch width           6
group capacity              4096
```

Successful workflow run on August 1, 2026:

```text
rows             5,000,000
instruments      9
warmup           1
measured runs    10
median           0.855752 M rows/s
mean             0.855213 M rows/s
best             0.856243 M rows/s
checksum        -0.790555667227
tail finite      100%
```

Ten measured runs:

```text
0.855827, 0.854544, 0.856243, 0.855371, 0.856076,
0.852580, 0.856137, 0.855779, 0.853851, 0.855725 M rows/s
```

The median corresponds to approximately:

```text
5.842 seconds per 5,000,000 rows
7.702 million instrument observations/second
```

The run distribution is tight: the slowest and fastest measurements differ by less
than 0.43% of the median. The benchmark output tail was entirely finite, so the
result is not an all-NaN or masked-output artifact.

### Correctness reference

The focused suite imports the same `roll_rets` object and runs it through both
cpp_stream and JAX-flat on identical session/tradability/missing-value data. It
asserts that each output contains finite values and then compares with:

```text
rtol = 2e-9
atol = 2e-9
equal_nan = true
```

The full focused suite passed 18 tests before the benchmark step.

## Generic implementation choices used by `roll_rets`

The implementation remains operator- and formula-agnostic:

- backend-neutral named stateless calls carry a stable native policy name;
- RBF values are lazy compile-time-width feature sources;
- nested Cat and basis values flatten into `FeatureList<Sources...>`;
- only the coefficient matrix needed by einsum is materialized;
- history nodes use `Execution::state_index` and fixed arrays;
- direct and grouped plans use the same node classes;
- explicit NaN/infinity source types avoid invalid generated C++ literals;
- no Python per-row execution and no heap allocation in operator `on_data`.

No `GroupedInstrumentBasisMean`, grouped history variants, or `roll_rets`-specific
native node was added.

## Regression methodology

1. Keep semantics and workload identical.
2. Run one warmup and at least ten measured executions for full benchmarks.
3. Exclude source generation and native compilation.
4. Report every run and a checksum, not only the best.
5. Verify finite/nontrivial output before accepting throughput.
6. Compare against an independent backend when available.
7. Prefer generic compile-time policy and layout changes over formula-specific code.

Hosted performance thresholds should be supplied through environment variables
rather than hard-coded globally because CPU model, filesystem, and contention vary.
