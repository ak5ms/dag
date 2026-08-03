# cpp_stream reductions and parallel benchmark

The parallel branch is based directly on `cpp-stream-backend` commit `a21928f57b79573b8985edf65a8b97d950d87905`.

## Full DSL reduction composition

Numeric outputs from reductions now retain their complete logical shape and may feed ordinary elementwise DSL operators, `cumsum`, `ewm`, `ffill`, and `shift`. Temporal reductions may also feed downstream algebra. A downstream graph that depends on a temporal reduction is evaluated cumulatively in one pass and implicitly emits only its final value.

The exact alpha-search topology is covered end to end:

```python
features = [xs_rank(ewm(returns, i)) for i in range(1, 5)]
pnls = cat(*[
    default_alpha_pnl(
        feature,
        roll_rets=returns,
        is_tradable=var("is_tradable_out0"),
        hl=1440,
    )
    for feature in features
])
pnl = pnls.sum(axis=[1])

path_sharpe = (
    pnl.cumsum() / (pnl ** 2).cumsum().pow(0.5)
).emit("last")

reduced_sharpe = pnl.sum(axis=[0, 1]) / pnl.std(axis=[0, 1])
```

Both forms compile and run as final-output streaming plans. Explicit `emit("last")` remains terminal; the implicit final emission applies only to downstream graphs that depend on a temporal reduction.

## Automatic shapes and direct NumPy output

The public compiler infers `n_instruments` from source row shapes. It selects the unique most frequent non-scalar leading row extent and rejects ambiguous ties or scalar-only mappings instead of guessing.

`runtime.run()` now creates a valid temporary `.npy` file when no output path is supplied. A supplied `.npy` path is also written directly: C++ maps the complete file and writes at the NumPy payload offset, so there is no conversion or full-output copy. `RunResult.load()` opens either `.npy` or raw output using the recorded complete logical shape.

Full 5M × 9 output benchmark, one warmup and ten alternating measured runs:

| Output | Native median | End-to-end median |
| --- | ---: | ---: |
| Raw mmap | 0.215533 s | 0.231861 s |
| Direct `.npy` | 0.213737 s | 0.229781 s |

The `.npy` payload began at byte 128. Direct `.npy` throughput was **1.0084x** raw for native execution and **1.0091x** raw end to end, so the shape-aware format added no performance regression.

## Key-hinted `roll_rets`

`flows.roll_rets_keys` wraps the redundantly lane-repeated `session_start0` key with:

```python
key(session_start0, row_scalar=True, dtype="float64")
```

No `num_keys` hint is used because absolute session timestamps have an unbounded domain. `row_scalar=True` allows one group lookup per row rather than repeating the same lookup for every instrument.

Full 5M × 9 `roll_rets` benchmark, one warmup and ten alternating runs:

| Threads | Baseline | Keyed | Keyed speedup |
| ---: | ---: | ---: | ---: |
| 1 | 6.286373 s | 5.722270 s | **1.0986x** |
| 4 | 3.918679 s | 3.796296 s | **1.0322x** |

Baseline and keyed outputs were bitwise equivalent with checksum `-344.029681514`.

## Eigen release and allocation benchmark

cpp_stream release builds now define `EIGEN_NO_DEBUG` explicitly in addition to `NDEBUG` and `EIGEN_DONT_PARALLELIZE`. Fixed-size Eigen expressions use `noalias()` where appropriate.

Following the `stulp/eigenrealtime` guidance, a separate allocation-audit build defines `EIGEN_RUNTIME_NO_MALLOC` and disables malloc inside the fixed-size solver. `EIGEN_NO_DEBUG` is intentionally absent from that audit build because it disables Eigen's runtime assertions. The audit passed 10,000 solves without a dynamic allocation.

Full benchmark: 5M rows × 9 instruments, one 3×3 SPD solve per row, one warmup and ten alternating runs:

| Solver | Median | Throughput |
| --- | ---: | ---: |
| Custom fixed-size Cholesky | 0.502774 s | 9.9448 M rows/s |
| Eigen fixed-size solver | 0.680973 s | 7.3424 M rows/s |

Eigen achieved **0.7383x** custom throughput; the custom hot-path solver remained approximately **1.354x faster**. Both produced checksum `12685927.6694`.

## Streaming terminal reductions

The terminal-reduction benchmark uses 5,000,000 rows, 9 instruments, and 3 computed features, with one warmup and ten measured runs. These reductions use the default `ignore_na=True`; selecting `ignore_na=False` changes missing-value propagation but not the planner's row, lane, or terminal scheduling rules.

| Execution | Median throughput | Median runtime | Output bytes |
| --- | ---: | ---: | ---: |
| Full 3-feature computed result | 12.952441 M rows/s | 0.386030 s | 1,080,000,000 |
| Fused 3-feature `sum(axis=0)` | 26.839841 M rows/s | 0.186290 s | 216 |
| Fused 3-feature `mean(axis=0)` | 26.584917 M rows/s | — | 216 |
| Fused 3-feature `std(axis=0)` | 19.658196 M rows/s | — | 216 |
| Simple `x.sum(axis=0)` | 116.006584 M rows/s | 0.043101 s | 72 |
| Simple `cumsum(x).emit("last")` | 114.549663 M rows/s | 0.043649 s | 72 |

The fused 3-feature temporal sum is **2.072x faster** than writing the full computed result. Including a NumPy post-hoc reduction of the materialized result, the fused native path is **4.993x faster** (`0.186290 s` versus `0.930111 s`).

The earlier table compared the simple one-feature cumsum/emit graph with a three-feature computed reduction, so its apparent large `emit` advantage was not an apples-to-apples operator comparison. On the same finite input, `cumsum(x).emit("last")` produced the same values as `x.sum(axis=0)` but was slightly slower: **0.987x** the sum throughput.

### `emit("last")` semantics

`emit("last")` is terminal output selection, not a reduction. For an expression `f(x)` evaluated row by row, it retains the latest row value and writes only that value during finalization. In NumPy terms, it is analogous to:

```python
values = f(x)
result = values[-1]
```

Thus:

```python
cumsum(x).emit("last")
```

is analogous to:

```python
np.cumsum(x, axis=0)[-1]
```

For entirely finite input, this equals `np.sum(x, axis=0)`. It is not generally equivalent under missing values. cpp_stream cumsum leaves its accumulated state unchanged when an observation is non-finite but emits NaN for that row. Consequently, if the final observation of a lane is NaN, `cumsum(x).emit("last")` returns NaN for that lane, while `x.sum(axis=0, ignore_na=True)` returns the sum of its finite observations. An all-NaN temporal sum also returns NaN rather than NumPy `nansum`'s zero.

Use `sum(axis=0)` when the requested operation is a reduction over time. Use `emit("last")` when the requested result is the final value of a temporal expression such as cumsum, EWM, ffill, shift, or a composed stateful graph. Any material performance advantage of cumsum/emit over an equivalent sum would be an implementation gap rather than a semantic reason to select `emit`.

## Parallel scheduling

- Temporal reductions and `emit("last")` have one final accumulator owner and remain single-threaded until a deterministic generic accumulator-merge layer is added.
- Row-only reductions are row-sharded when all rows are independent.
- A row reduction after lane-local temporal work may remain lane-sharded when it retains the instrument axis. Each worker accumulates and writes only its own lane interval.
- A row reduction that removes the instrument axis is cross-sectional and is not lane-sharded.

## Dedicated 5M × 9 reduction scaling benchmark

The permanent `scripts/benchmark_cpp_stream_parallel_reductions.py` benchmark uses 5,000,000 rows and 9 instruments. It performs one warmup and ten measured runs for 1, 2, and 4 requested threads. Thread-count order alternates forward and backward between repetitions, workers are pinned, output files are pre-sized, and asynchronous writeback is disabled. This matches the established 5M × 9 benchmark scale so fixed setup and scheduling costs are amortized consistently.

The row-sharded cases construct eight stateless features and reduce across the instrument axis. The lane-sharded cases use sixteen compute-heavy independent EWM feature streams and reduce only the feature axis. This intentionally tests the profitable lane-parallel regime; lightweight lane graphs remain subject to automatic serial fallback.

Every parallel output is compared exactly with the serial output, including its NaN mask. CI requires every measured multicore count—not only the largest one—to exceed serial throughput. The accepted median floors remain 1.15x for row-sharded reductions and 1.01x for lane-sharded reductions.

| Reduction graph | Planner | 1 thread | 2 threads | 2-thread speedup | 4 threads | 4-thread speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stateless feature `sum` | rows | 5.910180 M rows/s | 10.344889 M rows/s | **1.750x** | 11.638776 M rows/s | **1.969x** |
| Stateless feature `mean` | rows | 5.622389 M rows/s | 9.931293 M rows/s | **1.766x** | 11.207479 M rows/s | **1.993x** |
| Stateless feature `std` | rows | 3.861272 M rows/s | 7.583330 M rows/s | **1.964x** | 8.603668 M rows/s | **2.228x** |
| EWM feature `sum` | lanes | 0.937276 M rows/s | 1.419829 M rows/s | **1.515x** | 1.272564 M rows/s | **1.358x** |
| EWM feature `std` | lanes | 0.607844 M rows/s | 0.947984 M rows/s | **1.560x** | 1.041905 M rows/s | **1.714x** |

The runner exposes four logical CPUs as two physical cores with SMT. Worker pinning orders one logical CPU from each physical core before adding SMT siblings. On this topology, the selected order is `0, 2, 1, 3`, making the two-thread measurements use both physical cores.

## Existing parallel workloads after reduction integration

| Graph | 1 thread | 4 threads | Speedup |
| --- | ---: | ---: | ---: |
| Root Cat, 5M × 9 × 3 | 13.660 M rows/s | 39.781 M rows/s | 2.912x |
| Deep elementwise, N=64 | 2.588879 M rows/s | 7.253506 M rows/s | 2.802x |
| Optimized einsum, N=64 | 1.197148 M rows/s | 3.040936 M rows/s | 2.540x |
| Stateless K=3 Ridge | 2.224326 M rows/s | 6.083227 M rows/s | 2.735x |
| `roll_rets`, N=9 | 0.869686 M rows/s | 1.349784 M rows/s | 1.552x |

Light lane-local temporal graphs remain intentionally unattractive when parallelism is explicitly forced. The `threads=0` profitability policy keeps such graphs serial when the estimated work does not justify parallel execution.
