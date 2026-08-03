# cpp_stream reductions and parallel benchmark

The parallel branch is based directly on `cpp-stream-backend` commit `a21928f57b79573b8985edf65a8b97d950d87905`.

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

The row-sharded cases construct eight stateless features and reduce across the instrument axis. The lane-sharded cases construct six independent EWM feature streams and reduce only the feature axis, retaining instrument-local temporal state.

Every parallel output is compared exactly with the serial output, including its NaN mask. CI requires every measured multicore count—not only the largest one—to exceed serial throughput. The hosted-runner median floors are 1.15x for row-sharded reductions and 1.01x for lane-sharded reductions. The lower lane floor is deliberate: with only 9 lanes on a two-core/four-thread runner, the four-thread EWM sum is bandwidth- and SMT-limited and has repeatedly measured about 1.04–1.06x, while the two-thread physical-core configuration is materially faster.

| Reduction graph | Planner | 1 thread | 2 threads | 2-thread speedup | 4 threads | 4-thread speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stateless feature `sum` | rows | 5.818830 M rows/s | 10.471180 M rows/s | **1.800x** | 11.684285 M rows/s | **2.008x** |
| Stateless feature `mean` | rows | 5.849669 M rows/s | 10.148507 M rows/s | **1.735x** | 11.255023 M rows/s | **1.924x** |
| Stateless feature `std` | rows | 3.968740 M rows/s | 7.742193 M rows/s | **1.951x** | 8.621725 M rows/s | **2.172x** |
| EWM feature `sum` | lanes | 4.728132 M rows/s | 5.478824 M rows/s | **1.159x** | 5.004119 M rows/s | **1.058x** |
| EWM feature `std` | lanes | 2.587637 M rows/s | 3.913019 M rows/s | **1.512x** | 3.927134 M rows/s | **1.518x** |

The runner exposes four logical CPUs as two physical cores with SMT. Worker pinning orders one logical CPU from each physical core before adding SMT siblings. On this topology, the selected order is `0, 2, 1, 3`, making the two-thread measurements use both physical cores.

The EWM feature sum peaks at two threads because its 360 MB output and relatively light per-value computation become bandwidth/SMT constrained. Four threads remain faster than serial, but only narrowly; two threads are the preferred configuration for this graph on the hosted runner.

## Existing parallel workloads after reduction integration

| Graph | 1 thread | 4 threads | Speedup |
| --- | ---: | ---: | ---: |
| Root Cat, 5M × 9 × 3 | 13.660 M rows/s | 39.781 M rows/s | 2.912x |
| Deep elementwise, N=64 | 2.588879 M rows/s | 7.253506 M rows/s | 2.802x |
| Optimized einsum, N=64 | 1.197148 M rows/s | 3.040936 M rows/s | 2.540x |
| Stateless K=3 Ridge | 2.224326 M rows/s | 6.083227 M rows/s | 2.735x |
| `roll_rets`, N=9 | 0.869686 M rows/s | 1.349784 M rows/s | 1.552x |

Light lane-local temporal graphs remain intentionally unattractive when parallelism is explicitly forced. The `threads=0` profitability policy keeps such graphs serial when the estimated work does not justify parallel execution.
