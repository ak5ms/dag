# cpp_stream reductions and parallel benchmark

The final parallel branch is based directly on `cpp-stream-backend` commit `a21928f57b79573b8985edf65a8b97d950d87905`.

The reduction benchmark uses 1,000,000 rows, 9 instruments, and 3 computed features, with one warmup and seven measured runs on a four-CPU GitHub-hosted AMD EPYC runner. These reductions use the default `ignore_na=True`; selecting `ignore_na=False` changes missing-value propagation but not the planner's row, lane, or terminal scheduling rules.

| Execution | Median throughput | Median runtime | Output bytes |
| --- | ---: | ---: | ---: |
| Full computed result | 10.340193 M rows/s | 0.096710 s | 216,000,000 |
| Fused `sum(axis=0)` | 21.494070 M rows/s | 0.046524 s | 216 |
| Fused `mean(axis=0)` | 21.626119 M rows/s | — | 216 |
| Fused `std(axis=0)` | 14.824391 M rows/s | — | 216 |
| `cumsum(x).emit("last")` | 103.136654 M rows/s | — | 72 |

The fused temporal sum is **2.079x faster** than writing the full computed result. Including a NumPy post-hoc reduction of the materialized result, the fused native path is **5.307x faster** (`0.046524 s` versus `0.246888 s`). The native and post-hoc reductions produced the same checksum, `-106.013009915`.

## Parallel scheduling

- Temporal reductions and `emit("last")` have one final accumulator owner and remain single-threaded until a deterministic generic accumulator-merge layer is added.
- Row-only reductions are row-sharded when all rows are independent.
- A row reduction after lane-local temporal work may remain lane-sharded when it retains the instrument axis. Each worker accumulates and writes only its own lane interval.
- A row reduction that removes the instrument axis is cross-sectional and is not lane-sharded.

The final focused parallel suite passed 31 tests, including temporal reduction ownership, row sharding, lane-local feature reduction, final emission, Cat, groupby, `where`, and `roll_rets`.

## Dedicated reduction scaling benchmark

The permanent `scripts/benchmark_cpp_stream_parallel_reductions.py` benchmark uses 1,000,000 rows and 64 instruments. It performs one warmup and seven measured runs for 1, 2, and 4 requested threads. Thread-count order alternates forward and backward between repetitions, workers are pinned, output files are pre-sized, and asynchronous writeback is disabled.

The row-sharded cases construct eight stateless features and reduce across the instrument axis. The lane-sharded cases construct six independent EWM feature streams and reduce only the feature axis, retaining instrument-local temporal state.

Every parallel output is compared exactly with the serial output, including its NaN mask. CI requires every measured multicore count—not only the largest one—to exceed serial throughput. The minimum accepted median speedups are 1.15x for row-sharded reductions and 1.05x for lane-sharded reductions.

| Reduction graph | Planner | 1 thread | 2 threads | 2-thread speedup | 4 threads | 4-thread speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stateless feature `sum` | rows | 1.057190 M rows/s | 1.995096 M rows/s | **1.887x** | 2.148176 M rows/s | **2.032x** |
| Stateless feature `mean` | rows | 0.974521 M rows/s | 2.020273 M rows/s | **2.073x** | 2.139966 M rows/s | **2.196x** |
| Stateless feature `std` | rows | 0.749377 M rows/s | 1.564722 M rows/s | **2.088x** | 1.633382 M rows/s | **2.180x** |
| EWM feature `sum` | lanes | 0.697127 M rows/s | 1.020304 M rows/s | **1.464x** | 0.921344 M rows/s | **1.322x** |
| EWM feature `std` | lanes | 0.378995 M rows/s | 0.639478 M rows/s | **1.687x** | 0.690828 M rows/s | **1.823x** |

The runner exposes four logical CPUs as two physical cores with SMT. The initial benchmark revealed that numerical CPU order placed a two-thread run on sibling hardware threads of one physical core. Worker pinning now orders one logical CPU from each physical core before adding SMT siblings. On that topology, the selected order is `0, 2, 1, 3`, making the two-thread measurements use both physical cores.

The EWM feature sum peaks at two threads on this runner because the four-thread configuration adds SMT contention and writes a 512 MB output. It nevertheless remains 1.322x faster than serial and passes the permanent CI floor.

## Existing parallel workloads after reduction integration

| Graph | 1 thread | 4 threads | Speedup |
| --- | ---: | ---: | ---: |
| Root Cat, 5M × 9 × 3 | 10.729 M rows/s | 34.626 M rows/s | 3.227x |
| Deep elementwise, N=64 | 2.588879 M rows/s | 7.253506 M rows/s | 2.802x |
| Optimized einsum, N=64 | 1.197148 M rows/s | 3.040936 M rows/s | 2.540x |
| Stateless K=3 Ridge | 2.224326 M rows/s | 6.083227 M rows/s | 2.735x |
| `roll_rets`, N=9 | 0.869686 M rows/s | 1.349784 M rows/s | 1.552x |

Light lane-local temporal graphs remain intentionally unattractive when parallelism is explicitly forced: EWM measured `0.667x` and grouped state `0.698x` at four threads. The `threads=0` profitability policy keeps those graphs serial.
