# cpp_stream reductions and parallel benchmark

The final parallel branch is based directly on `cpp-stream-backend` commit `c607a6be39d481164433c2f84c1209de7e2ae8bd`.

The reduction benchmark uses 1,000,000 rows, 9 instruments, and 3 computed features, with one warmup and seven measured runs on a four-CPU GitHub-hosted AMD EPYC runner.

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

## Existing parallel workloads after reduction integration

| Graph | 1 thread | 4 threads | Speedup |
| --- | ---: | ---: | ---: |
| Root Cat, 5M × 9 × 3 | 10.729 M rows/s | 34.626 M rows/s | 3.227x |
| Deep elementwise, N=64 | 2.588879 M rows/s | 7.253506 M rows/s | 2.802x |
| Optimized einsum, N=64 | 1.197148 M rows/s | 3.040936 M rows/s | 2.540x |
| Stateless K=3 Ridge | 2.224326 M rows/s | 6.083227 M rows/s | 2.735x |
| `roll_rets`, N=9 | 0.869686 M rows/s | 1.349784 M rows/s | 1.552x |

Light lane-local temporal graphs remain intentionally unattractive when parallelism is explicitly forced: EWM measured `0.667x` and grouped state `0.698x` at four threads. The `threads=0` profitability policy keeps those graphs serial.
