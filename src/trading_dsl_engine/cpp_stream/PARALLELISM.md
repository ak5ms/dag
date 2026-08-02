# cpp_stream parallel execution

`cpp_stream` selects the safe partitioning strategy from the lowered physical graph,
but the caller controls whether parallel execution is used. The parallel branch
inherits the backend's fixed-size Eigen/NNQP Ridge implementation; Eigen remains
single-threaded so only the outer cpp_stream scheduler owns worker parallelism. The
backend's final same-host comparison preserved existing Ridge and `roll_rets`
throughput within 0.7% while improving stateless nonnegative Ridge by 65.1%.

```python
runtime = compile_formula(formula, sources, n_instruments=9)

# Serial by default.
serial = runtime.run(out_path="serial.bin")

# Explicit degree of parallelism.
parallel = runtime.run(
    out_path="parallel.bin",
    threads=4,
    pin_threads=True,
)

# Experimental automatic profitability policy.
automatic = runtime.run(
    out_path="automatic.bin",
    threads=0,
    pin_threads=True,
)
```

The thread policy is:

- omitted `threads` or `threads=1`: serial execution;
- `threads>1`: request that degree of parallelism;
- `threads=0`: explicitly opt into the retained automatic profitability heuristic.

Requested threads are capped by CPU affinity and by the available row or lane
parallelism. A positive thread count never bypasses dependency analysis: graphs that
cannot be partitioned safely remain serial.

`RunResult` reports wall time, process CPU time, actual thread count, available CPUs,
selected partitioning mode, and average busy cores.

## Why profitability is not inferred from operation implementation

The compiler can infer whether an operation is safe to partition from its lowered
inputs, outputs, shapes, and state dependencies. It cannot reliably infer whether
parallel execution will be faster from arbitrary C++ source alone.

A source-level or instruction-count estimate does not capture the main determinants:

- generated vectorization and unrolling;
- cache and memory-bandwidth pressure;
- input-source latency and page faults;
- repeated full-time-axis reads under lane sharding;
- state size and scratch traffic;
- CPU model, affinity, and competing system load.

A reliable automatic policy would require measured calibration, compiler profile
information, or runtime sampling rather than a manually assigned per-operation cost.
Until such a model exists, the current static work score is retained only behind the
explicit `threads=0` opt-in. It has no effect on default execution or on an explicit
positive thread count.

## Strategy selection

### Row sharding

Rows are divided into contiguous ranges and each worker owns a complete independent
plan instance. Workers write disjoint output rows directly to the final mmap.

This applies to elementwise graphs, stateless einsum, current-row cross-sectional
operators, stateless Ridge, and a root `cat(...)`.

### Lane sharding

Each worker owns a fixed contiguous instrument-lane interval and advances those lanes
through the complete time series. This preserves temporal order for EWM, cumulative
state, history, lane-local groupby, InstrumentBasisMean, and the complete
`flows.riskmodel.roll_rets` graph.

Workers own private scratch, group resolvers, and temporal state. They write disjoint
lane slices directly to the shared row-major mmap; no full-size per-worker output copy
or merge pass exists.

Lane sharding rereads the complete time axis in every worker and can lose the
single-thread loop's vectorization. This is why explicit thread control is the normal
policy. In the optional `threads=0` mode, the static graph-work threshold keeps light
temporal graphs such as a standalone EWM or small grouped cumulative graph serial,
while deeper graphs such as `roll_rets` can select lane multicore execution.

### Serial fallback

A graph remains serial when temporal state is followed by an operation requiring
other lanes from the same row, or when dependency analysis cannot prove lane
independence. Examples include `xs_rank(ewm(x, ...))`, stateful cross-sectional Ridge,
and einsum contractions that reduce the instrument label.

## Cat execution model

Cat does not create a nested task pool and does not schedule one task per child or
feature. For a root Cat graph, the whole plan is row-sharded when multicore execution
is requested: each worker executes the same fixed-width `CatNode` over a disjoint row
range. The local lane and feature loops remain compiler-unrollable and vectorizable.

When Cat feeds Ridge, InstrumentBasisMean, or einsum, lowering normally flattens it
into a compile-time `FeatureList`. In those plans there is no materialized Cat stage;
the selected row or lane strategy parallelizes the consumer while it reads the
original sources directly. If a Cat stage is materialized inside a lane-parallel plan,
it honors the worker's lane interval but still creates no nested tasks.

## Affinity and measurements

Linux CPU availability comes from `sched_getaffinity`. Optional pinning uses
`pthread_setaffinity_np`.

Native timing includes thread creation. Process CPU time is reported separately;
`cpu_seconds / wall_seconds` is the measured average number of busy cores.

## Correctness invariants

- Every worker owns a separate generated plan and all mutable operator state.
- Row workers process disjoint row ranges.
- Lane workers process disjoint lane ranges in original time order.
- Row-scalar nodes are recomputed locally by every lane worker.
- Cross-sectional temporal graphs are not lane-sharded.
- One-thread and multi-thread results must match exactly for temporal/grouped graphs.
- Omitting `threads` must execute with one thread even when the graph is parallelizable.
- Benchmarks validate checksums and finite output fractions at every thread count.

## Root Cat benchmark

A root `cat(x1, x2, x3)` with shape `(5_000_000, 9, 3)` was benchmarked on a
GitHub-hosted AMD EPYC 9V74 runner with four affinity CPUs, pinned workers, one warmup,
and ten measured executions.

| Mode | Threads | Median | Best | Busy cores | Minimum payload bandwidth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial | 1 | 10.722 M rows/s | 10.770 M rows/s | 1.000 | 4.632 GB/s |
| Automatic opt-in | 4 | 34.025 M rows/s | 34.424 M rows/s | 3.950 | 14.699 GB/s |

The measured median speedup was **3.173x**. Serial and multicore sampled outputs were
bitwise identical with checksum `-100.493612836`.

```bash
python scripts/benchmark_cpp_stream_cat_parallel.py
```

## Broad benchmark matrix

Pinned 1/2/4-thread runs on a four-affinity-CPU AMD EPYC runner produced:

| Graph | 1 thread | 4 threads | Speedup | Busy cores |
| --- | ---: | ---: | ---: | ---: |
| Deep elementwise, N=64 | 2.594 | 7.472 M rows/s | 2.88x | 3.94 |
| Optimized einsum, N=64 | 1.263 | 3.185 M rows/s | 2.52x | 3.98 |
| Stateless K=3 Ridge | 2.247 | 6.108 M rows/s | 2.72x | 3.94 |
| Stateful EWM, explicitly forced | 4.257 | 2.983 M rows/s | 0.70x | 2.97 |
| Grouped state, explicitly forced | 3.134 | 2.214 M rows/s | 0.71x | 3.07 |
| `roll_rets`, N=9 | 0.868 | 1.343 M rows/s | 1.55x | 3.63 |

The EWM and grouped-state regressions demonstrate why semantic parallelizability is
not equivalent to profitability and why automatic scoring is disabled by default.

```bash
python scripts/benchmark_cpp_stream_parallel_matrix.py
```

The suite covers deep elementwise graphs, optimized n-ary einsum, stateless Ridge,
stateful EWM, grouped state, a required serial fallback, and the exact `roll_rets`
graph. It reports each run, speedup, efficiency, CPU time, busy cores, checksum, and
finite output fraction.
