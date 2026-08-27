# cpp_stream parallel execution

`cpp_stream` selects a safe partitioning strategy from the lowered physical graph, while the caller controls whether parallel execution is requested:

```python
runtime = compile_formula(formula, sources, n_instruments=9)
serial = runtime.run(out_path="serial.bin")
parallel = runtime.run(out_path="parallel.bin", threads=4, pin_threads=True)
automatic = runtime.run(out_path="automatic.bin", threads=0, pin_threads=True)
```

- omitted `threads` or `threads=1`: serial execution;
- `threads>1`: request that degree of parallelism, capped by CPU affinity and safe work partitioning;
- `threads=0`: opt into the retained profitability heuristic.

Every worker owns an independent plan, scratch space, group resolver, and mutable operator state. Eigen remains internally single-threaded so only the outer `cpp_stream` scheduler owns worker parallelism.

## Strategies

### Row sharding

Independent rows are split into contiguous ranges. Workers write disjoint output rows directly to the final mmap. This covers elementwise graphs, stateless einsum, stateless Ridge, root Cat, and row-only reductions.

### Lane sharding

Each worker owns a fixed contiguous instrument interval and advances those lanes through the complete time series. This preserves temporal order for EWM, cumulative state, history, lane-local groupby, `InstrumentBasisMean`, and `roll_rets`.

A row reduction after temporal work can remain lane-sharded only when it retains the instrument axis. The reduction node constrains both its input reads and output writes to `lane_begin:lane_end`, so workers never overwrite another worker's result.

### Terminal reductions

A temporal reduction, meaning its axes include logical axis `0`, emits one fixed-size final result. `emit("last")` has the same final-output behavior. These plans currently use one accumulator owner even when more threads are requested. This preserves deterministic streaming semantics and avoids a formula-specific merge implementation. A future generic merge layer can parallelize these without changing the expression API.

### Serial fallback

A graph remains serial when temporal state is followed by a cross-sectional operation, when a row reduction removes the instrument axis after temporal work, or when dependency analysis cannot prove lane independence.

## Cat execution

Cat does not create a nested task pool. A root Cat is row-sharded at the whole-plan level. When Cat feeds Ridge, `InstrumentBasisMean`, einsum, or a reduction, lowering normally flattens it into a compile-time `FeatureList`, so no intermediate Cat tensor or additional file pass is created.

## Correctness invariants

- Every worker owns separate mutable state.
- Row workers process disjoint row ranges.
- Lane workers process disjoint lane ranges in original time order.
- Lane-aware reductions read and write only their owned lanes.
- Terminal reductions and final emission have one owner.
- Cross-sectional temporal graphs are not lane-sharded.
- Benchmarks validate checksums, NaN placement, finite output fractions, actual thread counts, and output byte counts.

See `REDUCTIONS_PARALLEL_BENCHMARK.md` for the reduction, Cat, Ridge, einsum, and `roll_rets` measurements from the final validation run.

<!-- native-gp-region-scheduler -->
## Independent runtime regions

`run_many(runtimes, ...)` is the native scheduling boundary for independent compiled DAG regions. It prepares all source pointers and outputs once, then submits tasks to one C++ worker pool. The pool uses an atomic task index for load balancing, optionally pins workers, and invokes each generated runner through its stable C ABI. Python performs orchestration and result construction only; it does not own worker threads during execution.

This path is particularly important for GP fitness graphs whose temporal reductions make each individual runtime sequential. The search groups related candidates into bounded multi-output programs for CSE, estimates formula cost, balances those programs across native tasks, and runs each task with one inner thread. This provides population-level parallelism without oversubscribing the machine.

`run_many` is component scheduling, not a claim that arbitrary temporal/cross-sectional boundaries inside one runtime are cache-tiled. Whole-plan row/lane scheduling remains the intra-runtime mechanism; independent-root scheduling covers the GP workload where graph-level dependencies otherwise force a serial final accumulator.

