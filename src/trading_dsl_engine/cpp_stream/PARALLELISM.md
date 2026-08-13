# cpp_stream parallel execution

`cpp_stream` selects a safe partitioning strategy from the lowered physical graph. Automatic execution is the runtime default:

```python
runtime = compile_formula(formula, sources, n_instruments=9)
automatic = runtime.run(out_path="automatic.bin")
serial = runtime.run(out_path="serial.bin", threads=1)
parallel = runtime.run(out_path="parallel.bin", threads=4, pin_threads=True)
```

- omitted `threads` or `threads=0`: select a useful count from the proven strategy, observed rows, instruments, fused-expression work, and CPU affinity;
- `threads=1`: force serial execution;
- `threads>1`: request that degree of parallelism, capped by CPU affinity and safe work partitioning.

Every worker owns an independent plan, scratch space, group resolver, and mutable operator state. Eigen remains internally single-threaded so only the outer `cpp_stream` scheduler owns worker parallelism.

## Strategies

### Row sharding

Independent rows are split into contiguous ranges. Workers write disjoint output rows directly to the final mmap. This covers elementwise graphs, stateless einsum, stateless Ridge, root Cat, and row-only reductions.

### Lane sharding

Each worker owns a fixed contiguous instrument interval and advances those lanes through the complete time series. This preserves temporal order for EWM, cumulative state, history, lane-local groupby, `InstrumentBasisMean`, and `roll_rets`.

A row reduction after temporal work can remain lane-sharded only when it retains the instrument axis. The reduction node constrains both its input reads and output writes to `lane_begin:lane_end`, so workers never overwrite another worker's result.

### Terminal reductions

Final-output plans may also use row or lane workers when the planner proves that their private results have a valid combination rule. The combination is implemented in ordinary compile-time C++, followed by one final write.

### Serial fallback

A graph remains serial when temporal state is followed by a cross-sectional operation, when a row reduction removes the instrument axis after temporal work, or when dependency analysis cannot prove lane independence.

## Cat execution

Cat does not create a nested task pool. A root Cat is row-sharded at the whole-plan level. When Cat feeds Ridge, `InstrumentBasisMean`, einsum, or a reduction, lowering normally flattens it into a compile-time `FeatureList`, so no intermediate Cat tensor or additional file pass is created.

## Correctness invariants

- Every worker owns separate mutable state.
- Row workers process disjoint row ranges.
- Lane workers process disjoint lane ranges in original time order.
- Lane-aware reductions read and write only their owned lanes.
- Final output is written once after worker results are combined.
- Cross-sectional temporal graphs are not lane-sharded.
- Benchmarks validate checksums, NaN placement, finite output fractions, actual thread counts, and output byte counts.

See `CEILING_ARCHITECTURE.md` for the generated-code boundary and the plan for closing remaining reference-ceiling gaps.
