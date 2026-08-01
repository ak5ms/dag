# cpp_stream agent guidance

This backend must remain independent of `jax_flat`.

## Shared IR and typing

- Shared formula semantics belong in `trading_dsl_engine.ir`; never import JAX or JAX-flat types into that package.
- Value kind and compile-time width belong in neutral `ValueType`. Matrix, fixed-width, and object values must not be disguised as ordinary vectors.
- Do not eagerly convert mapped inputs to float64. `InputSrc`, `SlotSrc`, and `SlotDst` carry native scalar types.
- Same-typed integer arithmetic stays integer through typed scratch. Mixed types promote only because the operation result requires it.
- Integral division in explicitly integral key graphs uses mathematical floor division; integral `floor` is an identity.
- Statistical operators may define float64 semantics, but conversions must occur at the operator boundary, not during input loading.
- `.npy` inputs are mmap-only. Inspect dtype, shape, payload offset, and C-order layout before compilation. Never add a per-element runtime dtype switch.
- Supported `.npy` shapes are `(rows,)`, `(rows,1)`, and `(rows,N)`; width one is row-scalar.

## Key metadata

- Per-key metadata belongs in neutral `GroupKeySpec`, exposed as `Key(expr, num_keys, offset, row_scalar, dtype)`.
- `num_keys=K` means exactly K consecutive categories `[offset, offset+K)`. Dense routing uses `value-offset`.
- `row_scalar=True` is a semantic assertion. Resolve such a key once per row and broadcast the slot.
- `dtype` validates the completed expression type; it never authorizes an implicit cast.
- If all keys are bounded, use generic mixed-radix dense routing. Otherwise hash the exact tuple.
- Preserve native 64-bit integer equality, floating NaN canonicalization, and `-0.0/+0.0` equivalence.

## Operator-agnostic groupby

- `groupby.hpp` owns only key resolution, grouped context creation, and inner-plan invocation. It must contain no indicator, regression, rolling, or cross-sectional operator implementation.
- Never create `GroupedFooNode` or `FastGroupedFooNode`.
- Every operator has one C++ node and receives its execution scope through the final template parameter.
- Supported scopes are `DirectExecution<N>` and `GroupedExecution<N, Capacity, PartitionCount>`.
- Stateful lane operators use `Execution::state_size` and `Execution::state_index`.
- Cross-sectional operators use `Execution::rank_group` or `Execution::cross_group` and size state from `Execution::cross_state_size`.
- The exact static partition count is compile-time execution metadata. Do not use conservative N-sized cross-state when the plan knows fewer partitions.
- Python codegen must emit the same node name inside and outside groupby; only the execution type changes.
- Adding a normal codegen mapping for a new operator should make it usable in groupby without a second class or grouped branch.

## Cat and fixed-width values

- `cat(...)` is represented by compile-time feature width.
- When Cat feeds Ridge, flatten it into `FeatureList<Sources...>` and read the original sources directly. Do not materialize an intermediate `N x K` matrix.
- Nested cat and separate Ridge feature arguments must lower to the same feature list and generated source.
- A standalone Cat root is row-major `(rows,N,K)` and may use `CatNode`.
- Do not introduce a Ridge-specific Cat implementation.

## Ridge

- Ridge is one generic `RidgeNode<..., Execution>` for direct and grouped execution. `GroupedRidgeNode` is forbidden.
- K is compile-time. Moment state, beta state, local systems, and solver workspaces use fixed `std::array` storage.
- Preserve weighted pairwise-missing moments and per-moment last-update timing.
- Positive-half-life predictions use the prior beta; `hl<=0` or nonfinite is current-row/stateless.
- Regularization is `XX + lambda * diag(diag(XX))`, with nonnegative lambda.
- Keep generic solver order: Cholesky first, pivoted solve second, symmetric pseudoinverse fallback last.
- Nonnegative Ridge uses the same fixed-size generic quadratic solver; do not add K-specific or formula-specific implementations.
- The finite-panel path may skip pairwise validity bookkeeping only when every required value in the row is finite. Missing-data rows must use the complete pairwise path.
- Finite-panel synchronized state is a semantic data-validity optimization, not a separate node. Maintain exact transition behavior across finite -> missing -> finite rows.
- Precompute literal decay and regularization constants during codegen where possible.
- `get_beta` and `get_preds` are projections of the neutral Ridge object. A raw Ridge object is not a file output.
- Direct beta output is `(rows,K)`; grouped beta output is `(rows,N,K)`.
- Current physical limitations must remain explicit: literal hl/lambda, scalar/vector weights, and no arbitrary downstream materialization of non-root matrix/fixed beta values.

## Existing operator performance invariants

- Preserve the all-finite small-width rank-count path. For N=9 it materially outperforms sorting.
- Preserve the common recursive-EWM policy in the single EWM implementation; avoid weight/count traffic when semantics permit it.
- Row-scalar facts must propagate through pure stateless graphs so timestamp expressions execute once per row.
- Reusable output files must not be retruncated when their size already matches.

## Code generation and runtime

- Translation-unit structure belongs in `python/templates/runner.cpp.j2`.
- Python codegen should build typed immutable template arguments/views, not concatenate complete row loops.
- Keep Jinja2 as a runtime dependency and package required `.j2` files.
- Physical choices such as scratch liveness, direct output writes, prefetching, mmap/writeback, state layout, and compiler flags belong in cpp_stream.
- Do not add Python per-row loops.
- Do not allocate from the heap in operator `on_data`. Compilation, setup, mapping, and error paths may allocate.

## Testing and benchmarks

- Correctness tests must cover native compilation/execution, NaNs, finite-to-missing transitions, grouped execution, and output shapes.
- Tests must assert groupby source contains no operator implementations and generated code contains no `Grouped*Node` classes.
- Hot-path changes should be benchmarked at 5M x 9 with one warmup and ten measured runs when practical.
- Report all runs and checksums, not only the best result.
- Compare direct and one-group grouped controls before attributing a slowdown to groupby.
- `scripts/benchmark_cpp_stream.py` covers timestamp/groupby/rank workloads.
- `scripts/benchmark_cpp_stream_ridge.py` covers Cat, direct Ridge, and grouped Ridge.
- Do not hard-code one hosted CPU's throughput as a universal test threshold; use environment-provided regression floors.
