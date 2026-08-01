# cpp_stream agent guidance

This backend is designed to remain independent of `jax_flat`.

- Shared formula semantics belong in `trading_dsl_engine.ir`; do not import JAX/JAX-flat types into that package.
- Per-key metadata belongs in neutral `GroupKeySpec`, exposed through `Key(expr=..., num_keys=..., offset=..., row_scalar=..., dtype=...)`. Do not replace it with backend-only global maps.
- `num_keys=K` means exactly `K` consecutive non-NaN integer categories. Valid values are `[offset, offset + K)`. Dense routing maps value `v` to digit `v - offset`. For example, `num_keys=12, offset=1` describes months 1..12. `offset` has no effect without `num_keys`.
- `row_scalar=True` asserts that one key value applies to all lanes in the row. Resolve such a tuple once and broadcast its slot. `row_scalar=None` means infer it from input shape and expression dependencies. Never silently treat a lane-varying key as row-scalar.
- `dtype` is the expected native type of the completed key expression. It is a validation assertion, not authorization to cast an input. A direct mapped input must exactly match it. An explicitly integral derived key graph must have matching integral input leaves and exactly representable integral constants.
- If every dynamic key has `num_keys`, use generic mixed-radix dense resolution. Capacity is `product(num_keys + 1)` because each floating key reserves one NaN digit. Reject capacities above the current uint16 slot limit.
- If any key is unbounded, hash the complete tuple exactly. Preserve floating NaN canonicalization and `-0.0/+0.0` equivalence. Preserve all bits of native integer keys; never route `int64` or `uint64` equality through `double`.
- Physical choices such as scratch liveness, direct root writes, grouped-state layout, dense key routing, prefetch distance, mmap/writeback, and native compiler flags belong in `cpp_stream`.
- C++ translation-unit structure belongs in `python/templates/runner.cpp.j2`. Python codegen should build typed template arguments and small immutable template views rather than concatenate complete C++ functions or row loops.
- Keep `Jinja2` as a runtime dependency and package every `.j2` file needed by installed wheels.
- `.npy` inputs must be mapped without copying. Inspect dtype, shape, payload offset, and C-order layout before native compilation. Keep typed input loads compile-time-specialized; do not add a per-element runtime dtype switch.
- Supported `.npy` shapes are `(rows,)`, `(rows, 1)`, and `(rows, N)`. Width one is row-scalar. Supported input dtypes are float32/64, int32/64, and uint32/64. Reject object, structured, big-endian, Fortran-order, and higher-rank arrays explicitly.
- Do not eagerly convert mapped inputs to float64. `InputSrc<Index, ValueType, RowWidth>`, `SlotSrc<Index, ValueType, RowScalar>`, and `SlotDst<Index, ValueType>` carry native scalar types. Stateless operators read through `read_native` and are templated on their result type.
- Same-typed integer arithmetic must remain integer through typed scratch. Mixed types may promote only because the operation's declared result type requires it. Do not hide conversions in `RowContext`.
- Integral division used inside an explicitly integral key graph follows mathematical floor-division semantics; integral `floor` is an identity. This lets the generic expanded `minute(_ev_ts)` graph remain `int64` without a calendar-specific node.
- Stateful/statistical operators currently define float64 semantics and may use `read()`: cumsum, EWM, rank, and grouped lhs/captures remain double-valued. The current root output file is also float64. Keep these conversion boundaries explicit.
- Propagate row-scalar information through pure stateless operations and instantiate producer stages at width one. Do not compute a row-scalar calendar expression once per instrument.
- Do not add Python loops to the per-row execution path.
- Do not allocate from the heap in operator `on_data`/row execution. Construction, compilation, mapping setup, and error paths may allocate.
- Keep stateful operators in their own headers. Stateless arithmetic and rank live in `ops/naryop.hpp`.
- `groupby.hpp` must remain operator agnostic. It owns key resolution, grouped context construction, and inner-plan invocation only. It must not define cumsum, EWM, rank, rolling, regression, or any other operator implementation.
- Never create `GroupedFooNode` or `FastGroupedFooNode`. Every node has one implementation and receives the plan execution scope through its final template parameter. The supported scopes are `DirectExecution<N>` and `GroupedExecution<N, Capacity>`.
- Python codegen must not branch by operator to select grouped types. `_stage_type` emits the same C++ node name inside and outside groupby; only the execution-scope argument changes.
- Stateful nodes obtain storage size and lane state addresses from `Execution::state_size` and `Execution::state_index(ctx, lane)`. Cross-sectional nodes obtain group identity from `Execution::rank_group(ctx, lane)`. Stateless nodes accept and ignore the same execution parameter.
- A newly added operator must have one C++ node implementation. Once its normal codegen mapping exists, it must work inside groupby without a second node class or grouped codegen branch.
- Preserve the all-finite small-width rank path. Checking `finite[j]` inside every N x N comparison was a measured regression for N=9.
- Preserve the `MinPeriods<=0 && IgnoreNa && !Adjust` EWM policy specialization inside the single EWM implementation. It is semantically equivalent to the general policy but avoids weight/count traffic for both direct and grouped state.
- Groupby uses the canonical shared form `groupby(key_tuple, lhs, rhs_using_self_)`. Tuple keys may combine one `univ(...)` component with arbitrary supported dynamic expressions.
- Calendar aliases such as `var("minute")` are derived from `_ev_ts` by the neutral frontend. Do not require pre-materialized calendar columns when the shared DSL already defines the derivation.
- The legacy `key_cardinalities` map may remain for compatibility with direct inputs, but new optimization metadata should use `Key` descriptors or inferred neutral-IR types/domains.
- Preserve EWM/cumsum/xs_rank semantics against the active repo reference tests, including EWM NaN carry behavior and upper-rank tie scoring.
- Reusable output files must not be truncated when their size is already correct. Every row is overwritten, and retruncation reintroduces page-allocation noise into repeated benchmarks.
- Any hot-path structural refactor should be benchmarked on the 5M x 9 workload and, where practical, compare emitted `.text` or assembly before/after. `scripts/benchmark_cpp_stream.py` supports an optional `CPP_STREAM_BENCH_MIN_MROWS` regression threshold.
