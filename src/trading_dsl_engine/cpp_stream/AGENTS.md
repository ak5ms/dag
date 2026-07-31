# cpp_stream agent guidance

This backend is designed to remain independent of `jax_flat`.

- Shared formula semantics belong in `trading_dsl_engine.ir`; do not import JAX/JAX-flat types into that package.
- Physical choices such as scratch liveness, direct root writes, grouped-state layout, dense key routing, prefetch distance, mmap/writeback, and native compiler flags belong in `cpp_stream`.
- C++ translation-unit structure belongs in `python/templates/runner.cpp.j2`. Python codegen should build typed template arguments and small immutable template views rather than concatenate complete C++ functions or row loops.
- Keep `Jinja2` as a runtime dependency and package every `.j2` file needed by installed wheels.
- Do not add Python loops to the per-row execution path.
- Do not allocate from the heap in operator `on_data`/row execution. Construction, compilation, mapping setup, and error paths may allocate.
- Keep stateful operators in their own headers. Stateless arithmetic and rank live in `ops/naryop.hpp`.
- Do not add `FastGroupedFooNode` classes. Grouped and ungrouped variants must share the same operator implementation through compile-time policies such as state indexing or rank-group identity. This keeps semantics and optimizations in one template and lets `if constexpr` remove irrelevant paths.
- Preserve the all-finite small-width rank path. Checking `finite[j]` inside every N x N comparison was a measured regression for N=9.
- Preserve the `MinPeriods<=0 && IgnoreNa && !Adjust` EWM policy specialization inside the shared EWM implementation. It is semantically equivalent to the general policy but avoids weight/count traffic for both direct and grouped state.
- Groupby uses the canonical shared form `groupby(key_tuple, lhs, rhs_using_self_)`. Tuple keys may combine one `univ(...)` component with arbitrary supported dynamic expressions. Preserve NaN-key canonicalization and +/-0 equivalence.
- Calendar aliases such as `var("minute")` are derived from `_ev_ts` by the neutral frontend. Do not require pre-materialized calendar columns when the shared DSL already defines the derivation.
- Dense bounded input keys should bypass hashing through `key_cardinalities`; preserve a dedicated NaN slot. Derived key-domain metadata should eventually provide the same optimization without user hints.
- Preserve EWM/cumsum/xs_rank semantics against the active repo reference tests, including EWM NaN carry behavior and upper-rank tie scoring.
- Reusable output files must not be truncated when their size is already correct. Every row is overwritten, and retruncation reintroduces page-allocation noise into repeated benchmarks.
- Any hot-path structural refactor should be benchmarked on the 5M x 9 workload and, where practical, compare emitted `.text` or assembly before/after. `scripts/benchmark_cpp_stream.py` supports an optional `CPP_STREAM_BENCH_MIN_MROWS` regression threshold.
