# cpp_stream agent guidance

This backend is designed to remain independent of `jax_flat`.

- Shared formula semantics belong in `trading_dsl_engine.ir`; do not import JAX/JAX-flat types into that package.
- Physical choices such as scratch liveness, direct root writes, grouped-state layout, dense key routing, prefetch distance, mmap/writeback, and native compiler flags belong in `cpp_stream`.
- Do not add Python loops to the per-row execution path.
- Do not allocate from the heap in operator `on_data`/row execution. Construction, compilation, mapping setup, and error paths may allocate.
- Keep stateful operators in their own headers. Stateless arithmetic and `xs_rank` live in `ops/naryop.hpp`; the generic variadic Nary-node experiment previously changed code generation, so do not reintroduce it without machine-code and throughput checks.
- Groupby uses the canonical shared form `groupby(key_tuple, lhs, rhs_using_self_)`. Tuple keys may combine one `univ(...)` component with arbitrary supported dynamic expressions. Preserve NaN-key canonicalization and +/-0 equivalence.
- Dense bounded input keys should bypass hashing through `key_cardinalities`; preserve a dedicated NaN slot.
- Preserve EWM/cumsum/xs_rank semantics against the active repo reference tests, including EWM NaN carry behavior and upper-rank tie scoring.
- Any hot-path structural refactor should be benchmarked on the 5M x 9 workload and, where practical, compare emitted `.text` or assembly before/after.
