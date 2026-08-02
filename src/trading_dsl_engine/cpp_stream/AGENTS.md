# cpp_stream agent guidance

This backend must remain independent of `jax_flat`.

## Shared IR and typing

- Shared formula semantics belong in `trading_dsl_engine.ir`; never import JAX or JAX-flat types into that package.
- Value kind, logical shape, dtype, and compile-time dimensions belong in neutral `ValueType`. Matrix, fixed-width, arbitrary tensor, and object values must not be disguised as ordinary vectors.
- Do not eagerly convert mapped inputs to float64. `InputSrc`, `SlotSrc`, and `SlotDst` carry native scalar types. Statistical/einsum operators may define float64 accumulation at their boundary.
- Never add a per-element runtime dtype or rank switch.

## Unified source layer

- There is one public compiler, `compile_formula`, and one public runner, `CppStreamRuntime.run`. Do not add `compile_<format>_formula`, `run_<format>_files`, or format-specific native entrypoints.
- Every formula input is resolved independently through `SourceAdapter`; a single formula may mix file formats, URI schemes, in-memory arrays, shared memory, and custom source objects.
- Adapter selection may use object type, URI scheme, extension, or explicit adapter name. Source selection belongs in `python/sources.py`, not the compiler, lowering, codegen, or operator nodes.
- Built-in `.npy`, raw, and ndarray adapters must remain zero-copy. `.npy` inspection preserves dtype, complete shape, payload offset, and C-order layout. Raw/headerless inputs require explicit `InputTypeSpec` metadata.
- `(rows,)` and `(rows, 1)` are row-scalar. Higher-rank row tensors preserve their complete positive C-order shape.
- All adapters return the same `PreparedSource` contract: compile-time `InputTypeSpec`, row count, stable contiguous pointer, owner, and deterministic close callback.
- The generated shared object receives only typed pointers, row counts, and row widths. It must not know whether an input came from `.npy`, raw mmap, Parquet, Arrow, a network feed, or another adapter.
- New source formats should register an adapter. A true chunked/network implementation may extend the source execution layer, but must not create another compiler or runner.
- Runtime replacement sources may use different formats from compile-time sources, but dtype and logical per-row shape must exactly match the compiled plan.

## NumPy-style einsum

- String-form subscript semantics belong in `trading_dsl_engine.ir.einsum`, not cpp_stream codegen.
- Match NumPy string behavior: case-sensitive ASCII labels, implicit/explicit output, empty scalar terms, repeated-label diagonals, arbitrary rank, and one ellipsis per term.
- Ordinary named dimensions must match exactly. Broadcasting is allowed only for ellipsis dimensions.
- NumPy's default is `optimize=False`; preserve left-to-right evaluation unless the caller requests `True`, `greedy`, or `optimal`.
- `greedy` and `optimal` choose a static pairwise path during lowering. `optimal` may exhaustively search only under a documented operand-count limit and must use a deterministic fallback afterward.
- Generated C++ must not parse subscript strings, inspect dynamic shapes, allocate a path, or search contraction order at runtime.
- Canonicalized labels become compile-time integer axis maps. Loop extents and output/reduction ordering are compile-time template arguments.
- Keep one generic `UnaryEinsumNode` and one generic `BinaryEinsumNode`; never add pattern-named nodes such as `EinsumNfNfToNNode` or formula-specific kernels.
- Preserve the contiguous identity-mapping FMA path. It must bulk-load a lazy feature row once, not recompute an RBF normalization for each feature access.
- General mapped loops must retain correct broadcasting, diagonal, permutation, scalar, and arbitrary-reduction semantics.
- Arbitrary-rank contiguous tensors use a zero-copy dense tensor source. Cat/RBF sources stay lazy. Only selected contraction intermediates use fixed tensor scratch.
- Scratch is compact by logical tensor size. Do not reserve `N * max_width` semantics for every arbitrary tensor unless its layout actually requires it.
- Native einsum currently accumulates/stores float64. Keep unsupported NumPy surfaces explicit: integer-sublist calls, precomputed path-list arguments, `out=`, `dtype=`, `order=`, `casting=`, and writeable-view behavior.
- Do not introduce Einsums, Eigen Tensor, TBLIS, or another contraction dependency unless same-host benchmarks show a material win without forcing copies/materialization of mapped, Cat, or RBF sources.

## Key metadata

- Per-key metadata belongs in neutral `GroupKeySpec`, exposed as `Key(expr, num_keys, offset, row_scalar, dtype)`.
- `num_keys=K` means exactly K consecutive categories `[offset, offset+K)`. Dense routing uses `value-offset`.
- `row_scalar=True` is a semantic assertion. Resolve such a key once per row and broadcast the slot.
- `dtype` validates the completed expression type; it never authorizes an implicit cast.
- If all keys are bounded, use generic mixed-radix dense routing. Otherwise hash the exact tuple.
- Preserve native 64-bit integer equality, floating NaN canonicalization, and `-0.0/+0.0` equivalence.

## Operator-agnostic groupby

- `groupby.hpp` owns only key resolution, grouped context creation, and inner-plan invocation. It must contain no operator implementation.
- Never create `GroupedFooNode` or `FastGroupedFooNode`.
- Every operator has one C++ node and receives `DirectExecution<N>` or `GroupedExecution<N, Capacity, PartitionCount>` through the final template parameter.
- Stateful lane operators use `Execution::state_size/state_index`; cross-sectional operators use `rank_group/cross_group`.
- Python codegen must emit the same node name inside and outside groupby; only the execution type changes.

## Cat, basis, and fixed-width values

- `cat(...)` is represented by compile-time feature width.
- Flatten Cat into `FeatureList<Sources...>` when consumed by Ridge, InstrumentBasisMean, or einsum. Read original sources directly instead of materializing `N x K`.
- Lazy RBF/future-RBF sources must support random feature reads and efficient full-row loads.
- Nested Cat and separate Ridge feature arguments must lower to the same feature list and generated source.
- A standalone Cat root is row-major `(rows,N,K)` and may use `CatNode`; do not add consumer-specific Cat implementations.

## Ridge

- Ridge is one generic `RidgeNode<..., Execution>` for direct and grouped execution. `GroupedRidgeNode` is forbidden.
- K is compile-time. Persistent grouped moment/beta state remains fixed `std::array` storage. Map local KxK/K workspaces into fixed-size Eigen matrices/vectors; do not use `Eigen::Dynamic` or Eigen Tensor in the row path.
- Compile Eigen with `EIGEN_DONT_PARALLELIZE`; cpp_stream owns any outer parallelism and must not permit nested Eigen worker teams.
- Preserve weighted pairwise-missing moments and per-moment last-update timing.
- Positive-half-life predictions use the prior beta; `hl<=0` or nonfinite is current-row/stateless.
- Regularization is `XX + lambda * diag(diag(XX))`, with nonnegative lambda.
- Unconstrained Ridge keeps the allocation-free fixed-array Cholesky, pivoted Gaussian, and Jacobi pseudoinverse chain because same-host benchmarks show a full Eigen replacement is materially slower at K=3. Fixed-size Eigen is used by the stateless NNQP path and may be used for cold numeric helpers only when benchmarks show no hot-path regression.
- Stateless nonnegative Ridge uses the fixed-size active-set NNQP implementation adapted from the repository's `jax_ffi/nnqp` solver. Stateful nonnegative Ridge keeps its exact warm-started fixed-array coordinate solver; stateless nonnegative Ridge uses fixed-size NNQP.
- `get_beta` and `get_preds` are projections of the neutral Ridge object. A raw object is not a file output.

## Existing performance invariants

- Preserve the all-finite small-width rank-count path.
- Preserve the common recursive-EWM policy and avoid weight/count traffic when semantics permit it.
- Row-scalar facts must propagate through pure stateless graphs.
- Reusable output files must not be retruncated when their size already matches.
- No operator may allocate from the heap in `on_data`; no Python per-row loop is allowed. Fixed-size Eigen objects are permitted, but dynamic Eigen matrices/vectors and hidden temporaries requiring heap storage are not.
- For source/I/O changes, inspect generated C++ for one outer row loop and use Linux `strace`/`perf stat` as supporting checks (`openat`, `mmap`, `munmap`, `read`/`pread64`, page faults). `strace` verifies syscall/mapping behavior, not every userspace load or allocation, so pair it with code inspection and allocation-aware tests.

## Code generation and runtime

- Translation-unit structure belongs in `python/templates/runner.cpp.j2`.
- Python codegen builds typed immutable template arguments/views, not complete string-concatenated row loops.
- Physical choices such as scratch liveness, direct output writes, prefetching, mmap/writeback, state layout, compiler flags, and contraction paths belong in cpp_stream lowering/codegen.
- Cache keys must include generated source, all packaged headers, compiler identity, flags, platform, and Python ABI.

## Testing and benchmarks

- Source tests must cover independent extension inference, mixed-format inputs in one formula, raw metadata validation, replacement source compatibility, and at least one registered custom adapter.
- Einsum correctness must cover implicit/explicit output, arbitrary labels, scalar operands/reductions, ellipsis broadcasting, rejection of non-ellipsis broadcasting, diagonals, transposes, rank-2/rank-4 mapped inputs, tensor scratch, and n-ary paths.
- Compare native output against NumPy for isolated einsum cases and against JAX-flat for the exact `roll_rets` graph.
- Tests must assert the old pattern-specific einsum node and format-specific native runner are absent from generated C++.
- Hot-path changes should be benchmarked at 5M x 9 with one warmup and ten measured runs when practical.
- Report every run, median/mean/best, checksums, finite fraction, estimated contraction work, and largest intermediate.
- Benchmark `optimize=False`, greedy, and optimal on the same n-ary expression. Reject an optimizer that changes the checksum or increases estimated work without a justified measured gain.
- Re-run `scripts/benchmark_cpp_stream_roll_rets.py` after source or einsum changes to detect end-to-end regressions.
- Do not hard-code one hosted CPU's throughput as a universal threshold; use environment-provided floors.
