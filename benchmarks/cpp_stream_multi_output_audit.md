# cpp_stream multi-output audit

This audit covers `compile_formula([f0, f1, ...])` and the unified 1..K-root
cpp_stream compiler/runtime path.

## Required invariants

- All roots share one neutral IR builder and CSE table.
- Lowering emits one physical plan, one C++ translation unit, one shared object,
  and one `cpp_stream_run_arrays` invocation.
- Each public output retains its exact logical shape. Packing occurs only at the
  native output boundary, and `RunResult.load()` returns ordered shaped views.
- Output offsets, sizes, row/final modes, and lane-partitionability are C++
  template arguments (`OutputSpec`) rather than runtime Python metadata.
- Scalar scratch is compacted independently by native dtype. Matrix/tensor
  scratch uses exact per-slot extents rather than a global maximum extent.
- A public subgraph that remains hot downstream keeps the same scratch-local
  producer/consumer path as the top-only graph; it is persisted without
  recomputation.
- Compatible EWM output projections remain eligible for EWM-bundle epilogue
  fusion, including the one-member case.
- Identical remaining Copy/TensorCopy roots are evaluated once and fanned out to
  distinct compile-time destinations. The output regions do not alias.
- Any plan with final-only state retains one final accumulator owner. `emit(last)`
  snapshots ordinary row sources and yields NaNs for an empty input.

## Representative hosted benchmark

Single thread, 5,000,000 rows × 9 instruments, one warmup plus ten measured
runs. The shared subgraph is `ewm(x + 1, span=32)` and the top-level formula is
`xs_rank(subgraph)`.

| Case | Median |
|---|---:|
| top only | about 0.412 s |
| `[subgraph, top]` | about 0.494 s |
| equivalent `cat(subgraph, top)` | about 0.546 s |

The formula-list form is approximately 9.7% faster than equal-byte Cat. Its
roughly 20% difference from top-only is the required additional 360 MB output
write, not another EWM computation. A heterogeneous `(N,) + (N, 8)` list is
within approximately 0.5% of equal-byte Cat, while compatible EWM-bundle list
and Cat plans are within approximately 0.3% and both lower to one
`ewm_bundle` stage.

The reproducible benchmark, including duplicate-input and duplicate-expensive-
expression fan-out cases, is
`scripts/benchmark_cpp_stream_multi_output_subgraph.py`.

## Validation

The PR workflow runs the focused IR/source/native/operator/roll-rets tests,
optimized stateless assembly audit, typed and unhinted monotonic-key comparison,
Cat/Ridge smoke benchmark, generic einsum benchmark, streaming-reduction
benchmark, roll-rets benchmark, and the full multi-output benchmark.
