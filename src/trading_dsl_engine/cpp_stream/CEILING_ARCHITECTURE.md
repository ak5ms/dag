# Native ceiling architecture

## Generated versus handwritten code

Jinja should describe only formula topology and runtime orchestration:

- declare the statically typed physical stages;
- bind each input row;
- divide rows or lanes among workers;
- invoke one generic worker-state merge protocol;
- finalize once.

It should not contain the arithmetic or state-combination law for a reduction,
EWM, regression, or another operator. Those laws belong in ordinary C++ where
they are type checked, documented, unit testable, and inlined by the compiler.

The generated runner passes a tuple of stage references and a row- or
lane-partition descriptor. At compile time the C++ customization point selects:

1. an operator-provided `merge_state_from(source, partition)` member;
2. a built-in adapter for an existing mergeable node;
3. a no-op for a stage with no worker state.

C++20 `requires`, `if constexpr`, index sequences, and fold expressions are
sufficient. C++26 reflection could enumerate fields, but cannot infer whether a
field must be summed, Welford-combined, overwritten in row order, or copied by
lane. The semantic merge law must remain explicit somewhere.

## Why a generic kernel has overhead

A direct reference loop can know the exact output and semantic options. The
compositional runtime may preserve arbitrary missing-value behavior,
intermediate consumers, generic destinations, shapes, and dtypes. These are
real instructions unless compile-time analysis proves that they are unnecessary.

## Preferred optimization model

Do not add a handwritten node for every slow formula. Derive simpler physical
policies from proven graph properties so equivalent GP expressions receive the
same optimization. The most important proofs are shared validity masks, dead
intermediate outputs, direct epilogue consumption, and compact state storage.

The implementation order should be:

1. prove equivalent validity masks and remove divergence machinery;
2. eliminate intermediate outputs with no observable consumer;
3. evaluate fused epilogues directly from current state;
4. select smaller state layouts from those proofs;
5. add bounded-history row halos where they remove artificial statefulness;
6. prototype a tiled hybrid scheduler only for important remaining pipelines.

## Implemented generic fusion passes

The projection-fusion work follows the same boundary: Python lowering proves
relationships, ordinary C++ templates encode the physical capability, and Jinja
only emits topology.

### Shared reduction projections

Adjacent reductions with the same lazy source, axes, missing-value policy, and
execution scope are represented by one `ReductionProjectionBundleNode`. The
requested projection set determines the minimum sufficient statistics at compile
time:

- sum requests total and count;
- mean requests total/count unless Welford moments are already required;
- standard deviation requests count, mean, and M2;
- min/max request only their extrema and count.

Thus mean plus standard deviation shares count/mean/M2 without maintaining a
redundant total. The mechanism is projection based rather than Sharpe based.

### Lazy producer-to-reduction fusion

A sole-consumer Cat or no-contraction einsum is represented as a typed lazy
tensor source. A dead current-row reduction can likewise become a lazy
`RowReductionTensorSource` feeding a temporal accumulator. Static einsum offset
maps and reduction inverse maps are computed before native compilation, so the
hot loop contains direct indexed reads rather than tensor/PnL scratch writes.

The pass is fail-closed: only consumers whose native source contract supports
arbitrary lazy tensors are eligible. Other stages retain the existing pointer
ABI.

### EWM observation proofs and compact fallback state

Sibling EWM expressions of the form `where(common_mask, value, NaN)` factor the
common predicate and evaluate it once. Per-component finite checks remain, so an
expression that overflows while a sibling remains finite still triggers exact
metadata divergence. If structural analysis proves observation equivalence, the
per-component fallback metadata is an empty `[[no_unique_address]]` member and
is compiled out completely.

### Dead projections and direct epilogues

Bundle members with no observable consumer receive compile-time discard
destinations. Scalar suffix algebra reads current bundle component values through
a typed epilogue context, avoiding scratch stores and reloads. The same mechanism
is used by heterogeneous reductions and EWM bundles.

See `PROJECTION_FUSION_BENCHMARK.md` for alternating-order before/after results
on direct ceilings, candidate batches, stateful formulas, and deep typed-GP
programs.
