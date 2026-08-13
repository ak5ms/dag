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
