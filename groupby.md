# Groupby execution plan for `src/jax_flat/`

## Step 1 (this PR)
Freeze semantics with JAX-flat-focused contract tests by adding `tests/jax_flat/test_groupby_semantics_spec.py`.

Contract scope:

- Canonical form only: `groupby(key_tuple, lhs, op_using_self_)`.
- Arbitrary tuple-key length.
- At most one `univ(...)` in the key tuple.
- NaN keys route to a dedicated NaN group.
- Streaming updates remain incremental (no full-history recompute).
- `xfail(strict=True)` is allowed only for not-yet-implemented `jax_flat` grouped runtime behavior.

Gate:

- This test module is the implementation contract for `jax_flat` groupby work.

## Remaining roadmap
1. Ensure lowering hands canonical grouped nodes to `jax_flat`.
2. Implement minimal grouped runtime in `src/jax_flat` (scalar key first).
3. Extend to arbitrary tuple keys.
4. Add tuple-key `univ(...)` support.
5. Add multi-arity grouped ops.
6. Performance hardening and regression checks.
7. Slice into PRs with explicit scope and minimal test commands.


## Step 2 (this PR)
Ensure lowering path into `jax_flat` is canonical-only and add lowering-shape tests in `tests/jax_flat/test_groupby_lowering_spec.py`.
