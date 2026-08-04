# Stateless NaN and code-generation audit

The permanent audit covers the expression:

```python
xs_rank((x + 5 + y) * 3)
```

and compares cpp_stream against the Python/JAX `jax_flat` implementation for:

- cross-sectional rank after composed arithmetic;
- equality and inequality comparisons;
- ordered comparisons;
- logical `and`, `or`, and `xor`;
- `fillna` and `where`;
- default NaN-skipping row mean, including an all-NaN row.

The comparison asserts both values and the exact NaN mask. Rank values use `rtol=2e-9` and `atol=2e-9` because cpp_stream uses its fixed native inverse-normal table while JAX uses its own `ndtri` implementation.

The audit identified and corrected one semantic difference: C++ comparisons and logical predicates previously converted NaN inputs to ordinary boolean values. They now propagate NaN exactly as `jax_flat._nan_cmp` does. Arithmetic, `fillna`, `where`, rank filtering, and default NaN-skipping mean already matched.

## Generated C++ structure

The generated translation unit has one outer row loop. For the audited expression it emits four adjacent stage calls in dependency order:

```text
s0.on_data(ctx)
s1.on_data(ctx)
s2.on_data(ctx)
s3.on_data(ctx)
```

These correspond to two additions, one multiplication, and cross-sectional rank. There is no additional row pass between operators.

## Optimized assembly

The permanent script `scripts/audit_cpp_stream_stateless_codegen.py` reconstructs the exact rendered stage types, compiles an isolated row kernel with the production optimization flags, and inspects `AuditKernel::run`.

On the Ubuntu 24.04 GitHub runner with GCC and `-O3 -march=native`:

- all four stage methods were inlined;
- there were no operator or stage calls in the optimized row kernel;
- the only call target was the compiler-inserted cold `__stack_chk_fail@PLT` guard;
- each arithmetic loop was vectorized with 32-byte vectors;
- the hot instruction sequence begins with adjacent `vaddpd` operations followed directly by `vmulpd` operations before rank evaluation;
- the logical plan has three scratch slots, but the optimizer scalar-replaced/folded the arithmetic stages into registers and the small rank working set rather than issuing separate stage function calls.

The focused audit run passed 11 tests, including cross-backend NaN compatibility and generated-source adjacency checks.
