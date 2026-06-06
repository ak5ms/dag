# Temporary plan: cpp/jax_flat integration hardening

- [x] 1. Add `cpp: bool = True` to `jax_flat.compile_formula` and flow it to `JaxFlatRuntime`.
- [x] 2. Add automatic fallback to JAX-flat for unsupported C++ formulas with a warning that identifies the unsupported part.
- [x] 3. Add/run reconciliation tests for C++ operators vs JAX-flat, including NaN/groupby behavior.
- [x] 4. Move `scripts/benchmark_groupby_matrix.py` into `tests/jax_flat` and mark it as a perf test when `n_rows > 200`.
- [~] 6. Implement remaining C++ operators requested here, especially `einsum` and `outer`, using a numerical library where practical. (`outer` and common row-aligned `einsum` patterns implemented; full arbitrary einsum/codegen remains future work.)
- [x] 7. Make state slots compile-time shaped by actually needed operators.
- [~] 8. Prefer fixed-size arrays/state buffers over dynamically growing vectors in hot state. (State slots are sized at `init_state`; container type remains `std::vector` for Python-compatible ownership.)
- [~] 9. Update C++ build to request C++26. (GCC 13 rejects `-std=c++26`; build uses `-std=c++2b`/C++23 as the available draft mode.)
- [ ] 10. Use BLAS/LAPACK/Eigen-style numerical kernels for solver-heavy numerical code where practical. (Not completed; no project BLAS/LAPACK binding was added.)
- [~] final. Check for no absolute runtime regression in JAX-flat on this branch vs main; fix if needed. (No local `main` branch or remote is available; targeted pure-JAX checks were run instead.)

Notes and command log will be appended while working.


## Notes
- C++26 request: GCC 13 rejects `-std=c++26`; build uses `cxx_std=23` with `-std=c++2b` as the available draft-standard mode in this container.
- Solver library request: no project-level BLAS/LAPACK binding was added in this pass; ridge still preserves existing sufficient-statistic behavior and solver semantics.
- Main-branch regression check: this checkout has no `main` branch or remote; compare is limited to targeted pure-JAX smoke/perf commands in this workspace.

## Command log
- `. .venv/bin/activate && python setup.py build_ext --inplace --force`
- `. .venv/bin/activate && pytest -q tests/jax_flat/test_cpp_flat_runtime.py tests/jax_flat/test_benchmark_groupby_matrix.py::test_groupby_matrix_perf_smoke tests/jax_flat/test_einsum.py -n 0`
- `. .venv/bin/activate && python tests/jax_flat/test_benchmark_groupby_matrix.py --rows 128 --cols 9 --runs 1 --warmups 1 --assert --key-mode all_same --univ 1 --lhs-kind stateless_lhs --rhs-kind stateful_rhs --rhs-nested 0 --root groupby_root`
- `. .venv/bin/activate && python - <<'PY' ... pure_jax_smoke ... PY`
