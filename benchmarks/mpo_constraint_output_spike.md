# MPO constraint-output timing spike

- Rows: 5,000
- Assets: 3
- Horizons: 8
- Timed runs per mode: 10 after one warmup
- `run_plus_load` includes native execution, output-file production, and eager `result.load(mmap_mode=None)`.

```text
CLARABEL {"include": "/home/runner/.cache/trading_dsl_engine/clarabel/v0.11.1-noalloc1-native/native/include", "library": "/home/runner/.cache/trading_dsl_engine/clarabel/v0.11.1-noalloc1-native/native/lib/libclarabel_c.a"}
RESULT {"checksum": 703454.5543362226, "compile_s": 3.3590901230000156, "mean_load_s": 0.00036966420000510424, "mean_native_s": 1.9540072146, "mean_run_plus_load_s": 1.955258775499999, "mean_run_wall_s": 1.954889111299994, "median_load_s": 0.0003591280000136976, "median_native_s": 1.952268337, "median_run_plus_load_s": 1.9535061274999919, "median_run_wall_s": 1.9531370399999872, "mode": "none", "output_bytes": 240128, "rows": 5000, "runs": 10, "setting": "default", "warmup_native_s": 1.971730865}
RESULT {"checksum": 739953.9188996291, "compile_s": 3.3691558240000177, "mean_load_s": 0.000701024099987535, "mean_native_s": 2.5413517936, "mean_run_plus_load_s": 2.543023731899993, "mean_run_wall_s": 2.5423227078000052, "median_load_s": 0.0005920584999898892, "median_native_s": 2.5400415935, "median_run_plus_load_s": 2.541570943499991, "median_run_wall_s": 2.5409965105000083, "mode": "augmented", "output_bytes": 1520128, "rows": 5000, "runs": 10, "setting": "default", "warmup_native_s": 2.658037733}

--- failure tail ---
Cloning into '/home/runner/.cache/trading_dsl_engine/clarabel/v0.11.1-noalloc1-native/Clarabel.cpp'...
Note: switching to '0de6259a3edfd5cc041ec42b2148599ce63e73cb'.

You are in 'detached HEAD' state. You can look around, make experimental
changes and commit them, and you can discard any commits you make in this
state without impacting any branches by switching back to a branch.

If you want to create a new branch to retain commits you create, you may
do so (now or later) by using -c with the switch command. Example:

  git switch -c <new-branch-name>

Or undo this operation with:

  git switch -

Turn off this advice by setting config variable advice.detachedHead to false

HEAD is now at 0de6259 cmake options cleanup (#69)
Cloning into '/home/runner/.cache/trading_dsl_engine/clarabel/v0.11.1-noalloc1-native/Clarabel.rs'...
Note: switching to '25540f559592068d0c8a80e46ded1b21760212a1'.

You are in 'detached HEAD' state. You can look around, make experimental
changes and commit them, and you can discard any commits you make in this
state without impacting any branches by switching back to a branch.

If you want to create a new branch to retain commits you create, you may
do so (now or later) by using -c with the switch command. Example:

  git switch -c <new-branch-name>

Or undo this operation with:

  git switch -

Turn off this advice by setting config variable advice.detachedHead to false

    Updating crates.io index
     Locking 49 packages to latest compatible versions
 Downloading crates ...
  Downloaded enum_dispatch v0.3.13
  Downloaded num-traits v0.2.19
  Downloaded derive_builder v0.11.2
  Downloaded serde_core v1.0.229
  Downloaded amd v0.2.2
  Downloaded itertools v0.11.0
  Downloaded paste v1.0.15
  Downloaded zmij v1.0.23
  Downloaded serde_derive v1.0.229
  Downloaded syn v1.0.109
  Downloaded darling_core v0.14.4
  Downloaded itoa v1.0.18
  Downloaded syn v2.0.119
  Downloaded serde_json v1.0.151
  Downloaded fnv v1.0.7
  Downloaded proc-macro2 v1.0.107
  Downloaded lazy_static v1.5.0
  Downloaded ident_case v1.0.1
  Downloaded thiserror-impl v1.0.69
  Downloaded unicode-ident v1.0.24
  Downloaded derive_builder_core v0.11.2
  Downloaded serde v1.0.229
  Downloaded serde-big-array v0.5.1
  Downloaded autocfg v1.5.1
  Downloaded darling_macro v0.14.4
  Downloaded once_cell v1.21.4
  Downloaded strsim v0.10.0
  Downloaded cfg-if v1.0.4
  Downloaded derive_builder_macro v0.11.2
  Downloaded quote v1.0.47
  Downloaded memchr v2.8.3
  Downloaded syn v3.0.4
  Downloaded darling v0.14.4
  Downloaded thiserror v1.0.69
  Downloaded either v1.18.0
   Compiling proc-macro2 v1.0.107
   Compiling unicode-ident v1.0.24
   Compiling quote v1.0.47
   Compiling syn v1.0.109
   Compiling fnv v1.0.7
   Compiling ident_case v1.0.1
   Compiling strsim v0.10.0
   Compiling serde_core v1.0.229
   Compiling autocfg v1.5.1
   Compiling zmij v1.0.23
   Compiling serde v1.0.229
   Compiling serde_json v1.0.151
   Compiling thiserror v1.0.69
   Compiling num-traits v0.2.19
   Compiling syn v2.0.119
   Compiling syn v3.0.4
   Compiling clarabel v0.11.1 (/home/runner/.cache/trading_dsl_engine/clarabel/v0.11.1-noalloc1-native/Clarabel.cpp/Clarabel.rs)
   Compiling paste v1.0.15
   Compiling once_cell v1.21.4
   Compiling either v1.18.0
   Compiling memchr v2.8.3
   Compiling darling_core v0.14.4
   Compiling serde_derive v1.0.229
   Compiling thiserror-impl v1.0.69
   Compiling itoa v1.0.18
   Compiling darling_macro v0.14.4
   Compiling darling v0.14.4
   Compiling derive_builder_core v0.11.2
   Compiling enum_dispatch v0.3.13
   Compiling derive_builder_macro v0.11.2
   Compiling itertools v0.11.0
   Compiling derive_builder v0.11.2
   Compiling amd v0.2.2
   Compiling serde-big-array v0.5.1
   Compiling cfg-if v1.0.4
   Compiling lazy_static v1.5.0
warning: unused import: `super::super::traits::*`
   --> /home/runner/.cache/trading_dsl_engine/clarabel/v0.11.1-noalloc1-native/Clarabel.cpp/Clarabel.rs/src/solver/core/solver.rs:470:9
    |
470 |     use super::super::traits::*;
    |         ^^^^^^^^^^^^^^^^^^^^^^^
    |
    = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default

warning: hiding a lifetime that's elided elsewhere is confusing
  --> /home/runner/.cache/trading_dsl_engine/clarabel/v0.11.1-noalloc1-native/Clarabel.cpp/Clarabel.rs/src/solver/core/kktsolvers/direct/quasidef/datamaps.rs:26:39
   |
26 |     pub(crate) fn to_sparse_expansion(&self) -> Option<SparseExpansionCone<T>> {
   |                                       ^^^^^            ^^^^^^^^^^^^^^^^^^^^^^ the same lifetime is hidden here
   |                                       |
   |                                       the lifetime is elided here
   |
   = help: the same lifetime is referred to in inconsistent ways, making the signature confusing
   = note: `#[warn(mismatched_lifetime_syntaxes)]` on by default
help: use `'_` for type paths
   |
26 |     pub(crate) fn to_sparse_expansion(&self) -> Option<SparseExpansionCone<'_, T>> {
   |                                                                            +++

warning: `clarabel` (lib) generated 2 warnings (run `cargo fix --lib -p clarabel` to apply 2 suggestions)
   Compiling clarabel_c v0.1.0 (/home/runner/.cache/trading_dsl_engine/clarabel/v0.11.1-noalloc1-native/Clarabel.cpp/rust_wrapper)
    Finished `release` profile [optimized] target(s) in 28.87s
CLARABEL {"include": "/home/runner/.cache/trading_dsl_engine/clarabel/v0.11.1-noalloc1-native/native/include", "library": "/home/runner/.cache/trading_dsl_engine/clarabel/v0.11.1-noalloc1-native/native/lib/libclarabel_c.a"}
/home/runner/work/dag/dag/src/trading_dsl_engine/cpp_stream/python/compiler_support.py:341: RuntimeWarning: ICX (icpx) is not available; cpp_stream is using g++ instead. Run trading_dsl_engine.cpp_stream.install_icx() to install ICX under ~/intel/oneapi without sudo.
  compiler = _compiler()
RESULT {"checksum": 703454.5543362226, "compile_s": 3.3590901230000156, "mean_load_s": 0.00036966420000510424, "mean_native_s": 1.9540072146, "mean_run_plus_load_s": 1.955258775499999, "mean_run_wall_s": 1.954889111299994, "median_load_s": 0.0003591280000136976, "median_native_s": 1.952268337, "median_run_plus_load_s": 1.9535061274999919, "median_run_wall_s": 1.9531370399999872, "mode": "none", "output_bytes": 240128, "rows": 5000, "runs": 10, "setting": "default", "warmup_native_s": 1.971730865}
/home/runner/work/dag/dag/src/trading_dsl_engine/cpp_stream/python/compiler_support.py:341: RuntimeWarning: ICX (icpx) is not available; cpp_stream is using g++ instead. Run trading_dsl_engine.cpp_stream.install_icx() to install ICX under ~/intel/oneapi without sudo.
  compiler = _compiler()
RESULT {"checksum": 739953.9188996291, "compile_s": 3.3691558240000177, "mean_load_s": 0.000701024099987535, "mean_native_s": 2.5413517936, "mean_run_plus_load_s": 2.543023731899993, "mean_run_wall_s": 2.5423227078000052, "median_load_s": 0.0005920584999898892, "median_native_s": 2.5400415935, "median_run_plus_load_s": 2.541570943499991, "median_run_wall_s": 2.5409965105000083, "mode": "augmented", "output_bytes": 1520128, "rows": 5000, "runs": 10, "setting": "default", "warmup_native_s": 2.658037733}
Traceback (most recent call last):
  File "/home/runner/work/dag/dag/scripts/benchmark_mpo_constraint_output_paths.py", line 469, in <module>
    _child()
  File "/home/runner/work/dag/dag/scripts/benchmark_mpo_constraint_output_paths.py", line 253, in _child
    runtime = compile_formula(expressions, data, n_instruments=N_ASSETS)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/dag/dag/src/trading_dsl_engine/cpp_stream/python/__init__.py", line 72, in compile_formula
    return _compile_formula(
           ^^^^^^^^^^^^^^^^^
  File "/home/runner/work/dag/dag/src/trading_dsl_engine/cpp_stream/python/compile.py", line 164, in compile_formula
    program = compile_ir(
              ^^^^^^^^^^^
  File "/home/runner/work/dag/dag/src/trading_dsl_engine/cpp_stream/python/frontend.py", line 122, in compile_ir
    roots = tuple(builder.build(expression) for expression in expressions)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/dag/dag/src/trading_dsl_engine/cpp_stream/python/frontend.py", line 122, in <genexpr>
    roots = tuple(builder.build(expression) for expression in expressions)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/dag/dag/src/trading_dsl_engine/ir/frontend.py", line 751, in build
    result = self._build_uncached(node)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/dag/dag/src/trading_dsl_engine/ir/frontend.py", line 894, in _build_uncached
    return self._build_call(node)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/dag/dag/src/trading_dsl_engine/ir/frontend.py", line 939, in _build_call
    widths = tuple(
             ^^^^^^
  File "/home/runner/work/dag/dag/src/trading_dsl_engine/ir/frontend.py", line 940, in <genexpr>
    _feature_width(self.nodes[index].value_type) for index in children
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/dag/dag/src/trading_dsl_engine/ir/frontend.py", line 310, in _feature_width
    raise FormulaIRCompileError(f"{value_type.kind!r} cannot be used as a feature")
trading_dsl_engine.ir.frontend.FormulaIRCompileError: 'fixed' cannot be used as a feature
Traceback (most recent call last):
  File "/home/runner/work/dag/dag/scripts/benchmark_mpo_constraint_output_paths.py", line 471, in <module>
    _parent()
  File "/home/runner/work/dag/dag/scripts/benchmark_mpo_constraint_output_paths.py", line 458, in _parent
    _run_child(setting, mode)
  File "/home/runner/work/dag/dag/scripts/benchmark_mpo_constraint_output_paths.py", line 372, in _run_child
    raise RuntimeError(
RuntimeError: constraint-output child failed: setting=default, mode=post, returncode=1
```

Exit status: 1
