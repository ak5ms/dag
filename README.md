# trading-dsl-engine

A high-performance Python DSL engine for streaming trading features on aligned minutely array data.

Current development targets the `trading_dsl_engine.jax_flat` runtime plus shared DSL/parser functionality. The older Numba and non-flat JAX implementations remain in the tree for compatibility, but they are deprecated: unless a task explicitly says otherwise, make code changes only in `jax_flat` and shared DSL modules, and run only the focused `tests/jax_flat/` and shared DSL tests needed for the behavior being changed.

## Core goals

- **Streaming-first stateful computation**: each op follows `on_data(...)` + `emit(...)`.
- **No interpreter hot loop**: `jax_flat` live tick and batch timestep paths execute through JAX JIT/`lax.scan`.
- **Composable formulas**: parse string expressions and support Python-level DSL macro composition.
- **Extensible operators**: registry/plugin model for adding new ops without central branching.
- **Scalable IO**: supports in-memory NumPy arrays and disk-backed memmaps.

## Project layout

- `src/trading_dsl_engine/base/`
  - Shared parser (`parse_formula`), Python DSL constructors, registry metadata, and compile/lower pipeline.
- `src/trading_dsl_engine/numba/`
  - Deprecated Numba built-in op implementations, jitclass state machines, and batch/live runtime helpers.
- `src/trading_dsl_engine/jax/`
  - Deprecated non-flat JAX + Equinox runtime.
- `src/trading_dsl_engine/jax_flat/`
  - Active JAX-flat runtime that lowers supported DSL expressions to a flat operator DAG and executes live ticks/batch scans through JIT-compiled JAX functions.
- `src/trading_dsl_engine/cpp_stream/`
  - Formula-specialized C++20 streaming backend with typed source adapters, generated native row loops, and fixed-size streaming reductions.
- `tests/numba/`
  - Deprecated Numba runtime tests; do not run or update unless explicitly requested.
- `tests/jax/`
  - Deprecated non-flat JAX backend tests.
- `tests/jax_flat/`
  - Active JAX-flat behavior, shape, state, and performance tests.

## Typical usage

```python
from trading_dsl_engine import compile_formula, build_engine, run_batch_from_mapping

artifact = compile_formula("xs_rank(ewm(div(close, open), 21))")
engine = build_engine("xs_rank(ewm(div(close, open), 21))")

# live tick update
out = engine.update_from_mapping({"open": open_t, "close": close_t})

# batch run (disk-backed output by default)
out2d = run_batch_from_mapping(engine, {"open": open_2d, "close": close_2d}, chunk_size=4096)  # memmap shape (time, n_instruments) for vector outputs
# opt into RAM materialization instead
out2d_ram = run_batch_from_mapping(engine, {"open": open_2d, "close": close_2d}, out_path=None)
```

Batch execution writes output to a NumPy memmap at `/tmp/trading_dsl_engine_out.memmap` by default to avoid materializing full results in RAM. Pass `out_path=None` to allocate in memory, or provide `out=` to write into a preallocated array.

Object-typed intermediate nodes are supported (e.g., stateful jitclass/structref emitters) as long as a downstream op projects them back to scalar/vector/matrix. Root object outputs are intentionally rejected in batch mode to keep the timestep loop on the compiled JIT path.

## DSL composition

You can define reusable macro-like composed functions with an explicit registry namespace:

```python
from trading_dsl_engine import DSLFunctionRegistry, register_dsl_function, add, div

my_registry = DSLFunctionRegistry()

@register_dsl_function("hlc3", registry=my_registry)
def hlc3(high, low, close):
    return div(add(add(high, low), close), 3.0)
```

Then compile with `compile_formula(..., dsl_registry=my_registry)`. Built-in composed DSL functions include `diff(x, nlag=1, max_size=1)`, which expands to `sub(x, shift(x, nlag, max_size))`.

Normal Python composition can use `var("close")`/`var("open")` identifiers and either prefix helpers or infix operator overloads: `xs_rank(ewm(var("close") / var("open"), 21))` is equivalent to `xs_rank(ewm(div(close, open), 21))`. Formula strings also support infix arithmetic/logical forms such as `close + open`, `close * 2`, `close % 5`, `close | open`, `close & open`, `close ^ open`, `close == open`, and `close != open`.

The returned artifact includes `stats` (`expanded_nodes`, `cache_hits`, `compile_seconds`) so compile-time CSE behavior and compile latency can be validated.


## Formula alpha search

`trading_dsl_engine.base.alpha_search` provides a DEAP-backed, Python-level formula alpha search scaffold. It uses standard DEAP typed GP (`gp.PrimitiveSetTyped`, `gp.PrimitiveTree`, toolbox registration, and `algorithms.eaSimple`) to evolve composed DSL `Expr` candidates from grouped terminals such as vector features, `PositiveScalar` halflives, and `PositiveIntScalar` shift lags, evaluates candidates through an injected objective callable, and admits candidates to the pool only through a configurable additive predicate. The search is staged by depth: round `i` only evaluates formulas whose expression depth is at most `i`, making it straightforward to grow complexity gradually.

The module includes compositional fitness helpers for the initial Sharpe-style objective (`alpha / ewm_var(roll_rets, HL)` with shifted, tradability-masked PnL) and a pool-aware ridge objective helper that combines the existing alpha pool with a candidate via `Ridge(...)`/`get_beta(...)`. `trading_dsl_engine.base.terminals` provides field terminal helpers: `futures_field_metadata()` expands the common futures field schema (types/ranges for prices, quantities, calendars, tradability flags, spreads, and cross-trade fields), and `feature_names_with_tags(...)` selects feature terminals by tags. Candidate filters are ordinary callables; `dimensionless_filter(...)` uses compile-time metadata so searches can restrict alphas by units or other static metadata without changing runtime hot paths.

## Data contract

- Inputs are aligned 2D arrays with shape `(time, n_instruments)`.
- Optional `column_names` passed to `compile_formula(...)`/`build_engine(...)` maps universe ticker names to input column positions for static column grouping.
- Live `update` expects 1D vectors with shape `(n_instruments,)`.
- Some ops may emit matrix outputs (e.g., `outer`, `bspline`), with shape `(n_instruments, width)` where `width` can differ from `n_instruments`.

The independent `trading_dsl_engine.cpp_stream` backend interprets reduction axes
against `(time, *row_shape)`. Its `sum`, `mean`, and `std` methods default to all
logical axes when `axis` is omitted, matching NumPy. Temporal reductions update
fixed-size state per row and project the result, plus any dependent algebraic
suffix, only once during finalization.

## NaN semantics (current)

- Binary ops propagate NaN values.
- `div` returns NaN on divide-by-zero.
- `shift(x, nlag, max_size)` stores a static ring capacity from numeric literal `max_size` while reading `x` and scalar `nlag` as sources; `shift(x, literal_nlag)` remains supported by using the literal lag as capacity.
- `ewm` skips updates for NaN inputs and can recover from NaN state.
- `xs_rank` ranks only valid values and emits NaN where input is NaN.
- `bspline(x, n_basis)` emits a per-instrument periodic basis matrix on `[0, 1]` with output width `n_basis` (inputs are clipped to `[0, 1]` and NaNs propagate).
- `InstrumentBasisMean(features, y, weights, hl)` keeps independent per-instrument basis means and emits an `(n_instruments, n_features)` beta matrix for high-throughput profile modeling without per-row matrix solves.
- `rbf_basis(ev_ts, session_start, session_end, n_basis)` derives a non-periodic normalized radial basis from epoch timestamps and explicit session bounds. `future_rbf_basis_sum(ev_ts, session_start, session_end, n_basis, n_steps)` emits suffix sums over the remaining discrete session grid without materializing a `(n_instruments, n_basis, n_steps)` horizon cube.
- `col(matrix, index)` extracts one matrix column as a vector for explicit feature selection/probing.
- `mod(a, b)` provides elementwise modulo for scalar/vector/matrix combinations supported by binary broadcasting rules.
- `groupby` now has one canonical form: `groupby((key1, key2, ..., maybe_univ, ...), lhs, op_using_self_)`.
- The tuple key supports arbitrary length. It may contain at most one `univ(...)` element plus any number of dynamic key expressions.
- The local op expression must consume the outer stream value via `self_`, e.g. `groupby((day, bucket), get_preds(Ridge(...)), cumsum(self_))`.
- Elementwise grouped RHS graphs such as nested arithmetic, `cumsum`, `ewm`, and `ffill` use a memberwise JAX-flat update path that keeps per-key/per-column state independent without invoking the heavier groupwise mask path reserved for cross-sectional RHS graphs.
- Python-composed formulas use the same canonical lowering via `lhs.groupby((...)).apply(op_expr_using_self_)`.
- Legacy groupby forms are intentionally removed (no `groupby(key, op)` and no alternate universe-only syntax path).
- Key NaNs are valid in groupby and are routed into a dedicated NaN-key group instead of raising.

## Ridge regression op (cross-sectional)

- `Ridge(x1, x2, ..., xk, y, weights, hl, lambda)` emits an object state and performs cross-sectional EWMA ridge updates each tick.
- `Ridge(x, y, hl, lambda)` omits weights and defaults them to 1.0 per instrument; Python-composed `Ridge(..., y=target, hl=21, lambda_=0.1)` supports the same default for multiple feature args without ambiguity.
- Ridge feature args can be vectors and/or matrices; matrix features are expanded by columns internally, so `Ridge(bspline(...), y, w, hl, lambda)` works without manual `col(...)` calls.
- Explicit `weights` can be a scalar, a vector `(n, 1)`/`(n,)` of sample weights, or a dense matrix `(n, n)` for `X'WX`/`X'Wy` weighting; scalar weights are broadcast per instrument.
- State uses pairwise-NaN-aware exponentially weighted sufficient statistics (`X'WX`, `X'Wy`) with per-statistic clocks, then solves `beta = (G + lambda * diag(G))^-1 h` fresh each tick.
- Pairwise clocks only advance when the corresponding `xx[j, k]` or `xy[j]` statistic receives at least one finite observation; missing rows or full outages leave only the affected statistics unchanged.
- A visual checklist for NaN behavior is available in `docs/ewm_regression_nan_handling.svg`.
- `get_preds(Ridge(...))` returns one-step-lagged predictions per instrument (`beta(t-1)·x(t)`).
- `get_beta(Ridge(...))` returns the current coefficient vector with shape `(k, 1)`.

## JAX backend

JAX and Equinox are required project dependencies. The backend mirrors the core runtime helpers under `trading_dsl_engine.jax`:

```python
from trading_dsl_engine.jax import build_jax_engine, run_batch_from_mapping

engine = build_jax_engine("xs_rank(ewm(div(close, open), 21))")
out = run_batch_from_mapping(engine, {"open": open_2d, "close": close_2d}, out_path=None)
```

The JAX backend accepts the same string formulas and Python-composed `Expr` trees as the Numba backend, including infix math/comparison operators and grouped-expression sugar such as `lhs.groupby((...)).apply(...)`. It stores formula structure in per-operator Equinox modules and wraps both the single-tick state transition and the batch `lax.scan` path with `eqx.filter_jit`, keeping the per-timestep hot paths compiled. It covers the scalar/vector/matrix stateless operators, `ewm`, `cumsum`, `shift`, `rolling_quantile`, `xs_rank`, `outer`, `bspline`, `rbf_basis`, `future_rbf_basis_sum`, `col`, `mean`, canonical tuple-key groupby `groupby((...), lhs, op_using_self_)` (including optional `univ(...)` inside the key tuple), and model projections via `get_beta(Ridge(...))`/`get_preds(Ridge(...))` plus `InstrumentBasisMean(...)` for low-cost basis-conditioned volume/profile estimates.

The active `trading_dsl_engine.jax_flat` backend additionally supports `buffer(shift(x, lag, max_size), min_lag, max_lag)` (or keyword form `buffer(shift(x, lag=..., max_lag=...), min=..., max=...)`). This JAX-flat-only operator emits a lag cube whose last axis is lags `1..max_lag`, preserving shift ring ordering and masking lags below dynamic `min_lag`, above dynamic `lag`, or beyond available history. Vector inputs produce `(time, n_instruments, max_lag)` outputs; matrix/ndarray inputs such as `bspline(...)` preserve their feature axes and append the lag axis, e.g. `(time, n_instruments, n_basis, max_lag)`.

`jax_flat.stateless(fn, ...)` wraps user-supplied stateless JAX callables as compositional variadic operators that still run inside the compiled tick and batch paths. If `output_kind`/`output_width` are omitted, the operator inherits shape metadata from its first child, which is suitable for shape-preserving transforms such as reversing the lag axis:

For JIT diagnostics, a compiled JAX-flat runtime exposes `inspect_jaxpr(state, *rows)` and `inspect_compiled_hlo(state, *rows)`, with `get_jaxpr`/`get_compiled_hlo` aliases. The arguments follow `tick`'s state-and-row ABI; the latter helper returns the compiled executable's optimized HLO text. `jit_compile_count` tracks observed tick-transition JAX traces (a per-runtime proxy for JIT compilation cache misses) across live ticks, compiled-HLO inspection, and batch scans, and `reset_jit_compile_count()` clears it. JAXPR inspection itself does not change the counter because it does not compile.

```python
import jax.numpy as jnp
from trading_dsl_engine.base.dsl import buffer, shift, var
from trading_dsl_engine.jax_flat import compile_formula, stateless

rev = stateless(lambda x: jnp.flip(x, axis=-1), name="rev")
close = var("close")
upper_lag = var("upper_lag")
min_lag = var("min_lag")
runtime = compile_formula(
    rev(buffer(shift(close, lag=upper_lag, max_lag=4), min=min_lag, max=4))
)
```


## Formula units, semantic types, and ranges

`trading_dsl_engine.jax_flat.compile_formula(...)` accepts optional static field metadata so users can inspect the physical/trading units and value range implied by a formula without changing the compiled streaming hot path. The helper API is intentionally lightweight: describe only new input fields, then let arithmetic propagate the metadata through the expression graph.

```python
from trading_dsl_engine.jax_flat import compile_formula, field, metadata

schema = metadata(
    {
        "close": field(units={"dollar": 1}, range="real", types=("price",)),
        "volume": field(units={"shares": 1}, range="nonnegative", types=("volume",)),
    },
    type_relations=(("price", "currency"),),
)

runtime = compile_formula("close * volume ** 3", metadata=schema, cpp=False)
runtime.get_units().as_dict()  # {"dollar": 1.0, "shares": 3.0}
runtime.get_range().as_tuple()  # (-inf, inf) for real price times nonnegative volume**3
runtime.get_type_relations().closure(("price",))  # {"price", "currency"}
runtime.get_node_metadata("div")  # intermediate node metadata for tracing/debugging
```

Unit metadata is represented as a sparse exponent vector over domain labels such as `dollar` or `shares`; compatible addition/subtraction preserves units when possible, incompatible combinations continue compiling with `UnitInfo.is_unknown()` set, and multiplication/division/power compose exponents. Algebraic reductions run before generic interval arithmetic by comparing normalized expression keys and literal identities, so forms such as `close / close`, `close - close`, `close * 1`, `close ** 0`, same-branch `where(...)`, and same-operand comparisons reduce to exact units/ranges instead of falling back to broad intervals. For operators without a hand-written range rule, the analyzer can automatically instantiate the JAX-flat `NaryOp` and trace ranges through the internal interval-inclusion adapter; this infers useful ranges and boolean output types for operators such as `ceil`, `floor`, `round`, `fraction`, `sign`, `arctan`, and comparisons. The semantic type graph is a small transitive Boolean relation matrix, so users can express Venn-style implications such as “every price is a currency-denominated value, but not every currency value is a price.” Range metadata uses interval arithmetic rules for common scalar/vector operators and exposes `ValueRange.to_interval()` for adapter-level interop. Unit metadata similarly exposes `UnitInfo.to_unxt_quantity()` for downstream `unxt` interop using the required `unxt` dependency when unit labels are registered/recognized by that environment, so callers can hand inferred formula units to unxt-aware tooling instead of relying only on the sparse exponent dictionary.

## Native C++ tick prototype

`trading_dsl_engine.jax_flat.compile_formula(..., cpp=True)` enables the optional native accelerator by default for supported grouped hot paths, while `cpp=False` forces the pure JAX-flat path. `trading_dsl_engine.jax_flat.engine_cpp.compile_formula(...)` lazily imports `trading_dsl_engine.jax_flat.engine_cpp` and exposes an experimental native tick-path runtime for flat formulas where C++ can currently preserve the same streaming semantics as JAX-flat. It now first lowers the shared `StreamingProgram` to a versioned typed `NativeExecutionPlan` containing resolved opcodes, shapes, widths, dtype policy, broadcast modes, state slots, purity/statefulness, grouping data, and liveness intervals. The plan applies dead-node removal, safe literal folding, stateless CSE, and cache-alias removal before recomputing liveness. During migration, the plan retains the flattened tuple table as a reference-evaluator adapter so optimized and unoptimized transitions remain exactly comparable. `runtime.inspect_native_plan()` exposes serialization-friendly plan and optimization diagnostics outside the hot path. `init_state(n_instruments)` preallocates per-node scratch buffers and operator-specific native state, while `tick_into(state, out, *rows)` reuses both the state and caller-owned output row. Validated native tick and batch compute release the Python GIL; force-cast tick input owners remain alive for the complete native call. The native wrapper also mirrors the JAX-flat batch API with `run_batch(...)` and `run_batch_into(...)`; `tick(...)` remains a convenience method that allocates only its returned row.

The native batch helper intentionally stays a repeated tick loop over the same `eval_row` transition rather than a separate vectorized semantic path. It binds contiguous row pointers once, calls the same non-batch evaluator for each row, and uses `__restrict`/flat contiguous buffers so C++ compilers can optimize the row loop without changing streaming state behavior. Native opcode metadata centrally declares output-row policy, preparation strategy, state family, rank scratch, and direct-root eligibility; runtime construction and state layout consume those traits generically instead of maintaining separate operator-specific allowlists. The benchmark script compiles and warms both C++ and JAX-flat runtimes before timing, so printed results exclude extension import, formula compilation, and JAX first-use compilation. When the optional extension is installed, `JaxFlatRuntime.run_batch(...)` may use the same native flat evaluator for fully supported dynamic-key groupby programs (including `univ(...)` column partitions and NumPy memmap-backed batch inputs) with no caller-supplied state. If the whole formula is not native-lowerable but contains coarse supported stateful/grouped subgraphs, batch execution can materialize multiple native islands, compute one or more JAX-only frontier values, and then run a final supported native root when the downstream graph becomes C++-lowerable again; this handles shapes such as `cpp(jax_only(cpp(...)), cpp(...))` while keeping user `jax_flat.stateless(...)` callables on the compiled JAX path. Set `TRADING_DSL_ENGINE_DISABLE_CPP_ACCEL=1` to force the pure JAX path for behavior checks. Native groupby lowering uses a nested RHS node table rather than a per-operator grouped-cumsum branch, so additional scalar/vector-width-1 RHS operators can compose without adding a new grouped hot-path case. If a grouped formula is not yet native-supported, `run_batch` emits a one-time `RuntimeWarning` identifying the unsupported node and automatically falls back to the JAX-flat implementation.

For a Python-oriented walkthrough of `Runtime`, `State`, `OpMetadata`, prepared data, shape layout, direct output writes, memory ownership, and the editable-extension rebuild process, see [Native C++ runtime guide](docs/native_cpp_runtime_guide.md).

Supported native operators now include inputs, literals, arithmetic/comparison/logical operators, `where`, `fillna`, `abs`, `ln`, `ceil`, `floor`, `round`, `exp`, `sign`, `arctan`, `isnan`, `purify`, `fraction`, `xstd`, `xs_rank`, `xs_sort`, `mean`, `outer`, an Eigen-backed `einsum` subset, `cat`, `bspline`, `col`, `cumsum`, literal-span `ewm`, static-limit `ffill`, `shift`, vector/`cat`-feature `Ridge` with scalar/vector weights solved via Eigen, `get_beta`, `get_preds`, the session-volume helpers used by `flows.pov.RollRets`, and dynamic-key `groupby(...)` with optional `univ(...)` column partitions for nested scalar/vector-width-1 RHS graphs over `self_` (for example `cumsum(self_)`, `cumsum(cumsum(self_))`, or `add(cumsum(self_), 1)`). A trusted `jax_flat.stateless(...)` wrapper can opt into a registered native kernel with `cpp_name=...`; arbitrary Python/JAX lambdas still cannot run in C++ and remain JAX residual nodes. The native batch path precomputes invariant future-basis tables, reuses pointer storage, and prefetches upcoming input rows rather than allocating these objects per tick. Explicit `trading_dsl_engine.jax_flat.engine_cpp.compile_formula(...)` still raises `NotImplementedError` for unsupported nodes, while automatic `compile_formula(..., cpp=True)` acceleration either runs a full native batch, runs supported native subgraphs before a JAX residual, or warns once and falls back to the default JAX-flat runtime.

Native lowering is inspectable without executing the formula. `runtime.get_lowering_plan()` returns a structured plan whose nodes identify their C++ or JAX island and the reason for any JAX-only node. `runtime.explain()` renders a Polars-style text plan; `runtime.explain("json")` produces machine-readable JSON and `runtime.explain("dot")` produces Graphviz DOT for graph visualization. When `cpp=True`, compilation warns once with the complete set of DSL functions in the formula that still lack native lowering, rather than waiting for batch execution to discover them. Capability detection probes the canonical `_cpp_node_specs`/nested spec lowerers themselves, so adding support there automatically removes the corresponding warning without maintaining a separate operator allowlist.

The extension build defaults to aggressive native optimization for local performance-sensitive installs: `-O3`, `-DNDEBUG`, `-DEIGEN_NO_DEBUG`, link-time optimization, `-march=native`, `-mtune=native`, `-fvisibility=hidden`, `-fno-math-errno`, and loop unrolling on Unix-like compilers. It intentionally does not enable `-ffast-math`, because the DSL has explicit NaN, infinity, and divide-by-zero semantics that must match JAX-flat behavior. In editable source checkouts, importing either native module hashes its complete repository-local quoted-include dependency closure plus `setup.py`, `pyproject.toml`, compiler-related environment settings, Python ABI, and platform. A changed or missing fingerprint automatically triggers one serialized forced build of both `.so` files and atomically refreshes both stamps; installed wheels without build sources remain immutable. Set `TRADING_DSL_ENGINE_CPP_NATIVE=0` before installation to omit CPU-specific `-march/-mtune` flags for redistributable wheels, set `TRADING_DSL_ENGINE_CPP_LTO=0` if the compiler/linker toolchain cannot use LTO, or append custom flags with `TRADING_DSL_ENGINE_CPP_EXTRA_FLAGS` and `TRADING_DSL_ENGINE_CPP_EXTRA_LINK_FLAGS`. To force a clean rebuild manually, remove `build/`, the in-place extensions, and their `_*.build.json` stamps, then reinstall with `python -m pip install -e . --no-build-isolation --force-reinstall --no-cache-dir` or run `python setup.py build_ext --inplace --force -v`.

Use the warmed comparison helper for quick local measurements:

```bash
python scripts/benchmark_cpp_flat.py --rows 100000 --cols 9 --runs 5
python tests/jax_flat/test_benchmark_groupby_matrix.py --rows 100000 --cols 9 --runs 1 --warmups 1 --assert
python tests/trading_dsl_engine/jax_flat/benchmark_cpp_flat_plan.py --rows 4096 --instruments 150
perf stat -e cycles,instructions,cache-misses,branch-misses -- python tests/trading_dsl_engine/jax_flat/benchmark_cpp_flat_plan.py --case elementwise
perf record -g -- python tests/trading_dsl_engine/jax_flat/benchmark_cpp_flat_plan.py --case groupby
```

The native-plan benchmark is an opt-in measurement tool rather than a correctness test. It reports cold construction, steady tick and batch throughput, peak RSS, output bytes per tick, Python-visible allocation count, graph size, and frontier-transfer time. Use instrument presets 150, 1,000, and 4,000 for normal, large, and stress measurements. Linux `perf` and allocator sampling are optional profiling aids and are never required by the test suite; the reported Python allocation count is not a claim about native heap allocations until the planned arena allocator adds native counters.

The benchmark accepts `--runs` (default 5), reports raw samples plus medians, and recreates state between samples. The latest controlled before/after results and profiling status are recorded in [`docs/native_performance_report.md`](docs/native_performance_report.md).

Native group lookup uses a preallocated open-addressed table per universe rather than scanning group capacity. Composite floating keys are hashed canonically: all NaN payloads share the dedicated NaN group and signed zeros compare/hash as the same key. The benchmark includes `groupby_locality` and `groupby_churn` cases to keep both ends of the lookup workload visible.

Fully native formulas remain native when `out_path` is set: validated batch execution writes scalar, vector, or statically-sized matrix roots directly into the disk-backed output. The `alpha_sharpes` benchmark reproduces the 29-feature alpha-PnL graph and reports source/optimized node counts, CSE totals, output bytes, cold construction, and repeated disk-backed throughput for `--backend cpp` or `--backend jax`.

Native `xs_rank` stores reusable `(value, instrument)` scratch in `State`, sorts each cross section once, and linearly scans equal-value runs to scatter the existing upper-rank score. This removes the former binary search for every instrument without changing NaN/nonfinite masking or tie behavior. The next whole-graph redesign target is feature-family lifting: sibling pipelines with identical topology and different static parameters (such as the 29 EWM spans in `alpha_sharpes`) should lower to lane-packed matrix kernels and structure-of-arrays state, reducing row dispatch from hundreds of scalar-vector nodes to a small number of prebound kernel descriptors while retaining the canonical sequential row transition.

Hybrid candidate selection estimates native work, materialized frontier bytes, conversion/copy requirements, and launch count, and rejects islands whose estimated transfer/launch cost exceeds their work. `inspect_hybrid_partition(program, rows, instruments)` returns the decision inputs and result without affecting runtime state or entering the hot path.

## Development quickstart

```bash
python -m venv .venv
. .venv/bin/activate
sudo apt-get install -y libeigen3-dev  # required to build the optional C++ jax_flat extension
PIP_CACHE_DIR=.pip-cache python -m pip install -e .
pytest -q  # configured to use pytest-xdist with 12 workers
```

The `.venv/` and `.pip-cache/` paths are gitignored so cloud/agent environments can reuse a repo-local virtualenv and wheel/download cache between iterations without committing environment artifacts.

By default, contributors and agents should run only targeted tests for active `jax_flat` and shared DSL functionality. Do not run the full suite, deprecated Numba tests, or deprecated non-flat JAX tests unless the task explicitly asks for them. Active performance tests (opt-in) live under `tests/jax_flat/`:

```bash
RUN_PERF_TESTS=1 pytest -n 0 tests/jax_flat/test_performance.py -q
```

## Notes for future work

- Add graph-level IR + CSE for shared subtrees across multi-feature workflows.
- Expand shape system for richer multi-output model/optimizer nodes.
- Continue reducing memory movement in batch paths for large memmap workloads.

### Experimental `cpp_new` formula-specialized tier

`trading_dsl_engine.cpp_new` consumes the same typed, topologically ordered
JAX-flat `StreamingProgram`; it does not add a parser.  Lowering produces an
immutable and hashable IR, applies the documented semantics-safe pass order,
plans 64-byte-aligned persistent and lifetime-colored scratch arenas, and emits
a straight-line formula transition through a typed C++ syntax tree (includes,
declarations, functions, blocks, and statements), rather than assembling the
translation unit by appending source fragments. Batch execution is defined as repeated
calls to that exact ordered row transition. Runtime-sized arrays live in owned
arenas—not the C++ call stack—so independently initialized stream states share
immutable formula data but never mutable state.

The compile-time descriptor registry currently recognizes `ewm`, `xs_rank`,
`cat`, `Ridge`, and the eliminable `get_beta` projection. A root `cat` of EWM
siblings over the same input is lifted into a fused native parameter-lane
kernel: it updates independent lane state in one native batch transition and
writes the instrument-by-lane root directly. Unsupported formulas retain
the generic flat-native tier as their cold-start/fallback implementation. Modes
are `generic-only`, `eagerly-specialized`, and `cached-specialized`. The initial
release publishes generated source into a locked, content-addressed, atomically
renamed cache; native execution remains on the equivalence-proven generic core
while the generated-module loader is completed.

Use `runtime.inspect_ir()`, `inspect_layout()`, and
`inspect_generated_source()` to inspect mappings, optimization counts, arena
layout, scratch lifetimes, scheduling traits, and source without running the
formula, tracing JAX, or loading a generated module. Formula specialization
removes interpretation and enables future fusion/lane lifting, but adds cold
compiler latency, cache space, and instruction-cache pressure; benchmarks must
therefore report cold compilation and cached loading separately from execution.

The opt-in `benchmark_cpp_new.py` benchmark validates outputs before timing and
reports the selected execution tier: ordinary formulas remain
`generic-flat-native-bridge`, while lifted EWM `cat` formulas report
`fused-ewm-lane-native`. On the 2026-07-29 development container (4,096 rows, 150
instruments, five samples), the EWM-chain baseline measured 548,713 rows/s
versus 570,547 rows/s through the bridge; `xs_rank` measured 101,199 rows/s
versus 107,650 rows/s. These differences include ordinary run/order variance,
**not** a specialization speedup. Cold source materialization was 1.91 ms/1.53
ms and cached materialization 0.91 ms/0.60 ms for EWM/rank respectively;
generated sources were 1,217 and 974 bytes.

For the lifted formula `cat(*[ewm(close, span_i) ...])` on 4,096 rows and 150
instruments, five-sample serial medians were:

| lanes | existing flat C++ | fused cpp_new | speedup |
| ---: | ---: | ---: | ---: |
| 4 | 256,406 rows/s | 544,964 rows/s | 2.13x |
| 16 | 57,735 rows/s | 86,023 rows/s | 1.49x |
| 32 | 26,800 rows/s | 39,924 rows/s | 1.49x |

The gain comes from eliminating per-node opcode traversal and intermediate
vector materialization for the sibling EWM family. Each lane retains separate
value, weight, count, and initialized arrays; the native batch loop releases
the GIL and allocates only at state/output boundaries, not per timestep.

#### Lane ablation and generalization

A CPU-pinned serial 16-lane ablation (4,096 × 150, seven samples) separated the
sources of the gap. Existing flat C++ improved from 47,822 to 53,415 rows/s
when given a reusable direct output; cpp_new improved from 79,024 to 107,370
rows/s. Within cpp_new, lane-major state traversal achieved 79,844 rows/s,
instrument-major traversal 111,259 rows/s, and a lane-major transition followed
by a separate transpose/materialization 85,710 rows/s. A store-only kernel
reached 474,574 rows/s (8.49 GiB/s). Both runtimes were single-threaded.

These results attribute the gap to four effects, in descending importance for
this workload: generic per-node evaluation/state-container overhead; output
allocation and first-touch costs; instrument-contiguous loop/output layout; and
the avoided intermediate `cat` materialization. Multithreading did not cause
the measured difference. The store-only ceiling is over 4.5 times the optimized
EWM rate, so raw output bandwidth is not yet the primary limit.

Lane discovery is descriptor-driven rather than EWM-specific: an operator opts
in with invariant topology and a declared set of lane-varying static parameters.
The same planner can therefore form lane families for elementwise operators,
independent `xs_rank` branches, and independent Ridge models. Their executors
remain operator-family-specific because barriers and state differ: elementwise
families can fuse instrument loops, rank families need one preallocated sort
scratch per active lane, and Ridge families need independent pairwise clocks
and deterministic reduction/solve state. Cross-sectional and solve barriers
must not be fused as if they were ordinary elementwise loops.

Lifting is automatic and does not require a lane construct in the DSL. Users
continue to write ordinary branches under `cat`; lowering recognizes compatible
siblings from the optimized graph. The first cross-sectional family,
`cat(xs_rank(ewm(x, p1)), xs_rank(ewm(x, p2)), ...)`, now uses the same fused EWM
transition followed by one preallocated sort barrier per lane. Time remains the
outer sequential batch dimension: “instrument-major” describes only loop order
*within one timestep*. Consequently all instruments for a row have been updated
before `xs_rank` compacts, sorts, scans ties, and scatters that row's scores.

The “store-only ceiling” is an ablation, not a usable formula result. It repeats
each input value into every output lane while skipping EWM arithmetic and state;
it estimates the best achievable output-store rate for the validated memory
layout. Comparing it with real kernels distinguishes memory-bandwidth limits
from computation/state-transition limits.

Lane extraction walks each `cat` branch recursively, so optimization is not
limited to one producer/barrier pair. A branch such as
`xs_rank(ewm(xs_rank(ewm(x, p)), q))` becomes one native pipeline with two EWM
state stages and two explicit rank barriers. Preallocated ping-pong row buffers
carry lane values between barriers without returning to the generic node
interpreter. Every barrier still observes the complete current timestep.

Pattern discovery itself has no EWM or rank special case. It builds a canonical
lane graph from operator descriptors, complete child topology, invariant static
parameters, and source-input identities. Registered native-family factories
probe that graph and either construct an executor or decline it. The built-in
EWM/rank factory is therefore an executor plugin, not a conditional in the
public runtime. New stateless, n-ary, model, or grouped families extend the same
mechanism by registering descriptors and a capability probe; mismatched branch
topology is rejected rather than partially fused.

The remaining native operator set, including canonical composite-key groupby,
continues to use the existing flat-native executor when no specialized family
matches. Groupby cannot safely be treated as an elementwise lane: its optimized
form needs formula-specific inner transitions plus preallocated open-addressing
tables per universe, canonical NaN/zero hashing, and independently owned bucket
state. This fallback is deliberate until that executor has equivalence and
churn/locality profiles; cpp_new does not advertise placeholder kernels as
specialized support.
