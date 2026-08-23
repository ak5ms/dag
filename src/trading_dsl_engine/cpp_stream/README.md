# `trading_dsl_engine.cpp_stream`

`cpp_stream` is a formula-specialized C++20 streaming backend. It consumes the
backend-neutral `trading_dsl_engine.ir` graph, generates a typed translation unit,
compiles a cached shared library, prepares each input through an independent source
adapter, and executes the row loop in native code. It does not depend on `jax_flat`
at runtime.

```text
shared DSL / parser
        -> backend-neutral ir.Program
        -> cpp_stream physical lowering
        -> generated C++20
        -> heterogeneous source adapters
        -> one typed-pointer native entrypoint
        -> mmap output
```

## One compile API and one run API

```python
from trading_dsl_engine.cpp_stream import compile_formula

runtime = compile_formula(
    "xs_rank(ewm(close / open, 21))",
    {
        "close": "/data/close.npy",
        "open": "/data/open.npy",
    },
    n_instruments=9,
)
runtime.run(out_path="/data/alpha.bin")
```

`compile_formula` also accepts a list or tuple of formulas. All roots are built into
one neutral IR DAG, so CSE and physical fusion are shared across formulas and the
backend still emits one C++ translation unit and invokes one native runner. The
formulas do not need compatible output shapes:

```python
runtime = compile_formula(
    [
        ewm(close, 21),
        cat(ewm(close, 21), ewm(close, 63)),
    ],
    {"close": close},
    n_instruments=9,
)
result = runtime.run(out_path="/data/alphas.npy")
fast, features = result.load()

assert fast.shape == (rows, 9)
assert features.shape == (rows, 9, 2)
```

For a formula list, `RunResult.load()` returns an ordered tuple of shaped views in
the same order as the input formulas. The native writer packs those logical outputs
into one backing mmap; callers should use `load()` rather than manually interpreting
the packed file. Row-emitting and final-only formulas may coexist in the same list;
a plan containing any final-only root stays single-owner during finalization. Passing
a single formula retains the existing ndarray result behavior, while a one-element
list returns a one-element tuple.

Public-output storage is chosen without changing the hot compute path. A leaf
materialized public root may write directly to its packed output slice. When that
same value is still consumed by another public root, it remains in the same
scratch-local producer/consumer path as the corresponding top-only formula, and an
ordinary `CopyNode` or `CatNode` persists the requested extra result. The subgraph is
never recomputed. Public projections also re-enter normal physical fusion: for
example, compatible EWM outputs are folded into `EwmBundleNode` epilogues instead of
forcing a second row traversal or dead scratch writes.

The packed output descriptor retains each formula's exact logical size, offset,
row/final mode, and lane-partitionability as compile-time C++ metadata. Scratch is
also exact: matrix/tensor slots use their individual extents rather than a global
maximum width, and scalar slots are compacted independently by native dtype.

A representative GitHub-hosted single-thread run for 5M x 9 with
`subgraph = ewm(x + 1, span=32)` and `top = xs_rank(subgraph)` measured about
0.4060 s median for `top`, 0.4971 s for `[subgraph, top]`, and 0.5419 s for
`cat(subgraph, top)`. The list form therefore returned the same 720 MB of values as
the Cat baseline about 8.3% faster while preserving the top-only `ewm -> xs_rank`
locality; its roughly 22% delta versus top-only is the cost of persisting the extra
360 MB result. A heterogeneous `(N,) + (N,8)` case with equal total output bytes was
within about 1% of the equivalent Cat baseline on the same class of hosted runner.
A same-span EWM-bundle case compiled both the Cat and formula-list forms to one
`ewm_bundle` stage and measured within about 0.2%, showing that public-output
projection does not block the existing EWM epilogue fusion. Full reproducible runs
live in `scripts/benchmark_cpp_stream_multi_output_subgraph.py`.

There is no `.npy`-specific compiler or runner. Every input independently selects a
source adapter from its object type, URI scheme, file extension, or explicit adapter
name. A single formula may therefore mix formats:

```python
from trading_dsl_engine.cpp_stream import InputTypeSpec, source

runtime = compile_formula(
    "left + right",
    {
        "left": "/data/left.npy",
        "right": source(
            "/data/right.bin",
            input_type=InputTypeSpec("float64", 9),
        ),
    },
    n_instruments=9,
)
runtime.run(out_path="/data/result.bin")
```

Built-in adapters currently cover zero-copy C-order `.npy`, headerless `.bin`/`.raw`
with explicit metadata, and C-contiguous in-memory NumPy arrays. Custom adapters can
match extensions such as `.parquet`, URI schemes such as `tcp://`, or application
source objects through `register_source_adapter(...)`. See `SOURCES.md`.

Sources supplied at compilation are bound to the runtime. `runtime.run(new_sources,
...)` may replace them with another compatible mapping, including different source
formats with the same dtype and per-row shape.

Any positive C-order per-row tensor shape is supported: `(rows,)`, `(rows, 1)`,
`(rows, N)`, `(rows, N, K)`, `(rows, B, N, K)`, and higher ranks. `(rows,)` and
`(rows, 1)` are row scalars. Supported native dtypes are `float32`, `float64`,
`int32`, `int64`, `uint32`, and `uint64`.

## NumPy-style `einsum`

The canonical call order matches NumPy:

```python
einsum("ij,jk->ik", left, right)
einsum("...ij,...jk->...ik", left, right)
einsum("ij,ij->", left, right)
einsum("ii->i", square)
einsum("ij,kj,kl->il", a, b, c, optimize="optimal")
```

Supported string-subscript behavior includes arbitrary case-sensitive ASCII labels,
implicit and explicit outputs, scalar operands and reductions, arbitrary rank,
diagonals, permutations, outer products, ellipsis broadcasting, and optimized n-ary
contraction paths. Named labels require equal dimensions; broadcasting is enabled
through `...`, matching NumPy.

The default is `optimize=False`. `True`, `"greedy"`, and `"optimal"` are supported.
`"optimal"` exhaustively searches paths through eight operands and falls back to
greedy for larger expressions.

Subscripts are parsed once in the neutral IR and lowered to static unary/binary
contraction stages. Generated C++ contains no runtime string parser, shape dispatch,
or path search. Contiguous inner reductions use bulk loads and FMA loops; generic
mapped loops cover diagonals, broadcasting, permutations, and arbitrary contraction
axes.

The native API does not yet implement NumPy's integer-sublist calling form,
precomputed path lists, `out=`, `dtype=`, `order=`, `casting=`, or writeable-view
semantics. Native einsum accumulation/output is currently `float64`.

## Streaming reductions

Reduction axes address `(time, *row_shape)`, so axis `0` is time. Omitting `axis`
matches NumPy and reduces all logical axes. Row-only reductions remain composable
per-row stages. Temporal `sum`, `mean`, and `std` update fixed-size state for every
row but call the result projection only once during finalization; any downstream
algebraic suffix also runs once from that final value. Neither a temporal reduction
nor `emit("last")` allocates or writes a time-sized output.

```python
features = cat(x, y, z)
per_instrument = features.sum(axis=2)
one_scalar = features.sum()
```

See `REDUCTIONS.md` for NaN, `ddof`, shape, and output-mode semantics.

## Cross-sectional and streaming statistics

All lookbacks are expressed as `periods`, meaning input rows. Cross-sectional
percentile rank is exposed as `xs_pct_rank`; finite ties receive their shared upper
rank and nonfinite lanes remain NaN.

Statistics with a natural exponentially weighted definition are compositions of
the existing `ewm` operator. They use its `span`, `min_periods`, `ignore_na`, and
`adjust` conventions, so there is only one EWM state machine to maintain:

```python
ewm_moment(x, span=32, k=3, min_periods=8, ignore_na=True, adjust=False)
ewm_var(x, span=32)
ewm_std(x, span=32)
ewm_skewness(x, span=32)
ewm_kurtosis(x, span=32)
ewm_cov(x, y, span=32)
ewm_corr(x, y, span=32)
ewm_co_skewness(y, x, span=32)
ewm_co_kurtosis(y, x, span=32)
ewm_triple_corr(x, y, z, span=32)
ewm_partial_corr(x, y, z, span=32)
```

Multivariate statistics use shared complete observations: any incomplete tuple is
passed to every component `ewm` as one missing observation, and its effect follows
the selected `ignore_na` mode. Variance and standardized higher moments are
population statistics; kurtosis is not excess kurtosis.

These remain ordinary DSL compositions. Physical lowering detects compatible
sibling `ewm` nodes and emits one variadic `EwmBundleNode`; it is not a co-moment
kernel. The bundle accepts any generated scalar expression graph, shares validity
metadata while observation masks agree, and splits to per-component metadata when
they diverge. If its only consumer is scalar algebra or a scalar-width `cat`, that
consumer becomes a generated epilogue over the live EWM state, avoiding a second
row traversal and raw-moment scratch writes.

Statistics without a useful EWM definition use fixed-row windows:

```python
rolling_sum(x, periods=20)
rolling_mean(x, periods=20)
rolling_std(x, periods=20, ddof=0)
rolling_min(x, periods=20)
rolling_max(x, periods=20)
rolling_median(x, periods=20)
rolling_quantile(x, periods=20, q=0.25)
rolling_pct_rank(x, periods=20)
rolling_argmin(x, periods=20)
rolling_argmax(x, periods=20)
rolling_theilsen(y, x, periods=63)
```

Rolling moments use removable stable state, min/max and their relative indices use
monotonic deques, and order statistics use an allocation-free order-statistics tree
for windows of at least 64 rows. Backfill walks a fixed recency list only to the
requested `k`; previous-different lookup adaptively changes from a bounded fast scan
to O(1) run state; entropy reuses the order tree for extrema and scans active values
once. Theil-Sen uses exact pairwise median selection through 512 rows and a
fixed-memory subquadratic inversion-count selector above that boundary. No
implementation allocates in `on_data`.

Cheap formulas such as `ewm_std`, `ewm_skewness`, `ewm_kurtosis`, `xs_zscore`,
`xs_scale`, `xs_vector_neut`, `rolling_range`, `rolling_zscore`, and `rolling_scale`
live in `cpp_stream.python.utils` and expand to the native primitives above.

## Compile-time CSE and physical fusion

The neutral IR deduplicates stateless expressions, including safely commutative
binary forms such as `x + y` and `y + x`. NaN literals share one semantic key, while
signed zero and order-sensitive minimum/maximum retain their original ordering.

Lowering then keeps stateless scalar and tensor expressions as typed sources instead
of assigning one scratch slot per AST node. They materialize only at an actual
pointer boundary such as grouped feeds or a dense model feature matrix. Generated
C++ exposes the nested expression type to the optimizer and uses a typed per-lane
cache when several fused consumers share a subexpression.

Compatible stateful siblings use one generic physical operation:

- EWM siblings share one traversal and pandas-style metadata;
- tensor reductions with equal shape/axes/policy share one source pass;
- projections of the same Ridge object share sufficient-statistic updates, one
  solve, and one inference calculation.

This is graph-driven rather than operator-name-driven: `ewm_co_kurtosis`, for
example, remains a utility composition but lowers to one eight-member EWM bundle.
Specialized cross-sectional algorithms (`xs_rank`, `xs_pct_rank`, and related
nodes) remain physical stages so fusion does not replace their tuned sort/ranking
implementation.

## Ridge projections and named regression results

`Ridge(...)` remains the model object. Native projection functions expose its fitted
values and inference without introducing a second regression implementation:

```python
from trading_dsl_engine import cat
import trading_dsl_engine.cpp_stream as cpp

model = cpp.Ridge(cat(1.0, x1, x2), y=y, weights=w, hl=32, lambda_=0.1)
cpp.get_beta(model)
cpp.get_preds(model)
cpp.get_residuals(model)
cpp.get_r2(model)
cpp.get_standard_errors(model)
cpp.get_tstats(model)
cpp.get_effective_df(model)
```

SSE, SST, R-squared, residual variance, individual coefficients/standard errors/
t-statistics, and effective sample size are also available. Inference uses positive,
finite, complete-case weights. Ridge covariance is
`sigma^2 A^-1 X'WX A^-1`, where `A` is this backend's regularized system, and its
residual degrees of freedom account for both `trace(H)` and `trace(H^2)`. Constrained
nonnegative Ridge exposes fit metrics but returns NaN for covariance-based inference.

`ts_regression` is an EWM weighted-Ridge composition. Its `periods` argument is the
half-life in rows, and `rettype` accepts descriptive values such as `"residual"`,
`"prediction"`, `"intercept"`, `"beta"`, `"r2"`, `"beta_stderr"`, and
`"beta_tstat"`; numeric selectors are rejected.

## Generated convex programs

`@cpp_stream.optimizer.cvxpy_program` is the normal DSL boundary. The decorated
function declares its explicitly named `cp.Parameter` objects and attributes,
then the same function binds DAG expressions. `previous_solution("weights[0]",
initial=...)` carries an actual prior primal into the next solve and is inferred
as temporal state; independent programs remain row-parallel with one generated
instance per worker.

`cpp_stream.optimizer.generate_clarabel_program(...)` compiles a static-shape,
DPP-compliant CVXPY problem in bounded parameter shards and emits an
instance-owned direct-Clarabel C++ class. CVXPY supplies cone canonicalization;
the generator merges compact parameter-to-`P/A/q/b` maps without building the
full DPP tensor. The generated class retains one Clarabel solver, updates dirty
fixed-sparsity blocks on subsequent solves, and frees the solver in its
destructor.

Every independent native worker must own a separate generated instance. Mutable
parameter/canonical/result buffers are per instance; immutable generated maps
and cone descriptors are shared. `GeneratedClarabelProgram.build_shared_kwargs()`
connects the generated headers and pinned Clarabel archive to cpp_stream's
normal translation-unit cache and build path. See
[`docs/cvxpygen_cpp_stream.md`](../../../docs/cvxpygen_cpp_stream.md) for the
compile-time and C++ interfaces.

## Execution model

Every operator has one native implementation and receives its execution scope as
the final template argument:

```cpp
DirectExecution<N>
GroupedExecution<N, Capacity, PartitionCount>
```

There are no `GroupedFooNode` or formula-specific fast-path classes. `groupby.hpp`
contains only key resolution, grouped-context construction, and inner-plan
invocation. No operator allocates from the heap during `on_data`.

`cat(...)`, RBF bases, coefficient matrices, and einsum use compile-time dimensions.
Lazy basis sources and nested Cat expressions flatten into `FeatureList<Sources...>`
so consumers read original inputs directly. Arbitrary intermediates use compact
fixed-size tensor scratch only when a contraction path requires them.

## `riskmodel.roll_rets`

```python
from flows.riskmodel import roll_rets
from trading_dsl_engine.cpp_stream import compile_formula

runtime = compile_formula(
    roll_rets,
    paths,
    n_instruments=9,
    default_group_capacity=4096,
)
runtime.run(out_path="roll_rets.bin")
```

The generated plan contains 50 scalar/vector scratch slots and one six-wide matrix
scratch slot. RBF and future-RBF basis values remain lazy. A native end-to-end test
compares the exact expression against JAX-flat with finite-output checks and
`rtol=2e-9`, `atol=2e-9`, equal-NaN semantics.

## 5M x 9 benchmarks

GitHub-hosted Ubuntu runner, GCC C++20 with
`-O3 -march=native -mtune=native -flto`, one warmup and ten measured executions:

| Workload | Median throughput |
| --- | ---: |
| `einsum("nf,nf->n", ...)`, six features | 12.076 M rows/s |
| equivalent ellipsis reduction | 12.097 M rows/s |
| scalar reduction `einsum("n,n->", ...)` | 85.027 M rows/s |
| three-operand contraction, `optimize=False` | 5.533 M rows/s |
| same contraction, greedy | 11.605 M rows/s |
| same contraction, optimal | 11.624 M rows/s |
| full `flows.riskmodel.roll_rets` | 0.865675 M rows/s |

For the n-ary case, planning reduced estimated work from 324 to 72 operations per
row and reduced the largest intermediate scratch width from 9 to 2. Checksums were
identical and every sampled output was finite. Full distributions and checksums are
in `PERFORMANCE.md`.

## Compilation cache

The cache key includes generated source, packaged headers, compiler identity,
compile/link flags, platform/machine, and Python ABI. The default cache is:

```text
~/.cache/trading_dsl_engine/cpp_stream
```

Override it with `TRADING_DSL_ENGINE_CPP_STREAM_CACHE`.

### Generated convex-program stages

A generated direct-Clarabel artifact can be bound to normal formulas with
`bind_program(...)` and projected with `get_field(...)`. The object stage is
lowered into the ordinary runner stage list: upstream Ridge/risk-model values,
the Clarabel solve, and downstream formulas all execute in the runner's single
row loop. Sibling field projections share one solve. See
`examples/cpp_stream_mpo_one_pass.py`.
