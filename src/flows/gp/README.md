# Strongly typed GP

`flows.gp` builds a row-preserving DEAP `PrimitiveSetTyped` for formulas that
compile through `cpp_stream`.

## Core rules

- InputData fields are typed as price, quantity, timestamp, duration, trading-day
  horizon, dimensionless, count, or boolean rows.
- `diff(TimestampRow, PositiveInt)` returns `DurationRow`.
- Positive numbers, booleans, axes, quantiles, frequencies, and other compile-time
  parameters use dedicated static GP types instead of pretending to be rows.
- The default static terminal grid includes dense intraday integer windows,
  positive/negative scalar floats, tail/interior quantiles, and an exact zero
  `ScalarNumber`, so neighboring-parameter mutations remain local.
- EWM and rolling `min_periods` are derived from their generated period. Other
  dependent arguments such as shift capacity and rolling-kth `k` are also fixed
  or derived so GP cannot create inconsistent combinations.
- Axis 0 and all-axis reductions are unavailable. Non-temporal reductions are
  broadcast back to row shape.
- Raw dimension-changing/search-structure primitives such as `emit`, `cat`,
  `einsum`, `groupby`, `cache`, `buffer`, basis expansions, raw Ridge objects,
  and raw Ridge getters are not GP node types.
- Dimension-changing intermediates may be used inside a composition when its
  final output is a lane-shaped row.
- All imports inside `flows.gp` are absolute.

## Geno and pheno robustness tests

`geno` tests inspect only the DEAP/DAG structure. `run_geno_tests()` always checks
prefix-tree type consistency and accepts additional zero-execution rules such as
`geno_max_depth`, `geno_max_nodes`, `geno_forbid_families`, or a custom
`GenoTest` predicate.

```python
from flows.gp import GenoTest, geno_max_depth, make_pset, random_formula, run_geno_tests

pset = make_pset()
tree, _ = random_formula(pset, seed=7)
report = run_geno_tests(
    tree,
    pset,
    tests=(
        geno_max_depth(6),
        GenoTest("has_terminal", lambda ctx: ctx.terminal_count > 0),
    ),
)
assert report.passed
```

`pheno` tests require an evaluator and therefore an actual baseline/trial run.
Each trial can perturb both kinds of leaves:

- sortable static terminals are replaced by a randomly selected value within
  `k` adjacent values of the current terminal in that type's sorted terminal
  grid;
- dynamic field leaves can be independently wrapped in seeded distribution noise.
  Noise parameters may themselves be DAG expressions or callables of the field
  leaf, which makes rolling/EWM scale estimates straightforward.

```python
from flows.gp import NoiseSpec, pheno_finite, run_pheno_tests
from trading_dsl_engine.base import dsl

noise = {
    "ap0_out0": NoiseSpec(
        "normal",
        params={
            "mu": 0.0,
            "sigma": lambda x: dsl.maximum(dsl.ewm_std(x, 60), 1e-8),
        },
        mode="add",
    )
}
report = run_pheno_tests(
    tree,
    pset,
    evaluator=run_formula,  # compile/run callback supplied by the search
    tests=(pheno_finite(),),
    n_trials=8,
    static_k=2,
    field_noise=noise,
    seed=11,
)
```

Every trial records its selected static replacements, dynamic field occurrences,
distribution names, seeds, evaluator result, and test outcomes. Runtime exceptions
are retained as failed trial outcomes rather than being discarded.

## Seeded random DSL compositions

Importing `trading_dsl_engine.base` registers `uniform`, `normal`, `lognormal`,
and `exponential` as ordinary DSL macros and exposes them on `dsl`. Their
parameters may be dynamic expressions. They are implemented entirely from
existing arithmetic primitives, so no backend-specific RNG operator or hot-loop
fallback is introduced.

```python
from trading_dsl_engine.base import dsl

x = dsl.var("ap0_out0")
mu = dsl.ewm(x, 60)
sigma = dsl.ewm_std(x, 60)
e = dsl.normal(mu=mu, sigma=sigma, key=x, seed=7)
shocked = x + e
```

The generator is keyed and deterministic for a given key/seed. Passing `key=x`
is recommended for per-field perturbations. If `key` is omitted, the first
non-literal distribution parameter is used; when all parameters are literals the
conventional `_ev_ts` input is used as the key.

## Interactive GP graph explorer

`explore_gp(pset)` writes and opens a standalone Plotly HTML explorer. Its initial
view shows type-to-type relations. Clicking a type drills into the connected
primitive variants and terminals; clicking any resulting node recenters the
neighborhood. The search box matches type names, operator families/primitive
names, grammar sections, terminal names, signatures, and terminal values.

```python
from flows.gp import explore_gp, make_pset

explore_gp(make_pset())
```

Use `build_gp_graph()` for the backend-neutral graph model,
`filter_gp_graph()` for programmatic search, or `gp_explorer_html()` when the
caller wants to embed the explorer rather than open it.

## Regression composites

The GP exposes row-valued compositions that internally use regression objects:

- `ts_regression(y, x, periods)` has separate primitive variants for residual,
  prediction, intercept, beta, SSE, SST, R2, residual variance, intercept/beta
  standard error, intercept/beta t-stat, effective degrees of freedom, and
  effective sample size.
- `ridge_<projection>(y, x1[, x2[, x3]])` builds a stateless rowwise Ridge and
  immediately projects it back to a row. The model object is never a GP value.
- `ts_poly_regression(y, x, periods)` has degree-1, degree-2, and degree-3
  residual variants.
- `xs_regression_neut(y, x)` exposes the predefined cross-sectional regression
  neutralization composition.

Scalar regression statistics are broadcast back across the target row before
returning from the GP primitive.

## DEAP generation

Tree construction uses DEAP's standard toolbox path rather than a custom subtree
generator:

```python
from flows.gp import make_pset, make_toolbox

pset = make_pset()
toolbox = make_toolbox(pset, min_depth=2, max_depth=5)
individual = toolbox.individual()
```

`make_toolbox()` registers `gp.genHalfAndHalf`, `tools.initIterate`, and
`tools.initRepeat`. `random_formula()` is a convenience wrapper around the same
DEAP path.

## Exact signatures

```python
from flows.gp import format_signature_table, make_pset

print(format_signature_table(make_pset()))
```

The compiler-fuzz tests generate random trees, wrap them through
`default_alpha_pnl`, and run both neutral-IR lowering and full native
`compile_formula`. `GP_FUZZ_MIN_DEPTH` and `GP_FUZZ_MAX_DEPTH` can be used for
deeper validation campaigns.
