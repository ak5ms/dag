# Strongly typed GP

`flows.gp` builds a row-preserving DEAP `PrimitiveSetTyped` for formulas that
compile through `cpp_stream`.

## Core rules

- InputData fields are typed as price, quantity, timestamp, duration, trading-day
  horizon, dimensionless, count, or boolean rows.
- `diff(TimestampRow, PositiveInt)` returns `DurationRow`.
- Positive numbers, booleans, axes, quantiles, frequencies, and other compile-time
  parameters use dedicated static GP types instead of pretending to be rows.
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
