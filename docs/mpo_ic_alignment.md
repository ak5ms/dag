# IC-aligned, volatility-scaled one-pass MPO

Source baseline: `agent/strongly-typed-gp` at
`834eebe165c2c6832c8328c76c00fa596ce815ae`.

## Feature and forecast units

For each instrument and configured EWM **span**:

```python
clean = where((abs(returns) <= 0.05) & (returns != 0), returns, nan)
alpha = xs_gauss(xs_rank(-ts_zscore(clean, span)))
sigma = ewm_std(clean, span=IC_VOL_SPAN)
x = alpha * sigma
```

This matches the supplied `scratch_10.py` expression, including its zero/outlier
filter and its span convention. The first span is 4. Other configured spans are
16, 64, and 256. These are not converted from half-lives.

Ridge learns a return-unit forecast `yhat_rate = x @ beta`. Its coefficients are
correlation-like loadings, not correlations mathematically constrained to
[-1, 1]. Ridge regularization remains `lambda_=0.1`, with the existing
`lambda * diag(XX)` penalty. No coefficient sign restriction is added.

The return-volatility warmup remains the reference's 30,240 observations.
The former 20,000-row default could never activate those diagnostics. The
example now defaults to 200,000 input rows and reports insufficient/no valid
volatility instead of silently producing an all-zero forecast run.

## One shared sample clock

For block `(a, a+b]`, `_ic1_terms` calls `_ic_terms`, then returns:

- `X_t = position[t-b]`, where `position[u]` updates from `x[u-a]` only on
  tradable rows with a finite candidate, otherwise holding the last value;
- `Y_t = mean(clean_return[t-b+1:t+1])`, with missing realized contributions
  replaced by zero, exactly as in the public IC accounting;
- `W_t = normalized_held_weight[t-b]`, using the same observation-time
  normalization, per-instrument session hold, and all-zero-weight-row behavior.

Away from gaps this reduces to `X_t = x[t-(a+b)]`. Across gaps, a plain
`shift(x, a+b)` is **not** equivalent. A current target-time spread weight is
also not equivalent to the held observation weight.

A window with no finite cleaned return observations is excluded from fitting.
Both X and Y are masked, and its sample weight is zero: masking only Y would
still allow Ridge's pairwise XX statistics to update.

After observing return `t`, the current fitted `beta_t` may be applied to the
current `x_t`. The resulting forecast concerns future returns only; an
additional shift around `get_beta` is neither needed nor inserted.

The MPO receives `mu_h = b * yhat_rate_h`, because the shared IC1 target is a
**mean**. This multiplication occurs once. Known entirely closed future
return windows receive zero expected return. A partly closed window that
contains reopening is not multiplied by its open fraction: its full reopening
return already belongs to the block target.

The optional `fit_weights` argument accepts `1` or a caller's liquidity
expression. The default is the original inverse-half-spread-squared candidate,
normalized and held by the shared IC helper. The named
`ridge_pool_capacity_share()` helper in the pasted script is absent from this
repository snapshot; no substitute claiming to reproduce that unavailable
helper has been invented.

## Diagnostic reconciliation

Public `ic` and `ic1` interpret their input as an alpha and internally divide
it by return volatility. Therefore a **return-unit** forecast is supplied to
those functions as `yhat_rate / sigma`, not as `yhat_rate`:

```
IC position for yhat = (yhat_rate / sigma) / sigma
                    = yhat_rate / sigma**2.
```

With one feature and `beta_override=1`, `yhat_rate = alpha * sigma` and the
supplied diagnostic alpha is exactly the original alpha. The identity holds
for every configured horizon, including session gaps and missing liquidity.
With several features, the scalar override sets every coefficient to that
scalar; use one feature to isolate each identity.

`xs_sum` is broadcast over instrument lanes in this DSL. The supplied
`scratch_10.py` then explicitly applies `np.nansum(..., axis=1)`. The example
now does the same NaN-to-zero instrument **sum**, not an instrument mean.
Consequently, these diagnostic plots preserve the reference's factor of N;
they should not be mistaken for independently normalized portfolio returns.

For `hz=1`, IC and IC1 use the same attribution timestamp. For `hz>1`, IC
attributes to each realized-return row and IC1 to the end of each target
window. Their paths need not agree row by row or day by day. The independent
pandas ablation confirms equal totals after a zero-return tail flush, while
also asserting that their interior rowwise paths differ.

## Actual execution and sessions

A signal observed at row t's VWAP cannot be filled retrospectively at that
VWAP. The first forecast block remains `(1,2]`; the removed `(0,1]` block is
not reintroduced.

| Row | Event |
|---|---|
| t | Observe VWAP and return t; update model; generate next-VWAP plan. |
| t+1 | Execute that queued plan when tradable with a valid quote. |
| t+2 | First return earned by that execution. |

The optimizer carries both its previous first-stage **plan** and its previous
**executed** holdings. Its `previous_weights` primal resolves the current
execution mask. Closed or missing-quote lanes retain the actual holding; an
unfilled queued change is not treated as a fill. These auxiliary primal
bindings keep the problem DPP-compliant.

The outputs distinguish `planned_weights`, full `planned_path`, and actual
`weights`. Gross PnL is yesterday's actual holding times today's raw finite
return. Realized cost is the current half-spread times the absolute change in
actual holdings; net PnL subtracts it once. The cumsum of the minimized MPO
objective remains a diagnostic of repeatedly solved plans, **not** a realized
PnL series.

`RollRets` ignores closed-session price marks and puts the entire close-to-open
move on the first tradable reopening row. The native ablation deliberately
uses bogus closed prices and verifies the complete +10%/-10% reopening moves.
The end-to-end test also includes a 7.5% gap: feature cleaning excludes it, but
realized PnL and risk retain it.

Risk observes the latest matured raw **total** for each block width. It does
not divide a gap by closed minutes or use the feature-cleaned series. Valid
zero returns while open remain risk observations; closed zero placeholders do
not advance a width-one risk observation. No second shift by the forecast
start is applied to the already matured covariance sample.

The calendar schedule assumes a regular minutely row grid and valid current
and next session boundaries. A missing timestamp is extrapolated from the last
observed timestamp and row count. The statistical risk/return models remain
empirical rolling models, not a separately estimated conditional weekend or
opening-auction model. This change establishes accounting and timing
consistency; it does not claim conditional forecast calibration or profitable
out-of-sample trading. Returns beyond the last configured horizon, 128 rows,
are outside this finite-horizon objective.

## Two lower-level correctness fixes

**Recycled expression IDs.** The shared IR previously memoized `id(expr)`
without validating that the referenced object still existed. Macro expansion
creates temporary ASTs, so Python could reuse an ID during one compilation.
Unrelated signals and normalized weights then shared a structural key. The
cache now uses weak identity references, checks the exact object, and removes
expired entries. A deterministic regression test reproduces the original
collision, without retaining all temporary ASTs indefinitely.

**Ridge observation-clock EWM.** The native Ridge implementation used
`alpha**gap` as the next observation's update weight after missing statistics.
For a long closure that can virtually discard the reopening sample. XX, XY,
and metric updates now freeze on missing data and resume with ordinary alpha,
matching pandas `adjust=False, ignore_na=True`. The test with a 50-row gap and
half-life 4 changes the expected coefficient from the old erroneous 1.0 to
approximately 0.522689. This correction applies to native `cpp_stream` Ridge,
not just to the example.

## Failed solves are not positions

Frozen holdings and a newly tightened/changed hard risk constraint can be
infeasible. This is a real modeling conflict, not permission to consume a
solver's infeasibility certificate as the next portfolio.

A sequential native optimizer now accepts only solved/almost-solved results
for feedback. A non-solution returns checked native error 7 before updating
any feedback cache. The Python runtime raises a clear error. The noexcept hot
loop is not made to throw, and no risk limit is silently relaxed.
Non-feedback optimizers retain their existing ability to expose status and
certificate outputs for inspection. Applications needing an emergency
liquidation, relaxed risk, or stale-solution policy must specify that policy
explicitly; the example does not invent one.

## Reproducible focused validation

```bash
PYTHONPATH=src:. python -m pytest -n 0 -q \
  tests/examples/test_cpp_stream_mpo_diagnostics.py \
  tests/trading_dsl_engine/cpp_stream/test_cat_ridge.py \
  tests/trading_dsl_engine/cpp_stream/test_ridge_projections.py \
  tests/trading_dsl_engine/cpp_stream/test_ridge_recompute.py \
  tests/trading_dsl_engine/cpp_stream/test_compile_optimizations.py \
  tests/trading_dsl_engine/cpp_stream/test_cvxpy_program_native.py \
  tests/trading_dsl_engine/cpp_stream/test_cvxpy_constraint_values_and_guard.py \
  tests/trading_dsl_engine/cpp_stream/test_where_broadcast.py \
  tests/trading_dsl_engine/cpp_stream/test_xs_gauss.py \
  tests/trading_dsl_engine/cpp_stream/test_codegen.py \
  tests/trading_dsl_engine/cpp_stream/test_multi_output_runtime.py \
  tests/trading_dsl_engine/ir tests/trading_dsl_engine/base \
  --deselect=tests/trading_dsl_engine/cpp_stream/test_compile_optimizations.py::test_known_instrument_count_uses_one_ir_build \
  --deselect=tests/trading_dsl_engine/cpp_stream/test_compile_optimizations.py::test_header_digest_cache_invalidates_after_header_edit
```

A network-enabled installation can build the native Clarabel dependency as
usual. The local audit environment had no DNS/network access, so source,
Python dependencies, and a portable native Clarabel build were retrieved as
artifacts. Only the installer function was redirected to that existing
include/library pair for the broader tests; the production generated C++,
parameter maps, solver, and row loop all ran **locally**, without mocking their
behavior. No full private InputData run was performed.

The test suite includes independent pandas IC terms, raw feature parity,
beta=1 at all seven horizons, independent pairwise normal equations, and a
future-data perturbation with bitwise-identical earlier coefficients/forecasts.
It also checks the existing column-major binding: DSL `(asset,horizon)`
buffers become CVXPY `(horizon,asset)`, and a factor L with S=L L' arrives as L',
so the SOC correctly evaluates sqrt(w' S w). No extra transpose was added.

### Baseline failures kept separate

The unmodified source archive was run in a separate checkout. It reproduces:

- `test_known_instrument_count_uses_one_ir_build`: the existing compiler makes
  two IR calls, while this older test expects one.
- `test_header_digest_cache_invalidates_after_header_edit`: the test refers to
  a removed private `_header_digest` function.
- The legacy `test_roll_rets.py` fixture supplies `mp_out0.close` and
  `mp_out1.close`, but the current production formula requires `vwap_mp_out0`
  and `vwap_mp_out1`. It is excluded from the focused command above; the new
  native gap test uses the actual current fields.
- Two `tests/flows/test_alpha_search.py` GP-search tests fail because the
  pre-existing toolbox map calls the evaluator on a population rather than
  mapping it over individuals. No GP search behavior is changed here.

The older generated-manifest test's expected schema version was updated from
4 to the existing version 5, and the Ridge signature fixture now includes
the existing `recompute_every` parameter. The allocation harness was migrated to the
checked solver-node entrypoint and still verifies zero warm-path allocations.
These known baseline failures are not presented as passing or newly fixed.

### Measured validation result

The final focused command completed locally with **116 passed, 2 explicitly
deselected baseline failures, and 2 CVXPY canonicalization warnings**. The
warnings describe CVXPY falling back to its SciPy canonicalization backend;
execution still uses the native generated Clarabel solver.

`python scripts/audit_mpo_ic_alignment.py` additionally runs the normal-equation,
beta=1, and return-unit scaling ablations on both **3 and 9 instruments**, at
all seven horizons. It writes the measured maximum errors to
[`../benchmarks/mpo_ic_alignment.json`](../benchmarks/mpo_ic_alignment.json).
The audit uses the same independent reference helpers as the regression tests,
shortens warmups explicitly for simulated data, and does not load private
market files. Doubling the units of returns must leave beta unchanged and
must double yhat; the 5% cleaning boundary is not crossed by any retained
observation in that scaling case.
