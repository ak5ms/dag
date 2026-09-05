# One-pass MPO: IC alignment, rho fitting, execution and gap-risk audit

Base: `agent/strongly-typed-gp` at
`834eebe165c2c6832c8328c76c00fa596ce815ae`.
Implementation: `examples/cpp_stream_mpo_one_pass.py`; deterministic numerical
regressions: `tests/examples/test_cpp_stream_mpo_diagnostics.py`.

## What was wrong

| Ablation / boundary | Reproduced defect | Fix |
|---|---|---|
| Raw alpha versus the supplied scratch recipe | Different cleaning, unranked/un-Gaussianized features, and halflife-to-span conversion | Use the exact zero/5% cleaning and `xs_gauss(xs_rank(-ts_zscore(clean, span)))`; `(4,16,64,256)` are literal spans |
| Canonical observation state versus Ridge inputs | Shifting raw features and sampling weights at label time skipped IC's execution/hold state | Obtain position and normalized, session-held weight from `_ic_terms`; use IC1's trailing return mean and horizon shift |
| Regression units | Dimensionless predictors were fitted directly to returns | Fit `alpha * sigma`; apply the same scaling when predicting |
| Missing labels | Ridge XX can update without a finite Y | Apply one complete-case eligibility mask to X, Y and weight in this workflow, without changing generic pairwise Ridge semantics |
| Post-outage update | The first new sample was weighted by `alpha**elapsed_rows`, nearly zero after a closure | Advance each sufficient statistic by one observation-time update; synchronize cpp_stream, JAX-flat scan/tick, and active JAX-flat native code |
| Temporary expression construction | An id-only cache returned a different, deallocated expression's structural key when Python reused its address | Retain the node alongside its key, use thread-local storage, clear at compilation boundaries |
| Plan versus actual portfolio | Today's plan earned the next observed VWAP return before its permitted fill | Separate previous plan from executed holdings; fill the prior plan only on a currently executable lane |
| Missing mask/quote predicates | NaN-valued comparisons could enter a true branch in DSL `where` | Explicit finite/zero-filled control predicates; distinguish current execution from calendar-known future opportunities |
| Reopening marks | Generic ordinary block risk/fit mixed large total gaps with minute observations | Separate ordinary calibration, ordinary risk, total event-time gap risk, and unfiltered realized PnL |
| Unsuccessful feedback solve | An infeasibility certificate could become next-row portfolio state | Quarantine unsuccessful feedback trajectories: NaN outputs plus the original failure status; `_run` raises instead of plotting them |
| Full control-graph compilation | Shared lazy subexpressions expanded repeatedly in generated C++ | Bound shared expanded work using generic preallocated materialization, retaining small-expression fusion |
| Diagnostics startup / scale | 20,000 rows could not initialize the 30,240-observation volatility; averaging did not match scratch's `nansum` | 90,000 default input rows plus explicit no-fit rejection; preserve scratch's displayed cross-sectional sum |

The cache and Ridge-clock bugs are below the example layer. An alignment-only
rewrite without fixing those defects still failed numerical ablations.

## Units and canonical training pairs

Let `a[t,i,f]` be an alpha and `sigma[t,i]` the **same** return-volatility expression
used by canonical IC, including its observation-count warmup. The example defines

```
x[t,i,f] = a[t,i,f] * sigma[t,i]
mu_rate[t,i,h] = sum_f x[t,i,f] * beta[t,f,h]
```

These coefficients are rho-like slopes, not necessarily correlations and not
constrained to [-1,1]. Multiple correlated predictors, ridge shrinkage, and
weighting all matter.

For horizon `(lag, lag+hz]`, `_ic_terms(x, lag=lag, ...)` supplies the tradability-
aware held position `p` and observation-time normalized, held weight `q`.
The exact canonical IC1 product is built from

```
X[t,i,f] = p[t-hz,i,f]
Y[t,i]   = mean(clean_return[t-hz+1:t+1,i])
W[t,i]   = q[t-hz,i]
```

No forward reads or negative shifts are used. In a continuous, fully observed
session this reduces to `X[t] = x[t-lag-hz]`. Across closures, a bare shift is not
an equivalent substitute for the held canonical state.

The example learns an **ordinary-open-bar rate**. It admits only fully observed,
non-gap, non-outlier target blocks, positive finite weights, and initialized
features. The same eligibility masks all three Ridge inputs. Both the canonical
pairs and the filtered fit pairs are exposed under `values['fit'][horizon]`.
Ridge remains pooled across instruments, with a separate model per horizon.

A coefficient is updated after the current realized label is available. That
coefficient predicts future blocks; it is never retroactively multiplied into an
old forecast for the diagnostic backtest. IC receives the timestamped forecast
stream normally.

### Comparing predicted and feature PnL

`ic` and `ic1` each divide their signal argument by sigma. A return forecast must
therefore be converted back to alpha units first:

```
alpha_hat = mu_rate / sigma
ic(alpha_hat, ...)    # exposure is mu_rate / sigma**2
ic1(alpha_hat, ...)
```

With one feature and `beta_override=1`, `alpha_hat == a`, and both diagnostics
reconcile with the original feature diagnostics at every timestamp, including
NaNs, mask closures and held weights. Multi-feature zero coefficients are omitted
rather than evaluating `0 * NaN`.

For `hz > 1`, IC and IC1 deliberately attribute the same products to different
timestamps. Their *totals* reconcile after a complete tail; requiring the two
per-timestamp series to be equal would be an incorrect test.

Canonical `xs_sum` currently broadcasts its result across instrument lanes.
The supplied scratch script then does `np.nansum(..., axis=1)`. The example
preserves that display convention, including its lane-count factor, so the plots
match; this display convention is not an additional leverage multiplier in MPO.

## Four return streams, not one ambiguous cleaned series

1. **Signal/IC stream:** exact scratch policy, zero -> NaN and abs(return) > 5%
   -> NaN. Moderate gap returns remain in this reference feature stream, exactly
   as in scratch. Sigma uses this stream too.
2. **Ordinary calibration:** excludes *every detected reopening gap*, even one
   below 5%, and excludes the >5% outliers. Missing target observations are not
   silently learned as zeros.
3. **Risk:** ordinary fully observed trailing block second moments are separate
   from event-time short/long gap second moments. Ordinary finite outliers remain
   in ordinary risk. Finite gap outliers remain in gap risk.
4. **Economic PnL:** every finite raw return is marked against previously executed
   holdings. The signal cutoff never erases an 8% or 12% reopening loss/profit.

A gap is detected using the timestamp of the last observed tradable mark, not
merely the current mask. Short closures and closures longer than 1,440 minutes
use distinct EWM event clocks. No total gap return is divided by closure length
or multiplied by a minute-count horizon normalization.

Future ordinary expected return is `mu_rate * number_of_ordinary_open_returns`.
Known calendar intervals supply this count; reopening returns are not ordinary
minute returns. Future gap expected return is zero in this example: there is no
separately demonstrated overnight alpha model.

## Portfolio execution and risk constraints

At row t, the preceding plan is executed only for lanes whose realized mask,
calendar state, and current quote allow execution. Other lanes retain their
previous actual holdings. The solver then plans the next trade, at t+1 or later.
Thus a signal first observed at t cannot earn `return[t+1]` at its just-observed
VWAP; its earliest new-position return is `return[t+2]`.

The objective remains **expected return minus spread cost**, with risk in
constraints. No spread-budget constraint was added. Planned positions are dollar
neutral; partial fills can make the actual portfolio temporarily non-neutral,
which is represented rather than retrospectively repaired.

For each disjoint block h, the risk constraint uses

```
|| [F_h @ w_h, sum_i gap_sigma[h,i] * abs(w[h,i])] ||_2 <= risk_radius
```

`F_h` is the ordinary covariance factor, scaled by the square root of the
ordinary-open-return count divided by the full block width. This partial-session
scaling is an explicit homogeneous ordinary-risk approximation, not a claim that
serially correlated returns scale exactly with time.

The gap term is a worst-correlation event-risk bound, so sparse/asynchronous
reopening observations do not grant unjustified overnight diversification.
A reopening beyond the finite 128-minute horizon is also charged to a terminal
position locked across the closure.

Bootstrap gap-volatility floors are **assumptions**: 0.5% for short and 2% for long
closures. They are configurable example constants, not estimated guarantees.
A controllable plan uses 90% of the hard risk radius to reserve room for next-row
risk/calendar revisions; unavoidable carry uses the full radius. This reserve
is not a proof of feasibility under arbitrary shocks or unexpected halts.

Unsuccessful sequential solves must not supply executable weights. The native
feedback node accepts Solved/AlmostSolved, otherwise preserves the first failure
status and emits NaNs thereafter without advancing feedback. `_run` rejects that
trajectory. Independent programs without feedback retain their prior status-
inspection behavior. There is deliberately no silently invented liquidation or
zero-position fallback when trading is impossible.

## Numerical verification

The committed tests cover the following independent checks:

- Native features exactly equal the supplied scratch expression (9 instruments).
- Independent Pandas position/weight state and canonical trailing products;
  weighted normal-equation beta oracle, including all-zero weight rows.
- Beta-one equality for IC and IC1 at every configured horizon.
- Known coefficients `(0.17, -0.09)` recovered across 100-fold cross-asset
  volatility differences, regime changes and missing labels (tolerance 1e-11).
- First valid Ridge observation after a 100-row outage; active native and JAX
  implementations use the same observation-time update.
- Brute-force calendar endpoint counts versus the compiled closed-form counts.
- Delayed actual fills, partial/NaN masks, invalid quotes, gross/net PnL, and
  large reopening marks.
- Learned beta and estimated volatility through a 2,880-minute closure, rather
  than only constant-beta/constant-volatility ablations. Future returns, masks
  and spreads are perturbed; all earlier outputs must remain bitwise identical.
- Real `RollRets`/POV entrypoint fed simulated price/volume/session fields.
- One generated temporal loop and one native solver stage, failed-feedback
  quarantine, native allocation/persistence tests, and expression-fusion tests.

The regular functional CI uses `-O1` to reduce compilation overhead; focused
three-instrument execution has also been run with default cpp_stream flags.
These are correctness checks, not production-throughput benchmarks.

## Reproducing and inspecting

With repository dependencies and native Clarabel configured:

```sh
python -m pytest -n 0 tests/examples/test_cpp_stream_mpo_diagnostics.py
python -m pytest -n 0 \
  tests/trading_dsl_engine/cpp_stream/test_cat_ridge.py \
  tests/trading_dsl_engine/cpp_stream/test_ridge_projections.py \
  tests/trading_dsl_engine/cpp_stream/test_ridge_recompute.py \
  tests/trading_dsl_engine/cpp_stream/test_operator_fusion.py \
  tests/trading_dsl_engine/ir/test_common_subexpressions.py
python -m pytest -n 0 \
  tests/trading_dsl_engine/cpp_stream/test_cvxpy_program_native.py \
  tests/trading_dsl_engine/cpp_stream/test_cvxpy_constraint_values_and_guard.py
TRADING_DSL_ENGINE_DISABLE_CPP_ACCEL=1 python -m pytest -n 0 \
  tests/trading_dsl_engine/jax_flat/test_ridge_bspline.py -m 'not perf'
```

The last file includes an explicit native-versus-pure-JAX outage test which
controls its own accelerator setting. The targeted native one-feature Ridge regression is also covered separately.

Useful outputs are `features`, `volatility`, `scaled_features`, `fit`, `yhat`,
`expected_returns`, `planned_weights`, `weights`, `execution_allowed`, `gap_event`,
`gap_sigma`, `planned_gap_sigma`, `status`, and gross/cost/net PnL.
`_formula` and `_run` accept `features`, `fit_weights`, `volatility`, and
`beta_override` for controlled ablations.

Example for a caller's own liquidity expression:

```python
formula = _formula(
    features=(my_alpha,),
    fit_weights=my_liquidity_weights,
    beta_override=1.0,
)
```

`ridge_pool_capacity_share()` was not present in the pinned repository source.
The example defaults to finite positive inverse-square-spread weights and permits
an explicit replacement. Matching a private scratch run requires supplying the
same available liquidity expression and the same input arrays.

## Scope and limitations

No private historical InputData files were available in this environment. The
synthetic fixtures exercise both direct-return and real RollRets entrypoints;
they do not establish profitability or parity with an unseen historical run.
The full repository suite was not certified. Two additional automatic
JAX-native object/fallback tests failed under that path in this environment (root Ridge object output
and grouped feature-vector fallback); pure-JAX tests and the directly changed
native Ridge behavior were verified separately. They were not papered over with
skips or unrelated implementation changes.
