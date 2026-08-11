# Complete RiskMiner pipeline on `cpp_stream`

The current implementation follows the RiskMiner search/training structure while
using the repository's requested Sharpe and online-Ridge pool objectives.

## End-to-end loop

Each outer mining iteration performs:

1. Create a fresh MCTS tree and replay buffer.
2. Use the persistent GRU policy for both PUCT priors and rollout sampling.
3. Assign an intermediate reward whenever a non-terminal RPN state is a valid
   dimensionless alpha.
4. On `END`, test the alpha in the persistent root-level Ridge pool and use the
   resulting validation pool Sharpe as the terminal reward.
5. Back up path-specific cumulative rewards, retaining `N`, `P`, `Q`, and the
   immediate reward `R` on every selected edge.
6. Update the reward quantile in trajectory order.
7. Train the GRU with the below-quantile risk-seeking policy objective.
8. Save a policy checkpoint and begin the next iteration with a fresh tree but
   the updated policy and alpha pool.

The pool has configurable capacity, default 100. When a candidate would exceed
capacity, the evaluator calculates the mean absolute online-Ridge coefficient for
each alpha over the validation period and proposes removal of the smallest one.
The replacement is committed only when the resulting pool Sharpe exceeds the
configured admission threshold.

## User-specific intermediate reward

The paper's IC-minus-correlation reward is replaced by the requested raw
cross-sectional orthogonalization:

```python
model = Ridge(
    *existing_pool,
    y=candidate,
    weights=1.0,
    hl=0.0,
    lambda_=0.0,
    nonneg=False,
)
orthogonal_alpha = get_residuals(model)
```

Neither the pool alphas nor the candidate are volatility-scaled. The residual is
then scored using:

```python
pnl = shift(orthogonal_alpha, 1, 1) * roll_rets
pnl = pnl.sum(axis=1)
reward = pnl.mean() / pnl.std(ddof=0)
```

The native Ridge solver uses its pseudo-inverse fallback when the cross-sectional
design is singular or has more columns than instruments.

### Pool size greater than instrument count

The pseudo-inverse makes the regression mathematically well-defined for
`K > N`, where `K` is pool size and `N` is instrument count. It does not preserve
a residual direction once the rowwise design reaches full instrument rank:

```text
rank(X_t) = N  =>  column_space(X_t) = R^N  =>  residual_t = 0
```

Therefore, with nine instruments, a sufficiently diverse pool of nine or more
raw alpha vectors can make every candidate's orthogonalized intermediate reward
zero. A larger pool can still leave a nonzero residual when its rowwise columns
are rank-deficient. The runtime emits `orthogonal_rank_saturation` once `K >= N`
so this condition is visible in logs.

## Neural policy

`JaxGRUPolicy` contains:

- one learned embedding per RPN token plus a learned `BEG` embedding;
- four GRU layers with hidden width 64;
- two 32-unit MLP layers;
- one output logit per vocabulary token;
- exact masking of illegal actions before softmax.

The policy output is used in both places required by MCTS:

```text
PUCT prior P(s,a)
rollout token-sampling probability
```

The quantile recursion is applied in trajectory order. Each replay item retains
the pre-update `q_i` threshold used for that trajectory, so batched policy
updates preserve the same below-quantile indicator as the coupled sequential
recursion. Dead-end and max-length episodes are retained with the configured
invalid reward, allowing the policy to learn to suppress them.

The initial neural output bias is set from the typed schema priors. This makes
the first small-budget search favor valid market terminals over literals while
remaining fully trainable and state-dependent after policy updates.

## Reward-dense backup

For a selected path with immediate rewards `r_1, ..., r_T`, the implementation
uses `discount=1` by default and calculates:

```text
G_k = r_k + r_(k+1) + ... + r_T
```

Each selected edge receives its own `G_k`; the same terminal value is not copied
to every edge. Rollout-only actions are retained in replay trajectories but are
not inserted into the permanent search tree.

## Operator grammar

The catalog contains the paper's operators:

```text
sign, abs, log, cross-sectional rank
add, subtract, multiply, divide, greater, less
shift/reference
rolling rank, skew, kurtosis, mean, median, sum
rolling standard deviation, variance, max, min
weighted moving average, exponential moving average
rolling covariance and correlation
```

It also retains native-safe helpers such as `purify`, `fillna`, `where`,
`minimum`, `maximum`, and percentile rank.

Dynamic temporal inputs are represented by an expression-valued selector over a
compile-time bank of native states (5, 60, and 1440 rows by default). For example, `dynamic_ewm(x, selector)`
constructs the configured static spans and chooses among them at each row and
instrument. This gives true expression-dependent output while avoiding a
runtime-sized native history buffer. More branches can be supplied through
`default_operator_catalog(dynamic_periods=...)`, but large banks materially increase
C++ template compile cost, especially for composed rolling correlation.

## Data separation

The generic pipeline and InputData runner use chronological, non-overlapping:

```text
training:   orthogonalized intermediate rewards
validation: exact terminal Ridge-pool score and admission
 test:      one final evaluation of the frozen pool
```

Derived return and volatility arrays are calculated causally before slicing, so
validation/test rows can inherit prior history without seeing future rows.
