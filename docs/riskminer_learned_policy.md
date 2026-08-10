# Risk-seeking learned policy checkpoint

`flows.riskminer.learned_policy` implements the learned-policy portion of the
RiskMiner design without changing the native alpha evaluator.

## Architecture

- learned token embedding plus a learned `BEG` embedding;
- configurable stacked GRU, default four layers and hidden size 64;
- two 32-unit MLP hidden layers;
- one logit per typed-RPN token;
- exact legal-action masking before softmax;
- `ActionPolicy.priors(...)` compatibility with `RiskMCTS`;
- stochastic reward-CDF quantile tracking;
- below-quantile trajectory probability suppression with manual SGD through JAX.

Candidate formulas and rewards continue to be computed exclusively by
`trading_dsl_engine.cpp_stream`; JAX is used only for token-policy inference and
training.

## Use the initialized policy in native search

```python
from flows.riskminer.config import RiskMinerConfig
from flows.riskminer.policy_search import search_cpp_stream_alphas_with_policy

result, policy = search_cpp_stream_alphas_with_policy(
    sources,
    n_instruments=9,
    work_dir="/tmp/riskminer-policy-search",
    config=RiskMinerConfig(
        max_depth=8,
        simulations=128,
        rollouts_per_expansion=8,
        evaluation_batch_size=32,
        archive_size=100,
        seed=42,
    ),
)
```

## Train one risk-seeking step

```python
from flows.riskminer.learned_policy import (
    PolicyTrajectory,
    RiskQuantileTracker,
    TrajectoryBatch,
)

tracker = RiskQuantileTracker(
    cdf_quantile=0.80,
    learning_rate=0.01,
)
tracker = tracker.update_many(episode_rewards)

batch = TrajectoryBatch(tuple(policy_trajectories))
policy, loss = policy.train_step(
    batch,
    reward_quantile=tracker.value,
)
```

A `PolicyTrajectory` stores the token prefix, chosen action, exact legal-action set,
and final reward at each decision. Gradient descent on the summed log probability
of trajectories at or below the tracked quantile decreases their probability.

## Current integration boundary

The GRU can already supply MCTS priors and its update is independently tested.
The next checkpoint will make `RiskMCTS` return a replay-ready trajectory record and
add the alternating loop:

1. freeze the current Ridge pool;
2. run a native-evaluated MCTS mining cycle;
3. evaluate exact candidate additions to the pool;
4. update the reward quantile;
5. train the GRU from the collected trajectories;
6. reset the tree for the changed pool and repeat.

This keeps all values within one MCTS tree conditional on one fixed pool snapshot.
