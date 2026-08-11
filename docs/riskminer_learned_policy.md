# Risk-seeking learned policy

The learned policy is fully integrated with the reward-dense MCTS pipeline.

## Architecture

- learned token embeddings and a learned `BEG` embedding;
- four GRU layers with hidden width 64;
- two 32-unit MLP hidden layers;
- one logit per typed-RPN token;
- legal-action masking before softmax.

The resulting probability distribution is used as both the PUCT prior and the
rollout policy.

## Training

Every complete episode records token states, selected actions, legal-action sets,
intermediate rewards, terminal reward, and whether the pool changed. A bounded
replay buffer is created fresh for each outer mining iteration.

The reward quantile is updated in trajectory order. Policy batches retain the
pre-update `q_i` associated with each trajectory, then suppress the probability
of trajectories at or below that threshold using the paper's risk-seeking
gradient direction. Invalid dead-end episodes remain in replay with a finite
negative reward instead of being silently discarded.

For a new run, the output bias is initialized from the typed token priors. The
first MCTS cycle therefore starts from a sensible schema distribution rather
than arbitrary random token preferences; subsequent updates remain fully
state-dependent through the GRU.

Policy checkpoints contain the GRU/MLP parameters, configuration, iteration,
quantile, and trajectory count. They can be restored with:

```python
policy, metadata = JaxGRUPolicy.load(path)
```

The plug-and-play runner accepts a checkpoint through
`RISKMINER_RESUME_POLICY`.

All alpha values and rewards are evaluated by `cpp_stream`. JAX is used only for
policy inference and gradient updates.
