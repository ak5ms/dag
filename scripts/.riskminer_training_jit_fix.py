from pathlib import Path

p = Path('src/flows/riskminer/learned_policy.py')
s = p.read_text()
if '    max_batch_size: int = 0\n' not in s:
    s = s.replace(
        '    max_sequence_length: int = 0\n    learning_rate: float = 1.0e-3\n',
        '    max_sequence_length: int = 0\n    max_batch_size: int = 0\n    learning_rate: float = 1.0e-3\n',
        1,
    )
    s = s.replace(
        '        if int(self.max_sequence_length) < 0:\n            raise ValueError("max_sequence_length must be nonnegative")\n',
        '        if int(self.max_sequence_length) < 0:\n            raise ValueError("max_sequence_length must be nonnegative")\n        if int(self.max_batch_size) < 0:\n            raise ValueError("max_batch_size must be nonnegative")\n',
        1,
    )

start = s.index('def _batched_trajectory_log_probabilities(')
end = s.index('\ndef masked_log_prob(', start)
replacement = '''def _trajectory_log_probabilities_from_arrays(\n    params: ArrayTree,\n    config: GRUPolicyConfig,\n    actions: jax.Array,\n    legal_mask: jax.Array,\n    step_mask: jax.Array,\n) -> jax.Array:\n    """Evaluate fixed-shape trajectory arrays with one recurrent scan/layer."""\n\n    batch_size = actions.shape[0]\n    max_steps = actions.shape[1]\n    if max_steps <= 0:\n        return jnp.zeros((batch_size,), dtype=DTYPE)\n\n    root_logits = _root_policy_logits(params, config)\n    state_logits = jnp.broadcast_to(\n        root_logits, (batch_size, 1, config.vocabulary_size)\n    )\n    if max_steps > 1:\n        previous_actions = actions[:, :-1]\n        recurrent_active = step_mask[:, :-1]\n        hidden = params["embedding"][previous_actions]\n        for layer in params["gru"]:\n            hidden = _gru_sequence_batched(layer, hidden, recurrent_active)\n        prefix_logits = _policy_head(params, hidden)\n        state_logits = jnp.concatenate((state_logits, prefix_logits), axis=1)\n\n    masked_logits = jnp.where(legal_mask, state_logits, -jnp.inf)\n    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)\n    gathered = jnp.take_along_axis(\n        log_probs, actions[..., None], axis=-1\n    )[..., 0]\n    return jnp.sum(jnp.where(step_mask, gathered, 0.0), axis=1)\n\n\ndef _pack_trajectory_batch(\n    batch: TrajectoryBatch,\n    config: GRUPolicyConfig,\n    reward_quantile: float,\n) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n    """Pack trajectories to one stable batch/token shape for JAX compilation."""\n\n    values = tuple(batch.trajectories)\n    for trajectory in values:\n        _validate_trajectory_prefixes(trajectory)\n    observed_batch = len(values)\n    batch_size = (\n        int(config.max_batch_size)\n        if int(config.max_batch_size) > 0\n        else observed_batch\n    )\n    if observed_batch > batch_size:\n        raise ValueError(\n            "trajectory batch exceeds configured max_batch_size: "\n            f"observed={observed_batch}, configured={batch_size}"\n        )\n    observed_max_steps = max((len(t.actions) for t in values), default=0)\n    max_steps = (\n        int(config.max_sequence_length)\n        if int(config.max_sequence_length) > 0\n        else observed_max_steps\n    )\n    if observed_max_steps > max_steps:\n        raise ValueError(\n            "trajectory exceeds configured max_sequence_length: "\n            f"observed={observed_max_steps}, configured={max_steps}"\n        )\n    if max_steps <= 0:\n        max_steps = 1\n\n    actions = np.zeros((batch_size, max_steps), dtype=np.int32)\n    legal_mask = np.zeros(\n        (batch_size, max_steps, config.vocabulary_size), dtype=np.bool_\n    )\n    legal_mask[:, :, 0] = True\n    step_mask = np.zeros((batch_size, max_steps), dtype=np.bool_)\n    selected_mask = np.zeros((batch_size,), dtype=np.bool_)\n    for row, trajectory in enumerate(values):\n        steps = len(trajectory.actions)\n        if steps:\n            actions[row, :steps] = np.asarray(trajectory.actions, dtype=np.int32)\n            step_mask[row, :steps] = True\n            for column, (action, legal) in enumerate(\n                zip(trajectory.actions, trajectory.legal_actions)\n            ):\n                if action not in legal:\n                    raise ValueError(f"action {action} is not legal")\n                legal_mask[row, column, :] = False\n                legal_mask[row, column, np.asarray(legal, dtype=np.int64)] = True\n        threshold = (\n            float(batch.reward_quantiles[row])\n            if batch.reward_quantiles\n            else float(reward_quantile)\n        )\n        selected_mask[row] = float(trajectory.reward) <= threshold\n    return actions, legal_mask, step_mask, selected_mask\n\n\ndef _risk_seeking_loss_arrays(\n    params: ArrayTree,\n    config: GRUPolicyConfig,\n    actions: jax.Array,\n    legal_mask: jax.Array,\n    step_mask: jax.Array,\n    selected_mask: jax.Array,\n) -> jax.Array:\n    log_probability = _trajectory_log_probabilities_from_arrays(\n        params, config, actions, legal_mask, step_mask\n    )\n    selected = selected_mask.astype(DTYPE)\n    count = jnp.sum(selected)\n    total = jnp.sum(log_probability * selected)\n    return jnp.where(\n        count > 0.0, total / count, jnp.asarray(0.0, dtype=DTYPE)\n    )\n\n\n_RISK_LOSS_AND_GRAD = jax.jit(\n    jax.value_and_grad(_risk_seeking_loss_arrays),\n    static_argnames=("config",),\n)\n\n\ndef _batched_trajectory_log_probabilities(\n    params: ArrayTree,\n    config: GRUPolicyConfig,\n    trajectories: Sequence[PolicyTrajectory],\n) -> jax.Array:\n    values = tuple(trajectories)\n    if not values:\n        return jnp.zeros((0,), dtype=DTYPE)\n    batch = TrajectoryBatch(values)\n    actions, legal_mask, step_mask, _ = _pack_trajectory_batch(\n        batch, config, reward_quantile=0.0\n    )\n    probabilities = _trajectory_log_probabilities_from_arrays(\n        params,\n        config,\n        jnp.asarray(actions),\n        jnp.asarray(legal_mask),\n        jnp.asarray(step_mask),\n    )\n    return probabilities[: len(values)]\n\n\n'''
s = s[:start] + replacement + s[end:]

start = s.index('def risk_seeking_loss(')
end = s.index('\n\n@dataclass(frozen=True)\nclass JaxGRUPolicy', start)
replacement = '''def risk_seeking_loss(\n    params: ArrayTree,\n    config: GRUPolicyConfig,\n    batch: TrajectoryBatch,\n    reward_quantile: float,\n) -> jax.Array:\n    """Equations 12/13 expressed as a gradient-descent loss."""\n\n    actions, legal_mask, step_mask, selected_mask = _pack_trajectory_batch(\n        batch, config, reward_quantile\n    )\n    return _risk_seeking_loss_arrays(\n        params,\n        config,\n        jnp.asarray(actions),\n        jnp.asarray(legal_mask),\n        jnp.asarray(step_mask),\n        jnp.asarray(selected_mask),\n    )\n'''
s = s[:start] + replacement + s[end:]

old = '''        loss_fn = lambda params: risk_seeking_loss(\n            params, self.config, batch, reward_quantile\n        )\n        loss, gradients = jax.value_and_grad(loss_fn)(self.params)\n'''
new = '''        actions, legal_mask, step_mask, selected_mask = _pack_trajectory_batch(\n            batch, self.config, reward_quantile\n        )\n        loss, gradients = _RISK_LOSS_AND_GRAD(\n            self.params,\n            self.config,\n            jnp.asarray(actions),\n            jnp.asarray(legal_mask),\n            jnp.asarray(step_mask),\n            jnp.asarray(selected_mask),\n        )\n'''
if old not in s:
    raise SystemExit('train_step block not found')
s = s.replace(old, new, 1)
p.write_text(s)

p = Path('src/flows/riskminer/trainer.py')
s = p.read_text()
if 'max_batch_size=config.policy_batch_size' not in s:
    s = s.replace(
        '                max_sequence_length=config.max_tokens,\n                learning_rate=config.policy_learning_rate,\n',
        '                max_sequence_length=config.max_tokens,\n                max_batch_size=config.policy_batch_size,\n                learning_rate=config.policy_learning_rate,\n',
        1,
    )
p.write_text(s)
