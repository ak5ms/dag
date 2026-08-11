from pathlib import Path

p = Path('src/flows/riskminer/learned_policy.py')
s = p.read_text()
if 'max_sequence_length: int = 0' not in s:
    s = s.replace(
        '    mlp_hidden_2: int = 32\n    learning_rate: float = 1.0e-3\n',
        '    mlp_hidden_2: int = 32\n    max_sequence_length: int = 0\n    learning_rate: float = 1.0e-3\n', 1)
    s = s.replace(
        '        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:\n',
        '        if int(self.max_sequence_length) < 0:\n            raise ValueError("max_sequence_length must be nonnegative")\n        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:\n', 1)
if 'def _policy_logits_numpy(' not in s:
    marker = '\ndef _validate_trajectory_prefixes(trajectory: PolicyTrajectory) -> None:\n'
    helper = '''

def _numpy_sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    out = np.empty_like(value)
    positive = value >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    negative_exp = np.exp(value[~positive])
    out[~positive] = negative_exp / (1.0 + negative_exp)
    return out


def _policy_logits_numpy(params: ArrayTree, config: GRUPolicyConfig, token_ids: Sequence[int]) -> np.ndarray:
    host = jax.tree_util.tree_map(np.asarray, params)
    ids = tuple(int(token_id) for token_id in token_ids)
    if not ids:
        ids = (config.vocabulary_size,)
    values = host["embedding"][np.asarray(ids, dtype=np.int32)]
    for layer in host["gru"]:
        hidden = np.zeros((config.hidden_size,), dtype=np.float32)
        outputs = np.empty((len(ids), config.hidden_size), dtype=np.float32)
        for index, value in enumerate(values):
            update = _numpy_sigmoid(value @ layer["w_z"] + hidden @ layer["u_z"] + layer["b_z"])
            reset = _numpy_sigmoid(value @ layer["w_r"] + hidden @ layer["u_r"] + layer["b_r"])
            candidate = np.tanh(value @ layer["w_n"] + (reset * hidden) @ layer["u_n"] + layer["b_n"])
            hidden = (1.0 - update) * candidate + update * hidden
            outputs[index] = hidden
        values = outputs
    hidden = values[-1]
    hidden = np.tanh(hidden @ host["mlp_1"]["w"] + host["mlp_1"]["b"])
    hidden = np.tanh(hidden @ host["mlp_2"]["w"] + host["mlp_2"]["b"])
    return hidden @ host["out"]["w"] + host["out"]["b"]
'''
    if marker not in s:
        raise SystemExit('trajectory marker not found')
    s = s.replace(marker, helper + marker, 1)
old = '        logits = np.asarray(\n            policy_logits(self.params, self.config, state.token_ids),\n            dtype=np.float64,\n        )\n'
new = '        logits = np.asarray(\n            _policy_logits_numpy(self.params, self.config, state.token_ids),\n            dtype=np.float64,\n        )\n'
if old in s:
    s = s.replace(old, new, 1)
elif '_policy_logits_numpy(self.params, self.config, state.token_ids)' not in s:
    raise SystemExit('priors block not found')
old = '    lengths = np.asarray([len(t.actions) for t in values], dtype=np.int32)\n    max_steps = int(lengths.max(initial=0))\n    if max_steps <= 0:\n        return jnp.zeros((batch_size,), dtype=DTYPE)\n'
new = '    lengths = np.asarray([len(t.actions) for t in values], dtype=np.int32)\n    observed_max_steps = int(lengths.max(initial=0))\n    max_steps = (\n        int(config.max_sequence_length)\n        if int(config.max_sequence_length) > 0\n        else observed_max_steps\n    )\n    if observed_max_steps > max_steps:\n        raise ValueError(\n            "trajectory exceeds configured max_sequence_length: "\n            f"observed={observed_max_steps}, configured={max_steps}"\n        )\n    if max_steps <= 0:\n        return jnp.zeros((batch_size,), dtype=DTYPE)\n'
if old in s:
    s = s.replace(old, new, 1)
p.write_text(s)

p = Path('src/flows/riskminer/trainer.py')
s = p.read_text()
old = '                vocabulary_size=int(vocabulary_size),\n                learning_rate=config.policy_learning_rate,\n                seed=config.seed,\n'
new = '                vocabulary_size=int(vocabulary_size),\n                max_sequence_length=config.max_tokens,\n                learning_rate=config.policy_learning_rate,\n                seed=config.seed,\n'
if old in s:
    s = s.replace(old, new, 1)
elif 'max_sequence_length=config.max_tokens' not in s:
    raise SystemExit('trainer block not found')
p.write_text(s)
