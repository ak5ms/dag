from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import pickle
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .rpn import RPNState, TypedRPNEnvironment

ArrayTree = Any
DTYPE = jnp.float32


@dataclass(frozen=True)
class GRUPolicyConfig:
    vocabulary_size: int
    embedding_dim: int = 32
    hidden_size: int = 64
    layers: int = 4
    mlp_hidden_1: int = 32
    mlp_hidden_2: int = 32
    max_sequence_length: int = 0
    learning_rate: float = 1.0e-3
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "vocabulary_size", "embedding_dim", "hidden_size", "layers",
            "mlp_hidden_1", "mlp_hidden_2",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.max_sequence_length) < 0:
            raise ValueError("max_sequence_length must be nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")


@dataclass(frozen=True)
class PolicyTrajectory:
    """One complete BEG-to-END episode used by risk-seeking training."""

    states: tuple[tuple[int, ...], ...]
    actions: tuple[int, ...]
    legal_actions: tuple[tuple[int, ...], ...]
    reward: float
    step_rewards: tuple[float, ...] = ()
    terminal_formula_key: tuple | None = None
    terminal_formula_rpn: str | None = None
    pool_changed: bool = False

    def __post_init__(self) -> None:
        if not (
            len(self.states) == len(self.actions) == len(self.legal_actions)
        ):
            raise ValueError("trajectory state/action/legal lengths must match")
        if self.step_rewards and len(self.step_rewards) != len(self.actions):
            raise ValueError("step_rewards must be empty or match action count")
        if not math.isfinite(float(self.reward)):
            raise ValueError("trajectory reward must be finite")
        for action, legal in zip(self.actions, self.legal_actions):
            if action not in legal:
                raise ValueError(f"chosen action {action} is not legal")


@dataclass(frozen=True)
class TrajectoryBatch:
    trajectories: tuple[PolicyTrajectory, ...]
    reward_quantiles: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise ValueError("trajectory batch cannot be empty")
        if self.reward_quantiles and (
            len(self.reward_quantiles) != len(self.trajectories)
        ):
            raise ValueError(
                "reward_quantiles must be empty or match trajectory count"
            )
        if any(not math.isfinite(float(value)) for value in self.reward_quantiles):
            raise ValueError("reward_quantiles must be finite")


@dataclass(frozen=True)
class RiskQuantileTracker:
    """Equation-11 stochastic recursion for a reward CDF quantile."""

    cdf_quantile: float = 0.80
    learning_rate: float = 0.01
    value: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.cdf_quantile < 1.0:
            raise ValueError("cdf_quantile must be in (0, 1)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.value):
            raise ValueError("quantile value must be finite")

    def update(self, reward: float) -> "RiskQuantileTracker":
        reward = float(reward)
        if not math.isfinite(reward):
            return self
        indicator = 1.0 if reward <= self.value else 0.0
        return replace(
            self,
            value=float(
                self.value
                + self.learning_rate * (self.cdf_quantile - indicator)
            ),
        )

    def update_many(self, rewards: Sequence[float]) -> "RiskQuantileTracker":
        tracker = self
        for reward in rewards:
            tracker = tracker.update(float(reward))
        return tracker


def _normal(key, shape, fan_in: int) -> jax.Array:
    scale = 1.0 / math.sqrt(max(1, fan_in))
    return scale * jax.random.normal(key, shape, dtype=DTYPE)


def initialize_gru_policy(
    config: GRUPolicyConfig,
    token_priors: Sequence[float] | None = None,
) -> ArrayTree:
    key = jax.random.PRNGKey(config.seed)
    key_count = 1 + 6 * config.layers + 3
    keys = iter(jax.random.split(key, key_count))
    embedding = _normal(
        next(keys),
        (config.vocabulary_size + 1, config.embedding_dim),
        config.embedding_dim,
    )
    gru_layers = []
    input_size = config.embedding_dim
    for _ in range(config.layers):
        layer = {}
        for gate in ("z", "r", "n"):
            layer[f"w_{gate}"] = _normal(
                next(keys), (input_size, config.hidden_size), input_size
            )
            layer[f"u_{gate}"] = _normal(
                next(keys),
                (config.hidden_size, config.hidden_size),
                config.hidden_size,
            )
            layer[f"b_{gate}"] = jnp.zeros(
                (config.hidden_size,), dtype=DTYPE
            )
        gru_layers.append(layer)
        input_size = config.hidden_size
    params = {
        "embedding": embedding,
        "gru": tuple(gru_layers),
        "mlp_1": {
            "w": _normal(
                next(keys),
                (config.hidden_size, config.mlp_hidden_1),
                config.hidden_size,
            ),
            "b": jnp.zeros((config.mlp_hidden_1,), dtype=DTYPE),
        },
        "mlp_2": {
            "w": _normal(
                next(keys),
                (config.mlp_hidden_1, config.mlp_hidden_2),
                config.mlp_hidden_1,
            ),
            "b": jnp.zeros((config.mlp_hidden_2,), dtype=DTYPE),
        },
        "out": {
            "w": _normal(
                next(keys),
                (config.mlp_hidden_2, config.vocabulary_size),
                config.mlp_hidden_2,
            ),
            "b": jnp.zeros((config.vocabulary_size,), dtype=DTYPE),
        },
    }
    if token_priors is not None:
        values = np.asarray(tuple(token_priors), dtype=np.float64)
        if values.shape != (config.vocabulary_size,):
            raise ValueError(
                "token_priors must have one value per vocabulary token"
            )
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("token_priors must be finite and nonnegative")
        values = np.maximum(values, 1.0e-12)
        values /= values.sum()
        params["out"]["w"] = 0.01 * params["out"]["w"]
        params["out"]["b"] = jnp.asarray(np.log(values), dtype=DTYPE)
    return params


def _gru_step(
    layer: Mapping[str, jax.Array],
    hidden: jax.Array,
    value: jax.Array,
) -> jax.Array:
    update = jax.nn.sigmoid(
        value @ layer["w_z"] + hidden @ layer["u_z"] + layer["b_z"]
    )
    reset = jax.nn.sigmoid(
        value @ layer["w_r"] + hidden @ layer["u_r"] + layer["b_r"]
    )
    candidate = jnp.tanh(
        value @ layer["w_n"]
        + (reset * hidden) @ layer["u_n"]
        + layer["b_n"]
    )
    return (1.0 - update) * candidate + update * hidden


def _gru_sequence(layer: Mapping[str, jax.Array], inputs: jax.Array) -> jax.Array:
    hidden_size = int(layer["b_z"].shape[0])

    def step(hidden, value):
        next_hidden = _gru_step(layer, hidden, value)
        return next_hidden, next_hidden

    initial = jnp.zeros((hidden_size,), dtype=DTYPE)
    _, outputs = jax.lax.scan(step, initial, inputs)
    return outputs


def _gru_sequence_batched(
    layer: Mapping[str, jax.Array],
    inputs: jax.Array,
    active: jax.Array,
) -> jax.Array:
    """Run one GRU layer for a padded batch of token sequences."""

    batch_size = int(inputs.shape[0])
    hidden_size = int(layer["b_z"].shape[0])

    def step(hidden, values):
        value, is_active = values
        proposed = _gru_step(layer, hidden, value)
        next_hidden = jnp.where(is_active[:, None], proposed, hidden)
        return next_hidden, next_hidden

    initial = jnp.zeros((batch_size, hidden_size), dtype=DTYPE)
    time_inputs = jnp.swapaxes(inputs, 0, 1)
    time_active = jnp.swapaxes(active, 0, 1)
    _, outputs = jax.lax.scan(step, initial, (time_inputs, time_active))
    return jnp.swapaxes(outputs, 0, 1)


def _policy_head(params: ArrayTree, hidden: jax.Array) -> jax.Array:
    """Apply the policy MLP to one or many GRU hidden states."""

    hidden = jnp.tanh(hidden @ params["mlp_1"]["w"] + params["mlp_1"]["b"])
    hidden = jnp.tanh(hidden @ params["mlp_2"]["w"] + params["mlp_2"]["b"])
    return hidden @ params["out"]["w"] + params["out"]["b"]


def policy_logits(
    params: ArrayTree,
    config: GRUPolicyConfig,
    token_ids: Sequence[int],
) -> jax.Array:
    ids = tuple(int(token_id) for token_id in token_ids)
    if not ids:
        ids = (config.vocabulary_size,)
    tokens = jnp.asarray(ids, dtype=jnp.int32)
    values = params["embedding"][tokens]
    for layer in params["gru"]:
        values = _gru_sequence(layer, values)
    return _policy_head(params, values[-1])



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

def _validate_trajectory_prefixes(trajectory: PolicyTrajectory) -> None:
    actions = tuple(int(action) for action in trajectory.actions)
    for index, state in enumerate(trajectory.states):
        expected = actions[:index]
        if tuple(int(token) for token in state) != expected:
            raise ValueError(
                "policy trajectory states must be action prefixes; "
                f"state {index}={state!r}, expected {expected!r}"
            )


def _root_policy_logits(
    params: ArrayTree,
    config: GRUPolicyConfig,
) -> jax.Array:
    """Evaluate the learned BEG state without constructing length-1 scans."""

    value = params["embedding"][config.vocabulary_size]
    for layer in params["gru"]:
        hidden = jnp.zeros((int(layer["b_z"].shape[0]),), dtype=DTYPE)
        value = _gru_step(layer, hidden, value)
    return _policy_head(params, value)


def _batched_trajectory_log_probabilities(
    params: ArrayTree,
    config: GRUPolicyConfig,
    trajectories: Sequence[PolicyTrajectory],
) -> jax.Array:
    """Compute complete trajectory log-probabilities with one scan per layer.

    The original implementation re-ran the entire stacked GRU separately for
    every RPN prefix.  That makes autodiff materialize O(B*T*L) scans for batch
    size B, trajectory length T, and L GRU layers.  This function pads the
    selected trajectories and runs all prefixes together, reducing the traced
    recurrent graph to O(L) scans.
    """

    values = tuple(trajectories)
    if not values:
        return jnp.zeros((0,), dtype=DTYPE)
    for trajectory in values:
        _validate_trajectory_prefixes(trajectory)

    batch_size = len(values)
    lengths = np.asarray([len(t.actions) for t in values], dtype=np.int32)
    observed_max_steps = int(lengths.max(initial=0))
    max_steps = (
        int(config.max_sequence_length)
        if int(config.max_sequence_length) > 0
        else observed_max_steps
    )
    if observed_max_steps > max_steps:
        raise ValueError(
            "trajectory exceeds configured max_sequence_length: "
            f"observed={observed_max_steps}, configured={max_steps}"
        )
    if max_steps <= 0:
        return jnp.zeros((batch_size,), dtype=DTYPE)

    actions = np.zeros((batch_size, max_steps), dtype=np.int32)
    legal_mask = np.zeros(
        (batch_size, max_steps, config.vocabulary_size), dtype=np.bool_
    )
    step_mask = np.zeros((batch_size, max_steps), dtype=np.bool_)
    for row, trajectory in enumerate(values):
        steps = len(trajectory.actions)
        if steps == 0:
            legal_mask[row, :, 0] = True
            continue
        actions[row, :steps] = np.asarray(trajectory.actions, dtype=np.int32)
        step_mask[row, :steps] = True
        for column, (action, legal) in enumerate(
            zip(trajectory.actions, trajectory.legal_actions)
        ):
            if action not in legal:
                raise ValueError(f"action {action} is not legal")
            legal_mask[row, column, np.asarray(legal, dtype=np.int64)] = True
        if steps < max_steps:
            legal_mask[row, steps:, 0] = True

    root_logits = _root_policy_logits(params, config)
    state_logits = jnp.broadcast_to(
        root_logits, (batch_size, 1, config.vocabulary_size)
    )

    if max_steps > 1:
        previous_actions = jnp.asarray(actions[:, :-1], dtype=jnp.int32)
        recurrent_active = jnp.asarray(step_mask[:, :-1], dtype=bool)
        hidden = params["embedding"][previous_actions]
        for layer in params["gru"]:
            hidden = _gru_sequence_batched(layer, hidden, recurrent_active)
        prefix_logits = _policy_head(params, hidden)
        state_logits = jnp.concatenate((state_logits, prefix_logits), axis=1)

    mask = jnp.asarray(legal_mask)
    masked_logits = jnp.where(mask, state_logits, -jnp.inf)
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    action_ids = jnp.asarray(actions, dtype=jnp.int32)
    gathered = jnp.take_along_axis(
        log_probs, action_ids[..., None], axis=-1
    )[..., 0]
    return jnp.sum(
        jnp.where(jnp.asarray(step_mask), gathered, 0.0), axis=1
    )


def masked_log_prob(
    params: ArrayTree,
    config: GRUPolicyConfig,
    state_tokens: Sequence[int],
    action: int,
    legal_actions: Sequence[int],
) -> jax.Array:
    if action not in legal_actions:
        raise ValueError(f"action {action} is not legal")
    logits = policy_logits(params, config, state_tokens)
    mask = jnp.zeros((config.vocabulary_size,), dtype=bool)
    mask = mask.at[jnp.asarray(legal_actions, dtype=jnp.int32)].set(True)
    masked = jnp.where(mask, logits, -jnp.inf)
    return jax.nn.log_softmax(masked)[int(action)]


def risk_seeking_loss(
    params: ArrayTree,
    config: GRUPolicyConfig,
    batch: TrajectoryBatch,
    reward_quantile: float,
) -> jax.Array:
    """Equations 12/13 expressed as a gradient-descent loss.

    The paper performs gradient ascent on ``-1{R<=q} sum grad log pi``.
    Minimizing the selected trajectories' summed log-probabilities is exactly
    the same update direction.
    """

    selected: list[PolicyTrajectory] = []
    for index, trajectory in enumerate(batch.trajectories):
        threshold = (
            float(batch.reward_quantiles[index])
            if batch.reward_quantiles
            else float(reward_quantile)
        )
        if float(trajectory.reward) <= threshold:
            selected.append(trajectory)
    if not selected:
        return jnp.asarray(0.0, dtype=DTYPE)
    return jnp.mean(
        _batched_trajectory_log_probabilities(
            params, config, tuple(selected)
        )
    )


@dataclass(frozen=True)
class JaxGRUPolicy:
    config: GRUPolicyConfig
    params: ArrayTree

    @classmethod
    def initialize(
        cls,
        config: GRUPolicyConfig,
        *,
        token_priors: Sequence[float] | None = None,
    ) -> "JaxGRUPolicy":
        return cls(
            config,
            initialize_gru_policy(config, token_priors=token_priors),
        )

    def priors(
        self,
        environment: TypedRPNEnvironment,
        state: RPNState,
        legal_actions: Sequence[int],
    ) -> Mapping[int, float]:
        if not legal_actions:
            return {}
        if len(environment.vocabulary) != self.config.vocabulary_size:
            raise ValueError("policy vocabulary size does not match the RPN environment")
        logits = np.asarray(
            _policy_logits_numpy(self.params, self.config, state.token_ids),
            dtype=np.float64,
        )
        legal = np.asarray(legal_actions, dtype=np.int64)
        selected = logits[legal]
        selected -= np.max(selected)
        weights = np.exp(selected)
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0.0:
            probability = 1.0 / len(legal_actions)
            return {int(action): probability for action in legal_actions}
        probabilities = weights / total
        return {
            int(action): float(probability)
            for action, probability in zip(legal, probabilities)
        }

    def loss(self, batch: TrajectoryBatch, reward_quantile: float) -> float:
        return float(risk_seeking_loss(self.params, self.config, batch, reward_quantile))

    def train_step(
        self,
        batch: TrajectoryBatch,
        reward_quantile: float,
    ) -> tuple["JaxGRUPolicy", float]:
        loss_fn = lambda params: risk_seeking_loss(
            params, self.config, batch, reward_quantile
        )
        loss, gradients = jax.value_and_grad(loss_fn)(self.params)
        updated = jax.tree_util.tree_map(
            lambda parameter, gradient: parameter - self.config.learning_rate * gradient,
            self.params,
            gradients,
        )
        return JaxGRUPolicy(self.config, updated), float(loss)

    def save(self, path: str | Path, **metadata: object) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "params": jax.tree_util.tree_map(np.asarray, self.params),
            "metadata": dict(metadata),
        }
        with destination.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> tuple["JaxGRUPolicy", dict[str, object]]:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        config = GRUPolicyConfig(**payload["config"])
        params = jax.tree_util.tree_map(
            lambda value: jnp.asarray(value, dtype=DTYPE), payload["params"]
        )
        return cls(config, params), dict(payload.get("metadata", {}))


__all__ = [
    "GRUPolicyConfig", "JaxGRUPolicy", "PolicyTrajectory",
    "RiskQuantileTracker", "TrajectoryBatch", "initialize_gru_policy",
    "masked_log_prob", "policy_logits", "risk_seeking_loss",
]
