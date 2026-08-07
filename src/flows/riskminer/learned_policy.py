from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
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
    learning_rate: float = 1.0e-3
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "vocabulary_size",
            "embedding_dim",
            "hidden_size",
            "layers",
            "mlp_hidden_1",
            "mlp_hidden_2",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")


@dataclass(frozen=True)
class PolicyTrajectory:
    states: tuple[tuple[int, ...], ...]
    actions: tuple[int, ...]
    legal_actions: tuple[tuple[int, ...], ...]
    reward: float

    def __post_init__(self) -> None:
        if not (
            len(self.states)
            == len(self.actions)
            == len(self.legal_actions)
        ):
            raise ValueError("trajectory state/action/legal lengths must match")
        for action, legal in zip(self.actions, self.legal_actions):
            if action not in legal:
                raise ValueError(f"chosen action {action} is not legal")


@dataclass(frozen=True)
class TrajectoryBatch:
    trajectories: tuple[PolicyTrajectory, ...]

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise ValueError("trajectory batch cannot be empty")


@dataclass(frozen=True)
class RiskQuantileTracker:
    """Stochastic approximation to a reward CDF quantile."""

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
                + self.learning_rate
                * (self.cdf_quantile - indicator)
            ),
        )

    def update_many(
        self,
        rewards: Sequence[float],
    ) -> "RiskQuantileTracker":
        tracker = self
        for reward in rewards:
            tracker = tracker.update(float(reward))
        return tracker


def _normal(key, shape, fan_in: int) -> jax.Array:
    scale = 1.0 / math.sqrt(max(1, fan_in))
    return scale * jax.random.normal(key, shape, dtype=DTYPE)


def initialize_gru_policy(config: GRUPolicyConfig) -> ArrayTree:
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
                next(keys),
                (input_size, config.hidden_size),
                input_size,
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
    return {
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


def _gru_sequence(layer: Mapping[str, jax.Array], inputs: jax.Array) -> jax.Array:
    hidden_size = int(layer["b_z"].shape[0])

    def step(hidden, value):
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
        next_hidden = (1.0 - update) * candidate + update * hidden
        return next_hidden, next_hidden

    initial = jnp.zeros((hidden_size,), dtype=DTYPE)
    _, outputs = jax.lax.scan(step, initial, inputs)
    return outputs


def policy_logits(
    params: ArrayTree,
    config: GRUPolicyConfig,
    token_ids: Sequence[int],
) -> jax.Array:
    ids = tuple(int(token_id) for token_id in token_ids)
    # The final embedding row is a learned BEG token for the empty root state.
    if not ids:
        ids = (config.vocabulary_size,)
    tokens = jnp.asarray(ids, dtype=jnp.int32)
    values = params["embedding"][tokens]
    for layer in params["gru"]:
        values = _gru_sequence(layer, values)
    hidden = values[-1]
    hidden = jnp.tanh(
        hidden @ params["mlp_1"]["w"] + params["mlp_1"]["b"]
    )
    hidden = jnp.tanh(
        hidden @ params["mlp_2"]["w"] + params["mlp_2"]["b"]
    )
    return hidden @ params["out"]["w"] + params["out"]["b"]


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
    """Decrease probability of trajectories at/below the reward quantile."""

    selected = []
    for trajectory in batch.trajectories:
        if float(trajectory.reward) > float(reward_quantile):
            continue
        log_probability = jnp.asarray(0.0, dtype=DTYPE)
        for state, action, legal in zip(
            trajectory.states,
            trajectory.actions,
            trajectory.legal_actions,
        ):
            log_probability = log_probability + masked_log_prob(
                params,
                config,
                state,
                action,
                legal,
            )
        selected.append(log_probability)
    if not selected:
        return jnp.asarray(0.0, dtype=DTYPE)
    return jnp.mean(jnp.stack(selected))


@dataclass(frozen=True)
class JaxGRUPolicy:
    config: GRUPolicyConfig
    params: ArrayTree

    @classmethod
    def initialize(cls, config: GRUPolicyConfig) -> "JaxGRUPolicy":
        return cls(config, initialize_gru_policy(config))

    def priors(
        self,
        environment: TypedRPNEnvironment,
        state: RPNState,
        legal_actions: Sequence[int],
    ) -> Mapping[int, float]:
        if not legal_actions:
            return {}
        if len(environment.vocabulary) != self.config.vocabulary_size:
            raise ValueError(
                "policy vocabulary size does not match the RPN environment"
            )
        logits = np.asarray(
            policy_logits(self.params, self.config, state.token_ids),
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

    def loss(
        self,
        batch: TrajectoryBatch,
        reward_quantile: float,
    ) -> float:
        return float(
            risk_seeking_loss(
                self.params,
                self.config,
                batch,
                reward_quantile,
            )
        )

    def train_step(
        self,
        batch: TrajectoryBatch,
        reward_quantile: float,
    ) -> tuple["JaxGRUPolicy", float]:
        loss_fn = lambda params: risk_seeking_loss(
            params,
            self.config,
            batch,
            reward_quantile,
        )
        loss, gradients = jax.value_and_grad(loss_fn)(self.params)
        updated = jax.tree_util.tree_map(
            lambda parameter, gradient: parameter
            - self.config.learning_rate * gradient,
            self.params,
            gradients,
        )
        return JaxGRUPolicy(self.config, updated), float(loss)


__all__ = [
    "GRUPolicyConfig",
    "JaxGRUPolicy",
    "PolicyTrajectory",
    "RiskQuantileTracker",
    "TrajectoryBatch",
    "initialize_gru_policy",
    "masked_log_prob",
    "policy_logits",
    "risk_seeking_loss",
]
