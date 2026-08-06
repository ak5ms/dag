from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from flows.riskminer.rpn import RPNState, Token


@dataclass(frozen=True)
class PolicyEpisode:
    token_ids: tuple[int, ...]
    reward: float


class RiskSeekingTokenPolicy:
    """Small deterministic fallback policy used by unit tests and ablations."""

    def __init__(self, *, risk_quantile: float = 0.80, learning_rate: float = 0.01, seed: int = 0) -> None:
        if not 0.0 < risk_quantile < 1.0:
            raise ValueError("risk_quantile must be in (0, 1)")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.risk_quantile = float(risk_quantile)
        self.learning_rate = float(learning_rate)
        self.rng = random.Random(seed)
        self.quantile_value = 0.0
        self._initialized_quantile = False
        self._global_bias: dict[int, float] = defaultdict(float)
        self._transition_bias: dict[tuple[int, int], float] = defaultdict(float)

    @staticmethod
    def _context(state: RPNState) -> int:
        return state.token_ids[-1] if state.token_ids else -1

    def priors(self, state: RPNState, legal_tokens: Sequence[Token]) -> dict[int, float]:
        if not legal_tokens:
            return {}
        context = self._context(state)
        logits = [
            math.log(max(token.prior, 1e-12))
            + self._global_bias[token.token_id]
            + self._transition_bias[(context, token.token_id)]
            for token in legal_tokens
        ]
        maximum = max(logits)
        weights = [math.exp(value - maximum) for value in logits]
        total = sum(weights)
        return {
            token.token_id: weight / total
            for token, weight in zip(legal_tokens, weights)
        }

    def sample(self, state: RPNState, legal_tokens: Sequence[Token]) -> Token:
        return _sample_from_priors(
            self.rng,
            legal_tokens,
            self.priors(state, legal_tokens),
        )

    def update(self, episodes: Iterable[PolicyEpisode]) -> None:
        finite = [episode for episode in episodes if math.isfinite(episode.reward)]
        if not finite:
            return
        rewards = np.asarray([episode.reward for episode in finite], dtype=float)
        empirical = float(np.quantile(rewards, self.risk_quantile))
        if not self._initialized_quantile:
            self.quantile_value = empirical
            self._initialized_quantile = True
        else:
            self.quantile_value += self.learning_rate * (
                empirical - self.quantile_value
            )
        for episode in finite:
            if episode.reward > self.quantile_value:
                continue
            previous = -1
            scale = self.learning_rate / max(1, len(episode.token_ids))
            for token_id in episode.token_ids:
                self._global_bias[token_id] -= scale
                self._transition_bias[(previous, token_id)] -= scale
                previous = token_id


def _sample_from_priors(
    rng: random.Random,
    legal_tokens: Sequence[Token],
    priors: dict[int, float],
) -> Token:
    target = rng.random()
    cumulative = 0.0
    for token in legal_tokens:
        cumulative += priors[token.token_id]
        if cumulative >= target:
            return token
    return legal_tokens[-1]


def _xavier(key, shape: tuple[int, ...]) -> jax.Array:
    fan_in = max(1, shape[0])
    fan_out = max(1, shape[-1])
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def _init_gru_policy(
    key: jax.Array,
    *,
    vocabulary_size: int,
    embedding_size: int,
    hidden_size: int,
    layers: int,
) -> dict[str, object]:
    keys = iter(jax.random.split(key, 3 + layers * 3))
    parameters: dict[str, object] = {
        "embedding": _xavier(next(keys), (vocabulary_size + 1, embedding_size)),
    }
    recurrent = []
    input_size = embedding_size
    for _ in range(layers):
        joined = input_size + hidden_size
        recurrent.append({
            "wz": _xavier(next(keys), (joined, hidden_size)),
            "wr": _xavier(next(keys), (joined, hidden_size)),
            "wh": _xavier(next(keys), (joined, hidden_size)),
            "bz": jnp.zeros((hidden_size,)),
            "br": jnp.zeros((hidden_size,)),
            "bh": jnp.zeros((hidden_size,)),
        })
        input_size = hidden_size
    parameters["recurrent"] = tuple(recurrent)
    parameters["w1"] = _xavier(next(keys), (hidden_size, 32))
    parameters["b1"] = jnp.zeros((32,))
    parameters["w2"] = _xavier(next(keys), (32, 32))
    parameters["b2"] = jnp.zeros((32,))
    parameters["wo"] = _xavier(next(keys), (32, vocabulary_size))
    parameters["bo"] = jnp.zeros((vocabulary_size,))
    return parameters


def _gru_step(parameters, hidden: jax.Array, value: jax.Array) -> jax.Array:
    joined = jnp.concatenate((value, hidden), axis=-1)
    update = jax.nn.sigmoid(joined @ parameters["wz"] + parameters["bz"])
    reset = jax.nn.sigmoid(joined @ parameters["wr"] + parameters["br"])
    candidate_input = jnp.concatenate((value, reset * hidden), axis=-1)
    candidate = jnp.tanh(candidate_input @ parameters["wh"] + parameters["bh"])
    return (1.0 - update) * hidden + update * candidate


def _policy_logits(parameters, token_ids: jax.Array, start_token: int) -> jax.Array:
    recurrent = parameters["recurrent"]
    hidden_size = recurrent[0]["bz"].shape[0]
    hidden = tuple(jnp.zeros((hidden_size,)) for _ in recurrent)
    previous = jnp.asarray(start_token, dtype=jnp.int32)

    def advance(carry, token_id):
        previous_token, hidden_layers = carry
        value = parameters["embedding"][previous_token]
        next_hidden = []
        for layer_index, layer in enumerate(recurrent):
            state = _gru_step(layer, hidden_layers[layer_index], value)
            next_hidden.append(state)
            value = state
        return (token_id, tuple(next_hidden)), value

    if token_ids.shape[0]:
        (previous, hidden), _ = jax.lax.scan(
            advance,
            (previous, hidden),
            token_ids,
        )
    value = hidden[-1]
    value = jax.nn.tanh(value @ parameters["w1"] + parameters["b1"])
    value = jax.nn.tanh(value @ parameters["w2"] + parameters["b2"])
    return value @ parameters["wo"] + parameters["bo"]


def _batch_sequence_log_probability(
    parameters,
    token_matrix: jax.Array,
    lengths: jax.Array,
    start_token: int,
) -> jax.Array:
    batch_size, max_length = token_matrix.shape
    recurrent = parameters["recurrent"]
    hidden_size = recurrent[0]["bz"].shape[0]
    hidden = tuple(
        jnp.zeros((batch_size, hidden_size))
        for _ in recurrent
    )
    previous = jnp.full((batch_size,), start_token, dtype=jnp.int32)
    total = jnp.zeros((batch_size,))

    def step(carry, time_index):
        previous_token, hidden_layers, accumulated = carry
        value = parameters["embedding"][previous_token]
        next_hidden = []
        for layer_index, layer in enumerate(recurrent):
            state = _gru_step(layer, hidden_layers[layer_index], value)
            next_hidden.append(state)
            value = state
        projected = jax.nn.tanh(value @ parameters["w1"] + parameters["b1"])
        projected = jax.nn.tanh(projected @ parameters["w2"] + parameters["b2"])
        logits = projected @ parameters["wo"] + parameters["bo"]
        target = token_matrix[:, time_index]
        selected = jnp.take_along_axis(
            jax.nn.log_softmax(logits),
            target[:, None],
            axis=1,
        )[:, 0]
        active = time_index < lengths
        accumulated = accumulated + jnp.where(active, selected, 0.0)
        previous_token = jnp.where(active, target, previous_token)
        return (previous_token, tuple(next_hidden), accumulated), None

    (_, _, total), _ = jax.lax.scan(
        step,
        (previous, hidden, total),
        jnp.arange(max_length),
    )
    return total


class GRURiskSeekingTokenPolicy:
    """Four-layer GRU policy with RiskMiner lower-tail suppression updates.

    The network follows the paper's 64-unit, four-layer GRU and 32/32 MLP.
    Candidate evaluation remains entirely in cpp_stream; JAX is used only for
    token-policy inference and training.
    """

    def __init__(
        self,
        vocabulary_size: int,
        *,
        embedding_size: int = 32,
        hidden_size: int = 64,
        layers: int = 4,
        risk_quantile: float = 0.80,
        learning_rate: float = 0.001,
        quantile_learning_rate: float = 0.01,
        seed: int = 0,
    ) -> None:
        if vocabulary_size <= 1:
            raise ValueError("vocabulary_size must exceed one")
        if not 0.0 < risk_quantile < 1.0:
            raise ValueError("risk_quantile must be in (0, 1)")
        self.vocabulary_size = int(vocabulary_size)
        self.start_token = self.vocabulary_size
        self.risk_quantile = float(risk_quantile)
        self.learning_rate = float(learning_rate)
        self.quantile_learning_rate = float(quantile_learning_rate)
        self.rng = random.Random(seed)
        self.parameters = _init_gru_policy(
            jax.random.PRNGKey(seed),
            vocabulary_size=self.vocabulary_size,
            embedding_size=int(embedding_size),
            hidden_size=int(hidden_size),
            layers=int(layers),
        )
        self.quantile_value = 0.0
        self._initialized_quantile = False
        self.training_steps = 0

    def priors(
        self,
        state: RPNState,
        legal_tokens: Sequence[Token],
    ) -> dict[int, float]:
        if not legal_tokens:
            return {}
        token_ids = jnp.asarray(state.token_ids, dtype=jnp.int32)
        logits = np.asarray(
            _policy_logits(self.parameters, token_ids, self.start_token),
            dtype=float,
        )
        legal_logits = np.asarray([
            logits[token.token_id] + math.log(max(token.prior, 1e-12))
            for token in legal_tokens
        ])
        legal_logits -= np.max(legal_logits)
        weights = np.exp(legal_logits)
        weights /= np.sum(weights)
        return {
            token.token_id: float(weight)
            for token, weight in zip(legal_tokens, weights)
        }

    def sample(self, state: RPNState, legal_tokens: Sequence[Token]) -> Token:
        return _sample_from_priors(
            self.rng,
            legal_tokens,
            self.priors(state, legal_tokens),
        )

    def update(self, episodes: Iterable[PolicyEpisode]) -> None:
        finite = [
            episode
            for episode in episodes
            if math.isfinite(episode.reward) and episode.token_ids
        ]
        if not finite:
            return
        rewards = np.asarray([episode.reward for episode in finite], dtype=float)
        empirical = float(np.quantile(rewards, self.risk_quantile))
        if not self._initialized_quantile:
            self.quantile_value = empirical
            self._initialized_quantile = True
        for reward in rewards:
            self.quantile_value += self.quantile_learning_rate * (
                self.risk_quantile
                - float(reward <= self.quantile_value)
            )
        lower_tail = [
            episode for episode in finite
            if episode.reward <= self.quantile_value
        ]
        if not lower_tail:
            return
        max_length = max(len(episode.token_ids) for episode in lower_tail)
        token_matrix = np.zeros((len(lower_tail), max_length), dtype=np.int32)
        lengths = np.empty((len(lower_tail),), dtype=np.int32)
        for row, episode in enumerate(lower_tail):
            lengths[row] = len(episode.token_ids)
            token_matrix[row, : lengths[row]] = episode.token_ids
        tokens = jnp.asarray(token_matrix)
        sequence_lengths = jnp.asarray(lengths)

        def loss_fn(parameters):
            # Log probabilities are non-positive. Minimizing their mean for the
            # lower tail makes those trajectories less probable.
            values = _batch_sequence_log_probability(
                parameters,
                tokens,
                sequence_lengths,
                self.start_token,
            )
            return jnp.mean(values)

        _, gradients = jax.value_and_grad(loss_fn)(self.parameters)
        self.parameters = jax.tree.map(
            lambda parameter, gradient: parameter - self.learning_rate * gradient,
            self.parameters,
            gradients,
        )
        self.training_steps += 1


__all__ = [
    "GRURiskSeekingTokenPolicy",
    "PolicyEpisode",
    "RiskSeekingTokenPolicy",
]
