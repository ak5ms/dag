from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

from flows.riskminer.rpn import RPNState, Token


@dataclass(frozen=True)
class PolicyEpisode:
    token_ids: tuple[int, ...]
    reward: float


class RiskSeekingTokenPolicy:
    """State-conditional lower-tail suppression policy for the initial checkpoint."""

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
        logits = [math.log(max(token.prior, 1e-12)) + self._global_bias[token.token_id] + self._transition_bias[(context, token.token_id)] for token in legal_tokens]
        maximum = max(logits)
        weights = [math.exp(value - maximum) for value in logits]
        total = sum(weights)
        return {token.token_id: weight / total for token, weight in zip(legal_tokens, weights)}

    def sample(self, state: RPNState, legal_tokens: Sequence[Token]) -> Token:
        priors = self.priors(state, legal_tokens)
        target = self.rng.random()
        cumulative = 0.0
        for token in legal_tokens:
            cumulative += priors[token.token_id]
            if cumulative >= target:
                return token
        return legal_tokens[-1]

    def update(self, episodes: Iterable[PolicyEpisode]) -> None:
        finite = [episode for episode in episodes if math.isfinite(episode.reward)]
        if not finite:
            return
        rewards = sorted(episode.reward for episode in finite)
        position = self.risk_quantile * (len(rewards) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        empirical = rewards[lower] * (1.0 - fraction) + rewards[upper] * fraction
        if not self._initialized_quantile:
            self.quantile_value = empirical
            self._initialized_quantile = True
        else:
            self.quantile_value += self.learning_rate * (empirical - self.quantile_value)
        for episode in finite:
            if episode.reward > self.quantile_value:
                continue
            previous = -1
            scale = self.learning_rate / max(1, len(episode.token_ids))
            for token_id in episode.token_ids:
                self._global_bias[token_id] -= scale
                self._transition_bias[(previous, token_id)] -= scale
                previous = token_id


__all__ = ["PolicyEpisode", "RiskSeekingTokenPolicy"]
