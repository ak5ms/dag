from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RiskMinerConfig:
    """Configuration for typed reward-dense RiskMiner search.

    A simulation selects/expands one permanent tree edge and then performs
    ``rollouts_per_expansion`` BEG-to-END completions.  The paper-style trainer
    creates a fresh tree/replay buffer per mining iteration while preserving the
    policy network and the alpha pool.
    """

    max_depth: int = 8
    min_formula_depth: int = 1
    max_tokens: int = 30
    max_stack: int = 8
    simulations: int = 128
    rollouts_per_expansion: int = 1
    evaluation_batch_size: int = 32
    archive_size: int = 500
    exploration: float = 1.25
    progressive_widening_k: float = 4.0
    progressive_widening_alpha: float = 0.5
    rollout_end_probability: float = 0.30
    dense_rewards: bool = True
    invalid_reward: float = -5.0
    discount: float = 1.0
    replay_capacity: int = 512
    policy_train_epochs: int = 1
    policy_batch_size: int = 32
    policy_learning_rate: float = 1.0e-3
    quantile_cdf: float = 0.80
    quantile_learning_rate: float = 0.01
    pool_capacity: int = 100
    pool_min_improvement: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        positive_ints = {
            "max_depth": self.max_depth,
            "min_formula_depth": self.min_formula_depth,
            "max_tokens": self.max_tokens,
            "max_stack": self.max_stack,
            "simulations": self.simulations,
            "rollouts_per_expansion": self.rollouts_per_expansion,
            "evaluation_batch_size": self.evaluation_batch_size,
            "archive_size": self.archive_size,
            "replay_capacity": self.replay_capacity,
            "policy_train_epochs": self.policy_train_epochs,
            "policy_batch_size": self.policy_batch_size,
            "pool_capacity": self.pool_capacity,
        }
        for name, value in positive_ints.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_formula_depth > self.max_depth:
            raise ValueError("min_formula_depth cannot exceed max_depth")
        if not math.isfinite(self.exploration) or self.exploration < 0.0:
            raise ValueError("exploration must be finite and nonnegative")
        if self.progressive_widening_k <= 0.0:
            raise ValueError("progressive_widening_k must be positive")
        if not 0.0 < self.progressive_widening_alpha <= 1.0:
            raise ValueError("progressive_widening_alpha must be in (0, 1]")
        if not 0.0 <= self.rollout_end_probability <= 1.0:
            raise ValueError("rollout_end_probability must be in [0, 1]")
        if not math.isfinite(self.invalid_reward):
            raise ValueError("invalid_reward must be finite")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if not math.isfinite(self.policy_learning_rate) or self.policy_learning_rate <= 0.0:
            raise ValueError("policy_learning_rate must be finite and positive")
        if not 0.0 < self.quantile_cdf < 1.0:
            raise ValueError("quantile_cdf must be in (0, 1)")
        if not math.isfinite(self.quantile_learning_rate) or self.quantile_learning_rate <= 0.0:
            raise ValueError("quantile_learning_rate must be finite and positive")
        if not math.isfinite(self.pool_min_improvement):
            raise ValueError("pool_min_improvement must be finite")
