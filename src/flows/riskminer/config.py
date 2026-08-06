from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskMinerConfig:
    """Configuration for grammar-guided RiskMiner search.

    A simulation is one tree selection/expansion followed by
    ``rollouts_per_expansion`` stochastic completions. Their mean reward is
    backed up through the selected path, while every unique completed formula
    remains eligible for the archive.
    """

    max_depth: int = 8
    max_tokens: int = 40
    max_stack: int = 8
    simulations: int = 128
    rollouts_per_expansion: int = 8
    evaluation_batch_size: int = 32
    archive_size: int = 100
    exploration: float = 1.25
    progressive_widening_k: float = 4.0
    progressive_widening_alpha: float = 0.5
    dense_rewards: bool = True
    invalid_reward: float = -1.0e6
    seed: int = 42

    def __post_init__(self) -> None:
        positive_ints = {
            "max_depth": self.max_depth,
            "max_tokens": self.max_tokens,
            "max_stack": self.max_stack,
            "simulations": self.simulations,
            "rollouts_per_expansion": self.rollouts_per_expansion,
            "evaluation_batch_size": self.evaluation_batch_size,
            "archive_size": self.archive_size,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.progressive_widening_k <= 0.0:
            raise ValueError("progressive_widening_k must be positive")
        if not 0.0 < self.progressive_widening_alpha <= 1.0:
            raise ValueError("progressive_widening_alpha must be in (0, 1]")
