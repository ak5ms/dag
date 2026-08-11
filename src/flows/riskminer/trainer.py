from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import statistics

from .config import RiskMinerConfig
from .learned_policy import GRUPolicyConfig, JaxGRUPolicy, RiskQuantileTracker
from .mcts import RewardDenseRiskMCTS, RewardDenseSearchResult
from .replay import ReplayBuffer
from .reward import RewardDensePoolModel
from .rpn import TypedRPNEnvironment


@dataclass(frozen=True)
class MiningIterationReport:
    iteration: int
    search: RewardDenseSearchResult
    reward_quantile_before: float
    reward_quantile_after: float
    trajectory_quantiles: tuple[float, ...]
    mean_trajectory_reward: float
    max_trajectory_reward: float
    policy_losses: tuple[float, ...]
    policy_checkpoint: str | None


@dataclass(frozen=True)
class RiskSeekingTrainingResult:
    policy: JaxGRUPolicy
    quantile: RiskQuantileTracker
    iterations: tuple[MiningIterationReport, ...]


class RiskSeekingTrainer:
    """Alternate fresh-tree MCTS sampling and risk-seeking policy updates."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        config: RiskMinerConfig,
        policy: JaxGRUPolicy | None = None,
        quantile: RiskQuantileTracker | None = None,
        initial_token_priors: Sequence[float] | None = None,
        output_dir: str | Path | None = None,
        on_event: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        self.config = config
        self.policy = policy or JaxGRUPolicy.initialize(
            GRUPolicyConfig(
                vocabulary_size=int(vocabulary_size),
                max_sequence_length=config.max_tokens,
                learning_rate=config.policy_learning_rate,
                seed=config.seed,
            ),
            token_priors=initial_token_priors,
        )
        if self.policy.config.vocabulary_size != int(vocabulary_size):
            raise ValueError("policy vocabulary size mismatch")
        self.quantile = quantile or RiskQuantileTracker(
            cdf_quantile=config.quantile_cdf,
            learning_rate=config.quantile_learning_rate,
        )
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.on_event = on_event
        self.iteration_counter = 0

    def _emit(self, event: str, **payload: object) -> None:
        if self.on_event is not None:
            self.on_event(event, dict(payload))

    def run_iteration(
        self,
        environment: TypedRPNEnvironment,
        reward_model: RewardDensePoolModel,
        *,
        config: RiskMinerConfig | None = None,
        iteration: int | None = None,
    ) -> MiningIterationReport:
        active = config or environment.config
        if len(environment.vocabulary) != self.policy.config.vocabulary_size:
            raise ValueError("environment vocabulary changed after policy initialization")
        self.iteration_counter = (
            self.iteration_counter + 1 if iteration is None else int(iteration)
        )
        index = self.iteration_counter
        self._emit(
            "mining_iteration_start",
            iteration=index,
            simulations=active.simulations,
            rollouts_per_expansion=active.rollouts_per_expansion,
            quantile=self.quantile.value,
        )

        # Algorithm 2 resets both tree and replay buffer each outer iteration;
        # the policy network and alpha pool persist.
        replay = ReplayBuffer(active.replay_capacity)
        search = RewardDenseRiskMCTS(
            environment, reward_model, config=active, policy=self.policy
        ).search()
        replay.extend(search.trajectories)
        rewards = [trajectory.reward for trajectory in replay.trajectories]
        before = self.quantile.value
        trajectory_quantiles: list[float] = []
        # Algorithm 2 applies the Equation-11 recursion in trajectory order.
        # We retain each trajectory's contemporaneous threshold so batched
        # Equation-12/13 updates are equivalent to the sequential indicator.
        for reward in rewards:
            # Equations 11 and 13 use q_i for trajectory i.  Record the
            # pre-update threshold, then advance the stochastic quantile
            # recursion to q_(i+1).
            trajectory_quantiles.append(float(self.quantile.value))
            self.quantile = self.quantile.update(float(reward))
        after = self.quantile.value

        losses: list[float] = []
        for batch in replay.batches(
            active.policy_batch_size,
            epochs=active.policy_train_epochs,
            seed=active.seed + index,
            shuffle=True,
            reward_quantiles=trajectory_quantiles,
        ):
            self.policy, loss = self.policy.train_step(batch, self.quantile.value)
            losses.append(float(loss))

        checkpoint: str | None = None
        if self.output_dir is not None:
            path = self.output_dir / f"policy_iteration_{index:04d}.pkl"
            self.policy.save(
                path,
                iteration=index,
                reward_quantile=self.quantile.value,
                trajectory_count=len(rewards),
            )
            checkpoint = str(path)

        mean_reward = statistics.fmean(rewards) if rewards else float("nan")
        max_reward = max(rewards) if rewards else float("nan")
        report = MiningIterationReport(
            iteration=index,
            search=search,
            reward_quantile_before=float(before),
            reward_quantile_after=float(after),
            trajectory_quantiles=tuple(trajectory_quantiles),
            mean_trajectory_reward=float(mean_reward),
            max_trajectory_reward=float(max_reward),
            policy_losses=tuple(losses),
            policy_checkpoint=checkpoint,
        )
        self._emit(
            "mining_iteration_done",
            iteration=index,
            trajectories=len(rewards),
            pool_updates=search.metrics.pool_updates,
            pool_size=len(reward_model.pool.entries),
            pool_score=getattr(reward_model.pool, "score", None),
            quantile_before=before,
            quantile_after=after,
            mean_reward=mean_reward,
            max_reward=max_reward,
            policy_losses=losses,
            checkpoint=checkpoint,
        )
        return report

    def run(
        self,
        environment_factory: Callable[[int], TypedRPNEnvironment],
        reward_model: RewardDensePoolModel,
        *,
        iterations: int,
    ) -> RiskSeekingTrainingResult:
        if int(iterations) <= 0:
            raise ValueError("iterations must be positive")
        reports = []
        for index in range(1, int(iterations) + 1):
            environment = environment_factory(index)
            reports.append(
                self.run_iteration(
                    environment,
                    reward_model,
                    config=environment.config,
                    iteration=index,
                )
            )
        return RiskSeekingTrainingResult(
            self.policy, self.quantile, tuple(reports)
        )


__all__ = [
    "MiningIterationReport", "RiskSeekingTrainer", "RiskSeekingTrainingResult",
]
