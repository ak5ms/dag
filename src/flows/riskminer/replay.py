from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
import random

from .learned_policy import PolicyTrajectory, TrajectoryBatch


class ReplayBuffer:
    """Bounded trajectory replay buffer used between MCTS and policy updates."""

    def __init__(self, capacity: int) -> None:
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._items: deque[PolicyTrajectory] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def trajectories(self) -> tuple[PolicyTrajectory, ...]:
        return tuple(self._items)

    def add(self, trajectory: PolicyTrajectory) -> None:
        self._items.append(trajectory)

    def extend(self, trajectories: Iterable[PolicyTrajectory]) -> None:
        for trajectory in trajectories:
            self.add(trajectory)

    def clear(self) -> None:
        self._items.clear()

    def batches(
        self,
        batch_size: int,
        *,
        epochs: int = 1,
        seed: int = 0,
        shuffle: bool = True,
        reward_quantiles: Sequence[float] | None = None,
    ) -> Iterator[TrajectoryBatch]:
        if int(batch_size) <= 0 or int(epochs) <= 0:
            raise ValueError("batch_size and epochs must be positive")
        trajectories = list(self._items)
        if not trajectories:
            return
        if reward_quantiles is not None and (
            len(reward_quantiles) != len(trajectories)
        ):
            raise ValueError(
                "reward_quantiles must match the replay trajectory count"
            )
        quantiles = (
            [float(value) for value in reward_quantiles]
            if reward_quantiles is not None
            else [None] * len(trajectories)
        )
        base = list(zip(trajectories, quantiles))
        rng = random.Random(int(seed))
        for _ in range(int(epochs)):
            items = list(base)
            if shuffle:
                rng.shuffle(items)
            for start in range(0, len(items), int(batch_size)):
                chunk = items[start:start + int(batch_size)]
                chunk_trajectories = tuple(item[0] for item in chunk)
                chunk_quantiles = (
                    tuple(float(item[1]) for item in chunk)
                    if reward_quantiles is not None
                    else ()
                )
                yield TrajectoryBatch(
                    chunk_trajectories,
                    reward_quantiles=chunk_quantiles,
                )


__all__ = ["ReplayBuffer"]
