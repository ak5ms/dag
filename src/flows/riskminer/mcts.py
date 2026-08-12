from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import random
import time
from typing import Protocol

from trading_dsl_engine.base.parser import Expr

from .config import RiskMinerConfig
from .rpn import RPNState, StackValue, TypedRPNEnvironment


class CandidateEvaluator(Protocol):
    def evaluate(self, candidates: Sequence[Expr]) -> Mapping[tuple, float]:
        ...


class ActionPolicy(Protocol):
    def priors(
        self,
        environment: TypedRPNEnvironment,
        state: RPNState,
        legal_actions: Sequence[int],
    ) -> Mapping[int, float]:
        ...


class SchemaPriorPolicy:
    """Deterministic policy using token-level priors.

    This is the checkpoint policy. A learned masked GRU can implement the same
    interface without changing MCTS or native evaluation.
    """

    def priors(
        self,
        environment: TypedRPNEnvironment,
        state: RPNState,
        legal_actions: Sequence[int],
    ) -> Mapping[int, float]:
        del state
        raw = {
            token_id: max(
                0.0,
                environment.vocabulary.by_id[token_id].prior,
            )
            for token_id in legal_actions
        }
        total = sum(raw.values())
        if total <= 0.0:
            uniform = 1.0 / max(1, len(raw))
            return {token_id: uniform for token_id in raw}
        return {token_id: value / total for token_id, value in raw.items()}


@dataclass
class EdgeStats:
    prior: float
    visits: int = 0
    virtual_visits: int = 0
    value_sum: float = 0.0
    child_key: tuple | None = None

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class TreeNode:
    state: RPNState
    visits: int = 0
    virtual_visits: int = 0
    edges: dict[int, EdgeStats] = field(default_factory=dict)


@dataclass(frozen=True)
class FormulaObservation:
    value: StackValue
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class Rollout:
    terminal_state: RPNState
    formulas: tuple[FormulaObservation, ...]


@dataclass(frozen=True)
class ArchiveEntry:
    expr: Expr
    score: float
    depth: int
    canonical_key: tuple
    rpn: str


class FormulaArchive:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("archive capacity must be positive")
        self.capacity = int(capacity)
        self._entries: dict[tuple, ArchiveEntry] = {}

    def add(self, entry: ArchiveEntry) -> None:
        previous = self._entries.get(entry.canonical_key)
        if previous is None or (
            entry.score,
            -entry.depth,
            entry.rpn,
        ) > (
            previous.score,
            -previous.depth,
            previous.rpn,
        ):
            self._entries[entry.canonical_key] = entry
        if len(self._entries) > self.capacity:
            worst_key = min(
                self._entries,
                key=lambda key: (
                    self._entries[key].score,
                    -self._entries[key].depth,
                    repr(key),
                ),
            )
            del self._entries[worst_key]

    @property
    def entries(self) -> tuple[ArchiveEntry, ...]:
        return tuple(
            sorted(
                self._entries.values(),
                key=lambda item: (
                    -item.score,
                    item.depth,
                    repr(item.canonical_key),
                ),
            )
        )


@dataclass(frozen=True)
class SearchMetrics:
    simulations: int
    rollouts: int
    unique_formula_requests: int
    finite_formula_scores: int
    invalid_rollouts: int
    tree_nodes: int
    wall_seconds: float


@dataclass(frozen=True)
class RiskMinerSearchResult:
    archive: tuple[ArchiveEntry, ...]
    metrics: SearchMetrics


@dataclass(frozen=True)
class _Selection:
    path: tuple[tuple[tuple, int], ...]
    leaf: RPNState


class RiskMCTS:
    def __init__(
        self,
        environment: TypedRPNEnvironment,
        evaluator: CandidateEvaluator,
        *,
        config: RiskMinerConfig | None = None,
        policy: ActionPolicy | None = None,
    ) -> None:
        self.environment = environment
        self.evaluator = evaluator
        self.config = config or environment.config
        self.policy = policy or SchemaPriorPolicy()
        self.rng = random.Random(self.config.seed)
        self.nodes: dict[tuple, TreeNode] = {}
        self.archive = FormulaArchive(self.config.archive_size)

    def search(self) -> RiskMinerSearchResult:
        started = time.perf_counter()
        root = self.environment.initial_state()
        self._node(root)

        simulations_completed = 0
        rollouts_completed = 0
        unique_formula_requests = 0
        finite_formula_scores = 0
        invalid_rollouts = 0

        while simulations_completed < self.config.simulations:
            wave_size = min(
                self.config.evaluation_batch_size,
                self.config.simulations - simulations_completed,
            )
            selections: list[_Selection] = []
            rollout_groups: list[list[Rollout]] = []
            all_observations: dict[tuple, FormulaObservation] = {}

            for _ in range(wave_size):
                selection = self._select_and_expand(root)
                selections.append(selection)
                group: list[Rollout] = []
                for _ in range(self.config.rollouts_per_expansion):
                    rollout = self._rollout(selection.leaf)
                    group.append(rollout)
                    rollouts_completed += 1
                    for observation in rollout.formulas:
                        all_observations.setdefault(
                            observation.value.canonical_key,
                            observation,
                        )
                rollout_groups.append(group)

            expressions = [
                observation.value.expr
                for observation in all_observations.values()
            ]
            unique_formula_requests += len(expressions)
            score_map = (
                dict(self.evaluator.evaluate(expressions))
                if expressions
                else {}
            )
            finite_formula_scores += sum(
                math.isfinite(value) for value in score_map.values()
            )

            for selection, group in zip(selections, rollout_groups):
                rollout_rewards: list[float] = []
                for rollout in group:
                    scored = []
                    seen: set[tuple] = set()
                    for observation in rollout.formulas:
                        key = observation.value.canonical_key
                        if key in seen:
                            continue
                        seen.add(key)
                        value = float(score_map.get(key, -math.inf))
                        if math.isfinite(value):
                            scored.append((observation, value))
                            self.archive.add(
                                ArchiveEntry(
                                    observation.value.expr,
                                    value,
                                    observation.value.depth,
                                    key,
                                    self._render_token_ids(
                                        observation.token_ids
                                    ),
                                )
                            )
                    if not scored:
                        invalid_rollouts += 1
                        rollout_rewards.append(self.config.invalid_reward)
                    elif self.config.dense_rewards:
                        rollout_rewards.append(
                            sum(value for _, value in scored)
                        )
                    else:
                        rollout_rewards.append(scored[-1][1])

                simulation_reward = (
                    sum(rollout_rewards) / len(rollout_rewards)
                    if rollout_rewards
                    else self.config.invalid_reward
                )
                self._backpropagate(selection.path, simulation_reward)
                simulations_completed += 1

        elapsed = time.perf_counter() - started
        return RiskMinerSearchResult(
            self.archive.entries,
            SearchMetrics(
                simulations=simulations_completed,
                rollouts=rollouts_completed,
                unique_formula_requests=unique_formula_requests,
                finite_formula_scores=finite_formula_scores,
                invalid_rollouts=invalid_rollouts,
                tree_nodes=len(self.nodes),
                wall_seconds=elapsed,
            ),
        )

    def _node(self, state: RPNState) -> TreeNode:
        key = self.environment.state_key(state)
        node = self.nodes.get(key)
        if node is None:
            node = TreeNode(state)
            self.nodes[key] = node
        return node

    def _select_and_expand(self, root: RPNState) -> _Selection:
        state = root
        path: list[tuple[tuple, int]] = []
        while True:
            node_key = self.environment.state_key(state)
            node = self._node(state)
            if state.terminated:
                return _Selection(tuple(path), state)
            legal = self.environment.legal_actions(state)
            if not legal:
                return _Selection(tuple(path), state)

            priors = self.policy.priors(
                self.environment,
                state,
                legal,
            )
            allowed_count = min(
                len(legal),
                max(
                    1,
                    int(
                        self.config.progressive_widening_k
                        * max(1, node.visits + node.virtual_visits)
                        ** self.config.progressive_widening_alpha
                    ),
                ),
            )
            ranked = sorted(
                legal,
                key=lambda token_id: (
                    -float(priors.get(token_id, 0.0)),
                    self.environment.vocabulary.by_id[token_id].name,
                ),
            )
            for token_id in ranked[:allowed_count]:
                node.edges.setdefault(
                    token_id,
                    EdgeStats(float(priors.get(token_id, 0.0))),
                )

            unvisited = [
                (token_id, edge)
                for token_id, edge in node.edges.items()
                if edge.visits + edge.virtual_visits == 0
            ]
            if unvisited:
                token_id, edge = max(
                    unvisited,
                    key=lambda item: (
                        item[1].prior,
                        -item[0],
                    ),
                )
                expanded = True
            else:
                total = max(1, node.visits + node.virtual_visits)

                def puct(item: tuple[int, EdgeStats]) -> tuple[float, int]:
                    token_id, edge = item
                    visits = edge.visits + edge.virtual_visits
                    exploration = (
                        self.config.exploration
                        * edge.prior
                        * math.sqrt(total)
                        / (1 + visits)
                    )
                    return edge.q + exploration, -token_id

                token_id, edge = max(node.edges.items(), key=puct)
                expanded = False

            child = self.environment.apply(state, token_id)
            child_key = self.environment.state_key(child)
            edge.child_key = child_key
            edge.virtual_visits += 1
            node.virtual_visits += 1
            path.append((node_key, token_id))
            self._node(child)
            state = child
            if expanded:
                return _Selection(tuple(path), state)

    def _rollout(self, leaf: RPNState) -> Rollout:
        state = leaf
        formulas: list[FormulaObservation] = []
        last_key: tuple | None = None

        def observe(current: RPNState) -> None:
            nonlocal last_key
            value = self.environment.formula_value(current)
            if value is not None and value.canonical_key != last_key:
                formulas.append(
                    FormulaObservation(value, current.token_ids)
                )
                last_key = value.canonical_key

        observe(state)
        while not state.terminated:
            legal = self.environment.legal_actions(state)
            if not legal:
                break
            priors = self.policy.priors(
                self.environment,
                state,
                legal,
            )
            end_id = self.environment.vocabulary.end.token_id
            if (
                end_id in legal
                and self.rng.random() < 0.30
            ):
                token_id = end_id
            else:
                non_end = [
                    token for token in legal if token != end_id
                ]
                choices = non_end or list(legal)
                weights = [
                    max(0.0, float(priors.get(token, 0.0)))
                    for token in choices
                ]
                token_id = self.rng.choices(
                    choices,
                    weights=weights,
                    k=1,
                )[0]
            state = self.environment.apply(state, token_id)
            observe(state)

        if not state.terminated and self.environment.can_terminate(state):
            state = self.environment.apply(
                state,
                self.environment.vocabulary.end.token_id,
            )
            observe(state)
        return Rollout(state, tuple(formulas))

    def _backpropagate(
        self,
        path: Sequence[tuple[tuple, int]],
        reward: float,
    ) -> None:
        finite_reward = (
            float(reward)
            if math.isfinite(reward)
            else self.config.invalid_reward
        )
        for node_key, token_id in reversed(path):
            node = self.nodes[node_key]
            edge = node.edges[token_id]
            node.virtual_visits = max(0, node.virtual_visits - 1)
            edge.virtual_visits = max(0, edge.virtual_visits - 1)
            node.visits += 1
            edge.visits += 1
            edge.value_sum += finite_reward

    def _render_token_ids(self, token_ids: Sequence[int]) -> str:
        return " ".join(
            self.environment.vocabulary.by_id[token_id].name
            for token_id in token_ids
        )


__all__ = [
    "ActionPolicy",
    "ArchiveEntry",
    "CandidateEvaluator",
    "FormulaArchive",
    "RiskMCTS",
    "RiskMinerSearchResult",
    "SchemaPriorPolicy",
    "SearchMetrics",
]
