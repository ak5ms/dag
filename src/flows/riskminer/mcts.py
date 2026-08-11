from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import random
import time
from typing import Protocol

from trading_dsl_engine.base.parser import Expr

from .config import RiskMinerConfig
from .learned_policy import PolicyTrajectory
from .rpn import RPNState, StackValue, TokenKind, TypedRPNEnvironment


class CandidateEvaluator(Protocol):
    def evaluate(self, candidates: Sequence[Expr]) -> Mapping[tuple, float]: ...


class ActionPolicy(Protocol):
    def priors(
        self,
        environment: TypedRPNEnvironment,
        state: RPNState,
        legal_actions: Sequence[int],
    ) -> Mapping[int, float]: ...


class DenseTerminalReward(Protocol):
    @property
    def reward(self) -> float: ...

    @property
    def transition(self): ...


class DenseRewardModel(Protocol):
    def intermediate_rewards(
        self, values: Sequence[StackValue]
    ) -> Mapping[tuple, float]: ...

    def terminal_reward(
        self,
        value: StackValue,
        *,
        rpn: str,
        individual_score: float = float("nan"),
    ) -> DenseTerminalReward: ...


class SchemaPriorPolicy:
    """Deterministic fallback using normalized token-level schema weights."""

    def priors(
        self,
        environment: TypedRPNEnvironment,
        state: RPNState,
        legal_actions: Sequence[int],
    ) -> Mapping[int, float]:
        del state
        raw = {
            token_id: max(0.0, environment.vocabulary.by_id[token_id].prior)
            for token_id in legal_actions
        }
        total = sum(raw.values())
        if total <= 0.0:
            probability = 1.0 / max(1, len(raw))
            return {token_id: probability for token_id in raw}
        return {token_id: value / total for token_id, value in raw.items()}


@dataclass
class EdgeStats:
    prior: float
    visits: int = 0
    virtual_visits: int = 0
    value_sum: float = 0.0
    child_key: tuple | None = None
    reward: float = 0.0

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
        if int(capacity) <= 0:
            raise ValueError("archive capacity must be positive")
        self.capacity = int(capacity)
        self._entries: dict[tuple, ArchiveEntry] = {}

    def add(self, entry: ArchiveEntry) -> None:
        previous = self._entries.get(entry.canonical_key)
        if previous is None or (
            entry.score, -entry.depth, entry.rpn
        ) > (
            previous.score, -previous.depth, previous.rpn
        ):
            self._entries[entry.canonical_key] = entry
        if len(self._entries) > self.capacity:
            worst = min(
                self._entries,
                key=lambda key: (
                    self._entries[key].score,
                    -self._entries[key].depth,
                    repr(key),
                ),
            )
            del self._entries[worst]

    @property
    def entries(self) -> tuple[ArchiveEntry, ...]:
        return tuple(
            sorted(
                self._entries.values(),
                key=lambda item: (-item.score, item.depth, repr(item.canonical_key)),
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
    trajectories: int = 0
    pool_updates: int = 0
    intermediate_formula_requests: int = 0


@dataclass(frozen=True)
class RiskMinerSearchResult:
    archive: tuple[ArchiveEntry, ...]
    metrics: SearchMetrics


@dataclass(frozen=True)
class RewardDenseSearchResult:
    archive: tuple[ArchiveEntry, ...]
    trajectories: tuple[PolicyTrajectory, ...]
    metrics: SearchMetrics


@dataclass(frozen=True)
class _Selection:
    path: tuple[tuple[tuple, int], ...]
    leaf: RPNState


@dataclass(frozen=True)
class _DenseSelection:
    path: tuple[tuple[tuple, int], ...]
    states: tuple[RPNState, ...]
    legal_actions: tuple[tuple[int, ...], ...]
    actions: tuple[int, ...]
    resulting_states: tuple[RPNState, ...]
    leaf: RPNState


class _TreeMixin:
    environment: TypedRPNEnvironment
    config: RiskMinerConfig
    policy: ActionPolicy
    rng: random.Random
    nodes: dict[tuple, TreeNode]

    def _node(self, state: RPNState) -> TreeNode:
        key = self.environment.state_key(state)
        node = self.nodes.get(key)
        if node is None:
            node = TreeNode(state)
            self.nodes[key] = node
        return node

    def _choose_edge(
        self,
        state: RPNState,
        legal: Sequence[int],
    ) -> tuple[int, EdgeStats, bool]:
        node = self._node(state)
        priors = self.policy.priors(self.environment, state, legal)
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
                token_id, EdgeStats(float(priors.get(token_id, 0.0)))
            )
        unvisited = [
            (token_id, edge)
            for token_id, edge in node.edges.items()
            if edge.visits + edge.virtual_visits == 0
        ]
        if unvisited:
            token_id, edge = max(
                unvisited, key=lambda item: (item[1].prior, -item[0])
            )
            return token_id, edge, True

        total = max(1, node.visits + node.virtual_visits)

        def puct(item: tuple[int, EdgeStats]) -> tuple[float, int]:
            token_id, edge = item
            visits = edge.visits + edge.virtual_visits
            bonus = (
                self.config.exploration
                * edge.prior
                * math.sqrt(total)
                / (1 + visits)
            )
            return edge.q + bonus, -token_id

        token_id, edge = max(node.edges.items(), key=puct)
        return token_id, edge, False

    def _sample_rollout_action(
        self,
        state: RPNState,
        legal: Sequence[int],
        rng: random.Random,
    ) -> int:
        end_id = self.environment.vocabulary.end.token_id
        if (
            end_id in legal
            and rng.random() < self.config.rollout_end_probability
        ):
            return end_id
        choices = [token_id for token_id in legal if token_id != end_id]
        if not choices:
            return end_id
        priors = self.policy.priors(self.environment, state, legal)

        # Correct action-class cardinality bias in rollout completion. With the
        # 69-field InputData grammar, direct token sampling gives terminals most
        # of the aggregate probability even when each terminal is individually
        # no more likely than an operator. Preserve learned token ranking within
        # each structural class, but give a class total weight equal to its mean
        # prior rather than the sum of all member priors.
        groups: dict[str, list[int]] = {}
        for token_id in choices:
            token = self.environment.vocabulary.by_id[token_id]
            if token.kind in {TokenKind.TERMINAL, TokenKind.LITERAL}:
                group = "push"
            else:
                operator = token.operator
                assert operator is not None
                group = "unary" if operator.arity == 1 else "reduce"
            groups.setdefault(group, []).append(token_id)

        weights_by_id: dict[int, float] = {}
        for token_ids in groups.values():
            cardinality = len(token_ids)
            for token_id in token_ids:
                weights_by_id[token_id] = (
                    max(0.0, float(priors.get(token_id, 0.0))) / cardinality
                )
        weights = [weights_by_id[token_id] for token_id in choices]
        if not any(weights):
            weights = [1.0] * len(choices)
        return rng.choices(choices, weights=weights, k=1)[0]

    def _render_token_ids(self, token_ids: Sequence[int]) -> str:
        return " ".join(
            self.environment.vocabulary.by_id[token_id].name
            for token_id in token_ids
        )


class RiskMCTS(_TreeMixin):
    """Backwards-compatible batched standalone-candidate MCTS."""

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
        simulations = rollouts = unique = finite = invalid = 0
        while simulations < self.config.simulations:
            wave_size = min(
                self.config.evaluation_batch_size,
                self.config.simulations - simulations,
            )
            selections: list[_Selection] = []
            rollout_groups: list[list[Rollout]] = []
            observations: dict[tuple, FormulaObservation] = {}
            for _ in range(wave_size):
                selection = self._select_and_expand(root)
                selections.append(selection)
                group = []
                for _ in range(self.config.rollouts_per_expansion):
                    rollout = self._rollout(selection.leaf)
                    group.append(rollout)
                    rollouts += 1
                    for observation in rollout.formulas:
                        observations.setdefault(observation.value.canonical_key, observation)
                rollout_groups.append(group)
            expressions = [observation.value.expr for observation in observations.values()]
            unique += len(expressions)
            scores = dict(self.evaluator.evaluate(expressions)) if expressions else {}
            finite += sum(math.isfinite(value) for value in scores.values())
            for selection, group in zip(selections, rollout_groups):
                rewards = []
                for rollout in group:
                    scored: list[tuple[FormulaObservation, float]] = []
                    seen: set[tuple] = set()
                    for observation in rollout.formulas:
                        key = observation.value.canonical_key
                        if key in seen:
                            continue
                        seen.add(key)
                        value = float(scores.get(key, -math.inf))
                        if math.isfinite(value):
                            scored.append((observation, value))
                            self.archive.add(
                                ArchiveEntry(
                                    observation.value.expr,
                                    value,
                                    observation.value.depth,
                                    key,
                                    self._render_token_ids(observation.token_ids),
                                )
                            )
                    if not scored:
                        invalid += 1
                        rewards.append(self.config.invalid_reward)
                    elif self.config.dense_rewards:
                        rewards.append(sum(value for _, value in scored))
                    else:
                        rewards.append(scored[-1][1])
                reward = (
                    sum(rewards) / len(rewards)
                    if rewards else self.config.invalid_reward
                )
                self._backpropagate(selection.path, reward)
                simulations += 1
        return RiskMinerSearchResult(
            self.archive.entries,
            SearchMetrics(
                simulations, rollouts, unique, finite, invalid, len(self.nodes),
                time.perf_counter() - started,
            ),
        )

    def _select_and_expand(self, root: RPNState) -> _Selection:
        state = root
        path: list[tuple[tuple, int]] = []
        while True:
            key = self.environment.state_key(state)
            node = self._node(state)
            if state.terminated:
                return _Selection(tuple(path), state)
            legal = self.environment.legal_actions(state)
            if not legal:
                return _Selection(tuple(path), state)
            token_id, edge, expanded = self._choose_edge(state, legal)
            child = self.environment.apply(state, token_id)
            edge.child_key = self.environment.state_key(child)
            edge.virtual_visits += 1
            node.virtual_visits += 1
            path.append((key, token_id))
            self._node(child)
            state = child
            if expanded:
                return _Selection(tuple(path), state)

    def _rollout(self, leaf: RPNState) -> Rollout:
        state = leaf
        formulas: list[FormulaObservation] = []
        last_key = None

        def observe(current: RPNState) -> None:
            nonlocal last_key
            value = self.environment.formula_value(current)
            if value is not None and value.canonical_key != last_key:
                formulas.append(FormulaObservation(value, current.token_ids))
                last_key = value.canonical_key

        observe(state)
        while not state.terminated:
            legal = self.environment.legal_actions(state)
            if not legal:
                break
            state = self.environment.apply(
                state, self._sample_rollout_action(state, legal, self.rng)
            )
            observe(state)
        if not state.terminated and self.environment.can_terminate(state):
            state = self.environment.apply(
                state, self.environment.vocabulary.end.token_id
            )
            observe(state)
        return Rollout(state, tuple(formulas))

    def _backpropagate(
        self,
        path: Sequence[tuple[tuple, int]],
        reward: float,
    ) -> None:
        value = float(reward) if math.isfinite(reward) else self.config.invalid_reward
        for node_key, token_id in reversed(path):
            node = self.nodes[node_key]
            edge = node.edges[token_id]
            node.virtual_visits = max(0, node.virtual_visits - 1)
            edge.virtual_visits = max(0, edge.virtual_visits - 1)
            node.visits += 1
            edge.visits += 1
            edge.value_sum += value


class RewardDenseRiskMCTS(_TreeMixin):
    """Paper-style MCTS with intermediate rewards and replay trajectories."""

    def __init__(
        self,
        environment: TypedRPNEnvironment,
        reward_model: DenseRewardModel,
        *,
        config: RiskMinerConfig | None = None,
        policy: ActionPolicy | None = None,
    ) -> None:
        self.environment = environment
        self.reward_model = reward_model
        self.config = config or environment.config
        self.policy = policy or SchemaPriorPolicy()
        self.rng = random.Random(self.config.seed)
        self.nodes: dict[tuple, TreeNode] = {}
        self.archive = FormulaArchive(self.config.archive_size)

    def search(self) -> RewardDenseSearchResult:
        started = time.perf_counter()
        root = self.environment.initial_state()
        self._node(root)
        trajectories: list[PolicyTrajectory] = []
        simulations = rollouts = invalid = pool_updates = 0
        intermediate_requests = finite_scores = 0
        while simulations < self.config.simulations:
            selection = self._select_and_expand(root)
            for _ in range(self.config.rollouts_per_expansion):
                completed, requested, finite = self._complete_episode(selection)
                rollouts += 1
                intermediate_requests += requested
                finite_scores += finite
                if completed is None:
                    invalid += 1
                    self._backpropagate_dense(
                        selection.path,
                        [self.config.invalid_reward] * len(selection.path),
                        [0.0] * len(selection.path),
                    )
                    continue
                trajectory, path_returns, path_rewards, changed = completed
                trajectories.append(trajectory)
                invalid += int(trajectory.terminal_formula_key is None)
                pool_updates += int(changed)
                self._backpropagate_dense(
                    selection.path, path_returns, path_rewards
                )
            simulations += 1
        return RewardDenseSearchResult(
            self.archive.entries,
            tuple(trajectories),
            SearchMetrics(
                simulations=simulations,
                rollouts=rollouts,
                unique_formula_requests=intermediate_requests,
                finite_formula_scores=finite_scores,
                invalid_rollouts=invalid,
                tree_nodes=len(self.nodes),
                wall_seconds=time.perf_counter() - started,
                trajectories=len(trajectories),
                pool_updates=pool_updates,
                intermediate_formula_requests=intermediate_requests,
            ),
        )

    def _select_and_expand(self, root: RPNState) -> _DenseSelection:
        state = root
        path: list[tuple[tuple, int]] = []
        states: list[RPNState] = []
        legal_history: list[tuple[int, ...]] = []
        actions: list[int] = []
        resulting: list[RPNState] = []
        while True:
            key = self.environment.state_key(state)
            node = self._node(state)
            if state.terminated:
                return _DenseSelection(
                    tuple(path), tuple(states), tuple(legal_history),
                    tuple(actions), tuple(resulting), state,
                )
            legal = self.environment.legal_actions(state)
            if not legal:
                return _DenseSelection(
                    tuple(path), tuple(states), tuple(legal_history),
                    tuple(actions), tuple(resulting), state,
                )
            token_id, edge, expanded = self._choose_edge(state, legal)
            child = self.environment.apply(state, token_id)
            edge.child_key = self.environment.state_key(child)
            edge.virtual_visits += 1
            node.virtual_visits += 1
            states.append(state)
            legal_history.append(tuple(legal))
            actions.append(token_id)
            resulting.append(child)
            path.append((key, token_id))
            self._node(child)
            state = child
            if expanded:
                return _DenseSelection(
                    tuple(path), tuple(states), tuple(legal_history),
                    tuple(actions), tuple(resulting), state,
                )

    def _rollout_steps(
        self, leaf: RPNState
    ) -> tuple[
        tuple[RPNState, ...], tuple[tuple[int, ...], ...],
        tuple[int, ...], tuple[RPNState, ...],
    ]:
        state = leaf
        states: list[RPNState] = []
        legal_history: list[tuple[int, ...]] = []
        actions: list[int] = []
        resulting: list[RPNState] = []
        while not state.terminated:
            legal = self.environment.legal_actions(state)
            if not legal:
                break
            token_id = self._sample_rollout_action(state, legal, self.rng)
            child = self.environment.apply(state, token_id)
            states.append(state)
            legal_history.append(tuple(legal))
            actions.append(token_id)
            resulting.append(child)
            state = child
        if not state.terminated and self.environment.can_terminate(state):
            legal = self.environment.legal_actions(state)
            end_id = self.environment.vocabulary.end.token_id
            if end_id in legal:
                child = self.environment.apply(state, end_id)
                states.append(state)
                legal_history.append(tuple(legal))
                actions.append(end_id)
                resulting.append(child)
        return tuple(states), tuple(legal_history), tuple(actions), tuple(resulting)

    def _complete_episode(
        self,
        selection: _DenseSelection,
    ) -> tuple[
        tuple[PolicyTrajectory, list[float], list[float], bool] | None,
        int,
        int,
    ]:
        roll_states, roll_legal, roll_actions, roll_resulting = self._rollout_steps(
            selection.leaf
        )
        states = selection.states + roll_states
        legal_history = selection.legal_actions + roll_legal
        actions = selection.actions + roll_actions
        resulting = selection.resulting_states + roll_resulting
        if not actions:
            return None, 0, 0
        end_id = self.environment.vocabulary.end.token_id
        if (
            not resulting
            or not resulting[-1].terminated
            or actions[-1] != end_id
        ):
            # Keep dead-end/max-length episodes in replay.  They are exactly
            # the low-reward trajectories that the risk-seeking update should
            # learn to suppress, rather than silently dropping them.
            step_rewards = [0.0] * len(actions)
            step_rewards[-1] = float(self.config.invalid_reward)
            returns = [0.0] * len(step_rewards)
            running = 0.0
            for index in range(len(step_rewards) - 1, -1, -1):
                running = (
                    step_rewards[index] + self.config.discount * running
                )
                returns[index] = running
            trajectory = PolicyTrajectory(
                states=tuple(state.token_ids for state in states),
                actions=tuple(actions),
                legal_actions=tuple(legal_history),
                reward=float(returns[0]),
                step_rewards=tuple(step_rewards),
                terminal_formula_key=None,
                terminal_formula_rpn=None,
                pool_changed=False,
            )
            return (
                trajectory,
                returns[:len(selection.path)],
                step_rewards[:len(selection.path)],
                False,
            ), 0, 0

        observations: dict[tuple, StackValue] = {}
        step_values: list[StackValue | None] = []
        for action, state in zip(actions, resulting):
            if action == end_id:
                step_values.append(None)
                continue
            value = self.environment.formula_value(state)
            step_values.append(value)
            if value is not None:
                observations.setdefault(value.canonical_key, value)
        intermediate = (
            dict(self.reward_model.intermediate_rewards(tuple(observations.values())))
            if observations else {}
        )
        finite_count = sum(math.isfinite(value) for value in intermediate.values())
        step_rewards: list[float] = []
        for action, state, value in zip(actions, resulting, step_values):
            if action == end_id or value is None:
                step_rewards.append(0.0)
                continue
            reward = float(intermediate.get(value.canonical_key, 0.0))
            reward = reward if math.isfinite(reward) else 0.0
            step_rewards.append(reward)
            self.archive.add(
                ArchiveEntry(
                    value.expr, reward, value.depth, value.canonical_key,
                    self._render_token_ids(state.token_ids),
                )
            )

        terminal_parent = states[-1]
        terminal_value = self.environment.formula_value(terminal_parent)
        if terminal_value is None:
            return None, len(observations), finite_count
        rpn = self._render_token_ids(terminal_parent.token_ids)
        individual_score = float(intermediate.get(terminal_value.canonical_key, 0.0))
        terminal = self.reward_model.terminal_reward(
            terminal_value, rpn=rpn, individual_score=individual_score
        )
        terminal_reward = float(terminal.reward)
        if not math.isfinite(terminal_reward):
            terminal_reward = self.config.invalid_reward
        step_rewards[-1] = terminal_reward
        self.archive.add(
            ArchiveEntry(
                terminal_value.expr, terminal_reward, terminal_value.depth,
                terminal_value.canonical_key, rpn,
            )
        )

        returns = [0.0] * len(step_rewards)
        running = 0.0
        for index in range(len(step_rewards) - 1, -1, -1):
            running = step_rewards[index] + self.config.discount * running
            returns[index] = running
        total_reward = returns[0] if returns else self.config.invalid_reward
        trajectory = PolicyTrajectory(
            states=tuple(state.token_ids for state in states),
            actions=tuple(actions),
            legal_actions=tuple(legal_history),
            reward=float(total_reward),
            step_rewards=tuple(step_rewards),
            terminal_formula_key=terminal_value.canonical_key,
            terminal_formula_rpn=rpn,
            pool_changed=bool(terminal.transition.committed),
        )
        return (
            trajectory,
            returns[:len(selection.path)],
            step_rewards[:len(selection.path)],
            bool(terminal.transition.committed),
        ), len(observations), finite_count

    def _backpropagate_dense(
        self,
        path: Sequence[tuple[tuple, int]],
        path_returns: Sequence[float],
        path_rewards: Sequence[float],
    ) -> None:
        if not (len(path) == len(path_returns) == len(path_rewards)):
            raise ValueError("one immediate and cumulative reward is required per selected edge")
        for (node_key, token_id), value, immediate in zip(
            path, path_returns, path_rewards
        ):
            node = self.nodes[node_key]
            edge = node.edges[token_id]
            node.virtual_visits = max(0, node.virtual_visits - 1)
            edge.virtual_visits = max(0, edge.virtual_visits - 1)
            node.visits += 1
            edge.visits += 1
            edge.reward = float(immediate)
            edge.value_sum += (
                float(value) if math.isfinite(value) else self.config.invalid_reward
            )


__all__ = [
    "ActionPolicy", "ArchiveEntry", "CandidateEvaluator", "DenseRewardModel",
    "EdgeStats", "FormulaArchive", "RewardDenseRiskMCTS",
    "RewardDenseSearchResult", "RiskMCTS", "RiskMinerSearchResult",
    "SchemaPriorPolicy", "SearchMetrics", "TreeNode",
]
