from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from statistics import mean
from typing import Protocol, Sequence

from trading_dsl_engine.base.parser import Expr

from flows.riskminer.policy import PolicyEpisode, RiskSeekingTokenPolicy
from flows.riskminer.rpn import RPNEnvironment, RPNState, StackValue, Token, TokenKind


class CandidateEvaluator(Protocol):
    def score_batch(self, candidates: Sequence[Expr]) -> list[float]: ...


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 512
    rollouts_per_expansion: int = 16
    selection_batch_size: int = 16
    exploration: float = 1.4
    progressive_k: float = 4.0
    progressive_alpha: float = 0.5
    archive_size: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        if min(self.simulations, self.rollouts_per_expansion, self.selection_batch_size, self.archive_size) <= 0:
            raise ValueError("simulation, rollout, batch, and archive sizes must be positive")
        if self.exploration < 0.0 or self.progressive_k <= 0.0:
            raise ValueError("invalid exploration/progressive-widening parameters")
        if not 0.0 < self.progressive_alpha <= 1.0:
            raise ValueError("progressive_alpha must be in (0, 1]")


@dataclass
class EdgeStats:
    token_id: int
    child_key: tuple
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    virtual_visits: int = 0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class TreeNode:
    state: RPNState
    visits: int = 0
    edges: dict[int, EdgeStats] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateRecord:
    expr: Expr
    score: float
    depth: int
    node_count: int
    token_ids: tuple[int, ...]
    canonical_key: str


@dataclass(frozen=True)
class SearchReport:
    candidates: tuple[CandidateRecord, ...]
    simulations: int
    rollout_proposals: int
    finite_proposals: int
    dead_rollouts: int
    tree_nodes: int
    policy_quantile: float


@dataclass
class _RolloutProposal:
    value: StackValue
    token_ids: tuple[int, ...]


@dataclass
class _SimulationPlan:
    path: list[tuple[tuple, int]]
    visited_nodes: list[tuple]
    proposals: list[_RolloutProposal]


class RiskMinerMCTS:
    def __init__(self, environment: RPNEnvironment, evaluator: CandidateEvaluator, *, policy: RiskSeekingTokenPolicy | None = None, config: MCTSConfig = MCTSConfig()) -> None:
        self.environment = environment
        self.evaluator = evaluator
        self.config = config
        self.rng = random.Random(config.seed)
        self.policy = policy or RiskSeekingTokenPolicy(seed=config.seed)
        self.nodes: dict[tuple, TreeNode] = {}
        self.archive: dict[str, CandidateRecord] = {}
        self.rollout_proposals = 0
        self.finite_proposals = 0
        self.dead_rollouts = 0

    def search(self) -> SearchReport:
        root_key = self._remember(self.environment.initial_state())
        completed = 0
        while completed < self.config.simulations:
            batch_size = min(self.config.selection_batch_size, self.config.simulations - completed)
            plans = [self._plan_simulation(root_key) for _ in range(batch_size)]
            self._evaluate_and_backup(plans)
            completed += batch_size
        candidates = tuple(sorted(self.archive.values(), key=lambda item: (-item.score, item.depth, item.canonical_key))[: self.config.archive_size])
        return SearchReport(candidates, completed, self.rollout_proposals, self.finite_proposals, self.dead_rollouts, len(self.nodes), self.policy.quantile_value)

    def _remember(self, state: RPNState) -> tuple:
        key = self.environment.state_key(state)
        self.nodes.setdefault(key, TreeNode(state))
        return key

    def _plan_simulation(self, root_key: tuple) -> _SimulationPlan:
        node_key = root_key
        path: list[tuple[tuple, int]] = []
        visited = [node_key]
        while True:
            node = self.nodes[node_key]
            state = node.state
            if state.terminated:
                break
            legal = self.environment.legal_tokens(state)
            if not legal:
                break
            priors = self.policy.priors(state, legal)
            widening_limit = min(len(legal), max(1, int(self.config.progressive_k * max(1, node.visits + sum(edge.virtual_visits for edge in node.edges.values())) ** self.config.progressive_alpha)))
            permitted = sorted(legal, key=lambda token: (-priors[token.token_id], token.token_id))[:widening_limit]
            unexpanded = [token for token in permitted if token.token_id not in node.edges]
            if unexpanded:
                token = unexpanded[0]
                child_key = self._remember(self.environment.step(state, token.token_id))
                node.edges[token.token_id] = EdgeStats(token.token_id, child_key, priors[token.token_id], virtual_visits=1)
                path.append((node_key, token.token_id))
                node_key = child_key
                visited.append(node_key)
                break
            total = max(1, node.visits + sum(edge.virtual_visits for edge in node.edges.values()))
            def puct(token: Token) -> tuple[float, int]:
                edge = node.edges[token.token_id]
                exploration = self.config.exploration * edge.prior * math.sqrt(total) / (1 + edge.visits + edge.virtual_visits)
                return edge.mean_value + exploration, -token.token_id
            token = max(permitted, key=puct)
            edge = node.edges[token.token_id]
            edge.virtual_visits += 1
            path.append((node_key, token.token_id))
            node_key = edge.child_key
            visited.append(node_key)
        leaf = self.nodes[node_key].state
        proposals: list[_RolloutProposal] = []
        for _ in range(self.config.rollouts_per_expansion):
            proposals.extend(self._rollout(leaf))
        if not proposals:
            self.dead_rollouts += 1
        return _SimulationPlan(path, visited, proposals)

    def _rollout(self, start: RPNState) -> list[_RolloutProposal]:
        state = start
        seen: set[str] = set()
        proposals: list[_RolloutProposal] = []
        def record(current: RPNState) -> None:
            value = self.environment.candidate(current)
            if value is not None and value.canonical_key not in seen:
                seen.add(value.canonical_key)
                proposals.append(_RolloutProposal(value, current.token_ids))
        record(state)
        guard = 0
        while not state.terminated and guard <= self.environment.max_tokens:
            legal = self.environment.legal_tokens(state)
            if not legal:
                break
            candidate = self.environment.candidate(state)
            if candidate is not None:
                end = [token for token in legal if token.kind is TokenKind.END]
                if end and (len(state.token_ids) >= self.environment.max_tokens - 2 or self.rng.random() < 0.28):
                    token = end[0]
                else:
                    token = self._sample_rollout_token(state, legal)
            else:
                token = self._sample_rollout_token(state, legal)
            state = self.environment.step(state, token.token_id)
            record(state)
            guard += 1
        return proposals

    def _sample_rollout_token(self, state: RPNState, legal: Sequence[Token]) -> Token:
        reducing = [token for token in legal if token.kind is TokenKind.OPERATOR and token.operator is not None and token.operator.arity >= 2]
        if len(state.stack) > 1 and reducing and self.rng.random() < 0.72:
            return self.policy.sample(state, reducing)
        operators = [token for token in legal if token.kind is TokenKind.OPERATOR]
        if len(state.stack) == 1 and operators and self.rng.random() < 0.58:
            return self.policy.sample(state, operators)
        return self.policy.sample(state, legal)

    def _evaluate_and_backup(self, plans: Sequence[_SimulationPlan]) -> None:
        unique: dict[str, Expr] = {}
        for plan in plans:
            for proposal in plan.proposals:
                unique.setdefault(proposal.value.canonical_key, proposal.value.expr)
        keys = list(unique)
        scores = self.evaluator.score_batch([unique[key] for key in keys]) if keys else []
        score_by_key = dict(zip(keys, scores))
        policy_episodes: list[PolicyEpisode] = []
        for plan in plans:
            rewards: list[float] = []
            for proposal in plan.proposals:
                self.rollout_proposals += 1
                score = float(score_by_key.get(proposal.value.canonical_key, -math.inf))
                if not math.isfinite(score):
                    continue
                self.finite_proposals += 1
                rewards.append(score)
                record = CandidateRecord(proposal.value.expr, score, proposal.value.depth, proposal.value.node_count, proposal.token_ids, proposal.value.canonical_key)
                previous = self.archive.get(record.canonical_key)
                if previous is None or (record.score, -record.depth) > (previous.score, -previous.depth):
                    self.archive[record.canonical_key] = record
                policy_episodes.append(PolicyEpisode(proposal.token_ids, score))
            reward = mean(rewards) if rewards else -1e6
            for visited_key in plan.visited_nodes:
                self.nodes[visited_key].visits += 1
            for parent_key, token_id in plan.path:
                edge = self.nodes[parent_key].edges[token_id]
                edge.virtual_visits = max(0, edge.virtual_visits - 1)
                edge.visits += 1
                edge.value_sum += reward
        self.policy.update(policy_episodes)


__all__ = ["CandidateRecord", "MCTSConfig", "RiskMinerMCTS", "SearchReport"]
