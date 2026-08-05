from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
import random
from typing import Any, Literal

from trading_dsl_engine.base.dsl import (
    abs as dsl_abs,
    add, arctan, clip, cumsum, diff, div, ewm, ewm_kurtosis,
    ewm_skewness, ewm_std, ewm_var, exp, ffill, fillna, fraction,
    ge, gt, le, ln, lt, mul, purify, rolling_argmax, rolling_argmin,
    rolling_max, rolling_mean, rolling_median, rolling_min,
    rolling_pct_rank, rolling_quantile, rolling_std, rolling_sum,
    rolling_theilsen, shift, sign, sub, var, where, xs_norm, xs_pct_rank,
    xs_rank,
)
from trading_dsl_engine.base.parser import Expr

Shape = Literal["scalar", "row", "matrix", "boolean", "unknown"]
Fitness = Callable[[Expr], float]


@dataclass(frozen=True)
class SemanticInfo:
    """Weak semantic facts used only to prune nonsensical programs.

    A value may carry several types simultaneously.  This deliberately does
    not depend on the unit/dimension checker.
    """

    types: frozenset[str] = frozenset({"numeric"})
    shape: Shape = "row"
    lower: float = -math.inf
    upper: float = math.inf
    integer: bool = False

    def intersects(self, other: "SemanticInfo") -> bool:
        ignored = {"numeric", "finite", "signed", "nonnegative"}
        lhs = self.types - ignored
        rhs = other.types - ignored
        return not lhs or not rhs or bool(lhs & rhs)

    def with_types(self, *types: str, shape: Shape | None = None) -> "SemanticInfo":
        return replace(self, types=frozenset(types), shape=self.shape if shape is None else shape)


@dataclass(frozen=True)
class Hole:
    required_types: frozenset[str] = frozenset()
    shape: Shape = "row"
    positive: bool = False
    integer: bool = False
    lower: float = -math.inf
    upper: float = math.inf
    max_depth: int = 4
    role: str = "value"

    def accepts(self, info: SemanticInfo) -> bool:
        if self.shape not in ("unknown", info.shape) and info.shape != "unknown":
            return False
        if self.required_types and not (self.required_types & info.types):
            return False
        if self.positive and info.lower <= 0.0:
            return False
        if self.integer and not info.integer:
            return False
        return info.lower >= self.lower and info.upper <= self.upper


@dataclass(frozen=True)
class SExpr:
    op: str | None = None
    children: tuple["SExpr", ...] = ()
    terminal: Expr | None = None
    info: SemanticInfo | None = None
    hole: Hole | None = None

    @staticmethod
    def unresolved(hole: Hole) -> "SExpr":
        return SExpr(hole=hole)

    @property
    def complete(self) -> bool:
        return self.hole is None and all(child.complete for child in self.children)

    def first_hole_path(self) -> tuple[int, ...] | None:
        if self.hole is not None:
            return ()
        for i, child in enumerate(self.children):
            suffix = child.first_hole_path()
            if suffix is not None:
                return (i,) + suffix
        return None

    def at(self, path: tuple[int, ...]) -> "SExpr":
        node = self
        for index in path:
            node = node.children[index]
        return node

    def replace_at(self, path: tuple[int, ...], node: "SExpr") -> "SExpr":
        if not path:
            return node
        i = path[0]
        children = list(self.children)
        children[i] = children[i].replace_at(path[1:], node)
        return replace(self, children=tuple(children))

    def key(self) -> str:
        if self.hole is not None:
            h = self.hole
            return f"?{h.role}:{sorted(h.required_types)}:{h.shape}:{h.max_depth}"
        if self.terminal is not None:
            return repr(self.terminal)
        return f"{self.op}({','.join(child.key() for child in self.children)})"


@dataclass(frozen=True)
class OperatorSchema:
    name: str
    build: Callable[..., Expr]
    family: str
    arity: int
    weight: float = 1.0
    min_depth: int = 1


@dataclass(frozen=True)
class Action:
    label: str
    replacement: SExpr
    prior: float


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 2000
    max_depth: int = 5
    exploration: float = 1.4
    prior_weight: float = 0.8
    progressive_k: float = 3.0
    progressive_alpha: float = 0.5
    rollout_candidates: int = 8
    seed: int = 0
    target_types: frozenset[str] = frozenset({"dimensionless"})
    dynamic_parameters: bool = True
    min_span: float = 2.0
    max_span: float = 10080.0
    min_lag: int = 1
    max_lag: int = 1440


@dataclass
class NodeStats:
    visits: int = 0
    value_sum: float = 0.0
    actions: list[Action] = field(default_factory=list)
    children: dict[str, str] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchResult:
    expr: Expr
    sharpe: float
    tree: SExpr
    simulations: int


class AlphaMCTS:
    """Grammar-guided MCTS over partially filled expression trees."""

    def __init__(
        self,
        terminals: Mapping[str, tuple[Expr, SemanticInfo]],
        fitness: Fitness,
        *,
        operators: Sequence[OperatorSchema] | None = None,
        literals: Sequence[float] = (0.0, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 240.0, 1440.0),
        config: SearchConfig = SearchConfig(),
    ) -> None:
        self.terminals = dict(terminals)
        self.fitness = fitness
        self.operators = tuple(operators or default_operator_schemas())
        self.literals = tuple(float(x) for x in literals)
        self.config = config
        self.rng = random.Random(config.seed)
        self.stats: dict[str, NodeStats] = defaultdict(NodeStats)
        self.states: dict[str, SExpr] = {}
        self.cache: dict[str, float] = {}
        self.best: SearchResult | None = None

    def search(self) -> SearchResult:
        root = SExpr.unresolved(Hole(
            required_types=self.config.target_types,
            shape="row",
            max_depth=self.config.max_depth,
            role="alpha",
        ))
        self.states[root.key()] = root
        for _ in range(self.config.simulations):
            path, leaf = self._select(root)
            completed = self._rollout(leaf)
            reward = self._evaluate(completed)
            self._backup(path, reward)
            if completed.complete and math.isfinite(reward):
                expr = compile_sexpr(completed)
                if self.best is None or reward > self.best.sharpe:
                    self.best = SearchResult(expr, reward, completed, self.config.simulations)
        if self.best is None:
            raise RuntimeError("MCTS did not produce a finite candidate")
        return self.best

    def _select(self, root: SExpr) -> tuple[list[str], SExpr]:
        state = root
        path: list[str] = []
        while True:
            key = state.key()
            path.append(key)
            self.states[key] = state
            if state.complete:
                return path, state
            stats = self.stats[key]
            all_actions = self._actions(state)
            limit = max(1, int(self.config.progressive_k * max(1, stats.visits) ** self.config.progressive_alpha))
            known = {a.label for a in stats.actions}
            for action in all_actions:
                if len(stats.actions) >= min(limit, len(all_actions)):
                    break
                if action.label not in known:
                    stats.actions.append(action)
                    known.add(action.label)
            unexpanded = [a for a in stats.actions if a.label not in stats.children]
            if unexpanded:
                action = self._weighted_choice(unexpanded)
                child = self._apply(state, action)
                child_key = child.key()
                stats.children[action.label] = child_key
                self.states[child_key] = child
                path.append(child_key)
                return path, child
            if not stats.actions:
                return path, state
            log_n = math.log(max(2, stats.visits))
            def score(action: Action) -> float:
                child_stats = self.stats[stats.children[action.label]]
                q = child_stats.mean
                u = self.config.exploration * math.sqrt(log_n / (1 + child_stats.visits))
                p = self.config.prior_weight * action.prior / (1 + child_stats.visits)
                return q + u + p
            action = max(stats.actions, key=score)
            state = self.states[stats.children[action.label]]

    def _rollout(self, state: SExpr) -> SExpr:
        best = state
        for _ in range(self.config.rollout_candidates):
            candidate = state
            guard = 0
            while not candidate.complete and guard < 256:
                actions = self._actions(candidate)
                if not actions:
                    break
                candidate = self._apply(candidate, self._weighted_choice(actions))
                guard += 1
            if candidate.complete:
                return candidate
            best = candidate
        return best

    def _evaluate(self, tree: SExpr) -> float:
        if not tree.complete:
            return -math.inf
        key = canonical_key(tree)
        if key not in self.cache:
            try:
                self.cache[key] = float(self.fitness(compile_sexpr(tree)))
            except Exception:
                self.cache[key] = -math.inf
        return self.cache[key]

    def _backup(self, path: Sequence[str], reward: float) -> None:
        value = reward if math.isfinite(reward) else -1e6
        for key in path:
            stats = self.stats[key]
            stats.visits += 1
            stats.value_sum += value

    def _actions(self, state: SExpr) -> list[Action]:
        path = state.first_hole_path()
        if path is None:
            return []
        hole = state.at(path).hole
        assert hole is not None
        actions: list[Action] = []
        for name, (expr, info) in self.terminals.items():
            if hole.accepts(info):
                actions.append(Action(f"terminal:{name}", SExpr(terminal=expr, info=info), 2.0))
        for value in self.literals:
            info = SemanticInfo(
                types=frozenset({"numeric", "dimensionless", "parameter"}),
                shape="scalar", lower=value, upper=value, integer=value.is_integer(),
            )
            if hole.accepts(info):
                actions.append(Action(f"literal:{value:g}", SExpr(terminal=_literal(value), info=info), 0.7))
        if hole.max_depth <= 0:
            return actions
        for schema in self.operators:
            replacement = instantiate_schema(schema, hole, self.config)
            if replacement is not None:
                actions.append(Action(f"op:{schema.name}:{replacement.key()}", replacement, schema.weight))
        actions.sort(key=lambda a: (-a.prior, a.label))
        return actions

    def _apply(self, state: SExpr, action: Action) -> SExpr:
        path = state.first_hole_path()
        assert path is not None
        return state.replace_at(path, action.replacement)

    def _weighted_choice(self, actions: Sequence[Action]) -> Action:
        total = sum(max(0.0, a.prior) for a in actions)
        if total <= 0.0:
            return self.rng.choice(list(actions))
        target = self.rng.random() * total
        cumulative = 0.0
        for action in actions:
            cumulative += max(0.0, action.prior)
            if cumulative >= target:
                return action
        return actions[-1]


def _literal(value: float) -> Expr:
    from trading_dsl_engine.base.dsl import ensure_expr
    return ensure_expr(value)


def instantiate_schema(schema: OperatorSchema, out: Hole, cfg: SearchConfig) -> SExpr | None:
    d = out.max_depth - 1
    value = Hole(shape=out.shape, max_depth=d, role="value")
    same = Hole(required_types=out.required_types, shape=out.shape, max_depth=d, role="same_type")
    dimless = Hole(required_types=frozenset({"dimensionless"}), shape=out.shape, max_depth=d, role="dimensionless")
    positive_scalar = Hole(required_types=frozenset({"dimensionless", "parameter"}), shape="scalar", positive=True,
                           lower=cfg.min_span, upper=cfg.max_span, max_depth=d if cfg.dynamic_parameters else 0, role="span")
    positive_int = Hole(required_types=frozenset({"dimensionless", "parameter"}), shape="scalar", positive=True, integer=True,
                        lower=cfg.min_lag, upper=cfg.max_lag, max_depth=d if cfg.dynamic_parameters else 0, role="lag")
    probability = Hole(required_types=frozenset({"dimensionless", "parameter"}), shape="scalar",
                       lower=0.0, upper=1.0, max_depth=d if cfg.dynamic_parameters else 0, role="probability")
    boolean = Hole(required_types=frozenset({"boolean"}), shape=out.shape, max_depth=d, role="condition")

    families: dict[str, tuple[Hole, ...]] = {
        "unary_same": (same,),
        "unary_dimensionless": (value,),
        "binary_same": (same, same),
        "binary_numeric": (value, value),
        "comparison": (value, value),
        "where": (boolean, same, same),
        "ewm": (same, positive_scalar),
        "rolling": (same, positive_int),
        "rolling_q": (same, positive_int, probability),
        "shift": (same, positive_int),
        "diff": (same, positive_int),
        "clip": (same, same, same),
    }
    holes = families.get(schema.family)
    if holes is None:
        return None
    if schema.family == "comparison" and "boolean" not in out.required_types:
        return None
    if schema.family == "unary_dimensionless" and out.required_types and "dimensionless" not in out.required_types:
        return None
    info = SemanticInfo(types=out.required_types or frozenset({"numeric"}), shape=out.shape)
    if schema.family == "comparison":
        info = SemanticInfo(frozenset({"boolean", "dimensionless"}), out.shape, 0.0, 1.0, True)
    elif schema.family == "unary_dimensionless":
        info = SemanticInfo(frozenset({"dimensionless", "numeric"}), out.shape)
    return SExpr(op=schema.name, children=tuple(SExpr.unresolved(h) for h in holes), info=info)


def compile_sexpr(node: SExpr) -> Expr:
    if node.hole is not None:
        raise ValueError("cannot compile an incomplete expression")
    if node.terminal is not None:
        return node.terminal
    args = [compile_sexpr(child) for child in node.children]
    schema = _SCHEMA_BY_NAME[node.op or ""]
    return schema.build(*args)


def canonical_key(node: SExpr) -> str:
    if node.terminal is not None:
        return repr(node.terminal)
    children = [canonical_key(child) for child in node.children]
    if node.op in {"add", "mul"}:
        children.sort()
    if node.op in {"xs_rank", "xs_pct_rank", "xs_norm", "purify", "ffill"} and node.children and node.children[0].op == node.op:
        return canonical_key(node.children[0])
    return f"{node.op}({','.join(children)})"


def default_operator_schemas() -> tuple[OperatorSchema, ...]:
    schemas = (
        OperatorSchema("add", add, "binary_same", 2, 1.4),
        OperatorSchema("sub", sub, "binary_same", 2, 1.3),
        OperatorSchema("mul", mul, "binary_numeric", 2, 1.1),
        OperatorSchema("div", div, "binary_numeric", 2, 1.1),
        OperatorSchema("abs", dsl_abs, "unary_same", 1, 0.8),
        OperatorSchema("sign", sign, "unary_dimensionless", 1, 0.8),
        OperatorSchema("fraction", fraction, "unary_dimensionless", 1, 0.7),
        OperatorSchema("arctan", arctan, "unary_dimensionless", 1, 0.7),
        OperatorSchema("exp", exp, "unary_dimensionless", 1, 0.5),
        OperatorSchema("ln", ln, "unary_dimensionless", 1, 0.5),
        OperatorSchema("purify", purify, "unary_same", 1, 1.0),
        OperatorSchema("ffill", ffill, "unary_same", 1, 0.8),
        OperatorSchema("fillna", lambda x, y: fillna(x, y), "binary_same", 2, 0.7),
        OperatorSchema("xs_rank", xs_rank, "unary_dimensionless", 1, 1.7),
        OperatorSchema("xs_pct_rank", xs_pct_rank, "unary_dimensionless", 1, 1.5),
        OperatorSchema("xs_norm", xs_norm, "unary_dimensionless", 1, 1.4),
        OperatorSchema("cumsum", cumsum, "unary_same", 1, 0.6),
        OperatorSchema("ewm", lambda x, s: ewm(x, span=s), "ewm", 2, 1.8),
        OperatorSchema("ewm_std", lambda x, s: ewm_std(x, span=s), "ewm", 2, 1.3),
        OperatorSchema("ewm_var", lambda x, s: ewm_var(x, span=s), "ewm", 2, 1.1),
        OperatorSchema("ewm_skewness", lambda x, s: ewm_skewness(x, span=s), "ewm", 2, 0.7),
        OperatorSchema("ewm_kurtosis", lambda x, s: ewm_kurtosis(x, span=s), "ewm", 2, 0.6),
        OperatorSchema("shift", lambda x, n: shift(x, n), "shift", 2, 1.1),
        OperatorSchema("diff", lambda x, n: diff(x, n), "diff", 2, 1.0),
        OperatorSchema("rolling_sum", rolling_sum, "rolling", 2, 0.8),
        OperatorSchema("rolling_mean", rolling_mean, "rolling", 2, 1.0),
        OperatorSchema("rolling_std", rolling_std, "rolling", 2, 0.9),
        OperatorSchema("rolling_min", rolling_min, "rolling", 2, 0.6),
        OperatorSchema("rolling_max", rolling_max, "rolling", 2, 0.6),
        OperatorSchema("rolling_median", rolling_median, "rolling", 2, 0.6),
        OperatorSchema("rolling_pct_rank", rolling_pct_rank, "rolling", 2, 0.8),
        OperatorSchema("rolling_argmin", rolling_argmin, "rolling", 2, 0.5),
        OperatorSchema("rolling_argmax", rolling_argmax, "rolling", 2, 0.5),
        OperatorSchema("rolling_theilsen", rolling_theilsen, "rolling", 2, 0.4),
        OperatorSchema("rolling_quantile", rolling_quantile, "rolling_q", 3, 0.7),
        OperatorSchema("clip", clip, "clip", 3, 0.7),
        OperatorSchema("lt", lt, "comparison", 2, 0.5),
        OperatorSchema("le", le, "comparison", 2, 0.5),
        OperatorSchema("gt", gt, "comparison", 2, 0.5),
        OperatorSchema("ge", ge, "comparison", 2, 0.5),
        OperatorSchema("where", where, "where", 3, 0.7),
    )
    return schemas


_SCHEMA_BY_NAME = {schema.name: schema for schema in default_operator_schemas()}


def market_terminal_semantics(fields: Mapping[str, Mapping[str, object]]) -> dict[str, tuple[Expr, SemanticInfo]]:
    out: dict[str, tuple[Expr, SemanticInfo]] = {}
    for name, spec in fields.items():
        types = frozenset(str(x) for x in spec.get("types", ())) | frozenset({"numeric"})
        lower, upper = _parse_range(spec.get("range", "real"))
        shape: Shape = "boolean" if "boolean" in types or "boolean_0_1" in types else "row"
        out[name] = (var(name), SemanticInfo(types, shape, lower, upper, "integer" in types))
    return out


def _parse_range(value: object) -> tuple[float, float]:
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]), float(value[1])
    if value == ">0":
        return math.nextafter(0.0, 1.0), math.inf
    if value == ">=0":
        return 0.0, math.inf
    if value == "boolean":
        return 0.0, 1.0
    return -math.inf, math.inf


__all__ = [
    "Action", "AlphaMCTS", "Hole", "OperatorSchema", "SearchConfig",
    "SearchResult", "SemanticInfo", "SExpr", "canonical_key",
    "compile_sexpr", "default_operator_schemas", "market_terminal_semantics",
]
