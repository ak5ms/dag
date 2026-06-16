from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import random
from typing import Any, TypeAlias

from deap import base as deap_base
from deap import tools

from trading_dsl_engine.base.dsl import abs as dsl_abs
from trading_dsl_engine.base.dsl import add, and_, cumsum, div, ewm, fillna, ffill, mean, mul, shift, where, xs_rank
from trading_dsl_engine.base.dsl import einsum, get_beta, Ridge, ensure_expr, var
from trading_dsl_engine.base.metadata import MetadataConfig, UnitInfo, analyze_formula_metadata
from trading_dsl_engine.base.parser import Call, Expr, Identifier, Number

AlphaKind: TypeAlias = str
VECTOR: AlphaKind = "vector"
SCALAR: AlphaKind = "scalar"
POSITIVE_SCALAR: AlphaKind = "positive_scalar"
BOOLEAN: AlphaKind = "boolean"

Objective = Callable[[Expr, Sequence[Expr]], float]
ExprFactory = Callable[..., Expr]
ExprPredicate = Callable[[Expr], bool]
AdditivePredicate = Callable[[Expr, Sequence[Expr], float], bool]


def ewm_var(x: Expr, span: Expr | float, min_periods: Expr | float | None = None) -> Expr:
    if min_periods is None:
        min_periods = 5 * span
    is_valid = cumsum(fillna(x == x, 0.0)) > (min_periods - 1.0)
    out = ewm(x * x, span) - ewm(x, span) ** 2
    return where(and_(is_valid, out > 0.0), out, float("nan"))


def ewm_std(x: Expr, span: Expr | float, min_periods: Expr | float | None = None) -> Expr:
    return ewm_var(x, span, min_periods) ** 0.5


def default_alpha_pnl(alpha: Expr, *, roll_rets: Expr, is_tradable: Expr, hl: Expr | float) -> Expr:
    w = alpha / ewm_var(roll_rets, hl)
    masked = ffill(where(is_tradable, w, float("nan")))
    return einsum(shift(masked, 1.0, 1.0) * roll_rets, "n->")


def default_sharpe_objective(alpha: Expr, *, roll_rets: Expr, is_tradable: Expr, hl: Expr | float) -> Expr:
    pnl = default_alpha_pnl(alpha, roll_rets=roll_rets, is_tradable=is_tradable, hl=hl)
    return mean(pnl) / ewm_std(pnl, hl)


def ridge_pool_alpha_pnl(
    alpha: Expr,
    pool: Sequence[Expr],
    *,
    roll_rets: Expr,
    hs: Expr,
    is_tradable: Expr,
    hl: Expr | float,
    ridge_hl: Expr | float = 1440.0 * 5.0,
    ridge_lambda: Expr | float = 0.0,
) -> Expr:
    alphas = tuple(pool) + (alpha,)
    reg = Ridge(*alphas, roll_rets, fillna(hs**-2.0, 0.0), ridge_hl, ridge_lambda)
    yhat = einsum(shift(get_beta(reg), 1.0, 1.0), alpha, "f,fn->n")
    w = yhat / (ewm_std(roll_rets, hl) ** 2.0)
    masked = ffill(where(is_tradable, w, float("nan")))
    return einsum(shift(masked, 1.0, 1.0) * roll_rets, "n->")


def dimensionless_filter(config: MetadataConfig | dict[str, Any] | None = None, *, allow_unknown: bool = False) -> ExprPredicate:
    def _predicate(expr: Expr) -> bool:
        units = analyze_formula_metadata(expr, config).get_units()
        return (allow_unknown and units.is_unknown()) or units == UnitInfo.dimensionless()

    return _predicate


@dataclass(frozen=True)
class AlphaTerminal:
    name: str
    expr: Expr
    kind: AlphaKind = VECTOR


@dataclass(frozen=True)
class AlphaPrimitive:
    name: str
    input_kinds: tuple[AlphaKind, ...]
    output_kind: AlphaKind
    fn: ExprFactory

    @property
    def arity(self) -> int:
        return len(self.input_kinds)


@dataclass(frozen=True)
class _TypedNode:
    expr: Expr
    kind: AlphaKind


@dataclass
class AlphaPrimitiveSet:
    terminals: dict[AlphaKind, list[AlphaTerminal]] = field(default_factory=dict)
    primitives: dict[AlphaKind, list[AlphaPrimitive]] = field(default_factory=dict)

    @classmethod
    def default(
        cls,
        features: Iterable[Expr | str],
        *,
        halflives: Iterable[Expr | int | float] = (5.0, 30.0, 120.0, 1440.0),
        scalars: Iterable[Expr | int | float] = (),
    ) -> "AlphaPrimitiveSet":
        pset = cls()
        for feature in features:
            pset.add_terminal(feature, VECTOR)
        for halflife in halflives:
            pset.add_terminal(_positive_scalar_expr(halflife), POSITIVE_SCALAR)
        for scalar in scalars:
            pset.add_terminal(_scalar_expr(scalar), SCALAR)
        pset.add_primitives(
            AlphaPrimitive("add", (VECTOR, VECTOR), VECTOR, add),
            AlphaPrimitive("sub", (VECTOR, VECTOR), VECTOR, lambda a, b: a - b),
            AlphaPrimitive("mul", (VECTOR, VECTOR), VECTOR, mul),
            AlphaPrimitive("div", (VECTOR, VECTOR), VECTOR, div),
            AlphaPrimitive("abs", (VECTOR,), VECTOR, dsl_abs),
            AlphaPrimitive("xs_rank", (VECTOR,), VECTOR, xs_rank),
            AlphaPrimitive("ewm", (VECTOR, POSITIVE_SCALAR), VECTOR, ewm),
            AlphaPrimitive("shift", (VECTOR, POSITIVE_SCALAR), VECTOR, lambda x, n: shift(x, n, n)),
        )
        return pset

    @classmethod
    def from_groups(
        cls,
        terminals: Mapping[AlphaKind, Iterable[Expr | str | int | float | AlphaTerminal]],
        primitives: Iterable[AlphaPrimitive] = (),
    ) -> "AlphaPrimitiveSet":
        pset = cls()
        for kind, values in terminals.items():
            for value in values:
                if isinstance(value, AlphaTerminal):
                    pset.add_terminal(value.expr, value.kind, name=value.name)
                elif kind == POSITIVE_SCALAR:
                    pset.add_terminal(_positive_scalar_expr(value), kind)
                elif kind == SCALAR:
                    pset.add_terminal(_scalar_expr(value), kind)
                else:
                    pset.add_terminal(value, kind)
        pset.add_primitives(*primitives)
        return pset

    def add_terminal(self, value: Expr | str | int | float, kind: AlphaKind = VECTOR, *, name: str | None = None) -> None:
        expr = _terminal_expr(value, kind)
        label = name or _terminal_name(value, expr)
        self.terminals.setdefault(kind, []).append(AlphaTerminal(label, expr, kind))

    def add_primitive(self, primitive: AlphaPrimitive) -> None:
        self.primitives.setdefault(primitive.output_kind, []).append(primitive)

    def add_primitives(self, *primitives: AlphaPrimitive) -> None:
        for primitive in primitives:
            self.add_primitive(primitive)

    def terminals_for(self, kind: AlphaKind) -> tuple[AlphaTerminal, ...]:
        return tuple(self.terminals.get(kind, ()))

    def primitives_for(self, kind: AlphaKind) -> tuple[AlphaPrimitive, ...]:
        return tuple(self.primitives.get(kind, ()))


def _terminal_expr(value: Expr | str | int | float, kind: AlphaKind) -> Expr:
    if isinstance(value, str):
        return var(value) if kind != POSITIVE_SCALAR else Identifier(value)
    if kind == POSITIVE_SCALAR:
        return _positive_scalar_expr(value)
    if kind == SCALAR:
        return _scalar_expr(value)
    return ensure_expr(value)


def _scalar_expr(value: Expr | str | int | float) -> Expr:
    if isinstance(value, str):
        return Identifier(value)
    return ensure_expr(value)


def _positive_scalar_expr(value: Expr | str | int | float) -> Expr:
    expr = _scalar_expr(value)
    if isinstance(expr, Number) and expr.value <= 0.0:
        raise ValueError(f"Positive scalar terminals must be > 0, got {expr.value:g}")
    return expr


def _terminal_name(value: Expr | str | int | float, expr: Expr) -> str:
    return value if isinstance(value, str) else repr(expr)


@dataclass(frozen=True)
class SearchScheme:
    population_size: int = 64
    generations_per_depth: int = 4
    cx_prob: float = 0.5
    mut_prob: float = 0.25
    terminal_prob: float = 0.35
    seed: int | None = None


@dataclass(frozen=True)
class AlphaCandidate:
    expr: Expr
    fitness: float
    depth: int
    kind: AlphaKind = VECTOR


def expr_depth(expr: Expr) -> int:
    if isinstance(expr, Call):
        return 1 + (max((expr_depth(arg) for arg in expr.args), default=0))
    return 0


def expr_key(expr: Expr) -> str:
    return repr(expr)


class FormulaAlphaSearch:
    def __init__(
        self,
        pset: AlphaPrimitiveSet,
        objective: Objective,
        *,
        output_kind: AlphaKind = VECTOR,
        filters: Sequence[ExprPredicate] = (),
        additive: AdditivePredicate | None = None,
        scheme: SearchScheme | None = None,
    ) -> None:
        self.pset = pset
        self.objective = objective
        self.output_kind = output_kind
        self.filters = tuple(filters)
        self.additive = additive or (lambda _expr, _pool, fitness: fitness > 0.0)
        self.scheme = scheme or SearchScheme()
        self._rng = random.Random(self.scheme.seed)

    def search(self, max_depth: int, *, initial_pool: Sequence[Expr] = ()) -> list[AlphaCandidate]:
        pool = list(initial_pool)
        selected: list[AlphaCandidate] = []
        seen = {expr_key(expr) for expr in pool}
        for depth in range(1, max_depth + 1):
            for node in self._evolve_depth(depth):
                expr = node.expr
                key = expr_key(expr)
                if key in seen or expr_depth(expr) > depth or not all(pred(expr) for pred in self.filters):
                    continue
                fitness = float(self.objective(expr, tuple(pool)))
                seen.add(key)
                if self.additive(expr, tuple(pool), fitness):
                    pool.append(expr)
                    selected.append(AlphaCandidate(expr, fitness, depth, node.kind))
        return selected

    def _evolve_depth(self, depth: int) -> list[_TypedNode]:
        toolbox = deap_base.Toolbox()
        toolbox.register("individual", lambda: self._random_node(self.output_kind, depth))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        pop = toolbox.population(n=self.scheme.population_size)
        for _ in range(self.scheme.generations_per_depth):
            offspring = list(pop)
            for i in range(1, len(offspring), 2):
                if self._rng.random() < self.scheme.cx_prob:
                    offspring[i - 1], offspring[i] = self._crossover(offspring[i - 1], offspring[i], depth)
            for i, node in enumerate(offspring):
                if self._rng.random() < self.scheme.mut_prob:
                    offspring[i] = self._random_node(node.kind, depth)
            pop = offspring
        return pop

    def _random_node(self, kind: AlphaKind, depth: int) -> _TypedNode:
        terminals = self.pset.terminals_for(kind)
        primitives = self.pset.primitives_for(kind)
        use_terminal = depth <= 0 or not primitives or (terminals and self._rng.random() < self.scheme.terminal_prob)
        if use_terminal:
            if not terminals:
                raise ValueError(f"No terminals registered for alpha kind {kind!r}")
            terminal = self._rng.choice(terminals)
            return _TypedNode(terminal.expr, terminal.kind)
        primitive = self._rng.choice(primitives)
        args = [self._random_node(arg_kind, depth - 1).expr for arg_kind in primitive.input_kinds]
        return _TypedNode(primitive.fn(*args), primitive.output_kind)

    def _crossover(self, left: _TypedNode, right: _TypedNode, depth: int) -> tuple[_TypedNode, _TypedNode]:
        if left.kind != right.kind or expr_depth(left.expr) > depth or expr_depth(right.expr) > depth:
            return left, right
        return right, left


__all__ = [
    "AlphaCandidate",
    "AlphaKind",
    "AlphaPrimitive",
    "AlphaPrimitiveSet",
    "AlphaTerminal",
    "BOOLEAN",
    "FormulaAlphaSearch",
    "POSITIVE_SCALAR",
    "SCALAR",
    "SearchScheme",
    "VECTOR",
    "default_alpha_pnl",
    "default_sharpe_objective",
    "dimensionless_filter",
    "ewm_std",
    "ewm_var",
    "expr_depth",
    "ridge_pool_alpha_pnl",
]
