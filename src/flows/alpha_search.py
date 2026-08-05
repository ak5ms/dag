from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import math
import random
import re
from typing import Any, Literal

from deap import algorithms, base as deap_base, creator, gp, tools

from trading_dsl_engine.base.dsl import *
from trading_dsl_engine.base.dsl import abs as dsl_abs
from trading_dsl_engine.base.dsl import (
    add,
    arctan,
    ceil,
    clip,
    div,
    ewm,
    exp,
    fillna,
    floor,
    fraction,
    ln,
    mul,
    norm_inv,
    purify,
    shift,
    sign,
    sub,
    var,
    where,
    xs_norm,
    xs_pct_rank,
    xs_rank,
)
from trading_dsl_engine.base.metadata import analyze_formula_metadata
from trading_dsl_engine.base.parser import Expr

Objective = Callable[[Expr, Sequence[Expr]], float]
ExprPredicate = Callable[[Expr], bool]
AdditivePredicate = Callable[[Expr, Sequence[Expr], float], bool]
Candidate = tuple[Expr, float, int]


@dataclass(frozen=True)
class PositiveScalar:
    expr: Expr


@dataclass(frozen=True)
class PositiveIntScalar:
    expr: Expr


@dataclass(frozen=True)
class OperatorSpec:
    """A search operator plus an algebraic compatibility family.

    The family selects a general constraint algorithm. It does not encode
    feature-specific combinations or units. Metadata types are propagated by
    the existing metadata analyzer, and nodes may carry multiple types.
    """

    name: str
    fn: Callable[..., Expr]
    arity: int
    family: Literal[
        "unary",
        "same_type",
        "numeric_binary",
        "comparison",
        "conditional",
    ] = "unary"


@dataclass(frozen=True)
class SemanticSearchConfig:
    metadata_config: dict[str, Any] | None = None
    target_types: frozenset[str] = frozenset()
    ignored_types: frozenset[str] = frozenset(
        {
            "numeric",
            "scalar",
            "vector",
            "matrix",
            "unknown",
            "finite",
            "bounded",
        }
    )
    require_known_types: bool = False


def default_alpha_pnl(alpha: Expr, *, roll_rets: Expr, is_tradable: Expr, hl: Expr | float) -> Expr:
    w = alpha / ewm_var(roll_rets, span=hl)
    masked = ffill(where(is_tradable, w, float("nan")))
    return shift(masked) * roll_rets


def default_sharpe_objective(alpha: Expr, *, roll_rets: Expr, is_tradable: Expr, hl: Expr | float) -> Expr:
    w = alpha / ewm_std(roll_rets, span=hl)
    masked = ffill(where(is_tradable, w, float("nan")))
    pnl = shift(masked) * roll_rets
    return einsum(fillna(pnl, 0), "n->")


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
    yhat = einsum(shift(get_beta(reg)), alpha, "f,fn->n")
    w = yhat / (ewm_std(roll_rets, span=hl) ** 2.0)
    masked = ffill(where(is_tradable, w, float("nan")))
    return einsum(shift(masked) * roll_rets, "n->")


def dimensionless_filter(config: dict[str, Any] | None = None, *, allow_unknown: bool = False) -> ExprPredicate:
    def _predicate(expr: Expr) -> bool:
        from trading_dsl_engine.base.metadata import UnitInfo

        units = analyze_formula_metadata(expr, config).get_units()
        return (allow_unknown and units.is_unknown()) or units == UnitInfo.dimensionless()

    return _predicate


def default_alpha_operator_specs() -> tuple[OperatorSpec, ...]:
    """Operators that are generally meaningful for scalar/vector alpha search.

    Compatibility is determined by a small set of algebraic families rather
    than a table of permitted feature pairs. More operators can be supplied by
    the caller without changing the generator.
    """

    unary = (
        ("abs", dsl_abs),
        ("sign", sign),
        ("fraction", fraction),
        ("purify", purify),
        ("arctan", arctan),
        ("exp", exp),
        ("ln", ln),
        ("floor", floor),
        ("ceil", ceil),
        ("xs_rank", xs_rank),
        ("xs_pct_rank", xs_pct_rank),
        ("xs_norm", xs_norm),
        ("norm_inv", norm_inv),
    )
    binary_same = (("add", add), ("sub", sub))
    binary_numeric = (("mul", mul), ("div", div))
    return tuple(OperatorSpec(name, fn, 1, "unary") for name, fn in unary) + tuple(
        OperatorSpec(name, fn, 2, "same_type") for name, fn in binary_same
    ) + tuple(OperatorSpec(name, fn, 2, "numeric_binary") for name, fn in binary_numeric)


def make_alpha_pset(
    features: Iterable[Expr | str],
    *,
    halflives: Iterable[Expr | int | float] = (5.0, 30.0, 120.0, 1440.0),
    shift_lags: Iterable[Expr | int | float] = (1.0,),
    operators: Iterable[OperatorSpec] | None = None,
    metadata_config: dict[str, Any] | None = None,
    target_types: Iterable[str] = (),
    dynamic_parameters: bool = True,
    name: str = "ALPHA",
) -> gp.PrimitiveSetTyped:
    """Build a DEAP grammar with metadata-directed semantic validation.

    Dynamic parameters are ordinary expression subtrees transformed into a
    positive scalar or positive integer expression. Thus EWM span and shift lag
    can depend on features or other generated expressions, not only literals.
    """

    config = SemanticSearchConfig(
        metadata_config=metadata_config,
        target_types=frozenset(target_types),
    )
    pset = gp.PrimitiveSetTyped(name, [], Expr)

    for spec in tuple(operators or default_alpha_operator_specs()):
        wrapped = _semantic_primitive(spec, config)
        pset.addPrimitive(wrapped, [Expr] * spec.arity, Expr, name=spec.name)

    pset.addPrimitive(
        lambda x, hl: ewm(x, span=hl.expr),
        [Expr, PositiveScalar],
        Expr,
        name="ewm",
    )
    pset.addPrimitive(
        lambda x, n: shift(x, n.expr, n.expr),
        [Expr, PositiveIntScalar],
        Expr,
        name="shift",
    )
    pset.addPrimitive(lambda hl: hl, [PositiveScalar], PositiveScalar, name="same_hl")
    pset.addPrimitive(lambda n: n, [PositiveIntScalar], PositiveIntScalar, name="same_lag")

    if dynamic_parameters:
        pset.addPrimitive(_as_positive_scalar, [Expr], PositiveScalar, name="positive")
        pset.addPrimitive(_as_positive_int_scalar, [Expr], PositiveIntScalar, name="positive_int")

    for idx, feature in enumerate(features):
        pset.addTerminal(_feature_expr(feature), Expr, name=_terminal_name("x", idx, feature))
    for idx, halflife in enumerate(halflives):
        pset.addTerminal(
            PositiveScalar(_positive_scalar_expr(halflife)),
            PositiveScalar,
            name=_terminal_name("hl", idx, halflife),
        )
    for idx, lag in enumerate(shift_lags):
        pset.addTerminal(
            PositiveIntScalar(_positive_int_expr(lag)),
            PositiveIntScalar,
            name=_terminal_name("lag", idx, lag),
        )

    pset.semantic_search_config = config
    return pset


def individual_to_expr(individual: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped) -> Expr:
    return ensure_expr(gp.compile(individual, pset))


def eval_multiple(alphas: list[Expr], *, roll_rets: Expr, is_tradable: Expr, hl: Expr | float):
    w = cat(*alphas) / ewm_std(roll_rets, span=hl)
    masked = ffill(where(is_tradable, w, float("nan")))
    return einsum(shift(masked), roll_rets, "nf,n->n")


def search_formulas(
    pset: gp.PrimitiveSetTyped,
    objective: Objective,
    *,
    max_depth: int,
    initial_pool: Sequence[Expr] = (),
    filters: Sequence[ExprPredicate] = (),
    additive: AdditivePredicate | None = None,
    population_size: int = 64,
    generations_per_depth: int = 4,
    cx_prob: float = 0.5,
    mut_prob: float = 0.25,
    tournament_size: int = 3,
    seed: int | None = None,
) -> list[Candidate]:
    _ensure_deap_creator()
    if seed is not None:
        random.seed(seed)
    pool = list(initial_pool)
    selected: list[Candidate] = []
    seen = {repr(expr) for expr in pool}
    for depth in range(1, max_depth + 1):
        toolbox = deap_base.Toolbox()
        toolbox.register("select", tools.selTournament, tournsize=tournament_size)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=depth)
        toolbox.register("individual", tools.initIterate, creator.AlphaIndividual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
        toolbox.register("map", lambda f, x: f(x))
        toolbox.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=depth))
        toolbox.decorate("mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=depth))
        toolbox.register(
            "evaluate",
            _evaluate_individual,
            pset=pset,
            objective=objective,
            pool=tuple(pool),
            filters=tuple(filters),
        )
        pop = toolbox.population(n=population_size)
        algorithms.eaSimple(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=generations_per_depth, verbose=False)
        for individual in tools.selBest(pop, k=len(pop)):
            try:
                expr = individual_to_expr(individual, pset)
            except (TypeError, ValueError, OverflowError):
                continue
            key = repr(expr)
            if key in seen or individual.height > depth:
                continue
            fitness = float(individual.fitness.values[0])
            seen.add(key)
            if additive is None or additive(expr, tuple(pool), fitness):
                pool.append(expr)
                selected.append((expr, fitness, depth))
    return selected


def _evaluate_individual(
    individual: gp.PrimitiveTree,
    *,
    pset: gp.PrimitiveSetTyped,
    objective: Objective,
    pool: Sequence[Expr],
    filters: Sequence[ExprPredicate],
) -> tuple[float]:
    try:
        expr = individual_to_expr(individual, pset)
        config = getattr(pset, "semantic_search_config", SemanticSearchConfig())
        if config.target_types and not (_semantic_types(expr, config) & config.target_types):
            return (float("-inf"),)
        if not all(predicate(expr) for predicate in filters):
            return (float("-inf"),)
        value = float(objective(expr, pool))
        return (value if math.isfinite(value) else float("-inf"),)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return (float("-inf"),)


def _semantic_primitive(spec: OperatorSpec, config: SemanticSearchConfig) -> Callable[..., Expr]:
    def apply(*args: Expr) -> Expr:
        exprs = tuple(ensure_expr(arg) for arg in args)
        if spec.family == "same_type" and len(exprs) > 1:
            common = _semantic_types(exprs[0], config)
            for expr in exprs[1:]:
                common &= _semantic_types(expr, config)
            if not common:
                raise ValueError(f"{spec.name} requires at least one intersecting semantic type")
        return ensure_expr(spec.fn(*exprs))

    return apply


def _semantic_types(expr: Expr, config: SemanticSearchConfig) -> frozenset[str]:
    types = frozenset(analyze_formula_metadata(expr, config.metadata_config).get_types())
    useful = types - config.ignored_types
    if config.require_known_types and not useful:
        raise ValueError(f"No known semantic type for {expr!r}")
    return useful or types


def _as_positive_scalar(expr: Expr) -> PositiveScalar:
    return PositiveScalar(dsl_abs(ensure_expr(expr)) + 1e-12)


def _as_positive_int_scalar(expr: Expr) -> PositiveIntScalar:
    return PositiveIntScalar(floor(dsl_abs(ensure_expr(expr))) + 1.0)


def _ensure_deap_creator() -> None:
    if not hasattr(creator, "AlphaFitnessMax"):
        creator.create("AlphaFitnessMax", deap_base.Fitness, weights=(1.0,))
    if not hasattr(creator, "AlphaIndividual"):
        creator.create("AlphaIndividual", gp.PrimitiveTree, fitness=creator.AlphaFitnessMax)


def _feature_expr(value: Expr | str) -> Expr:
    return var(value) if isinstance(value, str) else ensure_expr(value)


def _positive_scalar_expr(value: Expr | str | int | float) -> Expr:
    if isinstance(value, str):
        return var(value)
    expr = ensure_expr(value)
    if hasattr(expr, "value") and float(expr.value) <= 0.0:
        raise ValueError(f"Positive scalar terminals must be > 0, got {float(expr.value):g}")
    return expr


def _positive_int_expr(value: Expr | str | int | float) -> Expr:
    expr = _positive_scalar_expr(value)
    if hasattr(expr, "value") and not float(expr.value).is_integer():
        raise ValueError(f"Positive integer scalar terminals must be integer-valued, got {float(expr.value):g}")
    return expr


def _terminal_name(prefix: str, idx: int, value: Expr | str | int | float) -> str:
    if isinstance(value, str):
        name = re.sub(r"\W|^(?=\d)", "_", value)
        return name if name and name != "_" else f"{prefix}_{idx}"
    return f"{prefix}_{idx}"


__all__ = [
    "Candidate",
    "OperatorSpec",
    "PositiveScalar",
    "PositiveIntScalar",
    "SemanticSearchConfig",
    "default_alpha_operator_specs",
    "default_alpha_pnl",
    "default_sharpe_objective",
    "dimensionless_filter",
    "individual_to_expr",
    "make_alpha_pset",
    "ridge_pool_alpha_pnl",
    "search_formulas",
]
