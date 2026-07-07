from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import random
import re

from typing import Any

from deap import algorithms, base as deap_base, creator, gp, tools

from trading_dsl_engine.base.dsl import abs as dsl_abs
from trading_dsl_engine.base.dsl import add, and_, clip, cumsum, div, ewm, fillna, ffill, mean, mul, shift, where, xs_rank
from trading_dsl_engine.base.dsl import einsum, get_beta, Ridge, ensure_expr, var
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


def dimensionless_filter(config: dict[str, Any] | None = None, *, allow_unknown: bool = False) -> ExprPredicate:
    def _predicate(expr: Expr) -> bool:
        from trading_dsl_engine.base.metadata import UnitInfo, analyze_formula_metadata

        units = analyze_formula_metadata(expr, config).get_units()
        return (allow_unknown and units.is_unknown()) or units == UnitInfo.dimensionless()

    return _predicate


def make_alpha_pset(
    features: Iterable[Expr | str],
    *,
    halflives: Iterable[Expr | int | float] = (5.0, 30.0, 120.0, 1440.0),
    shift_lags: Iterable[Expr | int | float] = (1.0,),
    name: str = "ALPHA",
) -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped(name, [], Expr)
    pset.addPrimitive(add, [Expr, Expr], Expr, name="add")
    pset.addPrimitive(lambda a, b: a - b, [Expr, Expr], Expr, name="sub")
    pset.addPrimitive(mul, [Expr, Expr], Expr, name="mul")
    pset.addPrimitive(div, [Expr, Expr], Expr, name="div")
    pset.addPrimitive(dsl_abs, [Expr], Expr, name="abs")
    pset.addPrimitive(xs_rank, [Expr], Expr, name="xs_rank")
    pset.addPrimitive(lambda x, hl: ewm(x, hl.expr), [Expr, PositiveScalar], Expr, name="ewm")
    pset.addPrimitive(lambda x, n: shift(x, n.expr, n.expr), [Expr, PositiveIntScalar], Expr, name="shift")
    pset.addPrimitive(lambda hl: hl, [PositiveScalar], PositiveScalar, name="same_hl")
    pset.addPrimitive(lambda n: n, [PositiveIntScalar], PositiveIntScalar, name="same_lag")
    for idx, feature in enumerate(features):
        pset.addTerminal(_feature_expr(feature), Expr, name=_terminal_name("x", idx, feature))
    for idx, halflife in enumerate(halflives):
        pset.addTerminal(PositiveScalar(_positive_scalar_expr(halflife)), PositiveScalar, name=_terminal_name("hl", idx, halflife))
    for idx, lag in enumerate(shift_lags):
        pset.addTerminal(PositiveIntScalar(_positive_int_expr(lag)), PositiveIntScalar, name=_terminal_name("lag", idx, lag))
    return pset


def individual_to_expr(individual: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped) -> Expr:
    out = gp.compile(individual, pset)
    return ensure_expr(out)


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
        toolbox.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=depth))
        toolbox.decorate("mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=depth))
        toolbox.register("evaluate", _evaluate_individual, pset=pset, objective=objective, pool=tuple(pool), filters=tuple(filters))
        pop = toolbox.population(n=population_size)
        algorithms.eaSimple(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=generations_per_depth, verbose=False)
        for individual in tools.selBest(pop, k=len(pop)):
            expr = individual_to_expr(individual, pset)
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
    expr = individual_to_expr(individual, pset)
    if not all(predicate(expr) for predicate in filters):
        return (float("-inf"),)
    return (float(objective(expr, pool)),)


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



def _example_alpha_search_on_random_data() -> list[Candidate]:
    import numpy as np

    from flows.random_data import matrix_inputs_from_long_fields, simulate_requested_fields
    from flows.utils import pct_change
    from trading_dsl_engine.jax_flat.engine import compile_formula

    fields = simulate_requested_fields(n_minutes=6 * 60, seed=42)
    px = matrix_inputs_from_long_fields(fields, ("mp_out0.close",))["mp_out0.close"]
    realized_rets = px[1:] / px[:-1] - 1.0
    tradable = matrix_inputs_from_long_fields(fields, ("is_tradable_out0",))["is_tradable_out0"][1:] == 1.0

    returns = pct_change(var("mp_out0.close"))
    spread_bps = 1e4 * (var("ap_out0.close") - var("bp_out0.close")) / var("mp_out0.close")
    features = (
        ewm(returns, 15),
        ewm(returns, 60),
        xs_rank(returns),
        xs_rank(spread_bps),
        xs_rank(var("volume_out0")),
    )
    pset = make_alpha_pset(features, halflives=(5, 30), shift_lags=(1,))
    compiled: dict[str, Any] = {}

    def score(candidate: Expr, pool: Sequence[Expr]) -> float:
        key = repr(candidate)
        runtime = compiled.get(key)
        if runtime is None:
            runtime = compile_formula(candidate, cpp=False)
            compiled[key] = runtime
        inputs = matrix_inputs_from_long_fields(fields, runtime.program.input_names)
        _, alpha_path = runtime.run_batch(inputs)
        alpha = np.asarray(alpha_path, dtype=float)[:-1]
        pnl = np.where(tradable & np.isfinite(alpha), alpha * realized_rets, np.nan)
        pnl = pnl[np.isfinite(pnl)]
        if pnl.size < 2 or np.nanstd(pnl) == 0.0:
            return float("-inf")
        return float(np.nanmean(pnl) / np.nanstd(pnl))

    selected = search_formulas(
        pset,
        score,
        max_depth=1,
        filters=(lambda expr: "is_tradable_out0" not in repr(expr),),
        additive=lambda candidate, pool, fitness: np.isfinite(fitness) and len(pool) < 3,
        population_size=4,
        generations_per_depth=1,
        seed=7,
    )
    for expr, fitness, depth in selected:
        print(f"depth={depth} fitness={fitness:.6g} expr={expr!r}")
    return selected


if __name__ == "__main__":
    _example_alpha_search_on_random_data()


__all__ = [
    "Candidate",
    "PositiveScalar",
    "PositiveIntScalar",
    "default_alpha_pnl",
    "default_sharpe_objective",
    "dimensionless_filter",
    "ewm_std",
    "ewm_var",
    "individual_to_expr",
    "make_alpha_pset",
    "ridge_pool_alpha_pnl",
    "search_formulas",
]
