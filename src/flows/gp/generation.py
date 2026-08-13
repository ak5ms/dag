from __future__ import annotations

import random

from deap import base as deap_base
from deap import gp, tools

from flows.gp.types import ExprValue


def make_toolbox(pset: gp.PrimitiveSetTyped, *, min_depth: int = 1, max_depth: int = 4) -> deap_base.Toolbox:
    if min_depth < 0 or max_depth < min_depth:
        raise ValueError("require 0 <= min_depth <= max_depth")
    toolbox = deap_base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_depth, max_=max_depth)
    toolbox.register("individual", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox


def individual_to_expr(individual, pset: gp.PrimitiveSetTyped):
    value = gp.compile(expr=individual, pset=pset)
    if callable(value):
        value = value()
    if not isinstance(value, ExprValue):
        raise TypeError(f"GP individual compiled to {type(value).__name__}, expected ExprValue")
    return value.expr


def _generate(toolbox: deap_base.Toolbox, max_attempts: int) -> gp.PrimitiveTree:
    error = None
    for _ in range(max_attempts):
        try:
            return toolbox.individual()
        except IndexError as exc:
            error = exc
    raise RuntimeError(f"DEAP could not generate a typed tree after {max_attempts} attempts") from error


def random_tree(pset: gp.PrimitiveSetTyped, *, min_depth: int = 1, max_depth: int = 4, seed: int | None = None, max_attempts: int = 128) -> gp.PrimitiveTree:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    toolbox = make_toolbox(pset, min_depth=min_depth, max_depth=max_depth)
    if seed is None:
        return _generate(toolbox, max_attempts)
    state = random.getstate()
    random.seed(seed)
    try:
        return _generate(toolbox, max_attempts)
    finally:
        random.setstate(state)


def random_formula(pset: gp.PrimitiveSetTyped | None = None, *, config=None, min_depth: int = 1, max_depth: int = 4, seed: int | None = None):
    if pset is None:
        from flows.gp.pset import make_pset
        pset = make_pset(config)
    tree = random_tree(pset, min_depth=min_depth, max_depth=max_depth, seed=seed)
    return tree, individual_to_expr(tree, pset)


__all__ = ["individual_to_expr", "make_toolbox", "random_formula", "random_tree"]
