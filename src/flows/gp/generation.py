from __future__ import annotations

import random

from deap import base as deap_base
from deap import gp, tools

from flows.gp.types import ExprValue


_GROUP_WRAPPER_FAMILIES = frozenset({
    "xs_group_neutralize",
    "xs_market_neutralize",
})


def _is_group_utility_family(family: str | None) -> bool:
    return bool(family) and (
        family.startswith("group_") or family in _GROUP_WRAPPER_FAMILIES
    )


def _compiler_safe_tree(individual: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped) -> bool:
    """Reject compositions the cpp_stream groupby IR cannot represent.

    Canonical ``cpp_stream.python.utils`` group helpers are row-shaped and are
    valid GP primitives, but the current cpp_stream IR rejects nested groupby
    and non-terminal expression trees in groupby inputs. Keep those helpers in
    the grammar while requiring each group helper's GP children to terminate
    at fields/static terminals. Its output is still an ordinary row and can be
    composed by any outer GP primitive.
    """

    families = getattr(pset, "gp_primitive_family", {})

    def visit(index: int, inside_group_input: bool) -> tuple[bool, int]:
        node = individual[index]
        index += 1
        if not isinstance(node, gp.Primitive):
            return True, index
        if inside_group_input:
            return False, index

        family = families.get(node.name)
        children_are_group_inputs = _is_group_utility_family(family)
        for _ in range(node.arity):
            ok, index = visit(index, children_are_group_inputs)
            if not ok:
                return False, index
        return True, index

    ok, end = visit(0, False)
    return ok and end == len(individual)


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


def _generate(
    toolbox: deap_base.Toolbox,
    pset: gp.PrimitiveSetTyped,
    max_attempts: int,
) -> gp.PrimitiveTree:
    error = None
    rejected = 0
    for _ in range(max_attempts):
        try:
            individual = toolbox.individual()
        except IndexError as exc:
            error = exc
            continue
        if _compiler_safe_tree(individual, pset):
            return individual
        rejected += 1
    detail = f"; rejected {rejected} compiler-unsafe group compositions" if rejected else ""
    raise RuntimeError(
        f"DEAP could not generate a typed tree after {max_attempts} attempts{detail}"
    ) from error


def random_tree(pset: gp.PrimitiveSetTyped, *, min_depth: int = 1, max_depth: int = 4, seed: int | None = None, max_attempts: int = 128) -> gp.PrimitiveTree:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    toolbox = make_toolbox(pset, min_depth=min_depth, max_depth=max_depth)
    if seed is None:
        return _generate(toolbox, pset, max_attempts)
    state = random.getstate()
    random.seed(seed)
    try:
        return _generate(toolbox, pset, max_attempts)
    finally:
        random.setstate(state)


def random_formula(pset: gp.PrimitiveSetTyped | None = None, *, config=None, min_depth: int = 1, max_depth: int = 4, seed: int | None = None):
    if pset is None:
        from flows.gp.pset import make_pset
        pset = make_pset(config)
    tree = random_tree(pset, min_depth=min_depth, max_depth=max_depth, seed=seed)
    return tree, individual_to_expr(tree, pset)


__all__ = ["individual_to_expr", "make_toolbox", "random_formula", "random_tree"]
