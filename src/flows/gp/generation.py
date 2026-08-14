from __future__ import annotations

import random

from deap import base as deap_base
from deap import gp, tools

from flows.gp.types import ExprValue, StaticValue


_GROUP_WRAPPER_FAMILIES = frozenset({
    "xs_group_neutralize",
    "xs_market_neutralize",
})
_GROUP_RHS_REDUCTION_FAMILIES = frozenset({
    "sum",
    "mean",
    "std",
    "reduce_min",
    "reduce_max",
})
_GROUP_RHS_CAPTURE_ARGUMENTS = {
    "group_vector_proj": frozenset({1}),
    "group_vector_neut": frozenset({1}),
}


def _is_group_utility_family(family: str | None) -> bool:
    return bool(family) and (
        family.startswith("group_") or family in _GROUP_WRAPPER_FAMILIES
    )


def _forbidden_inside_group_rhs(family: str | None) -> bool:
    return (
        _is_group_utility_family(family)
        or family in _GROUP_RHS_REDUCTION_FAMILIES
        or bool(family) and family.startswith("vec_")
    )


def _compiler_safe_tree(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
) -> bool:
    """Reject only GP compositions the cpp_stream groupby IR cannot represent."""

    families = getattr(pset, "gp_primitive_family", {})

    def visit(index: int, inside_group_rhs: bool) -> tuple[bool, int]:
        node = individual[index]
        index += 1
        if not isinstance(node, gp.Primitive):
            return True, index
        family = families.get(node.name)
        valid = not (inside_group_rhs and _forbidden_inside_group_rhs(family))
        captured_positions = _GROUP_RHS_CAPTURE_ARGUMENTS.get(family, frozenset())
        for child_position in range(node.arity):
            child_valid, index = visit(
                index,
                inside_group_rhs or child_position in captured_positions,
            )
            valid = valid and child_valid
        return valid, index

    ok, end = visit(0, False)
    return ok and end == len(individual)


def _static_passthrough(value):
    """Generation-only identity used to extend compile-time parameter branches."""

    return value


def _ensure_static_generation_primitives(pset: gp.PrimitiveSetTyped) -> None:
    """Make every terminal-backed static GP type reachable at internal depths.

    DEAP's standard ``genFull`` requires a primitive whenever the requested
    minimum depth has not yet been reached. Static parameter types naturally
    only need terminals, so without this identity scaffold a perfectly valid
    operator such as ``vec_powersum(x, PositiveNumber)`` can make deep
    ``genHalfAndHalf`` generation fail before compilation is attempted.

    The identities are installed lazily by ``make_toolbox`` rather than by
    ``make_pset``. They are therefore generation scaffolding, not public DSL
    operator families, and they disappear when the tree is compiled.
    """

    for type_, terminals in tuple(pset.terminals.items()):
        if (
            not terminals
            or not isinstance(type_, type)
            or not issubclass(type_, StaticValue)
        ):
            continue
        name = f"__gp_static_passthrough_{type_.__name__}"
        if name in pset.mapping:
            continue
        pset.addPrimitive(
            _static_passthrough,
            (type_,),
            type_,
            name=name,
        )


def make_toolbox(
    pset: gp.PrimitiveSetTyped,
    *,
    min_depth: int = 1,
    max_depth: int = 4,
) -> deap_base.Toolbox:
    if min_depth < 0 or max_depth < min_depth:
        raise ValueError("require 0 <= min_depth <= max_depth")
    _ensure_static_generation_primitives(pset)
    toolbox = deap_base.Toolbox()
    toolbox.register(
        "expr",
        gp.genHalfAndHalf,
        pset=pset,
        min_=min_depth,
        max_=max_depth,
    )
    toolbox.register("individual", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox


def individual_to_expr(individual, pset: gp.PrimitiveSetTyped):
    value = gp.compile(expr=individual, pset=pset)
    if callable(value):
        value = value()
    if not isinstance(value, ExprValue):
        raise TypeError(
            f"GP individual compiled to {type(value).__name__}, expected ExprValue"
        )
    return value.expr


def _generate(toolbox, pset, max_attempts: int) -> gp.PrimitiveTree:
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
    detail = (
        f"; rejected {rejected} compiler-unsafe group compositions"
        if rejected
        else ""
    )
    raise RuntimeError(
        f"DEAP could not generate a typed tree after {max_attempts} attempts{detail}"
    ) from error


def random_tree(
    pset: gp.PrimitiveSetTyped,
    *,
    min_depth: int = 1,
    max_depth: int = 4,
    seed: int | None = None,
    max_attempts: int = 128,
) -> gp.PrimitiveTree:
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


def random_formula(
    pset: gp.PrimitiveSetTyped | None = None,
    *,
    config=None,
    min_depth: int = 1,
    max_depth: int = 4,
    seed: int | None = None,
):
    if pset is None:
        from flows.gp.factory import make_pset

        pset = make_pset(config)
    tree = random_tree(pset, min_depth=min_depth, max_depth=max_depth, seed=seed)
    return tree, individual_to_expr(tree, pset)


__all__ = ["individual_to_expr", "make_toolbox", "random_formula", "random_tree"]
