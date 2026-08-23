from __future__ import annotations

import random

from deap import base as deap_base
from deap import gp, tools

from flows.gp.types import ExprValue


def _generation_passthrough(value):
    """Identity used only when DEAP needs to extend a terminal-only branch."""

    return value


def _terminal_value(pset: gp.PrimitiveSetTyped, terminal: gp.Terminal):
    value = terminal.value
    if isinstance(value, str) and value in pset.context:
        return pset.context[value]
    return value


def _unique_public_primitives(pset: gp.PrimitiveSetTyped):
    seen: set[str] = set()
    family_of = getattr(pset, "gp_primitive_family", {})
    for primitives in pset.primitives.values():
        for primitive in primitives:
            if primitive.name in seen or primitive.name not in family_of:
                continue
            seen.add(primitive.name)
            yield primitive


def _ensure_leaf_witnesses(pset: gp.PrimitiveSetTyped) -> None:
    """Derive a valid terminal witness for every producible expression type.

    Standard DEAP generation requires a terminal when a branch reaches its leaf
    depth. Some semantic types (for example ``DurationRow``) are intentionally
    derived rather than raw input fields. Instead of retrying when DEAP lands
    on such a type, build one valid witness expression from the typed grammar and
    expose it as generation-only scaffolding.
    """

    scaffolding = set(getattr(pset, "gp_generation_scaffold", ()))
    primitives = list(_unique_public_primitives(pset))

    while True:
        missing = [
            type_
            for type_, values in pset.primitives.items()
            if values and not pset.terminals.get(type_)
        ]
        if not missing:
            break
        progress = False
        for type_ in missing:
            candidate = next(
                (
                    primitive
                    for primitive in primitives
                    if issubclass(primitive.ret, type_)
                    and all(pset.terminals.get(arg) for arg in primitive.args)
                ),
                None,
            )
            if candidate is None:
                continue
            args = [
                _terminal_value(pset, pset.terminals[arg][0])
                for arg in candidate.args
            ]
            value = pset.context[candidate.name](*args)
            name = f"__gp_leaf_witness_{type_.__name__}"
            if name not in pset.mapping:
                pset.addTerminal(value, type_, name=name)
            scaffolding.add(name)
            progress = True
        if not progress:
            names = ", ".join(sorted(type_.__name__ for type_ in missing))
            raise ValueError(
                "GP grammar has producible types with no terminal derivation: " + names
            )

    pset.gp_generation_scaffold = frozenset(scaffolding)


def _ensure_generation_reachability(pset: gp.PrimitiveSetTyped) -> None:
    """Make the typed grammar total for standard DEAP ``genHalfAndHalf``."""

    _ensure_leaf_witnesses(pset)
    scaffolding = set(getattr(pset, "gp_generation_scaffold", ()))
    for type_, terminals in tuple(pset.terminals.items()):
        if not terminals or pset.primitives.get(type_):
            continue
        name = f"__gp_generation_passthrough_{type_.__name__}"
        if name not in pset.mapping:
            pset.addPrimitive(
                _generation_passthrough,
                (type_,),
                type_,
                name=name,
            )
        scaffolding.add(name)
    pset.gp_generation_scaffold = frozenset(scaffolding)


def make_toolbox(
    pset: gp.PrimitiveSetTyped,
    *,
    min_depth: int = 1,
    max_depth: int = 4,
) -> deap_base.Toolbox:
    if min_depth < 0 or max_depth < min_depth:
        raise ValueError("require 0 <= min_depth <= max_depth")
    _ensure_generation_reachability(pset)
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


def random_tree(
    pset: gp.PrimitiveSetTyped,
    *,
    min_depth: int = 1,
    max_depth: int = 4,
    seed: int | None = None,
) -> gp.PrimitiveTree:
    """Generate exactly one tree; there is no reject/retry path."""

    toolbox = make_toolbox(pset, min_depth=min_depth, max_depth=max_depth)
    if seed is None:
        return toolbox.individual()
    state = random.getstate()
    random.seed(seed)
    try:
        return toolbox.individual()
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
