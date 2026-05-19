from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from numba import int64
from numba.experimental import jitclass

from trading_dsl_engine.base.dsl import DEFAULT_DSL_REGISTRY, DSLFunctionRegistry
from trading_dsl_engine.numba.ops import (
    _make_input_node,
    _make_literal_node,
    _make_local_value_node,
    _make_tuple_key_node,
    _make_universe_dynamic_groupby_node,
    _make_universe_groupby_node,
    register_builtin_ops,
)
from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple, Number, Universe, parse_formula
from trading_dsl_engine.base.registry import REGISTRY, CompiledNode


class FormulaCompileError(ValueError):
    pass


@dataclass(frozen=True)
class CompileStats:
    expanded_nodes: int
    cache_hits: int
    compile_seconds: float


@dataclass(frozen=True)
class CompiledFormulaArtifact:
    compiled: object
    compiled_type: object
    input_names: tuple[str, ...]
    output_kind: str
    stats: CompileStats


@dataclass(frozen=True)
class _CompiledFormulaPlan:
    formula_class: object
    feature_ctor: object
    compiled_type: object
    input_names: tuple[str, ...]
    output_kind: str
    output_code: int
    expanded_nodes: int
    cache_hits: int


_COMPILE_PLAN_CACHE: dict[tuple, _CompiledFormulaPlan] = {}


def _kind_to_code(kind: str) -> int:
    if kind == "scalar":
        return 0
    if kind == "vector":
        return 1
    if kind == "matrix":
        return 2
    if kind == "object":
        return 3
    raise FormulaCompileError(f"Unknown output kind: {kind}")


def _expr_key(node: Expr) -> tuple:
    if isinstance(node, Identifier):
        return ("id", node.name)
    if isinstance(node, Number):
        return ("num", node.value)
    if isinstance(node, Call):
        return ("call", node.fn, tuple(_expr_key(arg) for arg in node.args))
    if isinstance(node, Universe):
        return ("univ", node.groups)
    if isinstance(node, KeyTuple):
        return ("tuple", tuple(_expr_key(item) for item in node.items))
    raise FormulaCompileError(f"Unhandled expression node for hashing: {node}")


def _resolve_universe_groups(universe: Universe, column_name_to_index: dict[str, int]) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for group in universe.groups:
        resolved: list[int] = []
        for member in group:
            if isinstance(member, int):
                idx = member
            else:
                if member not in column_name_to_index:
                    raise FormulaCompileError(
                        f"Unknown universe column '{member}'. Pass column_names to compile_formula/build_engine."
                    )
                idx = column_name_to_index[member]
            if idx < 0:
                raise FormulaCompileError("Universe column indexes must be >= 0")
            if idx in seen:
                raise FormulaCompileError(f"Universe column index {idx} appears in more than one group")
            seen.add(idx)
            resolved.append(idx)
        groups.append(tuple(resolved))
    return tuple(groups)



def _canonical_groupby_key_items(key: Expr) -> tuple[Expr, ...]:
    if not isinstance(key, KeyTuple):
        key = KeyTuple((key,))
    universe_count = 0
    for item in key.items:
        if isinstance(item, Universe):
            universe_count += 1
    if universe_count > 1:
        raise FormulaCompileError("groupby key tuple may contain at most one univ(...) element")
    return key.items


def _replace_self_placeholder(node: Expr, lhs: Expr) -> Expr:
    if isinstance(node, Identifier) and node.name == "self_":
        return lhs
    if isinstance(node, Call):
        return Call(node.fn, tuple(_replace_self_placeholder(arg, lhs) for arg in node.args))
    if isinstance(node, KeyTuple):
        return KeyTuple(tuple(_replace_self_placeholder(item, lhs) for item in node.items))
    return node

def compile_formula(
    formula: str | Expr,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
) -> CompiledFormulaArtifact:
    started_at = perf_counter()
    register_builtin_ops()
    ast_expr = parse_formula(formula) if isinstance(formula, str) else formula
    inputs: dict[str, int] = {}
    column_name_to_index = {name: i for i, name in enumerate(column_names or ())}
    dsl_registry = dsl_registry or DEFAULT_DSL_REGISTRY
    plan_cache_key = None
    if dsl_registry is DEFAULT_DSL_REGISTRY:
        plan_cache_key = (_expr_key(ast_expr), tuple(column_names or ()))
        cached_plan = _COMPILE_PLAN_CACHE.get(plan_cache_key)
        if cached_plan is not None:
            compiled = cached_plan.formula_class(
                cached_plan.feature_ctor(),
                len(cached_plan.input_names),
                cached_plan.output_code,
            )
            return CompiledFormulaArtifact(
                compiled=compiled,
                compiled_type=cached_plan.compiled_type,
                input_names=cached_plan.input_names,
                output_kind=cached_plan.output_kind,
                stats=CompileStats(
                    expanded_nodes=cached_plan.expanded_nodes,
                    cache_hits=cached_plan.cache_hits + 1,
                    compile_seconds=perf_counter() - started_at,
                ),
            )
    cache: dict[tuple, CompiledNode] = {}
    cache_hits = 0
    expanded_nodes = 0

    def build(node: Expr, depth: int = 0, local_inputs: dict[str, CompiledNode] | None = None) -> CompiledNode:
        nonlocal cache_hits, expanded_nodes
        if depth > 256:
            raise FormulaCompileError("Exceeded max DSL expansion depth (256)")

        use_cache = not local_inputs
        key = _expr_key(node)
        if use_cache:
            cached = cache.get(key)
            if cached is not None:
                cache_hits += 1
                return cached

        if isinstance(node, Call):
            py_fn = dsl_registry.get(node.fn)
            if py_fn is not None:
                try:
                    expanded = py_fn(*node.args)
                except Exception as exc:
                    raise FormulaCompileError(f"Failed expanding DSL function '{node.fn}': {exc}") from exc
                compiled_expanded = build(expanded, depth + 1, local_inputs)
                if use_cache:
                    cache[key] = compiled_expanded
                return compiled_expanded

        expanded_nodes += 1
        if isinstance(node, Identifier):
            if local_inputs is not None:
                if node.name in local_inputs:
                    compiled = local_inputs[node.name]
                else:
                    if node.name not in inputs:
                        inputs[node.name] = len(inputs)
                    compiled = _make_input_node(inputs[node.name])
            else:
                if node.name not in inputs:
                    inputs[node.name] = len(inputs)
                compiled = _make_input_node(inputs[node.name])
        elif isinstance(node, Number):
            compiled = _make_literal_node(node.value)
        elif isinstance(node, Call):
            if node.fn == "groupby" and len(node.args) == 3:
                try:
                    spec = REGISTRY.get(node.fn)
                except KeyError as exc:
                    raise FormulaCompileError(str(exc)) from exc
                key_items = _canonical_groupby_key_items(node.args[0])
                universe_items = [item for item in key_items if isinstance(item, Universe)]
                dynamic_items = [item for item in key_items if not isinstance(item, Universe)]
                if universe_items:
                    op_expr = _replace_self_placeholder(node.args[2], node.args[1])
                    op_child = build(op_expr, depth + 1, local_inputs)
                    groups = _resolve_universe_groups(universe_items[0], column_name_to_index)
                    if len(dynamic_items) == 0:
                        compiled = _make_universe_groupby_node(op_child, groups)
                    else:
                        key_children = [build(item, depth + 1, local_inputs) for item in dynamic_items]
                        key_child = key_children[0] if len(key_children) == 1 else _make_tuple_key_node(key_children)
                        compiled = _make_universe_dynamic_groupby_node(key_child, op_child, groups)
                    if use_cache:
                        cache[key] = compiled
                    return compiled
                key_children = [build(item, depth + 1, local_inputs) for item in key_items]
                key_child = key_children[0] if len(key_children) == 1 else _make_tuple_key_node(key_children)
                lhs_child = build(node.args[1], depth + 1, local_inputs)
                local_value = _make_local_value_node(lhs_child.type_info)
                rhs_child = build(node.args[2], depth + 1, {"self_": local_value})
                children = [key_child, lhs_child, rhs_child]
                try:
                    _ = spec.validator([c.type_info for c in children])
                except ValueError as exc:
                    raise FormulaCompileError(f"Invalid call {node.fn}: {exc}") from exc
                literal_args = [float("nan"), float("nan"), float("nan")]
                compiled = spec.builder(children, literal_args)
            elif node.fn == "groupby":
                raise FormulaCompileError(
                    "groupby only supports canonical form: groupby((key1, ..., maybe_univ, ...), lhs, op_using_self_)"
                )
            else:
                try:
                    spec = REGISTRY.get(node.fn)
                except KeyError as exc:
                    raise FormulaCompileError(str(exc)) from exc
                children = [build(a, depth + 1, local_inputs) for a in node.args]
                try:
                    _ = spec.validator([c.type_info for c in children])
                except ValueError as exc:
                    raise FormulaCompileError(f"Invalid call {node.fn}: {exc}") from exc
                literal_args = [a.value if isinstance(a, Number) else float("nan") for a in node.args]
                compiled = spec.builder(children, literal_args)
        else:
            raise FormulaCompileError(f"Unhandled expression node: {node}")

        if use_cache:
            cache[key] = compiled
        return compiled

    root = build(ast_expr, 0)
    output_code = _kind_to_code(root.type_info.kind)

    spec = [
        ("feature", root.instance_type),
        ("n_inputs", int64),
        ("output_code", int64),
    ]

    @jitclass(spec)
    class CompiledFormula:  # noqa: N801
        def __init__(self, feature, n_inputs: int, output_code: int):
            self.feature = feature
            self.n_inputs = n_inputs
            self.output_code = output_code

        def on_data(self, frame2d):
            self.feature.on_data(frame2d)

        def emit(self):
            return self.feature.emit()

    ordered_names = tuple(inputs.keys())
    if plan_cache_key is not None:
        _COMPILE_PLAN_CACHE[plan_cache_key] = _CompiledFormulaPlan(
            formula_class=CompiledFormula,
            feature_ctor=root.ctor,
            compiled_type=CompiledFormula.class_type.instance_type,
            input_names=ordered_names,
            output_kind=root.type_info.kind,
            output_code=output_code,
            expanded_nodes=expanded_nodes,
            cache_hits=cache_hits,
        )

    compiled = CompiledFormula(root.ctor(), len(ordered_names), output_code)
    compile_seconds = perf_counter() - started_at
    return CompiledFormulaArtifact(
        compiled=compiled,
        compiled_type=CompiledFormula.class_type.instance_type,
        input_names=ordered_names,
        output_kind=root.type_info.kind,
        stats=CompileStats(
            expanded_nodes=expanded_nodes,
            cache_hits=cache_hits,
            compile_seconds=compile_seconds,
        ),
    )
