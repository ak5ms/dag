from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from numba import int64, types
from numba.experimental import jitclass
from numba.typed import List

from trading_dsl_engine.dsl import DEFAULT_DSL_REGISTRY, DSLFunctionRegistry
from trading_dsl_engine.ops import _make_input_node, _make_literal_node, _make_universe_groupby_node, register_builtin_ops
from trading_dsl_engine.parser import Call, Expr, Identifier, Number, Universe, parse_formula
from trading_dsl_engine.registry import REGISTRY, CompiledNode


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
    cache: dict[tuple, CompiledNode] = {}
    cache_hits = 0
    expanded_nodes = 0

    def build(node: Expr, depth: int = 0) -> CompiledNode:
        nonlocal cache_hits, expanded_nodes
        if depth > 256:
            raise FormulaCompileError("Exceeded max DSL expansion depth (256)")

        key = _expr_key(node)
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
                compiled_expanded = build(expanded, depth + 1)
                cache[key] = compiled_expanded
                return compiled_expanded

        expanded_nodes += 1
        if isinstance(node, Identifier):
            if node.name not in inputs:
                inputs[node.name] = len(inputs)
            compiled = _make_input_node(inputs[node.name])
        elif isinstance(node, Number):
            compiled = _make_literal_node(node.value)
        elif isinstance(node, Call):
            if node.fn == "groupby" and len(node.args) == 2 and isinstance(node.args[0], Universe):
                op_child = build(node.args[1], depth + 1)
                groups = _resolve_universe_groups(node.args[0], column_name_to_index)
                try:
                    compiled = _make_universe_groupby_node(op_child, groups)
                except ValueError as exc:
                    raise FormulaCompileError(f"Invalid call groupby: {exc}") from exc
            else:
                try:
                    spec = REGISTRY.get(node.fn)
                except KeyError as exc:
                    raise FormulaCompileError(str(exc)) from exc
                children = [build(a, depth + 1) for a in node.args]
                try:
                    _ = spec.validator([c.type_info for c in children])
                except ValueError as exc:
                    raise FormulaCompileError(f"Invalid call {node.fn}: {exc}") from exc
                literal_args = [a.value if isinstance(a, Number) else float("nan") for a in node.args]
                compiled = spec.builder(children, literal_args)
        else:
            raise FormulaCompileError(f"Unhandled expression node: {node}")

        cache[key] = compiled
        return compiled

    root = build(ast_expr, 0)
    output_code = _kind_to_code(root.type_info.kind)

    spec = [
        ("feature", root.instance_type),
        ("n_inputs", int64),
        ("output_code", int64),
        ("input_names", types.ListType(types.unicode_type)),
    ]

    @jitclass(spec)
    class CompiledFormula:  # noqa: N801
        def __init__(self, feature, names, output_code):
            self.feature = feature
            self.n_inputs = len(names)
            self.output_code = output_code
            self.input_names = names

        def on_data(self, frame2d):
            self.feature.on_data(frame2d)

        def emit(self):
            return self.feature.emit()

    ordered_names = tuple(inputs.keys())
    typed_names = List()
    for n in ordered_names:
        typed_names.append(n)

    compiled = CompiledFormula(root.ctor(), typed_names, output_code)
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
