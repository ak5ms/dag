from __future__ import annotations

from dataclasses import dataclass

from numba import int64, types
from numba.experimental import jitclass
from numba.typed import List

from trading_dsl_engine.dsl import DEFAULT_DSL_REGISTRY, DSLFunctionRegistry
from trading_dsl_engine.ops import _make_input_node, _make_literal_node, register_builtin_ops
from trading_dsl_engine.parser import Call, Expr, Identifier, Number, parse_formula
from trading_dsl_engine.registry import REGISTRY, CompiledNode


class FormulaCompileError(ValueError):
    pass


@dataclass(frozen=True)
class CompileStats:
    expanded_nodes: int
    cache_hits: int


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
    raise FormulaCompileError(f"Unknown output kind: {kind}")


def _expr_key(node: Expr) -> tuple:
    if isinstance(node, Identifier):
        return ("id", node.name)
    if isinstance(node, Number):
        return ("num", node.value)
    if isinstance(node, Call):
        return ("call", node.fn, tuple(_expr_key(arg) for arg in node.args))
    raise FormulaCompileError(f"Unhandled expression node for hashing: {node}")


def compile_formula(formula: str, dsl_registry: DSLFunctionRegistry | None = None) -> CompiledFormulaArtifact:
    register_builtin_ops()
    ast_expr = parse_formula(formula)
    inputs: dict[str, int] = {}
    dsl_registry = dsl_registry or DEFAULT_DSL_REGISTRY
    cache: dict[tuple, CompiledNode] = {}
    cache_hits = 0
    expanded_nodes = 0

    def build(node: Expr, depth: int = 0) -> CompiledNode:
        nonlocal cache_hits, expanded_nodes
        if depth > 256:
            raise FormulaCompileError("Exceeded max DSL expansion depth (256)")

        if isinstance(node, Call):
            py_fn = dsl_registry.get(node.fn)
            if py_fn is not None:
                try:
                    expanded = py_fn(*node.args)
                except Exception as exc:
                    raise FormulaCompileError(f"Failed expanding DSL function '{node.fn}': {exc}") from exc
                return build(expanded, depth + 1)

        key = _expr_key(node)
        cached = cache.get(key)
        if cached is not None:
            cache_hits += 1
            return cached

        expanded_nodes += 1
        if isinstance(node, Identifier):
            if node.name not in inputs:
                inputs[node.name] = len(inputs)
            compiled = _make_input_node(inputs[node.name])
        elif isinstance(node, Number):
            compiled = _make_literal_node(node.value)
        elif isinstance(node, Call):
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
    return CompiledFormulaArtifact(
        compiled=compiled,
        compiled_type=CompiledFormula.class_type.instance_type,
        input_names=ordered_names,
        output_kind=root.type_info.kind,
        stats=CompileStats(expanded_nodes=expanded_nodes, cache_hits=cache_hits),
    )
