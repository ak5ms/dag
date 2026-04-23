from __future__ import annotations

from dataclasses import dataclass

from .ops import _make_input_node, _make_literal_node, register_builtin_ops
from .parser import Call, Identifier, Number, Expr, parse_formula
from .registry import REGISTRY, CompiledNode, TypeInfo


class FormulaCompileError(ValueError):
    pass


@dataclass(frozen=True)
class CompiledFormula:
    formula: str
    input_names: tuple[str, ...]
    output_type: TypeInfo
    make_feature: callable


def compile_formula(formula: str) -> CompiledFormula:
    register_builtin_ops()
    ast_expr = parse_formula(formula)
    inputs: dict[str, int] = {}

    def build(node: Expr) -> CompiledNode:
        if isinstance(node, Identifier):
            if node.name not in inputs:
                inputs[node.name] = len(inputs)
            return _make_input_node(inputs[node.name])
        if isinstance(node, Number):
            return _make_literal_node(node.value)
        if isinstance(node, Call):
            try:
                spec = REGISTRY.get(node.fn)
            except KeyError as exc:
                raise FormulaCompileError(str(exc)) from exc
            children = [build(a) for a in node.args]
            try:
                _ = spec.validator([c.type_info for c in children])
            except ValueError as exc:
                raise FormulaCompileError(f"Invalid call {node.fn}: {exc}") from exc
            literal_args = [a.value if isinstance(a, Number) else float("nan") for a in node.args]
            return spec.builder(children, literal_args)
        raise FormulaCompileError(f"Unhandled expression node: {node}")

    root = build(ast_expr)
    return CompiledFormula(
        formula=formula,
        input_names=tuple(inputs.keys()),
        output_type=root.type_info,
        make_feature=root.ctor,
    )
