from __future__ import annotations

from dataclasses import dataclass

from numba import int64, types
from numba.experimental import jitclass
from numba.typed import List

from .ops import _make_input_node, _make_literal_node, register_builtin_ops
from .parser import Call, Expr, Identifier, Number, parse_formula
from .registry import REGISTRY, CompiledNode


class FormulaCompileError(ValueError):
    pass


@dataclass(frozen=True)
class CompiledFormulaArtifact:
    compiled: object
    compiled_type: object
    input_names: tuple[str, ...]
    output_kind: str


def _kind_to_code(kind: str) -> int:
    if kind == "scalar":
        return 0
    if kind == "vector":
        return 1
    if kind == "matrix":
        return 2
    raise FormulaCompileError(f"Unknown output kind: {kind}")


def compile_formula(formula: str) -> CompiledFormulaArtifact:
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
    )
