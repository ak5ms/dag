from __future__ import annotations

from collections.abc import Mapping
from threading import RLock

from trading_dsl_engine.base.dsl import DEFAULT_DSL_REGISTRY, DSLFunctionRegistry
from trading_dsl_engine.base.parser import Expr, parse_formula
from trading_dsl_engine.ir import frontend as neutral_frontend
from trading_dsl_engine.ir.ops import EmitOp, ReductionOp
from trading_dsl_engine.ir.program import Node, Program
from trading_dsl_engine.ir.types import ValueType, tensor


_COMPILE_LOCK = RLock()


def _broadcast_shapes(
    shapes: tuple[tuple[int | None, ...], ...],
) -> tuple[int | None, ...]:
    rank = max((len(shape) for shape in shapes), default=0)
    result: list[int | None] = []
    for output_axis in range(rank):
        aligned: list[int | None] = []
        for shape in shapes:
            input_axis = output_axis - (rank - len(shape))
            aligned.append(1 if input_axis < 0 else shape[input_axis])
        chosen: int | None = 1
        for extent in aligned:
            if extent == 1:
                continue
            if chosen == 1:
                chosen = extent
                continue
            if extent != chosen:
                raise neutral_frontend.FormulaIRCompileError(
                    f"operands could not be broadcast together: {shapes!r}"
                )
        result.append(chosen)
    return tuple(result)


def _nary_result_type(name: str, children: list[Node]) -> ValueType:
    del name
    if any(child.value_type.kind == "object" for child in children):
        raise neutral_frontend.FormulaIRCompileError(
            "elementwise operators cannot consume object values"
        )
    try:
        shape = _broadcast_shapes(
            tuple(child.value_type.logical_shape for child in children)
        )
    except ValueError as exc:
        raise neutral_frontend.FormulaIRCompileError(
            "elementwise operators require numeric tensor values"
        ) from exc
    dtype = (
        "float64"
        if len({child.value_type.dtype for child in children}) > 1
        else children[0].value_type.dtype
    )
    return tensor(shape, dtype=dtype)


def _lane_state_result_type(name: str, child: Node) -> ValueType:
    if child.value_type.kind == "object":
        raise neutral_frontend.FormulaIRCompileError(
            f"{name} cannot consume object values"
        )
    try:
        child.value_type.logical_shape
    except ValueError as exc:
        raise neutral_frontend.FormulaIRCompileError(
            f"{name} requires a numeric tensor input"
        ) from exc
    return child.value_type


def _depends_on_temporal_reduction(nodes: list[Node]) -> tuple[bool, ...]:
    result: list[bool] = []
    for node in nodes:
        result.append(
            (
                isinstance(node.op, ReductionOp)
                and node.op.temporal
            )
            or any(result[child_id] for child_id in node.child_ids)
        )
    return tuple(result)


def compile_ir(
    formula: str | Expr,
    *,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
    input_value_types: Mapping[str, ValueType] | None = None,
) -> Program:
    """Compile cpp_stream IR with tensor broadcasting and composable reductions.

    A temporal reduction used inside a larger expression exposes its cumulative
    value on each row. The complete downstream expression is implicitly wrapped
    in ``emit('last')`` so only the final value is materialized.
    """

    expression = parse_formula(formula) if isinstance(formula, str) else formula
    with _COMPILE_LOCK:
        original_nary = neutral_frontend._nary_result_type
        original_lane_state = neutral_frontend._lane_state_result_type
        neutral_frontend._nary_result_type = _nary_result_type
        neutral_frontend._lane_state_result_type = _lane_state_result_type
        try:
            builder = neutral_frontend._OuterBuilder(
                dsl_registry or DEFAULT_DSL_REGISTRY,
                {name: index for index, name in enumerate(column_names or ())},
                input_value_types or {},
            )
            root = builder.build(expression)
        finally:
            neutral_frontend._nary_result_type = original_nary
            neutral_frontend._lane_state_result_type = original_lane_state

    temporal = _depends_on_temporal_reduction(builder.nodes)
    if temporal[root] and not (
        isinstance(builder.nodes[root].op, EmitOp)
        or (
            isinstance(builder.nodes[root].op, ReductionOp)
            and builder.nodes[root].op.temporal
        )
    ):
        root = builder._append(
            EmitOp("last"),
            (root,),
            builder.nodes[root].value_type,
        )

    for node_id, node in enumerate(builder.nodes):
        if isinstance(node.op, EmitOp) and node_id != root:
            raise neutral_frontend.FormulaIRCompileError(
                "emit('last') must be the terminal output"
            )

    return Program(tuple(builder.nodes), (root,), tuple(builder.inputs))


FormulaIRCompileError = neutral_frontend.FormulaIRCompileError

# compile.py imports this module before importing lower_program from lowering.
# Replace that module attribute after its base definitions are loaded, while the
# full lowerer continues to reuse its public plan/source/stage data structures.
from trading_dsl_engine.cpp_stream.python import lowering as _lowering  # noqa: E402
from trading_dsl_engine.cpp_stream.python.lowering_full import (  # noqa: E402
    lower_program as _full_lower_program,
)

_lowering.lower_program = _full_lower_program


__all__ = ["FormulaIRCompileError", "compile_ir"]
