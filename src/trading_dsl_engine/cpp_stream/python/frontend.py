from __future__ import annotations

from collections.abc import Mapping
from threading import RLock

from trading_dsl_engine.base.dsl import DEFAULT_DSL_REGISTRY, DSLFunctionRegistry
from trading_dsl_engine.base.parser import Expr, parse_formula
from trading_dsl_engine.ir import frontend as neutral_frontend
from trading_dsl_engine.ir.ops import EmitOp, LiteralOp, ReductionOp
from trading_dsl_engine.ir.program import Node, Program
from trading_dsl_engine.ir.types import ValueType, tensor


_COMPILE_LOCK = RLock()

def _depends_on_temporal_reduction(nodes: list[Node]) -> tuple[bool, ...]:
    result: list[bool] = []
    for node in nodes:
        result.append(
            (isinstance(node.op, ReductionOp) and node.op.temporal)
            or any(result[child_id] for child_id in node.child_ids)
        )
    return tuple(result)


def compile_ir(
    formula: str | Expr | list[str | Expr] | tuple[str | Expr, ...],
    *,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
    input_value_types: Mapping[str, ValueType] | None = None,
) -> Program:
    """Build one neutral DAG and CSE table for one or many cpp_stream roots."""

    formulas = tuple(formula) if isinstance(formula, (list, tuple)) else (formula,)
    if not formulas:
        raise neutral_frontend.FormulaIRCompileError(
            "compile_ir requires at least one formula"
        )
    expressions = tuple(
        parse_formula(item) if isinstance(item, str) else item for item in formulas
    )

    # cpp_stream currently extends the neutral frontend's tensor broadcasting and
    # shape-preserving temporal semantics. Keep that compatibility shim confined to
    # IR construction; lowering/codegen are imported explicitly by compile.py.
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
            roots = tuple(builder.build(expression) for expression in expressions)
        finally:
            neutral_frontend._nary_result_type = original_nary
            neutral_frontend._lane_state_result_type = original_lane_state

    temporal = _depends_on_temporal_reduction(builder.nodes)
    resolved_roots: list[int] = []
    for root in roots:
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
        resolved_roots.append(root)

    root_set = frozenset(resolved_roots)
    child_ids = frozenset(
        child_id for node in builder.nodes for child_id in node.child_ids
    )
    for node_id, node in enumerate(builder.nodes):
        if isinstance(node.op, EmitOp) and (
            node_id not in root_set or node_id in child_ids
        ):
            raise neutral_frontend.FormulaIRCompileError(
                "emit('last') must be a terminal output"
            )

    return Program(
        tuple(builder.nodes),
        tuple(resolved_roots),
        tuple(builder.inputs),
    )


FormulaIRCompileError = neutral_frontend.FormulaIRCompileError


__all__ = ["FormulaIRCompileError", "compile_ir"]
