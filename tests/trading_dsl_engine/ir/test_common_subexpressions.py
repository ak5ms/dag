from __future__ import annotations

import math

from trading_dsl_engine.base.dsl import cat
from trading_dsl_engine.base.parser import Number
from trading_dsl_engine.ir import compile_ir
from trading_dsl_engine.ir.ops import LiteralOp, NaryOp


def test_commutative_scalar_expressions_share_one_ir_node() -> None:
    program = compile_ir("cat(x + y, y + x)")
    additions = [
        node
        for node in program.nodes
        if isinstance(node.op, NaryOp) and node.op.name == "add"
    ]
    assert len(additions) == 1
    assert program.nodes[program.output_id].child_ids[0] == (
        program.nodes[program.output_id].child_ids[1]
    )


def test_nan_literals_share_but_signed_zero_remains_distinct() -> None:
    nan_program = compile_ir(cat(Number(math.nan), Number(math.nan)))
    nan_literals = [
        node for node in nan_program.nodes if isinstance(node.op, LiteralOp)
    ]
    assert len(nan_literals) == 1

    zero_program = compile_ir("cat(-0.0, 0.0)")
    zero_literals = [
        node for node in zero_program.nodes if isinstance(node.op, LiteralOp)
    ]
    assert len(zero_literals) == 2


def test_order_sensitive_minimum_is_not_commutatively_deduplicated() -> None:
    program = compile_ir("cat(minimum(x, y), minimum(y, x))")
    minima = [
        node
        for node in program.nodes
        if isinstance(node.op, NaryOp) and node.op.name == "minimum"
    ]
    assert len(minima) == 2
