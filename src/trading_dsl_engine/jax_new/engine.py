from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from trading_dsl_engine.base.parser import Call, Expr, Identifier, Number, parse_formula
from trading_dsl_engine.jax_new.ops import (
    EwmOp,
    EmptyState,
    FfillOp,
    InputOp,
    LiteralOp,
    Op,
    OP_FACTORIES,
    RollingQuantileOp,
    ShiftOp,
    state_value,
)


@dataclass(frozen=True)
class DagNode:
    op: Any
    child_ids: tuple[int, ...]


@dataclass(frozen=True)
class DagProgram:
    nodes: tuple[DagNode, ...]
    outputs: tuple[int, ...]
    input_names: tuple[str, ...]


class JaxDagRuntime(eqx.Module):
    program: DagProgram = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        sample = jnp.zeros((n_instruments,), dtype=jnp.float64)
        return tuple(node.op.init_state(sample) for node in self.program.nodes)

    def tick(self, states, *input_rows):
        new_states: list[Any] = []
        for idx, node in enumerate(self.program.nodes):
            op = node.op
            if isinstance(op, InputOp):
                new_states.append(EmptyState(value=input_rows[op.input_index]))
                continue
            if isinstance(op, LiteralOp):
                new_states.append(states[idx])
                continue
            child_states = tuple(new_states[cid] for cid in node.child_ids)
            new_states.append(op.step(states[idx], *child_states))
        outputs = tuple(_format_output(state_value(new_states[i]), self.program.nodes[i].op.output_kind) for i in self.program.outputs)
        return tuple(new_states), outputs[0] if len(outputs) == 1 else jnp.stack(outputs, axis=0)


def _format_output(value, output_kind: str):
    if output_kind == "vector":
        return value[:, None]
    if output_kind == "scalar":
        return value.reshape(1, 1) if value.ndim == 0 else value
    return value

@eqx.filter_jit
def jit_tick(runtime: JaxDagRuntime, states, *input_rows):
    return runtime.tick(states, *input_rows)


def _expr_key(node: Expr):
    if isinstance(node, Identifier):
        return ("id", node.name)
    if isinstance(node, Number):
        return ("num", float(node.value))
    if isinstance(node, Call):
        return ("call", node.fn, tuple(_expr_key(a) for a in node.args))
    raise ValueError(f"Unsupported expression: {node}")

def _build_op(expr: Call) -> tuple[Op, int | None]:
    fn = expr.fn
    if fn == "ewm" and len(expr.args) == 2 and isinstance(expr.args[1], Number):
        return EwmOp(span=float(expr.args[1].value)), 1
    if fn == "shift" and len(expr.args) in (2, 3):
        max_size_expr = expr.args[2] if len(expr.args) == 3 else expr.args[1]
        if not isinstance(max_size_expr, Number):
            raise ValueError("shift requires literal max_size in jax_new")
        return ShiftOp(max_size=max(1, int(max_size_expr.value))), None
    if fn == "rolling_quantile" and len(expr.args) == 3 and isinstance(expr.args[1], Number):
        return RollingQuantileOp(window=max(1, int(expr.args[1].value))), None
    if fn == "ffill" and len(expr.args) == 2:
        return FfillOp(), None
    builder = OP_FACTORIES.get((fn, len(expr.args)))
    if builder is None:
        raise ValueError(f"Unsupported function {expr.fn}/{len(expr.args)} in jax_new")
    return builder(), None

def _compile_node(expr: Expr, memo: dict[tuple[Any, ...], int], nodes: list[DagNode], input_names: list[str]) -> int:
    key = _expr_key(expr)
    if key in memo:
        return memo[key]
    if isinstance(expr, Identifier):
        if expr.name not in input_names:
            input_names.append(expr.name)
        idx = len(nodes)
        nodes.append(DagNode(op=InputOp(input_index=input_names.index(expr.name)), child_ids=()))
        memo[key] = idx
        return idx
    if isinstance(expr, Number):
        idx = len(nodes)
        nodes.append(DagNode(op=LiteralOp(float(expr.value)), child_ids=()))
        memo[key] = idx
        return idx
    if not isinstance(expr, Call):
        raise ValueError(f"Unsupported node {expr}")
    child_ids = tuple(_compile_node(a, memo, nodes, input_names) for a in expr.args)
    op, drop_child_idx = _build_op(expr)
    if drop_child_idx is not None:
        child_ids = tuple(cid for i, cid in enumerate(child_ids) if i != drop_child_idx)
    idx = len(nodes)
    nodes.append(DagNode(op=op, child_ids=child_ids))
    memo[key] = idx
    return idx


def compile_formula(formula: str | Expr) -> JaxDagRuntime:
    expr = parse_formula(formula) if isinstance(formula, str) else formula
    nodes: list[DagNode] = []
    memo: dict[tuple[Any, ...], int] = {}
    input_names: list[str] = []
    out = _compile_node(expr, memo, nodes, input_names)
    program = DagProgram(nodes=tuple(nodes), outputs=(out,), input_names=tuple(input_names))
    return JaxDagRuntime(program=program)
