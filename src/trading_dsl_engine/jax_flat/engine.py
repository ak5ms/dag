from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple, Number, Universe, parse_formula
from trading_dsl_engine.jax_flat.ops import EwmOp, InputOp, LiteralOp, OP_FACTORIES, Op, GroupByOp


@dataclass(frozen=True)
class DagNode:
    op: Any
    child_ids: tuple[int, ...]


@dataclass(frozen=True)
class StateFieldRef:
    index: int


@dataclass(frozen=True)
class StateLayout:
    node_fields: tuple[StateFieldRef, ...]
    total_leaves: int


@dataclass(frozen=True)
class StreamingProgram:
    nodes: tuple[DagNode, ...]
    outputs: tuple[int, ...]
    input_names: tuple[str, ...]
    state_layout: StateLayout


class JaxFlatRuntime(eqx.Module):
    program: StreamingProgram = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        sample = jnp.zeros((n_instruments,), dtype=jnp.float64)
        states = []
        for node in self.program.nodes:
            if not node.op.is_stateful:
                continue
            states.append(node.op.init_state(sample))
        return tuple(states)

    @jax.jit
    def tick(self, state_leaves, *input_rows):
        values: list[jax.Array] = [jnp.array(0.0)] * len(self.program.nodes)
        new_state = list(state_leaves)

        for idx, node in enumerate(self.program.nodes):
            op = node.op
            if isinstance(op, InputOp):
                values[idx] = input_rows[op.input_index]
                continue
            if isinstance(op, LiteralOp):
                values[idx] = jnp.asarray(op.value, dtype=jnp.float64)
                continue

            child_values = tuple(values[cid] for cid in node.child_ids)
            field = self.program.state_layout.node_fields[idx]
            node_state = None if field.index < 0 else state_leaves[field.index]
            next_state, value = op.tick(node_state, *child_values)
            if field.index >= 0:
                new_state[field.index] = next_state
            values[idx] = value

        outs = tuple(values[i] for i in self.program.outputs)
        return tuple(new_state), outs[0] if len(outs) == 1 else jnp.stack(outs, axis=0)

    def run_batch(self, inputs, states=None):
        if not inputs:
            raise ValueError("run_batch requires at least one input array")
        n_steps = inputs[0].shape[0]
        for arr in inputs[1:]:
            if arr.shape[0] != n_steps:
                raise ValueError("All inputs must have identical timestep length")
        if not states:
            state0 = self.init_state(inputs[0].shape[1])
        else:
            state0 = states

        @jax.jit
        def step(states, rows):
            return self.tick(states, *rows)

        out = jax.lax.scan(step, state0, xs=inputs)
        return out

def _expr_key(node: Expr):
    if isinstance(node, Identifier):
        return ("id", node.name)
    if isinstance(node, Number):
        return ("num", float(node.value))
    if isinstance(node, Call):
        return ("call", node.fn, tuple(_expr_key(a) for a in node.args))
    if isinstance(node, KeyTuple):
        return ("tuple", tuple(_expr_key(item) for item in node.items))
    if isinstance(node, Universe):
        return ("univ", node.groups)
    raise ValueError(f"Unsupported expression: {node}")


def _canonical_groupby_key_items(key: Expr) -> tuple[Expr, ...]:
    if not isinstance(key, KeyTuple):
        key = KeyTuple((key,))
    if sum(1 for item in key.items if isinstance(item, Universe)) > 1:
        raise ValueError("groupby key tuple may contain at most one univ(...) element")
    return key.items




def _resolve_universe_groups(universe: Universe) -> tuple[tuple[int, ...], ...]:
    groups = []
    for g in universe.groups:
        cols = []
        for m in g:
            if not isinstance(m, int):
                raise ValueError("jax_flat univ currently supports integer column indexes only")
            cols.append(m)
        groups.append(tuple(cols))
    return tuple(groups)

def _validate_groupby_canonical_form(expr: Call) -> None:
    if len(expr.args) != 3:
        raise ValueError("groupby only supports canonical form: groupby((key1, ..., maybe_univ, ...), lhs, op_using_self_)")
    _canonical_groupby_key_items(expr.args[0])


def _replace_self(node: Expr, lhs: Expr) -> Expr:
    if isinstance(node, Identifier) and node.name == "self_":
        return lhs
    if isinstance(node, Call):
        return Call(node.fn, tuple(_replace_self(a, lhs) for a in node.args))
    if isinstance(node, KeyTuple):
        return KeyTuple(tuple(_replace_self(a, lhs) for a in node.items))
    return node


def _build_op(expr: Call) -> tuple[Op, int | None]:
    if expr.fn == "groupby":
        _validate_groupby_canonical_form(expr)
        key_items = _canonical_groupby_key_items(expr.args[0])
        universe_items = [item for item in key_items if isinstance(item, Universe)]
        dynamic_items = [item for item in key_items if not isinstance(item, Universe)]
        rhs = expr.args[2]
        if not isinstance(rhs, Call):
            raise ValueError("groupby rhs must be a call expression")
        inner_op, _ = _build_op(rhs)
        universe_groups = _resolve_universe_groups(universe_items[0]) if universe_items else ()
        return GroupByOp(inner_op=inner_op, n_keys=len(dynamic_items), universe_groups=universe_groups), None
    if expr.fn == "ewm" and len(expr.args) == 2 and isinstance(expr.args[1], Number):
        return EwmOp(span=float(expr.args[1].value)), 1
    builder = OP_FACTORIES.get((expr.fn, len(expr.args)))
    if builder is None:
        raise ValueError(f"Unsupported function {expr.fn}/{len(expr.args)} in jax_flat")
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
    if isinstance(expr, Call) and expr.fn == "groupby" and len(expr.args) == 3:
        key_items = _canonical_groupby_key_items(expr.args[0])
        rhs = expr.args[2]
        if isinstance(rhs, Call):
            dynamic_items = [item for item in key_items if not isinstance(item, Universe)]
            rhs_args = tuple(_replace_self(a, expr.args[1]) for a in rhs.args)
            child_ids = tuple(_compile_node(k, memo, nodes, input_names) for k in dynamic_items) + tuple(
                _compile_node(a, memo, nodes, input_names) for a in rhs_args
            )
        else:
            child_ids = tuple(_compile_node(a, memo, nodes, input_names) for a in expr.args)
    else:
        child_ids = tuple(_compile_node(a, memo, nodes, input_names) for a in expr.args)
    op, drop_child_idx = _build_op(expr)
    if drop_child_idx is not None:
        child_ids = tuple(cid for i, cid in enumerate(child_ids) if i != drop_child_idx)
    idx = len(nodes)
    nodes.append(DagNode(op=op, child_ids=child_ids))
    memo[key] = idx
    return idx


def _build_state_layout(nodes: tuple[DagNode, ...], sample: jax.Array) -> StateLayout:
    refs = []
    offset = 0
    for node in nodes:
        if node.op.is_stateful:
            refs.append(StateFieldRef(index=offset))
            offset += 1
        else:
            refs.append(StateFieldRef(index=-1))
    return StateLayout(node_fields=tuple(refs), total_leaves=offset)


def compile_formula(formula: str | Expr) -> JaxFlatRuntime:
    expr = parse_formula(formula) if isinstance(formula, str) else formula
    nodes: list[DagNode] = []
    memo: dict[tuple[Any, ...], int] = {}
    input_names: list[str] = []
    out = _compile_node(expr, memo, nodes, input_names)
    sample = jnp.zeros((1,), dtype=jnp.float64)
    layout = _build_state_layout(tuple(nodes), sample)
    return JaxFlatRuntime(
        program=StreamingProgram(
            nodes=tuple(nodes),
            outputs=(out,),
            input_names=tuple(input_names),
            state_layout=layout,
        )
    )


@eqx.filter_jit
def jit_tick_stream(runtime: JaxFlatRuntime, state_leaves, *input_rows):
    return runtime.tick(state_leaves, *input_rows)


@eqx.filter_jit
def jit_tick(runtime: JaxFlatRuntime, state_leaves, *input_rows):
    return runtime.tick(state_leaves, *input_rows)
