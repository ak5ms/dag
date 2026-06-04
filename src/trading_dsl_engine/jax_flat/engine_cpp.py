from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from trading_dsl_engine.base.dsl import DSLFunctionRegistry
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.jax_flat.engine import (
    DagNode,
    InnerGraphOp,
    StateLayout,
    StreamingProgram,
    compile_formula,
)
from trading_dsl_engine.jax_flat.ops import (
    CumsumOp,
    EwmOp,
    FFillOp,
    GroupByOp,
    InputOp,
    LiteralOp,
    NaryOp,
    Op,
    RidgeOp,
    ShiftOp,
)


@dataclass(frozen=True)
class CppFlatRuntime:
    """Allocation-conscious native tick-path runtime for supported flat formulas.

    The C++ core owns flattened node specs and preallocates per-node scratch in
    ``init_state``. ``tick_into`` is the intended hot path: callers pass the
    reusable state object, a reusable output row, and current input rows.
    ``tick`` remains a convenience wrapper that allocates only the returned row.
    """

    program: StreamingProgram
    core: Any
    supported_ops: tuple[str, ...]

    def init_state(self, n_instruments: int):
        return self.core.init_state(n_instruments)

    def tick(self, state, *input_rows):
        return self.core.tick(state, *input_rows)

    def tick_into(self, state, out, *input_rows) -> None:
        self.core.tick_into(state, out, *input_rows)

    def run_batch_tick(self, inputs, states=None, out=None):
        inputs = _normalize_batch_inputs_for_program(self.program, inputs)
        if not inputs:
            raise ValueError("run_batch_tick requires at least one input array")
        n_steps, n_instruments = inputs[0].shape
        for arr in inputs[1:]:
            if arr.shape != (n_steps, n_instruments):
                raise ValueError("All inputs must share aligned shape (time, n_instruments)")
        state = states or self.init_state(n_instruments)
        np_inputs = tuple(np.asarray(arr, dtype=np.float64) for arr in inputs)
        if out is None:
            raw = self.core.run_batch(state, *np_inputs)
            return state, _reshape_cpp_batch_output(self.program, raw, n_steps, n_instruments)
        self.core.run_batch_into(state, out, *np_inputs)
        return state, out


def compile_formula_cpp(formula: str | Expr, dsl_registry: DSLFunctionRegistry | None = None) -> CppFlatRuntime:
    """Compile a supported jax_flat formula to the native C++ tick runtime.

    This backend intentionally covers the allocation-sensitive scalar/vector tick
    subset first. Unsupported operators raise a clear ``NotImplementedError`` so
    callers can keep using the JAX-flat runtime for full DSL coverage.
    """
    from trading_dsl_engine.jax_flat import _cpp_flat

    runtime = compile_formula(formula, dsl_registry=dsl_registry)
    node_specs, supported = _cpp_node_specs(runtime.program)
    core = _cpp_flat.make_runtime(node_specs, runtime.program.outputs[0], runtime.program.state_layout.total_leaves)
    return CppFlatRuntime(program=runtime.program, core=core, supported_ops=tuple(sorted(set(supported))))


def _reshape_cpp_batch_output(program: StreamingProgram, raw, n_steps: int, n_instruments: int):
    root = program.nodes[program.outputs[0]].op
    arr = np.asarray(raw)
    if root.output_kind == "matrix" and root.output_width is not None:
        return arr.reshape((n_steps, n_instruments, int(root.output_width)))
    if root.output_kind == "scalar" and arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr


def _normalize_batch_inputs_for_program(program: StreamingProgram, inputs):
    if isinstance(inputs, dict):
        missing = [name for name in program.input_names if name not in inputs]
        if missing:
            raise ValueError(f"Missing jax_flat C++ input(s): {missing}")
        inputs = tuple(inputs[name] for name in program.input_names)
    else:
        inputs = tuple(inputs)
    if len(inputs) != len(program.input_names):
        raise ValueError(f"expected {len(program.input_names)} input array(s), got {len(inputs)}")
    for name, arr in zip(program.input_names, inputs):
        if np.asarray(arr).ndim != 2:
            raise ValueError(f"Expected 2D input for '{name}', got shape {np.asarray(arr).shape}")
    return inputs


def _cpp_node_specs(program: StreamingProgram):
    if len(program.outputs) != 1:
        raise NotImplementedError("C++ jax_flat currently supports exactly one output")
    specs = []
    supported = []

    def spec_tuple(name, children=(), input_index=-1, state_index=-1, literal=0.0, param=0.0, int_param=0, width=1, feature_widths=()):
        return (name, tuple(children), input_index, state_index, float(literal), float(param), int(int_param), int(width or 1), tuple(int(w) for w in feature_widths))

    for idx, node in enumerate(program.nodes):
        op = node.op
        field = program.state_layout.node_fields[idx]
        state_index = field.index
        width = op.output_width or 1
        if isinstance(op, InputOp):
            specs.append(spec_tuple("input", input_index=op.input_index, width=1))
            supported.append("input")
            continue
        if isinstance(op, LiteralOp):
            specs.append(spec_tuple("literal", literal=op.value, width=1))
            supported.append("literal")
            continue
        if isinstance(op, GroupByOp):
            group_spec = _cpp_groupby_spec(op, node.child_ids, state_index, spec_tuple)
            specs.append(group_spec)
            supported.append("group_cumsum")
            continue
        if isinstance(op, NaryOp) and op.cpp_name is not None:
            cpp_width = width
            if op.cpp_name == "get_beta":
                ridge_child = program.nodes[node.child_ids[0]].op
                if not isinstance(ridge_child, RidgeOp):
                    raise NotImplementedError("C++ jax_flat get_beta expects direct Ridge child")
                cpp_width = sum(ridge_child.feature_widths)
            specs.append(spec_tuple(op.cpp_name, node.child_ids, int_param=op.cpp_int_param, param=op.cpp_param, width=cpp_width))
            supported.append(op.cpp_name)
            continue
        if isinstance(op, CumsumOp):
            specs.append(spec_tuple("cumsum", node.child_ids, state_index=state_index, width=width))
            supported.append("cumsum")
            continue
        if isinstance(op, EwmOp) and op.span is not None:
            specs.append(spec_tuple("ewm", node.child_ids, state_index=state_index, param=op.span, width=width))
            supported.append("ewm")
            continue
        if isinstance(op, FFillOp) and not op.dynamic_limit:
            limit = -1 if op.limit is None else op.limit
            specs.append(spec_tuple("ffill", node.child_ids, state_index=state_index, int_param=limit, width=width))
            supported.append("ffill")
            continue
        if isinstance(op, ShiftOp):
            specs.append(spec_tuple("shift", node.child_ids, state_index=state_index, int_param=op.max_size, width=width))
            supported.append("shift")
            continue
        if isinstance(op, RidgeOp):
            k = sum(op.feature_widths)
            specs.append(spec_tuple("ridge", node.child_ids, state_index=state_index, width=1, feature_widths=op.feature_widths))
            supported.append("ridge")
            continue
        raise NotImplementedError(f"C++ jax_flat does not yet support node {idx}: {type(op).__name__}")
    return tuple(specs), supported


def _cpp_groupby_spec(op: GroupByOp, child_ids: tuple[int, ...], state_index: int, spec_tuple):
    if op.universe_groups is not None:
        raise NotImplementedError("C++ jax_flat groupby currently supports dynamic keys without univ(...) only")
    inner = op.inner_op
    if not isinstance(inner, InnerGraphOp):
        raise NotImplementedError("C++ jax_flat groupby expects lowered inner graph")
    if len(inner.nodes) != 2:
        raise NotImplementedError("C++ jax_flat groupby currently supports groupby(..., cumsum(self_))")
    if not isinstance(inner.nodes[0].op, InputOp) or not isinstance(inner.nodes[1].op, CumsumOp):
        raise NotImplementedError("C++ jax_flat groupby currently supports groupby(..., cumsum(self_))")
    return spec_tuple("group_cumsum", child_ids, state_index=state_index, param=op.capacity, int_param=op.n_keys, width=1)


def _require_vector_cpp_op(op: Op, name: str) -> None:
    if op.output_kind != "vector" or op.output_width not in (None, 1):
        raise NotImplementedError(f"C++ jax_flat op {name!r} currently supports vector outputs only")

