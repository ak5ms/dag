from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from trading_dsl_engine.dsl import DEFAULT_DSL_REGISTRY, DSLFunctionRegistry
from trading_dsl_engine.engine import build_engine as _build_compat_engine
from trading_dsl_engine.engine import run_batch_from_mapping as _compat_run_batch_from_mapping
from trading_dsl_engine.engine import update_from_mapping as _compat_update_from_mapping
from trading_dsl_engine.parser import Call, Expr, Identifier, Number, Universe, parse_formula
from trading_dsl_engine.registry import TypeInfo


class _Dim:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name


Time = _Dim("Time")
Instrument = _Dim("Instrument")


@dataclass(frozen=True)
class ArrayType:
    dtype: np.dtype
    dims: tuple[_Dim, ...]


class _DTypeFactory:
    def __init__(self, dtype: str):
        self.dtype = np.dtype(dtype)

    def __getitem__(self, dims) -> ArrayType:
        if not isinstance(dims, tuple):
            dims = (dims,)
        return ArrayType(self.dtype, dims)


Float64 = _DTypeFactory("float64")
Float32 = _DTypeFactory("float32")
Int32 = _DTypeFactory("int32")
Int64 = _DTypeFactory("int64")


@dataclass(frozen=True)
class Schema:
    inputs: dict[str, ArrayType]
    n_instruments: int
    columns: list[str] | tuple[str, ...] | None = None
    universes: dict[str, tuple[tuple[str | int, ...], ...]] | None = None
    key_domains: dict[str, tuple[int, ...] | list[int] | range] | None = None
    layout: str = "time_major"

    def __post_init__(self) -> None:
        if self.layout != "time_major":
            raise ValueError("Only time_major layout is currently supported by the schema-bound fast path")
        if self.n_instruments <= 0:
            raise ValueError("n_instruments must be positive")


@dataclass(frozen=True)
class ShapeInfo:
    kind: str
    rows: int
    cols: int

    @property
    def batch_tail(self) -> tuple[int, ...]:
        if self.kind == "scalar":
            return ()
        if self.kind == "vector":
            return (self.rows,)
        if self.kind == "matrix":
            return (self.rows, self.cols)
        raise ValueError(f"Object outputs are not batch-materializable: {self.kind}")


@dataclass(frozen=True)
class OutputSchema:
    kind: str
    width: int
    dtype: np.dtype
    shape: ShapeInfo


class OpKind(str, Enum):
    INPUT = "input"
    LITERAL = "literal"
    ELEMENTWISE = "elementwise"
    REDUCER = "reducer"
    CUMSUM = "cumsum"
    EWM = "ewm"
    SHIFT = "shift"
    XS_RANK = "xs_rank"
    ROLLING_QUANTILE = "rolling_quantile"
    OUTER = "outer"
    BSPLINE = "bspline"
    COL = "col"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class GroupScope:
    kind: str
    key_node: int | None = None
    universe: tuple[tuple[int, ...], ...] | None = None


@dataclass(frozen=True)
class IRNode:
    op: OpKind
    inputs: tuple[int, ...]
    type_info: TypeInfo
    shape: ShapeInfo
    dtype: np.dtype
    stateful: bool
    fusible: bool
    group_scope: GroupScope | None
    literals: tuple[float | int | str, ...] = ()
    name: str = ""


@dataclass(frozen=True)
class BufferPlan:
    node_buffers: dict[int, tuple[int, ...]]
    state_buffers: dict[int, dict[str, tuple[int, ...]]]
    scratch_buffers: dict[int, dict[str, tuple[int, ...]]]
    allocation_count: int


@dataclass(frozen=True)
class RuntimePlan:
    ir_nodes: tuple[IRNode, ...]
    input_names: tuple[str, ...]
    output_schema: OutputSchema
    buffers: BufferPlan
    fused_regions: tuple[tuple[int, ...], ...]
    fallback_reason: str | None = None


@dataclass
class ProgramState:
    buffers: dict[int, np.ndarray]
    state: dict[int, dict[str, Any]]
    initialized: bool = True


@dataclass
class ProgramWorkspace:
    scratch: dict[int, dict[str, np.ndarray]]
    allocation_count: int


_ELEMENTWISE = {
    "add", "sub", "mul", "div", "floordiv", "mod", "eq", "ne", "lt", "gt", "and", "or", "and_", "or_",
    "xor", "where", "abs", "isnan", "fillna", "ln", "ceil", "floor", "exp", "sign", "purify", "arctan", "pow",
}
_BINARY = {"add", "sub", "mul", "div", "floordiv", "mod", "eq", "ne", "lt", "gt", "and", "or", "and_", "or_", "xor", "fillna", "pow"}
_UNARY = {"abs", "isnan", "ln", "ceil", "floor", "exp", "sign", "purify", "arctan"}


class _FastPlanBuilder:
    def __init__(self, schema: Schema, dsl_registry: DSLFunctionRegistry | None):
        self.schema = schema
        self.dsl_registry = dsl_registry or DEFAULT_DSL_REGISTRY
        self.nodes: list[IRNode] = []
        self.input_names: list[str] = []
        self.input_indexes: dict[str, int] = {}
        self.cache: dict[tuple, int] = {}
        self.column_name_to_index = {name: i for i, name in enumerate(schema.columns or ())}

    def _key(self, node: Expr) -> tuple:
        if isinstance(node, Identifier):
            return ("id", node.name)
        if isinstance(node, Number):
            return ("num", node.value)
        if isinstance(node, Call):
            return ("call", node.fn, tuple(self._key(a) for a in node.args))
        if isinstance(node, Universe):
            return ("univ", node.groups)
        raise ValueError(f"Unsupported expression node: {node}")

    def build(self, node: Expr, depth: int = 0) -> int:
        if depth > 256:
            raise ValueError("Exceeded max DSL expansion depth (256)")
        if isinstance(node, Call):
            py_fn = self.dsl_registry.get(node.fn)
            if py_fn is not None:
                return self.build(py_fn(*node.args), depth + 1)
        key = self._key(node)
        if key in self.cache:
            return self.cache[key]

        if isinstance(node, Identifier):
            if node.name not in self.schema.inputs:
                raise ValueError(f"Input '{node.name}' is not declared in schema")
            if node.name not in self.input_indexes:
                self.input_indexes[node.name] = len(self.input_names)
                self.input_names.append(node.name)
            idx = len(self.nodes)
            ir = IRNode(
                op=OpKind.INPUT,
                inputs=(),
                type_info=TypeInfo("vector"),
                shape=ShapeInfo("vector", self.schema.n_instruments, 1),
                dtype=self.schema.inputs[node.name].dtype,
                stateful=False,
                fusible=True,
                group_scope=None,
                literals=(self.input_indexes[node.name],),
                name=node.name,
            )
        elif isinstance(node, Number):
            idx = len(self.nodes)
            ir = IRNode(OpKind.LITERAL, (), TypeInfo("scalar"), ShapeInfo("scalar", 1, 1), np.dtype("float64"), False, True, None, (node.value,))
        elif isinstance(node, Call):
            if node.fn == "groupby":
                raise ValueError("groupby is routed to compatibility runtime unless a specialized grouped fast path is selected")
            child_ids = tuple(self.build(a, depth + 1) for a in node.args)
            child_shapes = [self.nodes[i].shape for i in child_ids]
            idx = len(self.nodes)
            ir = self._call_node(node, child_ids, child_shapes)
        else:
            raise ValueError(f"Unsupported expression in fast runtime: {node}")

        self.nodes.append(ir)
        self.cache[key] = idx
        return idx

    def _broadcast_shape(self, shapes: list[ShapeInfo]) -> ShapeInfo:
        kind = "scalar"
        rows = 1
        cols = 1
        for s in shapes:
            if s.kind == "matrix":
                kind = "matrix"
            elif s.kind == "vector" and kind == "scalar":
                kind = "vector"
            rows = max(rows, s.rows)
            cols = max(cols, s.cols)
        if kind == "vector":
            cols = 1
        return ShapeInfo(kind, rows, cols)

    def _call_node(self, node: Call, child_ids: tuple[int, ...], child_shapes: list[ShapeInfo]) -> IRNode:
        fn = node.fn
        literals = tuple(a.value if isinstance(a, Number) else float("nan") for a in node.args)
        if fn in _ELEMENTWISE:
            arity = 3 if fn == "where" else 1 if fn in _UNARY else 2
            if len(child_ids) != arity:
                raise ValueError(f"{fn} expects {arity} args")
            shape = self._broadcast_shape(child_shapes)
            return IRNode(OpKind.ELEMENTWISE, child_ids, TypeInfo(shape.kind), shape, np.dtype("float64"), False, True, None, (fn,))
        if fn == "mean":
            return IRNode(OpKind.REDUCER, child_ids, TypeInfo("scalar"), ShapeInfo("scalar", 1, 1), np.dtype("float64"), False, False, None, (fn,))
        if fn == "cumsum":
            self._require_kind(fn, child_shapes, ["vector"])
            return IRNode(OpKind.CUMSUM, child_ids, TypeInfo("vector"), child_shapes[0], np.dtype("float64"), True, False, None)
        if fn == "ewm":
            self._require_kind(fn, child_shapes[:1], ["vector"])
            if len(literals) < 2 or np.isnan(literals[1]):
                raise ValueError("ewm span must be a static literal in the fast path")
            return IRNode(OpKind.EWM, child_ids, TypeInfo("vector"), child_shapes[0], np.dtype("float64"), True, False, None, (2.0 / (float(literals[1]) + 1.0),))
        if fn == "shift":
            self._require_kind(fn, child_shapes[:1], ["vector"])
            max_lit = literals[2] if len(literals) == 3 else literals[1]
            if np.isnan(max_lit):
                raise ValueError("shift max_size must be static in the fast path")
            return IRNode(OpKind.SHIFT, child_ids, TypeInfo("vector"), child_shapes[0], np.dtype("float64"), True, False, None, (int(round(max_lit)),))
        if fn == "xs_rank":
            self._require_kind(fn, child_shapes, ["vector"])
            return IRNode(OpKind.XS_RANK, child_ids, TypeInfo("vector"), child_shapes[0], np.dtype("float64"), False, False, None)
        if fn == "rolling_quantile":
            self._require_kind(fn, child_shapes[:1], ["vector"])
            if len(literals) != 3 or np.isnan(literals[1]) or np.isnan(literals[2]):
                raise ValueError("rolling_quantile window and q must be static literals in the fast path")
            window = int(round(literals[1]))
            q = float(literals[2])
            if window <= 0 or q < 0.0 or q > 1.0:
                raise ValueError("Invalid rolling_quantile window/q")
            return IRNode(OpKind.ROLLING_QUANTILE, child_ids, TypeInfo("vector"), child_shapes[0], np.dtype("float64"), True, False, None, (window, q))
        if fn == "outer":
            self._require_kind(fn, child_shapes, ["vector"])
            n = self.schema.n_instruments
            return IRNode(OpKind.OUTER, child_ids, TypeInfo("matrix"), ShapeInfo("matrix", n, n), np.dtype("float64"), False, False, None)
        if fn == "bspline":
            self._require_kind(fn, child_shapes[:1], ["vector"])
            if len(literals) != 2 or np.isnan(literals[1]):
                raise ValueError("bspline n_basis must be static in the fast path")
            width = int(round(literals[1]))
            if width <= 0:
                raise ValueError("bspline n_basis must be >= 1")
            return IRNode(OpKind.BSPLINE, child_ids, TypeInfo("matrix"), ShapeInfo("matrix", self.schema.n_instruments, width), np.dtype("float64"), False, False, None, (width,))
        if fn == "col":
            self._require_kind(fn, child_shapes[:1], ["matrix"])
            col = int(round(literals[1]))
            if col < 0 or col >= child_shapes[0].cols:
                raise ValueError("col index out of bounds in fast path")
            return IRNode(OpKind.COL, child_ids, TypeInfo("vector"), ShapeInfo("vector", self.schema.n_instruments, 1), np.dtype("float64"), False, True, None, (col,))
        raise ValueError(f"Operator '{fn}' is not implemented by schema-bound fast runtime")

    def _require_kind(self, fn: str, shapes: list[ShapeInfo], kinds: list[str]) -> None:
        if len(shapes) != len(kinds) or any(s.kind != k for s, k in zip(shapes, kinds)):
            raise ValueError(f"{fn} expects {', '.join(kinds)} input(s)")


def _plan_buffers(nodes: tuple[IRNode, ...]) -> BufferPlan:
    node_buffers: dict[int, tuple[int, ...]] = {}
    state_buffers: dict[int, dict[str, tuple[int, ...]]] = {}
    scratch_buffers: dict[int, dict[str, tuple[int, ...]]] = {}
    allocs = 0
    for i, node in enumerate(nodes):
        if node.op not in (OpKind.INPUT, OpKind.LITERAL):
            node_buffers[i] = node.shape.batch_tail or (1,)
            allocs += 1
        if node.op == OpKind.CUMSUM:
            state_buffers[i] = {"sum": (node.shape.rows,)}; allocs += 1
        elif node.op == OpKind.EWM:
            state_buffers[i] = {"value": node.shape.batch_tail, "has_state": (1,)}; allocs += 2
        elif node.op == OpKind.SHIFT:
            ring_len = int(node.literals[0]) + 1
            state_buffers[i] = {"ring": (node.shape.rows, ring_len), "meta": (2,)}; allocs += 2
        elif node.op == OpKind.XS_RANK:
            scratch_buffers[i] = {"values": (node.shape.rows,), "valid_index": (node.shape.rows,), "order": (node.shape.rows,)}; allocs += 3
        elif node.op == OpKind.ROLLING_QUANTILE:
            window = int(node.literals[0])
            state_buffers[i] = {"ring": (node.shape.rows, window), "meta": (2,)}; allocs += 2
            scratch_buffers[i] = {"values": (window,)}; allocs += 1
    return BufferPlan(node_buffers, state_buffers, scratch_buffers, allocs)


def _fusion_regions(nodes: tuple[IRNode, ...]) -> tuple[tuple[int, ...], ...]:
    regions: list[tuple[int, ...]] = []
    current: list[int] = []
    for i, node in enumerate(nodes):
        if node.fusible and node.op == OpKind.ELEMENTWISE:
            current.append(i)
        elif current:
            regions.append(tuple(current)); current = []
    if current:
        regions.append(tuple(current))
    return tuple(regions)


class CompiledProgram:
    def __init__(self, formula: str | Expr, schema: Schema, runtime_plan: RuntimePlan, ast_expr: Expr, fallback_engine=None):
        self.formula = formula
        self.schema = schema
        self.runtime_plan = runtime_plan
        self.output_schema = runtime_plan.output_schema
        self.input_names = runtime_plan.input_names
        self.ir_nodes = runtime_plan.ir_nodes
        self.ast_expr = ast_expr
        self._fallback_engine = fallback_engine
        self.fast_path = fallback_engine is None

    @property
    def allocation_count(self) -> int:
        return self.runtime_plan.buffers.allocation_count

    def new_state(self) -> ProgramState:
        if not self.fast_path:
            return ProgramState({}, {"engine": self._fresh_fallback_engine()})
        buffers: dict[int, np.ndarray] = {}
        state: dict[int, dict[str, Any]] = {}
        for node_id, shape in self.runtime_plan.buffers.node_buffers.items():
            buffers[node_id] = np.empty(shape, dtype=np.float64)
        for node_id, specs in self.runtime_plan.buffers.state_buffers.items():
            state[node_id] = {}
            node = self.ir_nodes[node_id]
            for name, shape in specs.items():
                if name == "ring":
                    arr = np.empty(shape, dtype=np.float64); arr[:] = np.nan
                elif name == "meta":
                    arr = np.zeros(shape, dtype=np.int64)
                elif name == "has_state":
                    arr = np.zeros(shape, dtype=np.bool_)
                else:
                    arr = np.zeros(shape, dtype=np.float64)
                state[node_id][name] = arr
            if node.op == OpKind.EWM and "value" in state[node_id]:
                state[node_id]["value"][:] = np.nan
        return ProgramState(buffers, state)

    def new_workspace(self) -> ProgramWorkspace:
        scratch: dict[int, dict[str, np.ndarray]] = {}
        for node_id, specs in self.runtime_plan.buffers.scratch_buffers.items():
            scratch[node_id] = {}
            for name, shape in specs.items():
                dtype = np.int64 if name in ("valid_index", "order") else np.float64
                scratch[node_id][name] = np.empty(shape, dtype=dtype)
        return ProgramWorkspace(scratch, self.runtime_plan.buffers.allocation_count)

    def initialize(self, n_time: int | None = None) -> tuple[ProgramState, ProgramWorkspace]:
        return self.new_state(), self.new_workspace()

    def bind(self, **inputs: np.ndarray) -> "BoundProgram":
        return BoundProgram(self, inputs)

    def step(self, state: ProgramState, tick_inputs: dict[str, np.ndarray] | tuple[np.ndarray, ...], tick_out: np.ndarray | None = None, workspace: ProgramWorkspace | None = None):
        workspace = workspace or self.new_workspace()
        if not self.fast_path:
            engine = state.state["engine"]
            data = tick_inputs if isinstance(tick_inputs, dict) else {name: tick_inputs[i] for i, name in enumerate(self.input_names)}
            y = _compat_update_from_mapping(engine, data)
            return self._copy_step_output(y, tick_out)
        if isinstance(tick_inputs, dict):
            rows = tuple(np.asarray(tick_inputs[name]) for name in self.input_names)
        else:
            rows = tuple(np.asarray(x) for x in tick_inputs)
        result = self._execute_step(state, workspace, rows)
        return self._copy_step_output(result, tick_out)

    def _copy_step_output(self, result, tick_out):
        if tick_out is None:
            if self.output_schema.kind == "scalar":
                return np.array(result[0] if np.ndim(result) == 1 else result[0, 0])
            return np.array(result[:, 0] if getattr(result, "ndim", 0) == 2 and result.shape[1] == 1 else result, copy=True)
        if self.output_schema.kind == "scalar":
            tick_out[...] = result[0] if np.ndim(result) == 1 else result[0, 0]
        elif self.output_schema.kind == "vector" and getattr(result, "ndim", 0) == 2:
            tick_out[...] = result[:, 0]
        else:
            tick_out[...] = result
        return tick_out

    def _fresh_fallback_engine(self):
        return _build_compat_engine(self.formula, column_names=self.schema.columns)

    def _execute_step(self, state: ProgramState, workspace: ProgramWorkspace, rows: tuple[np.ndarray, ...]):
        values: dict[int, Any] = {}
        for i, node in enumerate(self.ir_nodes):
            if node.op == OpKind.INPUT:
                row = rows[int(node.literals[0])]
                if row.shape[0] != self.schema.n_instruments:
                    raise ValueError("Live tick input width does not match schema.n_instruments")
                values[i] = row
            elif node.op == OpKind.LITERAL:
                values[i] = float(node.literals[0])
            else:
                out = state.buffers[i]
                self._eval_node(i, node, values, state, workspace, out)
                values[i] = out
        return values[len(self.ir_nodes) - 1]

    def _eval_node(self, node_id: int, node: IRNode, values: dict[int, Any], state: ProgramState, workspace: ProgramWorkspace, out: np.ndarray) -> None:
        if node.op == OpKind.ELEMENTWISE:
            _eval_elementwise(str(node.literals[0]), [values[c] for c in node.inputs], out)
        elif node.op == OpKind.REDUCER:
            out[0] = _nanmean(values[node.inputs[0]])
        elif node.op == OpKind.CUMSUM:
            x = values[node.inputs[0]]
            s = state.state[node_id]["sum"]
            for i in range(s.shape[0]):
                xv = _get1(x, i)
                if not np.isnan(xv):
                    s[i] += xv
                out[i] = s[i]
        elif node.op == OpKind.EWM:
            st = state.state[node_id]
            x = values[node.inputs[0]]; val = st["value"]; has = st["has_state"]
            alpha = float(node.literals[0]); beta = 1.0 - alpha
            if not has[0]:
                _copy_like(x, val); has[0] = True
            else:
                for idx in np.ndindex(val.shape):
                    xv = _get_nd(x, idx); sv = val[idx]
                    if np.isnan(xv):
                        val[idx] = sv
                    elif np.isnan(sv):
                        val[idx] = xv
                    else:
                        val[idx] = alpha * xv + beta * sv
            out[...] = val
        elif node.op == OpKind.SHIFT:
            st = state.state[node_id]
            ring = st["ring"]; meta = st["meta"]
            head = int(meta[0]); size = int(meta[1]); ring_len = ring.shape[1]
            x = values[node.inputs[0]]; lag_value = _scalar_value(values[node.inputs[1]])
            for i in range(ring.shape[0]):
                ring[i, head] = _get1(x, i)
            if np.isnan(lag_value):
                out[:] = np.nan
            else:
                lag = int(round(lag_value))
                if lag < 0 or lag > int(node.literals[0]):
                    raise ValueError("shift nlag must be between 0 and max_size")
                if size >= lag:
                    idx = head - lag
                    if idx < 0: idx += ring_len
                    out[:] = ring[:, idx]
                else:
                    out[:] = np.nan
            meta[0] = 0 if head + 1 == ring_len else head + 1
            meta[1] = min(size + 1, ring_len)
        elif node.op == OpKind.XS_RANK:
            _xs_rank_inplace(values[node.inputs[0]], out, workspace.scratch[node_id])
        elif node.op == OpKind.ROLLING_QUANTILE:
            st = state.state[node_id]
            ring = st["ring"]; meta = st["meta"]; window = ring.shape[1]
            head = int(meta[0]); size = int(meta[1]); x = values[node.inputs[0]]
            for i in range(ring.shape[0]):
                ring[i, head] = _get1(x, i)
            head = 0 if head + 1 == window else head + 1
            size = min(size + 1, window)
            q = float(node.literals[1]); scratch = workspace.scratch[node_id]["values"]
            for i in range(ring.shape[0]):
                for j in range(size):
                    idx = head - size + j
                    if idx < 0: idx += window
                    scratch[j] = ring[i, idx]
                out[i] = _quantile_inplace(scratch, size, q)
            meta[0] = head; meta[1] = size
        elif node.op == OpKind.OUTER:
            x = values[node.inputs[0]]
            for i in range(out.shape[0]):
                xi = _get1(x, i)
                for j in range(out.shape[1]):
                    out[i, j] = xi * _get1(x, j)
        elif node.op == OpKind.BSPLINE:
            x = values[node.inputs[0]]; n_basis = out.shape[1]; sigma = 1.0 / n_basis
            for i in range(out.shape[0]):
                xv = _get1(x, i)
                if np.isnan(xv):
                    out[i, :] = np.nan
                else:
                    if xv < 0.0: xv = 0.0
                    elif xv > 1.0: xv = 1.0
                    _basis_row(n_basis, sigma, xv, out[i])
        elif node.op == OpKind.COL:
            src = values[node.inputs[0]]; out[:] = src[:, int(node.literals[0])]
        else:
            raise ValueError(f"Unsupported runtime node: {node.op}")


class BoundProgram:
    def __init__(self, program: CompiledProgram, inputs: dict[str, np.ndarray]):
        self.program = program
        self.inputs = self._validate(inputs)
        self.n_time = self.inputs[0].shape[0]

    def _validate(self, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
        arrays = []
        for name in self.program.input_names:
            if name not in inputs:
                raise ValueError(f"Missing input '{name}'")
            arr = np.asarray(inputs[name])
            expected_dtype = self.program.schema.inputs[name].dtype
            if arr.dtype != expected_dtype:
                arr = arr.astype(expected_dtype, copy=False)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D input for '{name}', got {arr.shape}")
            if arr.shape[1] != self.program.schema.n_instruments:
                raise ValueError(f"Input '{name}' width {arr.shape[1]} does not match schema.n_instruments")
            arrays.append(arr)
        if not arrays:
            raise ValueError("Program has no inputs")
        t = arrays[0].shape[0]
        for arr in arrays[1:]:
            if arr.shape[0] != t:
                raise ValueError("All bound inputs must share the same time dimension")
        return tuple(arrays)

    def run_batch(self, out: np.ndarray | None = None, out_path: str | None = f"{tempfile.gettempdir()}/trading_dsl_engine_out.memmap"):
        if not self.program.fast_path:
            data = {name: self.inputs[i] for i, name in enumerate(self.program.input_names)}
            return _compat_run_batch_from_mapping(self.program._fresh_fallback_engine(), data, out=out, out_path=out_path)
        tail = self.program.output_schema.shape.batch_tail
        shape = (self.n_time,) + tail
        if out is None:
            out = np.empty(shape, dtype=np.float64) if out_path is None else np.memmap(out_path, mode="w+", dtype=np.float64, shape=shape)
        elif out.shape != shape:
            raise ValueError(f"Output shape must be {shape}, got {out.shape}")
        state = self.program.new_state(); workspace = self.program.new_workspace()
        for t in range(self.n_time):
            y = self.program._execute_step(state, workspace, tuple(arr[t] for arr in self.inputs))
            if self.program.output_schema.kind == "scalar":
                out[t] = y[0]
            else:
                out[t, ...] = y
        return out


def _get1(x, i: int) -> float:
    if np.isscalar(x):
        return float(x)
    if x.ndim == 1:
        return float(x[i])
    return float(x[i, 0])


def _get_nd(x, idx) -> float:
    if np.isscalar(x):
        return float(x)
    return float(x[idx])


def _scalar_value(x) -> float:
    if np.isscalar(x):
        return float(x)
    return float(x[0] if x.ndim == 1 else x[0, 0])


def _copy_like(src, dst) -> None:
    if np.isscalar(src):
        dst[...] = float(src)
    elif dst.ndim == 1 and src.ndim == 2 and src.shape[1] == 1:
        dst[:] = src[:, 0]
    else:
        dst[...] = src


def _eval_elementwise(fn: str, args: list[Any], out: np.ndarray) -> None:
    for idx in np.ndindex(out.shape):
        vals = [_value_for_shape(a, idx) for a in args]
        out[idx] = _apply_scalar(fn, vals)


def _value_for_shape(a, idx) -> float:
    if np.isscalar(a):
        return float(a)
    if a.ndim == 1:
        return float(a[0] if len(idx) == 0 else a[idx[0] if a.shape[0] > 1 else 0])
    r = idx[0] if a.shape[0] > 1 else 0
    c = idx[1] if len(idx) > 1 and a.shape[1] > 1 else 0
    return float(a[r, c])


def _apply_scalar(fn: str, v: list[float]) -> float:
    a = v[0]
    if fn == "abs": return abs(a)
    if fn == "isnan": return 1.0 if np.isnan(a) else 0.0
    if fn == "ln": return np.log(a)
    if fn == "ceil": return np.ceil(a)
    if fn == "floor": return np.floor(a)
    if fn == "exp": return np.exp(a)
    if fn == "sign": return np.sign(a)
    if fn == "purify": return a if np.isfinite(a) else np.nan
    if fn == "arctan": return np.arctan(a)
    b = v[1]
    if fn == "add": return a + b
    if fn == "sub": return a - b
    if fn == "mul": return a * b
    if fn == "div": return np.nan if b == 0.0 else a / b
    if fn == "floordiv": return np.nan if b == 0.0 else a // b
    if fn == "mod": return a % b
    if fn == "pow": return a ** b
    if fn == "eq": return np.nan if np.isnan(a) or np.isnan(b) else (1.0 if a == b else 0.0)
    if fn == "ne": return np.nan if np.isnan(a) or np.isnan(b) else (1.0 if a != b else 0.0)
    if fn == "lt": return a < b
    if fn == "gt": return a > b
    if fn in ("and", "and_"): return np.nan if np.isnan(a) or np.isnan(b) else (1.0 if (a != 0.0 and b != 0.0) else 0.0)
    if fn in ("or", "or_"): return np.nan if np.isnan(a) or np.isnan(b) else (1.0 if (a != 0.0 or b != 0.0) else 0.0)
    if fn == "xor": return np.nan if np.isnan(a) or np.isnan(b) else (1.0 if ((a != 0.0) != (b != 0.0)) else 0.0)
    if fn == "fillna": return b if np.isnan(a) else a
    if fn == "where": return v[1] if a != 0.0 else v[2]
    raise ValueError(fn)


def _nanmean(x) -> float:
    total = 0.0; count = 0
    arr = np.asarray(x)
    for v in arr.ravel():
        if not np.isnan(v):
            total += float(v); count += 1
    return np.nan if count == 0 else total / count


def _xs_rank_inplace(x, out, scratch: dict[str, np.ndarray]) -> None:
    values = scratch["values"]; valid_index = scratch["valid_index"]; order = scratch["order"]
    m = 0
    for i in range(out.shape[0]):
        v = _get1(x, i); values[i] = v
        if np.isnan(v):
            out[i] = np.nan
        else:
            valid_index[m] = i; order[m] = m; m += 1
    for i in range(1, m):
        key = order[i]; key_val = values[valid_index[key]]; j = i - 1
        while j >= 0 and values[valid_index[order[j]]] > key_val:
            order[j + 1] = order[j]; j -= 1
        order[j + 1] = key
    pos = 0
    while pos < m:
        start = pos; v = values[valid_index[order[pos]]]; pos += 1
        while pos < m and values[valid_index[order[pos]]] == v:
            pos += 1
        rank = pos / m
        for k in range(start, pos):
            out[valid_index[order[k]]] = rank


def _quantile_inplace(buf: np.ndarray, n: int, q: float) -> float:
    m = 0
    for i in range(n):
        v = buf[i]
        if not np.isnan(v):
            buf[m] = v; m += 1
    if m == 0:
        return np.nan
    for i in range(1, m):
        key = buf[i]; j = i - 1
        while j >= 0 and buf[j] > key:
            buf[j + 1] = buf[j]; j -= 1
        buf[j + 1] = key
    if m == 1:
        return buf[0]
    pos = q * (m - 1.0); lo = int(np.floor(pos)); hi = int(np.ceil(pos))
    if lo == hi:
        return buf[lo]
    w = pos - lo
    return buf[lo] * (1.0 - w) + buf[hi] * w


def _basis_row(n_basis: int, sigma: float, x: float, out: np.ndarray) -> None:
    total = 0.0; inv_sigma2 = 1.0 / (sigma * sigma)
    for i in range(n_basis):
        center = i / n_basis; d = abs(x - center)
        if 1.0 - d < d: d = 1.0 - d
        val = np.exp(-0.5 * d * d * inv_sigma2)
        out[i] = val; total += val
    if total <= 1e-18 or np.isnan(total):
        out[:] = 1.0 / n_basis
    else:
        out[:] = out[:] / total



def _infer_shape_compat(node: Expr, schema: Schema, dsl_registry: DSLFunctionRegistry | None = None) -> ShapeInfo:
    dsl_registry = dsl_registry or DEFAULT_DSL_REGISTRY
    if isinstance(node, Identifier):
        if node.name not in schema.inputs:
            raise ValueError(f"Input '{node.name}' is not declared in schema")
        return ShapeInfo("vector", schema.n_instruments, 1)
    if isinstance(node, Number):
        return ShapeInfo("scalar", 1, 1)
    if isinstance(node, Universe):
        return ShapeInfo("object", 1, 1)
    if not isinstance(node, Call):
        raise ValueError(f"Unsupported expression for shape inference: {node}")
    py_fn = dsl_registry.get(node.fn)
    if py_fn is not None:
        return _infer_shape_compat(py_fn(*node.args), schema, dsl_registry)
    fn = node.fn
    if fn == "groupby" and len(node.args) == 2 and isinstance(node.args[0], Universe):
        op_shape = _infer_shape_compat(node.args[1], schema, dsl_registry)
        return ShapeInfo("vector", schema.n_instruments, 1) if op_shape.kind == "scalar" else op_shape
    if fn == "groupby":
        return _infer_shape_compat(node.args[-1], schema, dsl_registry)
    child_shapes = [_infer_shape_compat(a, schema, dsl_registry) for a in node.args]
    if fn in _ELEMENTWISE:
        kind = "scalar"; rows = 1; cols = 1
        for sh in child_shapes:
            if sh.kind == "matrix": kind = "matrix"
            elif sh.kind == "vector" and kind == "scalar": kind = "vector"
            rows = max(rows, sh.rows); cols = max(cols, sh.cols)
        return ShapeInfo(kind, rows, cols if kind == "matrix" else 1)
    if fn == "mean":
        return ShapeInfo("scalar", 1, 1)
    if fn in ("cumsum", "ewm", "shift", "xs_rank", "rolling_quantile", "get_preds"):
        return ShapeInfo("vector", schema.n_instruments, 1)
    if fn == "outer":
        return ShapeInfo("matrix", schema.n_instruments, schema.n_instruments)
    if fn == "bspline" and len(node.args) >= 2 and isinstance(node.args[1], Number):
        return ShapeInfo("matrix", schema.n_instruments, int(round(node.args[1].value)))
    if fn == "col":
        return ShapeInfo("vector", schema.n_instruments, 1)
    if fn == "Ridge":
        has_explicit_weights = len(child_shapes) >= 5
        features = child_shapes[:-4] if has_explicit_weights else child_shapes[:-3]
        width = 0
        for sh in features:
            width += sh.cols if sh.kind == "matrix" else 1
        return ShapeInfo("object", width, 1)
    if fn == "get_beta":
        ridge_shape = child_shapes[0]
        return ShapeInfo("vector", ridge_shape.rows, 1)
    raise ValueError(f"Cannot infer shape for compatibility operator '{fn}'")

def compile_program(
    formula: str | Expr,
    schema: Schema,
    dsl_registry: DSLFunctionRegistry | None = None,
    *,
    allow_fallback: bool = True,
) -> CompiledProgram:
    ast_expr = parse_formula(formula) if isinstance(formula, str) else formula
    try:
        builder = _FastPlanBuilder(schema, dsl_registry)
        root = builder.build(ast_expr)
        nodes = tuple(builder.nodes)
        output_shape = nodes[root].shape
        output_schema = OutputSchema(output_shape.kind, output_shape.cols if output_shape.kind == "matrix" else output_shape.rows, np.dtype("float64"), output_shape)
        plan = RuntimePlan(nodes, tuple(builder.input_names), output_schema, _plan_buffers(nodes), _fusion_regions(nodes), None)
        return CompiledProgram(formula, schema, plan, ast_expr)
    except Exception as exc:
        if not allow_fallback:
            raise
        warnings.warn(f"Falling back to compatibility runtime for schema-bound program: {exc}", RuntimeWarning, stacklevel=2)
        eng = _build_compat_engine(formula, dsl_registry=dsl_registry, column_names=schema.columns)
        try:
            shape = _infer_shape_compat(ast_expr, schema, dsl_registry)
        except Exception:
            kind = {0: "scalar", 1: "vector", 2: "matrix", 3: "object"}[eng.output_code]
            shape = ShapeInfo(kind, 1 if kind == "scalar" else schema.n_instruments, schema.n_instruments if kind == "matrix" else 1)
        width = shape.cols if shape.kind == "matrix" else shape.rows
        plan = RuntimePlan((), tuple(eng.input_names), OutputSchema(shape.kind, width, np.dtype("float64"), shape), BufferPlan({}, {}, {}, 0), (), str(exc))
        return CompiledProgram(formula, schema, plan, ast_expr, fallback_engine=eng)
