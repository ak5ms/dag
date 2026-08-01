from __future__ import annotations

from dataclasses import dataclass, replace
import math
import struct
from typing import Mapping, TypeAlias

from trading_dsl_engine.ir.ops import (
    CatOp,
    CumsumOp,
    EwmOp,
    GroupByOp,
    GroupKeySpec,
    InputOp,
    LiteralOp,
    NaryOp,
    RidgeOp,
    RidgeProjectionOp,
    XsRankOp,
)
from trading_dsl_engine.ir.program import Program
from trading_dsl_engine.ir.types import SCALAR, VECTOR, ValueType, fixed, matrix


class CppStreamLoweringError(ValueError):
    pass


_SUPPORTED_DTYPES = {
    "float32",
    "float64",
    "int32",
    "int64",
    "uint32",
    "uint64",
}
_INTEGRAL_DTYPES = {"int32", "int64", "uint32", "uint64"}
_DTYPE_LIMITS = {
    "int32": (-(1 << 31), (1 << 31) - 1),
    "int64": (-(1 << 63), (1 << 63) - 1),
    "uint32": (0, (1 << 32) - 1),
    "uint64": (0, (1 << 64) - 1),
}


@dataclass(frozen=True, slots=True)
class Source:
    kind: str
    value: int | float
    value_type: ValueType
    row_scalar: bool = False
    dtype: str = "float64"
    pinned: bool = False

    @property
    def is_file_or_scratch(self) -> bool:
        return self.kind != "literal"


@dataclass(frozen=True, slots=True)
class RidgeValueRef:
    beta: Source
    preds: Source


PhysicalValue: TypeAlias = Source | RidgeValueRef


@dataclass(frozen=True, slots=True)
class Dest:
    slot: int | None
    value_type: ValueType
    dtype: str = "float64"


@dataclass(frozen=True, slots=True)
class RidgeDest:
    beta: Dest
    preds: Dest


@dataclass(frozen=True, slots=True)
class Stage:
    kind: str
    inputs: tuple[Source, ...]
    out: Dest | RidgeDest
    lane_count: int
    dtype: str = "float64"
    value_type: ValueType = VECTOR
    op_name: str | None = None
    ewm: EwmOp | None = None
    ridge: RidgeOp | None = None
    group: "GroupStage | None" = None


@dataclass(frozen=True, slots=True)
class Plan:
    stages: tuple[Stage, ...]
    scratch_slots: int
    scratch_stride: int
    input_count: int
    output_type: ValueType


@dataclass(frozen=True, slots=True)
class GroupStage:
    inner: Plan
    partitions: tuple[int, ...]
    capacity: int
    hash_capacity: int
    key_sources: tuple[Source, ...]
    key_specs: tuple[GroupKeySpec, ...]
    feed_sources: tuple[Source, ...]
    dense: bool


_BINARY_CPP = {
    "add": "stackdsl::AddOp",
    "sub": "stackdsl::SubOp",
    "mul": "stackdsl::MulOp",
    "div": "stackdsl::DivOp",
    "mod": "stackdsl::ModOp",
}
_UNARY_CPP = {"floor": "stackdsl::FloorOp"}


def double_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def value_elements(value_type: ValueType, n_instruments: int) -> int:
    if value_type.kind == "scalar":
        return 1
    if value_type.kind == "vector":
        return n_instruments
    if value_type.kind == "matrix":
        return n_instruments * int(value_type.width)
    if value_type.kind == "fixed":
        return int(value_type.width)
    raise CppStreamLoweringError(
        f"object value {value_type} must be projected before physical storage"
    )


def _partitions(groups: tuple[tuple[int, ...], ...] | None, n: int) -> tuple[int, ...]:
    if groups is None:
        return (0,) * n
    result = [-1] * n
    for group_id, group in enumerate(groups):
        for lane in group:
            if lane < 0 or lane >= n:
                raise CppStreamLoweringError(
                    f"univ column {lane} is outside n_instruments={n}"
                )
            if result[lane] != -1:
                raise CppStreamLoweringError(
                    f"univ column {lane} appears in multiple groups"
                )
            result[lane] = group_id
    missing = [i for i, value in enumerate(result) if value < 0]
    if missing:
        raise CppStreamLoweringError(
            f"cpp_stream requires univ(...) to partition every column; missing {missing}"
        )
    return tuple(result)


def _resolved_key_specs(
    program: Program,
    node_child_ids: tuple[int, ...],
    op: GroupByOp,
    key_cardinalities: Mapping[str, int] | None,
) -> tuple[GroupKeySpec, ...]:
    specs = list(op.key_specs)
    if key_cardinalities:
        for index, spec in enumerate(specs):
            if spec.num_keys is not None:
                continue
            key_op = program.nodes[node_child_ids[index]].op
            if isinstance(key_op, InputOp) and key_op.name in key_cardinalities:
                cardinality = int(key_cardinalities[key_op.name])
                if cardinality <= 0:
                    raise CppStreamLoweringError("key cardinality must be > 0")
                specs[index] = replace(spec, num_keys=cardinality)
    return tuple(specs)


def _dense_capacity(specs: tuple[GroupKeySpec, ...]) -> int:
    capacity = 1
    for spec in specs:
        assert spec.num_keys is not None
        capacity *= int(spec.num_keys) + 1
    if capacity > 65535:
        raise CppStreamLoweringError(
            f"dense composite key requires {capacity} slots, exceeding uint16 capacity"
        )
    return capacity


def _validate_dtype(dtype: str) -> str:
    dtype = str(dtype).lower()
    if dtype not in _SUPPORTED_DTYPES:
        raise CppStreamLoweringError(
            f"unsupported cpp_stream dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    return dtype


def _literal_dtype(value: int | float) -> str:
    return "int64" if isinstance(value, int) and not isinstance(value, bool) else "float64"


def _coerce_literal_for_dtype(value: int | float, dtype: str) -> int | float:
    """Compile a literal at the asserted expression type, never an input value."""
    if dtype in _INTEGRAL_DTYPES:
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise CppStreamLoweringError(
                f"integral expression cannot contain non-integral literal {value!r}"
            )
        integer = int(numeric)
        lower, upper = _DTYPE_LIMITS[dtype]
        if integer < lower or integer > upper:
            raise CppStreamLoweringError(
                f"literal {integer} is outside the representable range of {dtype}"
            )
        return integer
    return float(value)


def _promote_dtypes(children: tuple[str, ...]) -> str:
    if not children:
        return "float64"
    if all(dtype == children[0] for dtype in children):
        return children[0]
    if all(dtype == "float32" for dtype in children):
        return "float32"
    return "float64"


def _normal_nary_dtype(op: NaryOp, children: tuple[str, ...]) -> str:
    if op.arity == 1:
        return children[0]
    left, right = children
    if op.name == "div":
        if left == right == "float32":
            return "float32"
        if left == right and left in _INTEGRAL_DTYPES:
            return left
        return "float64"
    return _promote_dtypes(children)


def _forced_key_dtypes(
    program: Program,
    input_dtypes: tuple[str, ...],
) -> dict[int, str]:
    forced: dict[int, str] = {}

    def force(node_id: int, dtype: str) -> None:
        dtype = _validate_dtype(dtype)
        previous = forced.get(node_id)
        if previous is not None:
            if previous != dtype:
                raise CppStreamLoweringError(
                    f"node {node_id} is shared by incompatible key dtypes {previous!r} and {dtype!r}"
                )
            return
        forced[node_id] = dtype
        node = program.nodes[node_id]
        op = node.op
        if isinstance(op, InputOp):
            actual = input_dtypes[op.input_index]
            if actual != dtype:
                raise CppStreamLoweringError(
                    f"Key dtype {dtype!r} does not match input {op.name!r} dtype {actual!r}; "
                    "cpp_stream will not insert an implicit input conversion"
                )
            return
        if isinstance(op, LiteralOp):
            _coerce_literal_for_dtype(op.value, dtype)
            return
        if isinstance(op, NaryOp):
            for child in node.child_ids:
                force(child, dtype)
            return
        raise CppStreamLoweringError(
            "Key dtype assertions apply only to pure input/literal/arithmetic graphs; "
            f"node {node_id} is {type(op).__name__}"
        )

    for node in program.nodes:
        op = node.op
        if not isinstance(op, GroupByOp):
            continue
        for index, spec in enumerate(op.key_specs):
            if spec.dtype is not None:
                force(node.child_ids[index], spec.dtype)
    return forced


def infer_node_dtypes(
    program: Program,
    input_dtypes: tuple[str, ...],
) -> tuple[str, ...]:
    if len(input_dtypes) != len(program.input_names):
        raise CppStreamLoweringError("input dtype count does not match program inputs")
    input_dtypes = tuple(_validate_dtype(dtype) for dtype in input_dtypes)
    forced = _forced_key_dtypes(program, input_dtypes)
    result: list[str] = []
    for node_id, node in enumerate(program.nodes):
        op = node.op
        if isinstance(op, InputOp):
            dtype = input_dtypes[op.input_index]
        elif isinstance(op, LiteralOp):
            dtype = forced.get(node_id, _literal_dtype(op.value))
        elif isinstance(op, NaryOp):
            dtype = forced.get(
                node_id,
                _normal_nary_dtype(op, tuple(result[child] for child in node.child_ids)),
            )
        elif isinstance(op, CatOp):
            dtype = _promote_dtypes(tuple(result[child] for child in node.child_ids))
        elif isinstance(
            op,
            (CumsumOp, EwmOp, XsRankOp, RidgeOp, RidgeProjectionOp, GroupByOp),
        ):
            dtype = "float64"
        else:
            raise CppStreamLoweringError(f"unsupported IR op {type(op).__name__}")
        result.append(dtype)
    return tuple(result)


def _physical_type(
    semantic: ValueType,
    *,
    row_scalar: bool,
    is_root: bool,
    dtype: str,
) -> ValueType:
    if semantic.kind == "vector" and row_scalar and not is_root:
        return ValueType("scalar", 1, dtype)
    return ValueType(semantic.kind, semantic.width, dtype)


def _row_scalar_ids_from_inputs(
    program: Program,
    input_row_scalar: tuple[bool, ...],
) -> frozenset[int]:
    result: list[bool] = []
    for node in program.nodes:
        op = node.op
        if isinstance(op, InputOp):
            scalar = input_row_scalar[op.input_index]
        elif isinstance(op, LiteralOp):
            scalar = True
        elif isinstance(op, NaryOp):
            scalar = all(result[child] for child in node.child_ids)
        else:
            scalar = False
        result.append(scalar)
    return frozenset(index for index, scalar in enumerate(result) if scalar)


def _as_sources(values: tuple[PhysicalValue, ...], owner: str) -> tuple[Source, ...]:
    if any(not isinstance(value, Source) for value in values):
        raise CppStreamLoweringError(f"{owner} cannot consume an unprojected object value")
    return tuple(value for value in values if isinstance(value, Source))


def _build_plan(
    program: Program,
    *,
    n_instruments: int,
    default_group_capacity: int,
    key_cardinalities: Mapping[str, int] | None,
    grouped: bool,
    row_scalar_nodes: frozenset[int],
    input_dtypes: tuple[str, ...],
    node_dtypes: tuple[str, ...],
) -> Plan:
    uses = [0] * len(program.nodes)
    root = program.output_id
    for node in program.nodes:
        for child in node.child_ids:
            uses[child] += 1
    uses[root] += 1

    sources: dict[int, PhysicalValue] = {}
    free_slots: list[int] = []
    next_slot = 0
    scratch_stride = 1
    stages: list[Stage] = []

    def allocate(value_type: ValueType) -> int:
        nonlocal next_slot, scratch_stride
        scratch_stride = max(scratch_stride, value_elements(value_type, n_instruments))
        if free_slots:
            return free_slots.pop()
        slot = next_slot
        next_slot += 1
        return slot

    def release(node_id: int, value: PhysicalValue, protected: set[int] | None = None) -> None:
        uses[node_id] -= 1
        if uses[node_id] != 0 or not isinstance(value, Source):
            return
        if value.kind == "slot" and not value.pinned:
            slot = int(value.value)
            if protected is None or slot not in protected:
                free_slots.append(slot)

    for node_id, node in enumerate(program.nodes):
        op = node.op
        dtype = node_dtypes[node_id]
        row_scalar = node_id in row_scalar_nodes
        is_root = node_id == root
        semantic_type = node.value_type
        value_type = _physical_type(
            semantic_type,
            row_scalar=row_scalar,
            is_root=is_root,
            dtype=dtype,
        )
        lane_count = 1 if value_type.kind == "scalar" else n_instruments

        if isinstance(op, InputOp):
            sources[node_id] = Source(
                "input",
                op.input_index,
                value_type=value_type,
                row_scalar=row_scalar,
                dtype=input_dtypes[op.input_index],
            )
            continue
        if isinstance(op, LiteralOp):
            sources[node_id] = Source(
                "literal",
                _coerce_literal_for_dtype(op.value, dtype),
                value_type=ValueType("scalar", 1, dtype),
                row_scalar=True,
                dtype=dtype,
            )
            continue

        child_values = tuple(sources[child] for child in node.child_ids)

        if isinstance(op, RidgeProjectionOp):
            if len(child_values) != 1 or not isinstance(child_values[0], RidgeValueRef):
                raise CppStreamLoweringError("Ridge projection requires a Ridge value")
            ridge_value = child_values[0]
            sources[node_id] = ridge_value.beta if op.field == "beta" else ridge_value.preds
            release(node.child_ids[0], ridge_value)
            continue

        children = _as_sources(child_values, type(op).__name__)

        if isinstance(op, RidgeOp):
            beta_type = matrix(op.coefficient_width) if grouped else fixed(op.coefficient_width)
            beta_slot = allocate(beta_type)
            preds_slot = allocate(VECTOR)
            beta_dest = Dest(beta_slot, beta_type, "float64")
            preds_dest = Dest(preds_slot, VECTOR, "float64")
            stage = Stage(
                "ridge",
                children,
                RidgeDest(beta_dest, preds_dest),
                n_instruments,
                dtype="float64",
                value_type=semantic_type,
                ridge=op,
            )
            stages.append(stage)
            sources[node_id] = RidgeValueRef(
                beta=Source(
                    "slot",
                    beta_slot,
                    beta_type,
                    row_scalar=False,
                    dtype="float64",
                    pinned=True,
                ),
                preds=Source(
                    "slot",
                    preds_slot,
                    VECTOR,
                    row_scalar=False,
                    dtype="float64",
                    pinned=True,
                ),
            )
            for child_id, source in zip(node.child_ids, children):
                release(child_id, source, {beta_slot, preds_slot})
            continue

        reusable: int | None = None
        if not is_root and not isinstance(op, (CatOp, GroupByOp)):
            for child_id, source in zip(node.child_ids, children):
                if (
                    source.kind == "slot"
                    and not source.pinned
                    and uses[child_id] == 1
                    and value_elements(source.value_type, n_instruments)
                    >= value_elements(value_type, n_instruments)
                ):
                    reusable = int(source.value)
                    break
        out = Dest(
            None if is_root else (reusable if reusable is not None else allocate(value_type)),
            value_type,
            dtype,
        )

        if isinstance(op, NaryOp):
            if value_type.kind not in {"scalar", "vector"}:
                raise CppStreamLoweringError(
                    "cpp_stream matrix arithmetic is not implemented yet; cat and Ridge consume matrices directly"
                )
            if op.arity == 2 and op.name in _BINARY_CPP:
                stage = Stage(
                    "binary", children, out, lane_count, dtype=dtype,
                    value_type=value_type, op_name=op.name
                )
            elif op.arity == 1 and op.name in _UNARY_CPP:
                stage = Stage(
                    "unary", children, out, lane_count, dtype=dtype,
                    value_type=value_type, op_name=op.name
                )
            else:
                raise CppStreamLoweringError(f"unsupported nary op {op.name!r}/{op.arity}")
        elif isinstance(op, CatOp):
            stage = Stage(
                "cat", children, out, n_instruments, dtype=dtype,
                value_type=value_type, op_name="cat"
            )
        elif isinstance(op, CumsumOp):
            stage = Stage(
                "cumsum", children, out, lane_count, dtype="float64", value_type=VECTOR
            )
        elif isinstance(op, EwmOp):
            stage = Stage(
                "ewm", children, out, lane_count, dtype="float64",
                value_type=VECTOR, ewm=op
            )
        elif isinstance(op, XsRankOp):
            stage = Stage(
                "xs_rank", children, out, lane_count, dtype="float64", value_type=VECTOR
            )
        elif isinstance(op, GroupByOp):
            if grouped:
                raise CppStreamLoweringError("nested groupby is not supported")
            key_count = op.n_dynamic_keys
            key_sources = children[:key_count]
            feed_sources = children[key_count:]
            if any(source.value_type.kind not in {"scalar", "vector"} for source in key_sources):
                raise CppStreamLoweringError("group keys must be scalar or vector-valued")
            key_specs = _resolved_key_specs(program, node.child_ids, op, key_cardinalities)
            for key_index, (source, spec) in enumerate(zip(key_sources, key_specs)):
                if spec.dtype is not None and source.dtype != spec.dtype:
                    raise CppStreamLoweringError(
                        f"group key {key_index} inferred dtype {source.dtype!r}, "
                        f"but Key.dtype asserted {spec.dtype!r}"
                    )
            dense = bool(key_specs) and all(spec.num_keys is not None for spec in key_specs)
            capacity = _dense_capacity(key_specs) if dense else (
                op.capacity or default_group_capacity
            )
            inner_input_dtypes = tuple(source.dtype for source in feed_sources)
            inner_row_scalar_inputs = tuple(
                source.value_type.kind == "scalar" for source in feed_sources
            )
            inner_row_scalar_nodes = _row_scalar_ids_from_inputs(
                op.inner_program,
                inner_row_scalar_inputs,
            )
            inner = _build_plan(
                op.inner_program,
                n_instruments=n_instruments,
                default_group_capacity=default_group_capacity,
                key_cardinalities=key_cardinalities,
                grouped=True,
                row_scalar_nodes=inner_row_scalar_nodes,
                input_dtypes=inner_input_dtypes,
                node_dtypes=infer_node_dtypes(op.inner_program, inner_input_dtypes),
            )
            group_stage = GroupStage(
                inner=inner,
                partitions=_partitions(op.static_groups, n_instruments),
                capacity=capacity,
                hash_capacity=op.hash_capacity or 0,
                key_sources=key_sources,
                key_specs=key_specs,
                feed_sources=feed_sources,
                dense=dense,
            )
            stage = Stage(
                "groupby", children, out, n_instruments, dtype="float64",
                value_type=value_type, group=group_stage
            )
        else:
            raise CppStreamLoweringError(f"unsupported IR op {type(op).__name__}")

        stages.append(stage)
        sources[node_id] = Source(
            "slot",
            out.slot if out.slot is not None else -1,
            value_type=value_type,
            row_scalar=value_type.kind == "scalar",
            dtype=stage.dtype,
        )
        protected = set() if out.slot is None else {out.slot}
        for child_id, source in zip(node.child_ids, children):
            release(child_id, source, protected)

    root_value = sources[root]
    if isinstance(root_value, RidgeValueRef):
        raise CppStreamLoweringError(
            "Ridge object roots require get_beta(...) or get_preds(...)"
        )
    if not stages or (
        isinstance(program.nodes[root].op, (InputOp, LiteralOp, RidgeProjectionOp))
    ):
        stages.append(
            Stage(
                "copy",
                (root_value,),
                Dest(None, program.nodes[root].value_type, root_value.dtype),
                1 if root_value.value_type.kind == "scalar" else n_instruments,
                dtype=root_value.dtype,
                value_type=program.nodes[root].value_type,
            )
        )

    output_type = program.nodes[root].value_type
    if output_type.kind == "object":
        raise CppStreamLoweringError("object roots must be projected before output")
    return Plan(
        stages=tuple(stages),
        scratch_slots=next_slot,
        scratch_stride=max(scratch_stride, 1),
        input_count=len(program.input_names),
        output_type=output_type,
    )


def lower_program(
    program: Program,
    *,
    n_instruments: int,
    default_group_capacity: int = 64,
    key_cardinalities: Mapping[str, int] | None = None,
    row_scalar_nodes: frozenset[int] | None = None,
    input_dtypes: tuple[str, ...] | None = None,
) -> Plan:
    if n_instruments <= 0:
        raise CppStreamLoweringError("n_instruments must be > 0")
    if default_group_capacity <= 0:
        raise CppStreamLoweringError("default_group_capacity must be > 0")
    if input_dtypes is None:
        input_dtypes = ("float64",) * len(program.input_names)
    node_dtypes = infer_node_dtypes(program, input_dtypes)
    return _build_plan(
        program,
        n_instruments=n_instruments,
        default_group_capacity=default_group_capacity,
        key_cardinalities=key_cardinalities,
        grouped=False,
        row_scalar_nodes=row_scalar_nodes or frozenset(),
        input_dtypes=input_dtypes,
        node_dtypes=node_dtypes,
    )


__all__ = [
    "CppStreamLoweringError",
    "Source",
    "RidgeValueRef",
    "Dest",
    "RidgeDest",
    "Stage",
    "Plan",
    "GroupStage",
    "double_bits",
    "value_elements",
    "infer_node_dtypes",
    "lower_program",
]
