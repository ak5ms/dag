from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping
import math
import struct

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
from trading_dsl_engine.ir.types import ValueType


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
    value: int | float | None = None
    row_scalar: bool = False
    dtype: str = "float64"
    width: int = 1
    parts: tuple["Source", ...] = ()
    ridge: RidgeOp | None = None

    @property
    def is_vector(self) -> bool:
        return self.kind not in {"literal", "cat", "ridge"}


@dataclass(frozen=True, slots=True)
class Dest:
    slot: int | None


@dataclass(frozen=True, slots=True)
class Stage:
    kind: str
    inputs: tuple[Source, ...]
    out: Dest
    lane_count: int
    dtype: str = "float64"
    output_kind: str = "vector"
    output_width: int = 1
    op_name: str | None = None
    ewm: EwmOp | None = None
    ridge: RidgeOp | None = None
    projection: str | None = None
    half_life: float | None = None
    ridge_lambda: float | None = None
    group: "GroupStage | None" = None


@dataclass(frozen=True, slots=True)
class Plan:
    stages: tuple[Stage, ...]
    scratch_slots: int
    input_count: int
    output_kind: str
    output_width: int
    output_row_width: int


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


def _partitions(groups: tuple[tuple[int, ...], ...] | None, n: int) -> tuple[int, ...]:
    if groups is None:
        return (0,) * n
    result = [-1] * n
    for group_id, group in enumerate(groups):
        for lane in group:
            if lane < 0 or lane >= n:
                raise CppStreamLoweringError(f"univ column {lane} is outside n_instruments={n}")
            if result[lane] != -1:
                raise CppStreamLoweringError(f"univ column {lane} appears in multiple groups")
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
                f"integral key expression cannot contain non-integral literal {value!r}"
            )
        integer = int(numeric)
        lower, upper = _DTYPE_LIMITS[dtype]
        if integer < lower or integer > upper:
            raise CppStreamLoweringError(
                f"literal {integer} is outside the representable range of {dtype}"
            )
        return integer
    return float(value)


def _normal_nary_dtype(op: NaryOp, children: tuple[str, ...]) -> str:
    if op.arity == 1:
        return children[0]
    left, right = children
    if op.name == "div":
        if left == right == "float32":
            return "float32"
        return "float64"
    if left == right:
        return left
    return "float64"


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
            "Key dtype assertions currently apply only to pure "
            f"input/literal/arithmetic graphs; node {node_id} is {type(op).__name__}"
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
            child_types = {result[child] for child in node.child_ids}
            dtype = next(iter(child_types)) if len(child_types) == 1 else "float64"
        elif isinstance(op, (CumsumOp, EwmOp, XsRankOp, RidgeOp, RidgeProjectionOp, GroupByOp)):
            dtype = "float64"
        else:
            raise CppStreamLoweringError(f"unsupported IR op {type(op).__name__}")
        result.append(dtype)
    return tuple(result)


def _output_row_width(value_type: ValueType, n_instruments: int) -> int:
    if value_type.kind == "scalar":
        return 1
    if value_type.kind == "vector":
        return n_instruments
    if value_type.kind == "matrix":
        return n_instruments * int(value_type.width)
    if value_type.kind == "fixed":
        return int(value_type.width)
    raise CppStreamLoweringError(
        f"cpp_stream output must be projected before materialization; got {value_type.kind!r}"
    )


def _flatten_features(source: Source) -> tuple[Source, ...]:
    if source.kind != "cat":
        if source.width != 1:
            raise CppStreamLoweringError(
                f"cannot flatten feature source kind={source.kind!r}, width={source.width}"
            )
        return (source,)
    flattened: list[Source] = []
    for part in source.parts:
        flattened.extend(_flatten_features(part))
    return tuple(flattened)


def _literal_scalar(source: Source, name: str) -> float:
    if source.kind != "literal":
        raise CppStreamLoweringError(
            f"cpp_stream requires compile-time literal {name} for formula specialization"
        )
    return float(source.value)


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

    composite_plan = any(isinstance(node.op, (CatOp, RidgeOp)) for node in program.nodes)
    sources: dict[int, Source] = {}
    free_slots: list[int] = []
    next_slot = 0
    stages: list[Stage] = []

    def allocate() -> int:
        nonlocal next_slot
        if free_slots and not composite_plan:
            return free_slots.pop()
        slot = next_slot
        next_slot += 1
        return slot

    for node_id, node in enumerate(program.nodes):
        op = node.op
        dtype = node_dtypes[node_id]
        row_scalar = node_id in row_scalar_nodes
        lane_count = 1 if row_scalar else n_instruments
        if isinstance(op, InputOp):
            sources[node_id] = Source(
                "input",
                op.input_index,
                row_scalar=row_scalar,
                dtype=input_dtypes[op.input_index],
            )
            continue
        if isinstance(op, LiteralOp):
            sources[node_id] = Source(
                "literal",
                _coerce_literal_for_dtype(op.value, dtype),
                row_scalar=True,
                dtype=dtype,
            )
            continue

        children = tuple(sources[child] for child in node.child_ids)
        is_root = node_id == root

        if isinstance(op, CatOp):
            parts: list[Source] = []
            for child in children:
                parts.extend(_flatten_features(child))
            source = Source(
                "cat",
                row_scalar=False,
                dtype=dtype,
                width=len(parts),
                parts=tuple(parts),
            )
            sources[node_id] = source
            if is_root:
                stages.append(
                    Stage(
                        "cat",
                        tuple(parts),
                        Dest(None),
                        n_instruments,
                        dtype="float64",
                        output_kind="matrix",
                        output_width=len(parts),
                    )
                )
            continue

        if isinstance(op, RidgeOp):
            sources[node_id] = Source(
                "ridge",
                row_scalar=False,
                dtype="float64",
                width=op.coefficient_width,
                parts=children,
                ridge=op,
            )
            if is_root:
                raise CppStreamLoweringError(
                    "Ridge is an object value; materialize get_beta(Ridge(...)) or get_preds(Ridge(...))"
                )
            continue

        reusable: int | None = None
        if not is_root and not isinstance(op, GroupByOp) and not composite_plan:
            for child_id, source in zip(node.child_ids, children):
                if (
                    source.kind == "slot"
                    and uses[child_id] == 1
                    and (row_scalar or not source.row_scalar)
                ):
                    reusable = int(source.value)
                    break
        out = Dest(None if is_root else (reusable if reusable is not None else allocate()))

        if isinstance(op, NaryOp):
            if any(source.width != 1 for source in children):
                raise CppStreamLoweringError(
                    "matrix arithmetic is not materialized by cpp_stream yet; consume cat directly in Ridge"
                )
            if op.arity == 2 and op.name in _BINARY_CPP:
                stage = Stage("binary", children, out, lane_count, dtype=dtype, op_name=op.name)
            elif op.arity == 1 and op.name in _UNARY_CPP:
                stage = Stage("unary", children, out, lane_count, dtype=dtype, op_name=op.name)
            else:
                raise CppStreamLoweringError(f"unsupported nary op {op.name!r}/{op.arity}")
        elif isinstance(op, CumsumOp):
            if not children[0].is_vector:
                raise CppStreamLoweringError("cpp_stream cumsum requires vector input")
            stage = Stage("cumsum", children, out, lane_count, dtype="float64")
        elif isinstance(op, EwmOp):
            if not children[0].is_vector:
                raise CppStreamLoweringError("cpp_stream ewm requires vector input")
            stage = Stage("ewm", children, out, lane_count, dtype="float64", ewm=op)
        elif isinstance(op, XsRankOp):
            if not children[0].is_vector:
                raise CppStreamLoweringError("cpp_stream xs_rank requires vector input")
            stage = Stage("xs_rank", children, out, lane_count, dtype="float64")
        elif isinstance(op, RidgeProjectionOp):
            ridge_source = children[0]
            if ridge_source.kind != "ridge" or ridge_source.ridge is None:
                raise CppStreamLoweringError("Ridge projection lost its Ridge source")
            ridge_op = ridge_source.ridge
            ridge_children = ridge_source.parts
            feature_count = len(ridge_op.feature_widths)
            feature_sources: list[Source] = []
            for feature in ridge_children[:feature_count]:
                feature_sources.extend(_flatten_features(feature))
            if len(feature_sources) != ridge_op.coefficient_width:
                raise CppStreamLoweringError("Ridge feature width/source mismatch")
            y_source = ridge_children[feature_count]
            cursor = feature_count + 1
            if ridge_op.has_weights:
                weight_source = ridge_children[cursor]
                cursor += 1
            else:
                weight_source = Source("literal", 1.0, row_scalar=True, dtype="float64")
            half_life = _literal_scalar(ridge_children[cursor], "Ridge hl")
            ridge_lambda = _literal_scalar(ridge_children[cursor + 1], "Ridge lambda")
            if y_source.width != 1 or weight_source.width != 1:
                raise CppStreamLoweringError(
                    "cpp_stream Ridge currently supports scalar/vector y and scalar/vector weights"
                )
            if op.field == "beta" and not is_root:
                raise CppStreamLoweringError(
                    "Ridge beta is fixed/matrix-valued and currently must be a program/groupby root"
                )
            stage_inputs = tuple(feature_sources) + (y_source, weight_source)
            stage = Stage(
                "ridge",
                stage_inputs,
                out,
                n_instruments,
                dtype="float64",
                output_kind=node.value_type.kind,
                output_width=int(node.value_type.width),
                ridge=ridge_op,
                projection=op.field,
                half_life=half_life,
                ridge_lambda=ridge_lambda,
            )
        elif isinstance(op, GroupByOp):
            if grouped:
                raise CppStreamLoweringError("nested groupby is not supported")
            key_count = op.n_dynamic_keys
            key_sources = children[:key_count]
            feed_sources = children[key_count:]
            if any(source.width != 1 or not source.is_vector for source in key_sources + feed_sources):
                raise CppStreamLoweringError(
                    "cpp_stream groupby keys/lhs/captures must currently be scalar-width vectors"
                )
            key_specs = _resolved_key_specs(program, node.child_ids, op, key_cardinalities)
            for key_index, (source, spec) in enumerate(zip(key_sources, key_specs)):
                if spec.dtype is not None and source.dtype != spec.dtype:
                    raise CppStreamLoweringError(
                        f"group key {key_index} inferred dtype {source.dtype!r}, "
                        f"but Key.dtype asserted {spec.dtype!r}"
                    )
            non_double_feeds = [source.dtype for source in feed_sources if source.dtype != "float64"]
            if non_double_feeds:
                raise CppStreamLoweringError(
                    "groupby lhs/captures currently feed double-valued grouped operators; "
                    f"got native feed dtypes {non_double_feeds}. Key expressions remain typed."
                )
            dense = bool(key_specs) and all(spec.num_keys is not None for spec in key_specs)
            capacity = _dense_capacity(key_specs) if dense else (op.capacity or default_group_capacity)
            inner_input_dtypes = ("float64",) * len(op.inner_program.input_names)
            inner = _build_plan(
                op.inner_program,
                n_instruments=n_instruments,
                default_group_capacity=default_group_capacity,
                key_cardinalities=key_cardinalities,
                grouped=True,
                row_scalar_nodes=frozenset(),
                input_dtypes=inner_input_dtypes,
                node_dtypes=infer_node_dtypes(op.inner_program, inner_input_dtypes),
            )
            if node.value_type.kind in {"matrix", "fixed"} and not is_root:
                raise CppStreamLoweringError(
                    "matrix/fixed groupby results currently must be the program root"
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
                "groupby",
                children,
                out,
                n_instruments,
                dtype="float64",
                output_kind=node.value_type.kind,
                output_width=int(node.value_type.width),
                group=group_stage,
            )
        else:
            raise CppStreamLoweringError(f"unsupported IR op {type(op).__name__}")

        stages.append(stage)
        if not is_root:
            sources[node_id] = Source(
                "slot",
                out.slot,
                row_scalar=row_scalar,
                dtype=stage.dtype,
                width=1,
            )
        else:
            sources[node_id] = Source(
                "output",
                -1,
                row_scalar=False,
                dtype=stage.dtype,
                width=stage.output_width,
            )
        if not composite_plan:
            for child_id, source in zip(node.child_ids, children):
                uses[child_id] -= 1
                if uses[child_id] == 0 and source.kind == "slot":
                    slot = int(source.value)
                    if slot != out.slot:
                        free_slots.append(slot)

    root_type = program.nodes[root].value_type
    if isinstance(program.nodes[root].op, (InputOp, LiteralOp)):
        source = sources[root]
        stages.append(Stage("copy", (source,), Dest(None), n_instruments, dtype=source.dtype))
    return Plan(
        stages=tuple(stages),
        scratch_slots=next_slot,
        input_count=len(program.input_names),
        output_kind=root_type.kind,
        output_width=int(root_type.width),
        output_row_width=_output_row_width(root_type, n_instruments),
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
