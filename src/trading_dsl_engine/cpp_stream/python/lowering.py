from __future__ import annotations

from dataclasses import dataclass, replace
import math
import struct
from typing import Mapping

from trading_dsl_engine.ir.einsum import ContractionStep, build_contraction_plan
from trading_dsl_engine.ir.ops import (
    CatOp,
    ColumnOp,
    CumsumOp,
    CustomCallOp,
    EmitOp,
    EinsumOp,
    EwmOp,
    FFillOp,
    FutureRbfBasisSumOp,
    HumpOp,
    GroupByOp,
    GroupKeySpec,
    InputOp,
    InstrumentBasisMeanOp,
    InstrumentBasisProjectionOp,
    LiteralOp,
    LinearFilterOp,
    NaryOp,
    RbfBasisOp,
    PeriodsSinceChangeOp,
    RidgeOp,
    RidgeProjectionOp,
    RollingDecayOp,
    RollingEntropyOp,
    RollingKthOp,
    RollingOp,
    RollingPrevDiffOp,
    RollingProductOp,
    ReductionOp,
    ShiftOp,
    TheilSenOp,
    TradeWhenOp,
    VectorQuantileOp,
    XsPctRankOp,
    XsAggregateOp,
    XsWeightedMeanOp,
    XsProjectionOp,
    XsGeneralizedRankOp,
    XsDensifyOp,
    XsRankOp,
)
from trading_dsl_engine.ir.program import Program
from trading_dsl_engine.ir.types import ValueType, resolve_shape, shape_size


class CppStreamLoweringError(ValueError):
    pass


_SUPPORTED_DTYPES = {"float32", "float64", "int32", "int64", "uint32", "uint64"}
_INTEGRAL_DTYPES = {"int32", "int64", "uint32", "uint64"}
_DTYPE_LIMITS = {
    "int32": (-(1 << 31), (1 << 31) - 1),
    "int64": (-(1 << 63), (1 << 63) - 1),
    "uint32": (0, (1 << 32) - 1),
    "uint64": (0, (1 << 64) - 1),
}
_LOGICAL_NARY = {"eq", "ne", "lt", "gt", "le", "ge", "and_", "or_", "xor"}


@dataclass(frozen=True, slots=True)
class Source:
    kind: str
    value: int | float | str | None = None
    row_scalar: bool = False
    dtype: str = "float64"
    width: int = 1
    shape: tuple[int, ...] = ()
    parts: tuple["Source", ...] = ()
    op: object | None = None
    final_only: bool = False

    @property
    def is_scalar_width(self) -> bool:
        return self.width == 1 and self.kind not in {
            "cat",
            "rbf",
            "future_rbf",
            "matrix_slot",
            "tensor_slot",
        }


@dataclass(frozen=True, slots=True)
class Dest:
    slot: int | None
    matrix: bool = False
    tensor: bool = False
    width: int = 1
    size: int = 1
    shape: tuple[int, ...] = ()


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
    op: object | None = None
    projection: str | None = None
    projection_component: int | None = None
    half_life: float | None = None
    ridge_lambda: float | None = None
    group: "GroupStage | None" = None
    einsum_step: ContractionStep | None = None
    final_only: bool = False
    members: tuple["Stage", ...] = ()
    epilogues: tuple["Stage", ...] = ()


@dataclass(frozen=True, slots=True)
class Plan:
    stages: tuple[Stage, ...]
    scratch_slots: int
    matrix_scratch_slots: int
    matrix_scratch_width: int
    input_count: int
    output_kind: str
    output_width: int
    output_row_width: int
    output_shape: tuple[int, ...]
    output_mode: str


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


def double_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _validate_dtype(dtype: str) -> str:
    dtype = str(dtype).lower()
    if dtype not in _SUPPORTED_DTYPES:
        raise CppStreamLoweringError(f"unsupported cpp_stream dtype {dtype!r}")
    return dtype


def _literal_dtype(value: int | float) -> str:
    return "int64" if isinstance(value, int) and not isinstance(value, bool) else "float64"


def _coerce_literal_for_dtype(value: int | float, dtype: str) -> int | float:
    if dtype in _INTEGRAL_DTYPES:
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise CppStreamLoweringError(
                f"non-integral literal {value!r} in {dtype} expression"
            )
        integer = int(numeric)
        lower, upper = _DTYPE_LIMITS[dtype]
        if not lower <= integer <= upper:
            raise CppStreamLoweringError(f"literal {integer} outside {dtype} range")
        return integer
    return float(value)


def _normal_nary_dtype(op: NaryOp, children: tuple[str, ...]) -> str:
    if op.name in _LOGICAL_NARY or op.name in {"where", "fillna", "pow"}:
        return "float64"
    if op.arity == 1:
        return children[0]
    if op.name == "div":
        return "float32" if children == ("float32", "float32") else "float64"
    return children[0] if len(set(children)) == 1 else "float64"


def _forced_key_dtypes(program: Program, input_dtypes: tuple[str, ...]) -> dict[int, str]:
    forced: dict[int, str] = {}

    def force(node_id: int, dtype: str) -> None:
        dtype = _validate_dtype(dtype)
        previous = forced.get(node_id)
        if previous is not None:
            if previous != dtype:
                raise CppStreamLoweringError(
                    f"node {node_id} shared by incompatible key dtypes "
                    f"{previous!r}/{dtype!r}"
                )
            return
        forced[node_id] = dtype
        op = program.nodes[node_id].op
        if isinstance(op, InputOp):
            actual = input_dtypes[op.input_index]
            if actual != dtype:
                raise CppStreamLoweringError(
                    f"Key dtype {dtype!r} does not match input {op.name!r} "
                    f"dtype {actual!r}"
                )
        elif isinstance(op, LiteralOp):
            _coerce_literal_for_dtype(op.value, dtype)
        elif isinstance(op, NaryOp):
            for child in program.nodes[node_id].child_ids:
                force(child, dtype)
        else:
            raise CppStreamLoweringError(
                "Key dtype assertions require input/literal/arithmetic graphs, "
                f"got {type(op).__name__}"
            )

    for node in program.nodes:
        if isinstance(node.op, GroupByOp):
            for index, spec in enumerate(node.op.key_specs):
                if spec.dtype is not None:
                    force(node.child_ids[index], spec.dtype)
    return forced


def infer_node_dtypes(program: Program, input_dtypes: tuple[str, ...]) -> tuple[str, ...]:
    if len(input_dtypes) != len(program.input_names):
        raise CppStreamLoweringError("input dtype count does not match program inputs")
    input_dtypes = tuple(_validate_dtype(dtype) for dtype in input_dtypes)
    forced = _forced_key_dtypes(program, input_dtypes)
    result: list[str] = []
    float_ops = (
        CustomCallOp,
        CatOp,
        ColumnOp,
        CumsumOp,
        ReductionOp,
        EmitOp,
        FFillOp,
        ShiftOp,
        EwmOp,
        PeriodsSinceChangeOp,
        HumpOp,
        TradeWhenOp,
        LinearFilterOp,
        RollingProductOp,
        RollingKthOp,
        RollingPrevDiffOp,
        RollingDecayOp,
        RollingEntropyOp,
        RollingOp,
        TheilSenOp,
        VectorQuantileOp,
        XsRankOp,
        XsPctRankOp,
        XsAggregateOp,
        XsWeightedMeanOp,
        XsProjectionOp,
        XsGeneralizedRankOp,
        XsDensifyOp,
        RbfBasisOp,
        FutureRbfBasisSumOp,
        EinsumOp,
        InstrumentBasisMeanOp,
        InstrumentBasisProjectionOp,
        RidgeOp,
        RidgeProjectionOp,
        GroupByOp,
    )
    for node_id, node in enumerate(program.nodes):
        op = node.op
        if isinstance(op, InputOp):
            dtype = input_dtypes[op.input_index]
        elif isinstance(op, LiteralOp):
            dtype = forced.get(node_id, _literal_dtype(op.value))
        elif isinstance(op, NaryOp):
            dtype = forced.get(
                node_id,
                _normal_nary_dtype(
                    op, tuple(result[child] for child in node.child_ids)
                ),
            )
        elif isinstance(op, float_ops):
            dtype = "float64"
        else:
            raise CppStreamLoweringError(f"unsupported IR op {type(op).__name__}")
        result.append(dtype)
    return tuple(result)


def _output_row_width(value_type: ValueType, n: int) -> int:
    try:
        return shape_size(resolve_shape(value_type, n))
    except ValueError as exc:
        raise CppStreamLoweringError(
            f"cannot materialize output kind {value_type.kind!r}"
        ) from exc


def _partitions(
    groups: tuple[tuple[int, ...], ...] | None, n: int
) -> tuple[int, ...]:
    if groups is None:
        return (0,) * n
    result = [-1] * n
    for group_id, group in enumerate(groups):
        for lane in group:
            if lane < 0 or lane >= n or result[lane] != -1:
                raise CppStreamLoweringError(f"invalid universe lane {lane}")
            result[lane] = group_id
    missing = [lane for lane, value in enumerate(result) if value < 0]
    if missing:
        raise CppStreamLoweringError(f"univ does not partition lanes {missing}")
    return tuple(result)


def _resolved_key_specs(
    program: Program,
    child_ids: tuple[int, ...],
    op: GroupByOp,
    key_cardinalities: Mapping[str, int] | None,
) -> tuple[GroupKeySpec, ...]:
    specs = list(op.key_specs)
    if key_cardinalities:
        for index, spec in enumerate(specs):
            if spec.num_keys is not None:
                continue
            key_op = program.nodes[child_ids[index]].op
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
            f"dense key capacity {capacity} exceeds uint16"
        )
    return capacity


def _flatten_features(source: Source) -> tuple[Source, ...]:
    if source.kind == "cat":
        flattened: list[Source] = []
        for part in source.parts:
            flattened.extend(_flatten_features(part))
        return tuple(flattened)
    return (source,)


def _literal_scalar(source: Source, name: str) -> float:
    if source.kind != "literal":
        raise CppStreamLoweringError(f"{name} must be a compile-time literal")
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
    root = program.output_id
    sources: dict[int, Source] = {}
    stages: list[Stage] = []
    next_slot = 0
    next_matrix_slot = 0
    max_matrix_width = 1

    def scalar_dest(is_root: bool, shape: tuple[int, ...] = ()) -> Dest:
        nonlocal next_slot
        if is_root:
            return Dest(None, size=shape_size(shape), shape=shape)
        slot = next_slot
        next_slot += 1
        return Dest(slot, size=shape_size(shape), shape=shape)

    def matrix_dest(is_root: bool, width: int) -> Dest:
        nonlocal next_matrix_slot, max_matrix_width
        max_matrix_width = max(max_matrix_width, width)
        shape = (n_instruments, width)
        if is_root:
            return Dest(
                None,
                matrix=True,
                width=width,
                size=n_instruments * width,
                shape=shape,
            )
        slot = next_matrix_slot
        next_matrix_slot += 1
        return Dest(
            slot,
            matrix=True,
            width=width,
            size=n_instruments * width,
            shape=shape,
        )

    def tensor_dest(is_root: bool, shape: tuple[int, ...]) -> Dest:
        nonlocal next_matrix_slot, max_matrix_width
        size = shape_size(shape)
        width = max(1, (size + n_instruments - 1) // n_instruments)
        max_matrix_width = max(max_matrix_width, width)
        if is_root:
            return Dest(None, tensor=True, width=width, size=size, shape=shape)
        slot = next_matrix_slot
        next_matrix_slot += 1
        return Dest(
            slot,
            tensor=True,
            width=width,
            size=size,
            shape=shape,
        )

    def value_dest(is_root: bool, shape: tuple[int, ...]) -> Dest:
        if shape == () or shape == (n_instruments,):
            return scalar_dest(is_root, shape)
        if len(shape) == 2 and shape[0] == n_instruments:
            return matrix_dest(is_root, shape[1])
        return tensor_dest(is_root, shape)

    def source_from_dest(
        dest: Dest,
        shape: tuple[int, ...],
        *,
        dtype: str = "float64",
    ) -> Source:
        if dest.slot is None:
            return Source("output", -1, dtype=dtype, width=max(1, dest.width), shape=shape)
        if dest.tensor:
            return Source(
                "tensor_slot",
                dest.slot,
                dtype=dtype,
                width=dest.size,
                shape=shape,
            )
        if dest.matrix:
            return Source(
                "matrix_slot",
                dest.slot,
                dtype=dtype,
                width=dest.width,
                shape=shape,
            )
        return Source(
            "slot",
            dest.slot,
            row_scalar=shape == (),
            dtype=dtype,
            width=1,
            shape=shape,
        )

    for node_id, node in enumerate(program.nodes):
        op = node.op
        dtype = node_dtypes[node_id]
        row_scalar = node_id in row_scalar_nodes
        lane_count = 1 if row_scalar else n_instruments
        node_shape = resolve_shape(node.value_type, n_instruments)

        if isinstance(op, InputOp):
            sources[node_id] = Source(
                "input",
                op.input_index,
                row_scalar,
                input_dtypes[op.input_index],
                1,
                node_shape,
            )
            continue
        if isinstance(op, LiteralOp):
            sources[node_id] = Source(
                "literal",
                _coerce_literal_for_dtype(op.value, dtype),
                True,
                dtype,
                1,
                (),
            )
            continue

        children = tuple(sources[child] for child in node.child_ids)
        is_root = node_id == root

        if isinstance(op, RbfBasisOp):
            source = Source(
                "rbf",
                width=op.n_basis,
                shape=node_shape,
                parts=children,
                op=op,
            )
            sources[node_id] = source
            if is_root:
                stages.append(
                    Stage(
                        "cat",
                        (source,),
                        matrix_dest(True, op.n_basis),
                        n_instruments,
                        output_kind="matrix",
                        output_width=op.n_basis,
                    )
                )
            continue

        if isinstance(op, FutureRbfBasisSumOp):
            source = Source(
                "future_rbf",
                width=op.n_basis,
                shape=node_shape,
                parts=children,
                op=op,
            )
            sources[node_id] = source
            if is_root:
                stages.append(
                    Stage(
                        "cat",
                        (source,),
                        matrix_dest(True, op.n_basis),
                        n_instruments,
                        output_kind="matrix",
                        output_width=op.n_basis,
                    )
                )
            continue

        if isinstance(op, CatOp):
            parts: list[Source] = []
            for child in children:
                parts.extend(_flatten_features(child))
            width = sum(part.width for part in parts)
            source = Source(
                "cat",
                width=width,
                shape=node_shape,
                parts=tuple(parts),
                op=op,
            )
            sources[node_id] = source
            if is_root:
                stages.append(
                    Stage(
                        "cat",
                        tuple(parts),
                        matrix_dest(True, width),
                        n_instruments,
                        output_kind="matrix",
                        output_width=width,
                    )
                )
            continue

        if isinstance(op, ReductionOp):
            if op.temporal and not is_root:
                raise CppStreamLoweringError(
                    "temporal reductions must be the terminal output"
                )
            out = value_dest(is_root, node_shape)
            stages.append(
                Stage(
                    "reduce",
                    children,
                    out,
                    n_instruments,
                    output_kind=node.value_type.kind,
                    output_width=int(node.value_type.width),
                    op=op,
                )
            )
            sources[node_id] = source_from_dest(out, node_shape)
            continue

        if isinstance(op, EmitOp):
            if not is_root:
                raise CppStreamLoweringError("emit('last') must be the terminal output")
            out = value_dest(True, node_shape)
            stages.append(
                Stage(
                    "emit_last",
                    children,
                    out,
                    n_instruments,
                    output_kind=node.value_type.kind,
                    output_width=int(node.value_type.width),
                    op=op,
                )
            )
            sources[node_id] = source_from_dest(out, node_shape)
            continue

        if isinstance(op, InstrumentBasisMeanOp):
            sources[node_id] = Source(
                "instrument_basis",
                width=op.feature_width,
                shape=(),
                parts=children,
                op=op,
            )
            if is_root:
                raise CppStreamLoweringError(
                    "InstrumentBasisMean object must be projected with get_beta/get_preds"
                )
            continue

        if isinstance(op, RidgeOp):
            sources[node_id] = Source(
                "ridge",
                width=op.coefficient_width,
                shape=(),
                parts=children,
                op=op,
            )
            if is_root:
                raise CppStreamLoweringError("Ridge object must be projected")
            continue

        if isinstance(op, InstrumentBasisProjectionOp):
            object_source = children[0]
            if object_source.kind != "instrument_basis" or not isinstance(
                object_source.op, InstrumentBasisMeanOp
            ):
                raise CppStreamLoweringError(
                    "InstrumentBasis projection lost object source"
                )
            basis_op = object_source.op
            object_children = object_source.parts
            feature_sources = _flatten_features(object_children[0])
            y_source = object_children[1]
            if basis_op.has_weights:
                weight_source = object_children[2]
                hl_source = object_children[3]
            else:
                weight_source = Source(
                    "literal", 1.0, True, "float64", 1, ()
                )
                hl_source = object_children[2]
            half_life = _literal_scalar(hl_source, "InstrumentBasisMean hl")
            out = value_dest(is_root, node_shape)
            stages.append(
                Stage(
                    "instrument_basis",
                    tuple(feature_sources) + (y_source, weight_source),
                    out,
                    n_instruments,
                    output_kind=node.value_type.kind,
                    output_width=int(node.value_type.width),
                    op=basis_op,
                    projection=op.field,
                    half_life=half_life,
                )
            )
            sources[node_id] = source_from_dest(out, node_shape)
            continue

        if isinstance(op, RidgeProjectionOp):
            object_source = children[0]
            if object_source.kind != "ridge" or not isinstance(
                object_source.op, RidgeOp
            ):
                raise CppStreamLoweringError("Ridge projection lost object source")
            ridge_op = object_source.op
            object_children = object_source.parts
            feature_count = len(ridge_op.feature_widths)
            feature_sources: list[Source] = []
            for feature in object_children[:feature_count]:
                feature_sources.extend(_flatten_features(feature))
            y_source = object_children[feature_count]
            cursor = feature_count + 1
            if ridge_op.has_weights:
                weight_source = object_children[cursor]
                cursor += 1
            else:
                weight_source = Source(
                    "literal", 1.0, True, "float64", 1, ()
                )
            half_life = _literal_scalar(object_children[cursor], "Ridge hl")
            ridge_lambda = _literal_scalar(
                object_children[cursor + 1], "Ridge lambda"
            )
            out = value_dest(is_root, node_shape)
            stages.append(
                Stage(
                    "ridge",
                    tuple(feature_sources) + (y_source, weight_source),
                    out,
                    n_instruments,
                    output_kind=node.value_type.kind,
                    output_width=int(node.value_type.width),
                    op=ridge_op,
                    projection=op.field,
                    half_life=half_life,
                    ridge_lambda=ridge_lambda,
                )
            )
            sources[node_id] = source_from_dest(out, node_shape)
            continue

        if isinstance(op, EinsumOp):
            contraction = build_contraction_plan(
                op.spec, tuple(child.shape for child in children)
            )
            terms = list(children)
            final_source: Source | None = None
            for step_index, step in enumerate(contraction.steps):
                selected = tuple(terms[position] for position in step.operand_positions)
                final_step = step_index == len(contraction.steps) - 1
                out = value_dest(is_root and final_step, step.output_shape)
                stages.append(
                    Stage(
                        "einsum",
                        selected,
                        out,
                        n_instruments,
                        output_kind=(
                            node.value_type.kind if final_step else "tensor"
                        ),
                        output_width=(
                            int(node.value_type.width)
                            if final_step
                            else max(1, shape_size(step.output_shape))
                        ),
                        op=op,
                        einsum_step=step,
                    )
                )
                selected_positions = set(step.operand_positions)
                terms = [
                    term
                    for position, term in enumerate(terms)
                    if position not in selected_positions
                ]
                final_source = source_from_dest(out, step.output_shape)
                terms.append(final_source)
            assert final_source is not None
            sources[node_id] = final_source
            continue

        out = scalar_dest(is_root, node_shape)
        if isinstance(op, NaryOp):
            if any(child.width != 1 for child in children):
                raise CppStreamLoweringError(
                    "matrix/tensor nary operations are not implemented"
                )
            kind = (
                "unary"
                if op.arity == 1
                else "binary"
                if op.arity == 2
                else "ternary"
            )
            stage = Stage(
                kind,
                children,
                out,
                lane_count,
                dtype=dtype,
                op_name=op.name,
                output_kind=node.value_type.kind,
                output_width=int(node.value_type.width),
            )
        elif isinstance(op, CustomCallOp):
            if any(child.width != 1 for child in children):
                raise CppStreamLoweringError(
                    "named stateless calls currently require scalar-width inputs"
                )
            stage = Stage(
                "custom",
                children,
                out,
                lane_count,
                op_name=op.name,
                op=op,
                output_kind=node.value_type.kind,
                output_width=int(node.value_type.width),
            )
        elif isinstance(op, CumsumOp):
            stage = Stage("cumsum", children, out, lane_count, op=op)
        elif isinstance(op, FFillOp):
            stage = Stage("ffill", children, out, lane_count, op=op)
        elif isinstance(op, ShiftOp):
            stage = Stage("shift", children, out, lane_count, op=op)
        elif isinstance(op, EwmOp):
            stage = Stage("ewm", children, out, lane_count, op=op)
        elif isinstance(op, XsRankOp):
            stage = Stage("xs_rank", children, out, lane_count, op=op)
        elif isinstance(op, GroupByOp):
            if grouped:
                raise CppStreamLoweringError("nested groupby is not supported")
            key_count = op.n_dynamic_keys
            key_sources = children[:key_count]
            feed_sources = children[key_count:]
            if any(
                source.width != 1 for source in key_sources + feed_sources
            ):
                raise CppStreamLoweringError(
                    "groupby keys/lhs/captures must be scalar-width vectors"
                )
            specs = _resolved_key_specs(
                program, node.child_ids, op, key_cardinalities
            )
            dense = bool(specs) and all(
                spec.num_keys is not None for spec in specs
            )
            capacity = (
                _dense_capacity(specs)
                if dense
                else (op.capacity or default_group_capacity)
            )
            inner_dtypes = ("float64",) * len(op.inner_program.input_names)
            inner = _build_plan(
                op.inner_program,
                n_instruments=n_instruments,
                default_group_capacity=default_group_capacity,
                key_cardinalities=key_cardinalities,
                grouped=True,
                row_scalar_nodes=frozenset(),
                input_dtypes=inner_dtypes,
                node_dtypes=infer_node_dtypes(op.inner_program, inner_dtypes),
            )
            group_stage = GroupStage(
                inner,
                _partitions(op.static_groups, n_instruments),
                capacity,
                op.hash_capacity or 0,
                key_sources,
                specs,
                feed_sources,
                dense,
            )
            stage = Stage(
                "groupby",
                children,
                out,
                n_instruments,
                output_kind=node.value_type.kind,
                output_width=int(node.value_type.width),
                group=group_stage,
            )
        else:
            raise CppStreamLoweringError(f"unsupported IR op {type(op).__name__}")

        stages.append(stage)
        sources[node_id] = source_from_dest(out, node_shape, dtype=stage.dtype)

    root_type = program.nodes[root].value_type
    root_shape = resolve_shape(root_type, n_instruments)
    if isinstance(program.nodes[root].op, (InputOp, LiteralOp)):
        source = sources[root]
        stages.append(
            Stage(
                "copy",
                (source,),
                Dest(None, size=shape_size(root_shape), shape=root_shape),
                1 if root_shape == () else n_instruments,
                dtype=source.dtype,
                output_kind=root_type.kind,
                output_width=int(root_type.width),
            )
        )
    return Plan(
        tuple(stages),
        next_slot,
        next_matrix_slot,
        max_matrix_width,
        len(program.input_names),
        root_type.kind,
        int(root_type.width),
        _output_row_width(root_type, n_instruments),
        root_shape,
        "final"
        if isinstance(program.nodes[root].op, EmitOp)
        or (
            isinstance(program.nodes[root].op, ReductionOp)
            and program.nodes[root].op.temporal
        )
        else "rows",
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
    if n_instruments <= 0 or default_group_capacity <= 0:
        raise CppStreamLoweringError(
            "n_instruments and group capacity must be > 0"
        )
    if input_dtypes is None:
        input_dtypes = ("float64",) * len(program.input_names)
    return _build_plan(
        program,
        n_instruments=n_instruments,
        default_group_capacity=default_group_capacity,
        key_cardinalities=key_cardinalities,
        grouped=False,
        row_scalar_nodes=row_scalar_nodes or frozenset(),
        input_dtypes=input_dtypes,
        node_dtypes=infer_node_dtypes(program, input_dtypes),
    )
