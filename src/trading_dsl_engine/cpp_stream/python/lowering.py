from __future__ import annotations

from dataclasses import dataclass, replace
import math
import struct
from typing import Mapping

from trading_dsl_engine.ir.einsum import ContractionStep
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
SCALAR_SCRATCH_DTYPES = (
    "float64",
    "float32",
    "int64",
    "uint64",
    "int32",
    "uint32",
)


class ScalarScratchSlots(int):
    """Int-compatible total carrying exact per-dtype scratch counts."""

    counts: tuple[int, int, int, int, int, int]

    def __new__(cls, counts: tuple[int, ...]):
        if len(counts) != len(SCALAR_SCRATCH_DTYPES):
            raise ValueError("scalar scratch count length mismatch")
        normalized = tuple(int(value) for value in counts)
        if any(value < 0 for value in normalized):
            raise ValueError("scalar scratch counts must be nonnegative")
        value = int.__new__(cls, sum(normalized))
        value.counts = normalized
        return value


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


def scalar_scratch_slots(stages: tuple[Stage, ...] | list[Stage]) -> ScalarScratchSlots:
    """Return the scalar arrays generated C++ can physically address.

    Slot ids are compile-time indexes within each native dtype. EWM bundle members
    become component labels rather than RowContext storage whenever an epilogue is
    present because codegen binds every member to ``EwmDiscardDst``.
    """

    counts = {dtype: 0 for dtype in SCALAR_SCRATCH_DTYPES}
    for stage in stages:
        candidates = (
            stage.epilogues
            if stage.kind == "ewm_bundle" and stage.epilogues
            else (stage, *stage.members, *stage.epilogues)
        )
        for candidate in candidates:
            slot = candidate.out.slot
            if (
                slot is None
                or slot < 0
                or candidate.out.matrix
                or candidate.out.tensor
            ):
                continue
            counts[candidate.dtype] = max(
                counts[candidate.dtype], int(slot) + 1
            )
    return ScalarScratchSlots(
        tuple(counts[dtype] for dtype in SCALAR_SCRATCH_DTYPES)
    )


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

    def __post_init__(self) -> None:
        if isinstance(self.scratch_slots, ScalarScratchSlots):
            return
        object.__setattr__(
            self,
            "scratch_slots",
            scalar_scratch_slots(self.stages),
        )


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


def lower_program(
    program: Program,
    *,
    n_instruments: int,
    default_group_capacity: int = 64,
    key_cardinalities: Mapping[str, int] | None = None,
    row_scalar_nodes: frozenset[int] | None = None,
    input_dtypes: tuple[str, ...] | None = None,
) -> Plan:
    """Compatibility entrypoint for the unified 1..K output lowerer."""

    from trading_dsl_engine.cpp_stream.python.lowering_multi import (
        lower_program as lower_outputs,
    )

    return lower_outputs(
        program,
        n_instruments=n_instruments,
        default_group_capacity=default_group_capacity,
        key_cardinalities=key_cardinalities,
        row_scalar_nodes=row_scalar_nodes,
        input_dtypes=input_dtypes,
    )


__all__ = [
    "CppStreamLoweringError",
    "Dest",
    "GroupStage",
    "Plan",
    "ScalarScratchSlots",
    "SCALAR_SCRATCH_DTYPES",
    "Source",
    "Stage",
    "double_bits",
    "infer_node_dtypes",
    "lower_program",
    "scalar_scratch_slots",
]
