from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from trading_dsl_engine.cpp_stream.python.lowering import (
    GroupStage,
    Plan,
    Source,
    Stage,
    double_bits,
)
from trading_dsl_engine.cpp_stream.python.npy import InputTypeSpec
from trading_dsl_engine.ir.ops import (
    EmitOp,
    EwmOp,
    FFillOp,
    FutureRbfBasisSumOp,
    HumpOp,
    GroupKeySpec,
    InstrumentBasisMeanOp,
    LinearFilterOp,
    NaryOp,
    PeriodsSinceChangeOp,
    RbfBasisOp,
    ReductionOp,
    RollingDecayOp,
    RollingEntropyOp,
    RollingKthOp,
    RidgeOp,
    RollingOp,
    RollingPrevDiffOp,
    RollingProductOp,
    ShiftOp,
    TheilSenOp,
    TradeWhenOp,
    VectorQuantileOp,
    XsAggregateOp,
    XsDensifyOp,
    XsGeneralizedRankOp,
    XsProjectionOp,
    XsWeightedMeanOp,
)


_CPP_TYPES = {
    "float32": "float",
    "float64": "double",
    "int32": "std::int32_t",
    "int64": "std::int64_t",
    "uint32": "std::uint32_t",
    "uint64": "std::uint64_t",
}


@dataclass(frozen=True, slots=True)
class GeneratedSource:
    text: str


class CppType:
    def render(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Name(CppType):
    value: str

    def render(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class IntArg(CppType):
    value: int

    def render(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class SignedValueArg(CppType):
    value: int

    def render(self) -> str:
        return f"{self.value}LL"


@dataclass(frozen=True, slots=True)
class UnsignedValueArg(CppType):
    value: int

    def render(self) -> str:
        return f"{self.value}ULL"


@dataclass(frozen=True, slots=True)
class UInt64Arg(CppType):
    value: int

    def render(self) -> str:
        return f"0x{self.value:016x}ULL"


@dataclass(frozen=True, slots=True)
class BoolArg(CppType):
    value: bool

    def render(self) -> str:
        return "true" if self.value else "false"


@dataclass(frozen=True, slots=True)
class DoubleArg(CppType):
    value: float

    def render(self) -> str:
        value = float(self.value)
        if math.isnan(value):
            return "std::numeric_limits<double>::quiet_NaN()"
        if math.isinf(value):
            sign = "-" if value < 0.0 else ""
            return f"{sign}std::numeric_limits<double>::infinity()"
        text = repr(value)
        return text if any(char in text for char in ".eE") else text + ".0"


@dataclass(frozen=True, slots=True)
class FloatArg(CppType):
    value: float

    def render(self) -> str:
        text = repr(float(self.value))
        return text + ("f" if any(char in text for char in ".eE") else ".0f")


@dataclass(frozen=True, slots=True)
class TemplateType(CppType):
    name: str
    args: tuple[CppType, ...]

    def render(self) -> str:
        return f"{self.name}<" + ", ".join(arg.render() for arg in self.args) + ">"


def tmpl(name: str, *args: CppType) -> TemplateType:
    return TemplateType(name, tuple(args))


def _cpp_type(dtype: str) -> CppType:
    try:
        return Name(_CPP_TYPES[dtype])
    except KeyError as exc:
        raise ValueError(f"unsupported cpp_stream dtype {dtype!r}") from exc


def _literal_arg(source: Source) -> CppType:
    if source.dtype in {"int32", "int64"}:
        return SignedValueArg(int(source.value))
    if source.dtype in {"uint32", "uint64"}:
        return UnsignedValueArg(int(source.value))
    if source.dtype == "float32":
        return FloatArg(float(source.value))
    if source.dtype == "float64":
        return DoubleArg(float(source.value))
    raise ValueError(f"unsupported literal dtype {source.dtype!r}")


def _source_type(
    source: Source,
    *,
    n: int | CppType,
    input_types: tuple[InputTypeSpec, ...] | None,
) -> CppType:
    if source.kind == "expr":
        assert isinstance(source.op, NaryOp)
        inputs = tuple(
            _source_type(part, n=n, input_types=input_types)
            for part in source.parts
        )
        result = _cpp_type(source.dtype)
        if source.op.arity == 1:
            return tmpl(
                "stackdsl::UnaryExprSrc",
                inputs[0],
                result,
                Name(_UNARY_POLICIES[source.op.name]),
            )
        if source.op.arity == 2:
            return tmpl(
                "stackdsl::BinaryExprSrc",
                inputs[0],
                inputs[1],
                result,
                Name(_BINARY_POLICIES[source.op.name]),
            )
        if source.op.arity == 3:
            return tmpl(
                "stackdsl::TernaryExprSrc",
                inputs[0],
                inputs[1],
                inputs[2],
                result,
                Name(_TERNARY_POLICIES[source.op.name]),
            )
        raise ValueError(f"unsupported expression arity {source.op.arity}")
    if source.kind == "input":
        index = int(source.value)
        if input_types is None:
            cpp_type = _cpp_type(source.dtype)
            row_width = n if isinstance(n, CppType) else IntArg(n)
        else:
            spec = input_types[index]
            cpp_type = Name(spec.cpp_type)
            row_width = IntArg(spec.row_width)
        return tmpl("stackdsl::InputSrc", IntArg(index), cpp_type, row_width)
    if source.kind == "slot":
        return tmpl(
            "stackdsl::SlotSrc",
            IntArg(int(source.value)),
            _cpp_type(source.dtype),
            BoolArg(source.row_scalar),
        )
    if source.kind == "matrix_slot":
        return tmpl(
            "stackdsl::MatrixSlotSrc",
            IntArg(int(source.value)),
            IntArg(source.width),
        )
    if source.kind == "tensor_slot":
        return tmpl(
            "stackdsl::TensorSlotSrc",
            IntArg(int(source.value)),
            IntArg(math.prod(source.shape) if source.shape else 1),
        )
    if source.kind == "literal":
        if source.dtype in {"float32", "float64"}:
            value = float(source.value)
            if math.isnan(value):
                return Name("stackdsl::NaNLiteralSrc")
            if math.isinf(value):
                return Name(
                    "stackdsl::PositiveInfinityLiteralSrc"
                    if value > 0.0
                    else "stackdsl::NegativeInfinityLiteralSrc"
                )
        return tmpl("stackdsl::LiteralSrc", _literal_arg(source))
    if source.kind == "rbf":
        assert isinstance(source.op, RbfBasisOp) and len(source.parts) == 3
        return tmpl(
            "stackdsl::RbfBasisSrc",
            IntArg(source.op.n_basis),
            *(
                _source_type(part, n=n, input_types=input_types)
                for part in source.parts
            ),
        )
    if source.kind == "future_rbf":
        assert isinstance(source.op, FutureRbfBasisSumOp) and len(source.parts) == 3
        return tmpl(
            "stackdsl::FutureRbfBasisSumSrc",
            IntArg(source.op.n_basis),
            IntArg(source.op.n_steps),
            *(
                _source_type(part, n=n, input_types=input_types)
                for part in source.parts
            ),
        )
    raise ValueError(f"source kind {source.kind!r} is not directly renderable")


def _flatten_source(source: Source) -> tuple[Source, ...]:
    if source.kind != "cat":
        return (source,)
    flattened: list[Source] = []
    for part in source.parts:
        flattened.extend(_flatten_source(part))
    return tuple(flattened)


def _feature_list(
    sources: tuple[Source, ...],
    *,
    n: int | CppType,
    input_types: tuple[InputTypeSpec, ...] | None,
) -> CppType:
    flattened: list[Source] = []
    for source in sources:
        flattened.extend(_flatten_source(source))
    return tmpl(
        "stackdsl::FeatureList",
        *(
            _source_type(source, n=n, input_types=input_types)
            for source in flattened
        ),
    )


def _tensor_shape(shape: tuple[int, ...]) -> CppType:
    return tmpl("stackdsl::TensorShape", *(IntArg(extent) for extent in shape))


def _double_list(values: tuple[float, ...]) -> CppType:
    return tmpl(
        "stackdsl::DoubleList",
        *(UInt64Arg(double_bits(value)) for value in values),
    )


def _index_map(labels: tuple[str, ...], loop_labels: tuple[str, ...]) -> CppType:
    positions = {label: index for index, label in enumerate(loop_labels)}
    return tmpl(
        "stackdsl::IndexMap", *(IntArg(positions[label]) for label in labels)
    )


def _tensor_source_type(
    source: Source,
    *,
    n: int | CppType,
    input_types: tuple[InputTypeSpec, ...] | None,
) -> CppType:
    if source.kind == "tensor_expr":
        assert isinstance(source.op, NaryOp)
        inputs = tuple(
            _tensor_source_type(part, n=n, input_types=input_types)
            for part in source.parts
        )
        shape = _tensor_shape(source.shape)
        result = _cpp_type(source.dtype)
        if source.op.arity == 1:
            return tmpl(
                "stackdsl::TensorUnaryExpr",
                inputs[0],
                shape,
                result,
                Name(_UNARY_POLICIES[source.op.name]),
            )
        if source.op.arity == 2:
            return tmpl(
                "stackdsl::TensorBinaryExpr",
                inputs[0],
                inputs[1],
                shape,
                result,
                Name(_BINARY_POLICIES[source.op.name]),
            )
        if source.op.arity == 3:
            return tmpl(
                "stackdsl::TensorTernaryExpr",
                inputs[0],
                inputs[1],
                inputs[2],
                shape,
                result,
                Name(_TERNARY_POLICIES[source.op.name]),
            )
        raise ValueError(f"unsupported tensor expression arity {source.op.arity}")
    if source.kind == "input" and len(source.shape) >= 2:
        return tmpl(
            "stackdsl::DenseTensorSource",
            _source_type(source, n=n, input_types=input_types),
            _tensor_shape(source.shape),
        )
    if source.kind == "tensor_slot":
        return tmpl(
            "stackdsl::FlatTensorSource",
            _source_type(source, n=n, input_types=input_types),
            _tensor_shape(source.shape),
        )
    if source.shape == ():
        return tmpl(
            "stackdsl::ScalarTensorSource",
            _source_type(source, n=n, input_types=input_types),
        )
    if len(source.shape) == 1:
        return tmpl(
            "stackdsl::VectorTensorSource",
            IntArg(source.shape[0]),
            _source_type(source, n=n, input_types=input_types),
        )
    if len(source.shape) == 2 and source.kind in {
        "cat",
        "rbf",
        "future_rbf",
        "matrix_slot",
    }:
        return tmpl(
            "stackdsl::FeatureTensorSource",
            IntArg(source.shape[0]),
            _feature_list((source,), n=n, input_types=input_types),
        )
    raise ValueError(
        f"cannot render tensor source kind={source.kind!r} shape={source.shape!r}"
    )


def _dest_type(stage: Stage) -> CppType:
    if stage.out.slot is None:
        return Name("stackdsl::OutputDst")
    if stage.out.tensor:
        return tmpl(
            "stackdsl::TensorSlotDst",
            IntArg(stage.out.slot),
            IntArg(stage.out.size),
        )
    if stage.out.matrix:
        return tmpl(
            "stackdsl::MatrixSlotDst",
            IntArg(stage.out.slot),
            IntArg(stage.out.width),
        )
    return tmpl(
        "stackdsl::SlotDst", IntArg(stage.out.slot), _cpp_type(stage.dtype)
    )


_BINARY_POLICIES = {
    "add": "stackdsl::AddOp",
    "sub": "stackdsl::SubOp",
    "mul": "stackdsl::MulOp",
    "div": "stackdsl::DivOp",
    "mod": "stackdsl::ModOp",
    "pow": "stackdsl::PowOp",
    "minimum": "stackdsl::MinOp",
    "maximum": "stackdsl::MaxOp",
    "eq": "stackdsl::EqOp",
    "ne": "stackdsl::NeOp",
    "lt": "stackdsl::LtOp",
    "gt": "stackdsl::GtOp",
    "le": "stackdsl::LeOp",
    "ge": "stackdsl::GeOp",
    "and_": "stackdsl::AndOp",
    "or_": "stackdsl::OrOp",
    "xor": "stackdsl::XorOp",
    "fillna": "stackdsl::FillNaOp",
}
_UNARY_POLICIES = {
    "abs": "stackdsl::AbsOp",
    "ceil": "stackdsl::CeilOp",
    "floor": "stackdsl::FloorOp",
    "exp": "stackdsl::ExpOp",
    "ln": "stackdsl::LogOp",
    "round": "stackdsl::RoundOp",
    "sign": "stackdsl::SignOp",
    "fraction": "stackdsl::FractionOp",
    "purify": "stackdsl::PurifyOp",
    "arctan": "stackdsl::AtanOp",
    "acos": "stackdsl::AcosOp",
    "asin": "stackdsl::AsinOp",
    "sin": "stackdsl::SinOp",
    "cos": "stackdsl::CosOp",
    "tan": "stackdsl::TanOp",
    "tanh": "stackdsl::TanhOp",
    "sqrt": "stackdsl::SqrtOp",
    "isnan": "stackdsl::IsNanOp",
    "isfinite": "stackdsl::IsFiniteOp",
    "logical_not": "stackdsl::LogicalNotOp",
    "norm_inv": "stackdsl::NormInvOp",
    "pow2": "stackdsl::Pow2Op",
    "pow3": "stackdsl::Pow3Op",
    "pow4": "stackdsl::Pow4Op",
}
_TERNARY_POLICIES = {"where": "stackdsl::WhereOp"}
_CUSTOM_POLICIES = {
    "volume_for_fit_session": "stackdsl::VolumeForFitSessionPolicy",
    "volume_for_seen_session": "stackdsl::VolumeForSeenSessionPolicy",
    "nonnegative": "stackdsl::NonnegativePolicy",
    "pct_seen_session_volume": "stackdsl::PctSeenSessionVolumePolicy",
}


def _stateful_alpha(half_life: float) -> float:
    if not math.isfinite(half_life) or half_life <= 0.0:
        return 1.0
    return 1.0 - math.exp(math.log(0.5) / half_life)


def _ridge_projection_type(
    field: str, component: int | None
) -> CppType:
    if field in {"coefficient", "standard_error", "tstat"}:
        if component is None:
            raise ValueError(f"Ridge {field} projection requires a component")
        return tmpl(
            {
                "coefficient": "stackdsl::RidgeCoefficientProjection",
                "standard_error": "stackdsl::RidgeStandardErrorProjection",
                "tstat": "stackdsl::RidgeTStatProjection",
            }[field],
            IntArg(component),
        )
    return Name({
        "beta": "stackdsl::RidgeBetaProjection",
        "preds": "stackdsl::RidgePredsProjection",
        "residuals": "stackdsl::RidgeResidualsProjection",
        "standard_errors": "stackdsl::RidgeStandardErrorsProjection",
        "tstats": "stackdsl::RidgeTStatsProjection",
        "sse": "stackdsl::RidgeSseProjection",
        "sst": "stackdsl::RidgeSstProjection",
        "r2": "stackdsl::RidgeR2Projection",
        "residual_variance": "stackdsl::RidgeResidualVarianceProjection",
        "effective_df": "stackdsl::RidgeEffectiveDfProjection",
        "effective_n": "stackdsl::RidgeEffectiveNProjection",
    }[field])


def _stage_type(
    stage: Stage,
    n: CppType,
    execution: CppType,
    *,
    input_types: tuple[InputTypeSpec, ...] | None,
) -> CppType:
    stage_n: CppType = IntArg(1) if stage.lane_count == 1 else n
    out = _dest_type(stage)
    if stage.kind == "reduce_bundle":
        assert isinstance(stage.op, ReductionOp) and not stage.op.temporal
        if len(stage.bundle_outs) != len(stage.inputs) or len(stage.inputs) < 2:
            raise ValueError("reduction bundle inputs/outputs must have equal size > 1")
        tensors = tuple(
            _tensor_source_type(source, n=n, input_types=input_types)
            for source in stage.inputs
        )
        outputs = tuple(
            _dest_type(replace(stage, out=dest))
            for dest in stage.bundle_outs
        )
        row_axes = tuple(axis - 1 for axis in stage.op.axes if axis != 0)
        policy = {
            "sum": "stackdsl::SumReductionPolicy",
            "mean": "stackdsl::MeanReductionPolicy",
            "std": "stackdsl::StdReductionPolicy",
            "min": "stackdsl::MinReductionPolicy",
            "max": "stackdsl::MaxReductionPolicy",
        }[stage.op.kind]
        return tmpl(
            "stackdsl::ReductionBundleNode",
            tmpl("stackdsl::TypeList", *tensors),
            tmpl("stackdsl::TypeList", *outputs),
            tmpl("stackdsl::AxisList", *(IntArg(axis) for axis in row_axes)),
            Name(policy),
            IntArg(stage.op.ddof),
            BoolArg(stage.op.ignore_na),
        )
    if stage.kind == "reduce":
        assert isinstance(stage.op, ReductionOp)
        tensor_source = _tensor_source_type(
            stage.inputs[0], n=n, input_types=input_types
        )
        row_axes = tuple(axis - 1 for axis in stage.op.axes if axis != 0)
        policy = {
            "sum": "stackdsl::SumReductionPolicy",
            "mean": "stackdsl::MeanReductionPolicy",
            "std": "stackdsl::StdReductionPolicy",
            "min": "stackdsl::MinReductionPolicy",
            "max": "stackdsl::MaxReductionPolicy",
        }[stage.op.kind]
        return tmpl(
            "stackdsl::ReductionNode",
            tensor_source,
            out,
            tmpl("stackdsl::AxisList", *(IntArg(axis) for axis in row_axes)),
            Name(policy),
            IntArg(stage.op.ddof),
            BoolArg(stage.op.ignore_na),
            BoolArg(stage.op.temporal),
        )
    if stage.kind == "emit_last":
        assert isinstance(stage.op, EmitOp)
        return tmpl(
            "stackdsl::EmitLastNode",
            _tensor_source_type(stage.inputs[0], n=n, input_types=input_types),
            out,
        )
    if stage.kind == "cat":
        return tmpl(
            "stackdsl::CatNode",
            n,
            _feature_list(stage.inputs, n=n, input_types=input_types),
            out,
            execution,
        )
    if stage.kind == "einsum":
        step = stage.einsum_step
        if step is None:
            raise ValueError("einsum stage is missing its contraction step")
        tensors = tuple(
            _tensor_source_type(source, n=n, input_types=input_types)
            for source in stage.inputs
        )
        maps = tuple(
            _index_map(labels, step.loop_labels) for labels in step.input_labels
        )
        loop_shape = _tensor_shape(step.loop_extents)
        output_rank = IntArg(len(step.output_labels))
        if len(tensors) == 1:
            return tmpl(
                "stackdsl::UnaryEinsumNode",
                tensors[0],
                out,
                loop_shape,
                maps[0],
                output_rank,
                execution,
            )
        if len(tensors) == 2:
            return tmpl(
                "stackdsl::BinaryEinsumNode",
                tensors[0],
                tensors[1],
                out,
                loop_shape,
                maps[0],
                maps[1],
                output_rank,
                execution,
            )
        raise ValueError("physical einsum stages must be unary or binary")

    inputs = tuple(
        _source_type(source, n=n, input_types=input_types)
        for source in stage.inputs
    )
    if stage.kind == "copy":
        return tmpl("stackdsl::CopyNode", stage_n, inputs[0], out, execution)
    if stage.kind == "binary":
        return tmpl(
            "stackdsl::BinaryNode",
            stage_n,
            inputs[0],
            inputs[1],
            out,
            _cpp_type(stage.dtype),
            Name(_BINARY_POLICIES[stage.op_name or ""]),
            execution,
        )
    if stage.kind == "ternary":
        return tmpl(
            "stackdsl::TernaryNode",
            stage_n,
            inputs[0],
            inputs[1],
            inputs[2],
            out,
            _cpp_type(stage.dtype),
            Name(_TERNARY_POLICIES[stage.op_name or ""]),
            execution,
        )
    if stage.kind == "unary":
        return tmpl(
            "stackdsl::UnaryNode",
            stage_n,
            inputs[0],
            out,
            _cpp_type(stage.dtype),
            Name(_UNARY_POLICIES[stage.op_name or ""]),
            execution,
        )
    if stage.kind == "custom":
        try:
            policy = _CUSTOM_POLICIES[stage.op_name or ""]
        except KeyError as exc:
            raise ValueError(
                f"no native policy for stateless call {stage.op_name!r}"
            ) from exc
        return tmpl(
            "stackdsl::StatelessNode",
            stage_n,
            out,
            Name(policy),
            execution,
            *inputs,
        )
    if stage.kind == "cumsum":
        return tmpl("stackdsl::CumsumNode", stage_n, inputs[0], out, execution)
    if stage.kind == "ffill":
        assert isinstance(stage.op, FFillOp)
        limit = -1 if stage.op.limit is None else stage.op.limit
        return tmpl(
            "stackdsl::FFillNode",
            stage_n,
            inputs[0],
            out,
            SignedValueArg(limit),
            execution,
        )
    if stage.kind == "shift":
        assert isinstance(stage.op, ShiftOp)
        return tmpl(
            "stackdsl::ShiftNode",
            stage_n,
            inputs[0],
            out,
            IntArg(stage.op.lag),
            IntArg(stage.op.max_lag),
            execution,
        )
    if stage.kind == "ewm":
        assert isinstance(stage.op, EwmOp)
        return tmpl(
            "stackdsl::EwmNode",
            stage_n,
            inputs[0],
            out,
            UInt64Arg(double_bits(stage.op.span)),
            IntArg(stage.op.min_periods),
            BoolArg(stage.op.ignore_na),
            BoolArg(stage.op.adjust),
            execution,
        )
    if stage.kind == "ewm_bundle":
        assert isinstance(stage.op, EwmOp)
        if len(stage.bundle_outs) != len(inputs) or len(inputs) < 2:
            raise ValueError("EWM bundle inputs/outputs must have equal size > 1")
        outputs = tuple(
            _dest_type(replace(stage, out=dest))
            for dest in stage.bundle_outs
        )
        return tmpl(
            "stackdsl::EwmBundleNode",
            stage_n,
            tmpl("stackdsl::TypeList", *inputs),
            tmpl("stackdsl::TypeList", *outputs),
            UInt64Arg(double_bits(stage.op.span)),
            IntArg(stage.op.min_periods),
            BoolArg(stage.op.ignore_na),
            BoolArg(stage.op.adjust),
            execution,
        )
    if stage.kind == "periods_since_change":
        assert isinstance(stage.op, PeriodsSinceChangeOp)
        return tmpl(
            "stackdsl::PeriodsSinceChangeNode",
            stage_n,
            inputs[0],
            out,
            execution,
        )
    if stage.kind == "hump":
        assert isinstance(stage.op, HumpOp)
        return tmpl(
            "stackdsl::HumpNode",
            stage_n,
            inputs[0],
            out,
            UInt64Arg(double_bits(stage.op.threshold)),
            BoolArg(stage.op.relative),
            BoolArg(stage.op.move_by_threshold),
            execution,
        )
    if stage.kind == "trade_when":
        assert isinstance(stage.op, TradeWhenOp)
        return tmpl(
            "stackdsl::TradeWhenNode",
            stage_n,
            inputs[0],
            inputs[1],
            inputs[2],
            out,
            execution,
        )
    if stage.kind == "linear_filter":
        assert isinstance(stage.op, LinearFilterOp)
        return tmpl(
            "stackdsl::LinearFilterNode",
            stage_n,
            inputs[0],
            out,
            _double_list(stage.op.feedforward),
            _double_list(stage.op.recursive),
            execution,
        )
    if stage.kind == "rolling_product":
        assert isinstance(stage.op, RollingProductOp)
        return tmpl(
            "stackdsl::RollingProductNode",
            stage_n,
            inputs[0],
            out,
            IntArg(stage.op.periods),
            IntArg(stage.op.min_periods),
            execution,
        )
    if stage.kind == "rolling_kth":
        assert isinstance(stage.op, RollingKthOp)
        return tmpl(
            "stackdsl::RollingKthNode",
            stage_n,
            inputs[0],
            out,
            IntArg(stage.op.periods),
            IntArg(stage.op.min_periods),
            IntArg(stage.op.k),
            BoolArg(stage.op.ignore_zero),
            execution,
        )
    if stage.kind == "rolling_prev_diff":
        assert isinstance(stage.op, RollingPrevDiffOp)
        return tmpl(
            "stackdsl::RollingPrevDiffNode",
            stage_n,
            inputs[0],
            out,
            IntArg(stage.op.periods),
            execution,
        )
    if stage.kind == "rolling_decay":
        assert isinstance(stage.op, RollingDecayOp)
        return tmpl(
            "stackdsl::RollingLinearDecayNode",
            stage_n,
            inputs[0],
            out,
            IntArg(stage.op.periods),
            IntArg(stage.op.min_periods),
            execution,
        )
    if stage.kind == "rolling_entropy":
        assert isinstance(stage.op, RollingEntropyOp)
        return tmpl(
            "stackdsl::RollingEntropyNode",
            stage_n,
            inputs[0],
            out,
            IntArg(stage.op.periods),
            IntArg(stage.op.min_periods),
            IntArg(stage.op.buckets),
            execution,
        )
    if stage.kind == "xs_rank":
        return tmpl("stackdsl::XsRankNode", stage_n, inputs[0], out, execution)
    if stage.kind == "xs_pct_rank":
        return tmpl("stackdsl::XsPctRankNode", stage_n, inputs[0], out, execution)
    if stage.kind == "xs_aggregate":
        assert isinstance(stage.op, XsAggregateOp)
        policy = Name({
            "count": "stackdsl::XsCountProjection",
            "sum": "stackdsl::XsSumProjection",
            "mean": "stackdsl::XsMeanProjection",
            "std": "stackdsl::XsStdProjection",
            "min": "stackdsl::XsMinProjection",
            "max": "stackdsl::XsMaxProjection",
            "quantile": "stackdsl::XsQuantileProjection",
        }[stage.op.kind])
        return tmpl(
            "stackdsl::XsAggregateNode",
            stage_n,
            inputs[0],
            out,
            policy,
            UInt64Arg(double_bits(stage.op.quantile)),
            execution,
        )
    if stage.kind == "xs_weighted_mean":
        assert isinstance(stage.op, XsWeightedMeanOp)
        return tmpl(
            "stackdsl::XsWeightedMeanNode",
            stage_n,
            inputs[0],
            inputs[1],
            out,
            execution,
        )
    if stage.kind == "xs_projection":
        assert isinstance(stage.op, XsProjectionOp)
        return tmpl(
            "stackdsl::XsProjectionNode",
            stage_n,
            inputs[0],
            inputs[1],
            out,
            BoolArg(stage.op.intercept),
            execution,
        )
    if stage.kind == "xs_generalized_rank":
        assert isinstance(stage.op, XsGeneralizedRankOp)
        return tmpl(
            "stackdsl::XsGeneralizedRankNode",
            stage_n,
            inputs[0],
            out,
            UInt64Arg(double_bits(stage.op.power)),
            execution,
        )
    if stage.kind == "xs_densify":
        assert isinstance(stage.op, XsDensifyOp)
        return tmpl(
            "stackdsl::XsDensifyNode",
            stage_n,
            inputs[0],
            out,
            execution,
        )
    if stage.kind == "rolling":
        assert isinstance(stage.op, RollingOp)
        if stage.op.kind in {"sum", "mean", "std"}:
            projection = Name({
                "sum": "stackdsl::RollingSumProjection",
                "mean": "stackdsl::RollingMeanProjection",
                "std": "stackdsl::RollingStdProjection",
            }[stage.op.kind])
            return tmpl(
                "stackdsl::RollingMomentsNode",
                stage_n,
                inputs[0],
                out,
                IntArg(stage.op.periods),
                IntArg(stage.op.min_periods),
                IntArg(stage.op.ddof),
                projection,
                execution,
            )
        if stage.op.kind in {"min", "max", "argmin", "argmax"}:
            return tmpl(
                "stackdsl::RollingExtremaNode",
                stage_n,
                inputs[0],
                out,
                IntArg(stage.op.periods),
                IntArg(stage.op.min_periods),
                BoolArg(stage.op.kind in {"max", "argmax"}),
                BoolArg(stage.op.kind in {"argmin", "argmax"}),
                execution,
            )
        projection = Name(
            "stackdsl::RollingPctRankProjection"
            if stage.op.kind == "pct_rank"
            else "stackdsl::RollingQuantileProjection"
        )
        quantile = 0.5 if stage.op.kind == "median" else stage.op.quantile
        return tmpl(
            "stackdsl::RollingOrderNode",
            stage_n,
            inputs[0],
            out,
            IntArg(stage.op.periods),
            IntArg(stage.op.min_periods),
            UInt64Arg(double_bits(quantile)),
            projection,
            execution,
        )
    if stage.kind == "theilsen":
        assert isinstance(stage.op, TheilSenOp)
        return tmpl(
            "stackdsl::RollingTheilSenNode",
            stage_n,
            inputs[0],
            inputs[1],
            out,
            IntArg(stage.op.periods),
            IntArg(stage.op.min_periods),
            execution,
        )
    if stage.kind == "vector_quantile":
        assert isinstance(stage.op, VectorQuantileOp)
        return tmpl(
            "stackdsl::VectorQuantileNode",
            stage_n,
            _tensor_source_type(stage.inputs[0], n=n, input_types=input_types),
            out,
            UInt64Arg(double_bits(stage.op.quantile)),
            execution,
        )
    if stage.kind == "instrument_basis":
        assert isinstance(stage.op, InstrumentBasisMeanOp)
        assert stage.projection in {"beta", "preds"}
        assert stage.half_life is not None
        features = stage.inputs[:-2]
        y_source, weight_source = stage.inputs[-2:]
        projection = (
            "stackdsl::InstrumentBasisBetaProjection"
            if stage.projection == "beta"
            else "stackdsl::InstrumentBasisPredsProjection"
        )
        return tmpl(
            "stackdsl::InstrumentBasisMeanNode",
            n,
            _feature_list(features, n=n, input_types=input_types),
            _source_type(y_source, n=n, input_types=input_types),
            _source_type(weight_source, n=n, input_types=input_types),
            out,
            UInt64Arg(double_bits(_stateful_alpha(stage.half_life))),
            Name(projection),
            execution,
        )
    if stage.kind in {"ridge", "ridge_bundle"}:
        assert isinstance(stage.op, RidgeOp)
        assert stage.half_life is not None and stage.ridge_lambda is not None
        features = stage.inputs[:-2]
        y_source, weight_source = stage.inputs[-2:]
        stateful = (
            stage.op.is_stateful
            and math.isfinite(stage.half_life)
            and stage.half_life > 0.0
        )
        common = (
            n,
            _feature_list(features, n=n, input_types=input_types),
            _source_type(y_source, n=n, input_types=input_types),
            _source_type(weight_source, n=n, input_types=input_types),
        )
        trailing = (
            UInt64Arg(double_bits(_stateful_alpha(stage.half_life))),
            UInt64Arg(double_bits(stage.ridge_lambda)),
            BoolArg(stage.op.nonneg),
            BoolArg(stateful),
        )
        if stage.kind == "ridge_bundle":
            if (
                len(stage.bundle_outs) != len(stage.bundle_projections)
                or len(stage.bundle_outs) < 2
            ):
                raise ValueError(
                    "Ridge bundle outputs/projections must have equal size > 1"
                )
            bindings = tuple(
                tmpl(
                    "stackdsl::RidgeProjectionBinding",
                    _dest_type(replace(stage, out=dest)),
                    _ridge_projection_type(field, component),
                )
                for dest, (field, component) in zip(
                    stage.bundle_outs,
                    stage.bundle_projections,
                    strict=True,
                )
            )
            return tmpl(
                "stackdsl::RidgeBundleNode",
                *common,
                *trailing,
                tmpl("stackdsl::RidgeProjectionBundle", *bindings),
                execution,
            )
        assert stage.projection is not None
        projection = _ridge_projection_type(
            stage.projection, stage.projection_component
        )
        return tmpl(
            "stackdsl::RidgeNode",
            *common,
            out,
            *trailing,
            projection,
            execution,
        )
    if stage.kind == "groupby":
        raise AssertionError("groupby type is rendered separately")
    raise AssertionError(stage.kind)


def _source_list(
    sources: tuple[Source, ...],
    *,
    n: int | CppType,
    input_types: tuple[InputTypeSpec, ...] | None,
) -> CppType:
    return tmpl(
        "stackdsl::SourceList",
        *(
            _source_type(source, n=n, input_types=input_types)
            for source in sources
        ),
    )


def _key_type(
    source: Source,
    spec: GroupKeySpec,
    *,
    n: int,
    input_types: tuple[InputTypeSpec, ...],
) -> CppType:
    return tmpl(
        "stackdsl::KeySpec",
        _source_type(source, n=n, input_types=input_types),
        IntArg(0 if spec.num_keys is None else int(spec.num_keys)),
        IntArg(int(spec.offset)),
        BoolArg(bool(spec.row_scalar)),
    )


def _key_list(
    group: GroupStage,
    *,
    n: int,
    input_types: tuple[InputTypeSpec, ...],
) -> tuple[CppType, ...]:
    if len(group.key_sources) != len(group.key_specs):
        raise ValueError("group key source/spec length mismatch")
    return tuple(
        _key_type(source, spec, n=n, input_types=input_types)
        for source, spec in zip(group.key_sources, group.key_specs)
    )


@dataclass(frozen=True, slots=True)
class StageView:
    index: int
    cpp_type: str
    checked: bool = False
    final_on_data: bool = False
    finalizer: bool = False


@dataclass(frozen=True, slots=True)
class InnerView:
    name: str
    input_count: int
    scratch_slots: int
    matrix_scratch_slots: int
    matrix_scratch_width: int
    stages: tuple[StageView, ...]


@dataclass(frozen=True, slots=True)
class InputView:
    index: int
    cpp_type: str
    row_width: int


def _can_fuse_physical_pair(first: Stage, second: Stage) -> bool:
    return (
        first.kind in {"ewm_bundle", "reduce_bundle", "ridge_bundle"}
        and second.kind in {"copy", "cat", "tensor_copy"}
        and not first.final_only
        and not second.final_only
        and first.group is None
        and second.group is None
    )


def _inner_view(name: str, group: GroupStage) -> InnerView:
    n = Name("N")
    execution = tmpl(
        "stackdsl::GroupedExecution",
        n,
        Name("Capacity"),
        Name("PartitionCount"),
    )
    stages: list[StageView] = []
    stage_index = 0
    emitted_index = 0
    while stage_index < len(group.inner.stages):
        stage = group.inner.stages[stage_index]
        if stage.kind == "groupby":
            raise ValueError("nested groupby is not supported")
        if (
            stage_index + 1 < len(group.inner.stages)
            and _can_fuse_physical_pair(
                stage, group.inner.stages[stage_index + 1]
            )
        ):
            second = group.inner.stages[stage_index + 1]
            stages.append(
                StageView(
                    emitted_index,
                    tmpl(
                        "stackdsl::FusedStageNode",
                        _stage_type(stage, n, execution, input_types=None),
                        _stage_type(second, n, execution, input_types=None),
                    ).render(),
                )
            )
            stage_index += 2
            emitted_index += 1
            continue
        stages.append(
            StageView(
                emitted_index,
                _stage_type(stage, n, execution, input_types=None).render(),
            )
        )
        stage_index += 1
        emitted_index += 1
    return InnerView(
        name,
        group.inner.input_count,
        group.inner.scratch_slots,
        group.inner.matrix_scratch_slots,
        group.inner.matrix_scratch_width,
        tuple(stages),
    )


def _group_type(
    group_name: str,
    group: GroupStage,
    stage: Stage,
    n: int,
    input_types: tuple[InputTypeSpec, ...],
) -> CppType:
    keys = _key_list(group, n=n, input_types=input_types)
    if not keys:
        resolver = tmpl("stackdsl::NoKeyResolver", IntArg(n))
    elif group.dense:
        resolver = tmpl("stackdsl::DenseTupleGroupResolver", IntArg(n), *keys)
    else:
        resolver = tmpl(
            "stackdsl::HashGroupResolver",
            IntArg(n),
            IntArg(group.capacity),
            IntArg(group.hash_capacity),
            *keys,
        )
    partitions = tmpl(
        "stackdsl::StaticPartitions",
        IntArg(n),
        *(IntArg(value) for value in group.partitions),
    )
    inner = tmpl(
        group_name,
        IntArg(n),
        Name(f"{resolver.render()}::capacity"),
        IntArg(max(group.partitions, default=0) + 1),
    )
    return tmpl(
        "stackdsl::GroupByNode",
        IntArg(n),
        resolver,
        partitions,
        inner,
        _dest_type(stage),
        tmpl("stackdsl::KeyList", *keys),
        _source_list(group.feed_sources, n=n, input_types=input_types),
    )


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(Path(__file__).with_name("templates")),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_translation_unit(
    plan: Plan,
    *,
    n_instruments: int,
    prefetch_rows: int,
    input_types: tuple[InputTypeSpec, ...] | None = None,
) -> GeneratedSource:
    if input_types is None:
        input_types = tuple(
            InputTypeSpec("float64", n_instruments)
            for _ in range(plan.input_count)
        )
    if len(input_types) != plan.input_count:
        raise ValueError("input type count does not match the compiled program")

    group_names: dict[int, str] = {}
    inners: list[InnerView] = []
    for index, stage in enumerate(plan.stages):
        if stage.kind == "groupby":
            assert stage.group is not None
            name = f"CppStreamInner{index}"
            group_names[index] = name
            inners.append(_inner_view(name, stage.group))

    n = IntArg(n_instruments)
    direct = tmpl("stackdsl::DirectExecution", n)
    stages: list[StageView] = []
    original_index = 0
    emitted_index = 0
    while original_index < len(plan.stages):
        stage = plan.stages[original_index]
        if (
            original_index + 1 < len(plan.stages)
            and _can_fuse_physical_pair(
                stage, plan.stages[original_index + 1]
            )
        ):
            second = plan.stages[original_index + 1]
            stages.append(
                StageView(
                    emitted_index,
                    tmpl(
                        "stackdsl::FusedStageNode",
                        _stage_type(
                            stage, n, direct, input_types=input_types
                        ),
                        _stage_type(
                            second, n, direct, input_types=input_types
                        ),
                    ).render(),
                )
            )
            original_index += 2
            emitted_index += 1
            continue
        if stage.kind == "groupby":
            assert stage.group is not None
            stages.append(
                StageView(
                    emitted_index,
                    _group_type(
                        group_names[original_index],
                        stage.group,
                        stage,
                        n_instruments,
                        input_types,
                    ).render(),
                    checked=True,
                    final_on_data=stage.final_only,
                )
            )
        else:
            stages.append(
                StageView(
                    emitted_index,
                    _stage_type(
                        stage, n, direct, input_types=input_types
                    ).render(),
                    final_on_data=stage.final_only,
                    finalizer=stage.kind in {"reduce", "emit_last"}
                    and plan.output_mode == "final",
                )
            )
        original_index += 1
        emitted_index += 1
    if plan.input_count == 0:
        raise ValueError("cpp_stream requires at least one file-backed input")

    return GeneratedSource(
        _environment().get_template("runner.cpp.j2").render(
            n=n_instruments,
            input_count=plan.input_count,
            scratch_slots=plan.scratch_slots,
            matrix_scratch_slots=plan.matrix_scratch_slots,
            matrix_scratch_width=plan.matrix_scratch_width,
            output_row_width=plan.output_row_width,
            output_mode=plan.output_mode,
            prefetch_rows=prefetch_rows,
            inputs=tuple(
                InputView(index, spec.cpp_type, spec.row_width)
                for index, spec in enumerate(input_types)
            ),
            stages=tuple(stages),
            inners=tuple(inners),
        )
    )
