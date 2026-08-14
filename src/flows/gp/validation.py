from __future__ import annotations

from math import prod

from flows.gp.tensor_types import NumericTensor, TensorIndex, tensor_rank
from flows.gp.types import (
    AxisSpec,
    BoolParam,
    BoolRow,
    CountRow,
    DatetimeUnit,
    DerivedNumericRow,
    DimensionlessRow,
    DurationRow,
    FilterHSpec,
    FilterTSpec,
    FrequencySpec,
    GroupKeyInput,
    GroupVectorInput,
    KthIgnoreSpec,
    NumericRow,
    PeriodAtLeastTwo,
    PositiveFloat,
    PositiveInt,
    PositiveNumber,
    PriceRow,
    QuantileParam,
    QuantityRow,
    TimestampRow,
    TradingDayHorizonRow,
)
from trading_dsl_engine.base import dsl
from trading_dsl_engine.base.keys import key
from trading_dsl_engine.cpp_stream.python.source_types import InputTypeSpec
from trading_dsl_engine.ir.types import VECTOR, tensor


_ROW_SAMPLES = {
    PriceRow: ("ap0_out0", PriceRow),
    QuantityRow: ("volume_a0_out0", QuantityRow),
    TimestampRow: ("_ev_ts", TimestampRow),
    DurationRow: ("wdte_out0", DurationRow),
    TradingDayHorizonRow: ("wdte_out0", TradingDayHorizonRow),
    DimensionlessRow: ("vw_halfspread_out0", DimensionlessRow),
    BoolRow: ("is_tradable_out0", BoolRow),
    CountRow: ("trade_cross_pct_out0.count", CountRow),
    DerivedNumericRow: ("ap1_out0", DerivedNumericRow),
    NumericRow: ("ap0_out0", PriceRow),
}

_STATIC_SAMPLES = {
    PositiveInt: PositiveInt(20),
    PeriodAtLeastTwo: PeriodAtLeastTwo(20),
    PositiveFloat: PositiveFloat(0.5),
    PositiveNumber: PositiveInt(2),
    QuantileParam: QuantileParam(0.5),
    BoolParam: BoolParam(True),
    KthIgnoreSpec: KthIgnoreSpec("NAN 0"),
    TensorIndex: TensorIndex(0),
    AxisSpec: AxisSpec(1),
    DatetimeUnit: DatetimeUnit("us"),
    FrequencySpec: FrequencySpec("1min"),
    FilterHSpec: FilterHSpec("1,2,3,4"),
    FilterTSpec: FilterTSpec("0.5"),
}


def is_tensor_primitive(primitive) -> bool:
    return issubclass(primitive.ret, NumericTensor) or any(
        issubclass(type_, NumericTensor) for type_ in primitive.args
    )


def family_primitives(pset, family: str):
    names = getattr(pset, "gp_primitive_family", {})
    return [
        pset.mapping[name]
        for name, current in names.items()
        if current == family
    ]


def sample_argument(
    type_,
    ir_types: dict,
    input_specs: dict,
    n_instruments: int = 9,
):
    if issubclass(type_, GroupKeyInput):
        name = "group_key"
        ir_types[name] = VECTOR
        input_specs[name] = InputTypeSpec("float64", n_instruments)
        return GroupKeyInput(
            key(dsl.var(name), num_keys=2, offset=0, row_scalar=False)
        )
    if issubclass(type_, GroupVectorInput):
        name = "group_vector_input"
        ir_types[name] = VECTOR
        input_specs[name] = InputTypeSpec("float64", n_instruments)
        return GroupVectorInput(dsl.var(name))
    if issubclass(type_, NumericTensor):
        rank = tensor_rank(type_)
        name = f"tensor_rank_{rank}"
        logical_shape = (None, *(3 for _ in range(rank - 1)))
        row_shape = (n_instruments, *(3 for _ in range(rank - 1)))
        ir_types[name] = tensor(logical_shape)
        input_specs[name] = InputTypeSpec(
            "float64",
            prod(row_shape),
            row_shape=row_shape,
        )
        return type_(dsl.var(name))
    if issubclass(type_, NumericRow):
        name, concrete = _ROW_SAMPLES[type_]
        ir_types[name] = VECTOR
        input_specs[name] = InputTypeSpec("float64", n_instruments)
        return concrete(dsl.var(name))
    try:
        return _STATIC_SAMPLES[type_]
    except KeyError as exc:
        raise AssertionError(
            f"no validation sample for GP type {type_.__name__}"
        ) from exc


def sample_primitive(pset, primitive, n_instruments: int = 9):
    ir_types: dict = {}
    input_specs: dict = {}
    args = [
        sample_argument(type_, ir_types, input_specs, n_instruments)
        for type_ in primitive.args
    ]
    value = pset.context[primitive.name](*args)
    return value.expr, ir_types, input_specs


def expected_output_kind(return_type) -> str:
    if issubclass(return_type, NumericTensor):
        return "matrix" if tensor_rank(return_type) == 2 else "tensor"
    return "vector"


__all__ = [
    "expected_output_kind",
    "family_primitives",
    "is_tensor_primitive",
    "sample_argument",
    "sample_primitive",
]
