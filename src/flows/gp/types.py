from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trading_dsl_engine.base.parser import Expr


@dataclass(frozen=True)
class ExprValue:
    """Value carried by a DEAP GP tree and lowered to a DSL Expr."""

    expr: Expr

    def __str__(self) -> str:
        return str(self.expr)


class NumericRow(ExprValue):
    pass


class DerivedNumericRow(NumericRow):
    """Dimension-bearing result whose exact semantic unit is no longer tracked."""


class DimensionlessRow(DerivedNumericRow):
    pass


class BoolRow(DimensionlessRow):
    """0/1 numeric row that is also valid in boolean-only positions."""


class CountRow(DimensionlessRow):
    pass


class PriceRow(NumericRow):
    pass


class QuantityRow(NumericRow):
    pass


class TimestampRow(NumericRow):
    pass


class DurationRow(NumericRow):
    pass


class TradingDayHorizonRow(NumericRow):
    pass


@dataclass(frozen=True)
class StaticValue:
    value: Any


@dataclass(frozen=True)
class PositiveNumber(StaticValue):
    """Positive compile-time numeric literal usable in scalar-broadcast slots."""

    value: int | float

    def __post_init__(self) -> None:
        if float(self.value) <= 0.0:
            raise ValueError("PositiveNumber must be > 0")


@dataclass(frozen=True)
class PositiveInt(PositiveNumber):
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool):
            raise TypeError("PositiveInt cannot be bool")
        numeric = float(self.value)
        if not numeric.is_integer() or numeric <= 0:
            raise ValueError("PositiveInt must be a positive integer")
        object.__setattr__(self, "value", int(numeric))


@dataclass(frozen=True)
class PeriodAtLeastTwo(PositiveInt):
    """Window length for operators whose native implementation requires >= 2."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.value < 2:
            raise ValueError("PeriodAtLeastTwo must be >= 2")


@dataclass(frozen=True)
class PositiveFloat(PositiveNumber):
    value: float

    def __post_init__(self) -> None:
        value = float(self.value)
        if value <= 0.0:
            raise ValueError("PositiveFloat must be > 0")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class QuantileParam(StaticValue):
    """Compile-time probability used by quantile/rank operators."""

    value: float

    def __post_init__(self) -> None:
        value = float(self.value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("QuantileParam must be in [0, 1]")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class BoolParam(StaticValue):
    value: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", bool(self.value))


@dataclass(frozen=True)
class AxisSpec(StaticValue):
    """Non-temporal reduction axes only.

    Axis 0 is the streaming/time axis in the batch view and is deliberately
    forbidden. ``None`` is also forbidden because the DSL's default reduction
    over all axes would include time.
    """

    value: int | tuple[int, ...]

    def __post_init__(self) -> None:
        raw = self.value
        axes = raw if isinstance(raw, tuple) else (raw,)
        if not axes:
            raise ValueError("AxisSpec cannot be empty")
        normalized: list[int] = []
        for axis in axes:
            if isinstance(axis, bool) or not isinstance(axis, int):
                raise TypeError("AxisSpec axes must be integers")
            if axis <= 0:
                raise ValueError(
                    "AxisSpec forbids temporal/all-axis reductions; axes must be > 0"
                )
            if axis in normalized:
                raise ValueError(f"duplicate reduction axis {axis}")
            normalized.append(axis)
        object.__setattr__(
            self,
            "value",
            normalized[0] if not isinstance(raw, tuple) else tuple(normalized),
        )


@dataclass(frozen=True)
class DatetimeUnit(StaticValue):
    value: str


@dataclass(frozen=True)
class FrequencySpec(StaticValue):
    value: str | int | float


@dataclass(frozen=True)
class FilterHSpec(StaticValue):
    value: str


@dataclass(frozen=True)
class FilterTSpec(StaticValue):
    value: str


@dataclass(frozen=True)
class KthIgnoreSpec(StaticValue):
    value: str


@dataclass(frozen=True)
class RegressionReturnSpec(StaticValue):
    value: str


VALUE_TYPES: tuple[type[NumericRow], ...] = (
    PriceRow,
    QuantityRow,
    TimestampRow,
    DurationRow,
    TradingDayHorizonRow,
    DimensionlessRow,
    BoolRow,
    CountRow,
    DerivedNumericRow,
)


def unwrap(value: Any) -> Any:
    if isinstance(value, ExprValue):
        return value.expr
    if isinstance(value, StaticValue):
        return value.value
    return value


__all__ = [
    "AxisSpec",
    "BoolParam",
    "BoolRow",
    "CountRow",
    "DatetimeUnit",
    "DerivedNumericRow",
    "DimensionlessRow",
    "DurationRow",
    "ExprValue",
    "FilterHSpec",
    "FilterTSpec",
    "FrequencySpec",
    "KthIgnoreSpec",
    "NumericRow",
    "PeriodAtLeastTwo",
    "PositiveFloat",
    "PositiveInt",
    "PositiveNumber",
    "PriceRow",
    "QuantileParam",
    "QuantityRow",
    "RegressionReturnSpec",
    "StaticValue",
    "TimestampRow",
    "TradingDayHorizonRow",
    "VALUE_TYPES",
    "unwrap",
]
