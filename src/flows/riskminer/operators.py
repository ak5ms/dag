from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

from trading_dsl_engine.base.dsl import (
    abs as dsl_abs,
    add,
    arctan,
    div,
    ewm,
    fillna,
    fraction,
    maximum,
    minimum,
    mul,
    purify,
    rolling_mean,
    rolling_std,
    shift,
    sign,
    sub,
    where,
    xs_pct_rank,
    xs_rank,
)
from trading_dsl_engine.base.parser import Expr

from .semantics import (
    SearchShape,
    SemanticInfo,
    broadcast_shape,
    common_output,
    compatible,
    dimensionless_output,
    division_output,
    multiplication_output,
    unary_preserve,
)


Validator = Callable[[Sequence[SemanticInfo]], bool]
Infer = Callable[[Sequence[SemanticInfo]], SemanticInfo]
Builder = Callable[[Sequence[Expr], Sequence[float | None]], Expr]


@dataclass(frozen=True)
class OperatorSchema:
    name: str
    arity: int
    validate: Validator
    infer: Infer
    build: Builder
    prior: float = 1.0
    commutative: bool = False
    family: str = "generic"


def _numeric(info: SemanticInfo) -> bool:
    return "numeric" in info.types


def _all_numeric(args: Sequence[SemanticInfo]) -> bool:
    return all(_numeric(arg) for arg in args)


def _unary_numeric(args: Sequence[SemanticInfo]) -> bool:
    return len(args) == 1 and _numeric(args[0]) and args[0].shape in {
        SearchShape.ROW,
        SearchShape.SCALAR,
    }


def _cross_sectional_row(args: Sequence[SemanticInfo]) -> bool:
    """Cross-sectional transforms require an actual instrument row.

    In particular, ranking a compile-time scalar is not a meaningful alpha and
    produced degenerate formulas such as ``xs_pct_rank(60)`` in the first deep
    run. The shape rule removes those actions before rollout rather than adding
    a feature/operator blacklist.
    """

    return (
        len(args) == 1
        and _numeric(args[0])
        and args[0].shape is SearchShape.ROW
    )


def _same_type(args: Sequence[SemanticInfo]) -> bool:
    return (
        len(args) == 2
        and _all_numeric(args)
        and compatible(args[0], args[1])
    )


def _broadcast_numeric(args: Sequence[SemanticInfo]) -> bool:
    return (
        len(args) == 2
        and _all_numeric(args)
        and broadcast_shape(args[0].shape, args[1].shape) is not None
    )


def _where(args: Sequence[SemanticInfo]) -> bool:
    return (
        len(args) == 3
        and args[0].boolean
        and compatible(args[1], args[2])
        and broadcast_shape(args[0].shape, args[1].shape) is not None
    )


def _fillna(args: Sequence[SemanticInfo]) -> bool:
    return len(args) == 2 and compatible(args[0], args[1])


def _positive_static_parameter(
    info: SemanticInfo,
    *,
    integer: bool = False,
    minimum: float = 0.0,
    maximum_value: float = math.inf,
) -> bool:
    return (
        info.shape is SearchShape.SCALAR
        and info.static
        and info.lower > minimum
        and info.upper <= maximum_value
        and (not integer or info.integer)
    )


def _temporal(args: Sequence[SemanticInfo]) -> bool:
    return (
        len(args) == 2
        and _numeric(args[0])
        and args[0].shape is SearchShape.ROW
        and _positive_static_parameter(args[1])
    )


def _history(args: Sequence[SemanticInfo]) -> bool:
    return (
        len(args) == 2
        and _numeric(args[0])
        and args[0].shape is SearchShape.ROW
        and _positive_static_parameter(args[1], integer=True, maximum_value=4096)
    )


def _rolling(args: Sequence[SemanticInfo]) -> bool:
    return (
        len(args) == 2
        and _numeric(args[0])
        and args[0].shape is SearchShape.ROW
        and _positive_static_parameter(args[1], integer=True, maximum_value=4096)
    )


def _same_infer(args: Sequence[SemanticInfo]) -> SemanticInfo:
    return common_output(args[0], args[1])


def _preserve_first(args: Sequence[SemanticInfo]) -> SemanticInfo:
    value = unary_preserve(args[0])
    return SemanticInfo(
        value.types,
        value.shape,
        -math.inf,
        math.inf,
        integer=False,
        static=False,
        role="value",
    )


def _dimensionless(args: Sequence[SemanticInfo]) -> SemanticInfo:
    return dimensionless_output(args[0])


def _where_infer(args: Sequence[SemanticInfo]) -> SemanticInfo:
    return common_output(args[1], args[2])


def _literal_at(values: Sequence[float | None], index: int, name: str) -> float:
    value = values[index]
    if value is None:
        raise ValueError(f"{name} requires a compile-time literal")
    return float(value)


def default_operator_catalog() -> tuple[OperatorSchema, ...]:
    """Conservative catalog whose entries are known to lower to cpp_stream."""

    unary_preserving = (
        ("abs", dsl_abs, 0.8),
        ("purify", purify, 0.7),
    )
    ordinary_dimensionless = (
        ("sign", sign, 0.8),
        ("fraction", fraction, 0.35),
        ("arctan", arctan, 0.55),
    )
    cross_sectional_dimensionless = (
        ("xs_rank", xs_rank, 1.8),
        ("xs_pct_rank", xs_pct_rank, 1.6),
    )
    out: list[OperatorSchema] = []

    for name, fn, prior in unary_preserving:
        out.append(
            OperatorSchema(
                name,
                1,
                _unary_numeric,
                _preserve_first,
                lambda exprs, literals, fn=fn: fn(exprs[0]),
                prior=prior,
                family="unary_preserving",
            )
        )

    for name, fn, prior in ordinary_dimensionless:
        out.append(
            OperatorSchema(
                name,
                1,
                _unary_numeric,
                _dimensionless,
                lambda exprs, literals, fn=fn: fn(exprs[0]),
                prior=prior,
                family="normalization",
            )
        )

    for name, fn, prior in cross_sectional_dimensionless:
        out.append(
            OperatorSchema(
                name,
                1,
                _cross_sectional_row,
                _dimensionless,
                lambda exprs, literals, fn=fn: fn(exprs[0]),
                prior=prior,
                family="cross_sectional",
            )
        )

    out.extend(
        (
            OperatorSchema(
                "add",
                2,
                _same_type,
                _same_infer,
                lambda exprs, literals: add(exprs[0], exprs[1]),
                prior=1.25,
                commutative=True,
                family="compatible_binary",
            ),
            OperatorSchema(
                "sub",
                2,
                _same_type,
                _same_infer,
                lambda exprs, literals: sub(exprs[0], exprs[1]),
                prior=1.35,
                family="compatible_binary",
            ),
            OperatorSchema(
                "minimum",
                2,
                _same_type,
                _same_infer,
                lambda exprs, literals: minimum(exprs[0], exprs[1]),
                prior=0.5,
                commutative=True,
                family="compatible_binary",
            ),
            OperatorSchema(
                "maximum",
                2,
                _same_type,
                _same_infer,
                lambda exprs, literals: maximum(exprs[0], exprs[1]),
                prior=0.5,
                commutative=True,
                family="compatible_binary",
            ),
            OperatorSchema(
                "mul",
                2,
                _broadcast_numeric,
                lambda args: multiplication_output(args[0], args[1]),
                lambda exprs, literals: mul(exprs[0], exprs[1]),
                prior=1.05,
                commutative=True,
                family="numeric_binary",
            ),
            OperatorSchema(
                "div",
                2,
                _broadcast_numeric,
                lambda args: division_output(args[0], args[1]),
                lambda exprs, literals: div(exprs[0], exprs[1]),
                prior=1.3,
                family="numeric_binary",
            ),
            OperatorSchema(
                "fillna",
                2,
                _fillna,
                _same_infer,
                lambda exprs, literals: fillna(exprs[0], exprs[1]),
                prior=0.35,
                family="compatible_binary",
            ),
            OperatorSchema(
                "where",
                3,
                _where,
                _where_infer,
                lambda exprs, literals: where(exprs[0], exprs[1], exprs[2]),
                prior=0.35,
                family="conditional",
            ),
            OperatorSchema(
                "ewm",
                2,
                _temporal,
                _preserve_first,
                lambda exprs, literals: ewm(
                    exprs[0], span=_literal_at(literals, 1, "ewm")
                ),
                prior=1.45,
                family="temporal",
            ),
            OperatorSchema(
                "shift",
                2,
                _history,
                _preserve_first,
                lambda exprs, literals: shift(
                    exprs[0],
                    int(_literal_at(literals, 1, "shift")),
                    int(_literal_at(literals, 1, "shift")),
                ),
                prior=1.05,
                family="history",
            ),
            OperatorSchema(
                "rolling_mean",
                2,
                _rolling,
                _preserve_first,
                lambda exprs, literals: rolling_mean(
                    exprs[0],
                    int(_literal_at(literals, 1, "rolling_mean")),
                ),
                prior=0.9,
                family="rolling",
            ),
            OperatorSchema(
                "rolling_std",
                2,
                _rolling,
                _preserve_first,
                lambda exprs, literals: rolling_std(
                    exprs[0],
                    int(_literal_at(literals, 1, "rolling_std")),
                    ddof=0,
                ),
                prior=0.8,
                family="rolling",
            ),
        )
    )
    return tuple(out)


def catalog_by_name(
    schemas: Sequence[OperatorSchema] | None = None,
) -> dict[str, OperatorSchema]:
    values = tuple(default_operator_catalog() if schemas is None else schemas)
    out: dict[str, OperatorSchema] = {}
    for schema in values:
        if schema.name in out:
            raise ValueError(f"duplicate operator schema {schema.name!r}")
        out[schema.name] = schema
    return out


__all__ = [
    "OperatorSchema",
    "catalog_by_name",
    "default_operator_catalog",
]
