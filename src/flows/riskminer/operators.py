from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

from trading_dsl_engine.base.dsl import (
    abs as dsl_abs,
    add,
    arctan,
    clip,
    div,
    ewm,
    ewm_std,
    exp,
    fillna,
    fraction,
    gt,
    ln,
    lt,
    mul,
    purify,
    rolling_max,
    rolling_mean,
    rolling_median,
    rolling_min,
    rolling_pct_rank,
    rolling_std,
    rolling_sum,
    shift,
    sign,
    sub,
    where,
    xs_norm,
    xs_pct_rank,
    xs_rank,
)
from trading_dsl_engine.base.parser import Expr

from flows.riskminer.semantics import (
    DEFAULT_TYPE_RELATIONS,
    SemanticInfo,
    TypeRelations,
    branch_result,
    comparison_result,
    compatible_additive,
    dimensionless_result,
    numeric_product,
    numeric_ratio,
)


Builder = Callable[[tuple[Expr, ...], tuple[SemanticInfo, ...]], Expr]
Inference = Callable[[tuple[SemanticInfo, ...], TypeRelations], SemanticInfo | None]


@dataclass(frozen=True)
class OperatorSchema:
    name: str
    arity: int
    builder: Builder
    infer: Inference
    prior: float = 1.0
    family: str = "generic"
    native_parameter_mode: str = "runtime"

    def apply(
        self,
        expressions: Sequence[Expr],
        semantics: Sequence[SemanticInfo],
        relations: TypeRelations = DEFAULT_TYPE_RELATIONS,
    ) -> tuple[Expr, SemanticInfo] | None:
        expr_tuple = tuple(expressions)
        semantic_tuple = tuple(semantics)
        if len(expr_tuple) != self.arity or len(semantic_tuple) != self.arity:
            raise ValueError(f"{self.name} expects {self.arity} operands")
        output = self.infer(semantic_tuple, relations)
        if output is None:
            return None
        return self.builder(expr_tuple, semantic_tuple), output


def _same_type(values: tuple[SemanticInfo, ...], relations: TypeRelations) -> SemanticInfo | None:
    return compatible_additive(values[0], values[1], relations)


def _product(values: tuple[SemanticInfo, ...], relations: TypeRelations) -> SemanticInfo | None:
    return numeric_product(values[0], values[1], relations)


def _ratio(values: tuple[SemanticInfo, ...], relations: TypeRelations) -> SemanticInfo | None:
    return numeric_ratio(values[0], values[1], relations)


def _preserve(values: tuple[SemanticInfo, ...], relations: TypeRelations) -> SemanticInfo | None:
    del relations
    return values[0] if "numeric" in values[0].types else None


def _dimensionless(values: tuple[SemanticInfo, ...], relations: TypeRelations) -> SemanticInfo | None:
    del relations
    return dimensionless_result(values[0])


def _dimensionless_input(values: tuple[SemanticInfo, ...], relations: TypeRelations) -> SemanticInfo | None:
    if "dimensionless" not in relations.closure(values[0].types):
        return None
    return dimensionless_result(values[0])


def _comparison(values: tuple[SemanticInfo, ...], relations: TypeRelations) -> SemanticInfo | None:
    left, right = values
    if right.is_static_literal and "numeric" in left.types:
        return SemanticInfo(
            frozenset({"numeric", "boolean", "dimensionless"}),
            shape="row" if left.shape != "scalar" else "boolean",
            lower=0.0,
            upper=1.0,
            integer=True,
        )
    return comparison_result(left, right, relations)


def _where(values: tuple[SemanticInfo, ...], relations: TypeRelations) -> SemanticInfo | None:
    condition, true_value, false_value = values
    if false_value.is_static_literal and math.isnan(float(false_value.literal_value)):
        return true_value if condition.is_boolean else None
    return branch_result(condition, true_value, false_value, relations)


def _fillna(values: tuple[SemanticInfo, ...], relations: TypeRelations) -> SemanticInfo | None:
    value, replacement = values
    if replacement.is_static_literal:
        return value
    return compatible_additive(value, replacement, relations)


def _clip(values: tuple[SemanticInfo, ...], relations: TypeRelations) -> SemanticInfo | None:
    del relations
    value, lower, upper = values
    if "numeric" not in value.types:
        return None
    if not lower.is_static_literal or not upper.is_static_literal:
        return None
    if float(lower.literal_value) > float(upper.literal_value):
        return None
    return SemanticInfo(
        value.types,
        shape=value.shape,
        lower=max(value.lower, float(lower.literal_value)),
        upper=min(value.upper, float(upper.literal_value)),
        integer=value.integer,
        mode=value.mode,
        role=value.role,
    )


def _positive_literal(
    value: SemanticInfo,
    *,
    integer: bool,
    minimum: float,
    maximum: float | None = None,
) -> bool:
    if not value.is_static_literal:
        return False
    numeric = float(value.literal_value)
    if not math.isfinite(numeric) or numeric < minimum:
        return False
    if maximum is not None and numeric > maximum:
        return False
    return not integer or numeric.is_integer()


def _temporal_parameter(
    values: tuple[SemanticInfo, ...],
    relations: TypeRelations,
    *,
    integer: bool,
    minimum: float,
    maximum: float | None = None,
) -> SemanticInfo | None:
    del relations
    value, parameter = values
    if "numeric" not in value.types:
        return None
    if not _positive_literal(
        parameter,
        integer=integer,
        minimum=minimum,
        maximum=maximum,
    ):
        return None
    return value


def _ewm(values: tuple[SemanticInfo, ...], relations: TypeRelations) -> SemanticInfo | None:
    return _temporal_parameter(values, relations, integer=False, minimum=math.nextafter(0.0, 1.0))


def _lag(values: tuple[SemanticInfo, ...], relations: TypeRelations) -> SemanticInfo | None:
    return _temporal_parameter(values, relations, integer=True, minimum=0.0, maximum=4096.0)


def _window(values: tuple[SemanticInfo, ...], relations: TypeRelations) -> SemanticInfo | None:
    return _temporal_parameter(values, relations, integer=True, minimum=1.0, maximum=4096.0)


def _dimensionless_window(
    values: tuple[SemanticInfo, ...],
    relations: TypeRelations,
) -> SemanticInfo | None:
    preserved = _window(values, relations)
    return None if preserved is None else dimensionless_result(preserved)


def _literal_int(semantics: tuple[SemanticInfo, ...], index: int) -> int:
    value = semantics[index].literal_value
    if value is None:
        raise ValueError("operator parameter must be a compile-time literal")
    return int(value)


def _literal_float(semantics: tuple[SemanticInfo, ...], index: int) -> float:
    value = semantics[index].literal_value
    if value is None:
        raise ValueError("operator parameter must be a compile-time literal")
    return float(value)


def default_operator_schemas() -> tuple[OperatorSchema, ...]:
    return (
        OperatorSchema("add", 2, lambda e, s: add(e[0], e[1]), _same_type, 1.4, "arithmetic"),
        OperatorSchema("sub", 2, lambda e, s: sub(e[0], e[1]), _same_type, 1.5, "arithmetic"),
        OperatorSchema("mul", 2, lambda e, s: mul(e[0], e[1]), _product, 1.2, "arithmetic"),
        OperatorSchema("div", 2, lambda e, s: div(e[0], e[1]), _ratio, 1.5, "arithmetic"),
        OperatorSchema("abs", 1, lambda e, s: dsl_abs(e[0]), _preserve, 0.7, "unary"),
        OperatorSchema("purify", 1, lambda e, s: purify(e[0]), _preserve, 0.8, "unary"),
        OperatorSchema("fraction", 1, lambda e, s: fraction(e[0]), _dimensionless, 0.5, "unary"),
        OperatorSchema("sign", 1, lambda e, s: sign(e[0]), _dimensionless, 0.7, "unary"),
        OperatorSchema("arctan", 1, lambda e, s: arctan(e[0]), _dimensionless, 0.6, "unary"),
        OperatorSchema("ln", 1, lambda e, s: ln(e[0]), _dimensionless_input, 0.5, "unary"),
        OperatorSchema("exp", 1, lambda e, s: exp(e[0]), _dimensionless_input, 0.4, "unary"),
        OperatorSchema("xs_rank", 1, lambda e, s: xs_rank(e[0]), _dimensionless, 2.0, "cross_sectional"),
        OperatorSchema("xs_pct_rank", 1, lambda e, s: xs_pct_rank(e[0]), _dimensionless, 1.7, "cross_sectional"),
        OperatorSchema("xs_norm", 1, lambda e, s: xs_norm(e[0]), _dimensionless, 1.6, "cross_sectional"),
        OperatorSchema("ewm", 2, lambda e, s: ewm(e[0], span=_literal_float(s, 1)), _ewm, 1.7, "temporal", "compile_time_literal"),
        OperatorSchema("ewm_std", 2, lambda e, s: ewm_std(e[0], span=_literal_float(s, 1)), _ewm, 1.2, "temporal", "compile_time_literal"),
        OperatorSchema("shift", 2, lambda e, s: shift(e[0], _literal_int(s, 1), _literal_int(s, 1)), _lag, 1.2, "temporal", "compile_time_literal"),
        OperatorSchema("rolling_sum", 2, lambda e, s: rolling_sum(e[0], _literal_int(s, 1)), _window, 0.8, "rolling", "compile_time_literal"),
        OperatorSchema("rolling_mean", 2, lambda e, s: rolling_mean(e[0], _literal_int(s, 1)), _window, 1.1, "rolling", "compile_time_literal"),
        OperatorSchema("rolling_std", 2, lambda e, s: rolling_std(e[0], _literal_int(s, 1)), _window, 1.0, "rolling", "compile_time_literal"),
        OperatorSchema("rolling_min", 2, lambda e, s: rolling_min(e[0], _literal_int(s, 1)), _window, 0.5, "rolling", "compile_time_literal"),
        OperatorSchema("rolling_max", 2, lambda e, s: rolling_max(e[0], _literal_int(s, 1)), _window, 0.5, "rolling", "compile_time_literal"),
        OperatorSchema("rolling_median", 2, lambda e, s: rolling_median(e[0], _literal_int(s, 1)), _window, 0.4, "rolling", "compile_time_literal"),
        OperatorSchema("rolling_pct_rank", 2, lambda e, s: rolling_pct_rank(e[0], _literal_int(s, 1)), _dimensionless_window, 0.8, "rolling", "compile_time_literal"),
        OperatorSchema("lt", 2, lambda e, s: lt(e[0], e[1]), _comparison, 0.3, "comparison"),
        OperatorSchema("gt", 2, lambda e, s: gt(e[0], e[1]), _comparison, 0.3, "comparison"),
        OperatorSchema("where", 3, lambda e, s: where(e[0], e[1], e[2]), _where, 0.5, "conditional"),
        OperatorSchema("fillna", 2, lambda e, s: fillna(e[0], e[1]), _fillna, 0.5, "conditional"),
        OperatorSchema("clip", 3, lambda e, s: clip(e[0], e[1], e[2]), _clip, 0.5, "conditional"),
    )


def operator_inventory() -> dict[str, tuple[str, ...]]:
    searchable = tuple(schema.name for schema in default_operator_schemas())
    structured = (
        "cat", "einsum", "Ridge", "InstrumentBasisMean", "groupby",
        "grouped", "univ", "col", "outer",
    )
    evaluation_only = (
        "get_beta", "get_preds", "get_residuals", "get_r2",
        "get_standard_errors", "get_tstats",
    )
    return {
        "searchable": searchable,
        "structured": structured,
        "evaluation_only": evaluation_only,
    }


__all__ = ["OperatorSchema", "default_operator_schemas", "operator_inventory"]
