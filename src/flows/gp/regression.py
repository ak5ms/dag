from __future__ import annotations

from collections.abc import Sequence

from flows.gp.wrappers import broadcast_like
from trading_dsl_engine.base import dsl
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.cpp_stream.python import utils as cpp_stream_utils


REGRESSION_PROJECTIONS: tuple[str, ...] = (
    "residual",
    "prediction",
    "intercept",
    "beta",
    "sse",
    "sst",
    "r2",
    "residual_variance",
    "intercept_stderr",
    "beta_stderr",
    "intercept_tstat",
    "beta_tstat",
    "effective_df",
    "effective_n",
)

_VECTOR_PROJECTIONS = frozenset({"residual", "prediction"})


def _broadcast_projection(like: Expr, projection: Expr, name: str) -> Expr:
    if name in _VECTOR_PROJECTIONS:
        return projection
    return broadcast_like(like, projection)


def temporal_ridge_projection(name: str, y: Expr, x: Expr, periods: int) -> Expr:
    projection = cpp_stream_utils.ts_regression(
        y,
        x,
        periods,
        lag=0,
        rettype=name,
        weights=1.0,
        lambda_=0.0,
    )
    return _broadcast_projection(y, projection, name)


def temporal_poly_regression_residual(
    y: Expr,
    x: Expr,
    periods: int,
    degree: int,
) -> Expr:
    return cpp_stream_utils.ts_poly_regression(
        y,
        x,
        periods,
        k=degree,
        weights=1.0,
        lambda_=0.0,
    )


def xs_regression_neutralize(y: Expr, x: Expr) -> Expr:
    return cpp_stream_utils.xs_regression_neut(y, x)


def _rowwise_projection(model: Expr, name: str) -> Expr:
    projections = {
        "residual": dsl.get_residuals,
        "prediction": dsl.get_preds,
        "intercept": lambda value: dsl.get_coefficient(value, 0),
        "beta": lambda value: dsl.get_coefficient(value, 1),
        "sse": dsl.get_sse,
        "sst": dsl.get_sst,
        "r2": dsl.get_r2,
        "residual_variance": dsl.get_residual_variance,
        "intercept_stderr": lambda value: dsl.get_standard_error(value, 0),
        "beta_stderr": lambda value: dsl.get_standard_error(value, 1),
        "intercept_tstat": lambda value: dsl.get_tstat(value, 0),
        "beta_tstat": lambda value: dsl.get_tstat(value, 1),
        "effective_df": dsl.get_effective_df,
        "effective_n": dsl.get_effective_n,
    }
    try:
        return projections[name](model)
    except KeyError as exc:
        raise ValueError(f"unknown regression projection {name!r}") from exc


def rowwise_ridge_projection(
    name: str,
    y: Expr,
    regressors: Sequence[Expr],
) -> Expr:
    values = tuple(regressors)
    if not values:
        raise ValueError("rowwise Ridge requires at least one regressor")
    model = dsl.Ridge(
        1.0,
        *values,
        y=y,
        weights=1.0,
        hl=0.0,
        lambda_=0.0,
        nonneg=False,
    )
    projection = _rowwise_projection(model, name)
    return _broadcast_projection(y, projection, name)


__all__ = [
    "REGRESSION_PROJECTIONS",
    "rowwise_ridge_projection",
    "temporal_poly_regression_residual",
    "temporal_ridge_projection",
    "xs_regression_neutralize",
]
