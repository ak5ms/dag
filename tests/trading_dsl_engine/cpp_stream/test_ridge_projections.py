from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.ir import FormulaIRCompileError, compile_ir
from trading_dsl_engine.ir.ops import RidgeProjectionOp, ShiftOp


def _ridge_reference(
    features: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    ridge_lambda: float,
    *,
    weight_square: float | None = None,
) -> dict[str, np.ndarray | float]:
    xx = features.T @ (weights[:, None] * features)
    xy = features.T @ (weights * y)
    system = xx + ridge_lambda * np.diag(np.diag(xx))
    inverse = np.linalg.inv(system)
    beta = inverse @ xy
    residuals = y - features @ beta
    sse = float(np.sum(weights * residuals * residuals))
    weighted_mean = float(np.sum(weights * y) / np.sum(weights))
    sst = float(np.sum(weights * (y - weighted_mean) ** 2))
    q = float(np.sum(weights * weights) if weight_square is None else weight_square)
    effective_n = float(np.sum(weights) ** 2 / q)
    hat_core = inverse @ xx
    effective_df = float(np.trace(hat_core))
    residual_df = effective_n - 2.0 * effective_df + float(
        np.trace(hat_core @ hat_core)
    )
    residual_variance = sse / residual_df if residual_df > 0.0 else np.nan
    covariance = residual_variance * inverse @ xx @ inverse.T
    standard_errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    return {
        "beta": beta,
        "residuals": residuals,
        "sse": sse,
        "sst": sst,
        "r2": 1.0 - sse / sst,
        "residual_variance": residual_variance,
        "effective_df": effective_df,
        "effective_n": effective_n,
        "standard_errors": standard_errors,
        "tstats": beta / standard_errors,
    }


def _row_references(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    ridge_lambda: float,
) -> list[dict[str, np.ndarray | float]]:
    return [
        _ridge_reference(
            np.column_stack((np.ones(x1.shape[1]), x1[row], x2[row])),
            y[row],
            weights[row],
            ridge_lambda,
        )
        for row in range(x1.shape[0])
    ]


def test_weighted_ridge_inference_uses_penalized_covariance(tmp_path: Path) -> None:
    rng = np.random.default_rng(6102)
    rows, lanes = 9, 12
    x1 = rng.normal(size=(rows, lanes))
    x2 = rng.normal(size=(rows, lanes))
    y = 0.35 + 0.8 * x1 - 0.45 * x2 + rng.normal(scale=0.25, size=(rows, lanes))
    weights = rng.uniform(0.2, 2.5, size=(rows, lanes))
    ridge_lambda = 0.35
    data = {"x1": x1, "x2": x2, "y": y, "weights": weights}
    model = (
        "Ridge(cat(1.0, x1, x2), y=y, weights=weights, hl=0, lambda_=0.35)"
    )
    references = _row_references(x1, x2, y, weights, ridge_lambda)

    runtime = compile_formula(
        f"get_standard_errors({model})", data, n_instruments=lanes
    )
    output = tmp_path / "standard_errors.bin"
    runtime.run(out_path=output)
    actual_se = np.fromfile(output, dtype=np.float64).reshape(rows, 3)
    expected_se = np.stack([item["standard_errors"] for item in references])
    np.testing.assert_allclose(actual_se, expected_se, rtol=3e-9, atol=3e-9)

    metric_names = (
        "sse",
        "sst",
        "r2",
        "residual_variance",
        "effective_df",
        "effective_n",
    )
    metric_formula = "cat(" + ",".join(
        f"get_{name}({model})" for name in metric_names
    ) + ")"
    runtime = compile_formula(metric_formula, data, n_instruments=lanes)
    bundles = [
        stage for stage in runtime.plan.stages if stage.kind == "ridge_bundle"
    ]
    assert len(bundles) == 1
    assert len(bundles[0].members) == len(metric_names)
    output = tmp_path / "metrics.bin"
    runtime.run(out_path=output)
    actual_metrics = np.fromfile(output, dtype=np.float64).reshape(
        rows, lanes, len(metric_names)
    )
    expected_metrics = np.array(
        [[float(item[name]) for name in metric_names] for item in references]
    )
    np.testing.assert_allclose(
        actual_metrics[:, 0], expected_metrics, rtol=4e-9, atol=4e-9
    )
    np.testing.assert_allclose(
        actual_metrics,
        np.repeat(expected_metrics[:, None, :], lanes, axis=1),
        rtol=4e-9,
        atol=4e-9,
    )

    runtime = compile_formula(f"get_residuals({model})", data, n_instruments=lanes)
    output = tmp_path / "residuals.bin"
    runtime.run(out_path=output)
    actual_residuals = np.fromfile(output, dtype=np.float64).reshape(rows, lanes)
    expected_residuals = np.stack([item["residuals"] for item in references])
    np.testing.assert_allclose(
        actual_residuals, expected_residuals, rtol=3e-9, atol=3e-9
    )


def _stateful_ridge_references(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    halflife: float,
    ridge_lambda: float,
) -> list[dict[str, np.ndarray | float]]:
    alpha = 1.0 - np.exp(np.log(0.5) / halflife)
    old_factor = 1.0 - alpha
    state_xx: np.ndarray | None = None
    state_xy: np.ndarray | None = None
    state_ywy = state_wy = state_weight = state_weight_square = 0.0
    references: list[dict[str, np.ndarray | float]] = []
    for row in range(x.shape[0]):
        features = np.column_stack((np.ones(x.shape[1]), x[row]))
        row_xx = features.T @ (weights[row, :, None] * features)
        row_xy = features.T @ (weights[row] * y[row])
        row_ywy = float(np.sum(weights[row] * y[row] * y[row]))
        row_wy = float(np.sum(weights[row] * y[row]))
        row_weight = float(np.sum(weights[row]))
        row_weight_square = float(np.sum(weights[row] * weights[row]))
        if state_xx is None:
            state_xx = row_xx
            state_xy = row_xy
            state_ywy = row_ywy
            state_wy = row_wy
            state_weight = row_weight
            state_weight_square = row_weight_square
        else:
            state_xx += alpha * (row_xx - state_xx)
            state_xy += alpha * (row_xy - state_xy)
            state_ywy += alpha * (row_ywy - state_ywy)
            state_wy += alpha * (row_wy - state_wy)
            state_weight += alpha * (row_weight - state_weight)
            state_weight_square = (
                old_factor * old_factor * state_weight_square
                + alpha * alpha * row_weight_square
            )
        assert state_xx is not None and state_xy is not None
        system = state_xx + ridge_lambda * np.diag(np.diag(state_xx))
        inverse = np.linalg.inv(system)
        beta = inverse @ state_xy
        sse = float(
            state_ywy - 2.0 * beta @ state_xy + beta @ state_xx @ beta
        )
        sst = float(state_ywy - state_wy * state_wy / state_weight)
        effective_n = state_weight * state_weight / state_weight_square
        hat_core = inverse @ state_xx
        effective_df = float(np.trace(hat_core))
        residual_df = effective_n - 2.0 * effective_df + float(
            np.trace(hat_core @ hat_core)
        )
        residual_variance = max(0.0, sse) / residual_df if residual_df > 0.0 else np.nan
        covariance = residual_variance * inverse @ state_xx @ inverse.T
        standard_errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
        references.append(
            {
                "r2": 1.0 - max(0.0, sse) / sst,
                "standard_errors": standard_errors,
            }
        )
    return references


def test_stateful_weighted_ridge_standard_errors_and_named_regression(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(42)
    rows, lanes = 26, 9
    x = rng.normal(size=(rows, lanes))
    y = 0.5 + 1.15 * x + rng.normal(scale=0.3, size=(rows, lanes))
    weights = rng.uniform(0.3, 1.8, size=(rows, lanes))
    data = {"x": x, "y": y, "weights": weights}
    references = _stateful_ridge_references(
        x,
        y,
        weights,
        halflife=4.0,
        ridge_lambda=0.2,
    )
    span_references = _stateful_ridge_references(
        x,
        y,
        weights,
        halflife=np.log(0.5) / np.log1p(-2.0 / (4.0 + 1.0)),
        ridge_lambda=0.2,
    )

    model = "Ridge(1.0, x, y=y, weights=weights, hl=4, lambda_=0.2)"
    runtime = compile_formula(
        f"get_standard_errors({model})", data, n_instruments=lanes
    )
    output = tmp_path / "stateful_se.bin"
    runtime.run(out_path=output)
    actual_se = np.fromfile(output, dtype=np.float64).reshape(rows, 2)
    expected_se = np.stack([item["standard_errors"] for item in references])
    np.testing.assert_allclose(
        actual_se, expected_se, rtol=6e-9, atol=6e-9, equal_nan=True
    )

    runtime = compile_formula(
        'ts_regression(y, x, periods=4, rettype="r2", '
        "weights=weights, lambda_=0.2)",
        data,
        n_instruments=lanes,
    )
    output = tmp_path / "ts_regression_r2.bin"
    runtime.run(out_path=output)
    actual_r2 = np.fromfile(output, dtype=np.float64).reshape(rows)
    expected_r2 = np.array([float(item["r2"]) for item in span_references])
    np.testing.assert_allclose(
        actual_r2, expected_r2, rtol=6e-9, atol=6e-9, equal_nan=True
    )


@pytest.mark.parametrize(
    ("rettype", "field", "component"),
    [
        ("residual", "residuals", None),
        ("prediction", "preds", None),
        ("intercept", "coefficient", 0),
        ("beta", "coefficient", 1),
        ("sse", "sse", None),
        ("sst", "sst", None),
        ("r2", "r2", None),
        ("residual_variance", "residual_variance", None),
        ("intercept_stderr", "standard_error", 0),
        ("beta_stderr", "standard_error", 1),
        ("intercept_tstat", "tstat", 0),
        ("beta_tstat", "tstat", 1),
        ("effective_df", "effective_df", None),
        ("effective_n", "effective_n", None),
    ],
)
def test_ts_regression_rettype_names_are_semantic(
    rettype: str,
    field: str,
    component: int | None,
) -> None:
    program = compile_ir(
        f'ts_regression(y, x, periods=8, rettype="{rettype}", lambda_=0.1)'
    )
    op = program.nodes[program.output_id].op
    assert isinstance(op, RidgeProjectionOp)
    assert (op.field, op.component) == (field, component)


def test_ts_regression_rejects_numeric_rettype() -> None:
    with pytest.raises(FormulaIRCompileError, match="descriptive name"):
        compile_ir("ts_regression(y, x, periods=8, rettype=2)")


def test_ts_regression_validates_periods_and_elides_zero_lag() -> None:
    program = compile_ir('ts_regression(y, x, periods=8, rettype="beta")')
    assert not any(isinstance(node.op, ShiftOp) for node in program.nodes)

    lagged = compile_ir(
        'ts_regression(y, x, periods=8, lag=2, rettype="beta")'
    )
    assert any(isinstance(node.op, ShiftOp) for node in lagged.nodes)

    with pytest.raises(FormulaIRCompileError, match="periods must be finite and >= 1"):
        compile_ir('ts_regression(y, x, periods=0, rettype="beta")')


def test_ridge_projection_getters_accept_semantic_keywords() -> None:
    program = compile_ir(
        "get_standard_error("
        "model=Ridge(cat(x1, x2), y=y, hl=8, lambda_=0.1), component=1)"
    )
    op = program.nodes[program.output_id].op
    assert isinstance(op, RidgeProjectionOp)
    assert (op.field, op.component) == ("standard_error", 1)

    program = compile_ir(
        "get_r2(model=Ridge(cat(x1, x2), y=y, hl=8, lambda_=0.1))"
    )
    op = program.nodes[program.output_id].op
    assert isinstance(op, RidgeProjectionOp)
    assert (op.field, op.component) == ("r2", None)
