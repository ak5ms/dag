from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trading_dsl_engine.base.dsl import cat, cumsum, var
from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.ir.frontend import FormulaIRCompileError


def _run(tmp_path: Path, expression, data: dict[str, np.ndarray]):
    runtime = compile_formula(expression, data, n_instruments=data["x"].shape[1])
    result = runtime.run(out_path=tmp_path / "out.bin")
    values = np.fromfile(result.output_path, dtype=np.float64)
    return runtime, result, values.reshape(result.output_shape or ())


def test_temporal_sum_writes_one_final_vector(tmp_path: Path) -> None:
    x = np.arange(60, dtype=np.float64).reshape(10, 6)
    x[3, 2] = np.nan
    runtime, result, actual = _run(tmp_path, var("x").sum(axis=0), {"x": x})
    np.testing.assert_allclose(actual, np.nansum(x, axis=0))
    assert runtime.plan.output_mode == "final"
    assert result.rows == 10
    assert result.output_rows == 1
    assert result.output_shape == (6,)
    assert result.output_path.stat().st_size == 6 * 8


def test_row_reductions_compose_and_keep_row_emission(tmp_path: Path) -> None:
    x = np.arange(48, dtype=np.float64).reshape(8, 6)
    expression = var("x").sum(axis=1) + 2.0
    runtime, result, actual = _run(tmp_path, expression, {"x": x})
    np.testing.assert_allclose(actual, np.nansum(x, axis=1) + 2.0)
    assert runtime.plan.output_mode == "rows"
    assert result.output_shape == (8,)


def test_mixed_axis_mean_and_std_stream_without_materializing_time(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(37, 5))
    y = rng.normal(size=(37, 5))
    x[4, 1] = np.nan
    features = cat(var("x"), var("y"))

    _, mean_result, mean_value = _run(
        tmp_path, features.mean(axis=[0, 1]), {"x": x, "y": y}
    )
    expected = np.nanmean(np.stack((x, y), axis=-1), axis=(0, 1))
    np.testing.assert_allclose(mean_value, expected, rtol=1e-13, atol=1e-13)
    assert mean_result.output_shape == (2,)

    _, std_result, std_value = _run(
        tmp_path, features.std(axis=[0, 1], ddof=1), {"x": x, "y": y}
    )
    expected_std = np.nanstd(
        np.stack((x, y), axis=-1), axis=(0, 1), ddof=1
    )
    np.testing.assert_allclose(std_value, expected_std, rtol=1e-12, atol=1e-12)
    assert std_result.output_shape == (2,)


def test_reduction_axis_uses_full_materialized_rank_and_negative_axes(tmp_path: Path) -> None:
    x = np.arange(36, dtype=np.float64).reshape(6, 6)
    _, result, actual = _run(tmp_path, var("x").mean(axis=-1), {"x": x})
    np.testing.assert_allclose(actual, np.mean(x, axis=1))
    assert result.output_shape == (6,)


def test_emit_last_reuses_streaming_state_without_row_output(tmp_path: Path) -> None:
    x = np.arange(42, dtype=np.float64).reshape(7, 6)
    expression = cumsum(var("x")).emit("last")
    _, result, actual = _run(tmp_path, expression, {"x": x})
    np.testing.assert_allclose(actual, np.cumsum(x, axis=0)[-1])
    assert result.output_mode == "final"
    assert result.output_path.stat().st_size == 6 * 8


def test_row_then_temporal_reduction_composes(tmp_path: Path) -> None:
    x = np.arange(60, dtype=np.float64).reshape(10, 6)
    expression = (var("x") + 1.0).sum(axis=1).mean(axis=0)
    _, result, actual = _run(tmp_path, expression, {"x": x})
    np.testing.assert_allclose(actual, np.mean(np.sum(x + 1.0, axis=1)))
    assert result.output_shape == ()
    assert result.output_path.stat().st_size == 8


def test_temporal_reduction_composes_but_emit_remains_terminal(tmp_path: Path) -> None:
    x_values = np.arange(12, dtype=np.float64).reshape(4, 3)
    x = var("x")

    _, result, actual = _run(tmp_path, x.sum(axis=0) + 1.0, {"x": x_values})
    np.testing.assert_allclose(actual, np.sum(x_values, axis=0) + 1.0)
    assert result.output_mode == "final"

    with pytest.raises(FormulaIRCompileError, match="terminal output"):
        compile_formula(x.emit("last") + 1.0, {"x": x_values}, n_instruments=3)


def test_string_formula_accepts_list_axes(tmp_path: Path) -> None:
    x = np.arange(24, dtype=np.float64).reshape(4, 6)
    runtime = compile_formula("sum(x, axis=[0, 1])", {"x": x}, n_instruments=6)
    result = runtime.run(out_path=tmp_path / "scalar.bin")
    actual = np.fromfile(result.output_path, dtype=np.float64)
    np.testing.assert_allclose(actual, [np.sum(x)])
    assert result.output_shape == ()


@pytest.mark.parametrize("kind", ["sum", "mean", "std"])
def test_method_reduction_without_axis_defaults_to_all_axes(
    tmp_path: Path, kind: str
) -> None:
    x = np.arange(120, dtype=np.float64).reshape(5, 6, 4)
    expression = getattr(var("x"), kind)()
    _, result, actual = _run(tmp_path, expression, {"x": x})
    expected = getattr(np, kind)(x)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert result.output_mode == "final"
    assert result.output_shape == ()
    assert result.output_path.stat().st_size == np.dtype(np.float64).itemsize


@pytest.mark.parametrize("kind", ["sum", "mean", "std"])
def test_string_reduction_without_axis_defaults_to_all_axes(
    tmp_path: Path, kind: str
) -> None:
    x = np.arange(72, dtype=np.float64).reshape(4, 6, 3)
    runtime = compile_formula(f"{kind}(x)", {"x": x}, n_instruments=6)
    result = runtime.run(out_path=tmp_path / f"{kind}-all.bin")
    actual = np.fromfile(result.output_path, dtype=np.float64)

    np.testing.assert_allclose(actual, [getattr(np, kind)(x)])
    assert result.output_mode == "final"
    assert result.output_shape == ()


@pytest.mark.parametrize(
    ("kind", "kwargs"),
    [
        ("sum", {}),
        ("mean", {}),
        ("std", {"ddof": 1}),
    ],
)
def test_ignore_na_false_propagates_missing_values(
    tmp_path: Path, kind: str, kwargs: dict[str, int]
) -> None:
    x = np.arange(24, dtype=np.float64).reshape(8, 3)
    x[2, 1] = np.nan
    expression = getattr(var("x"), kind)(
        axis=0, ignore_na=False, **kwargs
    )
    _, _, actual = _run(tmp_path, expression, {"x": x})
    expected = getattr(np, kind)(x, axis=0, **kwargs)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert np.isfinite(actual[[0, 2]]).all()
    assert np.isnan(actual[1])


def test_string_reduction_ignore_na_defaults_true(tmp_path: Path) -> None:
    x = np.arange(24, dtype=np.float64).reshape(4, 6)
    x[1, 3] = np.nan
    runtime = compile_formula("sum(x, axis=0)", {"x": x}, n_instruments=6)
    result = runtime.run(out_path=tmp_path / "default-ignore-na.bin")
    actual = np.fromfile(result.output_path, dtype=np.float64)
    np.testing.assert_allclose(actual, np.nansum(x, axis=0))
