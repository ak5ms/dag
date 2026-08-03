from __future__ import annotations

from pathlib import Path

import numpy as np

from flows.alpha_search import default_alpha_pnl
from trading_dsl_engine.base.dsl import cat, ewm, var, xs_rank
from trading_dsl_engine.cpp_stream import compile_formula


def _run(tmp_path: Path, expression, x: np.ndarray) -> tuple[object, np.ndarray]:
    runtime = compile_formula(expression, {"x": x})
    result = runtime.run(out_path=tmp_path / "result.bin")
    values = np.fromfile(result.output_path, dtype=np.float64)
    return result, values.reshape(result.output_shape or ())


def test_fixed_reduction_supports_cumsum_pow_div_and_emit(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(29, 5, 4))
    pnl = var("x").sum(axis=1)
    expression = (
        pnl.cumsum() / (pnl**2).cumsum().pow(0.5)
    ).emit("last")

    result, actual = _run(tmp_path, expression, x)
    expected_pnl = np.sum(x, axis=1)
    expected = np.cumsum(expected_pnl, axis=0)[-1] / np.sqrt(
        np.cumsum(expected_pnl**2, axis=0)[-1]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert result.output_mode == "final"
    assert result.output_shape == (4,)


def test_temporal_reductions_compose_with_division_and_implicit_last(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(31, 6, 4))
    pnl = var("x").sum(axis=1)
    expression = pnl.sum(axis=[0, 1]) / pnl.std(axis=[0, 1])

    result, actual = _run(tmp_path, expression, x)
    expected_pnl = np.sum(x, axis=1)
    expected = np.sum(expected_pnl) / np.std(expected_pnl)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert result.output_mode == "final"
    assert result.output_shape == ()


def test_supplied_default_alpha_pnl_sharpe_graphs(tmp_path: Path) -> None:
    """Exercise the exact reduction topology from the alpha-search workflow."""
    rng = np.random.default_rng(123)
    rows, instruments = 96, 5
    returns_values = rng.normal(scale=0.002, size=(rows, instruments))
    tradable_values = np.ones((rows, instruments), dtype=np.float64)
    returns = var("returns")
    features = [xs_rank(ewm(returns, span)) for span in range(1, 5)]
    pnls = cat(
        *[
            default_alpha_pnl(
                feature,
                roll_rets=returns,
                is_tradable=var("is_tradable"),
                hl=4,
            )
            for feature in features
        ]
    )
    pnl = pnls.sum(axis=[1])
    data = {
        "returns": returns_values,
        "is_tradable": tradable_values,
    }

    pnl_runtime = compile_formula(pnl, data)
    pnl_result = pnl_runtime.run(out_path=tmp_path / "alpha-pnl.npy")
    pnl_values = np.asarray(pnl_result.load())
    assert pnl_values.shape == (rows, 4)

    path_sharpe = (
        pnl.cumsum() / (pnl**2).cumsum().pow(0.5)
    ).emit("last")
    path_runtime = compile_formula(path_sharpe, data)
    path_result = path_runtime.run(out_path=tmp_path / "path-sharpe.npy")
    path_actual = np.asarray(path_result.load())
    path_expected = np.nansum(pnl_values, axis=0) / np.sqrt(
        np.nansum(pnl_values**2, axis=0)
    )
    np.testing.assert_allclose(
        path_actual,
        path_expected,
        rtol=1e-11,
        atol=1e-11,
        equal_nan=True,
    )
    assert path_result.output_shape == (4,)
    assert path_result.output_mode == "final"

    reduced_sharpe = pnl.sum(axis=[0, 1]) / pnl.std(axis=[0, 1])
    reduced_runtime = compile_formula(reduced_sharpe, data)
    reduced_result = reduced_runtime.run(
        out_path=tmp_path / "reduced-sharpe.npy"
    )
    reduced_actual = np.asarray(reduced_result.load())
    reduced_expected = np.nansum(pnl_values) / np.nanstd(pnl_values)
    np.testing.assert_allclose(
        reduced_actual,
        reduced_expected,
        rtol=1e-11,
        atol=1e-11,
        equal_nan=True,
    )
    assert reduced_result.output_shape == ()
    assert reduced_result.output_mode == "final"


def test_fixed_tensor_elementwise_broadcasting_after_reduction(
    tmp_path: Path,
) -> None:
    x = np.arange(120, dtype=np.float64).reshape(5, 6, 4)
    reduced = var("x").sum(axis=1)
    expression = ((reduced + 2.0) * (reduced - 1.0)) / 3.0

    result, actual = _run(tmp_path, expression, x)
    expected_reduced = np.sum(x, axis=1)
    expected = ((expected_reduced + 2.0) * (expected_reduced - 1.0)) / 3.0

    np.testing.assert_allclose(actual, expected)
    assert result.output_mode == "rows"
    assert result.output_shape == (5, 4)


def test_default_output_is_direct_shape_aware_npy() -> None:
    x = np.arange(90, dtype=np.float64).reshape(5, 6, 3)
    runtime = compile_formula(var("x").sum(axis=1), {"x": x})
    result = runtime.run()
    try:
        actual = result.load()
        assert isinstance(actual, np.memmap)
        assert result.output_path.suffix == ".npy"
        assert actual.shape == (5, 3)
        assert actual.dtype == np.float64
        assert result.data_offset == actual.offset
        np.testing.assert_allclose(actual, np.sum(x, axis=1))
    finally:
        result.output_path.unlink(missing_ok=True)


def test_n_instruments_uses_dominant_leading_row_extent(tmp_path: Path) -> None:
    rows = 11
    x = np.arange(rows * 6 * 4, dtype=np.float64).reshape(rows, 6, 4)
    y = np.ones((rows, 6), dtype=np.float64)
    fixed = np.arange(rows * 4, dtype=np.float64).reshape(rows, 4)
    fixed_row_scalar = var("fixed").sum(axis=1)
    expression = var("x").sum(axis=2) + var("y") + fixed_row_scalar

    runtime = compile_formula(
        expression,
        {"x": x, "y": y, "fixed": fixed},
    )
    assert runtime.n_instruments == 6
    result = runtime.run(out_path=tmp_path / "automatic-n.npy")
    actual = result.load()
    expected = np.sum(x, axis=2) + y + np.sum(fixed, axis=1)[:, None]
    np.testing.assert_allclose(actual, expected)
