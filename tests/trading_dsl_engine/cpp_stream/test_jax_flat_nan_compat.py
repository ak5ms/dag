from __future__ import annotations

from pathlib import Path

import numpy as np

from trading_dsl_engine.cpp_stream import compile_formula as compile_cpp_stream
from trading_dsl_engine.jax_flat import engine as jax_flat_engine


def _run_cpp_stream(
    tmp_path: Path,
    formula: str,
    data: dict[str, np.ndarray],
) -> np.ndarray:
    n_instruments = next(iter(data.values())).shape[1]
    runtime = compile_cpp_stream(
        formula,
        data,
        n_instruments=n_instruments,
    )
    result = runtime.run(out_path=tmp_path / "cpp-stream.bin")
    return np.fromfile(result.output_path, dtype=np.float64).reshape(
        result.output_shape
    )


def _run_jax_flat(
    formula: str,
    data: dict[str, np.ndarray],
) -> np.ndarray:
    runtime = jax_flat_engine.compile_formula(formula, cpp=False)
    _, result = runtime.run_batch(data)
    return np.asarray(result)


def test_stateless_nan_semantics_match_jax_flat(tmp_path: Path) -> None:
    x = np.array(
        [
            [1.0, np.nan, 3.0, 3.0, -2.0, 7.0, 4.0, 0.0, 5.0],
            [np.nan, np.nan, 2.0, 2.0, 4.0, -1.0, 8.0, 8.0, 0.0],
            [5.0, 1.0, np.nan, -4.0, 2.0, 2.0, 9.0, np.nan, -1.0],
        ],
        dtype=np.float64,
    )
    y = np.array(
        [
            [0.5, 4.0, np.nan, 1.0, -2.0, 2.0, 1.0, np.nan, 5.0],
            [1.0, np.nan, 2.0, 3.0, np.nan, -1.0, 1.0, 7.0, 0.0],
            [5.0, np.nan, 4.0, -4.0, 3.0, 2.0, np.nan, 0.0, -2.0],
        ],
        dtype=np.float64,
    )
    data = {"x": x, "y": y}
    formula = (
        "cat("
        "xs_rank((x + 5 + y) * 3), "
        "x == y, x != y, x < y, x > y, x <= y, x >= y, "
        "x & y, x | y, x ^ y, "
        "fillna(x, y), where(x, y, 0)"
        ")"
    )

    actual = _run_cpp_stream(tmp_path, formula, data)
    expected = _run_jax_flat(formula, data)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-9,
        atol=2e-9,
        equal_nan=True,
    )
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))


def test_default_row_mean_nan_semantics_match_jax_flat(tmp_path: Path) -> None:
    x = np.array(
        [
            [1.0, np.nan, 3.0, 5.0],
            [np.nan, np.nan, np.nan, np.nan],
            [-1.0, 2.0, np.nan, 7.0],
        ],
        dtype=np.float64,
    )
    data = {"x": x}

    actual = _run_cpp_stream(tmp_path, "mean(x, axis=1)", data)
    expected = _run_jax_flat("mean(x)", data)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
