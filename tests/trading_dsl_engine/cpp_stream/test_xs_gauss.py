from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import numpy as np

import trading_dsl_engine.cpp_stream as cpp_stream
from trading_dsl_engine.base.dsl import DEFAULT_DSL_REGISTRY
from trading_dsl_engine.cpp_stream import compile_formula


_NORMAL = NormalDist()


def _bounded_location(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    rms = float(np.sqrt(np.mean(values * values)))
    if not rms > 0.0 or not np.isfinite(rms):
        return 0.0
    return float(np.clip(mean / rms, -1.0, 1.0))


def _reference(row: np.ndarray) -> np.ndarray:
    """Spacing-derived Gaussian shape plus bounded mean/RMS location."""

    row = np.asarray(row, dtype=np.float64)
    result = np.full_like(row, np.nan)
    valid = np.isfinite(row)
    values = row[valid]
    if values.size == 0:
        return result

    location = _bounded_location(values)
    spread = float(np.max(values) - np.min(values))
    if values.size < 2 or not spread > 0.0:
        result[valid] = location
        return result

    count = float(values.size)
    probabilities = (
        1.0 + (count - 1.0) * (values - np.min(values)) / spread
    ) / (count + 1.0)
    raw = np.array(
        [_NORMAL.inv_cdf(float(probability)) for probability in probabilities]
    )
    raw_std = float(np.std(raw, ddof=0))
    result[valid] = (raw - float(np.mean(raw))) / raw_std + location
    return result


def _run(tmp_path: Path, x: np.ndarray, name: str) -> np.ndarray:
    runtime = compile_formula("xs_gauss(x)", {"x": x}, n_instruments=x.shape[1])
    run = runtime.run(out_path=tmp_path / name)
    return np.fromfile(run.output_path, dtype=np.float64).reshape(run.output_shape)


def _assert_reference_location_and_scale(
    actual: np.ndarray,
    expected: np.ndarray,
    inputs: np.ndarray,
) -> None:
    np.testing.assert_allclose(
        actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True
    )
    for output_row, input_row in zip(actual, inputs):
        finite_output = output_row[np.isfinite(output_row)]
        finite_input = input_row[np.isfinite(input_row)]
        if finite_input.size == 0:
            continue

        location = _bounded_location(finite_input)
        np.testing.assert_allclose(
            np.mean(finite_output), location, rtol=2e-12, atol=2e-12
        )
        if finite_output.size > 1 and np.ptp(finite_input) > 0.0:
            np.testing.assert_allclose(
                np.std(finite_output, ddof=0), 1.0, rtol=2e-12, atol=2e-12
            )
        else:
            np.testing.assert_allclose(
                finite_output,
                np.full_like(finite_output, location),
                rtol=2e-12,
                atol=2e-12,
            )


def test_xs_gauss_matches_spacing_and_bounded_location_reference(
    tmp_path: Path,
) -> None:
    assert DEFAULT_DSL_REGISTRY.get("xs_gauss") is not None
    assert hasattr(cpp_stream, "xs_gauss")

    x = np.array(
        [
            [-3.0, -2.0, 10.0, np.nan, 1.0],
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [-1.25, -0.25, 0.75, 1.75, 2.75],
            [-2.0, 0.0, 0.0, 1.0, 5.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [-3.0, -3.0, -3.0, -3.0, -3.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    actual = _run(tmp_path, x, "xs_gauss.bin")
    expected = np.vstack([_reference(row) for row in x])

    _assert_reference_location_and_scale(actual, expected, x)
    np.testing.assert_array_equal(actual[-3], np.ones(x.shape[1]))
    np.testing.assert_array_equal(actual[-2], -np.ones(x.shape[1]))
    np.testing.assert_array_equal(actual[-1], np.zeros(x.shape[1]))


def test_xs_gauss_equal_spacing_is_scaled_xs_rank_and_shift_is_bounded(
    tmp_path: Path,
) -> None:
    centered = np.arange(-4.0, 5.0)
    shifted = centered + 0.75
    x = np.vstack((centered, shifted))
    actual = _run(tmp_path, x, "xs_gauss_equal_spacing.bin")

    rank_scores = np.array(
        [_NORMAL.inv_cdf(rank / 10.0) for rank in range(1, 10)]
    )
    scaled_rank = rank_scores / np.std(rank_scores, ddof=0)
    np.testing.assert_allclose(actual[0], scaled_rank, rtol=2e-12, atol=2e-12)

    expected_location = _bounded_location(shifted)
    np.testing.assert_allclose(
        actual[1] - actual[0],
        expected_location,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        np.mean(actual[1]), expected_location, rtol=2e-12, atol=2e-12
    )


def test_xs_gauss_nearly_constant_rows_remain_bounded(tmp_path: Path) -> None:
    offsets = np.arange(-4.0, 5.0)
    x = np.vstack(
        (
            1.0 + 1e-12 * offsets,
            -1.0 + 1e-12 * offsets,
            1e-12 * offsets,
        )
    )
    actual = _run(tmp_path, x, "xs_gauss_nearly_constant.bin")
    expected = np.vstack([_reference(row) for row in x])

    _assert_reference_location_and_scale(actual, expected, x)
    assert np.all(np.isfinite(actual))
    assert np.max(np.abs(actual)) < 3.0
    np.testing.assert_allclose(np.mean(actual[0]), 1.0, atol=2e-12)
    np.testing.assert_allclose(np.mean(actual[1]), -1.0, atol=2e-12)


def test_xs_gauss_wide_cross_sections_match_reference(tmp_path: Path) -> None:
    rng = np.random.default_rng(12345)
    cases = [
        rng.standard_t(df=2.0, size=17) + 0.4,
        rng.normal(loc=-0.35, size=150),
    ]
    for index, values in enumerate(cases):
        values[3] = np.nan
        values[7:9] = 1.25
        actual = _run(
            tmp_path,
            values[None, :],
            f"xs_gauss_wide_{index}.bin",
        )
        expected = _reference(values)[None, :]
        _assert_reference_location_and_scale(actual, expected, values[None, :])
