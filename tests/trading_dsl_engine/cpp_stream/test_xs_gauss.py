from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import numpy as np

import trading_dsl_engine.cpp_stream as cpp_stream
from trading_dsl_engine.base.dsl import DEFAULT_DSL_REGISTRY
from trading_dsl_engine.cpp_stream import compile_formula


_NORMAL = NormalDist()


def _reference(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64)
    result = np.full_like(row, np.nan)
    valid = np.isfinite(row)
    values = row[valid]
    if values.size == 0:
        return result

    magnitudes = np.sort(np.abs(values))
    total = float(np.sum(magnitudes))
    denominator = total + 0.5 * float(magnitudes[0] + magnitudes[-1])
    if denominator == 0.0:
        result[valid] = 0.0
        return result

    levels = np.cumsum(magnitudes) / denominator
    zero_count = int(np.searchsorted(magnitudes, 0.0, side="right"))
    if 0 < zero_count < magnitudes.size:
        first_level = magnitudes[zero_count] / denominator
        levels[:zero_count] = (
            first_level
            * np.arange(1, zero_count + 1, dtype=np.float64)
            / (zero_count + 1.0)
        )

    order = np.argsort(values, kind="stable")
    raw = np.empty_like(values)
    position = 0
    while position < values.size:
        upper = position + 1
        while upper < values.size and values[order[upper]] == values[order[position]]:
            upper += 1
        score = _NORMAL.inv_cdf(float(levels[upper - 1]))
        raw[order[position:upper]] = score
        position = upper

    stddev = float(np.std(raw, ddof=0))
    result[valid] = raw / stddev if stddev > 0.0 else 0.0
    return result


def _run(tmp_path: Path, x: np.ndarray, name: str) -> np.ndarray:
    runtime = compile_formula("xs_gauss(x)", {"x": x}, n_instruments=x.shape[1])
    run = runtime.run(out_path=tmp_path / name)
    return np.fromfile(run.output_path, dtype=np.float64).reshape(run.output_shape)


def _assert_reference_and_unit_std(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(
        actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True
    )
    for row in actual:
        finite = row[np.isfinite(row)]
        if finite.size and np.std(finite, ddof=0) > 0.0:
            np.testing.assert_allclose(
                np.std(finite, ddof=0), 1.0, rtol=2e-12, atol=2e-12
            )


def test_xs_gauss_matches_reference(tmp_path: Path) -> None:
    assert DEFAULT_DSL_REGISTRY.get("xs_gauss") is not None
    assert hasattr(cpp_stream, "xs_gauss")

    x = np.array(
        [
            [-3.0, -2.0, 10.0, np.nan],
            [-6.0, 5.0, 1.0, 2.0],
            [-1.0, 0.0, 2.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    actual = _run(tmp_path, x, "xs_gauss.bin")
    expected = np.vstack([_reference(row) for row in x])

    _assert_reference_and_unit_std(actual, expected)
    np.testing.assert_array_equal(actual[-1], np.zeros(x.shape[1]))


def test_xs_gauss_large_sort_path(tmp_path: Path) -> None:
    rng = np.random.default_rng(12345)
    x = np.vstack(
        (
            rng.standard_t(df=2.0, size=17),
            rng.normal(size=17),
        )
    )
    x[0, [1, 6]] = 0.0
    x[1, 3] = np.nan
    x[1, 7:9] = 1.25

    actual = _run(tmp_path, x, "xs_gauss_wide.bin")
    expected = np.vstack([_reference(row) for row in x])
    _assert_reference_and_unit_std(actual, expected)
