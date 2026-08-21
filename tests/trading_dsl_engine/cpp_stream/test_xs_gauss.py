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
    runtime = compile_formula("xs_gauss(x)", {"x": x}, n_instruments=x.shape[1])
    run = runtime.run(out_path=tmp_path / "xs_gauss.bin")
    actual = np.fromfile(run.output_path, dtype=np.float64).reshape(run.output_shape)
    expected = np.vstack([_reference(row) for row in x])

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True)
    for row in actual[:-1]:
        finite = row[np.isfinite(row)]
        np.testing.assert_allclose(np.std(finite, ddof=0), 1.0, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(actual[-1], np.zeros(x.shape[1]))
