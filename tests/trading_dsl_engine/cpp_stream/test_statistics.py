from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from trading_dsl_engine.cpp_stream import compile_formula


def _run(
    formula: str,
    data: dict[str, np.ndarray],
    tmp_path: Path,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, object]:
    runtime = compile_formula(
        formula,
        data,
        n_instruments=next(iter(data.values())).shape[1],
    )
    output = tmp_path / "output.bin"
    runtime.run(out_path=output)
    return np.fromfile(output, dtype=np.float64).reshape(shape), runtime


def _ewm_expectations(
    arrays: Sequence[np.ndarray],
    powers: Sequence[Sequence[int]],
    *,
    span: float,
    min_periods: int,
    ignore_na: bool,
    adjust: bool,
) -> np.ndarray:
    rows, lanes = arrays[0].shape
    alpha = 2.0 / (span + 1.0)
    old_weight_factor = 1.0 - alpha
    state = np.zeros((lanes, len(powers)), dtype=np.float64)
    weight = np.zeros(lanes, dtype=np.float64)
    count = np.zeros(lanes, dtype=np.int64)
    initialized = np.zeros(lanes, dtype=bool)
    result = np.full((rows, lanes, len(powers)), np.nan, dtype=np.float64)
    for row in range(rows):
        for lane in range(lanes):
            observations = np.array(
                [values[row, lane] for values in arrays], dtype=np.float64
            )
            valid = bool(np.all(np.isfinite(observations)))
            old_weight = weight[lane]
            if initialized[lane] and (valid or not ignore_na):
                old_weight *= old_weight_factor
            if valid:
                monomials = np.array(
                    [
                        np.prod(
                            [value**power for value, power in zip(observations, term)]
                        )
                        for term in powers
                    ]
                )
                if initialized[lane]:
                    new_weight = 1.0 if adjust else alpha
                    if not adjust and abs(alpha - 0.5) <= 1e-12:
                        new_weight = 1.0 - old_weight
                    state[lane] = (
                        old_weight * state[lane] + new_weight * monomials
                    ) / (old_weight + new_weight)
                    old_weight = old_weight + new_weight if adjust else 1.0
                else:
                    state[lane] = monomials
                    initialized[lane] = True
                    old_weight = 1.0
                count[lane] += 1
            weight[lane] = old_weight
            if initialized[lane] and count[lane] >= min_periods:
                result[row, lane] = state[lane]
    return result


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=(denominator > 0.0) & np.isfinite(denominator),
    )


def test_xs_pct_rank_uses_upper_tie_rank_and_ignores_nan(tmp_path: Path) -> None:
    x = np.array(
        [
            [3.0, 1.0, 1.0, np.nan, 2.0],
            [np.nan, np.nan, 5.0, np.nan, np.nan],
            [-2.0, 4.0, 0.0, 4.0, 1.0],
        ]
    )
    actual, _ = _run("xs_pct_rank(x)", {"x": x}, tmp_path, x.shape)
    expected = np.array(
        [
            [4.0 / 5.0, 2.0 / 5.0, 2.0 / 5.0, np.nan, 3.0 / 5.0],
            [np.nan, np.nan, 1.0 / 2.0, np.nan, np.nan],
            [1.0 / 6.0, 5.0 / 6.0, 2.0 / 6.0, 5.0 / 6.0, 3.0 / 6.0],
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)


def test_ewm_moment_family_matches_shared_complete_case_reference(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(20260804)
    rows, lanes = 32, 4
    x = rng.normal(size=(rows, lanes))
    y = 0.4 * x + rng.normal(scale=0.8, size=(rows, lanes))
    z = -0.2 * x + 0.3 * y + rng.normal(scale=0.6, size=(rows, lanes))
    x[3, 1] = np.nan
    y[7, 2] = np.nan
    z[11, 0] = np.nan
    span = 3.5
    min_periods = 3
    ignore_na = False
    adjust = True
    formula = (
        "cat("
        "ewm_moment(x, span=3.5, k=3, min_periods=3, ignore_na=False, adjust=True),"
        "ewm_var(x, span=3.5, min_periods=3, ignore_na=False, adjust=True),"
        "ewm_std(x, span=3.5, min_periods=3, ignore_na=False, adjust=True),"
        "ewm_skewness(x, span=3.5, min_periods=3, ignore_na=False, adjust=True),"
        "ewm_kurtosis(x, span=3.5, min_periods=3, ignore_na=False, adjust=True),"
        "ewm_cov(x, y, span=3.5, min_periods=3, ignore_na=False, adjust=True),"
        "ewm_corr(x, y, span=3.5, min_periods=3, ignore_na=False, adjust=True),"
        "ewm_co_skewness(y, x, span=3.5, min_periods=3, ignore_na=False, adjust=True),"
        "ewm_co_kurtosis(y, x, span=3.5, min_periods=3, ignore_na=False, adjust=True),"
        "ewm_triple_corr(x, y, z, span=3.5, min_periods=3, ignore_na=False, adjust=True),"
        "ewm_partial_corr(x, y, z, span=3.5, min_periods=3, ignore_na=False, adjust=True))"
    )
    actual, runtime = _run(
        formula,
        {"x": x, "y": y, "z": z},
        tmp_path,
        (rows, lanes, 11),
    )
    assert runtime.parallel_plan.mode == "lanes"

    univariate = _ewm_expectations(
        (x,),
        ((1,), (2,), (3,), (4,)),
        span=span,
        min_periods=min_periods,
        ignore_na=ignore_na,
        adjust=adjust,
    )
    mean_x, second_x, third_x, fourth_x = np.moveaxis(univariate, -1, 0)
    variance_x = np.maximum(0.0, second_x - mean_x * mean_x)
    moment_3 = third_x - 3.0 * mean_x * second_x + 2.0 * mean_x**3
    moment_4 = (
        fourth_x
        - 4.0 * mean_x * third_x
        + 6.0 * mean_x * mean_x * second_x
        - 3.0 * mean_x**4
    )

    pair_xy = _ewm_expectations(
        (x, y),
        ((1, 0), (0, 1), (2, 0), (0, 2), (1, 1)),
        span=span,
        min_periods=min_periods,
        ignore_na=ignore_na,
        adjust=adjust,
    )
    mx, my, x2, y2, xy = np.moveaxis(pair_xy, -1, 0)
    vx = np.maximum(0.0, x2 - mx * mx)
    vy = np.maximum(0.0, y2 - my * my)
    covariance = xy - mx * my

    pair_yx = _ewm_expectations(
        (y, x),
        (
            (1, 0),
            (0, 1),
            (2, 0),
            (0, 2),
            (1, 1),
            (0, 3),
            (1, 2),
            (1, 3),
            (0, 4),
        ),
        span=span,
        min_periods=min_periods,
        ignore_na=ignore_na,
        adjust=adjust,
    )
    myc, mxc, y2c, x2c, yx, x3c, yx2, yx3, _ = np.moveaxis(
        pair_yx, -1, 0
    )
    vyc = np.maximum(0.0, y2c - myc * myc)
    vxc = np.maximum(0.0, x2c - mxc * mxc)
    co_skew_central = (
        yx2 - 2.0 * mxc * yx - myc * x2c + 2.0 * myc * mxc * mxc
    )
    co_kurt_central = (
        yx3
        - 3.0 * mxc * yx2
        + 3.0 * mxc * mxc * yx
        - myc * x3c
        + 3.0 * myc * mxc * x2c
        - 3.0 * myc * mxc**3
    )

    triple = _ewm_expectations(
        (x, y, z),
        (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (2, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ),
        span=span,
        min_periods=min_periods,
        ignore_na=ignore_na,
        adjust=adjust,
    )
    tx, ty, tz, tx2, ty2, tz2, txy, txz, tyz, txyz = np.moveaxis(
        triple, -1, 0
    )
    tvx = np.maximum(0.0, tx2 - tx * tx)
    tvy = np.maximum(0.0, ty2 - ty * ty)
    tvz = np.maximum(0.0, tz2 - tz * tz)
    triple_central = txyz - tx * tyz - ty * txz - tz * txy + 2.0 * tx * ty * tz
    rxy = _safe_ratio(txy - tx * ty, np.sqrt(tvx * tvy))
    rxz = _safe_ratio(txz - tx * tz, np.sqrt(tvx * tvz))
    ryz = _safe_ratio(tyz - ty * tz, np.sqrt(tvy * tvz))

    expected = np.stack(
        (
            moment_3,
            variance_x,
            np.sqrt(variance_x),
            _safe_ratio(moment_3, variance_x * np.sqrt(variance_x)),
            _safe_ratio(moment_4, variance_x * variance_x),
            covariance,
            _safe_ratio(covariance, np.sqrt(vx * vy)),
            _safe_ratio(co_skew_central, np.sqrt(vyc) * vxc),
            _safe_ratio(co_kurt_central, np.sqrt(vyc) * vxc * np.sqrt(vxc)),
            _safe_ratio(triple_central, np.sqrt(tvx * tvy * tvz)),
            _safe_ratio(
                rxy - rxz * ryz,
                np.sqrt(np.maximum(0.0, 1.0 - rxz * rxz))
                * np.sqrt(np.maximum(0.0, 1.0 - ryz * ryz)),
            ),
        ),
        axis=-1,
    )
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=3e-10,
        atol=3e-10,
        equal_nan=True,
    )


def _rolling_reference(
    values: np.ndarray,
    periods: int,
    min_periods: int,
    projection: Callable[[np.ndarray, float], float],
) -> np.ndarray:
    rows, lanes = values.shape
    result = np.full_like(values, np.nan)
    for row in range(rows):
        start = max(0, row + 1 - periods)
        for lane in range(lanes):
            window = values[start : row + 1, lane]
            finite = window[np.isfinite(window)]
            if finite.size >= min_periods and finite.size:
                result[row, lane] = projection(finite, values[row, lane])
    return result


def test_rolling_family_uses_period_counts_and_nan_skipping(tmp_path: Path) -> None:
    x = np.array(
        [
            [1.0, np.nan, 3.0],
            [2.0, 1.0, 3.0],
            [2.0, 2.0, np.nan],
            [np.nan, 3.0, 1.0],
            [-1.0, 3.0, 2.0],
            [4.0, 0.0, 2.0],
            [4.0, -2.0, 5.0],
            [0.0, np.nan, 5.0],
        ]
    )
    rows, lanes = x.shape
    periods = 5
    min_periods = 2
    formula = (
        "cat("
        "rolling_sum(x, periods=5, min_periods=2),"
        "rolling_mean(x, periods=5, min_periods=2),"
        "rolling_std(x, periods=5, min_periods=2, ddof=1),"
        "rolling_min(x, periods=5, min_periods=2),"
        "rolling_max(x, periods=5, min_periods=2),"
        "rolling_median(x, periods=5, min_periods=2),"
        "rolling_quantile(x, periods=5, q=0.25, min_periods=2),"
        "rolling_pct_rank(x, periods=5, min_periods=2),"
        "rolling_argmin(x, periods=5, min_periods=2),"
        "rolling_argmax(x, periods=5, min_periods=2))"
    )
    actual, runtime = _run(formula, {"x": x}, tmp_path, (rows, lanes, 10))
    assert runtime.parallel_plan.mode == "lanes"

    simple = (
        lambda window, _: np.sum(window),
        lambda window, _: np.mean(window),
        lambda window, _: np.std(window, ddof=1),
        lambda window, _: np.min(window),
        lambda window, _: np.max(window),
        lambda window, _: np.quantile(window, 0.5),
        lambda window, _: np.quantile(window, 0.25),
        lambda window, current: (
            np.count_nonzero(window <= current) / (window.size + 1.0)
            if np.isfinite(current)
            else np.nan
        ),
    )
    expected_parts = [
        _rolling_reference(x, periods, min_periods, projection)
        for projection in simple
    ]
    for is_max in (False, True):
        expected = np.full_like(x, np.nan)
        for row in range(rows):
            start = max(0, row + 1 - periods)
            for lane in range(lanes):
                window = x[start : row + 1, lane]
                finite_positions = np.flatnonzero(np.isfinite(window))
                if finite_positions.size < min_periods:
                    continue
                finite_values = window[finite_positions]
                extreme = np.max(finite_values) if is_max else np.min(finite_values)
                latest = finite_positions[finite_values == extreme][-1]
                expected[row, lane] = row - (start + latest)
        expected_parts.append(expected)
    expected = np.stack(expected_parts, axis=-1)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-12,
        atol=2e-12,
        equal_nan=True,
    )


def test_cheap_xs_and_rolling_compositions_expand_to_native_primitives(
    tmp_path: Path,
) -> None:
    x = np.array(
        [
            [1.0, 2.0, -1.0, 4.0],
            [2.0, 4.0, 1.0, 3.0],
            [-1.0, 3.0, 2.0, 5.0],
            [4.0, 1.0, 3.0, 2.0],
            [5.0, -2.0, 4.0, 1.0],
        ]
    )
    y = np.array(
        [
            [2.0, -1.0, 3.0, 1.0],
            [1.0, 2.0, -2.0, 4.0],
            [3.0, 1.0, 2.0, -1.0],
            [-2.0, 4.0, 1.0, 3.0],
            [1.0, 3.0, -1.0, 2.0],
        ]
    )
    rows, lanes = x.shape
    xs_formula = (
        "cat(xs_demean(x), xs_zscore(x), xs_scale(x, 2.0), "
        "xs_direction(x), xs_vector_proj(x, y), xs_vector_neut(x, y))"
    )
    actual, _ = _run(
        xs_formula,
        {"x": x, "y": y},
        tmp_path,
        (rows, lanes, 6),
    )
    demeaned = x - np.mean(x, axis=1, keepdims=True)
    coefficient = np.sum(x * y, axis=1, keepdims=True) / np.sum(
        y * y, axis=1, keepdims=True
    )
    projection = coefficient * y
    expected = np.stack(
        (
            demeaned,
            demeaned / np.std(x, axis=1, keepdims=True),
            2.0 * x / np.sum(np.abs(x), axis=1, keepdims=True),
            x / np.sqrt(np.sum(x * x, axis=1, keepdims=True)),
            projection,
            x - projection,
        ),
        axis=-1,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)

    rolling_formula = (
        "cat(rolling_range(x, periods=3, min_periods=2), "
        "rolling_zscore(x, periods=3, min_periods=2), "
        "rolling_scale(x, periods=3, constant=0.5, min_periods=2))"
    )
    actual, _ = _run(
        rolling_formula,
        {"x": x},
        tmp_path,
        (rows, lanes, 3),
    )
    low = _rolling_reference(x, 3, 2, lambda window, _: np.min(window))
    high = _rolling_reference(x, 3, 2, lambda window, _: np.max(window))
    mean = _rolling_reference(x, 3, 2, lambda window, _: np.mean(window))
    std = _rolling_reference(x, 3, 2, lambda window, _: np.std(window))
    expected = np.stack(
        (high - low, (x - mean) / std, (x - low) / (high - low) + 0.5),
        axis=-1,
    )
    np.testing.assert_allclose(
        actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True
    )


def _theilsen_slope(y: np.ndarray, x: np.ndarray) -> float:
    slopes = [
        (y[right] - y[left]) / (x[right] - x[left])
        for left in range(x.size)
        for right in range(left + 1, x.size)
        if x[right] != x[left]
    ]
    return float(np.median(slopes)) if slopes else np.nan


def test_rolling_theilsen_matches_exact_pairwise_median(tmp_path: Path) -> None:
    rng = np.random.default_rng(91)
    rows, lanes = 25, 3
    x = rng.normal(size=(rows, lanes))
    y = 1.7 * x + rng.normal(scale=0.15, size=(rows, lanes))
    x[6, 1] = x[5, 1]
    x[11, 2] = np.nan
    y[15, 0] = np.nan
    periods = 7
    min_periods = 4
    actual, runtime = _run(
        "rolling_theilsen(y, x, periods=7, min_periods=4)",
        {"x": x, "y": y},
        tmp_path,
        (rows, lanes),
    )
    assert runtime.parallel_plan.mode == "lanes"
    expected = np.full_like(x, np.nan)
    for row in range(rows):
        start = max(0, row + 1 - periods)
        for lane in range(lanes):
            valid = np.isfinite(x[start : row + 1, lane]) & np.isfinite(
                y[start : row + 1, lane]
            )
            if np.count_nonzero(valid) >= min_periods:
                expected[row, lane] = _theilsen_slope(
                    y[start : row + 1, lane][valid],
                    x[start : row + 1, lane][valid],
                )
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-12,
        atol=2e-12,
        equal_nan=True,
    )


def test_large_rolling_theilsen_uses_subquadratic_selection_accurately(
    tmp_path: Path,
) -> None:
    periods = 513
    rng = np.random.default_rng(771)
    x = rng.normal(size=(periods, 1))
    y = 2.25 * x + rng.normal(scale=0.3, size=(periods, 1))
    y[::37] += rng.normal(scale=12.0, size=(len(y[::37]), 1))
    actual, _ = _run(
        "rolling_theilsen(y, x, periods=513, min_periods=513)",
        {"x": x, "y": y},
        tmp_path,
        (periods, 1),
    )
    expected = _theilsen_slope(y[:, 0], x[:, 0])
    assert np.isnan(actual[:-1]).all()
    np.testing.assert_allclose(actual[-1, 0], expected, rtol=2e-10, atol=2e-10)
