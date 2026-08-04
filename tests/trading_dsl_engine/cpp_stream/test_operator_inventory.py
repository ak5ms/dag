from __future__ import annotations

from pathlib import Path

import numpy as np

import trading_dsl_engine.cpp_stream as cpp_stream
from trading_dsl_engine.base.dsl import (
    DEFAULT_DSL_REGISTRY,
    get_dsl_op_signature,
)
from trading_dsl_engine.cpp_stream import compile_formula


def _run(
    tmp_path: Path,
    formula: str,
    data: dict[str, np.ndarray],
    *,
    key_cardinalities: dict[str, int] | None = None,
) -> np.ndarray:
    lanes = next(iter(data.values())).shape[1]
    runtime = compile_formula(
        formula,
        data,
        n_instruments=lanes,
        key_cardinalities=key_cardinalities,
    )
    result = runtime.run(out_path=tmp_path / "operators.bin")
    return np.fromfile(result.output_path, dtype=np.float64).reshape(
        result.output_shape
    )


def _rolling(
    values: np.ndarray,
    periods: int,
    projection,
    *,
    min_periods: int | None = None,
) -> np.ndarray:
    required = periods if min_periods is None else min_periods
    result = np.full_like(values, np.nan)
    for row in range(values.shape[0]):
        start = max(0, row + 1 - periods)
        for lane in range(values.shape[1]):
            finite = values[start : row + 1, lane]
            finite = finite[np.isfinite(finite)]
            if finite.size >= required:
                result[row, lane] = projection(finite, values[row, lane])
    return result


def test_full_operator_inventory_has_a_native_or_composed_entrypoint() -> None:
    names = {
        "abs", "ceil", "floor", "exp", "fraction", "inverse", "log",
        "log_diff", "max", "min", "maximum", "minimum", "nan_mask", "nan_out", "purify",
        "replace", "reverse", "round", "round_df", "round_down", "sign",
        "signed_power", "s_log_1p", "sqrt", "to_nan", "logical_and",
        "logical_or", "equal", "negate", "less", "if_else", "is_not_nan",
        "is_nan", "is_finite", "is_not_finite", "convert_float", "arc_cos",
        "arc_sin", "arc_tan", "sin", "cos", "tanh", "sigmoid", "clamp",
        "left_tail", "right_tail", "tail", "left_right_tail", "filter",
        "pasteurize", "get_df", "densify", "bucket", "vec_avg", "vec_choose",
        "vec_count", "vec_ir", "vec_kurtosis", "vec_max", "vec_min",
        "vec_norm", "vec_percentage", "vec_powersum", "vec_range",
        "vec_skewness", "vec_stddev", "vec_sum", "xs_rank", "xs_pct_rank",
        "xs_rank_by_side", "xs_generalized_rank", "xs_normalize", "xs_one_side",
        "xs_prob_density", "xs_scale", "xs_scale_down", "xs_truncate",
        "xs_vector_neut", "xs_vector_proj", "xs_winsorize", "xs_zscore",
        "xs_scale_by_side", "xs_direction", "xs_market_neutralize", "xs_filter",
        "xs_group_neutralize", "xs_quantile", "xs_generalized_rank",
        "xs_regression_neut", "xs_regression_proj", "xs_rank_gmean_amean_diff",
        "group_count", "group_na_count", "group_extra", "group_mean",
        "group_max", "group_median", "group_min", "group_rank", "group_scale",
        "group_std_dev", "group_sum", "group_zscore", "group_percentage",
        "group_vector_proj", "group_vector_neut", "group_backfill",
        "group_neutralize", "group_normalize", "periods_from_last_change",
        "hump", "hump_decay", "jump_decay", "keep", "trade_when",
        "ts_inst_tvr", "ts_backfill", "prev_diff_value", "ts_weighted_delay",
        "ts_shift", "ts_diff", "ts_returns", "ts_ln_change", "ts_pct_change",
        "ts_sum", "ts_product", "ts_mean", "ts_median", "ts_min", "ts_max",
        "ts_std", "ts_ir", "ts_rank", "ts_prob_density", "ts_percentage",
        "ewm_moment", "ewm_var", "ewm_std", "ewm_skewness", "ewm_kurtosis", "ewm_co_skewness",
        "ewm_co_kurtosis", "ewm_corr", "ewm_cov", "ewm_triple_corr",
        "ewm_partial_corr", "ts_regression", "ts_poly_regression",
        "ts_decay_linear", "ts_argmax", "ts_argmin", "ts_mean_diff",
        "ts_max_diff", "ts_min_diff", "ts_min_max_cps", "ts_min_max_diff",
        "ts_scale", "ts_zscore", "ts_count_nans", "ts_count_nonnumeric",
        "ts_entropy", "ewm_vector_neut", "ewm_vector_proj",
        "ts_rank_gmean_amean_diff", "ts_geomean", "slope", "ts_theilsen",
    }
    missing = sorted(
        name
        for name in names
        if DEFAULT_DSL_REGISTRY.get(name) is None
        and get_dsl_op_signature(name) is None
    )
    assert missing == []
    assert sorted(name for name in names if not hasattr(cpp_stream, name)) == []


def test_elementwise_compositions_and_bucket_match_reference(
    tmp_path: Path,
) -> None:
    x = np.array(
        [[-2.25, -0.5, 0.5, 2.25], [1.1, -1.1, 3.8, -3.8]],
        dtype=np.float64,
    )
    mask = np.array(
        [[1.0, -1.0, 0.0, -2.0], [-1.0, 1.0, 1.0, -1.0]],
        dtype=np.float64,
    )
    formula = (
        "cat(fraction(x), inverse(x), log(abs(x)+1), nan_mask(x, mask), "
        "nan_out(x, lower=-1, upper=1), reverse(x), round_df(x, 1), "
        "round_down(x, 0.5), signed_power(x, 2), s_log_1p(x), sigmoid(x), "
        "clamp(x, -1, 1), clamp(x, -1, 1, inverse=True, mask=9), "
        "left_right_tail(x, -2, 2), "
        "bucket(x, range=\"-2,2,1\", NANGroup=True))"
    )
    actual = _run(tmp_path, formula, {"x": x, "mask": mask})
    bucketed = sum((x >= boundary).astype(float) for boundary in (-1.0, 0.0, 1.0, 2.0))
    expected = np.stack(
        (
            np.modf(x)[0],
            1.0 / x,
            np.log(np.abs(x) + 1.0),
            np.where(mask < 0.0, np.nan, x),
            np.where((x < -1.0) | (x > 1.0), np.nan, x),
            -x,
            np.rint(x * 10.0) / 10.0,
            np.floor(x / 0.5) * 0.5,
            np.sign(x) * np.abs(x) ** 2,
            np.sign(x) * np.log1p(np.abs(x)),
            1.0 / (1.0 + np.exp(-x)),
            np.clip(x, -1.0, 1.0),
            np.where((x >= -1.0) & (x <= 1.0), 9.0, x),
            np.where((x >= -2.0) & (x <= 2.0), x, np.nan),
            bucketed,
        ),
        axis=-1,
    )
    np.testing.assert_allclose(
        actual, expected, rtol=2e-13, atol=2e-13, equal_nan=True
    )


def test_vector_cross_sectional_and_group_operators_match_reference(
    tmp_path: Path,
) -> None:
    vector = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 5.0, 2.0]],
            [[4.0, 2.0, 8.0, 6.0], [3.0, 3.0, 3.0, 3.0]],
        ]
    )
    vector_formula = (
        "cat(vec_avg(v), vec_choose(v, 2), vec_count(v), vec_ir(v), "
        "vec_kurtosis(v), vec_max(v), vec_min(v), vec_norm(v), "
        "vec_percentage(v, .25), vec_powersum(v, 2), vec_range(v), "
        "vec_skewness(v), vec_stddev(v), vec_sum(v))"
    )
    actual = _run(tmp_path, vector_formula, {"v": vector})
    mean = np.mean(vector, axis=-1)
    centered = vector - mean[..., None]
    variance = np.mean(centered**2, axis=-1)
    vector_std = np.std(vector, axis=-1)
    vector_ir = np.divide(
        mean,
        vector_std,
        out=np.full_like(mean, np.nan),
        where=vector_std > 0.0,
    )
    vector_kurtosis = np.divide(
        np.mean(centered**4, axis=-1),
        variance**2,
        out=np.full_like(mean, np.nan),
        where=variance > 0.0,
    )
    vector_skewness = np.divide(
        np.mean(centered**3, axis=-1),
        variance**1.5,
        out=np.full_like(mean, np.nan),
        where=variance > 0.0,
    )
    expected = np.stack(
        (
            mean,
            vector[..., 2],
            np.full_like(mean, 4.0),
            vector_ir,
            vector_kurtosis,
            np.max(vector, axis=-1),
            np.min(vector, axis=-1),
            np.sum(np.abs(vector), axis=-1),
            np.quantile(vector, 0.25, axis=-1),
            np.sum(vector**2, axis=-1),
            np.ptp(vector, axis=-1),
            vector_skewness,
            vector_std,
            np.sum(vector, axis=-1),
        ),
        axis=-1,
    )
    np.testing.assert_allclose(
        actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True
    )

    x = np.array([[1.0, 2.0, np.nan, 4.0], [4.0, 1.0, 3.0, 2.0]])
    y = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 1.0, 2.0]])
    group = np.array([[0.0, 0.0, 1.0, 1.0]] * 2)
    formula = (
        "cat(xs_count(x), xs_sum(x), xs_mean(x), xs_std(x), xs_min(x), "
        "xs_max(x), xs_median(x), densify(x), group_count(x, group), "
        "group_extra(x, 1, group), group_mean(x, 1, group), "
        "group_rank(x, group), group_neutralize(x, group), "
        "group_backfill(x, group, 2), xs_weighted_mean(x, y), "
        "xs_vector_proj(x, y), xs_regression_proj(x, y), "
        "xs_generalized_rank(x, 1))"
    )
    actual = _run(
        tmp_path,
        formula,
        {"x": x, "y": y, "group": group},
        key_cardinalities={"group": 2},
    )
    expected = np.empty_like(actual)
    for row, values in enumerate(x):
        finite = values[np.isfinite(values)]
        expected[row, :, 0] = finite.size
        expected[row, :, 1] = np.sum(finite)
        expected[row, :, 2] = np.mean(finite)
        expected[row, :, 3] = np.std(finite)
        expected[row, :, 4] = np.min(finite)
        expected[row, :, 5] = np.max(finite)
        expected[row, :, 6] = np.median(finite)
        unique = {value: index for index, value in enumerate(sorted(set(finite)))}
        expected[row, :, 7] = [unique.get(value, np.nan) for value in values]
        complete = np.isfinite(values) & np.isfinite(y[row])
        expected[row, :, 14] = np.sum(values[complete] * y[row, complete]) / np.sum(
            y[row, complete]
        )
        vector_beta = np.sum(values[complete] * y[row, complete]) / np.sum(
            y[row, complete] ** 2
        )
        expected[row, :, 15] = vector_beta * y[row]
        design = np.column_stack((np.ones(np.count_nonzero(complete)), y[row, complete]))
        regression_beta = np.linalg.solve(
            design.T @ design, design.T @ values[complete]
        )
        expected[row, :, 16] = regression_beta[0] + regression_beta[1] * y[row]
        expected[row, :, 17] = [
            (
                np.sum(value - finite) / finite.size
                if np.isfinite(value)
                else np.nan
            )
            for value in values
        ]
        for group_value in (0.0, 1.0):
            lanes = np.flatnonzero(group[row] == group_value)
            group_values = values[lanes]
            valid = group_values[np.isfinite(group_values)]
            count = valid.size
            group_mean = np.mean(valid)
            expected[row, lanes, 8] = count
            expected[row, lanes, 9] = np.where(
                np.isfinite(group_values), group_values, group_mean
            )
            expected[row, lanes, 10] = group_mean
            order = np.argsort(valid, kind="stable")
            sorted_values = valid[order]
            rank = {
                value: np.searchsorted(sorted_values, value, side="right")
                / (count + 1.0)
                for value in sorted_values
            }
            expected[row, lanes, 11] = [rank.get(value, np.nan) for value in group_values]
            expected[row, lanes, 12] = group_values - group_mean
            expected[row, lanes, 13] = np.where(
                np.isfinite(group_values), group_values, group_mean
            )
    np.testing.assert_allclose(
        actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True
    )


def test_native_history_operators_match_reference(tmp_path: Path) -> None:
    x = np.array([[1.0], [1.0], [2.0], [4.0], [3.0], [np.nan], [3.0]])
    trigger = np.array([[1.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
    exit_ = np.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
    formula = (
        "cat(periods_from_last_change(x), hump(x, hump=.5), "
        "hump_decay(x, p=.75), trade_when(trigger, x, exit_), "
        "filter(x, h=\"1,.5\", t=\"\"), "
        "rolling_product(x, 3, min_periods=1), "
        "rolling_kth(x, 3, k=2, ignore=\"NAN\", min_periods=2), "
        "rolling_prev_diff(x, 3), "
        "rolling_decay_linear(x, 3, min_periods=1), "
        "rolling_entropy(x, 3, buckets=3, min_periods=2))"
    )
    actual = _run(
        tmp_path,
        formula,
        {"x": x, "trigger": trigger, "exit_": exit_},
    )[:, 0]

    rows = x.shape[0]
    expected = np.full((rows, 10), np.nan)
    expected[:, 0] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    expected[:, 1] = [1.0, 1.0, 1.5, 2.0, 2.5, 2.5, 2.5]
    expected[:, 2] = [1.0, 1.0, 2.0, 4.0, 3.0, 3.0, 3.0]
    expected[:, 3] = [1.0, 1.0, np.nan, 4.0, 4.0, 4.0, 4.0]
    expected[:, 4] = [1.0, 1.5, 2.5, 5.0, 5.0, np.nan, np.nan]
    for row in range(rows):
        start = max(0, row - 2)
        window = x[start : row + 1, 0]
        finite = window[np.isfinite(window)]
        expected[row, 5] = np.prod(finite) if finite.size else np.nan
        if finite.size >= 2:
            expected[row, 6] = finite[-2]
        current = x[row, 0]
        if np.isfinite(current):
            prior = [
                value
                for value in window[:-1][::-1]
                if np.isfinite(value) and value != current
            ]
            expected[row, 7] = prior[0] if prior else np.nan
        finite_mask = np.isfinite(window)
        ages = np.arange(window.size - 1, -1, -1)
        weights = 3.0 - ages[finite_mask]
        expected[row, 8] = (
            np.dot(window[finite_mask], weights) / np.sum(weights)
        )
        if finite.size >= 2:
            if np.min(finite) == np.max(finite):
                expected[row, 9] = 0.0
            else:
                bins = np.minimum(
                    2,
                    ((finite - np.min(finite)) * 3 / np.ptp(finite)).astype(int),
                )
                counts = np.bincount(bins, minlength=3)
                probabilities = counts[counts > 0] / finite.size
                expected[row, 9] = -np.sum(probabilities * np.log(probabilities))
    np.testing.assert_allclose(
        actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True
    )


def test_time_series_compositions_match_fixed_period_references(
    tmp_path: Path,
) -> None:
    x = np.array(
        [[1.0, 2.0], [2.0, 4.0], [4.0, 8.0], [8.0, np.nan], [16.0, 16.0]]
    )
    formula = (
        "cat(ts_shift(x, 2), ts_diff(x, 2), ts_returns(x, 2), "
        "ts_returns(x, 2, mode=2), ts_sum(x, 3), ts_product(x, 3), "
        "ts_mean(x, 3), ts_std(x, 3), ts_rank(x, 3), "
        "ts_percentage(x, 3, percentage=.25), ts_decay_linear(x, 3), "
        "ts_count_nans(x, 3), ts_geomean(x, 3), slope(x, 2))"
    )
    actual = _run(tmp_path, formula, {"x": x})
    lag = np.full_like(x, np.nan)
    lag[2:] = x[:-2]
    change = x - lag
    parts = [
        lag,
        change,
        change / lag,
        change / (0.5 * (x + lag)),
        _rolling(x, 3, lambda values, _: np.sum(values)),
        _rolling(x, 3, lambda values, _: np.prod(values)),
        _rolling(x, 3, lambda values, _: np.mean(values)),
        _rolling(x, 3, lambda values, _: np.std(values)),
        _rolling(
            x,
            3,
            lambda values, current: np.count_nonzero(values <= current)
            / (values.size + 1.0),
        ),
        _rolling(x, 3, lambda values, _: np.quantile(values, 0.25)),
    ]
    decay = np.full_like(x, np.nan)
    for row in range(x.shape[0]):
        start = max(0, row - 2)
        for lane in range(x.shape[1]):
            window = x[start : row + 1, lane]
            ages = np.arange(window.size - 1, -1, -1)
            finite = np.isfinite(window)
            if np.count_nonzero(finite) == 3:
                weights = 3.0 - ages[finite]
                decay[row, lane] = np.dot(window[finite], weights) / np.sum(weights)
    parts.append(decay)
    parts.append(_rolling(np.isnan(x).astype(float), 3, lambda values, _: np.sum(values)))
    parts.append(_rolling(x, 3, lambda values, _: np.exp(np.mean(np.log(values)))))
    slope_reference = (x - np.vstack((np.full((1, 2), np.nan), x[:-1])))
    slope_reference += 0.5 * change
    parts.append(slope_reference)
    expected = np.stack(parts, axis=-1)
    np.testing.assert_allclose(
        actual, expected, rtol=3e-12, atol=3e-12, equal_nan=True
    )

    signed = np.array([[-2.0], [-4.0], [2.0]])
    signed_returns = _run(
        tmp_path,
        "ts_returns(x, 1)",
        {"x": signed},
    )
    np.testing.assert_allclose(
        signed_returns.reshape(-1),
        [np.nan, 1.0, -1.5],
        equal_nan=True,
    )


def test_remaining_composed_operator_groups_compile_and_run(
    tmp_path: Path,
) -> None:
    x = np.array(
        [
            [-2.0, -0.5, 1.0, 3.0],
            [-1.0, 2.0, 4.0, 1.0],
            [3.0, -2.0, 2.0, 5.0],
            [4.0, 1.0, -1.0, 2.0],
        ]
    )
    y = np.array(
        [
            [1.0, 2.0, -1.0, 3.0],
            [2.0, -1.0, 3.0, 1.0],
            [-1.0, 4.0, 1.0, 2.0],
            [3.0, 1.0, 2.0, -2.0],
        ]
    )
    group = np.array([[0.0, 0.0, 1.0, 1.0]] * x.shape[0])
    data = {"x": x, "y": y, "group": group}
    formulas = [
        (
            "cat(replace(x, target=\"1 2\", dest=\"7,8\"), to_nan(x, value=-2), "
            "to_nan(x, value=9, reverse=True), logical_and(x>0, y>0), "
            "logical_or(x>0, y>0), equal(x,y), negate(x>0), less(x,y), "
            "if_else(x>0,x,y), is_not_nan(x), is_nan(x), is_finite(x), "
            "is_not_finite(x), convert_float(x), arc_cos(tanh(x)), "
            "arc_sin(tanh(x)), arc_tan(x), sin(x), cos(x), tanh(x), "
            "pasteurize(x), get_df(x, 3))"
        ),
        (
            "cat(xs_rank_by_side(x), xs_normalize(x, useStd=True), "
            "xs_one_side(x, side=\"short\"), "
            "xs_prob_density(x, driver=\"cauchy\"), xs_scale(x, scale=2), "
            "xs_scale_down(x), xs_truncate(x), xs_vector_neut(x,y), "
            "xs_vector_proj(x,y), xs_winsorize(x), xs_zscore(x), "
            "xs_scale_by_side(x), xs_direction(x), "
            "xs_market_neutralize(x,group), xs_filter(x,.5), "
            "xs_regression_neut(x,y), xs_regression_proj(x,y), "
            "xs_rank_gmean_amean_diff(x,y))"
        ),
        (
            "cat(group_max(x,group), group_median(x,group), "
            "group_min(x,group), group_scale(x,group), "
            "group_std_dev(x,group), group_sum(x,group), "
            "group_zscore(x,group), group_percentage(x,group,.25), "
            "group_vector_proj(x,y,group), group_vector_neut(x,y,group), "
            "group_neutralize(x,group), group_normalize(x,group,scale=2))"
        ),
        (
            "cat(ewm_vector_proj(x,y,span=3,adjust=True), "
            "ewm_vector_neut(x,y,span=3,adjust=True), "
            "ts_poly_regression(y,x,periods=3,k=2,lambda_=0.1))"
        ),
        (
            "cat(jump_decay(x,2,stddev=False,sensitivity=.5,force=.1), "
            "keep(x,y,periods=2), ts_inst_tvr(x,2), "
            "ts_backfill(x,2,k=1,ignore=\"NAN\"), "
            "prev_diff_value(x,3), ts_weighted_delay(x,.25))"
        ),
        (
            "cat(ts_ln_change(abs(x)+3,1), ts_pct_change(abs(x)+3,1), "
            "ts_median(x,2), ts_min(x,2), ts_max(x,2), ts_ir(x,2), "
            "ts_rank(x,2), ts_prob_density(x,2,driver=\"uniform\"), "
            "ts_percentage(x,2,.25), ts_argmax(x,2), ts_argmin(x,2), "
            "ts_mean_diff(x,2), ts_max_diff(x,2), ts_min_diff(x,2), "
            "ts_min_max_cps(x,2), ts_min_max_diff(x,2), ts_scale(x,2), "
            "ts_zscore(x,2), ts_count_nans(x,2), "
            "ts_count_nonnumeric(x,2), ts_entropy(x,2,buckets=2), "
            "ts_rank_gmean_amean_diff(x,y,periods=2), "
            "ts_theilsen(y,x,3))"
        ),
    ]
    for formula in formulas:
        names = {
            name: values
            for name, values in data.items()
            if name in {"x", "y"}
            or (name == "group" and "group" in formula)
        }
        actual = _run(
            tmp_path,
            formula,
            names,
            key_cardinalities={"group": 2},
        )
        assert actual.shape[0] == x.shape[0]
        assert np.count_nonzero(np.isfinite(actual)) > 0
