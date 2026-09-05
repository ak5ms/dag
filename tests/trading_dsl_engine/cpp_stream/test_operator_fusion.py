from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_dsl_engine.cpp_stream import compile_formula


def _run(runtime, path: Path, shape: tuple[int, ...]) -> np.ndarray:
    runtime.run(out_path=path)
    return np.fromfile(path, dtype=np.float64).reshape(shape)


def test_lazy_scalar_graph_fuses_into_stateful_consumer(tmp_path: Path) -> None:
    rng = np.random.default_rng(9821)
    x = rng.normal(size=(45, 5))
    y = rng.normal(size=(45, 5))
    x[4, 2] = np.nan
    formula = "ewm((x + y) * (y + x), span=5, min_periods=2)"
    runtime = compile_formula(formula, {"x": x, "y": y}, n_instruments=5)
    actual = _run(runtime, tmp_path / "lazy-ewm.bin", x.shape)

    values = (x + y) ** 2
    expected = np.column_stack(
        [
            pd.Series(values[:, lane])
            .ewm(span=5, min_periods=2, ignore_na=True, adjust=False)
            .mean()
            .to_numpy()
            for lane in range(values.shape[1])
        ]
    )
    np.testing.assert_allclose(
        actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True
    )
    assert [stage.kind for stage in runtime.plan.stages] == ["ewm"]
    assert runtime.plan.scratch_slots == 0


def test_generic_ewm_bundle_splits_divergent_missing_value_metadata(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(1207)
    x = rng.normal(size=(53, 4))
    y = rng.normal(size=(53, 4))
    x[[2, 11, 20], [0, 1, 3]] = np.nan
    y[[5, 11, 31], [2, 0, 1]] = np.nan
    formula = (
        "cat("
        "ewm(x, span=3.5, min_periods=3, ignore_na=False, adjust=True),"
        "ewm(y, span=3.5, min_periods=3, ignore_na=False, adjust=True))"
    )
    runtime = compile_formula(formula, {"x": x, "y": y}, n_instruments=4)
    actual = _run(runtime, tmp_path / "ewm-bundle.bin", (53, 4, 2))
    expected = np.stack(
        [
            np.column_stack(
                [
                    pd.Series(values[:, lane])
                    .ewm(
                        span=3.5,
                        min_periods=3,
                        ignore_na=False,
                        adjust=True,
                    )
                    .mean()
                    .to_numpy()
                    for lane in range(values.shape[1])
                ]
            )
            for values in (x, y)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(
        actual, expected, rtol=3e-12, atol=3e-12, equal_nan=True
    )
    bundles = [stage for stage in runtime.plan.stages if stage.kind == "ewm_bundle"]
    assert len(bundles) == 1
    assert len(bundles[0].members) == 2


def test_comoment_expansion_uses_one_generic_ewm_traversal() -> None:
    values = np.ones((3, 2), dtype=np.float64)
    runtime = compile_formula(
        "ewm_co_kurtosis(y, x, span=8)",
        {"x": values, "y": values},
        n_instruments=2,
    )
    bundles = [stage for stage in runtime.plan.stages if stage.kind == "ewm_bundle"]
    assert len(bundles) == 1
    assert len(bundles[0].members) == 8
    # Final algebra is emitted directly from the shared moment-state bundle.
    assert runtime.plan.scratch_slots == 0
    assert len(bundles[0].epilogues) == 1
    assert not {
        "unary",
        "binary",
        "ternary",
        "custom",
        "tensor_unary",
        "tensor_binary",
        "tensor_ternary",
    }.intersection(stage.kind for stage in runtime.plan.stages)


def test_tensor_moment_reductions_share_one_pass(tmp_path: Path) -> None:
    rng = np.random.default_rng(713)
    values = rng.normal(size=(13, 3, 19))
    values[2, 1, 4] = np.nan
    values[8, 0, 7] = np.nan
    runtime = compile_formula(
        "cat(vec_skewness(v), vec_kurtosis(v))",
        {"v": values},
        n_instruments=3,
    )
    actual = _run(runtime, tmp_path / "vector-moments.bin", (13, 3, 2))
    mean = np.nanmean(values, axis=-1)
    centered = values - mean[..., None]
    variance = np.nanmean(centered**2, axis=-1)
    expected = np.stack(
        (
            np.nanmean(centered**3, axis=-1) / variance**1.5,
            np.nanmean(centered**4, axis=-1) / variance**2,
        ),
        axis=-1,
    )
    np.testing.assert_allclose(actual, expected, rtol=4e-12, atol=4e-12)
    bundles = [
        stage for stage in runtime.plan.stages if stage.kind == "reduction_bundle"
    ]
    assert len(bundles) == 1
    assert len(bundles[0].members) == 4
    assert runtime.plan.matrix_scratch_slots == 0


def test_specialized_cross_sectional_rank_node_is_preserved() -> None:
    values = np.arange(24, dtype=np.float64).reshape(6, 4)
    runtime = compile_formula(
        "xs_rank(ewm(x + 1, span=4))",
        {"x": values},
        n_instruments=4,
    )
    assert [stage.kind for stage in runtime.plan.stages] == ["ewm", "xs_rank"]


def test_large_rolling_order_and_history_paths_match_reference(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(9917)
    rows, lanes, periods = 150, 3, 67
    values = np.round(rng.normal(size=(rows, lanes)), 1)
    values[rng.random(values.shape) < 0.12] = np.nan
    values[rng.random(values.shape) < 0.10] = 0.0
    formula = (
        "cat("
        "rolling_median(x, 67, min_periods=11),"
        "rolling_quantile(x, 67, q=.27, min_periods=11),"
        "rolling_pct_rank(x, 67, min_periods=11),"
        "rolling_kth(x, 67, k=4, ignore=\"NAN 0\", min_periods=4),"
        "rolling_prev_diff(x, 67),"
        "rolling_entropy(x, 67, buckets=7, min_periods=5))"
    )
    runtime = compile_formula(formula, {"x": values}, n_instruments=lanes)
    actual = _run(
        runtime, tmp_path / "large-rolling.bin", (rows, lanes, 6)
    )
    expected = np.full_like(actual, np.nan)
    for row in range(rows):
        start = max(0, row + 1 - periods)
        for lane in range(lanes):
            window = values[start : row + 1, lane]
            finite = window[np.isfinite(window)]
            if finite.size >= 11:
                expected[row, lane, 0] = np.quantile(finite, 0.5)
                expected[row, lane, 1] = np.quantile(finite, 0.27)
                if np.isfinite(values[row, lane]):
                    expected[row, lane, 2] = (
                        np.count_nonzero(finite <= values[row, lane])
                        / (finite.size + 1.0)
                    )
            backfill = window[np.isfinite(window) & (window != 0.0)]
            if backfill.size >= 4:
                expected[row, lane, 3] = backfill[-4]
            current = values[row, lane]
            if np.isfinite(current):
                prior = window[:-1][
                    np.isfinite(window[:-1]) & (window[:-1] != current)
                ]
                if prior.size:
                    expected[row, lane, 4] = prior[-1]
            if finite.size >= 5:
                minimum, maximum = np.min(finite), np.max(finite)
                if minimum == maximum:
                    expected[row, lane, 5] = 0.0
                else:
                    bucket = np.minimum(
                        6,
                        ((finite - minimum) * (7.0 / (maximum - minimum))).astype(
                            np.int64
                        ),
                    )
                    counts = np.bincount(bucket, minlength=7)
                    probabilities = counts[counts > 0] / finite.size
                    expected[row, lane, 5] = -np.sum(
                        probabilities * np.log(probabilities)
                    )
    np.testing.assert_allclose(
        actual, expected, rtol=3e-12, atol=3e-12, equal_nan=True
    )


def test_previous_different_adaptive_run_state_handles_expiry_and_nan(
    tmp_path: Path,
) -> None:
    values = np.array(
        [
            *([1.0] * 5),
            *([2.0] * 24),
            np.nan,
            *([2.0] * 12),
            *([3.0] * 20),
            2.0,
        ],
        dtype=np.float64,
    )[:, None]
    periods = 17
    runtime = compile_formula(
        "rolling_prev_diff(x, periods=17)",
        {"x": values},
        n_instruments=1,
    )
    actual = _run(runtime, tmp_path / "prev-diff-runs.bin", values.shape)
    expected = np.full_like(values, np.nan)
    for row, current in enumerate(values[:, 0]):
        if not np.isfinite(current):
            continue
        start = max(0, row + 1 - periods)
        prior = values[start:row, 0]
        different = prior[np.isfinite(prior) & (prior != current)]
        if different.size:
            expected[row, 0] = different[-1]
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_large_shared_control_expression_has_bounded_inline_work(tmp_path):
    """Do not expand a shared decision DAG into exponentially large C++ trees."""
    from trading_dsl_engine.base.dsl import var, where
    from trading_dsl_engine.cpp_stream.python.frontend import compile_ir
    from trading_dsl_engine.cpp_stream.python.lowering_full import lower_program
    # Reused predicates magnify lazy control-flow work, even with IR CSE.
    x = var('x')
    expression = x
    for i in range(10):
        expression = where(expression > (i / 100.), expression + .01, expression - .01)
    ir = compile_ir(expression, n_instruments=3)
    # Lowering is enough for the red test; do not invoke an unbounded C++ compile.
    plan = lower_program(ir, n_instruments=3, default_group_capacity=128,
                      key_cardinalities=None)
    def work(source):
        return (1 + sum(work(p) for p in source.parts)
                if source.kind in {'expression', 'stateless_expression'} else 0)
    largest = max(work(source) for stage in plan.stages for source in stage.inputs)
    assert largest <= 256, largest
    data = np.linspace(-.3, .3, 90).reshape(30, 3)
    expected = data.copy()
    for i in range(10):
        expected = np.where(expected > i / 100., expected + .01, expected - .01)
    runtime = compile_formula(expression, {'x': data})
    actual = runtime.run(out_path=tmp_path / 'bounded-control.npy').load()
    np.testing.assert_allclose(actual, expected, atol=1e-15)
