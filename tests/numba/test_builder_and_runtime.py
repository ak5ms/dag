import numpy as np
import pytest
from numba import boolean, float64
from numba.experimental import jitclass

from trading_dsl_engine import build_engine, compile_formula, run_batch_from_mapping, update_from_mapping
from trading_dsl_engine.base.registry import CompiledNode, OpSpec, REGISTRY, TypeInfo


def _manual_formula(close, open_, span):
    ratio = close / open_
    alpha = 2.0 / (span + 1.0)
    s = ratio[0].copy()
    out = []
    for t in range(ratio.shape[0]):
        if t == 0:
            s = ratio[t]
        else:
            s = alpha * ratio[t] + (1 - alpha) * s
        row = np.empty((s.shape[0], 1), dtype=np.float64)
        order = np.argsort(s)
        pos = 0
        while pos < s.shape[0]:
            start = pos
            v = s[order[pos]]
            pos += 1
            while pos < s.shape[0] and s[order[pos]] == v:
                pos += 1
            rank = pos / s.shape[0]
            for k in range(start, pos):
                row[order[k], 0] = rank
        out.append(row)
    return np.array(out)[:, :, 0]




def _reference_weighted_snapshot(x, y, w):
    valid_x = np.isfinite(x)
    valid_y = np.isfinite(y)
    valid_w = np.isfinite(w)
    x0 = np.where(valid_x, x, 0.0)
    y0 = np.where(valid_y, y, 0.0)
    w0 = np.where(valid_w, w, 0.0)

    if w.ndim == 1:
        xw = x0 * w0[:, None]
        xx_new = x0.T @ xw
        xx_counts = valid_x.astype(np.int64).T @ (valid_x & valid_w[:, None]).astype(np.int64)
        xx_valid = xx_counts > 0
        xy_new = x0.T @ (w0 * y0)
        xy_counts = valid_x.astype(np.int64).T @ (valid_y & valid_w).astype(np.int64)
        xy_valid = xy_counts > 0
        return xx_new, xy_new, xx_valid, xy_valid

    n_features = x.shape[1]
    xx_new = np.zeros((n_features, n_features), dtype=np.float64)
    xx_valid = np.zeros((n_features, n_features), dtype=bool)
    xy_new = np.zeros(n_features, dtype=np.float64)
    xy_valid = np.zeros(n_features, dtype=bool)
    for j in range(n_features):
        xy_mask = valid_x[:, [j]] & valid_w & valid_y[None, :]
        xy_new[j] = np.sum(x0[:, [j]] * w0 * y0[None, :])
        xy_valid[j] = xy_mask.any()
        for k in range(n_features):
            xx_mask = valid_x[:, [j]] & valid_w & valid_x[:, k][None, :]
            xx_new[j, k] = np.sum(x0[:, [j]] * w0 * x0[:, k][None, :])
            xx_valid[j, k] = xx_mask.any()
    return xx_new, xy_new, xx_valid, xy_valid


def _reference_online_ewm_ridge(features, y, weights, hl, ridge):
    n_steps = y.shape[0]
    n_features = len(features)
    rho = 0.0 if hl <= 0.0 or np.isnan(hl) else np.exp(np.log(0.5) / hl)
    alpha = float(np.clip(1.0 - rho, 0.0, 1.0))
    xx = np.zeros((n_features, n_features), dtype=np.float64)
    xy = np.zeros(n_features, dtype=np.float64)
    last_xx = np.zeros((n_features, n_features), dtype=np.int64)
    last_xy = np.zeros(n_features, dtype=np.int64)
    has_xx = np.zeros((n_features, n_features), dtype=bool)
    has_xy = np.zeros(n_features, dtype=bool)
    beta = np.zeros(n_features, dtype=np.float64)
    out = np.empty((n_steps, n_features), dtype=np.float64)

    for t in range(n_steps):
        x = np.column_stack([feature[t] for feature in features])
        xx_new, xy_new, xx_valid, xy_valid = _reference_weighted_snapshot(x, y[t], weights[t])

        for j in range(n_features):
            if xy_valid[j]:
                if has_xy[j]:
                    a = alpha ** (t - last_xy[j])
                    xy[j] = xy[j] * (1.0 - a) + xy_new[j] * a
                else:
                    xy[j] = xy_new[j]
                    has_xy[j] = True
                last_xy[j] = t
            for k in range(n_features):
                if xx_valid[j, k]:
                    if has_xx[j, k]:
                        a = alpha ** (t - last_xx[j, k])
                        xx[j, k] = xx[j, k] * (1.0 - a) + xx_new[j, k] * a
                    else:
                        xx[j, k] = xx_new[j, k]
                        has_xx[j, k] = True
                    last_xx[j, k] = t

        xx = 0.5 * (xx + xx.T)
        last_xx = np.maximum(last_xx, last_xx.T)
        has_xx |= has_xx.T

        system = xx + ridge * np.diag(np.diag(xx))
        try:
            beta = np.linalg.solve(system, xy)
        except np.linalg.LinAlgError:
            beta = beta.copy()
        out[t] = beta
    return out


def test_compile_collects_inputs_and_runs_formula():
    c = compile_formula("xs_rank(ewm(div(close, open), 21))")
    assert c.input_names == ("close", "open")
    eng = build_engine("xs_rank(ewm(div(close, open), 21))")

    close = np.array([[10.0, 20.0, 30.0], [11.0, 22.0, 29.0], [12.0, 24.0, 28.0]])
    open_ = np.array([[5.0, 10.0, 15.0], [5.0, 11.0, 14.5], [6.0, 12.0, 14.0]])
    got = run_batch_from_mapping(eng, {"close": close, "open": open_})
    want = _manual_formula(close, open_, 21)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_streaming_state_persists_across_updates():
    eng = build_engine("ewm(div(close, open), 3)")
    y1 = update_from_mapping(eng, {"close": np.array([10.0, 20.0]), "open": np.array([5.0, 10.0])})
    y2 = update_from_mapping(eng, {"close": np.array([14.0, 18.0]), "open": np.array([7.0, 9.0])})
    np.testing.assert_allclose(y1[:, 0], np.array([2.0, 2.0]))
    np.testing.assert_allclose(y2[:, 0], np.array([2.0, 2.0]))


def test_shape_vector_and_matrix_emits():
    vec = build_engine("add(close, 1)")
    yv = update_from_mapping(vec, {"close": np.array([1.0, 2.0, 3.0])})
    assert yv.shape == (3, 1)

    mat = build_engine("outer(close)")
    ym = update_from_mapping(mat, {"close": np.array([1.0, 2.0, 3.0])})
    assert ym.shape == (3, 3)


def test_bspline_emits_matrix_with_expected_width():
    eng = build_engine("bspline(close, 7)")
    y = update_from_mapping(eng, {"close": np.array([0.0, 0.5, 1.0])})
    assert y.shape == (3, 7)
    assert np.isfinite(y).all()


def test_bspline_matches_periodic_reference_values():
    eng = build_engine("bspline(close, 96)")
    day_us = 24 * 60 * 60 * 1_000_000
    t_us = np.arange(0, day_us, 1_000_000, dtype=np.int64)
    tod = (t_us % day_us) / day_us
    close = tod.reshape(-1, 1).astype(np.float64)
    out = run_batch_from_mapping(eng, {"close": close}, out_path=None)[:, 0, :]

    k = out.shape[1]
    centers = np.linspace(0.0, 1.0, k, endpoint=False)
    sigma = 1.0 / k
    diff = np.abs(tod[:, None] - centers[None, :])
    circ_diff = np.minimum(diff, 1.0 - diff)
    ref = np.exp(-0.5 * (circ_diff / sigma) ** 2)
    ref /= ref.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_matrix_batch_output_supports_non_square_width():
    eng = build_engine("bspline(close, 5)")
    close = np.array([[0.0, 0.2, 0.6], [0.1, 0.4, 0.9]], dtype=np.float64)
    out = run_batch_from_mapping(eng, {"close": close}, out_path=None)
    assert out.shape == (2, 3, 5)


def test_col_unstacks_matrix_feature_for_ridge_input():
    formula = "get_beta(Ridge(col(bspline(close, 5), 0), col(bspline(close, 5), 1), target, weights, 4, 0.1))"
    eng = build_engine(formula)
    close = np.array([[0.1, 0.3], [0.2, 0.6], [0.4, 0.7]], dtype=np.float64)
    target = np.array([[1.0, 1.2], [1.1, 1.4], [1.5, 1.8]], dtype=np.float64)
    weights = np.ones_like(target)
    out = run_batch_from_mapping(eng, {"close": close, "target": target, "weights": weights}, out_path=None)
    assert out.shape == (3, 2)


def test_ridge_accepts_matrix_feature_without_manual_col_unstack():
    formula = "get_beta(Ridge(bspline(close, 5), target, weights, 4, 0.1))"
    eng = build_engine(formula)
    close = np.array([[0.1, 0.3], [0.2, 0.6], [0.4, 0.7]], dtype=np.float64)
    target = np.array([[1.0, 1.2], [1.1, 1.4], [1.5, 1.8]], dtype=np.float64)
    weights = np.ones_like(target)
    out = run_batch_from_mapping(eng, {"close": close, "target": target, "weights": weights}, out_path=None)
    assert out.shape == (3, 5)


def test_compiled_formula_and_engine_are_jitclasses():
    compiled = compile_formula("add(close, 1)")
    assert hasattr(compiled.compiled, "_numba_type_")
    eng = build_engine("add(close, 1)")
    assert hasattr(eng, "_numba_type_")


def test_nan_handling_div_ewm_and_xs_rank():
    eng = build_engine("xs_rank(ewm(div(close, open), 3))")
    close = np.array([[10.0, np.nan, 30.0], [12.0, 24.0, np.nan]], dtype=np.float64)
    open_ = np.array([[5.0, 10.0, 15.0], [6.0, 12.0, 15.0]], dtype=np.float64)

    out = run_batch_from_mapping(eng, {"close": close, "open": open_})
    assert np.isnan(out[0, 1])
    assert not np.isnan(out[1, 2])


def test_rolling_quantile_matches_reference_and_streams():
    eng = build_engine("rolling_quantile(close, 3, 0.5)")
    close = np.array(
        [
            [1.0, np.nan],
            [3.0, 2.0],
            [2.0, 1.0],
            [5.0, np.nan],
        ],
        dtype=np.float64,
    )
    out = run_batch_from_mapping(eng, {"close": close}, out_path=None)
    expected = np.array(
        [
            [1.0, np.nan],
            [2.0, 2.0],
            [2.0, 1.5],
            [3.0, 1.5],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out, expected, equal_nan=True)

    eng_stream = build_engine("rolling_quantile(close, 3, 0.25)")
    y1 = update_from_mapping(eng_stream, {"close": np.array([1.0, 2.0])}).copy()
    y2 = update_from_mapping(eng_stream, {"close": np.array([5.0, np.nan])}).copy()
    y3 = update_from_mapping(eng_stream, {"close": np.array([3.0, 4.0])}).copy()
    np.testing.assert_allclose(y1[:, 0], np.array([1.0, 2.0]), equal_nan=True)
    np.testing.assert_allclose(y2[:, 0], np.array([2.0, 2.0]), equal_nan=True)
    np.testing.assert_allclose(y3[:, 0], np.array([2.0, 2.5]), equal_nan=True)


def test_chunked_batch_matches_full_batch():
    eng1 = build_engine("ewm(div(close, open), 5)")
    eng2 = build_engine("ewm(div(close, open), 5)")
    rng = np.random.default_rng(0)
    close = rng.uniform(1, 2, size=(200, 8)).astype(np.float64)
    open_ = rng.uniform(1, 2, size=(200, 8)).astype(np.float64)

    full = run_batch_from_mapping(eng1, {"close": close, "open": open_}, chunk_size=1000)
    chunked = run_batch_from_mapping(eng2, {"close": close, "open": open_}, chunk_size=31)
    np.testing.assert_allclose(full, chunked)


def test_batch_path_does_not_require_np_stack(monkeypatch):
    eng = build_engine("add(close, 1)")
    close = np.arange(24, dtype=np.float64).reshape(6, 4)

    def _boom(*args, **kwargs):
        raise AssertionError("np.stack should not be used in batch path")

    monkeypatch.setattr(np, "stack", _boom)
    out = run_batch_from_mapping(eng, {"close": close}, chunk_size=2)
    np.testing.assert_allclose(out, close + 1.0)


def test_vector_batch_output_is_2d():
    vec = build_engine("add(close, 1)")
    close = np.arange(12, dtype=np.float64).reshape(3, 4)

    vec_out = run_batch_from_mapping(vec, {"close": close})

    assert vec_out.shape == (3, 4)
    np.testing.assert_allclose(vec_out, close + 1.0)


def test_batch_output_defaults_to_disk_memmap():
    vec = build_engine("add(close, 1)")
    close = np.arange(12, dtype=np.float64).reshape(3, 4)
    vec_out = run_batch_from_mapping(vec, {"close": close})
    assert isinstance(vec_out, np.memmap)
    np.testing.assert_allclose(vec_out, close + 1.0)


def test_batch_output_supports_in_memory_when_out_path_none():
    vec = build_engine("add(close, 1)")
    close = np.arange(12, dtype=np.float64).reshape(3, 4)
    vec_out = run_batch_from_mapping(vec, {"close": close}, out_path=None)
    assert not isinstance(vec_out, np.memmap)
    np.testing.assert_allclose(vec_out, close + 1.0)


def test_compile_stats_reports_cse_cache_hits():
    compiled = compile_formula("add(div(close, open), div(close, open))")
    assert compiled.stats.cache_hits > 0
    assert compiled.stats.expanded_nodes > 0
    assert compiled.stats.compile_seconds >= 0.0


def test_object_emitter_can_be_consumed_by_downstream_array_op():
    object_type = TypeInfo("object")

    state_spec = [("mean", float64)]

    @jitclass(state_spec)
    class MeanState:
        def __init__(self):
            self.mean = np.nan

    def mean_state_validator(types):
        if len(types) != 1 or types[0].kind != "vector":
            raise ValueError("mean_state_obj expects one vector arg")
        return object_type

    def mean_state_builder(children, literals):
        src = children[0]
        spec = [("src", src.instance_type), ("state", MeanState.class_type.instance_type)]

        @jitclass(spec)
        class MeanStateObjOp:
            def __init__(self, src):
                self.src = src
                self.state = MeanState()

            def on_data(self, frame2d):
                self.src.on_data(frame2d)
                x = self.src.emit()
                total = 0.0
                n = x.shape[0]
                for i in range(n):
                    total += x[i, 0]
                self.state.mean = total / n

            def emit(self):
                return self.state

        return CompiledNode(object_type, MeanStateObjOp.class_type.instance_type, lambda: MeanStateObjOp(src.ctor()))

    def get_mean_state_validator(types):
        if len(types) != 1 or types[0].kind != "object":
            raise ValueError("get_mean_state expects one object arg")
        return TypeInfo("vector")

    def get_mean_state_builder(children, literals):
        src = children[0]
        spec = [
            ("src", src.instance_type),
            ("initialized", boolean),
            ("out", float64[:, :]),
        ]

        @jitclass(spec)
        class GetMeanStateOp:
            def __init__(self, src):
                self.src = src
                self.initialized = False
                self.out = np.empty((1, 1), dtype=np.float64)

            def on_data(self, frame2d):
                self.src.on_data(frame2d)
                state = self.src.emit()
                row = frame2d[0]
                if (not self.initialized) or self.out.shape[0] != row.shape[0]:
                    self.out = np.empty((row.shape[0], 1), dtype=np.float64)
                    self.initialized = True
                for i in range(row.shape[0]):
                    self.out[i, 0] = state.mean

            def emit(self):
                return self.out

        return CompiledNode(TypeInfo("vector"), GetMeanStateOp.class_type.instance_type, lambda: GetMeanStateOp(src.ctor()))

    REGISTRY.register(OpSpec(name="mean_state_obj", validator=mean_state_validator, builder=mean_state_builder))
    REGISTRY.register(OpSpec(name="get_mean_state", validator=get_mean_state_validator, builder=get_mean_state_builder))

    eng = build_engine("get_mean_state(mean_state_obj(close))")
    close = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=np.float64)
    out = run_batch_from_mapping(eng, {"close": close}, out_path=None)

    expected = np.array([[1.5, 1.5], [4.0, 4.0], [9.0, 9.0]])
    np.testing.assert_allclose(out, expected)


def test_ridge_pairwise_nan_ewm_matches_numpy_reference():
    rng = np.random.default_rng(42)
    t = 8
    n = 5
    close = rng.normal(size=(t, n))
    open_ = rng.normal(size=(t, n))
    volume = rng.normal(size=(t, n))
    target = rng.normal(size=(t, n))
    weights = rng.uniform(0.5, 2.0, size=(t, n))

    # Exercise random NaNs, complete feature outages, complete weight outages,
    # and pairwise-only valid updates where one feature is missing but another
    # feature can still update its own sufficient statistics.
    close[1, [0, 3]] = np.nan
    open_[2, :] = np.nan
    volume[3, [1, 2, 4]] = np.nan
    target[4, [0, 2]] = np.nan
    weights[5, :] = np.nan
    close[6, 0] = np.nan
    open_[6, 1:] = np.nan

    formula = "get_beta(Ridge(close, open, volume, target, weights, 4, 0.1))"
    eng = build_engine(formula)
    actual = np.empty((t, 3), dtype=np.float64)
    for i in range(t):
        actual[i] = update_from_mapping(
            eng,
            {
                "close": close[i],
                "open": open_[i],
                "volume": volume[i],
                "target": target[i],
                "weights": weights[i],
            },
        )[:, 0]

    expected = _reference_online_ewm_ridge([close, open_, volume], target, weights, hl=4.0, ridge=0.1)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert np.isfinite(actual).all()


def _periodic_bspline_reference(values, n_basis):
    centers = np.linspace(0.0, 1.0, n_basis, endpoint=False)
    sigma = 1.0 / n_basis
    out = np.empty((values.shape[0], n_basis), dtype=np.float64)
    for i, value in enumerate(values):
        if np.isnan(value):
            out[i, :] = np.nan
            continue
        diff = np.abs(np.clip(value, 0.0, 1.0) - centers)
        circ_diff = np.minimum(diff, 1.0 - diff)
        row = np.exp(-0.5 * (circ_diff / sigma) ** 2)
        out[i] = row / row.sum()
    return out


def test_ridge_accepts_matrix_weights_and_matches_numpy_reference():
    close = np.array([[0.2, 0.8], [0.4, np.nan], [0.7, 0.1], [0.9, 0.3]], dtype=np.float64)
    open_ = np.array([[1.1, 1.4], [1.3, 1.5], [np.nan, 1.7], [1.8, 1.2]], dtype=np.float64)
    target = np.array([[0.5, 1.0], [0.7, 1.1], [0.8, np.nan], [1.0, 1.4]], dtype=np.float64)
    weight_key = np.array([[0.0, 0.5], [0.25, 0.75], [np.nan, 0.4], [0.6, 0.1]], dtype=np.float64)
    matrix_weights = np.stack([_periodic_bspline_reference(row, 2) for row in weight_key])

    eng = build_engine("get_beta(Ridge(close, open, target, bspline(weight_key, 2), 4, 0.1))")
    actual = run_batch_from_mapping(
        eng,
        {"close": close, "open": open_, "target": target, "weight_key": weight_key},
        out_path=None,
    )

    expected = _reference_online_ewm_ridge([close, open_], target, matrix_weights, hl=4.0, ridge=0.1)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_ridge_supports_variable_feature_arity_and_batch_beta_shape():
    formula_preds = "get_preds(Ridge(close, open, volume, target, weights, 4, 0.1))"
    formula_beta = "get_beta(Ridge(close, open, volume, target, weights, 4, 0.1))"
    eng_preds = build_engine(formula_preds)
    eng_beta = build_engine(formula_beta)

    t0_preds = update_from_mapping(
        eng_preds,
        {
            "close": np.array([1.0, 2.0]),
            "open": np.array([2.0, 3.0]),
            "volume": np.array([10.0, 11.0]),
            "target": np.array([5.0, 8.0]),
            "weights": np.array([1.0, 2.0]),
        },
    )
    t0_beta = update_from_mapping(
        eng_beta,
        {
            "close": np.array([1.0, 2.0]),
            "open": np.array([2.0, 3.0]),
            "volume": np.array([10.0, 11.0]),
            "target": np.array([5.0, 8.0]),
            "weights": np.array([1.0, 2.0]),
        },
    )
    np.testing.assert_allclose(t0_preds[:, 0], np.array([0.0, 0.0]))
    assert t0_beta.shape == (3, 1)

    t1_close = np.array([1.5, 2.5])
    t1_open = np.array([1.0, 2.0])
    t1_volume = np.array([12.0, 13.0])
    t1_target = np.array([3.0, 5.0])
    t1_preds = update_from_mapping(
        eng_preds,
        {
            "close": t1_close,
            "open": t1_open,
            "volume": t1_volume,
            "target": t1_target,
            "weights": np.array([1.0, 2.0]),
        },
    )
    expected = t0_beta[0, 0] * t1_close + t0_beta[1, 0] * t1_open + t0_beta[2, 0] * t1_volume
    np.testing.assert_allclose(t1_preds[:, 0], expected)

    close = np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]], dtype=np.float64)
    open_ = np.array([[2.0, 3.0], [2.1, 3.1], [2.2, 3.2]], dtype=np.float64)
    volume = np.array([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]], dtype=np.float64)
    target = np.array([[5.0, 8.0], [5.5, 8.5], [6.0, 9.0]], dtype=np.float64)
    eng_beta_batch = build_engine(formula_beta)
    out_beta = np.empty((close.shape[0], 3), dtype=np.float64)
    out = run_batch_from_mapping(
        eng_beta_batch,
        {
            "close": close,
            "open": open_,
            "volume": volume,
            "target": target,
            "weights": np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]], dtype=np.float64),
        },
        out=out_beta,
    )
    assert out.shape == (3, 3)

def test_groupby_with_nested_ewm_by_minute_of_day_matches_reference():
    formula = "groupby(mod(mod(ts, 86400000000), 60000000), ewm(close, 3))"
    eng = build_engine(formula)

    close = np.array(
        [
            [10.0, 20.0],
            [12.0, 18.0],
            [14.0, np.nan],
            [16.0, 14.0],
        ],
        dtype=np.float64,
    )
    ts = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [60_000_000.0, 60_000_000.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    out = run_batch_from_mapping(eng, {"ts": ts, "close": close}, out_path=None)

    alpha = 2.0 / (3.0 + 1.0)
    state = {}
    expected = np.empty_like(close)
    for t in range(close.shape[0]):
        for i in range(close.shape[1]):
            bucket = int((ts[t, i] % 86_400_000_000.0) % 60_000_000.0)
            key = (i, bucket)
            x = close[t, i]
            if key not in state:
                state[key] = x
            else:
                s = state[key]
                if np.isnan(x):
                    state[key] = s
                elif np.isnan(s):
                    state[key] = x
                else:
                    state[key] = alpha * x + (1.0 - alpha) * s
            expected[t, i] = state[key]

    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_groupby_supports_mixed_keys_within_tick_per_instrument():
    eng = build_engine("groupby(ts, ewm(close, 3))")

    first = update_from_mapping(
        eng,
        {"ts": np.array([0.0, 1.0], dtype=np.float64), "close": np.array([1.0, 10.0], dtype=np.float64)},
    ).copy()
    second = update_from_mapping(
        eng,
        {"ts": np.array([1.0, 0.0], dtype=np.float64), "close": np.array([3.0, 20.0], dtype=np.float64)},
    ).copy()
    third = update_from_mapping(
        eng,
        {"ts": np.array([0.0, 1.0], dtype=np.float64), "close": np.array([5.0, 30.0], dtype=np.float64)},
    ).copy()

    np.testing.assert_allclose(first[:, 0], np.array([1.0, 10.0]))
    np.testing.assert_allclose(second[:, 0], np.array([3.0, 20.0]))
    np.testing.assert_allclose(third[:, 0], np.array([3.0, 20.0]))


def test_groupby_supports_more_than_256_groups():
    eng = build_engine("groupby(ts, ewm(close, 3))")

    for key in range(300):
        out = update_from_mapping(
            eng,
            {"ts": np.array([float(key)], dtype=np.float64), "close": np.array([float(key + 1)], dtype=np.float64)},
        )
        np.testing.assert_allclose(out[:, 0], np.array([float(key + 1)]))

    out = update_from_mapping(
        eng,
        {"ts": np.array([0.0], dtype=np.float64), "close": np.array([5.0], dtype=np.float64)},
    )
    np.testing.assert_allclose(out[:, 0], np.array([3.0]))


def test_groupby_can_box_nested_ridge_slots_without_pickling_error():
    eng = build_engine("groupby(ts, get_beta(Ridge(x, y, w, 2, 0)))")

    assert eng.input_names == ("ts", "x", "y", "w")



def test_groupby_can_construct_nested_ridge_formula_from_compiled_path():
    formula = "groupby(ts, cumsum(get_preds(Ridge(x, y, w, 2, 0))))"
    eng = build_engine(formula)

    out = run_batch_from_mapping(
        eng,
        {
            "ts": np.arange(5.0, dtype=np.float64).reshape(5, 1),
            "x": np.arange(10.0, 15.0, dtype=np.float64).reshape(5, 1),
            "y": np.arange(1.0, 6.0, dtype=np.float64).reshape(5, 1),
            "w": np.ones((5, 1), dtype=np.float64),
        },
        out_path=None,
    )

    assert out.shape == (5, 1)
    assert np.all(np.isfinite(out))


def test_logical_eq_ne_and_mul_ops():
    eng = build_engine("and(eq(close, open), ne(volume, 0))")
    out = run_batch_from_mapping(
        eng,
        {
            "close": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "open": np.array([[1.0, 9.0], [3.0, 4.0]]),
            "volume": np.array([[10.0, 0.0], [1.0, 2.0]]),
        },
        out_path=None,
    )
    np.testing.assert_allclose(out, np.array([[1.0, 0.0], [1.0, 1.0]]), equal_nan=True)


def test_where_selects_true_false_branches():
    eng = build_engine("where(and(eq(close, open), ne(volume, 0)), mul(close, 2), 1)")
    out = run_batch_from_mapping(
        eng,
        {
            "close": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "open": np.array([[1.0, 9.0], [3.0, 4.0]]),
            "volume": np.array([[10.0, 0.0], [1.0, 2.0]]),
        },
        out_path=None,
    )
    np.testing.assert_allclose(out, np.array([[2.0, 1.0], [6.0, 8.0]]), equal_nan=True)


def test_logical_or_and_xor_ops():
    eng = build_engine("xor(or(close, 1), 0)")
    out = run_batch_from_mapping(eng, {"close": np.array([[0.0, 2.0]])}, out_path=None)
    np.testing.assert_allclose(out, np.array([[1.0, 1.0]]), equal_nan=True)


def test_isnan_and_abs_streaming():
    eng = build_engine("abs(where(isnan(close), 0, close))")
    y1 = update_from_mapping(eng, {"close": np.array([1.0, np.nan])}).copy()
    y2 = update_from_mapping(eng, {"close": np.array([-2.0, 4.0])}).copy()
    np.testing.assert_allclose(y1[:, 0], np.array([1.0, 0.0]), equal_nan=True)
    np.testing.assert_allclose(y2[:, 0], np.array([2.0, 4.0]), equal_nan=True)


def test_cumsum_shift_streaming():
    eng = build_engine("shift(cumsum(close), 1, 1)")
    y1 = update_from_mapping(eng, {"close": np.array([1.0, np.nan])}).copy()
    y2 = update_from_mapping(eng, {"close": np.array([-2.0, 4.0])}).copy()
    y3 = update_from_mapping(eng, {"close": np.array([3.0, -1.0])}).copy()

    np.testing.assert_allclose(y1[:, 0], np.array([np.nan, np.nan]), equal_nan=True)
    np.testing.assert_allclose(y2[:, 0], np.array([1.0, 0.0]), equal_nan=True)
    np.testing.assert_allclose(y3[:, 0], np.array([-1.0, 4.0]), equal_nan=True)


def test_shift_uses_static_max_size_for_lag_capacity():
    eng = build_engine("shift(close, 2, 2)")
    y1 = update_from_mapping(eng, {"close": np.array([1.0, 10.0])}).copy()
    y2 = update_from_mapping(eng, {"close": np.array([2.0, 20.0])}).copy()
    y3 = update_from_mapping(eng, {"close": np.array([3.0, 30.0])}).copy()

    np.testing.assert_allclose(y1[:, 0], np.array([np.nan, np.nan]), equal_nan=True)
    np.testing.assert_allclose(y2[:, 0], np.array([np.nan, np.nan]), equal_nan=True)
    np.testing.assert_allclose(y3[:, 0], np.array([1.0, 10.0]), equal_nan=True)


def test_diff_dsl_function_defaults_to_one_lag():
    eng = build_engine("diff(close)")
    y1 = update_from_mapping(eng, {"close": np.array([1.0, 2.0])}).copy()
    y2 = update_from_mapping(eng, {"close": np.array([3.0, 5.0])}).copy()

    np.testing.assert_allclose(y1[:, 0], np.array([np.nan, np.nan]), equal_nan=True)
    np.testing.assert_allclose(y2[:, 0], np.array([2.0, 3.0]), equal_nan=True)


def test_diff_dsl_function_accepts_lag_and_max_size():
    eng = build_engine("diff(close, 2, 2)")
    for values in (np.array([1.0, 10.0]), np.array([2.0, 20.0])):
        np.testing.assert_allclose(
            update_from_mapping(eng, {"close": values})[:, 0],
            np.array([np.nan, np.nan]),
            equal_nan=True,
        )
    y3 = update_from_mapping(eng, {"close": np.array([4.0, 25.0])}).copy()

    np.testing.assert_allclose(y3[:, 0], np.array([3.0, 15.0]), equal_nan=True)


def test_ridge_defaults_weights_to_one_and_broadcasts_scalar_weights():
    from trading_dsl_engine import Ridge, get_beta, var

    close = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]], dtype=np.float64)
    open_ = np.array([[2.0, 1.0, 0.5], [2.2, 1.2, 0.7], [2.4, 1.4, 0.9]], dtype=np.float64)
    target = np.array([[1.0, 1.5, 2.0], [1.2, 1.7, 2.2], [1.4, 1.9, 2.4]], dtype=np.float64)
    ones = np.ones_like(target)
    scalar_weights = np.full_like(target, 0.3)

    expected_one_feature = _reference_online_ewm_ridge([close], target, ones, hl=3.0, ridge=0.1)
    expected_two_features = _reference_online_ewm_ridge([close, open_], target, ones, hl=3.0, ridge=0.1)
    expected_scalar_weights = _reference_online_ewm_ridge([close, open_], target, scalar_weights, hl=3.0, ridge=0.1)
    string_engine = build_engine("get_beta(Ridge(close, target, 3, 0.1))")
    python_engine = build_engine(
        get_beta(Ridge(var("close"), var("open"), y=var("target"), hl=3.0, lambda_=0.1))
    )
    scalar_weight_engine = build_engine(
        get_beta(Ridge(var("close"), var("open"), y=var("target"), weights=0.3, hl=3.0, lambda_=0.1))
    )

    data = {"close": close, "open": open_, "target": target}
    np.testing.assert_allclose(run_batch_from_mapping(string_engine, data, out_path=None), expected_one_feature)
    np.testing.assert_allclose(run_batch_from_mapping(python_engine, data, out_path=None), expected_two_features)
    np.testing.assert_allclose(run_batch_from_mapping(scalar_weight_engine, data, out_path=None), expected_scalar_weights)


def test_dynamic_groupby_mean_uses_single_instrument_keyed_scope():
    eng = build_engine("groupby(ts, mean(close))")
    close = np.array([[1.0, 10.0], [3.0, 20.0]], dtype=np.float64)
    ts = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

    out = run_batch_from_mapping(eng, {"ts": ts, "close": close}, out_path=None)

    np.testing.assert_allclose(out, close)


def test_universe_groupby_mean_broadcasts_column_group_results():
    from trading_dsl_engine import groupby, mean, univ, var

    formula = groupby(univ(["6E", "6C"], ["6A"]), mean(var("close")))
    eng = build_engine(formula, column_names=["6E", "6C", "6A"])
    close = np.array(
        [
            [1.0, 3.0, 10.0],
            [2.0, np.nan, 20.0],
        ],
        dtype=np.float64,
    )

    out = run_batch_from_mapping(eng, {"close": close}, out_path=None)

    expected = np.array(
        [
            [2.0, 2.0, 10.0],
            [2.0, 2.0, 20.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_universe_groupby_string_formula_uses_column_names():
    eng = build_engine('groupby(univ(["6E", "6C"], ["6A"]), mean(close))', column_names=["6E", "6C", "6A"])
    close = np.array([[4.0, 8.0, 1.0]], dtype=np.float64)

    out = run_batch_from_mapping(eng, {"close": close}, out_path=None)

    np.testing.assert_allclose(out, np.array([[6.0, 6.0, 1.0]], dtype=np.float64))


def test_universe_groupby_preserves_state_per_column_group():
    from trading_dsl_engine import ewm, groupby, univ, var

    formula = groupby(univ([0, 1], [2]), ewm(var("close"), 3.0))
    eng = build_engine(formula)
    close = np.array(
        [
            [10.0, 20.0, 100.0],
            [12.0, 18.0, 200.0],
        ],
        dtype=np.float64,
    )

    out = run_batch_from_mapping(eng, {"close": close}, out_path=None)

    np.testing.assert_allclose(out[0], close[0])
    np.testing.assert_allclose(out[1], np.array([11.0, 19.0, 150.0], dtype=np.float64))



def test_tuple_key_groupby_combines_dynamic_keys():
    eng = build_engine("groupby((day, bucket), cumsum(close))")
    data = {
        "day": np.array([[1.0], [1.0], [1.0], [1.0]], dtype=np.float64),
        "bucket": np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float64),
        "close": np.array([[1.0], [10.0], [2.0], [20.0]], dtype=np.float64),
    }

    out = run_batch_from_mapping(eng, data, out_path=None)

    np.testing.assert_allclose(out, np.array([[1.0], [10.0], [3.0], [30.0]], dtype=np.float64))


def test_tuple_key_with_universe_groups_columns_before_dynamic_key():
    eng = build_engine("groupby((univ([0, 1]), ts), mean(close))")
    data = {
        "ts": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        "close": np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]], dtype=np.float64),
    }

    out = run_batch_from_mapping(eng, data, out_path=None)

    np.testing.assert_allclose(out, np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [6.0, 6.0]]))


def test_tuple_key_with_single_column_universes_preserves_column_local_grouping():
    eng = build_engine("groupby((univ([0], [1]), ts), mean(close))")
    data = {
        "ts": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        "close": np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]], dtype=np.float64),
    }

    out = run_batch_from_mapping(eng, data, out_path=None)

    np.testing.assert_allclose(out, data["close"])


def test_groupby_lhs_form_computes_lhs_outside_keyed_op_state():
    eng = build_engine("groupby(key, cumsum(x), cumsum(self_))")

    out = run_batch_from_mapping(
        eng,
        {
            "key": np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float64),
            "x": np.ones((4, 1), dtype=np.float64),
        },
        out_path=None,
    )

    np.testing.assert_allclose(out[:, 0], np.array([1.0, 2.0, 4.0, 6.0]))


def test_groupby_lhs_form_supports_other_stateful_ops():
    eng = build_engine("groupby(key, cumsum(x), ewm(self_, 3))")

    out = run_batch_from_mapping(
        eng,
        {
            "key": np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float64),
            "x": np.ones((4, 1), dtype=np.float64),
        },
        out_path=None,
    )

    np.testing.assert_allclose(out[:, 0], np.array([1.0, 2.0, 2.0, 3.0]))


def test_grouped_expr_apply_sugar_matches_three_arg_groupby():
    import trading_dsl_engine as tde
    from trading_dsl_engine import cumsum, var

    formula = cumsum(var("x")).groupby(var("key")).apply(cumsum(tde.self_))
    eng = build_engine(formula)

    out = run_batch_from_mapping(
        eng,
        {
            "key": np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float64),
            "x": np.ones((4, 1), dtype=np.float64),
        },
        out_path=None,
    )

    np.testing.assert_allclose(out[:, 0], np.array([1.0, 2.0, 4.0, 6.0]))


def test_grouped_expr_apply_accepts_nary_callable_args():
    from trading_dsl_engine import add, var

    formula = var("x").groupby(var("key")).apply(add, 2.0)
    out = run_batch_from_mapping(
        build_engine(formula),
        {
            "key": np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float64),
            "x": np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64),
        },
        out_path=None,
    )

    np.testing.assert_allclose(out[:, 0], np.array([3.0, 4.0, 5.0, 6.0]))


def test_universe_groupby_output_can_feed_keyed_apply_with_self_placeholder():
    import trading_dsl_engine as tde
    from trading_dsl_engine import cumsum, groupby, mean, univ, var

    lhs = groupby(univ([0, 1], [2]), mean(var("x")))
    formula = lhs.groupby(var("key")).apply(cumsum(tde.self_))
    out = run_batch_from_mapping(
        build_engine(formula),
        {
            "key": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=np.float64,
            ),
            "x": np.array(
                [
                    [1.0, 2.0, 10.0],
                    [3.0, 4.0, 20.0],
                    [5.0, 6.0, 30.0],
                    [7.0, 8.0, 40.0],
                ],
                dtype=np.float64,
            ),
        },
        out_path=None,
    )

    np.testing.assert_allclose(
        out,
        np.array(
            [
                [1.5, 1.5, 10.0],
                [3.5, 3.5, 20.0],
                [7.0, 7.0, 40.0],
                [11.0, 11.0, 60.0],
            ],
            dtype=np.float64,
        ),
    )


def test_grouped_expr_operator_method_sugar_matches_three_arg_groupby():
    from trading_dsl_engine import cumsum, var

    formula = cumsum(var("x")).groupby(var("key")).cumsum()
    eng = build_engine(formula)

    out = run_batch_from_mapping(
        eng,
        {
            "key": np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float64),
            "x": np.ones((4, 1), dtype=np.float64),
        },
        out_path=None,
    )

    np.testing.assert_allclose(out[:, 0], np.array([1.0, 2.0, 4.0, 6.0]))


def test_ffill_with_limit_matches_reference_and_streams():
    eng = build_engine("ffill(close, 2)")
    close = np.array(
        [
            [1.0, np.nan, 5.0],
            [np.nan, 2.0, np.nan],
            [np.nan, np.nan, np.nan],
            [4.0, np.nan, np.nan],
            [np.nan, np.nan, 9.0],
            [np.nan, 8.0, np.nan],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [1.0, np.nan, 5.0],
            [1.0, 2.0, 5.0],
            [1.0, 2.0, 5.0],
            [4.0, 2.0, np.nan],
            [4.0, np.nan, 9.0],
            [4.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )

    out_batch = run_batch_from_mapping(eng, {"close": close}, out_path=None)
    np.testing.assert_allclose(out_batch, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    eng_stream = build_engine("ffill(close, 2)")
    out_stream = np.vstack(
        [update_from_mapping(eng_stream, {"close": close[t]})[:, 0].copy() for t in range(close.shape[0])]
    )
    np.testing.assert_allclose(out_stream, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
