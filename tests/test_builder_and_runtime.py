import numpy as np
import pytest
from numba import boolean, float64
from numba.experimental import jitclass

from trading_dsl_engine import build_engine, compile_formula, run_batch_from_mapping, update_from_mapping
from trading_dsl_engine.registry import CompiledNode, OpSpec, REGISTRY, TypeInfo


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


def test_ridge_supports_variable_feature_arity_and_batch_beta_shape():
    formula_preds = "get_preds(Ridge(close, open, volume, target, 4, 0.1))"
    formula_beta = "get_beta(Ridge(close, open, volume, target, 4, 0.1))"
    eng_preds = build_engine(formula_preds)
    eng_beta = build_engine(formula_beta)

    t0_preds = update_from_mapping(
        eng_preds,
        {
            "close": np.array([1.0, 2.0]),
            "open": np.array([2.0, 3.0]),
            "volume": np.array([10.0, 11.0]),
            "target": np.array([5.0, 8.0]),
        },
    )
    t0_beta = update_from_mapping(
        eng_beta,
        {
            "close": np.array([1.0, 2.0]),
            "open": np.array([2.0, 3.0]),
            "volume": np.array([10.0, 11.0]),
            "target": np.array([5.0, 8.0]),
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
        {"close": t1_close, "open": t1_open, "volume": t1_volume, "target": t1_target},
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
        {"close": close, "open": open_, "volume": volume, "target": target},
        out=out_beta,
    )
    assert out.shape == (3, 3)
