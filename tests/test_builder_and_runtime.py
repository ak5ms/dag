import numpy as np

from trading_dsl_engine import compile_formula, StreamingFeatureEngine


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
        for i in range(s.shape[0]):
            row[i, 0] = np.count_nonzero(s <= s[i]) / s.shape[0]
        out.append(row)
    return np.array(out)


def test_compile_collects_inputs_and_runs_formula():
    c = compile_formula("xs_rank(ewm(div(close, open), 21))")
    assert c.input_names == ("close", "open")
    eng = StreamingFeatureEngine(c)

    close = np.array([[10.0, 20.0, 30.0], [11.0, 22.0, 29.0], [12.0, 24.0, 28.0]])
    open_ = np.array([[5.0, 10.0, 15.0], [5.0, 11.0, 14.5], [6.0, 12.0, 14.0]])
    got = eng.run_batch({"close": close, "open": open_})
    want = _manual_formula(close, open_, 21)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_streaming_state_persists_across_updates():
    eng = StreamingFeatureEngine.from_formula("ewm(div(close, open), 3)")
    y1 = eng.update({"close": np.array([10.0, 20.0]), "open": np.array([5.0, 10.0])})
    y2 = eng.update({"close": np.array([14.0, 18.0]), "open": np.array([7.0, 9.0])})
    np.testing.assert_allclose(y1[:, 0], np.array([2.0, 2.0]))
    # alpha=0.5, prev=2, new ratios=[2,2]
    np.testing.assert_allclose(y2[:, 0], np.array([2.0, 2.0]))


def test_shape_vector_and_matrix_emits():
    vec = StreamingFeatureEngine.from_formula("add(close, 1)")
    yv = vec.update({"close": np.array([1.0, 2.0, 3.0])})
    assert yv.shape == (3, 1)

    mat = StreamingFeatureEngine.from_formula("outer(close)")
    ym = mat.update({"close": np.array([1.0, 2.0, 3.0])})
    assert ym.shape == (3, 3)
