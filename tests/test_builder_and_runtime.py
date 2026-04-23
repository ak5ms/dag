import numpy as np

from trading_dsl_engine import build_engine, compile_formula, run_batch_from_mapping, update_from_mapping


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
    return np.array(out)


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
