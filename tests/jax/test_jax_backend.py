import numpy as np
import pytest

jax_backend = pytest.importorskip("trading_dsl_engine.jax")
pytest.importorskip("equinox")

from trading_dsl_engine import build_engine as build_numba_engine
from trading_dsl_engine import run_batch_from_mapping as run_numba_batch
from trading_dsl_engine.jax import build_jax_engine, run_batch_from_mapping, update_from_mapping


def _compare_batch(formula, data, **kwargs):
    numba_engine = build_numba_engine(formula, **kwargs)
    jax_engine = build_jax_engine(formula, **kwargs)
    expected = run_numba_batch(numba_engine, data, out_path=None)
    actual = run_batch_from_mapping(jax_engine, data, out_path=None)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_jax_backend_matches_numba_for_stateless_and_stateful_vector_formula():
    close = np.array([[10.0, 20.0, 30.0], [11.0, 22.0, 29.0], [12.0, 24.0, 28.0]])
    open_ = np.array([[5.0, 10.0, 15.0], [5.0, 11.0, 14.5], [6.0, 12.0, 14.0]])
    _compare_batch("xs_rank(ewm(div(close, open), 21))", {"close": close, "open": open_})


def test_jax_backend_live_updates_are_jit_compiled_and_stateful():
    eng = build_jax_engine("ewm(div(close, open), 3)")
    y1 = update_from_mapping(eng, {"close": np.array([10.0, 20.0]), "open": np.array([5.0, 10.0])})
    y2 = update_from_mapping(eng, {"close": np.array([14.0, 18.0]), "open": np.array([7.0, 9.0])})
    np.testing.assert_allclose(y1, np.array([2.0, 2.0]))
    np.testing.assert_allclose(y2, np.array([2.0, 2.0]))
    assert eng._state is not None




def test_jax_backend_shared_stateful_subtree_is_evaluated_once_per_tick():
    close = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    # If ewm(close, 3) is evaluated twice in one tick, this diverges from 2 * ewm(close, 3).
    shared = run_batch_from_mapping(build_jax_engine("add(ewm(close, 3), ewm(close, 3))"), {"close": close}, out_path=None)
    baseline = run_batch_from_mapping(build_jax_engine("mul(2, ewm(close, 3))"), {"close": close}, out_path=None)
    np.testing.assert_allclose(shared, baseline, rtol=1e-10, atol=1e-10, equal_nan=True)
def test_jax_backend_matches_numba_for_matrix_and_column_ops():
    close = np.array([[0.0, 0.2, 0.6], [0.1, 0.4, 0.9]], dtype=np.float64)
    _compare_batch("bspline(close, 5)", {"close": close})
    _compare_batch("col(bspline(close, 5), 2)", {"close": close})


def test_jax_backend_universe_groupby_mean_matches_numba():
    close = np.array([[1.0, 2.0, 10.0], [3.0, np.nan, 12.0]], dtype=np.float64)
    _compare_batch("groupby(univ([0, 1], [2]), mean(close))", {"close": close})


def test_jax_backend_dynamic_key_groupby_matches_numba():
    close = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]], dtype=np.float64)
    ts = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    _compare_batch("groupby(ts, ewm(close, 3))", {"ts": ts, "close": close})


def test_jax_backend_dynamic_groupby_stateless_child_uses_keyed_singleton_scope():
    close = np.array([[1.0, 10.0], [3.0, 20.0]], dtype=np.float64)
    ts = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    _compare_batch("groupby(ts, mean(close))", {"ts": ts, "close": close})



def test_jax_backend_tuple_key_with_universe_matches_numba():
    data = {
        "ts": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        "close": np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]], dtype=np.float64),
    }
    _compare_batch("groupby((univ([0, 1]), ts), mean(close))", data)
    _compare_batch("groupby((univ([0], [1]), ts), cumsum(close))", data)


def test_jax_backend_scoped_groupby_apply_matches_numba():
    close = np.array([[1.0], [2.0], [10.0], [3.0]], dtype=np.float64)
    ts = np.array([[0.0], [0.0], [1.0], [0.0]], dtype=np.float64)
    _compare_batch("groupby(ts, close, cumsum(self_))", {"ts": ts, "close": close})


def test_jax_backend_ridge_projection_shapes_match_numba():
    data = {
        "x": np.arange(10.0, 15.0, dtype=np.float64).reshape(5, 1),
        "y": np.arange(1.0, 6.0, dtype=np.float64).reshape(5, 1),
        "w": np.ones((5, 1), dtype=np.float64),
        "ts": np.arange(5.0, dtype=np.float64).reshape(5, 1),
    }
    _compare_batch("get_beta(Ridge(x, y, w, 2, 0))", data)
    out = run_batch_from_mapping(build_jax_engine("groupby(ts, get_preds(Ridge(x, y, w, 2, 0)))"), data, out_path=None)
    assert out.shape == (5, 1)
    assert np.all(np.isfinite(out))


def test_jax_backend_groupby_nested_ridge_cumsum_matches_numba():
    data = {
        "ts": np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        "x": np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0]], dtype=np.float64),
        "y": np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]], dtype=np.float64),
        "w": np.ones((4, 2), dtype=np.float64),
    }
    _compare_batch("groupby(ts, cumsum(get_preds(Ridge(x, y, w, 2, 0))))", data)


def test_jax_backend_python_composition_infix_and_dsl_function_match_string():
    from trading_dsl_engine import ewm, var, xs_rank

    def some_op(x):
        return xs_rank(ewm((x**2.0 + 1.0) // 2.0, 3.0))

    close = var("close")
    open_ = var("open")
    python_engine = build_jax_engine(some_op(close + open_))
    string_engine = build_jax_engine("xs_rank(ewm(floordiv(add(pow(add(close, open), 2), 1), 2), 3))")

    data = {
        "close": np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float64),
        "open": np.array([[0.5, 1.0, 1.5], [0.7, 1.2, 1.7]], dtype=np.float64),
    }
    np.testing.assert_allclose(
        run_batch_from_mapping(python_engine, data, out_path=None),
        run_batch_from_mapping(string_engine, data, out_path=None),
    )


def test_jax_backend_multiline_python_groupby_apply_composition_matches_numba():
    import trading_dsl_engine as tde
    from trading_dsl_engine import cumsum, ewm, var

    def some_op(x):
        return ewm(x, 3.0)

    close = var("close")
    open_ = var("open")
    ev_ts = var("ev_ts")
    bucket = 2.0

    t1 = some_op(close + open_)
    t2 = t1.groupby(ev_ts // bucket).apply(cumsum(tde.self_ + 1.0))

    data = {
        "close": np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64),
        "open": np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float64),
        "ev_ts": np.array([[0.0], [1.0], [2.0], [0.0]], dtype=np.float64),
    }
    jax_out = run_batch_from_mapping(build_jax_engine(t2), data, out_path=None)
    numba_out = run_numba_batch(build_numba_engine(t2), data, out_path=None)
    np.testing.assert_allclose(jax_out, numba_out, equal_nan=True)


def test_jax_backend_reverse_python_math_magics_match_string():
    from trading_dsl_engine import var

    close = var("close")
    formula = (10.0 // close) + (2.0**close) + (3.0 > close) + (close < 2.0)
    string_formula = "add(add(add(floordiv(10, close), pow(2, close)), gt(3, close)), lt(close, 2))"
    data = {"close": np.array([[1.0, 2.0, 4.0]], dtype=np.float64)}
    np.testing.assert_allclose(
        run_batch_from_mapping(build_jax_engine(formula), data, out_path=None),
        run_batch_from_mapping(build_jax_engine(string_formula), data, out_path=None),
    )


@pytest.mark.parametrize(
    ("close", "expected"),
    [
        (
            np.array([[2.0, 2.0, 1.0]], dtype=np.float64),
            np.array([[1.0, 1.0, 1.0 / 3.0]], dtype=np.float64),
        ),
        (
            np.array([[np.nan, np.nan, np.nan]], dtype=np.float64),
            np.array([[np.nan, np.nan, np.nan]], dtype=np.float64),
        ),
        (
            np.array([[3.0, np.nan, 1.0, 3.0]], dtype=np.float64),
            np.array([[1.0, np.nan, 1.0 / 3.0, 1.0]], dtype=np.float64),
        ),
    ],
)
def test_jax_backend_xs_rank_ties_and_nan_masking(close, expected):
    out = run_batch_from_mapping(build_jax_engine("xs_rank(close)"), {"close": close}, out_path=None)
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
