import time

import jax
import numpy as np

from trading_dsl_engine.jax_flat import compile_formula, compile_formula_cpp


def _assert_cpp_matches_jax(formula, data, *, rtol=1e-10, atol=1e-10):
    cpp_runtime = compile_formula_cpp(formula)
    jax_runtime = compile_formula(formula)
    _, cpp_out = cpp_runtime.run_batch_tick(data)
    _, jax_out = jax_runtime.run_batch(data)
    np.testing.assert_allclose(cpp_out, np.asarray(jax_out), rtol=rtol, atol=atol, equal_nan=True)


def test_cpp_flat_stateless_chain_matches_jax_flat():
    rows = 32
    cols = 5
    close = np.linspace(-2.0, 3.0, rows * cols, dtype=np.float64).reshape(rows, cols)
    open_ = np.linspace(1.0, 4.0, rows * cols, dtype=np.float64).reshape(rows, cols)
    close[3, 2] = np.nan
    open_[7, 1] = np.nan
    _assert_cpp_matches_jax(
        "xstd(add(abs(close), div(exp(fraction(open)), add(abs(close), 1.0))))",
        {"close": close, "open": open_},
    )


def test_cpp_flat_more_stateless_and_matrix_ops_match_jax_flat():
    rows = 24
    cols = 4
    close = np.linspace(-1.0, 1.0, rows * cols, dtype=np.float64).reshape(rows, cols)
    open_ = np.flip(close, axis=1) + 0.25
    close[4, 2] = np.nan
    open_[6, 1] = np.nan
    data = {"close": close, "open": open_}
    for formula in (
        "cat(close, open, bspline(fillna(close, 0.25), 4))",
        "col(cat(close, open, bspline(fillna(close, 0.25), 4)), 5)",
        "xs_rank(add(close, open))",
        "xs_sort(add(close, open))",
        "mean(add(close, open))",
        "where(gt(open, close), open, close)",
        "add(add(ln(abs(add(close, 2.0))), ceil(open)), add(floor(close), round(open)))",
        "add(add(sign(close), arctan(open)), add(isnan(close), purify(close)))",
        "add(add(mod(close, 0.7), pow(abs(open), 2.0)), floordiv(open, 0.5))",
        "add(add(and(gt(open, close), lt(close, open)), or(eq(open, close), ne(open, close))), xor(gt(open, close), lt(open, close)))",
    ):
        _assert_cpp_matches_jax(formula, data)


def test_cpp_flat_stateful_cumsum_ewm_shift_ffill_matches_jax_flat():
    rows = 40
    cols = 4
    row = np.arange(rows, dtype=np.float64)[:, None]
    col = np.arange(cols, dtype=np.float64)[None, :]
    close = row * 0.25 + col
    lag = np.mod(row + col, 4.0)
    close[5, 1] = np.nan
    close[9, 3] = np.nan
    data = {"close": close, "lag": lag}
    _assert_cpp_matches_jax("add(cumsum(close), shift(ewm(ffill(close, 1), 3.0), lag, 5))", data)


def test_cpp_flat_ridge_projections_match_jax_flat():
    rows = 16
    cols = 5
    row = np.arange(rows, dtype=np.float64)[:, None]
    col = np.arange(cols, dtype=np.float64)[None, :]
    close = 0.2 * row + col
    open_ = 1.0 + 0.1 * row - 0.05 * col
    close[3, 2] = np.nan
    open_[7, 1] = np.nan
    data = {"close": close, "open": open_}
    _assert_cpp_matches_jax("get_preds(Ridge(cat(close, open), open, 1.0, 8.0, 0.01))", data)
    _assert_cpp_matches_jax("get_beta(Ridge(cat(close, open), open, 1.0, 8.0, 0.01))", data)


def test_cpp_flat_groupby_cumsum_matches_jax_flat():
    rows = 18
    cols = 5
    close = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) * 0.1
    key0 = np.mod(np.arange(rows, dtype=np.float64)[:, None] + np.arange(cols, dtype=np.float64)[None, :], 3.0)
    key1 = np.mod(np.arange(rows, dtype=np.float64)[:, None], 2.0) + np.zeros((rows, cols), dtype=np.float64)
    key0[4, 2] = np.nan
    close[7, 3] = np.nan
    _assert_cpp_matches_jax("groupby((key0, key1), close, cumsum(self_))", {"close": close, "key0": key0, "key1": key1})


def test_cpp_flat_tick_into_reuses_output_buffer():
    runtime = compile_formula_cpp("add(close, open)")
    state = runtime.init_state(3)
    out = np.empty(3, dtype=np.float64)
    close = np.array([1.0, 2.0, 3.0])
    open_ = np.array([10.0, 20.0, 30.0])
    runtime.tick_into(state, out, close, open_)
    np.testing.assert_allclose(out, [11.0, 22.0, 33.0])
    out_id = id(out)
    runtime.tick_into(state, out, close + 1.0, open_)
    assert id(out) == out_id
    np.testing.assert_allclose(out, [12.0, 23.0, 34.0])


def test_cpp_flat_micro_runtime_comparison_smoke(capsys):
    rows = 256
    cols = 9
    rng = np.random.default_rng(123)
    data = {
        "close": rng.normal(size=(rows, cols)),
        "open": rng.normal(size=(rows, cols)),
    }
    formula = "xstd(add(abs(close), div(exp(fraction(open)), add(abs(close), 1.0))))"
    cpp_runtime = compile_formula_cpp(formula)
    jax_runtime = compile_formula(formula)

    # Warm both runtimes; benchmark timings below intentionally exclude compile/setup time.
    cpp_runtime.run_batch_tick(data)
    jax_runtime.run_batch(data)
    jax.block_until_ready(jax_runtime.run_batch(data)[1])

    t0 = time.perf_counter()
    _, cpp_out = cpp_runtime.run_batch_tick(data)
    cpp_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, jax_out = jax_runtime.run_batch(data)
    jax.block_until_ready(jax_out)
    jax_elapsed = time.perf_counter() - t0

    print(f"cpp_flat_smoke cpp_tick_s={cpp_elapsed:.6f} jax_flat_batch_s={jax_elapsed:.6f}")
    np.testing.assert_allclose(cpp_out, np.asarray(jax_out), rtol=1e-10, atol=1e-10, equal_nan=True)
    captured = capsys.readouterr()
    assert "cpp_flat_smoke" in captured.out
