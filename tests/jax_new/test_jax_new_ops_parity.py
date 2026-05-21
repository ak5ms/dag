import numpy as np

from trading_dsl_engine.jax.engine import compile_formula as compile_old
from trading_dsl_engine.jax_new.engine import compile_formula as compile_new


def _run(runtime, data):
    compiled = runtime.compiled if hasattr(runtime, "compiled") else runtime
    input_names = runtime.input_names if hasattr(runtime, "input_names") else runtime.program.input_names
    state = compiled.init_state(len(next(iter(data.values()))[0]))
    out = []
    for t in range(len(next(iter(data.values())))):
        row_inputs = [np.asarray(data[name][t], dtype=np.float64) for name in input_names]
        if hasattr(runtime, "compiled"):
            state, tick_out = compiled.tick(state, np.stack(row_inputs, axis=0))
        else:
            state, tick_out = compiled.tick(state, *row_inputs)
        out.append(np.asarray(tick_out, dtype=np.float64))
    return np.asarray(out)


def _assert_formula_parity(formula, data):
    old_rt = compile_old(formula)
    new_rt = compile_new(formula)
    old_out = _run(old_rt, data)
    new_out = _run(new_rt, data)
    if new_out.ndim == old_out.ndim + 1 and new_out.shape[-1] == 1:
        new_out = new_out[..., 0]
    np.testing.assert_allclose(new_out, old_out, equal_nan=True, atol=1e-10, rtol=1e-10)


def test_jax_new_parity_shift_ffill_rolling_quantile():
    data = {
        "close": [
            [1.0, np.nan, 3.0],
            [2.0, np.nan, 6.0],
            [3.0, 2.0, np.nan],
            [4.0, np.nan, 9.0],
        ]
    }
    _assert_formula_parity("shift(close, 1, 3)", data)
    _assert_formula_parity("ffill(close, 1)", data)
    _assert_formula_parity("rolling_quantile(close, 3, 0.5)", data)


def test_jax_new_parity_boolean_aliases():
    data = {
        "open": [[1.0, 0.0, np.nan], [2.0, 1.0, 0.0]],
        "close": [[1.0, 1.0, 2.0], [0.0, 1.0, 1.0]],
    }
    _assert_formula_parity("and_(open, close)", data)
    _assert_formula_parity("or_(open, close)", data)


def test_jax_new_parity_ridge_and_groupby_formulas():
    import pytest

    pytest.xfail("jax_new ridge/groupby parity port is pending")
    ridge_data = {
        "x": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
        "y": [[1.2, 2.2], [2.1, 2.9], [2.9, 4.1], [4.2, 5.1]],
        "w": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    }
    _assert_formula_parity("get_beta(Ridge(x, y, w, 2, 0))", ridge_data)
    _assert_formula_parity("get_preds(Ridge(x, y, w, 2, 0))", ridge_data)

    groupby_data = {
        "ts": [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
        "x": [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],
    }
    _assert_formula_parity("groupby(ts, x, cumsum(self_))", groupby_data)
