import numpy as np
import pytest

pytest.importorskip("trading_dsl_engine.jax_new")
pytest.importorskip("equinox")

from trading_dsl_engine import build_engine as build_numba_engine
from trading_dsl_engine import run_batch_from_mapping as run_numba_batch
from trading_dsl_engine.jax_new.engine import compile_formula as compile_jax_new_formula


def _run_jax_new_batch(formula, data):
    runtime = compile_jax_new_formula(formula)
    input_names = runtime.input_names if hasattr(runtime, "input_names") else runtime.program.input_names
    n_steps = np.asarray(data[input_names[0]]).shape[0]
    n_instruments = np.asarray(data[input_names[0]]).shape[1]
    compiled = runtime.compiled if hasattr(runtime, "compiled") else runtime
    state = compiled.init_state(n_instruments)
    out = []
    for t in range(n_steps):
        row_inputs = [np.asarray(data[name][t], dtype=np.float64) for name in input_names]
        if hasattr(runtime, "compiled"):
            state, tick_out = compiled.tick(state, np.stack(row_inputs, axis=0))
        else:
            state, tick_out = compiled.tick(state, *row_inputs)
        out.append(np.asarray(tick_out, dtype=np.float64))
    out_arr = np.asarray(out)
    if out_arr.ndim >= 3 and out_arr.shape[-1] == 1:
        out_arr = out_arr[..., 0]
    return out_arr


@pytest.mark.parametrize(
    ("case_name", "formula", "data"),
    [
        (
            "logical_eq_ne_where",
            "where(and(eq(close, open), ne(volume, 0)), mul(close, 2), 1)",
            {
                "close": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
                "open": np.array([[1.0, 9.0], [3.0, 4.0]], dtype=np.float64),
                "volume": np.array([[10.0, 0.0], [1.0, 2.0]], dtype=np.float64),
            },
        ),
        (
            "nan_abs_fill",
            "abs(where(isnan(close), 0, close))",
            {"close": np.array([[1.0, np.nan], [-2.0, 4.0]], dtype=np.float64)},
        ),
        (
            "cumsum_shift",
            "shift(cumsum(close), 1, 2)",
            {"close": np.array([[1.0, 2.0], [-2.0, 4.0], [3.0, 5.0]], dtype=np.float64)},
        ),
        (
            "rolling_quantile",
            "rolling_quantile(close, 3, 0.5)",
            {"close": np.array([[1.0, 3.0], [2.0, 2.0], [5.0, 1.0], [7.0, 4.0]], dtype=np.float64)},
        ),
        (
            "outer_matrix",
            "outer(close)",
            {"close": np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float64)},
        ),
        (
            "universe_groupby_state",
            "groupby(univ([0, 1], [2]), close, mean(ewm(self_, 3)))",
            {"close": np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]], dtype=np.float64)},
        ),
        (
            "grouped_method_sugar",
            None,
            {
                "key": np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float64),
                "x": np.ones((4, 1), dtype=np.float64),
            },
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_jax_new_corresponds_to_numba_runtime_cases(case_name, formula, data):
    if case_name == "grouped_method_sugar":
        pytest.xfail("jax_new groupby parity not ported yet")
        from trading_dsl_engine import cumsum, var

        formula = cumsum(var("x")).groupby((var("key"),)).cumsum()
    if case_name == "universe_groupby_state":
        pytest.xfail("jax_new universe/dynamic groupby parity not ported yet")
    numba_engine = build_numba_engine(formula)
    np.testing.assert_allclose(
        _run_jax_new_batch(formula, data),
        run_numba_batch(numba_engine, data, out_path=None),
        equal_nan=True,
        rtol=1e-10,
        atol=1e-10,
    )
