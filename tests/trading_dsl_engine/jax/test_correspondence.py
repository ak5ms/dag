import numpy as np
import pytest

pytest.importorskip("trading_dsl_engine.jax")
pytest.importorskip("equinox")

from trading_dsl_engine import build_engine as build_numba_engine
from trading_dsl_engine import run_batch_from_mapping as run_numba_batch
from trading_dsl_engine.jax import build_jax_engine, run_batch_from_mapping as run_jax_batch


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
def test_jax_corresponds_to_numba_runtime_cases(case_name, formula, data):
    if case_name == "grouped_method_sugar":
        from trading_dsl_engine import cumsum, var

        formula = cumsum(var("x")).groupby((var("key"),)).cumsum()
    numba_engine = build_numba_engine(formula)
    jax_engine = build_jax_engine(formula)
    np.testing.assert_allclose(
        run_jax_batch(jax_engine, data, out_path=None),
        run_numba_batch(numba_engine, data, out_path=None),
        equal_nan=True,
        rtol=1e-10,
        atol=1e-10,
    )
