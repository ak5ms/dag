from __future__ import annotations

import numpy as np
import pandas as pd

from flows.riskmodel import risk_covariance
from trading_dsl_engine.base.dsl import var
from trading_dsl_engine.cpp_stream import compile_formula


def _unwrap_run(result):
    if isinstance(result, tuple):
        return result[-1]
    return result


def test_risk_covariance_cpp_stream_matrix_path_handles_nans_and_changes():
    rng = np.random.default_rng(42)
    returns = rng.normal(scale=0.01, size=(80, 4))
    returns[rng.random(returns.shape) < 0.12] = np.nan
    returns[rng.random(returns.shape) < 0.05] = 0.0
    expression = risk_covariance(var("returns"), span=7)
    try:
        runtime = compile_formula(expression, n_instruments=returns.shape[1])
    except TypeError:
        runtime = compile_formula(expression)
    actual = np.asarray(_unwrap_run(runtime.run_batch({"returns": returns})))

    observed = np.nan_to_num(returns, nan=0.0)
    products = np.einsum("ti,tj->tij", observed, observed)
    products[products == 0.0] = np.nan
    expected = (
        pd.DataFrame(products.reshape(len(returns), -1))
        .ewm(span=7, ignore_na=True, adjust=False)
        .mean()
        .to_numpy()
        .reshape(products.shape)
    )
    assert actual.shape == expected.shape
    np.testing.assert_allclose(
        actual, expected, rtol=2e-9, atol=2e-12, equal_nan=True
    )
    assert np.all(np.isfinite(actual[-1]))
