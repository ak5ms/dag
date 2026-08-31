from __future__ import annotations

from types import SimpleNamespace

import cvxpy as cp
import numpy as np

from examples import cpp_stream_mpo_one_pass as example


def _shape(shape):
    return SimpleNamespace(shape=shape)


def test_mpo_preserves_nmv_across_all_planned_horizons() -> None:
    n_horizons = len(example.HORIZONS)
    n_assets = 3
    problem = example.MPO.factory(
        expected_returns=_shape((n_horizons, n_assets)),
        half_spread=_shape((n_assets,)),
        current_weights=_shape((n_assets,)),
        risk_factor_0=_shape((n_assets, n_assets)),
        risk_factor_1=_shape((n_assets, n_assets)),
        risk_factor_2=_shape((n_assets, n_assets)),
        risk_factor_3=_shape((n_assets, n_assets)),
        risk_factor_4=_shape((n_assets, n_assets)),
        risk_factor_5=_shape((n_assets, n_assets)),
        risk_factor_6=_shape((n_assets, n_assets)),
        risk_factor_7=_shape((n_assets, n_assets)),
        is_tradable=_shape((n_assets,)),
    )
    params = {p.name(): p for p in problem.parameters()}
    current = np.array([0.2, -0.1, 0.3])
    params["expected_returns"].value = np.tile([1.0, 0.2, -0.4], (n_horizons, 1))
    params["half_spread"].value = np.full(n_assets, 1e-4)
    params["current_weights"].value = current
    for h in range(n_horizons):
        params[f"risk_factor_{h}"].value = np.eye(n_assets)
    params["is_tradable"].value = np.ones(n_assets)
    params["risk_radius"].value = 1.0

    problem.solve(solver=cp.CLARABEL)
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    weights = next(v for v in problem.variables() if v.name() == "weights").value

    np.testing.assert_allclose(
        weights.sum(axis=1),
        current.sum(),
        rtol=0.0,
        atol=1e-7,
    )
