from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")
pytest.importorskip("clarabel")

from trading_dsl_engine.cpp_stream.optimizer import (
    CvxpyNodeBuild,
    constraint_dual,
    constraint_slack,
    cvxpy_node,
    expression_value,
    solver_metric,
    variable_value,
)


def _make_node(n: int = 3, h: int = 2):
    @cvxpy_node(
        outputs={
            "weights": variable_value("weights"),
            "first_weights": expression_value("first_weights"),
            "risk0_slack": constraint_slack("risk_0"),
            "turnover_dual": constraint_dual("turnover_pos"),
            "status": solver_metric("status"),
            "iterations": solver_metric("iterations"),
            "objective": solver_metric("objective"),
        },
        solver_settings={
            "max_iter": 200,
            "tol_gap_abs": 1e-8,
            "tol_gap_rel": 1e-8,
            "tol_feas": 1e-8,
        },
    )
    def node():
        er = cp.Parameter((h, n), name="er")
        hs = cp.Parameter((h, n), nonneg=True, name="hs")
        current = cp.Parameter(n, name="current")
        radius = cp.Parameter(h, nonneg=True, name="radius")
        factor = cp.Parameter((n, n), name="factor")
        weights = cp.Variable((h, n), name="weights")
        turnover = cp.Variable((h, n), name="turnover")
        delta = cp.vstack([weights[0] - current, weights[1:] - weights[:-1]])
        constraints = {
            "turnover_pos": turnover >= delta,
            "turnover_neg": turnover >= -delta,
        }
        for horizon in range(h):
            constraints[f"risk_{horizon}"] = cp.SOC(
                radius[horizon], factor @ weights[horizon]
            )
        problem = cp.Problem(
            cp.Maximize(
                cp.sum(
                    cp.multiply(er, weights)
                    - cp.multiply(hs, turnover)
                )
            ),
            list(constraints.values()),
        )
        return CvxpyNodeBuild(
            problem=problem,
            parameters={
                "er": er,
                "hs": hs,
                "current": current,
                "radius": radius,
                "factor": factor,
            },
            variables={"weights": weights, "turnover": turnover},
            constraints=constraints,
            expressions={"first_weights": weights[0]},
        )

    return node


def _data(seed: int, n: int = 3, h: int = 2):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, n))
    covariance = raw @ raw.T / n + np.eye(n) * 0.2
    return {
        "er": rng.normal(scale=0.01, size=(h, n)),
        "hs": np.full((h, n), 1e-4),
        "current": rng.normal(scale=0.01, size=n),
        "radius": np.full(h, 0.1),
        "factor": np.linalg.cholesky(covariance).T * (1.0 + seed * 1e-4),
    }


def _reference(node, values):
    build = node._fresh_build()
    for name, parameter in build.parameters.items():
        parameter.value = values[name]
    build.problem.solve(
        solver=cp.CLARABEL,
        max_iter=200,
        tol_gap_abs=1e-8,
        tol_gap_rel=1e-8,
        tol_feas=1e-8,
    )
    assert build.problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    return build


def test_changing_A_q_b_and_named_outputs_match_reference():
    node = _make_node()
    compiled = node.compile()
    for seed in range(3):
        values = _data(seed)
        actual = compiled.solve(**values)
        expected = _reference(node, values)
        np.testing.assert_allclose(
            actual["weights"],
            expected.variables["weights"].value,
            rtol=4e-6,
            atol=4e-7,
        )
        np.testing.assert_allclose(
            actual["first_weights"], actual["weights"][0], rtol=1e-8, atol=1e-9
        )
        assert str(actual["status"]) in {"Solved", "AlmostSolved"}
        assert int(actual["iterations"]) > 0
        assert np.asarray(actual["risk0_slack"]).shape == (4,)
        assert np.all(np.isfinite(actual["turnover_dual"]))


def test_parallel_batch_preserves_order_and_matches_serial():
    node = _make_node()
    cases = [_data(seed) for seed in range(8)]
    inputs = {
        name: np.stack([case[name] for case in cases]) for name in cases[0]
    }
    serial = node.compile(workers=1).solve_batch(inputs, workers=1)
    parallel = node.compile(workers=4).solve_batch(inputs, workers=4)
    assert serial.keys() == parallel.keys()
    for name in serial:
        if serial[name].dtype.kind in "USO":
            assert serial[name].tolist() == parallel[name].tolist()
        else:
            np.testing.assert_allclose(
                serial[name], parallel[name], rtol=4e-6, atol=4e-7
            )


def test_sequential_mode_rejects_parallel_workers():
    node = _make_node()
    cases = [_data(seed) for seed in range(2)]
    inputs = {
        name: np.stack([case[name] for case in cases]) for name in cases[0]
    }
    with pytest.raises(ValueError, match="sequential"):
        node.compile(workers=2).solve_batch(inputs, workers=2, sequential=True)


def test_multiple_node_instances_are_reentrant():
    node = _make_node()
    values = _data(17)
    left = node.compile(workers=2).solve(**values)
    right = node.compile(workers=2).solve(**values)
    np.testing.assert_allclose(
        left["weights"], right["weights"], rtol=2e-7, atol=2e-8
    )
