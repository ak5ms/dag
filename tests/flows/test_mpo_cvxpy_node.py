from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cvxpy")
pytest.importorskip("clarabel")

from flows.mpo_cvxpy_node import make_mpo_cvxpy_node
from trading_dsl_engine.cpp_stream.optimizer import (
    constraint_dual,
    constraint_slack,
    expression_value,
    solver_metric,
)
from trading_dsl_engine.cpp_stream.optimizer.codegen import (
    generate_cvxpygen_artifact,
)


def _parameters(seed: int, n: int = 3, h: int = 2):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, n))
    covariance = raw @ raw.T / n + np.eye(n) * 0.3
    return {
        "expected_returns": rng.normal(scale=2e-3, size=(h, n)),
        "half_spread_bps": rng.uniform(0.5, 2.5, size=(h, n)),
        "current_weights": rng.normal(scale=0.01, size=n),
        "risk_radius": np.full(h, 0.1),
        "risk_factor": np.linalg.cholesky(covariance).T,
    }


def test_mpo_factory_uses_shared_changing_risk_factor_and_bps_costs():
    node = make_mpo_cvxpy_node(3, 2)
    compiled = node.compile()
    first = compiled.solve(**_parameters(1))
    second_params = _parameters(2)
    second = compiled.solve(**second_params)
    assert first["weights"].shape == (2, 3)
    assert second["weights"].shape == (2, 3)
    np.testing.assert_allclose(second["next_weights"], second["weights"][0])
    covariance = second_params["risk_factor"].T @ second_params["risk_factor"]
    risk = np.einsum(
        "hi,ij,hj->h", second["weights"], covariance, second["weights"]
    )
    assert np.all(risk <= np.square(second_params["risk_radius"]) * (1 + 2e-5))
    assert np.asarray(second["transaction_cost"]).item() >= 0.0


def test_factory_accepts_named_constraint_and_expression_outputs():
    outputs = {
        "next_weights": expression_value("next_weights"),
        "risk_1_dual": constraint_dual("risk_1"),
        "risk_1_cone": constraint_slack("risk_1"),
        "objective": solver_metric("objective"),
    }
    result = make_mpo_cvxpy_node(3, 2, outputs=outputs).compile().solve(
        **_parameters(4)
    )
    assert result["next_weights"].shape == (3,)
    assert np.asarray(result["risk_1_cone"]).shape == (4,)
    assert np.all(np.isfinite(result["risk_1_dual"]))
    assert np.isfinite(np.asarray(result["objective"]).item())


def test_cvxpygen_artifact_writes_static_parameter_output_manifest(tmp_path, monkeypatch):
    class FakeCPG:
        @staticmethod
        def generate_code(problem, *, code_dir, solver, wrapper):
            destination = tmp_path / "generated"
            assert str(destination.resolve()) == code_dir
            assert solver == "CLARABEL"
            assert wrapper is False
            destination.mkdir(parents=True)
            (destination / "generated.c").write_text("/* generated */\n")

    monkeypatch.setitem(sys.modules, "cvxpygen", SimpleNamespace(cpg=FakeCPG))
    node = make_mpo_cvxpy_node(3, 2)
    artifact = generate_cvxpygen_artifact(
        node,
        _parameters(8),
        tmp_path / "generated",
    )
    manifest = json.loads(artifact.manifest_path.read_text())
    assert manifest["update_regime"] == "A/q/b"
    assert manifest["parameters"]["risk_factor"]["shape"] == [3, 3]
    assert manifest["outputs"]["risk_0_dual"] == {
        "kind": "ConstraintDual",
        "source": "risk_0",
    }
    assert manifest["runtime"]["workspace_ownership"].startswith("one persistent")
