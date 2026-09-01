from __future__ import annotations

import json
from pathlib import Path

import cvxpy as cp
import numpy as np

from trading_dsl_engine.base.dsl import var
from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.cpp_stream.optimizer import (
    build_current_clarabel,
    cvxpy_program,
    get_field,
)
from trading_dsl_engine.ir.types import SCALAR, VECTOR


def test_constraint_value_request_does_not_add_a_solver_primal() -> None:
    @cvxpy_program
    def RiskTarget(target, radius=0.5) -> cp.Problem:
        target = cp.Parameter(target.shape, name="target")
        radius = cp.Parameter(name="radius", nonneg=True)
        weights = cp.Variable(target.shape, name="weights")
        risk = cp.SOC(radius, weights)
        risk.set_label("risk")
        return cp.Problem(cp.Minimize(cp.sum_squares(weights - target)), [risk])

    prototype = RiskTarget.resolve_for_types(
        {"target": VECTOR, "radius": SCALAR},
        requested_fields=frozenset({"risk.value"}),
        n_instruments=None,
    )

    assert [primal.name for primal in prototype.primals] == ["weights"]
    field = prototype.resolve_field("risk.value")
    assert field.kind == "constraint_value"
    assert field.logical_shape == (prototype.instrument_count + 1,)


def test_native_constraint_value_is_evaluated_after_the_original_solve(
    tmp_path: Path,
    monkeypatch,
) -> None:
    native = build_current_clarabel(cache_dir=tmp_path / "clarabel-native")

    @cvxpy_program(
        cache_dir=tmp_path / "program-cache",
        clarabel=native,
    )
    def RiskTarget(target, risk_factor, radius=0.5) -> cp.Problem:
        target = cp.Parameter(target.shape, name="target")
        risk_factor = cp.Parameter(risk_factor.shape, name="risk_factor")
        radius = cp.Parameter(name="radius", nonneg=True)
        weights = cp.Variable(target.shape, name="weights")
        risk = cp.SOC(radius, risk_factor @ weights)
        risk.set_label("risk")
        return cp.Problem(cp.Minimize(cp.sum_squares(weights - target)), [risk])

    rows, assets = 7, 3
    rng = np.random.default_rng(812)
    data = {
        "target": rng.normal(scale=0.04, size=(rows, assets)),
        "risk_factor": np.broadcast_to(
            np.eye(assets), (rows, assets, assets)
        ).copy(),
    }
    mpo = RiskTarget(target=var("target"), risk_factor=var("risk_factor"))
    fields = [
        get_field(mpo, "weights"),
        get_field(mpo, "risk.value"),
        get_field(mpo, "status"),
    ]
    monkeypatch.setenv(
        "TRADING_DSL_ENGINE_CPP_STREAM_CACHE", str(tmp_path / "runner-cache")
    )
    runtime = compile_formula(fields, data, n_instruments=assets)

    generated = runtime.generated_cpp.read_text()
    assert "ClarabelResultKind::ConstraintValue" in generated
    assert "cpp_stream_constraint_value" not in generated

    manifests = tuple(
        (tmp_path / "program-cache").rglob("clarabel_program_manifest.json")
    )
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["schema_version"] == 5
    assert [item["name"] for item in manifest["primals"]] == ["weights"]
    assert [item["constraint_index"] for item in manifest["constraint_values"]] == [0]

    weights, risk_values, statuses = runtime.run(
        out_path=tmp_path / "postsolve-values.npy"
    ).load(mmap_mode=None)
    expected = np.concatenate(
        [
            np.full((rows, 1), 0.5),
            np.einsum("rij,rj->ri", data["risk_factor"], weights),
        ],
        axis=1,
    )
    np.testing.assert_allclose(risk_values, expected, rtol=2e-6, atol=2e-8)
    assert np.isin(np.asarray(statuses).reshape(-1), [1.0, 4.0]).all()
