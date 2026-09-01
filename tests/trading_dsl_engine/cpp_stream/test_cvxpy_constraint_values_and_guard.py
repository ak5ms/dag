from __future__ import annotations

import cvxpy as cp

from trading_dsl_engine.cpp_stream.optimizer import cvxpy_program
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
