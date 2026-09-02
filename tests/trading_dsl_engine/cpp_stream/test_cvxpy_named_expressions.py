from __future__ import annotations

from pathlib import Path

import cvxpy as cp
import numpy as np

from trading_dsl_engine.base.dsl import var
from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.cpp_stream.optimizer import cvxpy_program, get_field


def test_factory_can_return_named_scalar_expression_without_adding_problem_rows(tmp_path: Path):
    @cvxpy_program(cache_dir=tmp_path / "named-expression")
    def program(target):
        target = cp.Parameter(target.shape, name="target")
        x = cp.Variable(target.shape, name="x")
        abs_loss = cp.sum(cp.abs(x - target))
        problem = cp.Problem(
            cp.Minimize(abs_loss + 0.1 * cp.sum_squares(x)),
            [x >= 0.0],
        )
        return problem, {"abs_loss": abs_loss}

    result = program.factory(np.zeros(3))
    assert isinstance(result, tuple) and len(result) == 2
    problem, named = result
    assert isinstance(problem, cp.Problem)
    assert set(named) == {"abs_loss"}
    assert [variable.name() for variable in problem.variables()] == ["x"]
    assert len(problem.constraints) == 1


def test_named_scalar_expression_projects_native_postsolve_value(tmp_path: Path):
    @cvxpy_program(cache_dir=tmp_path / "named-expression")
    def program(target):
        target = cp.Parameter(target.shape, name="target")
        x = cp.Variable(target.shape, name="x")
        abs_loss = cp.sum(cp.abs(x - target))
        problem = cp.Problem(
            cp.Minimize(abs_loss + 0.1 * cp.sum_squares(x)),
            [x >= 0.0],
        )
        return problem, {"abs_loss": abs_loss}

    data = {
        "target": np.array(
            [
                [-1.0, 2.0, -3.0],
                [0.5, -0.25, 1.25],
                [3.0, 2.0, 1.0],
            ],
            dtype=np.float64,
        )
    }
    call = program(var("target"))
    runtime = compile_formula(
        {
            "x": get_field(call, "x"),
            "abs_loss": get_field(call, "abs_loss"),
            "objective": get_field(call, "objective"),
        },
        data,
        n_instruments=3,
    )
    values = runtime.run().load(mmap_mode=None)

    expected_x = []
    expected_loss = []
    expected_objective = []
    for row in data["target"]:
        target = cp.Parameter(3, name="target")
        target.value = row
        x = cp.Variable(3, name="x")
        abs_loss = cp.sum(cp.abs(x - target))
        reference = cp.Problem(
            cp.Minimize(abs_loss + 0.1 * cp.sum_squares(x)),
            [x >= 0.0],
        )
        reference.solve(
            solver=cp.CLARABEL,
            presolve_enable=False,
            tol_gap_abs=1e-10,
            tol_gap_rel=1e-10,
            tol_feas=1e-10,
        )
        expected_x.append(np.asarray(x.value))
        expected_loss.append(float(abs_loss.value))
        expected_objective.append(float(reference.value))

    np.testing.assert_allclose(values["x"], np.asarray(expected_x), rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(values["abs_loss"], expected_loss, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        values["objective"], expected_objective, rtol=2e-6, atol=2e-7
    )

    generated = runtime.generated_cpp.read_text()
    assert generated.count("stackdsl::ClarabelNode<") == 1
    assert "ClarabelResultKind::ExpressionValue" in generated
