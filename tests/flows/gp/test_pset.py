from __future__ import annotations

import ast
from pathlib import Path

import pytest
from deap import gp

from flows.gp import (
    ALL_DSL_OPERATOR_NAMES,
    AxisSpec,
    BoolParam,
    BoolRow,
    DimensionlessRow,
    DurationRow,
    EXCLUDED_DSL_OPERATOR_NAMES,
    EXPECTED_DSL_OPERATOR_NAMES,
    EXPECTED_GP_OPERATOR_NAMES,
    ExprValue,
    GPConfig,
    NumericRow,
    NumericTensor,
    PeriodAtLeastTwo,
    PositiveInt,
    PositiveNumber,
    PriceRow,
    QuantileParam,
    REGRESSION_PROJECTIONS,
    TimestampRow,
    make_pset,
    make_toolbox,
    primitive_names_for_operator,
    random_formula,
)
from trading_dsl_engine.base import dsl
from trading_dsl_engine.base.parser import Call, Expr, Number
from trading_dsl_engine.cpp_stream.python.frontend import compile_ir


def _primitives(pset, family):
    return [pset.mapping[name] for name in primitive_names_for_operator(pset, family)]


def _primitive(pset, family, args):
    return next(value for value in _primitives(pset, family) if tuple(value.args) == args)


def test_complete_family_coverage_and_exclusions():
    pset = make_pset()
    assert pset.gp_operator_families == EXPECTED_GP_OPERATOR_NAMES
    assert EXPECTED_DSL_OPERATOR_NAMES == ALL_DSL_OPERATOR_NAMES - EXCLUDED_DSL_OPERATOR_NAMES
    for family in EXPECTED_GP_OPERATOR_NAMES:
        assert primitive_names_for_operator(pset, family), family
    for family in EXCLUDED_DSL_OPERATOR_NAMES:
        assert not primitive_names_for_operator(pset, family), family
    unique = {
        primitive.name: primitive
        for values in pset.primitives.values()
        for primitive in values
    }
    assert all(
        issubclass(primitive.ret, (NumericRow, NumericTensor))
        for primitive in unique.values()
    )


def test_boolean_rows_and_tensors_are_numeric_masks():
    assert issubclass(BoolRow, DimensionlessRow)
    assert issubclass(BoolRow, ExprValue)
    assert not issubclass(BoolParam, ExprValue)
    pset = make_pset()
    row_results = [
        primitive for primitive in _primitives(pset, "lt")
        if issubclass(primitive.ret, NumericRow)
    ]
    assert row_results and all(primitive.ret is BoolRow for primitive in row_results)
    assert any(issubclass(primitive.ret, NumericTensor) for primitive in _primitives(pset, "lt"))


def test_scalar_terminals_do_not_masquerade_as_rows():
    assert not issubclass(PositiveNumber, ExprValue)
    pset = make_pset()
    price = PriceRow(dsl.var("ap0_out0"))
    primitive = _primitive(pset, "mul", (PriceRow, PositiveNumber))
    result = pset.context[primitive.name](price, PositiveInt(2))
    assert isinstance(result, PriceRow)
    assert isinstance(result.expr, Call) and result.expr.fn == "mul"
    assert isinstance(result.expr.args[1], Number)


def test_parameter_constraints_and_derived_windows():
    assert QuantileParam(0.0).value == 0.0
    assert QuantileParam(1.0).value == 1.0
    with pytest.raises(ValueError):
        QuantileParam(-0.01)
    assert PeriodAtLeastTwo(2).value == 2
    with pytest.raises(ValueError):
        PeriodAtLeastTwo(1)
    with pytest.raises(ValueError):
        AxisSpec(0)
    with pytest.raises(ValueError):
        GPConfig(axes=(0,))

    pset = make_pset()
    price = PriceRow(dsl.var("ap0_out0"))
    period = PositiveInt(20)
    rolling = _primitive(pset, "rolling_mean", (PriceRow, PositiveInt))
    assert dict(pset.context[rolling.name](price, period).expr.kwargs)["min_periods"].value == 20.0
    ewm = _primitive(pset, "ewm", (PriceRow, PositiveInt))
    assert dict(pset.context[ewm.name](price, period).expr.kwargs)["min_periods"].value == 20.0


def test_timestamp_diff_returns_duration():
    pset = make_pset()
    primitive = _primitive(pset, "diff", (TimestampRow, PositiveInt))
    result = pset.context[primitive.name](TimestampRow(dsl.var("_ev_ts")), PositiveInt(2))
    assert primitive.ret is DurationRow
    assert isinstance(result, DurationRow)
    compile_ir(result.expr)


def test_non_temporal_reductions_remain_rows():
    pset = make_pset()
    price = PriceRow(dsl.var("ap0_out0"))
    for family in ("sum", "mean", "std", "reduce_min", "reduce_max"):
        primitive = _primitive(pset, family, (PriceRow, AxisSpec, BoolParam))
        result = pset.context[primitive.name](price, AxisSpec(1), BoolParam(True))
        assert compile_ir(result.expr).nodes[-1].value_type.kind == "vector"


def test_row_regression_variants_still_lower_after_matrix_overloads():
    pset = make_pset()
    y = PriceRow(dsl.var("ap0_out0"))
    x = PriceRow(dsl.var("bp0_out0"))
    period = PositiveInt(20)
    temporal = [
        primitive for primitive in _primitives(pset, "ts_regression")
        if tuple(primitive.args) == (NumericRow, NumericRow, PositiveInt)
    ]
    assert len(temporal) == len(REGRESSION_PROJECTIONS)
    for primitive in temporal:
        assert compile_ir(pset.context[primitive.name](y, x, period).expr).nodes[-1].value_type.kind == "vector"
    for projection in REGRESSION_PROJECTIONS:
        rowwise = [
            primitive for primitive in _primitives(pset, f"ridge_{projection}")
            if all(type_ is NumericRow for type_ in primitive.args)
        ]
        assert len(rowwise) == 3


def test_generation_uses_standard_deap_toolbox():
    pset = make_pset()
    toolbox = make_toolbox(pset, min_depth=1, max_depth=3)
    assert toolbox.expr.func is gp.genHalfAndHalf
    for seed in range(12):
        tree, expr = random_formula(pset, min_depth=1, max_depth=4, seed=seed)
        assert isinstance(tree, gp.PrimitiveTree)
        assert isinstance(expr, Expr)


def test_gp_package_uses_only_absolute_imports():
    root = Path(__file__).resolve().parents[3] / "src" / "flows" / "gp"
    for path in root.glob("*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert not [
            node for node in ast.walk(module)
            if isinstance(node, ast.ImportFrom) and node.level
        ], path
