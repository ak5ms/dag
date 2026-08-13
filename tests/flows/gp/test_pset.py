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
    CountRow,
    DimensionlessRow,
    DurationRow,
    EXCLUDED_DSL_OPERATOR_NAMES,
    EXPECTED_DSL_OPERATOR_NAMES,
    EXPECTED_GP_OPERATOR_NAMES,
    ExprValue,
    GP_COMPOSITE_OPERATOR_NAMES,
    GPConfig,
    NumericRow,
    PeriodAtLeastTwo,
    PositiveInt,
    PositiveNumber,
    PriceRow,
    QuantileParam,
    QuantityRow,
    REGRESSION_PROJECTIONS,
    ROWWISE_RIDGE_COMPOSITE_NAMES,
    TimestampRow,
    TradingDayHorizonRow,
    make_pset,
    make_toolbox,
    primitive_names_for_operator,
    random_formula,
)
from flows.riskminer.semantics import inputdata_alpha_terminal_metadata
from trading_dsl_engine.base import dsl
from trading_dsl_engine.base.parser import Call, Expr, Number
from trading_dsl_engine.cpp_stream.python.frontend import compile_ir


def _family_primitives(pset, family: str):
    return [pset.mapping[name] for name in primitive_names_for_operator(pset, family)]


def _primitive(pset, family: str, args: tuple[type, ...]):
    return next(
        value
        for value in _family_primitives(pset, family)
        if tuple(value.args) == args
    )


def test_all_allowed_dsl_and_row_preserving_composite_families_are_exposed():
    pset = make_pset()
    assert pset.gp_dsl_operator_families == EXPECTED_DSL_OPERATOR_NAMES
    assert pset.gp_composite_operator_families == GP_COMPOSITE_OPERATOR_NAMES
    assert pset.gp_operator_families == EXPECTED_GP_OPERATOR_NAMES
    assert EXPECTED_DSL_OPERATOR_NAMES == ALL_DSL_OPERATOR_NAMES - EXCLUDED_DSL_OPERATOR_NAMES
    assert ROWWISE_RIDGE_COMPOSITE_NAMES <= GP_COMPOSITE_OPERATOR_NAMES
    for operator in EXPECTED_GP_OPERATOR_NAMES:
        names = primitive_names_for_operator(pset, operator)
        assert names, operator
        assert all(pset.gp_primitive_family[name] == operator for name in names)


def test_raw_dimension_changing_or_unsupported_operators_are_not_exposed():
    pset = make_pset()
    required_exclusions = {
        "emit", "einsum", "groupby", "cat", "cache", "buffer", "outer",
        "bspline", "rbf_basis", "future_rbf_basis_sum", "col", "Ridge",
        "InstrumentBasisMean", "get_beta", "get_preds", "get_residuals",
        "get_sse", "get_sst", "get_r2", "get_residual_variance",
        "get_standard_errors", "get_standard_error", "get_tstats", "get_tstat",
        "get_effective_df", "get_effective_n", "get_coefficient", "xstd",
        "xs_sort", "xs_norm",
    }
    assert required_exclusions <= EXCLUDED_DSL_OPERATOR_NAMES
    for operator in EXCLUDED_DSL_OPERATOR_NAMES:
        assert not primitive_names_for_operator(pset, operator), operator

    unique_primitives = {
        primitive.name: primitive
        for primitives in pset.primitives.values()
        for primitive in primitives
    }
    assert unique_primitives
    assert all(issubclass(primitive.ret, NumericRow) for primitive in unique_primitives.values())


def test_inputdata_fields_are_typed_terminals():
    pset = make_pset()
    metadata = inputdata_alpha_terminal_metadata()
    assert set(pset.gp_field_terminals) == set(metadata)
    expected = {
        "_ev_ts": TimestampRow,
        "ap0_out0": PriceRow,
        "volume_a0_out0": QuantityRow,
        "is_tradable_out0": BoolRow,
        "trade_cross_pct_out0.count": CountRow,
        "vw_halfspread_out0": DimensionlessRow,
        "wdte_out0": TradingDayHorizonRow,
    }
    for field_name, return_type in expected.items():
        terminal = pset.mapping[pset.gp_field_terminals[field_name]]
        assert terminal.ret is return_type


def test_boolean_rows_are_numeric_masks_and_static_bools_are_not_rows():
    assert issubclass(BoolRow, DimensionlessRow)
    assert issubclass(BoolRow, ExprValue)
    assert not issubclass(BoolParam, ExprValue)
    pset = make_pset()
    assert all(primitive.ret is BoolRow for primitive in _family_primitives(pset, "lt"))
    assert any(primitive.args[0] is BoolRow for primitive in _family_primitives(pset, "where"))


def test_numeric_constants_are_static_and_only_broadcast_through_typed_slots():
    assert not issubclass(PositiveNumber, ExprValue)
    pset = make_pset()
    price = PriceRow(dsl.var("ap0_out0"))
    primitive = _primitive(pset, "mul", (PriceRow, PositiveNumber))
    result = pset.context[primitive.name](price, PositiveInt(2))
    assert isinstance(result, PriceRow)
    assert isinstance(result.expr, Call) and result.expr.fn == "mul"
    assert isinstance(result.expr.args[1], Number)
    for family in ("xs_rank", "xs_mean", "rolling_mean"):
        assert all(PositiveNumber not in primitive.args[:1] for primitive in _family_primitives(pset, family))


def test_quantile_and_window_parameter_constraints():
    assert QuantileParam(0.0).value == 0.0
    assert QuantileParam(1.0).value == 1.0
    with pytest.raises(ValueError):
        QuantileParam(-0.01)
    with pytest.raises(ValueError):
        QuantileParam(1.01)
    assert PeriodAtLeastTwo(2).value == 2
    with pytest.raises(ValueError):
        PeriodAtLeastTwo(1)

    pset = make_pset()
    assert all(
        QuantileParam in primitive.args
        for family in ("rolling_quantile", "xs_quantile_value", "vec_quantile")
        for primitive in _family_primitives(pset, family)
    )
    assert all(
        primitive.args[-1] is PeriodAtLeastTwo
        for family in ("rolling_prev_diff", "rolling_theilsen")
        for primitive in _family_primitives(pset, family)
    )


def test_min_periods_and_shift_capacity_are_derived_from_period():
    pset = make_pset()
    price = PriceRow(dsl.var("ap0_out0"))
    period = PositiveInt(20)

    rolling = _primitive(pset, "rolling_mean", (PriceRow, PositiveInt))
    rolling_result = pset.context[rolling.name](price, period).expr
    assert dict(rolling_result.kwargs)["min_periods"].value == 20.0

    ewm = _primitive(pset, "ewm", (PriceRow, PositiveInt))
    ewm_result = pset.context[ewm.name](price, period).expr
    ewm_kwargs = dict(ewm_result.kwargs)
    assert ewm_kwargs["min_periods"].value == 20.0
    assert ewm_kwargs["ignore_na"].value == 1.0
    assert ewm_kwargs["adjust"].value == 0.0

    shift = _primitive(pset, "shift", (PriceRow, PositiveInt))
    shifted = pset.context[shift.name](price, PositiveInt(60)).expr
    assert [arg.value for arg in shifted.args[1:]] == [60.0, 60.0]


def test_timestamp_diff_returns_duration_row():
    pset = make_pset()
    timestamp = TimestampRow(dsl.var("_ev_ts"))
    primitive = _primitive(pset, "diff", (TimestampRow, PositiveInt))
    assert primitive.ret is DurationRow
    result = pset.context[primitive.name](timestamp, PositiveInt(2))
    assert isinstance(result, DurationRow)
    assert isinstance(result.expr, Call) and result.expr.fn == "sub"
    compile_ir(result.expr)


def test_round_and_clip_expand_to_supported_cpp_stream_nodes():
    pset = make_pset()
    price = PriceRow(dsl.var("ap0_out0"))
    rounded = _primitive(pset, "round", (NumericRow, PositiveInt))
    round_expr = pset.context[rounded.name](price, PositiveInt(2)).expr
    assert isinstance(round_expr, Call) and round_expr.fn == "div"
    assert isinstance(round_expr.args[0], Call) and round_expr.args[0].fn == "round"
    assert len(round_expr.args[0].args) == 1
    compile_ir(round_expr)

    clipped = _primitive(pset, "clip", (PriceRow, PositiveNumber, PositiveNumber))
    clip_expr = pset.context[clipped.name](price, PositiveInt(1), PositiveInt(2)).expr
    assert isinstance(clip_expr, Call) and clip_expr.fn == "minimum"
    assert isinstance(clip_expr.args[0], Call) and clip_expr.args[0].fn == "maximum"
    compile_ir(clip_expr)


def test_reduction_axes_cannot_include_time_and_results_remain_rows():
    assert AxisSpec(1).value == 1
    with pytest.raises((TypeError, ValueError)):
        AxisSpec(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AxisSpec(0)
    with pytest.raises(ValueError):
        GPConfig(axes=(0,))

    pset = make_pset()
    price = PriceRow(dsl.var("ap0_out0"))
    for family in ("sum", "mean", "std", "reduce_min", "reduce_max"):
        primitive = _primitive(pset, family, (PriceRow, AxisSpec, BoolParam))
        result = pset.context[primitive.name](price, AxisSpec(1), BoolParam(True))
        assert isinstance(result, PriceRow)
        program = compile_ir(result.expr)
        assert program.nodes[program.output_id].value_type.kind == "vector"
        assert all(type(node.op).__name__ != "EmitOp" for node in program.nodes)


def test_all_regression_composites_lower_to_row_outputs():
    pset = make_pset()
    y = PriceRow(dsl.var("ap0_out0"))
    x1 = PriceRow(dsl.var("bp0_out0"))
    x2 = PriceRow(dsl.var("mp_out0.close"))
    x3 = DimensionlessRow(dsl.var("vw_halfspread_out0"))
    period = PositiveInt(20)

    ts_primitives = _family_primitives(pset, "ts_regression")
    assert len(ts_primitives) == len(REGRESSION_PROJECTIONS)
    for primitive in ts_primitives:
        result = pset.context[primitive.name](y, x1, period)
        program = compile_ir(result.expr)
        assert program.nodes[program.output_id].value_type.kind == "vector"

    for projection in REGRESSION_PROJECTIONS:
        family = f"ridge_{projection}"
        primitives = _family_primitives(pset, family)
        assert len(primitives) == 3
        inputs = {2: (y, x1), 3: (y, x1, x2), 4: (y, x1, x2, x3)}
        for primitive in primitives:
            result = pset.context[primitive.name](*inputs[len(primitive.args)])
            program = compile_ir(result.expr)
            assert program.nodes[program.output_id].value_type.kind == "vector"

    poly = _family_primitives(pset, "ts_poly_regression")
    assert len(poly) == 3
    for primitive in poly:
        result = pset.context[primitive.name](y, x1, period)
        assert compile_ir(result.expr).nodes[-1].value_type.kind == "vector"

    neut = _family_primitives(pset, "xs_regression_neut")
    assert len(neut) == 1
    result = pset.context[neut[0].name](y, x1)
    assert compile_ir(result.expr).nodes[-1].value_type.kind == "vector"


def test_generation_uses_standard_deap_toolbox_and_gen_half_and_half():
    pset = make_pset()
    toolbox = make_toolbox(pset, min_depth=1, max_depth=3)
    assert toolbox.expr.func is gp.genHalfAndHalf
    for seed in range(24):
        tree, expr = random_formula(pset, min_depth=1, max_depth=4, seed=seed)
        assert isinstance(tree, gp.PrimitiveTree)
        assert isinstance(expr, Expr)
        assert 1 <= tree.height <= 4


def test_gp_package_uses_only_absolute_imports():
    root = Path(__file__).resolve().parents[3] / "src" / "flows" / "gp"
    files = sorted(root.glob("*.py"))
    assert files
    for path in files:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = [node for node in ast.walk(module) if isinstance(node, ast.ImportFrom) and node.level]
        assert not relative, f"relative imports in {path}: {relative}"
