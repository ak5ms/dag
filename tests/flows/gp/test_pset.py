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
    GenoTest,
    GPConfig,
    NoiseSpec,
    NumericRow,
    NumericTensor,
    PeriodAtLeastTwo,
    PositiveInt,
    PositiveNumber,
    PriceRow,
    QuantileParam,
    REGRESSION_PROJECTIONS,
    ScalarNumber,
    TimestampRow,
    build_gp_graph,
    filter_gp_graph,
    geno_max_depth,
    geno_max_nodes,
    gp_explorer_html,
    make_pset,
    make_toolbox,
    pheno_finite,
    primitive_names_for_operator,
    random_formula,
    run_geno_tests,
    run_pheno_tests,
    shock_dynamic_leaves,
    shock_static_terminals,
)
from trading_dsl_engine.base import dsl
from trading_dsl_engine.base.parser import Call, Expr, Number
from trading_dsl_engine.cpp_stream.python.frontend import compile_ir


def _primitives(pset, family):
    return [pset.mapping[name] for name in primitive_names_for_operator(pset, family)]


def _primitive(pset, family, args):
    return next(value for value in _primitives(pset, family) if tuple(value.args) == args)


def _terminal_value(pset, terminal):
    value = terminal.value
    if isinstance(value, str) and value in pset.context:
        return pset.context[value]
    return value


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


def test_default_static_terminal_grid_is_dense_and_includes_zero():
    config = GPConfig()
    assert {4, 8, 16, 32, 64, 128, 256, 720, 2880} <= set(config.positive_ints)
    assert {0.0001, 0.025, 0.2, 0.75, 1.5, 5.0, 20.0} <= set(config.positive_floats)
    assert {-20.0, -1.5, -0.2, -0.025, -0.0001} <= set(config.negative_floats)
    assert {0.01, 0.33, 0.67, 0.99} <= set(config.quantiles)

    pset = make_pset(config)
    zero_name = pset.gp_scalar_terminals[0.0]
    zero = _terminal_value(pset, pset.mapping[zero_name])
    assert zero == ScalarNumber(0.0)


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


def test_geno_tests_are_structural_and_composable():
    pset = make_pset()
    tree, _ = random_formula(pset, min_depth=1, max_depth=3, seed=7)
    report = run_geno_tests(
        tree,
        pset,
        tests=(
            geno_max_depth(6),
            geno_max_nodes(200),
            GenoTest("has_terminal", lambda ctx: ctx.terminal_count > 0),
        ),
    )
    assert report.passed
    assert report.outcomes[0].name == "well_typed"
    assert report.context.primitive_count + report.context.terminal_count == len(tree)


def test_static_pheno_shock_moves_only_within_k_sorted_terminal_neighbors():
    pset = make_pset()
    terminals = [
        terminal
        for terminal in pset.terminals[PositiveInt]
        if terminal.ret is PositiveInt
        and _terminal_value(pset, terminal).value == 20
    ]
    assert terminals
    tree = gp.PrimitiveTree([terminals[0]])
    shocked, changes = shock_static_terminals(tree, pset, k=2, seed=13)
    assert len(changes) == 1

    values = sorted({
        _terminal_value(pset, terminal).value
        for terminal in pset.terminals[PositiveInt]
        if terminal.ret is PositiveInt
    })
    change = changes[0]
    assert abs(values.index(change.after) - values.index(change.before)) <= 2
    assert change.after != change.before
    assert _terminal_value(pset, shocked[0]).value == change.after


def test_dynamic_field_shocks_accept_dynamic_distribution_parameters_and_lower():
    leaf = dsl.var("ap0_out0")
    expr = dsl.add(leaf, dsl.var("bp0_out0"))
    shocked, changes = shock_dynamic_leaves(
        expr,
        {
            "ap0_out0": NoiseSpec(
                "normal",
                params={
                    "mu": 0.0,
                    "sigma": lambda x: dsl.maximum(dsl.ewm_std(x, 20), 1e-8),
                },
                mode="add",
            )
        },
        seed=11,
    )
    assert len(changes) == 1
    assert changes[0].field == "ap0_out0"
    assert "sin" in str(shocked) and "ln" in str(shocked)
    compile_ir(shocked)


def test_random_distribution_dsl_helpers_are_pure_compositions():
    x = dsl.var("ap0_out0")
    draws = (
        dsl.uniform(-1.0, 1.0, key=x, seed=1),
        dsl.normal(mu=dsl.ewm(x, 20), sigma=1.0, key=x, seed=2),
        dsl.lognormal(mu=0.0, sigma=dsl.ewm_std(x, 20), key=x, seed=3),
        dsl.exponential(scale=dsl.maximum(dsl.abs(x), 1e-8), key=x, seed=4),
    )
    for draw in draws:
        program = compile_ir(draw)
        assert program.nodes[-1].value_type.kind == "vector"
        assert "random" not in str(draw).lower()


def test_pheno_tests_execute_baseline_and_each_shocked_trial():
    pset = make_pset()
    tree, _ = random_formula(pset, min_depth=1, max_depth=2, seed=3)
    calls = []

    def evaluator(expr):
        calls.append(expr)
        return float(len(compile_ir(expr).nodes))

    report = run_pheno_tests(
        tree,
        pset,
        evaluator,
        tests=(pheno_finite(),),
        n_trials=3,
        static_k=2,
        seed=19,
    )
    assert len(calls) == 4
    assert len(report.trials) == 3
    assert report.passed
    assert all(trial.outcomes[0].name == "execution" for trial in report.trials)


def test_gp_graph_explorer_drills_from_types_and_searches_all_node_kinds():
    pset = make_pset()
    model = build_gp_graph(pset)
    assert {node.kind for node in model.nodes} == {"type", "operator", "terminal"}
    assert model.type_relations

    filtered = filter_gp_graph(model, "PriceRow")
    assert any(node.kind == "type" and node.label == "PriceRow" for node in filtered.nodes)
    assert any(node.kind in {"operator", "terminal"} for node in filtered.nodes)

    page = gp_explorer_html(pset, include_plotlyjs=False)
    assert 'id="gp-search"' in page
    assert "plotly_click" in page
    assert "GP type relations" in page


def test_gp_package_uses_only_absolute_imports():
    root = Path(__file__).resolve().parents[3] / "src" / "flows" / "gp"
    for path in root.glob("*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert not [
            node for node in ast.walk(module)
            if isinstance(node, ast.ImportFrom) and node.level
        ], path


def test_negative_scalar_float_terminals_support_from_string():
    from flows.gp import GrammarPolicy, individual_to_expr
    from flows.gp.generation import make_toolbox
    from flows.riskminer.semantics import gp_alpha_search_terminal_metadata

    pset = make_pset(
        GPConfig(
            fields=gp_alpha_search_terminal_metadata(),
            grammar=GrammarPolicy(exclude_sections=("utils.group",)),
        )
    )
    make_toolbox(pset, min_depth=1, max_depth=2)
    assert "-1" in pset.mapping
    tree = gp.PrimitiveTree.from_string(
        "mul_scalar_dimensionless(-1, xs_rank_numeric(field_roll_rets))",
        pset,
    )
    expr = individual_to_expr(tree, pset)
    assert "xs_rank" in str(expr)
    assert "mul" in str(expr)
