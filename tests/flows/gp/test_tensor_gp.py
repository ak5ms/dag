from __future__ import annotations

from flows.gp import (
    BookPriceMatrix,
    BookVolumeMatrix,
    GPConfig,
    PriceRow,
    TensorFieldSpec,
    gp_input_types,
    make_pset,
    primitive_names_for_operator,
    tensor_type,
)
from trading_dsl_engine.base import dsl
from trading_dsl_engine.cpp_stream.python.frontend import compile_ir
from trading_dsl_engine.ir.types import tensor


def _primitives(pset, family):
    return [pset.mapping[name] for name in primitive_names_for_operator(pset, family)]


def _primitive(pset, family, args):
    return next(value for value in _primitives(pset, family) if tuple(value.args) == args)


def test_default_books_are_composed_matrix_terminals():
    config = GPConfig()
    pset = make_pset(config)
    assert pset.gp_tensor_ranks == (2,)
    assert set(pset.gp_tensor_field_terminals) == {"book_price", "book_volume"}
    price_terminal = pset.mapping[pset.gp_tensor_field_terminals["book_price"]]
    volume_terminal = pset.mapping[pset.gp_tensor_field_terminals["book_volume"]]
    assert price_terminal.ret is BookPriceMatrix
    assert volume_terminal.ret is BookVolumeMatrix
    assert gp_input_types(config, 9) == {}
    price = pset.context[price_terminal.value]
    volume = pset.context[volume_terminal.value]
    assert price.expr.fn == "cat"
    assert volume.expr.fn == "cat"
    assert len(price.expr.args) == 20
    assert len(volume.expr.args) == 20


def test_vec_reductions_turn_book_matrices_into_rows():
    pset = make_pset()
    source = BookPriceMatrix(dsl.var("book"))
    families = (
        "vec_avg", "vec_choose", "vec_count", "vec_ir", "vec_kurtosis",
        "vec_max", "vec_min", "vec_norm", "vec_percentage", "vec_powersum",
        "vec_range", "vec_skewness", "vec_stddev", "vec_sum",
    )
    assert all(_primitives(pset, family) for family in families)
    avg = _primitive(pset, "vec_avg", (BookPriceMatrix,))
    assert avg.ret is PriceRow
    expr = pset.context[avg.name](source).expr
    program = compile_ir(expr, input_value_types={"book": tensor((None, 20))})
    assert program.nodes[program.output_id].value_type.kind == "vector"


def test_higher_rank_tensors_reduce_one_axis_at_a_time():
    config = GPConfig(
        tensor_fields=(TensorFieldSpec("book4", "price", (4, 3)),),
        tensor_indices=(0, 1, 2),
    )
    pset = make_pset(config)
    rank3 = tensor_type(3, "price")
    rank2 = tensor_type(2, "price")
    first = _primitive(pset, "vec_avg", (rank3,))
    second = _primitive(pset, "vec_sum", (rank2,))
    assert first.ret is rank2
    assert second.ret is PriceRow
    matrix = pset.context[first.name](rank3(dsl.var("book4")))
    expr = pset.context[second.name](matrix).expr
    program = compile_ir(expr, input_value_types={"book4": tensor((None, 4, 3))})
    assert program.nodes[program.output_id].value_type.kind == "vector"
    spec = gp_input_types(config, 9)["book4"]
    assert spec.row_shape == (9, 4, 3)
    assert spec.row_width == 108


def test_tensor_runtime_suite(tmp_path):
    from tests.flows.gp import test_tensor_runtime as runtime

    checks = (
        runtime.test_matrix_vec_average_matches_numpy,
        runtime.test_rank3_tensor_reduces_one_final_axis_per_vec_call,
        runtime.test_rank3_elementwise_broadcasting_matches_numpy,
        runtime.test_matrix_temporal_diff_then_vec_sum_matches_numpy,
        runtime.test_matrix_regression_composite_compiles_and_runs,
    )
    for check in checks:
        case = tmp_path / check.__name__
        case.mkdir()
        check(case)
