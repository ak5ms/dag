from __future__ import annotations

import numpy as np

from flows.gp import (
    GPConfig,
    NumericRow,
    PositiveInt,
    PriceRow,
    TensorFieldSpec,
    make_pset,
    tensor_type,
)
from flows.gp.validation import family_primitives
from trading_dsl_engine.base import dsl
from trading_dsl_engine.cpp_stream import compile_formula


def _primitive(pset, family, args):
    return next(
        value
        for value in family_primitives(pset, family)
        if tuple(value.args) == args
    )


def _run(tmp_path, expression, data, n_instruments):
    runtime = compile_formula(
        expression,
        data,
        n_instruments=n_instruments,
        default_group_capacity=256,
        prefetch_rows=4,
    )
    result = runtime.run(out_path=tmp_path / "tensor-result.npy", threads=0)
    return np.asarray(result.load())


def test_matrix_vec_average_matches_numpy(tmp_path):
    rows, instruments, levels = 23, 4, 5
    values = np.arange(rows * instruments * levels, dtype=np.float64).reshape(
        rows, instruments, levels
    )
    config = GPConfig(
        tensor_fields=(TensorFieldSpec("book", "price", (levels,)),),
        tensor_indices=(0, 1, 2),
    )
    pset = make_pset(config)
    matrix = tensor_type(2, "price")
    primitive = _primitive(pset, "vec_avg", (matrix,))
    expression = pset.context[primitive.name](matrix(dsl.var("book"))).expr
    actual = _run(tmp_path, expression, {"book": values}, instruments)
    np.testing.assert_allclose(actual, np.mean(values, axis=-1))


def test_rank3_tensor_reduces_one_final_axis_per_vec_call(tmp_path):
    rng = np.random.default_rng(3)
    rows, instruments, levels, channels = 19, 4, 3, 2
    values = rng.normal(size=(rows, instruments, levels, channels))
    config = GPConfig(
        tensor_fields=(TensorFieldSpec("book4", "price", (levels, channels)),),
        tensor_indices=(0, 1),
    )
    pset = make_pset(config)
    rank3 = tensor_type(3, "price")
    rank2 = tensor_type(2, "price")
    average = _primitive(pset, "vec_avg", (rank3,))
    total = _primitive(pset, "vec_sum", (rank2,))
    matrix = pset.context[average.name](rank3(dsl.var("book4")))
    expression = pset.context[total.name](matrix).expr
    actual = _run(tmp_path, expression, {"book4": values}, instruments)
    expected = np.sum(np.mean(values, axis=-1), axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_rank3_elementwise_broadcasting_matches_numpy(tmp_path):
    rng = np.random.default_rng(7)
    rows, instruments, levels, channels = 17, 4, 3, 2
    values = rng.normal(size=(rows, instruments, levels, channels))
    config = GPConfig(
        tensor_fields=(TensorFieldSpec("book4", "price", (levels, channels)),),
        tensor_indices=(0, 1),
    )
    pset = make_pset(config)
    price3 = tensor_type(3, "price")
    derived3 = tensor_type(3, "derived")
    derived2 = tensor_type(2, "derived")
    add = _primitive(pset, "add", (price3, PositiveInt.__bases__[0]))
    multiply = _primitive(pset, "mul", (price3, price3))
    average = _primitive(pset, "vec_avg", (derived3,))
    total = _primitive(pset, "vec_sum", (derived2,))
    source = price3(dsl.var("book4"))
    shifted = pset.context[add.name](source, PositiveInt(2))
    product = pset.context[multiply.name](shifted, source)
    matrix = pset.context[average.name](product)
    expression = pset.context[total.name](matrix).expr
    actual = _run(tmp_path, expression, {"book4": values}, instruments)
    expected = np.sum(np.mean((values + 2.0) * values, axis=-1), axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_matrix_temporal_diff_then_vec_sum_matches_numpy(tmp_path):
    rng = np.random.default_rng(11)
    rows, instruments, levels = 21, 4, 5
    values = rng.normal(size=(rows, instruments, levels))
    config = GPConfig(
        tensor_fields=(TensorFieldSpec("book", "price", (levels,)),),
        tensor_indices=(0, 1, 2),
    )
    pset = make_pset(config)
    matrix = tensor_type(2, "price")
    diff = _primitive(pset, "diff", (matrix, PositiveInt))
    total = _primitive(pset, "vec_sum", (matrix,))
    changed = pset.context[diff.name](matrix(dsl.var("book")), PositiveInt(2))
    expression = pset.context[total.name](changed).expr
    actual = _run(tmp_path, expression, {"book": values}, instruments)
    expected = np.full((rows, instruments), np.nan)
    expected[2:] = np.sum(values[2:] - values[:-2], axis=-1)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_matrix_regression_composite_compiles_and_runs(tmp_path):
    rng = np.random.default_rng(13)
    rows, instruments, levels = 40, 4, 3
    x = rng.normal(size=(rows, instruments, levels))
    y = 0.2 + 0.4 * x[..., 0] - 0.1 * x[..., 1] + rng.normal(
        scale=0.01, size=(rows, instruments)
    )
    config = GPConfig(
        tensor_fields=(TensorFieldSpec("book", "price", (levels,)),),
        tensor_indices=(0, 1, 2),
    )
    pset = make_pset(config)
    matrix = tensor_type(2, "numeric")
    primitive = _primitive(
        pset,
        "ts_regression",
        (NumericRow, matrix, PositiveInt),
    )
    expression = pset.context[primitive.name](
        PriceRow(dsl.var("y")),
        tensor_type(2, "price")(dsl.var("book")),
        PositiveInt(5),
    ).expr
    actual = _run(tmp_path, expression, {"book": x, "y": y}, instruments)
    assert actual.shape == y.shape
    assert np.isfinite(actual[10:]).any()
