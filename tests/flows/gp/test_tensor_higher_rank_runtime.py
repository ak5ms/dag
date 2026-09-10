from __future__ import annotations

import numpy as np

from flows.gp import GPConfig, PositiveInt, PositiveNumber, TensorFieldSpec, make_pset, tensor_type
from flows.gp.validation import family_primitives
from trading_dsl_engine.base import dsl
from trading_dsl_engine.cpp_stream import compile_formula


def _primitive(pset, family, args):
    return next(
        primitive
        for primitive in family_primitives(pset, family)
        if tuple(primitive.args) == args
    )


def _run(tmp_path, expression, data, n_instruments):
    runtime = compile_formula(
        expression,
        data,
        n_instruments=n_instruments,
        default_group_capacity=256,
        prefetch_rows=4,
    )
    result = runtime.run(out_path=tmp_path / "higher-rank-result.npy", threads=0)
    return np.asarray(result.load())


def _sum_all_feature_axes(pset, value, rank, semantic):
    current = value
    current_rank = rank
    current_semantic = semantic
    while current_rank >= 2:
        input_type = tensor_type(current_rank, current_semantic)
        primitive = _primitive(pset, "vec_sum", (input_type,))
        current = pset.context[primitive.name](current)
        current_rank -= 1
        if current_rank >= 2:
            current_semantic = current.__class__.tensor_semantic
    return current


def test_rank3_temporal_shift_and_reductions_match_numpy(tmp_path):
    rng = np.random.default_rng(101)
    rows, instruments, levels, channels = 18, 4, 5, 3
    values = rng.normal(size=(rows, instruments, levels, channels))
    config = GPConfig(
        tensor_fields=(TensorFieldSpec("book4", "price", (levels, channels)),),
        tensor_indices=(0, 1, 2),
    )
    pset = make_pset(config)
    price3 = tensor_type(3, "price")
    shift = _primitive(pset, "shift", (price3, PositiveInt))
    shifted = pset.context[shift.name](price3(dsl.var("book4")), PositiveInt(2))
    expression = _sum_all_feature_axes(pset, shifted, 3, "price").expr
    actual = _run(tmp_path, expression, {"book4": values}, instruments)

    expected = np.full((rows, instruments), np.nan)
    expected[2:] = np.sum(values[:-2], axis=(-1, -2))
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_rank4_elementwise_and_repeated_vec_reduction_match_numpy(tmp_path):
    rng = np.random.default_rng(103)
    rows, instruments, levels, channels, sides = 15, 3, 4, 3, 2
    values = rng.normal(size=(rows, instruments, levels, channels, sides))
    config = GPConfig(
        tensor_fields=(
            TensorFieldSpec("book5", "price", (levels, channels, sides)),
        ),
        tensor_indices=(0, 1),
    )
    pset = make_pset(config)
    price4 = tensor_type(4, "price")
    add = _primitive(pset, "add", (price4, PositiveNumber))
    shifted = pset.context[add.name](price4(dsl.var("book5")), PositiveInt(2))
    expression = _sum_all_feature_axes(pset, shifted, 4, "price").expr
    actual = _run(tmp_path, expression, {"book5": values}, instruments)
    expected = np.sum(values + 2.0, axis=(-1, -2, -3))
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
