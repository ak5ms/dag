from __future__ import annotations

import os

import pytest

from flows.alpha_search import default_alpha_pnl
from flows.gp import ALL_CPP_STREAM_UTIL_NAMES, GPConfig, NumericRow, PositiveInt, PriceRow, REGRESSION_PROJECTIONS, TensorFieldSpec, make_pset, random_formula
from flows.gp.signatures import format_signature_table
from flows.gp.validation import expected_output_kind, family_primitives, is_tensor_primitive, sample_primitive
from trading_dsl_engine.base import dsl
from trading_dsl_engine.base.dsl import ffill, shift, var
from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.cpp_stream.python.frontend import compile_ir

MIN_DEPTH = int(os.environ.get("GP_FUZZ_MIN_DEPTH", "1"))
MAX_DEPTH = int(os.environ.get("GP_FUZZ_MAX_DEPTH", "3"))
if MIN_DEPTH < 0 or MAX_DEPTH < MIN_DEPTH:
    raise ValueError("require 0 <= GP_FUZZ_MIN_DEPTH <= GP_FUZZ_MAX_DEPTH")


def _returns():
    close = ffill(var("mp_out0.close"))
    return close / shift(close) - 1.0


def _wrapped(alpha):
    return default_alpha_pnl(alpha, roll_rets=_returns(), is_tradable=var("is_tradable_out0"), hl=1440)


def _failure_text(failures):
    return "\n\n".join(
        f"seed={seed}\ndepth={tree.height}\ntree={tree}\nalpha={alpha!r}\nerror={type(error).__name__}: {error}"
        for seed, tree, alpha, error in failures
    )


def _unique_primitives(pset):
    return {p.name: p for values in pset.primitives.values() for p in values}.values()


def _compile_bundle(pset, primitives):
    expressions, specs = [], {}
    for primitive in primitives:
        expression, _, current = sample_primitive(pset, primitive)
        expressions.append(expression)
        specs.update(current)
    runtime = compile_formula(
        dsl.cat(*expressions), n_instruments=9, input_types=specs,
        default_group_capacity=365 * 15, prefetch_rows=16,
    )
    assert runtime.program.output_id >= 0


def test_print_exact_gp_signature_table():
    print(format_signature_table(make_pset()))


def test_all_cpp_stream_utils_are_exposed():
    pset = make_pset()
    assert pset.gp_cpp_stream_utility_families == ALL_CPP_STREAM_UTIL_NAMES
    assert not pset.gp_non_row_cpp_stream_utility_families
    for family in ALL_CPP_STREAM_UTIL_NAMES:
        assert family_primitives(pset, family), family


@pytest.mark.parametrize("config", [
    GPConfig(),
    GPConfig(tensor_fields=(TensorFieldSpec("book4", "price", (3, 2)),), tensor_indices=(0, 1)),
])
def test_every_concrete_primitive_lowers_with_declared_type(config):
    pset, failures, checked = make_pset(config), [], 0
    for primitive in _unique_primitives(pset):
        checked += 1
        try:
            expression, input_types, _ = sample_primitive(pset, primitive)
            program = compile_ir(expression, input_value_types=input_types)
            actual = program.nodes[program.output_id].value_type.kind
            expected = expected_output_kind(primitive.ret)
            if actual != expected:
                raise AssertionError(f"output kind={actual}, expected={expected}")
        except Exception as exc:
            failures.append((primitive.name, primitive.args, primitive.ret, exc))
            if len(failures) >= 40:
                break
    message = "\n\n".join(
        f"primitive={name}\nargs={[t.__name__ for t in args]}\nret={ret.__name__}\nerror={type(error).__name__}: {error}"
        for name, args, ret, error in failures
    )
    assert checked > 0
    assert not failures, message


def test_one_row_primitive_per_added_utility_family_compiles_natively():
    pset = make_pset()
    selected = [next(p for p in family_primitives(pset, family) if not is_tensor_primitive(p)) for family in sorted(pset.gp_added_cpp_stream_utility_families)]
    _compile_bundle(pset, selected)


def test_one_primitive_per_tensor_family_compiles_natively():
    pset = make_pset()
    selected = [next(p for p in family_primitives(pset, family) if is_tensor_primitive(p)) for family in sorted(pset.gp_tensor_operator_families)]
    _compile_bundle(pset, selected)


def test_random_gp_formulas_compile_to_cpp_stream_ir():
    pset, failures = make_pset(), []
    for seed in range(int(os.environ.get("GP_IR_FUZZ_SAMPLES", "2000"))):
        tree, alpha = random_formula(pset, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, seed=seed)
        assert MIN_DEPTH <= tree.height <= MAX_DEPTH
        try:
            compile_ir(_wrapped(alpha))
        except Exception as exc:
            failures.append((seed, tree, alpha, exc))
            if len(failures) >= 25:
                break
    assert not failures, _failure_text(failures)


@pytest.mark.parametrize("seed", range(int(os.environ.get("GP_NATIVE_FUZZ_SAMPLES", "40"))))
def test_random_gp_formulas_compile_natively(seed):
    pset = make_pset()
    tree, alpha = random_formula(pset, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, seed=seed)
    assert MIN_DEPTH <= tree.height <= MAX_DEPTH
    try:
        runtime = compile_formula(_wrapped(alpha), n_instruments=9, default_group_capacity=365 * 15, prefetch_rows=16)
    except Exception as exc:
        pytest.fail(_failure_text([(seed, tree, alpha, exc)]), pytrace=True)
    assert runtime.program.output_id >= 0


def test_all_row_regression_composites_compile_natively():
    pset = make_pset()
    y, x, period = PriceRow(dsl.var("ap0_out0")), PriceRow(dsl.var("bp0_out0")), PositiveInt(20)
    expressions = []
    for primitive in family_primitives(pset, "ts_regression"):
        if tuple(primitive.args) == (NumericRow, NumericRow, PositiveInt):
            expressions.append(pset.context[primitive.name](y, x, period).expr)
    for projection in REGRESSION_PROJECTIONS:
        for primitive in family_primitives(pset, f"ridge_{projection}"):
            if all(type_ is NumericRow for type_ in primitive.args):
                expressions.append(pset.context[primitive.name](*(y, x, x, x)[:len(primitive.args)]).expr)
    for primitive in family_primitives(pset, "ts_poly_regression"):
        if tuple(primitive.args) == (NumericRow, NumericRow, PositiveInt):
            expressions.append(pset.context[primitive.name](y, x, period).expr)
    runtime = compile_formula(dsl.cat(*expressions), n_instruments=9, default_group_capacity=365 * 15, prefetch_rows=16)
    assert runtime.program.output_id >= 0
