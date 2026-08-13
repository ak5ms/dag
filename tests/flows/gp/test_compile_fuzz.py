from __future__ import annotations

import os

import pytest

from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.cpp_stream.python import utils as cpp_stream_utils
from trading_dsl_engine.cpp_stream.python.frontend import compile_ir
from trading_dsl_engine.base import dsl
from trading_dsl_engine.base.dsl import ffill, shift, var
from trading_dsl_engine.base.parser import Call
from flows.alpha_search import default_alpha_pnl
from flows.gp import (
    PositiveInt,
    PriceRow,
    REGRESSION_PROJECTIONS,
    make_pset,
    primitive_names_for_operator,
    random_formula,
)
from flows.gp.regression import (
    temporal_poly_regression_residual,
    temporal_ridge_projection,
    xs_regression_neutralize,
)
from flows.gp.signatures import format_signature_table


_MIN_DEPTH = int(os.environ.get("GP_FUZZ_MIN_DEPTH", "1"))
_MAX_DEPTH = int(os.environ.get("GP_FUZZ_MAX_DEPTH", "3"))
if _MIN_DEPTH < 0 or _MAX_DEPTH < _MIN_DEPTH:
    raise ValueError("require 0 <= GP_FUZZ_MIN_DEPTH <= GP_FUZZ_MAX_DEPTH")


def _returns():
    close = ffill(var("mp_out0.close"))
    return close / shift(close) - 1.0


def _wrapped_formula(alpha):
    return default_alpha_pnl(
        alpha,
        roll_rets=_returns(),
        is_tradable=var("is_tradable_out0"),
        hl=1440,
    )


def _failure_text(failures):
    chunks = []
    for seed, tree, alpha, error in failures:
        chunks.append(
            f"seed={seed}\n"
            f"depth={tree.height}\n"
            f"tree={tree}\n"
            f"alpha={alpha!r}\n"
            f"error={type(error).__name__}: {error}"
        )
    return "\n\n".join(chunks)


def test_print_exact_gp_signature_table():
    print(format_signature_table(make_pset()))


def test_regression_gp_adapters_reuse_cpp_stream_utils():
    y = dsl.var("y")
    x = dsl.var("x")

    expected_residual = cpp_stream_utils.ts_regression(
        y,
        x,
        20,
        lag=0,
        rettype="residual",
        weights=1.0,
        lambda_=0.0,
    )
    assert temporal_ridge_projection("residual", y, x, 20) == expected_residual

    expected_r2 = cpp_stream_utils.ts_regression(
        y,
        x,
        20,
        lag=0,
        rettype="r2",
        weights=1.0,
        lambda_=0.0,
    )
    actual_r2 = temporal_ridge_projection("r2", y, x, 20)
    assert isinstance(actual_r2, Call) and actual_r2.fn == "where"
    assert actual_r2.args[1] == expected_r2
    assert actual_r2.args[2] == expected_r2

    for degree in (1, 2, 3):
        expected_poly = cpp_stream_utils.ts_poly_regression(
            y,
            x,
            20,
            k=degree,
            weights=1.0,
            lambda_=0.0,
        )
        assert temporal_poly_regression_residual(y, x, 20, degree) == expected_poly

    assert xs_regression_neutralize(y, x) == cpp_stream_utils.xs_regression_neut(y, x)


def test_random_gp_formulas_compile_to_cpp_stream_ir():
    pset = make_pset()
    failures = []
    samples = int(os.environ.get("GP_IR_FUZZ_SAMPLES", "2000"))
    for seed in range(samples):
        tree, alpha = random_formula(
            pset,
            min_depth=_MIN_DEPTH,
            max_depth=_MAX_DEPTH,
            seed=seed,
        )
        assert _MIN_DEPTH <= tree.height <= _MAX_DEPTH
        formula = _wrapped_formula(alpha)
        try:
            compile_ir(formula)
        except Exception as exc:  # pragma: no cover - failure diagnostics
            failures.append((seed, tree, alpha, exc))
            if len(failures) >= 25:
                break
    assert not failures, _failure_text(failures)


_NATIVE_SAMPLES = int(os.environ.get("GP_NATIVE_FUZZ_SAMPLES", "40"))


@pytest.mark.parametrize("seed", range(_NATIVE_SAMPLES))
def test_random_gp_formulas_compile_natively(seed):
    pset = make_pset()
    tree, alpha = random_formula(
        pset,
        min_depth=_MIN_DEPTH,
        max_depth=_MAX_DEPTH,
        seed=seed,
    )
    assert _MIN_DEPTH <= tree.height <= _MAX_DEPTH
    formula = _wrapped_formula(alpha)
    try:
        runtime = compile_formula(
            formula,
            n_instruments=9,
            default_group_capacity=365 * 15,
            prefetch_rows=16,
        )
    except Exception as exc:  # pragma: no cover - failure diagnostics
        pytest.fail(
            _failure_text([(seed, tree, alpha, exc)]),
            pytrace=True,
        )
    assert runtime.program.output_id >= 0


def _family_primitives(pset, family: str):
    return [pset.mapping[name] for name in primitive_names_for_operator(pset, family)]


def test_all_regression_composite_families_compile_natively():
    pset = make_pset()
    y = PriceRow(dsl.var("ap0_out0"))
    x = PriceRow(dsl.var("bp0_out0"))
    period = PositiveInt(20)
    expressions = []

    for primitive in _family_primitives(pset, "ts_regression"):
        expressions.append(pset.context[primitive.name](y, x, period).expr)

    for projection in REGRESSION_PROJECTIONS:
        family = f"ridge_{projection}"
        primitive = next(
            item for item in _family_primitives(pset, family) if len(item.args) == 2
        )
        expressions.append(pset.context[primitive.name](y, x).expr)

    for primitive in _family_primitives(pset, "ts_poly_regression"):
        expressions.append(pset.context[primitive.name](y, x, period).expr)

    neut = _family_primitives(pset, "xs_regression_neut")
    assert len(neut) == 1
    expressions.append(pset.context[neut[0].name](y, x).expr)

    runtime = compile_formula(
        dsl.cat(*expressions),
        n_instruments=9,
        default_group_capacity=365 * 15,
        prefetch_rows=16,
    )
    assert runtime.program.output_id >= 0
