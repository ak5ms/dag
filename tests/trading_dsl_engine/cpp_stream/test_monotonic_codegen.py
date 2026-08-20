from __future__ import annotations

import pytest

from trading_dsl_engine.base.dsl import cumsum, groupby, self_, var
from trading_dsl_engine.base.keys import Key
from trading_dsl_engine.cpp_stream.python.codegen import render_translation_unit
from trading_dsl_engine.cpp_stream.python.frontend import compile_ir
from trading_dsl_engine.cpp_stream.python.lowering import CppStreamLoweringError
from trading_dsl_engine.cpp_stream.python.lowering_multi import lower_program
from trading_dsl_engine.cpp_stream.python.npy import InputTypeSpec
from trading_dsl_engine.cpp_stream.python.outputs import build_output_layout


def _group(plan):
    return next(stage.group for stage in plan.stages if stage.group is not None)


def _render(program, plan, *, n: int, input_types) -> str:
    return render_translation_unit(
        plan,
        n_instruments=n,
        prefetch_rows=16,
        input_types=input_types,
        output_layout=build_output_layout(program, n),
    ).text


def test_monotonic_only_key_uses_capacity_one_reset_resolver() -> None:
    formula = groupby(
        Key(
            var("session"),
            row_scalar=True,
            dtype="float64",
            monotonic=True,
        ),
        var("close"),
        cumsum(self_),
    )
    program = compile_ir(formula)
    assert program.nodes[program.output_id].op.key_specs[0].monotonic is True
    input_types = (
        InputTypeSpec("float64", 9),
        InputTypeSpec("float64", 9),
    )
    plan = lower_program(
        program,
        n_instruments=9,
        default_group_capacity=4096,
        input_dtypes=tuple(spec.dtype for spec in input_types),
    )
    group = _group(plan)
    assert group.capacity == 1
    assert group.dense is False
    source = _render(program, plan, n=9, input_types=input_types)
    assert "stackdsl::MonotonicGroupResolver<" in source
    assert "stackdsl::NoKeyResolver<9>" in source


def test_monotonic_epoch_can_wrap_retained_dense_keys() -> None:
    formula = groupby(
        (
            Key(
                var("session"),
                row_scalar=True,
                dtype="float64",
                monotonic=True,
            ),
            Key(var("bucket"), num_keys=4, dtype="int32"),
        ),
        var("close"),
        cumsum(self_),
    )
    program = compile_ir(formula)
    input_types = (
        InputTypeSpec("float64", 1),
        InputTypeSpec("int32", 5),
        InputTypeSpec("float64", 5),
    )
    plan = lower_program(
        program,
        n_instruments=5,
        input_dtypes=tuple(spec.dtype for spec in input_types),
    )
    group = _group(plan)
    assert group.capacity == 5
    assert group.dense is True
    source = _render(program, plan, n=5, input_types=input_types)
    assert "stackdsl::MonotonicGroupResolver<" in source
    assert "stackdsl::DenseTupleGroupResolver<5" in source


def test_monotonic_key_requires_row_scalar_routing() -> None:
    formula = groupby(
        Key(var("session"), row_scalar=False, monotonic=True),
        var("close"),
        cumsum(self_),
    )
    with pytest.raises(CppStreamLoweringError, match="row_scalar=True"):
        lower_program(compile_ir(formula), n_instruments=5)
