from __future__ import annotations

from trading_dsl_engine.cpp_stream.python import codegen as base
from trading_dsl_engine.cpp_stream.python.codegen import (
    BoolArg,
    CppType,
    IntArg,
    Name,
    SignedValueArg,
    Stage,
    UInt64Arg,
    tmpl,
)
from trading_dsl_engine.cpp_stream.python.npy import InputTypeSpec
from trading_dsl_engine.ir.ops import ColumnOp, EwmOp, FFillOp, ShiftOp


_original_stage_type = base._stage_type


def _tensor_inputs(
    stage: Stage,
    n: CppType,
    input_types: tuple[InputTypeSpec, ...] | None,
) -> tuple[CppType, ...]:
    return tuple(
        base._tensor_source_type(source, n=n, input_types=input_types)
        for source in stage.inputs
    )


def _stage_type(
    stage: Stage,
    n: CppType,
    execution: CppType,
    *,
    input_types: tuple[InputTypeSpec, ...] | None,
) -> CppType:
    if stage.kind not in {
        "tensor_copy",
        "tensor_unary",
        "tensor_binary",
        "tensor_ternary",
        "tensor_cumsum",
        "tensor_ffill",
        "tensor_shift",
        "tensor_ewm",
        "tensor_column",
    }:
        return _original_stage_type(
            stage, n, execution, input_types=input_types
        )

    tensors = _tensor_inputs(stage, n, input_types)
    out = base._dest_type(stage)
    shape = base._tensor_shape(stage.out.shape)

    if stage.kind == "tensor_copy":
        return tmpl("stackdsl::TensorCopyNode", tensors[0], out, execution)
    if stage.kind == "tensor_unary":
        return tmpl(
            "stackdsl::TensorUnaryNode",
            tensors[0],
            out,
            shape,
            base._cpp_type(stage.dtype),
            Name(base._UNARY_POLICIES[stage.op_name or ""]),
            execution,
        )
    if stage.kind == "tensor_binary":
        return tmpl(
            "stackdsl::TensorBinaryNode",
            tensors[0],
            tensors[1],
            out,
            shape,
            base._cpp_type(stage.dtype),
            Name(base._BINARY_POLICIES[stage.op_name or ""]),
            execution,
        )
    if stage.kind == "tensor_ternary":
        return tmpl(
            "stackdsl::TensorTernaryNode",
            tensors[0],
            tensors[1],
            tensors[2],
            out,
            shape,
            base._cpp_type(stage.dtype),
            Name(base._TERNARY_POLICIES[stage.op_name or ""]),
            execution,
        )
    if stage.kind == "tensor_cumsum":
        return tmpl("stackdsl::TensorCumsumNode", tensors[0], out, execution)
    if stage.kind == "tensor_ffill":
        assert isinstance(stage.op, FFillOp)
        limit = -1 if stage.op.limit is None else stage.op.limit
        return tmpl(
            "stackdsl::TensorFFillNode",
            tensors[0],
            out,
            SignedValueArg(limit),
            execution,
        )
    if stage.kind == "tensor_shift":
        assert isinstance(stage.op, ShiftOp)
        return tmpl(
            "stackdsl::TensorShiftNode",
            tensors[0],
            out,
            IntArg(stage.op.lag),
            IntArg(stage.op.max_lag),
            execution,
        )
    if stage.kind == "tensor_ewm":
        assert isinstance(stage.op, EwmOp)
        return tmpl(
            "stackdsl::TensorEwmNode",
            tensors[0],
            out,
            UInt64Arg(base.double_bits(stage.op.span)),
            IntArg(stage.op.min_periods),
            BoolArg(stage.op.ignore_na),
            BoolArg(stage.op.adjust),
            execution,
        )
    if stage.kind == "tensor_column":
        assert isinstance(stage.op, ColumnOp)
        return tmpl(
            "stackdsl::TensorColumnNode",
            tensors[0],
            out,
            IntArg(stage.op.index),
            execution,
        )
    raise AssertionError(stage.kind)


def install() -> None:
    base._stage_type = _stage_type


__all__ = ["install"]
