from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable

from trading_dsl_engine.base.parser import Call, Number
from trading_dsl_engine.cpp_stream.python import codegen as codegen_base
from trading_dsl_engine.cpp_stream.python.codegen import (
    BoolArg,
    CppType,
    IntArg,
    Stage,
    tmpl,
)
from trading_dsl_engine.cpp_stream.python.lowering import GroupStage, Plan
from trading_dsl_engine.cpp_stream.python.npy import InputTypeSpec
from trading_dsl_engine.ir.ops import EwmOp


@dataclass(frozen=True, slots=True)
class DynamicEwmOp(EwmOp):
    """EWM whose span is supplied by a scalar or lane-valued child expression."""


def install_frontend(neutral_frontend) -> None:
    """Teach the neutral builder to preserve non-literal EWM span expressions."""
    original = neutral_frontend._BaseBuilder._build_call
    if getattr(original, "_dynamic_ewm_installed", False):
        return

    def build_call(self, node):
        if not isinstance(node, Call) or node.fn != "ewm":
            return original(self, node)

        names = ("x", "span", "min_periods", "ignore_na", "adjust")
        values = {
            "min_periods": Number(0.0),
            "ignore_na": Number(1.0),
            "adjust": Number(0.0),
        }
        explicit: set[str] = set()
        if len(node.args) > len(names):
            raise neutral_frontend.FormulaIRCompileError("ewm received too many arguments")
        for name, value in zip(names, node.args):
            values[name] = value
            explicit.add(name)
        for name, value in node.kwargs:
            if name not in names or name in explicit:
                raise neutral_frontend.FormulaIRCompileError(
                    f"invalid ewm argument {name!r}"
                )
            values[name] = value
            explicit.add(name)
        if "x" not in values or "span" not in values:
            raise neutral_frontend.FormulaIRCompileError("ewm requires x and span")

        span_expr = values["span"]
        if isinstance(span_expr, Number):
            return original(self, node)

        child = self.build(values["x"])
        span_child = self.build(span_expr)
        child_type = self.nodes[child].value_type
        span_type = self.nodes[span_child].value_type
        try:
            child_shape = child_type.logical_shape
            span_shape = span_type.logical_shape
        except ValueError as exc:
            raise neutral_frontend.FormulaIRCompileError(
                "dynamic ewm span requires numeric scalar/vector expressions"
            ) from exc
        if span_shape not in {(), child_shape}:
            raise neutral_frontend.FormulaIRCompileError(
                "dynamic ewm span must be row-scalar or match the input shape; "
                f"got input={child_shape!r} span={span_shape!r}"
            )

        minimum = neutral_frontend._literal_int(
            values["min_periods"], "min_periods", 0
        )
        ignore = neutral_frontend._literal_bool(values["ignore_na"], "ignore_na")
        adjust = neutral_frontend._literal_bool(values["adjust"], "adjust")
        return self._append(
            DynamicEwmOp(math.nan, minimum, ignore, adjust),
            (child, span_child),
            neutral_frontend._lane_state_result_type("ewm", self.nodes[child]),
        )

    build_call._dynamic_ewm_installed = True
    neutral_frontend._BaseBuilder._build_call = build_call


def _rewrite_plan(plan: Plan) -> Plan:
    stages: list[Stage] = []
    for stage in plan.stages:
        group = stage.group
        if group is not None:
            group = replace(group, inner=_rewrite_plan(group.inner))
            stage = replace(stage, group=group)
        if isinstance(stage.op, DynamicEwmOp):
            if stage.kind == "ewm_bundle":
                raise ValueError("dynamic-span EWMs cannot use the fixed-span bundle")
            if stage.kind == "ewm":
                stage = replace(stage, kind="dynamic_ewm")
            elif stage.kind == "tensor_ewm":
                stage = replace(stage, kind="tensor_dynamic_ewm")
        stages.append(stage)
    return replace(plan, stages=tuple(stages))


def wrap_lower_program(original: Callable) -> Callable:
    def lower_program(*args, **kwargs):
        return _rewrite_plan(original(*args, **kwargs))

    return lower_program


def install_codegen() -> None:
    """Route dynamic EWM stages to the runtime-span native state machine."""
    original = codegen_base._stage_type
    if getattr(original, "_dynamic_ewm_installed", False):
        return

    def stage_type(
        stage: Stage,
        n: CppType,
        execution: CppType,
        *,
        input_types: tuple[InputTypeSpec, ...] | None,
    ) -> CppType:
        if stage.kind == "tensor_dynamic_ewm":
            raise ValueError("tensor-valued dynamic EWM spans are not yet supported")
        if stage.kind != "dynamic_ewm":
            return original(stage, n, execution, input_types=input_types)
        if not isinstance(stage.op, DynamicEwmOp) or len(stage.inputs) != 2:
            raise ValueError("malformed dynamic EWM stage")
        stage_n: CppType = IntArg(1) if stage.lane_count == 1 else n
        inputs = tuple(
            codegen_base._source_type(source, n=n, input_types=input_types)
            for source in stage.inputs
        )
        return tmpl(
            "stackdsl::DynamicEwmNode",
            stage_n,
            inputs[0],
            inputs[1],
            codegen_base._dest_type(stage),
            IntArg(stage.op.min_periods),
            BoolArg(stage.op.ignore_na),
            BoolArg(stage.op.adjust),
            execution,
        )

    stage_type._dynamic_ewm_installed = True
    codegen_base._stage_type = stage_type


__all__ = [
    "DynamicEwmOp",
    "install_codegen",
    "install_frontend",
    "wrap_lower_program",
]
