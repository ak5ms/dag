from __future__ import annotations

from functools import partial

from flows.gp.tensor_types import tensor_type, tensor_types_for_rank
from flows.gp.types import BoolParam, ExprValue, PositiveFloat, PositiveInt, PositiveNumber
from flows.gp.utils_primitives import _call, _nan_out, _replace
from trading_dsl_engine.cpp_stream.python import utils as cpp_utils

TENSOR_ELEMENTWISE_UTILS = frozenset({
    "arc_cos", "arc_sin", "arc_tan", "bucket", "clamp", "convert_float",
    "elementwise_max", "elementwise_min", "equal", "get_df", "if_else",
    "inverse", "is_finite", "is_nan", "is_not_finite", "is_not_nan",
    "left_right_tail", "left_tail", "less", "log", "log_diff",
    "logical_and", "logical_or", "nan_mask", "nan_out", "negate",
    "pasteurize", "replace", "reverse", "right_tail", "round_df",
    "round_down", "s_log_1p", "sigmoid", "signed_power", "tail", "to_nan",
})


def _add(reg, name, ret, args, variant):
    reg.add(name, partial(_call, getattr(cpp_utils, name), ret), args, ret, variant=variant)


def _tensor_bucket(spec, ret: type[ExprValue], value: ExprValue):
    mode, text = spec
    kwargs = {"buckets": text} if mode == "buckets" else {"range_": text}
    return ret(cpp_utils.bucket(value.expr, **kwargs))


def register_tensor_elementwise_utils(reg, ranks, config):
    for rank in sorted(set(ranks)):
        root = tensor_type(rank, "numeric")
        derived = tensor_type(rank, "derived")
        dim = tensor_type(rank, "dimensionless")
        boolean = tensor_type(rank, "bool")
        count = tensor_type(rank, "count")
        for type_ in tensor_types_for_rank(rank):
            tag = f"tensor_{type_.__name__.lower()}"
            for name in ("reverse", "pasteurize", "convert_float"):
                _add(reg, name, type_, (type_,), tag)
            for name in ("round_down", "left_tail", "right_tail"):
                _add(reg, name, type_, (type_,), tag + "_default")
                _add(reg, name, type_, (type_, PositiveFloat), tag + "_value")
            _add(reg, "round_df", type_, (type_, PositiveInt), tag)
            _add(reg, "to_nan", type_, (type_,), tag + "_default")
            _add(reg, "to_nan", type_, (type_, PositiveNumber, BoolParam), tag + "_full")
            _add(reg, "tail", type_, (type_,), tag + "_default")
            _add(reg, "tail", type_, (type_, PositiveNumber, PositiveNumber, PositiveNumber), tag + "_full")
            _add(reg, "left_right_tail", type_, (type_, PositiveNumber, PositiveNumber), tag)
            _add(reg, "clamp", type_, (type_,), tag + "_default")
            _add(reg, "clamp", type_, (type_, PositiveNumber, PositiveNumber), tag + "_bounds")
            _add(reg, "clamp", type_, (type_, PositiveNumber, PositiveNumber, BoolParam), tag + "_inverse")
            _add(reg, "nan_mask", type_, (type_, root), tag)
            for mode, args in (("lower", (type_, PositiveNumber)), ("upper", (type_, PositiveNumber)), ("both", (type_, PositiveNumber, PositiveNumber))):
                reg.add("nan_out", partial(_nan_out, mode, type_), args, type_, variant=tag + "_" + mode)
            for index, spec in enumerate(config.replace_specs):
                reg.add("replace", partial(_replace, tuple(spec), type_), (type_,), type_, variant=f"{tag}_{index}")
            for name in ("elementwise_max", "elementwise_min"):
                for arity in (2, 3, 4):
                    _add(reg, name, type_, (type_,) * arity, f"{tag}_{arity}")
            _add(reg, "if_else", type_, (boolean, type_, type_), tag)
        for name, ret in (("log", dim), ("log_diff", dim), ("s_log_1p", dim), ("sigmoid", dim), ("arc_cos", dim), ("arc_sin", dim), ("arc_tan", dim), ("inverse", derived)):
            _add(reg, name, ret, (root,), f"tensor_rank_{rank}")
        _add(reg, "signed_power", derived, (root, root), f"tensor_rank_{rank}_row")
        _add(reg, "signed_power", derived, (root, PositiveNumber), f"tensor_rank_{rank}_scalar")
        for name in ("is_not_nan", "is_nan", "is_finite", "is_not_finite"):
            _add(reg, name, boolean, (root,), f"tensor_rank_{rank}")
        for name in ("equal", "less"):
            _add(reg, name, boolean, (root, root), f"tensor_rank_{rank}")
        _add(reg, "negate", boolean, (boolean,), f"tensor_rank_{rank}")
        for name in ("logical_and", "logical_or"):
            _add(reg, name, boolean, (boolean, boolean), f"tensor_rank_{rank}")
        _add(reg, "get_df", derived, (root, PositiveNumber), f"tensor_rank_{rank}")
        for index, spec in enumerate(config.bucket_specs):
            reg.add("bucket", partial(_tensor_bucket, tuple(spec), count), (root,), count, variant=f"tensor_rank_{rank}_{index}")
    return TENSOR_ELEMENTWISE_UTILS


__all__ = ["TENSOR_ELEMENTWISE_UTILS", "register_tensor_elementwise_utils"]
