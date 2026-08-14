from __future__ import annotations

from functools import partial

from flows.gp.tensor_types import tensor_type, tensor_types_for_rank
from flows.gp.types import ExprValue, PositiveFloat, PositiveInt
from flows.gp.utils_primitives import _call, _ewm_vector
from trading_dsl_engine.cpp_stream.python import utils as cpp_utils

TENSOR_TEMPORAL_UTILS = frozenset({
    "ewm_vector_neut", "ewm_vector_proj", "slope", "ts_diff", "ts_ln_change",
    "ts_pct_change", "ts_returns", "ts_shift", "ts_vector_neut",
    "ts_vector_proj", "ts_weighted_delay",
})


def _add(reg, name, ret, args, variant):
    reg.add(name, partial(_call, getattr(cpp_utils, name), ret), args, ret, variant=variant)


def _tensor_returns(mode: int, ret: type[ExprValue], value: ExprValue, periods: PositiveInt):
    return ret(cpp_utils.ts_returns(value.expr, periods.value, mode=mode))


def register_tensor_temporal_utils(reg, ranks):
    for rank in sorted(set(ranks)):
        root = tensor_type(rank, "numeric")
        derived = tensor_type(rank, "derived")
        dim = tensor_type(rank, "dimensionless")
        for type_ in tensor_types_for_rank(rank):
            tag = f"tensor_{type_.__name__.lower()}"
            _add(reg, "ts_shift", type_, (type_, PositiveInt), tag)
            _add(reg, "ts_diff", type_, (type_, PositiveInt), tag)
            _add(reg, "ts_weighted_delay", type_, (type_,), tag + "_default")
            _add(reg, "ts_weighted_delay", type_, (type_, PositiveFloat), tag + "_weight")
        for mode in (1, 2):
            reg.add("ts_returns", partial(_tensor_returns, mode, dim), (root, PositiveInt), dim, variant=f"tensor_rank_{rank}_mode_{mode}")
        for name in ("ts_pct_change", "ts_ln_change"):
            _add(reg, name, dim, (root, PositiveInt), f"tensor_rank_{rank}")
        for name in ("ewm_vector_proj", "ewm_vector_neut", "ts_vector_proj", "ts_vector_neut"):
            reg.add(name, partial(_ewm_vector, getattr(cpp_utils, name), derived), (root, root, PositiveInt), derived, variant=f"tensor_rank_{rank}")
        _add(reg, "slope", derived, (root, PositiveInt), f"tensor_rank_{rank}")
    return TENSOR_TEMPORAL_UTILS


__all__ = ["TENSOR_TEMPORAL_UTILS", "register_tensor_temporal_utils"]
