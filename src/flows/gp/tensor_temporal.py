from __future__ import annotations

from functools import partial

from flows.gp import pset as row_pset
from flows.gp.tensor_types import TensorIndex, reduced_type, tensor_semantic, tensor_type, tensor_types_for_rank
from flows.gp.types import PositiveInt, PositiveNumber, QuantileParam
from flows.gp.utils_primitives import _call
from trading_dsl_engine.cpp_stream.python import utils as cpp_stream_utils


TENSOR_VEC_FAMILIES = frozenset({
    "vec_avg", "vec_choose", "vec_count", "vec_ir", "vec_kurtosis",
    "vec_max", "vec_min", "vec_norm", "vec_percentage", "vec_powersum",
    "vec_range", "vec_skewness", "vec_stddev", "vec_sum",
})


def _vec_output(type_, family):
    semantic = tensor_semantic(type_)
    if family in {"vec_ir", "vec_skewness", "vec_kurtosis"}:
        return reduced_type(type_, "dimensionless")
    if family == "vec_count":
        return reduced_type(type_, "count")
    if family == "vec_powersum":
        return reduced_type(type_, "derived")
    if family == "vec_sum" and semantic in {"bool", "count"}:
        return reduced_type(type_, "count")
    if family in {"vec_avg", "vec_norm", "vec_range", "vec_stddev"} and semantic in {"bool", "count"}:
        return reduced_type(type_, "dimensionless")
    return reduced_type(type_)


def register_tensor_temporal(reg, ranks):
    families = set(TENSOR_VEC_FAMILIES)
    for rank in sorted(set(ranks)):
        root = tensor_type(rank, "numeric")
        derived = tensor_type(rank, "derived")
        dim = tensor_type(rank, "dimensionless")
        for type_ in tensor_types_for_rank(rank):
            tag = f"tensor_{type_.__name__.lower()}"
            reg.add("cumsum", partial(row_pset._core_call, "cumsum", type_), (type_,), type_, variant=tag)
            reg.add("ffill", partial(row_pset._core_call, "ffill", type_), (type_, PositiveInt), type_, variant=tag)
            reg.add("shift", partial(row_pset._safe_shift, type_), (type_, PositiveInt), type_, variant=tag)
            reg.add("diff", partial(row_pset._safe_diff, type_), (type_, PositiveInt), type_, variant=tag)
            reg.add("ewm", partial(row_pset._ewm1, "ewm", type_), (type_, PositiveInt), type_, variant=tag)
            reg.add("ewm_std", partial(row_pset._ewm1, "ewm_std", type_), (type_, PositiveInt), type_, variant=tag)
            reg.add("ewm_var", partial(row_pset._ewm1, "ewm_var", derived), (type_, PositiveInt), derived, variant=tag)
            reg.add("ewm_moment", partial(row_pset._ewm_moment, derived), (type_, PositiveInt), derived, variant=tag)
            for name in ("ewm_skewness", "ewm_kurtosis"):
                reg.add(name, partial(row_pset._ewm1, name, dim), (type_, PositiveInt), dim, variant=tag)
            for family in TENSOR_VEC_FAMILIES:
                ret = _vec_output(type_, family)
                if family == "vec_choose":
                    args = (type_, TensorIndex)
                elif family == "vec_percentage":
                    args = (type_, QuantileParam)
                elif family == "vec_powersum":
                    args = (type_, PositiveNumber)
                else:
                    args = (type_,)
                reg.add(family, partial(_call, getattr(cpp_stream_utils, family), ret), args, ret, variant=tag)
        for name, ret in (("ewm_cov", derived), ("ewm_corr", dim), ("ewm_co_skewness", derived), ("ewm_co_kurtosis", derived)):
            reg.add(name, partial(row_pset._ewm2, name, ret), (root, root, PositiveInt), ret, variant=f"tensor_rank_{rank}")
        for name in ("ewm_triple_corr", "ewm_partial_corr"):
            reg.add(name, partial(row_pset._ewm3, name, dim), (root, root, root, PositiveInt), dim, variant=f"tensor_rank_{rank}")
    families.update((
        "cumsum", "ffill", "shift", "diff", "ewm", "ewm_std", "ewm_var",
        "ewm_moment", "ewm_skewness", "ewm_kurtosis", "ewm_cov", "ewm_corr",
        "ewm_co_skewness", "ewm_co_kurtosis", "ewm_triple_corr", "ewm_partial_corr",
    ))
    return frozenset(families)


__all__ = ["TENSOR_VEC_FAMILIES", "register_tensor_temporal"]
