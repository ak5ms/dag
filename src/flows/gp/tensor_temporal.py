from __future__ import annotations

from functools import partial

from flows.gp import pset as row_pset
from flows.gp.tensor_types import tensor_type, tensor_types_for_rank
from flows.gp.types import PositiveInt


def register_tensor_temporal(reg, ranks):
    families = set()
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


__all__ = ["register_tensor_temporal"]
