from __future__ import annotations

from functools import partial

from flows.gp import pset as row_pset
from flows.gp.tensor_types import tensor_type, tensor_types_for_rank
from flows.gp.types import PositiveNumber


def _core(reg, family, args, ret, variant, op=None):
    reg.add(family, partial(row_pset._core_call, op or family, ret), args, ret, variant=variant)


def register_tensor_elementwise(reg, ranks):
    families = set()
    preserve = ("add", "sub", "minimum", "maximum", "fillna", "mod")
    compare = ("eq", "ne", "lt", "gt", "le", "ge")
    unitless = ("ln", "exp", "sign", "fraction", "arctan", "acos", "asin", "sin", "cos", "tan", "tanh", "norm_inv")
    for rank in sorted(set(ranks)):
        boolean = tensor_type(rank, "bool")
        dim = tensor_type(rank, "dimensionless")
        derived = tensor_type(rank, "derived")
        root = tensor_type(rank, "numeric")
        for type_ in tensor_types_for_rank(rank):
            tag = f"tensor_{type_.__name__.lower()}"
            for family in preserve:
                _core(reg, family, (type_, type_), type_, tag)
                _core(reg, family, (type_, PositiveNumber), type_, tag + "_scalar")
            for family in compare:
                _core(reg, family, (type_, type_), boolean, tag)
                _core(reg, family, (type_, PositiveNumber), boolean, tag + "_scalar")
            _core(reg, "where", (boolean, type_, type_), type_, tag)
            _core(reg, "where", (boolean, type_, PositiveNumber), type_, tag + "_scalar_false")
            _core(reg, "where", (boolean, PositiveNumber, type_), type_, tag + "_scalar_true")
            reg.add("clip", partial(row_pset._clip_call, type_), (type_, type_, type_), type_, variant=tag)
            reg.add("clip", partial(row_pset._clip_call, type_), (type_, PositiveNumber, PositiveNumber), type_, variant=tag + "_scalar")
            for family in ("abs", "purify", "floor", "ceil", "round"):
                _core(reg, family, (type_,), type_, tag)
            for family in ("isnan", "isfinite"):
                _core(reg, family, (type_,), boolean, tag)
            for family in unitless:
                _core(reg, family, (type_,), dim, tag)
            _core(reg, "sqrt", (type_,), derived, tag)
            for family in ("mul", "div"):
                _core(reg, family, (type_, PositiveNumber), type_, tag + "_scalar")
            reg.add("floordiv", partial(row_pset._floordiv_call, type_), (type_, PositiveNumber), type_, variant=tag + "_scalar")
            _core(reg, "pow", (type_, PositiveNumber), dim if issubclass(type_, dim) else derived, tag + "_scalar")
            _core(reg, "mul", (type_, type_), derived, tag)
            _core(reg, "div", (type_, type_), dim, tag)
            reg.add("floordiv", partial(row_pset._floordiv_call, dim), (type_, type_), dim, variant=tag)
            if not issubclass(type_, dim):
                _core(reg, "mul", (type_, dim), type_, tag + "_dim")
                _core(reg, "mul", (dim, type_), type_, "dim_" + tag)
                _core(reg, "div", (type_, dim), type_, tag + "_dim")
        for family in ("and", "and_", "or", "or_", "xor"):
            _core(reg, family, (boolean, boolean), boolean, f"tensor_rank_{rank}", {"and": "and_", "or": "or_"}.get(family, family))
        _core(reg, "logical_not", (boolean,), boolean, f"tensor_rank_{rank}")
        _core(reg, "mul", (root, root), derived, f"tensor_rank_{rank}_generic")
        _core(reg, "div", (root, root), derived, f"tensor_rank_{rank}_generic")
    families.update(preserve + compare + unitless + ("where", "clip", "abs", "purify", "floor", "ceil", "round", "isnan", "isfinite", "sqrt", "mul", "div", "floordiv", "pow", "and", "and_", "or", "or_", "xor", "logical_not"))
    return frozenset(families)


__all__ = ["register_tensor_elementwise"]
