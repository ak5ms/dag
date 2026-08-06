from __future__ import annotations

import math
import struct

from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple, Number, String, Universe


_COMMUTATIVE = frozenset({"add", "mul", "eq", "ne", "and", "and_", "or", "or_", "xor"})
_ASSOCIATIVE = frozenset({"add", "mul"})


def _number_key(value: float) -> tuple[str, int] | tuple[str]:
    numeric = float(value)
    if math.isnan(numeric):
        return ("nan",)
    return ("bits", struct.unpack("!Q", struct.pack("!d", numeric))[0])


def expression_key(expr: Expr) -> tuple:
    if isinstance(expr, Identifier):
        return ("id", expr.name)
    if isinstance(expr, Number):
        return ("num", _number_key(expr.value))
    if isinstance(expr, String):
        return ("str", expr.value)
    if isinstance(expr, Universe):
        return ("univ", expr.groups)
    if isinstance(expr, KeyTuple):
        return ("tuple", tuple(expression_key(item) for item in expr.items))
    if isinstance(expr, Call):
        args = [expression_key(arg) for arg in expr.args]
        if expr.fn in _ASSOCIATIVE:
            flattened: list[tuple] = []
            for arg, key in zip(expr.args, args):
                if isinstance(arg, Call) and arg.fn == expr.fn and not arg.kwargs:
                    flattened.extend(expression_key(child) for child in arg.args)
                else:
                    flattened.append(key)
            args = flattened
        if expr.fn in _COMMUTATIVE:
            args.sort(key=repr)
        return (
            "call",
            expr.fn,
            tuple(args),
            tuple((name, expression_key(value)) for name, value in expr.kwargs),
        )
    if hasattr(expr, "args"):
        return (
            type(expr).__name__,
            getattr(expr, "name", None),
            tuple(expression_key(arg) for arg in expr.args),
        )
    return (type(expr).__name__, repr(expr))


def expression_identifiers(expr: Expr) -> frozenset[str]:
    names: set[str] = set()

    def visit(node: Expr) -> None:
        if isinstance(node, Identifier):
            if node.name not in {"True", "False", "self_"}:
                names.add(node.name)
            return
        if isinstance(node, KeyTuple):
            for item in node.items:
                visit(item)
            return
        if isinstance(node, Call):
            for argument in node.args:
                visit(argument)
            for _, value in node.kwargs:
                visit(value)
            return
        if hasattr(node, "args"):
            for argument in node.args:
                visit(argument)

    visit(expr)
    return frozenset(names)


def canonical_string(expr: Expr) -> str:
    return repr(expression_key(expr))


__all__ = ["canonical_string", "expression_identifiers", "expression_key"]
