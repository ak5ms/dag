from __future__ import annotations

from dataclasses import dataclass
from typing import Union
import ast
import re


@dataclass(frozen=True, eq=False)
class Expr:
    def _call(self, name: str, *args):
        from trading_dsl_engine.base.dsl import call

        return call(name, self, *args)

    def __add__(self, other):
        return self._call("add", other)

    def __radd__(self, other):
        from trading_dsl_engine.base.dsl import call

        return call("add", other, self)

    def __sub__(self, other):
        return self._call("sub", other)

    def __rsub__(self, other):
        from trading_dsl_engine.base.dsl import call

        return call("sub", other, self)

    def __mul__(self, other):
        return self._call("mul", other)

    def __rmul__(self, other):
        from trading_dsl_engine.base.dsl import call

        return call("mul", other, self)

    def __truediv__(self, other):
        return self._call("div", other)

    def __rtruediv__(self, other):
        from trading_dsl_engine.base.dsl import call

        return call("div", other, self)

    def __mod__(self, other):
        return self._call("mod", other)

    def __rmod__(self, other):
        from trading_dsl_engine.base.dsl import call

        return call("mod", other, self)

    def __and__(self, other):
        return self._call("and_", other)

    def __rand__(self, other):
        from trading_dsl_engine.base.dsl import call

        return call("and_", other, self)

    def __or__(self, other):
        return self._call("or_", other)

    def __ror__(self, other):
        from trading_dsl_engine.base.dsl import call

        return call("or_", other, self)

    def __xor__(self, other):
        return self._call("xor", other)

    def __rxor__(self, other):
        from trading_dsl_engine.base.dsl import call

        return call("xor", other, self)

    def __neg__(self):
        from trading_dsl_engine.base.dsl import call

        return call("sub", 0.0, self)

    def __invert__(self):
        from trading_dsl_engine.base.dsl import call

        return call("sub", 1.0, self)

    def __abs__(self):
        return self._call("abs")

    def __eq__(self, other):
        return self._call("eq", other)

    def __ne__(self, other):
        return self._call("ne", other)

    def __floordiv__(self, other):
        return self._call("floordiv", other)

    def __rfloordiv__(self, other):
        from trading_dsl_engine.base.dsl import call

        return call("floordiv", other, self)

    def __pow__(self, other):
        return self._call("pow", other)

    def __rpow__(self, other):
        from trading_dsl_engine.base.dsl import call

        return call("pow", other, self)

    def __lt__(self, other):
        return self._call("lt", other)

    def __gt__(self, other):
        return self._call("gt", other)

    def __le__(self, other):
        return self._call("le", other)

    def __ge__(self, other):
        return self._call("ge", other)

    def sum(self, axis=None, ignore_na=True):
        from trading_dsl_engine.base.dsl import reduction

        return reduction("sum", self, axis=axis, ignore_na=ignore_na)

    def mean(self, axis=None, ignore_na=True):
        from trading_dsl_engine.base.dsl import reduction

        return reduction("mean", self, axis=axis, ignore_na=ignore_na)

    def std(self, axis=None, ddof=0, ignore_na=True):
        from trading_dsl_engine.base.dsl import reduction

        return reduction(
            "std", self, axis=axis, ddof=ddof, ignore_na=ignore_na
        )

    def emit(self, mode="last"):
        from trading_dsl_engine.base.dsl import emit

        return emit(self, mode=mode)

    def groupby(self, key, rhs=None, *args):
        from trading_dsl_engine.base.dsl import grouped

        grouped_expr = grouped(self, key)
        if rhs is None:
            return grouped_expr
        return grouped_expr.apply(rhs, *args)

    def plot(
        self,
        backend: str = "pydot",
        *,
        show: bool = True,
        rankdir: str = "LR",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot this uncompiled formula graph and return the backend graph object."""
        from trading_dsl_engine.visualization import plot

        return plot(
            self,
            backend=backend,
            show=show,
            rankdir=rankdir,
            figsize=figsize,
        )

    def __getattr__(self, name: str):
        from trading_dsl_engine.base.dsl import call, get_dsl_op_signature

        if get_dsl_op_signature(name) is None:
            raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

        def _method_chained_op(*args, **kwargs):
            return call(name, self, *args, **kwargs)

        return _method_chained_op


@dataclass(frozen=True, eq=False)
class Identifier(Expr):
    name: str


@dataclass(frozen=True, eq=False)
class Number(Expr):
    value: int | float


@dataclass(frozen=True, eq=False)
class String(Expr):
    value: str


@dataclass(frozen=True, eq=False)
class Call(Expr):
    fn: str
    args: tuple[Expr, ...]
    kwargs: tuple[tuple[str, Expr], ...] = ()


UniverseItem = Union[str, int]


@dataclass(frozen=True, eq=False)
class Universe(Expr):
    groups: tuple[tuple[UniverseItem, ...], ...]


@dataclass(frozen=True, eq=False)
class KeyTuple(Expr):
    items: tuple[Expr, ...]


class FormulaParseError(ValueError):
    pass


_BINOP_NAMES: dict[type[ast.operator], str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.Mod: "mod",
    ast.BitAnd: "and_",
    ast.BitOr: "or_",
    ast.BitXor: "xor",
    ast.FloorDiv: "floordiv",
    ast.Pow: "pow",
}

_CMP_NAMES: dict[type[ast.cmpop], str] = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Gt: "gt",
    ast.Lt: "lt",
    ast.GtE: "ge",
    ast.LtE: "le",
}


class _AstParser:
    def parse(self, text: str) -> Expr:
        source = text.strip()
        source = re.sub(r"\band\s*\(", "and_(", source)
        source = re.sub(r"\bor\s*\(", "or_(", source)
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as exc:
            raise FormulaParseError(f"Syntax error at line {exc.lineno}, col {exc.offset}: {exc.msg}") from exc
        return self._expr(tree.body)

    def _expr(self, node: ast.AST) -> Expr:
        if isinstance(node, ast.Name):
            return Identifier(node.id)
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float)
        ):
            value = node.value
            return Number(float(value) if isinstance(value, bool) else value)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return String(node.value)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                if isinstance(node.operand, ast.Constant):
                    v = node.operand.value
                    if isinstance(v, (int, float)):
                        return Number(
                            -float(v) if isinstance(v, bool) else -v
                        )
                return Call("sub", (Number(0.0), self._expr(node.operand)))
            if isinstance(node.op, ast.UAdd):
                return self._expr(node.operand)
            if isinstance(node.op, ast.Invert):
                return Call("sub", (Number(1.0), self._expr(node.operand)))
        if isinstance(node, ast.BinOp):
            op_name = self._binop_name(node.op)
            return Call(op_name, (self._expr(node.left), self._expr(node.right)))
        if isinstance(node, (ast.Tuple, ast.List)):
            if len(node.elts) == 0:
                raise FormulaParseError("Expression lists cannot be empty")
            return KeyTuple(tuple(self._expr(item) for item in node.elts))
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise FormulaParseError("Chained comparisons are not supported")
            op_name = self._cmpop_name(node.ops[0])
            return Call(op_name, (self._expr(node.left), self._expr(node.comparators[0])))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FormulaParseError("Only direct function names are supported")
            if node.func.id == "univ":
                if node.keywords:
                    raise FormulaParseError("univ keyword arguments are not supported")
                return self._universe(node)
            kwargs: list[tuple[str, Expr]] = []
            for kw in node.keywords:
                if kw.arg is None:
                    raise FormulaParseError("**kwargs are not supported")
                kwargs.append((kw.arg, self._expr(kw.value)))
            return Call(node.func.id, tuple(self._expr(arg) for arg in node.args), tuple(kwargs))
        raise FormulaParseError(f"Unsupported syntax: {ast.dump(node, include_attributes=False)}")

    def _universe(self, node: ast.Call) -> Universe:
        groups: list[tuple[UniverseItem, ...]] = []
        for arg in node.args:
            if isinstance(arg, (ast.List, ast.Tuple)):
                members = tuple(self._universe_item(item) for item in arg.elts)
            else:
                members = (self._universe_item(arg),)
            if len(members) == 0:
                raise FormulaParseError("Universe groups cannot be empty")
            groups.append(members)
        if len(groups) == 0:
            raise FormulaParseError("univ expects at least one group")
        return Universe(tuple(groups))

    def _universe_item(self, node: ast.AST) -> UniverseItem:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value
            if isinstance(node.value, int):
                return node.value
        raise FormulaParseError(f"Unsupported universe member: {ast.dump(node, include_attributes=False)}")

    def _binop_name(self, op: ast.operator) -> str:
        try:
            return _BINOP_NAMES[type(op)]
        except KeyError as exc:
            raise FormulaParseError(f"Unsupported binary operator: {op.__class__.__name__}") from exc

    def _cmpop_name(self, op: ast.cmpop) -> str:
        try:
            return _CMP_NAMES[type(op)]
        except KeyError as exc:
            raise FormulaParseError(f"Unsupported comparison operator: {op.__class__.__name__}") from exc


def parse_formula(text: str) -> Expr:
    return _AstParser().parse(text)
