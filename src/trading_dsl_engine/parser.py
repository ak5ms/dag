from __future__ import annotations

from dataclasses import dataclass
import ast


@dataclass(frozen=True)
class Expr:
    pass


@dataclass(frozen=True)
class Identifier(Expr):
    name: str


@dataclass(frozen=True)
class Number(Expr):
    value: float


@dataclass(frozen=True)
class Call(Expr):
    fn: str
    args: tuple[Expr, ...]


class FormulaParseError(ValueError):
    pass


class _AstParser:
    def parse(self, text: str) -> Expr:
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise FormulaParseError(f"Syntax error at line {exc.lineno}, col {exc.offset}: {exc.msg}") from exc
        return self._expr(tree.body)

    def _expr(self, node: ast.AST) -> Expr:
        if isinstance(node, ast.Name):
            return Identifier(node.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Number(float(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            v = node.operand.value
            if isinstance(v, (int, float)):
                return Number(-float(v))
        if isinstance(node, ast.Call):
            if node.keywords:
                raise FormulaParseError("Keyword arguments are not supported")
            if not isinstance(node.func, ast.Name):
                raise FormulaParseError("Only direct function names are supported")
            return Call(node.func.id, tuple(self._expr(arg) for arg in node.args))
        raise FormulaParseError(f"Unsupported syntax: {ast.dump(node, include_attributes=False)}")


def parse_formula(text: str) -> Expr:
    return _AstParser().parse(text)
