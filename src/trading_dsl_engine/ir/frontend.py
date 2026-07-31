from __future__ import annotations

from dataclasses import dataclass

from trading_dsl_engine.base.dsl import DEFAULT_DSL_REGISTRY, DSLFunctionRegistry
from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple, Number, String, Universe, parse_formula
from trading_dsl_engine.ir.ops import CumsumOp, EwmOp, GroupByOp, InputOp, LiteralOp, NaryOp, XsRankOp
from trading_dsl_engine.ir.program import Node, Program
from trading_dsl_engine.ir.types import SCALAR, VECTOR, ValueType


class FormulaIRCompileError(ValueError):
    pass


_NARY_ARITY = {"add": 2, "sub": 2, "mul": 2, "div": 2}


def _expr_key(node: Expr) -> tuple:
    if isinstance(node, Identifier):
        return ("id", node.name)
    if isinstance(node, Number):
        return ("num", node.value)
    if isinstance(node, String):
        return ("str", node.value)
    if isinstance(node, Universe):
        return ("univ", node.groups)
    if isinstance(node, KeyTuple):
        return ("tuple", tuple(_expr_key(item) for item in node.items))
    if isinstance(node, Call):
        return ("call", node.fn, tuple(_expr_key(arg) for arg in node.args), tuple((name, _expr_key(value)) for name, value in node.kwargs))
    raise FormulaIRCompileError(f"unhandled expression for key: {node!r}")


def _contains_self(node: Expr) -> bool:
    if isinstance(node, Identifier):
        return node.name == "self_"
    if isinstance(node, Call):
        return any(_contains_self(arg) for arg in node.args) or any(_contains_self(v) for _, v in node.kwargs)
    if isinstance(node, KeyTuple):
        return any(_contains_self(item) for item in node.items)
    return False


def _literal_number(node: Expr, name: str) -> float:
    if not isinstance(node, Number):
        raise FormulaIRCompileError(f"{name} must be a numeric literal")
    return float(node.value)


def _literal_bool(node: Expr, name: str) -> bool:
    value = _literal_number(node, name)
    if value not in (0.0, 1.0):
        raise FormulaIRCompileError(f"{name} must be a boolean/numeric literal")
    return bool(value)


def _resolve_universe_groups(universe: Universe, column_name_to_index: dict[str, int]) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for group in universe.groups:
        resolved: list[int] = []
        for member in group:
            if isinstance(member, int):
                index = member
            else:
                try:
                    index = column_name_to_index[member]
                except KeyError as exc:
                    raise FormulaIRCompileError(f"unknown universe column {member!r}; pass column_names when using ticker names") from exc
            if index < 0:
                raise FormulaIRCompileError("universe column indexes must be >= 0")
            if index in seen:
                raise FormulaIRCompileError(f"universe column index {index} appears in more than one group")
            seen.add(index)
            resolved.append(index)
        if not resolved:
            raise FormulaIRCompileError("universe groups cannot be empty")
        groups.append(tuple(resolved))
    return tuple(groups)


def _result_type(children: list[Node]) -> ValueType:
    return VECTOR if any(child.value_type.kind == "vector" for child in children) else SCALAR


def _normalize_ewm(call: Call) -> tuple[Expr, float, int, bool, bool]:
    names = ("x", "span", "min_periods", "ignore_na", "adjust")
    if len(call.args) > len(names):
        raise FormulaIRCompileError("ewm accepts at most five positional arguments")
    values: dict[str, Expr] = {"min_periods": Number(0.0), "ignore_na": Number(1.0), "adjust": Number(0.0)}
    explicitly_set: set[str] = set()
    for name, value in zip(names, call.args):
        values[name] = value
        explicitly_set.add(name)
    for name, value in call.kwargs:
        if name not in names:
            raise FormulaIRCompileError(f"unsupported ewm keyword {name!r}")
        if name in explicitly_set:
            raise FormulaIRCompileError(f"ewm argument {name!r} specified more than once")
        values[name] = value
        explicitly_set.add(name)
    if "x" not in values or "span" not in values:
        raise FormulaIRCompileError("ewm requires x and span")
    span = _literal_number(values["span"], "ewm span")
    if not span > 0.0:
        raise FormulaIRCompileError("ewm span must be > 0")
    min_periods = int(round(_literal_number(values["min_periods"], "ewm min_periods")))
    if min_periods < 0:
        raise FormulaIRCompileError("ewm min_periods must be >= 0")
    ignore_na = _literal_bool(values["ignore_na"], "ewm ignore_na")
    adjust = _literal_bool(values["adjust"], "ewm adjust")
    return values["x"], span, min_periods, ignore_na, adjust


@dataclass
class _Builder:
    dsl_registry: DSLFunctionRegistry
    column_name_to_index: dict[str, int]

    def __post_init__(self) -> None:
        self.nodes: list[Node] = []
        self.inputs: dict[str, int] = {}
        self.memo: dict[tuple, int] = {}

    def _append(self, op, child_ids: tuple[int, ...], value_type: ValueType) -> int:
        index = len(self.nodes)
        self.nodes.append(Node(op=op, child_ids=child_ids, value_type=value_type))
        return index

    def build(self, node: Expr, *, use_cache: bool = True) -> int:
        key = _expr_key(node)
        if use_cache and key in self.memo:
            return self.memo[key]
        if isinstance(node, Call):
            macro = self.dsl_registry.get(node.fn)
            if macro is not None:
                try:
                    expanded = macro(*node.args, **dict(node.kwargs))
                except Exception as exc:
                    raise FormulaIRCompileError(f"failed expanding DSL function {node.fn!r}: {exc}") from exc
                if _expr_key(expanded) != key:
                    result = self.build(expanded, use_cache=use_cache)
                    if use_cache:
                        self.memo[key] = result
                    return result
        if isinstance(node, Identifier):
            if node.name == "self_":
                raise FormulaIRCompileError("self_ is only valid inside groupby RHS")
            input_index = self.inputs.setdefault(node.name, len(self.inputs))
            result = self._append(InputOp(input_index, node.name), (), VECTOR)
        elif isinstance(node, Number):
            result = self._append(LiteralOp(float(node.value)), (), SCALAR)
        elif isinstance(node, Call) and node.fn in _NARY_ARITY:
            arity = _NARY_ARITY[node.fn]
            if node.kwargs or len(node.args) != arity:
                raise FormulaIRCompileError(f"{node.fn} expects exactly {arity} positional arguments")
            child_ids = tuple(self.build(arg) for arg in node.args)
            result = self._append(NaryOp(node.fn, arity), child_ids, _result_type([self.nodes[i] for i in child_ids]))
        elif isinstance(node, Call) and node.fn == "cumsum":
            if node.kwargs or len(node.args) != 1:
                raise FormulaIRCompileError("cumsum expects exactly one positional argument")
            child = self.build(node.args[0])
            result = self._append(CumsumOp(), (child,), self.nodes[child].value_type)
        elif isinstance(node, Call) and node.fn == "ewm":
            x, span, min_periods, ignore_na, adjust = _normalize_ewm(node)
            child = self.build(x)
            result = self._append(EwmOp(span, min_periods, ignore_na, adjust), (child,), self.nodes[child].value_type)
        elif isinstance(node, Call) and node.fn == "xs_rank":
            if node.kwargs or len(node.args) != 1:
                raise FormulaIRCompileError("xs_rank expects exactly one positional argument")
            child = self.build(node.args[0])
            if self.nodes[child].value_type.kind != "vector":
                raise FormulaIRCompileError("xs_rank requires a vector input")
            result = self._append(XsRankOp(), (child,), VECTOR)
        elif isinstance(node, Call) and node.fn == "groupby":
            result = self._build_groupby(node)
        else:
            raise FormulaIRCompileError(f"cpp_stream neutral IR does not yet support {getattr(node, 'fn', type(node).__name__)!r}")
        if use_cache:
            self.memo[key] = result
        return result

    def _build_groupby(self, call: Call) -> int:
        if len(call.args) != 3:
            raise FormulaIRCompileError("groupby requires groupby(key_tuple, lhs, op_using_self_)")
        kw = dict(call.kwargs)
        unknown = set(kw) - {"capacity", "hash_capacity"}
        if unknown:
            raise FormulaIRCompileError(f"unsupported groupby keyword argument(s): {sorted(unknown)}")
        capacity = None if "capacity" not in kw else int(round(_literal_number(kw["capacity"], "groupby capacity")))
        hash_capacity = None if "hash_capacity" not in kw else int(round(_literal_number(kw["hash_capacity"], "groupby hash_capacity")))
        if capacity is not None and capacity <= 0:
            raise FormulaIRCompileError("groupby capacity must be > 0")
        if hash_capacity is not None and hash_capacity <= 0:
            raise FormulaIRCompileError("groupby hash_capacity must be > 0")
        key = call.args[0]
        key_items = key.items if isinstance(key, KeyTuple) else (key,)
        universes = [item for item in key_items if isinstance(item, Universe)]
        if len(universes) > 1:
            raise FormulaIRCompileError("groupby key tuple may contain at most one univ(...) element")
        static_groups = None if not universes else _resolve_universe_groups(universes[0], self.column_name_to_index)
        dynamic_items = [item for item in key_items if not isinstance(item, Universe)]
        key_ids = tuple(self.build(item) for item in dynamic_items)
        lhs_id = self.build(call.args[1])
        if self.nodes[lhs_id].value_type.kind != "vector":
            raise FormulaIRCompileError("cpp_stream groupby lhs must be vector-valued")
        inner_builder = _InnerBuilder(self)
        inner_root = inner_builder.build(call.args[2])
        inner_program = Program(nodes=tuple(inner_builder.nodes), outputs=(inner_root,), input_names=("__self__",) + tuple(f"__capture_{i}__" for i in range(len(inner_builder.capture_ids))))
        if inner_program.nodes[inner_root].value_type.kind != "vector":
            raise FormulaIRCompileError("cpp_stream groupby RHS must emit a vector")
        op = GroupByOp(n_dynamic_keys=len(key_ids), static_groups=static_groups, inner_program=inner_program, capacity=capacity, hash_capacity=hash_capacity)
        children = key_ids + (lhs_id,) + tuple(inner_builder.capture_ids)
        return self._append(op, children, VECTOR)


class _InnerBuilder:
    def __init__(self, outer: _Builder) -> None:
        self.outer = outer
        self.nodes: list[Node] = []
        self.memo: dict[tuple, int] = {}
        self.capture_map: dict[tuple, int] = {}
        self.capture_ids: list[int] = []
        self._self_input: int | None = None

    def _append(self, op, child_ids: tuple[int, ...], value_type: ValueType) -> int:
        index = len(self.nodes)
        self.nodes.append(Node(op=op, child_ids=child_ids, value_type=value_type))
        return index

    def _input(self, input_index: int, name: str) -> int:
        return self._append(InputOp(input_index, name), (), VECTOR)

    def build(self, node: Expr) -> int:
        key = _expr_key(node)
        if key in self.memo:
            return self.memo[key]
        if isinstance(node, Identifier) and node.name == "self_":
            if self._self_input is None:
                self._self_input = self._input(0, "__self__")
            return self._self_input
        if isinstance(node, Number):
            result = self._append(LiteralOp(float(node.value)), (), SCALAR)
            self.memo[key] = result
            return result
        if not _contains_self(node):
            capture_pos = self.capture_map.get(key)
            if capture_pos is None:
                outer_id = self.outer.build(node)
                capture_pos = len(self.capture_ids)
                self.capture_map[key] = capture_pos
                self.capture_ids.append(outer_id)
            result = self._input(capture_pos + 1, f"__capture_{capture_pos}__")
            self.memo[key] = result
            return result
        if isinstance(node, Call):
            macro = self.outer.dsl_registry.get(node.fn)
            if macro is not None:
                try:
                    expanded = macro(*node.args, **dict(node.kwargs))
                except Exception as exc:
                    raise FormulaIRCompileError(f"failed expanding grouped DSL function {node.fn!r}: {exc}") from exc
                if _expr_key(expanded) != key:
                    result = self.build(expanded)
                    self.memo[key] = result
                    return result
        if isinstance(node, Call) and node.fn in _NARY_ARITY:
            arity = _NARY_ARITY[node.fn]
            if node.kwargs or len(node.args) != arity:
                raise FormulaIRCompileError(f"{node.fn} expects exactly {arity} positional arguments")
            children = tuple(self.build(arg) for arg in node.args)
            result = self._append(NaryOp(node.fn, arity), children, _result_type([self.nodes[i] for i in children]))
        elif isinstance(node, Call) and node.fn == "cumsum":
            if node.kwargs or len(node.args) != 1:
                raise FormulaIRCompileError("cumsum expects exactly one positional argument")
            child = self.build(node.args[0])
            result = self._append(CumsumOp(), (child,), self.nodes[child].value_type)
        elif isinstance(node, Call) and node.fn == "ewm":
            x, span, min_periods, ignore_na, adjust = _normalize_ewm(node)
            child = self.build(x)
            result = self._append(EwmOp(span, min_periods, ignore_na, adjust), (child,), self.nodes[child].value_type)
        elif isinstance(node, Call) and node.fn == "xs_rank":
            if node.kwargs or len(node.args) != 1:
                raise FormulaIRCompileError("xs_rank expects exactly one positional argument")
            child = self.build(node.args[0])
            result = self._append(XsRankOp(), (child,), VECTOR)
        elif isinstance(node, Call) and node.fn == "groupby":
            raise FormulaIRCompileError("nested groupby inside groupby RHS is not yet supported by cpp_stream")
        else:
            raise FormulaIRCompileError(f"cpp_stream grouped IR does not yet support {getattr(node, 'fn', type(node).__name__)!r}")
        self.memo[key] = result
        return result


def compile_ir(formula: str | Expr, *, dsl_registry: DSLFunctionRegistry | None = None, column_names: list[str] | tuple[str, ...] | None = None) -> Program:
    """Compile the shared DSL AST into backend-neutral streaming IR."""
    expr = parse_formula(formula) if isinstance(formula, str) else formula
    builder = _Builder(dsl_registry=dsl_registry or DEFAULT_DSL_REGISTRY, column_name_to_index={name: i for i, name in enumerate(column_names or ())})
    root = builder.build(expr)
    return Program(nodes=tuple(builder.nodes), outputs=(root,), input_names=tuple(builder.inputs))
