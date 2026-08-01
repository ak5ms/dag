from __future__ import annotations

from dataclasses import dataclass

from trading_dsl_engine.base.custom import StatelessCall
from trading_dsl_engine.base.dsl import DEFAULT_DSL_REGISTRY, DSLFunctionRegistry
from trading_dsl_engine.base.keys import Key
from trading_dsl_engine.base.parser import (
    Call,
    Expr,
    Identifier,
    KeyTuple,
    Number,
    String,
    Universe,
    parse_formula,
)
from trading_dsl_engine.ir.ops import (
    CatOp,
    CumsumOp,
    CustomCallOp,
    EinsumOp,
    EwmOp,
    FFillOp,
    FutureRbfBasisSumOp,
    GroupByOp,
    GroupKeySpec,
    InputOp,
    InstrumentBasisMeanOp,
    InstrumentBasisProjectionOp,
    LiteralOp,
    NaryOp,
    RbfBasisOp,
    RidgeOp,
    RidgeProjectionOp,
    ShiftOp,
    XsRankOp,
)
from trading_dsl_engine.ir.program import Node, Program
from trading_dsl_engine.ir.types import SCALAR, VECTOR, ValueType, fixed, matrix, object_value


class FormulaIRCompileError(ValueError):
    pass


_NARY_ARITY = {
    "floor": 1,
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "mod": 2,
    "pow": 2,
    "eq": 2,
    "ne": 2,
    "lt": 2,
    "gt": 2,
    "le": 2,
    "ge": 2,
    "and_": 2,
    "or_": 2,
    "xor": 2,
    "fillna": 2,
    "where": 3,
}
_LOGICAL_OPS = {"eq", "ne", "lt", "gt", "le", "ge", "and_", "or_", "xor"}
_DERIVED_TERMINALS: dict[str, Expr] = {
    "minute": Call("minute", (Identifier("_ev_ts"),), ()),
}


def _expr_key(node: Expr) -> tuple:
    if isinstance(node, Identifier):
        return ("id", node.name)
    if isinstance(node, Number):
        return ("num", node.value)
    if isinstance(node, String):
        return ("str", node.value)
    if isinstance(node, Universe):
        return ("univ", node.groups)
    if isinstance(node, Key):
        return ("key", _expr_key(node.expr), node.num_keys, node.offset, node.row_scalar, node.dtype)
    if isinstance(node, KeyTuple):
        return ("tuple", tuple(_expr_key(item) for item in node.items))
    if isinstance(node, StatelessCall):
        return (
            "stateless",
            node.cpp_name or node.name,
            node.output_kind,
            node.output_width,
            tuple(_expr_key(arg) for arg in node.args),
        )
    if isinstance(node, Call):
        return (
            "call",
            node.fn,
            tuple(_expr_key(arg) for arg in node.args),
            tuple((name, _expr_key(value)) for name, value in node.kwargs),
        )
    raise FormulaIRCompileError(f"unhandled expression for key: {node!r}")


def _contains_self(node: Expr) -> bool:
    if isinstance(node, Identifier):
        return node.name == "self_"
    if isinstance(node, Key):
        return _contains_self(node.expr)
    if isinstance(node, StatelessCall):
        return any(_contains_self(arg) for arg in node.args)
    if isinstance(node, Call):
        return any(_contains_self(arg) for arg in node.args) or any(
            _contains_self(value) for _, value in node.kwargs
        )
    if isinstance(node, KeyTuple):
        return any(_contains_self(item) for item in node.items)
    return False


def _literal_number(node: Expr, name: str) -> float:
    if not isinstance(node, Number):
        raise FormulaIRCompileError(f"{name} must be a numeric literal")
    return float(node.value)


def _literal_int(node: Expr, name: str, *, minimum: int | None = None) -> int:
    value = _literal_number(node, name)
    rounded = int(round(value))
    if value != rounded:
        raise FormulaIRCompileError(f"{name} must be an integer literal")
    if minimum is not None and rounded < minimum:
        raise FormulaIRCompileError(f"{name} must be >= {minimum}")
    return rounded


def _literal_bool(node: Expr, name: str) -> bool:
    if isinstance(node, Identifier) and node.name in {"True", "False"}:
        return node.name == "True"
    value = _literal_number(node, name)
    if value not in (0.0, 1.0, 2.0, 3.0):
        raise FormulaIRCompileError(f"{name} must be a boolean/numeric literal")
    if value in (2.0, 3.0):
        return bool(int(value) - 2)
    return bool(value)


def _literal_string(node: Expr, name: str) -> str:
    if not isinstance(node, String):
        raise FormulaIRCompileError(f"{name} must be a string literal")
    return node.value


def _resolve_universe_groups(
    universe: Universe,
    column_name_to_index: dict[str, int],
) -> tuple[tuple[int, ...], ...]:
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
                    raise FormulaIRCompileError(
                        f"unknown universe column {member!r}; pass column_names when using ticker names"
                    ) from exc
            if index < 0:
                raise FormulaIRCompileError("universe column indexes must be >= 0")
            if index in seen:
                raise FormulaIRCompileError(f"universe column index {index} appears in multiple groups")
            seen.add(index)
            resolved.append(index)
        if not resolved:
            raise FormulaIRCompileError("universe groups cannot be empty")
        groups.append(tuple(resolved))
    return tuple(groups)


def _feature_width(value_type: ValueType) -> int:
    if value_type.kind in {"scalar", "vector"}:
        return 1
    if value_type.kind == "matrix":
        return int(value_type.width)
    raise FormulaIRCompileError(f"value kind {value_type.kind!r} cannot be used as a feature")


def _nary_result_type(name: str, children: list[Node]) -> ValueType:
    if name == "where":
        children = children[1:]
    if name in _LOGICAL_OPS:
        if any(child.value_type.kind == "matrix" for child in children):
            raise FormulaIRCompileError(f"{name} matrix values are not supported")
        return VECTOR if any(child.value_type.kind == "vector" for child in children) else SCALAR
    matrices = [child.value_type for child in children if child.value_type.kind == "matrix"]
    vectors = [child for child in children if child.value_type.kind == "vector"]
    unsupported = [
        child.value_type.kind
        for child in children
        if child.value_type.kind not in {"scalar", "vector", "matrix"}
    ]
    if unsupported:
        raise FormulaIRCompileError(f"arithmetic does not support {sorted(set(unsupported))}")
    if matrices:
        widths = {value.width for value in matrices}
        if len(widths) != 1 or vectors:
            raise FormulaIRCompileError(
                "matrix arithmetic requires equal-width matrices and/or scalars"
            )
        return matrix(next(iter(widths)))
    return VECTOR if vectors else SCALAR


def _custom_value_type(node: StatelessCall, children: list[Node]) -> ValueType:
    kind = node.output_kind
    if kind is None:
        if not children:
            raise FormulaIRCompileError("stateless call has no children for output inference")
        return children[0].value_type
    if kind == "scalar":
        return SCALAR
    if kind == "vector":
        return VECTOR
    if kind == "matrix":
        if node.output_width is None or int(node.output_width) <= 0:
            raise FormulaIRCompileError("matrix stateless calls require output_width > 0")
        return matrix(int(node.output_width))
    if kind == "object":
        return object_value(int(node.output_width or 1))
    raise FormulaIRCompileError(f"unsupported stateless output kind {kind!r}")


def _normalize_ewm(call: Call) -> tuple[Expr, float, int, bool, bool]:
    names = ("x", "span", "min_periods", "ignore_na", "adjust")
    if len(call.args) > len(names):
        raise FormulaIRCompileError("ewm accepts at most five positional arguments")
    values: dict[str, Expr] = {
        "min_periods": Number(0.0),
        "ignore_na": Number(1.0),
        "adjust": Number(0.0),
    }
    explicitly_set: set[str] = set()
    for name, value in zip(names, call.args):
        values[name] = value
        explicitly_set.add(name)
    for name, value in call.kwargs:
        if name not in names:
            raise FormulaIRCompileError(f"unsupported ewm keyword {name!r}")
        if name in explicitly_set:
            raise FormulaIRCompileError(f"ewm argument {name!r} specified twice")
        values[name] = value
        explicitly_set.add(name)
    if "x" not in values or "span" not in values:
        raise FormulaIRCompileError("ewm requires x and span")
    span = _literal_number(values["span"], "ewm span")
    if not span > 0.0:
        raise FormulaIRCompileError("ewm span must be > 0")
    min_periods = _literal_int(values["min_periods"], "ewm min_periods", minimum=0)
    return (
        values["x"],
        span,
        min_periods,
        _literal_bool(values["ignore_na"], "ewm ignore_na"),
        _literal_bool(values["adjust"], "ewm adjust"),
    )


def _normalize_shift(call: Call) -> tuple[Expr, int, int]:
    if call.kwargs or not 1 <= len(call.args) <= 3:
        raise FormulaIRCompileError("shift expects x[, lag[, max_lag]]")
    lag = 1 if len(call.args) < 2 else _literal_int(call.args[1], "shift lag", minimum=0)
    max_lag = lag if len(call.args) < 3 else _literal_int(call.args[2], "shift max_lag", minimum=0)
    if lag > max_lag:
        raise FormulaIRCompileError("shift lag cannot exceed max_lag")
    return call.args[0], lag, max_lag


def _normalize_ffill(call: Call) -> tuple[Expr, int | None]:
    if call.kwargs or not 1 <= len(call.args) <= 2:
        raise FormulaIRCompileError("ffill expects x[, limit]")
    limit = None if len(call.args) == 1 else _literal_int(call.args[1], "ffill limit", minimum=0)
    return call.args[0], limit


def _flatten_cat_features(expressions: tuple[Expr, ...]) -> tuple[Expr, ...]:
    flattened: list[Expr] = []
    for expression in expressions:
        if isinstance(expression, Call) and expression.fn == "cat" and not expression.kwargs:
            flattened.extend(_flatten_cat_features(expression.args))
        else:
            flattened.append(expression)
    return tuple(flattened)


def _normalize_ridge(
    call: Call,
) -> tuple[tuple[Expr, ...], Expr, Expr | None, Expr, Expr, bool, bool]:
    if call.kwargs:
        allowed = {"y", "weights", "hl", "lambda_", "nonneg"}
        values: dict[str, Expr] = {}
        for name, value in call.kwargs:
            if name not in allowed or name in values:
                raise FormulaIRCompileError(f"invalid Ridge keyword {name!r}")
            values[name] = value
        missing = [name for name in ("y", "hl", "lambda_") if name not in values]
        if missing:
            raise FormulaIRCompileError(f"Ridge keyword form is missing {missing}")
        features = call.args
        y = values["y"]
        weights = values.get("weights")
        hl = values["hl"]
        lam = values["lambda_"]
        nonneg = _literal_bool(values.get("nonneg", Number(0.0)), "Ridge nonneg")
    else:
        args = call.args
        sentinel = (
            len(args) >= 5 and isinstance(args[-1], Number) and float(args[-1].value) in (2.0, 3.0)
        )
        nonneg = _literal_bool(args[-1], "Ridge nonneg") if sentinel else False
        if sentinel:
            args = args[:-1]
        if len(args) < 4:
            raise FormulaIRCompileError("Ridge expects features..., y, [weights,] hl, lambda")
        has_weights = len(args) >= 5
        if has_weights:
            features = args[:-4]
            y, weights, hl, lam = args[-4:]
        else:
            features = args[:-3]
            y, hl, lam = args[-3:]
            weights = None
    features = _flatten_cat_features(tuple(features))
    if not features:
        raise FormulaIRCompileError("Ridge expects at least one feature")
    is_stateful = not (isinstance(hl, Number) and float(hl.value) == 0.0)
    return features, y, weights, hl, lam, nonneg, is_stateful


def _group_key(item: Expr) -> tuple[Expr, GroupKeySpec]:
    if isinstance(item, Key):
        if isinstance(item.expr, Universe):
            raise FormulaIRCompileError("Key may not wrap univ")
        return item.expr, GroupKeySpec(item.num_keys, item.offset, item.row_scalar, item.dtype)
    return item, GroupKeySpec()


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

    def _expand_macro(self, node: Call) -> Expr | None:
        macro = self.dsl_registry.get(node.fn)
        if macro is None:
            return None
        try:
            expanded = macro(*node.args, **dict(node.kwargs))
        except Exception as exc:
            raise FormulaIRCompileError(f"failed expanding DSL function {node.fn!r}: {exc}") from exc
        return expanded if _expr_key(expanded) != _expr_key(node) else None

    def _build_custom(self, node: StatelessCall) -> int:
        name = node.cpp_name or node.name
        if not name:
            raise FormulaIRCompileError("cpp_stream stateless calls require cpp_name or name")
        children = tuple(self.build(arg) for arg in node.args)
        child_nodes = [self.nodes[index] for index in children]
        return self._append(CustomCallOp(name, len(children)), children, _custom_value_type(node, child_nodes))

    def build(self, node: Expr, *, use_cache: bool = True) -> int:
        key = _expr_key(node)
        if use_cache and key in self.memo:
            return self.memo[key]
        if isinstance(node, Key):
            result = self.build(node.expr, use_cache=use_cache)
        elif isinstance(node, StatelessCall):
            result = self._build_custom(node)
        elif isinstance(node, Call) and (expanded := self._expand_macro(node)) is not None:
            result = self.build(expanded, use_cache=use_cache)
        elif isinstance(node, Identifier):
            derived = _DERIVED_TERMINALS.get(node.name)
            if derived is not None:
                result = self.build(derived, use_cache=use_cache)
            else:
                if node.name == "self_":
                    raise FormulaIRCompileError("self_ is only valid inside groupby RHS")
                input_index = self.inputs.setdefault(node.name, len(self.inputs))
                result = self._append(InputOp(input_index, node.name), (), VECTOR)
        elif isinstance(node, Number):
            result = self._append(LiteralOp(node.value), (), SCALAR)
        elif isinstance(node, String):
            raise FormulaIRCompileError(f"string literal {node.value!r} is invalid here")
        elif isinstance(node, Call):
            result = self._build_call(node, grouped=False)
        else:
            raise FormulaIRCompileError(f"unsupported expression {node!r}")
        if use_cache:
            self.memo[key] = result
        return result

    def _build_call(self, node: Call, *, grouped: bool) -> int:
        if node.fn in _NARY_ARITY:
            arity = _NARY_ARITY[node.fn]
            if node.kwargs or len(node.args) != arity:
                raise FormulaIRCompileError(f"{node.fn} expects {arity} positional arguments")
            children = tuple(self.build(arg) for arg in node.args)
            return self._append(
                NaryOp(node.fn, arity),
                children,
                _nary_result_type(node.fn, [self.nodes[index] for index in children]),
            )
        if node.fn == "cat":
            if node.kwargs or not node.args:
                raise FormulaIRCompileError("cat expects at least one argument")
            children = tuple(self.build(arg) for arg in node.args)
            widths = tuple(_feature_width(self.nodes[child].value_type) for child in children)
            return self._append(CatOp(widths), children, matrix(sum(widths)))
        if node.fn == "cumsum":
            if node.kwargs or len(node.args) != 1:
                raise FormulaIRCompileError("cumsum expects one argument")
            child = self.build(node.args[0])
            if self.nodes[child].value_type.kind != "vector":
                raise FormulaIRCompileError("cumsum requires vector input")
            return self._append(CumsumOp(), (child,), VECTOR)
        if node.fn == "ffill":
            x, limit = _normalize_ffill(node)
            child = self.build(x)
            if self.nodes[child].value_type.kind != "vector":
                raise FormulaIRCompileError("ffill requires vector input")
            return self._append(FFillOp(limit), (child,), VECTOR)
        if node.fn == "shift":
            x, lag, max_lag = _normalize_shift(node)
            child = self.build(x)
            if self.nodes[child].value_type.kind != "vector":
                raise FormulaIRCompileError("shift requires vector input")
            return self._append(ShiftOp(lag, max_lag), (child,), VECTOR)
        if node.fn == "ewm":
            x, span, min_periods, ignore_na, adjust = _normalize_ewm(node)
            child = self.build(x)
            if self.nodes[child].value_type.kind != "vector":
                raise FormulaIRCompileError("ewm requires vector input")
            return self._append(EwmOp(span, min_periods, ignore_na, adjust), (child,), VECTOR)
        if node.fn == "xs_rank":
            if node.kwargs or len(node.args) != 1:
                raise FormulaIRCompileError("xs_rank expects one argument")
            child = self.build(node.args[0])
            return self._append(XsRankOp(), (child,), VECTOR)
        if node.fn == "rbf_basis":
            if node.kwargs or len(node.args) != 4:
                raise FormulaIRCompileError("rbf_basis expects ts, start, end, n_basis")
            n_basis = _literal_int(node.args[3], "rbf_basis n_basis", minimum=1)
            children = tuple(self.build(arg) for arg in node.args[:3])
            return self._append(RbfBasisOp(n_basis), children, matrix(n_basis))
        if node.fn == "future_rbf_basis_sum":
            if node.kwargs or len(node.args) != 5:
                raise FormulaIRCompileError(
                    "future_rbf_basis_sum expects ts, start, end, n_basis, n_steps"
                )
            n_basis = _literal_int(node.args[3], "future_rbf_basis_sum n_basis", minimum=1)
            n_steps = _literal_int(node.args[4], "future_rbf_basis_sum n_steps", minimum=1)
            children = tuple(self.build(arg) for arg in node.args[:3])
            return self._append(FutureRbfBasisSumOp(n_basis, n_steps), children, matrix(n_basis))
        if node.fn == "einsum":
            if node.kwargs or len(node.args) < 3:
                raise FormulaIRCompileError("einsum expects operands and a subscript string")
            subscripts = _literal_string(node.args[-1], "einsum subscripts")
            children = tuple(self.build(arg) for arg in node.args[:-1])
            if subscripts != "nf,nf->n" or len(children) != 2:
                raise FormulaIRCompileError(
                    "cpp_stream currently supports einsum('nf,nf->n')"
                )
            left, right = (self.nodes[index].value_type for index in children)
            if left.kind != "matrix" or right.kind != "matrix" or left.width != right.width:
                raise FormulaIRCompileError("einsum nf,nf->n requires equal-width matrices")
            return self._append(EinsumOp(subscripts), children, VECTOR)
        if node.fn == "InstrumentBasisMean":
            if node.kwargs or len(node.args) not in {3, 4}:
                raise FormulaIRCompileError(
                    "InstrumentBasisMean expects features, y, [weights,] hl"
                )
            feature_id = self.build(node.args[0])
            feature_width = _feature_width(self.nodes[feature_id].value_type)
            y_id = self.build(node.args[1])
            if self.nodes[y_id].value_type.kind != "vector":
                raise FormulaIRCompileError("InstrumentBasisMean y must be vector")
            children = [feature_id, y_id]
            has_weights = len(node.args) == 4
            if has_weights:
                children.append(self.build(node.args[2]))
                hl = node.args[3]
            else:
                hl = node.args[2]
            children.append(self.build(hl))
            op = InstrumentBasisMeanOp(feature_width, has_weights)
            return self._append(op, tuple(children), object_value(feature_width))
        if node.fn == "Ridge":
            features, y, weights, hl, lam, nonneg, is_stateful = _normalize_ridge(node)
            feature_ids = tuple(self.build(expression) for expression in features)
            feature_widths = tuple(_feature_width(self.nodes[child].value_type) for child in feature_ids)
            y_id = self.build(y)
            if self.nodes[y_id].value_type.kind != "vector":
                raise FormulaIRCompileError("Ridge y must be vector")
            children = list(feature_ids) + [y_id]
            if weights is not None:
                children.append(self.build(weights))
            children.extend((self.build(hl), self.build(lam)))
            op = RidgeOp(feature_widths, weights is not None, nonneg, is_stateful)
            return self._append(op, tuple(children), object_value(op.coefficient_width))
        if node.fn in {"get_beta", "get_preds"}:
            if node.kwargs or len(node.args) != 1:
                raise FormulaIRCompileError(f"{node.fn} expects one object value")
            child = self.build(node.args[0])
            child_node = self.nodes[child]
            field = "beta" if node.fn == "get_beta" else "preds"
            if isinstance(child_node.op, RidgeOp):
                value_type = (
                    matrix(child_node.op.coefficient_width)
                    if field == "beta" and grouped
                    else fixed(child_node.op.coefficient_width)
                    if field == "beta"
                    else VECTOR
                )
                return self._append(RidgeProjectionOp(field), (child,), value_type)
            if isinstance(child_node.op, InstrumentBasisMeanOp):
                value_type = matrix(child_node.op.feature_width) if field == "beta" else VECTOR
                return self._append(InstrumentBasisProjectionOp(field), (child,), value_type)
            raise FormulaIRCompileError(f"{node.fn} requires Ridge or InstrumentBasisMean")
        if node.fn == "groupby":
            if grouped:
                raise FormulaIRCompileError("nested groupby is not supported")
            return self._build_groupby(node)
        raise FormulaIRCompileError(f"cpp_stream neutral IR does not yet support {node.fn!r}")

    def _build_groupby(self, call: Call) -> int:
        if len(call.args) != 3:
            raise FormulaIRCompileError("groupby requires key_tuple, lhs, rhs")
        kw = dict(call.kwargs)
        unknown = set(kw) - {"capacity", "hash_capacity"}
        if unknown:
            raise FormulaIRCompileError(f"unsupported groupby keywords {sorted(unknown)}")
        capacity = None if "capacity" not in kw else _literal_int(
            kw["capacity"], "groupby capacity", minimum=1
        )
        hash_capacity = None if "hash_capacity" not in kw else _literal_int(
            kw["hash_capacity"], "groupby hash_capacity", minimum=1
        )
        key = call.args[0]
        key_items = key.items if isinstance(key, KeyTuple) else (key,)
        universes = [item for item in key_items if isinstance(item, Universe)]
        if len(universes) > 1:
            raise FormulaIRCompileError("groupby may contain at most one univ")
        static_groups = None if not universes else _resolve_universe_groups(
            universes[0], self.column_name_to_index
        )
        dynamic_items: list[Expr] = []
        key_specs: list[GroupKeySpec] = []
        for item in key_items:
            if isinstance(item, Universe):
                continue
            expression, spec = _group_key(item)
            dynamic_items.append(expression)
            key_specs.append(spec)
        key_ids = tuple(self.build(item) for item in dynamic_items)
        lhs_id = self.build(call.args[1])
        if self.nodes[lhs_id].value_type.kind != "vector":
            raise FormulaIRCompileError("groupby lhs must be vector")
        inner_builder = _InnerBuilder(self)
        inner_root = inner_builder.build(call.args[2])
        inner_program = Program(
            nodes=tuple(inner_builder.nodes),
            outputs=(inner_root,),
            input_names=("__self__",)
            + tuple(f"__capture_{i}__" for i in range(len(inner_builder.capture_ids))),
        )
        inner_type = inner_program.nodes[inner_root].value_type
        if inner_type.kind not in {"vector", "matrix"}:
            raise FormulaIRCompileError("groupby RHS must emit vector or matrix")
        op = GroupByOp(
            tuple(key_specs),
            static_groups,
            inner_program,
            capacity,
            hash_capacity,
        )
        children = key_ids + (lhs_id,) + tuple(inner_builder.capture_ids)
        return self._append(op, children, inner_type)


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

    def _input(self, input_index: int, name: str, value_type: ValueType) -> int:
        return self._append(InputOp(input_index, name), (), value_type)

    def _expand_macro(self, node: Call) -> Expr | None:
        macro = self.outer.dsl_registry.get(node.fn)
        if macro is None:
            return None
        try:
            expanded = macro(*node.args, **dict(node.kwargs))
        except Exception as exc:
            raise FormulaIRCompileError(f"failed expanding grouped function {node.fn!r}: {exc}") from exc
        return expanded if _expr_key(expanded) != _expr_key(node) else None

    def build(self, node: Expr) -> int:
        key = _expr_key(node)
        if key in self.memo:
            return self.memo[key]
        if isinstance(node, Key):
            result = self.build(node.expr)
        elif isinstance(node, Identifier) and node.name == "self_":
            if self._self_input is None:
                self._self_input = self._input(0, "__self__", VECTOR)
            result = self._self_input
        elif isinstance(node, Number):
            result = self._append(LiteralOp(node.value), (), SCALAR)
        elif not _contains_self(node):
            capture_pos = self.capture_map.get(key)
            if capture_pos is None:
                outer_id = self.outer.build(node)
                capture_pos = len(self.capture_ids)
                self.capture_map[key] = capture_pos
                self.capture_ids.append(outer_id)
            outer_id = self.capture_ids[capture_pos]
            result = self._input(
                capture_pos + 1,
                f"__capture_{capture_pos}__",
                self.outer.nodes[outer_id].value_type,
            )
        elif isinstance(node, StatelessCall):
            children = tuple(self.build(arg) for arg in node.args)
            name = node.cpp_name or node.name
            if not name:
                raise FormulaIRCompileError("grouped stateless call requires a native name")
            result = self._append(
                CustomCallOp(name, len(children)),
                children,
                _custom_value_type(node, [self.nodes[index] for index in children]),
            )
        elif isinstance(node, Call) and (expanded := self._expand_macro(node)) is not None:
            result = self.build(expanded)
        elif isinstance(node, Call):
            result = self._build_call(node)
        else:
            raise FormulaIRCompileError(f"unsupported grouped expression {node!r}")
        self.memo[key] = result
        return result

    def _build_call(self, node: Call) -> int:
        if node.fn in _NARY_ARITY:
            arity = _NARY_ARITY[node.fn]
            if node.kwargs or len(node.args) != arity:
                raise FormulaIRCompileError(f"{node.fn} expects {arity} arguments")
            children = tuple(self.build(arg) for arg in node.args)
            return self._append(
                NaryOp(node.fn, arity),
                children,
                _nary_result_type(node.fn, [self.nodes[index] for index in children]),
            )
        if node.fn == "cat":
            children = tuple(self.build(arg) for arg in node.args)
            widths = tuple(_feature_width(self.nodes[child].value_type) for child in children)
            return self._append(CatOp(widths), children, matrix(sum(widths)))
        if node.fn == "cumsum":
            child = self.build(node.args[0])
            return self._append(CumsumOp(), (child,), VECTOR)
        if node.fn == "ffill":
            x, limit = _normalize_ffill(node)
            child = self.build(x)
            return self._append(FFillOp(limit), (child,), VECTOR)
        if node.fn == "shift":
            x, lag, max_lag = _normalize_shift(node)
            child = self.build(x)
            return self._append(ShiftOp(lag, max_lag), (child,), VECTOR)
        if node.fn == "ewm":
            x, span, min_periods, ignore_na, adjust = _normalize_ewm(node)
            child = self.build(x)
            return self._append(EwmOp(span, min_periods, ignore_na, adjust), (child,), VECTOR)
        if node.fn == "xs_rank":
            child = self.build(node.args[0])
            return self._append(XsRankOp(), (child,), VECTOR)
        if node.fn == "groupby":
            raise FormulaIRCompileError("nested groupby is not supported")
        raise FormulaIRCompileError(f"cpp_stream grouped IR does not support {node.fn!r}")


def compile_ir(
    formula: str | Expr,
    *,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
) -> Program:
    expr = parse_formula(formula) if isinstance(formula, str) else formula
    builder = _Builder(
        dsl_registry=dsl_registry or DEFAULT_DSL_REGISTRY,
        column_name_to_index={name: index for index, name in enumerate(column_names or ())},
    )
    root = builder.build(expr)
    return Program(nodes=tuple(builder.nodes), outputs=(root,), input_names=tuple(builder.inputs))


__all__ = ["FormulaIRCompileError", "compile_ir"]
