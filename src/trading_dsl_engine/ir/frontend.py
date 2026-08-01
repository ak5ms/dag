from __future__ import annotations

from dataclasses import dataclass

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
    EwmOp,
    GroupByOp,
    GroupKeySpec,
    InputOp,
    LiteralOp,
    NaryOp,
    RidgeOp,
    RidgeProjectionOp,
    XsRankOp,
)
from trading_dsl_engine.ir.program import Node, Program
from trading_dsl_engine.ir.types import (
    SCALAR,
    VECTOR,
    ValueType,
    fixed,
    matrix,
    object_value,
)


class FormulaIRCompileError(ValueError):
    pass


_NARY_ARITY = {"add": 2, "sub": 2, "mul": 2, "div": 2, "mod": 2, "floor": 1}
_DERIVED_TERMINALS: dict[str, Expr] = {
    # Calendar aliases are derived from the canonical microseconds-since-epoch
    # event timestamp rather than requiring pre-materialized input columns.
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
        return (
            "key",
            _expr_key(node.expr),
            node.num_keys,
            node.offset,
            node.row_scalar,
            node.dtype,
        )
    if isinstance(node, KeyTuple):
        return ("tuple", tuple(_expr_key(item) for item in node.items))
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


def _literal_bool(node: Expr, name: str) -> bool:
    if isinstance(node, Identifier) and node.name in {"True", "False"}:
        return node.name == "True"
    value = _literal_number(node, name)
    if value not in (0.0, 1.0, 2.0, 3.0):
        raise FormulaIRCompileError(f"{name} must be a boolean/numeric literal")
    if value in (2.0, 3.0):
        return bool(int(value) - 2)
    return bool(value)


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
                raise FormulaIRCompileError(
                    f"universe column index {index} appears in more than one group"
                )
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
    raise FormulaIRCompileError(
        f"value kind {value_type.kind!r} cannot be used as a Ridge/cat feature"
    )


def _nary_result_type(children: list[Node]) -> ValueType:
    matrices = [child.value_type for child in children if child.value_type.kind == "matrix"]
    vectors = [child for child in children if child.value_type.kind == "vector"]
    unsupported = [
        child.value_type.kind
        for child in children
        if child.value_type.kind not in {"scalar", "vector", "matrix"}
    ]
    if unsupported:
        raise FormulaIRCompileError(
            f"arithmetic does not support value kinds {sorted(set(unsupported))}"
        )
    if matrices:
        widths = {value.width for value in matrices}
        if len(widths) != 1 or vectors:
            raise FormulaIRCompileError(
                "matrix arithmetic currently requires equal-width matrices and/or scalars"
            )
        return matrix(next(iter(widths)))
    return VECTOR if vectors else SCALAR


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
    """Return features, y, weights, hl, lambda, nonneg, is_stateful."""
    if call.kwargs:
        allowed = {"y", "weights", "hl", "lambda_", "nonneg"}
        values: dict[str, Expr] = {}
        for name, value in call.kwargs:
            if name not in allowed:
                raise FormulaIRCompileError(f"unsupported Ridge keyword {name!r}")
            if name in values:
                raise FormulaIRCompileError(f"Ridge argument {name!r} specified twice")
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
        has_nonneg_sentinel = (
            len(args) >= 5
            and isinstance(args[-1], Number)
            and float(args[-1].value) in (2.0, 3.0)
        )
        nonneg = _literal_bool(args[-1], "Ridge nonneg") if has_nonneg_sentinel else False
        if has_nonneg_sentinel:
            args = args[:-1]
        if len(args) < 4:
            raise FormulaIRCompileError(
                "Ridge expects features..., y, hl, lambda or features..., y, weights, hl, lambda"
            )
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
            raise FormulaIRCompileError("Key(...) may only wrap dynamic group keys, not univ(...)")
        return item.expr, GroupKeySpec(
            num_keys=item.num_keys,
            offset=item.offset,
            row_scalar=item.row_scalar,
            dtype=item.dtype,
        )
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

    def build(self, node: Expr, *, use_cache: bool = True) -> int:
        key = _expr_key(node)
        if use_cache and key in self.memo:
            return self.memo[key]
        if isinstance(node, Key):
            result = self.build(node.expr, use_cache=use_cache)
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
            raise FormulaIRCompileError(
                f"string literal {node.value!r} is not valid in this cpp_stream expression"
            )
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
                raise FormulaIRCompileError(f"{node.fn} expects exactly {arity} positional arguments")
            child_ids = tuple(self.build(arg) for arg in node.args)
            return self._append(
                NaryOp(node.fn, arity),
                child_ids,
                _nary_result_type([self.nodes[i] for i in child_ids]),
            )
        if node.fn == "cat":
            if node.kwargs or not node.args:
                raise FormulaIRCompileError("cat expects at least one positional argument")
            child_ids = tuple(self.build(arg) for arg in node.args)
            widths = tuple(_feature_width(self.nodes[child].value_type) for child in child_ids)
            return self._append(CatOp(widths), child_ids, matrix(sum(widths)))
        if node.fn == "cumsum":
            if node.kwargs or len(node.args) != 1:
                raise FormulaIRCompileError("cumsum expects exactly one positional argument")
            child = self.build(node.args[0])
            if self.nodes[child].value_type.kind != "vector":
                raise FormulaIRCompileError("cumsum currently requires a vector input")
            return self._append(CumsumOp(), (child,), VECTOR)
        if node.fn == "ewm":
            x, span, min_periods, ignore_na, adjust = _normalize_ewm(node)
            child = self.build(x)
            if self.nodes[child].value_type.kind != "vector":
                raise FormulaIRCompileError("ewm currently requires a vector input")
            return self._append(
                EwmOp(span, min_periods, ignore_na, adjust),
                (child,),
                VECTOR,
            )
        if node.fn == "xs_rank":
            if node.kwargs or len(node.args) != 1:
                raise FormulaIRCompileError("xs_rank expects exactly one positional argument")
            child = self.build(node.args[0])
            if self.nodes[child].value_type.kind != "vector":
                raise FormulaIRCompileError("xs_rank requires a vector input")
            return self._append(XsRankOp(), (child,), VECTOR)
        if node.fn == "Ridge":
            features, y, weights, hl, lam, nonneg, is_stateful = _normalize_ridge(node)
            feature_ids = tuple(self.build(expression) for expression in features)
            feature_widths = tuple(
                _feature_width(self.nodes[child].value_type) for child in feature_ids
            )
            y_id = self.build(y)
            if self.nodes[y_id].value_type.kind != "vector":
                raise FormulaIRCompileError("Ridge y must be vector-valued")
            children = list(feature_ids)
            children.append(y_id)
            has_weights = weights is not None
            if weights is not None:
                weight_id = self.build(weights)
                if self.nodes[weight_id].value_type.kind not in {
                    "scalar",
                    "vector",
                    "matrix",
                }:
                    raise FormulaIRCompileError("Ridge weights must be scalar, vector, or matrix")
                children.append(weight_id)
            children.extend((self.build(hl), self.build(lam)))
            op = RidgeOp(
                feature_widths=feature_widths,
                has_weights=has_weights,
                nonneg=nonneg,
                is_stateful=is_stateful,
            )
            return self._append(op, tuple(children), object_value(op.coefficient_width))
        if node.fn in {"get_beta", "get_preds"}:
            if node.kwargs or len(node.args) != 1:
                raise FormulaIRCompileError(f"{node.fn} expects exactly one Ridge value")
            child = self.build(node.args[0])
            child_node = self.nodes[child]
            if not isinstance(child_node.op, RidgeOp):
                raise FormulaIRCompileError(f"{node.fn} currently requires a Ridge value")
            if node.fn == "get_preds":
                value_type = VECTOR
                field = "preds"
            else:
                value_type = matrix(child_node.op.coefficient_width) if grouped else fixed(
                    child_node.op.coefficient_width
                )
                field = "beta"
            return self._append(RidgeProjectionOp(field), (child,), value_type)
        if node.fn == "groupby":
            if grouped:
                raise FormulaIRCompileError("nested groupby inside groupby RHS is not supported")
            return self._build_groupby(node)
        raise FormulaIRCompileError(
            f"cpp_stream neutral IR does not yet support {node.fn!r}"
        )

    def _build_groupby(self, call: Call) -> int:
        if len(call.args) != 3:
            raise FormulaIRCompileError("groupby requires groupby(key_tuple, lhs, op_using_self_)")
        kw = dict(call.kwargs)
        unknown = set(kw) - {"capacity", "hash_capacity"}
        if unknown:
            raise FormulaIRCompileError(f"unsupported groupby keyword argument(s): {sorted(unknown)}")
        capacity = None if "capacity" not in kw else int(
            round(_literal_number(kw["capacity"], "groupby capacity"))
        )
        hash_capacity = None if "hash_capacity" not in kw else int(
            round(_literal_number(kw["hash_capacity"], "groupby hash_capacity"))
        )
        if capacity is not None and capacity <= 0:
            raise FormulaIRCompileError("groupby capacity must be > 0")
        if hash_capacity is not None and hash_capacity <= 0:
            raise FormulaIRCompileError("groupby hash_capacity must be > 0")

        key = call.args[0]
        key_items = key.items if isinstance(key, KeyTuple) else (key,)
        universes = [item for item in key_items if isinstance(item, Universe)]
        if len(universes) > 1:
            raise FormulaIRCompileError("groupby key tuple may contain at most one univ(...) element")
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
            raise FormulaIRCompileError("cpp_stream groupby lhs must be vector-valued")
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
            raise FormulaIRCompileError(
                "cpp_stream groupby RHS must emit a per-instrument vector or matrix"
            )
        op = GroupByOp(
            key_specs=tuple(key_specs),
            static_groups=static_groups,
            inner_program=inner_program,
            capacity=capacity,
            hash_capacity=hash_capacity,
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
            raise FormulaIRCompileError(
                f"failed expanding grouped DSL function {node.fn!r}: {exc}"
            ) from exc
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
        elif isinstance(node, Call) and (expanded := self._expand_macro(node)) is not None:
            result = self.build(expanded)
        elif isinstance(node, Call):
            # Reuse the outer builder's call semantics with this builder's node
            # storage and grouped projection shape.
            result = self._build_call(node)
        else:
            raise FormulaIRCompileError(f"unsupported grouped expression {node!r}")
        self.memo[key] = result
        return result

    def _build_call(self, node: Call) -> int:
        if node.fn in _NARY_ARITY:
            arity = _NARY_ARITY[node.fn]
            if node.kwargs or len(node.args) != arity:
                raise FormulaIRCompileError(f"{node.fn} expects exactly {arity} positional arguments")
            children = tuple(self.build(arg) for arg in node.args)
            return self._append(
                NaryOp(node.fn, arity),
                children,
                _nary_result_type([self.nodes[i] for i in children]),
            )
        if node.fn == "cat":
            if node.kwargs or not node.args:
                raise FormulaIRCompileError("cat expects at least one positional argument")
            children = tuple(self.build(arg) for arg in node.args)
            widths = tuple(_feature_width(self.nodes[child].value_type) for child in children)
            return self._append(CatOp(widths), children, matrix(sum(widths)))
        if node.fn == "cumsum":
            if node.kwargs or len(node.args) != 1:
                raise FormulaIRCompileError("cumsum expects one argument")
            child = self.build(node.args[0])
            return self._append(CumsumOp(), (child,), VECTOR)
        if node.fn == "ewm":
            x, span, min_periods, ignore_na, adjust = _normalize_ewm(node)
            child = self.build(x)
            return self._append(
                EwmOp(span, min_periods, ignore_na, adjust),
                (child,),
                VECTOR,
            )
        if node.fn == "xs_rank":
            if node.kwargs or len(node.args) != 1:
                raise FormulaIRCompileError("xs_rank expects one argument")
            child = self.build(node.args[0])
            return self._append(XsRankOp(), (child,), VECTOR)
        if node.fn == "Ridge":
            features, y, weights, hl, lam, nonneg, is_stateful = _normalize_ridge(node)
            feature_ids = tuple(self.build(expression) for expression in features)
            feature_widths = tuple(
                _feature_width(self.nodes[child].value_type) for child in feature_ids
            )
            children = list(feature_ids)
            y_id = self.build(y)
            children.append(y_id)
            if weights is not None:
                children.append(self.build(weights))
            children.extend((self.build(hl), self.build(lam)))
            op = RidgeOp(feature_widths, weights is not None, nonneg, is_stateful)
            return self._append(op, tuple(children), object_value(op.coefficient_width))
        if node.fn in {"get_beta", "get_preds"}:
            if node.kwargs or len(node.args) != 1:
                raise FormulaIRCompileError(f"{node.fn} expects one Ridge value")
            child = self.build(node.args[0])
            child_node = self.nodes[child]
            if not isinstance(child_node.op, RidgeOp):
                raise FormulaIRCompileError(f"{node.fn} requires a Ridge value")
            field = "beta" if node.fn == "get_beta" else "preds"
            value_type = matrix(child_node.op.coefficient_width) if field == "beta" else VECTOR
            return self._append(RidgeProjectionOp(field), (child,), value_type)
        if node.fn == "groupby":
            raise FormulaIRCompileError("nested groupby inside groupby RHS is not supported")
        raise FormulaIRCompileError(
            f"cpp_stream grouped IR does not yet support {node.fn!r}"
        )


def compile_ir(
    formula: str | Expr,
    *,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
) -> Program:
    """Compile the shared DSL AST into backend-neutral streaming IR."""
    expr = parse_formula(formula) if isinstance(formula, str) else formula
    builder = _Builder(
        dsl_registry=dsl_registry or DEFAULT_DSL_REGISTRY,
        column_name_to_index={name: i for i, name in enumerate(column_names or ())},
    )
    root = builder.build(expr)
    return Program(nodes=tuple(builder.nodes), outputs=(root,), input_names=tuple(builder.inputs))
