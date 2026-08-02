from __future__ import annotations

from collections.abc import Mapping
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
from trading_dsl_engine.ir.einsum import EinsumParseError, parse_einsum
from trading_dsl_engine.ir.ops import (
    CatOp,
    CumsumOp,
    CustomCallOp,
    EmitOp,
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
    ReductionOp,
    ShiftOp,
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
    tensor,
)


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
    "minute": Call("minute", (Identifier("_ev_ts"),), ())
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
    raise FormulaIRCompileError(f"unhandled expression {node!r}")


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


def _literal_int(node: Expr, name: str, minimum: int | None = None) -> int:
    value = _literal_number(node, name)
    result = int(round(value))
    if value != result or (minimum is not None and result < minimum):
        raise FormulaIRCompileError(f"{name} must be an integer >= {minimum}")
    return result


def _literal_bool(node: Expr, name: str) -> bool:
    if isinstance(node, Identifier) and node.name in {"True", "False"}:
        return node.name == "True"
    value = _literal_number(node, name)
    if value in (2.0, 3.0):
        return bool(int(value) - 2)
    if value in (0.0, 1.0):
        return bool(value)
    raise FormulaIRCompileError(f"{name} must be a boolean literal")


def _literal_string(node: Expr, name: str) -> str:
    if not isinstance(node, String):
        raise FormulaIRCompileError(f"{name} must be a string literal")
    return node.value


def _literal_optimize(node: Expr) -> object:
    if isinstance(node, String):
        return node.value
    if isinstance(node, Identifier) and node.name in {"True", "False"}:
        return node.name == "True"
    if isinstance(node, Number):
        value = float(node.value)
        if value in (0.0, 1.0):
            return bool(value)
        if value in (2.0, 3.0):
            return bool(int(value) - 2)
    raise FormulaIRCompileError(
        "einsum optimize must be True, False, 'greedy', 'optimal', or 'none'"
    )


def _feature_width(value_type: ValueType) -> int:
    if value_type.kind in {"scalar", "vector"}:
        return 1
    if value_type.kind == "matrix":
        return int(value_type.width)
    raise FormulaIRCompileError(f"{value_type.kind!r} cannot be used as a feature")


def _lane_state_result_type(name: str, child: Node) -> ValueType:
    """Return the shape-preserving type of a per-lane temporal operator.

    Cumulative/history/EWM operators apply independently to each logical lane. A
    row-scalar input therefore remains a row scalar; a vector remains a vector.
    Matrices, tensors, and object values require a separate explicit operator.
    """

    if child.value_type.kind not in {"scalar", "vector"}:
        raise FormulaIRCompileError(
            f"{name} requires a scalar or vector input, got {child.value_type.kind!r}"
        )
    return child.value_type


def _nary_result_type(name: str, children: list[Node]) -> ValueType:
    # Every where operand participates in broadcast shape. A vector condition
    # with scalar branches therefore produces a vector, matching NumPy/JAX.
    values = children
    if name in _LOGICAL_OPS:
        if any(child.value_type.kind == "matrix" for child in values):
            raise FormulaIRCompileError(f"{name} matrix values are unsupported")
        return VECTOR if any(child.value_type.kind == "vector" for child in values) else SCALAR
    matrices = [child.value_type for child in values if child.value_type.kind == "matrix"]
    vectors = [child for child in values if child.value_type.kind == "vector"]
    if any(
        child.value_type.kind not in {"scalar", "vector", "matrix"}
        for child in values
    ):
        raise FormulaIRCompileError(f"{name} cannot consume object/fixed/tensor values")
    if matrices:
        widths = {value.width for value in matrices}
        if len(widths) != 1 or vectors:
            raise FormulaIRCompileError(
                "matrix arithmetic requires equal-width matrices and scalars"
            )
        return matrix(next(iter(widths)))
    return VECTOR if vectors else SCALAR


def _custom_value_type(node: StatelessCall, children: list[Node]) -> ValueType:
    if node.output_kind is None:
        return children[0].value_type
    if node.output_kind == "scalar":
        return SCALAR
    if node.output_kind == "vector":
        return VECTOR
    if node.output_kind == "matrix":
        return matrix(int(node.output_width or 0))
    if node.output_kind == "object":
        return object_value(int(node.output_width or 1))
    raise FormulaIRCompileError(f"invalid stateless output kind {node.output_kind!r}")


def _reduction_arguments(call: Call) -> tuple[Expr, Expr | None, int]:
    if call.fn not in {"sum", "mean", "std"}:
        raise FormulaIRCompileError(f"invalid reduction {call.fn!r}")
    names = ("x", "axis", "ddof") if call.fn == "std" else ("x", "axis")
    values: dict[str, Expr] = {}
    explicit: set[str] = set()
    for name, value in zip(names, call.args):
        values[name] = value
        explicit.add(name)
    if len(call.args) > len(names):
        raise FormulaIRCompileError(f"{call.fn} received too many arguments")
    for name, value in call.kwargs:
        if name not in names or name in explicit:
            raise FormulaIRCompileError(f"invalid {call.fn} argument {name!r}")
        values[name] = value
        explicit.add(name)
    if "x" not in values:
        raise FormulaIRCompileError(f"{call.fn} requires x")
    ddof = _literal_int(values.get("ddof", Number(0.0)), "std ddof", 0)
    return values["x"], values.get("axis"), ddof


def _reduction_axes(axis: Expr | None, stream_rank: int) -> tuple[int, ...]:
    if stream_rank <= 0:
        raise FormulaIRCompileError("reduction stream rank must be positive")
    items = (
        tuple(range(stream_rank))
        if axis is None
        else axis.items
        if isinstance(axis, KeyTuple)
        else (axis,)
    )
    normalized: list[int] = []
    for item in items:
        value = _literal_int(item, "reduction axis")
        if value < 0:
            value += stream_rank
        if value < 0 or value >= stream_rank:
            raise FormulaIRCompileError(
                f"reduction axis {value} outside rank {stream_rank}"
            )
        if value in normalized:
            raise FormulaIRCompileError(f"duplicate reduction axis {value}")
        normalized.append(value)
    return tuple(sorted(normalized))


def _normalize_emit(call: Call) -> tuple[Expr, str]:
    if len(call.args) != 1:
        raise FormulaIRCompileError("emit expects one expression")
    values = dict(call.kwargs)
    if len(values) != len(call.kwargs) or set(values) - {"mode"}:
        raise FormulaIRCompileError("emit supports only the mode keyword")
    mode = _literal_string(values.get("mode", String("last")), "emit mode")
    if mode != "last":
        raise FormulaIRCompileError("emit currently supports only mode='last'")
    return call.args[0], mode


def _normalize_ewm(call: Call) -> tuple[Expr, float, int, bool, bool]:
    names = ("x", "span", "min_periods", "ignore_na", "adjust")
    values: dict[str, Expr] = {
        "min_periods": Number(0.0),
        "ignore_na": Number(1.0),
        "adjust": Number(0.0),
    }
    explicit: set[str] = set()
    for name, value in zip(names, call.args):
        values[name] = value
        explicit.add(name)
    for name, value in call.kwargs:
        if name not in names or name in explicit:
            raise FormulaIRCompileError(f"invalid ewm argument {name!r}")
        values[name] = value
        explicit.add(name)
    if "x" not in values or "span" not in values:
        raise FormulaIRCompileError("ewm requires x and span")
    span = _literal_number(values["span"], "ewm span")
    if span <= 0.0:
        raise FormulaIRCompileError("ewm span must be > 0")
    return (
        values["x"],
        span,
        _literal_int(values["min_periods"], "min_periods", 0),
        _literal_bool(values["ignore_na"], "ignore_na"),
        _literal_bool(values["adjust"], "adjust"),
    )


def _normalize_shift(call: Call) -> tuple[Expr, int, int]:
    if call.kwargs or not 1 <= len(call.args) <= 3:
        raise FormulaIRCompileError("shift expects x[,lag[,max_lag]]")
    lag = 1 if len(call.args) < 2 else _literal_int(call.args[1], "shift lag", 0)
    maximum = lag if len(call.args) < 3 else _literal_int(
        call.args[2], "shift max_lag", 0
    )
    if lag > maximum:
        raise FormulaIRCompileError("shift lag exceeds max_lag")
    return call.args[0], lag, maximum


def _normalize_ffill(call: Call) -> tuple[Expr, int | None]:
    if call.kwargs or not 1 <= len(call.args) <= 2:
        raise FormulaIRCompileError("ffill expects x[,limit]")
    return (
        call.args[0],
        None if len(call.args) == 1 else _literal_int(call.args[1], "ffill limit", 0),
    )


def _normalize_einsum(call: Call) -> tuple[str, tuple[Expr, ...], object]:
    if not call.args:
        raise FormulaIRCompileError("einsum requires subscripts and operands")
    kwargs = dict(call.kwargs)
    if len(kwargs) != len(call.kwargs):
        raise FormulaIRCompileError("einsum got duplicate keyword arguments")
    unknown = set(kwargs) - {"optimize"}
    if unknown:
        raise FormulaIRCompileError(
            f"unsupported einsum keyword argument(s): {sorted(unknown)}"
        )
    optimize: object = False if "optimize" not in kwargs else _literal_optimize(
        kwargs["optimize"]
    )

    if isinstance(call.args[0], String):
        subscripts = call.args[0].value
        operands = call.args[1:]
        if operands and isinstance(operands[-1], String):
            raise FormulaIRCompileError("einsum received multiple subscript strings")
    elif isinstance(call.args[-1], String):
        subscripts = call.args[-1].value
        operands = call.args[:-1]
    else:
        raise FormulaIRCompileError(
            "einsum expects a string subscript as its first argument"
        )
    if not operands:
        raise FormulaIRCompileError("einsum requires at least one operand")
    return subscripts, tuple(operands), optimize


def _flatten_cat_features(expressions: tuple[Expr, ...]) -> tuple[Expr, ...]:
    result: list[Expr] = []
    for expression in expressions:
        if isinstance(expression, Call) and expression.fn == "cat" and not expression.kwargs:
            result.extend(_flatten_cat_features(expression.args))
        else:
            result.append(expression)
    return tuple(result)


def _normalize_ridge(
    call: Call,
) -> tuple[tuple[Expr, ...], Expr, Expr | None, Expr, Expr, bool, bool]:
    if call.kwargs:
        values = dict(call.kwargs)
        if set(values) - {"y", "weights", "hl", "lambda_", "nonneg"}:
            raise FormulaIRCompileError("invalid Ridge keyword")
        if any(name not in values for name in ("y", "hl", "lambda_")):
            raise FormulaIRCompileError("Ridge missing y/hl/lambda_")
        features = call.args
        y = values["y"]
        weights = values.get("weights")
        hl = values["hl"]
        lam = values["lambda_"]
        nonneg = _literal_bool(values.get("nonneg", Number(0.0)), "Ridge nonneg")
    else:
        args = call.args
        sentinel = (
            len(args) >= 5
            and isinstance(args[-1], Number)
            and float(args[-1].value) in (2.0, 3.0)
        )
        nonneg = _literal_bool(args[-1], "Ridge nonneg") if sentinel else False
        if sentinel:
            args = args[:-1]
        if len(args) >= 5:
            features, (y, weights, hl, lam) = args[:-4], args[-4:]
        elif len(args) >= 4:
            features, (y, hl, lam), weights = args[:-3], args[-3:], None
        else:
            raise FormulaIRCompileError(
                "Ridge expects features,y,[weights,]hl,lambda"
            )
    features = _flatten_cat_features(tuple(features))
    return (
        features,
        y,
        weights,
        hl,
        lam,
        nonneg,
        not (isinstance(hl, Number) and float(hl.value) == 0.0),
    )


def _resolve_universe_groups(
    universe: Universe, columns: dict[str, int]
) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for group in universe.groups:
        resolved: list[int] = []
        for member in group:
            index = member if isinstance(member, int) else columns.get(member, -1)
            if index < 0 or index in seen:
                raise FormulaIRCompileError(f"invalid universe member {member!r}")
            seen.add(index)
            resolved.append(index)
        groups.append(tuple(resolved))
    return tuple(groups)


class _BaseBuilder:
    nodes: list[Node]
    memo: dict[tuple, int]
    grouped: bool

    def _append(self, op, children: tuple[int, ...], value_type: ValueType) -> int:
        index = len(self.nodes)
        self.nodes.append(Node(op=op, child_ids=children, value_type=value_type))
        return index

    def _expand(self, node: Call) -> Expr | None:
        macro = self.registry.get(node.fn)
        if macro is None:
            return None
        try:
            expanded = macro(*node.args, **dict(node.kwargs))
        except Exception as exc:
            raise FormulaIRCompileError(f"failed expanding {node.fn!r}: {exc}") from exc
        return expanded if _expr_key(expanded) != _expr_key(node) else None

    def build(self, node: Expr) -> int:
        key = _expr_key(node)
        if key in self.memo:
            return self.memo[key]
        result = self._build_uncached(node)
        self.memo[key] = result
        return result

    def _build_uncached(self, node: Expr) -> int:
        if isinstance(node, Key):
            return self.build(node.expr)
        if isinstance(node, Number):
            return self._append(LiteralOp(node.value), (), SCALAR)
        if isinstance(node, StatelessCall):
            children = tuple(self.build(arg) for arg in node.args)
            name = node.cpp_name or node.name
            if not name:
                raise FormulaIRCompileError("stateless call requires a native name")
            return self._append(
                CustomCallOp(name, len(children)),
                children,
                _custom_value_type(node, [self.nodes[index] for index in children]),
            )
        if isinstance(node, Call):
            expanded = self._expand(node)
            if expanded is not None:
                return self.build(expanded)
            return self._build_call(node)
        return self._build_terminal_or_capture(node)

    def _build_call(self, node: Call) -> int:
        if node.fn in _NARY_ARITY:
            arity = _NARY_ARITY[node.fn]
            if node.kwargs or len(node.args) != arity:
                raise FormulaIRCompileError(f"{node.fn} expects {arity} args")
            children = tuple(self.build(arg) for arg in node.args)
            return self._append(
                NaryOp(node.fn, arity),
                children,
                _nary_result_type(
                    node.fn, [self.nodes[index] for index in children]
                ),
            )
        if node.fn in {"sum", "mean", "std"}:
            expression, axis, ddof = _reduction_arguments(node)
            child = self.build(expression)
            child_type = self.nodes[child].value_type
            try:
                row_shape = child_type.logical_shape
            except ValueError as exc:
                raise FormulaIRCompileError(
                    f"{node.fn} cannot reduce object values"
                ) from exc
            axes = _reduction_axes(axis, 1 + len(row_shape))
            output_shape = tuple(
                extent
                for full_axis, extent in enumerate(row_shape, start=1)
                if full_axis not in axes
            )
            return self._append(
                ReductionOp(node.fn, axes, ddof),
                (child,),
                tensor(output_shape, dtype=child_type.dtype),
            )
        if node.fn == "emit":
            expression, mode = _normalize_emit(node)
            child = self.build(expression)
            return self._append(
                EmitOp(mode), (child,), self.nodes[child].value_type
            )
        if node.fn == "cat":
            children = tuple(self.build(arg) for arg in node.args)
            widths = tuple(
                _feature_width(self.nodes[index].value_type) for index in children
            )
            return self._append(CatOp(widths), children, matrix(sum(widths)))
        if node.fn == "cumsum":
            child = self.build(node.args[0])
            return self._append(
                CumsumOp(),
                (child,),
                _lane_state_result_type("cumsum", self.nodes[child]),
            )
        if node.fn == "ffill":
            x, limit = _normalize_ffill(node)
            child = self.build(x)
            return self._append(
                FFillOp(limit),
                (child,),
                _lane_state_result_type("ffill", self.nodes[child]),
            )
        if node.fn == "shift":
            x, lag, maximum = _normalize_shift(node)
            child = self.build(x)
            return self._append(
                ShiftOp(lag, maximum),
                (child,),
                _lane_state_result_type("shift", self.nodes[child]),
            )
        if node.fn == "ewm":
            x, span, minimum, ignore, adjust = _normalize_ewm(node)
            child = self.build(x)
            return self._append(
                EwmOp(span, minimum, ignore, adjust),
                (child,),
                _lane_state_result_type("ewm", self.nodes[child]),
            )
        if node.fn == "xs_rank":
            return self._append(XsRankOp(), (self.build(node.args[0]),), VECTOR)
        if node.fn == "rbf_basis":
            width = _literal_int(node.args[3], "n_basis", 1)
            return self._append(
                RbfBasisOp(width),
                tuple(self.build(arg) for arg in node.args[:3]),
                matrix(width),
            )
        if node.fn == "future_rbf_basis_sum":
            width = _literal_int(node.args[3], "n_basis", 1)
            steps = _literal_int(node.args[4], "n_steps", 1)
            return self._append(
                FutureRbfBasisSumOp(width, steps),
                tuple(self.build(arg) for arg in node.args[:3]),
                matrix(width),
            )
        if node.fn == "einsum":
            subscripts, operand_exprs, optimize = _normalize_einsum(node)
            children = tuple(self.build(arg) for arg in operand_exprs)
            child_types = tuple(self.nodes[index].value_type for index in children)
            try:
                shapes = tuple(value_type.logical_shape for value_type in child_types)
            except ValueError as exc:
                raise FormulaIRCompileError(
                    "einsum operands must be scalar/vector/matrix/fixed/tensor values"
                ) from exc
            try:
                spec = parse_einsum(subscripts, shapes, optimize=optimize)
            except EinsumParseError as exc:
                raise FormulaIRCompileError(str(exc)) from exc
            return self._append(EinsumOp(spec), children, tensor(spec.output_shape))
        if node.fn == "InstrumentBasisMean":
            if len(node.args) not in {3, 4}:
                raise FormulaIRCompileError(
                    "InstrumentBasisMean expects features,y,[weights,]hl"
                )
            feature = self.build(node.args[0])
            y = self.build(node.args[1])
            has_weights = len(node.args) == 4
            children = [feature, y]
            if has_weights:
                children.append(self.build(node.args[2]))
                hl = node.args[3]
            else:
                hl = node.args[2]
            children.append(self.build(hl))
            width = _feature_width(self.nodes[feature].value_type)
            return self._append(
                InstrumentBasisMeanOp(width, has_weights),
                tuple(children),
                object_value(width),
            )
        if node.fn == "Ridge":
            features, y, weights, hl, lam, nonneg, stateful = _normalize_ridge(node)
            feature_ids = tuple(self.build(feature) for feature in features)
            widths = tuple(
                _feature_width(self.nodes[index].value_type)
                for index in feature_ids
            )
            children = list(feature_ids) + [self.build(y)]
            if weights is not None:
                children.append(self.build(weights))
            children.extend((self.build(hl), self.build(lam)))
            op = RidgeOp(widths, weights is not None, nonneg, stateful)
            return self._append(
                op, tuple(children), object_value(op.coefficient_width)
            )
        if node.fn in {"get_beta", "get_preds"}:
            child = self.build(node.args[0])
            child_node = self.nodes[child]
            field = "beta" if node.fn == "get_beta" else "preds"
            if isinstance(child_node.op, RidgeOp):
                value_type = (
                    matrix(child_node.op.coefficient_width)
                    if field == "beta" and self.grouped
                    else fixed(child_node.op.coefficient_width)
                    if field == "beta"
                    else VECTOR
                )
                return self._append(RidgeProjectionOp(field), (child,), value_type)
            if isinstance(child_node.op, InstrumentBasisMeanOp):
                return self._append(
                    InstrumentBasisProjectionOp(field),
                    (child,),
                    matrix(child_node.op.feature_width) if field == "beta" else VECTOR,
                )
            raise FormulaIRCompileError(
                f"{node.fn} requires Ridge or InstrumentBasisMean"
            )
        if node.fn == "groupby":
            if self.grouped:
                raise FormulaIRCompileError("nested groupby is unsupported")
            return self._build_groupby(node)
        raise FormulaIRCompileError(f"neutral IR does not support {node.fn!r}")


@dataclass
class _OuterBuilder(_BaseBuilder):
    registry: DSLFunctionRegistry
    columns: dict[str, int]
    input_value_types: Mapping[str, ValueType]
    grouped: bool = False

    def __post_init__(self) -> None:
        self.nodes = []
        self.memo = {}
        self.inputs: dict[str, int] = {}

    def _build_terminal_or_capture(self, node: Expr) -> int:
        if isinstance(node, String):
            raise FormulaIRCompileError("string literal invalid here")
        if not isinstance(node, Identifier):
            raise FormulaIRCompileError(f"unsupported expression {node!r}")
        derived = _DERIVED_TERMINALS.get(node.name)
        if derived is not None:
            return self.build(derived)
        if node.name == "self_":
            raise FormulaIRCompileError("self_ only valid in groupby RHS")
        input_index = self.inputs.setdefault(node.name, len(self.inputs))
        return self._append(
            InputOp(input_index, node.name),
            (),
            self.input_value_types.get(node.name, VECTOR),
        )

    def _build_groupby(self, call: Call) -> int:
        if len(call.args) != 3:
            raise FormulaIRCompileError("groupby requires key,lhs,rhs")
        kw = dict(call.kwargs)
        capacity = (
            _literal_int(kw["capacity"], "capacity", 1)
            if "capacity" in kw
            else None
        )
        hash_capacity = (
            _literal_int(kw["hash_capacity"], "hash_capacity", 1)
            if "hash_capacity" in kw
            else None
        )
        unsupported = set(kw) - {"capacity", "hash_capacity"}
        if unsupported:
            raise FormulaIRCompileError(
                f"unsupported groupby keyword(s): {sorted(unsupported)}"
            )
        key_expr = call.args[0]
        key_items = key_expr.items if isinstance(key_expr, KeyTuple) else (key_expr,)
        universes = [item for item in key_items if isinstance(item, Universe)]
        if len(universes) > 1:
            raise FormulaIRCompileError("only one universe key is allowed")
        static_groups = (
            _resolve_universe_groups(universes[0], self.columns)
            if universes
            else None
        )
        dynamic: list[Expr] = []
        specs: list[GroupKeySpec] = []
        for item in key_items:
            if isinstance(item, Universe):
                continue
            if isinstance(item, Key):
                dynamic.append(item.expr)
                specs.append(
                    GroupKeySpec(
                        item.num_keys,
                        item.offset,
                        item.row_scalar,
                        item.dtype,
                    )
                )
            else:
                dynamic.append(item)
                specs.append(GroupKeySpec())
        key_ids = tuple(self.build(item) for item in dynamic)
        lhs = self.build(call.args[1])
        inner = _InnerBuilder(self)
        inner_root = inner.build(call.args[2])
        inner_program = Program(
            tuple(inner.nodes),
            (inner_root,),
            ("__self__",)
            + tuple(f"__capture_{index}__" for index in range(len(inner.capture_ids))),
        )
        if any(
            isinstance(inner_node.op, (ReductionOp, EmitOp))
            for inner_node in inner_program.nodes
        ):
            raise FormulaIRCompileError(
                "reductions and emit are not yet supported inside groupby"
            )
        op = GroupByOp(
            tuple(specs), static_groups, inner_program, capacity, hash_capacity
        )
        return self._append(
            op,
            key_ids + (lhs,) + tuple(inner.capture_ids),
            inner_program.nodes[inner_root].value_type,
        )


class _InnerBuilder(_BaseBuilder):
    grouped = True

    def __init__(self, outer: _OuterBuilder) -> None:
        self.outer = outer
        self.registry = outer.registry
        self.nodes = []
        self.memo = {}
        self.capture_ids: list[int] = []
        self.capture_map: dict[tuple, int] = {}
        self.self_input: int | None = None

    def _build_terminal_or_capture(self, node: Expr) -> int:
        if isinstance(node, Identifier) and node.name == "self_":
            if self.self_input is None:
                self.self_input = self._append(
                    InputOp(0, "__self__"), (), VECTOR
                )
            return self.self_input
        if _contains_self(node):
            raise FormulaIRCompileError(
                f"unsupported self-dependent expression {node!r}"
            )
        key = _expr_key(node)
        position = self.capture_map.get(key)
        if position is None:
            position = len(self.capture_ids)
            self.capture_map[key] = position
            self.capture_ids.append(self.outer.build(node))
        outer_node = self.outer.nodes[self.capture_ids[position]]
        return self._append(
            InputOp(position + 1, f"__capture_{position}__"),
            (),
            outer_node.value_type,
        )

    def _build_groupby(self, call: Call) -> int:
        raise FormulaIRCompileError("nested groupby is unsupported")


def compile_ir(
    formula: str | Expr,
    *,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
    input_value_types: Mapping[str, ValueType] | None = None,
) -> Program:
    expression = parse_formula(formula) if isinstance(formula, str) else formula
    builder = _OuterBuilder(
        dsl_registry or DEFAULT_DSL_REGISTRY,
        {name: index for index, name in enumerate(column_names or ())},
        input_value_types or {},
    )
    root = builder.build(expression)
    for node_id, node in enumerate(builder.nodes):
        terminal = isinstance(node.op, EmitOp) or (
            isinstance(node.op, ReductionOp) and node.op.temporal
        )
        if terminal and node_id != root:
            raise FormulaIRCompileError(
                "temporal reductions and emit('last') must be the terminal output"
            )
    return Program(tuple(builder.nodes), (root,), tuple(builder.inputs))


__all__ = ["FormulaIRCompileError", "compile_ir"]
