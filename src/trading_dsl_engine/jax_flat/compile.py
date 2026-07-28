"""Formula lowering from DSL expressions to JAX-flat streaming programs.

Parses and normalizes formula AST nodes, expands DSL macros, builds operator
DAGs (including groupby RHS inner graphs), and returns a ``JaxFlatRuntime``
wrapping the compiled ``StreamingProgram``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from inspect import Parameter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import DEFAULT_DSL_REGISTRY, DSLFunctionRegistry, ensure_expr, get_dsl_op_signature
from trading_dsl_engine.base.metadata import MetadataConfig
from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple, Number, String, Universe, parse_formula
from trading_dsl_engine.jax_flat.custom import RollingJaxCall, StatelessJaxCall
from trading_dsl_engine.jax_flat.ops import (
    InstrumentBasisMeanOp,
    RbfBasisOp,
    FutureRbfBasisSumOp,
    BufferShiftOp,
    CacheOp,
    EwmOp,
    FFillOp,
    GroupByOp,
    InputOp,
    LiteralOp,
    NaryOp,
    RollingMeanOp,
    RollingOp,
    ANY_ARITY,
    OP_FACTORIES,
    Op,
    RidgeOp,
    ShiftOp,
    _bspline,
    _cat,
    _col,
    _einsum,
)
from trading_dsl_engine.jax_flat.program import (
    DagNode,
    InnerGraphOp,
    StateFieldRef,
    StateLayout,
    StreamingProgram,
)
from trading_dsl_engine.jax_flat.runtime import JaxFlatRuntime


# --- Keyword / call-site normalization (compile-time only) ---


def _normalize_variadic_kwargs(fn: str, args: tuple[Expr, ...], kwargs: tuple[tuple[str, Expr], ...]) -> tuple[Expr, ...]:
    values: list[Expr | None] = list(args)
    seen_positions: set[int] = set()
    for name, value in kwargs:
        if not name.startswith("arg") or not name[3:].isdigit():
            raise ValueError(f"Unsupported {fn} keyword argument(s): {[name]}")
        position = int(name[3:])
        if position < len(args) or position in seen_positions:
            raise ValueError(f"{fn} got multiple values for argument {name!r}")
        while len(values) <= position:
            values.append(None)
        values[position] = value
        seen_positions.add(position)
    if any(value is None for value in values):
        missing_position = next(i for i, value in enumerate(values) if value is None)
        raise ValueError(f"{fn} missing required argument 'arg{missing_position}' before keyword arguments")
    return tuple(value for value in values if value is not None)


def _normalize_ridge_kwargs(args: tuple[Expr, ...], kwargs: tuple[tuple[str, Expr], ...]) -> tuple[Expr, ...]:
    values: dict[str, Expr] = {}
    for name, value in kwargs:
        if name not in {"y", "weights", "hl", "lambda_", "nonneg"}:
            raise ValueError(f"Unsupported Ridge keyword argument(s): {[name]}")
        if name in values:
            raise ValueError(f"Ridge got multiple values for argument {name!r}")
        values[name] = value

    y = values.get("y")
    hl = values.get("hl")
    lambda_value = values.get("lambda_")
    weights = values.get("weights")
    nonneg = values.get("nonneg", Number(2.0))
    if isinstance(nonneg, Identifier) and nonneg.name in {"True", "False"}:
        nonneg = Number(3.0 if nonneg.name == "True" else 2.0)
    elif isinstance(nonneg, Number):
        nonneg = Number(3.0 if bool(nonneg.value) else 2.0)
    tail = (y, weights if weights is not None else Number(1.0), hl, lambda_value, nonneg)
    if any(value is None for value in tail[:-1]):
        raise ValueError("Ridge keyword form requires y, hl, and lambda_")
    return args + tail


def _normalize_call_kwargs(fn: str, args: tuple[Expr, ...], kwargs: tuple[tuple[str, Expr], ...]) -> tuple[Expr, ...]:
    if not kwargs or fn == "groupby":
        return args
    if fn == "Ridge":
        return _normalize_ridge_kwargs(args, kwargs)
    if fn in {"cat", "einsum"}:
        return _normalize_variadic_kwargs(fn, args, kwargs)

    signature = get_dsl_op_signature(fn)
    if signature is None:
        return args
    parameter_names = tuple(
        name for name, param in signature.parameters.items() if param.kind is not param.VAR_POSITIONAL
    )
    positions_by_name = {name: idx for idx, name in enumerate(parameter_names)}
    values: list[Expr | None] = list(args)
    seen_positions: set[int] = set()
    for name, value in kwargs:
        if name not in positions_by_name:
            raise ValueError(f"Unsupported {fn} keyword argument(s): {[name]}")
        position = positions_by_name[name]
        if position < len(args) or position in seen_positions:
            raise ValueError(f"{fn} got multiple values for argument {name!r}")
        while len(values) <= position:
            values.append(None)
        values[position] = value
        seen_positions.add(position)

    for idx, value in enumerate(values):
        if value is None:
            default = signature.parameters[parameter_names[idx]].default
            if default is Parameter.empty:
                expected = parameter_names[idx]
                raise ValueError(f"{fn} missing required argument {expected!r} before keyword arguments")
            values[idx] = None if default is None else ensure_expr(default)
    return tuple(value for value in values if value is not None)


def _normalize_static_jax_flat_kwargs(node: Expr) -> Expr:
    if isinstance(node, Call):
        args = tuple(_normalize_static_jax_flat_kwargs(arg) for arg in node.args)
        kwargs = tuple((key, _normalize_static_jax_flat_kwargs(value)) for key, value in node.kwargs)
        if not kwargs:
            return Call(node.fn, args)
        return Call(node.fn, _normalize_call_kwargs(node.fn, args, kwargs))
    if isinstance(node, RollingJaxCall):
        return RollingJaxCall(
            fn=node.fn,
            args=tuple(_normalize_static_jax_flat_kwargs(arg) for arg in node.args),
            lookback=node.lookback,
            min_periods=node.min_periods,
            output_kind=node.output_kind,
            output_width=node.output_width,
            name=node.name,
        )
    if isinstance(node, StatelessJaxCall):
        return StatelessJaxCall(
            fn=node.fn,
            args=tuple(_normalize_static_jax_flat_kwargs(arg) for arg in node.args),
            output_kind=node.output_kind,
            output_width=node.output_width,
            name=node.name,
            cpp_name=node.cpp_name,
        )
    if isinstance(node, KeyTuple):
        return KeyTuple(tuple(_normalize_static_jax_flat_kwargs(item) for item in node.items))
    return node

# --- Expression keys and groupby canonical-form helpers ---


def _expr_key(node: Expr):
    if isinstance(node, Identifier):
        return ("id", node.name)
    if isinstance(node, Number):
        return ("num", float(node.value))
    if isinstance(node, String):
        return ("str", node.value)
    if isinstance(node, Call):
        return (
            "call",
            node.fn,
            tuple(_expr_key(a) for a in node.args),
            tuple((k, _expr_key(v)) for k, v in node.kwargs),
        )
    if isinstance(node, RollingJaxCall):
        return (
            "jax_rolling",
            id(node.fn),
            node.lookback,
            node.min_periods,
            node.output_kind,
            node.output_width,
            node.name,
            tuple(_expr_key(a) for a in node.args),
        )
    if isinstance(node, StatelessJaxCall):
        return (
            "jax_stateless",
            id(node.fn),
            node.output_kind,
            node.output_width,
            node.name,
            node.cpp_name,
            tuple(_expr_key(a) for a in node.args),
        )
    if isinstance(node, KeyTuple):
        return ("tuple", tuple(_expr_key(item) for item in node.items))
    if isinstance(node, Universe):
        return ("univ", node.groups)
    raise ValueError(f"Unsupported expression: {node}")


def _canonical_groupby_key_items(key: Expr) -> tuple[Expr, ...]:
    if not isinstance(key, KeyTuple):
        key = KeyTuple((key,))
    if sum(1 for item in key.items if isinstance(item, Universe)) > 1:
        raise ValueError("groupby key tuple may contain at most one univ(...) element")
    return key.items


def _resolve_universe_groups(universe: Universe) -> tuple[tuple[int, ...], ...]:
    groups = []
    for g in universe.groups:
        cols = []
        for m in g:
            if not isinstance(m, int):
                raise ValueError("jax_flat univ currently supports integer column indexes only")
            cols.append(m)
        groups.append(tuple(cols))
    return tuple(groups)


def _validate_groupby_canonical_form(expr: Call) -> None:
    if len(expr.args) != 3:
        raise ValueError(
            "groupby only supports canonical form: "
            "groupby((key1, ..., maybe_univ, ...), lhs, op_using_self_)"
        )
    unsupported_kwargs = {name for name, _ in expr.kwargs} - {"capacity", "hash_capacity"}
    if unsupported_kwargs:
        raise ValueError(f"Unsupported groupby keyword argument(s): {sorted(unsupported_kwargs)}")
    for name, value in expr.kwargs:
        if not isinstance(value, Number):
            raise ValueError(f"groupby {name} must be a numeric literal")
    _canonical_groupby_key_items(expr.args[0])


def _replace_self(node: Expr, lhs: Expr) -> Expr:
    if isinstance(node, Identifier) and node.name == "self_":
        return lhs
    if isinstance(node, Call):
        return Call(
            node.fn,
            tuple(_replace_self(a, lhs) for a in node.args),
            tuple((key, _replace_self(value, lhs)) for key, value in node.kwargs),
        )
    if isinstance(node, RollingJaxCall):
        return RollingJaxCall(
            fn=node.fn,
            args=tuple(_replace_self(arg, lhs) for arg in node.args),
            lookback=node.lookback,
            min_periods=node.min_periods,
            output_kind=node.output_kind,
            output_width=node.output_width,
            name=node.name,
        )
    if isinstance(node, StatelessJaxCall):
        return StatelessJaxCall(
            fn=node.fn,
            args=tuple(_replace_self(a, lhs) for a in node.args),
            output_kind=node.output_kind,
            output_width=node.output_width,
            name=node.name,
            cpp_name=node.cpp_name,
        )
    if isinstance(node, KeyTuple):
        return KeyTuple(tuple(_replace_self(a, lhs) for a in node.items))
    return node


def _literal_int_arg(expr: Expr, fn: str, position: int) -> int:
    if not isinstance(expr, Number):
        raise ValueError(f"{fn} argument {position} must be a numeric literal")
    return int(round(float(expr.value)))


def _op_width(op: Op) -> int:
    if op.output_width is None:
        raise ValueError(f"Cannot infer static output width for {op.output_kind} op in jax_flat")
    return int(op.output_width)



def _shape_source_op(child_ops: tuple[Op, ...]) -> Op | None:
    for child in child_ops:
        if child.output_kind != "scalar":
            return child
    return child_ops[0] if child_ops else None


def _shape_preserving_nary(op: Op, child_ops: tuple[Op, ...]) -> Op:
    if not isinstance(op, NaryOp):
        return op
    shape_source = _shape_source_op(child_ops)
    if shape_source is None:
        return op
    shape_preserving = {
        "abs", "ln", "ceil", "floor", "round", "exp", "sign", "arctan",
        "isnan", "purify", "fraction", "add", "sub", "mul", "mod", "pow",
        "div", "floordiv", "eq", "ne", "lt", "gt", "le", "ge", "and", "or", "xor",
        "fillna", "where", "clip", "norm_inv", "xs_norm", "cache",
    }
    if op.cpp_name not in shape_preserving:
        return op
    return replace(op, output_kind=shape_source.output_kind, output_width=shape_source.output_width)


def _object_projection_op(expr: Call, child_ops: tuple[Op, ...]) -> tuple[Op, int | tuple[int, ...] | None] | None:
    if expr.fn not in {"get_beta", "get_preds"} or len(expr.args) != 1 or not child_ops:
        return None
    child = child_ops[0]
    op = OP_FACTORIES[(expr.fn, 1)]()
    if expr.fn == "get_preds":
        return replace(op, output_kind="vector", output_width=1), None
    if isinstance(child, InstrumentBasisMeanOp):
        return replace(op, output_kind="matrix", output_width=child.feature_width), None
    if isinstance(child, RidgeOp):
        return replace(op, output_kind="vector", output_width=sum(child.feature_widths)), None
    return replace(op, output_kind=child.output_kind, output_width=child.output_width), None


# --- Per-call operator lowering (`Call` -> `Op`) ---


def _build_rolling_jax_op(expr: RollingJaxCall, child_ops: tuple[Op, ...]) -> Op:
    if len(child_ops) != 1:
        raise ValueError("rolling JAX call expects one child")
    output_kind = expr.output_kind if expr.output_kind is not None else child_ops[0].output_kind
    output_width = expr.output_width if expr.output_width is not None else child_ops[0].output_width
    return RollingOp(expr.lookback, expr.min_periods, expr.fn, output_kind=output_kind, output_width=output_width)


def _build_stateless_jax_op(expr: StatelessJaxCall, child_ops: tuple[Op, ...]) -> Op:
    if not child_ops:
        raise ValueError("stateless JAX call expects at least one child")
    output_kind = expr.output_kind if expr.output_kind is not None else child_ops[0].output_kind
    output_width = expr.output_width if expr.output_width is not None else child_ops[0].output_width
    return NaryOp(
        expr.fn,
        output_kind=output_kind,
        output_width=output_width,
        cpp_name=expr.cpp_name,
        diagnostic_name=expr.name,
    )


def _build_op(expr: Call, child_ops: tuple[Op, ...] = ()) -> tuple[Op, int | tuple[int, ...] | None]:
    if expr.fn == "groupby":
        raise ValueError("groupby must be compiled with its RHS subgraph")
    if expr.kwargs:
        raise ValueError(f"Keyword arguments are not supported for {expr.fn}")
    projection = _object_projection_op(expr, child_ops)
    if projection is not None:
        return projection
    if expr.fn == "cache" and len(expr.args) in (1, 2):
        storage = "ram"
        if len(expr.args) == 2:
            if not isinstance(expr.args[1], String):
                raise ValueError("cache storage must be a string literal")
            storage = expr.args[1].value
        if storage not in {"ram", "disk"}:
            raise ValueError("cache storage must be 'ram' or 'disk'")
        return CacheOp(storage=storage, output_kind=child_ops[0].output_kind, output_width=child_ops[0].output_width), 1 if len(expr.args) == 2 else None
    if expr.fn == "cat" and len(expr.args) >= 1:
        return NaryOp(_cat, output_kind="matrix", output_width=sum(_op_width(op) for op in child_ops), cpp_name="cat"), None
    if expr.fn == "einsum":
        static_args = tuple(arg.value for arg in expr.args if isinstance(arg, String))
        if len(static_args) != 1:
            raise ValueError("einsum expects one string subscript literal")
        subscripts = str(static_args[0])
        output = subscripts.split("->", 1)[1] if "->" in subscripts else ""
        if output == "":
            return NaryOp(lambda *child_values, subscripts=subscripts: _einsum(subscripts, *child_values), output_kind="scalar", output_width=1, cpp_name="einsum", cpp_str_param=subscripts), None
        index_widths = {}
        input_terms = subscripts.split("->", 1)[0].split(",")
        for term, op in zip(input_terms, child_ops):
            compact = term.replace("...", "")
            if len(compact) >= 1:
                index_widths.setdefault(compact[0], 1)
            if len(compact) >= 2 and op.output_width is not None:
                index_widths[compact[1]] = int(op.output_width)
        if len(output) == 1:
            width = index_widths.get(output, 1)
            return NaryOp(lambda *child_values, subscripts=subscripts: _einsum(subscripts, *child_values), output_kind="vector", output_width=width, cpp_name="einsum", cpp_str_param=subscripts), None
        if len(output) == 2:
            # Matrix width is inferred from the last output index when possible;
            # dynamic widths (for outer-like i,j output) use None.
            width = index_widths.get(output[-1])
            return NaryOp(lambda *child_values, subscripts=subscripts: _einsum(subscripts, *child_values), output_kind="matrix", output_width=width, cpp_name="einsum", cpp_str_param=subscripts), None
    variadic_builder = OP_FACTORIES.get((expr.fn, ANY_ARITY))
    if variadic_builder is not None:
        static_args = tuple(arg.value for arg in expr.args if isinstance(arg, String))
        return variadic_builder(*static_args), None
    if expr.fn == "round":
        if len(expr.args) == 1:
            return NaryOp(jnp.round, cpp_name="round"), None
        if len(expr.args) == 2 and isinstance(expr.args[1], Number):
            decimals = int(round(float(expr.args[1].value)))
            return NaryOp(lambda x, decimals=decimals: jnp.round(x, decimals=decimals)), 1
    if expr.fn == "ewm" and len(expr.args) in (2, 3, 4, 5):
        span = float(expr.args[1].value) if isinstance(expr.args[1], Number) else None
        min_periods = None
        drop: int | tuple[int, ...] | None = 1 if span is not None else None
        if len(expr.args) >= 3:
            if isinstance(expr.args[2], Number):
                min_periods = float(expr.args[2].value)
                drop = (1, 2) if span is not None else 2
            else:
                drop = 1 if span is not None else None
        ignore_na = True
        adjust = False
        drop_indices = set()
        if isinstance(drop, tuple):
            drop_indices.update(drop)
        elif isinstance(drop, int):
            drop_indices.add(drop)
        if len(expr.args) >= 4:
            ignore_na = bool(_literal_int_arg(expr.args[3], "ewm", 4))
            drop_indices.add(3)
        if len(expr.args) >= 5:
            adjust = bool(_literal_int_arg(expr.args[4], "ewm", 5))
            drop_indices.add(4)
        return EwmOp(span=span, min_periods=min_periods, ignore_na=ignore_na, adjust=adjust), tuple(sorted(drop_indices)) if drop_indices else None
    if expr.fn == "roll_mean" and len(expr.args) in (2, 3):
        lookback = _literal_int_arg(expr.args[1], "roll_mean", 2)
        min_periods = lookback if len(expr.args) == 2 else _literal_int_arg(expr.args[2], "roll_mean", 3)
        if lookback <= 0 or min_periods <= 0 or min_periods > lookback:
            raise ValueError("roll_mean expects 0 < min_periods <= lookback")
        return (
            RollingMeanOp(
                lookback=lookback,
                min_periods=min_periods,
                output_kind=child_ops[0].output_kind,
                output_width=child_ops[0].output_width,
            ),
            tuple(range(1, len(expr.args))),
        )
    if expr.fn == "ffill" and len(expr.args) in (1, 2):
        limit = None
        dynamic_limit = False
        drop_child_idx = None
        if len(expr.args) == 2:
            if isinstance(expr.args[1], Number):
                limit = int(round(float(expr.args[1].value)))
                if limit < 0:
                    raise ValueError("ffill limit must be >= 0")
                drop_child_idx = 1
            else:
                dynamic_limit = True
        return (
            FFillOp(
                limit=limit,
                dynamic_limit=dynamic_limit,
                output_kind=child_ops[0].output_kind,
                output_width=child_ops[0].output_width,
            ),
            drop_child_idx,
        )
    if expr.fn == "shift" and len(expr.args) in (1, 2, 3):
        lag_arg = expr.args[1] if len(expr.args) >= 2 else Number(1.0)
        max_size_arg = expr.args[2] if len(expr.args) == 3 else lag_arg
        max_size = max(0, _literal_int_arg(max_size_arg, "shift", 3 if len(expr.args) == 3 else 2))
        return (
            ShiftOp(
                max_size=max_size,
                output_kind=child_ops[0].output_kind,
                output_width=child_ops[0].output_width,
            ),
            2 if len(expr.args) == 3 else None,
        )
    if expr.fn == "bspline" and len(expr.args) == 2:
        n_basis = _literal_int_arg(expr.args[1], "bspline", 2)
        if n_basis <= 0:
            raise ValueError("bspline n_basis must be >= 1")
        return NaryOp(lambda x, n_basis=n_basis: _bspline(x, n_basis), output_kind="matrix", output_width=n_basis, cpp_name="bspline", cpp_int_param=n_basis), 1
    if expr.fn == "rbf_basis" and len(expr.args) == 4:
        n_basis = _literal_int_arg(expr.args[3], "rbf_basis", 4)
        if n_basis <= 0:
            raise ValueError("rbf_basis n_basis must be >= 1")
        return RbfBasisOp(n_basis=n_basis), 3
    if expr.fn == "future_rbf_basis_sum" and len(expr.args) == 5:
        n_basis = _literal_int_arg(expr.args[3], "future_rbf_basis_sum", 4)
        n_steps = _literal_int_arg(expr.args[4], "future_rbf_basis_sum", 5)
        if n_basis <= 0:
            raise ValueError("future_rbf_basis_sum n_basis must be >= 1")
        if n_steps <= 0:
            raise ValueError("future_rbf_basis_sum n_steps must be >= 1")
        return FutureRbfBasisSumOp(n_basis=n_basis, n_steps=n_steps), (3, 4)
    if expr.fn == "col" and len(expr.args) == 2:
        index = _literal_int_arg(expr.args[1], "col", 2)
        if index < 0:
            raise ValueError("col index must be >= 0")
        return NaryOp(lambda x, index=index: _col(x, index), output_kind="vector", output_width=1, cpp_name="col", cpp_int_param=index), 1
    if expr.fn == "InstrumentBasisMean" and len(expr.args) in (3, 4):
        has_weights = len(expr.args) == 4
        feature_op = child_ops[0]
        return InstrumentBasisMeanOp(feature_width=_op_width(feature_op), has_weights=has_weights), None
    if expr.fn == "Ridge" and len(expr.args) >= 4:
        has_nonneg = len(expr.args) >= 5 and isinstance(expr.args[-1], Number) and float(expr.args[-1].value) in (2.0, 3.0)
        nonneg = bool(_literal_int_arg(expr.args[-1], "Ridge", len(expr.args)) - 2) if has_nonneg else False
        data_child_ops = child_ops[:-1] if has_nonneg else child_ops
        data_args = expr.args[:-1] if has_nonneg else expr.args
        has_weights = len(data_args) >= 5
        feature_ops = data_child_ops[:-4] if has_weights else data_child_ops[:-3]
        if not feature_ops:
            raise ValueError("Ridge expects at least one feature arg")
        feature_widths = tuple(_op_width(op) for op in feature_ops)
        hl_arg = data_args[-2]
        is_stateful = not (isinstance(hl_arg, Number) and float(hl_arg.value) == 0.0)
        return RidgeOp(feature_widths=feature_widths, has_weights=has_weights, nonneg=nonneg, is_stateful=is_stateful), ((len(expr.args) - 1,) if has_nonneg else None)
    builder = OP_FACTORIES.get((expr.fn, len(expr.args)))
    if builder is None:
        raise ValueError(f"Unsupported function {expr.fn}/{len(expr.args)} in jax_flat")
    return builder(), None


def _build_state_layout(nodes: tuple[DagNode, ...], sample: jax.Array | None = None) -> StateLayout:
    del sample
    refs = []
    offset = 0
    for node in nodes:
        if node.op.is_stateful:
            refs.append(StateFieldRef(index=offset))
            offset += 1
        else:
            refs.append(StateFieldRef(index=-1))
    return StateLayout(node_fields=tuple(refs), total_leaves=offset)


# --- Groupby / buffer specialized DAG compilation ---


def _compile_groupby_inner_op(rhs: Expr, lhs: Expr) -> tuple[InnerGraphOp, tuple[Expr, ...]]:
    if not isinstance(rhs, Call):
        raise ValueError("groupby rhs must be a call expression")

    nodes: list[DagNode] = []
    memo: dict[tuple[Any, ...], int] = {}
    feed_exprs: list[Expr] = []
    feed_index_by_key: dict[tuple[Any, ...], int] = {}

    def feed_node(feed_expr: Expr) -> int:
        feed_key = _expr_key(feed_expr)
        input_index = feed_index_by_key.get(feed_key)
        if input_index is None:
            input_index = len(feed_exprs)
            feed_index_by_key[feed_key] = input_index
            feed_exprs.append(feed_expr)

        node_key = ("feed", feed_key)
        cached = memo.get(node_key)
        if cached is not None:
            return cached
        idx = len(nodes)
        nodes.append(DagNode(op=InputOp(input_index=input_index), child_ids=()))
        memo[node_key] = idx
        return idx

    def build(node: Expr) -> int:
        if isinstance(node, Identifier):
            return feed_node(lhs if node.name == "self_" else node)

        key = ("expr", _expr_key(_replace_self(node, lhs)))
        cached = memo.get(key)
        if cached is not None:
            return cached

        if isinstance(node, Number):
            idx = len(nodes)
            nodes.append(DagNode(op=LiteralOp(float(node.value)), child_ids=()))
            memo[key] = idx
            return idx
        if isinstance(node, String):
            raise ValueError(f"String literal {node.value!r} is only supported as a static keyword argument")

        if isinstance(node, RollingJaxCall):
            child_ids = tuple(build(a) for a in node.args)
            idx = len(nodes)
            nodes.append(DagNode(op=_build_rolling_jax_op(node, tuple(nodes[cid].op for cid in child_ids)), child_ids=child_ids))
            memo[key] = idx
            return idx

        if isinstance(node, StatelessJaxCall):
            child_ids = tuple(build(a) for a in node.args)
            idx = len(nodes)
            nodes.append(
                DagNode(
                    op=_build_stateless_jax_op(node, tuple(nodes[cid].op for cid in child_ids)),
                    child_ids=child_ids,
                )
            )
            memo[key] = idx
            return idx

        if isinstance(node, Call):
            if node.fn == "groupby":
                raise ValueError("nested groupby inside a groupby rhs is not supported in jax_flat")
            child_ids = tuple(
                build(a)
                for a in node.args
                if not ((node.fn, ANY_ARITY) in OP_FACTORIES and isinstance(a, String))
            )
            op, drop_child_idx = _build_op(node, tuple(nodes[cid].op for cid in child_ids))
            if drop_child_idx is not None:
                drop_child_idxs = (drop_child_idx,) if isinstance(drop_child_idx, int) else tuple(drop_child_idx)
                child_ids = tuple(cid for i, cid in enumerate(child_ids) if i not in drop_child_idxs)
            idx = len(nodes)
            nodes.append(DagNode(op=op, child_ids=child_ids))
            memo[key] = idx
            return idx

        raise ValueError(f"Unsupported groupby rhs node: {node}")

    output_id = build(rhs)
    node_tuple = tuple(nodes)
    state_layout = _build_state_layout(node_tuple)
    return (
        InnerGraphOp(
            nodes=node_tuple,
            output_id=output_id,
            state_layout=state_layout,
            n_inputs=len(feed_exprs),
            is_stateful=state_layout.total_leaves > 0,
            output_kind=node_tuple[output_id].op.output_kind,
        ),
        tuple(feed_exprs),
    )


def _groupby_capacity_kwargs(expr: Call, has_universe_groups: bool) -> dict[str, int]:
    capacity = 2048 if has_universe_groups else 4096
    hash_capacity = None
    for name, value in expr.kwargs:
        literal = int(value.value)
        if literal <= 0:
            raise ValueError(f"groupby {name} must be positive")
        if name == "capacity":
            capacity = literal
        elif name == "hash_capacity":
            hash_capacity = literal
    return {
        "capacity": capacity,
        "hash_capacity": max(hash_capacity if hash_capacity is not None else capacity * 2, capacity),
    }


def _typed_groupby_inner_op(inner_op: InnerGraphOp, feed_ops: tuple[Op, ...]) -> InnerGraphOp:
    typed_nodes = []
    changed = False
    for node in inner_op.nodes:
        op = node.op
        if isinstance(op, InputOp) and op.input_index < len(feed_ops):
            feed_op = feed_ops[op.input_index]
            op = InputOp(op.input_index, output_kind=feed_op.output_kind, output_width=feed_op.output_width)
            changed = True
        child_ops = tuple(typed_nodes[cid].op for cid in node.child_ids)
        shaped_op = _shape_preserving_nary(op, child_ops)
        changed = changed or shaped_op is not op
        typed_nodes.append(replace(node, op=shaped_op) if shaped_op is not node.op else node)
    node_tuple = tuple(typed_nodes)
    output_kind = node_tuple[inner_op.output_id].op.output_kind
    return replace(inner_op, nodes=node_tuple, output_kind=output_kind) if changed or output_kind != inner_op.output_kind else inner_op


def _compile_groupby_node(
    expr: Call,
    memo: dict[tuple[Any, ...], int],
    nodes: list[DagNode],
    input_names: list[str],
    external_cache_names: dict[tuple[Any, ...], str] | None = None,
) -> int:
    _validate_groupby_canonical_form(expr)
    key_items = _canonical_groupby_key_items(expr.args[0])
    universe_items = [item for item in key_items if isinstance(item, Universe)]

    dynamic_items = [item for item in key_items if not isinstance(item, Universe)]
    inner_op, feed_exprs = _compile_groupby_inner_op(expr.args[2], expr.args[1])
    universe_groups = _resolve_universe_groups(universe_items[0]) if universe_items else None

    key_child_ids = tuple(_compile_node(k, memo, nodes, input_names, external_cache_names) for k in dynamic_items)
    feed_child_ids = tuple(_compile_node(feed, memo, nodes, input_names, external_cache_names) for feed in feed_exprs)
    child_ids = key_child_ids + feed_child_ids
    inner_op = _typed_groupby_inner_op(inner_op, tuple(nodes[cid].op for cid in feed_child_ids))
    idx = len(nodes)
    groupby_kwargs = _groupby_capacity_kwargs(expr, universe_groups is not None)
    nodes.append(
        DagNode(
            op=GroupByOp(
                inner_op=inner_op,
                n_keys=len(dynamic_items),
                universe_groups=universe_groups,
                **groupby_kwargs,
            ),
            child_ids=child_ids,
        )
    )
    memo[_expr_key(expr)] = idx
    return idx


def _compile_buffer_shift_node(
    expr: Call,
    memo: dict[tuple[Any, ...], int],
    nodes: list[DagNode],
    input_names: list[str],
    external_cache_names: dict[tuple[Any, ...], str] | None = None,
) -> int:
    if len(expr.args) != 3:
        raise ValueError("buffer expects args: shift_expr, min_lag, max_lag")
    shift_expr = expr.args[0]
    if not isinstance(shift_expr, Call) or shift_expr.fn != "shift" or len(shift_expr.args) not in (2, 3):
        raise ValueError("buffer first arg must be a direct shift(...) expression")
    max_size_arg = shift_expr.args[2] if len(shift_expr.args) == 3 else shift_expr.args[1]
    max_size = max(0, _literal_int_arg(max_size_arg, "shift", 3 if len(shift_expr.args) == 3 else 2))
    max_lag = _literal_int_arg(expr.args[2], "buffer", 3)
    if max_lag < 1:
        raise ValueError("buffer max_lag must be >= 1")
    if max_lag > max_size:
        raise ValueError("buffer max_lag must be <= shift max_size")

    child_ids = (
        _compile_node(shift_expr.args[0], memo, nodes, input_names, external_cache_names),
        _compile_node(shift_expr.args[1], memo, nodes, input_names, external_cache_names),
        _compile_node(expr.args[1], memo, nodes, input_names, external_cache_names),
    )
    idx = len(nodes)
    nodes.append(DagNode(op=BufferShiftOp(max_size=max_size, max_lag=max_lag), child_ids=child_ids))
    memo[_expr_key(expr)] = idx
    return idx

def _compile_node(
    expr: Expr,
    memo: dict[tuple[Any, ...], int],
    nodes: list[DagNode],
    input_names: list[str],
    external_cache_names: dict[tuple[Any, ...], str] | None = None,
) -> int:
    key = _expr_key(expr)
    if key in memo:
        return memo[key]
    if external_cache_names and key in external_cache_names:
        name = external_cache_names[key]
        if name not in input_names:
            input_names.append(name)
        idx = len(nodes)
        nodes.append(DagNode(op=InputOp(input_index=input_names.index(name)), child_ids=()))
        memo[key] = idx
        return idx
    if isinstance(expr, Identifier):
        if expr.name not in input_names:
            input_names.append(expr.name)
        idx = len(nodes)
        nodes.append(DagNode(op=InputOp(input_index=input_names.index(expr.name)), child_ids=()))
        memo[key] = idx
        return idx
    if isinstance(expr, Number):
        idx = len(nodes)
        nodes.append(DagNode(op=LiteralOp(float(expr.value)), child_ids=()))
        memo[key] = idx
        return idx
    if isinstance(expr, String):
        raise ValueError(f"String literal {expr.value!r} is only supported as a static keyword argument")
    if isinstance(expr, RollingJaxCall):
        child_ids = tuple(_compile_node(a, memo, nodes, input_names, external_cache_names) for a in expr.args)
        idx = len(nodes)
        nodes.append(DagNode(op=_build_rolling_jax_op(expr, tuple(nodes[cid].op for cid in child_ids)), child_ids=child_ids))
        memo[key] = idx
        return idx
    if isinstance(expr, StatelessJaxCall):
        child_ids = tuple(_compile_node(a, memo, nodes, input_names, external_cache_names) for a in expr.args)
        idx = len(nodes)
        nodes.append(
            DagNode(
                op=_build_stateless_jax_op(expr, tuple(nodes[cid].op for cid in child_ids)),
                child_ids=child_ids,
            )
        )
        memo[key] = idx
        return idx
    if isinstance(expr, Call) and expr.fn == "groupby":
        return _compile_groupby_node(expr, memo, nodes, input_names, external_cache_names)
    if isinstance(expr, Call) and expr.fn == "buffer":
        return _compile_buffer_shift_node(expr, memo, nodes, input_names, external_cache_names)
    if not isinstance(expr, Call):
        raise ValueError(f"Unsupported node {expr}")

    child_ids = tuple(
        _compile_node(a, memo, nodes, input_names, external_cache_names)
        for a in expr.args
        if not (((expr.fn, ANY_ARITY) in OP_FACTORIES or expr.fn == "cache") and isinstance(a, String))
    )
    op, drop_child_idx = _build_op(expr, tuple(nodes[cid].op for cid in child_ids))
    if drop_child_idx is not None:
        drop_child_idxs = (drop_child_idx,) if isinstance(drop_child_idx, int) else tuple(drop_child_idx)
        child_ids = tuple(cid for i, cid in enumerate(child_ids) if i not in drop_child_idxs)
    idx = len(nodes)
    nodes.append(DagNode(op=op, child_ids=child_ids))
    memo[key] = idx
    return idx


# --- DSL macro expansion and cross-runtime cache wiring ---


def _expand_dsl(node: Expr, dsl_registry: DSLFunctionRegistry, depth: int = 0) -> Expr:
    if depth > 256:
        raise ValueError("Exceeded max DSL expansion depth (256)")
    if isinstance(node, Call):
        expanded_args = tuple(_expand_dsl(arg, dsl_registry, depth) for arg in node.args)
        expanded_kwargs = tuple((key, _expand_dsl(value, dsl_registry, depth)) for key, value in node.kwargs)
        expanded_node = Call(node.fn, expanded_args, expanded_kwargs)
        macro = dsl_registry.get(expanded_node.fn)
        if macro is None:
            return expanded_node
        result = macro(*expanded_node.args, **dict(expanded_node.kwargs))
        if _expr_key(result) == _expr_key(expanded_node):
            return expanded_node
        return _expand_dsl(result, dsl_registry, depth + 1)
    if isinstance(node, RollingJaxCall):
        return RollingJaxCall(
            fn=node.fn,
            args=tuple(_expand_dsl(arg, dsl_registry, depth) for arg in node.args),
            lookback=node.lookback,
            min_periods=node.min_periods,
            output_kind=node.output_kind,
            output_width=node.output_width,
            name=node.name,
        )
    if isinstance(node, StatelessJaxCall):
        return StatelessJaxCall(
            fn=node.fn,
            args=tuple(_expand_dsl(arg, dsl_registry, depth) for arg in node.args),
            output_kind=node.output_kind,
            output_width=node.output_width,
            name=node.name,
            cpp_name=node.cpp_name,
        )
    if isinstance(node, KeyTuple):
        return KeyTuple(tuple(_expand_dsl(item, dsl_registry, depth) for item in node.items))
    return node




def _normalize_runtime_tuple(runtimes: JaxFlatRuntime | Iterable[JaxFlatRuntime] | None) -> tuple[JaxFlatRuntime, ...]:
    if runtimes is None:
        return ()
    if isinstance(runtimes, JaxFlatRuntime):
        return (runtimes,)
    return tuple(runtimes)


def _external_cache_inputs(
    runtimes: JaxFlatRuntime | Iterable[JaxFlatRuntime] | None,
) -> tuple[dict[tuple[Any, ...], str], dict[str, np.ndarray]]:
    names_by_key: dict[tuple[Any, ...], str] = {}
    values_by_name: dict[str, np.ndarray] = {}
    for runtime_idx, runtime in enumerate(_normalize_runtime_tuple(runtimes)):
        cached_values = runtime.get_cached_values()
        missing = [node_id for node_id in runtime.program.cache_nodes if node_id not in cached_values]
        if missing:
            raise ValueError(
                "runtimes passed to compile_formula must have materialized cache values; "
                f"runtime {runtime_idx} is missing cache node(s) {missing}. Run run_batch first."
            )
        for node_id, expr_key in zip(runtime.program.cache_nodes, runtime.program.cache_expr_keys):
            if expr_key in names_by_key:
                continue
            name = f"__cache_runtime_{runtime_idx}_node_{node_id}"
            names_by_key[expr_key] = name
            values_by_name[name] = cached_values[node_id]
    return names_by_key, values_by_name


# --- Public compile entrypoint ---


def compile_formula(
    formula: str | Expr,
    dsl_registry: DSLFunctionRegistry | None = None,
    cpp: bool = True,
    metadata: MetadataConfig | dict | None = None,
    type_relations=(),
    runtimes: JaxFlatRuntime | Iterable[JaxFlatRuntime] | None = None,
    workers: int | None = None,
) -> JaxFlatRuntime:
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0):
        raise ValueError("workers must be a positive integer or None")
    expr = parse_formula(formula) if isinstance(formula, str) else formula
    expr = _normalize_static_jax_flat_kwargs(expr)
    expr = _expand_dsl(expr, dsl_registry or DEFAULT_DSL_REGISTRY)
    expr = _normalize_static_jax_flat_kwargs(expr)
    # metadata_config = MetadataConfig.from_value(metadata, type_relations=type_relations)
    # formula_metadata = analyze_formula_metadata(expr, metadata_config)
    external_cache_names, external_cache_values = _external_cache_inputs(runtimes)
    nodes: list[DagNode] = []
    memo: dict[tuple[Any, ...], int] = {}
    input_names: list[str] = []
    out = _compile_node(expr, memo, nodes, input_names, external_cache_names)
    node_tuple = tuple(nodes)
    layout = _build_state_layout(node_tuple)
    cache_nodes = tuple(idx for idx, node in enumerate(node_tuple) if isinstance(node.op, CacheOp))
    cache_key_by_node = {node_id: key for key, node_id in memo.items() if key[0] == "call" and key[1] == "cache"}
    cache_expr_keys = tuple(cache_key_by_node[idx][2][0] for idx in cache_nodes)
    runtime = JaxFlatRuntime(
        program=StreamingProgram(
            nodes=node_tuple,
            outputs=(out,),
            input_names=tuple(input_names),
            state_layout=layout,
            metadata=None,
            cache_nodes=cache_nodes,
            cache_expr_keys=cache_expr_keys,
            external_cache_inputs=external_cache_values or None,
        ),
        cpp=cpp,
        cpp_workers=workers,
    )
    if cpp:
        import warnings
        from trading_dsl_engine.jax_flat.engine_cpp import explain_cpp_plan

        missing = explain_cpp_plan(runtime.program).missing_cpp_functions
        if missing:
            warnings.warn(
                "C++ jax_flat lowering requires native implementations for DSL function(s): "
                + ", ".join(missing)
                + "; those nodes will run as JAX islands. Call runtime.explain() for the lowered plan.",
                RuntimeWarning,
                stacklevel=2,
            )
    return runtime
