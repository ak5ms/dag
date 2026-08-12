from __future__ import annotations

from dataclasses import fields, is_dataclass
from io import BytesIO

from trading_dsl_engine.base.parser import (
    Call,
    Expr,
    Identifier,
    KeyTuple,
    Number,
    String,
    Universe,
)
from trading_dsl_engine.ir.ops import CustomCallOp, GroupByOp, InputOp, LiteralOp, NaryOp
from trading_dsl_engine.ir.program import Program
from trading_dsl_engine.ir.types import ValueType


_SUPPORTED_BACKENDS = {"pydot"}
_SUPPORTED_RANKDIRS = {"LR", "RL", "TB", "BT"}


def _short_repr(value: object, limit: int = 72) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _expr_details(node: Expr) -> tuple[str, str, tuple[tuple[str | None, Expr], ...]]:
    from trading_dsl_engine.base.custom import StatelessCall
    from trading_dsl_engine.base.keys import Key

    if isinstance(node, Call):
        children: list[tuple[str | None, Expr]] = []
        positional_labels = len(node.args) > 1
        children.extend(
            (str(index) if positional_labels else None, child)
            for index, child in enumerate(node.args)
        )
        children.extend((name, child) for name, child in node.kwargs)
        return node.fn, "ellipse", tuple(children)
    if isinstance(node, Identifier):
        return f"input\n{node.name}", "box", ()
    if isinstance(node, Number):
        return f"literal\n{_short_repr(node.value)}", "box", ()
    if isinstance(node, String):
        return f"string\n{_short_repr(node.value)}", "box", ()
    if isinstance(node, Universe):
        return f"univ\n{_short_repr(node.groups)}", "box", ()
    if isinstance(node, KeyTuple):
        return (
            "key tuple",
            "ellipse",
            tuple((str(index), child) for index, child in enumerate(node.items)),
        )
    if isinstance(node, Key):
        metadata = []
        if node.num_keys is not None:
            metadata.append(f"num_keys={node.num_keys}")
        if node.offset:
            metadata.append(f"offset={node.offset}")
        if node.row_scalar is not None:
            metadata.append(f"row_scalar={node.row_scalar}")
        if node.dtype is not None:
            metadata.append(f"dtype={node.dtype}")
        if node.monotonic:
            metadata.append("monotonic=True")
        label = "Key" + ("\n" + ", ".join(metadata) if metadata else "")
        return label, "ellipse", ((None, node.expr),)
    if isinstance(node, StatelessCall):
        name = node.cpp_name or node.name or getattr(node.fn, "__name__", "stateless")
        positional_labels = len(node.args) > 1
        return (
            f"stateless\n{name}",
            "ellipse",
            tuple(
                (str(index) if positional_labels else None, child)
                for index, child in enumerate(node.args)
            ),
        )
    raise TypeError(f"unsupported expression node for plotting: {type(node).__name__}")


def _expr_to_pydot(root: Expr, *, rankdir: str):
    import pydot

    graph = pydot.Dot("formula", graph_type="digraph", rankdir=rankdir)
    node_ids: dict[int, str] = {}

    def visit(node: Expr) -> str:
        identity = id(node)
        existing = node_ids.get(identity)
        if existing is not None:
            return existing
        node_id = f"n{len(node_ids)}"
        node_ids[identity] = node_id
        label, shape, children = _expr_details(node)
        graph.add_node(pydot.Node(node_id, label=label, shape=shape))
        for edge_label, child in children:
            child_id = visit(child)
            attrs = {} if edge_label is None else {"label": edge_label}
            graph.add_edge(pydot.Edge(child_id, node_id, **attrs))
        return node_id

    root_id = visit(root)
    graph.get_node(root_id)[0].set_peripheries("2")
    return graph


def _simple_detail(value: object, *, depth: int = 0) -> bool:
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if depth >= 2 or not isinstance(value, (tuple, list)) or len(value) > 6:
        return False
    return all(_simple_detail(item, depth=depth + 1) for item in value)


def _op_label(op: object) -> str:
    if isinstance(op, InputOp):
        return f"input\n{op.name}"
    if isinstance(op, LiteralOp):
        return f"literal\n{_short_repr(op.value)}"
    if isinstance(op, (NaryOp, CustomCallOp)):
        return op.name
    if isinstance(op, GroupByOp):
        details = [f"dynamic_keys={op.n_dynamic_keys}"]
        if op.static_groups is not None:
            details.append(f"static_groups={len(op.static_groups)}")
        if op.capacity is not None:
            details.append(f"capacity={op.capacity}")
        if op.hash_capacity is not None:
            details.append(f"hash_capacity={op.hash_capacity}")
        return "GroupBy\n" + "\n".join(details)

    type_name = type(op).__name__
    title = type_name[:-2] if type_name.endswith("Op") else type_name
    details: list[str] = []
    subscripts = getattr(op, "subscripts", None)
    if isinstance(subscripts, str):
        details.append(f"subscripts={subscripts!r}")
    if is_dataclass(op):
        for field in fields(op):
            if field.name in {"inner_program", "spec", "input_index", "name", "arity"}:
                continue
            value = getattr(op, field.name)
            if _simple_detail(value):
                details.append(f"{field.name}={_short_repr(value, 48)}")
            if len(details) >= 6:
                break
    return title + ("\n" + "\n".join(details) if details else "")


def _value_type_label(value_type: ValueType) -> str:
    if value_type.shape is None:
        return f"{value_type.dtype} object"
    shape = value_type.logical_shape
    if not shape:
        shape_text = "()"
    else:
        pieces = ["?" if extent is None else str(extent) for extent in shape]
        shape_text = "(" + ", ".join(pieces) + ("," if len(pieces) == 1 else "") + ")"
    return f"{value_type.dtype} {shape_text}"


def _program_to_pydot(program: Program, *, rankdir: str):
    import pydot

    graph = pydot.Dot("formula_ir", graph_type="digraph", rankdir=rankdir)

    def add_program(
        current: Program,
        *,
        prefix: str,
        container,
        mark_outputs: bool,
    ) -> None:
        outputs = set(current.outputs) if mark_outputs else set()
        for node_id, node in enumerate(current.nodes):
            attrs: dict[str, str] = {
                "label": f"{node_id}: {_op_label(node.op)}\n{_value_type_label(node.value_type)}",
                "shape": "box" if isinstance(node.op, (InputOp, LiteralOp)) else "ellipse",
            }
            if node_id in outputs:
                attrs["peripheries"] = "2"
            container.add_node(pydot.Node(f"{prefix}n{node_id}", **attrs))

        for node_id, node in enumerate(current.nodes):
            target = f"{prefix}n{node_id}"
            if not isinstance(node.op, GroupByOp):
                label_edges = len(node.child_ids) > 1
                for child_index, child_id in enumerate(node.child_ids):
                    attrs = {"label": str(child_index)} if label_edges else {}
                    graph.add_edge(pydot.Edge(f"{prefix}n{child_id}", target, **attrs))
                continue

            n_keys = node.op.n_dynamic_keys
            for key_index, child_id in enumerate(node.child_ids[:n_keys]):
                graph.add_edge(
                    pydot.Edge(f"{prefix}n{child_id}", target, label=f"key {key_index}")
                )

            inner = node.op.inner_program
            inner_prefix = f"{prefix}g{node_id}_"
            cluster = pydot.Cluster(
                f"{prefix}groupby_{node_id}",
                label=f"GroupBy {node_id} RHS",
            )
            add_program(inner, prefix=inner_prefix, container=cluster, mark_outputs=False)
            graph.add_subgraph(cluster)

            inner_inputs = {
                inner_node.op.input_index: inner_id
                for inner_id, inner_node in enumerate(inner.nodes)
                if isinstance(inner_node.op, InputOp)
            }
            for input_index, inner_id in inner_inputs.items():
                outer_child_index = n_keys + input_index
                if outer_child_index >= len(node.child_ids):
                    continue
                source = f"{prefix}n{node.child_ids[outer_child_index]}"
                input_target = f"{inner_prefix}n{inner_id}"
                role = "self" if input_index == 0 else f"capture {input_index - 1}"
                graph.add_edge(pydot.Edge(source, input_target, label=role))

            for inner_output in inner.outputs:
                graph.add_edge(
                    pydot.Edge(f"{inner_prefix}n{inner_output}", target, label="rhs")
                )

    add_program(program, prefix="", container=graph, mark_outputs=True)
    return graph


def _show_pydot(graph, *, figsize: tuple[float, float] | None) -> None:
    import matplotlib.pyplot as plt

    try:
        png = graph.create_png()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "pydot rendering requires the Graphviz 'dot' executable on PATH"
        ) from exc
    image = plt.imread(BytesIO(png), format="png")
    if figsize is None:
        height, width = image.shape[:2]
        figsize = (
            max(6.0, min(20.0, width / 120.0)),
            max(4.0, min(20.0, height / 120.0)),
        )
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.set_axis_off()
    fig.tight_layout(pad=0.1)
    plt.show()


def plot(
    value: Expr | Program,
    backend: str = "pydot",
    *,
    show: bool = True,
    rankdir: str = "LR",
    figsize: tuple[float, float] | None = None,
):
    """Plot a formula AST or neutral IR DAG and return the backend graph object."""
    backend_name = str(backend).lower()
    if backend_name not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported plot backend {backend!r}; expected one of {sorted(_SUPPORTED_BACKENDS)}"
        )
    direction = str(rankdir).upper()
    if direction not in _SUPPORTED_RANKDIRS:
        raise ValueError(
            f"unsupported rankdir {rankdir!r}; expected one of {sorted(_SUPPORTED_RANKDIRS)}"
        )
    if isinstance(value, Expr):
        graph = _expr_to_pydot(value, rankdir=direction)
    elif isinstance(value, Program):
        graph = _program_to_pydot(value, rankdir=direction)
    else:
        raise TypeError(f"plot expects Expr or Program, got {type(value).__name__}")
    if show:
        _show_pydot(graph, figsize=figsize)
    return graph


__all__ = ["plot"]
