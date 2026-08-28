from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path
import tempfile
from typing import Any
import uuid
import webbrowser


@dataclass(frozen=True)
class GPGraphNode:
    id: str
    label: str
    kind: str
    detail: str
    search_text: str


@dataclass(frozen=True)
class GPGraphEdge:
    source: str
    target: str
    label: str


def _type_name(value: Any) -> str:
    return getattr(value, "__name__", str(value))


def _short(value: Any, limit: int = 150) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<{type(value).__name__}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _terminal_value(pset: Any, terminal: Any) -> Any:
    value = getattr(terminal, "value", terminal)
    context = getattr(pset, "context", {})
    if isinstance(value, str) and isinstance(context, dict) and value in context:
        return context[value]
    return value


def gp_graph_data(pset: Any) -> tuple[tuple[GPGraphNode, ...], tuple[GPGraphEdge, ...]]:
    primitives_by_type = getattr(pset, "primitives", None)
    terminals_by_type = getattr(pset, "terminals", None)
    if not isinstance(primitives_by_type, dict) or not isinstance(terminals_by_type, dict):
        raise TypeError("expected a DEAP PrimitiveSetTyped-like object")

    types = set(primitives_by_type) | set(terminals_by_type)
    primitives = {}
    terminals = {}
    for values in primitives_by_type.values():
        for primitive in values:
            key = (
                getattr(primitive, "name", None),
                tuple(getattr(primitive, "args", ())),
                getattr(primitive, "ret", None),
            )
            primitives.setdefault(key, primitive)
            types.update(getattr(primitive, "args", ()))
            if getattr(primitive, "ret", None) is not None:
                types.add(primitive.ret)
    for values in terminals_by_type.values():
        for terminal in values:
            key = (
                getattr(terminal, "name", None),
                getattr(terminal, "ret", None),
                _short(_terminal_value(pset, terminal)),
            )
            terminals.setdefault(key, terminal)
            if getattr(terminal, "ret", None) is not None:
                types.add(terminal.ret)

    nodes = []
    edges = []
    type_ids = {}
    for type_ in sorted(types, key=lambda item: _type_name(item).lower()):
        label = _type_name(type_)
        node_id = f"type:{getattr(type_, '__module__', '')}.{label}"
        type_ids[type_] = node_id
        nodes.append(GPGraphNode(node_id, label, "type", f"Type: {label}", f"type {label}".lower()))

    families = getattr(pset, "gp_primitive_family", {})
    sections = getattr(pset, "gp_primitive_section", {})
    for primitive in sorted(primitives.values(), key=lambda item: str(item.name).lower()):
        name = str(primitive.name)
        args = tuple(getattr(primitive, "args", ()))
        ret = getattr(primitive, "ret", None)
        family = families.get(name, name.split("__", 1)[0])
        section = sections.get(name, "")
        signature = f"({', '.join(_type_name(arg) for arg in args)}) -> {_type_name(ret)}"
        detail = f"Operator: {name}\nFamily: {family}\nSignature: {signature}"
        if section:
            detail += f"\nSection: {section}"
        node_id = f"operator:{name}"
        nodes.append(
            GPGraphNode(
                node_id,
                name,
                "operator",
                detail,
                f"operator primitive {name} {family} {section} {signature}".lower(),
            )
        )
        for index, arg in enumerate(args):
            if arg in type_ids:
                edges.append(GPGraphEdge(type_ids[arg], node_id, f"arg {index}"))
        if ret in type_ids:
            edges.append(GPGraphEdge(node_id, type_ids[ret], "returns"))

    for index, terminal in enumerate(
        sorted(terminals.values(), key=lambda item: str(getattr(item, "name", "")).lower())
    ):
        name = str(getattr(terminal, "name", f"terminal_{index}"))
        ret = getattr(terminal, "ret", None)
        value = _terminal_value(pset, terminal)
        node_id = f"terminal:{index}:{name}"
        detail = f"Terminal: {name}\nType: {_type_name(ret)}\nValue: {_short(value)}"
        nodes.append(
            GPGraphNode(
                node_id,
                name,
                "terminal",
                detail,
                f"terminal {name} {_type_name(ret)} {_short(value)}".lower(),
            )
        )
        if ret in type_ids:
            edges.append(GPGraphEdge(node_id, type_ids[ret], "has type"))
    return tuple(nodes), tuple(edges)


def _positions(nodes: tuple[GPGraphNode, ...]) -> dict[str, tuple[float, float]]:
    result = {}
    x_by_kind = {"terminal": 0.0, "type": 1.0, "operator": 2.0}
    for kind in ("terminal", "type", "operator"):
        values = sorted((node for node in nodes if node.kind == kind), key=lambda node: node.label.lower())
        for index, node in enumerate(values):
            y = 0.0 if len(values) < 2 else 1.0 - 2.0 * index / (len(values) - 1)
            result[node.id] = (x_by_kind[kind], y)
    return result


class GPGraphExplorer:
    """Interactive Plotly type/operator/terminal explorer for a GP pset."""

    def __init__(self, pset: Any, *, title: str = "GP grammar explorer") -> None:
        self.pset = pset
        self.title = title
        self.nodes, self.edges = gp_graph_data(pset)
        self.positions = _positions(self.nodes)
        self.div_id = f"gp-grammar-{uuid.uuid4().hex}"
        self.figure = self._figure()

    def _figure(self):
        import plotly.graph_objects as go

        edge_x = []
        edge_y = []
        for edge in self.edges:
            source = self.positions[edge.source]
            target = self.positions[edge.target]
            edge_x.extend((source[0], target[0], None))
            edge_y.extend((source[1], target[1], None))
        edges = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="skip", line={"width": 0.7})
        symbols = {"terminal": "square", "type": "diamond", "operator": "circle"}
        sizes = {"terminal": 9, "type": 14, "operator": 10}
        nodes = go.Scatter(
            x=[self.positions[node.id][0] for node in self.nodes],
            y=[self.positions[node.id][1] for node in self.nodes],
            mode="markers+text",
            text=[node.label for node in self.nodes],
            textposition="middle right",
            customdata=[node.id for node in self.nodes],
            hovertext=[node.detail.replace("\n", "<br>") for node in self.nodes],
            hovertemplate="%{hovertext}<extra></extra>",
            marker={
                "symbol": [symbols[node.kind] for node in self.nodes],
                "size": [sizes[node.kind] for node in self.nodes],
                "line": {"width": 0.5},
            },
        )
        figure = go.Figure((edges, nodes))
        figure.update_layout(
            title=self.title,
            showlegend=False,
            hovermode="closest",
            dragmode="pan",
            margin={"l": 25, "r": 230, "t": 55, "b": 25},
            xaxis={"visible": False, "range": (-0.15, 2.75)},
            yaxis={"visible": False},
            annotations=[
                {"x": 0, "y": 1.08, "text": "Terminals", "showarrow": False},
                {"x": 1, "y": 1.08, "text": "Types", "showarrow": False},
                {"x": 2, "y": 1.08, "text": "Operators", "showarrow": False},
            ],
        )
        return figure

    def to_html(self, *, full_html: bool = True, include_plotlyjs: str | bool = "cdn") -> str:
        from plotly.io import to_html

        plot = to_html(
            self.figure,
            full_html=False,
            include_plotlyjs=include_plotlyjs,
            div_id=self.div_id,
            config={"scrollZoom": True, "displaylogo": False, "responsive": True},
        )
        nodes_json = json.dumps([asdict(node) for node in self.nodes])
        edges_json = json.dumps([asdict(edge) for edge in self.edges])
        positions_json = json.dumps({key: list(value) for key, value in self.positions.items()})
        controls = f'''<div class="gp-controls"><label>Search terminals, operators, and types
<input id="{self.div_id}-search" type="search" placeholder="rolling, PriceRow, field_close"></label>
<button id="{self.div_id}-reset" type="button">Reset</button><span id="{self.div_id}-count"></span></div>
<div class="gp-layout"><div>{plot}</div><pre id="{self.div_id}-details">Click a node to inspect it and its direct relations.</pre></div>'''
        script = r'''<script>(() => {
const graph = document.getElementById("__DIV__");
const search = document.getElementById("__DIV__-search");
const reset = document.getElementById("__DIV__-reset");
const details = document.getElementById("__DIV__-details");
const count = document.getElementById("__DIV__-count");
const nodes = __NODES__, edges = __EDGES__, positions = __POSITIONS__;
const byId = new Map(nodes.map(node => [node.id, node]));
const neighbors = new Map(nodes.map(node => [node.id, new Set()]));
edges.forEach(edge => { neighbors.get(edge.source).add(edge.target); neighbors.get(edge.target).add(edge.source); });
function apply(visible, selected=null) {
  const opacity = nodes.map(node => visible.has(node.id) ? 1 : 0.04);
  const size = nodes.map(node => node.id === selected ? 20 : (node.kind === 'type' ? 14 : (node.kind === 'operator' ? 10 : 9)));
  const x=[], y=[];
  edges.forEach(edge => { if (visible.has(edge.source) && visible.has(edge.target)) { x.push(positions[edge.source][0], positions[edge.target][0], null); y.push(positions[edge.source][1], positions[edge.target][1], null); }});
  Plotly.restyle(graph, {x:[x], y:[y]}, [0]);
  Plotly.restyle(graph, {'marker.opacity':[opacity], 'marker.size':[size]}, [1]);
  count.textContent = `${visible.size} / ${nodes.length} nodes`;
}
const all = () => new Set(nodes.map(node => node.id));
function filter() {
  const query = search.value.trim().toLowerCase();
  if (!query) { details.textContent='Click a node to inspect it and its direct relations.'; apply(all()); return; }
  const matches = nodes.filter(node => node.search_text.includes(query));
  const visible = new Set();
  matches.forEach(node => { visible.add(node.id); neighbors.get(node.id).forEach(id => visible.add(id)); });
  details.textContent = matches.length ? `Search: ${query}\nDirect matches: ${matches.length}\n` + matches.slice(0,30).map(node => `${node.kind}: ${node.label}`).join('\n') : `No match for “${query}”.`;
  apply(visible);
}
search.addEventListener('input', filter);
reset.addEventListener('click', () => { search.value=''; filter(); });
graph.on('plotly_click', event => {
  const point = event.points && event.points.find(item => item.curveNumber === 1);
  if (!point) return;
  const id=point.customdata, node=byId.get(id), visible=new Set([id, ...neighbors.get(id)]);
  details.textContent=node.detail+'\n\nDirect relations:\n'+[...neighbors.get(id)].map(other => `- ${byId.get(other).kind}: ${byId.get(other).label}`).join('\n');
  apply(visible,id);
});
apply(all());
})()</script>'''.replaceAll('__DIV__', self.div_id).replace('__NODES__', nodes_json).replace('__EDGES__', edges_json).replace('__POSITIONS__', positions_json)
        style = '''<style>.gp-controls{display:flex;gap:.6rem;align-items:center;margin:.4rem 0}.gp-controls input{min-width:24rem;padding:.4rem}.gp-layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(16rem,24rem);gap:.8rem}.gp-layout pre{white-space:pre-wrap;overflow:auto;max-height:78vh;padding:.7rem;border:1px solid #bbb}@media(max-width:850px){.gp-layout{grid-template-columns:1fr}}</style>'''
        body = style + controls + script
        if not full_html:
            return body
        return f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(self.title)}</title></head><body>{body}</body></html>'

    def write_html(self, path: str | Path, *, include_plotlyjs: str | bool = True) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_html(include_plotlyjs=include_plotlyjs), encoding="utf-8")
        return destination

    def show(self) -> Path:
        destination = Path(tempfile.mkdtemp(prefix="gp-grammar-")) / "index.html"
        self.write_html(destination, include_plotlyjs=True)
        webbrowser.open(destination.resolve().as_uri())
        return destination

    def _repr_html_(self) -> str:
        return self.to_html(full_html=False, include_plotlyjs="cdn")


def explore_pset(pset: Any, *, show: bool = False, title: str = "GP grammar explorer") -> GPGraphExplorer:
    explorer = GPGraphExplorer(pset, title=title)
    if show:
        explorer.show()
    return explorer


plot_pset = explore_pset
visualize_pset = explore_pset
plot_gp_graph = explore_pset
explore_gp = explore_pset

__all__ = [
    "GPGraphEdge", "GPGraphExplorer", "GPGraphNode", "explore_gp",
    "explore_pset", "gp_graph_data", "plot_gp_graph", "plot_pset",
    "visualize_pset",
]
