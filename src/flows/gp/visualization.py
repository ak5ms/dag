from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
import tempfile
import webbrowser

from deap import gp

from flows.gp.types import ExprValue, StaticValue


@dataclass(frozen=True)
class GPGraphNode:
    id: str
    label: str
    kind: str
    search_text: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class GPGraphEdge:
    source: str
    target: str
    role: str


@dataclass(frozen=True)
class GPTypeRelation:
    source: str
    target: str
    operators: tuple[str, ...]


@dataclass(frozen=True)
class GPGraphModel:
    nodes: tuple[GPGraphNode, ...]
    edges: tuple[GPGraphEdge, ...]
    type_relations: tuple[GPTypeRelation, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "kind": node.kind,
                    "search_text": node.search_text,
                    "metadata": node.metadata,
                }
                for node in self.nodes
            ],
            "edges": [
                {"source": edge.source, "target": edge.target, "role": edge.role}
                for edge in self.edges
            ],
            "type_relations": [
                {
                    "source": relation.source,
                    "target": relation.target,
                    "operators": list(relation.operators),
                }
                for relation in self.type_relations
            ],
        }


def _terminal_value(pset, terminal: gp.Terminal):
    value = terminal.value
    if isinstance(value, str) and value in pset.context:
        return pset.context[value]
    return value


def _terminal_text(value: object) -> str:
    if isinstance(value, StaticValue):
        return repr(value.value)
    if isinstance(value, ExprValue):
        return str(value.expr)
    return repr(value)


def build_gp_graph(pset: gp.PrimitiveSetTyped) -> GPGraphModel:
    """Build a searchable type/operator/terminal graph from a DEAP pset."""

    primitives = {
        name: value
        for name, value in pset.mapping.items()
        if isinstance(value, gp.Primitive)
    }
    terminals = {
        name: value
        for name, value in pset.mapping.items()
        if isinstance(value, gp.Terminal)
    }
    types: set[type] = {pset.ret}
    for primitive in primitives.values():
        types.add(primitive.ret)
        types.update(primitive.args)
    for terminal in terminals.values():
        types.add(terminal.ret)

    nodes: list[GPGraphNode] = []
    for type_ in sorted(types, key=lambda value: value.__name__):
        name = type_.__name__
        nodes.append(
            GPGraphNode(
                id=f"type:{name}",
                label=name,
                kind="type",
                search_text=f"type {name}".lower(),
                metadata={"type": name},
            )
        )

    family_of = getattr(pset, "gp_primitive_family", {})
    section_of = getattr(pset, "gp_primitive_section", {})
    relation_ops: dict[tuple[str, str], set[str]] = {}
    edges: list[GPGraphEdge] = []
    for name, primitive in sorted(primitives.items()):
        family = family_of.get(name, name)
        section = section_of.get(name, "")
        args = tuple(type_.__name__ for type_ in primitive.args)
        ret = primitive.ret.__name__
        signature = f"{family}({', '.join(args)}) -> {ret}"
        operator_id = f"operator:{name}"
        nodes.append(
            GPGraphNode(
                id=operator_id,
                label=family,
                kind="operator",
                search_text=(
                    f"operator primitive {name} family {family} section {section} "
                    f"{signature}"
                ).lower(),
                metadata={
                    "primitive": name,
                    "family": family,
                    "section": section,
                    "args": args,
                    "return": ret,
                    "signature": signature,
                },
            )
        )
        for index, arg in enumerate(primitive.args):
            source = f"type:{arg.__name__}"
            edges.append(GPGraphEdge(source, operator_id, f"arg {index}"))
            relation_ops.setdefault((source, f"type:{ret}"), set()).add(family)
        edges.append(GPGraphEdge(operator_id, f"type:{ret}", "returns"))

    for name, terminal in sorted(terminals.items()):
        value = _terminal_value(pset, terminal)
        value_text = _terminal_text(value)
        type_name = terminal.ret.__name__
        terminal_id = f"terminal:{name}"
        nodes.append(
            GPGraphNode(
                id=terminal_id,
                label=name,
                kind="terminal",
                search_text=f"terminal {name} {type_name} {value_text}".lower(),
                metadata={
                    "terminal": name,
                    "type": type_name,
                    "value": value_text,
                },
            )
        )
        edges.append(GPGraphEdge(terminal_id, f"type:{type_name}", "has type"))

    relations = tuple(
        GPTypeRelation(source, target, tuple(sorted(operators)))
        for (source, target), operators in sorted(relation_ops.items())
    )
    return GPGraphModel(tuple(nodes), tuple(edges), relations)


def filter_gp_graph(
    model: GPGraphModel,
    query: str,
    *,
    include_neighbors: bool = True,
) -> GPGraphModel:
    """Filter graph nodes by search text, optionally retaining one-hop context."""

    needle = query.strip().lower()
    if not needle:
        return model
    matched = {node.id for node in model.nodes if needle in node.search_text}
    visible = set(matched)
    if include_neighbors:
        for edge in model.edges:
            if edge.source in matched or edge.target in matched:
                visible.update((edge.source, edge.target))
    nodes = tuple(node for node in model.nodes if node.id in visible)
    edges = tuple(
        edge
        for edge in model.edges
        if edge.source in visible and edge.target in visible
    )
    relations = tuple(
        relation
        for relation in model.type_relations
        if relation.source in visible and relation.target in visible
    )
    return GPGraphModel(nodes, edges, relations)


def gp_explorer_html(
    pset: gp.PrimitiveSetTyped,
    *,
    include_plotlyjs: bool = True,
) -> str:
    """Return a standalone searchable/clickable Plotly GP graph explorer."""

    model_json = json.dumps(build_gp_graph(pset).as_dict(), separators=(",", ":")).replace("</", "<\\/")
    if include_plotlyjs:
        from plotly.offline import get_plotlyjs

        plotly_script = f"<script>{get_plotlyjs()}</script>"
    else:
        plotly_script = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>GP graph explorer</title>
{plotly_script}
<style>
body {{ font-family: system-ui, sans-serif; margin: 0; }}
#controls {{ display: flex; gap: .6rem; padding: .7rem; border-bottom: 1px solid #ddd; }}
#gp-search {{ flex: 1; min-width: 16rem; padding: .45rem .6rem; }}
#gp-reset {{ padding: .45rem .8rem; }}
#gp-status {{ padding: .45rem .2rem; min-width: 14rem; color: #555; }}
#gp-graph {{ width: 100vw; height: calc(100vh - 58px); }}
</style>
</head>
<body>
<div id="controls">
  <input id="gp-search" type="search" placeholder="Search types, operators, terminals, sections…">
  <button id="gp-reset" type="button">Reset</button>
  <span id="gp-status"></span>
</div>
<div id="gp-graph"></div>
<script>
const MODEL = {model_json};
const BY_ID = Object.fromEntries(MODEL.nodes.map(node => [node.id, node]));
let focus = null;

function neighbors(ids, rounds=1) {{
  const visible = new Set(ids);
  for (let round = 0; round < rounds; round++) {{
    const frontier = new Set(visible);
    for (const edge of MODEL.edges) {{
      if (frontier.has(edge.source) || frontier.has(edge.target)) {{
        visible.add(edge.source); visible.add(edge.target);
      }}
    }}
  }}
  return visible;
}}

function visibleIds() {{
  const query = document.getElementById('gp-search').value.trim().toLowerCase();
  if (query) {{
    const hits = MODEL.nodes.filter(node => node.search_text.includes(query)).map(node => node.id);
    return neighbors(hits, 1);
  }}
  if (focus) return neighbors([focus], 2);
  return new Set(MODEL.nodes.filter(node => node.kind === 'type').map(node => node.id));
}}

function positions(nodes) {{
  const groups = {{type: [], operator: [], terminal: []}};
  for (const node of nodes) groups[node.kind].push(node);
  const y = {{type: 1, operator: 0, terminal: -1}};
  const result = {{}};
  for (const kind of ['type', 'operator', 'terminal']) {{
    const values = groups[kind];
    values.forEach((node, index) => {{
      result[node.id] = {{x: index - (values.length - 1) / 2, y: y[kind]}};
    }});
  }}
  return result;
}}

function render() {{
  const ids = visibleIds();
  const nodes = MODEL.nodes.filter(node => ids.has(node.id));
  const pos = positions(nodes);
  const query = document.getElementById('gp-search').value.trim();
  const overview = !focus && !query;
  const selectedEdges = overview ? MODEL.type_relations : MODEL.edges;
  const edgeX = [], edgeY = [];
  for (const edge of selectedEdges) {{
    if (!ids.has(edge.source) || !ids.has(edge.target)) continue;
    edgeX.push(pos[edge.source].x, pos[edge.target].x, null);
    edgeY.push(pos[edge.source].y, pos[edge.target].y, null);
  }}
  const edgeTrace = {{
    type: 'scatter', mode: 'lines', x: edgeX, y: edgeY,
    hoverinfo: 'skip', line: {{width: 1}}, showlegend: false
  }};
  const symbols = {{type: 'circle', operator: 'diamond', terminal: 'square'}};
  const nodeTrace = {{
    type: 'scatter', mode: 'markers+text',
    x: nodes.map(node => pos[node.id].x),
    y: nodes.map(node => pos[node.id].y),
    text: nodes.map(node => node.label),
    textposition: 'top center',
    customdata: nodes.map(node => node.id),
    marker: {{size: nodes.map(node => node.kind === 'type' ? 18 : 13), symbol: nodes.map(node => symbols[node.kind])}},
    hovertemplate: nodes.map(node => Object.entries(node.metadata).map(([k,v]) => `<b>${{k}}</b>: ${{v}}`).join('<br>') + '<extra></extra>'),
    showlegend: false
  }};
  const layout = {{
    title: overview ? 'GP type relations — click a type to drill down' : 'GP graph drill-down',
    margin: {{l: 30, r: 30, t: 55, b: 30}},
    xaxis: {{visible: false}}, yaxis: {{visible: false, range: [-1.45, 1.45]}},
    hovermode: 'closest'
  }};
  Plotly.react('gp-graph', [edgeTrace, nodeTrace], layout, {{responsive: true}});
  document.getElementById('gp-status').textContent = `${{nodes.length}} nodes` + (focus ? ` · focused: ${{BY_ID[focus].label}}` : '');
}}

document.getElementById('gp-search').addEventListener('input', () => {{ focus = null; render(); }});
document.getElementById('gp-reset').addEventListener('click', () => {{
  focus = null; document.getElementById('gp-search').value = ''; render();
}});
render();
document.getElementById('gp-graph').on('plotly_click', event => {{
  const id = event.points?.[0]?.customdata;
  if (id && BY_ID[id]) {{ focus = id; document.getElementById('gp-search').value = ''; render(); }}
}});
</script>
</body>
</html>"""


def explore_gp(
    pset: gp.PrimitiveSetTyped,
    path: str | Path | None = None,
    *,
    open_browser: bool = True,
) -> Path:
    """Write the interactive Plotly explorer and optionally open it in a browser."""

    if path is None:
        directory = Path(tempfile.mkdtemp(prefix="gp-explorer-"))
        output = directory / "index.html"
    else:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(gp_explorer_html(pset), encoding="utf-8")
    if open_browser:
        webbrowser.open(output.as_uri())
    return output


__all__ = [
    "GPGraphEdge",
    "GPGraphModel",
    "GPGraphNode",
    "GPTypeRelation",
    "build_gp_graph",
    "explore_gp",
    "filter_gp_graph",
    "gp_explorer_html",
]
