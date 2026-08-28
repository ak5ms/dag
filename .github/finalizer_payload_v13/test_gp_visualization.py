from __future__ import annotations

from flows.gp import explore_pset, make_pset
from flows.gp.types import IntegerParam, NonNegativeFloat, NonNegativeInt, ScalarNumber
from flows.gp.visualization import gp_graph_data


def _terminal_values(pset, type_):
    values = set()
    for terminal in pset.terminals[type_]:
        value = getattr(terminal, "value", None)
        if isinstance(value, str) and value in pset.context:
            value = pset.context[value]
        values.add(getattr(value, "value", value))
    return values


def test_default_pset_has_richer_exact_static_numeric_terminals():
    pset = make_pset()
    assert {-1, 0, 1, 30, 1440} <= _terminal_values(pset, IntegerParam)
    assert {0, 1, 30, 1440} <= _terminal_values(pset, NonNegativeInt)
    assert {0.0, 0.001, 0.5, 10.0} <= _terminal_values(pset, NonNegativeFloat)
    assert {-1.0, 0.0, 0.5, 1.0, 30.0} <= _terminal_values(pset, ScalarNumber)


def test_gp_graph_explorer_has_relations_search_and_click_drilldown():
    pset = make_pset()
    nodes, edges = gp_graph_data(pset)
    assert {node.kind for node in nodes} == {"type", "operator", "terminal"}
    assert any(edge.label.startswith("arg") for edge in edges)
    assert any(edge.label == "returns" for edge in edges)
    assert any(edge.label == "has type" for edge in edges)
    explorer = explore_pset(pset)
    rendered = explorer.to_html(full_html=False, include_plotlyjs=False)
    assert 'type="search"' in rendered
    assert "Search terminals, operators, and types" in rendered
    assert "plotly_click" in rendered
    assert "Direct relations" in rendered
