"""Descriptor-driven lane graph discovery and native-family factories.

Discovery is deliberately independent of any concrete operator family.  It
compares complete branch topology while allowing descriptors to identify static
parameters that form lanes.  Native factories then probe the discovered graph;
adding an executor never adds another pattern conditional to the public runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from trading_dsl_engine.cpp_new.registry import descriptor


@dataclass(frozen=True)
class LaneGraph:
    output_nodes: tuple[int, ...]
    signatures: tuple[tuple, ...]
    source_inputs: tuple[int, ...]

    @property
    def topology(self) -> tuple:
        return self.signatures[0]


def _semantic_parameters(node) -> tuple[tuple[str, str], ...]:
    spec = descriptor(node.opcode)
    ignored = {*spec.lane_parameters, "scratch_offset"}
    return tuple((name, value) for name, value in node.parameters if name not in ignored)


def _signature(ir, node_id: int, sources: set[int]) -> tuple | None:
    node = ir.nodes[node_id]
    spec = descriptor(node.opcode)
    if node.opcode == "input":
        sources.add(int(dict(node.parameters)["input_index"]))
        return ("input", int(dict(node.parameters)["input_index"]))
    if not spec.lane_lift:
        return None
    children = tuple(_signature(ir, child, sources) for child in node.children)
    if any(child is None for child in children):
        return None
    return node.opcode, _semantic_parameters(node), children


def discover_lane_graph(ir) -> LaneGraph | None:
    root = ir.nodes[ir.outputs[0].node]
    if not descriptor(root.opcode).lane_root or len(root.children) < 2:
        return None
    signatures, input_sets = [], []
    for child in root.children:
        sources: set[int] = set()
        signature = _signature(ir, child, sources)
        if signature is None:
            return None
        signatures.append(signature)
        input_sets.append(tuple(sorted(sources)))
    if len(set(signatures)) != 1 or len(set(input_sets)) != 1:
        return None
    return LaneGraph(tuple(root.children), tuple(signatures), input_sets[0])


@dataclass(frozen=True)
class AcceleratorFactory:
    name: str
    probe: Callable[[object, LaneGraph], object | None]


_FACTORIES: list[AcceleratorFactory] = []


def register_accelerator_factory(factory: AcceleratorFactory) -> None:
    if any(existing.name == factory.name for existing in _FACTORIES):
        raise ValueError(f"duplicate cpp_new accelerator factory {factory.name!r}")
    _FACTORIES.append(factory)


def build_accelerator(ir):
    graph = discover_lane_graph(ir)
    if graph is None:
        return None
    for factory in _FACTORIES:
        accelerator = factory.probe(ir, graph)
        if accelerator is not None:
            return accelerator
    return None
