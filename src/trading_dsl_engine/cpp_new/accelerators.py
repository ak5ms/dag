"""Built-in native accelerator factories.

Factories are capability probes over a generic :class:`LaneGraph`.  Operator
names belong here, next to the executor they configure, rather than in runtime
selection or graph discovery.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path

from trading_dsl_engine._native_build import ensure_native_extension_current
from trading_dsl_engine.cpp_new.lanes import AcceleratorFactory, register_accelerator_factory


@dataclass(frozen=True)
class NativeAccelerator:
    core: object
    tier: str
    input_indices: tuple[int, ...]

    def __getattr__(self, name):
        return getattr(self.core, name)


def _lane_module():
    module_name = "trading_dsl_engine.cpp_new._cpp_new_lanes"
    spec = importlib.util.find_spec(module_name)
    extension = Path(spec.origin) if spec is not None and spec.origin is not None else None
    ensure_native_extension_current(Path(__file__).resolve().parents[3], "cpp_new_lanes", extension)
    return importlib.import_module(module_name)


def _unary_ewm_rank_pipeline(ir, graph):
    if len(graph.source_inputs) != 1:
        return None
    branches = []
    for output in graph.output_nodes:
        cursor = ir.nodes[output]
        stages = []
        while True:
            ranked = cursor.opcode == "xs_rank" and len(cursor.children) == 1
            if ranked:
                cursor = ir.nodes[cursor.children[0]]
            if cursor.opcode != "ewm" or len(cursor.children) != 1:
                break
            stages.append((float(dict(cursor.parameters)["param"]), ranked))
            cursor = ir.nodes[cursor.children[0]]
        if cursor.opcode != "input" or not stages:
            return None
        branches.append(tuple(reversed(stages)))
    stage_count = len(branches[0])
    if any(len(branch) != stage_count for branch in branches):
        return None
    rank_after = [branches[0][stage][1] for stage in range(stage_count)]
    if any([branch[stage][1] for stage in range(stage_count)] != rank_after for branch in branches):
        return None
    stage_spans = [[branch[stage][0] for branch in branches] for stage in range(stage_count)]
    core = _lane_module().EwmLaneRuntime(stage_spans, rank_after)
    if stage_count > 1:
        tier = "fused-ewm-cross-sectional-pipeline-native"
    elif rank_after[-1]:
        tier = "fused-ewm-rank-lane-native"
    else:
        tier = "fused-ewm-lane-native"
    return NativeAccelerator(core, tier, graph.source_inputs)


register_accelerator_factory(AcceleratorFactory("ewm-rank-unary-pipeline", _unary_ewm_rank_pipeline))
