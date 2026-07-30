"""Public optimized-tier runtime with generic native execution fallback."""
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import numpy as np
from trading_dsl_engine._native_build import ensure_native_extension_current
from trading_dsl_engine.cpp_new.build import materialize
from trading_dsl_engine.cpp_new.codegen import emit_source
from trading_dsl_engine.cpp_new.lowering import lower
from trading_dsl_engine.jax_flat.engine_cpp import compile_formula as compile_generic


@dataclass(frozen=True)
class SpecializedRuntime:
    generic: object
    ir: object
    artifact: object | None
    mode: str
    accelerator: object | None = None

    @property
    def program(self): return self.generic.program
    def init_state(self, n_instruments): return self.accelerator.init_state(n_instruments) if self.accelerator else self.generic.init_state(n_instruments)
    def tick(self, state, *rows):
        if not self.accelerator: return self.generic.tick(state, *rows)
        out = np.empty((np.asarray(rows[0]).shape[0], self.ir.nodes[-1].value_type.width))
        self.accelerator.tick_into(state, out, np.asarray(rows[0], dtype=np.float64))
        return state, out
    def tick_into(self, state, out, *rows):
        if self.accelerator: return self.accelerator.tick_into(state, out, np.asarray(rows[0], dtype=np.float64))
        return self.generic.tick_into(state, out, *rows)
    def run_batch(self, inputs, states=None, out=None):
        if not self.accelerator: return self.generic.run_batch(inputs, states, out)
        values = tuple(inputs.values()) if isinstance(inputs, dict) else tuple(inputs)
        data = np.asarray(values[0], dtype=np.float64, order="C")
        state = states or self.init_state(data.shape[1])
        if out is not None:
            self.accelerator.run_batch_into(state, out, data)
            result = out
        else:
            result = self.accelerator.run_batch(state, data)
        return state, result
    def run_batch_into(self, state, out, inputs):
        return self.run_batch(inputs, states=state, out=out)
    def run_batch_ablation(self, state, out, inputs, variant):
        if not self.accelerator:
            raise ValueError("ablations require a selected native lane accelerator")
        values = tuple(inputs.values()) if isinstance(inputs, dict) else tuple(inputs)
        data = np.asarray(values[0], dtype=np.float64, order="C")
        self.accelerator.run_batch_ablation(state, out, data, variant)
        return state, out
    def inspect_ir(self): return self.ir.inspect()
    def inspect_generated_source(self): return emit_source(self.ir)
    def inspect_layout(self): return {"state_bytes": self.ir.state_bytes, "scratch_bytes": self.ir.scratch_bytes, "states": self.ir.inspect()["states"], "scratch": self.ir.inspect()["scratch"]}
    @property
    def execution_tier(self): return "fused-ewm-lane-native" if self.accelerator else "generic-flat-native-bridge"


def compile_formula(formula, dsl_registry=None, *, mode="cached-specialized", cache_dir=None, n_instruments=None):
    if mode not in {"generic-only", "eagerly-specialized", "cached-specialized"}: raise ValueError("invalid cpp_new mode")
    generic = compile_generic(formula, dsl_registry=dsl_registry)
    ir = lower(generic.program, n_instruments=n_instruments)
    artifact = None if mode == "generic-only" else materialize(ir, cache_dir)
    accelerator = None if mode == "generic-only" else _ewm_lane_accelerator(ir)
    return SpecializedRuntime(generic, ir, artifact, mode, accelerator)


def _ewm_lane_accelerator(ir):
    root = ir.nodes[ir.outputs[0].node]
    if root.opcode != "cat" or not root.children:
        return None
    children = tuple(ir.nodes[child] for child in root.children)
    if any(node.opcode != "ewm" or len(node.children) != 1 for node in children):
        return None
    if len({node.children for node in children}) != 1:
        return None
    source = ir.nodes[children[0].children[0]]
    if source.opcode != "input":
        return None
    module_name = "trading_dsl_engine.cpp_new._cpp_new_lanes"
    spec = importlib.util.find_spec(module_name)
    extension = Path(spec.origin) if spec is not None and spec.origin is not None else None
    ensure_native_extension_current(Path(__file__).resolve().parents[3], "cpp_new_lanes", extension)
    _cpp_new_lanes = importlib.import_module(module_name)

    spans = [float(dict(node.parameters)["param"]) for node in children]
    return _cpp_new_lanes.EwmLaneRuntime(spans)
