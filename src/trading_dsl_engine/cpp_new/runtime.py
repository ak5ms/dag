"""Public optimized-tier runtime with generic native execution fallback."""
from dataclasses import dataclass
from pathlib import Path
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

    def init_state(self, n_instruments): return self.generic.init_state(n_instruments)
    def tick(self, state, *rows): return self.generic.tick(state, *rows)
    def tick_into(self, state, out, *rows): return self.generic.tick_into(state, out, *rows)
    def run_batch(self, inputs, states=None, out=None): return self.generic.run_batch(inputs, states, out)
    def run_batch_into(self, state, out, inputs): return self.generic.run_batch_into(state, out, inputs)
    def inspect_ir(self): return self.ir.inspect()
    def inspect_generated_source(self): return emit_source(self.ir)
    def inspect_layout(self): return {"state_bytes": self.ir.state_bytes, "scratch_bytes": self.ir.scratch_bytes, "states": self.ir.inspect()["states"], "scratch": self.ir.inspect()["scratch"]}


def compile_formula(formula, dsl_registry=None, *, mode="cached-specialized", cache_dir=None, n_instruments=None):
    if mode not in {"generic-only", "eagerly-specialized", "cached-specialized"}: raise ValueError("invalid cpp_new mode")
    generic = compile_generic(formula, dsl_registry=dsl_registry)
    ir = lower(generic.program, n_instruments=n_instruments)
    artifact = None if mode == "generic-only" else materialize(ir, cache_dir)
    return SpecializedRuntime(generic, ir, artifact, mode)
