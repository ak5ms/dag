from __future__ import annotations

from trading_dsl_engine.jax_flat import optimized as _optimized
from trading_dsl_engine.jax_flat import optimized_pair_fusion as _pair
from trading_dsl_engine.jax_flat import optimized_planner as _planner


def _has_ewm_pair(program) -> bool:
    consumers = _planner._consumer_lists(program)
    return any(
        _planner._paired_ewm_consumer(program, node_id, consumers) is not None
        for node_id in range(len(program.nodes))
    )


def _execution_strategy_measured(self) -> str:
    if self.strategy == "auto":
        plan = _planner._detect_ewm_branch_plan(self.program)
        if plan is not None:
            return "ewm_branch_pair_batch"
        if _pair._pure_ewm_program(self.program) and _has_ewm_pair(self.program):
            return "pair_fused_node_batch"
    return _pair._BASE_EXECUTION_STRATEGY(self)


_optimized.OptimizedJaxFlatRuntime.execution_strategy = _execution_strategy_measured
