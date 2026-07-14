from __future__ import annotations

from trading_dsl_engine.jax_flat import optimized as _optimized
from trading_dsl_engine.jax_flat.ops import CumsumOp, EwmOp


def _replace_parallel_ops_cpu(program):
    """Keep associative EWM out of the default CPU plan.

    The lowering remains available in optimized.py for experimentation, but local
    CPU HLO/runtime measurements show that the tree scan is slower and allocates
    substantially more temporary memory than the sequential scan for the current
    aligned feature shapes.
    """
    return program


def _node_batch_pad_safe(program) -> bool:
    for node in program.nodes:
        op = node.op
        if not op.is_stateful:
            continue
        if isinstance(op, CumsumOp):
            continue
        if isinstance(op, EwmOp) and op.ignore_na:
            continue
        return False
    return True


def _choose_cpu_strategy(program, requested: str) -> str:
    if requested not in {"auto", "compound", "node_batch"}:
        raise ValueError("strategy must be 'auto', 'compound', or 'node_batch'")
    if requested != "auto":
        return requested

    # On the measured 56-core CPU with nine instruments, the compound scan cut
    # HLO loop count and temporary memory but consistently reduced throughput for
    # narrow stateful chains by serializing the entire per-tick DAG onto roughly
    # one effective core. Retain compound as an explicit memory-first option;
    # default to specialized node-batch kernels whenever padded chunks preserve
    # semantics. Multi-root formulas still share one DAG and receive graph-level
    # CSE under node_batch.
    return "node_batch" if _node_batch_pad_safe(program) else "compound"


_optimized._replace_parallel_ops = _replace_parallel_ops_cpu
_optimized._choose_strategy = _choose_cpu_strategy
