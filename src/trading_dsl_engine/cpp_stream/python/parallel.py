from __future__ import annotations

import math
from dataclasses import dataclass

from trading_dsl_engine.cpp_stream.python.lowering import Plan, Source, Stage


@dataclass(frozen=True, slots=True)
class ParallelPlan:
    """Static execution strategy selected for one generated formula.

    ``rows`` shards independent rows and is valid only when no stage carries state
    between rows. ``lanes`` gives each worker a fixed instrument-lane interval and
    lets that worker advance the complete time series, preserving temporal state.
    ``serial`` is the conservative fallback for stateful cross-sectional graphs.

    ``auto_multicore`` controls only the explicit ``threads=0`` mode. Positive
    thread counts remain available for caller-selected parallelism. Lane sharding
    rereads every row once per worker, so the optional automatic mode keeps low-work
    lane graphs single-threaded even though they are semantically partitionable.
    """

    mode: str
    reason: str
    auto_multicore: bool
    work_score: int


_TEMPORAL_KINDS = {
    "cumsum",
    "ffill",
    "shift",
    "ewm",
    "instrument_basis",
    "groupby",
}

_LANE_LOCAL_KINDS = {
    "copy",
    "unary",
    "binary",
    "ternary",
    "custom",
    "cumsum",
    "ffill",
    "shift",
    "ewm",
    "cat",
    "instrument_basis",
}

# This heuristic is retained only for the explicit threads=0 opt-in. Lane workers
# traverse the complete time axis independently, trading redundant input reads for
# parallel compute. The threshold was chosen from the broad 4-CPU matrix: one-stage
# EWM (score 3) and cumsum+EWM groupby (score 9) regress, while the real roll_rets
# graph is well above the threshold and improves materially.
_LANE_AUTO_MULTICORE_MIN_SCORE = 16

_EXPERIMENTAL_STAGE_WORK = {
    "copy": 1,
    "unary": 1,
    "binary": 1,
    "ternary": 1,
    "custom": 2,
    "cumsum": 2,
    "ffill": 2,
    "shift": 2,
    "ewm": 3,
    "cat": 2,
    "einsum": 5,
    "instrument_basis": 10,
    "ridge": 12,
    "xs_rank": 8,
}


def _ridge_is_stateful(stage: Stage) -> bool:
    if stage.kind != "ridge":
        return False
    op = stage.op
    return bool(
        getattr(op, "is_stateful", False)
        and stage.half_life is not None
        and math.isfinite(stage.half_life)
        and stage.half_life > 0.0
    )


def _plan_work_score(plan: Plan) -> int:
    score = 0
    for stage in plan.stages:
        if stage.kind == "groupby" and stage.group is not None:
            score += 4 + _plan_work_score(stage.group.inner)
        else:
            score += _EXPERIMENTAL_STAGE_WORK.get(stage.kind, 2)
    return score


def plan_is_row_independent(plan: Plan) -> bool:
    for stage in plan.stages:
        if stage.kind in _TEMPORAL_KINDS or _ridge_is_stateful(stage):
            return False
    return True


def _source_is_available(
    source: Source,
    scalar_slots: set[int],
    tensor_slots: set[int],
) -> bool:
    if source.kind in {"input", "literal"}:
        return True
    if source.kind == "slot":
        return int(source.value) in scalar_slots
    if source.kind in {"matrix_slot", "tensor_slot"}:
        return int(source.value) in tensor_slots
    if source.kind in {"cat", "rbf", "future_rbf"}:
        return all(
            _source_is_available(part, scalar_slots, tensor_slots)
            for part in source.parts
        )
    return False


def _scratch_cross_lane_read(source: Source, lane_label: str) -> bool:
    """Whether a generated einsum input needs another worker's scratch lanes."""
    if source.kind not in {"slot", "matrix_slot", "tensor_slot"}:
        return False
    if not source.shape or source.shape[0] == 1:
        return False
    return source.shape[0] > 1 and not lane_label


def _einsum_lane_local(
    stage: Stage,
    n_instruments: int,
    scalar_slots: set[int],
    tensor_slots: set[int],
) -> bool:
    step = stage.einsum_step
    if step is None or not step.output_shape:
        return False
    if step.output_shape[0] != n_instruments or not step.output_labels:
        return False
    lane_label = step.output_labels[0]
    for source, labels in zip(stage.inputs, step.input_labels):
        if not _source_is_available(source, scalar_slots, tensor_slots):
            return False
        if source.kind in {"slot", "matrix_slot", "tensor_slot"}:
            if source.shape and source.shape[0] == n_instruments:
                if not labels or labels[0] != lane_label:
                    return False
                if labels.count(lane_label) != 1:
                    return False
        if _scratch_cross_lane_read(
            source,
            lane_label if labels and labels[0] == lane_label else "",
        ):
            return False
    return True


def _group_lane_local(stage: Stage, n_instruments: int) -> bool:
    group = stage.group
    if group is None:
        return False
    # Grouped inner IR values are logically scalar per member even though the
    # generated physical plan processes N lanes. Its file-output shape therefore
    # must not be used as the lane-partitionability test.
    return _plan_is_lane_independent(
        group.inner,
        n_instruments,
        require_partitionable_output=False,
    )


def _plan_is_lane_independent(
    plan: Plan,
    n_instruments: int,
    *,
    require_partitionable_output: bool,
) -> bool:
    if n_instruments <= 1:
        return False
    if require_partitionable_output:
        if not plan.output_shape or plan.output_shape[0] != n_instruments:
            return False
        if plan.output_row_width % n_instruments:
            return False

    scalar_slots: set[int] = set()
    tensor_slots: set[int] = set()
    for stage in plan.stages:
        inputs_ready = all(
            _source_is_available(source, scalar_slots, tensor_slots)
            for source in stage.inputs
        )
        if not inputs_ready:
            return False

        if stage.kind in _LANE_LOCAL_KINDS:
            local = True
        elif stage.kind == "einsum":
            local = _einsum_lane_local(
                stage, n_instruments, scalar_slots, tensor_slots
            )
        elif stage.kind == "groupby":
            local = _group_lane_local(stage, n_instruments)
        else:
            # xs_rank and Ridge are cross-sectional. Stateless instances still
            # receive row parallelism through the higher-priority rows strategy.
            local = False

        if not local:
            return False
        if stage.out.slot is not None:
            if stage.out.matrix or stage.out.tensor:
                tensor_slots.add(stage.out.slot)
            else:
                scalar_slots.add(stage.out.slot)
    return True


def plan_is_lane_independent(plan: Plan, n_instruments: int) -> bool:
    return _plan_is_lane_independent(
        plan,
        n_instruments,
        require_partitionable_output=True,
    )


def select_parallel_plan(plan: Plan, n_instruments: int) -> ParallelPlan:
    score = _plan_work_score(plan)
    if plan_is_row_independent(plan):
        return ParallelPlan(
            "rows",
            "all rows are independent",
            True,
            score,
        )
    if plan_is_lane_independent(plan, n_instruments):
        profitable = score >= _LANE_AUTO_MULTICORE_MIN_SCORE
        reason = "temporal state is independent across instrument lanes"
        if not profitable:
            reason += (
                f"; work score {score} is below automatic multicore threshold "
                f"{_LANE_AUTO_MULTICORE_MIN_SCORE}"
            )
        return ParallelPlan("lanes", reason, profitable, score)
    return ParallelPlan(
        "serial",
        "graph contains temporal cross-sectional or conservatively unsupported state",
        False,
        score,
    )


__all__ = [
    "ParallelPlan",
    "plan_is_lane_independent",
    "plan_is_row_independent",
    "select_parallel_plan",
]
