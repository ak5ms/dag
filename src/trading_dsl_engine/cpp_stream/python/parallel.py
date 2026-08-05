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
    "ewm_bundle",
    "rolling",
    "theilsen",
    "periods_since_change",
    "hump",
    "trade_when",
    "linear_filter",
    "rolling_product",
    "rolling_kth",
    "rolling_prev_diff",
    "rolling_decay",
    "rolling_entropy",
    "tensor_cumsum",
    "tensor_ffill",
    "tensor_shift",
    "tensor_ewm",
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
    "ewm_bundle",
    "rolling",
    "theilsen",
    "periods_since_change",
    "hump",
    "trade_when",
    "linear_filter",
    "rolling_product",
    "rolling_kth",
    "rolling_prev_diff",
    "rolling_decay",
    "rolling_entropy",
    "vector_quantile",
    "cat",
    "instrument_basis",
    "tensor_copy",
    "tensor_unary",
    "tensor_binary",
    "tensor_ternary",
    "tensor_cumsum",
    "tensor_ffill",
    "tensor_shift",
    "tensor_ewm",
    "tensor_column",
}

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
    "ewm_bundle": 8,
    "rolling": 6,
    "theilsen": 16,
    "periods_since_change": 2,
    "hump": 2,
    "trade_when": 2,
    "linear_filter": 6,
    "rolling_product": 4,
    "rolling_kth": 6,
    "rolling_prev_diff": 5,
    "rolling_decay": 6,
    "rolling_entropy": 10,
    "vector_quantile": 8,
    "tensor_copy": 1,
    "tensor_unary": 1,
    "tensor_binary": 1,
    "tensor_ternary": 1,
    "tensor_cumsum": 2,
    "tensor_ffill": 2,
    "tensor_shift": 2,
    "tensor_ewm": 3,
    "tensor_column": 1,
    "cat": 2,
    "reduce": 2,
    "reduce_bundle": 5,
    "emit_last": 1,
    "einsum": 5,
    "instrument_basis": 10,
    "ridge": 12,
    "ridge_bundle": 18,
    "xs_rank": 8,
    "xs_pct_rank": 8,
    "xs_aggregate": 8,
    "xs_weighted_mean": 8,
    "xs_projection": 10,
    "xs_generalized_rank": 16,
    "xs_densify": 8,
}


def _ridge_is_stateful(stage: Stage) -> bool:
    if stage.kind not in {"ridge", "ridge_bundle"}:
        return False
    op = stage.op
    return bool(
        getattr(op, "is_stateful", False)
        and stage.half_life is not None
        and math.isfinite(stage.half_life)
        and stage.half_life > 0.0
    )


def _reduction_is_temporal(stage: Stage) -> bool:
    return stage.kind == "reduce" and bool(
        getattr(stage.op, "temporal", False)
    )


def _reduction_is_lane_local(stage: Stage, n_instruments: int) -> bool:
    """A row reduction is lane-local only when it retains the instrument axis."""
    if stage.kind not in {"reduce", "reduce_bundle"} or _reduction_is_temporal(stage):
        return False
    axes = tuple(getattr(stage.op, "axes", ()))
    if 1 in axes:
        return False
    if not stage.inputs or not stage.inputs[0].shape:
        return False
    return stage.inputs[0].shape[0] == n_instruments


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
        if (
            stage.kind in _TEMPORAL_KINDS
            or _ridge_is_stateful(stage)
            or _reduction_is_temporal(stage)
            or stage.kind == "emit_last"
        ):
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
    if source.kind in {
        "cat", "rbf", "future_rbf", "expr", "tensor_expr"
    }:
        return all(
            _source_is_available(part, scalar_slots, tensor_slots)
            for part in source.parts
        )
    return False


def _scratch_cross_lane_read(source: Source, lane_label: str) -> bool:
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
    return _plan_is_lane_independent(
        group.inner,
        n_instruments,
        require_partitionable_output=False,
    )


def _tensor_stage_lane_local(stage: Stage, n_instruments: int) -> bool:
    """Tensor stages are lane-local only with an instrument-leading shape."""
    return bool(
        stage.out.shape
        and stage.out.shape[0] == n_instruments
        and stage.out.size % n_instruments == 0
        and all(
            source.shape == ()
            or (
                source.shape
                and source.shape[0] in {1, n_instruments}
            )
            for source in stage.inputs
        )
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

        if stage.kind.startswith("tensor_"):
            local = _tensor_stage_lane_local(stage, n_instruments)
        elif stage.kind in _LANE_LOCAL_KINDS:
            local = True
        elif stage.kind in {"reduce", "reduce_bundle"}:
            local = _reduction_is_lane_local(stage, n_instruments)
        elif stage.kind == "einsum":
            local = _einsum_lane_local(
                stage, n_instruments, scalar_slots, tensor_slots
            )
        elif stage.kind == "groupby":
            local = _group_lane_local(stage, n_instruments)
        else:
            local = False

        if not local:
            return False
        if stage.out.slot is not None:
            if stage.out.matrix or stage.out.tensor:
                tensor_slots.add(stage.out.slot)
            else:
                scalar_slots.add(stage.out.slot)
        for dest in stage.bundle_outs:
            if dest.slot is None:
                continue
            if dest.matrix or dest.tensor:
                tensor_slots.add(dest.slot)
            else:
                scalar_slots.add(dest.slot)
    return True


def plan_is_lane_independent(plan: Plan, n_instruments: int) -> bool:
    return _plan_is_lane_independent(
        plan,
        n_instruments,
        require_partitionable_output=True,
    )


def select_parallel_plan(plan: Plan, n_instruments: int) -> ParallelPlan:
    score = _plan_work_score(plan)
    if plan.output_mode == "final":
        return ParallelPlan(
            "serial",
            "terminal streaming reduction or emit has one final accumulator owner",
            False,
            score,
        )
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
