from __future__ import annotations

import math
from dataclasses import dataclass

from trading_dsl_engine.cpp_stream.python.lowering import Plan, Source, Stage


@dataclass(frozen=True, slots=True)
class ParallelPlan:
    mode: str
    reason: str
    auto_multicore: bool
    work_score: int


_TEMPORAL_KINDS = {
    "cumsum", "ffill", "shift", "ewm", "ewm_bundle", "rolling",
    "theilsen", "periods_since_change", "hump", "trade_when",
    "linear_filter", "rolling_product", "rolling_kth",
    "rolling_prev_diff", "rolling_decay", "rolling_entropy",
    "tensor_cumsum", "tensor_ffill", "tensor_shift", "tensor_ewm",
    "instrument_basis", "groupby",
}

_LANE_LOCAL_KINDS = {
    "copy", "unary", "binary", "ternary", "custom", "cumsum", "ffill",
    "shift", "ewm", "ewm_bundle", "rolling", "theilsen",
    "periods_since_change", "hump", "trade_when", "linear_filter",
    "rolling_product", "rolling_kth", "rolling_prev_diff", "rolling_decay",
    "rolling_entropy", "vector_quantile", "cat", "instrument_basis",
    "tensor_copy", "tensor_unary", "tensor_binary", "tensor_ternary",
    "tensor_cumsum", "tensor_ffill", "tensor_shift", "tensor_ewm",
    "tensor_column",
}

_EXPERIMENTAL_STAGE_WORK = {
    "copy": 1, "unary": 1, "binary": 1, "ternary": 1, "custom": 2,
    "cumsum": 2, "ffill": 2, "shift": 2, "ewm": 3, "ewm_bundle": 3,
    "rolling": 6, "theilsen": 16, "periods_since_change": 2, "hump": 2,
    "trade_when": 2, "linear_filter": 6, "rolling_product": 4,
    "rolling_kth": 6, "rolling_prev_diff": 5, "rolling_decay": 6,
    "rolling_entropy": 10, "vector_quantile": 8, "tensor_copy": 1,
    "tensor_unary": 1, "tensor_binary": 1, "tensor_ternary": 1,
    "tensor_cumsum": 2, "tensor_ffill": 2, "tensor_shift": 2,
    "tensor_ewm": 3, "tensor_column": 1, "cat": 2, "reduce": 2,
    "reduction_bundle": 2, "emit_last": 1, "einsum": 5,
    "instrument_basis": 10, "ridge": 12, "ridge_bundle": 12,
    "xs_rank": 8, "xs_pct_rank": 8, "xs_aggregate": 8,
    "xs_weighted_mean": 8, "xs_projection": 10,
    "xs_generalized_rank": 16, "xs_densify": 8,
}


def _ridge_is_stateful(stage: Stage) -> bool:
    if stage.kind not in {"ridge", "ridge_bundle"}:
        return False
    return bool(
        getattr(stage.op, "is_stateful", False)
        and stage.half_life is not None
        and math.isfinite(stage.half_life)
        and stage.half_life > 0.0
    )


def _reduction_is_temporal(stage: Stage) -> bool:
    return stage.kind in {"reduce", "reduction_bundle"} and bool(
        getattr(stage.op, "temporal", False)
    )


def _reduction_retains_instrument_axis(stage: Stage, n_instruments: int) -> bool:
    if stage.kind not in {"reduce", "reduction_bundle"}:
        return False
    axes = tuple(getattr(stage.op, "axes", ()))
    if 1 in axes or not stage.inputs or not stage.inputs[0].shape:
        return False
    if stage.inputs[0].shape[0] != n_instruments:
        return False
    return bool(stage.out.shape and stage.out.shape[0] == n_instruments)


def _reduction_is_lane_local(stage: Stage, n_instruments: int) -> bool:
    return not _reduction_is_temporal(stage) and _reduction_retains_instrument_axis(stage, n_instruments)


def _temporal_reduction_is_lane_mergeable(stage: Stage, n_instruments: int) -> bool:
    return _reduction_is_temporal(stage) and _reduction_retains_instrument_axis(stage, n_instruments)


def _emit_is_lane_mergeable(stage: Stage, n_instruments: int) -> bool:
    return bool(
        stage.kind == "emit_last"
        and stage.inputs
        and stage.inputs[0].shape
        and stage.inputs[0].shape[0] == n_instruments
    )


def _source_work_score(source: Source) -> int:
    child_score = sum(_source_work_score(part) for part in source.parts)
    if source.kind in {"expression", "stateless_expression"}:
        return 1 + child_score
    if source.kind == "cat":
        return max(1, int(source.width)) + child_score
    if source.kind in {"rbf", "future_rbf"}:
        return max(2, int(source.width)) + child_score
    return child_score


def _stage_work_score(stage: Stage) -> int:
    score = _EXPERIMENTAL_STAGE_WORK.get(stage.kind, 2)
    score += sum(_source_work_score(source) for source in stage.inputs)
    if stage.kind == "groupby" and stage.group is not None:
        score += 4 + _plan_work_score(stage.group.inner)
    score += sum(_stage_work_score(member) for member in stage.members)
    score += sum(_stage_work_score(item) for item in stage.epilogues)
    return score


def _plan_work_score(plan: Plan) -> int:
    return sum(_stage_work_score(stage) for stage in plan.stages)


def _row_independence(plan: Plan) -> tuple[bool, str]:
    for index, stage in enumerate(plan.stages):
        if stage.kind in _TEMPORAL_KINDS or _ridge_is_stateful(stage):
            return False, f"stage {index} ({stage.kind}) carries state across rows"
        if _reduction_is_temporal(stage):
            return False, f"stage {index} ({stage.kind}) reduces the time axis"
        if stage.kind == "emit_last":
            return False, f"stage {index} (emit_last) retains the last row"
    return True, "all rows are independent"


def plan_is_row_independent(plan: Plan) -> bool:
    return _row_independence(plan)[0]


def _final_row_mergeability(plan: Plan) -> tuple[bool, str]:
    saw_mergeable_state = False
    for index, stage in enumerate(plan.stages):
        if stage.final_only:
            continue
        if _reduction_is_temporal(stage) or stage.kind == "emit_last":
            saw_mergeable_state = True
            continue
        if stage.kind in _TEMPORAL_KINDS or _ridge_is_stateful(stage):
            return False, f"stage {index} ({stage.kind}) carries state across row shards"
    if not saw_mergeable_state:
        return False, "the final plan has no mergeable reduction or emit state"
    return True, "pre-final rows are independent and terminal worker states are mergeable"


def _source_is_available(source: Source, scalar_slots: set[int], tensor_slots: set[int]) -> bool:
    if source.kind in {"input", "literal"}:
        return True
    if source.kind == "slot":
        return int(source.value) in scalar_slots
    if source.kind in {"matrix_slot", "tensor_slot"}:
        return int(source.value) in tensor_slots
    if source.kind in {"cat", "rbf", "future_rbf", "expression", "stateless_expression"}:
        return all(_source_is_available(part, scalar_slots, tensor_slots) for part in source.parts)
    return False


def _scratch_cross_lane_read(source: Source, lane_label: str) -> bool:
    if source.kind not in {"slot", "matrix_slot", "tensor_slot"}:
        return False
    if not source.shape or source.shape[0] == 1:
        return False
    return source.shape[0] > 1 and not lane_label


def _einsum_lane_local(stage: Stage, n_instruments: int,
                       scalar_slots: set[int], tensor_slots: set[int]) -> bool:
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
                if not labels or labels[0] != lane_label or labels.count(lane_label) != 1:
                    return False
        if _scratch_cross_lane_read(source, lane_label if labels and labels[0] == lane_label else ""):
            return False
    return True


def _group_lane_local(stage: Stage, n_instruments: int) -> bool:
    return bool(
        stage.group is not None
        and _plan_is_lane_independent(
            stage.group.inner,
            n_instruments,
            require_partitionable_output=False,
        )
    )


def _tensor_stage_lane_local(stage: Stage, n_instruments: int) -> bool:
    return bool(
        stage.out.shape
        and stage.out.shape[0] == n_instruments
        and stage.out.size % n_instruments == 0
        and all(
            source.shape == ()
            or (source.shape and source.shape[0] in {1, n_instruments})
            for source in stage.inputs
        )
    )


def _lane_independence(
    plan: Plan,
    n_instruments: int,
    *,
    require_partitionable_output: bool,
    skip_final_only: bool = False,
    allow_temporal_reductions: bool = False,
    allow_emit_last: bool = False,
) -> tuple[bool, str]:
    if n_instruments <= 1:
        return False, "the plan has only one instrument lane"
    if require_partitionable_output:
        if not plan.output_shape or plan.output_shape[0] != n_instruments:
            return False, "the output does not retain the instrument axis"
        if plan.output_row_width % n_instruments:
            return False, "the output width is not divisible by instrument lanes"

    scalar_slots: set[int] = set()
    tensor_slots: set[int] = set()
    for index, stage in enumerate(plan.stages):
        if skip_final_only and stage.final_only:
            continue
        if not all(_source_is_available(source, scalar_slots, tensor_slots) for source in stage.inputs):
            return False, f"stage {index} ({stage.kind}) reads unavailable scratch"

        if stage.kind.startswith("tensor_"):
            local = _tensor_stage_lane_local(stage, n_instruments)
            detail = "tensor shape is not instrument-leading"
        elif stage.kind in _LANE_LOCAL_KINDS:
            local, detail = True, ""
        elif stage.kind in {"reduce", "reduction_bundle"}:
            if _reduction_is_temporal(stage):
                local = allow_temporal_reductions and _temporal_reduction_is_lane_mergeable(stage, n_instruments)
                detail = "temporal reduction removes the instrument axis"
            else:
                local = _reduction_is_lane_local(stage, n_instruments)
                detail = "row reduction removes the instrument axis"
        elif stage.kind == "emit_last":
            local = allow_emit_last and _emit_is_lane_mergeable(stage, n_instruments)
            detail = "emit output is not partitionable by instrument lane"
        elif stage.kind == "einsum":
            local = _einsum_lane_local(stage, n_instruments, scalar_slots, tensor_slots)
            detail = "einsum contracts or permutes across instrument lanes"
        elif stage.kind == "groupby":
            local = _group_lane_local(stage, n_instruments)
            detail = "groupby inner plan couples instrument lanes"
        else:
            local = False
            detail = "operator couples instrument lanes"
        if not local:
            return False, f"stage {index} ({stage.kind}): {detail}"

        outputs = (*stage.members, *stage.epilogues) if stage.members or stage.epilogues else (stage,)
        for member in outputs:
            if member.out.slot is not None:
                if member.out.matrix or member.out.tensor:
                    tensor_slots.add(member.out.slot)
                else:
                    scalar_slots.add(member.out.slot)
    return True, "temporal state is independent across instrument lanes"


def _plan_is_lane_independent(plan: Plan, n_instruments: int, *,
                              require_partitionable_output: bool,
                              skip_final_only: bool = False,
                              allow_temporal_reductions: bool = False,
                              allow_emit_last: bool = False) -> bool:
    return _lane_independence(
        plan, n_instruments,
        require_partitionable_output=require_partitionable_output,
        skip_final_only=skip_final_only,
        allow_temporal_reductions=allow_temporal_reductions,
        allow_emit_last=allow_emit_last,
    )[0]


def plan_is_lane_independent(plan: Plan, n_instruments: int) -> bool:
    return _plan_is_lane_independent(plan, n_instruments, require_partitionable_output=True)


def _final_lane_mergeability(plan: Plan, n_instruments: int) -> tuple[bool, str]:
    saw_mergeable_state = any(
        not stage.final_only
        and (
            _temporal_reduction_is_lane_mergeable(stage, n_instruments)
            or _emit_is_lane_mergeable(stage, n_instruments)
        )
        for stage in plan.stages
    )
    if not saw_mergeable_state:
        return False, "no terminal state can be merged by instrument lane"
    return _lane_independence(
        plan,
        n_instruments,
        require_partitionable_output=False,
        skip_final_only=True,
        allow_temporal_reductions=True,
        allow_emit_last=True,
    )


def _lane_plan(reason: str, score: int) -> ParallelPlan:
    return ParallelPlan(
        "lanes",
        reason + "; observed row count selects the automatic thread count",
        True,
        score,
    )


def select_parallel_plan(plan: Plan, n_instruments: int) -> ParallelPlan:
    score = _plan_work_score(plan)
    if plan.output_mode == "final":
        row_ok, row_reason = _final_row_mergeability(plan)
        if row_ok:
            return ParallelPlan("rows", row_reason, True, score)
        lane_ok, lane_reason = _final_lane_mergeability(plan, n_instruments)
        if lane_ok:
            return _lane_plan("terminal worker state is mergeable; " + lane_reason, score)
        return ParallelPlan(
            "serial",
            "final output cannot be partitioned safely: "
            f"row sharding blocked by {row_reason}; "
            f"lane sharding blocked by {lane_reason}",
            False,
            score,
        )

    row_ok, row_reason = _row_independence(plan)
    if row_ok:
        return ParallelPlan("rows", row_reason, True, score)
    lane_ok, lane_reason = _lane_independence(
        plan,
        n_instruments,
        require_partitionable_output=True,
    )
    if lane_ok:
        return _lane_plan(lane_reason, score)
    return ParallelPlan(
        "serial",
        "row sharding blocked by "
        f"{row_reason}; lane sharding blocked by {lane_reason}",
        False,
        score,
    )


__all__ = [
    "ParallelPlan",
    "plan_is_lane_independent",
    "plan_is_row_independent",
    "select_parallel_plan",
]
