from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Mapping

from trading_dsl_engine.cpp_stream.python.lowering import (
    GroupStage,
    Plan,
    Source,
    Stage,
    scalar_scratch_slots,
)


SlotKey = tuple[str, int]
ScalarSlotKey = tuple[str, int]
_TERMINAL_PROJECTION_KINDS = frozenset(
    {"copy", "tensor_copy", "cat", "emit_last"}
)


def _source_slot_key(source: Source) -> SlotKey | None:
    if source.kind == "slot":
        return ("scalar", int(source.value))
    if source.kind in {"matrix_slot", "tensor_slot"}:
        return ("tensor", int(source.value))
    return None


def _dest_slot_key(stage: Stage) -> SlotKey | None:
    slot = stage.out.slot
    if slot is None or slot < 0:
        return None
    return (
        "tensor" if stage.out.matrix or stage.out.tensor else "scalar",
        int(slot),
    )


def _public_offset(stage: Stage) -> int | None:
    slot = stage.out.slot
    return None if slot is None or slot >= 0 else -int(slot) - 1


def _collect_source_slots(source: Source, result: set[SlotKey]) -> None:
    key = _source_slot_key(source)
    if key is not None:
        result.add(key)
    for part in source.parts:
        _collect_source_slots(part, result)


def _stage_input_slots(stage: Stage) -> frozenset[SlotKey]:
    result: set[SlotKey] = set()
    for candidate in (stage, *stage.members, *stage.epilogues):
        for source in candidate.inputs:
            _collect_source_slots(source, result)
    if stage.group is not None:
        for source in (*stage.group.key_sources, *stage.group.feed_sources):
            _collect_source_slots(source, result)
    return frozenset(result)


def _packed_source(source: Source, offset: int) -> Source:
    if source.dtype != "float64":
        raise TypeError("packed public-output reuse requires a float64 source")
    return Source(
        "packed_output",
        offset,
        row_scalar=source.row_scalar,
        dtype=source.dtype,
        width=source.width,
        shape=source.shape,
        final_only=False,
    )


def _replace_slot_source(
    source: Source,
    key: SlotKey,
    replacement: Source,
) -> Source:
    if _source_slot_key(source) == key:
        return replacement
    if not source.parts:
        return source
    parts = tuple(
        _replace_slot_source(part, key, replacement)
        for part in source.parts
    )
    return source if parts == source.parts else replace(source, parts=parts)


def _is_public_ewm_projection(
    stage: Stage,
    member_outputs: frozenset[SlotKey],
    final_only: bool,
) -> bool:
    public_dest = _public_offset(stage) is not None
    scalar_projection = (
        stage.kind == "copy"
        and len(stage.inputs) == 1
        and stage.inputs[0].width == 1
    ) or (
        stage.kind == "cat"
        and stage.inputs
        and all(source.width == 1 for source in stage.inputs)
    )
    dependencies = _stage_input_slots(stage)
    return bool(
        public_dest
        and scalar_projection
        and dependencies
        and dependencies <= member_outputs
        and stage.final_only == final_only
    )


def _fuse_nonadjacent_ewm_epilogues(stages: list[Stage]) -> list[Stage]:
    """Attach output-only EWM consumers even across independent stages.

    The original lowering pass only saw an immediately following Copy/Cat. Public
    projections are terminal and their packed offsets encode API order, so an
    unrelated physical stage between an EWM producer and its projection must not
    force the EWM values into RowContext scratch. We scan all later direct users:
    fusion is allowed only when every use of every member output is an eligible
    public projection. A real downstream consumer keeps the ordinary scratch path.
    """

    removed: set[int] = set()
    result: list[Stage] = []

    for cursor, stage in enumerate(stages):
        if cursor in removed:
            continue
        if stage.kind not in {"ewm", "ewm_bundle"} or stage.epilogues:
            result.append(stage)
            continue

        members = stage.members if stage.members else (stage,)
        member_outputs = frozenset(
            key
            for member in members
            if (key := _dest_slot_key(member)) is not None
        )
        if not member_outputs:
            result.append(stage)
            continue

        epilogue_indices: list[int] = []
        blocked = False
        for following in range(cursor + 1, len(stages)):
            candidate = stages[following]
            dependencies = _stage_input_slots(candidate)
            if not (dependencies & member_outputs):
                continue
            if _is_public_ewm_projection(
                candidate,
                member_outputs,
                stage.final_only,
            ):
                epilogue_indices.append(following)
            else:
                blocked = True
                break

        if blocked or not epilogue_indices:
            result.append(stage)
            continue

        bundle = (
            stage
            if stage.kind == "ewm_bundle"
            else replace(stage, kind="ewm_bundle", members=members)
        )
        result.append(
            replace(
                bundle,
                epilogues=tuple(stages[index] for index in epilogue_indices),
            )
        )
        removed.update(epilogue_indices)

    return result


def _split_singleton_ewm_bundle(stage: Stage) -> tuple[Stage, ...]:
    """Remove a one-member EWM bundle introduced only by output epilogues."""

    if stage.kind != "ewm_bundle" or len(stage.members) != 1:
        return (stage,)

    member = stage.members[0]
    member_key = _dest_slot_key(member)
    if member_key is None:
        return (member, *stage.epilogues)

    anchor_index: int | None = None
    for index, epilogue in enumerate(stage.epilogues):
        if (
            epilogue.kind == "copy"
            and len(epilogue.inputs) == 1
            and _source_slot_key(epilogue.inputs[0]) == member_key
            and _public_offset(epilogue) is not None
            and not epilogue.final_only
        ):
            anchor_index = index
            break

    if anchor_index is None:
        return (member, *stage.epilogues)

    anchor = stage.epilogues[anchor_index]
    offset = _public_offset(anchor)
    assert offset is not None
    packed = _packed_source(anchor.inputs[0], offset)
    producer = replace(
        member,
        out=anchor.out,
        members=(),
        epilogues=(),
    )
    projections: list[Stage] = []
    for index, epilogue in enumerate(stage.epilogues):
        if index == anchor_index:
            continue
        inputs = tuple(
            _replace_slot_source(source, member_key, packed)
            for source in epilogue.inputs
        )
        projections.append(
            epilogue if inputs == epilogue.inputs else replace(epilogue, inputs=inputs)
        )
    return (producer, *projections)


def _source_contains(container: Source, candidate: Source) -> bool:
    if container == candidate:
        return True
    return any(_source_contains(part, candidate) for part in container.parts)


def _row_anchor_source(stage: Stage) -> Source | None:
    if (
        _public_offset(stage) is None
        or stage.final_only
        or stage.kind not in {"copy", "tensor_copy"}
        or len(stage.inputs) != 1
        or stage.inputs[0].dtype != "float64"
    ):
        return None
    return stage.inputs[0]


def _terminal_projection_suffix_start(stages: list[Stage]) -> int:
    start = len(stages)
    while start > 0:
        stage = stages[start - 1]
        if (
            _public_offset(stage) is None
            or stage.kind not in _TERMINAL_PROJECTION_KINDS
        ):
            break
        start -= 1
    return start


def _schedule_terminal_projections(stages: list[Stage]) -> list[Stage]:
    """Materialize requested lazy subgraphs before their public descendants.

    Output offsets encode API order independently of execution order. A stable
    topological ordering of the terminal projection suffix can therefore execute
    ``[parent, subgraph]`` as ``subgraph -> parent``. Stateful stages are outside
    this suffix and never move.
    """

    start = _terminal_projection_suffix_start(stages)
    suffix = stages[start:]
    if len(suffix) < 2:
        return stages

    anchors = tuple(_row_anchor_source(stage) for stage in suffix)
    edges: list[set[int]] = [set() for _ in suffix]
    indegree = [0] * len(suffix)
    for producer, source in enumerate(anchors):
        if source is None:
            continue
        for consumer, stage in enumerate(suffix):
            if producer == consumer:
                continue
            # Equal roots are already handled in stable API order. Only strict
            # containment requires reordering.
            depends = any(
                input_source != source
                and _source_contains(input_source, source)
                for input_source in stage.inputs
            )
            if depends and consumer not in edges[producer]:
                edges[producer].add(consumer)
                indegree[consumer] += 1

    remaining = set(range(len(suffix)))
    order: list[int] = []
    while remaining:
        ready = next(
            (
                index
                for index in range(len(suffix))
                if index in remaining and indegree[index] == 0
            ),
            None,
        )
        if ready is None:
            return stages
        remaining.remove(ready)
        order.append(ready)
        for consumer in edges[ready]:
            indegree[consumer] -= 1

    if order == list(range(len(suffix))):
        return stages
    return [*stages[:start], *(suffix[index] for index in order)]


def _rewrite_anchored_source(
    source: Source,
    anchors: dict[Source, Source],
) -> Source:
    anchor = anchors.get(source)
    if anchor is not None:
        return anchor
    if not source.parts:
        return source
    parts = tuple(
        _rewrite_anchored_source(part, anchors) for part in source.parts
    )
    return source if parts == source.parts else replace(source, parts=parts)


def _reuse_public_row_storage(stages: list[Stage]) -> list[Stage]:
    """Reuse an earlier float64 public row result instead of recomputing it.

    Public storage is float64, so integral values remain on their original typed
    source path and never round-trip through a potentially inexact output value.
    """

    source_anchors: dict[Source, Source] = {}
    cat_anchors: dict[
        tuple[tuple[Source, ...], tuple[int, ...], str],
        Source,
    ] = {}
    result: list[Stage] = []

    for original in stages:
        original_inputs = original.inputs
        inputs = tuple(
            _rewrite_anchored_source(source, source_anchors)
            for source in original_inputs
        )
        stage = original if inputs == original_inputs else replace(original, inputs=inputs)
        offset = _public_offset(stage)
        public_row_projection = (
            offset is not None
            and not stage.final_only
            and stage.kind in {"copy", "tensor_copy", "cat"}
        )
        if not public_row_projection:
            result.append(stage)
            continue

        assert offset is not None
        if stage.kind in {"copy", "tensor_copy"} and len(original_inputs) == 1:
            source = original_inputs[0]
            if source.dtype != "float64":
                result.append(stage)
                continue
            anchor = source_anchors.get(source)
            if anchor is None:
                source_anchors[source] = _packed_source(source, offset)
            else:
                stage = replace(stage, inputs=(anchor,))
            result.append(stage)
            continue

        if stage.kind == "cat":
            if stage.dtype != "float64":
                result.append(stage)
                continue
            key = (original_inputs, stage.out.shape, stage.dtype)
            anchor = cat_anchors.get(key)
            if anchor is None:
                cat_anchors[key] = Source(
                    "packed_output",
                    offset,
                    row_scalar=False,
                    dtype=stage.dtype,
                    width=stage.out.width,
                    shape=stage.out.shape,
                    final_only=False,
                )
            else:
                stage = replace(
                    stage,
                    kind="tensor_copy",
                    inputs=(anchor,),
                    members=(),
                    epilogues=(),
                )
            result.append(stage)
            continue

        result.append(stage)

    return result


def _copy_bundle_candidate(stage: Stage) -> bool:
    return bool(
        stage.kind == "copy"
        and _public_offset(stage) is not None
        and not stage.final_only
        and stage.dtype == "float64"
        and len(stage.inputs) == 1
        and stage.inputs[0].dtype == "float64"
        and stage.inputs[0].width == 1
        and not stage.out.matrix
        and not stage.out.tensor
    )


def _bundle_terminal_copies(stages: list[Stage]) -> list[Stage]:
    """Fuse adjacent public float64 vector copies into one cached lane loop."""

    result: list[Stage] = []
    cursor = 0
    while cursor < len(stages):
        first = stages[cursor]
        if not _copy_bundle_candidate(first):
            result.append(first)
            cursor += 1
            continue

        members = [first]
        following = cursor + 1
        while following < len(stages):
            candidate = stages[following]
            if not (
                _copy_bundle_candidate(candidate)
                and candidate.lane_count == first.lane_count
            ):
                break
            members.append(candidate)
            following += 1

        if len(members) == 1:
            result.append(first)
        else:
            result.append(
                replace(
                    first,
                    kind="copy_bundle",
                    inputs=tuple(member.inputs[0] for member in members),
                    members=tuple(members),
                    epilogues=(),
                )
            )
        cursor = following
    return result


def _physical_scalar_candidates(stage: Stage) -> tuple[Stage, ...]:
    # EWM members become compile-time component labels when epilogues are present;
    # only the epilogue destinations address RowContext scalar storage.
    if stage.kind == "ewm_bundle" and stage.epilogues:
        return stage.epilogues
    return (stage, *stage.members, *stage.epilogues)


def _physical_scalar_slot_map(stages: list[Stage]) -> dict[ScalarSlotKey, int]:
    slots_by_dtype: dict[str, set[int]] = defaultdict(set)
    for stage in stages:
        for candidate in _physical_scalar_candidates(stage):
            slot = candidate.out.slot
            if (
                slot is None
                or slot < 0
                or candidate.out.matrix
                or candidate.out.tensor
            ):
                continue
            slots_by_dtype[candidate.dtype].add(int(slot))
    return {
        (dtype, old): new
        for dtype, slots in slots_by_dtype.items()
        for new, old in enumerate(sorted(slots))
    }


def _remap_scalar_source(
    source: Source,
    slots: Mapping[ScalarSlotKey, int],
) -> Source:
    value = source.value
    if source.kind == "slot":
        value = slots.get((source.dtype, int(value)), int(value))
    parts = tuple(_remap_scalar_source(part, slots) for part in source.parts)
    if value == source.value and parts == source.parts:
        return source
    return replace(source, value=value, parts=parts)


def _remap_scalar_dest(
    stage: Stage,
    slots: Mapping[ScalarSlotKey, int],
) -> Stage:
    slot = stage.out.slot
    if (
        slot is None
        or slot < 0
        or stage.out.matrix
        or stage.out.tensor
    ):
        return stage
    mapped = slots.get((stage.dtype, int(slot)))
    return stage if mapped is None or mapped == slot else replace(
        stage,
        out=replace(stage.out, slot=mapped),
    )


def _remap_scalar_child(
    stage: Stage,
    slots: Mapping[ScalarSlotKey, int],
) -> Stage:
    remapped = _remap_scalar_dest(stage, slots)
    inputs = tuple(_remap_scalar_source(source, slots) for source in stage.inputs)
    return remapped if inputs == remapped.inputs else replace(remapped, inputs=inputs)


def _remap_scalar_group(
    group: GroupStage | None,
    slots: Mapping[ScalarSlotKey, int],
) -> GroupStage | None:
    if group is None:
        return None
    return replace(
        group,
        key_sources=tuple(
            _remap_scalar_source(source, slots) for source in group.key_sources
        ),
        feed_sources=tuple(
            _remap_scalar_source(source, slots) for source in group.feed_sources
        ),
    )


def _recompact_physical_scalar_slots(stages: list[Stage]) -> list[Stage]:
    """Remove scalar holes without treating EWM component labels as arrays."""

    slots = _physical_scalar_slot_map(stages)
    result: list[Stage] = []
    for stage in stages:
        remapped = _remap_scalar_child(stage, slots)
        result.append(
            replace(
                remapped,
                group=_remap_scalar_group(stage.group, slots),
                members=tuple(
                    _remap_scalar_child(member, slots)
                    for member in stage.members
                ),
                epilogues=tuple(
                    _remap_scalar_child(epilogue, slots)
                    for epilogue in stage.epilogues
                ),
            )
        )
    return result


def optimize_public_projections(plan: Plan) -> Plan:
    """Optimize terminal fan-out and recompact any released scalar slots."""

    fused = _fuse_nonadjacent_ewm_epilogues(list(plan.stages))
    expanded: list[Stage] = []
    for stage in fused:
        expanded.extend(_split_singleton_ewm_bundle(stage))
    scheduled = _schedule_terminal_projections(expanded)
    reused = _reuse_public_row_storage(scheduled)
    bundled = _bundle_terminal_copies(reused)
    compact = _recompact_physical_scalar_slots(bundled)
    return replace(
        plan,
        stages=tuple(compact),
        scratch_slots=scalar_scratch_slots(compact),
    )


__all__ = ["optimize_public_projections"]
