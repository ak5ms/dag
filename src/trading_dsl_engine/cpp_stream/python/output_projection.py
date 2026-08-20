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


def _split_singleton_ewm_bundle(stage: Stage) -> tuple[Stage, ...]:
    """Remove the invalid one-member bundle introduced by output epilogues.

    A normal EWM bundle has at least two independent state machines. When one EWM
    feeds several public projections, the first plain Copy projection can instead
    become the EWM destination and later projections can read that packed output.
    This keeps one EWM evaluation, removes scratch, and avoids teaching the native
    multi-state bundle about a degenerate one-member case. If there is no plain
    public Copy anchor, restore the ordinary EWM plus its projection stages.
    """

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
    `[parent, subgraph]` as `subgraph -> parent`, allowing the parent projection to
    read the requested packed subgraph instead of evaluating the same pure graph a
    second time. Stateful physical stages are outside this suffix and never move.
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
            (index for index in range(len(suffix))
             if index in remaining and indegree[index] == 0),
            None,
        )
        if ready is None:
            # Structural source containment should be acyclic. Preserve the
            # original schedule rather than guessing if a future source kind
            # violates that invariant.
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
    """Reuse an earlier public row result instead of recomputing the same source.

    Neutral-IR CSE can leave a pure expression lazy. Two separate Copy stages would
    then place that same expression in two different C++ loops, outside the
    compiler's local CSE region. The first requested row output is a natural
    materialization point: later equal roots, emits, and descendant expressions
    read its packed row slice. The incremental cost is therefore an ordinary copy,
    matching the memory traffic of returning the extra result rather than repeating
    expensive arithmetic.

    Native outputs are float64. Reusing that storage as an internal source is
    therefore restricted to float64 values; integral scratch remains typed so large
    integers and integer operators cannot be changed by a float64 round trip.
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

    expanded: list[Stage] = []
    for stage in plan.stages:
        expanded.extend(_split_singleton_ewm_bundle(stage))
    scheduled = _schedule_terminal_projections(expanded)
    optimized = _reuse_public_row_storage(scheduled)
    compact = _recompact_physical_scalar_slots(optimized)
    return replace(
        plan,
        stages=tuple(compact),
        scratch_slots=scalar_scratch_slots(compact),
    )


__all__ = ["optimize_public_projections"]
