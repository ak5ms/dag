from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from typing import Mapping

from trading_dsl_engine.cpp_stream.python.lowering import (
    Dest,
    GroupStage,
    Plan,
    Source,
    Stage,
    scalar_scratch_slots,
)
from trading_dsl_engine.cpp_stream.python.lowering_full import lower_graph
from trading_dsl_engine.cpp_stream.python.outputs import (
    FormulaOutput,
    OutputLayout,
    build_output_layout,
)
from trading_dsl_engine.ir.ops import EmitOp, LiteralOp
from trading_dsl_engine.ir.program import Node, Program


SlotKey = tuple[str, int]
ScalarSlotKey = tuple[str, int]


def _physical_program(program: Program) -> tuple[Program, tuple[int, ...]]:
    """Remove only terminal Emit scheduling markers before physical lowering."""

    nodes = list(program.nodes)
    exposed_ids: list[int] = []
    for root_id in program.outputs:
        node = nodes[root_id]
        if isinstance(node.op, EmitOp):
            exposed_ids.append(node.child_ids[0])
            nodes[root_id] = Node(LiteralOp(0.0), (), node.value_type)
        else:
            exposed_ids.append(root_id)

    return (
        Program(tuple(nodes), (exposed_ids[0],), program.input_names),
        tuple(exposed_ids),
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


def _stage_candidates(stage: Stage) -> tuple[Stage, ...]:
    return (stage, *stage.members, *stage.epilogues)


def _promotable_producers(stages: list[Stage]) -> frozenset[SlotKey]:
    result: set[SlotKey] = set()
    for stage in stages:
        for candidate in _stage_candidates(stage):
            key = _dest_slot_key(candidate)
            if key is not None and candidate.dtype == "float64":
                result.add(key)
    return frozenset(result)


def _collect_source_slots(source: Source, result: set[SlotKey]) -> None:
    key = _source_slot_key(source)
    if key is not None:
        result.add(key)
    for part in source.parts:
        _collect_source_slots(part, result)


def _source_slots(source: Source) -> frozenset[SlotKey]:
    result: set[SlotKey] = set()
    _collect_source_slots(source, result)
    return frozenset(result)


def _public_dependency_slots(
    exposed_sources: tuple[Source, ...],
) -> frozenset[SlotKey]:
    """Public physical roots that are dependencies of another public root."""

    roots = frozenset(
        key
        for source in exposed_sources
        if (key := _source_slot_key(source)) is not None
    )
    dependencies: set[SlotKey] = set()
    for source in exposed_sources:
        slots = set(_source_slots(source))
        own = _source_slot_key(source)
        if own is not None:
            slots.discard(own)
        dependencies.update(slots & roots)
    return frozenset(dependencies)


def _consumed_slot_keys(stages: list[Stage]) -> frozenset[SlotKey]:
    """Find physical scratch values that remain hot inputs downstream."""

    result: set[SlotKey] = set()
    for stage in stages:
        for candidate in _stage_candidates(stage):
            for source in candidate.inputs:
                _collect_source_slots(source, result)
        if stage.group is not None:
            for source in (*stage.group.key_sources, *stage.group.feed_sources):
                _collect_source_slots(source, result)
    return frozenset(result)


def _packed_source(source: Source, output: FormulaOutput) -> Source:
    return Source(
        "packed_output",
        output.offset,
        row_scalar=source.row_scalar,
        dtype=source.dtype,
        width=source.width,
        shape=source.shape,
        final_only=source.final_only,
    )


def _rewrite_source(
    source: Source,
    promotions: Mapping[SlotKey, FormulaOutput],
) -> Source:
    key = _source_slot_key(source)
    if key is not None and key in promotions:
        return _packed_source(source, promotions[key])
    if not source.parts:
        return source
    parts = tuple(_rewrite_source(part, promotions) for part in source.parts)
    return source if parts == source.parts else replace(source, parts=parts)


def _promoted_dest(
    stage: Stage,
    promotions: Mapping[SlotKey, FormulaOutput],
) -> Dest:
    key = _dest_slot_key(stage)
    if key is None or key not in promotions:
        return stage.out
    return replace(stage.out, slot=-(promotions[key].offset + 1))


def _rewrite_child(
    stage: Stage,
    promotions: Mapping[SlotKey, FormulaOutput],
) -> Stage:
    return replace(
        stage,
        inputs=tuple(
            _rewrite_source(source, promotions) for source in stage.inputs
        ),
        out=_promoted_dest(stage, promotions),
    )


def _rewrite_group(
    group: GroupStage | None,
    promotions: Mapping[SlotKey, FormulaOutput],
) -> GroupStage | None:
    if group is None:
        return None
    return replace(
        group,
        key_sources=tuple(
            _rewrite_source(source, promotions) for source in group.key_sources
        ),
        feed_sources=tuple(
            _rewrite_source(source, promotions) for source in group.feed_sources
        ),
    )


def _rewrite_stage(
    stage: Stage,
    promotions: Mapping[SlotKey, FormulaOutput],
) -> Stage:
    rewritten = _rewrite_child(stage, promotions)
    return replace(
        rewritten,
        group=_rewrite_group(stage.group, promotions),
        members=tuple(
            _rewrite_child(member, promotions) for member in stage.members
        ),
        epilogues=tuple(
            _rewrite_child(epilogue, promotions) for epilogue in stage.epilogues
        ),
    )


def _output_dest(output: FormulaOutput, n_instruments: int) -> Dest:
    slot = -(output.offset + 1)
    shape = output.shape
    if shape == () or shape == (n_instruments,):
        return Dest(slot, size=output.size, shape=shape)
    if len(shape) == 2 and shape[0] == n_instruments:
        return Dest(
            slot,
            matrix=True,
            width=shape[1],
            size=output.size,
            shape=shape,
        )
    return Dest(
        slot,
        tensor=True,
        width=max(1, (output.size + n_instruments - 1) // n_instruments),
        size=output.size,
        shape=shape,
    )


def _output_stage(
    source: Source,
    output: FormulaOutput,
    n_instruments: int,
) -> Stage:
    out = _output_dest(output, n_instruments)
    final_only = output.mode == "final"
    scalar_width = output.shape == () or output.shape == (n_instruments,)

    # A final public output backed by an ordinary row source is the semantic
    # `emit(last)` case. Reuse EmitLastNode so it snapshots each row and returns
    # NaNs when the input has zero rows. Sources that are already final-only
    # (temporal reductions and their suffixes) still project exactly once during
    # finalization through the ordinary Copy/TensorCopy path below.
    if final_only and not source.final_only:
        return Stage(
            "emit_last",
            (source,),
            out,
            n_instruments,
            dtype=source.dtype,
            output_kind=("scalar" if output.shape == () else "tensor"),
            output_width=output.size,
            op=EmitOp("last"),
            final_only=False,
        )

    if source.kind == "cat":
        return Stage(
            "cat",
            source.parts,
            out,
            n_instruments,
            dtype=source.dtype,
            output_kind="matrix",
            output_width=source.width,
            final_only=final_only,
        )
    if source.kind in {"rbf", "future_rbf"}:
        return Stage(
            "cat",
            (source,),
            out,
            n_instruments,
            dtype=source.dtype,
            output_kind="matrix",
            output_width=source.width,
            final_only=final_only,
        )
    return Stage(
        "copy" if scalar_width else "tensor_copy",
        (source,),
        out,
        1 if output.shape == () else n_instruments,
        dtype=source.dtype,
        output_kind=("scalar" if output.shape == () else "tensor"),
        output_width=output.size,
        final_only=final_only,
    )


def _stage_input_slots(stage: Stage) -> frozenset[SlotKey]:
    result: set[SlotKey] = set()
    for candidate in _stage_candidates(stage):
        for source in candidate.inputs:
            _collect_source_slots(source, result)
    if stage.group is not None:
        for source in (*stage.group.key_sources, *stage.group.feed_sources):
            _collect_source_slots(source, result)
    return frozenset(result)


def _fuse_public_ewm_epilogues(stages: list[Stage]) -> list[Stage]:
    """Fuse output-only projections into a single EWM or EWM bundle.

    Public Copy/Cat stages are attached after the normal physical bundling pass.
    Treat a lone EWM as a one-member bundle when those projections are its only
    consumers, so both single and multi-member EWM producers use the same native
    epilogue machinery and avoid scratch/output-copy traffic.
    """

    result: list[Stage] = []
    cursor = 0
    while cursor < len(stages):
        stage = stages[cursor]
        if stage.kind not in {"ewm", "ewm_bundle"} or stage.epilogues:
            result.append(stage)
            cursor += 1
            continue

        members = stage.members if stage.members else (stage,)
        member_outputs = frozenset(
            key
            for member in members
            if (key := _dest_slot_key(member)) is not None
        )
        if not member_outputs:
            result.append(stage)
            cursor += 1
            continue

        epilogues: list[Stage] = []
        following = cursor + 1
        while following < len(stages):
            candidate = stages[following]
            public_dest = candidate.out.slot is not None and candidate.out.slot < 0
            scalar_projection = (
                candidate.kind == "copy"
                and len(candidate.inputs) == 1
                and candidate.inputs[0].width == 1
            ) or (
                candidate.kind == "cat"
                and candidate.inputs
                and all(source.width == 1 for source in candidate.inputs)
            )
            dependencies = _stage_input_slots(candidate)
            if not (
                public_dest
                and scalar_projection
                and dependencies
                and dependencies <= member_outputs
                and candidate.final_only == stage.final_only
            ):
                break
            epilogues.append(candidate)
            following += 1

        if not epilogues:
            result.append(stage)
            cursor += 1
            continue

        future_dependencies = frozenset().union(
            *(_stage_input_slots(later) for later in stages[following:])
        )
        if member_outputs & future_dependencies:
            result.append(stage)
            cursor += 1
            continue

        bundle = (
            stage
            if stage.kind == "ewm_bundle"
            else replace(stage, kind="ewm_bundle", members=members)
        )
        result.append(replace(bundle, epilogues=tuple(epilogues)))
        cursor = following
    return result


def _plan_output_shape(layout: OutputLayout, n_instruments: int) -> tuple[int, ...]:
    if len(layout.outputs) == 1:
        return layout.outputs[0].shape
    if layout.mode != "rows":
        return (layout.row_width + layout.final_width,)
    if layout.row_lane_partitionable:
        return (n_instruments, layout.row_width // n_instruments)
    return (layout.row_width,)


def _slot_maps(
    stages: list[Stage],
) -> tuple[dict[ScalarSlotKey, int], dict[int, int]]:
    scalar_by_dtype: dict[str, set[int]] = defaultdict(set)
    tensor: set[int] = set()

    def add_dest(stage: Stage) -> None:
        slot = stage.out.slot
        if slot is None or slot < 0:
            return
        if stage.out.matrix or stage.out.tensor:
            tensor.add(int(slot))
        else:
            scalar_by_dtype[stage.dtype].add(int(slot))

    for stage in stages:
        for candidate in _stage_candidates(stage):
            add_dest(candidate)

    scalar = {
        (dtype, old): new
        for dtype, slots in scalar_by_dtype.items()
        for new, old in enumerate(sorted(slots))
    }
    return scalar, {old: new for new, old in enumerate(sorted(tensor))}


def _remap_source(
    source: Source,
    scalar_slots: Mapping[ScalarSlotKey, int],
    tensor_slots: Mapping[int, int],
) -> Source:
    value = source.value
    if source.kind == "slot":
        value = scalar_slots[(source.dtype, int(value))]
    elif source.kind in {"matrix_slot", "tensor_slot"}:
        value = tensor_slots[int(value)]
    parts = tuple(
        _remap_source(part, scalar_slots, tensor_slots) for part in source.parts
    )
    if value == source.value and parts == source.parts:
        return source
    return replace(source, value=value, parts=parts)


def _remap_dest(
    dest: Dest,
    dtype: str,
    scalar_slots: Mapping[ScalarSlotKey, int],
    tensor_slots: Mapping[int, int],
) -> Dest:
    slot = dest.slot
    if slot is None or slot < 0:
        return dest
    mapped = (
        tensor_slots[int(slot)]
        if dest.matrix or dest.tensor
        else scalar_slots[(dtype, int(slot))]
    )
    return dest if mapped == slot else replace(dest, slot=mapped)


def _remap_group(
    group: GroupStage | None,
    scalar_slots: Mapping[ScalarSlotKey, int],
    tensor_slots: Mapping[int, int],
) -> GroupStage | None:
    if group is None:
        return None
    return replace(
        group,
        key_sources=tuple(
            _remap_source(source, scalar_slots, tensor_slots)
            for source in group.key_sources
        ),
        feed_sources=tuple(
            _remap_source(source, scalar_slots, tensor_slots)
            for source in group.feed_sources
        ),
    )


def _compact_stage(
    stage: Stage,
    scalar_slots: Mapping[ScalarSlotKey, int],
    tensor_slots: Mapping[int, int],
) -> Stage:
    def child(value: Stage) -> Stage:
        return replace(
            value,
            inputs=tuple(
                _remap_source(source, scalar_slots, tensor_slots)
                for source in value.inputs
            ),
            out=_remap_dest(
                value.out, value.dtype, scalar_slots, tensor_slots
            ),
        )

    return replace(
        stage,
        inputs=tuple(
            _remap_source(source, scalar_slots, tensor_slots)
            for source in stage.inputs
        ),
        out=_remap_dest(stage.out, stage.dtype, scalar_slots, tensor_slots),
        group=_remap_group(stage.group, scalar_slots, tensor_slots),
        members=tuple(child(member) for member in stage.members),
        epilogues=tuple(child(epilogue) for epilogue in stage.epilogues),
    )


def _compact_scratch(stages: list[Stage]) -> tuple[list[Stage], int, int, int]:
    scalar_slots, tensor_slots = _slot_maps(stages)
    compact = [
        _compact_stage(stage, scalar_slots, tensor_slots) for stage in stages
    ]
    matrix_width = 1
    for stage in compact:
        for candidate in _stage_candidates(stage):
            if (
                candidate.out.slot is not None
                and candidate.out.slot >= 0
                and (candidate.out.matrix or candidate.out.tensor)
            ):
                matrix_width = max(matrix_width, candidate.out.width)
    return compact, len(scalar_slots), len(tensor_slots), matrix_width


def _can_promote(source: Source, output: FormulaOutput) -> bool:
    if source.dtype != "float64":
        return False
    if output.mode == "rows":
        return not source.final_only
    return source.final_only


def lower_program(
    program: Program,
    *,
    n_instruments: int,
    default_group_capacity: int = 64,
    key_cardinalities: Mapping[str, int] | None = None,
    row_scalar_nodes: frozenset[int] | None = None,
    input_dtypes: tuple[str, ...] | None = None,
) -> Plan:
    """Lower one or many public roots through one physical DAG and output pass."""

    layout = build_output_layout(program, n_instruments)
    physical, exposed_ids = _physical_program(program)
    plan, exposed_sources = lower_graph(
        physical,
        exposed_node_ids=exposed_ids,
        n_instruments=n_instruments,
        default_group_capacity=default_group_capacity,
        key_cardinalities=key_cardinalities,
        row_scalar_nodes=row_scalar_nodes,
        input_dtypes=input_dtypes,
    )

    stages = list(plan.stages)
    producers = _promotable_producers(stages)
    consumed = _consumed_slot_keys(stages)
    public_dependencies = _public_dependency_slots(exposed_sources)
    public_key_counts = Counter(
        key
        for source in exposed_sources
        if (key := _source_slot_key(source)) is not None
    )
    promotions: dict[SlotKey, FormulaOutput] = {}
    for source, output in zip(exposed_sources, layout.outputs):
        key = _source_slot_key(source)
        if (
            key is not None
            and key in producers
            and key not in consumed
            and key not in public_dependencies
            and key not in promotions
            # PackedOutputSrc is deliberately a row-region source. A final-only
            # producer may own final storage directly only when no second public
            # root needs to read the same physical value during finalization.
            and (output.mode == "rows" or public_key_counts[key] == 1)
            and _can_promote(source, output)
        ):
            promotions[key] = output

    rewritten = [_rewrite_stage(stage, promotions) for stage in stages]

    # Output projections run after the physical DAG. For exact duplicate row
    # roots that do not already have a promoted physical owner, let the first
    # projection own the packed row result and make later duplicates read that
    # result. This preserves IR CSE for lazy stateless roots without inserting a
    # scratch materialization or making any hot downstream computation read mmap.
    row_projection_owners: dict[Source, FormulaOutput] = {}
    for source, output in zip(exposed_sources, layout.outputs):
        key = _source_slot_key(source)
        if key is not None and promotions.get(key) == output:
            continue
        rewritten_source = _rewrite_source(source, promotions)
        if output.mode == "rows" and not rewritten_source.final_only:
            owner = row_projection_owners.get(rewritten_source)
            if owner is None:
                row_projection_owners[rewritten_source] = output
            else:
                rewritten_source = _packed_source(rewritten_source, owner)
        rewritten.append(_output_stage(rewritten_source, output, n_instruments))

    rewritten = _fuse_public_ewm_epilogues(rewritten)
    rewritten, _, matrix_slots, matrix_width = _compact_scratch(rewritten)
    scratch_slots = scalar_scratch_slots(rewritten)
    total_width = layout.row_width + layout.final_width
    if len(layout.outputs) == 1:
        root_type = program.nodes[layout.outputs[0].root_id].value_type
        output_kind = root_type.kind
        output_width = int(root_type.width)
    else:
        output_kind = "tensor"
        output_width = total_width

    return replace(
        plan,
        stages=tuple(rewritten),
        scratch_slots=scratch_slots,
        matrix_scratch_slots=matrix_slots,
        matrix_scratch_width=matrix_width,
        output_kind=output_kind,
        output_width=output_width,
        output_row_width=(layout.row_width or layout.final_width),
        output_shape=_plan_output_shape(layout, n_instruments),
        output_mode=layout.mode,
    )


__all__ = ["lower_program"]