from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from trading_dsl_engine.cpp_stream.python import lowering as base
from trading_dsl_engine.cpp_stream.python.lowering import (
    CppStreamLoweringError,
    Dest,
    GroupStage,
    Plan,
    Source,
    Stage,
    infer_node_dtypes,
)
from trading_dsl_engine.ir.einsum import build_contraction_plan
from trading_dsl_engine.ir.ops import (
    CatOp,
    ColumnOp,
    CumsumOp,
    CustomCallOp,
    EmitOp,
    EinsumOp,
    EwmOp,
    FFillOp,
    FutureRbfBasisSumOp,
    HumpOp,
    GroupByOp,
    InputOp,
    InstrumentBasisMeanOp,
    InstrumentBasisProjectionOp,
    LiteralOp,
    LinearFilterOp,
    NaryOp,
    RbfBasisOp,
    PeriodsSinceChangeOp,
    ReductionOp,
    RidgeOp,
    RidgeProjectionOp,
    RollingDecayOp,
    RollingEntropyOp,
    RollingKthOp,
    RollingOp,
    RollingPrevDiffOp,
    RollingProductOp,
    ShiftOp,
    TheilSenOp,
    TradeWhenOp,
    VectorQuantileOp,
    XsPctRankOp,
    XsAggregateOp,
    XsWeightedMeanOp,
    XsProjectionOp,
    XsGeneralizedRankOp,
    XsDensifyOp,
    XsRankOp,
)
from trading_dsl_engine.ir.program import Program
from trading_dsl_engine.ir.types import resolve_shape, shape_size


def _build_plan(
    program: Program,
    *,
    n_instruments: int,
    default_group_capacity: int,
    key_cardinalities: Mapping[str, int] | None,
    grouped: bool,
    row_scalar_nodes: frozenset[int],
    input_dtypes: tuple[str, ...],
    node_dtypes: tuple[str, ...],
    materialize_root: bool = True,
    exposed_node_ids: frozenset[int] = frozenset(),
    exposed_sources: dict[int, Source] | None = None,
) -> Plan:
    """Lower one DAG.

    ``materialize_root`` preserves the established grouped/single-plan behavior.
    The public multi-output layer sets it to ``False`` and asks for the lowered
    Sources of its roots instead, so there is no synthetic root or output anchor.
    ``exposed_node_ids`` also protects externally consumed scratch values from
    epilogue fusion that would otherwise discard them.
    """

    root = program.output_id
    sources: dict[int, Source] = {}
    stages: list[Stage] = []
    next_slot = 0
    next_matrix_slot = 0
    max_matrix_width = 1
    materialized_sources: dict[Source, Source] = {}

    def source_slot_dependencies(source: Source) -> frozenset[tuple[str, int]]:
        dependencies: set[tuple[str, int]] = set()
        if source.kind == "slot":
            dependencies.add(("scalar", int(source.value)))
        elif source.kind in {"matrix_slot", "tensor_slot"}:
            dependencies.add(("tensor", int(source.value)))
        for part in source.parts:
            dependencies.update(source_slot_dependencies(part))
        return frozenset(dependencies)

    def scalar_dest(is_root: bool, shape: tuple[int, ...] = ()) -> Dest:
        nonlocal next_slot
        if is_root:
            return Dest(None, size=shape_size(shape), shape=shape)
        slot = next_slot
        next_slot += 1
        return Dest(slot, size=shape_size(shape), shape=shape)

    def matrix_dest(is_root: bool, width: int) -> Dest:
        nonlocal next_matrix_slot, max_matrix_width
        max_matrix_width = max(max_matrix_width, width)
        shape = (n_instruments, width)
        if is_root:
            return Dest(
                None,
                matrix=True,
                width=width,
                size=n_instruments * width,
                shape=shape,
            )
        slot = next_matrix_slot
        next_matrix_slot += 1
        return Dest(
            slot,
            matrix=True,
            width=width,
            size=n_instruments * width,
            shape=shape,
        )

    def tensor_dest(is_root: bool, shape: tuple[int, ...]) -> Dest:
        nonlocal next_matrix_slot, max_matrix_width
        size = shape_size(shape)
        width = max(1, (size + n_instruments - 1) // n_instruments)
        max_matrix_width = max(max_matrix_width, width)
        if is_root:
            return Dest(None, tensor=True, width=width, size=size, shape=shape)
        slot = next_matrix_slot
        next_matrix_slot += 1
        return Dest(
            slot,
            tensor=True,
            width=width,
            size=size,
            shape=shape,
        )

    def value_dest(is_root: bool, shape: tuple[int, ...]) -> Dest:
        if shape == () or shape == (n_instruments,):
            return scalar_dest(is_root, shape)
        if len(shape) == 2 and shape[0] == n_instruments:
            return matrix_dest(is_root, shape[1])
        return tensor_dest(is_root, shape)

    def source_from_dest(
        dest: Dest,
        shape: tuple[int, ...],
        *,
        dtype: str = "float64",
        final_only: bool = False,
    ) -> Source:
        if dest.slot is None:
            return Source(
                "output",
                -1,
                dtype=dtype,
                width=max(1, dest.width),
                shape=shape,
                final_only=final_only,
            )
        if dest.tensor:
            return Source(
                "tensor_slot",
                dest.slot,
                dtype=dtype,
                width=dest.size,
                shape=shape,
                final_only=final_only,
            )
        if dest.matrix:
            return Source(
                "matrix_slot",
                dest.slot,
                dtype=dtype,
                width=dest.width,
                shape=shape,
                final_only=final_only,
            )
        return Source(
            "slot",
            dest.slot,
            row_scalar=shape == (),
            dtype=dtype,
            width=1,
            shape=shape,
            final_only=final_only,
        )

    def scalar_width_shape(shape: tuple[int, ...]) -> bool:
        return shape == () or shape == (n_instruments,)

    def materialize_source(
        source: Source,
        shape: tuple[int, ...],
        *,
        dtype: str,
    ) -> Source:
        """Materialize a lazy expression at a pointer ABI boundary."""

        if source.kind not in {"expression", "stateless_expression"}:
            return source
        previous = materialized_sources.get(source)
        if previous is not None:
            return previous
        scalar_width = scalar_width_shape(shape)
        out = value_dest(False, shape)
        stages.append(
            Stage(
                "copy" if scalar_width else "tensor_copy",
                (source,),
                out,
                1 if shape == () else n_instruments,
                dtype=dtype,
                output_kind="scalar" if shape == () else "vector",
                final_only=source.final_only,
            )
        )
        materialized = source_from_dest(
            out,
            shape,
            dtype=dtype,
            final_only=source.final_only,
        )
        materialized_sources[source] = materialized
        return materialized

    def bundle_physical_stages(
        items: list[Stage],
        preserve_slots: frozenset[tuple[str, int]],
    ) -> list[Stage]:
        """Bundle compatible sibling state machines without reordering stages."""

        bundled: list[Stage] = []
        cursor = 0
        while cursor < len(items):
            first = items[cursor]
            bundle_kind = {
                "ewm": "ewm_bundle",
                "reduce": "reduction_bundle",
                "ridge": "ridge_bundle",
            }.get(first.kind)
            if bundle_kind is None:
                bundled.append(first)
                cursor += 1
                continue
            members = [first]
            produced: set[tuple[str, int]] = set()
            if first.out.slot is not None:
                produced.add(
                    (
                        "tensor"
                        if first.out.matrix or first.out.tensor
                        else "scalar",
                        first.out.slot,
                    )
                )
            following = cursor + 1
            while following < len(items):
                candidate = items[following]
                if (
                    candidate.kind != first.kind
                    or candidate.op != first.op
                    or candidate.lane_count != first.lane_count
                    or candidate.dtype != first.dtype
                    or candidate.final_only != first.final_only
                    or (
                        first.kind == "reduce"
                        and candidate.inputs[0].shape != first.inputs[0].shape
                    )
                    or (
                        first.kind == "ridge"
                        and (
                            candidate.inputs != first.inputs
                            or candidate.half_life != first.half_life
                            or candidate.ridge_lambda != first.ridge_lambda
                        )
                    )
                    or any(
                        source_slot_dependencies(source) & produced
                        for source in candidate.inputs
                    )
                ):
                    break
                members.append(candidate)
                if candidate.out.slot is not None:
                    produced.add(
                        (
                            "tensor"
                            if candidate.out.matrix or candidate.out.tensor
                            else "scalar",
                            candidate.out.slot,
                        )
                    )
                following += 1
            if len(members) == 1:
                bundled.append(first)
            else:
                bundled.append(
                    replace(
                        first,
                        kind=bundle_kind,
                        inputs=tuple(
                            source
                            for member in members
                            for source in member.inputs
                        ),
                        members=tuple(members),
                    )
                )
            cursor = following

        fused: list[Stage] = []
        cursor = 0
        while cursor < len(bundled):
            stage = bundled[cursor]
            if stage.kind != "ewm_bundle" or cursor + 1 >= len(bundled):
                fused.append(stage)
                cursor += 1
                continue
            epilogue = bundled[cursor + 1]
            scalar_epilogue = (
                epilogue.kind == "copy"
                and len(epilogue.inputs) == 1
                and epilogue.inputs[0].width == 1
            ) or (
                epilogue.kind == "cat"
                and epilogue.inputs
                and all(source.width == 1 for source in epilogue.inputs)
            )
            member_outputs = {
                ("scalar", member.out.slot)
                for member in stage.members
                if member.out.slot is not None
                and not member.out.matrix
                and not member.out.tensor
            }
            epilogue_dependencies = frozenset().union(
                *(
                    source_slot_dependencies(source)
                    for source in epilogue.inputs
                )
            )
            future_dependencies = frozenset().union(
                *(
                    source_slot_dependencies(source)
                    for later in bundled[cursor + 2 :]
                    for source in later.inputs
                )
            )
            if (
                scalar_epilogue
                and epilogue_dependencies
                and epilogue_dependencies <= member_outputs
                and not (member_outputs & future_dependencies)
                and not (member_outputs & preserve_slots)
                and epilogue.final_only == stage.final_only
            ):
                fused.append(replace(stage, epilogues=(epilogue,)))
                cursor += 2
            else:
                fused.append(stage)
                cursor += 1
        return fused

    for node_id, node in enumerate(program.nodes):
        op = node.op
        dtype = node_dtypes[node_id]
        row_scalar = node_id in row_scalar_nodes
        lane_count = 1 if row_scalar else n_instruments
        node_shape = resolve_shape(node.value_type, n_instruments)

        if isinstance(op, InputOp):
            sources[node_id] = Source(
                "input",
                op.input_index,
                row_scalar,
                input_dtypes[op.input_index],
                1,
                node_shape,
            )
            continue
        if isinstance(op, LiteralOp):
            sources[node_id] = Source(
                "literal",
                base._coerce_literal_for_dtype(op.value, dtype),
                True,
                dtype,
                1,
                (),
            )
            continue

        children = tuple(sources[child] for child in node.child_ids)
        final_only = any(child.final_only for child in children)
        is_root = materialize_root and node_id == root

        if isinstance(op, RbfBasisOp):
            source = Source(
                "rbf",
                width=op.n_basis,
                shape=node_shape,
                parts=children,
                op=op,
                final_only=final_only,
            )
            sources[node_id] = source
            if is_root:
                stages.append(
                    Stage(
                        "cat",
                        (source,),
                        matrix_dest(True, op.n_basis),
                        n_instruments,
                        output_kind="matrix",
                        output_width=op.n_basis,
                        final_only=final_only,
                    )
                )
            continue

        if isinstance(op, FutureRbfBasisSumOp):
            source = Source(
                "future_rbf",
                width=op.n_basis,
                shape=node_shape,
                parts=children,
                op=op,
                final_only=final_only,
            )
            sources[node_id] = source
            if is_root:
                stages.append(
                    Stage(
                        "cat",
                        (source,),
                        matrix_dest(True, op.n_basis),
                        n_instruments,
                        output_kind="matrix",
                        output_width=op.n_basis,
                        final_only=final_only,
                    )
                )
            continue

        if isinstance(op, CatOp):
            parts: list[Source] = []
            for child in children:
                feature = (
                    materialize_source(
                        child, child.shape, dtype=child.dtype
                    )
                    if child.kind in {"expression", "stateless_expression"}
                    and not scalar_width_shape(child.shape)
                    else child
                )
                parts.extend(base._flatten_features(feature))
            width = sum(part.width for part in parts)
            source = Source(
                "cat",
                width=width,
                shape=node_shape,
                parts=tuple(parts),
                op=op,
                final_only=final_only,
            )
            sources[node_id] = source
            if is_root:
                stages.append(
                    Stage(
                        "cat",
                        tuple(parts),
                        matrix_dest(True, width),
                        n_instruments,
                        output_kind="matrix",
                        output_width=width,
                        final_only=final_only,
                    )
                )
            continue

        if isinstance(op, ReductionOp):
            out = value_dest(is_root, node_shape)
            stages.append(
                Stage(
                    "reduce",
                    children,
                    out,
                    n_instruments,
                    output_kind=node.value_type.kind,
                    output_width=int(node.value_type.width),
                    op=op,
                    final_only=final_only,
                )
            )
            sources[node_id] = source_from_dest(
                out,
                node_shape,
                final_only=final_only or op.temporal,
            )
            continue

        if isinstance(op, EmitOp):
            if not is_root:
                raise CppStreamLoweringError(
                    "emit('last') must be the terminal output"
                )
            out = value_dest(True, node_shape)
            stages.append(
                Stage(
                    "emit_last",
                    children,
                    out,
                    n_instruments,
                    output_kind=node.value_type.kind,
                    output_width=int(node.value_type.width),
                    op=op,
                    final_only=final_only,
                )
            )
            sources[node_id] = source_from_dest(
                out, node_shape, final_only=final_only
            )
            continue

        if isinstance(op, InstrumentBasisMeanOp):
            sources[node_id] = Source(
                "instrument_basis",
                width=op.feature_width,
                shape=(),
                parts=children,
                op=op,
                final_only=final_only,
            )
            if is_root:
                raise CppStreamLoweringError(
                    "InstrumentBasisMean object must be projected with get_beta/get_preds"
                )
            continue

        if isinstance(op, RidgeOp):
            sources[node_id] = Source(
                "ridge",
                width=op.coefficient_width,
                shape=(),
                parts=children,
                op=op,
                final_only=final_only,
            )
            if is_root:
                raise CppStreamLoweringError("Ridge object must be projected")
            continue

        if isinstance(op, InstrumentBasisProjectionOp):
            object_source = children[0]
            if object_source.kind != "instrument_basis" or not isinstance(
                object_source.op, InstrumentBasisMeanOp
            ):
                raise CppStreamLoweringError(
                    "InstrumentBasis projection lost object source"
                )
            basis_op = object_source.op
            object_children = object_source.parts
            feature_object = object_children[0]
            if (
                feature_object.kind in {"expression", "stateless_expression"}
                and not scalar_width_shape(feature_object.shape)
            ):
                feature_object = materialize_source(
                    feature_object,
                    feature_object.shape,
                    dtype=feature_object.dtype,
                )
            feature_sources = base._flatten_features(feature_object)
            y_source = object_children[1]
            if basis_op.has_weights:
                weight_source = object_children[2]
                hl_source = object_children[3]
            else:
                weight_source = Source(
                    "literal", 1.0, True, "float64", 1, ()
                )
                hl_source = object_children[2]
            half_life = base._literal_scalar(
                hl_source, "InstrumentBasisMean hl"
            )
            out = value_dest(is_root, node_shape)
            stages.append(
                Stage(
                    "instrument_basis",
                    tuple(feature_sources) + (y_source, weight_source),
                    out,
                    n_instruments,
                    output_kind=node.value_type.kind,
                    output_width=int(node.value_type.width),
                    op=basis_op,
                    projection=op.field,
                    half_life=half_life,
                    final_only=final_only,
                )
            )
            sources[node_id] = source_from_dest(
                out, node_shape, final_only=final_only
            )
            continue

        if isinstance(op, RidgeProjectionOp):
            object_source = children[0]
            if object_source.kind != "ridge" or not isinstance(
                object_source.op, RidgeOp
            ):
                raise CppStreamLoweringError("Ridge projection lost object source")
            ridge_op = object_source.op
            object_children = object_source.parts
            feature_count = len(ridge_op.feature_widths)
            feature_sources: list[Source] = []
            for feature in object_children[:feature_count]:
                if (
                    feature.kind in {"expression", "stateless_expression"}
                    and not scalar_width_shape(feature.shape)
                ):
                    feature = materialize_source(
                        feature, feature.shape, dtype=feature.dtype
                    )
                feature_sources.extend(base._flatten_features(feature))
            y_source = object_children[feature_count]
            cursor = feature_count + 1
            if ridge_op.has_weights:
                weight_source = object_children[cursor]
                cursor += 1
            else:
                weight_source = Source(
                    "literal", 1.0, True, "float64", 1, ()
                )
            half_life = base._literal_scalar(
                object_children[cursor], "Ridge hl"
            )
            ridge_lambda = base._literal_scalar(
                object_children[cursor + 1], "Ridge lambda"
            )
            out = value_dest(is_root, node_shape)
            stages.append(
                Stage(
                    "ridge",
                    tuple(feature_sources) + (y_source, weight_source),
                    out,
                    n_instruments,
                    output_kind=node.value_type.kind,
                    output_width=int(node.value_type.width),
                    op=ridge_op,
                    projection=op.field,
                    projection_component=op.component,
                    half_life=half_life,
                    ridge_lambda=ridge_lambda,
                    final_only=final_only,
                )
            )
            sources[node_id] = source_from_dest(
                out, node_shape, final_only=final_only
            )
            continue

        if isinstance(op, EinsumOp):
            contraction = build_contraction_plan(
                op.spec, tuple(child.shape for child in children)
            )
            terms = list(children)
            final_source: Source | None = None
            for step_index, step in enumerate(contraction.steps):
                selected = tuple(
                    terms[position] for position in step.operand_positions
                )
                final_step = step_index == len(contraction.steps) - 1
                out = value_dest(is_root and final_step, step.output_shape)
                stages.append(
                    Stage(
                        "einsum",
                        selected,
                        out,
                        n_instruments,
                        output_kind=(
                            node.value_type.kind if final_step else "tensor"
                        ),
                        output_width=(
                            int(node.value_type.width)
                            if final_step
                            else max(1, shape_size(step.output_shape))
                        ),
                        op=op,
                        einsum_step=step,
                        final_only=final_only,
                    )
                )
                selected_positions = set(step.operand_positions)
                terms = [
                    term
                    for position, term in enumerate(terms)
                    if position not in selected_positions
                ]
                final_source = source_from_dest(
                    out,
                    step.output_shape,
                    final_only=final_only,
                )
                terms.append(final_source)
            assert final_source is not None
            sources[node_id] = final_source
            continue

        if isinstance(op, NaryOp):
            scalar_width = (
                scalar_width_shape(node_shape)
                and all(child.width == 1 for child in children)
            )
            expression_width = 1
            if not scalar_width:
                expression_width = (
                    node_shape[1]
                    if len(node_shape) == 2 and node_shape[0] == n_instruments
                    else shape_size(node_shape)
                )
            expression = Source(
                "expression",
                row_scalar=row_scalar,
                dtype=dtype,
                width=expression_width,
                shape=node_shape,
                parts=children,
                op=op,
                final_only=final_only,
            )
            if not is_root:
                sources[node_id] = expression
                continue
            out = value_dest(True, node_shape)
            stage = Stage(
                "copy" if scalar_width else "tensor_copy",
                (expression,),
                out,
                lane_count,
                dtype=dtype,
                output_kind=node.value_type.kind,
                output_width=int(node.value_type.width),
                final_only=final_only,
            )
        elif isinstance(op, CustomCallOp):
            if any(child.width != 1 for child in children):
                raise CppStreamLoweringError(
                    "named stateless calls currently require scalar-width inputs"
                )
            expression = Source(
                "stateless_expression",
                row_scalar=row_scalar,
                dtype=dtype,
                width=1,
                shape=node_shape,
                parts=children,
                op=op,
                final_only=final_only,
            )
            if not is_root:
                sources[node_id] = expression
                continue
            out = scalar_dest(True, node_shape)
            stage = Stage(
                "copy",
                (expression,),
                out,
                lane_count,
                dtype=dtype,
                output_kind=node.value_type.kind,
                output_width=int(node.value_type.width),
                final_only=final_only,
            )
        elif isinstance(op, CumsumOp):
            out = value_dest(is_root, node_shape)
            stage = Stage(
                "cumsum" if scalar_width_shape(node_shape) else "tensor_cumsum",
                children,
                out,
                lane_count,
                op=op,
            )
        elif isinstance(op, FFillOp):
            out = value_dest(is_root, node_shape)
            stage = Stage(
                "ffill" if scalar_width_shape(node_shape) else "tensor_ffill",
                children,
                out,
                lane_count,
                op=op,
            )
        elif isinstance(op, ShiftOp):
            out = value_dest(is_root, node_shape)
            stage = Stage(
                "shift" if scalar_width_shape(node_shape) else "tensor_shift",
                children,
                out,
                lane_count,
                op=op,
            )
        elif isinstance(op, EwmOp):
            out = value_dest(is_root, node_shape)
            stage = Stage(
                "ewm" if scalar_width_shape(node_shape) else "tensor_ewm",
                children,
                out,
                lane_count,
                op=op,
            )
        elif isinstance(
            op,
            (
                PeriodsSinceChangeOp,
                HumpOp,
                TradeWhenOp,
                LinearFilterOp,
                RollingProductOp,
                RollingKthOp,
                RollingPrevDiffOp,
                RollingDecayOp,
                RollingEntropyOp,
            ),
        ):
            if not scalar_width_shape(node_shape):
                raise CppStreamLoweringError(
                    f"{type(op).__name__} currently requires scalar/vector inputs"
                )
            stage_kind = {
                PeriodsSinceChangeOp: "periods_since_change",
                HumpOp: "hump",
                TradeWhenOp: "trade_when",
                LinearFilterOp: "linear_filter",
                RollingProductOp: "rolling_product",
                RollingKthOp: "rolling_kth",
                RollingPrevDiffOp: "rolling_prev_diff",
                RollingDecayOp: "rolling_decay",
                RollingEntropyOp: "rolling_entropy",
            }[type(op)]
            out = value_dest(is_root, node_shape)
            stage = Stage(stage_kind, children, out, lane_count, op=op)
        elif isinstance(op, RollingOp):
            if not scalar_width_shape(node_shape):
                raise CppStreamLoweringError(
                    f"rolling_{op.kind} currently requires a scalar/vector input"
                )
            out = value_dest(is_root, node_shape)
            stage = Stage(
                "rolling",
                children,
                out,
                lane_count,
                op=op,
            )
        elif isinstance(op, VectorQuantileOp):
            out = value_dest(is_root, node_shape)
            stage = Stage(
                "vector_quantile",
                children,
                out,
                lane_count,
                op=op,
            )
        elif isinstance(op, ColumnOp):
            out = value_dest(is_root, node_shape)
            stage = Stage("tensor_column", children, out, lane_count, op=op)
        elif isinstance(op, XsAggregateOp):
            out = value_dest(is_root, node_shape)
            stage = Stage(
                "xs_aggregate", children, out, lane_count, op=op
            )
        elif isinstance(op, XsWeightedMeanOp):
            out = value_dest(is_root, node_shape)
            stage = Stage(
                "xs_weighted_mean", children, out, lane_count, op=op
            )
        elif isinstance(op, XsProjectionOp):
            out = value_dest(is_root, node_shape)
            stage = Stage(
                "xs_projection", children, out, lane_count, op=op
            )
        elif isinstance(op, XsGeneralizedRankOp):
            out = value_dest(is_root, node_shape)
            stage = Stage(
                "xs_generalized_rank", children, out, lane_count, op=op
            )
        elif isinstance(op, XsDensifyOp):
            out = value_dest(is_root, node_shape)
            stage = Stage(
                "xs_densify", children, out, lane_count, op=op
            )
        elif isinstance(op, TheilSenOp):
            if not scalar_width_shape(node_shape):
                raise CppStreamLoweringError(
                    "rolling_theilsen currently requires scalar/vector inputs"
                )
            out = value_dest(is_root, node_shape)
            stage = Stage(
                "theilsen",
                children,
                out,
                lane_count,
                op=op,
            )
        elif isinstance(op, XsRankOp):
            if node_shape != (n_instruments,):
                raise CppStreamLoweringError(
                    "xs_rank requires an instrument vector"
                )
            out = scalar_dest(is_root, node_shape)
            stage = Stage("xs_rank", children, out, lane_count, op=op)
        elif isinstance(op, XsPctRankOp):
            if node_shape != (n_instruments,):
                raise CppStreamLoweringError(
                    "xs_pct_rank requires an instrument vector"
                )
            out = scalar_dest(is_root, node_shape)
            stage = Stage("xs_pct_rank", children, out, lane_count, op=op)
        elif isinstance(op, GroupByOp):
            if grouped:
                raise CppStreamLoweringError("nested groupby is not supported")
            key_count = op.n_dynamic_keys
            materialized_children = tuple(
                materialize_source(
                    source,
                    source.shape,
                    dtype=source.dtype,
                )
                for source in children
            )
            key_sources = materialized_children[:key_count]
            feed_sources = materialized_children[key_count:]
            if any(
                source.width != 1 for source in key_sources + feed_sources
            ):
                raise CppStreamLoweringError(
                    "groupby keys/lhs/captures must be scalar-width vectors"
                )
            specs = base._resolved_key_specs(
                program, node.child_ids, op, key_cardinalities
            )
            monotonic_specs = tuple(spec for spec in specs if spec.monotonic)
            if any(spec.row_scalar is not True for spec in monotonic_specs):
                raise CppStreamLoweringError(
                    "monotonic group keys require row_scalar=True"
                )
            retained_specs = tuple(spec for spec in specs if not spec.monotonic)
            dense = bool(retained_specs) and all(
                spec.num_keys is not None for spec in retained_specs
            )
            capacity = (
                1
                if not retained_specs
                else base._dense_capacity(retained_specs)
                if dense
                else (op.capacity or default_group_capacity)
            )
            inner_dtypes = ("float64",) * len(op.inner_program.input_names)
            inner = _build_plan(
                op.inner_program,
                n_instruments=n_instruments,
                default_group_capacity=default_group_capacity,
                key_cardinalities=key_cardinalities,
                grouped=True,
                row_scalar_nodes=frozenset(),
                input_dtypes=inner_dtypes,
                node_dtypes=infer_node_dtypes(op.inner_program, inner_dtypes),
            )
            group_stage = GroupStage(
                inner,
                base._partitions(op.static_groups, n_instruments),
                capacity,
                op.hash_capacity or 0,
                key_sources,
                specs,
                feed_sources,
                dense,
            )
            out = scalar_dest(is_root, node_shape)
            stage = Stage(
                "groupby",
                materialized_children,
                out,
                n_instruments,
                output_kind=node.value_type.kind,
                output_width=int(node.value_type.width),
                group=group_stage,
            )
        else:
            raise CppStreamLoweringError(
                f"unsupported IR op {type(op).__name__}"
            )

        if final_only:
            stage = replace(stage, final_only=True)
        stages.append(stage)
        sources[node_id] = source_from_dest(
            stage.out,
            node_shape,
            dtype=stage.dtype,
            final_only=final_only,
        )

    root_type = program.nodes[root].value_type
    root_shape = resolve_shape(root_type, n_instruments)
    if materialize_root and isinstance(program.nodes[root].op, (InputOp, LiteralOp)):
        source = sources[root]
        tensor_copy = not scalar_width_shape(root_shape)
        out = value_dest(True, root_shape)
        stages.append(
            Stage(
                "tensor_copy" if tensor_copy else "copy",
                (source,),
                out,
                1 if root_shape == () else n_instruments,
                dtype=source.dtype,
                output_kind=root_type.kind,
                output_width=int(root_type.width),
                final_only=source.final_only,
            )
        )

    if exposed_sources is not None:
        exposed_sources.update(sources)

    preserve_slots = frozenset().union(
        *(
            source_slot_dependencies(sources[node_id])
            for node_id in exposed_node_ids
            if node_id in sources
        )
    )
    return Plan(
        tuple(bundle_physical_stages(stages, preserve_slots)),
        next_slot,
        next_matrix_slot,
        max_matrix_width,
        len(program.input_names),
        root_type.kind,
        int(root_type.width),
        base._output_row_width(root_type, n_instruments),
        root_shape,
        "final"
        if isinstance(program.nodes[root].op, EmitOp)
        or (
            isinstance(program.nodes[root].op, ReductionOp)
            and program.nodes[root].op.temporal
        )
        else "rows",
    )


def _validate_lowering_args(
    program: Program,
    n_instruments: int,
    default_group_capacity: int,
    input_dtypes: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if n_instruments <= 0 or default_group_capacity <= 0:
        raise CppStreamLoweringError(
            "n_instruments and group capacity must be > 0"
        )
    return (
        ("float64",) * len(program.input_names)
        if input_dtypes is None
        else input_dtypes
    )


def lower_program(
    program: Program,
    *,
    n_instruments: int,
    default_group_capacity: int = 64,
    key_cardinalities: Mapping[str, int] | None = None,
    row_scalar_nodes: frozenset[int] | None = None,
    input_dtypes: tuple[str, ...] | None = None,
) -> Plan:
    input_dtypes = _validate_lowering_args(
        program, n_instruments, default_group_capacity, input_dtypes
    )
    return _build_plan(
        program,
        n_instruments=n_instruments,
        default_group_capacity=default_group_capacity,
        key_cardinalities=key_cardinalities,
        grouped=False,
        row_scalar_nodes=row_scalar_nodes or frozenset(),
        input_dtypes=input_dtypes,
        node_dtypes=infer_node_dtypes(program, input_dtypes),
    )


def lower_graph(
    program: Program,
    *,
    exposed_node_ids: tuple[int, ...],
    n_instruments: int,
    default_group_capacity: int = 64,
    key_cardinalities: Mapping[str, int] | None = None,
    row_scalar_nodes: frozenset[int] | None = None,
    input_dtypes: tuple[str, ...] | None = None,
) -> tuple[Plan, tuple[Source, ...]]:
    """Lower one complete DAG and expose selected lowered values as Sources."""

    input_dtypes = _validate_lowering_args(
        program, n_instruments, default_group_capacity, input_dtypes
    )
    table: dict[int, Source] = {}
    plan = _build_plan(
        program,
        n_instruments=n_instruments,
        default_group_capacity=default_group_capacity,
        key_cardinalities=key_cardinalities,
        grouped=False,
        row_scalar_nodes=row_scalar_nodes or frozenset(),
        input_dtypes=input_dtypes,
        node_dtypes=infer_node_dtypes(program, input_dtypes),
        materialize_root=False,
        exposed_node_ids=frozenset(exposed_node_ids),
        exposed_sources=table,
    )
    try:
        values = tuple(table[node_id] for node_id in exposed_node_ids)
    except KeyError as exc:
        raise CppStreamLoweringError(
            f"requested lowered source {exc.args[0]} was not produced"
        ) from exc
    return plan, values


__all__ = ["lower_graph", "lower_program"]