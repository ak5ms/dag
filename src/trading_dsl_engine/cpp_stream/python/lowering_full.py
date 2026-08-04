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
    CumsumOp,
    CustomCallOp,
    EmitOp,
    EinsumOp,
    EwmOp,
    FFillOp,
    FutureRbfBasisSumOp,
    GroupByOp,
    InputOp,
    InstrumentBasisMeanOp,
    InstrumentBasisProjectionOp,
    LiteralOp,
    NaryOp,
    RbfBasisOp,
    ReductionOp,
    RidgeOp,
    RidgeProjectionOp,
    ShiftOp,
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
) -> Plan:
    root = program.output_id
    sources: dict[int, Source] = {}
    stages: list[Stage] = []
    next_slot = 0
    next_matrix_slot = 0
    max_matrix_width = 1

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
        is_root = node_id == root

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
                parts.extend(base._flatten_features(child))
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
            feature_sources = base._flatten_features(object_children[0])
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
            out = value_dest(is_root, node_shape)
            scalar_width = (
                scalar_width_shape(node_shape)
                and all(child.width == 1 for child in children)
            )
            if scalar_width:
                kind = (
                    "unary"
                    if op.arity == 1
                    else "binary"
                    if op.arity == 2
                    else "ternary"
                )
            else:
                kind = (
                    "tensor_unary"
                    if op.arity == 1
                    else "tensor_binary"
                    if op.arity == 2
                    else "tensor_ternary"
                )
            stage = Stage(
                kind,
                children,
                out,
                lane_count,
                dtype=dtype,
                op_name=op.name,
                output_kind=node.value_type.kind,
                output_width=int(node.value_type.width),
            )
        elif isinstance(op, CustomCallOp):
            if any(child.width != 1 for child in children):
                raise CppStreamLoweringError(
                    "named stateless calls currently require scalar-width inputs"
                )
            out = scalar_dest(is_root, node_shape)
            stage = Stage(
                "custom",
                children,
                out,
                lane_count,
                op_name=op.name,
                op=op,
                output_kind=node.value_type.kind,
                output_width=int(node.value_type.width),
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
        elif isinstance(op, XsRankOp):
            if node_shape != (n_instruments,):
                raise CppStreamLoweringError(
                    "xs_rank requires an instrument vector"
                )
            out = scalar_dest(is_root, node_shape)
            stage = Stage("xs_rank", children, out, lane_count, op=op)
        elif isinstance(op, GroupByOp):
            if grouped:
                raise CppStreamLoweringError("nested groupby is not supported")
            key_count = op.n_dynamic_keys
            key_sources = children[:key_count]
            feed_sources = children[key_count:]
            if any(
                source.width != 1 for source in key_sources + feed_sources
            ):
                raise CppStreamLoweringError(
                    "groupby keys/lhs/captures must be scalar-width vectors"
                )
            specs = base._resolved_key_specs(
                program, node.child_ids, op, key_cardinalities
            )
            dense = bool(specs) and all(
                spec.num_keys is not None for spec in specs
            )
            capacity = (
                base._dense_capacity(specs)
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
                children,
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
    if isinstance(program.nodes[root].op, (InputOp, LiteralOp)):
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

    return Plan(
        tuple(stages),
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


def lower_program(
    program: Program,
    *,
    n_instruments: int,
    default_group_capacity: int = 64,
    key_cardinalities: Mapping[str, int] | None = None,
    row_scalar_nodes: frozenset[int] | None = None,
    input_dtypes: tuple[str, ...] | None = None,
) -> Plan:
    if n_instruments <= 0 or default_group_capacity <= 0:
        raise CppStreamLoweringError(
            "n_instruments and group capacity must be > 0"
        )
    if input_dtypes is None:
        input_dtypes = ("float64",) * len(program.input_names)
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


__all__ = ["lower_program"]
