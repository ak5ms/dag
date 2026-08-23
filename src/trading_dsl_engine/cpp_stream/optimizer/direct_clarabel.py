from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping
import warnings

import numpy as np
from scipy import sparse


_CANONICAL_BLOCKS = ("P", "q", "A", "b", "d")
_SETTING_TYPES = {
    "verbose": "bool",
    "max_iter": "std::uint32_t",
    "tol_gap_abs": "double",
    "tol_gap_rel": "double",
    "tol_feas": "double",
    "presolve_enable": "bool",
}


@dataclass(frozen=True, slots=True)
class _SparseMap:
    rows: int
    values: np.ndarray
    columns: np.ndarray
    row_ptr: np.ndarray


@dataclass(frozen=True, slots=True)
class _MatrixStructure:
    rows: int
    columns: int
    row_indices: np.ndarray
    column_ptr: np.ndarray


@dataclass(frozen=True, slots=True)
class _PrimalView:
    name: str
    size: int
    offset: int


@dataclass(frozen=True, slots=True)
class _DualView:
    name: str
    size: int
    offset: int


@dataclass(frozen=True, slots=True)
class _CompiledCanonicalProgram:
    parameter_offsets: tuple[int, ...]
    parameter_maps: Mapping[str, _SparseMap]
    P: _MatrixStructure
    A: _MatrixStructure
    cone_initializers: tuple[str, ...]
    primals: tuple[_PrimalView, ...]
    duals: tuple[_DualView, ...]
    parameter_shards: int


def _parameter_attributes(parameter: Any) -> dict[str, object]:
    unsupported = [
        name
        for name in (
            "complex",
            "imag",
            "symmetric",
            "diag",
            "PSD",
            "NSD",
            "hermitian",
            "boolean",
            "integer",
            "sparsity",
            "bounds",
        )
        if parameter.attributes.get(name) not in (False, None)
    ]
    if unsupported:
        raise ValueError(
            f"cp.Parameter {parameter.name()!r} uses attributes not yet "
            f"supported by direct Clarabel parameter sharding: {unsupported}"
        )
    return {
        name: value
        for name, value in parameter.attributes.items()
        if value is not False and value is not None and name != "sparsity"
    }


def _zero_constant(cp: Any, shape: tuple[int, ...]):
    return cp.Constant(0.0 if not shape else np.zeros(shape, dtype=np.float64))


def _copy_constraint(constraint: Any, replacements: Mapping[int, Any]):
    args = [
        argument.tree_copy(id_objects=replacements)
        for argument in constraint.args
    ]
    data = constraint.get_data()
    copied = (
        type(constraint)(*(args + data))
        if data is not None
        else type(constraint)(*args)
    )
    label = getattr(constraint, "label", None)
    if label is not None and hasattr(copied, "set_label"):
        copied.set_label(label)
    return copied


def _parameter_shard_problem(
    cp: Any,
    problem: Any,
    parameters: tuple[Any, ...],
    offsets: tuple[int, ...],
    global_start: int,
    global_stop: int,
) -> tuple[Any, dict[int, np.ndarray]]:
    replacements: dict[int, Any] = {}
    global_indices_by_parameter_id: dict[int, np.ndarray] = {}
    for parameter, offset in zip(parameters, offsets):
        shape = tuple(int(extent) for extent in parameter.shape)
        local_start = max(0, global_start - offset)
        local_stop = min(int(parameter.size), global_stop - offset)
        if local_start >= local_stop:
            replacements[id(parameter)] = _zero_constant(cp, shape)
            continue
        selected = np.arange(local_start, local_stop, dtype=np.int64)
        attributes = _parameter_attributes(parameter)
        name = f"tdsl_shard_{parameter.id}_{global_start}_{global_stop}"
        if not shape:
            shard = cp.Parameter(name=name, **attributes)
            ordered_global_indices = np.asarray([offset], dtype=np.int64)
        else:
            coordinates = np.unravel_index(selected, shape, order="F")
            shard = cp.Parameter(
                shape,
                name=name,
                sparsity=coordinates,
                **attributes,
            )
            ordered_global_indices = offset + np.ravel_multi_index(
                shard.sparse_idx,
                shape,
                order="F",
            )
        replacements[id(parameter)] = shard
        global_indices_by_parameter_id[shard.id] = np.asarray(
            ordered_global_indices,
            dtype=np.int64,
        )

    objective = type(problem.objective)(
        problem.objective.expr.tree_copy(id_objects=replacements)
    )
    constraints = [
        _copy_constraint(constraint, replacements)
        for constraint in problem.constraints
    ]
    return cp.Problem(objective, constraints), global_indices_by_parameter_id


def _composed_parameter_ids(chain: Any, parameter_id: int) -> tuple[int, ...]:
    mapped = chain.compose_param_id_map().get(parameter_id, (parameter_id,))
    if isinstance(mapped, int):
        return (mapped,)
    return tuple(int(item) for item in mapped)


def _local_to_global_columns(
    chain: Any,
    param_problem: Any,
    original_parameter_columns: Mapping[int, np.ndarray],
    global_constant_column: int,
) -> np.ndarray:
    result = np.full(
        param_problem.total_param_size + 1,
        -1,
        dtype=np.int64,
    )
    for parameter_id, global_columns in original_parameter_columns.items():
        reduced_ids = _composed_parameter_ids(chain, parameter_id)
        if len(reduced_ids) != 1:
            raise ValueError(
                "one real CVXPY parameter shard unexpectedly mapped to "
                f"{len(reduced_ids)} canonical parameters"
            )
        reduced_id = reduced_ids[0]
        if reduced_id not in param_problem.param_id_to_col:
            raise ValueError("CVXPY omitted a declared parameter shard")
        start = int(param_problem.param_id_to_col[reduced_id])
        size = int(param_problem.param_id_to_size[reduced_id])
        if size != global_columns.size:
            raise ValueError(
                "CVXPY changed the scalar order of a sparse parameter shard"
            )
        result[start : start + size] = global_columns
    if np.any(result[:-1] < 0):
        raise ValueError("could not map every canonical parameter column")
    result[-1] = global_constant_column
    return result


def _problem_data_keys(reduced: Any) -> np.ndarray:
    reduced.cache()
    indices, column_ptr, shape = reduced.problem_data_index
    indices = np.asarray(indices, dtype=np.int64)
    column_ptr = np.asarray(column_ptr, dtype=np.int64)
    columns = np.repeat(
        np.arange(int(shape[1]), dtype=np.int64),
        np.diff(column_ptr),
    )
    return columns * int(shape[0]) + indices


def _append_mapping(
    target: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    mapping: sparse.spmatrix,
    output_keys: np.ndarray,
    local_to_global: np.ndarray,
    *,
    include_constant: bool,
    output_scales: np.ndarray | None = None,
) -> None:
    coordinate = mapping.tocoo(copy=False)
    if coordinate.nnz == 0:
        return
    keep = (
        np.ones(coordinate.nnz, dtype=bool)
        if include_constant
        else coordinate.col != mapping.shape[1] - 1
    )
    if not np.any(keep):
        return
    selected_rows = np.asarray(coordinate.row[keep], dtype=np.int64)
    selected_values = np.asarray(coordinate.data[keep], dtype=np.float64)
    if output_scales is not None:
        selected_values = selected_values * output_scales[selected_rows]
    target.append(
        (
            output_keys[selected_rows],
            local_to_global[np.asarray(coordinate.col[keep], dtype=np.int64)],
            selected_values,
        )
    )


def _unformatted_param_problem(
    cp: Any,
    problem: Any,
    canon_backend: str,
) -> tuple[Any, Any]:
    chain = problem._construct_chain(
        solver=cp.CLARABEL,
        enforce_dpp=True,
        canon_backend=canon_backend,
    )
    reduced = problem
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Reading from a sparse CVXPY expression",
            category=RuntimeWarning,
        )
        for reduction in chain.reductions[:-1]:
            reduced, _ = reduction.apply(reduced)
    if not hasattr(reduced, "reduced_A"):
        raise TypeError("CVXPY did not produce a parameterized cone program")
    return reduced, chain


def _constraint_row_format(param_problem: Any) -> tuple[np.ndarray, np.ndarray]:
    from cvxpy.constraints import (
        ExpCone,
        NonNeg,
        PowCone3D,
        PowConeND,
        PSD,
        SOC,
        Zero,
    )

    try:
        from cvxpy.constraints import SvecPSD
    except ImportError:  # pragma: no cover - only older compatible CVXPY.
        SvecPSD = ()

    row_count = int(param_problem.constr_size)
    destination = np.empty(row_count, dtype=np.int64)
    scales = np.ones(row_count, dtype=np.float64)
    offset = 0
    for constraint in param_problem.constraints:
        size = int(constraint.size)
        local_destination = np.arange(size, dtype=np.int64)
        if type(constraint) is Zero:
            scales[offset : offset + size] = -1.0
        elif type(constraint) in (NonNeg, PSD, SvecPSD):
            pass
        elif type(constraint) is SOC:
            if constraint.axis != 0:
                raise ValueError("CVXPY did not lower SOC to axis zero")
            t_size = int(constraint.args[0].size)
            x_dim = (
                int(constraint.args[1].shape[0])
                if constraint.args[1].shape
                else 1
            )
            local_destination[:t_size] = (
                np.arange(t_size, dtype=np.int64) * (x_dim + 1)
            )
            x_index = np.arange(int(constraint.args[1].size), dtype=np.int64)
            local_destination[t_size:] = (
                (x_index // x_dim) * (x_dim + 1)
                + 1
                + x_index % x_dim
            )
        elif type(constraint) in (ExpCone, PowCone3D):
            arity = len(constraint.args)
            argument_size = int(constraint.args[0].size)
            for argument_index in range(arity):
                source = slice(
                    argument_index * argument_size,
                    (argument_index + 1) * argument_size,
                )
                local_destination[source] = (
                    np.arange(argument_size, dtype=np.int64) * arity
                    + argument_index
                )
        elif type(constraint) is PowConeND:
            if constraint.args[0].ndim == 1:
                m, n = int(constraint.args[0].shape[0]), 1
            else:
                m, n = (int(value) for value in constraint.args[0].shape)
            weights_size = int(constraint.args[0].size)
            weight_index = np.arange(weights_size, dtype=np.int64)
            local_destination[:weights_size] = (
                (weight_index // m) * (m + 1) + weight_index % m
            )
            local_destination[weights_size:] = (
                np.arange(n, dtype=np.int64) * (m + 1) + m
            )
        else:
            raise ValueError(
                f"unsupported canonical constraint {type(constraint).__name__}"
            )
        destination[offset : offset + size] = offset + local_destination
        offset += size
    if offset != row_count or np.unique(destination).size != row_count:
        raise ValueError("invalid canonical constraint row permutation")
    return destination, scales


def _combine_mapping(
    entries: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    rows: int,
    columns: int,
) -> _SparseMap:
    if not entries:
        return _SparseMap(
            rows,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.uint32),
            np.zeros(rows + 1, dtype=np.uint32),
        )
    output_rows = np.concatenate([entry[0] for entry in entries])
    parameter_columns = np.concatenate([entry[1] for entry in entries])
    values = np.concatenate([entry[2] for entry in entries])
    matrix = sparse.csr_matrix(
        (values, (output_rows, parameter_columns)),
        shape=(rows, columns),
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.nnz >= np.iinfo(np.uint32).max:
        raise OverflowError("canonical parameter map exceeds uint32 storage")
    if columns > np.iinfo(np.uint32).max:
        raise OverflowError("canonical parameter vector exceeds uint32 storage")
    return _SparseMap(
        rows,
        np.asarray(matrix.data, dtype=np.float64),
        np.asarray(matrix.indices, dtype=np.uint32),
        np.asarray(matrix.indptr, dtype=np.uint32),
    )


def _compress_structural_mapping(
    entries: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    matrix_rows: int,
    matrix_columns: int,
    parameter_columns: int,
    upper_triangle: bool = False,
) -> tuple[_MatrixStructure, _SparseMap]:
    if not entries:
        structure = _MatrixStructure(
            matrix_rows,
            matrix_columns,
            np.empty(0, dtype=np.uint64),
            np.zeros(matrix_columns + 1, dtype=np.uint64),
        )
        return structure, _combine_mapping(
            [], rows=0, columns=parameter_columns
        )
    keys = np.concatenate([entry[0] for entry in entries])
    parameter_indices = np.concatenate([entry[1] for entry in entries])
    values = np.concatenate([entry[2] for entry in entries])
    if upper_triangle:
        row = keys % matrix_rows
        column = keys // matrix_rows
        keep = row <= column
        keys = keys[keep]
        parameter_indices = parameter_indices[keep]
        values = values[keep]
    unique_keys = np.unique(keys)
    output_rows = np.searchsorted(unique_keys, keys)
    mapping = _combine_mapping(
        [(output_rows, parameter_indices, values)],
        rows=unique_keys.size,
        columns=parameter_columns,
    )
    structural_columns = unique_keys // matrix_rows
    column_ptr = np.zeros(matrix_columns + 1, dtype=np.uint64)
    np.add.at(column_ptr, structural_columns + 1, 1)
    np.cumsum(column_ptr, out=column_ptr)
    structure = _MatrixStructure(
        matrix_rows,
        matrix_columns,
        np.asarray(unique_keys % matrix_rows, dtype=np.uint64),
        column_ptr,
    )
    return structure, mapping


def _split_A_and_b(
    entries: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    rows: int,
    columns: int,
    parameter_columns: int,
) -> tuple[_MatrixStructure, _SparseMap, _SparseMap]:
    a_entries: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    b_entries: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    boundary = rows * columns
    for keys, parameter_indices, values in entries:
        is_a = keys < boundary
        if np.any(is_a):
            # ParamConeProg stores cone expressions as A*x + b. Clarabel's
            # solver interface uses (-A)*x + s = b.
            a_entries.append(
                (keys[is_a], parameter_indices[is_a], -values[is_a])
            )
        if np.any(~is_a):
            b_entries.append(
                (
                    keys[~is_a] - boundary,
                    parameter_indices[~is_a],
                    values[~is_a],
                )
            )
    structure, a_mapping = _compress_structural_mapping(
        a_entries,
        matrix_rows=rows,
        matrix_columns=columns,
        parameter_columns=parameter_columns,
    )
    b_mapping = _combine_mapping(
        b_entries,
        rows=rows,
        columns=parameter_columns,
    )
    return structure, a_mapping, b_mapping


def _cone_initializers(dims: Any) -> tuple[str, ...]:
    if getattr(dims, "pnd", ()):
        raise ValueError("generalized power cones are not supported by the C emitter")
    result: list[str] = []
    if dims.zero:
        result.append(f"ClarabelZeroConeT({int(dims.zero)})")
    if dims.nonneg:
        result.append(f"ClarabelNonnegativeConeT({int(dims.nonneg)})")
    result.extend(
        f"ClarabelSecondOrderConeT({int(size)})"
        for size in dims.soc
    )
    result.extend(
        f"ClarabelPSDTriangleConeT({int(size)})"
        for size in dims.psd
    )
    result.extend("ClarabelExponentialConeT()" for _ in range(int(dims.exp)))
    result.extend(
        f"ClarabelPowerConeT({float(power):.17g})"
        for power in dims.p3d
    )
    return tuple(result)


def _canonical_layouts(
    original_problem: Any,
    shard_problem: Any,
    param_problem: Any,
) -> tuple[tuple[_PrimalView, ...], tuple[_DualView, ...]]:
    primals: list[_PrimalView] = []
    for variable in original_problem.variables():
        offset = param_problem.var_id_to_col.get(variable.id)
        if offset is None:
            raise ValueError(
                f"CVXPY variable {variable.name()!r} requires a nontrivial "
                "inverse map that the direct Clarabel emitter does not support"
            )
        primals.append(
            _PrimalView(variable.name(), int(variable.size), int(offset))
        )

    canonical_constraints: dict[int, tuple[int, int]] = {}
    offset = 0
    for constraint in param_problem.constraints:
        canonical_constraints[int(constraint.id)] = (offset, int(constraint.size))
        offset += int(constraint.size)
    duals: list[_DualView] = []
    for index, constraint in enumerate(shard_problem.constraints):
        view = canonical_constraints.get(int(constraint.id))
        if view is None:
            continue
        duals.append(_DualView(f"d{index}", view[1], view[0]))
    return tuple(primals), tuple(duals)


def compile_sharded_canonical_program(
    problem: Any,
    *,
    parameter_shard_size: int = 512,
    canon_backend: str = "COO",
) -> _CompiledCanonicalProgram:
    """Compile compact affine maps without forming CVXPY's full DPP tensor."""

    import cvxpy as cp

    if parameter_shard_size <= 0:
        raise ValueError("parameter_shard_size must be positive")
    parameters = tuple(problem.parameters())
    offsets_list: list[int] = []
    total_parameters = 0
    for parameter in parameters:
        _parameter_attributes(parameter)
        offsets_list.append(total_parameters)
        total_parameters += int(parameter.size)
    offsets = tuple(offsets_list)
    global_columns = total_parameters + 1
    shard_ranges = list(range(0, total_parameters, parameter_shard_size)) or [0]
    ab_entries: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    p_entries: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    q_entries: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    expected_signature = None
    first_shard_problem = None
    first_param_problem = None
    n_variables = 0
    n_constraints = 0
    dims = None

    for shard_index, global_start in enumerate(shard_ranges):
        global_stop = min(
            total_parameters,
            global_start + parameter_shard_size,
        )
        if total_parameters:
            shard_problem, parameter_columns = _parameter_shard_problem(
                cp,
                problem,
                parameters,
                offsets,
                global_start,
                global_stop,
            )
        else:
            shard_problem = problem
            parameter_columns = {}
        param_problem, chain = _unformatted_param_problem(
            cp,
            shard_problem,
            canon_backend,
        )
        local_to_global = _local_to_global_columns(
            chain,
            param_problem,
            parameter_columns,
            total_parameters,
        )
        current_n = int(param_problem.x.size)
        current_m = int(param_problem.constr_size)
        current_dims = (
            int(param_problem.cone_dims.zero),
            int(param_problem.cone_dims.nonneg),
            int(param_problem.cone_dims.exp),
            tuple(int(size) for size in param_problem.cone_dims.soc),
            tuple(int(size) for size in param_problem.cone_dims.psd),
            tuple(float(size) for size in param_problem.cone_dims.p3d),
        )
        signature = (current_n, current_m, current_dims)
        if expected_signature is None:
            expected_signature = signature
            n_variables = current_n
            n_constraints = current_m
            dims = param_problem.cone_dims
            first_shard_problem = shard_problem
            first_param_problem = param_problem
        elif signature != expected_signature:
            raise ValueError("parameter sharding changed CVXPY's canonical layout")

        include_constant = shard_index == 0
        ab_keys = _problem_data_keys(param_problem.reduced_A)
        row_destination, row_scales = _constraint_row_format(param_problem)
        ab_rows = ab_keys % current_m
        ab_keys = (
            (ab_keys // current_m) * current_m
            + row_destination[ab_rows]
        )
        _append_mapping(
            ab_entries,
            param_problem.reduced_A.reduced_mat,
            ab_keys,
            local_to_global,
            include_constant=include_constant,
            output_scales=row_scales[ab_rows],
        )
        q_keys = np.arange(param_problem.q.shape[0], dtype=np.int64)
        _append_mapping(
            q_entries,
            param_problem.q,
            q_keys,
            local_to_global,
            include_constant=include_constant,
        )
        if param_problem.P is not None:
            p_keys = _problem_data_keys(param_problem.reduced_P)
            _append_mapping(
                p_entries,
                param_problem.reduced_P.reduced_mat,
                p_keys,
                local_to_global,
                include_constant=include_constant,
            )
        del chain, param_problem
        if shard_index:
            del shard_problem

    assert dims is not None
    assert first_shard_problem is not None
    assert first_param_problem is not None
    A, A_map, b_map = _split_A_and_b(
        ab_entries,
        rows=n_constraints,
        columns=n_variables,
        parameter_columns=global_columns,
    )
    P, P_map = _compress_structural_mapping(
        p_entries,
        matrix_rows=n_variables,
        matrix_columns=n_variables,
        parameter_columns=global_columns,
        upper_triangle=True,
    )
    q_full = _combine_mapping(
        q_entries,
        rows=n_variables + 1,
        columns=global_columns,
    )
    q_matrix = sparse.csr_matrix(
        (q_full.values, q_full.columns, q_full.row_ptr),
        shape=(q_full.rows, global_columns),
    )
    q_map_matrix = q_matrix[:n_variables].tocsr()
    d_map_matrix = q_matrix[n_variables:].tocsr()
    q_map = _SparseMap(
        n_variables,
        np.asarray(q_map_matrix.data, dtype=np.float64),
        np.asarray(q_map_matrix.indices, dtype=np.uint32),
        np.asarray(q_map_matrix.indptr, dtype=np.uint32),
    )
    d_map = _SparseMap(
        1,
        np.asarray(d_map_matrix.data, dtype=np.float64),
        np.asarray(d_map_matrix.indices, dtype=np.uint32),
        np.asarray(d_map_matrix.indptr, dtype=np.uint32),
    )
    primals, duals = _canonical_layouts(
        problem,
        first_shard_problem,
        first_param_problem,
    )
    return _CompiledCanonicalProgram(
        offsets,
        {"P": P_map, "q": q_map, "A": A_map, "b": b_map, "d": d_map},
        P,
        A,
        _cone_initializers(dims),
        primals,
        duals,
        len(shard_ranges),
    )


def _cpp_float(value: float) -> str:
    if np.isnan(value) or np.isinf(value):
        raise ValueError("generated canonical map contains a non-finite value")
    result = f"{float(value):.17g}"
    if result == "-0":
        return "0.0"
    if "." not in result and "e" not in result:
        result += ".0"
    return result


def _cpp_array(values: np.ndarray, formatter, *, per_line: int = 12) -> str:
    rendered = [formatter(value) for value in values]
    if not rendered:
        return "{}"
    lines = [
        ", ".join(rendered[start : start + per_line])
        for start in range(0, len(rendered), per_line)
    ]
    return "{\n      " + ",\n      ".join(lines) + "\n  }"


def _parameter_dirty_blocks(
    mapping_by_block: Mapping[str, _SparseMap],
    offset: int,
    size: int,
) -> tuple[str, ...]:
    result = []
    for block in _CANONICAL_BLOCKS:
        columns = mapping_by_block[block].columns
        if np.any((columns >= offset) & (columns < offset + size)):
            result.append(block)
    return tuple(result)


def _emit_direct_header(
    root: Path,
    *,
    class_name: str,
    prefix: str,
    problem: Any,
    compiled: _CompiledCanonicalProgram,
    enable_settings: Iterable[str],
) -> Path:
    from .clarabel_native import _safe_identifier, _template_environment

    parameter_specs = []
    for index, (parameter, offset) in enumerate(
        zip(problem.parameters(), compiled.parameter_offsets)
    ):
        name = _safe_identifier(parameter.name(), label="CVXPY parameter name")
        parameter_specs.append(
            {
                "name": name,
                "index": index,
                "offset": offset,
                "size": int(parameter.size),
                "dirty_blocks": _parameter_dirty_blocks(
                    compiled.parameter_maps,
                    offset,
                    int(parameter.size),
                ),
            }
        )
    primal_specs = []
    for index, view in enumerate(compiled.primals):
        primal_specs.append(
            {
                "name": _safe_identifier(view.name, label="CVXPY variable name"),
                "index": index,
                "offset": view.offset,
                "size": view.size,
            }
        )
    dual_specs = [
        {
            "name": view.name,
            "index": index,
            "offset": view.offset,
            "size": view.size,
        }
        for index, view in enumerate(compiled.duals)
    ]
    settings = []
    for name in enable_settings:
        if name not in _SETTING_TYPES:
            raise ValueError(f"unsupported Clarabel setting {name!r}")
        settings.append({"name": name, "type": _SETTING_TYPES[name]})
    maps = {
        name: {
            "name": name,
            "rows": mapping.rows,
            "nnz": int(mapping.values.size),
            "values": _cpp_array(mapping.values, _cpp_float, per_line=6),
            "columns": _cpp_array(
                mapping.columns,
                lambda value: str(int(value)),
            ),
            "row_ptr": _cpp_array(
                mapping.row_ptr,
                lambda value: str(int(value)),
            ),
        }
        for name, mapping in compiled.parameter_maps.items()
    }
    matrices = {
        "P": {
            "rows": compiled.P.rows,
            "columns": compiled.P.columns,
            "nnz": int(compiled.P.row_indices.size),
            "row_indices": _cpp_array(
                compiled.P.row_indices,
                lambda value: str(int(value)),
            ),
            "column_ptr": _cpp_array(
                compiled.P.column_ptr,
                lambda value: str(int(value)),
            ),
        },
        "A": {
            "rows": compiled.A.rows,
            "columns": compiled.A.columns,
            "nnz": int(compiled.A.row_indices.size),
            "row_indices": _cpp_array(
                compiled.A.row_indices,
                lambda value: str(int(value)),
            ),
            "column_ptr": _cpp_array(
                compiled.A.column_ptr,
                lambda value: str(int(value)),
            ),
        },
    }
    header_dir = root / "cpp" / "include"
    header_dir.mkdir(parents=True, exist_ok=True)
    environment = _template_environment()
    header = header_dir / f"{prefix}instance.hpp"
    header.write_text(
        environment.get_template("direct_clarabel_instance.hpp.j2").render(
            class_name=class_name,
            parameter_count=len(parameter_specs),
            parameter_scalar_count=sum(item["size"] for item in parameter_specs),
            parameters=parameter_specs,
            primals=primal_specs,
            duals=dual_specs,
            maps=maps,
            matrices=matrices,
            cones=compiled.cone_initializers,
            settings=settings,
            info_fields=(
                ("objective", "obj_val"),
                ("iterations", "iterations"),
                ("status", "status"),
                ("primal_residual", "r_prim"),
                ("dual_residual", "r_dual"),
            ),
        )
    )
    (header_dir / "cpg_instance.hpp").write_text(
        environment.get_template("cpg_instance_alias.hpp.j2").render(
            instance_header=header.name,
        )
    )
    return header


def generate_direct_clarabel_program(
    problem: Any,
    *,
    code_dir: str | os.PathLike[str],
    clarabel: Any,
    class_name: str,
    prefix: str,
    instrument_count: int | None,
    enable_settings: Iterable[str],
    field_aliases: Mapping[str, str] | None,
    force: bool,
    parameter_shard_size: int = 512,
):
    from .clarabel_native import (
        DualLayout,
        FieldAlias,
        GeneratedCvxpygenProgram,
        ParameterLayout,
        PrimalLayout,
        _constraint_dual_shape,
        _constraint_label,
        _package_version,
        _safe_identifier,
    )

    _safe_identifier(class_name, label="C++ class name")
    _safe_identifier(prefix, label="generated prefix")
    if not problem.is_dcp(dpp=True):
        raise ValueError("CVXPY problem must be DPP-compliant")
    labels = tuple(_constraint_label(constraint) for constraint in problem.constraints)
    duplicate_labels = sorted(
        label
        for label in set(labels)
        if label is not None and labels.count(label) > 1
    )
    if duplicate_labels:
        raise ValueError(f"constraint labels must be unique: {duplicate_labels}")
    root = Path(code_dir).expanduser().resolve()
    if root.exists():
        if not force:
            raise FileExistsError(
                f"code directory already exists: {root}; pass force=True to replace it"
            )
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    compiled = compile_sharded_canonical_program(
        problem,
        parameter_shard_size=parameter_shard_size,
    )
    header = _emit_direct_header(
        root,
        class_name=class_name,
        prefix=prefix,
        problem=problem,
        compiled=compiled,
        enable_settings=enable_settings,
    )
    clarabel = clarabel.normalized()
    parameters = tuple(problem.parameters())
    public_parameters = tuple(
        ParameterLayout(
            parameter.name(),
            tuple(int(extent) for extent in parameter.shape),
            int(parameter.size),
            compiled.parameter_offsets[index],
            _parameter_dirty_blocks(
                compiled.parameter_maps,
                compiled.parameter_offsets[index],
                int(parameter.size),
            ),
        )
        for index, parameter in enumerate(parameters)
    )
    variable_by_name = {variable.name(): variable for variable in problem.variables()}
    public_primals = tuple(
        PrimalLayout(
            view.name,
            tuple(int(extent) for extent in variable_by_name[view.name].shape),
            view.size,
        )
        for view in compiled.primals
    )
    dual_by_name = {view.name: view for view in compiled.duals}
    public_duals = tuple(
        DualLayout(
            f"d{index}",
            index,
            labels[index],
            _constraint_dual_shape(
                constraint,
                dual_by_name[f"d{index}"].size,
            ),
            dual_by_name[f"d{index}"].size,
        )
        for index, constraint in enumerate(problem.constraints)
        if f"d{index}" in dual_by_name
    )
    alias_mapping = dict(field_aliases or {})
    primal_names = {primal.name for primal in public_primals}
    for alias_name, primal_name in alias_mapping.items():
        if primal_name not in primal_names:
            raise ValueError(
                f"generated field alias {alias_name!r} targets unknown primal "
                f"{primal_name!r}"
            )
    aliases = tuple(
        FieldAlias(name, primal_name)
        for name, primal_name in sorted(alias_mapping.items())
    )
    manifest = {
        "schema_version": 3,
        "backend": "cvxpy-direct-clarabel",
        "class_name": class_name,
        "prefix": prefix,
        "cvxpy_version": _package_version("cvxpy"),
        "solver": "CLARABEL",
        "clarabel_version": clarabel.version,
        "canonicalization_backend": "COO",
        "parameter_shard_size": parameter_shard_size,
        "parameter_shards": compiled.parameter_shards,
        "instance_owned": True,
        "persistent_solver": True,
        "instrument_count": instrument_count,
        "parameters": [
            {
                "name": item.name,
                "shape": list(item.shape),
                "size": item.size,
                "offset": item.offset,
                "dirty_blocks": list(item.dirty_blocks),
                "column_major": item.column_major,
            }
            for item in public_parameters
        ],
        "primals": [
            {"name": item.name, "shape": list(item.shape), "size": item.size}
            for item in public_primals
        ],
        "duals": [
            {
                "name": item.name,
                "constraint_index": item.constraint_index,
                "label": item.label,
                "shape": list(item.shape),
                "size": item.size,
            }
            for item in public_duals
        ],
        "aliases": [
            {"name": item.name, "primal_name": item.primal_name}
            for item in aliases
        ],
    }
    manifest_path = root / "cpp" / "cpg_instance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return GeneratedCvxpygenProgram(
        root,
        header,
        manifest_path,
        class_name,
        prefix,
        public_parameters,
        public_primals,
        public_duals,
        aliases,
        clarabel,
        instrument_count,
    )


def load_direct_clarabel_program(
    code_dir: str | os.PathLike[str],
    *,
    clarabel: Any,
):
    from .clarabel_native import (
        DualLayout,
        FieldAlias,
        GeneratedCvxpygenProgram,
        ParameterLayout,
        PrimalLayout,
        _package_version,
    )

    root = Path(code_dir).expanduser().resolve()
    manifest_path = root / "cpp" / "cpg_instance_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 3:
        raise ValueError(
            f"unsupported generated-program manifest schema in {manifest_path}"
        )
    if manifest.get("backend") != "cvxpy-direct-clarabel":
        raise ValueError(f"cached program in {manifest_path} uses another backend")
    if manifest.get("cvxpy_version") != _package_version("cvxpy"):
        raise ValueError(
            "cached generated program uses CVXPY "
            f"{manifest.get('cvxpy_version')!r}, found "
            f"{_package_version('cvxpy')!r}"
        )
    clarabel = clarabel.normalized()
    if manifest.get("clarabel_version") != clarabel.version:
        raise ValueError(
            "cached generated program targets Clarabel "
            f"{manifest.get('clarabel_version')!r}, found {clarabel.version!r}"
        )
    prefix = str(manifest["prefix"])
    instance_header = root / "cpp" / "include" / f"{prefix}instance.hpp"
    if not instance_header.is_file():
        raise FileNotFoundError(
            f"cached generated instance header not found: {instance_header}"
        )
    parameters = tuple(
        ParameterLayout(
            str(item["name"]),
            tuple(int(extent) for extent in item["shape"]),
            int(item["size"]),
            int(item["offset"]),
            tuple(str(block) for block in item["dirty_blocks"]),
            bool(item.get("column_major", True)),
        )
        for item in manifest["parameters"]
    )
    primals = tuple(
        PrimalLayout(
            str(item["name"]),
            tuple(int(extent) for extent in item["shape"]),
            int(item["size"]),
        )
        for item in manifest["primals"]
    )
    duals = tuple(
        DualLayout(
            str(item["name"]),
            int(item["constraint_index"]),
            None if item.get("label") is None else str(item["label"]),
            tuple(int(extent) for extent in item["shape"]),
            int(item["size"]),
        )
        for item in manifest["duals"]
    )
    aliases = tuple(
        FieldAlias(str(item["name"]), str(item["primal_name"]))
        for item in manifest.get("aliases", ())
    )
    instrument_count = manifest.get("instrument_count")
    return GeneratedCvxpygenProgram(
        root,
        instance_header,
        manifest_path,
        str(manifest["class_name"]),
        prefix,
        parameters,
        primals,
        duals,
        aliases,
        clarabel,
        None if instrument_count is None else int(instrument_count),
    )


__all__ = [
    "compile_sharded_canonical_program",
    "generate_direct_clarabel_program",
    "load_direct_clarabel_program",
]
