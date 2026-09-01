from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one patch anchor, found {count}: {old[:100]!r}"
        )
    path.write_text(text.replace(old, new, 1))


def patch_direct_clarabel() -> None:
    path = ROOT / "src/trading_dsl_engine/cpp_stream/optimizer/direct_clarabel.py"
    replace_once(
        path,
        '    "presolve_enable": "bool",\n',
        '    "presolve_enable": "bool",\n    "iterative_refinement_enable": "bool",\n',
    )
    replace_once(
        path,
        """@dataclass(frozen=True, slots=True)\nclass _CompiledCanonicalProgram:\n""",
        """@dataclass(frozen=True, slots=True)\nclass _ConstraintValueView:\n    name: str\n    constraint_index: int\n    label: str | None\n    shape: tuple[int, ...]\n    size: int\n    offset: int\n\n\n@dataclass(frozen=True, slots=True)\nclass _ConstraintValueProgram:\n    values: tuple[_ConstraintValueView, ...]\n    rhs_map: _SparseMap\n    denominator_map: _SparseMap\n    coefficient_map: _SparseMap\n    term_row_ptr: np.ndarray\n    term_primal_columns: np.ndarray\n\n    @property\n    def scalar_count(self) -> int:\n        return sum(value.size for value in self.values)\n\n\n@dataclass(frozen=True, slots=True)\nclass _CompiledCanonicalProgram:\n""",
    )
    insertion = r'''

def _constraint_value_expression(cp: Any, constraint: Any):
    try:
        return constraint.expr
    except (AttributeError, ValueError):
        pass
    parts = tuple(
        cp.reshape(argument, (argument.size,), order="F")
        for argument in constraint.args
    )
    if not parts:
        raise ValueError(
            f"constraint {type(constraint).__name__} exposes no numeric arguments"
        )
    return parts[0] if len(parts) == 1 else cp.hstack(parts)


def _select_sparse_map_rows(
    mapping: _SparseMap,
    rows: Iterable[int],
) -> _SparseMap:
    values: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    row_ptr = [0]
    for raw_row in rows:
        row = int(raw_row)
        if row < 0 or row >= mapping.rows:
            raise IndexError(f"sparse-map row {row} outside [0, {mapping.rows})")
        begin = int(mapping.row_ptr[row])
        end = int(mapping.row_ptr[row + 1])
        values.append(mapping.values[begin:end])
        columns.append(mapping.columns[begin:end])
        row_ptr.append(row_ptr[-1] + end - begin)
    return _SparseMap(
        len(row_ptr) - 1,
        np.concatenate(values) if values else np.empty(0, dtype=np.float64),
        np.concatenate(columns) if columns else np.empty(0, dtype=np.uint32),
        np.asarray(row_ptr, dtype=np.uint32),
    )


def _remap_sparse_map_columns(
    mapping: _SparseMap,
    column_map: np.ndarray,
) -> _SparseMap:
    if mapping.columns.size:
        mapped = column_map[np.asarray(mapping.columns, dtype=np.int64)]
        if np.any(mapped < 0):
            missing = np.unique(mapping.columns[mapped < 0]).tolist()
            raise ValueError(
                f"constraint evaluator references unmapped parameter columns {missing}"
            )
        columns = np.asarray(mapped, dtype=np.uint32)
    else:
        columns = np.empty(0, dtype=np.uint32)
    return _SparseMap(
        mapping.rows,
        np.asarray(mapping.values, dtype=np.float64),
        columns,
        np.asarray(mapping.row_ptr, dtype=np.uint32),
    )


def _matrix_entries_by_row(
    structure: _MatrixStructure,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    rows: list[list[tuple[int, int]]] = [
        [] for _ in range(structure.rows)
    ]
    for column in range(structure.columns):
        begin = int(structure.column_ptr[column])
        end = int(structure.column_ptr[column + 1])
        for entry in range(begin, end):
            row = int(structure.row_indices[entry])
            rows[row].append((column, entry))
    return tuple(tuple(items) for items in rows)


def _empty_constraint_value_program() -> _ConstraintValueProgram:
    return _ConstraintValueProgram(
        (),
        _SparseMap(
            0,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.uint32),
            np.zeros(1, dtype=np.uint32),
        ),
        _SparseMap(
            0,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.uint32),
            np.zeros(1, dtype=np.uint32),
        ),
        _SparseMap(
            0,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.uint32),
            np.zeros(1, dtype=np.uint32),
        ),
        np.zeros(1, dtype=np.uint32),
        np.empty(0, dtype=np.uint32),
    )


def _compile_constraint_value_program(
    problem: Any,
    compiled: _CompiledCanonicalProgram,
    constraint_value_indices: Iterable[int],
    *,
    parameter_shard_size: int,
    canon_backend: str = "COO",
) -> _ConstraintValueProgram:
    import cvxpy as cp

    indices = tuple(dict.fromkeys(int(index) for index in constraint_value_indices))
    if not indices:
        return _empty_constraint_value_program()

    variable_names = {variable.name() for variable in problem.variables()}
    evaluator_constraints = []
    requested: list[tuple[str, int, str | None, tuple[int, ...], int]] = []
    for index in indices:
        if index < 0 or index >= len(problem.constraints):
            raise IndexError(
                f"constraint value index {index} outside "
                f"[0, {len(problem.constraints)})"
            )
        constraint = problem.constraints[index]
        expression = _constraint_value_expression(cp, constraint)
        if not expression.is_affine():
            raise ValueError(
                f"constraint value {index} must be affine for native "
                "post-solve evaluation"
            )
        base_name = f"cpp_stream_constraint_eval_{index}"
        name = base_name
        suffix = 1
        while name in variable_names:
            suffix += 1
            name = f"{base_name}_{suffix}"
        variable_names.add(name)
        value = cp.Variable(expression.shape, name=name)
        evaluator_constraints.append(value == expression)
        requested.append(
            (
                name,
                index,
                getattr(constraint, "label", None),
                tuple(int(extent) for extent in expression.shape),
                int(expression.size),
            )
        )

    evaluator_problem = cp.Problem(cp.Minimize(0.0), evaluator_constraints)
    evaluator = compile_sharded_canonical_program(
        evaluator_problem,
        parameter_shard_size=parameter_shard_size,
        canon_backend=canon_backend,
    )

    main_parameters = {parameter.name(): parameter for parameter in problem.parameters()}
    main_offsets = {
        parameter.name(): int(offset)
        for parameter, offset in zip(problem.parameters(), compiled.parameter_offsets)
    }
    main_parameter_count = sum(int(parameter.size) for parameter in problem.parameters())
    evaluator_parameter_count = sum(
        int(parameter.size) for parameter in evaluator_problem.parameters()
    )
    parameter_column_map = np.full(
        evaluator_parameter_count + 1, -1, dtype=np.int64
    )
    for parameter, offset in zip(
        evaluator_problem.parameters(), evaluator.parameter_offsets
    ):
        name = parameter.name()
        main = main_parameters.get(name)
        if main is None:
            raise ValueError(
                f"constraint evaluator introduced unknown parameter {name!r}"
            )
        if tuple(main.shape) != tuple(parameter.shape):
            raise ValueError(
                f"constraint evaluator changed shape of parameter {name!r}"
            )
        start = int(offset)
        stop = start + int(parameter.size)
        main_start = main_offsets[name]
        parameter_column_map[start:stop] = np.arange(
            main_start, main_start + int(parameter.size), dtype=np.int64
        )
    parameter_column_map[evaluator_parameter_count] = main_parameter_count

    evaluator_A_map = _remap_sparse_map_columns(
        evaluator.parameter_maps["A"], parameter_column_map
    )
    evaluator_b_map = _remap_sparse_map_columns(
        evaluator.parameter_maps["b"], parameter_column_map
    )

    main_primal_by_name = {view.name: view for view in compiled.primals}
    evaluator_primal_by_name = {view.name: view for view in evaluator.primals}
    auxiliary_names = {item[0] for item in requested}
    evaluator_to_main_column: dict[int, int] = {}
    for view in evaluator.primals:
        if view.name in auxiliary_names:
            continue
        main = main_primal_by_name.get(view.name)
        if main is None or main.size != view.size:
            raise ValueError(
                f"constraint evaluator cannot map primal {view.name!r} "
                "back to the original solver"
            )
        for local in range(view.size):
            evaluator_to_main_column[view.offset + local] = main.offset + local

    dual_by_name = {view.name: view for view in evaluator.duals}
    A_entries = _matrix_entries_by_row(evaluator.A)
    rhs_rows: list[int] = []
    denominator_entries: list[int] = []
    coefficient_entries: list[int] = []
    term_primal_columns: list[int] = []
    term_row_ptr = [0]
    value_views: list[_ConstraintValueView] = []
    output_offset = 0

    for position, (name, index, label, shape, size) in enumerate(requested):
        value_view = evaluator_primal_by_name.get(name)
        dual_view = dual_by_name.get(f"d{position}")
        if value_view is None or dual_view is None:
            raise ValueError(
                f"constraint evaluator lost requested constraint {index}"
            )
        if value_view.size != size or dual_view.size != size:
            raise ValueError(
                f"constraint evaluator changed size of constraint {index}"
            )
        value_views.append(
            _ConstraintValueView(
                f"v{index}", index, label, shape, size, output_offset
            )
        )
        output_offset += size
        value_columns = set(range(value_view.offset, value_view.offset + size))
        for local in range(size):
            row = dual_view.offset + local
            expected_value_column = value_view.offset + local
            denominator = None
            terms: list[tuple[int, int]] = []
            for column, entry in A_entries[row]:
                if column == expected_value_column:
                    denominator = entry
                    continue
                if column in value_columns:
                    raise ValueError(
                        f"constraint evaluator couples output rows for constraint {index}"
                    )
                main_column = evaluator_to_main_column.get(column)
                if main_column is None:
                    raise ValueError(
                        f"constraint evaluator row {row} references unmapped "
                        f"canonical variable column {column}"
                    )
                terms.append((main_column, entry))
            if denominator is None:
                raise ValueError(
                    f"constraint evaluator has no output coefficient for "
                    f"constraint {index}, scalar {local}"
                )
            rhs_rows.append(row)
            denominator_entries.append(denominator)
            for main_column, entry in terms:
                term_primal_columns.append(main_column)
                coefficient_entries.append(entry)
            term_row_ptr.append(len(coefficient_entries))

    return _ConstraintValueProgram(
        tuple(value_views),
        _select_sparse_map_rows(evaluator_b_map, rhs_rows),
        _select_sparse_map_rows(evaluator_A_map, denominator_entries),
        _select_sparse_map_rows(evaluator_A_map, coefficient_entries),
        np.asarray(term_row_ptr, dtype=np.uint32),
        np.asarray(term_primal_columns, dtype=np.uint32),
    )
'''
    text = path.read_text()
    anchor = "\n\ndef _cpp_float(value: float) -> str:\n"
    if "def _compile_constraint_value_program(" not in text:
        if text.count(anchor) != 1:
            raise RuntimeError("direct_clarabel.py: missing evaluator insertion anchor")
        path.write_text(text.replace(anchor, insertion + anchor, 1))

    replace_once(
        path,
        """def _parameter_dirty_blocks(\n    mapping_by_block: Mapping[str, _SparseMap],\n""",
        """def _parameter_affects_constraint_values(\n    program: _ConstraintValueProgram,\n    offset: int,\n    size: int,\n) -> bool:\n    return any(\n        np.any(\n            (mapping.columns >= offset)\n            & (mapping.columns < offset + size)\n        )\n        for mapping in (\n            program.rhs_map,\n            program.denominator_map,\n            program.coefficient_map,\n        )\n    )\n\n\ndef _parameter_dirty_blocks(\n    mapping_by_block: Mapping[str, _SparseMap],\n""",
    )
    replace_once(
        path,
        """    compiled: _CompiledCanonicalProgram,\n    enable_settings: Iterable[str],\n) -> Path:\n""",
        """    compiled: _CompiledCanonicalProgram,\n    constraint_program: _ConstraintValueProgram,\n    enable_settings: Iterable[str],\n    clarabel_settings: Mapping[str, Any],\n) -> Path:\n""",
    )
    replace_once(
        path,
        """                \"dirty_blocks\": _parameter_dirty_blocks(\n                    compiled.parameter_maps,\n                    offset,\n                    int(parameter.size),\n                ),\n""",
        """                \"dirty_blocks\": _parameter_dirty_blocks(\n                    compiled.parameter_maps,\n                    offset,\n                    int(parameter.size),\n                ),\n                \"constraint_value_dirty\": _parameter_affects_constraint_values(\n                    constraint_program, offset, int(parameter.size)\n                ),\n""",
    )
    maps_anchor = """    matrices = {\n"""
    constraint_map_render = """    constraint_maps = {\n        name: {\n            \"name\": name,\n            \"rows\": mapping.rows,\n            \"nnz\": int(mapping.values.size),\n            \"values\": _cpp_array(mapping.values, _cpp_float, per_line=6),\n            \"columns\": _cpp_array(\n                mapping.columns, lambda value: str(int(value))\n            ),\n            \"row_ptr\": _cpp_array(\n                mapping.row_ptr, lambda value: str(int(value))\n            ),\n        }\n        for name, mapping in {\n            \"rhs\": constraint_program.rhs_map,\n            \"denominator\": constraint_program.denominator_map,\n            \"coefficient\": constraint_program.coefficient_map,\n        }.items()\n    }\n    constraint_values = [\n        {\n            \"name\": value.name,\n            \"index\": index,\n            \"offset\": value.offset,\n            \"size\": value.size,\n        }\n        for index, value in enumerate(constraint_program.values)\n    ]\n    fixed_settings = []\n    for name, value in sorted(clarabel_settings.items()):\n        setting_type = _SETTING_TYPES.get(name)\n        if setting_type is None:\n            raise ValueError(f\"unsupported Clarabel setting {name!r}\")\n        if setting_type == \"bool\":\n            if not isinstance(value, bool):\n                raise TypeError(f\"Clarabel setting {name!r} must be bool\")\n            literal = \"true\" if value else \"false\"\n        elif setting_type == \"std::uint32_t\":\n            if not isinstance(value, int) or isinstance(value, bool) or value < 0:\n                raise TypeError(\n                    f\"Clarabel setting {name!r} must be a nonnegative int\"\n                )\n            literal = str(value)\n        else:\n            literal = _cpp_float(float(value))\n        fixed_settings.append({\"name\": name, \"value\": literal})\n\n"""
    replace_once(path, maps_anchor, constraint_map_render + maps_anchor)
    replace_once(
        path,
        """            maps=maps,\n            matrices=matrices,\n            cones=compiled.cone_initializers,\n            settings=settings,\n""",
        """            maps=maps,\n            constraint_maps=constraint_maps,\n            constraint_values=constraint_values,\n            constraint_value_scalar_count=constraint_program.scalar_count,\n            constraint_value_term_count=int(\n                constraint_program.term_primal_columns.size\n            ),\n            constraint_value_term_row_ptr=_cpp_array(\n                constraint_program.term_row_ptr,\n                lambda value: str(int(value)),\n            ),\n            constraint_value_term_primal_columns=_cpp_array(\n                constraint_program.term_primal_columns,\n                lambda value: str(int(value)),\n            ),\n            matrices=matrices,\n            cones=compiled.cone_initializers,\n            settings=settings,\n            fixed_settings=fixed_settings,\n""",
    )
    replace_once(
        path,
        """    field_aliases: Mapping[str, str] | None = None,\n    force: bool = False,\n""",
        """    constraint_value_indices: Iterable[int] = (),\n    clarabel_settings: Mapping[str, Any] | None = None,\n    field_aliases: Mapping[str, str] | None = None,\n    force: bool = False,\n""",
    )
    replace_once(
        path,
        """        DualLayout,\n        FieldAlias,\n""",
        """        ConstraintValueLayout,\n        DualLayout,\n        FieldAlias,\n""",
    )
    replace_once(
        path,
        """    header = _emit_direct_header(\n        root,\n""",
        """    constraint_program = _compile_constraint_value_program(\n        problem,\n        compiled,\n        constraint_value_indices,\n        parameter_shard_size=parameter_shard_size,\n    )\n    header = _emit_direct_header(\n        root,\n""",
    )
    replace_once(
        path,
        """        compiled=compiled,\n        enable_settings=enable_settings,\n    )\n""",
        """        compiled=compiled,\n        constraint_program=constraint_program,\n        enable_settings=enable_settings,\n        clarabel_settings=dict(clarabel_settings or {}),\n    )\n""",
    )
    replace_once(
        path,
        """    aliases = tuple(\n        FieldAlias(name, primal_name)\n        for name, primal_name in sorted(alias_mapping.items())\n    )\n    manifest = {\n        \"schema_version\": 4,\n""",
        """    aliases = tuple(\n        FieldAlias(name, primal_name)\n        for name, primal_name in sorted(alias_mapping.items())\n    )\n    public_constraint_values = tuple(\n        ConstraintValueLayout(\n            value.name,\n            value.constraint_index,\n            value.label,\n            value.shape,\n            value.size,\n        )\n        for value in constraint_program.values\n    )\n    manifest = {\n        \"schema_version\": 5,\n""",
    )
    replace_once(
        path,
        """        \"aliases\": [\n            {\"name\": item.name, \"primal_name\": item.primal_name}\n            for item in aliases\n        ],\n""",
        """        \"aliases\": [\n            {\"name\": item.name, \"primal_name\": item.primal_name}\n            for item in aliases\n        ],\n        \"constraint_values\": [\n            {\n                \"name\": item.name,\n                \"constraint_index\": item.constraint_index,\n                \"label\": item.label,\n                \"shape\": list(item.shape),\n                \"size\": item.size,\n            }\n            for item in public_constraint_values\n        ],\n        \"clarabel_settings\": dict(clarabel_settings or {}),\n""",
    )
    replace_once(
        path,
        """        clarabel,\n        instrument_count,\n    )\n""",
        """        clarabel,\n        instrument_count,\n        public_constraint_values,\n    )\n""",
    )
    replace_once(
        path,
        """        DualLayout,\n        FieldAlias,\n        GeneratedClarabelProgram,\n""",
        """        ConstraintValueLayout,\n        DualLayout,\n        FieldAlias,\n        GeneratedClarabelProgram,\n""",
    )
    replace_once(
        path,
        """    if manifest.get(\"schema_version\") != 4:\n""",
        """    if manifest.get(\"schema_version\") != 5:\n""",
    )
    replace_once(
        path,
        """    instrument_count = manifest.get(\"instrument_count\")\n    return GeneratedClarabelProgram(\n""",
        """    constraint_values = tuple(\n        ConstraintValueLayout(\n            str(item[\"name\"]),\n            int(item[\"constraint_index\"]),\n            None if item.get(\"label\") is None else str(item[\"label\"]),\n            tuple(int(extent) for extent in item[\"shape\"]),\n            int(item[\"size\"]),\n        )\n        for item in manifest.get(\"constraint_values\", ())\n    )\n    instrument_count = manifest.get(\"instrument_count\")\n    return GeneratedClarabelProgram(\n""",
    )
    replace_once(
        path,
        """        clarabel,\n        None if instrument_count is None else int(instrument_count),\n    )\n""",
        """        clarabel,\n        None if instrument_count is None else int(instrument_count),\n        constraint_values,\n    )\n""",
    )


def patch_template() -> None:
    path = ROOT / (
        "src/trading_dsl_engine/cpp_stream/optimizer/templates/"
        "direct_clarabel_instance.hpp.j2"
    )
    replace_once(
        path,
        """{% endfor %}\n{% for name, matrix in matrices.items() %}\n""",
        """{% endfor %}\n{% for name, mapping in constraint_maps.items() %}\n  alignas(64) inline static constexpr std::array<double, {{ mapping.nnz }}>\n      constraint_{{ name }}_map_values_{{ mapping[\"values\"] }};\n  alignas(64) inline static constexpr std::array<std::uint32_t, {{ mapping.nnz }}>\n      constraint_{{ name }}_map_columns_{{ mapping.columns }};\n  alignas(64) inline static constexpr std::array<std::uint32_t, {{ mapping.rows + 1 }}>\n      constraint_{{ name }}_map_row_ptr_{{ mapping.row_ptr }};\n{% endfor %}\n  alignas(64) inline static constexpr std::array<std::uint32_t, {{ constraint_value_scalar_count + 1 }}>\n      constraint_value_term_row_ptr_{{ constraint_value_term_row_ptr }};\n  alignas(64) inline static constexpr std::array<std::uint32_t, {{ constraint_value_term_count }}>\n      constraint_value_term_primal_columns_{{ constraint_value_term_primal_columns }};\n{% for name, matrix in matrices.items() %}\n""",
    )
    replace_once(
        path,
        """  std::array<double, 1> d_value_{};\n""",
        """  std::array<double, 1> d_value_{};\n  alignas(64) std::array<double, {{ constraint_value_scalar_count }}> constraint_values_{};\n  alignas(64) std::array<double, {{ constraint_value_scalar_count }}> constraint_rhs_values_{};\n  alignas(64) std::array<double, {{ constraint_value_scalar_count }}> constraint_denominator_values_{};\n  alignas(64) std::array<double, {{ constraint_value_term_count }}> constraint_coefficient_values_{};\n""",
    )
    replace_once(
        path,
        """  bool d_dirty_{true};\n""",
        """  bool d_dirty_{true};\n  bool constraint_values_dirty_{{ \"{true}\" if constraint_value_scalar_count else \"{false}\" }};\n""",
    )
    replace_once(
        path,
        """{% endfor %}\n  void initialize_solver() {\n""",
        """{% endfor %}\n{% for name, mapping in constraint_maps.items() %}\n  void canonicalize_constraint_{{ name }}() noexcept {\n    auto& output = constraint_{{ name }}_values_;\n    for (std::size_t row = 0; row < {{ mapping.rows }}; ++row) {\n      double value = 0.0;\n      const auto begin = constraint_{{ name }}_map_row_ptr_[row];\n      const auto end = constraint_{{ name }}_map_row_ptr_[row + 1];\n      for (std::uint32_t index = begin; index < end; ++index) {\n        value += constraint_{{ name }}_map_values_[index]\n               * parameters_[constraint_{{ name }}_map_columns_[index]];\n      }\n      output[row] = value;\n    }\n  }\n\n{% endfor %}\n  void evaluate_constraint_values() noexcept {\n{% if constraint_value_scalar_count %}\n    canonicalize_constraint_rhs();\n    canonicalize_constraint_denominator();\n    canonicalize_constraint_coefficient();\n    for (std::size_t row = 0; row < {{ constraint_value_scalar_count }}; ++row) {\n      double value = constraint_rhs_values_[row];\n      const auto begin = constraint_value_term_row_ptr_[row];\n      const auto end = constraint_value_term_row_ptr_[row + 1];\n      for (std::uint32_t term = begin; term < end; ++term) {\n        value -= constraint_coefficient_values_[term]\n               * solution_.x[constraint_value_term_primal_columns_[term]];\n      }\n      constraint_values_[row] =\n          value / constraint_denominator_values_[row];\n    }\n{% endif %}\n    constraint_values_dirty_ = false;\n  }\n\n  void initialize_solver() {\n""",
    )
    replace_once(
        path,
        """    settings_.verbose = false;\n    settings_.presolve_enable = false;\n""",
        """    settings_.verbose = false;\n    settings_.presolve_enable = false;\n{% for setting in fixed_settings %}\n    settings_.{{ setting.name }} = {{ setting.value }};\n{% endfor %}\n""",
    )
    replace_once(
        path,
        """      d_dirty_ = false;\n      return;\n""",
        """      if (constraint_values_dirty_) evaluate_constraint_values();\n      d_dirty_ = false;\n      return;\n""",
    )
    replace_once(
        path,
        """    solution_ = clarabel_DefaultSolver_solution(solver_);\n    P_dirty_ = q_dirty_ = A_dirty_ = b_dirty_ = d_dirty_ = false;\n""",
        """    solution_ = clarabel_DefaultSolver_solution(solver_);\n    evaluate_constraint_values();\n    P_dirty_ = q_dirty_ = A_dirty_ = b_dirty_ = d_dirty_ = false;\n""",
    )
    replace_once(
        path,
        """{% for block in parameter.dirty_blocks %}\n    {{ block }}_dirty_ = true;\n{% endfor %}\n  }\n""",
        """{% for block in parameter.dirty_blocks %}\n    {{ block }}_dirty_ = true;\n{% endfor %}\n{% if parameter.constraint_value_dirty %}\n    constraint_values_dirty_ = true;\n{% endif %}\n  }\n""",
    )
    replace_once(
        path,
        """{% for block in parameter.dirty_blocks %}\n      {{ block }}_dirty_ = true;\n{% endfor %}\n    } else\n""",
        """{% for block in parameter.dirty_blocks %}\n      {{ block }}_dirty_ = true;\n{% endfor %}\n{% if parameter.constraint_value_dirty %}\n      constraint_values_dirty_ = true;\n{% endif %}\n    } else\n""",
    )
    replace_once(
        path,
        """  template <std::size_t Index>\n  [[nodiscard]] double info() const noexcept {\n""",
        """{% for value in constraint_values %}\n  [[nodiscard]] std::span<const double> constraint_value_{{ value.name }}() noexcept {\n    return std::span<const double>(\n        constraint_values_.data() + {{ value.offset }}, {{ value.size }});\n  }\n\n{% endfor %}\n  template <std::size_t Index>\n  [[nodiscard]] std::span<const double> constraint_value() noexcept {\n{% for value in constraint_values %}\n    if constexpr (Index == {{ value.index }}) return constraint_value_{{ value.name }}();\n    else\n{% endfor %}\n    {\n      static_assert(Index < {{ constraint_values|length }}, \"invalid constraint value index\");\n      return {};\n    }\n  }\n\n  template <std::size_t Index>\n  [[nodiscard]] double info() const noexcept {\n""",
    )


def patch_codegen() -> None:
    path = ROOT / "src/trading_dsl_engine/cpp_stream/python/codegen.py"
    replace_once(
        path,
        """                            \"dual\": \"stackdsl::ClarabelResultKind::Dual\",\n                            \"info\": \"stackdsl::ClarabelResultKind::Info\",\n""",
        """                            \"dual\": \"stackdsl::ClarabelResultKind::Dual\",\n                            \"constraint_value\": \"stackdsl::ClarabelResultKind::ConstraintValue\",\n                            \"info\": \"stackdsl::ClarabelResultKind::Info\",\n""",
    )


def patch_clarabel_node() -> None:
    path = ROOT / (
        "src/trading_dsl_engine/cpp_stream/cpp/stackdsl/ops/clarabel_program.hpp"
    )
    replace_once(
        path,
        """    Dual,\n    Info,\n""",
        """    Dual,\n    ConstraintValue,\n    Info,\n""",
    )
    replace_once(
        path,
        """                if constexpr (\n                    Projection::kind == ClarabelResultKind::Primal\n                ) {\n                    return program_.template primal<Projection::source_index>();\n                } else {\n                    return program_.template dual<Projection::source_index>();\n                }\n""",
        """                if constexpr (\n                    Projection::kind == ClarabelResultKind::Primal\n                ) {\n                    return program_.template primal<Projection::source_index>();\n                } else if constexpr (\n                    Projection::kind == ClarabelResultKind::Dual\n                ) {\n                    return program_.template dual<Projection::source_index>();\n                } else {\n                    static_assert(\n                        Projection::kind == ClarabelResultKind::ConstraintValue\n                    );\n                    return program_.template constraint_value<\n                        Projection::source_index\n                    >();\n                }\n""",
    )


if __name__ == "__main__":
    patch_direct_clarabel()
    patch_template()
    patch_codegen()
    patch_clarabel_node()
