from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap

ROOT = Path.cwd()


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one anchor, found {count}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, 1))


def run(*args: str, check: bool = True, env: dict[str, str] | None = None):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, cwd=ROOT, env=merged, check=check)


def write_red_tests() -> None:
    p = ROOT / "tests/examples/test_cpp_stream_mpo_diagnostics.py"
    text = p.read_text()
    text = text.replace(
        '        "mpo_spread_cost",\n        "mpo_gross_pnl",\n',
        '        "mpo_spread_cost",\n        "mpo_objective",\n        "mpo_gross_pnl",\n',
        1,
    )
    marker = "def test_mpo_exposes_spread_cost_as_generic_named_primal():"
    if marker not in text:
        raise RuntimeError("missing previous spread-cost test")
    prefix = text.split(marker, 1)[0]
    replacement = r'''
def test_mpo_exposes_spread_cost_as_named_objective_term_only():
    assert example.HORIZONS == (2, 4, 8, 16, 32, 64, 128)
    assert example.TRADE_STARTS == (0, 2, 4, 8, 16, 32, 64)
    source = __import__("inspect").getsource(example._formula)
    assert "-ts_zscore(" in source

    n_horizons = len(example.HORIZONS)
    n_assets = 3
    zeros = np.zeros((n_horizons, n_assets))
    factors = [np.eye(n_assets) for _ in range(n_horizons)]
    problem, named_values = example.MPO.factory(
        zeros,
        np.full(n_assets, 1e-4),
        np.zeros(n_assets),
        *factors,
        np.ones((n_horizons, n_assets)),
        example.RISK_RADIUS,
    )
    assert problem.is_dpp()
    assert set(named_values) == {"spread_cost"}
    variables = {variable.name(): variable for variable in problem.variables()}
    assert set(variables) == {"weights", "previous_weights"}
    assert len(problem.constraints) == 3 + n_horizons

    rng = np.random.default_rng(51)
    parameter_values = {
        "expected_returns": rng.normal(scale=2e-4, size=(n_horizons, n_assets)),
        "half_spread": np.array([4e-5, 6e-5, 8e-5]),
        "current_weights": np.array([0.01, -0.02, 0.01]),
        "trade_allowed": np.ones((n_horizons, n_assets)),
        "risk_radius": example.RISK_RADIUS,
        **{f"risk_factor_{h}": np.eye(n_assets) for h in range(n_horizons)},
    }
    for parameter in problem.parameters():
        parameter.value = parameter_values[parameter.name()]
    problem.solve(
        solver=cp.CLARABEL,
        presolve_enable=False,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
    )
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    weights = np.asarray(variables["weights"].value)
    current = parameter_values["current_weights"]
    delta = weights - np.vstack([current, weights[:-1]])
    direct_cost = np.sum(parameter_values["half_spread"] * np.abs(delta))
    np.testing.assert_allclose(
        float(named_values["spread_cost"].value),
        direct_cost,
        rtol=2e-7,
        atol=2e-10,
    )


def test_plotting_shows_every_tight_layout_and_includes_objective():
    import inspect

    source = inspect.getsource(example._plot_diagnostics)
    assert source.count("fig.tight_layout()") == source.count("plt.show()")
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if "fig.tight_layout()" in line:
            assert lines[index + 1].strip() == "plt.show()"
    assert 'values["mpo_objective"]' in source
    assert '"mpo_objective.png"' in source
'''
    p.write_text(prefix + textwrap.dedent(replacement).lstrip())

    q = ROOT / "tests/trading_dsl_engine/cpp_stream/test_cvxpy_constraint_values_and_guard.py"
    qtext = q.read_text()
    qtext += textwrap.dedent(r'''


def test_named_objective_term_projects_without_adding_a_solver_primal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    native = build_current_clarabel(cache_dir=tmp_path / "clarabel-native")

    @cvxpy_program(
        cache_dir=tmp_path / "program-cache",
        clarabel=native,
    )
    def L1Target(target):
        target = cp.Parameter(target.shape, name="target")
        weights = cp.Variable(target.shape, name="weights")
        l1_term = 0.25 * cp.norm1(weights)
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(weights - target) + l1_term)
        )
        return problem, {"l1_term": l1_term}

    prototype = L1Target.resolve_for_types(
        {"target": VECTOR},
        requested_fields=frozenset({"l1_term"}),
        n_instruments=None,
    )
    assert [primal.name for primal in prototype.primals] == ["weights"]
    assert prototype.resolve_field("l1_term").kind == "constraint_value"

    rows, assets = 5, 3
    data = {"target": np.asarray([
        [0.4, -0.2, 0.1],
        [0.2, -0.4, 0.3],
        [-0.1, 0.2, -0.3],
        [0.5, 0.1, -0.2],
        [0.0, 0.3, -0.1],
    ])}
    expression = L1Target(target=var("target"))
    monkeypatch.setenv(
        "TRADING_DSL_ENGINE_CPP_STREAM_CACHE", str(tmp_path / "runner-cache")
    )
    runtime = compile_formula(
        [get_field(expression, "weights"), get_field(expression, "l1_term")],
        data,
        n_instruments=assets,
    )
    weights, l1_values = runtime.run(
        out_path=tmp_path / "named-objective.npy"
    ).load(mmap_mode=None)
    expected = 0.25 * np.abs(weights).sum(axis=1)
    np.testing.assert_allclose(l1_values, expected, rtol=2e-6, atol=2e-8)

    manifests = tuple(
        (tmp_path / "program-cache").rglob("clarabel_program_manifest.json")
    )
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert [item["name"] for item in manifest["primals"]] == ["weights"]
    assert any(
        item["name"] == "l1_term" and item["constraint_index"] == -1
        for item in manifest["constraint_values"]
    )
''')
    q.write_text(qtext)


def implement() -> None:
    factory = "src/trading_dsl_engine/cpp_stream/optimizer/factory.py"
    replace_once(
        factory,
        "def _call_with_named_values(factory, signature, values):\n",
        r'''def _split_problem_result(cp, value):
    if isinstance(value, cp.Problem):
        return value, {}
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], cp.Problem)
        and isinstance(value[1], Mapping)
    ):
        problem, named_values = value
        normalized = {}
        for raw_name, expression in named_values.items():
            name = str(raw_name)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise KeyError(f"invalid named CVXPY objective term {name!r}")
            if not isinstance(expression, cp.Expression):
                raise TypeError(
                    f"named CVXPY objective term {name!r} must be an Expression"
                )
            normalized[name] = expression
        return problem, normalized
    raise TypeError(
        "CVXPY program factory must return cp.Problem or "
        "(cp.Problem, mapping_of_named_objective_terms)"
    )


def _contains_expression_identity(root, target):
    if root is target:
        return True
    return any(
        _contains_expression_identity(argument, target)
        for argument in getattr(root, "args", ())
    )


def _requested_objective_values(problem, named_values, requested_fields):
    reserved = {variable.name() for variable in problem.variables()} | {
        "objective",
        "objective_value",
        "obj_val",
        "iterations",
        "status",
        "primal_residual",
        "dual_residual",
    }
    collisions = sorted(set(named_values) & reserved)
    if collisions:
        raise KeyError(
            f"named CVXPY objective terms collide with result fields: {collisions}"
        )
    result = {}
    for name in sorted(set(named_values) & set(requested_fields)):
        expression = named_values[name]
        if tuple(int(extent) for extent in expression.shape) != ():
            raise ValueError(f"named CVXPY objective term {name!r} must be scalar")
        if not _contains_expression_identity(problem.objective.expr, expression):
            raise ValueError(
                f"named CVXPY objective term {name!r} must be a subexpression "
                "of the problem objective"
            )
        result[name] = expression
    return result


def _call_with_named_values(factory, signature, values):
''',
    )
    replace_once(
        factory,
        r'''        problem = _call_with_named_values(
            self.factory, self.signature, arguments
        )
        if not isinstance(problem, cp.Problem):
            raise TypeError(
                f"{self.factory.__qualname__} must return cvxpy.Problem, "
                f"got {type(problem).__name__}"
            )
        constraint_values = _constraint_value_layouts(
            cp, problem, requested_fields
        )
''',
        r'''        raw_result = _call_with_named_values(
            self.factory, self.signature, arguments
        )
        problem, named_values = _split_problem_result(cp, raw_result)
        constraint_values = _constraint_value_layouts(
            cp, problem, requested_fields
        )
        objective_values = _requested_objective_values(
            problem, named_values, requested_fields
        )
''',
    )
    replace_once(factory, "        return problem, constraint_values\n", "        return problem, constraint_values, objective_values\n")
    replace_once(factory, "    def _prototype(self, problem, constraint_values, n_instruments):\n", "    def _prototype(self, problem, constraint_values, objective_values, n_instruments):\n")
    replace_once(
        factory,
        r'''            constraint_values,
        )

    def _cache_key(
''',
        r'''            constraint_values
            + tuple(
                ConstraintValueLayout(name, -1, None, (), 1)
                for name in objective_values
            ),
        )

    def _cache_key(
''',
    )
    replace_once(
        factory,
        r'''        constraint_values: tuple[ConstraintValueLayout, ...],
    ) -> str:
''',
        r'''        constraint_values: tuple[ConstraintValueLayout, ...],
        objective_values: Mapping[str, Any],
    ) -> str:
''',
    )
    replace_once(
        factory,
        r'''            "constraint_values": [
                (value.constraint_index, value.label, value.shape)
                for value in constraint_values
            ],
''',
        r'''            "constraint_values": [
                (value.constraint_index, value.label, value.shape)
                for value in constraint_values
            ],
            "objective_values": [
                (name, str(expression))
                for name, expression in sorted(objective_values.items())
            ],
''',
    )
    replace_once(factory, "        problem, constraint_values = self._instantiate_problem(\n", "        problem, constraint_values, objective_values = self._instantiate_problem(\n")
    replace_once(
        factory,
        r'''            return self._prototype(problem, constraint_values, resolved_n)

        cache_key = self._cache_key(
            problem, int(n_instruments), constraint_values
        )
''',
        r'''            return self._prototype(
                problem, constraint_values, objective_values, resolved_n
            )

        cache_key = self._cache_key(
            problem, int(n_instruments), constraint_values, objective_values
        )
''',
    )
    replace_once(
        factory,
        r'''                    constraint_value_indices=tuple(
                        value.constraint_index for value in constraint_values
                    ),
                    field_aliases={},
''',
        r'''                    constraint_value_indices=tuple(
                        value.constraint_index for value in constraint_values
                    ),
                    objective_value_expressions=objective_values,
                    field_aliases={},
''',
    )

    direct = "src/trading_dsl_engine/cpp_stream/optimizer/direct_clarabel.py"
    replace_once(
        direct,
        "@dataclass(frozen=True, slots=True)\nclass _CompiledCanonicalProgram:\n",
        r'''@dataclass(frozen=True, slots=True)
class _ObjectiveValueView:
    name: str
    P_map: _SparseMap
    q_map: _SparseMap
    d_map: _SparseMap


@dataclass(frozen=True, slots=True)
class _CompiledCanonicalProgram:
''',
    )
    replace_once(
        direct,
        "def _cpp_float(value: float) -> str:\n",
        r'''def _sparse_map_matrix(mapping: _SparseMap, columns: int):
    return sparse.csr_matrix(
        (mapping.values, mapping.columns, mapping.row_ptr),
        shape=(mapping.rows, columns),
    )


def _sparse_map_difference(
    left: _SparseMap,
    right: _SparseMap,
    *,
    columns: int,
    scale: float = 1.0,
) -> _SparseMap:
    if left.rows != right.rows:
        raise ValueError("named objective term changed canonical map row count")
    matrix = (
        _sparse_map_matrix(left, columns) - _sparse_map_matrix(right, columns)
    ) * scale
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return _SparseMap(
        matrix.shape[0],
        np.asarray(matrix.data, dtype=np.float64),
        np.asarray(matrix.indices, dtype=np.uint32),
        np.asarray(matrix.indptr, dtype=np.uint32),
    )


def _compile_objective_value_program(
    problem: Any,
    compiled: _CompiledCanonicalProgram,
    objective_value_expressions: Mapping[str, Any],
    *,
    parameter_shard_size: int,
) -> tuple[_ObjectiveValueView, ...]:
    import cvxpy as cp

    if not objective_value_expressions:
        return ()
    parameter_columns = sum(int(p.size) for p in problem.parameters()) + 1
    result = []
    for name, expression in sorted(objective_value_expressions.items()):
        tagged_expr = problem.objective.expr.tree_copy(
            id_objects={id(expression): 2.0 * expression}
        )
        tagged_problem = cp.Problem(
            type(problem.objective)(tagged_expr), list(problem.constraints)
        )
        tagged = compile_sharded_canonical_program(
            tagged_problem, parameter_shard_size=parameter_shard_size
        )
        same_structure = (
            np.array_equal(tagged.P.row_indices, compiled.P.row_indices)
            and np.array_equal(tagged.P.column_ptr, compiled.P.column_ptr)
            and np.array_equal(tagged.A.row_indices, compiled.A.row_indices)
            and np.array_equal(tagged.A.column_ptr, compiled.A.column_ptr)
            and tagged.cone_initializers == compiled.cone_initializers
            and tagged.primals == compiled.primals
            and tagged.parameter_offsets == compiled.parameter_offsets
        )
        if not same_structure:
            raise ValueError(
                f"named objective term {name!r} changes the canonical problem; "
                "name an existing scalar objective subexpression"
            )
        for block in ("A", "b"):
            difference = _sparse_map_difference(
                tagged.parameter_maps[block],
                compiled.parameter_maps[block],
                columns=parameter_columns,
            )
            if difference.values.size:
                raise ValueError(
                    f"named objective term {name!r} unexpectedly changes {block}"
                )
        sign = -1.0 if isinstance(problem.objective, cp.Maximize) else 1.0
        result.append(
            _ObjectiveValueView(
                str(name),
                _sparse_map_difference(
                    tagged.parameter_maps["P"], compiled.parameter_maps["P"],
                    columns=parameter_columns, scale=sign,
                ),
                _sparse_map_difference(
                    tagged.parameter_maps["q"], compiled.parameter_maps["q"],
                    columns=parameter_columns, scale=sign,
                ),
                _sparse_map_difference(
                    tagged.parameter_maps["d"], compiled.parameter_maps["d"],
                    columns=parameter_columns, scale=sign,
                ),
            )
        )
    return tuple(result)


def _cpp_float(value: float) -> str:
''',
    )
    replace_once(direct, "    constraint_program: _ConstraintValueProgram,\n    enable_settings: Iterable[str],\n", "    constraint_program: _ConstraintValueProgram,\n    objective_values: tuple[_ObjectiveValueView, ...],\n    enable_settings: Iterable[str],\n")
    replace_once(
        direct,
        "    fixed_settings = []\n",
        r'''    objective_value_specs = []
    for index, value in enumerate(objective_values):
        maps_by_name = {}
        for map_name, mapping in (
            ("P", value.P_map), ("q", value.q_map), ("d", value.d_map)
        ):
            maps_by_name[map_name] = {
                "name": map_name,
                "rows": mapping.rows,
                "nnz": int(mapping.values.size),
                "values": _cpp_array(mapping.values, _cpp_float, per_line=6),
                "columns": _cpp_array(
                    mapping.columns, lambda item: str(int(item))
                ),
                "row_ptr": _cpp_array(
                    mapping.row_ptr, lambda item: str(int(item))
                ),
            }
        objective_value_specs.append(
            {
                "name": _safe_identifier(value.name, label="named objective term"),
                "index": index,
                "source_index": len(constraint_program.values) + index,
                "maps": maps_by_name,
            }
        )

    fixed_settings = []
''',
    )
    replace_once(direct, "            constraint_values=constraint_values,\n            constraint_value_scalar_count=constraint_program.scalar_count,\n", "            constraint_values=constraint_values,\n            objective_values=objective_value_specs,\n            constraint_value_scalar_count=constraint_program.scalar_count,\n")
    replace_once(direct, "    constraint_value_indices: Iterable[int] = (),\n    clarabel_settings: Mapping[str, Any] | None = None,\n", "    constraint_value_indices: Iterable[int] = (),\n    objective_value_expressions: Mapping[str, Any] | None = None,\n    clarabel_settings: Mapping[str, Any] | None = None,\n")
    replace_once(
        direct,
        r'''    constraint_program = _compile_constraint_value_program(
        problem,
        compiled,
        constraint_value_indices,
        parameter_shard_size=parameter_shard_size,
    )
    header = _emit_direct_header(
''',
        r'''    constraint_program = _compile_constraint_value_program(
        problem,
        compiled,
        constraint_value_indices,
        parameter_shard_size=parameter_shard_size,
    )
    objective_values = _compile_objective_value_program(
        problem,
        compiled,
        dict(objective_value_expressions or {}),
        parameter_shard_size=parameter_shard_size,
    )
    header = _emit_direct_header(
''',
    )
    replace_once(direct, "        compiled=compiled,\n        constraint_program=constraint_program,\n        enable_settings=enable_settings,\n", "        compiled=compiled,\n        constraint_program=constraint_program,\n        objective_values=objective_values,\n        enable_settings=enable_settings,\n")
    replace_once(
        direct,
        r'''    public_constraint_values = tuple(
        ConstraintValueLayout(
            value.name,
            value.constraint_index,
            value.label,
            value.shape,
            value.size,
        )
        for value in constraint_program.values
    )
''',
        r'''    public_constraint_values = tuple(
        ConstraintValueLayout(
            value.name,
            value.constraint_index,
            value.label,
            value.shape,
            value.size,
        )
        for value in constraint_program.values
    ) + tuple(
        ConstraintValueLayout(value.name, -1, None, (), 1)
        for value in objective_values
    )
''',
    )

    template = "src/trading_dsl_engine/cpp_stream/optimizer/templates/direct_clarabel_instance.hpp.j2"
    replace_once(
        template,
        "{% endfor %}\n  alignas(64) inline static constexpr std::array<std::uint32_t, {{ constraint_value_scalar_count + 1 }}>\n",
        r'''{% endfor %}
{% for value in objective_values %}
{% for name, mapping in value.maps.items() %}
  alignas(64) inline static constexpr std::array<double, {{ mapping.nnz }}>
      objective_{{ value.index }}_{{ name }}_map_values_{{ mapping["values"] }};
  alignas(64) inline static constexpr std::array<std::uint32_t, {{ mapping.nnz }}>
      objective_{{ value.index }}_{{ name }}_map_columns_{{ mapping.columns }};
  alignas(64) inline static constexpr std::array<std::uint32_t, {{ mapping.rows + 1 }}>
      objective_{{ value.index }}_{{ name }}_map_row_ptr_{{ mapping.row_ptr }};
{% endfor %}
{% endfor %}
  alignas(64) inline static constexpr std::array<std::uint32_t, {{ constraint_value_scalar_count + 1 }}>
''',
    )
    replace_once(template, "  alignas(64) std::array<double, {{ constraint_value_term_count }}> constraint_coefficient_values_{};\n", "  alignas(64) std::array<double, {{ constraint_value_term_count }}> constraint_coefficient_values_{};\n  alignas(64) std::array<double, {{ objective_values|length }}> objective_values_{};\n")
    replace_once(
        template,
        "  void evaluate_constraint_values() noexcept {\n",
        r'''  template <std::size_t Nnz, std::size_t RowsPlusOne>
  [[nodiscard]] double evaluate_parameter_map_row(
      const std::array<double, Nnz>& values,
      const std::array<std::uint32_t, Nnz>& columns,
      const std::array<std::uint32_t, RowsPlusOne>& row_ptr,
      std::size_t row) const noexcept {
    double value = 0.0;
    for (std::uint32_t index = row_ptr[row]; index < row_ptr[row + 1]; ++index) {
      value += values[index] * parameters_[columns[index]];
    }
    return value;
  }

  void evaluate_objective_values() noexcept {
{% for value in objective_values %}
    double value_{{ value.index }} = evaluate_parameter_map_row(
        objective_{{ value.index }}_d_map_values_,
        objective_{{ value.index }}_d_map_columns_,
        objective_{{ value.index }}_d_map_row_ptr_, 0);
    for (std::size_t row = 0; row < {{ value.maps.q.rows }}; ++row) {
      value_{{ value.index }} += evaluate_parameter_map_row(
          objective_{{ value.index }}_q_map_values_,
          objective_{{ value.index }}_q_map_columns_,
          objective_{{ value.index }}_q_map_row_ptr_, row) * solution_.x[row];
    }
    for (std::size_t column = 0; column < {{ matrices.P.columns }}; ++column) {
      for (std::size_t entry = P_column_ptr_[column]; entry < P_column_ptr_[column + 1]; ++entry) {
        const std::size_t row = P_row_indices_[entry];
        const double coefficient = evaluate_parameter_map_row(
            objective_{{ value.index }}_P_map_values_,
            objective_{{ value.index }}_P_map_columns_,
            objective_{{ value.index }}_P_map_row_ptr_, entry);
        value_{{ value.index }} += coefficient * solution_.x[row] * solution_.x[column]
            * (row == column ? 0.5 : 1.0);
      }
    }
    objective_values_[{{ value.index }}] = value_{{ value.index }};
{% endfor %}
  }

  void evaluate_constraint_values() noexcept {
''',
    )
    replace_once(template, "      if (constraint_values_dirty_) evaluate_constraint_values();\n      d_dirty_ = false;\n      return;\n", "      if (constraint_values_dirty_) evaluate_constraint_values();\n      evaluate_objective_values();\n      d_dirty_ = false;\n      return;\n")
    replace_once(template, "    solution_ = clarabel_DefaultSolver_solution(solver_);\n    evaluate_constraint_values();\n    P_dirty_ = q_dirty_ = A_dirty_ = b_dirty_ = d_dirty_ = false;\n", "    solution_ = clarabel_DefaultSolver_solution(solver_);\n    evaluate_constraint_values();\n    evaluate_objective_values();\n    P_dirty_ = q_dirty_ = A_dirty_ = b_dirty_ = d_dirty_ = false;\n")
    replace_once(
        template,
        r'''{% endfor %}
  template <std::size_t Index>
  [[nodiscard]] std::span<const double> constraint_value() noexcept {
{% for value in constraint_values %}
    if constexpr (Index == {{ value.index }}) return constraint_value_{{ value.name }}();
    else
{% endfor %}
    {
      static_assert(Index < {{ constraint_values|length }}, "invalid constraint value index");
      return {};
    }
  }
''',
        r'''{% endfor %}
{% for value in objective_values %}
  [[nodiscard]] std::span<const double> objective_value_{{ value.name }}() noexcept {
    return std::span<const double>(objective_values_.data() + {{ value.index }}, 1);
  }

{% endfor %}
  template <std::size_t Index>
  [[nodiscard]] std::span<const double> constraint_value() noexcept {
{% for value in constraint_values %}
    if constexpr (Index == {{ value.index }}) return constraint_value_{{ value.name }}();
    else
{% endfor %}
{% for value in objective_values %}
    if constexpr (Index == {{ value.source_index }}) return objective_value_{{ value.name }}();
    else
{% endfor %}
    {
      static_assert(Index < {{ constraint_values|length + objective_values|length }}, "invalid constraint value index");
      return {};
    }
  }
''',
    )

    example = "examples/cpp_stream_mpo_one_pass.py"
    replace_once(example, "HORIZONS = (1, 2, 4, 8, 16, 32, 64, 128)\n", "HORIZONS = (2, 4, 8, 16, 32, 64, 128)\n")
    replace_once(example, "    risk_factor_7,\n", "")
    replace_once(example, "                risk_factor_7,\n", "")
    replace_once(example, "        risk_factor_7=factors[7],\n", "")
    replace_once(example, "    feature_list = tuple(\n        ts_zscore(\n", "    feature_list = tuple(\n        -ts_zscore(\n")
    replace_once(
        example,
        "    weights = cp.Variable((n_horizons, n_assets), name=\"weights\")\n    previous_weights = cp.Variable((n_assets,), name=\"previous_weights\")\n    spread_cost = cp.Variable(name=\"spread_cost\")\n    delta = weights - cp.vstack([previous_weights, weights[:-1]])\n",
        "    weights = cp.Variable((n_horizons, n_assets), name=\"weights\")\n    previous_weights = cp.Variable((n_assets,), name=\"previous_weights\")\n    delta = weights - cp.vstack([previous_weights, weights[:-1]])\n",
    )
    replace_once(example, "        abs_delta <= TRADE_BIG_M * trade_allowed,\n        cp.sum(cp.multiply(half_spread, abs_delta)) <= spread_cost,\n    ]\n", "        abs_delta <= TRADE_BIG_M * trade_allowed,\n    ]\n")
    replace_once(
        example,
        "    return cp.Problem(\n        cp.Minimize(\n            -cp.sum(cp.multiply(expected_returns, weights))\n            + spread_cost\n        ),\n        constraints,\n    )\n",
        "    spread_cost = cp.sum(cp.multiply(half_spread, abs_delta))\n    problem = cp.Problem(\n        cp.Minimize(\n            -cp.sum(cp.multiply(expected_returns, weights))\n            + spread_cost\n        ),\n        constraints,\n    )\n    return problem, {\"spread_cost\": spread_cost}\n",
    )
    replace_once(
        example,
        "    status = where(\n        session_open,\n        get_field(mpo, \"status\"),\n        float(\"nan\"),\n    )\n",
        "    status = where(\n        session_open,\n        get_field(mpo, \"status\"),\n        float(\"nan\"),\n    )\n    mpo_objective = where(\n        session_open,\n        get_field(mpo, \"objective\"),\n        float(\"nan\"),\n    )\n",
    )
    replace_once(example, '        "mpo_spread_cost": mpo_spread_cost,\n        "mpo_gross_pnl": mpo_gross_pnl,\n', '        "mpo_spread_cost": mpo_spread_cost,\n        "mpo_objective": mpo_objective,\n        "mpo_gross_pnl": mpo_gross_pnl,\n')

    p = ROOT / example
    text = p.read_text()
    lines = text.splitlines(keepends=True)
    shown = []
    for line in lines:
        shown.append(line)
        if line.strip() == "fig.tight_layout()":
            shown.append(line[: len(line) - len(line.lstrip())] + "plt.show()\n")
    text = "".join(shown)
    anchor = '''    path = plot_dir / "mpo_spread_cost.png"\n    fig.savefig(path, dpi=150)\n    plt.close(fig)\n    paths.append(path)\n\n    fig, ax = plt.subplots(figsize=(10, 5))\n    for start, end in zip(TRADE_STARTS, HORIZONS):\n'''
    replacement = '''    path = plot_dir / "mpo_spread_cost.png"\n    fig.savefig(path, dpi=150)\n    plt.close(fig)\n    paths.append(path)\n\n    fig, ax = plt.subplots(figsize=(10, 5))\n    ax.plot(index, _cum(values["mpo_objective"]), label="objective")\n    ax.set_title("MPO objective")\n    ax.set_ylabel("Cumulative objective")\n    ax.legend()\n    ax.grid(alpha=0.2)\n    fig.tight_layout()\n    plt.show()\n    path = plot_dir / "mpo_objective.png"\n    fig.savefig(path, dpi=150)\n    plt.close(fig)\n    paths.append(path)\n\n    fig, ax = plt.subplots(figsize=(10, 5))\n    for start, end in zip(TRADE_STARTS, HORIZONS):\n'''
    if text.count(anchor) != 1:
        raise RuntimeError(f"objective plot anchor count={text.count(anchor)}")
    text = text.replace(anchor, replacement, 1)
    text = text.replace("MPO objective spread cost", "MPO spread cost")
    p.write_text(text)


def verify() -> None:
    env = {"PYTHONPATH": "src:.", "MPLBACKEND": "Agg"}
    run("python", "-m", "compileall", "-q", "src", "examples", "tests")
    run("git", "diff", "--check")
    run(
        "python", "-m", "pytest", "-q", "-o", "addopts=",
        "tests/examples/test_cpp_stream_mpo_diagnostics.py",
        "tests/trading_dsl_engine/cpp_stream/test_cvxpy_constraint_values_and_guard.py",
        env=env,
    )
    smoke = r'''
from pathlib import Path
import tempfile
import numpy as np
import examples.cpp_stream_mpo_one_pass as example

assert example.HORIZONS == (2, 4, 8, 16, 32, 64, 128)
rows, assets = 320, 3
rng = np.random.default_rng(73)
returns = rng.normal(scale=2e-4, size=(rows, assets))
tradable = np.ones((rows, assets), dtype=float)
tradable[140:150] = 0.0
returns[140:150] = 0.0
hs = np.full((rows, assets), 8e-5)
ts = np.arange(rows, dtype=float) * example.MINUTE_US + 1_800_000_000_000_000.0
broad_start = np.full((rows, assets), ts[0] - 10 * example.MINUTE_US)
broad_end = np.full((rows, assets), ts[-1] + 500 * example.MINUTE_US)
data = {
    "returns": returns,
    "is_tradable_out0": tradable,
    "vw_halfspread_out0": hs,
    "_ev_ts": ts,
    "session_start0": broad_start,
    "session_end0": broad_end,
    "next_session_start0": broad_start,
    "next_session_end0": broad_end,
}
example.FEATURE_HLS = (2, 4, 8, 16)
example.IC_VOL_SPAN = 32
example.RIDGE_HL = 64
example.RISK_SPAN = 32
example.RISK_MIN_PERIODS = 8
with tempfile.TemporaryDirectory() as directory:
    result, paths = example._run(
        data, returns=example.var("returns"), output_dir=Path(directory)
    )
    values = result.load(mmap_mode=None)
    assert result.rows == rows
    assert len(paths) == 18, len(paths)
    assert (Path(directory) / "plots" / "mpo_objective.png").is_file()
    assert values["mpo_spread_cost"].shape == (rows,)
    assert values["mpo_objective"].shape == (rows,)
    assert np.isnan(values["mpo_spread_cost"][140:150]).all()
    assert np.isnan(values["mpo_objective"][140:150]).all()
    assert np.isnan(values["status"][140:150]).all()
    print("rows", result.rows, "seconds", result.seconds, "plots", len(paths))
'''
    run("python", "-c", textwrap.dedent(smoke), env=env)


def main() -> None:
    write_red_tests()
    run("python", "-m", "compileall", "-q", "tests")
    red = run(
        "python", "-m", "pytest", "-q", "-o", "addopts=",
        "tests/examples/test_cpp_stream_mpo_diagnostics.py",
        "tests/trading_dsl_engine/cpp_stream/test_cvxpy_constraint_values_and_guard.py::test_named_objective_term_projects_without_adding_a_solver_primal",
        check=False,
        env={"PYTHONPATH": "src:.", "MPLBACKEND": "Agg"},
    )
    if red.returncode == 0:
        raise RuntimeError("RED tests unexpectedly passed")
    print("RED confirmed", flush=True)
    implement()
    verify()
    run("git", "config", "user.name", "github-actions[bot]")
    run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    paths = [
        "examples/cpp_stream_mpo_one_pass.py",
        "tests/examples/test_cpp_stream_mpo_diagnostics.py",
        "tests/trading_dsl_engine/cpp_stream/test_cvxpy_constraint_values_and_guard.py",
        "src/trading_dsl_engine/cpp_stream/optimizer/factory.py",
        "src/trading_dsl_engine/cpp_stream/optimizer/direct_clarabel.py",
        "src/trading_dsl_engine/cpp_stream/optimizer/templates/direct_clarabel_instance.hpp.j2",
    ]
    run("git", "add", *paths)
    run("git", "commit", "-m", "Expose named CVXPY objective terms and update MPO diagnostics")
    run("git", "push", "origin", "HEAD:agent/mpo-postsolve-values-session-gate-clean")


if __name__ == "__main__":
    main()
