from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math

import flows.gp.pset as row_pset
from flows.gp.regression import REGRESSION_PROJECTIONS
from flows.gp.tensor_elementwise import register_tensor_elementwise
from flows.gp.tensor_temporal import TENSOR_VEC_FAMILIES, register_tensor_temporal
from flows.gp.tensor_types import DEFAULT_TENSOR_FIELDS, TensorFieldSpec, TensorIndex, tensor_input_types, tensor_type
from flows.gp.tensor_utils_elementwise import TENSOR_ELEMENTWISE_UTILS, register_tensor_elementwise_utils
from flows.gp.tensor_utils_temporal import TENSOR_TEMPORAL_UTILS, register_tensor_temporal_utils
from flows.gp.types import NumericRow, PositiveInt
from flows.gp.utils_primitives import ALL_CPP_STREAM_UTIL_NAMES, NON_ROW_CPP_STREAM_UTIL_NAMES, ROW_SHAPED_CPP_STREAM_UTIL_NAMES
from trading_dsl_engine.base import dsl


ALL_DSL_OPERATOR_NAMES = row_pset.ALL_DSL_OPERATOR_NAMES
EXCLUDED_DSL_OPERATOR_NAMES = row_pset.EXCLUDED_DSL_OPERATOR_NAMES
EXPECTED_DSL_OPERATOR_NAMES = row_pset.EXPECTED_DSL_OPERATOR_NAMES
EXTRA_DSL_OPERATOR_NAMES = row_pset.EXTRA_DSL_OPERATOR_NAMES
GP_COMPOSITE_OPERATOR_NAMES = row_pset.GP_COMPOSITE_OPERATOR_NAMES
ROWWISE_RIDGE_COMPOSITE_NAMES = row_pset.ROWWISE_RIDGE_COMPOSITE_NAMES
BASE_EXPECTED_GP_OPERATOR_NAMES = row_pset.EXPECTED_GP_OPERATOR_NAMES
EXPECTED_GP_OPERATOR_NAMES = BASE_EXPECTED_GP_OPERATOR_NAMES | TENSOR_VEC_FAMILIES
primitive_names_for_operator = row_pset.primitive_names_for_operator

TENSOR_EWM_UTILS = frozenset({
    "ewm_co_kurtosis", "ewm_co_skewness", "ewm_corr", "ewm_cov",
    "ewm_kurtosis", "ewm_moment", "ewm_partial_corr", "ewm_skewness",
    "ewm_std", "ewm_triple_corr", "ewm_var",
})
TENSOR_CPP_STREAM_UTIL_NAMES = (
    TENSOR_ELEMENTWISE_UTILS | TENSOR_TEMPORAL_UTILS | TENSOR_VEC_FAMILIES | TENSOR_EWM_UTILS
)


@dataclass(frozen=True)
class GPConfig(row_pset.GPConfig):
    # A denser default terminal grid makes static-parameter search and local
    # phenotypic shocks meaningfully local instead of jumping 2x-10x at a time.
    positive_ints: tuple[int, ...] = (
        1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 32, 48, 60, 64,
        90, 120, 128, 180, 240, 256, 360, 480, 720, 960, 1440, 2880,
    )
    positive_floats: tuple[float, ...] = (
        0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.25,
        0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0,
    )
    negative_floats: tuple[float, ...] = (
        -20.0, -10.0, -5.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5,
        -0.25, -0.2, -0.1, -0.05, -0.025, -0.01, -0.005, -0.001,
        -0.0001,
    )
    quantiles: tuple[float, ...] = (
        0.01, 0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.8,
        0.9, 0.95, 0.99,
    )
    # Exact ScalarNumber terminals supplement the typed positive/negative grids;
    # zero is especially useful for offsets, clipping and robustness shocks.
    scalar_numbers: tuple[float, ...] = (0.0,)
    tensor_fields: tuple[TensorFieldSpec, ...] = DEFAULT_TENSOR_FIELDS
    tensor_indices: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        super().__post_init__()
        scalars = tuple(float(value) for value in self.scalar_numbers)
        if any(not math.isfinite(value) for value in scalars):
            raise ValueError("scalar_numbers must contain only finite values")
        if len(scalars) != len(set(scalars)):
            raise ValueError("scalar_numbers must be unique")
        object.__setattr__(self, "scalar_numbers", scalars)

        fields = tuple(self.tensor_fields)
        names = [field.name for field in fields]
        if len(names) != len(set(names)):
            raise ValueError("tensor field names must be unique")
        shapes: dict[int, tuple[int, ...]] = {}
        for field in fields:
            for rank in range(2, field.logical_rank + 1):
                feature_shape = field.feature_shape[: rank - 1]
                previous = shapes.setdefault(rank, feature_shape)
                if previous != feature_shape:
                    raise ValueError(
                        f"incompatible shapes at logical rank {rank}: "
                        f"{previous} versus {feature_shape}"
                    )
        indices = tuple(int(value) for value in self.tensor_indices)
        if any(value < 0 for value in indices) or len(indices) != len(set(indices)):
            raise ValueError("tensor_indices must be unique nonnegative integers")
        extents = [extent for field in fields for extent in field.feature_shape]
        if extents and any(value >= min(extents) for value in indices):
            raise ValueError("tensor index exceeds at least one configured feature axis")
        object.__setattr__(self, "tensor_fields", fields)
        object.__setattr__(self, "tensor_indices", indices)


def _tensor_expr(field: TensorFieldSpec):
    if field.columns:
        return dsl.cat(*(dsl.var(name) for name in field.columns))
    return dsl.var(field.name)


def _register_tensor_terminals(pset, config: GPConfig) -> dict[str, str]:
    names: dict[str, str] = {}
    if not config.grammar.allows_section("tensor.terminals"):
        return names
    for field in config.tensor_fields:
        type_ = field.gp_type()
        terminal_name = row_pset._safe_name(f"tensor_field_{field.name}")
        pset.addTerminal(type_(_tensor_expr(field)), type_, name=terminal_name)
        names[field.name] = terminal_name
    for value in config.tensor_indices:
        pset.addTerminal(TensorIndex(value), TensorIndex, name=f"tensor_index_{value}")
    return names


def _register_scalar_terminals(pset, config: GPConfig) -> dict[float, str]:
    names: dict[float, str] = {}
    for value in config.scalar_numbers:
        literal = f"{value:g}"
        name = f"scalar_number_{row_pset._safe_name(literal)}"
        if name not in pset.mapping:
            row_pset._add_terminal(
                pset,
                row_pset.ScalarNumber(value),
                row_pset.ScalarNumber,
                name,
            )
        names[value] = name
    return names


def _active_ranks(config: GPConfig) -> tuple[int, ...]:
    if not config.grammar.allows_section("tensor.terminals"):
        return ()
    return tuple(
        sorted(
            {
                rank
                for field in config.tensor_fields
                for rank in range(2, field.logical_rank + 1)
            }
        )
    )


def _register_matrix_regression(reg) -> frozenset[str]:
    matrix = tensor_type(2, "numeric")
    families: set[str] = set()
    for projection in REGRESSION_PROJECTIONS:
        ret = row_pset._regression_ret_type(projection)
        reg.add(
            "ts_regression",
            partial(row_pset._temporal_regression_call, projection),
            (NumericRow, matrix, PositiveInt),
            ret,
            variant=f"{projection}_matrix",
        )
        family = f"ridge_{projection}"
        reg.add(
            family,
            partial(row_pset._rowwise_regression_call, projection),
            (NumericRow, matrix),
            ret,
            variant="matrix",
        )
        families.update(("ts_regression", family))
    for degree in (1, 2, 3):
        reg.add(
            "ts_poly_regression",
            partial(row_pset._poly_regression_call, degree),
            (NumericRow, matrix, PositiveInt),
            row_pset.DerivedNumericRow,
            variant=f"matrix_degree_{degree}",
        )
    families.add("ts_poly_regression")
    return frozenset(families)


def make_pset(config: GPConfig | None = None):
    config = config or GPConfig()
    pset = row_pset.make_pset(config)
    scalar_terminals = _register_scalar_terminals(pset, config)
    reg = row_pset._Registrar(pset, policy=config.grammar)
    reg.families = set(pset.gp_operator_families)
    reg.primitive_family = dict(pset.gp_primitive_family)
    reg.primitive_section = dict(pset.gp_primitive_section)
    reg._names = set(pset.mapping)
    terminals = _register_tensor_terminals(pset, config)
    ranks = _active_ranks(config)
    tensor_families: set[str] = set()
    if ranks:
        reg.set_section("tensor.elementwise")
        tensor_families.update(register_tensor_elementwise(reg, ranks))
        reg.set_section("tensor.temporal")
        tensor_families.update(register_tensor_temporal(reg, ranks))
        reg.set_section("tensor.utils.elementwise")
        tensor_families.update(register_tensor_elementwise_utils(reg, ranks, config))
        reg.set_section("tensor.utils.temporal")
        tensor_families.update(register_tensor_temporal_utils(reg, ranks))
        if 2 in ranks:
            reg.set_section("tensor.regression")
            tensor_families.update(_register_matrix_regression(reg))

    if config.grammar.is_default:
        expected = BASE_EXPECTED_GP_OPERATOR_NAMES | TENSOR_VEC_FAMILIES
        missing = expected - reg.families
        unexpected = reg.families - expected
        if missing or unexpected:
            raise AssertionError(
                f"tensor GP coverage mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

    pset.gp_operator_families = frozenset(reg.families)
    pset.gp_full_operator_families = BASE_EXPECTED_GP_OPERATOR_NAMES | (TENSOR_VEC_FAMILIES if ranks else frozenset())
    pset.gp_primitive_family = dict(reg.primitive_family)
    pset.gp_primitive_section = dict(reg.primitive_section)
    pset.gp_sections = frozenset(reg.primitive_section.values())
    pset.gp_tensor_operator_families = frozenset(tensor_families & reg.families)
    pset.gp_tensor_field_terminals = terminals
    pset.gp_scalar_terminals = scalar_terminals
    pset.gp_tensor_ranks = ranks
    pset.gp_tensor_feature_shapes = {
        rank: next(
            field.feature_shape[: rank - 1]
            for field in config.tensor_fields
            if field.logical_rank >= rank
        )
        for rank in ranks
    }
    pset.gp_cpp_stream_utility_families = (
        (ALL_CPP_STREAM_UTIL_NAMES if ranks else ROW_SHAPED_CPP_STREAM_UTIL_NAMES)
        & reg.families
    )
    pset.gp_non_row_cpp_stream_utility_families = NON_ROW_CPP_STREAM_UTIL_NAMES - (
        TENSOR_VEC_FAMILIES if ranks else frozenset()
    )
    pset.gp_tensor_cpp_stream_utility_families = (
        TENSOR_CPP_STREAM_UTIL_NAMES & reg.families if ranks else frozenset()
    )
    pset.gp_policy_excluded_families = pset.gp_full_operator_families - reg.families
    return pset


def gp_input_types(config: GPConfig | None = None, n_instruments: int = 9):
    config = config or GPConfig()
    return tensor_input_types(config.tensor_fields, n_instruments)


__all__ = [
    "ALL_CPP_STREAM_UTIL_NAMES", "ALL_DSL_OPERATOR_NAMES", "BASE_EXPECTED_GP_OPERATOR_NAMES",
    "EXCLUDED_DSL_OPERATOR_NAMES", "EXPECTED_DSL_OPERATOR_NAMES", "EXPECTED_GP_OPERATOR_NAMES",
    "EXTRA_DSL_OPERATOR_NAMES", "GP_COMPOSITE_OPERATOR_NAMES", "GPConfig",
    "ROWWISE_RIDGE_COMPOSITE_NAMES", "TENSOR_CPP_STREAM_UTIL_NAMES",
    "gp_input_types", "make_pset", "primitive_names_for_operator",
]
