from __future__ import annotations

import flows.gp.pset as base_pset
from flows.gp.utils_primitives import (
    ALL_CPP_STREAM_UTIL_NAMES,
    NON_ROW_CPP_STREAM_UTIL_NAMES,
    ROW_SHAPED_CPP_STREAM_UTIL_NAMES,
    register_cpp_stream_utils,
)


BASE_EXPECTED_GP_OPERATOR_NAMES = base_pset.EXPECTED_GP_OPERATOR_NAMES
ALREADY_EXPOSED_CPP_STREAM_UTIL_NAMES = (
    ROW_SHAPED_CPP_STREAM_UTIL_NAMES & BASE_EXPECTED_GP_OPERATOR_NAMES
)
ADDED_CPP_STREAM_UTIL_OPERATOR_NAMES = (
    ROW_SHAPED_CPP_STREAM_UTIL_NAMES - BASE_EXPECTED_GP_OPERATOR_NAMES
)
EXPECTED_GP_OPERATOR_NAMES = (
    BASE_EXPECTED_GP_OPERATOR_NAMES | ROW_SHAPED_CPP_STREAM_UTIL_NAMES
)

ALL_DSL_OPERATOR_NAMES = base_pset.ALL_DSL_OPERATOR_NAMES
EXCLUDED_DSL_OPERATOR_NAMES = base_pset.EXCLUDED_DSL_OPERATOR_NAMES
EXPECTED_DSL_OPERATOR_NAMES = base_pset.EXPECTED_DSL_OPERATOR_NAMES
EXTRA_DSL_OPERATOR_NAMES = base_pset.EXTRA_DSL_OPERATOR_NAMES
GP_COMPOSITE_OPERATOR_NAMES = base_pset.GP_COMPOSITE_OPERATOR_NAMES
GPConfig = base_pset.GPConfig
ROWWISE_RIDGE_COMPOSITE_NAMES = base_pset.ROWWISE_RIDGE_COMPOSITE_NAMES
primitive_names_for_operator = base_pset.primitive_names_for_operator


def make_pset(config: GPConfig | None = None):
    pset = base_pset.make_pset(config)
    reg = base_pset._Registrar(pset)
    reg.families = set(pset.gp_operator_families)
    reg.primitive_family = dict(pset.gp_primitive_family)
    reg._names = set(pset.mapping)
    added = register_cpp_stream_utils(
        reg,
        pset.gp_config,
        skip_names=reg.families,
    )
    if added != ADDED_CPP_STREAM_UTIL_OPERATOR_NAMES:
        raise AssertionError(
            f"utility registration mismatch: expected={sorted(ADDED_CPP_STREAM_UTIL_OPERATOR_NAMES)}, actual={sorted(added)}"
        )
    missing = EXPECTED_GP_OPERATOR_NAMES - reg.families
    unexpected = reg.families - EXPECTED_GP_OPERATOR_NAMES
    if missing or unexpected:
        raise AssertionError(
            f"full GP coverage mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    pset.gp_operator_families = frozenset(reg.families)
    pset.gp_primitive_family = dict(reg.primitive_family)
    pset.gp_cpp_stream_utility_families = ROW_SHAPED_CPP_STREAM_UTIL_NAMES
    pset.gp_already_exposed_cpp_stream_utility_families = ALREADY_EXPOSED_CPP_STREAM_UTIL_NAMES
    pset.gp_added_cpp_stream_utility_families = added
    pset.gp_non_row_cpp_stream_utility_families = NON_ROW_CPP_STREAM_UTIL_NAMES
    return pset


__all__ = [
    "ADDED_CPP_STREAM_UTIL_OPERATOR_NAMES",
    "ALREADY_EXPOSED_CPP_STREAM_UTIL_NAMES",
    "ALL_CPP_STREAM_UTIL_NAMES",
    "ALL_DSL_OPERATOR_NAMES",
    "BASE_EXPECTED_GP_OPERATOR_NAMES",
    "EXCLUDED_DSL_OPERATOR_NAMES",
    "EXPECTED_DSL_OPERATOR_NAMES",
    "EXPECTED_GP_OPERATOR_NAMES",
    "EXTRA_DSL_OPERATOR_NAMES",
    "GP_COMPOSITE_OPERATOR_NAMES",
    "GPConfig",
    "NON_ROW_CPP_STREAM_UTIL_NAMES",
    "ROWWISE_RIDGE_COMPOSITE_NAMES",
    "ROW_SHAPED_CPP_STREAM_UTIL_NAMES",
    "make_pset",
    "primitive_names_for_operator",
]
