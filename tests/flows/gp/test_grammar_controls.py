from __future__ import annotations

import inspect

from flows.gp import (
    GPConfig,
    GrammarPolicy,
    GroupVectorInput,
    format_grammar_table,
    grammar_families,
    grammar_rows,
    individual_to_expr,
    make_pset,
    primitive_names_for_operator,
    random_tree,
)
from trading_dsl_engine.cpp_stream.python.frontend import compile_ir


def _primitives(pset, family):
    return [pset.mapping[name] for name in primitive_names_for_operator(pset, family)]


def test_top_level_section_exclusion_is_single_source_of_truth():
    config = GPConfig(grammar=GrammarPolicy(exclude_sections=("utils.group",)))
    pset = make_pset(config)
    assert "utils.group" not in pset.gp_sections
    assert not [family for family in pset.gp_operator_families if family.startswith("group_")]
    assert "xs_group_neutralize" not in pset.gp_operator_families
    assert "xs_market_neutralize" not in pset.gp_operator_families
    assert "group_mean" in pset.gp_policy_excluded_families


def test_family_patterns_apply_across_row_and_tensor_registrars():
    config = GPConfig(grammar=GrammarPolicy(exclude_families=("ewm*",)))
    pset = make_pset(config)
    assert not [family for family in pset.gp_operator_families if family.startswith("ewm")]
    assert not primitive_names_for_operator(pset, "ewm")
    assert "add" in pset.gp_operator_families


def test_only_sections_builds_small_valid_grammar():
    config = GPConfig(
        grammar=GrammarPolicy(
            include_sections=("row.elementwise", "row.ewm", "tensor.terminals")
        )
    )
    pset = make_pset(config)
    assert pset.gp_sections <= {"row.elementwise", "row.ewm"}
    assert "ewm" in pset.gp_operator_families
    assert "rolling_mean" not in pset.gp_operator_families
    assert "group_mean" not in pset.gp_operator_families


def test_group_vector_rhs_is_structurally_terminal_only():
    pset = make_pset()
    for family in ("group_vector_proj", "group_vector_neut"):
        primitives = _primitives(pset, family)
        assert primitives
        assert all(primitive.args[1] is GroupVectorInput for primitive in primitives)
    assert pset.terminals[GroupVectorInput]
    assert not pset.primitives[GroupVectorInput]


def test_generation_has_no_retry_or_rejection_path():
    assert "max_attempts" not in inspect.signature(random_tree).parameters
    pset = make_pset()
    for seed in range(100):
        tree = random_tree(pset, min_depth=1, max_depth=4, seed=seed)
        compile_ir(individual_to_expr(tree, pset))


def test_grammar_introspection_reports_effective_configuration():
    pset = make_pset(
        GPConfig(grammar=GrammarPolicy(exclude_sections=("utils.group",)))
    )
    section_rows = grammar_rows(pset, level="section")
    family_rows = grammar_rows(pset, level="family")
    assert section_rows
    assert family_rows
    assert "utils.group" not in {row["section"] for row in section_rows}
    assert "group_mean" not in grammar_families(pset)
    text = format_grammar_table(pset, level="section")
    assert "| Section | Families | Overloads | Outputs |" in text
    assert "row.elementwise" in text


def test_group_utilities_use_bounded_key_terminals():
    from flows.gp import GroupKeyInput, make_pset, primitive_names_for_operator

    pset = make_pset()
    keys = pset.terminals.get(GroupKeyInput, ())
    assert keys, "at least one bounded semantic field should be a group key"
    for family in ("group_mean", "group_sum", "group_rank", "group_vector_proj"):
        for name in primitive_names_for_operator(pset, family):
            primitive = pset.mapping[name]
            assert GroupKeyInput in primitive.args

