from __future__ import annotations

import pytest

from trading_dsl_engine.ir.einsum import (
    EinsumParseError,
    build_contraction_plan,
    parse_einsum,
)


def test_explicit_implicit_ellipsis_and_diagonal_shapes() -> None:
    explicit = parse_einsum("ij,jk->ik", ((None, 3), (3, 4)))
    assert explicit.input_labels == (("i", "j"), ("j", "k"))
    assert explicit.output_labels == ("i", "k")
    assert explicit.output_shape == (None, 4)

    implicit = parse_einsum("ji,jk", ((3, 5), (3, 7)))
    assert implicit.output_labels == ("i", "k")
    assert implicit.output_shape == (5, 7)

    ellipsis = parse_einsum("...j,...j->...", ((2, None, 3), (1, None, 3)))
    assert ellipsis.output_shape == (2, None)

    diagonal = parse_einsum("ii->i", ((None, None),))
    assert diagonal.output_shape == (None,)


def test_empty_scalar_terms_and_arbitrary_ascii_symbols() -> None:
    scalar = parse_einsum(",Q->Q", ((), (None,)))
    assert scalar.input_labels == ((), ("Q",))
    assert scalar.output_shape == (None,)

    reduction = parse_einsum("Aq,Aq->", ((5, 3), (5, 3)))
    assert reduction.output_shape == ()


def test_invalid_subscripts_match_numpy_style_constraints() -> None:
    with pytest.raises(EinsumParseError, match="subscript terms"):
        parse_einsum("i,j->ij", ((3,),))
    with pytest.raises(EinsumParseError, match="output labels may not repeat"):
        parse_einsum("ij->ii", ((3, 4),))
    with pytest.raises(EinsumParseError, match="does not appear"):
        parse_einsum("ij->k", ((3, 4),))
    with pytest.raises(EinsumParseError, match="not broadcastable"):
        parse_einsum("ij,ij->ij", ((3, 4), (3, 5)))
    with pytest.raises(EinsumParseError, match="repeated einsum label"):
        parse_einsum("ii->i", ((3, 4),))


def test_greedy_and_optimal_paths_reduce_nary_work() -> None:
    shapes = ((40, 2), (30, 2), (30, 50))
    greedy_spec = parse_einsum("ij,kj,kl->il", shapes, optimize="greedy")
    optimal_spec = parse_einsum("ij,kj,kl->il", shapes, optimize="optimal")
    none_spec = parse_einsum("ij,kj,kl->il", shapes, optimize=False)

    greedy = build_contraction_plan(greedy_spec, shapes)
    optimal = build_contraction_plan(optimal_spec, shapes)
    left_to_right = build_contraction_plan(none_spec, shapes)

    assert len(greedy.steps) == 2
    assert len(optimal.steps) == 2
    assert optimal.estimated_flops <= greedy.estimated_flops
    assert greedy.estimated_flops <= left_to_right.estimated_flops
    assert optimal.output_shape == (40, 50)


def test_batched_matmul_ellipsis_plan() -> None:
    spec = parse_einsum("...ij,...jk->...ik", ((2, 3, 4), (1, 4, 5)))
    plan = build_contraction_plan(spec, ((2, 3, 4), (1, 4, 5)))
    assert plan.output_shape == (2, 3, 5)
    assert plan.steps[-1].loop_extents == (2, 3, 5, 4)
