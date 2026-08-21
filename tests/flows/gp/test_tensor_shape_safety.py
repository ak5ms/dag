from __future__ import annotations

import pytest

from flows.gp import GPConfig, TensorFieldSpec, make_pset


def test_incompatible_shapes_reachable_at_same_reduced_rank_are_rejected():
    with pytest.raises(ValueError, match="incompatible shapes at logical rank 2"):
        GPConfig(
            tensor_fields=(
                TensorFieldSpec("rank3", "price", (5, 3)),
                TensorFieldSpec("rank2", "volume", (20,)),
            ),
            tensor_indices=(0, 1, 2),
        )


def test_compatible_prefix_shapes_can_share_intermediate_tensor_types():
    config = GPConfig(
        tensor_fields=(
            TensorFieldSpec("rank3", "price", (5, 3)),
            TensorFieldSpec("rank2", "volume", (5,)),
        ),
        tensor_indices=(0, 1, 2),
    )
    pset = make_pset(config)
    assert pset.gp_tensor_ranks == (2, 3)
    assert pset.gp_tensor_feature_shapes == {2: (5,), 3: (5, 3)}


def test_arbitrary_higher_rank_fields_register_every_reduction_prefix():
    config = GPConfig(
        tensor_fields=(TensorFieldSpec("rank4", "price", (4, 3, 2)),),
        tensor_indices=(0, 1),
    )
    pset = make_pset(config)
    assert pset.gp_tensor_ranks == (2, 3, 4)
    assert pset.gp_tensor_feature_shapes == {
        2: (4,),
        3: (4, 3),
        4: (4, 3, 2),
    }
