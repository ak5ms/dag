from pathlib import Path


def test_generated_runner_keeps_operator_merge_laws_out_of_jinja() -> None:
    template = (
        Path(__file__).resolve().parents[3]
        / "src/trading_dsl_engine/cpp_stream/python/templates/runner.cpp.j2"
    ).read_text()

    # The generated file owns topology and worker orchestration only. Concrete
    # state laws remain in ordinary C++ where they are type checked and inlined.
    for forbidden in (
        "merge_reduction_state_range",
        "ReductionNode<",
        "ReductionBundleNode<",
        "EmitLastNode<",
    ):
        assert forbidden not in template

    assert "stackdsl::merge_stage_states" in template
    assert "stackdsl::RowShardStateMerge" in template
    assert "stackdsl::LaneShardStateMerge" in template
