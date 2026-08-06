from __future__ import annotations

from pathlib import Path

import numpy as np

from flows.riskminer import (
    CppStreamCandidateEvaluator,
    CppStreamRidgePoolEvaluator,
    GRURiskSeekingTokenPolicy,
    MCTSConfig,
    PolicyEpisode,
    RPNEnvironment,
    RiskMinerMCTS,
    default_market_semantics,
    operator_inventory,
)
from flows.riskminer.semantics import (
    DEFAULT_TYPE_RELATIONS,
    compatible_additive,
    numeric_ratio,
)
from trading_dsl_engine.base.dsl import var


def _score_reference(alpha: np.ndarray, returns: np.ndarray) -> float:
    shifted = np.full_like(alpha, np.nan)
    shifted[1:] = alpha[:-1]
    pnl = np.nansum(shifted * returns, axis=1)
    return float(np.mean(pnl) / np.std(pnl))


def test_market_semantics_and_generic_type_relations() -> None:
    fields = default_market_semantics()
    price_sum = compatible_additive(
        fields["ap0"], fields["bp0"], DEFAULT_TYPE_RELATIONS
    )
    quantity_sum = compatible_additive(
        fields["av0"], fields["volume"], DEFAULT_TYPE_RELATIONS
    )
    invalid = compatible_additive(
        fields["close"], fields["volume"], DEFAULT_TYPE_RELATIONS
    )
    price_ratio = numeric_ratio(
        fields["ap0"], fields["bp0"], DEFAULT_TYPE_RELATIONS
    )
    assert price_sum is not None and "price" in price_sum.types
    assert quantity_sum is not None and "quantity" in quantity_sum.types
    assert invalid is None
    assert price_ratio is not None and "dimensionless" in price_ratio.types


def test_typed_rpn_builds_valid_formula_and_rejects_bad_addition() -> None:
    environment = RPNEnvironment(max_depth=8, max_tokens=16)
    assert "roll_rets" not in environment.terminals
    assert "hs" not in environment.terminals
    assert "vol" not in environment.terminals
    assert "is_tradable" not in environment.terminals

    state = environment.parse(["ap0", "bp0", "div", "xs_rank", "END"])
    assert state.complete
    assert environment.candidate(state) is not None
    assert "xs_rank" in repr(state.stack[0].expr)

    bad = environment.initial_state()
    for token in ("close", "volume", "add"):
        bad = environment.step(
            bad,
            environment.token_by_name[token].token_id,
        )
    assert bad.invalid_reason is not None
    assert "illegal token" in bad.invalid_reason


def test_rpn_static_temporal_parameter_is_cpp_stream_compatible() -> None:
    environment = RPNEnvironment(max_depth=8, max_tokens=16)
    state = environment.parse(
        ["close", "literal:20", "ewm", "xs_rank", "END"]
    )
    assert state.complete
    assert "ewm" in repr(state.stack[0].expr)


def test_operator_inventory_exposes_direct_and_structured_groups() -> None:
    inventory = operator_inventory()
    assert len(inventory["searchable"]) >= 25
    assert {
        "add", "div", "ewm", "rolling_mean", "xs_rank"
    }.issubset(inventory["searchable"])
    assert {"cat", "einsum", "Ridge"}.issubset(
        inventory["structured"]
    )


def test_gru_policy_masks_actions_and_performs_risk_update() -> None:
    environment = RPNEnvironment(
        literals=(-1.0, 0.0, 1.0, 2.0, 5.0, 20.0),
        max_depth=4,
        max_tokens=12,
    )
    legal = environment.legal_tokens(environment.initial_state())
    first = GRURiskSeekingTokenPolicy(
        len(environment.tokens),
        seed=13,
    )
    second = GRURiskSeekingTokenPolicy(
        len(environment.tokens),
        seed=13,
    )
    first_priors = first.priors(environment.initial_state(), legal)
    second_priors = second.priors(environment.initial_state(), legal)
    assert set(first_priors) == {token.token_id for token in legal}
    assert np.isclose(sum(first_priors.values()), 1.0)
    np.testing.assert_allclose(
        [first_priors[token.token_id] for token in legal],
        [second_priors[token.token_id] for token in legal],
        rtol=0.0,
        atol=0.0,
    )

    soft_side = environment.token_by_name["soft_side_wavg"].token_id
    rank = environment.token_by_name["xs_rank"].token_id
    end = environment.token_by_name["END"].token_id
    first.update(
        (
            PolicyEpisode((soft_side, end), 2.0),
            PolicyEpisode((soft_side, rank, end), 0.25),
            PolicyEpisode((soft_side, rank, end), -0.5),
        )
    )
    assert first.training_steps == 1
    assert np.isfinite(first.quantile_value)


def test_cpp_stream_candidate_batch_matches_numpy_reference(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    rows, instruments = 128, 3
    signal = np.tanh(rng.normal(size=(rows, instruments)))
    returns = rng.normal(scale=1e-3, size=(rows, instruments))
    returns[1:] += 5e-4 * signal[:-1]
    sources = {
        "soft_side_wavg": signal,
        "roll_rets": returns,
    }
    candidates = [
        var("soft_side_wavg"),
        var("soft_side_wavg") * 2.0,
    ]
    evaluator = CppStreamCandidateEvaluator(
        sources,
        n_instruments=instruments,
        work_dir=tmp_path,
    )
    actual = evaluator.score_batch(candidates)
    expected = [
        _score_reference(signal, returns),
        _score_reference(signal * 2.0, returns),
    ]
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    assert evaluator.stats.last_output_mode == "final"
    assert evaluator.stats.last_output_shape == (2,)
    assert "cpp_stream" in (evaluator.stats.last_runtime_type or "")


class _FakeEvaluator:
    def score_batch(self, candidates):
        return [
            2.0 if "soft_side_wavg" in repr(candidate) else 0.1
            for candidate in candidates
        ]


def test_mcts_is_deterministic_and_discovers_finite_family() -> None:
    config = MCTSConfig(
        simulations=16,
        rollouts_per_expansion=3,
        selection_batch_size=4,
        archive_size=20,
        seed=11,
    )
    first = RiskMinerMCTS(
        RPNEnvironment(max_depth=4, max_tokens=12),
        _FakeEvaluator(),
        config=config,
    ).search()
    second = RiskMinerMCTS(
        RPNEnvironment(max_depth=4, max_tokens=12),
        _FakeEvaluator(),
        config=config,
    ).search()
    assert first.candidates
    assert first.simulations == 16
    assert first.rollout_proposals >= 16
    assert [
        item.canonical_key for item in first.candidates
    ] == [
        item.canonical_key for item in second.candidates
    ]


def test_cpp_stream_ridge_pool_runs_final_only(tmp_path: Path) -> None:
    rng = np.random.default_rng(19)
    rows, instruments = 256, 3
    signal = np.tanh(rng.normal(size=(rows, instruments)))
    returns = rng.normal(scale=2e-4, size=(rows, instruments))
    returns[1:] += 4e-4 * signal[:-1]
    mid = 100.0 * np.exp(np.cumsum(returns, axis=0))
    spread = np.full_like(mid, 5e-5)
    sources = {
        "soft_side_wavg": signal,
        "ap0": mid * (1.0 + spread),
        "bp0": mid * (1.0 - spread),
        "close": mid,
        "roll_rets": returns,
        "hs": spread,
        "vol": np.full_like(mid, 0.01),
        "is_tradable": np.ones_like(mid),
    }
    alphas = (
        var("soft_side_wavg"),
        (var("ap0") - var("bp0")) / var("close"),
    )
    evaluator = CppStreamRidgePoolEvaluator(
        sources,
        n_instruments=instruments,
        work_dir=tmp_path,
        ridge_halflife=16.0,
    )
    result = evaluator.evaluate(alphas)
    assert result.output_mode == "final"
    assert result.output_shape == ()
    assert np.isfinite(result.sharpe)
    assert "cpp_stream" in result.runtime_type
