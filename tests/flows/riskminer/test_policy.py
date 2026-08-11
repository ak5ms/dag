import math

import jax
import numpy as np

from flows.riskminer.config import RiskMinerConfig
from flows.riskminer.learned_policy import (
    GRUPolicyConfig,
    JaxGRUPolicy,
    PolicyTrajectory,
    RiskQuantileTracker,
    TrajectoryBatch,
)
from flows.riskminer.mcts import RiskMCTS
from flows.riskminer.rpn import TypedRPNEnvironment, canonical_expr_key


jax.config.update("jax_enable_x64", True)


def _environment():
    config = RiskMinerConfig(
        max_depth=4,
        max_tokens=12,
        max_stack=5,
        simulations=4,
        rollouts_per_expansion=2,
        evaluation_batch_size=2,
        archive_size=6,
        seed=11,
    )
    return TypedRPNEnvironment(config=config), config


def test_gru_policy_masks_illegal_actions_and_normalizes_root_priors():
    environment, _ = _environment()
    policy = JaxGRUPolicy.initialize(
        GRUPolicyConfig(
            vocabulary_size=len(environment.vocabulary),
            hidden_size=16,
            layers=2,
            mlp_hidden_1=8,
            mlp_hidden_2=8,
            seed=3,
        )
    )
    state = environment.initial_state()
    legal = environment.legal_actions(state)
    priors = policy.priors(environment, state, legal)
    assert set(priors) == set(legal)
    assert all(value > 0.0 and math.isfinite(value) for value in priors.values())
    assert math.isclose(sum(priors.values()), 1.0, rel_tol=1e-7, abs_tol=1e-7)
    assert environment.vocabulary.end.token_id not in legal


def test_risk_quantile_tracker_moves_in_the_correct_direction():
    tracker = RiskQuantileTracker(
        cdf_quantile=0.8,
        learning_rate=0.1,
        value=0.0,
    )
    higher = tracker.update(1.0)
    assert higher.value > tracker.value
    lower = higher.update(-1.0)
    assert lower.value < higher.value
    sample = np.linspace(-1.0, 1.0, 101)
    rng = np.random.default_rng(17)
    fitted = RiskQuantileTracker(
        cdf_quantile=0.8,
        learning_rate=0.01,
        value=0.0,
    )
    for _ in range(50):
        fitted = fitted.update_many(rng.permutation(sample))
    assert abs(fitted.value - np.quantile(sample, 0.8)) < 0.08


def test_below_quantile_training_reduces_selected_action_probability():
    environment, _ = _environment()
    root = environment.initial_state()
    legal = environment.legal_actions(root)
    chosen = environment.vocabulary.by_name["soft_side_wavg"].token_id
    assert chosen in legal
    policy = JaxGRUPolicy.initialize(
        GRUPolicyConfig(
            vocabulary_size=len(environment.vocabulary),
            hidden_size=16,
            layers=2,
            mlp_hidden_1=8,
            mlp_hidden_2=8,
            learning_rate=0.1,
            seed=5,
        )
    )
    before = policy.priors(environment, root, legal)[chosen]
    batch = TrajectoryBatch(
        (
            PolicyTrajectory(
                states=((),),
                actions=(chosen,),
                legal_actions=(legal,),
                reward=-1.0,
            ),
        )
    )
    updated, loss = policy.train_step(batch, reward_quantile=0.0)
    after = updated.priors(environment, root, legal)[chosen]
    assert loss < 0.0
    assert after < before


class StubEvaluator:
    def evaluate(self, candidates):
        return {
            canonical_expr_key(expr): (
                1.0 if "soft_side_wavg" in repr(expr) else 0.1
            )
            for expr in candidates
        }


def test_learned_policy_implements_the_mcts_action_prior_interface():
    environment, config = _environment()
    policy = JaxGRUPolicy.initialize(
        GRUPolicyConfig(
            vocabulary_size=len(environment.vocabulary),
            hidden_size=16,
            layers=2,
            mlp_hidden_1=8,
            mlp_hidden_2=8,
            seed=config.seed,
        )
    )
    # The interface test should not depend on a random initialized policy
    # ranking a valid market terminal ahead of the enlarged literal inventory.
    chosen = environment.vocabulary.by_name["soft_side_wavg"].token_id
    policy.params["out"]["b"] = (
        policy.params["out"]["b"].at[chosen].set(20.0)
    )
    result = RiskMCTS(
        environment,
        StubEvaluator(),
        config=config,
        policy=policy,
    ).search()
    assert result.metrics.simulations == config.simulations
    assert result.metrics.rollouts == (
        config.simulations * config.rollouts_per_expansion
    )
    assert result.archive


def test_per_trajectory_quantile_thresholds_are_used_in_batched_loss():
    environment, _ = _environment()
    root = environment.initial_state()
    legal = environment.legal_actions(root)
    chosen = environment.vocabulary.by_name["soft_side_wavg"].token_id
    policy = JaxGRUPolicy.initialize(
        GRUPolicyConfig(
            vocabulary_size=len(environment.vocabulary),
            hidden_size=16,
            layers=2,
            mlp_hidden_1=8,
            mlp_hidden_2=8,
            seed=13,
        )
    )
    excluded = PolicyTrajectory(
        states=((),), actions=(chosen,), legal_actions=(legal,), reward=-1.0
    )
    selected = PolicyTrajectory(
        states=((),), actions=(chosen,), legal_actions=(legal,), reward=1.0
    )
    batched = TrajectoryBatch(
        (excluded, selected),
        reward_quantiles=(-2.0, 2.0),
    )
    selected_only = TrajectoryBatch((selected,), reward_quantiles=(2.0,))
    assert math.isclose(
        policy.loss(batched, reward_quantile=999.0),
        policy.loss(selected_only, reward_quantile=-999.0),
        rel_tol=1e-7,
        abs_tol=1e-7,
    )
