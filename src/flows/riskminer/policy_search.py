from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import RiskMinerConfig
from .cpp_stream_eval import CppStreamCandidateEvaluator
from .learned_policy import GRUPolicyConfig, JaxGRUPolicy
from .mcts import ActionPolicy, RiskMCTS
from .rpn import TypedRPNEnvironment, build_vocabulary
from .search import CppStreamRiskMinerResult
from .semantics import SemanticInfo, alpha_terminal_metadata


def search_cpp_stream_alphas_with_policy(
    sources: Mapping[str, Any],
    *,
    n_instruments: int,
    work_dir: str | Path,
    config: RiskMinerConfig = RiskMinerConfig(),
    terminals: Mapping[str, SemanticInfo] | None = None,
    policy: ActionPolicy | None = None,
    compile_kwargs: Mapping[str, Any] | None = None,
) -> tuple[CppStreamRiskMinerResult, ActionPolicy]:
    """Run native RiskMiner search with a supplied or initialized token policy."""

    terminal_values = dict(
        alpha_terminal_metadata() if terminals is None else terminals
    )
    missing = sorted(set(terminal_values) - set(sources))
    if missing:
        raise KeyError(f"missing alpha terminal sources: {missing}")
    vocabulary = build_vocabulary(terminals=terminal_values)
    environment = TypedRPNEnvironment(
        config=config,
        vocabulary=vocabulary,
        target_types=("dimensionless",),
    )
    resolved_policy: ActionPolicy = policy or JaxGRUPolicy.initialize(
        GRUPolicyConfig(
            vocabulary_size=len(vocabulary),
            learning_rate=1.0e-3,
            seed=config.seed,
        )
    )
    evaluator = CppStreamCandidateEvaluator(
        sources,
        n_instruments=n_instruments,
        work_dir=work_dir,
        batch_size=config.evaluation_batch_size,
        compile_kwargs=compile_kwargs,
    )
    search = RiskMCTS(
        environment,
        evaluator,
        config=config,
        policy=resolved_policy,
    ).search()
    return CppStreamRiskMinerResult(search, evaluator.summary), resolved_policy


__all__ = ["search_cpp_stream_alphas_with_policy"]