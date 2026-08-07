from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RiskMinerConfig
from .cpp_stream_eval import CppStreamCandidateEvaluator, EvaluationSummary
from .mcts import RiskMCTS, RiskMinerSearchResult, SchemaPriorPolicy
from .rpn import TypedRPNEnvironment, build_vocabulary
from .semantics import SemanticInfo, alpha_terminal_metadata


@dataclass(frozen=True)
class CppStreamRiskMinerResult:
    search: RiskMinerSearchResult
    evaluation: EvaluationSummary


def search_cpp_stream_alphas(
    sources: Mapping[str, Any],
    *,
    n_instruments: int,
    work_dir: str | Path,
    config: RiskMinerConfig = RiskMinerConfig(),
    terminals: Mapping[str, SemanticInfo] | None = None,
    compile_kwargs: Mapping[str, Any] | None = None,
) -> CppStreamRiskMinerResult:
    """Run typed RPN MCTS with batched native cpp_stream scoring."""

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
        policy=SchemaPriorPolicy(),
    ).search()
    return CppStreamRiskMinerResult(search, evaluator.summary)


__all__ = ["CppStreamRiskMinerResult", "search_cpp_stream_alphas"]
