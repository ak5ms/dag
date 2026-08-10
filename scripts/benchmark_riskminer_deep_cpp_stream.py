from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import tempfile
import time

from benchmark_riskminer_cpp_stream import generate_synthetic_sources
from flows.riskminer import (
    CppStreamCandidateEvaluator,
    RiskMCTS,
    RiskMinerConfig,
    SchemaPriorPolicy,
    TokenKind,
    TypedRPNEnvironment,
    build_vocabulary,
)
from flows.riskminer.mcts import FormulaObservation, Rollout
from flows.riskminer.semantics import SearchShape


ROWS = int(os.environ.get("RISKMINER_ROWS", "5000"))
INSTRUMENTS = int(os.environ.get("RISKMINER_INSTRUMENTS", "9"))
SIMULATIONS = int(os.environ.get("RISKMINER_SIMULATIONS", "48"))
ROLLOUTS = int(os.environ.get("RISKMINER_ROLLOUTS", "4"))
EVALUATION_BATCH = int(os.environ.get("RISKMINER_EVALUATION_BATCH", "8"))
ARCHIVE_SIZE = int(os.environ.get("RISKMINER_ARCHIVE_SIZE", "100"))
MAX_DEPTH = int(os.environ.get("RISKMINER_MAX_DEPTH", "8"))
MIN_FORMULA_DEPTH = int(os.environ.get("RISKMINER_MIN_FORMULA_DEPTH", "5"))
MAX_TOKENS = int(os.environ.get("RISKMINER_MAX_TOKENS", "28"))
SEED = int(os.environ.get("RISKMINER_SEED", "42"))
OUTPUT_DIR = os.environ.get("RISKMINER_OUTPUT_DIR")
KEEP_DATA = os.environ.get("RISKMINER_KEEP_DATA", "0") == "1"


class DeepTypedRPNEnvironment(TypedRPNEnvironment):
    """Require a nontrivial minimum expression depth before scoring or END."""

    def formula_value(self, state):
        value = super().formula_value(state)
        if value is None or value.depth < self.config.min_formula_depth:
            return None
        return value


class GuidedDeepRiskMCTS(RiskMCTS):
    """Use typed RPN macro completions instead of blind stack growth.

    The tree still expands one token at a time. During a rollout, however, a
    one-expression stack is extended by a legal unary transform, a
    ``literal + temporal operator`` pair, or a ``terminal + binary operator``
    pair. This keeps the stack near one complete expression and makes depth 5-8
    programs common without hand-writing alpha templates.
    """

    def _weighted_choice(self, token_ids, multipliers=None):
        multipliers = multipliers or {}
        weights = []
        for token_id in token_ids:
            token = self.environment.vocabulary.by_id[token_id]
            weights.append(
                max(1.0e-12, token.prior)
                * float(multipliers.get(token.name, 1.0))
            )
        return self.rng.choices(list(token_ids), weights=weights, k=1)[0]

    def _rollout(self, leaf):
        state = leaf
        formulas: list[FormulaObservation] = []
        last_key = None

        def observe(current):
            nonlocal last_key
            value = self.environment.formula_value(current)
            if value is not None and value.canonical_key != last_key:
                formulas.append(FormulaObservation(value, current.token_ids))
                last_key = value.canonical_key

        def apply(token_id):
            nonlocal state
            state = self.environment.apply(state, token_id)
            observe(state)

        def legal_operator_ids(current, *, minimum_arity=1, maximum_arity=3):
            result = []
            for token_id in self.environment.legal_actions(current):
                token = self.environment.vocabulary.by_id[token_id]
                if (
                    token.kind is TokenKind.OPERATOR
                    and token.operator is not None
                    and minimum_arity <= token.operator.arity <= maximum_arity
                ):
                    result.append(token_id)
            return result

        def normalizer_ids(current):
            names = {"xs_rank", "xs_pct_rank", "sign", "arctan", "fraction"}
            return [
                token_id
                for token_id in legal_operator_ids(current, maximum_arity=1)
                if self.environment.vocabulary.by_id[token_id].name in names
            ]

        def temporal_macros(current):
            pairs = []
            legal = set(self.environment.legal_actions(current))
            for token in self.environment.vocabulary:
                if token.token_id not in legal or token.kind is not TokenKind.LITERAL:
                    continue
                if token.value is None or token.value.literal_value is None:
                    continue
                if not math.isfinite(token.value.literal_value) or token.value.literal_value <= 0:
                    continue
                pushed = self.environment.apply(current, token.token_id)
                for operator_id in legal_operator_ids(pushed, minimum_arity=2, maximum_arity=2):
                    operator = self.environment.vocabulary.by_id[operator_id]
                    if operator.operator is not None and operator.operator.family in {
                        "temporal", "history", "rolling"
                    }:
                        pairs.append((token.token_id, operator_id))
            return pairs

        def binary_macros(current):
            pairs = []
            legal = set(self.environment.legal_actions(current))
            for token in self.environment.vocabulary:
                if token.token_id not in legal or token.kind is not TokenKind.TERMINAL:
                    continue
                pushed = self.environment.apply(current, token.token_id)
                for operator_id in legal_operator_ids(pushed, minimum_arity=2, maximum_arity=2):
                    operator = self.environment.vocabulary.by_id[operator_id]
                    if operator.operator is not None and operator.operator.family in {
                        "compatible_binary", "numeric_binary"
                    }:
                        pairs.append((token.token_id, operator_id))
            return pairs

        observe(state)
        while not state.terminated and state.token_count < self.config.max_tokens:
            legal = self.environment.legal_actions(state)
            if not legal:
                break

            if not state.stack:
                terminals = [
                    token_id
                    for token_id in legal
                    if self.environment.vocabulary.by_id[token_id].kind
                    is TokenKind.TERMINAL
                ]
                if not terminals:
                    break
                apply(
                    self._weighted_choice(
                        terminals,
                        {"soft_side_wavg": 3.0},
                    )
                )
                continue

            if len(state.stack) > 1:
                reducers = legal_operator_ids(state, minimum_arity=2)
                if reducers:
                    apply(
                        self._weighted_choice(
                            reducers,
                            {
                                "div": 2.0,
                                "sub": 1.8,
                                "mul": 1.2,
                                "add": 1.0,
                            },
                        )
                    )
                    continue
                unary = legal_operator_ids(state, maximum_arity=1)
                if unary:
                    apply(self._weighted_choice(unary))
                    continue
                break

            value = state.stack[0]
            if (
                self.environment.can_terminate(state)
                and (
                    value.depth >= self.config.max_depth
                    or self.rng.random() < self.config.rollout_end_probability
                )
            ):
                apply(self.environment.vocabulary.end.token_id)
                break

            if value.depth >= self.config.max_depth:
                normalizers = normalizer_ids(state)
                if normalizers and not self.environment.can_terminate(state):
                    apply(self._weighted_choice(normalizers))
                    continue
                if self.environment.can_terminate(state):
                    apply(self.environment.vocabulary.end.token_id)
                break

            unary = legal_operator_ids(state, maximum_arity=1)
            temporal = temporal_macros(state)
            binary = binary_macros(state)

            categories = []
            category_weights = []
            if unary:
                categories.append("unary")
                # A non-dimensionless expression needs a normalizer more urgently.
                category_weights.append(
                    3.0 if "dimensionless" not in value.semantics.types else 1.5
                )
            if temporal:
                categories.append("temporal")
                category_weights.append(4.0)
            if binary:
                categories.append("binary")
                category_weights.append(3.0)
            if not categories:
                break

            category = self.rng.choices(
                categories,
                weights=category_weights,
                k=1,
            )[0]
            if category == "unary":
                candidates = normalizer_ids(state) or unary
                apply(
                    self._weighted_choice(
                        candidates,
                        {"xs_rank": 2.5, "xs_pct_rank": 2.2, "arctan": 1.2},
                    )
                )
            elif category == "temporal":
                literal_id, operator_id = self.rng.choice(temporal)
                apply(literal_id)
                apply(operator_id)
            else:
                terminal_id, operator_id = self.rng.choice(binary)
                apply(terminal_id)
                apply(operator_id)

        # Deterministically repair a nearly complete rollout when enough token
        # budget remains. This uses only legal typed actions.
        while not state.terminated and state.token_count < self.config.max_tokens:
            if len(state.stack) > 1:
                reducers = legal_operator_ids(state, minimum_arity=2)
                if not reducers:
                    break
                apply(self._weighted_choice(reducers, {"div": 2.0, "sub": 1.5}))
                continue
            if len(state.stack) != 1:
                break
            value = state.stack[0]
            if self.environment.can_terminate(state):
                apply(self.environment.vocabulary.end.token_id)
                break
            if value.semantics.shape is SearchShape.ROW:
                candidates = normalizer_ids(state)
                if candidates:
                    apply(self._weighted_choice(candidates, {"xs_rank": 2.0}))
                    continue
            unary = legal_operator_ids(state, maximum_arity=1)
            if unary and value.depth < self.config.min_formula_depth:
                apply(self._weighted_choice(unary))
                continue
            break

        return Rollout(state, tuple(formulas))


def _entry_payload(entry) -> dict[str, object]:
    return {
        "score": float(entry.score),
        "depth": int(entry.depth),
        "rpn": entry.rpn,
        "expr": repr(entry.expr),
    }


def main() -> None:
    if MIN_FORMULA_DEPTH > MAX_DEPTH:
        raise ValueError("RISKMINER_MIN_FORMULA_DEPTH exceeds RISKMINER_MAX_DEPTH")

    temporary = None
    if OUTPUT_DIR:
        root = Path(OUTPUT_DIR)
        root.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="riskminer_deep_cpp_stream_")
        root = Path(temporary.name)

    data_started = time.perf_counter()
    sources = generate_synthetic_sources(root / "data")
    data_seconds = time.perf_counter() - data_started

    config = RiskMinerConfig(
        max_depth=MAX_DEPTH,
        min_formula_depth=MIN_FORMULA_DEPTH,
        max_tokens=MAX_TOKENS,
        max_stack=8,
        simulations=SIMULATIONS,
        rollouts_per_expansion=ROLLOUTS,
        evaluation_batch_size=EVALUATION_BATCH,
        archive_size=ARCHIVE_SIZE,
        rollout_end_probability=0.35,
        seed=SEED,
    )
    environment = DeepTypedRPNEnvironment(
        config=config,
        vocabulary=build_vocabulary(),
        target_types=("dimensionless",),
    )
    evaluator = CppStreamCandidateEvaluator(
        sources,
        n_instruments=INSTRUMENTS,
        work_dir=root / "candidate_outputs",
        batch_size=EVALUATION_BATCH,
    )

    search_started = time.perf_counter()
    search = GuidedDeepRiskMCTS(
        environment,
        evaluator,
        config=config,
        policy=SchemaPriorPolicy(),
    ).search()
    search_wall_seconds = time.perf_counter() - search_started
    entries = search.archive
    if not entries:
        reasons = list(evaluator.summary.rejection_reasons.values())[:10]
        raise RuntimeError(
            "higher-depth RiskMiner search produced no finite native candidates; "
            f"sample rejections={reasons}"
        )

    depth_histogram = Counter(entry.depth for entry in entries)
    deepest = sorted(
        entries,
        key=lambda entry: (-entry.depth, -entry.score, entry.rpn),
    )[:20]
    first_batch = evaluator.summary.batches[0] if evaluator.summary.batches else None
    max_achieved_depth = max(entry.depth for entry in entries)

    report = {
        "backend": "trading_dsl_engine.cpp_stream",
        "guided_rpn_rollouts": True,
        "rows": ROWS,
        "instruments": INSTRUMENTS,
        "seed": SEED,
        "max_depth": MAX_DEPTH,
        "min_formula_depth": MIN_FORMULA_DEPTH,
        "max_achieved_depth": max_achieved_depth,
        "max_tokens": MAX_TOKENS,
        "simulations": search.metrics.simulations,
        "rollouts_per_expansion": ROLLOUTS,
        "rollouts": search.metrics.rollouts,
        "tree_nodes": search.metrics.tree_nodes,
        "formula_requests": search.metrics.unique_formula_requests,
        "finite_formula_scores": search.metrics.finite_formula_scores,
        "invalid_rollouts": search.metrics.invalid_rollouts,
        "archive_size": len(entries),
        "depth_histogram": {
            str(depth): count
            for depth, count in sorted(depth_histogram.items())
        },
        "data_seconds": data_seconds,
        "search_wall_seconds": search_wall_seconds,
        "candidate_compile_seconds": evaluator.summary.compile_seconds,
        "candidate_run_seconds": evaluator.summary.run_seconds,
        "compile_rejected": evaluator.summary.compile_rejected,
        "nonfinite": evaluator.summary.nonfinite,
        "candidate_runtime_type": (
            first_batch.runtime_type if first_batch is not None else None
        ),
        "candidate_output_shape": (
            first_batch.output_shape if first_batch is not None else None
        ),
        "candidate_native_cache_path": (
            first_batch.cache_path if first_batch is not None else None
        ),
        "top_by_score": [_entry_payload(entry) for entry in entries[:20]],
        "deepest": [_entry_payload(entry) for entry in deepest],
        "sample_rejections": list(
            evaluator.summary.rejection_reasons.values()
        )[:10],
    }
    result_path = root / "riskminer_deep_benchmark.json"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    print("=== Higher-depth RiskMiner / cpp_stream run ===")
    print("backend=trading_dsl_engine.cpp_stream")
    print("guided_rpn_rollouts=True")
    print(
        f"shape={ROWS:,}x{INSTRUMENTS} min_depth={MIN_FORMULA_DEPTH} "
        f"max_depth={MAX_DEPTH} achieved={max_achieved_depth}"
    )
    print(
        f"simulations={search.metrics.simulations} "
        f"rollouts={search.metrics.rollouts} "
        f"tree_nodes={search.metrics.tree_nodes}"
    )
    print(
        f"formula_requests={search.metrics.unique_formula_requests} "
        f"finite_scores={search.metrics.finite_formula_scores} "
        f"archive={len(entries)} invalid_rollouts={search.metrics.invalid_rollouts}"
    )
    print(f"depth_histogram={dict(sorted(depth_histogram.items()))}")
    print(
        f"data_seconds={data_seconds:.6f} "
        f"search_wall_seconds={search_wall_seconds:.6f} "
        f"cpp_compile_seconds={evaluator.summary.compile_seconds:.6f} "
        f"cpp_run_seconds={evaluator.summary.run_seconds:.6f}"
    )
    if first_batch is not None:
        print(f"runtime_type={first_batch.runtime_type}")
        print(f"output_shape={first_batch.output_shape}")
        print(f"native_cache_path={first_batch.cache_path}")

    print("--- top by score ---")
    for index, entry in enumerate(entries[:20], start=1):
        print(
            f"{index:02d}. score={entry.score:.10g} depth={entry.depth} "
            f"rpn={entry.rpn}"
        )
        print(f"    expr={entry.expr!r}")

    print("--- deepest formulas ---")
    for index, entry in enumerate(deepest, start=1):
        print(
            f"{index:02d}. depth={entry.depth} score={entry.score:.10g} "
            f"rpn={entry.rpn}"
        )
        print(f"    expr={entry.expr!r}")

    print(f"result_json={result_path}")
    if KEEP_DATA:
        print(f"data_directory={root / 'data'}")
        if temporary is not None:
            temporary.cleanup = lambda: None  # type: ignore[method-assign]


if __name__ == "__main__":
    main()
