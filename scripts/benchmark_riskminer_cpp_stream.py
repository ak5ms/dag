from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from flows.riskminer import (
    CppStreamCandidateEvaluator,
    CppStreamRidgePoolEvaluator,
    MCTSConfig,
    RPNEnvironment,
    RiskMinerMCTS,
    RiskSeekingTokenPolicy,
    default_market_semantics,
)


ROWS = int(os.environ.get("RISKMINER_ROWS", "4096"))
INSTRUMENTS = int(os.environ.get("RISKMINER_INSTRUMENTS", "9"))
SIMULATIONS = int(os.environ.get("RISKMINER_SIMULATIONS", "48"))
ROLLOUTS = int(os.environ.get("RISKMINER_ROLLOUTS", "4"))
SELECTION_BATCH = int(os.environ.get("RISKMINER_SELECTION_BATCH", "8"))
MAX_DEPTH = int(os.environ.get("RISKMINER_MAX_DEPTH", "8"))
MAX_TOKENS = int(os.environ.get("RISKMINER_MAX_TOKENS", "20"))
ARCHIVE_SIZE = int(os.environ.get("RISKMINER_ARCHIVE_SIZE", "100"))
POOL_SIZE = int(os.environ.get("RISKMINER_POOL_SIZE", "6"))
SEED = int(os.environ.get("RISKMINER_SEED", "42"))
WORK_DIR = Path(os.environ.get("RISKMINER_WORK_DIR", ".riskminer-benchmark"))
RESULT_JSON = Path(os.environ.get("RISKMINER_RESULT_JSON", WORK_DIR / "riskminer-benchmark.json"))


def _synthetic_market_data(rows: int, instruments: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    soft_side = np.tanh(rng.normal(size=(rows, instruments))).astype(np.float64)
    innovations = rng.normal(scale=1.6e-4, size=(rows, instruments))
    roll_rets = innovations
    roll_rets[1:] += 2.8e-4 * soft_side[:-1]
    roll_rets[::997] = 0.0
    log_mid = np.log(100.0) + np.cumsum(roll_rets, axis=0)
    mid = np.exp(log_mid)
    half_spread_fraction = 4.0e-5 + 3.0e-5 * np.abs(rng.normal(size=(rows, instruments)))
    ap0 = mid * (1.0 + half_spread_fraction)
    bp0 = mid * (1.0 - half_spread_fraction)
    close = mid * (1.0 + rng.normal(scale=2.0e-5, size=mid.shape))
    open_ = np.vstack((close[:1], close[:-1]))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(scale=4.0e-5, size=mid.shape)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(scale=4.0e-5, size=mid.shape)))
    vwap = np.clip(mid * (1.0 + rng.normal(scale=1.5e-5, size=mid.shape)), low, high)
    volume = rng.lognormal(mean=8.0, sigma=0.45, size=mid.shape)
    av0 = rng.lognormal(mean=7.4, sigma=0.35, size=mid.shape)
    bv0 = rng.lognormal(mean=7.4, sigma=0.35, size=mid.shape)
    is_tradable = np.ones((rows, instruments), dtype=np.float64)
    cycle = np.arange(rows) % 512
    is_tradable[cycle >= 496] = 0.0
    hs = np.maximum(half_spread_fraction, 1.0e-6)
    vol = np.broadcast_to(np.linspace(0.0075, 0.015, instruments, dtype=np.float64), (rows, instruments)).copy()
    return {
        "ap0": ap0,
        "bp0": bp0,
        "av0": av0,
        "bv0": bv0,
        "volume": volume,
        "vwap": vwap,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "soft_side_wavg": soft_side,
        "roll_rets": roll_rets,
        "hs": hs,
        "vol": vol,
        "is_tradable": is_tradable,
    }


def main() -> None:
    if ROWS < 32 or INSTRUMENTS < 2:
        raise ValueError("benchmark requires at least 32 rows and 2 instruments")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    generation_started = time.perf_counter()
    sources = _synthetic_market_data(ROWS, INSTRUMENTS, SEED)
    generation_seconds = time.perf_counter() - generation_started
    environment = RPNEnvironment(terminals=default_market_semantics(), target_types=("dimensionless",), max_depth=MAX_DEPTH, max_tokens=MAX_TOKENS)
    evaluator = CppStreamCandidateEvaluator(sources, n_instruments=INSTRUMENTS, work_dir=WORK_DIR / "candidate-evaluation", max_batch_size=64)
    policy = RiskSeekingTokenPolicy(risk_quantile=0.80, learning_rate=0.01, seed=SEED)
    search = RiskMinerMCTS(
        environment,
        evaluator,
        policy=policy,
        config=MCTSConfig(
            simulations=SIMULATIONS,
            rollouts_per_expansion=ROLLOUTS,
            selection_batch_size=SELECTION_BATCH,
            archive_size=ARCHIVE_SIZE,
            seed=SEED,
        ),
    )
    search_started = time.perf_counter()
    report = search.search()
    search_seconds = time.perf_counter() - search_started
    if not report.candidates:
        examples = list(evaluator.stats.rejection_messages.values())[:5]
        raise RuntimeError(f"RiskMiner produced no finite cpp_stream candidates; sample rejections={examples}")
    pool_count = min(POOL_SIZE, len(report.candidates))
    pool_alphas = [record.expr for record in report.candidates[:pool_count]]
    pool_evaluator = CppStreamRidgePoolEvaluator(sources, n_instruments=INSTRUMENTS, work_dir=WORK_DIR / "pool-evaluation")
    pool_started = time.perf_counter()
    pool = pool_evaluator.evaluate(pool_alphas)
    pool_total_seconds = time.perf_counter() - pool_started
    top = [
        {
            "rank": rank,
            "score": record.score,
            "depth": record.depth,
            "nodes": record.node_count,
            "tokens": environment.format_tokens(record.token_ids),
            "expression": repr(record.expr),
        }
        for rank, record in enumerate(report.candidates[:10], start=1)
    ]
    payload = {
        "backend": "trading_dsl_engine.cpp_stream",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rows": ROWS,
        "instruments": INSTRUMENTS,
        "seed": SEED,
        "max_depth": MAX_DEPTH,
        "max_tokens": MAX_TOKENS,
        "simulations": report.simulations,
        "rollouts_per_expansion": ROLLOUTS,
        "selection_batch_size": SELECTION_BATCH,
        "rollout_proposals": report.rollout_proposals,
        "finite_proposals": report.finite_proposals,
        "dead_rollouts": report.dead_rollouts,
        "tree_nodes": report.tree_nodes,
        "archive_count": len(report.candidates),
        "policy_quantile": report.policy_quantile,
        "generation_seconds": generation_seconds,
        "search_seconds": search_seconds,
        "candidate_compile_seconds": evaluator.stats.compile_seconds,
        "candidate_run_seconds": evaluator.stats.run_seconds,
        "candidate_compiled_batches": evaluator.stats.compiled_batches,
        "candidate_compile_failures": evaluator.stats.compile_failures,
        "candidate_execution_failures": evaluator.stats.execution_failures,
        "candidate_nonfinite_scores": evaluator.stats.nonfinite_scores,
        "runtime_type": evaluator.stats.last_runtime_type,
        "runtime_output_mode": evaluator.stats.last_output_mode,
        "runtime_output_shape": evaluator.stats.last_output_shape,
        "runtime_input_names": evaluator.stats.last_input_names,
        "runtime_native_path": evaluator.stats.last_native_path,
        "pool_size": pool_count,
        "pool_sharpe": pool.sharpe,
        "pool_compile_seconds": pool.compile_seconds,
        "pool_run_seconds": pool.run_seconds,
        "pool_total_seconds": pool_total_seconds,
        "pool_output_mode": pool.output_mode,
        "pool_output_shape": pool.output_shape,
        "pool_runtime_type": pool.runtime_type,
        "top_candidates": top,
        "sample_rejections": list(evaluator.stats.rejection_messages.values())[:10],
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("=== RiskMiner / cpp_stream smoke benchmark ===")
    print(f"rows={ROWS:,} instruments={INSTRUMENTS} max_depth={MAX_DEPTH} simulations={report.simulations} rollouts={ROLLOUTS}")
    print(f"proposals={report.rollout_proposals} finite={report.finite_proposals} archive={len(report.candidates)} tree_nodes={report.tree_nodes}")
    print(f"search={search_seconds:.3f}s cpp_compile={evaluator.stats.compile_seconds:.3f}s cpp_run={evaluator.stats.run_seconds:.3f}s compiled_batches={evaluator.stats.compiled_batches}")
    print(f"backend={evaluator.stats.last_runtime_type} output_mode={evaluator.stats.last_output_mode} output_shape={evaluator.stats.last_output_shape}")
    print("--- top formulas ---")
    for item in top:
        print(f"#{item['rank']:02d} score={item['score']:.6g} depth={item['depth']} nodes={item['nodes']} tokens={item['tokens']}")
        print(f"    {item['expression']}")
    print("--- Ridge pool ---")
    print(f"pool_size={pool_count} sharpe={pool.sharpe:.6g} compile={pool.compile_seconds:.3f}s run={pool.run_seconds:.3f}s output_mode={pool.output_mode}")
    print(f"result_json={RESULT_JSON}")


if __name__ == "__main__":
    main()
