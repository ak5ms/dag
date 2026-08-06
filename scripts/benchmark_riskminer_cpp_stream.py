from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from flows.riskminer import (
    CppStreamPoolEvaluator,
    RiskMinerConfig,
    search_cpp_stream_alphas,
)


ROWS = int(os.environ.get("RISKMINER_ROWS", "25000"))
INSTRUMENTS = int(os.environ.get("RISKMINER_INSTRUMENTS", "9"))
SIMULATIONS = int(os.environ.get("RISKMINER_SIMULATIONS", "24"))
ROLLOUTS = int(os.environ.get("RISKMINER_ROLLOUTS", "4"))
EVALUATION_BATCH = int(os.environ.get("RISKMINER_EVALUATION_BATCH", "8"))
ARCHIVE_SIZE = int(os.environ.get("RISKMINER_ARCHIVE_SIZE", "100"))
POOL_SIZE = int(os.environ.get("RISKMINER_POOL_SIZE", "8"))
SEED = int(os.environ.get("RISKMINER_SEED", "42"))
OUTPUT_DIR = os.environ.get("RISKMINER_OUTPUT_DIR")
KEEP_DATA = os.environ.get("RISKMINER_KEEP_DATA", "0") == "1"


def _open(path: Path, rows: int, instruments: int):
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float64,
        shape=(rows, instruments),
    )


def generate_synthetic_sources(root: Path) -> dict[str, str]:
    """Generate coherent hypothetical market data in bounded memory."""

    root.mkdir(parents=True, exist_ok=True)
    names = (
        "ap0", "bp0", "av0", "bv0", "volume", "vwap", "open", "high",
        "low", "close", "soft_side_wavg", "roll_rets", "hs", "vol",
        "is_tradable",
    )
    arrays = {
        name: _open(root / f"{name}.npy", ROWS, INSTRUMENTS)
        for name in names
    }
    rng = np.random.default_rng(SEED)
    chunk_rows = min(65536, ROWS)
    log_mid = np.log(np.linspace(50.0, 150.0, INSTRUMENTS))
    previous_side = np.zeros(INSTRUMENTS)

    for start in range(0, ROWS, chunk_rows):
        stop = min(ROWS, start + chunk_rows)
        count = stop - start
        side = np.clip(
            0.78 * rng.normal(size=(count, INSTRUMENTS))
            + 0.22 * rng.normal(size=(count, 1)),
            -1.0,
            1.0,
        )
        lagged_side = np.vstack((previous_side[None, :], side[:-1]))
        returns = (
            2.5e-4 * lagged_side
            + rng.normal(0.0, 4.0e-4, (count, INSTRUMENTS))
        )

        price_innovations = rng.normal(
            0.0,
            3.0e-4,
            (count, INSTRUMENTS),
        )
        log_paths = log_mid[None, :] + np.cumsum(price_innovations, axis=0)
        mid = np.exp(log_paths)
        log_mid = log_paths[-1]
        previous_side = side[-1]

        bar_move = rng.normal(0.0, 1.5e-4, (count, INSTRUMENTS))
        open_ = mid * (1.0 - 0.5 * bar_move)
        close = mid * (1.0 + 0.5 * bar_move)
        excursion = np.abs(
            rng.normal(2.0e-4, 8.0e-5, (count, INSTRUMENTS))
        )
        high = np.maximum(open_, close) * (1.0 + excursion)
        low = np.minimum(open_, close) * (1.0 - excursion)
        vwap = np.clip((open_ + close + mid) / 3.0, low, high)

        half_spread_fraction = np.clip(
            rng.lognormal(-9.0, 0.25, (count, INSTRUMENTS)),
            2.0e-5,
            8.0e-4,
        )
        ap0 = mid * (1.0 + half_spread_fraction)
        bp0 = mid * (1.0 - half_spread_fraction)
        quoted_base = rng.lognormal(5.5, 0.6, (count, INSTRUMENTS))
        av0 = quoted_base * np.exp(-0.25 * side)
        bv0 = quoted_base * np.exp(0.25 * side)
        volume = rng.lognormal(5.0, 0.7, (count, INSTRUMENTS))
        vol = np.clip(
            0.008
            + np.abs(rng.normal(0.0, 0.002, (count, INSTRUMENTS))),
            0.002,
            0.03,
        )
        row_index = np.arange(start, stop)[:, None]
        tradable = np.broadcast_to(
            ((row_index % 1440) < 1200).astype(np.float64),
            (count, INSTRUMENTS),
        )

        values = {
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
            "soft_side_wavg": side,
            "roll_rets": returns,
            "hs": half_spread_fraction,
            "vol": vol,
            "is_tradable": tradable,
        }
        for name, value in values.items():
            arrays[name][start:stop] = value

    for array in arrays.values():
        array.flush()
    del arrays
    return {name: str(root / f"{name}.npy") for name in names}


def main() -> None:
    if INSTRUMENTS != 9:
        print(
            "note: the requested production shape is 9 instruments; "
            f"this run uses {INSTRUMENTS}"
        )
    temporary = None
    if OUTPUT_DIR:
        root = Path(OUTPUT_DIR)
        root.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="riskminer_cpp_stream_")
        root = Path(temporary.name)

    data_started = time.perf_counter()
    sources = generate_synthetic_sources(root / "data")
    data_seconds = time.perf_counter() - data_started

    config = RiskMinerConfig(
        max_depth=8,
        max_tokens=32,
        max_stack=7,
        simulations=SIMULATIONS,
        rollouts_per_expansion=ROLLOUTS,
        evaluation_batch_size=EVALUATION_BATCH,
        archive_size=ARCHIVE_SIZE,
        seed=SEED,
    )

    search_result = search_cpp_stream_alphas(
        sources,
        n_instruments=INSTRUMENTS,
        work_dir=root / "candidate_outputs",
        config=config,
    )
    entries = search_result.search.archive
    if not entries:
        raise RuntimeError("RiskMiner produced no finite native candidates")

    pool_entries = entries[: min(POOL_SIZE, len(entries))]
    pool_result = CppStreamPoolEvaluator(
        sources,
        n_instruments=INSTRUMENTS,
        work_dir=root / "pool_output",
    ).evaluate(
        tuple(entry.expr for entry in pool_entries),
        ridge_halflife=min(1440.0 * 5.0, max(32.0, ROWS / 4.0)),
        risk_halflife=min(1440.0 * 5.0, max(32.0, ROWS / 4.0)),
    )

    evaluation = search_result.evaluation
    metrics = search_result.search.metrics
    first_batch = evaluation.batches[0] if evaluation.batches else None
    report = {
        "rows": ROWS,
        "instruments": INSTRUMENTS,
        "seed": SEED,
        "max_depth": config.max_depth,
        "simulations": metrics.simulations,
        "rollouts": metrics.rollouts,
        "tree_nodes": metrics.tree_nodes,
        "formula_requests": metrics.unique_formula_requests,
        "finite_formula_scores": metrics.finite_formula_scores,
        "archive_size": len(entries),
        "data_seconds": data_seconds,
        "search_wall_seconds": metrics.wall_seconds,
        "candidate_compile_seconds": evaluation.compile_seconds,
        "candidate_run_seconds": evaluation.run_seconds,
        "compile_rejected": evaluation.compile_rejected,
        "pool_alpha_count": pool_result.alpha_count,
        "pool_score": pool_result.score,
        "pool_compile_seconds": pool_result.compile_seconds,
        "pool_run_seconds": pool_result.run_seconds,
        "candidate_runtime_type": (
            first_batch.runtime_type if first_batch is not None else None
        ),
        "pool_runtime_type": pool_result.runtime_type,
        "top": [
            {
                "score": entry.score,
                "depth": entry.depth,
                "rpn": entry.rpn,
                "expr": repr(entry.expr),
            }
            for entry in entries[:10]
        ],
    }
    result_path = root / "riskminer_benchmark.json"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    print("backend=trading_dsl_engine.cpp_stream")
    print(
        f"shape={ROWS:,}x{INSTRUMENTS} max_depth={config.max_depth} "
        f"simulations={metrics.simulations} rollouts={metrics.rollouts}"
    )
    print(
        f"data_seconds={data_seconds:.6f} "
        f"search_wall_seconds={metrics.wall_seconds:.6f}"
    )
    print(
        f"candidate_compile_seconds={evaluation.compile_seconds:.6f} "
        f"candidate_run_seconds={evaluation.run_seconds:.6f}"
    )
    print(
        f"tree_nodes={metrics.tree_nodes} "
        f"formula_requests={metrics.unique_formula_requests} "
        f"finite_scores={metrics.finite_formula_scores} "
        f"archive={len(entries)} "
        f"compile_rejected={evaluation.compile_rejected}"
    )
    if first_batch is not None:
        print(f"candidate_runtime={first_batch.runtime_type}")
        print(f"candidate_output_shape={first_batch.output_shape}")
    print("top_formulas:")
    for index, entry in enumerate(entries[:10], start=1):
        print(
            f"  {index:02d} score={entry.score:.8g} "
            f"depth={entry.depth} rpn={entry.rpn}"
        )
        print(f"     expr={entry.expr!r}")
    print(
        f"pool_alpha_count={pool_result.alpha_count} "
        f"pool_score={pool_result.score:.8g} "
        f"pool_compile_seconds={pool_result.compile_seconds:.6f} "
        f"pool_run_seconds={pool_result.run_seconds:.6f}"
    )
    print(f"pool_runtime={pool_result.runtime_type}")
    print(f"result_json={result_path}")
    if KEEP_DATA:
        print(f"data_directory={root / 'data'}")
        if temporary is not None:
            temporary.cleanup = lambda: None  # type: ignore[method-assign]


if __name__ == "__main__":
    main()
