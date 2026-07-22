from __future__ import annotations

import csv
import gc
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.jax_flat import compile_features, compile_formula
from trading_dsl_engine.jax_flat import engine
from trading_dsl_engine.jax_flat import engine_legacy

jax.config.update("jax_enable_x64", True)

ASSETS = 9
RUNS = 10
GROUP = os.environ.get("BENCH_GROUP", "stateless")
VALUES: dict[int, jax.Array] = {}
RESULTS: list[dict[str, object]] = []


def values_for(rows: int) -> jax.Array:
    value = VALUES.get(rows)
    if value is None:
        value = jax.random.normal(
            jax.random.PRNGKey(rows % (2**31 - 1)),
            (rows, ASSETS),
            dtype=jnp.float64,
        )
        value.block_until_ready()
        VALUES[rows] = value
    return value


def affine(expr: str, level: int, branch: int) -> str:
    scale = 1.0001 + 0.00007 * level + 0.000011 * branch
    offset = 0.001 * (level + 1) + 0.0001 * branch
    return f"add(mul({expr}, {scale:.12f}), {offset:.12f})"


def formula(kind: str, depth: int, branch: int) -> str:
    expr = "x"
    for level in range(depth):
        if kind == "stateless":
            expr = affine(expr, level, branch)
        elif kind == "stateful":
            span = 5.0 + 6.0 * level + 2.0 * branch
            expr = f"ewm({expr}, {span:.1f})"
        elif kind == "mixed":
            if level % 2 == 0:
                expr = affine(expr, level, branch)
            else:
                span = 7.0 + 8.0 * level + 2.0 * branch
                expr = f"ewm({expr}, {span:.1f})"
        else:
            raise ValueError(kind)
    return expr


def checksum(output) -> float:
    total = 0.0
    for leaf in jax.tree_util.tree_leaves(output):
        arr = np.asarray(jax.device_get(leaf))
        if arr.size:
            sample = arr.reshape(-1)[:: max(1, arr.size // 1024)]
            total += float(np.nansum(sample))
    return total


def paired_timings(before_fn, after_fn) -> tuple[list[float], list[float]]:
    before_fn()
    after_fn()
    before_samples: list[float] = []
    after_samples: list[float] = []
    for run in range(RUNS):
        ordered = ((before_fn, before_samples), (after_fn, after_samples))
        if run % 2:
            ordered = tuple(reversed(ordered))
        for fn, samples in ordered:
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
    return before_samples, after_samples


def summarize(
    *,
    case: str,
    family: str,
    rows: int,
    depth: int | None,
    breadth: int | None,
    strategy: str,
    before_samples: list[float],
    after_samples: list[float],
    notes: str = "",
) -> None:
    before = statistics.median(before_samples)
    after = statistics.median(after_samples)
    result = {
        "case": case,
        "family": family,
        "rows": rows,
        "assets": ASSETS,
        "depth": depth,
        "breadth": breadth,
        "strategy": strategy,
        "before_median_s": before,
        "after_median_s": after,
        "speedup_x": before / after,
        "delta_pct": 100.0 * (after / before - 1.0),
        "before_min_s": min(before_samples),
        "after_min_s": min(after_samples),
        "before_samples": before_samples,
        "after_samples": after_samples,
        "notes": notes,
    }
    RESULTS.append(result)
    print(json.dumps(result, sort_keys=True), flush=True)


def benchmark_formula_family(kind: str, rows: int, depth: int, breadth: int) -> None:
    values = values_for(rows)
    data = {"x": values}
    formulas = {f"f{branch}": formula(kind, depth, branch) for branch in range(breadth)}

    if breadth == 1:
        expression = formulas["f0"]
        before_runtime = engine_legacy.compile_formula(expression, cpp=False)
        after_runtime = compile_formula(expression, cpp=False)
        _, before_check = before_runtime.run_batch(data)
        _, after_check = after_runtime.run_batch(data)
        np.testing.assert_allclose(
            np.asarray(before_check), np.asarray(after_check),
            rtol=1e-10, atol=1e-10, equal_nan=True,
        )

        def before_fn() -> float:
            return checksum(before_runtime.run_batch(data)[1])

        def after_fn() -> float:
            return checksum(after_runtime.run_batch(data)[1])

        plan = engine.build_execution_plan(after_runtime.program)
    else:
        before_runtimes = {
            name: engine_legacy.compile_formula(expression, cpp=False)
            for name, expression in formulas.items()
        }
        after_runtime = compile_features(formulas, cpp=False)
        before_check = {
            name: runtime.run_batch(data)[1]
            for name, runtime in before_runtimes.items()
        }
        after_check = after_runtime.run_batch(data)[1]
        for name in formulas:
            np.testing.assert_allclose(
                np.asarray(before_check[name]), np.asarray(after_check[name]),
                rtol=1e-10, atol=1e-10, equal_nan=True,
            )

        def before_fn() -> float:
            total = 0.0
            for runtime in before_runtimes.values():
                total += checksum(runtime.run_batch(data)[1])
            return total

        def after_fn() -> float:
            return checksum(after_runtime.run_batch(data)[1])

        plan = engine.build_execution_plan(after_runtime.program)

    before_samples, after_samples = paired_timings(before_fn, after_fn)
    summarize(
        case=f"{kind}_r{rows}_d{depth}_b{breadth}",
        family=kind,
        rows=rows,
        depth=depth,
        breadth=breadth,
        strategy=plan.strategy,
        before_samples=before_samples,
        after_samples=after_samples,
        notes=(
            "Breadth>1 before runs independently compiled formulas; after uses one shared multi-root DAG. "
            "Mixed depth alternates stateless affine and EWM stages."
        ),
    )
    gc.collect()
    jax.clear_caches()


def benchmark_single_formula(
    *,
    case: str,
    family: str,
    rows: int,
    expression: str,
    data: dict[str, object],
    depth: int | None = None,
    breadth: int | None = 1,
    notes: str = "",
) -> None:
    before_runtime = engine_legacy.compile_formula(expression, cpp=False)
    after_runtime = compile_formula(expression, cpp=False)
    before_check = before_runtime.run_batch(data)[1]
    after_check = after_runtime.run_batch(data)[1]
    np.testing.assert_allclose(
        np.asarray(before_check), np.asarray(after_check),
        rtol=1e-9, atol=1e-9, equal_nan=True,
    )

    def before_fn() -> float:
        return checksum(before_runtime.run_batch(data)[1])

    def after_fn() -> float:
        return checksum(after_runtime.run_batch(data)[1])

    before_samples, after_samples = paired_timings(before_fn, after_fn)
    eligible = all(
        type(node.op).__name__ not in {"GroupByOp", "RidgeOp", "InstrumentBasisMeanOp", "CacheOp"}
        for node in after_runtime.program.nodes
    )
    strategy = engine.build_execution_plan(after_runtime.program).strategy if eligible else "legacy_fallback"
    summarize(
        case=case,
        family=family,
        rows=rows,
        depth=depth,
        breadth=breadth,
        strategy=strategy,
        before_samples=before_samples,
        after_samples=after_samples,
        notes=notes,
    )
    gc.collect()
    jax.clear_caches()


def benchmark_core(kind: str) -> None:
    for depth in (3, 5, 8):
        for breadth in (1, 4, 8):
            benchmark_formula_family(kind, 1_000_000, depth, breadth)
    for rows in (100_000, 3_000_000):
        benchmark_formula_family(kind, rows, 5, 1)


def benchmark_operators() -> None:
    values = values_for(1_000_000)
    benchmark_single_formula(
        case="prefix_cumsum_depth5",
        family="prefix",
        rows=1_000_000,
        depth=5,
        expression="cumsum(cumsum(cumsum(cumsum(cumsum(x)))))",
        data={"x": values},
        notes="Five chained prefix sums.",
    )
    benchmark_single_formula(
        case="lookback_shift_depth3",
        family="lookback",
        rows=1_000_000,
        depth=3,
        expression="shift(shift(shift(x, 1), 2), 3)",
        data={"x": values},
        notes="Three static shifts.",
    )
    benchmark_single_formula(
        case="rolling_mean_depth3",
        family="lookback",
        rows=1_000_000,
        depth=3,
        expression="roll_mean(roll_mean(roll_mean(x, 16), 16), 16)",
        data={"x": values},
        notes="Three rolling means with lookback 16.",
    )


def benchmark_blockers() -> None:
    rows = 100_000
    x = values_for(rows)
    row = jnp.arange(rows, dtype=jnp.float64)[:, None]
    col = jnp.arange(ASSETS, dtype=jnp.float64)[None, :]
    key = jnp.mod(row + col, 128.0)
    z = jnp.cos(x * 0.3) + 0.1 * x
    y = 0.4 * x - 0.2 * z + 0.05
    cases = (
        (
            "groupby_dynamic_cumsum", "groupby",
            "groupby((key,), x, cumsum(self_))", {"x": x, "key": key},
            "Dynamic-key groupby blocker; planner falls back to the original executor.",
        ),
        (
            "groupby_dynamic_nested_cumsum", "groupby",
            "groupby((key,), x, cumsum(cumsum(self_)))", {"x": x, "key": key},
            "Nested stateful groupby blocker; planner falls back.",
        ),
        (
            "groupby_universe_cumsum", "universe_groupby",
            "groupby((univ([0, 1, 2], [3, 4, 5], [6, 7, 8]), key), x, cumsum(self_))",
            {"x": x, "key": key},
            "Three universe partitions plus a dynamic key; planner falls back.",
        ),
        (
            "ridge_instant_preds", "ridge",
            "get_preds(Ridge(cat(x, z), y, 1.0, 0.0, 0.01))", {"x": x, "z": z, "y": y},
            "Instantaneous two-feature cross-sectional Ridge; object-producing node falls back.",
        ),
        (
            "ridge_ewm_preds", "ridge",
            "get_preds(Ridge(cat(x, z), y, 1.0, 32.0, 0.01))", {"x": x, "z": z, "y": y},
            "Stateful EWM Ridge with halflife 32; object-producing node falls back.",
        ),
    )
    for case, family, expression, data, notes in cases:
        benchmark_single_formula(
            case=case,
            family=family,
            rows=rows,
            expression=expression,
            data=data,
            notes=notes,
        )


def benchmark_memmap() -> None:
    rows = 500_000
    for kind in ("stateless", "stateful"):
        expression = formula(kind, 5, 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.dat"
            mapped = np.memmap(input_path, mode="w+", dtype=np.float64, shape=(rows, ASSETS))
            mapped[:] = np.asarray(values_for(rows))
            mapped.flush()
            del mapped
            mapped = np.memmap(input_path, mode="r", dtype=np.float64, shape=(rows, ASSETS))
            data = {"x": mapped}
            before_runtime = engine_legacy.compile_formula(expression, cpp=False)
            after_runtime = compile_formula(expression, cpp=False)
            before_path = root / "before.dat"
            after_path = root / "after.dat"

            def run(runtime, path: Path) -> float:
                if path.exists():
                    path.unlink()
                output = runtime.run_batch(data, out_path=str(path))[1]
                total = float(np.nansum(np.asarray(output)[::4096]))
                del output
                return total

            before_check = run(before_runtime, before_path)
            after_check = run(after_runtime, after_path)
            np.testing.assert_allclose(before_check, after_check, rtol=1e-10, atol=1e-10)
            before_samples, after_samples = paired_timings(
                lambda: run(before_runtime, before_path),
                lambda: run(after_runtime, after_path),
            )
            summarize(
                case=f"memmap_{kind}_r{rows}_d5_b1",
                family="host_io",
                rows=rows,
                depth=5,
                breadth=1,
                strategy=engine.build_execution_plan(after_runtime.program).strategy,
                before_samples=before_samples,
                after_samples=after_samples,
                notes="Host memmap input and incrementally flushed memmap output; includes I/O and synchronization.",
            )


def write_results() -> None:
    fields = [
        "case", "family", "rows", "assets", "depth", "breadth", "strategy",
        "before_median_s", "after_median_s", "speedup_x", "delta_pct",
        "before_min_s", "after_min_s", "notes",
    ]
    stem = f"benchmark_{GROUP}"
    with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in RESULTS:
            writer.writerow({field: result.get(field) for field in fields})
    with open(f"{stem}.json", "w", encoding="utf-8") as handle:
        json.dump(RESULTS, handle, indent=2)

    lines = [
        f"# Integrated JAX-flat runtime matrix: {GROUP}",
        "",
        "GitHub Actions CPU, JAX float64, 9 assets, one warmup plus 10 measured runs; medians shown; compilation excluded.",
        "Before is the preserved original JAX-flat batch executor. After is the integrated public runtime.",
        "",
        "| Case | Rows | Depth | Breadth | Strategy | Before (s) | After (s) | Speedup | Δ runtime |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for result in RESULTS:
        depth = result["depth"] if result["depth"] is not None else "—"
        breadth = result["breadth"] if result["breadth"] is not None else "—"
        lines.append(
            f"| {result['case']} | {int(result['rows']):,} | {depth} | {breadth} | `{result['strategy']}` | "
            f"{float(result['before_median_s']):.4f} | {float(result['after_median_s']):.4f} | "
            f"{float(result['speedup_x']):.2f}× | {float(result['delta_pct']):+.1f}% |"
        )
    with open(f"{stem}.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    speedups = [float(result["speedup_x"]) for result in RESULTS]
    print(json.dumps({
        "group_summary": {
            "group": GROUP,
            "cases": len(RESULTS),
            "geomean_speedup_x": math.exp(statistics.mean(math.log(value) for value in speedups)),
            "median_speedup_x": statistics.median(speedups),
            "min_speedup_x": min(speedups),
            "max_speedup_x": max(speedups),
        }
    }, sort_keys=True), flush=True)


if GROUP in {"stateless", "stateful", "mixed"}:
    benchmark_core(GROUP)
elif GROUP == "operators":
    benchmark_operators()
elif GROUP == "blockers":
    benchmark_blockers()
elif GROUP == "memmap":
    benchmark_memmap()
else:
    raise ValueError(f"Unknown BENCH_GROUP={GROUP!r}")

write_results()
