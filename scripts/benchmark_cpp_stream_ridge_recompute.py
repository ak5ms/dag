from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import re
from statistics import mean, median
import subprocess
import tempfile
from time import perf_counter

import numpy as np

from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.ir import compile_ir


N = int(os.environ.get("CPP_STREAM_RIDGE_RECOMPUTE_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_RIDGE_RECOMPUTE_RUNS", "10"))
WARMUPS = int(os.environ.get("CPP_STREAM_RIDGE_RECOMPUTE_WARMUPS", "1"))
INTERVALS = tuple(
    int(value)
    for value in os.environ.get(
        "CPP_STREAM_RIDGE_RECOMPUTE_INTERVALS", "1,2,4,8,16,64"
    ).split(",")
    if value.strip()
)
SCALING_INTERVALS = tuple(
    int(value)
    for value in os.environ.get(
        "CPP_STREAM_RIDGE_RECOMPUTE_SCALING_INTERVALS", "1,8,64"
    ).split(",")
    if value.strip()
)
OUTPUT_DIR = os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR")
REPORT_DIR = Path(
    os.environ.get(
        "CPP_STREAM_RIDGE_RECOMPUTE_REPORT_DIR",
        "src/trading_dsl_engine/cpp_stream",
    )
)
BASELINE_LOG = os.environ.get("CPP_STREAM_RIDGE_BASELINE_LOG")
HEAD_LOG = os.environ.get("CPP_STREAM_RIDGE_HEAD_LOG")


@dataclass(frozen=True)
class Case:
    features: int
    field: str
    rows: int
    recompute_every: int


@dataclass
class Result:
    features: int
    field: str
    rows: int
    instruments: int
    recompute_every: int
    rates_mrows: list[float]
    median_mrows: float
    mean_mrows: float
    best_mrows: float
    median_seconds: float
    speedup: float
    checksum: float
    finite_fraction: float
    output_row_width: int
    generated_cpp: str


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _case_rows(features: int, field: str) -> int:
    defaults = {
        (9, "beta"): 200_000,
        (9, "preds"): 200_000,
        (9, "r2"): 150_000,
        (9, "standard_errors"): 15_000,
        (3, "beta"): 400_000,
        (3, "r2"): 300_000,
        (3, "standard_errors"): 100_000,
        (16, "beta"): 30_000,
        (16, "r2"): 25_000,
        (16, "standard_errors"): 4_000,
    }
    env_name = (
        f"CPP_STREAM_RIDGE_RECOMPUTE_ROWS_K{features}_"
        f"{field.upper()}"
    )
    return int(os.environ.get(env_name, defaults[(features, field)]))


def _cases() -> list[Case]:
    cases: list[Case] = []
    for field in ("beta", "preds", "r2", "standard_errors"):
        rows = _case_rows(9, field)
        cases.extend(
            Case(9, field, rows, interval) for interval in INTERVALS
        )
    for features in (3, 16):
        for field in ("beta", "r2", "standard_errors"):
            rows = _case_rows(features, field)
            cases.extend(
                Case(features, field, rows, interval)
                for interval in SCALING_INTERVALS
            )
    return cases


def _build_data(
    root: Path,
    *,
    rows: int,
    instruments: int,
    features: int,
) -> dict[str, Path]:
    case_root = root / f"k{features}_rows{rows}"
    case_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7_000 + features * 101 + rows)
    coefficients = np.linspace(0.5, -0.2, features, dtype=np.float64)
    y_path = case_root / "y.npy"
    y = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.float64,
        shape=(rows, instruments),
    )
    chunk = 32_768
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        y[start:stop] = rng.normal(
            scale=0.05, size=(stop - start, instruments)
        )
    paths: dict[str, Path] = {"y": y_path}
    for feature in range(features):
        path = case_root / f"x{feature}.npy"
        x = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float64,
            shape=(rows, instruments),
        )
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            values = rng.normal(size=(stop - start, instruments))
            x[start:stop] = values
            y[start:stop] += coefficients[feature] * values
        x.flush()
        del x
        paths[f"x{feature}"] = path
    y.flush()
    del y
    return paths


def _formula(case: Case) -> str:
    features = ", ".join(f"x{index}" for index in range(case.features))
    model = (
        f"Ridge(cat({features}), y=y, hl=64, lambda_=0.1, "
        f"recompute_every={case.recompute_every})"
    )
    projection = {
        "beta": "get_beta",
        "preds": "get_preds",
        "r2": "get_r2",
        "standard_errors": "get_standard_errors",
    }[case.field]
    return f"{projection}({model})"


def _benchmark(
    case: Case,
    paths: dict[str, Path],
    output_root: Path,
) -> Result:
    formula = _formula(case)
    input_names = compile_ir(formula).input_names
    data = {name: paths[name] for name in input_names}
    runtime = compile_formula(
        formula,
        data,
        n_instruments=N,
        prefetch_rows=16,
    )
    output = output_root / (
        f"ridge_recompute_k{case.features}_{case.field}_"
        f"every{case.recompute_every}.bin"
    )
    for _ in range(WARMUPS):
        runtime.run(out_path=output, async_writeback_mb=0)
    rates: list[float] = []
    elapsed: list[float] = []
    for _ in range(RUNS):
        start = perf_counter()
        run = runtime.run(out_path=output, async_writeback_mb=0)
        elapsed.append(perf_counter() - start)
        rates.append(run.rows_per_second)
    values = np.memmap(
        output,
        mode="r",
        dtype=np.float64,
        shape=(case.rows, runtime.plan.output_row_width),
    )
    tail = np.asarray(values[-min(1024, case.rows):])
    finite = np.isfinite(tail)
    finite_fraction = float(finite.mean())
    checksum = float(np.nansum(tail))
    del values
    output.unlink(missing_ok=True)
    rates_mrows = [rate / 1e6 for rate in rates]
    return Result(
        features=case.features,
        field=case.field,
        rows=case.rows,
        instruments=N,
        recompute_every=case.recompute_every,
        rates_mrows=rates_mrows,
        median_mrows=median(rates_mrows),
        mean_mrows=mean(rates_mrows),
        best_mrows=max(rates_mrows),
        median_seconds=median(elapsed),
        speedup=1.0,
        checksum=checksum,
        finite_fraction=finite_fraction,
        output_row_width=runtime.plan.output_row_width,
        generated_cpp=str(runtime.generated_cpp),
    )


def _assign_speedups(results: list[Result]) -> None:
    baselines = {
        (result.features, result.field, result.rows): result.median_mrows
        for result in results
        if result.recompute_every == 1
    }
    for result in results:
        baseline = baselines[(result.features, result.field, result.rows)]
        result.speedup = result.median_mrows / baseline


def _parse_existing_benchmark(path: str | None) -> dict[str, dict[str, object]]:
    if not path or not Path(path).exists():
        return {}
    cases: dict[str, dict[str, object]] = {}
    current: str | None = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("case="):
            current = line.split("=", 1)[1].strip()
            cases[current] = {}
        elif current and line.startswith("median="):
            match = re.search(r"median=([0-9.]+)", line)
            if match:
                cases[current]["median_mrows"] = float(match.group(1))
        elif current and line.startswith("runs="):
            values = [
                float(value)
                for value in re.findall(r"([0-9.]+) M rows/s", line)
            ]
            cases[current]["rates_mrows"] = values
        elif current and line.startswith("checksum="):
            cases[current]["checksum"] = float(line.split("=", 1)[1])
    return cases


def _k1_comparison() -> list[dict[str, object]]:
    before = _parse_existing_benchmark(BASELINE_LOG)
    after = _parse_existing_benchmark(HEAD_LOG)
    rows: list[dict[str, object]] = []
    for case in sorted(before.keys() & after.keys()):
        before_median = float(before[case]["median_mrows"])
        after_median = float(after[case]["median_mrows"])
        rows.append(
            {
                "case": case,
                "before_median_mrows": before_median,
                "after_median_mrows": after_median,
                "ratio": after_median / before_median,
                "before_rates_mrows": before[case].get("rates_mrows", []),
                "after_rates_mrows": after[case].get("rates_mrows", []),
                "before_checksum": before[case].get("checksum"),
                "after_checksum": after[case].get("checksum"),
            }
        )
    return rows


def _fmt_runs(values: list[float]) -> str:
    return ", ".join(f"{value:.3f}" for value in values)


def _write_reports(
    results: list[Result],
    k1_comparison: list[dict[str, object]],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "git_sha_before_benchmark_commit": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "instruments": N,
        "warmups": WARMUPS,
        "runs": RUNS,
        "intervals": INTERVALS,
        "scaling_intervals": SCALING_INTERVALS,
    }
    payload = {
        "metadata": metadata,
        "k1_comparison": k1_comparison,
        "results": [asdict(result) for result in results],
    }
    json_path = REPORT_DIR / "RIDGE_RECOMPUTE_BENCHMARK.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Ridge periodic recomputation benchmark",
        "",
        (
            "`recompute_every=k` updates Ridge sufficient statistics on every "
            "bar but refreshes the solved beta and requested diagnostics on "
            "bars 0, k, 2k, ... . Between refreshes, projections use the last "
            "coherent solved snapshot."
        ),
        "",
        "## Environment",
        "",
        f"- CPU: `{metadata['cpu']}`",
        f"- Platform: `{metadata['platform']}`",
        f"- Python: `{metadata['python']}`",
        f"- Instruments per bar: **{N}**",
        f"- Measurements: **{WARMUPS} warmup + {RUNS} timed runs**",
        "",
    ]
    if k1_comparison:
        lines.extend(
            [
                "## Default-path regression check (`recompute_every=1`)",
                "",
                "| Existing benchmark case | Before M rows/s | After M rows/s | After / before |",
                "|---|---:|---:|---:|",
            ]
        )
        for item in k1_comparison:
            lines.append(
                f"| {item['case']} | "
                f"{item['before_median_mrows']:.3f} | "
                f"{item['after_median_mrows']:.3f} | "
                f"{item['ratio']:.3f}x |"
            )
        lines.append("")

    lines.extend(
        [
            "## Main matrix: 9 coefficients",
            "",
            "| Projection | Rows | Every | Median M rows/s | Speedup | Median seconds | Finite tail |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        if result.features != 9:
            continue
        lines.append(
            f"| {result.field} | {result.rows:,} | "
            f"{result.recompute_every} | {result.median_mrows:.3f} | "
            f"{result.speedup:.2f}x | {result.median_seconds:.6f} | "
            f"{result.finite_fraction:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Coefficient-count scaling",
            "",
            "| Coefficients | Projection | Rows | Every | Median M rows/s | Speedup |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        if result.features == 9:
            continue
        lines.append(
            f"| {result.features} | {result.field} | {result.rows:,} | "
            f"{result.recompute_every} | {result.median_mrows:.3f} | "
            f"{result.speedup:.2f}x |"
        )

    lines.extend(
        [
            "",
            "## Raw timed runs",
            "",
            "Rates are M rows/s. Checksums cover the last 1,024 output rows.",
            "",
            "| Coefficients | Projection | Every | Runs | Mean | Best | Checksum |",
            "|---:|---|---:|---|---:|---:|---:|",
        ]
    )
    for result in results:
        lines.append(
            f"| {result.features} | {result.field} | "
            f"{result.recompute_every} | {_fmt_runs(result.rates_mrows)} | "
            f"{result.mean_mrows:.3f} | {result.best_mrows:.3f} | "
            f"{result.checksum:.12g} |"
        )
    lines.append("")
    (REPORT_DIR / "RIDGE_RECOMPUTE_BENCHMARK.md").write_text(
        "\n".join(lines)
    )


def main() -> None:
    if not INTERVALS or INTERVALS[0] != 1 or 1 not in SCALING_INTERVALS:
        raise ValueError("benchmark intervals must include 1 as the baseline")
    cases = _cases()
    with tempfile.TemporaryDirectory(
        prefix="cpp_stream_ridge_recompute_"
    ) as temporary:
        root = Path(temporary)
        output_root = Path(OUTPUT_DIR) if OUTPUT_DIR else root
        output_root.mkdir(parents=True, exist_ok=True)
        data: dict[tuple[int, int], dict[str, Path]] = {}
        results: list[Result] = []
        print(
            f"instruments={N} warmups={WARMUPS} runs={RUNS} "
            f"cases={len(cases)}"
        )
        for index, case in enumerate(cases, 1):
            key = (case.features, case.rows)
            if key not in data:
                data[key] = _build_data(
                    root,
                    rows=case.rows,
                    instruments=N,
                    features=case.features,
                )
            print(
                f"[{index}/{len(cases)}] K={case.features} "
                f"field={case.field} rows={case.rows:,} "
                f"every={case.recompute_every}"
            )
            result = _benchmark(case, data[key], output_root)
            results.append(result)
            print(
                f"  median={result.median_mrows:.3f} M rows/s "
                f"runs={_fmt_runs(result.rates_mrows)}"
            )
    _assign_speedups(results)
    comparison = _k1_comparison()
    _write_reports(results, comparison)
    print(f"wrote {REPORT_DIR / 'RIDGE_RECOMPUTE_BENCHMARK.md'}")
    print(f"wrote {REPORT_DIR / 'RIDGE_RECOMPUTE_BENCHMARK.json'}")


if __name__ == "__main__":
    main()
