"""End-to-end strongly typed GP alpha search with timed cpp_stream execution.

The search uses Sharpe as individual fitness and a persistent nonnegative
rolling Ridge as a marginal-contribution screen.  Configuration is controlled
by the constants below or by the matching ``GP_*`` environment variables so
the same file can be used for full data and reproducible benchmark runs.
"""

from __future__ import annotations

import copy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import partial
import json
import math
import operator
import os
from pathlib import Path
import random
import shutil
import time

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import base, creator, gp, tools

from flows.gp import (
    GPConfig,
    GrammarPolicy,
    individual_to_expr,
    make_pset,
    make_toolbox,
)
from flows.load import InputData
from flows.riskminer.semantics import inputdata_alpha_terminal_metadata
from flows.riskmodel import roll_rets
from flows.utils import ewm_std, replace
from trading_dsl_engine.base.dsl import (
    Ridge,
    cat,
    ffill,
    get_beta,
    get_coefficient,
    get_residuals,
    purify,
    shift,
    var,
    where,
)
from trading_dsl_engine.cpp_stream import compile_formula


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_names(name: str) -> tuple[str, ...]:
    value = os.environ.get(name, "")
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _effective_cpu_count() -> int:
    """Return the process-usable CPU count, including cgroup quotas."""

    counts = [os.cpu_count() or 1]
    try:
        counts.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota_text, period_text = cpu_max.read_text().split()
        if quota_text != "max":
            quota = int(quota_text)
            period = int(period_text)
            if quota > 0 and period > 0:
                counts.append(max(1, math.ceil(quota / period)))
    except (OSError, ValueError):
        pass
    return max(1, min(counts))


def _available_memory_bytes() -> int | None:
    """Return usable memory, bounded by the active cgroup when available."""

    limits: list[int] = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                limits.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass

    memory_max = Path("/sys/fs/cgroup/memory.max")
    memory_current = Path("/sys/fs/cgroup/memory.current")
    try:
        maximum = memory_max.read_text().strip()
        if maximum != "max":
            remaining = int(maximum) - int(memory_current.read_text().strip())
            limits.append(max(0, remaining))
    except (OSError, ValueError):
        pass
    return min(limits) if limits else None


def _default_worker_count() -> int:
    cpus = _effective_cpu_count()
    available = _available_memory_bytes()
    if available is not None:
        # Concurrent C++ template compilation is the memory peak. Reserve about
        # 1.1 GiB per compiler process and leave one GiB for Python/mmaps.
        memory_workers = max(1, int(max(0, available - (1 << 30)) / (1.1 * (1 << 30))))
        cpus = min(cpus, memory_workers)
    return max(1, min(16, cpus))


def _select_gp_compiler() -> str:
    """Prefer Clang for GP translation units when the user did not choose CXX."""

    explicit = os.environ.get("GP_CXX")
    if explicit and explicit.lower() != "auto":
        os.environ["CXX"] = explicit
        return explicit
    if "CXX" in os.environ:
        return os.environ["CXX"]
    selected = "clang++" if shutil.which("clang++") else "g++"
    os.environ["CXX"] = selected
    return selected


GP_COMPILER = _select_gp_compiler()
os.environ.setdefault("TRADING_DSL_ENGINE_CPP_PCH", "1")
_GP_CLANG_LLD = (
    "clang" in Path(GP_COMPILER).name.lower()
    and shutil.which("ld.lld") is not None
)
if _GP_CLANG_LLD:
    # Full LTO is disproportionately expensive for many one-shot GP shared
    # objects and interacts poorly with Clang PCH consumption. ThinLTO retains
    # link-time optimization while allowing the reusable PCH and parallel link.
    # Explicit user toolchain flags always win.
    if "TRADING_DSL_ENGINE_CPP_LTO" not in os.environ:
        os.environ["TRADING_DSL_ENGINE_CPP_LTO"] = "0"
        os.environ.setdefault(
            "TRADING_DSL_ENGINE_CPP_EXTRA_FLAGS",
            "-flto=thin",
        )
    os.environ.setdefault(
        "TRADING_DSL_ENGINE_CPP_EXTRA_LINK_FLAGS",
        "-fuse-ld=lld",
    )


# Search controls. Depth is 1 for the first DEPTH_GROW_EVERY generations, then
# increases by one for each subsequent block of that many generations.
N_INSTRUMENTS = int(os.environ.get("GP_N_INSTRUMENTS", "9"))
ROWS = int(os.environ.get("GP_ROWS", "5000000"))
POPULATION_SIZE = int(os.environ.get("GP_POPULATION_SIZE", "64"))
GENERATIONS = int(os.environ.get("GP_GENERATIONS", "50"))
DEPTH_GROW_EVERY = int(os.environ.get("GP_DEPTH_GROW_EVERY", "5"))
ELITE_COUNT = int(os.environ.get("GP_ELITE_COUNT", "8"))
TOURNAMENT_SIZE = int(os.environ.get("GP_TOURNAMENT_SIZE", "3"))
CROSSOVER_PROB = float(os.environ.get("GP_CROSSOVER_PROB", "0.50"))
MUTATION_PROB = float(os.environ.get("GP_MUTATION_PROB", "0.40"))
IMMIGRANTS = int(os.environ.get("GP_IMMIGRANTS", "8"))
SEED = int(os.environ.get("GP_SEED", "42"))

# Fitness / execution controls. Candidate microbatches are independent native
# programs because terminal temporal reductions have one accumulator owner.
# More tasks than workers permit dynamic scheduling when formula costs differ.
ALPHA_PNL_HL = int(os.environ.get("GP_ALPHA_PNL_HL", str(1440 * 21)))
PREFETCH_ROWS = int(os.environ.get("GP_PREFETCH_ROWS", "16"))
THREADS = int(os.environ.get("GP_THREADS", "1"))
_DEFAULT_WORKERS = _default_worker_count()
FITNESS_WORKERS = int(
    os.environ.get(
        "GP_FITNESS_WORKERS",
        os.environ.get("GP_FITNESS_SHARDS", str(_DEFAULT_WORKERS)),
    )
)
# Preserve the old name in reports/environment overrides.
FITNESS_SHARDS = FITNESS_WORKERS
FITNESS_BATCH_SIZE = int(os.environ.get("GP_FITNESS_BATCH_SIZE", "8"))
FITNESS_TASKS_PER_WORKER = int(
    os.environ.get("GP_FITNESS_TASKS_PER_WORKER", "1")
)
FITNESS_PROBE_ROWS = int(
    os.environ.get(
        "GP_FITNESS_PROBE_ROWS",
        "100000" if ROWS >= 1_000_000 else "0",
    )
)
FITNESS_MAX_PROJECTED_BATCH_SECONDS = float(
    os.environ.get("GP_FITNESS_MAX_PROJECTED_BATCH_SECONDS", "45")
)
PARALLEL_DIAGNOSTIC = _env_bool("GP_PARALLEL_DIAGNOSTIC", False)
DIAGNOSTIC_CANDIDATES = int(os.environ.get("GP_DIAGNOSTIC_CANDIDATES", "16"))
INPUT_GLOB = os.environ.get(
    "GP_INPUT_GLOB",
    "/mnt/extra/qrt/data/aks_out3/*.npy",
)
OUTPUT_DIR = Path(os.environ.get("GP_OUTPUT_DIR", "/tmp/gp-alpha-search"))
SHOW_PLOT = _env_bool("GP_SHOW_PLOT", True)
FIELD_NAMES = _env_names("GP_FIELD_NAMES")
DISABLE_TENSORS = _env_bool("GP_DISABLE_TENSORS", False)

# A bounded run can extrapolate rather than consume an hour. Projection excludes
# the optional one-off serial-vs-sharded diagnostic and preprocessing.
PROJECTED_GENERATIONS = int(os.environ.get("GP_PROJECTED_GENERATIONS", "50"))
STOP_IF_PROJECTED_OVER_SECONDS = float(
    os.environ.get("GP_STOP_IF_PROJECTED_OVER_SECONDS", "0")
)
MIN_GENERATIONS_BEFORE_STOP = int(
    os.environ.get("GP_MIN_GENERATIONS_BEFORE_STOP", "1")
)
MAX_SEARCH_WALL_SECONDS = float(
    os.environ.get("GP_MAX_SEARCH_WALL_SECONDS", "0")
)

# Persistent Ridge pool. mean(abs(beta)) is only the marginal-contribution
# screen; individual evolutionary fitness remains candidate Sharpe.
POOL_SIZE = int(os.environ.get("GP_POOL_SIZE", "16"))
POOL_CANDIDATES_PER_GENERATION = int(
    os.environ.get("GP_POOL_CANDIDATES_PER_GENERATION", "8")
)
POOL_RIDGE_HL = int(os.environ.get("GP_POOL_RIDGE_HL", str(1440 * 5)))
POOL_RIDGE_LAMBDA = float(os.environ.get("GP_POOL_RIDGE_LAMBDA", "1e-3"))
POOL_RIDGE_RECOMPUTE_EVERY = 1
POOL_SCREEN_MODE = os.environ.get("GP_POOL_SCREEN_MODE", "orthogonal").strip().lower()
POOL_SCREEN_WORKERS = int(
    os.environ.get("GP_POOL_SCREEN_WORKERS", str(FITNESS_WORKERS))
)
POOL_SCREEN_BATCH_SIZE = int(
    os.environ.get("GP_POOL_SCREEN_BATCH_SIZE", "4")
)
POOL_SCREEN_TASKS_PER_WORKER = int(
    os.environ.get("GP_POOL_SCREEN_TASKS_PER_WORKER", "1")
)
POOL_ORTHOGONAL_BASIS_SIZE = int(
    os.environ.get(
        "GP_POOL_ORTHOGONAL_BASIS_SIZE",
        str(max(1, N_INSTRUMENTS - 1)),
    )
)
POOL_XS_LAMBDA = float(os.environ.get("GP_POOL_XS_LAMBDA", "1e-3"))
POOL_RESCORE_EXISTING = int(os.environ.get("GP_POOL_RESCORE_EXISTING", "8"))

# Group utilities remain enabled. Their GP key arguments are bounded Key
# terminals, so no generic default_group_capacity override is required.
GRAMMAR = GrammarPolicy()


def l1_norm(x):
    """Cross-sectionally normalize a signal with finite-value purification."""

    return purify(x / abs(x).sum(axis=-1))


def clean_returns_expr():
    """The exact cleaned return expression used by the alpha-PnL denominator."""

    roll_rets_value = var("roll_rets")
    return where(
        abs(roll_rets_value) <= 0.05,
        replace(roll_rets_value, 0, float("nan")),
        float("nan"),
    )


def ridge_weights_expr():
    """The invariant inverse-half-spread weights used by every pool screen."""

    hs = var("vw_halfspread_out0")
    return purify(1.0 / (hs * hs))


def precomputed_alpha_pnl(alpha):
    """Exact ``default_alpha_pnl`` using invariant materialized inputs.

    ``clean_rets``, its EWM standard deviation, and half-spread weights depend
    only on the input dataset. Keeping them in every evolved formula repeats
    both C++ template instantiation and native work. Materializing them once
    preserves numerical semantics while making candidate programs smaller.
    """

    weights = alpha / var("volatility")
    held = shift(
        ffill(
            where(
                var("is_tradable_out0"),
                weights,
                float("nan"),
            )
        ),
        1,
        1,
    )
    return held * var("clean_rets")


def depth_for_generation(generation: int) -> int:
    return 1 + (generation - 1) // DEPTH_GROW_EVERY


def _slice_sources(data, rows: int):
    if rows <= 0:
        return dict(data)
    sliced = {}
    for name, value in data.items():
        shape = tuple(getattr(value, "shape", ()))
        if not shape:
            sliced[name] = value
            continue
        if int(shape[0]) < rows:
            raise ValueError(
                f"source {name!r} has {int(shape[0]):,} rows; "
                f"requested {rows:,}"
            )
        sliced[name] = value[:rows]
    return sliced


def _run_summary(result, runtime, *, compile_seconds: float, wall_seconds: float):
    return {
        "compile_seconds": float(compile_seconds),
        "wall_seconds": float(wall_seconds),
        "native_seconds": float(result.seconds),
        "cpu_seconds": float(result.cpu_seconds),
        "average_busy_cores": float(result.average_busy_cores),
        "threads": int(result.threads),
        "available_cpus": int(result.available_cpus),
        "parallel_mode": str(result.parallel_mode),
        "parallel_plan_mode": str(runtime.parallel_plan.mode),
        "parallel_plan_reason": str(runtime.parallel_plan.reason),
        "work_score": int(runtime.parallel_plan.work_score),
    }


def _derived_source(name, formula, sources):
    out_path = OUTPUT_DIR / "derived" / f"{name}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    compile_started = time.perf_counter()
    runtime = compile_formula(
        formula,
        sources,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=PREFETCH_ROWS,
    )
    compile_seconds = time.perf_counter() - compile_started

    run_started = time.perf_counter()
    result = runtime.run(out_path=out_path, threads=THREADS)
    wall_seconds = time.perf_counter() - run_started

    metrics = _run_summary(
        result,
        runtime,
        compile_seconds=compile_seconds,
        wall_seconds=wall_seconds,
    )
    metrics.update({"name": name, "output_path": str(out_path)})
    print(
        f"derived={name} compile={compile_seconds:.3f}s "
        f"run={wall_seconds:.3f}s mode={metrics['parallel_mode']} "
        f"busy_cores={metrics['average_busy_cores']:.2f}"
    )
    return result.load(), metrics


def load_sources():
    """Load exactly ROWS rows and materialize invariant source transforms."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    data = InputData(fp=INPUT_GLOB, idx=None, nrows=None)
    raw_sources = data.get_data()
    if not raw_sources:
        raise FileNotFoundError(f"no input arrays matched {INPUT_GLOB!r}")
    sources = _slice_sources(raw_sources, ROWS)
    load_seconds = time.perf_counter() - load_started

    derived_metrics = {}

    def ensure(name, formula):
        nonlocal sources
        if name in sources:
            print(f"derived={name} reused precomputed input")
            return
        values, metrics = _derived_source(name, formula, sources)
        sources = sources | {name: values}
        derived_metrics[name] = metrics

    if "roll_rets" not in sources:
        ensure("roll_rets", roll_rets)
    else:
        print("derived=roll_rets reused precomputed input")

    # These three expressions are dataset invariants. They are deliberately
    # file-backed once so every GP candidate and pool screen reads them as
    # ordinary sources instead of rebuilding identical state/algebra.
    ensure("clean_rets", clean_returns_expr())
    ensure(
        "volatility",
        ewm_std(var("clean_rets"), span=ALPHA_PNL_HL),
    )
    ensure("ridge_weights", ridge_weights_expr())

    return sources, {
        "load_seconds": float(load_seconds),
        "derived": derived_metrics,
        "rows": int(ROWS),
        "n_instruments": int(N_INSTRUMENTS),
    }


def build_search_state():
    config_kwargs = {"grammar": GRAMMAR}
    if FIELD_NAMES:
        available = inputdata_alpha_terminal_metadata()
        missing = sorted(set(FIELD_NAMES) - set(available))
        if missing:
            raise KeyError(f"unknown GP field names: {missing}")
        config_kwargs["fields"] = {
            name: available[name] for name in FIELD_NAMES
        }
    if DISABLE_TENSORS:
        config_kwargs["tensor_fields"] = ()

    pset = make_pset(GPConfig(**config_kwargs))

    # Installs generation-only typed leaf witnesses for standard DEAP
    # generation. There is no reject/retry path.
    make_toolbox(pset, min_depth=1, max_depth=1)

    if not hasattr(creator, "GPAlphaFitness"):
        creator.create("GPAlphaFitness", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "GPAlphaIndividual"):
        creator.create(
            "GPAlphaIndividual",
            gp.PrimitiveTree,
            fitness=creator.GPAlphaFitness,
        )

    toolbox = base.Toolbox()
    toolbox.register("clone", copy.deepcopy)
    toolbox.register(
        "select",
        tools.selTournament,
        tournsize=TOURNAMENT_SIZE,
    )
    return pset, toolbox


def new_individual(pset, depth: int):
    nodes = gp.genHalfAndHalf(pset=pset, min_=1, max_=depth)
    return creator.GPAlphaIndividual(nodes)


def raw_alpha_expr(individual, pset):
    return individual_to_expr(individual, pset)


def alpha_expr(individual, pset):
    return l1_norm(raw_alpha_expr(individual, pset))


@dataclass(frozen=True, slots=True)
class _CandidateSpec:
    key: str
    alpha: object
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class _PoolScreenSpec:
    key: str
    individual: object
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class _BatchOutcome:
    batch: tuple[object, ...]
    values: np.ndarray | None
    metrics: dict[str, object]
    retry_as_smaller_batches: bool = False


def _individual_work_estimate(individual) -> float:
    """Cheap compile-time proxy used only to balance independent batches."""

    total = 1.0
    for node in individual:
        name = str(getattr(node, "name", "")).lower()
        weight = 1.0
        if "theilsen" in name:
            weight = 80.0
        elif any(
            token in name
            for token in ("rolling_kth", "rolling_quantile", "rolling_entropy")
        ):
            weight = 35.0
        elif any(token in name for token in ("groupby", "group_", "future_rbf")):
            weight = 20.0
        elif any(
            token in name
            for token in (
                "ridge",
                "regression",
                "einsum",
                "tensor",
                "bspline",
                "rbf",
            )
        ):
            weight = 10.0
        elif any(token in name for token in ("rolling", "ewm_", "ewm")):
            weight = 4.0
        elif any(token in name for token in ("xs_rank", "xs_sort", "quantile")):
            weight = 3.0
        total += weight
    height = float(getattr(individual, "height", 1))
    return total * (1.0 + 0.15 * height)


def _make_cost_balanced_batches(
    items,
    workers: int,
    *,
    max_batch: int,
    tasks_per_worker: int,
):
    """Largest-processing-time bin packing with bounded fusion width."""

    if not items:
        return []
    workers = max(1, min(int(workers), len(items)))
    max_batch = max(1, int(max_batch))
    task_target = max(
        math.ceil(len(items) / max_batch),
        min(len(items), workers * max(1, int(tasks_per_worker))),
    )
    task_count = min(len(items), task_target)
    bins: list[list[_CandidateSpec]] = [[] for _ in range(task_count)]
    loads = [0.0] * task_count
    for item in sorted(items, key=lambda value: value.estimated_cost, reverse=True):
        eligible = [
            index for index, values in enumerate(bins)
            if len(values) < max_batch
        ]
        index = min(eligible, key=lambda value: (loads[value], len(bins[value])))
        bins[index].append(item)
        loads[index] += item.estimated_cost
    ordered = sorted(
        zip(bins, loads),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [values for values, _ in ordered if values]


def _make_microbatches(
    items: list[_CandidateSpec],
    workers: int,
) -> list[list[_CandidateSpec]]:
    return _make_cost_balanced_batches(
        items,
        workers,
        max_batch=FITNESS_BATCH_SIZE,
        tasks_per_worker=FITNESS_TASKS_PER_WORKER,
    )


def _balanced_halves(batch):
    left = []
    right = []
    left_cost = 0.0
    right_cost = 0.0
    for item in sorted(batch, key=lambda value: value.estimated_cost, reverse=True):
        if left_cost <= right_cost:
            left.append(item)
            left_cost += item.estimated_cost
        else:
            right.append(item)
            right_cost += item.estimated_cost
    return left, right


def _fitness_batch(
    batch,
    sources,
    generation: int,
    task_id: int,
    label: str,
):
    specs = tuple(batch)
    pnls = [precomputed_alpha_pnl(spec.alpha) for spec in specs]
    pnl = (
        pnls[0].sum(axis=1)
        if len(pnls) == 1
        else cat(*pnls).sum(axis=1)
    )
    score = pnl.mean(axis=0) / pnl.std(axis=0)

    task_started = time.perf_counter()
    compile_started = time.perf_counter()
    runtime = compile_formula(
        score,
        sources,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=PREFETCH_ROWS,
    )
    compile_ended = time.perf_counter()
    compile_seconds = compile_ended - compile_started

    scratch = OUTPUT_DIR / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    stem = f"fitness_{label}_g{generation:03d}_t{task_id:03d}"
    probe_result = None
    probe_wall_seconds = 0.0
    projected_full_seconds = None
    probe_rows = min(max(0, FITNESS_PROBE_ROWS), ROWS)
    enforce_budget = not label.startswith("diagnostic")
    if 0 < probe_rows < ROWS:
        probe_path = scratch / f"{stem}_probe.npy"
        probe_started = time.perf_counter()
        probe_result = runtime.run(
            _slice_sources(sources, probe_rows),
            out_path=probe_path,
            threads=THREADS,
        )
        probe_wall_seconds = time.perf_counter() - probe_started
        probe_path.unlink(missing_ok=True)
        projected_full_seconds = (
            float(probe_result.seconds) * ROWS / probe_rows
        )
        if (
            enforce_budget
            and FITNESS_MAX_PROJECTED_BATCH_SECONDS > 0
            and projected_full_seconds > FITNESS_MAX_PROJECTED_BATCH_SECONDS
        ):
            base = _run_summary(
                probe_result,
                runtime,
                compile_seconds=compile_seconds,
                wall_seconds=probe_wall_seconds,
            )
            base.update(
                {
                    "candidate_count": len(specs),
                    "task_id": task_id,
                    "label": label,
                    "task_started_at": task_started,
                    "compile_started_at": compile_started,
                    "compile_ended_at": compile_ended,
                    "execution_started_at": probe_started,
                    "execution_ended_at": time.perf_counter(),
                    "estimated_cost": sum(item.estimated_cost for item in specs),
                    "probe_rows": probe_rows,
                    "probe_native_seconds": float(probe_result.seconds),
                    "probe_wall_seconds": probe_wall_seconds,
                    "projected_full_seconds": projected_full_seconds,
                    "execution_wall_seconds": probe_wall_seconds,
                    "total_native_seconds": float(probe_result.seconds),
                    "runtime_rejected": len(specs) == 1,
                    "probe_requested_split": len(specs) > 1,
                    "task_wall_seconds": time.perf_counter() - task_started,
                }
            )
            return _BatchOutcome(
                specs,
                (
                    np.full(len(specs), -np.inf, dtype=np.float64)
                    if len(specs) == 1
                    else None
                ),
                base,
                retry_as_smaller_batches=len(specs) > 1,
            )

    out_path = scratch / f"{stem}.npy"
    run_started = time.perf_counter()
    result = runtime.run(out_path=out_path, threads=THREADS)
    run_ended = time.perf_counter()
    run_wall_seconds = run_ended - run_started
    values = np.asarray(
        result.load(mmap_mode=None),
        dtype=np.float64,
    ).reshape(-1)
    out_path.unlink(missing_ok=True)

    if values.size != len(specs):
        raise RuntimeError(
            f"fitness returned {values.size} values for {len(specs)} candidates"
        )

    metrics = _run_summary(
        result,
        runtime,
        compile_seconds=compile_seconds,
        wall_seconds=run_wall_seconds,
    )
    probe_native = 0.0 if probe_result is None else float(probe_result.seconds)
    metrics.update(
        {
            "candidate_count": len(specs),
            "task_id": task_id,
            "label": label,
            "task_started_at": task_started,
            "compile_started_at": compile_started,
            "compile_ended_at": compile_ended,
            "execution_started_at": (
                probe_started if probe_result is not None else run_started
            ),
            "execution_ended_at": run_ended,
            "estimated_cost": sum(item.estimated_cost for item in specs),
            "probe_rows": probe_rows if probe_result is not None else 0,
            "probe_native_seconds": probe_native,
            "probe_wall_seconds": probe_wall_seconds,
            "projected_full_seconds": projected_full_seconds,
            "execution_wall_seconds": run_wall_seconds + probe_wall_seconds,
            "total_native_seconds": float(result.seconds) + probe_native,
            "runtime_rejected": False,
            "probe_requested_split": False,
            "task_wall_seconds": time.perf_counter() - task_started,
        }
    )
    return _BatchOutcome(specs, values, metrics)


def _evaluate_alpha_batches(
    batch,
    sources,
    clean_rets,
    generation: int,
    shards: int,
    label: str,
):
    del clean_rets  # The exact cleaned returns are now a materialized source.
    items = list(batch)
    workers = max(1, min(int(shards), len(items))) if items else 0
    chunks = _make_microbatches(items, workers) if items else []
    started = time.perf_counter()
    outcomes: list[_BatchOutcome] = []
    stages: list[dict[str, object]] = []
    task_sequence = 0

    if chunks:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {}

            def submit(chunk):
                nonlocal task_sequence
                current = task_sequence
                task_sequence += 1
                future = executor.submit(
                    _fitness_batch,
                    chunk,
                    sources,
                    generation,
                    current,
                    label,
                )
                pending[future] = tuple(chunk)

            for chunk in chunks:
                submit(chunk)

            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    original = pending.pop(future)
                    outcome = future.result()
                    stages.append(outcome.metrics)
                    if outcome.retry_as_smaller_batches:
                        left, right = _balanced_halves(original)
                        if not left or not right:
                            raise RuntimeError(
                                "probe split did not reduce a fitness batch"
                            )
                        submit(left)
                        submit(right)
                    else:
                        outcomes.append(outcome)

    wall_seconds = time.perf_counter() - started
    scores = {}
    for outcome in outcomes:
        assert outcome.values is not None
        for spec, value in zip(outcome.batch, outcome.values):
            scores[spec.key] = float(value) if np.isfinite(value) else -np.inf

    native_sum = sum(float(item["total_native_seconds"]) for item in stages)
    run_wall_sum = sum(float(item["execution_wall_seconds"]) for item in stages)
    cpu_sum = sum(float(item["cpu_seconds"]) for item in stages)
    return scores, {
        "wall_seconds": float(wall_seconds),
        "shards": workers,
        "candidate_count": len(items),
        "task_count": len(stages),
        "initial_task_count": len(chunks),
        "batch_sizes": [int(item["candidate_count"]) for item in stages],
        "compile_seconds_sum": float(
            sum(float(item["compile_seconds"]) for item in stages)
        ),
        "compile_seconds_max": float(
            max((float(item["compile_seconds"]) for item in stages), default=0.0)
        ),
        "compile_wall_seconds_union": _interval_union_seconds(
            stages, "compile_started_at", "compile_ended_at"
        ),
        "run_wall_seconds_sum": float(run_wall_sum),
        "execution_wall_seconds_union": _interval_union_seconds(
            stages, "execution_started_at", "execution_ended_at"
        ),
        "native_seconds_sum": float(native_sum),
        "cpu_seconds_sum": float(cpu_sum),
        "effective_native_concurrency": float(
            native_sum / wall_seconds if wall_seconds else 0.0
        ),
        "effective_cpu_concurrency": float(
            cpu_sum / wall_seconds if wall_seconds else 0.0
        ),
        "runtime_rejections": sum(bool(item["runtime_rejected"]) for item in stages),
        "probe_splits": sum(bool(item["probe_requested_split"]) for item in stages),
        "plans": sorted(
            {
                f"{item['parallel_plan_mode']}: {item['parallel_plan_reason']}"
                for item in stages
            }
        ),
        "stages": stages,
    }


def _interval_union_seconds(
    stages: list[dict[str, object]],
    start_key: str,
    end_key: str,
) -> float:
    intervals = sorted(
        (float(stage[start_key]), float(stage[end_key]))
        for stage in stages
        if start_key in stage and end_key in stage
    )
    total = 0.0
    current_start = None
    current_end = None
    for start, end in intervals:
        if end <= start:
            continue
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    if current_start is not None:
        total += current_end - current_start
    return total


def _empty_batch_metrics() -> dict[str, object]:
    return {
        "wall_seconds": 0.0,
        "steady_state_wall_seconds": 0.0,
        "shards": 0,
        "candidate_count": 0,
        "task_count": 0,
        "initial_task_count": 0,
        "batch_sizes": [],
        "compile_seconds_sum": 0.0,
        "compile_seconds_max": 0.0,
        "compile_wall_seconds_union": 0.0,
        "run_wall_seconds_sum": 0.0,
        "execution_wall_seconds_union": 0.0,
        "native_seconds_sum": 0.0,
        "cpu_seconds_sum": 0.0,
        "effective_native_concurrency": 0.0,
        "effective_cpu_concurrency": 0.0,
        "runtime_rejections": 0,
        "probe_splits": 0,
        "plans": [],
        "stages": [],
    }


def _merge_batch_metrics(
    parts: list[dict[str, object]],
    *,
    wall_seconds: float,
    candidate_count: int,
    shards: int,
) -> dict[str, object]:
    stages = [stage for part in parts for stage in part["stages"]]
    native_sum = sum(float(part["native_seconds_sum"]) for part in parts)
    cpu_sum = sum(float(part["cpu_seconds_sum"]) for part in parts)
    return {
        "wall_seconds": float(wall_seconds),
        "shards": int(shards),
        "candidate_count": int(candidate_count),
        "task_count": sum(int(part["task_count"]) for part in parts),
        "initial_task_count": sum(
            int(part["initial_task_count"]) for part in parts
        ),
        "batch_sizes": [
            int(size) for part in parts for size in part["batch_sizes"]
        ],
        "compile_seconds_sum": sum(
            float(part["compile_seconds_sum"]) for part in parts
        ),
        "compile_seconds_max": max(
            (float(part["compile_seconds_max"]) for part in parts),
            default=0.0,
        ),
        "compile_wall_seconds_union": _interval_union_seconds(
            stages, "compile_started_at", "compile_ended_at"
        ),
        "run_wall_seconds_sum": sum(
            float(part["run_wall_seconds_sum"]) for part in parts
        ),
        "execution_wall_seconds_union": _interval_union_seconds(
            stages, "execution_started_at", "execution_ended_at"
        ),
        "native_seconds_sum": native_sum,
        "cpu_seconds_sum": cpu_sum,
        "effective_native_concurrency": (
            native_sum / wall_seconds if wall_seconds else 0.0
        ),
        "effective_cpu_concurrency": (
            cpu_sum / wall_seconds if wall_seconds else 0.0
        ),
        "runtime_rejections": sum(
            int(part["runtime_rejections"]) for part in parts
        ),
        "probe_splits": sum(int(part["probe_splits"]) for part in parts),
        "plans": sorted(
            {plan for part in parts for plan in part["plans"]}
        ),
        "stages": stages,
    }


def _compare_score_maps(left, right):
    if set(left) != set(right):
        raise RuntimeError("serial and sharded fitness keys differ")
    for key, left_value in left.items():
        right_value = right[key]
        if left_value == right_value:
            continue
        if not np.isclose(
            left_value,
            right_value,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"serial/sharded fitness mismatch for {key}: "
                f"{left_value} versus {right_value}"
            )


def evaluate_individuals(
    individuals,
    pset,
    sources,
    clean_rets,
    generation: int,
    fitness_cache,
):
    """Evaluate unique invalid formulas in concurrent cpp_stream batches."""

    pending = [
        individual
        for individual in individuals
        if not individual.fitness.valid
    ]
    representatives = {}
    duplicate_groups = {}
    cached = 0
    for individual in pending:
        key = str(individual)
        if key in fitness_cache:
            individual.fitness.values = (fitness_cache[key],)
            cached += 1
            continue
        duplicate_groups.setdefault(key, []).append(individual)
        representatives.setdefault(key, individual)

    batch = [
        _CandidateSpec(
            key=key,
            alpha=alpha_expr(individual, pset),
            estimated_cost=_individual_work_estimate(individual),
        )
        for key, individual in representatives.items()
    ]

    diagnostic = None
    if (
        batch
        and PARALLEL_DIAGNOSTIC
        and generation == 1
        and FITNESS_SHARDS > 1
    ):
        count = max(1, min(DIAGNOSTIC_CANDIDATES, len(batch)))
        diagnostic_batch = batch[:count]
        remainder = batch[count:]

        serial_scores, serial_metrics = _evaluate_alpha_batches(
            diagnostic_batch,
            sources,
            clean_rets,
            generation,
            1,
            "diagnostic_serial",
        )
        sharded_scores, sharded_metrics = _evaluate_alpha_batches(
            diagnostic_batch,
            sources,
            clean_rets,
            generation,
            FITNESS_SHARDS,
            "diagnostic_sharded",
        )
        _compare_score_maps(serial_scores, sharded_scores)

        scores = dict(sharded_scores)
        if remainder:
            remainder_scores, remainder_metrics = _evaluate_alpha_batches(
                remainder,
                sources,
                clean_rets,
                generation,
                FITNESS_SHARDS,
                "search_remainder",
            )
            scores.update(remainder_scores)
        else:
            remainder_metrics = _empty_batch_metrics()


        steady_wall = (
            sharded_metrics["wall_seconds"]
            + remainder_metrics["wall_seconds"]
        )
        metrics = _merge_batch_metrics(
            [sharded_metrics, remainder_metrics],
            wall_seconds=steady_wall,
            candidate_count=len(batch),
            shards=FITNESS_SHARDS,
        )
        metrics["steady_state_wall_seconds"] = steady_wall
        metrics["wall_seconds"] = (
            float(serial_metrics["wall_seconds"]) + steady_wall
        )
        diagnostic = {
            "candidate_count": count,
            "serial_wall_seconds": serial_metrics["wall_seconds"],
            "sharded_wall_seconds": sharded_metrics["wall_seconds"],
            "speedup": (
                serial_metrics["wall_seconds"]
                / sharded_metrics["wall_seconds"]
                if sharded_metrics["wall_seconds"]
                else float("inf")
            ),
            "serial": serial_metrics,
            "sharded": sharded_metrics,
        }
    elif batch:
        scores, metrics = _evaluate_alpha_batches(
            batch,
            sources,
            clean_rets,
            generation,
            FITNESS_SHARDS,
            "search",
        )
        metrics["steady_state_wall_seconds"] = metrics["wall_seconds"]
    else:
        scores = {}
        metrics = _empty_batch_metrics()

    for key, group in duplicate_groups.items():
        score = scores[key]
        fitness_cache[key] = score
        for individual in group:
            individual.fitness.values = (score,)

    metrics.update(
        {
            "pending_count": len(pending),
            "unique_evaluated": len(batch),
            "cache_hits": cached,
            "duplicates_within_batch": (
                sum(len(group) for group in duplicate_groups.values())
                - len(duplicate_groups)
            ),
            "diagnostic": diagnostic,
        }
    )
    return metrics


def _full_ridge_contributions(
    individuals,
    pset,
    sources,
    clean_rets,
    generation: int,
):
    """Reference multivariate pool screen retained for A/B benchmarking."""

    del clean_rets
    normalized_alphas = [alpha_expr(individual, pset) for individual in individuals]
    volatility = var("volatility")
    ridge_alphas = [
        shift(alpha, 1, 1) * volatility
        for alpha in normalized_alphas
    ]
    regression = Ridge(
        *ridge_alphas,
        y=var("clean_rets"),
        weights=var("ridge_weights"),
        hl=float(POOL_RIDGE_HL),
        lambda_=POOL_RIDGE_LAMBDA,
        nonneg=True,
        recompute_every=POOL_RIDGE_RECOMPUTE_EVERY,
    )
    mean_abs_beta = abs(get_beta(regression)).mean(axis=0)

    task_started = time.perf_counter()
    compile_started = time.perf_counter()
    runtime = compile_formula(
        mean_abs_beta,
        sources,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=PREFETCH_ROWS,
    )
    compile_ended = time.perf_counter()
    compile_seconds = compile_ended - compile_started

    out_path = OUTPUT_DIR / "scratch" / f"ridge_full_g{generation:03d}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    result = runtime.run(out_path=out_path, threads=THREADS)
    run_ended = time.perf_counter()
    wall_seconds = run_ended - run_started
    values = np.asarray(result.load(mmap_mode=None), dtype=np.float64).reshape(-1)
    out_path.unlink(missing_ok=True)

    if values.size != len(individuals):
        raise RuntimeError(
            f"Ridge returned {values.size} coefficients for {len(individuals)} alphas"
        )

    metrics = _run_summary(
        result,
        runtime,
        compile_seconds=compile_seconds,
        wall_seconds=wall_seconds,
    )
    metrics.update(
        {
            "candidate_count": len(individuals),
            "task_started_at": task_started,
            "compile_started_at": compile_started,
            "compile_ended_at": compile_ended,
            "execution_started_at": run_started,
            "execution_ended_at": run_ended,
            "compile_seconds_max": compile_seconds,
            "compile_wall_seconds_union": compile_seconds,
            "execution_wall_seconds_union": wall_seconds,
            "task_count": 1,
            "basis_size": len(individuals),
            "screen_mode": "full_ridge",
            "runtime_rejections": 0,
            "stages": [],
        }
    )
    return (
        np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
        metrics,
    )


def _orthogonal_score_expressions(specs, basis, pset):
    """Build independent incremental-value scores with one shared pool design."""

    # Reuse the same Expr objects across every candidate in the batch. The IR
    # builder can then CSE even stateful pool subgraphs before lowering the
    # separate cross-sectional residual models.
    design = [alpha_expr(value, pset) for value in basis]
    scores = []
    for spec in specs:
        candidate = alpha_expr(spec.individual, pset)
        if design:
            xs_model = Ridge(
                *design,
                y=candidate,
                weights=1.0,
                hl=0.0,
                lambda_=POOL_XS_LAMBDA,
                nonneg=False,
                recompute_every=1,
            )
            candidate = purify(get_residuals(xs_model))

        # Orthogonalize before volatility scaling, then restore a stable L1
        # scale. Each candidate retains an independent K=1 temporal screen.
        feature = shift(l1_norm(candidate), 1, 1) * var("volatility")
        temporal = Ridge(
            feature,
            y=var("clean_rets"),
            weights=var("ridge_weights"),
            hl=float(POOL_RIDGE_HL),
            lambda_=POOL_RIDGE_LAMBDA,
            nonneg=True,
            recompute_every=POOL_RIDGE_RECOMPUTE_EVERY,
        )
        # get_coefficient(K=1) is scalar-shaped. Multiple final scalars can be
        # concatenated into one native program; cpp_stream broadcasts them over
        # the instrument row, and the loader below verifies and removes that
        # redundant lane dimension.
        scores.append(abs(get_coefficient(temporal, 0)).mean(axis=0))
    return scores[0] if len(scores) == 1 else cat(*scores)


def _pool_batch_values(result, candidate_count: int) -> np.ndarray:
    raw = np.asarray(result.load(mmap_mode=None), dtype=np.float64)
    if candidate_count == 1:
        values = raw.reshape(-1)
        if values.size != 1:
            raise RuntimeError(
                f"pool screen returned {values.size} values for one candidate"
            )
        return values

    if raw.size % candidate_count:
        raise RuntimeError(
            f"pool screen returned shape {raw.shape} for {candidate_count} candidates"
        )
    rows = raw.reshape(-1, candidate_count)
    if rows.shape[0] > 1 and not np.allclose(
        rows,
        rows[:1],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
    ):
        raise RuntimeError("broadcast pool scores differ across instrument lanes")
    return rows[0]


def _orthogonal_score_batch(
    batch,
    basis,
    pset,
    sources,
    generation: int,
    task_id: int,
):
    specs = tuple(batch)
    formula = _orthogonal_score_expressions(specs, basis, pset)
    task_started = time.perf_counter()
    compile_started = time.perf_counter()
    runtime = compile_formula(
        formula,
        sources,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=PREFETCH_ROWS,
    )
    compile_ended = time.perf_counter()
    compile_seconds = compile_ended - compile_started

    scratch = OUTPUT_DIR / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    stem = f"ridge_orthogonal_g{generation:03d}_t{task_id:03d}"
    probe_result = None
    probe_wall = 0.0
    projected = None
    probe_rows = min(max(0, FITNESS_PROBE_ROWS), ROWS)
    if 0 < probe_rows < ROWS:
        probe_path = scratch / f"{stem}_probe.npy"
        probe_started = time.perf_counter()
        probe_result = runtime.run(
            _slice_sources(sources, probe_rows),
            out_path=probe_path,
            threads=THREADS,
        )
        probe_ended = time.perf_counter()
        probe_wall = probe_ended - probe_started
        probe_path.unlink(missing_ok=True)
        projected = float(probe_result.seconds) * ROWS / probe_rows
        if (
            FITNESS_MAX_PROJECTED_BATCH_SECONDS > 0
            and projected > FITNESS_MAX_PROJECTED_BATCH_SECONDS
        ):
            metrics = _run_summary(
                probe_result,
                runtime,
                compile_seconds=compile_seconds,
                wall_seconds=probe_wall,
            )
            metrics.update(
                {
                    "candidate_count": len(specs),
                    "task_id": task_id,
                    "task_started_at": task_started,
                    "compile_started_at": compile_started,
                    "compile_ended_at": compile_ended,
                    "execution_started_at": probe_started,
                    "execution_ended_at": probe_ended,
                    "estimated_cost": sum(
                        item.estimated_cost for item in specs
                    ),
                    "probe_rows": probe_rows,
                    "probe_native_seconds": float(probe_result.seconds),
                    "execution_wall_seconds": probe_wall,
                    "total_native_seconds": float(probe_result.seconds),
                    "projected_full_seconds": projected,
                    "runtime_rejected": len(specs) == 1,
                    "probe_requested_split": len(specs) > 1,
                    "task_wall_seconds": time.perf_counter() - task_started,
                }
            )
            return _BatchOutcome(
                specs,
                (
                    np.zeros(1, dtype=np.float64)
                    if len(specs) == 1
                    else None
                ),
                metrics,
                retry_as_smaller_batches=len(specs) > 1,
            )

    out_path = scratch / f"{stem}.npy"
    run_started = time.perf_counter()
    result = runtime.run(out_path=out_path, threads=THREADS)
    run_ended = time.perf_counter()
    run_wall = run_ended - run_started
    values = _pool_batch_values(result, len(specs))
    out_path.unlink(missing_ok=True)

    metrics = _run_summary(
        result,
        runtime,
        compile_seconds=compile_seconds,
        wall_seconds=run_wall,
    )
    probe_native = 0.0 if probe_result is None else float(probe_result.seconds)
    metrics.update(
        {
            "candidate_count": len(specs),
            "task_id": task_id,
            "task_started_at": task_started,
            "compile_started_at": compile_started,
            "compile_ended_at": compile_ended,
            "execution_started_at": (
                probe_started if probe_result is not None else run_started
            ),
            "execution_ended_at": run_ended,
            "estimated_cost": sum(item.estimated_cost for item in specs),
            "probe_rows": probe_rows if probe_result is not None else 0,
            "probe_native_seconds": probe_native,
            "execution_wall_seconds": run_wall + probe_wall,
            "total_native_seconds": float(result.seconds) + probe_native,
            "projected_full_seconds": projected,
            "runtime_rejected": False,
            "probe_requested_split": False,
            "task_wall_seconds": time.perf_counter() - task_started,
        }
    )
    return _BatchOutcome(specs, values, metrics)


def _orthogonal_contributions(
    candidates,
    basis,
    pset,
    sources,
    generation: int,
):
    if not candidates:
        return {}, {
            "candidate_count": 0,
            "compile_seconds": 0.0,
            "compile_seconds_max": 0.0,
            "compile_wall_seconds_union": 0.0,
            "wall_seconds": 0.0,
            "execution_wall_seconds_union": 0.0,
            "native_seconds": 0.0,
            "cpu_seconds": 0.0,
            "average_busy_cores": 0.0,
            "threads": 0,
            "available_cpus": _effective_cpu_count(),
            "parallel_mode": "candidate_batches",
            "parallel_plan_mode": "outer",
            "parallel_plan_reason": "no candidates",
            "work_score": 0,
            "task_count": 0,
            "initial_task_count": 0,
            "batch_sizes": [],
            "basis_size": len(basis),
            "screen_mode": "orthogonal",
            "runtime_rejections": 0,
            "probe_splits": 0,
            "stages": [],
        }

    specs = [
        _PoolScreenSpec(
            key=key,
            individual=individual,
            estimated_cost=_individual_work_estimate(individual),
        )
        for key, individual in candidates
    ]
    workers = max(1, min(POOL_SCREEN_WORKERS, len(specs)))
    chunks = _make_cost_balanced_batches(
        specs,
        workers,
        max_batch=POOL_SCREEN_BATCH_SIZE,
        tasks_per_worker=POOL_SCREEN_TASKS_PER_WORKER,
    )
    started = time.perf_counter()
    outcomes: list[_BatchOutcome] = []
    stages: list[dict[str, object]] = []
    task_sequence = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}

        def submit(chunk):
            nonlocal task_sequence
            current = task_sequence
            task_sequence += 1
            future = executor.submit(
                _orthogonal_score_batch,
                chunk,
                basis,
                pset,
                sources,
                generation,
                current,
            )
            pending[future] = tuple(chunk)

        for chunk in chunks:
            submit(chunk)

        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                original = pending.pop(future)
                outcome = future.result()
                stages.append(outcome.metrics)
                if outcome.retry_as_smaller_batches:
                    left, right = _balanced_halves(original)
                    if not left or not right:
                        raise RuntimeError(
                            "probe split did not reduce a pool-screen batch"
                        )
                    submit(left)
                    submit(right)
                else:
                    outcomes.append(outcome)

    wall_seconds = time.perf_counter() - started
    values = {}
    for outcome in outcomes:
        assert outcome.values is not None
        for spec, value in zip(outcome.batch, outcome.values):
            values[spec.key] = float(value) if np.isfinite(value) else 0.0

    compile_sum = sum(float(item["compile_seconds"]) for item in stages)
    native_sum = sum(float(item["total_native_seconds"]) for item in stages)
    cpu_sum = sum(float(item["cpu_seconds"]) for item in stages)
    return values, {
        "candidate_count": len(specs),
        "compile_seconds": compile_sum,
        "compile_seconds_max": max(
            (float(item["compile_seconds"]) for item in stages),
            default=0.0,
        ),
        "compile_wall_seconds_union": _interval_union_seconds(
            stages,
            "compile_started_at",
            "compile_ended_at",
        ),
        "wall_seconds": wall_seconds,
        "execution_wall_seconds_union": _interval_union_seconds(
            stages,
            "execution_started_at",
            "execution_ended_at",
        ),
        "native_seconds": native_sum,
        "cpu_seconds": cpu_sum,
        "average_busy_cores": cpu_sum / wall_seconds if wall_seconds else 0.0,
        "threads": workers,
        "available_cpus": _effective_cpu_count(),
        "parallel_mode": "candidate_batches",
        "parallel_plan_mode": "outer",
        "parallel_plan_reason": (
            "cost-balanced batches of independent cross-sectional residual "
            "and K=1 temporal Ridge scores"
        ),
        "work_score": sum(int(item["work_score"]) for item in stages),
        "task_count": len(stages),
        "initial_task_count": len(chunks),
        "batch_sizes": [int(item["candidate_count"]) for item in stages],
        "basis_size": len(basis),
        "screen_mode": "orthogonal",
        "runtime_rejections": sum(
            bool(item["runtime_rejected"]) for item in stages
        ),
        "probe_splits": sum(
            bool(item["probe_requested_split"]) for item in stages
        ),
        "stages": stages,
    }

def _empty_pool_metrics(reason: str):
    return {
        "candidate_count": 0,
        "compile_seconds": 0.0,
        "compile_seconds_max": 0.0,
        "compile_wall_seconds_union": 0.0,
        "wall_seconds": 0.0,
        "execution_wall_seconds_union": 0.0,
        "native_seconds": 0.0,
        "cpu_seconds": 0.0,
        "average_busy_cores": 0.0,
        "threads": 0,
        "available_cpus": _effective_cpu_count(),
        "parallel_mode": "serial",
        "parallel_plan_mode": "serial",
        "parallel_plan_reason": reason,
        "work_score": 0,
        "task_count": 0,
        "basis_size": 0,
        "screen_mode": POOL_SCREEN_MODE,
        "runtime_rejections": 0,
        "stages": [],
    }


def update_pool(
    pool,
    pool_contribution,
    population,
    pset,
    sources,
    clean_rets,
    toolbox,
    generation: int,
):
    """Update the persistent pool with a bounded-rank incremental screen."""

    proposals = tools.selBest(
        population,
        min(POOL_CANDIDATES_PER_GENERATION, len(population)),
    )
    proposal_map = {}
    for individual in proposals:
        key = str(individual)
        if key not in pool:
            proposal_map.setdefault(key, individual)

    if POOL_SCREEN_MODE == "full_ridge":
        candidates = list(pool.values()) + list(proposal_map.values())
        unique = {str(individual): individual for individual in candidates}
        candidates = list(unique.values())
        if not candidates:
            return {}, {}, _empty_pool_metrics("no candidates")
        contribution, metrics = _full_ridge_contributions(
            candidates,
            pset,
            sources,
            clean_rets,
            generation,
        )
        order = np.argsort(contribution)[::-1][:POOL_SIZE]
        next_pool = {
            str(candidates[index]): toolbox.clone(candidates[index])
            for index in order
            if contribution[index] > 0.0
        }
        next_contribution = {
            str(candidates[index]): float(contribution[index])
            for index in order
            if contribution[index] > 0.0
        }
        return next_pool, next_contribution, metrics

    if POOL_SCREEN_MODE != "orthogonal":
        raise ValueError(
            "GP_POOL_SCREEN_MODE must be 'orthogonal' or 'full_ridge', got "
            f"{POOL_SCREEN_MODE!r}"
        )

    ranked_pool = sorted(
        pool.items(),
        key=lambda item: pool_contribution.get(item[0], 0.0),
        reverse=True,
    )
    basis_size = min(
        len(ranked_pool),
        max(0, POOL_ORTHOGONAL_BASIS_SIZE),
        max(0, N_INSTRUMENTS - 1),
    )
    basis_keys = {key for key, _ in ranked_pool[:basis_size]}
    basis = [individual for _, individual in ranked_pool[:basis_size]]

    # Re-score a bounded number of weakest non-basis incumbents under the same
    # current basis so stale entry scores do not freeze the dynamic pool slots.
    non_basis = [
        (key, individual)
        for key, individual in reversed(ranked_pool)
        if key not in basis_keys
    ][: max(0, POOL_RESCORE_EXISTING)]
    screen_map = {key: individual for key, individual in non_basis}
    screen_map.update(proposal_map)
    screened, metrics = _orthogonal_contributions(
        list(screen_map.items()),
        basis,
        pset,
        sources,
        generation,
    )

    combined_pool = {key: individual for key, individual in pool.items()}
    combined_pool.update(proposal_map)
    combined_scores = {
        key: float(pool_contribution.get(key, 0.0))
        for key in combined_pool
    }
    combined_scores.update(screened)
    order = sorted(
        combined_pool,
        key=lambda key: combined_scores.get(key, 0.0),
        reverse=True,
    )[:POOL_SIZE]
    next_pool = {
        key: toolbox.clone(combined_pool[key])
        for key in order
        if combined_scores.get(key, 0.0) > 0.0
    }
    next_contribution = {
        key: float(combined_scores[key])
        for key in order
        if combined_scores.get(key, 0.0) > 0.0
    }
    metrics.update(
        {
            "proposal_count": len(proposal_map),
            "rescored_existing": len(non_basis),
            "basis_keys": sorted(basis_keys),
        }
    )
    return next_pool, next_contribution, metrics


def vary(population, pset, toolbox, next_depth: int):
    elites = [
        toolbox.clone(x)
        for x in tools.selBest(population, ELITE_COUNT)
    ]
    child_count = POPULATION_SIZE - ELITE_COUNT - IMMIGRANTS
    if child_count < 0:
        raise ValueError(
            "POPULATION_SIZE must be at least ELITE_COUNT + IMMIGRANTS"
        )
    children = [
        toolbox.clone(x)
        for x in toolbox.select(population, child_count)
    ]

    mate = gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=next_depth,
    )(gp.cxOnePoint)
    mutation_expr = partial(
        gp.genFull,
        min_=0,
        max_=min(2, next_depth),
    )
    mutate = gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=next_depth,
    )(
        partial(
            gp.mutUniform,
            expr=mutation_expr,
            pset=pset,
        )
    )

    for index in range(1, len(children), 2):
        if random.random() < CROSSOVER_PROB:
            left, right = mate(
                children[index - 1],
                children[index],
            )
            children[index - 1], children[index] = left, right
            if left.fitness.valid:
                del left.fitness.values
            if right.fitness.valid:
                del right.fitness.values

    for index, child in enumerate(children):
        if random.random() < MUTATION_PROB:
            (child,) = mutate(child)
            children[index] = child
            if child.fitness.valid:
                del child.fitness.values

    immigrants = [
        new_individual(pset, next_depth)
        for _ in range(IMMIGRANTS)
    ]
    return elites + children + immigrants


def _settings():
    return {
        "n_instruments": N_INSTRUMENTS,
        "rows": ROWS,
        "population_size": POPULATION_SIZE,
        "generations_requested": GENERATIONS,
        "projected_generations": PROJECTED_GENERATIONS,
        "depth_grow_every": DEPTH_GROW_EVERY,
        "elite_count": ELITE_COUNT,
        "tournament_size": TOURNAMENT_SIZE,
        "crossover_probability": CROSSOVER_PROB,
        "mutation_probability": MUTATION_PROB,
        "immigrants": IMMIGRANTS,
        "seed": SEED,
        "alpha_pnl_span": ALPHA_PNL_HL,
        "prefetch_rows": PREFETCH_ROWS,
        "native_threads_per_task": THREADS,
        "effective_cpus": _effective_cpu_count(),
        "available_memory_bytes": _available_memory_bytes(),
        "compiler": GP_COMPILER,
        "lto": _env_bool("TRADING_DSL_ENGINE_CPP_LTO", False),
        "extra_compile_flags": os.environ.get(
            "TRADING_DSL_ENGINE_CPP_EXTRA_FLAGS", ""
        ),
        "extra_link_flags": os.environ.get(
            "TRADING_DSL_ENGINE_CPP_EXTRA_LINK_FLAGS", ""
        ),
        "precompiled_header": _env_bool(
            "TRADING_DSL_ENGINE_CPP_PCH", True
        ),
        "fitness_workers": FITNESS_WORKERS,
        "fitness_batch_size": FITNESS_BATCH_SIZE,
        "fitness_tasks_per_worker": FITNESS_TASKS_PER_WORKER,
        "fitness_probe_rows": FITNESS_PROBE_ROWS,
        "fitness_max_projected_batch_seconds": (
            FITNESS_MAX_PROJECTED_BATCH_SECONDS
        ),
        "parallel_diagnostic": PARALLEL_DIAGNOSTIC,
        "diagnostic_candidates": DIAGNOSTIC_CANDIDATES,
        "field_names": list(FIELD_NAMES),
        "disable_tensors": DISABLE_TENSORS,
        "materialized_invariants": [
            "clean_rets",
            "volatility",
            "ridge_weights",
        ],
        "pool_size": POOL_SIZE,
        "pool_candidates_per_generation": (
            POOL_CANDIDATES_PER_GENERATION
        ),
        "pool_screen_mode": POOL_SCREEN_MODE,
        "pool_screen_workers": POOL_SCREEN_WORKERS,
        "pool_screen_batch_size": POOL_SCREEN_BATCH_SIZE,
        "pool_screen_tasks_per_worker": POOL_SCREEN_TASKS_PER_WORKER,
        "pool_orthogonal_basis_size": POOL_ORTHOGONAL_BASIS_SIZE,
        "pool_xs_lambda": POOL_XS_LAMBDA,
        "pool_rescore_existing": POOL_RESCORE_EXISTING,
        "pool_ridge_span": POOL_RIDGE_HL,
        "pool_ridge_lambda": POOL_RIDGE_LAMBDA,
        "pool_ridge_recompute_every": POOL_RIDGE_RECOMPUTE_EVERY,
        "pool_ridge_nonnegative": True,
        "ridge_weights": "materialized purify(1 / hs**2)",
        "ridge_feature": (
            "shift(l1_norm(alpha), 1, 1) "
            "* materialized ewm_std(clean_rets)"
        ),
        "stop_if_projected_over_seconds": (
            STOP_IF_PROJECTED_OVER_SECONDS
        ),
        "max_search_wall_seconds": MAX_SEARCH_WALL_SECONDS,
    }


def _write_outputs(history, summary):
    frame = pd.DataFrame(history)
    csv_path = OUTPUT_DIR / "gp_search_history.csv"
    json_path = OUTPUT_DIR / "gp_search_summary.json"
    plot_path = OUTPUT_DIR / "gp_search_history.png"

    frame.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    if not frame.empty:
        plt.figure(figsize=(9, 5))
        plt.plot(
            frame["generation"],
            frame["best_sharpe"],
            label="best Sharpe",
        )
        plt.plot(
            frame["generation"],
            frame["mean_sharpe"],
            label="mean Sharpe",
        )
        plt.plot(
            frame["generation"],
            frame["median_sharpe"],
            label="median Sharpe",
        )
        plt.xlabel("Generation")
        plt.ylabel("Fitness (Sharpe)")
        plt.title(
            f"Strongly typed GP fitness: "
            f"{ROWS:,} rows × {N_INSTRUMENTS}"
        )
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        if SHOW_PLOT:
            plt.show()
        plt.close()

    return csv_path, json_path, plot_path


def main():
    if GENERATIONS <= 0:
        raise ValueError("GP_GENERATIONS must be positive")
    if POPULATION_SIZE <= 0:
        raise ValueError("GP_POPULATION_SIZE must be positive")
    if FITNESS_WORKERS <= 0:
        raise ValueError("GP_FITNESS_WORKERS must be positive")
    if FITNESS_BATCH_SIZE <= 0:
        raise ValueError("GP_FITNESS_BATCH_SIZE must be positive")
    if FITNESS_TASKS_PER_WORKER <= 0:
        raise ValueError("GP_FITNESS_TASKS_PER_WORKER must be positive")
    if FITNESS_PROBE_ROWS < 0:
        raise ValueError("GP_FITNESS_PROBE_ROWS must be nonnegative")
    if POOL_SCREEN_WORKERS <= 0:
        raise ValueError("GP_POOL_SCREEN_WORKERS must be positive")
    if POOL_SCREEN_BATCH_SIZE <= 0:
        raise ValueError("GP_POOL_SCREEN_BATCH_SIZE must be positive")
    if POOL_SCREEN_TASKS_PER_WORKER <= 0:
        raise ValueError("GP_POOL_SCREEN_TASKS_PER_WORKER must be positive")
    if POOL_ORTHOGONAL_BASIS_SIZE < 0:
        raise ValueError("GP_POOL_ORTHOGONAL_BASIS_SIZE must be nonnegative")
    if POOL_RESCORE_EXISTING < 0:
        raise ValueError("GP_POOL_RESCORE_EXISTING must be nonnegative")
    if POOL_SCREEN_MODE not in {"orthogonal", "full_ridge"}:
        raise ValueError(
            "GP_POOL_SCREEN_MODE must be 'orthogonal' or 'full_ridge'"
        )
    if POOL_RIDGE_RECOMPUTE_EVERY != 1:
        raise AssertionError("Ridge recompute_every must remain 1")

    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"compiler={GP_COMPILER} pch="
        f"{_env_bool('TRADING_DSL_ENGINE_CPP_PCH', True)} "
        f"lto={_env_bool('TRADING_DSL_ENGINE_CPP_LTO', False)} "
        f"compile_flags={os.environ.get('TRADING_DSL_ENGINE_CPP_EXTRA_FLAGS', '')!r} "
        f"link_flags={os.environ.get('TRADING_DSL_ENGINE_CPP_EXTRA_LINK_FLAGS', '')!r} "
        f"effective_cpus={_effective_cpu_count()} "
        f"fitness_workers={FITNESS_WORKERS} "
        f"pool_workers={POOL_SCREEN_WORKERS} "
        f"fitness_batch_size={FITNESS_BATCH_SIZE}"
    )

    total_started = time.perf_counter()
    sources, preprocessing = load_sources()
    clean_rets = var("clean_rets")
    pset, toolbox = build_search_state()

    population = [
        new_individual(pset, 1)
        for _ in range(POPULATION_SIZE)
    ]
    hall_of_fame = tools.HallOfFame(20)
    pool = {}
    pool_contribution = {}
    fitness_cache = {}
    history = []
    stop_reason = None
    search_started = time.perf_counter()
    steady_state_cumulative = 0.0

    summary = {
        "settings": _settings(),
        "preprocessing": preprocessing,
        "operator_family_count": len(
            getattr(pset, "gp_operator_families", ())
        ),
        "history": history,
        "stop_reason": None,
    }

    for generation in range(1, GENERATIONS + 1):
        generation_started = time.perf_counter()
        depth = depth_for_generation(generation)

        fitness_metrics = evaluate_individuals(
            population,
            pset,
            sources,
            clean_rets,
            generation,
            fitness_cache,
        )
        hall_of_fame.update(population)
        pool, pool_contribution, ridge_metrics = update_pool(
            pool,
            pool_contribution,
            population,
            pset,
            sources,
            clean_rets,
            toolbox,
            generation,
        )

        generation_wall = time.perf_counter() - generation_started
        diagnostic_extra = max(
            0.0,
            fitness_metrics["wall_seconds"]
            - fitness_metrics["steady_state_wall_seconds"],
        )
        steady_generation_wall = max(
            0.0,
            generation_wall - diagnostic_extra,
        )
        steady_state_cumulative += steady_generation_wall
        cumulative_search = time.perf_counter() - search_started
        projected_total = (
            steady_state_cumulative
            / generation
            * PROJECTED_GENERATIONS
        )

        fitness = np.array(
            [x.fitness.values[0] for x in population],
            dtype=np.float64,
        )
        finite = fitness[np.isfinite(fitness)]
        generation_compile_wall = (
            float(fitness_metrics.get("compile_wall_seconds_union", 0.0))
            + float(ridge_metrics.get("compile_wall_seconds_union", 0.0))
        )
        generation_compile_work = (
            float(fitness_metrics.get("compile_seconds_sum", 0.0))
            + float(ridge_metrics.get("compile_seconds", 0.0))
        )
        row = {
            "generation": generation,
            "max_depth": depth,
            "best_sharpe": (
                float(np.max(fitness))
                if fitness.size
                else float("nan")
            ),
            "mean_sharpe": (
                float(np.mean(finite))
                if finite.size
                else float("nan")
            ),
            "median_sharpe": (
                float(np.median(finite))
                if finite.size
                else float("nan")
            ),
            "pool_size": len(pool),
            "fitness_pending": fitness_metrics["pending_count"],
            "fitness_unique_evaluated": (
                fitness_metrics["unique_evaluated"]
            ),
            "fitness_cache_hits": fitness_metrics["cache_hits"],
            "fitness_workers": fitness_metrics["shards"],
            "fitness_tasks": fitness_metrics["task_count"],
            "fitness_initial_tasks": fitness_metrics["initial_task_count"],
            "fitness_batch_sizes": json.dumps(fitness_metrics["batch_sizes"]),
            "fitness_wall_seconds": fitness_metrics["wall_seconds"],
            "fitness_steady_state_wall_seconds": (
                fitness_metrics["steady_state_wall_seconds"]
            ),
            "fitness_compile_seconds_sum": (
                fitness_metrics["compile_seconds_sum"]
            ),
            "fitness_compile_seconds_max": (
                fitness_metrics["compile_seconds_max"]
            ),
            "fitness_compile_wall_seconds": (
                fitness_metrics["compile_wall_seconds_union"]
            ),
            "fitness_run_wall_seconds_sum": (
                fitness_metrics["run_wall_seconds_sum"]
            ),
            "fitness_execution_wall_seconds": (
                fitness_metrics["execution_wall_seconds_union"]
            ),
            "fitness_runtime_rejections": (
                fitness_metrics["runtime_rejections"]
            ),
            "fitness_probe_splits": fitness_metrics["probe_splits"],
            "fitness_effective_native_concurrency": (
                fitness_metrics["effective_native_concurrency"]
            ),
            "fitness_effective_cpu_concurrency": (
                fitness_metrics["effective_cpu_concurrency"]
            ),
            "fitness_parallel_plans": " | ".join(
                fitness_metrics["plans"]
            ),
            "ridge_candidates": ridge_metrics["candidate_count"],
            "ridge_tasks": ridge_metrics.get("task_count", 1),
            "ridge_basis_size": ridge_metrics.get("basis_size", 0),
            "ridge_screen_mode": ridge_metrics.get(
                "screen_mode", POOL_SCREEN_MODE
            ),
            "ridge_runtime_rejections": ridge_metrics.get(
                "runtime_rejections", 0
            ),
            "ridge_compile_seconds": ridge_metrics["compile_seconds"],
            "ridge_compile_seconds_max": ridge_metrics.get(
                "compile_seconds_max", ridge_metrics["compile_seconds"]
            ),
            "ridge_compile_wall_seconds": ridge_metrics.get(
                "compile_wall_seconds_union", ridge_metrics["compile_seconds"]
            ),
            "ridge_wall_seconds": ridge_metrics["wall_seconds"],
            "ridge_execution_wall_seconds": ridge_metrics.get(
                "execution_wall_seconds_union", ridge_metrics["wall_seconds"]
            ),
            "ridge_native_seconds": ridge_metrics["native_seconds"],
            "ridge_cpu_seconds": ridge_metrics["cpu_seconds"],
            "ridge_average_busy_cores": (
                ridge_metrics["average_busy_cores"]
            ),
            "ridge_parallel_mode": ridge_metrics["parallel_mode"],
            "ridge_parallel_plan": (
                f"{ridge_metrics['parallel_plan_mode']}: "
                f"{ridge_metrics['parallel_plan_reason']}"
            ),
            "generation_compile_work_seconds": generation_compile_work,
            "generation_compile_wall_seconds": generation_compile_wall,
            "generation_compile_wall_percent": (
                100.0 * generation_compile_wall / steady_generation_wall
                if steady_generation_wall
                else 0.0
            ),
            "generation_noncompile_wall_seconds": max(
                0.0,
                steady_generation_wall - generation_compile_wall,
            ),
            "generation_wall_seconds": generation_wall,
            "generation_steady_state_seconds": steady_generation_wall,
            "cumulative_search_seconds": cumulative_search,
            "projected_50_generation_seconds": projected_total,
            "projected_generation_seconds": projected_total,
        }
        history.append(row)

        diagnostic = fitness_metrics.get("diagnostic")
        if diagnostic is not None:
            print(
                f"parallel_diagnostic candidates="
                f"{diagnostic['candidate_count']} "
                f"serial={diagnostic['serial_wall_seconds']:.3f}s "
                f"sharded={diagnostic['sharded_wall_seconds']:.3f}s "
                f"speedup={diagnostic['speedup']:.3f}x"
            )

        print(
            f"generation={generation:3d} depth={depth:2d} "
            f"best={row['best_sharpe']:9.5f} "
            f"mean={row['mean_sharpe']:9.5f} "
            f"pool={len(pool):2d} "
            f"fitness={row['fitness_steady_state_wall_seconds']:.2f}s "
            f"screen={row['ridge_wall_seconds']:.2f}s "
            f"compile={row['generation_compile_wall_percent']:.1f}% "
            f"generation={steady_generation_wall:.2f}s "
            f"projected_{PROJECTED_GENERATIONS}="
            f"{projected_total / 60.0:.2f}min"
        )

        summary["history"] = history
        if diagnostic is not None:
            summary["parallel_diagnostic"] = diagnostic
        summary["latest_projection_seconds"] = projected_total
        summary["actual_search_seconds"] = cumulative_search
        _write_outputs(history, summary)

        if (
            MAX_SEARCH_WALL_SECONDS > 0
            and cumulative_search >= MAX_SEARCH_WALL_SECONDS
        ):
            stop_reason = (
                f"search wall {cumulative_search:.3f}s reached "
                f"GP_MAX_SEARCH_WALL_SECONDS="
                f"{MAX_SEARCH_WALL_SECONDS:.3f}s"
            )
        elif (
            STOP_IF_PROJECTED_OVER_SECONDS > 0
            and generation >= MIN_GENERATIONS_BEFORE_STOP
            and projected_total > STOP_IF_PROJECTED_OVER_SECONDS
        ):
            stop_reason = (
                f"steady-state projection {projected_total:.3f}s for "
                f"{PROJECTED_GENERATIONS} generations exceeds "
                f"{STOP_IF_PROJECTED_OVER_SECONDS:.3f}s"
            )

        if stop_reason is not None:
            print(f"stopping: {stop_reason}")
            break

        if generation < GENERATIONS:
            population = vary(
                population,
                pset,
                toolbox,
                depth_for_generation(generation + 1),
            )

    print("\nBest formulas by Sharpe:")
    for rank, individual in enumerate(hall_of_fame[:10], 1):
        print(
            f"{rank:2d}  "
            f"sharpe={individual.fitness.values[0]:9.5f}  "
            f"{individual}"
        )

    print("\nPersistent Ridge pool:")
    for rank, (text, individual) in enumerate(
        sorted(
            pool.items(),
            key=lambda item: pool_contribution[item[0]],
            reverse=True,
        ),
        1,
    ):
        print(
            f"{rank:2d}  "
            f"mean_abs_beta={pool_contribution[text]:.8g}  "
            f"sharpe={individual.fitness.values[0]:9.5f}  "
            f"{text}"
        )

    summary.update(
        {
            "history": history,
            "stop_reason": stop_reason,
            "completed_generations": len(history),
            "actual_search_seconds": (
                time.perf_counter() - search_started
            ),
            "total_wall_seconds": time.perf_counter() - total_started,
            "best_formulas": [
                {
                    "rank": rank,
                    "fitness": float(individual.fitness.values[0]),
                    "formula": str(individual),
                }
                for rank, individual in enumerate(
                    hall_of_fame[:10],
                    1,
                )
            ],
            "ridge_pool": [
                {
                    "rank": rank,
                    "mean_abs_beta": (
                        pool_contribution[text]
                    ),
                    "fitness": float(
                        individual.fitness.values[0]
                    ),
                    "formula": text,
                }
                for rank, (text, individual) in enumerate(
                    sorted(
                        pool.items(),
                        key=lambda item: pool_contribution[item[0]],
                        reverse=True,
                    ),
                    1,
                )
            ],
        }
    )
    csv_path, json_path, plot_path = _write_outputs(history, summary)
    print(f"\nhistory_csv={csv_path}")
    print(f"summary_json={json_path}")
    print(f"fitness_plot={plot_path}")


if __name__ == "__main__":
    main()
