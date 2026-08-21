"""Fast strongly typed GP alpha search on the cpp_stream backend.

The expensive paths are deliberately organized around two primitives:

* ``compile_formula([f0, f1, ...])`` builds one CSE'd native program for related
  outputs, including heterogeneous row/final outputs.
* ``run_many(...)`` executes independent final-reduction programs on one native
  C++ task pool.  Python does not create worker threads in the hot search loop.

Set ``GP_WALK_FORWARD_FOLDS`` above zero to enable anchored walk-forward
validation.  Every fold trains from row zero and reserves its final window for a
simple two-Sharpe z test.  The last fold always ends at the last loaded row.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from functools import partial
import json
import math
import operator
import os
from pathlib import Path
import random
from statistics import NormalDist
import time
from typing import Any

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
from flows.riskminer.semantics import (
    gp_alpha_search_terminal_metadata,
    inputdata_alpha_terminal_metadata,
)
from flows.riskmodel import roll_rets
from flows.utils import ewm_std, replace
from trading_dsl_engine.base.dsl import (
    Ridge,
    cat,
    einsum,
    fillna,
    ffill,
    get_beta,
    purify,
    shift,
    var,
    where,
)
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.cpp_stream import compile_formula, run_many


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_names(name: str) -> tuple[str, ...]:
    value = os.environ.get(name, "")
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _env_first(names: tuple[str, ...], default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# Search controls.
N_INSTRUMENTS = int(os.environ.get("GP_N_INSTRUMENTS", "9"))
ROWS = int(os.environ.get("GP_ROWS", "1000000"))
POPULATION_SIZE = int(os.environ.get("GP_POPULATION_SIZE", "64"))
GENERATIONS = int(os.environ.get("GP_GENERATIONS", "50"))
DEPTH_GROW_EVERY = int(os.environ.get("GP_DEPTH_GROW_EVERY", "5"))
ELITE_COUNT = int(os.environ.get("GP_ELITE_COUNT", "8"))
TOURNAMENT_SIZE = int(os.environ.get("GP_TOURNAMENT_SIZE", "3"))
CROSSOVER_PROB = float(os.environ.get("GP_CROSSOVER_PROB", "0.20"))
MUTATION_PROB = float(os.environ.get("GP_MUTATION_PROB", "0.80"))
IMMIGRANTS = int(os.environ.get("GP_IMMIGRANTS", "8"))
SEED = int(os.environ.get("GP_SEED", "40"))

# Fitness and native execution controls.
LAG = int(os.environ.get("GP_ALPHA_LAG", "1"))
ALPHA_PNL_HL = int(os.environ.get("GP_ALPHA_PNL_HL", str(1440 * 21)))
PREFETCH_ROWS = int(os.environ.get("GP_PREFETCH_ROWS", "16"))
# Zero opts into the cpp_stream planner for safe row/lane parallel graphs.
THREADS = int(os.environ.get("GP_THREADS", "0"))
NATIVE_WORKERS = int(
    _env_first(
        ("GP_NATIVE_WORKERS", "GP_FITNESS_SHARDS"),
        "0",
    )
)
FITNESS_BATCH_SIZE = int(os.environ.get("GP_FITNESS_BATCH_SIZE", "8"))
FITNESS_TASKS_PER_WORKER = int(
    os.environ.get("GP_FITNESS_TASKS_PER_WORKER", "2")
)
PIN_NATIVE_WORKERS = _env_bool("GP_PIN_NATIVE_WORKERS", False)
PARALLEL_DIAGNOSTIC = _env_bool("GP_PARALLEL_DIAGNOSTIC", False)
DIAGNOSTIC_CANDIDATES = int(os.environ.get("GP_DIAGNOSTIC_CANDIDATES", "16"))
INPUT_GLOB = os.environ.get(
    "GP_INPUT_GLOB",
    "/mnt/extra/qrt/data/aks_out3/*.npy",
)
OUTPUT_DIR = Path(os.environ.get("GP_OUTPUT_DIR", "/tmp/gp-alpha-search"))
FIELD_NAMES = _env_names("GP_FIELD_NAMES")
DISABLE_TENSORS = _env_bool("GP_DISABLE_TENSORS", False)

# Anchored walk-forward controls.  FOLDS=0 preserves full-sample search.
WALK_FORWARD_FOLDS = int(os.environ.get("GP_WALK_FORWARD_FOLDS", "0"))
WALK_FORWARD_VALIDATION_FRACTION = float(
    _env_first(
        (
            "GP_WALK_FORWARD_VALIDATION_FRACTION",
            "GP_VALIDATION_FRACTION",
        ),
        "0.10",
    )
)
WALK_FORWARD_VALIDATION_ROWS = int(
    os.environ.get("GP_WALK_FORWARD_VALIDATION_ROWS", "0")
)
WALK_FORWARD_STEP_ROWS = int(
    os.environ.get("GP_WALK_FORWARD_STEP_ROWS", "0")
)
WALK_FORWARD_MIN_TRAIN_ROWS = int(
    os.environ.get("GP_WALK_FORWARD_MIN_TRAIN_ROWS", "0")
)
OOS_TEST_ALPHA = float(os.environ.get("GP_OOS_TEST_ALPHA", "0.05"))
OOS_MIN_SHARPE_RATIO = float(
    os.environ.get("GP_OOS_MIN_SHARPE_RATIO", "0.50")
)
OOS_MIN_PASS_FRACTION = float(
    os.environ.get("GP_OOS_MIN_PASS_FRACTION", "1.0")
)
OOS_REQUIRE_POSITIVE = _env_bool("GP_OOS_REQUIRE_POSITIVE", True)
OOS_FILTER_FITNESS = _env_bool("GP_OOS_FILTER_FITNESS", True)
WALK_FORWARD_FITNESS = os.environ.get(
    "GP_WALK_FORWARD_FITNESS",
    "median_is",
).strip().lower()

# Plotting.  All per-generation PnL outputs share one multi-output DAG/run.
SHOW_PLOT = _env_bool("GP_SHOW_PLOT", False)
PLOT_EVERY = int(os.environ.get("GP_PLOT_EVERY", "5"))
PLOT_FINAL_GENERATION = _env_bool("GP_PLOT_FINAL_GENERATION", True)
PNL_PLOT_DOWNSAMPLE = int(os.environ.get("GP_PNL_PLOT_DOWNSAMPLE", "2000"))
PLOT_MAX_ALPHAS = int(os.environ.get("GP_PLOT_MAX_ALPHAS", "16"))
PLOT_PNL_BY_ALPHA = _env_bool("GP_PLOT_PNL_BY_ALPHA", True)
PLOT_PNL_BY_POOL = _env_bool("GP_PLOT_PNL_BY_POOL", True)
PLOT_RIDGE_BETA = _env_bool("GP_PLOT_RIDGE_BETA", True)

# Bounded execution / projection controls.
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

# Persistent Ridge pool.
POOL_SIZE = int(os.environ.get("GP_POOL_SIZE", "16"))
POOL_CANDIDATES_PER_GENERATION = int(
    os.environ.get("GP_POOL_CANDIDATES_PER_GENERATION", "8")
)
POOL_RIDGE_HL = int(os.environ.get("GP_POOL_RIDGE_HL", str(1440 * 5)))
POOL_RIDGE_LAMBDA = float(os.environ.get("GP_POOL_RIDGE_LAMBDA", "1e-3"))
POOL_RIDGE_RECOMPUTE_EVERY = 1
POOL_ROW_THRESHOLD = int(os.environ.get("GP_POOL_ROW_THRESHOLD", "5000000"))
ENABLE_POOL = _env_bool("GP_ENABLE_POOL", True)

# Group utilities remain enabled; their key terminals are bounded Key objects.
GRAMMAR = GrammarPolicy(exclude_sections=("utils.group",))
_NORMAL = NormalDist()


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    index: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int

    @property
    def train_rows(self) -> int:
        return self.train_end - self.train_start

    @property
    def validation_rows(self) -> int:
        return self.validation_end - self.validation_start


@dataclass(frozen=True, slots=True)
class SharpeComparison:
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    in_sample_se: float
    out_of_sample_se: float
    noninferiority_z: float
    noninferiority_p: float
    equality_z: float
    equality_two_sided_p: float
    passed: bool


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    fitness: float
    full_sharpe: float
    fold_comparisons: tuple[SharpeComparison, ...]
    validation_pass_fraction: float
    validation_passed: bool

    @classmethod
    def failed(cls) -> "CandidateAssessment":
        return cls(
            fitness=-math.inf,
            full_sharpe=-math.inf,
            fold_comparisons=(),
            validation_pass_fraction=0.0,
            validation_passed=False,
        )


@dataclass(frozen=True, slots=True)
class _CandidateSpec:
    key: str
    alpha: Expr
    estimated_cost: float


@dataclass(slots=True)
class _CompiledFitnessBatch:
    specs: list[_CandidateSpec]
    runtime: Any
    output_path: Path
    compile_started: float
    compile_ended: float


@dataclass(frozen=True, slots=True)
class _ExecutedFitnessBatch:
    compiled: _CompiledFitnessBatch
    values: tuple[float, ...]
    native_seconds: float
    cpu_seconds: float
    threads: int
    available_cpus: int
    parallel_mode: str
    parallel_plan: str


def l1_norm(x: Expr) -> Expr:
    """Cross-sectionally normalize a signal with finite-value purification."""

    return purify(x / abs(x).sum(axis=-1))


def clean_returns_expr(returns: Expr | None = None) -> Expr:
    """Clean raw roll returns exactly once for all search paths."""

    value = var("roll_rets") if returns is None else returns
    return where(
        abs(value) <= 0.05,
        replace(value, 0, float("nan")),
        float("nan"),
    )


def precomputed_alpha_pnl(alpha: Expr) -> Expr:
    """Equivalent to default_alpha_pnl using materialized clean/vol sources."""

    position = alpha / var("volatility")
    if LAG:
        position = shift(position, LAG)
    held = ffill(
        where(
            var("is_tradable_out0"),
            position,
            float("nan"),
        )
    )
    return shift(held) * var("clean_rets")


def depth_for_generation(generation: int) -> int:
    return 1 + (generation - 1) // DEPTH_GROW_EVERY


def build_anchored_walk_forward(
    rows: int,
    *,
    folds: int,
    validation_fraction: float,
    validation_rows: int = 0,
    step_rows: int = 0,
    min_train_rows: int = 0,
) -> tuple[WalkForwardFold, ...]:
    """Return expanding training folds whose last OOS window ends at ``rows``."""

    rows = int(rows)
    folds = int(folds)
    if rows <= 0:
        raise ValueError("walk-forward rows must be positive")
    if folds < 0:
        raise ValueError("GP_WALK_FORWARD_FOLDS must be >= 0")
    if folds == 0:
        return ()
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "GP_WALK_FORWARD_VALIDATION_FRACTION must be between 0 and 1"
        )

    valid = int(validation_rows)
    if valid <= 0:
        valid = max(2, int(round(rows * validation_fraction)))
    step = int(step_rows) if int(step_rows) > 0 else valid
    minimum_train = max(2, int(min_train_rows))
    first_end = rows - (folds - 1) * step
    first_validation_start = first_end - valid
    if first_validation_start < minimum_train:
        raise ValueError(
            "anchored walk-forward does not fit: "
            f"rows={rows:,}, folds={folds}, validation_rows={valid:,}, "
            f"step_rows={step:,}, first_train_rows={first_validation_start:,}, "
            f"required_min_train_rows={minimum_train:,}"
        )

    result: list[WalkForwardFold] = []
    for index in range(folds):
        validation_end = first_end + index * step
        validation_start = validation_end - valid
        result.append(
            WalkForwardFold(
                index=index,
                train_start=0,
                train_end=validation_start,
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )
    if result[-1].validation_end != rows:
        raise AssertionError("last anchored fold must end at the last row")
    return tuple(result)


def _sharpe_standard_error(sharpe: float, observations: int) -> float:
    if not math.isfinite(sharpe) or observations <= 1:
        return math.inf
    # IID delta-method approximation for the unannualized sample Sharpe.
    return math.sqrt((1.0 + 0.5 * sharpe * sharpe) / observations)


def compare_sharpes(
    in_sample_sharpe: float,
    out_of_sample_sharpe: float,
    *,
    in_sample_rows: int,
    out_of_sample_rows: int,
    min_ratio: float,
    alpha: float,
    require_positive: bool,
) -> SharpeComparison:
    """One-sided noninferiority plus a reported two-sided two-Sharpe z test."""

    in_sample_sharpe = float(in_sample_sharpe)
    out_of_sample_sharpe = float(out_of_sample_sharpe)
    in_se = _sharpe_standard_error(in_sample_sharpe, in_sample_rows)
    out_se = _sharpe_standard_error(out_of_sample_sharpe, out_of_sample_rows)
    finite = all(
        math.isfinite(value)
        for value in (in_sample_sharpe, out_of_sample_sharpe, in_se, out_se)
    )
    if not finite:
        return SharpeComparison(
            in_sample_sharpe,
            out_of_sample_sharpe,
            in_se,
            out_se,
            -math.inf,
            0.0,
            math.inf,
            0.0,
            False,
        )

    noninferiority_denominator = math.hypot(out_se, min_ratio * in_se)
    noninferiority_z = (
        (out_of_sample_sharpe - min_ratio * in_sample_sharpe)
        / noninferiority_denominator
        if noninferiority_denominator > 0.0
        else math.inf
    )
    noninferiority_p = _NORMAL.cdf(noninferiority_z)

    equality_denominator = math.hypot(in_se, out_se)
    equality_z = (
        (out_of_sample_sharpe - in_sample_sharpe) / equality_denominator
        if equality_denominator > 0.0
        else 0.0
    )
    equality_two_sided_p = 2.0 * (1.0 - _NORMAL.cdf(abs(equality_z)))
    passed = bool(
        noninferiority_p >= 1.0 - alpha
        and (not require_positive or out_of_sample_sharpe > 0.0)
    )
    return SharpeComparison(
        in_sample_sharpe,
        out_of_sample_sharpe,
        in_se,
        out_se,
        noninferiority_z,
        noninferiority_p,
        equality_z,
        equality_two_sided_p,
        passed,
    )


def _walk_forward_fitness(comparisons: tuple[SharpeComparison, ...]) -> float:
    if not comparisons:
        return -math.inf
    in_sample = np.asarray(
        [item.in_sample_sharpe for item in comparisons],
        dtype=np.float64,
    )
    out_of_sample = np.asarray(
        [item.out_of_sample_sharpe for item in comparisons],
        dtype=np.float64,
    )
    choices = {
        "median_is": lambda: np.median(in_sample),
        "mean_is": lambda: np.mean(in_sample),
        "min_is": lambda: np.min(in_sample),
        "median_oos": lambda: np.median(out_of_sample),
        "mean_oos": lambda: np.mean(out_of_sample),
        "min_oos": lambda: np.min(out_of_sample),
    }
    try:
        value = float(choices[WALK_FORWARD_FITNESS]())
    except KeyError as exc:
        raise ValueError(
            "GP_WALK_FORWARD_FITNESS must be one of "
            f"{sorted(choices)}, got {WALK_FORWARD_FITNESS!r}"
        ) from exc
    return value if math.isfinite(value) else -math.inf


def _assessment_from_values(
    values: tuple[float, ...],
    folds: tuple[WalkForwardFold, ...],
) -> CandidateAssessment:
    expected = 1 + 2 * len(folds)
    if len(values) != expected:
        raise RuntimeError(
            f"candidate returned {len(values)} statistics; expected {expected}"
        )
    full_sharpe = float(values[0])
    if not folds:
        finite = math.isfinite(full_sharpe)
        return CandidateAssessment(
            fitness=full_sharpe if finite else -math.inf,
            full_sharpe=full_sharpe if finite else -math.inf,
            fold_comparisons=(),
            validation_pass_fraction=1.0 if finite else 0.0,
            validation_passed=finite,
        )

    comparisons = tuple(
        compare_sharpes(
            values[1 + 2 * index],
            values[2 + 2 * index],
            in_sample_rows=fold.train_rows,
            out_of_sample_rows=fold.validation_rows,
            min_ratio=OOS_MIN_SHARPE_RATIO,
            alpha=OOS_TEST_ALPHA,
            require_positive=OOS_REQUIRE_POSITIVE,
        )
        for index, fold in enumerate(folds)
    )
    pass_fraction = float(np.mean([item.passed for item in comparisons]))
    validation_passed = pass_fraction + 1e-15 >= OOS_MIN_PASS_FRACTION
    fitness = _walk_forward_fitness(comparisons)
    if OOS_FILTER_FITNESS and not validation_passed:
        fitness = -math.inf
    return CandidateAssessment(
        fitness=fitness,
        full_sharpe=(full_sharpe if math.isfinite(full_sharpe) else -math.inf),
        fold_comparisons=comparisons,
        validation_pass_fraction=pass_fraction,
        validation_passed=validation_passed,
    )


def _slice_sources(data: dict[str, Any], rows: int) -> tuple[dict[str, Any], int]:
    shaped = [
        int(value.shape[0])
        for value in data.values()
        if tuple(getattr(value, "shape", ()))
    ]
    if not shaped:
        raise ValueError("input source mapping contains no row-shaped arrays")
    available = min(shaped)
    requested = available if rows <= 0 else int(rows)
    if requested > available:
        raise ValueError(
            f"requested {requested:,} rows, but shortest source has {available:,}"
        )

    sliced: dict[str, Any] = {}
    for name, value in data.items():
        shape = tuple(getattr(value, "shape", ()))
        sliced[name] = value[:requested] if shape else value
    return sliced, requested


def _persist_contiguous(name: str, values: np.ndarray) -> np.memmap:
    path = OUTPUT_DIR / "derived" / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    source = np.asarray(values, dtype=np.float64)
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float64,
        shape=source.shape,
    )
    chunk_rows = max(1, min(131_072, source.shape[0]))
    for start in range(0, source.shape[0], chunk_rows):
        stop = min(start + chunk_rows, source.shape[0])
        output[start:stop] = source[start:stop]
    output.flush()
    del output
    return np.load(path, mmap_mode="r")


def _run_summary(
    result,
    runtime,
    *,
    compile_seconds: float,
    wall_seconds: float,
) -> dict[str, Any]:
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


def _materialize_derived_sources(
    source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute missing roll_rets/clean_rets/volatility in one multi-output DAG."""

    missing = [
        name
        for name in ("roll_rets", "clean_rets", "volatility")
        if name not in source
    ]
    if not missing:
        return source, {"reused": list(("roll_rets", "clean_rets", "volatility"))}

    roll_expr = var("roll_rets") if "roll_rets" in source else roll_rets
    clean_expr = (
        var("clean_rets")
        if "clean_rets" in source
        else clean_returns_expr(roll_expr)
    )
    volatility_expr = (
        var("volatility")
        if "volatility" in source
        else ewm_std(clean_expr, span=ALPHA_PNL_HL)
    )
    formulas_by_name = {
        "roll_rets": roll_expr,
        "clean_rets": clean_expr,
        "volatility": volatility_expr,
    }
    formulas = [formulas_by_name[name] for name in missing]

    compile_started = time.perf_counter()
    runtime = compile_formula(
        formulas,
        source,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=PREFETCH_ROWS,
    )
    compile_seconds = time.perf_counter() - compile_started
    packed_path = OUTPUT_DIR / "derived" / "derived_multi_output.npy"
    run_started = time.perf_counter()
    result = runtime.run(
        out_path=packed_path,
        threads=THREADS,
        pin_threads=PIN_NATIVE_WORKERS,
    )
    wall_seconds = time.perf_counter() - run_started
    loaded = result.load(mmap_mode="r")
    if not isinstance(loaded, tuple):
        loaded = (loaded,)
    if len(loaded) != len(missing):
        raise RuntimeError("derived multi-output count mismatch")

    updated = dict(source)
    output_shapes: dict[str, tuple[int, ...]] = {}
    for name, values in zip(missing, loaded):
        updated[name] = _persist_contiguous(name, values)
        output_shapes[name] = tuple(int(value) for value in values.shape)
    packed_path.unlink(missing_ok=True)

    metrics = _run_summary(
        result,
        runtime,
        compile_seconds=compile_seconds,
        wall_seconds=wall_seconds,
    )
    metrics.update(
        {
            "computed": missing,
            "reused": [name for name in formulas_by_name if name not in missing],
            "output_shapes": output_shapes,
        }
    )
    print(
        f"derived_multi outputs={','.join(missing)} "
        f"compile={compile_seconds:.3f}s run={wall_seconds:.3f}s "
        f"mode={result.parallel_mode} busy_cores={result.average_busy_cores:.2f}"
    )
    return updated, metrics


def load_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load once, slice once, and build all reusable derived arrays once."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    data = InputData(fp=INPUT_GLOB, idx=None, nrows=None)
    raw = data.get_data()
    if not raw:
        raise FileNotFoundError(f"no input arrays matched {INPUT_GLOB!r}")
    source, rows = _slice_sources(raw, ROWS)
    load_seconds = time.perf_counter() - load_started
    source, derived = _materialize_derived_sources(source)
    source["gp_row_index"] = np.arange(rows, dtype=np.int64)
    return source, {
        "load_seconds": float(load_seconds),
        "derived": derived,
        "rows": int(rows),
        "n_instruments": int(N_INSTRUMENTS),
    }


def build_search_state():
    config_kwargs: dict[str, Any] = {"grammar": GRAMMAR}
    available = gp_alpha_search_terminal_metadata()
    if FIELD_NAMES:
        inputdata_fields = inputdata_alpha_terminal_metadata()
        missing = sorted(set(FIELD_NAMES) - set(inputdata_fields))
        if missing:
            raise KeyError(f"unknown GP field names: {missing}")
        config_kwargs["fields"] = {
            name: available[name] for name in FIELD_NAMES
        }
        config_kwargs["fields"]["roll_rets"] = available["roll_rets"]
    else:
        config_kwargs["fields"] = available
    if DISABLE_TENSORS:
        config_kwargs["tensor_fields"] = ()

    pset = make_pset(GPConfig(**config_kwargs))
    # Install typed leaf witnesses used by DEAP's standard generators.
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


def raw_alpha_expr(individual, pset) -> Expr:
    return individual_to_expr(individual, pset)


def alpha_expr(individual, pset) -> Expr:
    return l1_norm(raw_alpha_expr(individual, pset))


def _make_microbatches(
    items: list[_CandidateSpec],
    *,
    workers: int,
) -> list[list[_CandidateSpec]]:
    """Bound fusion size and balance estimated formula cost across native tasks."""

    if not items:
        return []
    if FITNESS_BATCH_SIZE <= 0:
        raise ValueError("GP_FITNESS_BATCH_SIZE must be positive")
    if FITNESS_TASKS_PER_WORKER <= 0:
        raise ValueError("GP_FITNESS_TASKS_PER_WORKER must be positive")
    worker_count = max(1, int(workers))
    minimum_batches = math.ceil(len(items) / FITNESS_BATCH_SIZE)
    requested_batches = worker_count * FITNESS_TASKS_PER_WORKER
    batch_count = min(len(items), max(minimum_batches, requested_batches))

    batches: list[list[_CandidateSpec]] = [[] for _ in range(batch_count)]
    loads = [0.0] * batch_count
    for item in sorted(items, key=lambda value: value.estimated_cost, reverse=True):
        choices = [
            index
            for index, batch in enumerate(batches)
            if len(batch) < FITNESS_BATCH_SIZE
        ]
        if not choices:
            raise AssertionError("microbatch capacity accounting failed")
        target = min(choices, key=lambda index: (loads[index], len(batches[index])))
        batches[target].append(item)
        loads[target] += item.estimated_cost
    return [batch for batch in batches if batch]


def _interval_union_seconds(
    stages: list[dict[str, float]],
    start_key: str,
    end_key: str,
) -> float:
    intervals = sorted(
        (float(stage[start_key]), float(stage[end_key]))
        for stage in stages
        if float(stage[end_key]) >= float(stage[start_key])
    )
    if not intervals:
        return 0.0
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _masked_sharpe(pnl: Expr, start: int, end: int) -> Expr:
    row = var("gp_row_index")
    masked = where(
        row >= int(start),
        where(row < int(end), pnl, float("nan")),
        float("nan"),
    )
    return masked.mean(axis=0) / masked.std(axis=0)


def _fitness_formulas(
    specs: list[_CandidateSpec],
    folds: tuple[WalkForwardFold, ...],
) -> list[Expr]:
    formulas: list[Expr] = []
    for spec in specs:
        pnl = precomputed_alpha_pnl(spec.alpha).sum(axis=1)
        formulas.append(pnl.mean(axis=0) / pnl.std(axis=0))
        for fold in folds:
            formulas.append(
                _masked_sharpe(pnl, fold.train_start, fold.train_end)
            )
            formulas.append(
                _masked_sharpe(
                    pnl,
                    fold.validation_start,
                    fold.validation_end,
                )
            )
    return formulas


def _compile_fitness_batch_or_bisect(
    specs: list[_CandidateSpec],
    source: dict[str, Any],
    folds: tuple[WalkForwardFold, ...],
    generation: int,
    compiled: list[_CompiledFitnessBatch],
    failures: dict[str, CandidateAssessment],
    counter: list[int],
) -> None:
    batch_index = counter[0]
    counter[0] += 1
    started = time.perf_counter()
    try:
        runtime = compile_formula(
            _fitness_formulas(specs, folds),
            source,
            n_instruments=N_INSTRUMENTS,
            prefetch_rows=PREFETCH_ROWS,
        )
    except Exception as exc:
        if len(specs) > 1:
            midpoint = len(specs) // 2
            _compile_fitness_batch_or_bisect(
                specs[:midpoint],
                source,
                folds,
                generation,
                compiled,
                failures,
                counter,
            )
            _compile_fitness_batch_or_bisect(
                specs[midpoint:],
                source,
                folds,
                generation,
                compiled,
                failures,
                counter,
            )
            return
        spec = specs[0]
        failures[spec.key] = CandidateAssessment.failed()
        print(
            f"fitness_skip candidate={spec.key!r} "
            f"reason={type(exc).__name__}: {exc}"
        )
        return

    ended = time.perf_counter()
    compiled.append(
        _CompiledFitnessBatch(
            specs=specs,
            runtime=runtime,
            output_path=(
                OUTPUT_DIR
                / "scratch"
                / f"fitness_g{generation:03d}_b{batch_index:03d}.npy"
            ),
            compile_started=started,
            compile_ended=ended,
        )
    )


def _flatten_multi_output(result) -> tuple[float, ...]:
    loaded = result.load(mmap_mode=None)
    values = loaded if isinstance(loaded, tuple) else (loaded,)
    flattened: list[float] = []
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        if array.size != 1:
            raise RuntimeError(
                "fitness statistic must be scalar, got "
                f"shape={array.shape}"
            )
        flattened.append(float(array.reshape(-1)[0]))
    return tuple(flattened)


def _execute_compiled_fitness_batches(
    compiled: list[_CompiledFitnessBatch],
    *,
    generation: int,
) -> tuple[list[_ExecutedFitnessBatch], dict[str, Any]]:
    if not compiled:
        return [], {
            "wall_seconds": 0.0,
            "workers": 0,
            "native_seconds_sum": 0.0,
            "effective_native_concurrency": 0.0,
            "fallback_serial": False,
        }

    for item in compiled:
        item.output_path.parent.mkdir(parents=True, exist_ok=True)

    fallback_serial = False
    native_started = time.perf_counter()
    try:
        batch_result = run_many(
            [item.runtime for item in compiled],
            out_paths=[item.output_path for item in compiled],
            workers=NATIVE_WORKERS,
            threads_per_runtime=1,
            pin_workers=PIN_NATIVE_WORKERS,
        )
        run_results = list(batch_result.results)
        wall_seconds = batch_result.wall_seconds
        workers = batch_result.workers
        effective_concurrency = batch_result.effective_concurrency
    except Exception as exc:
        # Runtime failures are rare after successful compilation.  Serial replay
        # isolates the bad batch while retaining a useful search result.
        fallback_serial = True
        print(
            f"native_batch_fallback generation={generation} "
            f"reason={type(exc).__name__}: {exc}"
        )
        run_results = []
        for item in compiled:
            run_results.append(
                item.runtime.run(
                    out_path=item.output_path,
                    threads=1,
                )
            )
        wall_seconds = time.perf_counter() - native_started
        workers = 1
        effective_concurrency = (
            sum(result.seconds for result in run_results) / wall_seconds
            if wall_seconds > 0.0
            else 0.0
        )

    executed: list[_ExecutedFitnessBatch] = []
    for item, result in zip(compiled, run_results):
        values = _flatten_multi_output(result)
        executed.append(
            _ExecutedFitnessBatch(
                compiled=item,
                values=values,
                native_seconds=float(result.seconds),
                cpu_seconds=float(result.cpu_seconds),
                threads=int(result.threads),
                available_cpus=int(result.available_cpus),
                parallel_mode=str(result.parallel_mode),
                parallel_plan=(
                    f"{item.runtime.parallel_plan.mode}: "
                    f"{item.runtime.parallel_plan.reason}"
                ),
            )
        )
        item.output_path.unlink(missing_ok=True)

    native_sum = float(sum(item.native_seconds for item in executed))
    return executed, {
        "wall_seconds": float(wall_seconds),
        "workers": int(workers),
        "native_seconds_sum": native_sum,
        "cpu_seconds_sum": float(sum(item.cpu_seconds for item in executed)),
        "effective_native_concurrency": float(effective_concurrency),
        "fallback_serial": fallback_serial,
        "plans": sorted({item.parallel_plan for item in executed}),
        "stages": [
            {
                "candidate_count": len(item.compiled.specs),
                "compile_seconds": (
                    item.compiled.compile_ended - item.compiled.compile_started
                ),
                "native_seconds": item.native_seconds,
                "parallel_mode": item.parallel_mode,
                "threads": item.threads,
            }
            for item in executed
        ],
    }


def evaluate_individuals(
    individuals,
    pset,
    source,
    folds: tuple[WalkForwardFold, ...],
    generation: int,
    assessment_cache: dict[str, CandidateAssessment],
) -> dict[str, Any]:
    """Evaluate unique invalid formulas using multi-output native microbatches."""

    pending = [item for item in individuals if not item.fitness.valid]
    representatives: dict[str, Any] = {}
    duplicate_groups: dict[str, list[Any]] = {}
    cached = 0
    for individual in pending:
        key = str(individual)
        cached_assessment = assessment_cache.get(key)
        if cached_assessment is not None:
            individual.fitness.values = (cached_assessment.fitness,)
            cached += 1
            continue
        duplicate_groups.setdefault(key, []).append(individual)
        representatives.setdefault(key, individual)

    specs: list[_CandidateSpec] = []
    failures: dict[str, CandidateAssessment] = {}
    for key, individual in representatives.items():
        try:
            specs.append(
                _CandidateSpec(
                    key=key,
                    alpha=alpha_expr(individual, pset),
                    estimated_cost=float(max(1, len(individual))),
                )
            )
        except Exception as exc:
            failures[key] = CandidateAssessment.failed()
            print(
                f"fitness_skip candidate={key!r} "
                f"reason={type(exc).__name__}: {exc}"
            )

    worker_hint = _available_cpus() if NATIVE_WORKERS == 0 else NATIVE_WORKERS
    microbatches = _make_microbatches(specs, workers=worker_hint)
    compiled: list[_CompiledFitnessBatch] = []
    counter = [0]
    compile_wall_started = time.perf_counter()
    for batch in microbatches:
        _compile_fitness_batch_or_bisect(
            batch,
            source,
            folds,
            generation,
            compiled,
            failures,
            counter,
        )
    compile_wall_seconds = time.perf_counter() - compile_wall_started

    executed, run_metrics = _execute_compiled_fitness_batches(
        compiled,
        generation=generation,
    )
    assessments = dict(failures)
    statistics_per_candidate = 1 + 2 * len(folds)
    for executed_batch in executed:
        expected = statistics_per_candidate * len(executed_batch.compiled.specs)
        if len(executed_batch.values) != expected:
            raise RuntimeError(
                f"fitness batch returned {len(executed_batch.values)} values; "
                f"expected {expected}"
            )
        for index, spec in enumerate(executed_batch.compiled.specs):
            start = index * statistics_per_candidate
            stop = start + statistics_per_candidate
            assessments[spec.key] = _assessment_from_values(
                executed_batch.values[start:stop],
                folds,
            )

    for key, group in duplicate_groups.items():
        assessment = assessments.get(key, CandidateAssessment.failed())
        assessment_cache[key] = assessment
        for individual in group:
            individual.fitness.values = (assessment.fitness,)

    compile_seconds_sum = float(
        sum(item.compile_ended - item.compile_started for item in compiled)
    )
    validation_values = [
        item.validation_passed
        for key, item in assessments.items()
        if key in representatives
    ]
    return {
        "wall_seconds": float(compile_wall_seconds + run_metrics["wall_seconds"]),
        "steady_state_wall_seconds": float(
            compile_wall_seconds + run_metrics["wall_seconds"]
        ),
        "compile_wall_seconds": float(compile_wall_seconds),
        "compile_seconds_sum": compile_seconds_sum,
        "run_wall_seconds_sum": float(run_metrics["wall_seconds"]),
        "native_seconds_sum": float(run_metrics["native_seconds_sum"]),
        "cpu_seconds_sum": float(run_metrics["cpu_seconds_sum"]),
        "effective_native_concurrency": float(
            run_metrics["effective_native_concurrency"]
        ),
        "effective_cpu_concurrency": float(
            run_metrics["cpu_seconds_sum"] / run_metrics["wall_seconds"]
            if run_metrics["wall_seconds"] > 0.0
            else 0.0
        ),
        "native_workers": int(run_metrics["workers"]),
        "microbatches": len(compiled),
        "plans": run_metrics.get("plans", []),
        "stages": run_metrics.get("stages", []),
        "fallback_serial": bool(run_metrics["fallback_serial"]),
        "pending_count": len(pending),
        "unique_evaluated": len(representatives),
        "cache_hits": cached,
        "duplicates_within_batch": (
            sum(len(group) for group in duplicate_groups.values())
            - len(duplicate_groups)
        ),
        "validation_pass_count": int(sum(validation_values)),
        "validation_tested_count": len(validation_values),
        "diagnostic": None,
    }


def _pool_scaled_alphas(individuals, pset) -> list[Expr]:
    volatility = var("volatility")
    return [alpha_expr(individual, pset) * volatility for individual in individuals]


def _pool_ridge_expr(
    scaled_alphas: list[Expr],
    clean_rets: Expr,
    lag: int = 0,
) -> Expr:
    hs = var("vw_halfspread_out0")
    ridge_weights = purify(
        var("volume_out0") * var("vwap_mp_out0") / (hs * hs)
    )
    return Ridge(
        *(shift(alpha, 1 + lag) for alpha in scaled_alphas),
        y=clean_rets,
        weights=ridge_weights,
        hl=float(POOL_RIDGE_HL),
        lambda_=POOL_RIDGE_LAMBDA,
        nonneg=True,
        recompute_every=POOL_RIDGE_RECOMPUTE_EVERY,
    )


_pool_ridge_expr = partial(_pool_ridge_expr, lag=LAG)


def _pool_yhat_expr(individuals, pset) -> Expr:
    scaled = _pool_scaled_alphas(individuals, pset)
    regression = _pool_ridge_expr(scaled, var("clean_rets"))
    return einsum(
        "f,nf->n",
        fillna(get_beta(regression), 0),
        fillna(cat(*scaled), 0),
    )


def _pool_pnl_expr(individuals, pset) -> Expr:
    yhat = _pool_yhat_expr(individuals, pset)
    normalized = purify(yhat / ewm_std(yhat, ALPHA_PNL_HL))
    return precomputed_alpha_pnl(normalized).sum(axis=1)


def _pool_batch_values(values: Any, expected: int) -> np.ndarray:
    """Normalize a final vector and reject accidental lane-dependent copies."""

    if hasattr(values, "load"):
        values = values.load(mmap_mode=None)
    array = np.asarray(values, dtype=np.float64)
    if array.size == expected:
        return array.reshape(expected)
    if array.ndim >= 2 and array.shape[-1] == expected:
        rows = array.reshape(-1, expected)
        first = rows[0]
        if not np.allclose(rows, first, rtol=0.0, atol=0.0, equal_nan=True):
            raise RuntimeError("final Ridge values differ across instrument lanes")
        return first.copy()
    raise RuntimeError(
        f"Ridge returned shape {array.shape}; expected {expected} values"
    )


def _plot_ridge_betas(beta: np.ndarray, generation: int) -> Path | None:
    if beta.size == 0:
        return None
    values = np.asarray(beta, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    step = max(1, math.ceil(values.shape[0] / 5000))
    sampled = values[::step]
    path = OUTPUT_DIR / f"gp_ridge_beta_g{generation:03d}.png"
    plt.figure(figsize=(10, 5))
    for index in range(sampled.shape[1]):
        plt.plot(sampled[:, index], label=f"beta {index + 1}")
    plt.xlabel(f"Time (every {step:,} rows)")
    plt.ylabel("Ridge beta")
    plt.title(f"Persistent-pool Ridge betas — generation {generation}")
    if sampled.shape[1] <= 16:
        plt.legend(fontsize=8)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    if SHOW_PLOT:
        plt.show()
    plt.close()
    return path


def ridge_contributions(
    individuals,
    pset,
    source,
    generation: int,
    *,
    capture_beta: bool,
) -> tuple[np.ndarray, dict[str, Any], Path | None]:
    """Compute contribution and optional beta history from one Ridge state/run."""

    scaled = _pool_scaled_alphas(individuals, pset)
    regression = _pool_ridge_expr(scaled, var("clean_rets"))
    formulas = [abs(get_beta(regression)).mean(axis=0)]
    if capture_beta:
        formulas.append(get_beta(regression))

    compile_started = time.perf_counter()
    runtime = compile_formula(
        formulas,
        source,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=PREFETCH_ROWS,
    )
    compile_seconds = time.perf_counter() - compile_started
    output_path = OUTPUT_DIR / "scratch" / f"ridge_g{generation:03d}.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    result = runtime.run(out_path=output_path, threads=THREADS)
    wall_seconds = time.perf_counter() - run_started
    loaded = result.load(mmap_mode=None)
    outputs = loaded if isinstance(loaded, tuple) else (loaded,)
    contribution = _pool_batch_values(outputs[0], len(individuals))
    beta_path = None
    if capture_beta:
        beta_path = _plot_ridge_betas(
            np.asarray(outputs[1], dtype=np.float64),
            generation,
        )
    output_path.unlink(missing_ok=True)

    metrics = _run_summary(
        result,
        runtime,
        compile_seconds=compile_seconds,
        wall_seconds=wall_seconds,
    )
    metrics["candidate_count"] = len(individuals)
    metrics["multi_output_beta"] = bool(capture_beta)
    return (
        np.nan_to_num(contribution, nan=0.0, posinf=0.0, neginf=0.0),
        metrics,
        beta_path,
    )


def _empty_ridge_metrics(*, reason: str = "no candidates") -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "compile_seconds": 0.0,
        "wall_seconds": 0.0,
        "native_seconds": 0.0,
        "cpu_seconds": 0.0,
        "average_busy_cores": 0.0,
        "threads": THREADS,
        "available_cpus": _available_cpus(),
        "parallel_mode": "serial",
        "parallel_plan_mode": "serial",
        "parallel_plan_reason": reason,
        "work_score": 0,
        "multi_output_beta": False,
    }


def _candidate_has_positive_fitness(individual) -> bool:
    if not individual.fitness.valid:
        return False
    value = float(individual.fitness.values[0])
    return math.isfinite(value) and value > 0.0


def update_pool(
    pool,
    population,
    pset,
    source,
    toolbox,
    generation: int,
    *,
    capture_beta: bool,
):
    """Merge strong candidates and rank once with one multi-output Ridge run."""

    if not ENABLE_POOL:
        return (
            pool,
            {},
            _empty_ridge_metrics(reason="pool disabled"),
            None,
        )

    candidates = list(pool.values()) + tools.selBest(
        population,
        min(POOL_CANDIDATES_PER_GENERATION, len(population)),
    )
    unique: dict[str, Any] = {}
    for individual in candidates:
        unique.setdefault(str(individual), individual)
    candidates = list(unique.values())
    if not candidates:
        return {}, {}, _empty_ridge_metrics(), None

    contribution, metrics, beta_path = ridge_contributions(
        candidates,
        pset,
        source,
        generation,
        capture_beta=capture_beta,
    )
    order = np.argsort(contribution)[::-1]
    previous = set(pool)
    next_pool: dict[str, Any] = {}
    next_contribution: dict[str, float] = {}
    for index in order:
        if contribution[index] <= 0.0:
            continue
        individual = candidates[int(index)]
        key = str(individual)
        if key not in previous and not _candidate_has_positive_fitness(individual):
            continue
        next_pool[key] = toolbox.clone(individual)
        next_contribution[key] = float(contribution[index])
        if len(next_pool) >= POOL_SIZE:
            break

    # The previous implementation repeatedly recompiled/re-ran the whole pool
    # after dropping one member at a time merely to test for an all-zero PnL.
    # Positive finite fitness plus positive Ridge contribution is the inexpensive
    # invariant; the plotted pool PnL remains a direct diagnostic.
    return next_pool, next_contribution, metrics, beta_path


def _alpha_pnl_matrix_expr(individuals, pset) -> Expr:
    pnls = [precomputed_alpha_pnl(alpha_expr(item, pset)) for item in individuals]
    if len(pnls) == 1:
        return pnls[0].sum(axis=1)
    return cat(*pnls).sum(axis=1)


def _portfolio_cumulative(pnl: np.ndarray, step: int) -> np.ndarray:
    """Downsample and cumulate in NumPy without constructing a DataFrame."""

    values = np.asarray(pnl, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[0] == 0:
        return np.empty((0, values.shape[1]), dtype=np.float64)
    step = max(1, int(step))
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    starts = np.arange(0, clean.shape[0], step, dtype=np.int64)
    blocks = np.add.reduceat(clean, starts, axis=0)
    return np.cumsum(blocks, axis=0)


def _plot_alpha_pnls(
    pnl: np.ndarray,
    *,
    generation: int,
) -> Path:
    values = np.asarray(pnl, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    cumulative = _portfolio_cumulative(values, PNL_PLOT_DOWNSAMPLE)
    path = OUTPUT_DIR / f"gp_alpha_pnl_g{generation:03d}.png"
    plt.figure(figsize=(10, 5))
    x = np.arange(cumulative.shape[0])
    for index in range(cumulative.shape[1]):
        plt.plot(x, cumulative[:, index], label=f"alpha {index + 1}")
    plt.xlabel(f"Time (every {PNL_PLOT_DOWNSAMPLE:,} rows)")
    plt.ylabel("Cumulative PnL")
    plt.title(f"Pool alpha PnLs — generation {generation}")
    if cumulative.shape[1] <= 16:
        plt.legend(fontsize=8)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    if SHOW_PLOT:
        plt.show()
    plt.close()
    return path


def _plot_pool_pnl(pnl: np.ndarray, *, generation: int) -> Path:
    portfolio = np.asarray(pnl, dtype=np.float64).reshape(-1)
    cumulative = _portfolio_cumulative(
        portfolio.reshape(-1, 1),
        PNL_PLOT_DOWNSAMPLE,
    )[:, 0]
    final = float(cumulative[-1]) if cumulative.size else math.nan
    path = OUTPUT_DIR / f"gp_pool_pnl_g{generation:03d}.png"
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(cumulative.shape[0]), cumulative)
    plt.xlabel(f"Time (every {PNL_PLOT_DOWNSAMPLE:,} rows)")
    plt.ylabel("Cumulative PnL")
    plt.title(f"Pool PnL — generation {generation} (cum={final:.3f})")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    if SHOW_PLOT:
        plt.show()
    plt.close()
    print(
        f"pool_pnl_plot generation={generation} rows={portfolio.size:,} "
        f"final_cum={final:.4f}"
    )
    return path


def _plot_search_pnls(
    pool,
    pool_contribution,
    pset,
    source,
    *,
    generation: int,
) -> tuple[dict[str, str | None], dict[str, Any]]:
    outputs = {"alpha_pnl_plot": None, "pool_pnl_plot": None}
    if not pool or not (PLOT_PNL_BY_ALPHA or PLOT_PNL_BY_POOL):
        return outputs, {
            "compile_seconds": 0.0,
            "wall_seconds": 0.0,
            "native_seconds": 0.0,
            "output_count": 0,
        }

    individuals = [
        pool[key]
        for key, _ in sorted(
            pool.items(),
            key=lambda item: pool_contribution[item[0]],
            reverse=True,
        )[:PLOT_MAX_ALPHAS]
    ]
    formulas: list[Expr] = []
    names: list[str] = []
    if PLOT_PNL_BY_ALPHA:
        formulas.append(_alpha_pnl_matrix_expr(individuals, pset))
        names.append("alpha")
    if PLOT_PNL_BY_POOL:
        formulas.append(_pool_pnl_expr(individuals, pset))
        names.append("pool")

    compile_started = time.perf_counter()
    runtime = compile_formula(
        formulas,
        source,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=PREFETCH_ROWS,
    )
    compile_seconds = time.perf_counter() - compile_started
    output_path = OUTPUT_DIR / "scratch" / f"plot_g{generation:03d}.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    result = runtime.run(out_path=output_path, threads=THREADS)
    wall_seconds = time.perf_counter() - run_started
    loaded = result.load(mmap_mode=None)
    values = loaded if isinstance(loaded, tuple) else (loaded,)
    for name, value in zip(names, values):
        if name == "alpha":
            outputs["alpha_pnl_plot"] = str(
                _plot_alpha_pnls(value, generation=generation)
            )
        else:
            outputs["pool_pnl_plot"] = str(
                _plot_pool_pnl(value, generation=generation)
            )
    output_path.unlink(missing_ok=True)
    metrics = _run_summary(
        result,
        runtime,
        compile_seconds=compile_seconds,
        wall_seconds=wall_seconds,
    )
    metrics["output_count"] = len(formulas)
    print(
        f"plot_multi generation={generation} outputs={','.join(names)} "
        f"compile={compile_seconds:.3f}s run={wall_seconds:.3f}s"
    )
    return outputs, metrics


def vary(population, pset, toolbox, next_depth: int):
    elites = [toolbox.clone(item) for item in tools.selBest(population, ELITE_COUNT)]
    child_count = POPULATION_SIZE - ELITE_COUNT - IMMIGRANTS
    if child_count < 0:
        raise ValueError(
            "POPULATION_SIZE must be at least ELITE_COUNT + IMMIGRANTS"
        )
    children = [
        toolbox.clone(item)
        for item in toolbox.select(population, child_count)
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
            left, right = mate(children[index - 1], children[index])
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

    immigrants = [new_individual(pset, next_depth) for _ in range(IMMIGRANTS)]
    return elites + children + immigrants


def _settings(folds: tuple[WalkForwardFold, ...]) -> dict[str, Any]:
    return {
        "n_instruments": N_INSTRUMENTS,
        "rows_requested": ROWS,
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
        "alpha_lag": LAG,
        "alpha_pnl_span": ALPHA_PNL_HL,
        "prefetch_rows": PREFETCH_ROWS,
        "threads": THREADS,
        "native_workers": NATIVE_WORKERS,
        "fitness_batch_size": FITNESS_BATCH_SIZE,
        "fitness_tasks_per_worker": FITNESS_TASKS_PER_WORKER,
        "pin_native_workers": PIN_NATIVE_WORKERS,
        "field_names": list(FIELD_NAMES),
        "disable_tensors": DISABLE_TENSORS,
        "walk_forward_folds": [asdict(fold) for fold in folds],
        "walk_forward_validation_fraction": WALK_FORWARD_VALIDATION_FRACTION,
        "walk_forward_validation_rows": WALK_FORWARD_VALIDATION_ROWS,
        "walk_forward_step_rows": WALK_FORWARD_STEP_ROWS,
        "walk_forward_min_train_rows": WALK_FORWARD_MIN_TRAIN_ROWS,
        "oos_test": {
            "kind": "one-sided Sharpe noninferiority z test",
            "iid_standard_error": "sqrt((1 + 0.5 * sharpe**2) / rows)",
            "alpha": OOS_TEST_ALPHA,
            "minimum_oos_to_is_ratio": OOS_MIN_SHARPE_RATIO,
            "minimum_pass_fraction": OOS_MIN_PASS_FRACTION,
            "require_positive_oos": OOS_REQUIRE_POSITIVE,
            "filter_fitness": OOS_FILTER_FITNESS,
            "fitness_aggregation": WALK_FORWARD_FITNESS,
        },
        "pool_size": POOL_SIZE,
        "pool_candidates_per_generation": POOL_CANDIDATES_PER_GENERATION,
        "pool_ridge_span": POOL_RIDGE_HL,
        "pool_ridge_lambda": POOL_RIDGE_LAMBDA,
        "pool_ridge_recompute_every": POOL_RIDGE_RECOMPUTE_EVERY,
        "pool_ridge_nonnegative": True,
        "enable_pool": ENABLE_POOL,
        "pool_row_threshold": POOL_ROW_THRESHOLD,
        "plot_every": PLOT_EVERY,
        "plot_final_generation": PLOT_FINAL_GENERATION,
        "plot_pnl_by_alpha": PLOT_PNL_BY_ALPHA,
        "plot_pnl_by_pool": PLOT_PNL_BY_POOL,
        "plot_ridge_beta": PLOT_RIDGE_BETA,
        "pnl_plot_downsample": PNL_PLOT_DOWNSAMPLE,
        "stop_if_projected_over_seconds": STOP_IF_PROJECTED_OVER_SECONDS,
        "max_search_wall_seconds": MAX_SEARCH_WALL_SECONDS,
    }


def _write_outputs(history, summary, *, render_plot: bool):
    frame = pd.DataFrame(history)
    csv_path = OUTPUT_DIR / "gp_search_history.csv"
    json_path = OUTPUT_DIR / "gp_search_summary.json"
    plot_path = OUTPUT_DIR / "gp_search_history.png"
    frame.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    if render_plot and not frame.empty:
        generations = frame["generation"].to_numpy()
        plt.figure(figsize=(9, 5))
        for column, label in (
            ("best_sharpe", "best fitness"),
            ("mean_sharpe", "mean fitness"),
            ("median_sharpe", "median fitness"),
        ):
            plt.plot(generations, frame[column].to_numpy(), label=label)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title(
            f"Strongly typed GP: {summary['preprocessing']['rows']:,} rows "
            f"× {N_INSTRUMENTS}"
        )
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        if SHOW_PLOT:
            plt.show()
        plt.close()
    return csv_path, json_path, plot_path


def _plot_due(generation: int) -> bool:
    scheduled = PLOT_EVERY > 0 and generation % PLOT_EVERY == 0
    final = PLOT_FINAL_GENERATION and generation == GENERATIONS
    return scheduled or final


def _assessment_json(assessment: CandidateAssessment | None) -> dict[str, Any] | None:
    if assessment is None:
        return None
    return {
        "fitness": assessment.fitness,
        "full_sharpe": assessment.full_sharpe,
        "validation_pass_fraction": assessment.validation_pass_fraction,
        "validation_passed": assessment.validation_passed,
        "folds": [asdict(item) for item in assessment.fold_comparisons],
    }


def main() -> None:
    if GENERATIONS <= 0:
        raise ValueError("GP_GENERATIONS must be positive")
    if POPULATION_SIZE <= 0:
        raise ValueError("GP_POPULATION_SIZE must be positive")
    if NATIVE_WORKERS < 0:
        raise ValueError("GP_NATIVE_WORKERS must be >= 0")
    if THREADS < 0:
        raise ValueError("GP_THREADS must be >= 0")
    if not 0.0 < OOS_TEST_ALPHA < 1.0:
        raise ValueError("GP_OOS_TEST_ALPHA must be between 0 and 1")
    if OOS_MIN_SHARPE_RATIO < 0.0:
        raise ValueError("GP_OOS_MIN_SHARPE_RATIO must be >= 0")
    if not 0.0 <= OOS_MIN_PASS_FRACTION <= 1.0:
        raise ValueError("GP_OOS_MIN_PASS_FRACTION must be between 0 and 1")
    if POOL_RIDGE_RECOMPUTE_EVERY != 1:
        raise AssertionError("Ridge recompute_every must remain 1")

    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    source, preprocessing = load_sources()
    folds = build_anchored_walk_forward(
        preprocessing["rows"],
        folds=WALK_FORWARD_FOLDS,
        validation_fraction=WALK_FORWARD_VALIDATION_FRACTION,
        validation_rows=WALK_FORWARD_VALIDATION_ROWS,
        step_rows=WALK_FORWARD_STEP_ROWS,
        min_train_rows=WALK_FORWARD_MIN_TRAIN_ROWS,
    )
    if folds:
        print("anchored_walk_forward:")
        for fold in folds:
            print(
                f"  fold={fold.index + 1} train=[0,{fold.train_end:,}) "
                f"validation=[{fold.validation_start:,},{fold.validation_end:,})"
            )
    else:
        print("anchored_walk_forward=disabled")

    if not ENABLE_POOL:
        print(
            f"pool_updates=disabled rows={preprocessing['rows']:,} "
            f"threshold={POOL_ROW_THRESHOLD:,} "
            "(set GP_ENABLE_POOL=1 to force)"
        )

    pset, toolbox = build_search_state()
    population = [new_individual(pset, 1) for _ in range(POPULATION_SIZE)]
    hall_of_fame = tools.HallOfFame(20)
    pool: dict[str, Any] = {}
    pool_contribution: dict[str, float] = {}
    assessment_cache: dict[str, CandidateAssessment] = {}
    history: list[dict[str, Any]] = []
    stop_reason = None
    search_started = time.perf_counter()
    steady_state_cumulative = 0.0

    summary: dict[str, Any] = {
        "settings": _settings(folds),
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
            source,
            folds,
            generation,
            assessment_cache,
        )
        hall_of_fame.update(population)
        plotting = _plot_due(generation)
        pool, pool_contribution, ridge_metrics, beta_path = update_pool(
            pool,
            population,
            pset,
            source,
            toolbox,
            generation,
            capture_beta=(plotting and PLOT_RIDGE_BETA),
        )
        core_generation_wall = time.perf_counter() - generation_started
        steady_state_cumulative += core_generation_wall
        cumulative_search = time.perf_counter() - search_started
        projected_total = (
            steady_state_cumulative / generation * PROJECTED_GENERATIONS
        )

        fitness = np.asarray(
            [item.fitness.values[0] for item in population],
            dtype=np.float64,
        )
        finite = fitness[np.isfinite(fitness)]
        population_assessments = [
            assessment_cache.get(str(item)) for item in population
        ]
        tested = [item for item in population_assessments if item is not None]
        passed = [item for item in tested if item.validation_passed]
        oos_values = [
            comparison.out_of_sample_sharpe
            for item in tested
            for comparison in item.fold_comparisons
            if math.isfinite(comparison.out_of_sample_sharpe)
        ]

        validation_pass_rate = (
            len(passed) / len(tested) if folds and tested else math.nan
        )
        row: dict[str, Any] = {
            "generation": generation,
            "max_depth": depth,
            "best_sharpe": float(np.max(finite)) if finite.size else math.nan,
            "mean_sharpe": float(np.mean(finite)) if finite.size else math.nan,
            "median_sharpe": (
                float(np.median(finite)) if finite.size else math.nan
            ),
            "mean_oos_sharpe": (
                float(np.mean(oos_values)) if oos_values else math.nan
            ),
            "validation_pass_rate": validation_pass_rate,
            "pool_size": len(pool),
            "fitness_pending": fitness_metrics["pending_count"],
            "fitness_unique_evaluated": fitness_metrics["unique_evaluated"],
            "fitness_cache_hits": fitness_metrics["cache_hits"],
            "fitness_microbatches": fitness_metrics["microbatches"],
            "fitness_native_workers": fitness_metrics["native_workers"],
            "fitness_wall_seconds": fitness_metrics["wall_seconds"],
            "fitness_compile_wall_seconds": fitness_metrics[
                "compile_wall_seconds"
            ],
            "fitness_compile_seconds_sum": fitness_metrics[
                "compile_seconds_sum"
            ],
            "fitness_run_wall_seconds": fitness_metrics[
                "run_wall_seconds_sum"
            ],
            "fitness_native_seconds_sum": fitness_metrics[
                "native_seconds_sum"
            ],
            "fitness_effective_native_concurrency": fitness_metrics[
                "effective_native_concurrency"
            ],
            "fitness_effective_cpu_concurrency": fitness_metrics[
                "effective_cpu_concurrency"
            ],
            "fitness_parallel_plans": " | ".join(fitness_metrics["plans"]),
            "fitness_fallback_serial": fitness_metrics["fallback_serial"],
            "ridge_candidates": ridge_metrics["candidate_count"],
            "ridge_compile_seconds": ridge_metrics["compile_seconds"],
            "ridge_wall_seconds": ridge_metrics["wall_seconds"],
            "ridge_native_seconds": ridge_metrics["native_seconds"],
            "ridge_cpu_seconds": ridge_metrics["cpu_seconds"],
            "ridge_average_busy_cores": ridge_metrics["average_busy_cores"],
            "ridge_parallel_mode": ridge_metrics["parallel_mode"],
            "ridge_parallel_plan": (
                f"{ridge_metrics['parallel_plan_mode']}: "
                f"{ridge_metrics['parallel_plan_reason']}"
            ),
            "ridge_beta_plot": str(beta_path) if beta_path else None,
            "generation_wall_seconds": core_generation_wall,
            "generation_steady_state_seconds": core_generation_wall,
            "cumulative_search_seconds": cumulative_search,
            "projected_50_generation_seconds": projected_total,
        }

        if plotting:
            pnl_plots, plot_metrics = _plot_search_pnls(
                pool,
                pool_contribution,
                pset,
                source,
                generation=generation,
            )
        else:
            pnl_plots = {"alpha_pnl_plot": None, "pool_pnl_plot": None}
            plot_metrics = {
                "compile_seconds": 0.0,
                "wall_seconds": 0.0,
                "native_seconds": 0.0,
                "output_count": 0,
            }
        row.update(pnl_plots)
        row["plot_compile_seconds"] = plot_metrics["compile_seconds"]
        row["plot_wall_seconds"] = plot_metrics["wall_seconds"]
        history.append(row)

        print(
            f"generation={generation:3d} depth={depth:2d} "
            f"best={row['best_sharpe']:9.5f} "
            f"mean={row['mean_sharpe']:9.5f} "
            f"oos_pass="
            f"{row['validation_pass_rate']:.1%} "
            f"pool={len(pool):2d} "
            f"fitness={row['fitness_wall_seconds']:.2f}s "
            f"native_workers={row['fitness_native_workers']} "
            f"native_concurrency="
            f"{row['fitness_effective_native_concurrency']:.2f}x "
            f"ridge={row['ridge_wall_seconds']:.2f}s "
            f"generation={core_generation_wall:.2f}s "
            f"projected_{PROJECTED_GENERATIONS}="
            f"{projected_total / 60.0:.2f}min"
        )

        summary["history"] = history
        summary["latest_projection_seconds"] = projected_total
        summary["actual_search_seconds"] = cumulative_search
        summary["latest_pnl_plots"] = pnl_plots
        _write_outputs(history, summary, render_plot=plotting)

        if (
            MAX_SEARCH_WALL_SECONDS > 0
            and cumulative_search >= MAX_SEARCH_WALL_SECONDS
        ):
            stop_reason = (
                f"search wall {cumulative_search:.3f}s reached "
                f"GP_MAX_SEARCH_WALL_SECONDS={MAX_SEARCH_WALL_SECONDS:.3f}s"
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

    print("\nBest formulas by fitness:")
    for rank, individual in enumerate(hall_of_fame[:10], 1):
        assessment = assessment_cache.get(str(individual))
        pass_text = (
            f" oos_pass={assessment.validation_pass_fraction:.0%}"
            if assessment is not None and folds
            else ""
        )
        print(
            f"{rank:2d}  fitness={individual.fitness.values[0]:9.5f}"
            f"{pass_text}  {individual}"
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
            f"{rank:2d}  mean_abs_beta={pool_contribution[text]:.8g}  "
            f"fitness={individual.fitness.values[0]:9.5f}  {text}"
        )

    summary.update(
        {
            "history": history,
            "stop_reason": stop_reason,
            "completed_generations": len(history),
            "actual_search_seconds": time.perf_counter() - search_started,
            "total_wall_seconds": time.perf_counter() - total_started,
            "best_formulas": [
                {
                    "rank": rank,
                    "fitness": float(individual.fitness.values[0]),
                    "formula": str(individual),
                    "assessment": _assessment_json(
                        assessment_cache.get(str(individual))
                    ),
                }
                for rank, individual in enumerate(hall_of_fame[:10], 1)
            ],
            "ridge_pool": [
                {
                    "rank": rank,
                    "mean_abs_beta": pool_contribution[text],
                    "fitness": float(individual.fitness.values[0]),
                    "formula": text,
                    "assessment": _assessment_json(
                        assessment_cache.get(text)
                    ),
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
            "pnl_plots": summary.get("latest_pnl_plots"),
        }
    )
    csv_path, json_path, plot_path = _write_outputs(
        history,
        summary,
        render_plot=True,
    )
    print(f"\nhistory_csv={csv_path}")
    print(f"summary_json={json_path}")
    print(f"fitness_plot={plot_path}")
    latest = summary.get("latest_pnl_plots", {})
    if latest.get("alpha_pnl_plot"):
        print(f"alpha_pnl_plot={latest['alpha_pnl_plot']}")
    if latest.get("pool_pnl_plot"):
        print(f"pool_pnl_plot={latest['pool_pnl_plot']}")


if __name__ == "__main__":
    main()
