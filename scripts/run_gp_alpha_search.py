"""End-to-end strongly typed GP alpha search with timed cpp_stream execution.

The search uses Sharpe as individual fitness and a persistent nonnegative
rolling Ridge as a marginal-contribution screen.  Configuration is controlled
by the constants below or by the matching ``GP_*`` environment variables so
the same file can be used for full data and reproducible benchmark runs.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import operator
import os
from pathlib import Path
import random
import time

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import base, creator, gp, tools

from flows.alpha_search import default_alpha_pnl
from flows.gp import GPConfig, GrammarPolicy, individual_to_expr, make_pset, make_toolbox
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
    div,
    einsum,
    fillna,
    ffill,
    get_beta,
    mul,
    purify,
    reduction,
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


# Search controls. Depth is 1 for the first DEPTH_GROW_EVERY generations, then
# increases by one for each subsequent block of that many generations.
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

# Fitness / execution controls. Terminal temporal reductions are single-owner
# cpp_stream plans, so independent candidate batches are the useful unit of
# concurrency for fitness evaluation.
LAG = 1
ALPHA_PNL_HL = int(os.environ.get("GP_ALPHA_PNL_HL", str(1440 * 21)))
PREFETCH_ROWS = int(os.environ.get("GP_PREFETCH_ROWS", "16"))
THREADS = int(os.environ.get("GP_THREADS", "1"))
FITNESS_SHARDS = int(
    os.environ.get(
        "GP_FITNESS_SHARDS",
        str(max(1, min(4, os.cpu_count() or 1))),
    )
)
PARALLEL_DIAGNOSTIC = _env_bool("GP_PARALLEL_DIAGNOSTIC", False)
DIAGNOSTIC_CANDIDATES = int(os.environ.get("GP_DIAGNOSTIC_CANDIDATES", "16"))
INPUT_GLOB = os.environ.get(
    "GP_INPUT_GLOB",
    "/mnt/extra/qrt/data/aks_out3/*.npy",
)
OUTPUT_DIR = Path(os.environ.get("GP_OUTPUT_DIR", "/tmp/gp-alpha-search"))
SHOW_PLOT = _env_bool("GP_SHOW_PLOT", True)
PNL_PLOT_DOWNSAMPLE = int(os.environ.get("GP_PNL_PLOT_DOWNSAMPLE", "2000"))
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
POOL_ROW_THRESHOLD = int(os.environ.get("GP_POOL_ROW_THRESHOLD", "5000000"))
_explicit_pool = os.environ.get("GP_ENABLE_POOL", str(True))
ENABLE_POOL = (
    _explicit_pool.strip().lower() in {"1", "true", "yes", "on"}
    if _explicit_pool is not None
    else ROWS < POOL_ROW_THRESHOLD
)
# Pool PnL plots read from the Ridge pool; disable alongside pool updates.
PLOT_PNL_BY_ALPHA = _env_bool("GP_PLOT_PNL_BY_ALPHA", True)
PLOT_PNL_BY_POOL = _env_bool("GP_PLOT_PNL_BY_POOL", True)

# Group utilities remain enabled. Their GP key arguments are bounded Key
# terminals, so no generic default_group_capacity override is required.
GRAMMAR = GrammarPolicy(exclude_sections=("utils.group",),)

default_alpha_pnl = partial(default_alpha_pnl, lag=LAG)

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


def _portfolio_cumulative(pnl: np.ndarray, step: int) -> np.ndarray:
    """Downsample row-wise PnL and cumulate, treating NaNs as zero."""
    df = pd.DataFrame(pnl)
    df = df.groupby(df.index // step).sum()
    return df.cumsum().values
    # values = np.asarray(pnl, dtype=np.float64)
    # if values.ndim == 1:
    #     values = values.reshape(-1, 1)
    # downsampled = np.nan_to_num(
    #     _downsample_rows(values, step),
    #     nan=0.0,
    # )
    # return np.cumsum(downsampled, axis=0)


def _run_expr_array(expr, sources, label: str) -> np.ndarray:
    compile_started = time.perf_counter()
    runtime = compile_formula(
        expr,
        sources,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=PREFETCH_ROWS,
    )
    compile_seconds = time.perf_counter() - compile_started

    out_path = OUTPUT_DIR / "scratch" / f"{label}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    result = runtime.run(out_path=out_path, threads=THREADS)
    wall_seconds = time.perf_counter() - run_started
    values = np.asarray(result.load(mmap_mode=None), dtype=np.float64)
    out_path.unlink(missing_ok=True)

    print(
        f"pnl_plot={label} compile={compile_seconds:.3f}s "
        f"run={wall_seconds:.3f}s shape={values.shape}"
    )
    return values


def _alpha_pnl_matrix_expr(individuals, pset, clean_rets):
    pnls = [
        default_alpha_pnl(
            alpha_expr(individual, pset),
            roll_rets=clean_rets,
            is_tradable=var("is_tradable_out0"),
            hl=ALPHA_PNL_HL,
        )
        for individual in individuals
    ]
    if len(pnls) == 1:
        return pnls[0].sum(axis=1)
    return cat(*pnls).sum(axis=1)


def _pool_scaled_alphas(individuals, pset):
    volatility = var("volatility")
    return [
        alpha_expr(individual, pset) * volatility
        for individual in individuals
    ]


def _pool_ridge_expr(scaled_alphas, clean_rets, lag = 0):
    hs = var("vw_halfspread_out0")
    ridge_weights = purify(var("volume_out0")*var("vwap_mp_out0") / (hs * hs))
    return Ridge(
        *(shift(scaled_alpha, 1 + lag) for scaled_alpha in scaled_alphas),
        y=clean_rets,
        weights=ridge_weights,
        hl=float(POOL_RIDGE_HL),
        lambda_=POOL_RIDGE_LAMBDA,
        nonneg=True,
        recompute_every=POOL_RIDGE_RECOMPUTE_EVERY,
    )

_pool_ridge_expr = partial(_pool_ridge_expr, lag=LAG)

def _pool_yhat_expr(individuals, pset, clean_rets):
    scaled_alphas = _pool_scaled_alphas(individuals, pset)
    regression = _pool_ridge_expr(scaled_alphas, clean_rets)
    # Ridge fits lagged features but applies betas to current scaled alphas.
    return einsum(
            "f,nf->n",
            fillna(get_beta(regression),0),
            fillna(cat(*scaled_alphas), 0),
        )


def _pool_portfolio_pnl_array(pnl: np.ndarray) -> np.ndarray:
    values = np.asarray(pnl, dtype=np.float64)
    if values.ndim == 1:
        return values.reshape(-1)
    if values.ndim == 2:
        return np.nansum(values, axis=1).reshape(-1)
    raise ValueError(f"expected portfolio PnL vector, got shape {pnl.shape}")


def _pool_pnl_has_signal(pnl: np.ndarray) -> bool:
    portfolio = _pool_portfolio_pnl_array(pnl)
    return bool(np.count_nonzero(portfolio))


def _candidate_has_positive_fitness(individual) -> bool:
    if not individual.fitness.valid:
        return False
    fitness = float(individual.fitness.values[0])
    return np.isfinite(fitness) and fitness > 0.0


def _pool_pnl_expr(individuals, pset, clean_rets):
    """Portfolio PnL for the Ridge pool, matching riskminer pool semantics."""

    yhat = _pool_yhat_expr(individuals, pset, clean_rets)
    # denominator = mul(
    #     ewm_std(yhat, span=ALPHA_PNL_HL),
    #     ewm_std(clean_rets, span=ALPHA_PNL_HL),
    # )
    # session_position = ffill(
    #     where(
    #         var("is_tradable_out0"),
    #         div(yhat, denominator),
    #         float("nan"),
    #     )
    # )
    # pool_contributions = fillna(
    #     mul(shift(session_position, 1, 1), clean_rets),
    #     0.0,
    # )
    # return reduction("sum", pool_contributions, axis=1)
    pnl = default_alpha_pnl(
        purify(yhat / ewm_std(yhat, ALPHA_PNL_HL)),
        roll_rets=clean_rets,
        is_tradable=var("is_tradable_out0"),
        hl=ALPHA_PNL_HL,
    ).sum(axis=1)
    return pnl

def _plot_alpha_pnls(
    individuals,
    pset,
    sources,
    clean_rets,
    *,
    generation: int,
) -> Path | None:
    if not individuals:
        return None

    pnl = _run_expr_array(
        _alpha_pnl_matrix_expr(individuals, pset, clean_rets),
        sources,
        f"alpha_pnl_g{generation:03d}",
    )
    if pnl.ndim == 1:
        pnl = pnl.reshape(-1, 1)

    plot_path = OUTPUT_DIR / f"gp_alpha_pnl_g{generation:03d}.png"
    plt.figure(figsize=(10, 5))
    cumulative = _portfolio_cumulative(pnl, PNL_PLOT_DOWNSAMPLE)
    x_values = np.arange(cumulative.shape[0])
    for index in range(cumulative.shape[1]):
        plt.plot(x_values, cumulative[:, index], label=f"alpha {index + 1}")
    plt.xlabel(f"Time (every {PNL_PLOT_DOWNSAMPLE:,} rows)")
    plt.ylabel("Cumulative PnL")
    plt.title(
        f"Alpha PnLs — generation {generation} "
        f"({len(individuals)} alphas × {N_INSTRUMENTS})"
    )
    plt.grid(True, alpha=0.25)
    if cumulative.shape[1] <= 16:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    if SHOW_PLOT:
        plt.show()
    plt.close()
    return plot_path


def _plot_pool_pnl(
    individuals,
    pset,
    sources,
    clean_rets,
    *,
    generation: int,
) -> Path | None:
    if not individuals:
        return None

    pnl = _run_expr_array(
        _pool_pnl_expr(individuals, pset, clean_rets),
        sources,
        f"pool_pnl_g{generation:03d}",
    )
    portfolio = _pool_portfolio_pnl_array(pnl)
    cumulative = _portfolio_cumulative(
        portfolio.reshape(-1, 1),
        PNL_PLOT_DOWNSAMPLE,
    )[:, 0]
    final_cumulative = float(cumulative[-1]) if cumulative.size else float("nan")
    print(
        f"pool_pnl_plot generation={generation} "
        f"rows={portfolio.size:,} "
        f"final_cum={final_cumulative:.4f} "
        f"range=[{float(np.min(cumulative)):.4f}, "
        f"{float(np.max(cumulative)):.4f}]"
    )

    plot_path = OUTPUT_DIR / f"gp_pool_pnl_g{generation:03d}.png"
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(cumulative.shape[0]), cumulative)
    plt.xlabel(f"Time (every {PNL_PLOT_DOWNSAMPLE:,} rows)")
    plt.ylabel("Cumulative PnL")
    plt.title(
        f"Pool PnL — generation {generation} "
        f"(cum={final_cumulative:.3f})"
    )
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    if SHOW_PLOT:
        plt.show()
    plt.close()
    return plot_path


def _plot_search_pnls(
    pool,
    pool_contribution,
    pset,
    sources,
    clean_rets,
    *,
    generation: int,
) -> dict[str, str | None]:
    # if not pool:
    #     return {"alpha_pnl_plot": None, "pool_pnl_plot": None}

    individuals = [
        pool[key]
        for key, _ in sorted(
            pool.items(),
            key=lambda item: pool_contribution[item[0]],
            reverse=True,
        )
    ]
    outputs = {
        "alpha_pnl_plot": None,
        "pool_pnl_plot": None,
    }
    if PLOT_PNL_BY_ALPHA:
        alpha_path = _plot_alpha_pnls(
            individuals,
            pset,
            sources,
            clean_rets,
            generation=generation,
        )
        outputs["alpha_pnl_plot"] = (
            str(alpha_path) if alpha_path is not None else None
        )
    if PLOT_PNL_BY_POOL:
        pool_path = _plot_pool_pnl(
            individuals,
            pset,
            sources,
            clean_rets,
            generation=generation,
        )
        outputs["pool_pnl_plot"] = (
            str(pool_path) if pool_path is not None else None
        )
    return outputs


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


def load_sources(rows = None):
    """Load exactly ROWS rows and materialize reusable derived sources once."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    data = InputData(fp=INPUT_GLOB, idx=None, nrows=None)
    raw_sources = data.get_data()
    if not raw_sources:
        raise FileNotFoundError(f"no input arrays matched {INPUT_GLOB!r}")
    sources = _slice_sources(raw_sources, rows if rows else ROWS)
    load_seconds = time.perf_counter() - load_started

    derived_metrics = {}
    if "roll_rets" not in sources:
        values, metrics = _derived_source("roll_rets", roll_rets, sources)
        sources = sources | {"roll_rets": values}
        derived_metrics["roll_rets"] = metrics
    else:
        print("derived=roll_rets reused precomputed input")

    # This is the exact ewm_std denominator used by default_alpha_pnl. It is
    # materialized once so the Ridge feature scaling does not rebuild it in
    # every generation.
    if "volatility" not in sources:
        volatility_formula = ewm_std(
            clean_returns_expr(),
            span=ALPHA_PNL_HL,
        )
        values, metrics = _derived_source(
            "volatility",
            volatility_formula,
            sources,
        )
        sources = sources | {"volatility": values}
        derived_metrics["volatility"] = metrics
    else:
        print("derived=volatility reused precomputed input")

    return sources, {
        "load_seconds": float(load_seconds),
        "derived": derived_metrics,
        "rows": int(ROWS),
        "n_instruments": int(N_INSTRUMENTS),
    }


def build_search_state():
    config_kwargs = {"grammar": GRAMMAR}
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


def _split_batches(items, requested_shards: int):
    if not items:
        return []
    count = max(1, min(int(requested_shards), len(items)))
    width = (len(items) + count - 1) // count
    return [
        items[start : start + width]
        for start in range(0, len(items), width)
    ]


def _run_fitness_score(
    score,
    candidate_count: int,
    generation: int,
    shard: int,
    label: str,
    sources,
):
    compile_started = time.perf_counter()
    runtime = compile_formula(
        score,
        sources,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=PREFETCH_ROWS,
    )
    compile_seconds = time.perf_counter() - compile_started

    out_path = (
        OUTPUT_DIR
        / "scratch"
        / f"fitness_{label}_g{generation:03d}_s{shard:02d}.npy"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    result = runtime.run(out_path=out_path, threads=THREADS)
    wall_seconds = time.perf_counter() - run_started
    values = np.asarray(
        result.load(mmap_mode=None),
        dtype=np.float64,
    ).reshape(-1)
    out_path.unlink(missing_ok=True)

    if values.size != candidate_count:
        raise RuntimeError(
            f"fitness returned {values.size} values for "
            f"{candidate_count} candidates"
        )

    metrics = _run_summary(
        result,
        runtime,
        compile_seconds=compile_seconds,
        wall_seconds=wall_seconds,
    )
    return values, metrics


def _fitness_batch_fallback(
    batch,
    sources,
    clean_rets,
    generation: int,
    shard: int,
    label: str,
):
    keys_out = []
    values_out = []
    stages = []
    for key, alpha in batch:
        try:
            pnl = default_alpha_pnl(
                alpha,
                roll_rets=clean_rets,
                is_tradable=var("is_tradable_out0"),
                hl=ALPHA_PNL_HL,
            ).sum(axis=1)
            score = pnl.mean(axis=0) / pnl.std(axis=0)
            values, metrics = _run_fitness_score(
                score,
                1,
                generation,
                shard,
                f"{label}_single",
                sources,
            )
            value = float(values[0])
            keys_out.append(key)
            values_out.append(
                value if np.isfinite(value) else -np.inf
            )
            stages.append(metrics)
        except Exception as exc:
            print(
                f"fitness_skip candidate={key!r} "
                f"reason={type(exc).__name__}: {exc}"
            )
            keys_out.append(key)
            values_out.append(-np.inf)

    metrics = {
        "compile_seconds": float(
            sum(item["compile_seconds"] for item in stages)
        ),
        "wall_seconds": float(
            sum(item["wall_seconds"] for item in stages)
        ),
        "native_seconds": float(
            sum(item["native_seconds"] for item in stages)
        ),
        "cpu_seconds": float(
            sum(item["cpu_seconds"] for item in stages)
        ),
        "average_busy_cores": float(
            np.mean([item["average_busy_cores"] for item in stages])
            if stages
            else 0.0
        ),
        "threads": THREADS,
        "available_cpus": int(stages[0]["available_cpus"]) if stages else (
            os.cpu_count() or 1
        ),
        "parallel_mode": (
            stages[0]["parallel_mode"] if stages else "serial"
        ),
        "parallel_plan_mode": (
            stages[0]["parallel_plan_mode"] if stages else "serial"
        ),
        "parallel_plan_reason": (
            stages[0]["parallel_plan_reason"]
            if stages
            else "per-candidate fallback"
        ),
        "work_score": int(
            sum(item["work_score"] for item in stages)
        ),
        "candidate_count": len(batch),
        "shard": shard,
        "label": label,
    }
    return keys_out, np.asarray(values_out, dtype=np.float64), metrics


def _fitness_batch(
    batch,
    sources,
    clean_rets,
    generation: int,
    shard: int,
    label: str,
):
    pnls = [
        default_alpha_pnl(
            alpha,
            roll_rets=clean_rets,
            is_tradable=var("is_tradable_out0"),
            hl=ALPHA_PNL_HL,
        )
        for _, alpha in batch
    ]
    pnl = (
        pnls[0].sum(axis=1)
        if len(pnls) == 1
        else cat(*pnls).sum(axis=1)
    )
    score = pnl.mean(axis=0) / pnl.std(axis=0)

    try:
        values, metrics = _run_fitness_score(
            score,
            len(batch),
            generation,
            shard,
            label,
            sources,
        )
    except Exception as exc:
        print(
            f"fitness_batch shard={shard} label={label!r} "
            f"fallback after {type(exc).__name__}: {exc}"
        )
        return _fitness_batch_fallback(
            batch,
            sources,
            clean_rets,
            generation,
            shard,
            label,
        )

    metrics.update(
        {
            "candidate_count": len(batch),
            "shard": shard,
            "label": label,
        }
    )
    return [key for key, _ in batch], values, metrics


def _evaluate_alpha_batches(
    batch,
    sources,
    clean_rets,
    generation: int,
    shards: int,
    label: str,
):
    chunks = _split_batches(batch, shards)
    started = time.perf_counter()
    if len(chunks) == 1:
        outcomes = [
            _fitness_batch(
                chunks[0],
                sources,
                clean_rets,
                generation,
                0,
                label,
            )
        ]
    else:
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(
                    _fitness_batch,
                    chunk,
                    sources,
                    clean_rets,
                    generation,
                    index,
                    label,
                )
                for index, chunk in enumerate(chunks)
            ]
            outcomes = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - started

    scores = {}
    stages = []
    for keys, values, metrics in outcomes:
        stages.append(metrics)
        for key, value in zip(keys, values):
            scores[key] = (
                float(value)
                if np.isfinite(value)
                else -np.inf
            )

    native_sum = sum(item["native_seconds"] for item in stages)
    run_wall_sum = sum(item["wall_seconds"] for item in stages)
    cpu_sum = sum(item["cpu_seconds"] for item in stages)
    return scores, {
        "wall_seconds": float(wall_seconds),
        "shards": len(chunks),
        "candidate_count": len(batch),
        "compile_seconds_sum": float(
            sum(item["compile_seconds"] for item in stages)
        ),
        "run_wall_seconds_sum": float(run_wall_sum),
        "native_seconds_sum": float(native_sum),
        "cpu_seconds_sum": float(cpu_sum),
        "effective_native_concurrency": float(
            native_sum / wall_seconds if wall_seconds else 0.0
        ),
        "effective_cpu_concurrency": float(
            cpu_sum / wall_seconds if wall_seconds else 0.0
        ),
        "plans": sorted(
            {
                (
                    f"{item['parallel_plan_mode']}: "
                    f"{item['parallel_plan_reason']}"
                )
                for item in stages
            }
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
        (key, alpha_expr(individual, pset))
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
            remainder_metrics = {
                "wall_seconds": 0.0,
                "shards": 0,
                "candidate_count": 0,
                "compile_seconds_sum": 0.0,
                "run_wall_seconds_sum": 0.0,
                "native_seconds_sum": 0.0,
                "cpu_seconds_sum": 0.0,
                "effective_native_concurrency": 0.0,
                "effective_cpu_concurrency": 0.0,
                "plans": [],
                "stages": [],
            }

        steady_wall = (
            sharded_metrics["wall_seconds"]
            + remainder_metrics["wall_seconds"]
        )
        metrics = {
            "wall_seconds": (
                serial_metrics["wall_seconds"] + steady_wall
            ),
            "steady_state_wall_seconds": steady_wall,
            "shards": FITNESS_SHARDS,
            "candidate_count": len(batch),
            "compile_seconds_sum": (
                sharded_metrics["compile_seconds_sum"]
                + remainder_metrics["compile_seconds_sum"]
            ),
            "run_wall_seconds_sum": (
                sharded_metrics["run_wall_seconds_sum"]
                + remainder_metrics["run_wall_seconds_sum"]
            ),
            "native_seconds_sum": (
                sharded_metrics["native_seconds_sum"]
                + remainder_metrics["native_seconds_sum"]
            ),
            "cpu_seconds_sum": (
                sharded_metrics["cpu_seconds_sum"]
                + remainder_metrics["cpu_seconds_sum"]
            ),
            "effective_native_concurrency": (
                (
                    sharded_metrics["native_seconds_sum"]
                    + remainder_metrics["native_seconds_sum"]
                )
                / steady_wall
                if steady_wall
                else 0.0
            ),
            "effective_cpu_concurrency": (
                (
                    sharded_metrics["cpu_seconds_sum"]
                    + remainder_metrics["cpu_seconds_sum"]
                )
                / steady_wall
                if steady_wall
                else 0.0
            ),
            "plans": sorted(
                set(sharded_metrics["plans"])
                | set(remainder_metrics["plans"])
            ),
            "stages": (
                sharded_metrics["stages"]
                + remainder_metrics["stages"]
            ),
        }
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
        metrics = {
            "wall_seconds": 0.0,
            "steady_state_wall_seconds": 0.0,
            "shards": 0,
            "candidate_count": 0,
            "compile_seconds_sum": 0.0,
            "run_wall_seconds_sum": 0.0,
            "native_seconds_sum": 0.0,
            "cpu_seconds_sum": 0.0,
            "effective_native_concurrency": 0.0,
            "effective_cpu_concurrency": 0.0,
            "plans": [],
            "stages": [],
        }

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


def ridge_contributions(
    individuals,
    pset,
    sources,
    clean_rets,
    generation: int,
):
    """Return mean(abs(beta)) using the requested scaled nonnegative Ridge."""

    # alpha_expr is explicitly l1_norm(raw_alpha). Ridge therefore receives
    # shift(l1_norm(alpha), 1, 1) multiplied by the exact volatility used in the
    # default_alpha_pnl denominator.
    scaled_alphas = _pool_scaled_alphas(individuals, pset)
    regression = _pool_ridge_expr(scaled_alphas, clean_rets)
    mean_abs_beta = abs(get_beta(regression)).mean(axis=0)

    compile_started = time.perf_counter()
    runtime = compile_formula(
        mean_abs_beta,
        sources,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=PREFETCH_ROWS,
    )
    compile_seconds = time.perf_counter() - compile_started

    out_path = (
        OUTPUT_DIR
        / "scratch"
        / f"ridge_g{generation:03d}.npy"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    result = runtime.run(out_path=out_path, threads=THREADS)
    wall_seconds = time.perf_counter() - run_started
    values = np.asarray(
        result.load(mmap_mode=None),
        dtype=np.float64,
    ).reshape(-1)
    out_path.unlink(missing_ok=True)

    ## plotting
    runtime = compile_formula(
        get_beta(regression),
        sources,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=PREFETCH_ROWS,
    )
    result_beta = runtime.run(out_path=out_path, threads=THREADS)
    pd.DataFrame(result_beta.load()).plot(); plt.show()

    if values.size != len(individuals):
        raise RuntimeError(
            f"Ridge returned {values.size} coefficients for "
            f"{len(individuals)} alphas"
        )

    metrics = _run_summary(
        result,
        runtime,
        compile_seconds=compile_seconds,
        wall_seconds=wall_seconds,
    )
    metrics["candidate_count"] = len(individuals)
    return (
        np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        metrics,
    )


def _empty_ridge_metrics(*, reason: str = "no candidates"):
    return {
        "candidate_count": 0,
        "compile_seconds": 0.0,
        "wall_seconds": 0.0,
        "native_seconds": 0.0,
        "cpu_seconds": 0.0,
        "average_busy_cores": 0.0,
        "threads": THREADS,
        "available_cpus": os.cpu_count() or 1,
        "parallel_mode": "serial",
        "parallel_plan_mode": "serial",
        "parallel_plan_reason": reason,
        "work_score": 0,
    }


def update_pool(
    pool,
    population,
    pset,
    sources,
    clean_rets,
    toolbox,
    generation: int,
):
    """Merge strong population members into the pool and rerank by Ridge beta."""

    if not ENABLE_POOL:
        return pool, {}, _empty_ridge_metrics(reason="pool disabled")

    candidates = list(pool.values()) + tools.selBest(
        population,
        min(POOL_CANDIDATES_PER_GENERATION, len(population)),
    )
    unique = {}
    for individual in candidates:
        unique.setdefault(str(individual), individual)
    candidates = list(unique.values())

    if not candidates:
        return {}, {}, _empty_ridge_metrics()

    contribution, ridge_metrics = ridge_contributions(
        candidates,
        pset,
        sources,
        clean_rets,
        generation,
    )
    order = np.argsort(contribution)[::-1]
    previous_keys = set(pool.keys())
    next_pool = {}
    next_contribution = {}
    for index in order:
        if contribution[index] <= 0.0:
            continue
        individual = candidates[index]
        key = str(individual)
        if key not in previous_keys and not _candidate_has_positive_fitness(
            individual
        ):
            continue
        next_pool[key] = toolbox.clone(individual)
        next_contribution[key] = float(contribution[index])
        if len(next_pool) >= POOL_SIZE:
            break

    while next_pool:
        ordered = [
            next_pool[key]
            for key, _ in sorted(
                next_pool.items(),
                key=lambda item: next_contribution[item[0]],
                reverse=True,
            )
        ]
        combined_pnl = _run_expr_array(
            _pool_pnl_expr(ordered, pset, clean_rets),
            sources,
            f"pool_check_g{generation:03d}",
        )
        if _pool_pnl_has_signal(combined_pnl):
            break
        worst_key = min(
            next_pool,
            key=lambda item: next_contribution[item],
        )
        del next_pool[worst_key]
        del next_contribution[worst_key]

    return next_pool, next_contribution, ridge_metrics


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
        "threads": THREADS,
        "fitness_shards": FITNESS_SHARDS,
        "parallel_diagnostic": PARALLEL_DIAGNOSTIC,
        "diagnostic_candidates": DIAGNOSTIC_CANDIDATES,
        "field_names": list(FIELD_NAMES),
        "disable_tensors": DISABLE_TENSORS,
        "pool_size": POOL_SIZE,
        "pool_candidates_per_generation": (
            POOL_CANDIDATES_PER_GENERATION
        ),
        "pool_ridge_span": POOL_RIDGE_HL,
        "pool_ridge_lambda": POOL_RIDGE_LAMBDA,
        "pool_ridge_recompute_every": POOL_RIDGE_RECOMPUTE_EVERY,
        "pool_ridge_nonnegative": True,
        "enable_pool": ENABLE_POOL,
        "pool_row_threshold": POOL_ROW_THRESHOLD,
        "plot_pnl_by_alpha": PLOT_PNL_BY_ALPHA,
        "plot_pnl_by_pool": PLOT_PNL_BY_POOL,
        "pnl_plot_downsample": PNL_PLOT_DOWNSAMPLE,
        "ridge_weights": "purify(1 / (vw_halfspread_out0 ** 2))",
        "ridge_feature": (
            "shift(l1_norm(alpha), 1, 1) "
            "* ewm_std(clean_roll_rets, span=alpha_pnl_span)"
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
        generations = frame["generation"].to_numpy()
        plt.figure(figsize=(9, 5))
        plt.plot(
            generations,
            frame["best_sharpe"].to_numpy(),
            label="best Sharpe",
        )
        plt.plot(
            generations,
            frame["mean_sharpe"].to_numpy(),
            label="mean Sharpe",
        )
        plt.plot(
            generations,
            frame["median_sharpe"].to_numpy(),
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

if __name__ == "__main__":
    if GENERATIONS <= 0:
        raise ValueError("GP_GENERATIONS must be positive")
    if POPULATION_SIZE <= 0:
        raise ValueError("GP_POPULATION_SIZE must be positive")
    if FITNESS_SHARDS <= 0:
        raise ValueError("GP_FITNESS_SHARDS must be positive")
    if POOL_RIDGE_RECOMPUTE_EVERY != 1:
        raise AssertionError("Ridge recompute_every must remain 1")

    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ENABLE_POOL:
        print(
            f"pool_updates=disabled rows={ROWS:,} "
            f"threshold={POOL_ROW_THRESHOLD:,} "
            f"(set GP_ENABLE_POOL=1 to force)"
        )

    total_started = time.perf_counter()
    sources, preprocessing = load_sources()
    sources_all, preprocessing_all = load_sources(int(5E6))
    clean_rets = clean_returns_expr()
    pset, toolbox = build_search_state()

    population = [
        new_individual(pset, 1)
        for _ in range(POPULATION_SIZE)
    ]

    # ind = gp.PrimitiveTree.from_string("xs_rank_numeric(field_roll_rets)", pset)
    # ind = gp.PrimitiveTree.from_string("mul_scalar_dimensionless(-1, xs_rank_numeric(field_roll_rets))", pset)
    # feature = gp.compile(ind, pset).expr.plot()
    # population = [creator.GPAlphaIndividual(ind)]

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
            "fitness_shards": fitness_metrics["shards"],
            "fitness_wall_seconds": fitness_metrics["wall_seconds"],
            "fitness_steady_state_wall_seconds": (
                fitness_metrics["steady_state_wall_seconds"]
            ),
            "fitness_compile_seconds_sum": (
                fitness_metrics["compile_seconds_sum"]
            ),
            "fitness_run_wall_seconds_sum": (
                fitness_metrics["run_wall_seconds_sum"]
            ),
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
            "ridge_compile_seconds": ridge_metrics["compile_seconds"],
            "ridge_wall_seconds": ridge_metrics["wall_seconds"],
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
            "generation_wall_seconds": generation_wall,
            "generation_steady_state_seconds": steady_generation_wall,
            "cumulative_search_seconds": cumulative_search,
            "projected_50_generation_seconds": projected_total,
        }
        pnl_plots = _plot_search_pnls(
            pool,
            pool_contribution,
            pset,
            sources_all,
            clean_rets,
            generation=generation,
        )
        row["alpha_pnl_plot"] = pnl_plots["alpha_pnl_plot"]
        row["pool_pnl_plot"] = pnl_plots["pool_pnl_plot"]
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
            f"ridge={row['ridge_wall_seconds']:.2f}s "
            f"generation={steady_generation_wall:.2f}s "
            f"projected_{PROJECTED_GENERATIONS}="
            f"{projected_total / 60.0:.2f}min"
        )

        summary["history"] = history
        if diagnostic is not None:
            summary["parallel_diagnostic"] = diagnostic
        summary["latest_projection_seconds"] = projected_total
        summary["actual_search_seconds"] = cumulative_search
        summary["latest_pnl_plots"] = pnl_plots
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
            "pnl_plots": summary.get("latest_pnl_plots"),
        }
    )
    csv_path, json_path, plot_path = _write_outputs(history, summary)
    latest_pnl_plots = summary.get("latest_pnl_plots", {})
    print(f"\nhistory_csv={csv_path}")
    print(f"summary_json={json_path}")
    print(f"fitness_plot={plot_path}")
    if latest_pnl_plots.get("alpha_pnl_plot"):
        print(f"alpha_pnl_plot={latest_pnl_plots['alpha_pnl_plot']}")
    if latest_pnl_plots.get("pool_pnl_plot"):
        print(f"pool_pnl_plot={latest_pnl_plots['pool_pnl_plot']}")