"""Simple end-to-end strongly typed GP alpha search.

Edit the constants below and run this file directly. The search uses Sharpe as
individual fitness and a rolling Ridge over normalized signals to keep a small
persistent pool of candidates with meaningful marginal coefficients.
"""

from __future__ import annotations

import copy
from functools import partial
import operator
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import base, creator, gp, tools

from flows.alpha_search import default_alpha_pnl
from flows.gp import GPConfig, GrammarPolicy, individual_to_expr, make_pset, make_toolbox
from flows.load import InputData
from flows.riskmodel import roll_rets
from flows.utils import replace
from trading_dsl_engine.base.dsl import Ridge, cat, get_beta, purify, shift, var, where
from trading_dsl_engine.cpp_stream import compile_formula


# Search controls. Depth is 1 for the first DEPTH_GROW_EVERY generations, then
# increases by one for each subsequent block of that many generations.
N_INSTRUMENTS = 9
POPULATION_SIZE = 64
GENERATIONS = 40
DEPTH_GROW_EVERY = 5
ELITE_COUNT = 8
TOURNAMENT_SIZE = 3
CROSSOVER_PROB = 0.50
MUTATION_PROB = 0.40
IMMIGRANTS = 8
SEED = 42

# Fitness / execution controls.
ALPHA_PNL_HL = 1440 * 21
PREFETCH_ROWS = 16

# Persistent Ridge pool. The Ridge coefficient is not the GP fitness; it is a
# simple marginal-contribution screen among the best candidates seen recently.
POOL_SIZE = 16
POOL_CANDIDATES_PER_GENERATION = 8
POOL_RIDGE_HL = 1440 * 5
POOL_RIDGE_LAMBDA = 1e-3
POOL_RIDGE_RECOMPUTE_EVERY = 60

# This is the one place to trim the grammar. Group utilities are enabled: their
# GP key arguments are bounded Key terminals, so they do not need a large generic
# group-capacity override.
GRAMMAR = GrammarPolicy()


def l1_norm(x):
    return purify(x / abs(x).sum(axis=-1))


def depth_for_generation(generation: int) -> int:
    return 1 + (generation - 1) // DEPTH_GROW_EVERY


def load_sources():
    """Load InputData and materialize roll_rets once for all GP evaluations."""

    data = InputData()
    sources = data.get_data()
    runtime = compile_formula(
        roll_rets,
        sources,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=PREFETCH_ROWS,
    )
    values = runtime.run().load()
    return sources | {"roll_rets": values}


def build_search_state():
    pset = make_pset(GPConfig(grammar=GRAMMAR))

    # Installs generation-only typed leaf witnesses for standard DEAP generation.
    # There is no reject/retry path.
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
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    return pset, toolbox


def new_individual(pset, depth: int):
    nodes = gp.genHalfAndHalf(pset=pset, min_=1, max_=depth)
    return creator.GPAlphaIndividual(nodes)


def alpha_expr(individual, pset):
    return l1_norm(individual_to_expr(individual, pset))


def evaluate_individuals(individuals, pset, sources, clean_rets):
    """Evaluate all invalid individuals together as a vector of Sharpes."""

    pending = [individual for individual in individuals if not individual.fitness.valid]
    if not pending:
        return

    pnls = [
        default_alpha_pnl(
            alpha_expr(individual, pset),
            roll_rets=clean_rets,
            is_tradable=var("is_tradable_out0"),
            hl=ALPHA_PNL_HL,
        )
        for individual in pending
    ]

    # cat gives (instrument, candidate) per streaming row. axis=1 therefore
    # aggregates instruments independently for every candidate, while axis=0
    # below is the temporal reduction that produces one Sharpe per candidate.
    pnl = pnls[0].sum(axis=1) if len(pnls) == 1 else cat(*pnls).sum(axis=1)
    score = pnl.mean(axis=0) / pnl.std(axis=0)
    runtime = compile_formula(
        score,
        sources,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=PREFETCH_ROWS,
    )
    values = np.asarray(runtime.run().load(), dtype=np.float64).reshape(-1)
    if values.size != len(pending):
        raise RuntimeError(
            f"fitness returned {values.size} values for {len(pending)} candidates"
        )

    for individual, value in zip(pending, values):
        individual.fitness.values = (
            float(value) if np.isfinite(value) else -np.inf,
        )


def ridge_contributions(individuals, pset, sources, clean_rets):
    """Return mean absolute rolling Ridge beta for each supplied alpha."""

    alphas = [alpha_expr(individual, pset) for individual in individuals]
    regression = Ridge(
        *(shift(alpha, 1, 1) for alpha in alphas),
        y=clean_rets,
        hl=float(POOL_RIDGE_HL),
        lambda_=POOL_RIDGE_LAMBDA,
        nonneg=False,
        recompute_every=POOL_RIDGE_RECOMPUTE_EVERY,
    )
    mean_abs_beta = abs(get_beta(regression)).mean(axis=0)
    runtime = compile_formula(
        mean_abs_beta,
        sources,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=PREFETCH_ROWS,
    )
    values = np.asarray(runtime.run().load(), dtype=np.float64).reshape(-1)
    if values.size != len(individuals):
        raise RuntimeError(
            f"Ridge returned {values.size} coefficients for {len(individuals)} alphas"
        )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def update_pool(pool, population, pset, sources, clean_rets, toolbox):
    """Merge strong population members into the pool and rerank by Ridge beta."""

    candidates = list(pool.values()) + tools.selBest(
        population,
        min(POOL_CANDIDATES_PER_GENERATION, len(population)),
    )
    unique = {}
    for individual in candidates:
        unique.setdefault(str(individual), individual)
    candidates = list(unique.values())

    contribution = ridge_contributions(candidates, pset, sources, clean_rets)
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
    return next_pool, next_contribution


def vary(population, pset, toolbox, next_depth: int):
    elites = [toolbox.clone(x) for x in tools.selBest(population, ELITE_COUNT)]
    child_count = POPULATION_SIZE - ELITE_COUNT - IMMIGRANTS
    children = [
        toolbox.clone(x)
        for x in toolbox.select(population, child_count)
    ]

    mate = gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=next_depth,
    )(gp.cxOnePoint)
    mutation_expr = partial(gp.genFull, min_=0, max_=min(2, next_depth))
    mutate = gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=next_depth,
    )(partial(gp.mutUniform, expr=mutation_expr, pset=pset))

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


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    sources = load_sources()
    clean_rets = where(
        abs(var("roll_rets")) <= 0.05,
        replace(var("roll_rets"), 0, float("nan")),
        float("nan"),
    )
    pset, toolbox = build_search_state()

    population = [new_individual(pset, 1) for _ in range(POPULATION_SIZE)]
    hall_of_fame = tools.HallOfFame(20)
    pool = {}
    pool_contribution = {}
    history = []

    for generation in range(1, GENERATIONS + 1):
        depth = depth_for_generation(generation)
        evaluate_individuals(population, pset, sources, clean_rets)
        hall_of_fame.update(population)
        pool, pool_contribution = update_pool(
            pool,
            population,
            pset,
            sources,
            clean_rets,
            toolbox,
        )

        fitness = np.array([x.fitness.values[0] for x in population])
        finite = fitness[np.isfinite(fitness)]
        row = {
            "generation": generation,
            "max_depth": depth,
            "best_sharpe": float(np.max(fitness)),
            "mean_sharpe": float(np.mean(finite)) if finite.size else np.nan,
            "median_sharpe": float(np.median(finite)) if finite.size else np.nan,
            "pool_size": len(pool),
        }
        history.append(row)
        print(
            f"generation={generation:3d} depth={depth} "
            f"best={row['best_sharpe']:8.4f} mean={row['mean_sharpe']:8.4f} "
            f"pool={len(pool):2d}"
        )

        if generation < GENERATIONS:
            population = vary(
                population,
                pset,
                toolbox,
                depth_for_generation(generation + 1),
            )

    frame = pd.DataFrame(history)
    frame.to_csv("gp_search_history.csv", index=False)

    print("\nBest formulas by Sharpe:")
    for rank, individual in enumerate(hall_of_fame[:10], 1):
        print(f"{rank:2d}  sharpe={individual.fitness.values[0]:9.5f}  {individual}")

    print("\nPersistent Ridge pool:")
    for rank, (text, individual) in enumerate(
        sorted(pool.items(), key=lambda item: pool_contribution[item[0]], reverse=True),
        1,
    ):
        print(
            f"{rank:2d}  mean_abs_beta={pool_contribution[text]:.8g}  "
            f"sharpe={individual.fitness.values[0]:9.5f}  {text}"
        )

    frame.plot(x="generation", y=["best_sharpe", "mean_sharpe"])
    plt.tight_layout()
    plt.savefig("gp_search_history.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
