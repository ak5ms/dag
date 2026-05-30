from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

from trading_dsl_engine.base.dsl import ewm, shift, var
from trading_dsl_engine.jax_flat.engine import compile_formula

jax.config.update("jax_enable_x64", True)


N_ROWS = 5_000
N_ASSETS = 5


def generate_autocorrelated_returns(seed=0, n_rows=N_ROWS, n_assets=N_ASSETS, rho=0.15):
    """Generate stationary unit-variance Gaussian AR(1) returns."""
    key = jax.random.PRNGKey(seed)
    shocks = jax.random.normal(key, (n_rows, n_assets), dtype=jnp.float64)
    shock_scale = jnp.sqrt(1.0 - rho**2)

    def step(previous, shock):
        value = rho * previous + shock_scale * shock
        return value, value

    initial = shocks[0]
    _, tail = jax.lax.scan(step, initial, shocks[1:])
    return jnp.vstack([initial[None, :], tail])


def ts_zscore(x, hl):
    """Composed DSL expression: no dedicated ts_zscore operator is required."""
    return (x - ewm(x, hl)) / (ewm(x**2, hl) ** 0.5)


def build_momentum_runtime():
    returns = var("returns")
    half_life = var("hl")
    # One-step lag avoids lookahead: today's return is multiplied by yesterday's signal.
    signal = shift(ts_zscore(returns, half_life), 1, 2)
    return compile_formula(signal)


@dataclass(frozen=True)
class OptimizationResult:
    initial_half_life: jax.Array
    optimized_half_life: jax.Array
    initial_sharpe: jax.Array
    optimized_sharpe: jax.Array
    sharpe_path: jax.Array


def make_sharpe_objective(returns, runtime):
    def sharpe_for_raw_half_life(raw_half_life):
        half_life = jax.nn.softplus(raw_half_life) + 2.0
        inputs_by_name = {"returns": returns, "hl": jnp.broadcast_to(half_life, returns.shape)}
        _, weights = runtime.run_batch(tuple(inputs_by_name[name] for name in runtime.program.input_names))
        pnl = (jnp.nan_to_num(weights) * returns).sum(axis=1)
        return pnl.mean() / (pnl.std() + 1e-12)

    return sharpe_for_raw_half_life


def optimize_half_life(returns, steps=25, learning_rate=0.5):
    runtime = build_momentum_runtime()
    sharpe = make_sharpe_objective(returns, runtime)
    opt_init, opt_update, get_params = optimizers.adam(step_size=learning_rate)
    initial_raw_half_life = jnp.array(2.0, dtype=jnp.float64)

    def loss(raw_half_life):
        return -sharpe(raw_half_life)

    @jax.jit
    def run_optimizer(opt_state):
        def step(state, step_index):
            raw_half_life = get_params(state)
            loss_value, grad = jax.value_and_grad(loss)(raw_half_life)
            next_state = opt_update(step_index, grad, state)
            return next_state, -loss_value

        return jax.lax.scan(step, opt_state, jnp.arange(steps))

    initial_state = opt_init(initial_raw_half_life)
    final_state, sharpe_path = run_optimizer(initial_state)
    optimized_raw_half_life = get_params(final_state)
    return OptimizationResult(
        initial_half_life=jax.nn.softplus(initial_raw_half_life) + 2.0,
        optimized_half_life=jax.nn.softplus(optimized_raw_half_life) + 2.0,
        initial_sharpe=sharpe(initial_raw_half_life),
        optimized_sharpe=sharpe(optimized_raw_half_life),
        sharpe_path=sharpe_path,
    )


def validate_optimization(result):
    assert jnp.isfinite(result.initial_sharpe)
    assert jnp.isfinite(result.optimized_sharpe)
    assert jnp.all(jnp.isfinite(result.sharpe_path))
    assert result.optimized_half_life > 2.0
    assert result.optimized_sharpe > result.initial_sharpe + 1e-3


def main():
    result = optimize_half_life(generate_autocorrelated_returns())
    validate_optimization(result)
    print(
        {
            "initial_half_life": float(result.initial_half_life),
            "optimized_half_life": float(result.optimized_half_life),
            "initial_sharpe": float(result.initial_sharpe),
            "optimized_sharpe": float(result.optimized_sharpe),
        }
    )


if __name__ == "__main__":
    main()
