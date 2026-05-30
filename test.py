import jax
import jax.numpy as jnp

from trading_dsl_engine.base.dsl import ewm, shift, var
from trading_dsl_engine.jax import compile_formula

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


def build_momentum_program():
    returns = var("returns")
    half_life = var("hl")
    # One-step lag avoids lookahead: today's return is multiplied by yesterday's signal.
    signal = shift(ts_zscore(returns, half_life), 1, 2)
    return compile_formula(signal)


def test_jax_autodiff_optimizes_momentum_half_life():
    returns = generate_autocorrelated_returns()
    artifact = build_momentum_program()
    program = artifact.compiled

    def sharpe_for_raw_half_life(raw_half_life):
        half_life = jax.nn.softplus(raw_half_life) + 2.0
        half_life_frame = jnp.broadcast_to(half_life, returns.shape)
        inputs = tuple({"returns": returns, "hl": half_life_frame}[name] for name in artifact.input_names)
        weights = program.run_batch(inputs)
        pnl = (jnp.nan_to_num(weights) * returns).sum(axis=1)
        return pnl.mean() / (pnl.std() + 1e-12)

    initial_raw_half_life = jnp.array(2.0, dtype=jnp.float64)
    initial_sharpe, initial_grad = jax.value_and_grad(sharpe_for_raw_half_life)(initial_raw_half_life)

    @jax.jit
    def optimize(raw_half_life):
        def step(raw, _):
            sharpe, grad = jax.value_and_grad(sharpe_for_raw_half_life)(raw)
            return raw + 5.0 * grad, sharpe

        return jax.lax.scan(step, raw_half_life, None, length=25)

    optimized_raw_half_life, sharpe_path = optimize(initial_raw_half_life)
    optimized_sharpe = sharpe_for_raw_half_life(optimized_raw_half_life)
    optimized_half_life = jax.nn.softplus(optimized_raw_half_life) + 2.0

    assert jnp.isfinite(initial_sharpe)
    assert jnp.isfinite(initial_grad)
    assert jnp.all(jnp.isfinite(sharpe_path))
    assert optimized_half_life > 2.0
    assert optimized_sharpe > initial_sharpe + 1e-3


if __name__ == "__main__":
    returns = generate_autocorrelated_returns()
    artifact = build_momentum_program()
    program = artifact.compiled

    def sharpe(raw_half_life):
        half_life = jax.nn.softplus(raw_half_life) + 2.0
        inputs = tuple(
            {"returns": returns, "hl": jnp.broadcast_to(half_life, returns.shape)}[name]
            for name in artifact.input_names
        )
        weights = program.run_batch(inputs)
        pnl = (jnp.nan_to_num(weights) * returns).sum(axis=1)
        return pnl.mean() / (pnl.std() + 1e-12)

    raw = jnp.array(2.0, dtype=jnp.float64)
    for _ in range(25):
        raw = raw + 5.0 * jax.grad(sharpe)(raw)
    print({"half_life": float(jax.nn.softplus(raw) + 2.0), "sharpe": float(sharpe(raw))})
