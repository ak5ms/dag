
import jax
import jax.numpy as jnp
import pandas as pd

from trading_dsl_engine.base.dsl import ewm, shift, var
from trading_dsl_engine.jax_flat.engine import compile_formula

jax.config.update("jax_enable_x64", True)


N_ROWS = 5_000
N_ASSETS = 5


def generate_autocorrelated_returns(seed=0, n_rows=N_ROWS, n_assets=N_ASSETS, windows=(5, 10)):

    noise = pd.DataFrame(jax.random.normal (jax.random.PRNGKey(seed), (n_rows+max(windows), n_assets)))
    returns = noise.rolling(windows[0]).mean() - noise.rolling(windows[1]).mean()
    returns = returns.dropna()

    import matplotlib.pyplot as plt
    def acf(returns):
        pd.DataFrame(returns).iloc[:,0].pipe(lambda x: pd.Series([x.corr(x.shift(i)) for i in range(20)])).plot();
        plt.show()
    acf(returns)
    return jnp.array(returns)

def ts_zscore(x, hl):
    """Composed DSL expression: no dedicated ts_zscore operator is required."""
    return (x - ewm(x, hl)) / (ewm(x**2, hl) ** 0.5)


def build_momentum_runtime():
    returns = var("returns")
    half_life = var("hl")
    # One-step lag avoids lookahead: today's return is multiplied by yesterday's signal.
    signal = shift(ts_zscore(returns, half_life), 1, 2)
    return compile_formula(signal)


def test_jax_flat_autodiff_optimizes_momentum_half_life():
    returns = generate_autocorrelated_returns()
    runtime = build_momentum_runtime()

    def sharpe_for_raw_half_life(raw_half_life):
        half_life = jax.nn.softplus(raw_half_life) + 2.0
        half_life_frame = jnp.broadcast_to(half_life, returns.shape)
        inputs_by_name = {"returns": returns, "hl": half_life_frame}
        _, weights = runtime.run_batch(tuple(inputs_by_name[name] for name in runtime.program.input_names))
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
    runtime = build_momentum_runtime()

    def sharpe(raw_half_life):
        half_life = jax.nn.softplus(raw_half_life) + 2.0
        inputs_by_name = {"returns": returns, "hl": jnp.broadcast_to(half_life, returns.shape)}
        _, weights = runtime.run_batch(tuple(inputs_by_name[name] for name in runtime.program.input_names))
        pnl = (jnp.nan_to_num(weights) * returns).sum(axis=1)
        sharpe = pnl.mean() / (pnl.std() + 1e-12)
        print(sharpe)
        return sharpe

    raw = jnp.array(2.0, dtype=jnp.float64)
    for _ in range(25):
        # jax.hessian(sharpe)(raw)
        raw = raw + 50.0 * jax.grad(sharpe)(raw)

    import pandas as pd
    import matplotlib.pyplot as plt
    sharpe_vmap = jax.vmap(jax.jit(sharpe))
    s1 = sharpe_vmap(jnp.arange(0,100, dtype=jnp.float64))

    s2 = []
    for x in range(100):
        sharpe(1.1*x) - sharpe(x)
        s2.append(jax.grad(sharpe)(0.1*x) + 0.5*jax.hessian(sharpe)(0.1*x)**2)
    s2 = jnp.stack(s2)
    pd.Series(s2).plot(); plt.show()

    print({"half_life": float(jax.nn.softplus(raw) + 2.0), "sharpe": float(sharpe(raw))})
