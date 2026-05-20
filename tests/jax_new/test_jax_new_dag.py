import jax
import jax.numpy as jnp

from trading_dsl_engine.jax_new.engine import compile_formula, jit_batch


def test_jax_new_full_graph_cse_reuses_subexpression_node_once():
    runtime = compile_formula("mul(xstd(ewm(div(open, close), 60)), ewm(div(open, close), 60))")
    op_names = [type(node.op).__name__ for node in runtime.program.nodes]
    assert op_names.count("NaryOp") >= 2
    assert op_names.count("EwmOp") == 1

    state0 = runtime.init_state(4)
    open_row = jnp.array([10.0, 20.0, 30.0, 40.0])
    close_row = jnp.array([11.0, 19.0, 31.0, 39.0])
    _, out = runtime.tick(state0, open_row, close_row)
    assert out.shape == (4, 1)
    jaxpr = jax.make_jaxpr(runtime.tick)(state0, open_row, close_row)
    txt = str(jaxpr)
    assert "sqrt" in txt
    assert "searchsorted" not in txt
    assert "concatenate" not in txt


def test_jax_new_supports_unary_binary_where_and_cumsum():
    runtime = compile_formula(
        "where(gt(abs(sub(open, close)), 0.1), cumsum(div(open, close)), fillna(open, close))")
    state0 = runtime.init_state(3)
    open_row = jnp.array([1.0, 2.0, jnp.nan])
    close_row = jnp.array([1.0, 1.0, 4.0])
    state1, out1 = runtime.tick(state0, open_row, close_row)
    _, out2 = runtime.tick(state1, open_row, close_row)
    assert out1.shape == (3, 1)
    assert out2.shape == (3, 1)
    assert jnp.isfinite(out2[0, 0])

def test_perf():
    import numpy as np
    from pathlib import Path
    T_1Y_MINUTES, N_INSTRUMENTS = 1440*365*10, 9
    rng = np.random.default_rng(7)
    tmp_path = Path("C:\\Users\\ak40s\\Downloads")
    close_path = tmp_path / "close.dat"
    open_path = tmp_path / "open.dat"
    # if not open_path.exists() or not close_path.exists():
    #     close = np.memmap(close_path, mode="w+", shape=(T_1Y_MINUTES, N_INSTRUMENTS), dtype=np.float64)
    #     open_ = np.memmap(open_path, mode="w+", shape=(T_1Y_MINUTES, N_INSTRUMENTS), dtype=np.float64)
    #     close[:] = rng.lognormal(mean=0.0, sigma=0.03, size=(T_1Y_MINUTES, N_INSTRUMENTS))
    #     open_[:] = rng.lognormal(mean=0.0, sigma=0.03, size=(T_1Y_MINUTES, N_INSTRUMENTS))
    #     close.flush()
    #     open_.flush()

    close = rng.lognormal(mean=0.0, sigma=0.03, size=(T_1Y_MINUTES, N_INSTRUMENTS))
    open_ = rng.lognormal(mean=0.0, sigma=0.03, size=(T_1Y_MINUTES, N_INSTRUMENTS))

    runtime = compile_formula("xs_rank(close)")# * ewm(div(close, open), 21)* ewm(div(close, open), 21)* ewm(div(close, open), 21)* ewm(div(close, open), 21)")
    # from jax import make_jaxpr
    # make_jaxpr(runtime.run_batch)(states, (close, open_))
    # print(jnp.argsort(close, axis=1))
    import time
    import jax
    import jax.numpy as jnp
    from jax import make_jaxpr

    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end = time.perf_counter()
            self.elapsed = self.end - self.start
            print(self.elapsed)

    print("dsl sort")
    with Timer():
        states, out = jit_batch(runtime, (close,), states=())
        # states, out = runtime.run_batch(states, (close, open_))
        out.block_until_ready()
    print(jax.jit(runtime.run_batch).trace(states, (close, open_)).lower().as_text())
    print(make_jaxpr(runtime.run_batch)(states, (close, open_)))

    print("vectorized sort")
    with Timer():
        print(jnp.sort(close, axis=1))
    print(jax.jit(jnp.sort).trace(close).lower().as_text())
    print(make_jaxpr(jnp.sort)(close))

    def sort_step(carry, row):
        """
        carry: Unused in this example (None), but required by lax.scan.
        row: The current 1D array sliced from the input matrix.
        """
        # Find the indices that would sort the current row
        sorted_indices = jnp.sort(row)

        # Return (new_carry, output)
        return None, sorted_indices

    # Data to scan over
    print("scan sort")
    def scan_only(xs_input):
        return jax.lax.scan(sort_step, init=None, xs=xs_input)

    with Timer():
        final_carry, all_sorted_indices = scan_only(close)
        all_sorted_indices.block_until_ready()

    print(jax.jit(scan_only).trace( close).lower().as_text())
    print(make_jaxpr(scan_only)(close))


