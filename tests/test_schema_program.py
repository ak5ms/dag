import numpy as np

from trading_dsl_engine import Float64, Instrument, Schema, Time, compile_program
from trading_dsl_engine.engine import build_engine, run_batch_from_mapping, update_from_mapping


def _schema(*names, n=3):
    return Schema(inputs={name: Float64[Time, Instrument] for name in names}, n_instruments=n)


def test_compile_program_exposes_schema_bound_plan_and_fuses_elementwise_chain():
    program = compile_program("xs_rank(ewm(div(close, open), 21))", _schema("close", "open"), allow_fallback=False)

    assert program.fast_path
    assert program.input_names == ("close", "open")
    assert program.output_schema.kind == "vector"
    assert program.output_schema.width == 3
    assert program.runtime_plan.fused_regions == ((2,),)
    assert program.runtime_plan.buffers.allocation_count > 0


def test_schema_bound_batch_matches_compat_runtime_without_tick_frame_probe():
    close = np.array([[10.0, 20.0, 30.0], [11.0, 22.0, 29.0], [12.0, 24.0, 28.0]])
    open_ = np.array([[5.0, 10.0, 15.0], [5.0, 11.0, 14.5], [6.0, 12.0, 14.0]])
    formula = "xs_rank(ewm(div(close, open), 21))"

    fast = compile_program(formula, _schema("close", "open"), allow_fallback=False)
    got = fast.bind(close=close, open=open_).run_batch(out_path=None)
    want = run_batch_from_mapping(build_engine(formula), {"close": close, "open": open_}, out_path=None)

    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_allocation_plan_counter_is_stable_across_live_steps():
    program = compile_program("rolling_quantile(close, 4, 0.5)", _schema("close", n=2), allow_fallback=False)
    state, workspace = program.initialize()
    before = workspace.allocation_count
    out = np.empty(2, dtype=np.float64)

    for tick in (np.array([1.0, 4.0]), np.array([2.0, np.nan]), np.array([3.0, 6.0])):
        program.step(state, {"close": tick}, tick_out=out, workspace=workspace)

    assert workspace.allocation_count == before
    np.testing.assert_allclose(out, np.array([2.0, 5.0]))


def test_build_engine_schema_wrapper_and_mapping_helpers_use_program_runtime():
    schema = _schema("close", "open", n=2)
    engine = build_engine("close + open", schema=schema)
    assert engine.fast_path

    close = np.array([[1.0, 2.0], [3.0, 4.0]])
    open_ = np.array([[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_allclose(run_batch_from_mapping(engine, {"close": close, "open": open_}, out_path=None), close + open_)

    y1 = update_from_mapping(engine, {"close": np.array([1.0, 2.0]), "open": np.array([3.0, 4.0])})
    np.testing.assert_allclose(y1, np.array([4.0, 6.0]))
