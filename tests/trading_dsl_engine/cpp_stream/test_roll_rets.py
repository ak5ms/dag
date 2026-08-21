from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from flows.riskmodel import roll_rets
from trading_dsl_engine.cpp_stream import compile_formula


N = 9
MINUTE_US = 60_000_000.0
DAY_US = 86_400_000_000.0


def _data(rows: int) -> dict[str, np.ndarray]:
    base = 1_700_000_000_000_000.0
    t = np.arange(rows, dtype=np.float64)[:, None]
    lane = np.arange(N, dtype=np.float64)[None, :]
    session_start = np.full((rows, N), base, dtype=np.float64)
    session_end = session_start + DAY_US
    event_ts = base + (60.0 + t) * MINUTE_US
    event_ts = event_ts * np.ones((1, N))
    phase = (event_ts - session_start) / DAY_US
    tradable0 = np.ones((rows, N), dtype=np.float64)
    tradable1 = np.ones((rows, N), dtype=np.float64)
    tradable0[17:20, 2] = 0.0
    tradable1[31:33, 5] = 0.0
    volume = 100.0 + 25.0 * np.sin(2.0 * np.pi * phase) + lane
    volume[11, 3] = np.nan
    close0 = 100.0 + 0.002 * t + 0.01 * lane
    close1 = 101.0 + 0.0022 * t + 0.01 * lane
    close0[tradable0 != 1.0] = np.nan
    close1[tradable1 != 1.0] = np.nan
    wdte = np.where((t // 32) % 2 == 0, 1.0, 2.0) * np.ones((1, N))
    return {
        "_ev_ts": event_ts,
        "session_start0": session_start,
        "session_end0": session_end,
        "volume_out0": volume,
        "is_tradable_out0": tradable0,
        "is_tradable_out1": tradable1,
        "wdte_out0": wdte,
        # Current roll_rets consumes VWAP fields. Retain legacy close aliases as
        # harmless extras so this validation-only fixture remains compatible
        # with either expression shape.
        "vwap_mp_out0": close0,
        "vwap_mp_out1": close1,
        "mp_out0.close": close0,
        "mp_out1.close": close1,
    }


def _save_npy(root: Path, data: dict[str, np.ndarray]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for index, (name, values) in enumerate(data.items()):
        path = root / f"input_{index}.npy"
        np.save(path, values)
        paths[name] = path
    return paths


def _assert_roll_rets_codegen(runtime) -> None:
    generated = runtime.generated_cpp.read_text()
    assert "RbfBasisSrc<6" in generated
    assert "FutureRbfBasisSumSrc<6, 1440" in generated
    assert "InstrumentBasisMeanNode" in generated
    assert "BinaryEinsumNode" in generated
    assert "EinsumNfNfToNNode" not in generated
    assert "FFillNode" in generated
    assert "stackdsl::MonotonicGroupResolver<" in generated
    assert "ShiftNode" in generated
    assert "GroupedInstrumentBasis" not in generated
    assert runtime.plan.matrix_scratch_slots == 1
    assert runtime.plan.matrix_scratch_width == 6


def test_roll_rets_uses_monotonic_session_key(tmp_path: Path) -> None:
    """Production roll_rets must compile without a large fallback group capacity."""

    paths = _save_npy(tmp_path, _data(32))
    runtime = compile_formula(
        roll_rets,
        paths,
        n_instruments=N,
    )
    _assert_roll_rets_codegen(runtime)


@pytest.mark.skip(
    reason=(
        "inherited jax_flat reference compiler does not support Key expressions; "
        "native roll_rets compilation is covered above"
    )
)
def test_roll_rets_native_matches_jax_flat(tmp_path: Path) -> None:
    from trading_dsl_engine.jax_flat.engine import compile_formula as compile_jax_formula

    jax.config.update("jax_enable_x64", True)
    rows = 160
    data = _data(rows)
    paths = _save_npy(tmp_path, data)

    cpp_runtime = compile_formula(
        roll_rets,
        paths,
        n_instruments=N,
    )
    cpp_output_path = tmp_path / "roll_rets.bin"
    cpp_runtime.run(out_path=cpp_output_path)
    cpp_output = np.asarray(
        np.memmap(cpp_output_path, mode="r", dtype=np.float64, shape=(rows, N))
    ).copy()

    jax_runtime = compile_jax_formula(roll_rets)
    state, jax_output = jax_runtime.run_batch(data)
    jax.block_until_ready(state)
    expected = np.asarray(jax_output)

    assert np.isfinite(expected).any(), "reference output must contain finite values"
    assert np.isfinite(cpp_output).any(), "native output must contain finite values"
    np.testing.assert_allclose(
        cpp_output, expected, rtol=2e-9, atol=2e-9, equal_nan=True
    )
    _assert_roll_rets_codegen(cpp_runtime)
