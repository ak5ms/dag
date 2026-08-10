from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.ir import compile_ir
from trading_dsl_engine.ir.frontend import FormulaIRCompileError


def _save_inputs(
    root: Path,
    *,
    rows: int = 17,
    instruments: int = 9,
    features: int = 3,
) -> tuple[dict[str, Path], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260810)
    x = rng.normal(size=(rows, instruments, features))
    coefficients = np.linspace(0.45, -0.15, features)
    y = np.einsum("tnk,k->tn", x, coefficients)
    y += rng.normal(scale=0.03, size=(rows, instruments))
    paths: dict[str, Path] = {}
    for index in range(features):
        path = root / f"x{index}.npy"
        np.save(path, x[:, :, index])
        paths[f"x{index}"] = path
    y_path = root / "y.npy"
    np.save(y_path, y)
    paths["y"] = y_path
    return paths, x, y


def _model(features: int, *, half_life: int, recompute_every: int | None) -> str:
    feature_text = ", ".join(f"x{index}" for index in range(features))
    interval = (
        ""
        if recompute_every is None
        else f", recompute_every={recompute_every}"
    )
    return (
        f"Ridge(cat({feature_text}), y=y, hl={half_life}, "
        f"lambda_=0.1{interval})"
    )


def _run(
    root: Path,
    paths: dict[str, Path],
    formula: str,
    *,
    instruments: int,
    label: str,
) -> tuple[np.ndarray, str]:
    runtime = compile_formula(formula, paths, n_instruments=instruments)
    output = root / f"{label}.bin"
    runtime.run(out_path=output, async_writeback_mb=0)
    values = np.memmap(
        output,
        mode="r",
        dtype=np.float64,
        shape=(17, runtime.plan.output_row_width),
    )
    result = np.asarray(values).copy()
    del values
    return result, runtime.generated_cpp.read_text()


def _hold_refresh_rows(values: np.ndarray, every: int) -> np.ndarray:
    indices = (np.arange(values.shape[0]) // every) * every
    return values[indices]


@pytest.mark.parametrize(
    ("projection", "width"),
    [
        ("get_beta", 3),
        ("get_r2", 1),
        ("get_standard_errors", 3),
        ("get_tstats", 3),
    ],
)
def test_recompute_every_one_preserves_existing_results(
    tmp_path: Path,
    projection: str,
    width: int,
) -> None:
    paths, _, _ = _save_inputs(tmp_path)
    default, _ = _run(
        tmp_path,
        paths,
        f"{projection}({_model(3, half_life=8, recompute_every=None)})",
        instruments=9,
        label=f"{projection}_default",
    )
    explicit, generated = _run(
        tmp_path,
        paths,
        f"{projection}({_model(3, half_life=8, recompute_every=1)})",
        instruments=9,
        label=f"{projection}_one",
    )
    assert default.shape == explicit.shape == (17, width)
    np.testing.assert_array_equal(explicit, default)
    assert "stackdsl::RidgeNode<" in generated
    assert re.search(r"stackdsl::DirectExecution<9>, 1>", generated)


@pytest.mark.parametrize(
    "projection",
    [
        "get_beta",
        "get_r2",
        "get_standard_errors",
        "get_tstats",
    ],
)
def test_periodic_ridge_holds_coherent_solved_snapshot(
    tmp_path: Path,
    projection: str,
) -> None:
    paths, _, _ = _save_inputs(tmp_path)
    baseline, _ = _run(
        tmp_path,
        paths,
        f"{projection}({_model(3, half_life=8, recompute_every=1)})",
        instruments=9,
        label=f"{projection}_baseline",
    )
    periodic, generated = _run(
        tmp_path,
        paths,
        f"{projection}({_model(3, half_life=8, recompute_every=4)})",
        instruments=9,
        label=f"{projection}_periodic",
    )
    expected = _hold_refresh_rows(baseline, 4)
    np.testing.assert_allclose(
        periodic,
        expected,
        rtol=5e-10,
        atol=5e-10,
        equal_nan=True,
    )
    assert re.search(r"stackdsl::DirectExecution<9>, 4>", generated)


def test_periodic_stateful_predictions_keep_prior_beta_timing(
    tmp_path: Path,
) -> None:
    paths, features, _ = _save_inputs(tmp_path)
    beta, _ = _run(
        tmp_path,
        paths,
        f"get_beta({_model(3, half_life=8, recompute_every=4)})",
        instruments=9,
        label="periodic_beta",
    )
    predictions, _ = _run(
        tmp_path,
        paths,
        f"get_preds({_model(3, half_life=8, recompute_every=4)})",
        instruments=9,
        label="periodic_predictions",
    )
    expected = np.zeros_like(predictions)
    for row in range(1, predictions.shape[0]):
        expected[row] = features[row] @ beta[row - 1]
    np.testing.assert_allclose(predictions, expected, rtol=5e-10, atol=5e-10)


def test_periodic_stateless_ridge_holds_last_current_row_fit(
    tmp_path: Path,
) -> None:
    paths, _, _ = _save_inputs(tmp_path)
    baseline, _ = _run(
        tmp_path,
        paths,
        f"get_beta({_model(3, half_life=0, recompute_every=1)})",
        instruments=9,
        label="stateless_baseline",
    )
    periodic, _ = _run(
        tmp_path,
        paths,
        f"get_beta({_model(3, half_life=0, recompute_every=4)})",
        instruments=9,
        label="stateless_periodic",
    )
    np.testing.assert_allclose(
        periodic,
        _hold_refresh_rows(baseline, 4),
        rtol=5e-10,
        atol=5e-10,
    )


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_recompute_every_requires_positive_integer(value: float) -> None:
    with pytest.raises(FormulaIRCompileError, match="recompute_every"):
        compile_ir(
            "get_beta(Ridge(x, y=y, hl=8, lambda_=0.1, "
            f"recompute_every={value}))"
        )
