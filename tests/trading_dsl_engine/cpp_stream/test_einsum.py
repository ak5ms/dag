from __future__ import annotations

from pathlib import Path

import numpy as np

from trading_dsl_engine.base.dsl import cat, einsum, var
from trading_dsl_engine.cpp_stream import compile_npy_formula


N = 5
ROWS = 11


def _data() -> dict[str, np.ndarray]:
    t = np.arange(ROWS, dtype=np.float64)[:, None]
    lane = np.arange(N, dtype=np.float64)[None, :]
    return {
        "w": 0.25 + 0.03 * t + 0.02 * lane,
        "x": 1.0 + 0.10 * t + 0.01 * lane,
        "y": 2.0 - 0.04 * t + 0.03 * lane,
        "z": -0.5 + 0.07 * t - 0.02 * lane,
    }


def _run(
    tmp_path: Path,
    formula,
    data: dict[str, np.ndarray],
    name: str,
) -> tuple[np.ndarray, object]:
    paths: dict[str, Path] = {}
    for index, input_name in enumerate(sorted(data)):
        path = tmp_path / f"{name}_{index}.npy"
        np.save(path, data[input_name])
        paths[input_name] = path
    runtime = compile_npy_formula(formula, paths, n_instruments=N)
    output_path = tmp_path / f"{name}.bin"
    runtime.run_npy_files(paths, out_path=output_path)
    output_shape = (ROWS,) + tuple(runtime.plan.output_shape)
    output = np.asarray(
        np.memmap(output_path, mode="r", dtype=np.float64, shape=output_shape)
    ).copy()
    return output, runtime


def _rowwise(subscripts: str, *arrays: np.ndarray, optimize=True) -> np.ndarray:
    values = [
        np.einsum(
            subscripts,
            *(array[row] for array in arrays),
            optimize=optimize,
        )
        for row in range(ROWS)
    ]
    return np.asarray(values)


def test_scalar_reduction_and_arbitrary_labels(tmp_path: Path) -> None:
    data = _data()
    output, runtime = _run(
        tmp_path,
        einsum("Q,Q->", var("x"), var("y")),
        {"x": data["x"], "y": data["y"]},
        "scalar_dot",
    )
    expected = _rowwise("Q,Q->", data["x"], data["y"])
    np.testing.assert_allclose(output, expected, rtol=1e-13, atol=1e-13)
    generated = runtime.generated_cpp.read_text()
    assert "BinaryEinsumNode" in generated
    assert "EinsumNfNfToNNode" not in generated


def test_legacy_trailing_subscript_outer_product(tmp_path: Path) -> None:
    data = _data()
    formula = einsum(var("x"), var("y"), "i,j->ij")
    output, _ = _run(
        tmp_path,
        formula,
        {"x": data["x"], "y": data["y"]},
        "outer",
    )
    expected = _rowwise("i,j->ij", data["x"], data["y"])
    np.testing.assert_allclose(output, expected, rtol=1e-13, atol=1e-13)


def test_ellipsis_feature_reduction(tmp_path: Path) -> None:
    data = _data()
    left = cat(var("x"), var("y"), var("z"))
    right = cat(var("z"), var("y"), var("x"))
    output, _ = _run(
        tmp_path,
        einsum("...j,...j->...", left, right),
        {"x": data["x"], "y": data["y"], "z": data["z"]},
        "ellipsis",
    )
    left_np = np.stack((data["x"], data["y"], data["z"]), axis=-1)
    right_np = np.stack((data["z"], data["y"], data["x"]), axis=-1)
    expected = _rowwise("...j,...j->...", left_np, right_np)
    np.testing.assert_allclose(output, expected, rtol=1e-13, atol=1e-13)


def test_unary_transpose_and_diagonal_tensor_slot(tmp_path: Path) -> None:
    data = _data()
    outer = einsum("i,j->ij", var("x"), var("y"))
    diagonal = einsum("ii->i", outer)
    output, runtime = _run(
        tmp_path,
        diagonal,
        {"x": data["x"], "y": data["y"]},
        "diagonal",
    )
    np.testing.assert_allclose(
        output, data["x"] * data["y"], rtol=1e-13, atol=1e-13
    )
    generated = runtime.generated_cpp.read_text()
    assert "UnaryEinsumNode" in generated
    assert "TensorSlotSrc" in generated

    matrix = cat(var("x"), var("y"), var("z"))
    transposed, _ = _run(
        tmp_path,
        einsum("ij->ji", matrix),
        {"x": data["x"], "y": data["y"], "z": data["z"]},
        "transpose",
    )
    expected = np.transpose(
        np.stack((data["x"], data["y"], data["z"]), axis=-1),
        (0, 2, 1),
    )
    np.testing.assert_allclose(transposed, expected, rtol=1e-13, atol=1e-13)


def test_implicit_scalar_output_and_scalar_operand(tmp_path: Path) -> None:
    data = _data()
    left = cat(var("x"), var("y"))
    right = cat(var("y"), var("z"))
    output, _ = _run(
        tmp_path,
        einsum("ij,ij", left, right),
        {"x": data["x"], "y": data["y"], "z": data["z"]},
        "implicit",
    )
    left_np = np.stack((data["x"], data["y"]), axis=-1)
    right_np = np.stack((data["y"], data["z"]), axis=-1)
    expected = _rowwise("ij,ij", left_np, right_np)
    np.testing.assert_allclose(output, expected, rtol=1e-13, atol=1e-13)

    scaled, _ = _run(
        tmp_path,
        einsum(",i->i", 2.5, var("x")),
        {"x": data["x"]},
        "scalar_operand",
    )
    np.testing.assert_allclose(
        scaled, 2.5 * data["x"], rtol=1e-13, atol=1e-13
    )


def test_nary_optimal_contraction_matches_numpy(tmp_path: Path) -> None:
    data = _data()
    left = cat(var("x"), var("y"))
    middle = cat(var("y"), var("z"))
    right = einsum("k,l->kl", var("z"), var("w"))
    formula = einsum(
        "ij,kj,kl->il",
        left,
        middle,
        right,
        optimize="optimal",
    )
    output, runtime = _run(tmp_path, formula, data, "nary")
    left_np = np.stack((data["x"], data["y"]), axis=-1)
    middle_np = np.stack((data["y"], data["z"]), axis=-1)
    right_np = _rowwise("k,l->kl", data["z"], data["w"])
    expected = _rowwise(
        "ij,kj,kl->il",
        left_np,
        middle_np,
        right_np,
        optimize="optimal",
    )
    np.testing.assert_allclose(output, expected, rtol=2e-13, atol=2e-13)
    stages = [stage for stage in runtime.plan.stages if stage.kind == "einsum"]
    assert len(stages) == 3
    assert runtime.plan.matrix_scratch_slots >= 2
