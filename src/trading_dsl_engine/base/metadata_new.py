"""Range analysis through JAX -> ONNX -> auto_LiRPA.

This module is intentionally narrow: it provides range finding for pure JAX
functions and exposes results through ``get_range()``.  It does not replace the
legacy unit/type metadata propagation in ``metadata.py`` yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Sequence

import jax.numpy as jnp
import numpy as np
import onnx
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from jax2onnx import to_onnx
from onnx2torch import convert

ArrayFn = Callable[..., jnp.ndarray]


@dataclass(frozen=True)
class ValueRange:
    """Scalar lower/upper output range."""

    lower: float
    upper: float

    def as_tuple(self) -> tuple[float, float]:
        return self.lower, self.upper


@dataclass(frozen=True)
class InputRange:
    """Input shape and elementwise lower/upper bounds for LiRPA analysis."""

    shape: tuple[int, ...]
    lower: float | np.ndarray
    upper: float | np.ndarray
    dtype: np.dtype = np.dtype(np.float32)

    def lower_array(self) -> np.ndarray:
        return _bound_array(self.lower, self.shape, self.dtype)

    def upper_array(self) -> np.ndarray:
        return _bound_array(self.upper, self.shape, self.dtype)

    def midpoint_array(self) -> np.ndarray:
        return (self.lower_array() + self.upper_array()) / np.asarray(2.0, dtype=self.dtype)


@dataclass(frozen=True)
class LiRPARangeMetadata:
    """Range-analysis result compatible with the ``get_range()`` convention."""

    range: ValueRange
    method: str
    onnx_path: Path | None = None

    def get_range(self) -> ValueRange:
        return self.range


def range_field(
    shape: Sequence[int],
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    dtype=np.float32,
) -> InputRange:
    """Convenience constructor for ``InputRange``."""

    return InputRange(tuple(int(dim) for dim in shape), lower, upper, np.dtype(dtype))


def analyze_jax_range(
    fn: ArrayFn,
    inputs: Sequence[InputRange],
    *,
    method: str = "IBP",
    keep_onnx_path: str | Path | None = None,
) -> LiRPARangeMetadata:
    """Compute a conservative output range for a JAX function with auto_LiRPA.

    Parameters
    ----------
    fn:
        Pure JAX callable.  It should accept one argument per ``InputRange`` and
        return a tensor output.
    inputs:
        Elementwise input bounds and concrete shapes used for JAX tracing and
        LiRPA perturbation construction.
    method:
        auto_LiRPA bound method, for example ``"IBP"`` or ``"CROWN"``.
    keep_onnx_path:
        Optional path where the intermediate ONNX model should be written and
        retained.  If omitted, a temporary ONNX file is used.
    """

    specs = tuple(inputs)
    if not specs:
        raise ValueError("analyze_jax_range requires at least one input range")
    if keep_onnx_path is None:
        with TemporaryDirectory() as tmpdir:
            return _analyze_jax_range_to_path(fn, specs, method, Path(tmpdir) / "range.onnx", None)
    target = Path(keep_onnx_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return _analyze_jax_range_to_path(fn, specs, method, target, target)


def _analyze_jax_range_to_path(
    fn: ArrayFn,
    specs: tuple[InputRange, ...],
    method: str,
    export_path: Path,
    retained_path: Path | None,
) -> LiRPARangeMetadata:
    example_inputs = [jnp.asarray(spec.midpoint_array()) for spec in specs]
    to_onnx(fn, inputs=example_inputs, return_mode="file", output_path=str(export_path))
    torch_model = convert(onnx.load(export_path)).eval()
    nominal = tuple(torch.from_numpy(spec.midpoint_array().astype(np.float32, copy=False)) for spec in specs)
    bounded_model = BoundedModule(torch_model, nominal, device="cpu")
    bounded_inputs = tuple(_bounded_tensor(value, spec) for value, spec in zip(nominal, specs))
    lower, upper = bounded_model.compute_bounds(x=bounded_inputs, method=method)
    lower_np = lower.detach().cpu().numpy()
    upper_np = upper.detach().cpu().numpy()
    return LiRPARangeMetadata(
        range=ValueRange(float(np.nanmin(lower_np)), float(np.nanmax(upper_np))),
        method=method,
        onnx_path=retained_path,
    )


def _bounded_tensor(nominal: torch.Tensor, spec: InputRange) -> BoundedTensor:
    perturbation = PerturbationLpNorm(
        norm=float("inf"),
        x_L=torch.from_numpy(spec.lower_array().astype(np.float32, copy=False)),
        x_U=torch.from_numpy(spec.upper_array().astype(np.float32, copy=False)),
    )
    return BoundedTensor(nominal, perturbation)


def _bound_array(value: float | np.ndarray, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape == ():
        return np.full(shape, array, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"Expected bound shape {shape}, got {array.shape}")
    return array


__all__ = [
    "InputRange",
    "LiRPARangeMetadata",
    "ValueRange",
    "analyze_jax_range",
    "range_field",
]
