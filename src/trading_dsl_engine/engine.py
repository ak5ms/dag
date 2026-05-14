from __future__ import annotations

import tempfile

import numpy as np
from numba import int64
from numba.experimental import jitclass

from trading_dsl_engine.compiler import compile_formula
from trading_dsl_engine.dsl import DSLFunctionRegistry
from trading_dsl_engine.parser import Expr


_ENGINE_CLASS_CACHE: dict[object, object] = {}


class EngineHandle:
    def __init__(self, engine, input_names, output_code):
        self._engine = engine
        self.compiled = self._engine.compiled
        self.input_names = input_names
        self.output_code = output_code
        self.input_schema = _input_schema(self.input_names)

    @property
    def _numba_type_(self):
        return self._engine._numba_type_

    def bind(self, **arrays: np.ndarray) -> tuple[np.ndarray, ...]:
        return _bind_inputs(self.input_names, arrays)

    def on_data(self, inputs, t: int = 0):
        return self._engine.on_data(inputs, t)

    def emit(self):
        return self._engine.emit()

    def run_batch_scalar_aligned(self, inputs, out1d, start: int64, stop: int64):
        return self._engine.run_batch_scalar_aligned(inputs, out1d, start, stop)

    def run_batch_vector_aligned(self, inputs, out2d, start: int64, stop: int64):
        return self._engine.run_batch_vector_aligned(inputs, out2d, start, stop)

    def run_batch_matrix_aligned(self, inputs, out3d, start: int64, stop: int64):
        return self._engine.run_batch_matrix_aligned(inputs, out3d, start, stop)


def _input_schema(input_names: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {"name": name, "index": i, "dtype": np.float64, "ndim": 2, "layout": "C"}
        for i, name in enumerate(input_names)
    )


def _validate_array(name: str, arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Input '{name}' must be a numpy.ndarray")
    if arr.dtype != np.float64:
        raise TypeError(f"Input '{name}' must have dtype float64, got {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D input for '{name}', got shape {arr.shape}")
    if not arr.flags.c_contiguous:
        raise ValueError(f"Input '{name}' must be C-contiguous for row-aligned batch execution")
    return arr


def _validate_bound_inputs(input_names: tuple[str, ...], inputs: tuple[np.ndarray, ...]) -> tuple[int, int]:
    if len(inputs) != len(input_names):
        raise ValueError(f"Expected {len(input_names)} input arrays, got {len(inputs)}")
    if len(inputs) == 0:
        raise ValueError("No input arrays provided")
    first = _validate_array(input_names[0], inputs[0])
    t = first.shape[0]
    n = first.shape[1]
    for i in range(1, len(inputs)):
        arr = _validate_array(input_names[i], inputs[i])
        if arr.shape[0] != t or arr.shape[1] != n:
            raise ValueError("All inputs must share aligned shape (time, n_instruments)")
    return t, n


def _bind_inputs(input_names: tuple[str, ...], arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    expected = set(input_names)
    provided = set(arrays)
    missing = tuple(name for name in input_names if name not in arrays)
    extra = tuple(sorted(provided - expected))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing inputs: {missing}")
        if extra:
            details.append(f"unexpected inputs: {extra}")
        raise ValueError("Input names do not match compiled schema (" + "; ".join(details) + ")")
    inputs = tuple(_validate_array(name, arrays[name]) for name in input_names)
    _validate_bound_inputs(input_names, inputs)
    return inputs


def _bind_tick(input_names: tuple[str, ...], arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    expected = set(input_names)
    provided = set(arrays)
    missing = tuple(name for name in input_names if name not in arrays)
    extra = tuple(sorted(provided - expected))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing inputs: {missing}")
        if extra:
            details.append(f"unexpected inputs: {extra}")
        raise ValueError("Input names do not match compiled schema (" + "; ".join(details) + ")")
    rows = []
    width = -1
    for name in input_names:
        arr = arrays[name]
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Input '{name}' must be a numpy.ndarray")
        if arr.dtype != np.float64:
            raise TypeError(f"Input '{name}' must have dtype float64, got {arr.dtype}")
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D tick input for '{name}', got shape {arr.shape}")
        if not arr.flags.c_contiguous:
            raise ValueError(f"Input '{name}' must be C-contiguous")
        if width < 0:
            width = arr.shape[0]
        elif arr.shape[0] != width:
            raise ValueError("All tick inputs must share instrument width")
        rows.append(arr.reshape(1, arr.shape[0]))
    return tuple(rows)


def _as_aligned_inputs(engine, data: dict[str, np.ndarray] | tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    if isinstance(data, tuple):
        _validate_bound_inputs(engine.input_names, data)
        return data
    return engine.bind(**data)


def _validate_aligned_inputs(inputs: tuple[np.ndarray, ...]) -> tuple[int, int]:
    if len(inputs) == 0:
        raise ValueError("No input arrays provided")
    t = inputs[0].shape[0]
    n = inputs[0].shape[1]
    for i in range(1, len(inputs)):
        if inputs[i].shape[0] != t or inputs[i].shape[1] != n:
            raise ValueError("All inputs must share aligned shape (time, n_instruments)")
    return t, n


def update_from_mapping(engine, data: dict[str, np.ndarray]) -> np.ndarray:
    inputs = _bind_tick(engine.input_names, data)
    engine.compiled.on_data(inputs, 0)
    return engine.compiled.emit()


def _alloc_output(engine, t: int, n_instruments: int):
    output_code = engine.output_code
    if output_code == 0:
        return np.empty(t, dtype=np.float64)
    if output_code == 1:
        return np.empty((t, n_instruments), dtype=np.float64)
    if output_code == 2:
        return np.empty((t, n_instruments, n_instruments), dtype=np.float64)
    raise ValueError(f"Unknown output code: {output_code}")


def _probe_vector_output(engine, inputs: tuple[np.ndarray, ...]) -> np.ndarray:
    engine.compiled.on_data(inputs, 0)
    y = engine.compiled.emit()
    return y[:, 0].copy()


def _output_shape(engine, t: int, n_instruments: int) -> tuple[int, ...]:
    output_code = engine.output_code
    if output_code == 0:
        return (t,)
    if output_code == 1:
        return (t, n_instruments)
    if output_code == 2:
        return (t, n_instruments, n_instruments)
    raise ValueError(f"Unknown output code: {output_code}")


def _alloc_memmap_output(engine, t: int, n_instruments: int, out_path: str):
    return np.memmap(out_path, mode="w+", dtype=np.float64, shape=_output_shape(engine, t, n_instruments))


def _probe_matrix_output(engine, inputs: tuple[np.ndarray, ...]) -> int:
    engine.compiled.on_data(inputs, 0)
    y = engine.compiled.emit()
    return y.shape[1]


def run_batch_from_mapping(
    engine,
    data: dict[str, np.ndarray] | tuple[np.ndarray, ...],
    out: np.ndarray | None = None,
    out_path: str | None = f"{tempfile.gettempdir()}/trading_dsl_engine_out.memmap",
    chunk_size: int = 8192,
):
    inputs = _as_aligned_inputs(engine, data)
    t, n_instruments = _validate_aligned_inputs(inputs)

    output_code = engine.output_code
    if output_code == 3:
        raise ValueError(
            "Root object outputs are not supported in batch mode. "
            "Project object state to scalar/vector/matrix via a downstream op."
        )

    inferred_vector_t0 = None
    inferred_vector_width = n_instruments
    inferred_matrix_width = n_instruments
    if output_code == 1:
        inferred_vector_t0 = _probe_vector_output(engine, inputs)
        inferred_vector_width = inferred_vector_t0.shape[0]
    elif output_code == 2:
        inferred_matrix_width = _probe_matrix_output(engine, inputs)

    if out is None:
        if out_path is None:
            if output_code == 1:
                out = np.empty((t, inferred_vector_width), dtype=np.float64)
            elif output_code == 2:
                out = np.empty((t, n_instruments, inferred_matrix_width), dtype=np.float64)
            else:
                out = _alloc_output(engine, t, n_instruments)
        else:
            if output_code == 1:
                out = np.memmap(out_path, mode="w+", dtype=np.float64, shape=(t, inferred_vector_width))
            elif output_code == 2:
                out = np.memmap(
                    out_path,
                    mode="w+",
                    dtype=np.float64,
                    shape=(t, n_instruments, inferred_matrix_width),
                )
            else:
                out = _alloc_memmap_output(engine, t, n_instruments, out_path)

    if output_code == 0:
        if out.ndim != 1 or out.shape[0] != t:
            raise ValueError("Scalar output requires out.shape == (time,)")
    elif output_code == 1:
        if out.ndim != 2 or out.shape[0] != t:
            raise ValueError("Vector output requires out.shape == (time, width)")
    elif output_code == 2:
        if out.ndim != 3 or out.shape[0] != t or out.shape[1] != n_instruments:
            raise ValueError("Matrix output requires out.shape == (time, n_instruments, width)")
    else:
        raise ValueError(f"Unknown output code: {output_code}")

    start_idx = 0
    if output_code == 1 and inferred_vector_t0 is not None:
        out[0, :] = inferred_vector_t0
        start_idx = 1

    for i in range(start_idx, t, chunk_size):
        j = min(t, i + chunk_size)
        if output_code == 0:
            engine.run_batch_scalar_aligned(inputs, out, i, j)
        elif output_code == 1:
            engine.run_batch_vector_aligned(inputs, out, i, j)
        else:
            engine.run_batch_matrix_aligned(inputs, out, i, j)
    return out


def _engine_class_for(compiled_type):
    cached = _ENGINE_CLASS_CACHE.get(compiled_type)
    if cached is not None:
        return cached

    spec = [("compiled", compiled_type)]

    @jitclass(spec)
    class EngineArtifact:  # noqa: N801
        def __init__(self, compiled):
            self.compiled = compiled

        def on_data(self, inputs, t: int64):
            self.compiled.on_data(inputs, t)

        def emit(self):
            return self.compiled.emit()

        def run_batch_scalar_aligned(self, inputs, out1d, start: int64, stop: int64):
            for t in range(start, stop):
                self.compiled.on_data(inputs, t)
                y = self.compiled.emit()
                out1d[t] = y[0, 0]
            return out1d

        def run_batch_vector_aligned(self, inputs, out2d, start: int64, stop: int64):
            for t in range(start, stop):
                self.compiled.on_data(inputs, t)
                y = self.compiled.emit()
                out2d[t, :] = y[:, 0]
            return out2d

        def run_batch_matrix_aligned(self, inputs, out3d, start: int64, stop: int64):
            for t in range(start, stop):
                self.compiled.on_data(inputs, t)
                y = self.compiled.emit()
                out3d[t, :, :] = y
            return out3d

    _ENGINE_CLASS_CACHE[compiled_type] = EngineArtifact
    return EngineArtifact

def build_engine(
    formula: str | Expr,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
):
    compiled_artifact = compile_formula(formula, dsl_registry=dsl_registry, column_names=column_names)
    engine_class = _engine_class_for(compiled_artifact.compiled_type)
    engine = engine_class(compiled_artifact.compiled)
    return EngineHandle(engine, compiled_artifact.input_names, compiled_artifact.compiled.output_code)
