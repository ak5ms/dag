from __future__ import annotations

import tempfile

import numpy as np
from numba import boolean, float64, int64
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

    @property
    def _numba_type_(self):
        return self._engine._numba_type_

    def on_data(self, frame2d):
        return self._engine.on_data(frame2d)

    def emit(self):
        return self._engine.emit()

    def run_batch_scalar_aligned(self, inputs, out1d, start: int64, stop: int64):
        return self._engine.run_batch_scalar_aligned(inputs, out1d, start, stop)

    def run_batch_vector_aligned(self, inputs, out2d, start: int64, stop: int64):
        return self._engine.run_batch_vector_aligned(inputs, out2d, start, stop)

    def run_batch_matrix_aligned(self, inputs, out3d, start: int64, stop: int64):
        return self._engine.run_batch_matrix_aligned(inputs, out3d, start, stop)


def _pack_tick(engine, data: dict[str, np.ndarray]) -> np.ndarray:
    names = engine.input_names
    frame = np.empty((len(names), data[names[0]].shape[0]), dtype=np.float64)
    for i in range(len(names)):
        frame[i] = data[names[i]]
    return frame


def _as_aligned_inputs(engine, data: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    names = engine.input_names
    inputs = []
    for i in range(len(names)):
        arr = np.asarray(data[names[i]], dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input for '{names[i]}', got shape {arr.shape}")
        inputs.append(arr)
    return tuple(inputs)


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
    frame = _pack_tick(engine, data)
    engine.compiled.on_data(frame)
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


def _first_tick_frame(inputs: tuple[np.ndarray, ...]) -> np.ndarray:
    n_inputs = len(inputs)
    n_instruments = inputs[0].shape[1]
    frame = np.empty((n_inputs, n_instruments), dtype=np.float64)
    for k in range(n_inputs):
        frame[k, :] = inputs[k][0, :]
    return frame


def _probe_vector_output(engine, inputs: tuple[np.ndarray, ...]) -> np.ndarray:
    engine.compiled.on_data(_first_tick_frame(inputs))
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
    engine.compiled.on_data(_first_tick_frame(inputs))
    y = engine.compiled.emit()
    return y.shape[1]


def run_batch_from_mapping(
    engine,
    data: dict[str, np.ndarray],
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

    spec = [
        ("compiled", compiled_type),
        ("frame_initialized", boolean),
        ("frame", float64[:, :]),
    ]

    @jitclass(spec)
    class EngineArtifact:  # noqa: N801
        def __init__(self, compiled):
            self.compiled = compiled
            self.frame_initialized = False
            self.frame = np.empty((1, 1), dtype=np.float64)

        def _ensure_frame(self, n_inputs: int, n_instruments: int):
            if (not self.frame_initialized) or self.frame.shape[0] != n_inputs or self.frame.shape[1] != n_instruments:
                self.frame = np.empty((n_inputs, n_instruments), dtype=np.float64)
                self.frame_initialized = True

        def on_data(self, frame2d):
            self.compiled.on_data(frame2d)

        def emit(self):
            return self.compiled.emit()

        def _load_tick(self, inputs, t: int64):
            n_inputs = len(inputs)
            n_instruments = inputs[0].shape[1]
            self._ensure_frame(n_inputs, n_instruments)
            for k in range(n_inputs):
                source = inputs[k]
                self.frame[k, :] = source[t, :]

        def run_batch_scalar_aligned(self, inputs, out1d, start: int64, stop: int64):
            for t in range(start, stop):
                self._load_tick(inputs, t)
                self.compiled.on_data(self.frame)
                y = self.compiled.emit()
                out1d[t] = y[0, 0]
            return out1d

        def run_batch_vector_aligned(self, inputs, out2d, start: int64, stop: int64):
            for t in range(start, stop):
                self._load_tick(inputs, t)
                self.compiled.on_data(self.frame)
                y = self.compiled.emit()
                out2d[t, :] = y[:, 0]
            return out2d

        def run_batch_matrix_aligned(self, inputs, out3d, start: int64, stop: int64):
            for t in range(start, stop):
                self._load_tick(inputs, t)
                self.compiled.on_data(self.frame)
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
