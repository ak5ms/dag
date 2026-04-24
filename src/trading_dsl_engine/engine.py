from __future__ import annotations

import numpy as np
from numba import boolean, float64, int64
from numba.experimental import jitclass
from numba.typed import List

from trading_dsl_engine.compiler import compile_formula
from trading_dsl_engine.dsl import DSLFunctionRegistry


def _pack_tick(engine, data: dict[str, np.ndarray]) -> np.ndarray:
    names = engine.compiled.input_names
    frame = np.empty((len(names), data[names[0]].shape[0]), dtype=np.float64)
    for i in range(len(names)):
        frame[i] = data[names[i]]
    return frame


def _as_aligned_inputs(engine, data: dict[str, np.ndarray]) -> List:
    names = engine.compiled.input_names
    inputs = List()
    for i in range(len(names)):
        arr = np.asarray(data[names[i]], dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input for '{names[i]}', got shape {arr.shape}")
        inputs.append(arr)
    return inputs


def _validate_aligned_inputs(inputs: List) -> tuple[int, int]:
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
    engine.on_data(frame)
    return engine.emit().copy()


def _alloc_output(engine, t: int, n_instruments: int):
    output_code = engine.compiled.output_code
    if output_code == 0:
        return np.empty(t, dtype=np.float64)
    if output_code == 1:
        return np.empty((t, n_instruments), dtype=np.float64)
    if output_code == 2:
        return np.empty((t, n_instruments, n_instruments), dtype=np.float64)
    raise ValueError(f"Unknown output code: {output_code}")


def run_batch_from_mapping(
    engine,
    data: dict[str, np.ndarray],
    out: np.ndarray | None = None,
    chunk_size: int = 8192,
):
    inputs = _as_aligned_inputs(engine, data)
    t, n_instruments = _validate_aligned_inputs(inputs)

    output_code = engine.compiled.output_code
    if out is None:
        out = _alloc_output(engine, t, n_instruments)

    if output_code == 0:
        if out.ndim != 1 or out.shape[0] != t:
            raise ValueError("Scalar output requires out.shape == (time,)")
    elif output_code == 1:
        if out.ndim != 2 or out.shape[0] != t or out.shape[1] != n_instruments:
            raise ValueError("Vector output requires out.shape == (time, n_instruments)")
    elif output_code == 2:
        if out.ndim != 3 or out.shape[0] != t or out.shape[1] != n_instruments or out.shape[2] != n_instruments:
            raise ValueError("Matrix output requires out.shape == (time, n_instruments, n_instruments)")
    else:
        raise ValueError(f"Unknown output code: {output_code}")

    for i in range(0, t, chunk_size):
        j = min(t, i + chunk_size)
        if output_code == 0:
            engine.run_batch_scalar_aligned(inputs, out, i, j)
        elif output_code == 1:
            engine.run_batch_vector_aligned(inputs, out, i, j)
        else:
            engine.run_batch_matrix_aligned(inputs, out, i, j)
    return out


def build_engine(formula: str, dsl_registry: DSLFunctionRegistry | None = None):
    compiled_artifact = compile_formula(formula, dsl_registry=dsl_registry)

    spec = [
        ("compiled", compiled_artifact.compiled_type),
        ("initialized", boolean),
        ("frame_initialized", boolean),
        ("last", float64[:, :]),
        ("frame", float64[:, :]),
    ]

    @jitclass(spec)
    class EngineArtifact:  # noqa: N801
        def __init__(self, compiled):
            self.compiled = compiled
            self.initialized = False
            self.frame_initialized = False
            self.last = np.empty((1, 1), dtype=np.float64)
            self.frame = np.empty((1, 1), dtype=np.float64)

        def _copy_last(self, y):
            if not self.initialized or self.last.shape[0] != y.shape[0] or self.last.shape[1] != y.shape[1]:
                self.last = np.empty((y.shape[0], y.shape[1]), dtype=np.float64)
                self.initialized = True
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    self.last[i, j] = y[i, j]

        def _ensure_frame(self, n_inputs: int, n_instruments: int):
            if (not self.frame_initialized) or self.frame.shape[0] != n_inputs or self.frame.shape[1] != n_instruments:
                self.frame = np.empty((n_inputs, n_instruments), dtype=np.float64)
                self.frame_initialized = True

        def on_data(self, frame2d):
            self.compiled.on_data(frame2d)
            self._copy_last(self.compiled.emit())

        def emit(self):
            return self.last

        def _load_tick(self, inputs, t: int64):
            n_inputs = len(inputs)
            n_instruments = inputs[0].shape[1]
            self._ensure_frame(n_inputs, n_instruments)
            for k in range(n_inputs):
                source = inputs[k]
                for j in range(n_instruments):
                    self.frame[k, j] = source[t, j]

        def run_batch_scalar_aligned(self, inputs, out1d, start: int64, stop: int64):
            for t in range(start, stop):
                self._load_tick(inputs, t)
                self.compiled.on_data(self.frame)
                y = self.compiled.emit()
                self._copy_last(y)
                out1d[t] = y[0, 0]
            return out1d

        def run_batch_vector_aligned(self, inputs, out2d, start: int64, stop: int64):
            for t in range(start, stop):
                self._load_tick(inputs, t)
                self.compiled.on_data(self.frame)
                y = self.compiled.emit()
                self._copy_last(y)
                for i in range(y.shape[0]):
                    out2d[t, i] = y[i, 0]
            return out2d

        def run_batch_matrix_aligned(self, inputs, out3d, start: int64, stop: int64):
            for t in range(start, stop):
                self._load_tick(inputs, t)
                self.compiled.on_data(self.frame)
                y = self.compiled.emit()
                self._copy_last(y)
                for i in range(y.shape[0]):
                    for j in range(y.shape[1]):
                        out3d[t, i, j] = y[i, j]
            return out3d

    return EngineArtifact(compiled_artifact.compiled)
