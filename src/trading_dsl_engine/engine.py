from __future__ import annotations

import numpy as np
from numba import boolean, float64, int64
from numba.experimental import jitclass
from numba.typed import List

from .compiler import compile_formula
from .dsl import DSLFunctionRegistry


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


def run_batch_from_mapping(
    engine,
    data: dict[str, np.ndarray],
    out: np.ndarray | None = None,
    chunk_size: int = 8192,
) -> np.ndarray:
    inputs = _as_aligned_inputs(engine, data)
    t, _ = _validate_aligned_inputs(inputs)

    if out is None:
        frame0 = _pack_tick(engine, {name: data[name][0] for name in engine.compiled.input_names})
        engine.on_data(frame0)
        y0 = engine.emit()
        out = np.empty((t, y0.shape[0], y0.shape[1]), dtype=np.float64)
        out[0] = y0
        start = 1
    else:
        if out.shape[0] != t:
            raise ValueError("Output leading dimension must match input time dimension")
        start = 0

    for i in range(start, t, chunk_size):
        j = min(t, i + chunk_size)
        engine.run_batch_aligned(inputs, out, i, j)
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

        def run_batch_aligned(self, inputs, out3d, start: int64, stop: int64):
            n_inputs = len(inputs)
            n_instruments = inputs[0].shape[1]
            self._ensure_frame(n_inputs, n_instruments)

            for t in range(start, stop):
                for k in range(n_inputs):
                    row = inputs[k]
                    for j in range(n_instruments):
                        self.frame[k, j] = row[t, j]
                self.compiled.on_data(self.frame)
                y = self.compiled.emit()
                self._copy_last(y)
                for i in range(y.shape[0]):
                    for j in range(y.shape[1]):
                        out3d[t, i, j] = y[i, j]
            return out3d

    return EngineArtifact(compiled_artifact.compiled)
