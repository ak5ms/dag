from __future__ import annotations

import numpy as np
from numba import boolean, float64
from numba.experimental import jitclass

from .compiler import compile_formula
from .dsl import DSLFunctionRegistry


def _pack_tick(engine, data: dict[str, np.ndarray]) -> np.ndarray:
    names = engine.compiled.input_names
    frame = np.empty((len(names), data[names[0]].shape[0]), dtype=np.float64)
    for i in range(len(names)):
        frame[i] = data[names[i]]
    return frame


def pack_cube(engine, data: dict[str, np.ndarray], start: int = 0, stop: int | None = None) -> np.ndarray:
    names = engine.compiled.input_names
    end = data[names[0]].shape[0] if stop is None else stop
    stacked = [np.asarray(data[names[i]][start:end], dtype=np.float64) for i in range(len(names))]
    return np.stack(stacked, axis=1)


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
    names = engine.compiled.input_names
    t = data[names[0]].shape[0]

    if out is None:
        frame0 = _pack_tick(engine, {n: data[n][0] for n in names})
        engine.on_data(frame0)
        y0 = engine.emit()
        out = np.empty((t, y0.shape[0], y0.shape[1]), dtype=np.float64)
        out[0] = y0
        start = 1
    else:
        start = 0

    for i in range(start, t, chunk_size):
        j = min(t, i + chunk_size)
        cube = pack_cube(engine, data, i, j)
        engine.run_batch(cube, out[i:j])
    return out


def build_engine(formula: str, dsl_registry: DSLFunctionRegistry | None = None):
    compiled_artifact = compile_formula(formula, dsl_registry=dsl_registry)

    spec = [
        ("compiled", compiled_artifact.compiled_type),
        ("initialized", boolean),
        ("last", float64[:, :]),
    ]

    @jitclass(spec)
    class EngineArtifact:  # noqa: N801
        def __init__(self, compiled):
            self.compiled = compiled
            self.initialized = False
            self.last = np.empty((1, 1), dtype=np.float64)

        def _copy_last(self, y):
            if not self.initialized or self.last.shape[0] != y.shape[0] or self.last.shape[1] != y.shape[1]:
                self.last = np.empty((y.shape[0], y.shape[1]), dtype=np.float64)
                self.initialized = True
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    self.last[i, j] = y[i, j]

        def on_data(self, frame2d):
            self.compiled.on_data(frame2d)
            self._copy_last(self.compiled.emit())

        def emit(self):
            return self.last

        def run_batch(self, cube3d, out3d):
            for t in range(cube3d.shape[0]):
                self.compiled.on_data(cube3d[t])
                y = self.compiled.emit()
                self._copy_last(y)
                for i in range(y.shape[0]):
                    for j in range(y.shape[1]):
                        out3d[t, i, j] = y[i, j]
            return out3d

    return EngineArtifact(compiled_artifact.compiled)
