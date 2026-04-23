from __future__ import annotations

import numpy as np
from numba import boolean, float64
from numba.experimental import jitclass

from .compiler import compile_formula


def _pack_tick(engine, data: dict[str, np.ndarray]) -> np.ndarray:
    names = engine.compiled.input_names
    frame = np.empty((len(names), data[names[0]].shape[0]), dtype=np.float64)
    for i in range(len(names)):
        frame[i] = data[names[i]]
    return frame


def pack_cube(engine, data: dict[str, np.ndarray]) -> np.ndarray:
    names = engine.compiled.input_names
    stacked = [np.asarray(data[names[i]], dtype=np.float64) for i in range(len(names))]
    return np.stack(stacked, axis=1)


def update_from_mapping(engine, data: dict[str, np.ndarray]) -> np.ndarray:
    frame = _pack_tick(engine, data)
    engine.on_data(frame)
    return engine.emit().copy()


def run_batch_from_mapping(engine, data: dict[str, np.ndarray], out: np.ndarray | None = None) -> np.ndarray:
    cube = pack_cube(engine, data)
    if out is None:
        engine.on_data(cube[0])
        y0 = engine.emit()
        out = np.empty((cube.shape[0], y0.shape[0], y0.shape[1]), dtype=np.float64)
    return engine.run_batch(cube, out)


def build_engine(formula: str):
    compiled_artifact = compile_formula(formula)

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
