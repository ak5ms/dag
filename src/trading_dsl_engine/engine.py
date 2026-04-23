from __future__ import annotations

import numpy as np

from .compiler import CompiledFormula, compile_formula


class StreamingFeatureEngine:
    def __init__(self, compiled: CompiledFormula):
        self.compiled = compiled
        self.feature = compiled.make_feature()
        self._frame = None

    @classmethod
    def from_formula(cls, formula: str) -> "StreamingFeatureEngine":
        return cls(compile_formula(formula))

    def update(self, data: dict[str, np.ndarray]) -> np.ndarray:
        if self._frame is None:
            first = data[self.compiled.input_names[0]]
            self._frame = np.empty((len(self.compiled.input_names), first.shape[0]), dtype=np.float64)
        for idx, name in enumerate(self.compiled.input_names):
            self._frame[idx] = data[name]
        self.feature.on_data(self._frame)
        return self.feature.emit().copy()

    def run_batch(self, data: dict[str, np.ndarray], out: np.ndarray | None = None) -> np.ndarray:
        t = data[self.compiled.input_names[0]].shape[0]
        # prime output shape from first tick
        first_tick = {k: data[k][0] for k in self.compiled.input_names}
        y0 = self.update(first_tick)
        if out is None:
            out = np.empty((t, *y0.shape), dtype=np.float64)
        out[0] = y0
        for i in range(1, t):
            tick = {k: data[k][i] for k in self.compiled.input_names}
            out[i] = self.update(tick)
        return out
