import glob
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_dsl_engine.jax_flat.engine import compile_formula

_DATA_CACHE: dict[tuple[str, int | None], dict[str, np.ndarray]] = {}


def _close_memmap(arr: np.ndarray) -> None:
    base = getattr(arr, "base", None)
    if base is not None and getattr(base, "_mmap", None) is not None:
        base._mmap.close()


@dataclass
class InputData:
    fp: str = "/mnt/extra/qrt/data/aks_out3/*.npy"
    idx: str = "_ev_ts"
    nrows: int | float | None = None
    data: dict[str, np.ndarray] = None

    def __post_init__(self):
        if self.nrows:
            self.nrows = int(self.nrows)


    def _load_memmap(self, fps: list[str]) -> dict[str, np.ndarray]:
        cache_key = (self.fp, self.nrows)
        cached = _DATA_CACHE.get(cache_key)
        if cached is not None:
            return cached

        numba_data: dict[str, np.ndarray] = {}
        for fp in fps:
            key = fp.split("/")[-1].removesuffix(".npy")
            mmap = np.load(fp, mmap_mode="r")
            view = mmap if self.nrows is None else mmap[: self.nrows]
            if self.nrows is None:
                numba_data[key] = view
            else:
                numba_data[key] = np.array(view, copy=True)
                _close_memmap(mmap)
        _DATA_CACHE[cache_key] = numba_data
        return numba_data

    def get_data(self):
        fps = sorted(glob.glob(self.fp))
        if self.data is None:
            self.data = self._load_memmap(fps)
        if isinstance(self.idx, str):
            self.idx = pd.to_datetime(
                pd.Series(data=self.data[self.idx][:, 0]).interpolate(),
                unit="us",
            )
        return self.data

    def run(self, formula, cpp=True, runtimes=None) -> pd.DataFrame | np.ndarray:
        # TODO: attach cached values to call objects directly (careful about tracking memory)?
        if self.data is None:
            self.get_data()
        start = time.perf_counter()
        runtime = compile_formula(formula, cpp=cpp, runtimes=runtimes)
        out = runtime.run_batch(self.data)[-1]
        self.runtime = runtime
        end = time.perf_counter()
        print(end-start)
        if len(out.shape) <= 2 and len(out) == len(self.idx):
            out = pd.DataFrame(out, index=self.idx)
        return out