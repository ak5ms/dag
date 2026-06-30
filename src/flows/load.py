import glob
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_dsl_engine.jax_flat.engine import compile_formula


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
        numba_data = {
            fp.split('/')[-1].removesuffix('.npy'): np.load(fp, mmap_mode='r')[:self.nrows]
            for fp in fps
        }
        return numba_data

    def get_data(self):
        fps = sorted(glob.glob(self.fp))
        if not self.data:
            data = self._load_memmap(fps)
        self.idx = pd.to_datetime(pd.Series(data=data[self.idx][:, 0]).interpolate(), unit='us')
        self.data = data
        return data

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