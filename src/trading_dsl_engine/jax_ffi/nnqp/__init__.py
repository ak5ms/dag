from pathlib import Path
import importlib
import importlib.util

from trading_dsl_engine._native_build import ensure_native_extension_current


def _ensure_eigen_nnqp() -> None:
    module_name = __name__ + "._eigen_nnqp"
    spec = importlib.util.find_spec(module_name)
    extension = Path(spec.origin) if spec is not None and spec.origin is not None else None
    root = Path(__file__).resolve().parents[4]
    ensure_native_extension_current(root, "eigen_nnqp", extension)
    importlib.invalidate_caches()
    importlib.import_module(module_name)


_ensure_eigen_nnqp()

from trading_dsl_engine.jax_ffi.nnqp.solver import nnqp, nnqp_raw, solve_direct

__all__ = ["nnqp", "nnqp_raw", "solve_direct"]
