from pathlib import Path
import importlib
import importlib.util
import subprocess
import sys


def _ensure_eigen_nnqp() -> None:
    module_name = __name__ + "._eigen_nnqp"
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError as exc:
        if exc.name not in {"_eigen_nnqp", module_name}:
            raise
    if importlib.util.find_spec("setuptools") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "setuptools", "wheel"], check=True)
    root = Path(__file__).resolve().parents[4]
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=root, check=True)
    importlib.invalidate_caches()
    importlib.import_module(module_name)


_ensure_eigen_nnqp()

from trading_dsl_engine.jax_ffi.nnqp.solver import nnqp, nnqp_raw, solve_direct

__all__ = ["nnqp", "nnqp_raw", "solve_direct"]
