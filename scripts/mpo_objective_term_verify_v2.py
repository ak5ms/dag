from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("mpo_objective_term_verify.py")
spec = importlib.util.spec_from_file_location("mpo_objective_term_verify_base", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load {SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

original_replace_once = base.replace_once


def replace_once(path: str, old: str, new: str) -> None:
    if (
        path == "examples/cpp_stream_mpo_one_pass.py"
        and old == "    risk_factor_7,\n"
    ):
        target = base.ROOT / path
        text = target.read_text()
        signature = "    risk_factor_6,\n    risk_factor_7,\n    trade_allowed,\n"
        if text.count(signature) != 1:
            raise RuntimeError(
                f"{path}: expected exactly one risk_factor_7 signature anchor"
            )
        target.write_text(
            text.replace(
                signature,
                "    risk_factor_6,\n    trade_allowed,\n",
                1,
            )
        )
        return
    original_replace_once(path, old, new)


base.replace_once = replace_once
base.main()
