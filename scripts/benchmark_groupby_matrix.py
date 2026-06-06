from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable

import jax
import numpy as np

from trading_dsl_engine.jax_flat import compile_formula


def _time(fn: Callable[[], object], runs: int) -> tuple[float, object]:
    best = float("inf")
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        best = min(best, time.perf_counter() - t0)
    return best, result


def _set_cpp_disabled(disabled: bool):
    old = os.environ.get("TRADING_DSL_ENGINE_DISABLE_CPP_ACCEL")
    if disabled:
        os.environ["TRADING_DSL_ENGINE_DISABLE_CPP_ACCEL"] = "1"
    else:
        os.environ.pop("TRADING_DSL_ENGINE_DISABLE_CPP_ACCEL", None)
    return old


def _restore_cpp_disabled(old):
    if old is None:
        os.environ.pop("TRADING_DSL_ENGINE_DISABLE_CPP_ACCEL", None)
    else:
        os.environ["TRADING_DSL_ENGINE_DISABLE_CPP_ACCEL"] = old


def _run_default(runtime, data):
    return runtime.run_batch(data)[1]


def _run_pure(runtime, data):
    old = _set_cpp_disabled(True)
    try:
        return runtime.run_batch(data)[1]
    finally:
        _restore_cpp_disabled(old)


def _formula(*, key_mode: str, univ: bool, lhs_kind: str, rhs_kind: str, rhs_nested: bool, composed_root: bool) -> str:
    key_name = "key_same" if key_mode == "all_same" else "key_mixed"
    key = f"(univ([0, 1, 2], [3, 4, 5, 6, 7, 8]), {key_name})" if univ else f"({key_name},)"
    lhs = "add(close, open)" if lhs_kind == "stateless_lhs" else "cumsum(close)"
    if rhs_kind == "stateless_rhs":
        rhs = "add(add(self_, 1.0), 2.0)" if rhs_nested else "add(self_, 1.0)"
    else:
        rhs = "cumsum(cumsum(self_))" if rhs_nested else "cumsum(self_)"
    group = f"groupby({key}, {lhs}, {rhs})"
    return f"add({group}, mul(open, 0.0))" if composed_root else group


def _make_data(rows: int, cols: int):
    rng = np.random.default_rng(42)
    close = rng.normal(size=(rows, cols)).astype(np.float64)
    open_ = rng.normal(size=(rows, cols)).astype(np.float64)
    row = np.arange(rows, dtype=np.float64)[:, None]
    col = np.arange(cols, dtype=np.float64)[None, :]
    # all_same has a uniform key across columns for each row, changing by run.
    key_same = np.broadcast_to((row // 32.0) % 128.0, (rows, cols)).copy()
    # key_mixed intentionally has repeated keys in some columns and different
    # keys in others, while remaining dense/integer for native-supported cases.
    key_mixed = ((row // 32.0) + np.mod(col, 3.0)) % 128.0
    return {"close": close, "open": open_, "key_same": key_same, "key_mixed": key_mixed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark groupby cartesian matrix default runtime vs pure JAX path.")
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--assert", dest="assert_outputs", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--key-mode", choices=("all_same", "mixed_cols"), action="append")
    parser.add_argument("--univ", choices=("0", "1"), action="append")
    parser.add_argument("--lhs-kind", choices=("stateless_lhs", "stateful_lhs"), action="append")
    parser.add_argument("--rhs-kind", choices=("stateless_rhs", "stateful_rhs"), action="append")
    parser.add_argument("--rhs-nested", choices=("0", "1"), action="append")
    parser.add_argument("--root", choices=("groupby_root", "composed_root"), action="append")
    args = parser.parse_args()

    key_modes = tuple(args.key_mode or ("all_same", "mixed_cols"))
    univ_modes = tuple(args.univ or ("0", "1"))
    lhs_kinds = tuple(args.lhs_kind or ("stateless_lhs", "stateful_lhs"))
    rhs_kinds = tuple(args.rhs_kind or ("stateless_rhs", "stateful_rhs"))
    rhs_nested_modes = tuple(args.rhs_nested or ("0", "1"))
    roots = tuple(args.root or ("groupby_root", "composed_root"))

    data = _make_data(args.rows, args.cols)
    print(
        "case,key_mode,univ,lhs_kind,rhs_kind,rhs_nested,root,default_s,pure_jax_s,default_over_pure,pure_over_default",
        flush=True,
    )
    case_i = 0
    for key_mode in key_modes:
        for univ_arg in univ_modes:
            univ = univ_arg == "1"
            for lhs_kind in lhs_kinds:
                for rhs_kind in rhs_kinds:
                    for rhs_nested_arg in rhs_nested_modes:
                        rhs_nested = rhs_nested_arg == "1"
                        for root in roots:
                            case_i += 1
                            formula = _formula(
                                key_mode=key_mode,
                                univ=univ,
                                lhs_kind=lhs_kind,
                                rhs_kind=rhs_kind,
                                rhs_nested=rhs_nested,
                                composed_root=root == "composed_root",
                            )
                            runtime = compile_formula(formula)
                            # Warm both paths outside timing so compile/import/first-use costs are excluded.
                            for _ in range(args.warmups):
                                jax.block_until_ready(_run_default(runtime, data))
                                jax.block_until_ready(_run_pure(runtime, data))
                            default_s, default_out = _time(lambda: _run_default(runtime, data), args.runs)
                            pure_s, pure_out = _time(lambda: _run_pure(runtime, data), args.runs)
                            if args.assert_outputs:
                                np.testing.assert_allclose(
                                    np.asarray(default_out),
                                    np.asarray(pure_out),
                                    rtol=args.rtol,
                                    atol=args.atol,
                                    equal_nan=True,
                                )
                            default_over_pure = default_s / pure_s if pure_s else float("nan")
                            pure_over_default = pure_s / default_s if default_s else float("nan")
                            print(
                                f"{case_i},{key_mode},{int(univ)},{lhs_kind},{rhs_kind},{int(rhs_nested)},{root},"
                                f"{default_s:.6f},{pure_s:.6f},{default_over_pure:.3f},{pure_over_default:.3f}",
                                flush=True,
                            )


if __name__ == "__main__":
    main()
