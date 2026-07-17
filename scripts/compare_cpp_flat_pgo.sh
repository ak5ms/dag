#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
CXX_BIN="${CXX:-c++}"
PROFILE_DIR="${TRADING_DSL_ENGINE_CPP_PGO_DIR:-$ROOT/.pgo-data}"
ROWS="${ROWS:-5000000}"
COLS="${COLS:-9}"
RUNS="${RUNS:-6}"
WARMUPS="${WARMUPS:-1}"
CPU="${CPU:-}"

if ! printf '' | "$CXX_BIN" -dM -E -x c++ - 2>/dev/null | grep -q '^#define __GNUC__ '; then
    echo "error: the automated PGO flow currently requires a GCC-compatible compiler; set CXX accordingly" >&2
    exit 2
fi

bench_args=(--rows "$ROWS" --cols "$COLS" --runs "$RUNS" --warmups "$WARMUPS" --json)
train_args=(--rows "$ROWS" --cols "$COLS" --runs 1 --warmups 0 --json)
if [[ -n "$CPU" ]]; then
    bench_args+=(--cpu "$CPU")
    train_args+=(--cpu "$CPU")
fi

clean_native() {
    rm -rf build
    rm -f src/trading_dsl_engine/jax_flat/_cpp_flat*.so
}

build_native() {
    clean_native
    "$PYTHON" setup.py build_ext --inplace --force -v
}

rm -rf "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"

printf '\n== baseline build ==\n'
unset TRADING_DSL_ENGINE_CPP_PGO
build_native
"$PYTHON" scripts/benchmark_cpp_flat_pgo.py "${bench_args[@]}" | tee "$PROFILE_DIR/baseline.json"

printf '\n== profile-generate build and training ==\n'
export TRADING_DSL_ENGINE_CPP_PGO=generate
export TRADING_DSL_ENGINE_CPP_PGO_DIR="$PROFILE_DIR"
build_native
"$PYTHON" scripts/benchmark_cpp_flat_pgo.py "${train_args[@]}" | tee "$PROFILE_DIR/training.json"
if ! find "$PROFILE_DIR" -type f -name '*.gcda' -print -quit | grep -q .; then
    echo "error: training completed without producing GCC .gcda profile data" >&2
    exit 3
fi

printf '\n== profile-use build ==\n'
export TRADING_DSL_ENGINE_CPP_PGO=use
build_native
"$PYTHON" scripts/benchmark_cpp_flat_pgo.py "${bench_args[@]}" | tee "$PROFILE_DIR/pgo.json"

"$PYTHON" - "$PROFILE_DIR/baseline.json" "$PROFILE_DIR/pgo.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

baseline = json.loads(Path(sys.argv[1]).read_text().splitlines()[-1])
pgo = json.loads(Path(sys.argv[2]).read_text().splitlines()[-1])
base_s = float(baseline["median_seconds"])
pgo_s = float(pgo["median_seconds"])
speedup = base_s / pgo_s
print("\n== comparison ==")
print(f"baseline_median_seconds={base_s:.9f}")
print(f"pgo_median_seconds={pgo_s:.9f}")
print(f"speedup={speedup:.6f}x")
print(f"time_reduction_percent={(1.0 - pgo_s / base_s) * 100.0:.3f}")
PY
