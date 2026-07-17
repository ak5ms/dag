# Native C++ profile-guided optimization

The optional `jax_flat` C++ extension supports an opt-in GCC profile-guided optimization build. PGO is limited to `trading_dsl_engine.jax_flat._cpp_flat`; the unrelated NNQP extension keeps the normal native flags.

## Automated comparison

Run the full baseline, training, and profile-use sequence from the repository root:

```bash
CPU=0 bash scripts/compare_cpp_flat_pgo.sh
```

The default workload is 5,000,000 rows by 9 instruments and an eight-stateful-level alpha:

```text
ewm -> cumsum -> ewm -> shift -> ewm -> cumsum -> ewm -> shift
```

The script performs these phases:

1. Build the normal `-O3`/native/LTO extension and benchmark it.
2. Rebuild `_cpp_flat` with `-fprofile-generate`, then execute one full training pass.
3. Verify that GCC produced `.gcda` files.
4. Rebuild `_cpp_flat` with `-fprofile-use` and `-fprofile-correction`.
5. Benchmark the optimized extension and report the median speedup.

Useful overrides:

```bash
PYTHON=.venv/bin/python CXX=g++ CPU=3 RUNS=10 WARMUPS=2 \
  TRADING_DSL_ENGINE_CPP_PGO_DIR=/tmp/dag-pgo \
  bash scripts/compare_cpp_flat_pgo.sh
```

`ROWS` and `COLS` may be overridden for development, but the production training profile should use a representative workload. The default in-memory arrays require roughly 720 MB for one float64 input and one float64 output, excluding extension state and Python overhead. `scripts/benchmark_cpp_flat_pgo.py` also accepts `--input-memmap PATH` to keep the input disk-backed.

## Manual build phases

```bash
rm -rf .pgo-data build
rm -f src/trading_dsl_engine/jax_flat/_cpp_flat*.so

TRADING_DSL_ENGINE_CPP_PGO=generate \
TRADING_DSL_ENGINE_CPP_PGO_DIR="$PWD/.pgo-data" \
python setup.py build_ext --inplace --force -v

python scripts/benchmark_cpp_flat_pgo.py --rows 5000000 --cols 9 --runs 1 --warmups 0

rm -rf build
rm -f src/trading_dsl_engine/jax_flat/_cpp_flat*.so
TRADING_DSL_ENGINE_CPP_PGO=use \
TRADING_DSL_ENGINE_CPP_PGO_DIR="$PWD/.pgo-data" \
python setup.py build_ext --inplace --force -v
```

PGO data is tied to the compiler, binary layout, CPU target, and training workload. Regenerate it after meaningful native-code changes, compiler upgrades, or changes to the production formula mix. Do not commit `.pgo-data/`.
