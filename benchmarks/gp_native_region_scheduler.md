# Native GP region scheduler benchmark

Validated on commit `975de60d090396767e3f1ef8771fbde861f44077` before this evidence-only commit.

## Host

- GitHub-hosted Ubuntu 24.04 runner
- AMD EPYC 7763
- 2 physical cores / 4 logical CPUs (SMT2)
- Python 3.11.16
- clang 18.1.3

## Workload

`scripts/benchmark_gp_native_scheduler.py` generated 12 deterministic random strongly typed GP formulas, wrapped each in the GP fitness graph, and compiled six bounded multi-output batches of two candidates each.

- rows: 150,000
- instruments: 9
- candidates: 12
- batches: 6
- measured runs: 3
- warmups: 1
- native workers: 4
- inner runtime threads: 1

Compilation was excluded from execution timings. The benchmark checked every Python-pool and native-pool result against serial output with `numpy.testing.assert_allclose` before reporting timings.

## Results

| Mode | Median wall time | Median busy cores | Speedup vs serial |
| --- | ---: | ---: | ---: |
| Serial | 0.565840 s | 0.999 | 1.000x |
| Python `ThreadPoolExecutor` | 0.327693 s | 2.302 | 1.727x |
| Native `run_many` scheduler | 0.323817 s | 2.297 | **1.747x** |

The native scheduler was **1.012x** faster than the existing Python-threaded baseline on this host. Its more important architectural result is that execution parallelism is owned by one native C++ worker pool rather than Python threads, while bounded multi-output batches preserve CSE and avoid nested pools.

Cold compilation of the six random batches took 21.8733 s. Recompiling the identical batches from the native cache took 0.2865 s.

## Interpretation

The hosted runner exposes four logical CPUs but only two physical cores. A median of 2.30 busy cores indicates physical-core saturation with a modest SMT contribution. The limiting speedup is therefore consistent with the host topology rather than a four-physical-core ceiling.

All six fitness programs were internally serial final-reduction plans. The improvement comes from scheduling independent GP DAG components concurrently. This implementation does not claim cache-tiled parallel execution across arbitrary temporal-to-cross-sectional boundaries inside one compiled runtime; existing row/lane planning remains the intra-runtime mechanism.

## Correctness validation

The final focused workflow passed 35 tests in 103.48 seconds, including:

- native batch output ordering and path validation;
- multi-output runtime correctness;
- GP search batching and walk-forward invariants;
- monotonic `roll_rets` behavior;
- 200 random GP IR fuzz samples; and
- 8 random GP native compile samples.
