# Native flat-runtime performance report

Measured on 2026-07-27 in the project Linux container (CPython 3.14, GCC `-O3`
with LTO and `-march=native`). Each throughput number is the median of repeated
runs over 150 instruments, 1,024 batch rows, and 2,000 steady-state ticks. State
is recreated between samples; construction and warm-up are excluded.

The baseline is commit `495919d`, before typed-plan and native-call changes. The
final measurement includes arithmetic kernel specialization, direct writes from
compatible root kernels into the caller's output, and a measured GIL-release
cutoff for short tick vectors.

| Graph | Baseline tick/s | Final tick/s | Tick change | Baseline batch rows/s | Final batch rows/s | Batch change |
|---|---:|---:|---:|---:|---:|---:|
| Elementwise chain | 185,817 | 222,319 | **+19.6%** | 261,002 | 349,022 | **+33.7%** |
| Chained EWM | 298,371 | 320,681 | **+7.5%** | 533,011 | 561,489 | **+5.3%** |
| Composite-key groupby | 53,437 | 114,186 | **+113.7%** | 41,687 | 135,082 | **+224.0%** |
| Ridge projection | 124,003 | 155,283 | **+25.2%** | 214,784 | 221,059 | **+2.9%** |

## Iterations

The first optimization iteration specialized the common arithmetic opcodes and
allowed compatible roots to write directly into `tick_into`/`run_batch_into`
destinations. The second iteration avoided release/reacquire overhead for tick
vectors below 1,024 instruments while retaining GIL release for large ticks and
the entire validated batch loop.

| Elementwise-chain stage | Tick/s | Change from prior | Batch rows/s | Change from prior |
|---|---:|---:|---:|---:|
| Pre-optimization typed runtime | 178,358 | — | 267,031 | — |
| Specialized kernels + direct root | 211,177 | +18.4% | 353,544 | +32.4% |
| GIL cutoff (final iteration) | 223,947 | +6.0% | 376,168 | +6.4% |

The final table above uses separate seven-sample runs, rather than selecting the
best iteration sample. Group scratch preallocation alone did not resolve the
groupby regression. Profiling the lookup showed that each instrument linearly
scanned the complete preallocated capacity—often 1,024 slots—even when only a
small number of slots were occupied. The final iteration replaced both lookup
and insertion scans with allocation-free open addressing over canonical hashes.
NaNs use one dedicated bit representation, and signed zeros hash identically,
matching the existing key equality contract. Against the immediately preceding
linear implementation, groupby improved 117.0% for ticks and 213.1% for batch.

| Final group-key workload | Tick/s | Batch rows/s | Python-visible steady allocations |
|---|---:|---:|---:|
| Random composite keys | 114,186 | 135,082 | 2 |
| High locality (one composite key) | 147,685 | 211,605 | 2 |
| Adversarial churn (128 × 8 key pattern) | 111,271 | 93,970 | 2 |

The remaining gap between locality and churn is now collision/key-comparison
work rather than a capacity-wide scan. The keyed index remains allocation-free
after state construction for all three measured patterns.

## Linux `perf`

Ubuntu's `linux-tools-generic` was installed. Its 6.8 perf binary runs on the
container's 6.12 host kernel when invoked directly. Software events and sampled
recording work; virtualized hardware events are reported as unsupported.

| Profile | Command | Result |
|---|---|---|
| Counter statistics | `perf stat -r 3 -e task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions,cache-references,cache-misses,branches,branch-misses -- ... --case groupby --rows 4096 --instruments 150 --ticks 10000 --runs 1` | 2,683.07 ms task clock (±1.32%), 1.089 CPUs, 0 context switches, 0 migrations, 56,274 page faults (±0.34%); hardware events unsupported |
| Call-graph recording | `perf record -e cpu-clock:u -F 999 -g --call-graph dwarf -o /tmp/perf-groupby.data -- ... --case groupby --rows 4096 --instruments 150 --ticks 50000 --runs 1` | 2,436 samples, zero lost, 20.047 MB recording; `Runtime::eval_group` was the largest resolved symbol at 16.17% |

On a Linux host with perf enabled, use the repository benchmark directly:

```bash
perf stat -r 5 -e cycles,instructions,cache-references,cache-misses,branches,branch-misses -- \
  python tests/trading_dsl_engine/jax_flat/benchmark_cpp_flat_plan.py \
  --case elementwise --rows 4096 --instruments 150 --ticks 10000 --runs 1
perf record -g --call-graph dwarf -- \
  python tests/trading_dsl_engine/jax_flat/benchmark_cpp_flat_plan.py \
  --case groupby --rows 4096 --instruments 150 --ticks 10000 --runs 1
perf report
```

Raw sample arrays remain in the JSON emitted by the benchmark CLI so medians,
variance, and outliers can be reviewed rather than relying on a single timing.
