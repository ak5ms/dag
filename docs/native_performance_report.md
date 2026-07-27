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

## 29-feature alpha-sharpe workload

The workload builds 29 `xs_rank(ewm(pct_change(mid), span))` features and then
applies `default_alpha_pnl` to each before concatenating a `(time, instrument,
29)` result. At 150 instruments its unoptimized streaming program contains 473
nodes.

The largest initial problem was not the arithmetic: passing `out_path` disabled
automatic native acceleration entirely, despite `cpp=True`, and selected the
chunked JAX scan. A 512-row, 150-instrument JAX run did not complete within a
120-second limit. The corrected full-native path validates once and writes the
matrix root directly into the three-dimensional memmap. Its first run completes
in 0.608 seconds after 0.106 seconds of construction (843 rows/s), over 168×
faster end-to-end than the timed-out path's demonstrated upper bound.

Plan inspection found repeated independently-built denominator expressions.
Deterministic stateful CSE now merges transitions only when opcode, parameters,
and already-remapped children are identical; keyed group nodes remain unique.
Together with pure CSE and DCE this reduces the native plan from 473 to 220
nodes: 225 common subexpressions, including 56 stateful transitions, and 28 dead
nodes.

`xs_rank` was the next clear kernel bottleneck. Its first optimization moved
scratch and the full-finite score table into `State`, but the partial-finite path
still sorted values and then called `upper_bound` once per instrument. The
second optimization stores `(value, instrument)` pairs, sorts once, scans equal
value runs linearly, and scatters the upper-rank score. It therefore removes an
additional O(n log n) search phase while retaining the exact tie and nonfinite
semantics. Common vector/scalar arithmetic and `where` broadcasts remain
resolved outside their loops, and the emitted assembly contains AVX `vmulpd`
and `vdivpd` instructions.

| Stage (512 rows × 150 instruments) | Rows/s | Relative change |
|---|---:|---:|
| Old `cpp=True, out_path=...` route | <4.27 (120 s timeout) | — |
| First direct native memmap iteration | 595.7 | >139× |
| Preallocated-rank native plan, 7-run median | 848.9 | +42.5% vs first native |
| Single-sort rank kernel, 7-run median | 2,231.5 | +162.9% vs preallocated-rank plan |

At 2,048 rows (71.27 MB output), the new five-run samples were 2,232.6, 2,324.5,
2,328.5, 2,280.2, and 1,937.4 rows/s. The 2,280.2 median is 2.86× the former
798.5 rows/s median. Extrapolating only that measured compute rate, 525,600 rows
would take about 230.5 seconds and produce 18.29 GB. Actual storage performance
can change that result.

`perf stat -r 3` on the 2,048-row case measured 9,412.24 ms user task-clock
(±0.99%), 0.987 CPUs, zero context switches/migrations, and 79,848 page faults;
hardware counters remain unavailable in this VM. A 499 Hz `cpu-clock:u`
recording captured 3,691 samples without loss: generic `Runtime::eval_row` was
27.36%, `std::__introsort_loop` 6.10%, and the resolved binary division kernel
0.85%. After the single-sort rank change, `perf stat -r 3` fell to 4,254.14 ms
user task-clock (0.976 CPUs, ±0.72%) for the same three-process measurement;
hardware events are still unavailable in this VM. The remaining cost is spread
across the 220-node row transition and the 29 required sorts, rather than Python
timestep execution.

The post-change 499 Hz `cpu-clock:u` recording captured about 1,600 samples
without loss. `Runtime::eval_row` accounted for 16.55% and the pair
`std::__introsort_loop` for 10.06%; the removed per-instrument `upper_bound`
phase no longer appears as a separate hot function. Percentages include process
startup and should be used to locate work, while the warmed benchmark medians
above are the throughput comparison.

## Whole-graph redesign direction

The plan still represents the 29 parameterized alpha branches as 29 separate
scalar/vector pipelines. The highest-leverage next redesign is to detect sibling
subgraphs with identical topology but different static constants and **lift the
parameter dimension into a feature lane**. For this formula that means an
instrument-by-29 EWM/state matrix followed by lane-packed rank, shift, ffill,
where, multiply, and output kernels. Persistent state should use structure of
arrays, with the instrument index contiguous inside each lane; ephemeral values
should use liveness-colored aligned arenas and input/root views rather than
per-node buffers.

This changes dispatch complexity from roughly 220 node visits per row to a small
prebound sequence of family kernels, without changing the one canonical ordered
row transition. Stateful family kernels must still perform `on_data` before
`emit` lane-by-lane, and each lane retains its independent clocks and NaN state.
Ranking can parallelize across lanes above a measured cutoff because lanes are
independent, while time remains sequential. A persistent executor/thread pool is
required to avoid per-row launch overhead. The same typed plan can then support
an optional cached whole-island specialization that emits these family kernels
through a versioned C ABI; the generic plan remains the cold-start fallback.

The measurement gates for that redesign are separate: node/kernel dispatch,
rank sorting, state traffic, root write bandwidth, and cold compilation. The
current root alone writes 34,800 bytes per row (18.29 GB/year), so output
bandwidth is a hard floor and must be reported separately from compute time.
