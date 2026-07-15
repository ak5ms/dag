# Alpha DAG pair-fusion benchmark

- JAX 0.9.0.1, CPU backend, float64
- 9 assets
- One warmup plus 10 measured runs per case; medians shown
- Compilation excluded
- Timings include execution, synchronization, fixed-size chunking, and NumPy output materialization

## Aggregate

| Group | Geometric-mean speedup | Median speedup |
|---|---:|---:|
| All cases | 1.76× | 1.75× |
| Stateless | 1.71× | 1.65× |
| Stateful | 2.00× | 1.86× |
| Mixed | 1.59× | 1.54× |
| Breadth 1 | 1.61× | 1.55× |
| Breadth 4 | 1.88× | 1.88× |
| Breadth 8 | 1.88× | 1.96× |

## 1,000,000 rows — breadth 1

| Formula | Depth | Before | After | Speedup | Strategy |
|---|---:|---:|---:|---:|---|
| Stateless | 3 | 0.1591 s | 0.1084 s | **1.47×** | `node_batch` |
| Stateless | 5 | 0.1169 s | 0.0881 s | **1.33×** | `node_batch` |
| Stateless | 8 | 0.1598 s | 0.0911 s | **1.75×** | `node_batch` |
| Stateful | 3 | 0.2197 s | 0.1444 s | **1.52×** | `pair_fused` |
| Stateful | 5 | 0.2872 s | 0.1678 s | **1.71×** | `pair_fused` |
| Stateful | 8 | 0.4291 s | 0.2021 s | **2.12×** | `pair_fused` |
| Mixed | 3 | 0.1675 s | 0.0891 s | **1.88×** | `node_batch` |
| Mixed | 5 | 0.2406 s | 0.1564 s | **1.54×** | `node_batch` |
| Mixed | 8 | 0.2861 s | 0.2196 s | **1.30×** | `node_batch` |

## 1,000,000 rows — breadth 4

| Formula | Depth | Before | After | Speedup | Strategy |
|---|---:|---:|---:|---:|---|
| Stateless | 3 | 0.5526 s | 0.3356 s | **1.65×** | `branch_batched` |
| Stateless | 5 | 0.5212 s | 0.2618 s | **1.99×** | `branch_batched` |
| Stateless | 8 | 0.6158 s | 0.3057 s | **2.01×** | `branch_batched` |
| Stateful | 3 | 0.9886 s | 0.5271 s | **1.88×** | `branch_batched_pair_fused` |
| Stateful | 5 | 1.3582 s | 0.7409 s | **1.83×** | `branch_batched_pair_fused` |
| Stateful | 8 | 1.9830 s | 1.0693 s | **1.85×** | `branch_batched_pair_fused` |
| Mixed | 3 | 0.6757 s | 0.3406 s | **1.98×** | `node_batch` |
| Mixed | 5 | 0.9046 s | 0.3582 s | **2.53×** | `branch_batched` |
| Mixed | 8 | 1.2024 s | 0.8559 s | **1.40×** | `node_batch` |

## 1,000,000 rows — breadth 8

| Formula | Depth | Before | After | Speedup | Strategy |
|---|---:|---:|---:|---:|---|
| Stateless | 3 | 1.1500 s | 0.7582 s | **1.52×** | `branch_batched` |
| Stateless | 8 | 1.4029 s | 0.5822 s | **2.41×** | `branch_batched` |
| Stateful | 3 | 1.9543 s | 0.7227 s | **2.70×** | `branch_batched_pair_fused` |
| Stateful | 8 | 4.1121 s | 1.4465 s | **2.84×** | `branch_batched_pair_fused` |
| Mixed | 3 | 1.4492 s | 1.0778 s | **1.34×** | `node_batch` |
| Mixed | 8 | 2.4254 s | 2.0991 s | **1.16×** | `node_batch` |

## 3,000,000 rows — depth 5, breadth 1

| Formula | Before | After | Speedup |
|---|---:|---:|---:|
| Stateless | 0.3935 s | 0.2631 s | **1.50×** |
| Stateful | 0.9096 s | 0.4886 s | **1.86×** |
| Mixed | 0.6222 s | 0.4000 s | **1.56×** |

## HLO and compiler temporary memory

Memory analysis uses one configured CPU chunk (4,096 rows for narrow stateful cases; 8,192 for depth-5 breadth-4). For breadth 4, the before value is the sum across four separately compiled formulas.

| Stateful case | HLO while ops | Temporary memory | Reduction |
|---|---:|---:|---:|
| Depth 3, breadth 1 | 3 → 2 | 0.563 MiB → 0.282 MiB | 50.0% |
| Depth 5, breadth 1 | 5 → 3 | 1.127 MiB → 0.563 MiB | 50.0% |
| Depth 8, breadth 1 | 8 → 4 | 1.971 MiB → 0.844 MiB | 57.2% |
| Depth 5, breadth 4 | 20 → 3 | 9.006 MiB → 5.063 MiB | 43.8% |

## Rejected full affine block-prefix path

For two nested EWMs at 1,000,000 × 9 with block size 512, the two-pass block-affine path was numerically exact but ran in 0.2939 s versus 0.1037 s for two sequential scans (0.35× relative speed). Compiler temporary memory increased from 4.501 MiB to 9.150 MiB. It is not selected on CPU.

## Implementation conclusion

- Fuse only eligible adjacent EWM pairs, reducing a depth-D chain to `ceil(D / 2)` scans.
- Pack homogeneous branches into a breadth axis.
- Retain vectorized stateless and specialized node-batch kernels.
- Keep fixed-shape CPU tiles and donated recurrence state.
- Do not use a full compound scan or two-pass associative block prefix by default for 9-asset CPU workloads.
