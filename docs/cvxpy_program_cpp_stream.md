# Direct CVXPY/Clarabel programs inside `cpp_stream`

`cpp_stream.optimizer.cvxpy_program` turns a static-shape, DPP-compliant
CVXPY problem factory into a normal callable in the formula DSL. The numerical
row path calls compact generated affine maps and Clarabel's C ABI directly; it
does not invoke Python or CVXPY and has no CVXPYgen dependency.

## Ownership boundary

CVXPY owns DCP/DPP validation and the initial cone canonicalization. The native
compiler then:

- shards the user parameter vector into at most 512 symbolic scalars per pass;
- extracts CVXPY's sparse affine maps without ever constructing the full DPP
  parameter/variable tensor;
- merges the shards into compact CSR maps for `P/q/A/b` and the objective
  offset;
- applies CVXPY's signed cone-row permutation directly to sparse coordinates;
- derives dirty-block metadata, fixed CSC sparsity, cone layout, and direct
  primal/dual spans; and
- renders one header that invokes Clarabel's C API without a generated wrapper.

The generated C++ class changes only runtime ownership and solver lifetime:

- mutable parameter, canonical, solution, and diagnostic buffers are class
  members;
- immutable sparse maps, CSC index arrays, and cone descriptors are shared
  `inline static` data;
- the first `solve()` constructs Clarabel;
- later `solve()` calls canonicalize only dirty blocks and update fixed-sparsity
  data through
  `clarabel_DefaultSolver_update_P/A/q/b`;
- the destructor frees the solver; and
- copy and move are disabled so a solver workspace cannot be duplicated
  accidentally.

This permits one generated object per cpp_stream operator state or independent
worker, with no global mutable generated solver state and no lock around a
shared Clarabel object.

## Problem-factory DSL

Install the optional Python code-generation dependencies:

```bash
python -m pip install -e '.[optimizer]'
```

At formula compile time each function argument is a small descriptor exposing
its concrete CVXPY shape. The function explicitly declares `cp.Parameter`
objects, so attributes such as `nonneg=True` live next to the CVXPY model
rather than in a second decorator dictionary.
The direct emitter currently accepts ordinary real parameters and sign
attributes (`nonneg`, `nonpos`, `pos`, or `neg`); dimension-reducing parameter
attributes such as `sparsity`, `diag`, `symmetric`, and `PSD`, and parameter
bounds, fail generation explicitly rather than producing a wrong ABI.
Primal fields likewise require ordinary CVXPY variables with a direct canonical
offset; dimension-reducing variable attributes such as `symmetric=True` are
rejected until an allocation-free inverse map is implemented.
Consequently the decorated function is the sole public binding and generation
boundary:

```python
import cvxpy as cp

from trading_dsl_engine.base.dsl import var
from trading_dsl_engine.cpp_stream.optimizer import (
    cvxpy_program,
    get_field,
    previous_solution,
)

@cvxpy_program(sequential=None)
def MPO(
    expected_returns,
    half_spread_bps,
    current_weights,
    risk_factor,
    risk_radius=0.08,
) -> cp.Problem:
    horizons, assets = expected_returns.shape
    expected_returns = cp.Parameter(
        expected_returns.shape, name="expected_returns"
    )
    half_spread_bps = cp.Parameter(
        half_spread_bps.shape,
        name="half_spread_bps",
        nonneg=True,
    )
    current_weights = cp.Parameter(
        (assets,), name="current_weights"
    )
    risk_factor = cp.Parameter(risk_factor.shape, name="risk_factor")
    risk_radius = cp.Parameter(name="risk_radius", nonneg=True)
    weights = cp.Variable((horizons, assets), name="weights")
    turnover = cp.Variable((horizons, assets), name="turnover")
    delta = weights - cp.vstack([current_weights, weights[:-1]])

    turnover_limit = turnover >= delta
    turnover_limit.set_label("turnover_limit")
    risk_limit = cp.SOC(risk_radius, risk_factor @ weights[0])
    risk_limit.set_label("risk_limit")
    return cp.Problem(
        cp.Minimize(
            -cp.sum(cp.multiply(expected_returns, weights))
            + cp.sum(cp.multiply(half_spread_bps * 1e-4, turnover))
        ),
        [turnover_limit, turnover >= -delta, risk_limit],
    )

mpo = MPO(
    expected_returns=var("expected_returns"),
    half_spread_bps=var("half_spread_bps"),
    current_weights=previous_solution("weights[0]", initial=0.0),
    risk_factor=var("risk_factor"),
    risk_radius=0.08,
)
next_weights = get_field(mpo, "weights[0]")
risk_lagrangian = get_field(mpo, "risk_limit.lagrangian")
solver_iterations = get_field(mpo, "iterations")
```

Concrete instrument count and fixed widths specialize the cached sub-program.
For multi-dimensional DSL values, the adapter accounts for CVXPY's
column-major parameter ABI, so an `(assets, horizons)` DSL value is presented to
the problem factory as `(horizons, assets)`. Every problem parameter must be
explicitly named after its function argument. Shape mismatches and missing or
extra parameters fail during IR construction.

### Prior-solution state and parallelism

`previous_solution("weights[0]", initial=0.0)` is an internal delayed edge. On
the first row, a scalar initializer broadcasts over `current_weights`; a
non-scalar initializer must match its logical shape. On every later row, the
generated node retrieves the prior solve's first-horizon weights directly from
its instance-owned result buffer and writes them into `current_weights` before
the next solve. It does not materialize history and it does not create a second
loop.

The decorator's `sequential` setting is tri-state:

- `None` (default): infer dependencies. `previous_solution()` makes the stage
  sequential; otherwise the complete DAG decides whether rows are independent.
- `True`: force ordered solves even without an explicit feedback edge.
- `False`: assert that the solver itself has no cross-row state dependency.
  Combining it with `previous_solution()` is a compile-time error.

`sequential=False` cannot override a real dependency elsewhere in the DAG. For
example, an MPO fed by stateful Ridge/EWM nodes still runs in temporal order.
An entirely independent DAG is row-parallel and gives each native worker its
own persistent generated object and Clarabel solver.

### Result fields

`get_field()` resolves fields at compile time. Unknown names fail compilation;
the row loop retrieves only the primal, dual, or info structures actually
projected by the formula.

| Request | Result |
|---|---|
| `weights`, `weights[0]` | named primal, or its CVXPY axis-0 slice |
| `constraint[2].dual` | dual for the third original constraint |
| `risk_limit.dual` | dual for a constraint labeled with `set_label()` |
| `risk_limit.lagrangian` | alias for the same dual |
| `risk_limit.value` | numeric constraint expression: signed residual for equality/inequality, `[t; x]` for SOC |
| `objective` | solver objective value |
| `iterations`, `status` | solver iteration count and status code |
| `primal_residual`, `dual_residual` | solver residual diagnostics |

Constraint-value projections add a requested-only auxiliary primal to the
sub-program. Merely requesting a dual or solver diagnostic does not change the
optimization problem.

### Cache boundary

The CVXPY/Clarabel sub-compilation is cached independently of the surrounding
formula. Its key covers the factory implementation, declared parameter shapes
and attributes, resulting problem structure, and enabled solver settings. Direct
bindings and prior-solution bindings therefore reuse the same solver artifact;
their source dtype/layout remains part of the surrounding native-runner cache.
The same MPO can be reused in another outer formula without regenerating its
native parameter maps. A requested constraint value gets a distinct entry only
because it changes the generated problem by adding the auxiliary projection
variable.

Compiling the complete outer DAG still matters for CSE, scheduling, native
link-cache invalidation, and placing upstream formulas, the solve, and downstream
consumers in one row loop. It does not make the isolated cone canonicalization
faster, so the MPO sub-program and the full fused runner use
separate cache layers.

## Parallel independent problems

For an embarrassingly parallel batch, allocate one generated object in each
native cpp_stream worker. Immutable generated maps are shared, while all mutable
buffers and Clarabel state remain worker-local:

```text
worker 0 -> GeneratedProblem instance 0 -> Clarabel solver 0
worker 1 -> GeneratedProblem instance 1 -> Clarabel solver 1
worker 2 -> GeneratedProblem instance 2 -> Clarabel solver 2
```

Do not share one instance between workers. Sequential portfolio problems that
carry prior weights remain one ordered state transition and must not be split
across rows.

## Performance

`benchmarks/clarabel_persistent_instances.md` covers 9–150 assets at eight
horizons in all-changing, objective-only, and bitwise-unchanged regimes. It
includes parameter change detection, any required copies and canonicalization,
Clarabel update/solve, requested-only projection, allocation wrapping, resident
growth, and 2/4/8-worker independent-problem throughput. The companion JSON
retains every raw timing and checksum.

The original CVXPYgen 1.0 route attempted a 34.895 GiB dense temporary while
signing a `186,900 × 25,059` affine map. A sparse patch removed that allocation,
but fresh 150-asset × 8-horizon generation still took 36.4–36.8 seconds, peaked
at 4.46 GiB RSS, and emitted a 26.28 MiB header. Extracting CVXPY's full sparse
map directly improved time but still peaked above 4 GiB because CVXPY formed a
large intermediate parameter/variable tensor.

The selected backend canonicalizes 512-scalar parameter shards, merges only
their sparse nonzeros, and applies cone formatting as a signed coordinate
permutation. In the complete project environment, including JAX 0.4.38, five
fresh guarded 150×8 generations had a 1.001-second median. Median process peak
was 405,068 KiB (395.6 MiB), of which only 40,132 KiB (39.2 MiB) was growth above
the already-loaded CVXPY/JAX/project baseline. The generated header was
5,449,354 bytes (5.20 MiB).

The audit rejects any individual `numpy.zeros` or SciPy sparse `toarray()`
allocation at or above 512 MiB; all five runs recorded zero rejected attempts.
Using the conservative absolute process peak, this is a 97.3% generation-time
reduction and a 91.3% peak-RSS reduction versus the patched CVXPYgen route.
`scripts/audit_clarabel_sparse_generation.py` reproduces it;
`CLARABEL_AUDIT_PARAMETER_SHARD_SIZE` tunes the compile-time time/memory
tradeoff. The generator collects cyclic CVXPY objects once after all shards,
releases shard graphs through normal reference counting, avoiding explicit
full-GC scans of CVXPY's eagerly imported optional solver modules. Ten repeated
generations reached a stable resident-memory plateau rather than growing.

## One temporal loop with upstream and downstream formulas

Calling a decorated problem factory creates an object-valued IR node rather than
a batch callback. Its bound parameters remain ordinary DAG expressions. During
lowering, all requested `get_field()` projections from the same object are
collected into one physical native optimizer node, which is placed among the
normal cpp_stream stages.
Consequently the generated runner executes, for each row:

```text
Ridge/EWM/risk-model updates
    -> native PSD factor
    -> compact parameter maps
    -> persistent Clarabel update/solve
    -> requested result projections
    -> downstream shift/PnL expressions
```

There is no historical parameter materialization and no optimizer-specific
second loop. `examples/cpp_stream_mpo_one_pass.py` asserts that the generated
translation unit contains exactly one temporal `for (t)` loop and one generated
optimizer stage even though it requests both weights and turnover.

Before copying a bound parameter, the native optimizer node bitwise-compares it
with the instance-owned parameter buffer. Unchanged parameters do not dirty
their canonical blocks. Primals and duals are direct zero-copy solution spans, and
only requested fields are projected. The pinned Clarabel build preserves timer
storage across resets; allocator
wrapping verifies zero `malloc`, `calloc`, `realloc`, or `aligned_alloc` calls
after warm-up.

The C++ instance and persistent-solve source are rendered from Jinja templates
under `cpp_stream/optimizer/templates`; Python code supplies structured template
context rather than assembling C++ files with handwritten print/string loops.
