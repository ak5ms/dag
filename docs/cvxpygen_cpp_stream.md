# CVXPYgen programs inside `cpp_stream`

`cpp_stream.optimizer.cvxpy_program` turns a static-shape, DPP-compliant
CVXPY problem factory into a normal callable in the formula DSL. The numerical
row path calls CVXPYgen's generated parameter maps and result maps directly; it
does not invoke Python, CVXPY, or the CVXPYgen Python wrapper.

## Ownership boundary

CVXPY and CVXPYgen continue to own:

- DCP/DPP validation;
- user-parameter to canonical `P/A/q/b` maps;
- dirty-block metadata;
- cone layout and fixed CSC sparsity;
- primal/dual inverse mappings; and
- generated solver settings and diagnostics.

The generated C++ class changes only runtime ownership and solver lifetime:

- mutable parameter, canonical, solution, and diagnostic buffers are class
  members;
- immutable sparse maps, CSC index arrays, and cone descriptors are shared
  `inline static` data;
- the first `solve()` constructs Clarabel;
- later `solve()` calls update the dirty fixed-sparsity blocks through
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
objects, so attributes such as `nonneg=True`, sparsity, bounds, and symmetry
live next to the CVXPY model rather than in a second decorator dictionary.
Consequently the same function name binds streaming expressions; there is no
separate `bind_program()` or `generate_clarabel_program()` call:

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

The CVXPYgen sub-compilation is cached independently of the surrounding formula.
Its key covers the factory implementation, declared parameter shapes and
attributes, resulting problem structure, and enabled solver settings. Direct
bindings and prior-solution bindings therefore reuse the same solver artifact;
their source dtype/layout remains part of the surrounding native-runner cache.
The same MPO can be reused in another outer formula without regenerating its
native parameter maps. A requested constraint value gets a distinct entry only
because it changes the generated problem by adding the auxiliary projection
variable.

Compiling the complete outer DAG still matters for CSE, scheduling, native
link-cache invalidation, and placing upstream formulas, the solve, and downstream
consumers in one row loop. It does not make CVXPYgen's isolated DPP
canonicalization faster, so the MPO sub-program and the full fused runner use
separate cache layers.

## Explicit generation API

`generate_clarabel_program()` remains available when a generated solver class
is needed outside the formula DSL. Provide a current Clarabel C header and
static library explicitly, or use the allocation-free pinned build returned by
`build_current_clarabel()`.

```python
x = cp.Variable(3, name="x")
A = cp.Parameter((4, 3), name="A")
b = cp.Parameter(4, name="b")
problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))
program = generate_clarabel_program(
    problem,
    code_dir=".generated/least_squares",
    clarabel=build_current_clarabel(),
    class_name="GeneratedLeastSquares",
    prefix="least_squares_",
)
```

The generated header exposes bulk, column-major parameter setters, a persistent
`solve()`, lazy named primal/dual/info views, and the normal CVXPYgen result
object:

```cpp
#include "cpg_instance.hpp"

GeneratedLeastSquares solver;
solver.set_A(a_values);
solver.set_b(b_values);
solver.solve();
auto x = solver.primal_x();
auto const& info = solver.result().info;
```

`GeneratedCvxpygenProgram.build_shared_kwargs()` supplies the include, link, and
cache-fingerprint arguments required by cpp_stream's existing native build:

```python
from trading_dsl_engine.cpp_stream.python.compiler_support import build_shared

library, source = build_shared(
    generated_translation_unit,
    **program.build_shared_kwargs(),
)
```

The cache key includes every generated public header, the generated manifest,
and the linked Clarabel archive. Changing the CVXPY problem, generated ABI, or
native solver therefore produces a new compiled artifact.

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

`benchmarks/cvxpygen_persistent_instances.md` covers 9–150 assets at eight
horizons in all-changing, objective-only, and bitwise-unchanged regimes. It
includes parameter change detection, any required copies and canonicalization,
Clarabel update/solve, requested-only projection, allocation wrapping, resident
growth, and 2/4/8-worker independent-problem throughput. The companion JSON
retains every raw timing and checksum.

CVXPYgen 1.0's original canonicalizer had two dense-materialization sites. Its
`_update_to_dense_mapping()` constructed a sparse LIL matrix from
`np.zeros(dense_shape)`, and `_process_canonical_parameters()` evaluated
`affine_map.mapping.toarray() * affine_map.sign`. The latter requested about
34.9 GiB for the 150-asset, eight-horizon benchmark: the canonical `A` affine
map had shape `186,900 × 25,059`, or 37,468,216,800 float64 bytes if dense. The
version-pinned adapter
now scatters COO/CSR nonzeros directly into their destination rows and applies
signs in place to CSR `data` slices. Neither path calls `np.zeros(dense_shape)`,
`toarray()`, nor `todense()`.

The fresh 150×8 audit completed with a 4.46 GiB peak RSS, so the 34.9 GiB dense
array was not resident or lazily retained. That remaining generation-time peak
comes from CVXPYgen's canonical and code-writer structures; it is separate from
the native row hot path, whose allocation wrappers remain at zero after warm-up.
`scripts/audit_cvxpygen_sparse_generation.py` reproduces the guarded generation.

## One temporal loop with upstream and downstream formulas

Calling a decorated problem factory creates an object-valued IR node rather than
a batch callback. Its bound parameters remain ordinary DAG expressions. During
lowering, all requested `get_field()` projections from the same object are
collected into one physical `CvxpygenNode`, which is placed among the normal
cpp_stream stages.
Consequently the generated runner executes, for each row:

```text
Ridge/EWM/risk-model updates
    -> native PSD factor
    -> CVXPYgen parameter maps
    -> persistent Clarabel update/solve
    -> requested result projections
    -> downstream shift/PnL expressions
```

There is no historical parameter materialization and no optimizer-specific
second loop. `examples/cpp_stream_mpo_one_pass.py` asserts that the generated
translation unit contains exactly one temporal `for (t)` loop and one generated
optimizer stage even though it requests both weights and turnover.

Before copying a bound parameter, `CvxpygenNode` bitwise-compares it with the
instance-owned parameter buffer. Unchanged parameters do not dirty their
canonical blocks. Result inverse maps are likewise lazy and requested-field
only. The pinned Clarabel build preserves timer storage across resets; allocator
wrapping verifies zero `malloc`, `calloc`, `realloc`, or `aligned_alloc` calls
after warm-up.

The C++ instance and persistent-solve source are rendered from Jinja templates
under `cpp_stream/optimizer/templates`; Python code supplies structured template
context rather than assembling C++ files with handwritten print/string loops.
