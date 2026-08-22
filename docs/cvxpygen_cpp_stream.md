# CVXPYgen programs inside `cpp_stream`

`cpp_stream.optimizer.generate_clarabel_program` turns a static-shape,
DPP-compliant CVXPY problem into one C++ class suitable for formula-specific
native builds. The numerical row path calls CVXPYgen's generated parameter maps
and result maps directly; it does not invoke Python, CVXPY, or the CVXPYgen
Python wrapper.

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

## Compile-time use

Install the optional Python code-generation dependencies:

```bash
python -m pip install -e '.[optimizer]'
```

Provide a current Clarabel C header and static library. They may be built and
cached with `build_current_clarabel()`, or supplied explicitly when builds are
offline:

```python
from pathlib import Path

import cvxpy as cp

from trading_dsl_engine.cpp_stream.optimizer import (
    ClarabelNativePaths,
    generate_clarabel_program,
)

x = cp.Variable(3, name="x")
A = cp.Parameter((4, 3), name="A")
b = cp.Parameter(4, name="b")
problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

program = generate_clarabel_program(
    problem,
    code_dir=".generated/least_squares",
    clarabel=ClarabelNativePaths(
        Path("/opt/clarabel/include"),
        Path("/opt/clarabel/lib/libclarabel_c.a"),
    ),
    class_name="GeneratedLeastSquares",
    prefix="least_squares_",
)
```

The generated header exposes bulk, column-major parameter setters, a persistent
`solve()`, named primal views, and the normal CVXPYgen result object:

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

`benchmarks/cvxpygen_persistent_instances.md` measures changing covariance, so
canonical `A`, `q`, and `b` all change on every problem. The benchmark includes
bulk parameter copies, CVXPYgen canonicalization, Clarabel updates and solve,
and primal projection. It also measures 2- and 4-worker independent-problem
throughput and resident-memory growth.
