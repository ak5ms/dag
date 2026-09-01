# MPO Post-Solve Constraint Values and Session Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve `get_field(mpo, "constraint.value")` while removing requested constraint values from the Clarabel KKT system, skip sequential MPO solves on closed-session rows through `where(session_open, field, NaN)`, and expose solver status in the one-pass example.

**Architecture:** Compile the original CVXPY problem unchanged. Requested affine constraint expressions are canonicalized only in a separate compile-time evaluation problem, producing sparse maps that evaluate the expression from the original parameter vector and returned primal solution after Clarabel solves. A guarded optimizer projection is recognized in the IR for scalar `where(condition, get_field(...), NaN)` and lowered to a Clarabel node that does not load parameters, solve, or advance previous-solution state when the condition is false.

**Tech Stack:** Python 3.11, CVXPY, NumPy/SciPy sparse matrices, Jinja2-generated C++20, Clarabel C ABI, pytest, GitHub Actions.

**Spec:** User-approved design in the conversation preceding this plan.

## Global Constraints

- Keep the public `get_field(mpo, "risk_0.value")` interface unchanged.
- Do not add constraint-value variables or equalities to the problem passed to Clarabel.
- Preserve one temporal row loop and one bundled generated optimizer node.
- Closed rows must return NaN for guarded optimizer fields and must not advance `previous_solution()` state.
- Return `get_field(mpo, "status")` from the one-pass MPO example.
- Keep iterative refinement enabled globally; expose/disable it only through an explicit program setting if selected for this MPO.
- Retain the existing host-native Clarabel compiler flag.

---

### Task 1: Add failing behavioral tests

**Files:**
- Create: `tests/trading_dsl_engine/cpp_stream/test_cvxpy_constraint_values_and_guard.py`
- Create: `.github/workflows/mpo-postsolve-values-session-gate.yml`

**Interfaces:**
- Consumes: `cvxpy_program`, `get_field`, `previous_solution`, `compile_formula`, and DSL `where`.
- Produces: executable requirements for unchanged solver dimensions, post-solve value accuracy, lazy closed-row gating, preserved feedback state, and status output.

- [ ] Write a pure compilation test asserting that requesting an SOC value creates a `constraint_value` field and no `cpp_stream_constraint_value_*` primal.
- [ ] Write a native runtime test comparing the returned SOC vector with `[radius; factor @ weights]`.
- [ ] Write a sequential open/closed/open test asserting closed weights/status are NaN and the next open solve consumes the last open solution.
- [ ] Run the focused workflow and confirm the tests fail because post-solve fields and guarded projections are not implemented.

### Task 2: Compile constraint expressions outside the solver problem

**Files:**
- Modify: `src/trading_dsl_engine/cpp_stream/optimizer/factory.py`
- Modify: `src/trading_dsl_engine/cpp_stream/optimizer/clarabel_native.py`
- Modify: `src/trading_dsl_engine/cpp_stream/optimizer/direct_clarabel.py`
- Modify: `src/trading_dsl_engine/cpp_stream/optimizer/templates/direct_clarabel_instance.hpp.j2`

**Interfaces:**
- Produces: `ConstraintValueLayout`, `GeneratedClarabelProgram.constraint_values`, and `constraint_value<Index>()` spans.
- Consumes: existing sharded canonical maps and original generated parameter/primal layouts.

- [ ] Replace `_augment_constraint_values` with request collection that leaves the solver problem unchanged.
- [ ] Create compile-time auxiliary equality problems only for affine value expressions and extract sparse maps from parameters and original primal columns to requested values.
- [ ] Emit fixed sparse post-solve evaluators and evaluate them after each actual solve.
- [ ] Resolve `constraint[index].value` and labeled `.value` fields as `constraint_value`, including axis-zero indexing.
- [ ] Bump cache/manifest schemas and persist evaluator metadata.
- [ ] Run the focused tests until constraint-value tests pass.

### Task 3: Push scalar `where(..., optimizer_field, NaN)` into the optimizer node

**Files:**
- Modify: `src/trading_dsl_engine/ir/frontend.py`
- Modify: `src/trading_dsl_engine/cpp_stream/python/lowering.py`
- Modify: `src/trading_dsl_engine/cpp_stream/python/lowering_full.py`
- Modify: `src/trading_dsl_engine/cpp_stream/python/lowering_multi.py`
- Modify: `src/trading_dsl_engine/cpp_stream/python/codegen.py`
- Modify: `src/trading_dsl_engine/cpp_stream/cpp/stackdsl/ops/clarabel_program.hpp`

**Interfaces:**
- Produces: optional scalar guard source on a Clarabel stage.
- Behavior: false guard writes NaN to all projected outputs and returns before parameter loading, solve, feedback caching, or state mutation.

- [ ] Recognize only scalar `where(condition, CvxpyFieldExpr, NaN)` as a guarded optimizer projection; leave all other `where` expressions unchanged.
- [ ] Bundle projections only when they share the same program object and guard.
- [ ] Carry/remap the guard through lowering and code generation.
- [ ] Add `ConstraintValue` projection support and false-guard NaN projection in `ClarabelNode`.
- [ ] Run the open/closed/open test and confirm feedback state is preserved.

### Task 4: Update the one-pass MPO flow and documentation

**Files:**
- Modify: `examples/cpp_stream_mpo_one_pass.py`
- Modify: `docs/cvxpy_program_cpp_stream.md`
- Modify: `src/trading_dsl_engine/cpp_stream/AGENTS.md`

**Interfaces:**
- Produces: formula outputs `(returns, features, yhat_signals, weights, status, *risks)`.

- [ ] Build a scalar `session_open` from the instrument tradability mask.
- [ ] Wrap weights, status, and every risk value as `where(session_open, raw_field, NaN)`.
- [ ] Update result unpacking and diagnostics for the status output.
- [ ] Document post-solve evaluation and lazy guarded rows.

### Task 5: Verify and publish

**Files:**
- Test: focused optimizer and example tests.

- [ ] Run the focused native workflow with a freshly built host-native Clarabel library.
- [ ] Run existing optimizer unit/native tests and check generated code still has one row loop and one optimizer node.
- [ ] Remove the temporary workflow from the final target branch.
- [ ] Fast-forward `agent/mpo-inputdata-example` to the verified implementation commit.
- [ ] Confirm the target branch commit and CI result before reporting completion.
