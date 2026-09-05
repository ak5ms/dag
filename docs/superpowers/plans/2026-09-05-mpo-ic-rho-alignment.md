# MPO IC / rho alignment implementation plan

**Goal:** Match scratch feature/backtest semantics, fit volatility-scaled predictors
with canonical observation timing, and execute a causally correct gap-aware MPO.

**Architecture:** One generated temporal loop and one persistent Clarabel stage.
Canonical IC position/weight state is the fit-alignment authority; planned and
actual executed portfolios are separate state variables. Ordinary returns and
total reopening events use distinct calibration/risk policies.

**Tech stack:** Python DSL, cpp_stream C++/Eigen, CVXPY DPP, native Clarabel,
NumPy/Pandas numerical oracles, active JAX-flat Python/native parity tests.

**Specification and results:** `docs/mpo_ic_alignment.md`.

## Verified implementation steps

- [x] Pin base `834eebe165c2c6832c8328c76c00fa596ce815ae`; create a feature branch.
- [x] Reproduce feature-cleaning/span/rank mismatches against the supplied recipe.
- [x] Add canonical X/Y/W and beta-one ablations before replacing fit alignment.
- [x] Isolate missing-observation Ridge updates; synchronize all active kernels.
- [x] Reproduce temporary-AST identity aliasing; retain and scope memo ownership.
- [x] Separate current realized fills from future plans and verify return timing.
- [x] Test NaN control masks, invalid quotes, closures, and partial fills.
- [x] Model unclipped short/long gap risk and ordinary-only calibration separately.
- [x] Reproduce failed-solve state poisoning; quarantine unsuccessful feedback.
- [x] Bound duplicated lazy-expression work without changing operator semantics.
- [x] Run an estimated-volatility/learned-beta native weekend simulation and
      perturb future returns/masks/spreads to verify exact prefix causality.
- [x] Update README, AGENTS, regression tests and the investigation report.
- [ ] Verify the remotely published feature-branch tree and final targeted CI.
