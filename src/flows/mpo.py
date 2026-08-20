from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from flows.riskmodel import covariance_diagnostics, sanitize_covariance


@dataclass(frozen=True)
class MPOProblem:
    P: sparse.csr_array
    q: np.ndarray
    A: sparse.csr_array
    b: np.ndarray
    num_nonneg_cones: int
    so_cone_dims: tuple[int, ...]
    expected_returns: np.ndarray
    covariance: np.ndarray
    half_spread: np.ndarray
    current_weights: np.ndarray
    risk_constraint: np.ndarray

    @property
    def n_horizons(self) -> int:
        return self.expected_returns.shape[0]

    @property
    def n_assets(self) -> int:
        return self.expected_returns.shape[1]

    @property
    def weight_size(self) -> int:
        return self.n_horizons * self.n_assets


@dataclass(frozen=True)
class MPOResult:
    weights: np.ndarray
    turnover: np.ndarray
    expected_return: np.ndarray
    transaction_cost: np.ndarray
    risk_variance: np.ndarray
    objective: float
    status: Any
    iterations: int
    solve_time: float
    setup_time: float
    construction_time: float
    solution: Any
    covariance_diagnostics: tuple[dict[str, float | bool], ...]


def _as_horizon_array(value, n_horizons: int, n_assets: int, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full((n_horizons, n_assets), float(arr), dtype=np.float64)
    elif arr.ndim == 1 and arr.shape == (n_assets,):
        arr = np.broadcast_to(arr, (n_horizons, n_assets)).copy()
    elif arr.shape != (n_horizons, n_assets):
        raise ValueError(f"{name} must be scalar, ({n_assets},), or ({n_horizons}, {n_assets})")
    return arr


def _as_risk_constraint(value, n_horizons: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n_horizons, float(arr), dtype=np.float64)
    elif arr.shape != (n_horizons,):
        raise ValueError(f"risk_constraint must be scalar or ({n_horizons},)")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("risk_constraint must be finite and strictly positive")
    return arr


def _clean_covariances(covariance, n_horizons: int, n_assets: int) -> np.ndarray:
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape == (n_assets, n_assets):
        cov = np.broadcast_to(cov, (n_horizons, n_assets, n_assets))
    elif cov.shape != (n_horizons, n_assets, n_assets):
        raise ValueError(
            f"covariance must have shape ({n_assets}, {n_assets}) or "
            f"({n_horizons}, {n_assets}, {n_assets})"
        )
    return np.stack([sanitize_covariance(cov[t]) for t in range(n_horizons)], axis=0)


def _covariance_factor(covariance: np.ndarray) -> np.ndarray:
    """Return C such that C.T @ C == covariance up to roundoff."""
    evals, evecs = np.linalg.eigh(covariance)
    if np.min(evals) <= 0.0:
        raise ValueError("sanitized covariance must be positive definite")
    return np.sqrt(evals)[:, None] * evecs.T


def build_mpo_problem(
    expected_returns,
    covariance,
    half_spread_bps,
    risk_constraint,
    *,
    current_weights=None,
) -> MPOProblem:
    """Build the Moreau conic form for multi-period turnover-aware optimization.

    Maximizes, over horizons t,

        expected_returns[t]' w[t]
        - half_spread' abs(w[t] - w[t-1])

    subject to ``w[t]' covariance[t] w[t] <= risk_constraint[t]``.

    Expected returns and covariance must use the same per-period return units.
    ``risk_constraint`` is therefore a *variance* limit, matching the expression
    above. ``half_spread_bps`` is converted from basis points to decimal return.
    """
    er = np.asarray(expected_returns, dtype=np.float64)
    if er.ndim == 1:
        er = er[None, :]
    if er.ndim != 2 or er.shape[0] == 0 or er.shape[1] == 0:
        raise ValueError("expected_returns must have shape (horizons, assets)")
    n_horizons, n_assets = er.shape
    if np.any(np.isinf(er)):
        raise ValueError("expected_returns cannot contain infinities")
    # A missing alpha means no expected-return contribution for that asset/horizon.
    er = np.nan_to_num(er, nan=0.0)

    hs_bps = _as_horizon_array(half_spread_bps, n_horizons, n_assets, name="half_spread_bps")
    if not np.all(np.isfinite(hs_bps)) or np.any(hs_bps < 0.0):
        raise ValueError("half_spread_bps must be finite and nonnegative")
    half_spread = hs_bps * 1e-4

    if current_weights is None:
        current = np.zeros(n_assets, dtype=np.float64)
    else:
        current = np.asarray(current_weights, dtype=np.float64)
        if current.shape != (n_assets,) or not np.all(np.isfinite(current)):
            raise ValueError(f"current_weights must be finite with shape ({n_assets},)")

    risk = _as_risk_constraint(risk_constraint, n_horizons)
    clean_cov = _clean_covariances(covariance, n_horizons, n_assets)
    factors = np.stack([_covariance_factor(clean_cov[t]) for t in range(n_horizons)], axis=0)

    # x = [w_0, ..., w_{H-1}, u_0, ..., u_{H-1}], where u >= |delta w|.
    nw = n_horizons * n_assets
    nvar = 2 * nw
    num_nonneg = 2 * nw
    soc_dims = tuple([n_assets + 1] * n_horizons)
    m = num_nonneg + sum(soc_dims)

    A = sparse.lil_array((m, nvar), dtype=np.float64)
    b = np.zeros(m, dtype=np.float64)

    def widx(t: int, i: int) -> int:
        return t * n_assets + i

    def uidx(t: int, i: int) -> int:
        return nw + t * n_assets + i

    row = 0
    for t in range(n_horizons):
        for i in range(n_assets):
            # w_t - w_{t-1} - u_t <= 0
            A[row, widx(t, i)] = 1.0
            A[row, uidx(t, i)] = -1.0
            if t == 0:
                b[row] = current[i]
            else:
                A[row, widx(t - 1, i)] = -1.0
            row += 1

            # -w_t + w_{t-1} - u_t <= 0
            A[row, widx(t, i)] = -1.0
            A[row, uidx(t, i)] = -1.0
            if t == 0:
                b[row] = -current[i]
            else:
                A[row, widx(t - 1, i)] = 1.0
            row += 1

    # SOC slack is [sqrt(risk variance), C @ w], so ||Cw||_2 <= sqrt(risk).
    for t in range(n_horizons):
        b[row] = np.sqrt(risk[t])
        row += 1
        w_slice = slice(t * n_assets, (t + 1) * n_assets)
        A[row:row + n_assets, w_slice] = -factors[t]
        row += n_assets

    if row != m:
        raise AssertionError("internal MPO constraint row mismatch")

    q = np.concatenate([-er.reshape(-1), half_spread.reshape(-1)])
    P = sparse.csr_array((nvar, nvar), dtype=np.float64)
    return MPOProblem(
        P=P,
        q=q,
        A=sparse.csr_array(A),
        b=b,
        num_nonneg_cones=num_nonneg,
        so_cone_dims=soc_dims,
        expected_returns=er,
        covariance=clean_cov,
        half_spread=half_spread,
        current_weights=current,
        risk_constraint=risk,
    )


def solve_mpo(
    expected_returns,
    covariance,
    half_spread_bps,
    risk_constraint,
    *,
    current_weights=None,
    settings=None,
    warm_start=None,
    allow_almost_solved: bool = True,
) -> MPOResult:
    """Solve the MPO with Moreau and return weights plus objective diagnostics."""
    import moreau

    problem = build_mpo_problem(
        expected_returns,
        covariance,
        half_spread_bps,
        risk_constraint,
        current_weights=current_weights,
    )
    cones = moreau.Cones(
        num_nonneg_cones=problem.num_nonneg_cones,
        so_cone_dims=list(problem.so_cone_dims),
    )
    solver_kwargs = {"cones": cones}
    if settings is not None:
        solver_kwargs["settings"] = settings
    solver = moreau.Solver(problem.P, problem.q, problem.A, problem.b, **solver_kwargs)
    solution = solver.solve() if warm_start is None else solver.solve(warm_start=warm_start)

    status = solver.info.status
    solved = status == moreau.SolverStatus.Solved
    if allow_almost_solved:
        solved = solved or status == moreau.SolverStatus.AlmostSolved
    if not solved:
        raise RuntimeError(f"Moreau MPO solve failed: {status}")

    h, n = problem.n_horizons, problem.n_assets
    weights = np.asarray(solution.x[:problem.weight_size], dtype=np.float64).reshape(h, n)
    prev = np.vstack([problem.current_weights, weights[:-1]])
    delta = weights - prev
    turnover = np.sum(np.abs(delta), axis=1)
    expected_return = np.einsum("hi,hi->h", problem.expected_returns, weights)
    transaction_cost = np.sum(np.abs(delta) * problem.half_spread, axis=1)
    risk_variance = np.einsum("hi,hij,hj->h", weights, problem.covariance, weights)
    objective = float(np.sum(expected_return - transaction_cost))
    cov_diag = tuple(covariance_diagnostics(c) for c in problem.covariance)

    return MPOResult(
        weights=weights,
        turnover=turnover,
        expected_return=expected_return,
        transaction_cost=transaction_cost,
        risk_variance=risk_variance,
        objective=objective,
        status=solver.info.status,
        iterations=int(solver.info.iterations),
        solve_time=float(solver.info.solve_time),
        setup_time=float(solver.info.setup_time),
        construction_time=float(solver.info.construction_time),
        solution=solution,
        covariance_diagnostics=cov_diag,
    )
