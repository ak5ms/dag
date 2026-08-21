from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class CvxpyOutput:
    """Marker base for compile-time optimizer output projections."""


@dataclass(frozen=True)
class VariableValue(CvxpyOutput):
    name: str


@dataclass(frozen=True)
class ExpressionValue(CvxpyOutput):
    name: str


@dataclass(frozen=True)
class ConstraintDual(CvxpyOutput):
    name: str


@dataclass(frozen=True)
class ConstraintSlack(CvxpyOutput):
    name: str


@dataclass(frozen=True)
class SolverMetric(CvxpyOutput):
    name: Literal["status", "iterations", "objective", "solve_time"]


def variable_value(name: str) -> VariableValue:
    return VariableValue(name)


def expression_value(name: str) -> ExpressionValue:
    return ExpressionValue(name)


def constraint_dual(name: str) -> ConstraintDual:
    return ConstraintDual(name)


def constraint_slack(name: str) -> ConstraintSlack:
    return ConstraintSlack(name)


def solver_metric(
    name: Literal["status", "iterations", "objective", "solve_time"],
) -> SolverMetric:
    return SolverMetric(name)


@dataclass
class CvxpyNodeBuild:
    """Static-shape objects returned by a ``@cvxpy_node`` factory.

    ``parameters``, ``variables``, ``expressions`` and ``constraints`` are named
    dictionaries. Runtime values are supplied by name; CVXPY is used only for
    canonicalization/inverse projection at the batch-stage boundary.
    """

    problem: Any
    parameters: Mapping[str, Any]
    variables: Mapping[str, Any]
    constraints: Mapping[str, Any] = field(default_factory=dict)
    expressions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolveDiagnostics:
    status: str
    iterations: int
    objective: float
    solve_time: float


def _normalize_build(value: Any) -> CvxpyNodeBuild:
    if isinstance(value, CvxpyNodeBuild):
        return value
    if not isinstance(value, tuple):
        raise TypeError(
            "cvxpy_node factory must return CvxpyNodeBuild or a compatible tuple"
        )
    if len(value) == 3:
        return CvxpyNodeBuild(value[0], value[1], value[2])
    if len(value) == 4:
        return CvxpyNodeBuild(value[0], value[1], value[2], value[3])
    if len(value) == 5:
        return CvxpyNodeBuild(*value)
    raise TypeError("unsupported cvxpy_node factory return shape")


def _constraint_slack_value(constraint: Any) -> np.ndarray:
    """Return original-constraint primal slack/residual.

    For ``lhs <= rhs`` this returns ``rhs-lhs``. Equality output is the signed
    residual. SOC output is the cone vector ``[t, x...]`` so callers can inspect
    both radius and norm components without an additional solve.
    """
    import cvxpy as cp

    if isinstance(constraint, cp.constraints.nonpos.Inequality):
        lhs, rhs = constraint.args
        return np.asarray(rhs.value, dtype=np.float64) - np.asarray(
            lhs.value, dtype=np.float64
        )
    if isinstance(constraint, cp.constraints.zero.Equality):
        lhs, rhs = constraint.args
        return np.asarray(lhs.value, dtype=np.float64) - np.asarray(
            rhs.value, dtype=np.float64
        )
    if isinstance(constraint, cp.constraints.second_order.SOC):
        t, x = constraint.args
        return np.concatenate(
            [
                np.asarray(t.value, dtype=np.float64).reshape(-1),
                np.asarray(x.value, dtype=np.float64).reshape(-1),
            ]
        )
    violation = constraint.violation()
    return np.asarray(np.nan if violation is None else -violation, dtype=np.float64)


class _ClarabelWorkspace:
    """One persistent Clarabel solver/workspace.

    Covariance changes imply that the canonical A matrix changes on every problem.
    Repeated solves therefore always update P/A/q/b using fixed sparsity. Presolve
    and chordal decomposition are disabled because they can alter structure.
    """

    def __init__(self, definition: "CvxpyNodeDefinition") -> None:
        import clarabel
        import cvxpy as cp

        self.definition = definition
        self.clarabel = clarabel
        self.cp = cp
        self.build = definition._fresh_build()
        self.solver: Any | None = None
        self.structure: tuple[Any, ...] | None = None
        self.lock = threading.Lock()

    def _assign(self, values: Mapping[str, Any]) -> None:
        missing = set(self.build.parameters) - set(values)
        extra = set(values) - set(self.build.parameters)
        if missing or extra:
            raise KeyError(
                f"optimizer parameter mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        for name, parameter in self.build.parameters.items():
            value = np.asarray(values[name], dtype=np.float64)
            expected = tuple(int(v) for v in parameter.shape)
            if value.shape != expected:
                raise ValueError(
                    f"parameter {name!r} has shape {value.shape}; expected {expected}"
                )
            parameter.value = value

    def _settings(self) -> Any:
        settings = self.clarabel.DefaultSettings()
        settings.verbose = bool(self.definition.solver_settings.get("verbose", False))
        for name in ("max_iter", "tol_gap_abs", "tol_gap_rel", "tol_feas"):
            if name in self.definition.solver_settings:
                setattr(settings, name, self.definition.solver_settings[name])
        if hasattr(settings, "presolve_enable"):
            settings.presolve_enable = False
        if hasattr(settings, "chordal_decomposition_enable"):
            settings.chordal_decomposition_enable = False
        return settings

    def _canonical_data(self) -> tuple[Any, ...]:
        from scipy import sparse

        data, chain, inverse = self.build.problem.get_problem_data(
            self.cp.CLARABEL,
            solver_opts=self.definition.solver_settings,
        )
        q = np.asarray(data["c"], dtype=np.float64)
        b = np.asarray(data["b"], dtype=np.float64)
        A = data["A"].tocsc(copy=False)
        P = data.get("P")
        if P is None:
            P = sparse.csc_matrix((q.size, q.size), dtype=np.float64)
        else:
            P = P.tocsc(copy=False)

        dims = data["dims"]
        cones: list[Any] = []
        if int(dims.zero):
            cones.append(self.clarabel.ZeroConeT(int(dims.zero)))
        if int(dims.nonneg):
            cones.append(self.clarabel.NonnegativeConeT(int(dims.nonneg)))
        cones.extend(self.clarabel.SecondOrderConeT(int(dim)) for dim in dims.soc)
        cones.extend(self.clarabel.ExponentialConeT() for _ in range(int(dims.exp)))
        for alpha in getattr(dims, "p3d", []) or []:
            cones.append(self.clarabel.PowerConeT(float(alpha)))
        if getattr(dims, "psd", []) or []:
            raise NotImplementedError("Clarabel optimizer nodes do not support PSD cones")

        structure = (
            P.shape,
            P.indptr.tobytes(),
            P.indices.tobytes(),
            A.shape,
            A.indptr.tobytes(),
            A.indices.tobytes(),
            tuple(type(cone).__name__ for cone in cones),
            tuple(getattr(cone, "dim", None) for cone in cones),
        )
        return P, q, A, b, cones, data, chain, inverse, structure

    @staticmethod
    def _update(solver: Any, *, P: Any, q: np.ndarray, A: Any, b: np.ndarray) -> None:
        try:
            solver.update(P=P, q=q, A=A, b=b)
            return
        except TypeError:
            pass
        for label, value in (("P", P), ("q", q), ("A", A), ("b", b)):
            method = getattr(solver, f"update_{label}", None) or getattr(
                solver, f"update_{label.lower()}", None
            )
            if method is None:
                raise RuntimeError(
                    "installed Clarabel lacks the fixed-sparsity update API"
                )
            method(value)

    def _unpack(self, solution: Any, data: Any, chain: Any, inverse: Any) -> None:
        """Populate original CVXPY primal and dual values after a native solve."""
        try:
            original_solution = chain.invert(solution, inverse)
            self.build.problem.unpack(original_solution)
            return
        except Exception:
            pass

        inverse_items = list(reversed(inverse)) if isinstance(inverse, list) else [inverse]
        metadata = next(
            (
                item
                for item in inverse_items
                if hasattr(item, "var_offsets") and hasattr(item, "var_shapes")
            ),
            None,
        )
        if metadata is None:
            raise RuntimeError("CVXPY inverse metadata unavailable")
        flat = np.asarray(solution.x, dtype=np.float64)
        by_id: dict[int, np.ndarray] = {}
        for var_id, offset in metadata.var_offsets.items():
            shape = tuple(metadata.var_shapes[var_id])
            size = int(np.prod(shape, dtype=int)) if shape else 1
            value = flat[int(offset) : int(offset) + size]
            by_id[var_id] = value.reshape(shape, order="F") if shape else value[0]
        for variable in self.build.problem.variables():
            if variable.id in by_id:
                variable.save_value(by_id[variable.id])

    def solve(self, values: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        self._assign(values)
        P, q, A, b, cones, data, chain, inverse, structure = self._canonical_data()
        with self.lock:
            if self.solver is None or structure != self.structure:
                self.solver = self.clarabel.DefaultSolver(
                    P, q, A, b, cones, self._settings()
                )
                self.structure = structure
            else:
                self._update(self.solver, P=P, q=q, A=A, b=b)
            solution = self.solver.solve()
        elapsed = time.perf_counter() - started
        status = str(solution.status)
        if status not in {"Solved", "AlmostSolved"}:
            raise RuntimeError(f"Clarabel optimizer node failed: {status}")
        self._unpack(solution, data, chain, inverse)
        maximize = self.build.problem.objective.NAME == "maximize"
        diagnostics = SolveDiagnostics(
            status=status,
            iterations=int(solution.iterations),
            objective=(-1.0 if maximize else 1.0) * float(solution.obj_val),
            solve_time=elapsed,
        )
        return self.definition._project(self.build, diagnostics)


@dataclass
class CompiledCvxpyNode:
    definition: "CvxpyNodeDefinition"
    workers: int = 1

    def __post_init__(self) -> None:
        self.workers = max(1, int(self.workers))
        self._workspaces = [
            _ClarabelWorkspace(self.definition) for _ in range(self.workers)
        ]

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.definition._prototype.parameters)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self.definition.outputs)

    def solve(self, **parameters: Any) -> dict[str, Any]:
        return self._workspaces[0].solve(parameters)

    def solve_batch(
        self,
        parameters: Mapping[str, Any],
        *,
        workers: int | None = None,
        sequential: bool = False,
    ) -> dict[str, np.ndarray]:
        missing = set(self.parameter_names) - set(parameters)
        extra = set(parameters) - set(self.parameter_names)
        if missing or extra:
            raise KeyError(
                f"optimizer parameter mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        arrays = {name: np.asarray(parameters[name]) for name in self.parameter_names}
        sizes = {value.shape[0] for value in arrays.values()}
        if len(sizes) != 1:
            raise ValueError("all optimizer batch inputs must share axis-0 length")
        batch_size = sizes.pop()
        requested_workers = max(1, int(workers or self.workers))
        if sequential and requested_workers != 1:
            raise ValueError("sequential optimizer batches require workers=1")
        n_workers = 1 if sequential else min(requested_workers, batch_size)
        while len(self._workspaces) < n_workers:
            self._workspaces.append(_ClarabelWorkspace(self.definition))

        results: list[dict[str, Any] | None] = [None] * batch_size

        def run_partition(worker: int) -> None:
            workspace = self._workspaces[worker]
            for index in range(worker, batch_size, n_workers):
                results[index] = workspace.solve(
                    {name: arrays[name][index] for name in self.parameter_names}
                )

        if n_workers == 1:
            run_partition(0)
        else:
            # Clarabel's numerical solve is native and releases the GIL. Stable
            # strided partitions give each worker exclusive ownership of one
            # persistent solver and preserve deterministic output ordering.
            with ThreadPoolExecutor(
                max_workers=n_workers, thread_name_prefix="cvxpy-optimizer"
            ) as pool:
                list(pool.map(run_partition, range(n_workers)))

        completed = [result for result in results if result is not None]
        if len(completed) != batch_size:
            raise AssertionError("optimizer batch did not produce every output")
        return {
            name: np.stack([np.asarray(result[name]) for result in completed], axis=0)
            for name in completed[0]
        }


class CvxpyNodeDefinition:
    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        outputs: Mapping[str, CvxpyOutput | str],
        solver_settings: Mapping[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        self.factory = factory
        self.name = name or factory.__name__
        self.solver_settings = dict(solver_settings or {})
        self.outputs = {
            output_name: VariableValue(spec) if isinstance(spec, str) else spec
            for output_name, spec in outputs.items()
        }
        self._prototype = self._fresh_build()
        self._validate()

    def _fresh_build(self) -> CvxpyNodeBuild:
        return _normalize_build(self.factory())

    def _validate(self) -> None:
        if not self._prototype.problem.is_dcp(dpp=True):
            raise ValueError(f"CVXPY node {self.name!r} must be DPP compliant")
        for output_name, spec in self.outputs.items():
            if isinstance(spec, VariableValue) and spec.name not in self._prototype.variables:
                raise KeyError(
                    f"output {output_name!r} references unknown variable {spec.name!r}"
                )
            if isinstance(spec, ExpressionValue) and spec.name not in self._prototype.expressions:
                raise KeyError(
                    f"output {output_name!r} references unknown expression {spec.name!r}"
                )
            if isinstance(spec, (ConstraintDual, ConstraintSlack)) and spec.name not in self._prototype.constraints:
                raise KeyError(
                    f"output {output_name!r} references unknown constraint {spec.name!r}"
                )

    def compile(self, *, workers: int = 1) -> CompiledCvxpyNode:
        return CompiledCvxpyNode(self, workers=workers)

    def _project(
        self, build: CvxpyNodeBuild, diagnostics: SolveDiagnostics
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for output_name, spec in self.outputs.items():
            if isinstance(spec, VariableValue):
                value = build.variables[spec.name].value
            elif isinstance(spec, ExpressionValue):
                value = build.expressions[spec.name].value
            elif isinstance(spec, ConstraintDual):
                value = build.constraints[spec.name].dual_value
            elif isinstance(spec, ConstraintSlack):
                value = _constraint_slack_value(build.constraints[spec.name])
            elif isinstance(spec, SolverMetric):
                value = getattr(diagnostics, spec.name)
            else:
                raise TypeError(f"unsupported output spec {type(spec).__name__}")
            if value is None:
                raise RuntimeError(f"optimizer output {output_name!r} is unavailable")
            result[output_name] = np.asarray(value).copy()
        return result

    def __call__(self, **parameters: Any) -> dict[str, Any]:
        return self.compile().solve(**parameters)


def cvxpy_node(
    *,
    outputs: Mapping[str, CvxpyOutput | str],
    solver_settings: Mapping[str, Any] | None = None,
    name: str | None = None,
) -> Callable[[Callable[[], Any]], CvxpyNodeDefinition]:
    """Declare a static-shape, DPP-compliant CVXPY optimizer node."""

    def decorate(factory: Callable[[], Any]) -> CvxpyNodeDefinition:
        return CvxpyNodeDefinition(
            factory,
            outputs=outputs,
            solver_settings=solver_settings,
            name=name,
        )

    return decorate


@dataclass
class OptimizerPipeline:
    """cpp_stream parameter stages followed by one optimizer batch stage.

    Every parameter producer executes once over the aligned batch; the optimizer
    then consumes all problem inputs in one ordered dispatch. There is no Python
    per-row formula evaluation. Sequential dependence remains explicit and single
    worker; independent problems may use all requested workers.
    """

    optimizer: CompiledCvxpyNode
    parameter_producers: Mapping[str, Any]

    def run_batch(
        self,
        data: Mapping[str, Any],
        *,
        workers: int | None = None,
        sequential: bool = False,
    ) -> dict[str, np.ndarray]:
        parameters: dict[str, np.ndarray] = {}
        for name, producer in self.parameter_producers.items():
            if hasattr(producer, "run_batch"):
                produced = producer.run_batch(data)
                if isinstance(produced, tuple):
                    produced = produced[-1]
            else:
                produced = producer(data)
            parameters[name] = np.asarray(produced)
        return self.optimizer.solve_batch(
            parameters, workers=workers, sequential=sequential
        )


def optimizer_pipeline(
    optimizer: CompiledCvxpyNode, **parameter_producers: Any
) -> OptimizerPipeline:
    missing = set(optimizer.parameter_names) - set(parameter_producers)
    extra = set(parameter_producers) - set(optimizer.parameter_names)
    if missing or extra:
        raise KeyError(
            f"parameter producer mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return OptimizerPipeline(optimizer, parameter_producers)
