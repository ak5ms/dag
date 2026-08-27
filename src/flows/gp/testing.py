from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import math
import random
from typing import Any

from deap import gp
import numpy as np

from flows.gp.generation import individual_to_expr
from flows.gp.types import StaticValue
from trading_dsl_engine.base import dsl
from trading_dsl_engine.base import random_dsl
from trading_dsl_engine.base.custom import StatelessCall
from trading_dsl_engine.base.keys import Key
from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple


CheckValue = bool | tuple[bool, str]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class GenoContext:
    tree: gp.PrimitiveTree
    pset: gp.PrimitiveSetTyped
    family_counts: Mapping[str, int]
    type_counts: Mapping[str, int]
    primitive_count: int
    terminal_count: int


@dataclass(frozen=True)
class GenoTest:
    name: str
    check: Callable[[GenoContext], CheckValue | CheckResult]


@dataclass(frozen=True)
class GenoReport:
    context: GenoContext
    outcomes: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(outcome.passed for outcome in self.outcomes)


@dataclass(frozen=True)
class StaticShock:
    index: int
    type_name: str
    before: Any
    after: Any
    before_name: str
    after_name: str


@dataclass(frozen=True)
class NoiseSpec:
    """Per-field noise used by phenotypic robustness trials.

    Parameter values may be constants/expressions or callables taking the field
    leaf and returning a constant/expression. ``mode`` controls how the generated
    draw is combined with the original leaf: ``add``, ``mul``, or ``replace``.
    """

    distribution: str = "normal"
    params: Mapping[str, object] = field(default_factory=dict)
    mode: str = "add"
    seed: int | None = None

    def __post_init__(self) -> None:
        distribution = self.distribution.lower()
        if distribution not in {"normal", "lognormal", "exponential", "uniform"}:
            raise ValueError(
                "distribution must be one of normal, lognormal, exponential, uniform"
            )
        if self.mode not in {"add", "mul", "replace"}:
            raise ValueError("noise mode must be add, mul, or replace")
        if "seed" in self.params:
            raise ValueError("put the random seed in NoiseSpec.seed, not params")
        object.__setattr__(self, "distribution", distribution)


@dataclass(frozen=True)
class DynamicShock:
    field: str
    occurrence: int
    distribution: str
    mode: str
    seed: int


@dataclass(frozen=True)
class PhenoContext:
    trial_index: int
    baseline_expr: Expr
    shocked_expr: Expr
    baseline: object
    shocked: object
    static_shocks: tuple[StaticShock, ...]
    dynamic_shocks: tuple[DynamicShock, ...]


@dataclass(frozen=True)
class PhenoTest:
    name: str
    check: Callable[[PhenoContext], CheckValue | CheckResult]


@dataclass(frozen=True)
class PhenoTrial:
    index: int
    expr: Expr
    value: object | None
    static_shocks: tuple[StaticShock, ...]
    dynamic_shocks: tuple[DynamicShock, ...]
    outcomes: tuple[CheckResult, ...]
    execution_error: str | None = None

    @property
    def passed(self) -> bool:
        return self.execution_error is None and all(
            outcome.passed for outcome in self.outcomes
        )


@dataclass(frozen=True)
class PhenoReport:
    baseline_expr: Expr
    baseline: object
    trials: tuple[PhenoTrial, ...]

    @property
    def passed(self) -> bool:
        return all(trial.passed for trial in self.trials)


def _normalize_result(name: str, value: CheckValue | CheckResult) -> CheckResult:
    if isinstance(value, CheckResult):
        if value.name == name:
            return value
        return CheckResult(name=name, passed=value.passed, detail=value.detail)
    if isinstance(value, tuple):
        passed, detail = value
        return CheckResult(name=name, passed=bool(passed), detail=str(detail))
    return CheckResult(name=name, passed=bool(value))


def _run_check(name: str, check: Callable, context) -> CheckResult:
    try:
        return _normalize_result(name, check(context))
    except Exception as exc:  # a failing robustness predicate is a failed test
        return CheckResult(
            name=name,
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _compatible(actual: type, expected: type) -> bool:
    try:
        return issubclass(actual, expected)
    except TypeError:
        return actual == expected


def _well_typed(tree: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped) -> CheckResult:
    if not tree:
        return CheckResult("well_typed", False, "empty GP tree")

    def consume(index: int, expected: type) -> int:
        if index >= len(tree):
            raise ValueError(f"missing child of expected type {expected.__name__}")
        node = tree[index]
        if not _compatible(node.ret, expected):
            raise TypeError(
                f"node {index} {node.name!r} returns {node.ret.__name__}, "
                f"expected {expected.__name__}"
            )
        next_index = index + 1
        if isinstance(node, gp.Primitive):
            for arg_type in node.args:
                next_index = consume(next_index, arg_type)
        return next_index

    try:
        end = consume(0, pset.ret)
        if end != len(tree):
            raise ValueError(f"prefix tree has {len(tree) - end} trailing node(s)")
    except Exception as exc:
        return CheckResult("well_typed", False, str(exc))
    return CheckResult("well_typed", True, f"typed prefix tree; {len(tree)} nodes")


def geno_context(
    tree: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
) -> GenoContext:
    family_of = getattr(pset, "gp_primitive_family", {})
    primitives = [node for node in tree if isinstance(node, gp.Primitive)]
    terminals = [node for node in tree if isinstance(node, gp.Terminal)]
    families = Counter(family_of.get(node.name, node.name) for node in primitives)
    types = Counter(node.ret.__name__ for node in tree)
    return GenoContext(
        tree=tree,
        pset=pset,
        family_counts=dict(families),
        type_counts=dict(types),
        primitive_count=len(primitives),
        terminal_count=len(terminals),
    )


def run_geno_tests(
    tree: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    tests: Sequence[GenoTest] = (),
) -> GenoReport:
    """Run zero-execution structural/rational tests against a GP tree."""

    context = geno_context(tree, pset)
    outcomes = [_well_typed(tree, pset)]
    outcomes.extend(_run_check(test.name, test.check, context) for test in tests)
    return GenoReport(context=context, outcomes=tuple(outcomes))


def geno_max_depth(max_depth: int) -> GenoTest:
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    return GenoTest(
        f"max_depth<={max_depth}",
        lambda ctx: (
            ctx.tree.height <= max_depth,
            f"depth={ctx.tree.height}, limit={max_depth}",
        ),
    )


def geno_max_nodes(max_nodes: int) -> GenoTest:
    if max_nodes < 1:
        raise ValueError("max_nodes must be >= 1")
    return GenoTest(
        f"max_nodes<={max_nodes}",
        lambda ctx: (
            len(ctx.tree) <= max_nodes,
            f"nodes={len(ctx.tree)}, limit={max_nodes}",
        ),
    )


def geno_forbid_families(*families: str) -> GenoTest:
    forbidden = frozenset(families)

    def check(ctx: GenoContext) -> CheckValue:
        present = sorted(forbidden & ctx.family_counts.keys())
        return not present, "present=" + ",".join(present) if present else "none present"

    return GenoTest("forbid:" + ",".join(sorted(forbidden)), check)


def geno_require_families(*families: str) -> GenoTest:
    required = frozenset(families)

    def check(ctx: GenoContext) -> CheckValue:
        missing = sorted(required - ctx.family_counts.keys())
        return not missing, "missing=" + ",".join(missing) if missing else "all present"

    return GenoTest("require:" + ",".join(sorted(required)), check)


def _terminal_value(pset: gp.PrimitiveSetTyped, terminal: gp.Terminal):
    value = terminal.value
    if isinstance(value, str) and value in pset.context:
        return pset.context[value]
    return value


def _sortable_static_value(value: object):
    if not isinstance(value, StaticValue):
        return None
    raw = value.value
    if isinstance(raw, bool):
        return "number", float(raw)
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        return "number", float(raw)
    return None


def _slot_types(
    tree: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
) -> tuple[type, ...]:
    slots: list[type | None] = [None] * len(tree)
    if not tree:
        return ()

    def consume(index: int, expected: type) -> int:
        if index >= len(tree):
            return index
        node = tree[index]
        slots[index] = expected
        next_index = index + 1
        if isinstance(node, gp.Primitive):
            for arg_type in node.args:
                next_index = consume(next_index, arg_type)
        return next_index

    root_expected = pset.ret if _compatible(tree[0].ret, pset.ret) else tree[0].ret
    consume(0, root_expected)
    return tuple(slot or tree[index].ret for index, slot in enumerate(slots))


def _shock_candidates(
    pset: gp.PrimitiveSetTyped,
    terminal: gp.Terminal,
    expected_type: type,
    *,
    k: int,
) -> list[gp.Terminal]:
    current_value = _terminal_value(pset, terminal)
    current_key = _sortable_static_value(current_value)
    if current_key is None:
        return []

    by_key: dict[tuple[str, object], gp.Terminal] = {}
    for candidate in pset.terminals.get(expected_type, ()):  # includes compatible subtypes
        key = _sortable_static_value(_terminal_value(pset, candidate))
        if key is not None:
            by_key.setdefault(key, candidate)
    if current_key not in by_key:
        return []
    ordered = sorted(by_key)
    index = ordered.index(current_key)
    keys = ordered[max(0, index - k) : min(len(ordered), index + k + 1)]
    return [by_key[key] for key in keys if key != current_key]


def shock_static_terminals(
    tree: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    *,
    k: int = 1,
    probability: float = 1.0,
    seed: int | None = None,
) -> tuple[gp.PrimitiveTree, tuple[StaticShock, ...]]:
    """Randomly replace sortable static leaves by one of their +/- ``k`` neighbors."""

    if k < 1:
        raise ValueError("k must be >= 1")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    rng = random.Random(seed)
    shocked = gp.PrimitiveTree(tree)
    slot_types = _slot_types(shocked, pset)
    changes: list[StaticShock] = []
    for index, terminal in enumerate(tuple(shocked)):
        if not isinstance(terminal, gp.Terminal) or rng.random() > probability:
            continue
        expected_type = slot_types[index]
        candidates = _shock_candidates(pset, terminal, expected_type, k=k)
        if not candidates:
            continue
        replacement = rng.choice(candidates)
        before_value = _terminal_value(pset, terminal)
        after_value = _terminal_value(pset, replacement)
        shocked[index] = replacement
        changes.append(
            StaticShock(
                index=index,
                type_name=expected_type.__name__,
                before=before_value.value if isinstance(before_value, StaticValue) else before_value,
                after=after_value.value if isinstance(after_value, StaticValue) else after_value,
                before_name=terminal.name,
                after_name=replacement.name,
            )
        )
    return shocked, tuple(changes)


def _derived_seed(base_seed: int | None, field_name: str, occurrence: int, spec: NoiseSpec) -> int:
    root = int(spec.seed if spec.seed is not None else (base_seed or 0))
    digest = int.from_bytes(
        hashlib.blake2b(field_name.encode("utf-8"), digest_size=4).digest(),
        "little",
    )
    modulus = 2**31 - 1
    return (root + digest + occurrence * 104_729) % modulus


def _resolve_noise_params(spec: NoiseSpec, leaf: Expr) -> dict[str, object]:
    params: dict[str, object] = {}
    for name, value in spec.params.items():
        params[name] = value(leaf) if callable(value) else value
    return params


def _noise_expr(leaf: Expr, spec: NoiseSpec, *, seed: int) -> Expr:
    generator = getattr(random_dsl, spec.distribution)
    params = _resolve_noise_params(spec, leaf)
    params.setdefault("key", leaf)
    params["seed"] = seed
    noise = generator(**params)
    if spec.mode == "add":
        return dsl.add(leaf, noise)
    if spec.mode == "mul":
        return dsl.mul(leaf, noise)
    return noise


def _coerce_noise_spec(value: NoiseSpec | Mapping[str, object]) -> NoiseSpec:
    if isinstance(value, NoiseSpec):
        return value
    if isinstance(value, Mapping):
        return NoiseSpec(**dict(value))
    raise TypeError("field_noise values must be NoiseSpec or mapping configs")


def shock_dynamic_leaves(
    expr: Expr,
    field_noise: Mapping[str, NoiseSpec | Mapping[str, object]],
    *,
    seed: int | None = None,
) -> tuple[Expr, tuple[DynamicShock, ...]]:
    """Rewrite matching field identifiers with independently seeded noise expressions."""

    occurrences: Counter[str] = Counter()
    changes: list[DynamicShock] = []

    def visit(node: Expr) -> Expr:
        if isinstance(node, Identifier):
            raw_spec = field_noise.get(node.name)
            if raw_spec is None:
                return node
            spec = _coerce_noise_spec(raw_spec)
            occurrence = occurrences[node.name]
            occurrences[node.name] += 1
            draw_seed = _derived_seed(seed, node.name, occurrence, spec)
            changes.append(
                DynamicShock(
                    field=node.name,
                    occurrence=occurrence,
                    distribution=spec.distribution,
                    mode=spec.mode,
                    seed=draw_seed,
                )
            )
            return _noise_expr(node, spec, seed=draw_seed)
        if isinstance(node, Call):
            return Call(
                node.fn,
                tuple(visit(child) for child in node.args),
                tuple((name, visit(child)) for name, child in node.kwargs),
            )
        if isinstance(node, KeyTuple):
            return KeyTuple(tuple(visit(child) for child in node.items))
        if isinstance(node, Key):
            return replace(node, expr=visit(node.expr))
        if isinstance(node, StatelessCall):
            return replace(node, args=tuple(visit(child) for child in node.args))
        return node

    return visit(expr), tuple(changes)


def run_pheno_tests(
    tree: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    evaluator: Callable[[Expr], object],
    tests: Sequence[PhenoTest] = (),
    *,
    n_trials: int = 8,
    static_k: int | None = 1,
    static_probability: float = 1.0,
    field_noise: Mapping[str, NoiseSpec | Mapping[str, object]] | None = None,
    seed: int | None = None,
) -> PhenoReport:
    """Execute baseline + perturbed formulas and apply run-dependent tests."""

    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if static_k is not None and static_k < 1:
        raise ValueError("static_k must be >= 1 or None")
    baseline_expr = individual_to_expr(tree, pset)
    try:
        baseline = evaluator(baseline_expr)
    except Exception as exc:
        raise RuntimeError(f"baseline pheno evaluation failed: {exc}") from exc

    rng = random.Random(seed)
    noise = field_noise or {}
    trials: list[PhenoTrial] = []
    for trial_index in range(n_trials):
        trial_seed = rng.randrange(0, 2**31 - 1)
        if static_k is None:
            shocked_tree = gp.PrimitiveTree(tree)
            static_shocks: tuple[StaticShock, ...] = ()
        else:
            shocked_tree, static_shocks = shock_static_terminals(
                tree,
                pset,
                k=static_k,
                probability=static_probability,
                seed=trial_seed,
            )
        shocked_expr = individual_to_expr(shocked_tree, pset)
        shocked_expr, dynamic_shocks = shock_dynamic_leaves(
            shocked_expr,
            noise,
            seed=trial_seed,
        )
        try:
            shocked_value = evaluator(shocked_expr)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            trials.append(
                PhenoTrial(
                    index=trial_index,
                    expr=shocked_expr,
                    value=None,
                    static_shocks=static_shocks,
                    dynamic_shocks=dynamic_shocks,
                    outcomes=(CheckResult("execution", False, error),),
                    execution_error=error,
                )
            )
            continue

        context = PhenoContext(
            trial_index=trial_index,
            baseline_expr=baseline_expr,
            shocked_expr=shocked_expr,
            baseline=baseline,
            shocked=shocked_value,
            static_shocks=static_shocks,
            dynamic_shocks=dynamic_shocks,
        )
        outcomes = [CheckResult("execution", True)]
        outcomes.extend(_run_check(test.name, test.check, context) for test in tests)
        trials.append(
            PhenoTrial(
                index=trial_index,
                expr=shocked_expr,
                value=shocked_value,
                static_shocks=static_shocks,
                dynamic_shocks=dynamic_shocks,
                outcomes=tuple(outcomes),
            )
        )
    return PhenoReport(
        baseline_expr=baseline_expr,
        baseline=baseline,
        trials=tuple(trials),
    )


def pheno_finite() -> PhenoTest:
    def check(ctx: PhenoContext) -> CheckValue:
        values = np.asarray(ctx.shocked)
        finite = bool(np.all(np.isfinite(values)))
        return finite, f"finite={finite}, shape={values.shape}"

    return PhenoTest("finite", check)


def pheno_stability(max_relative_change: float, *, atol: float = 1e-12) -> PhenoTest:
    if max_relative_change < 0.0:
        raise ValueError("max_relative_change must be >= 0")
    if atol <= 0.0:
        raise ValueError("atol must be > 0")

    def check(ctx: PhenoContext) -> CheckValue:
        baseline = np.asarray(ctx.baseline, dtype=float)
        shocked = np.asarray(ctx.shocked, dtype=float)
        if baseline.shape != shocked.shape:
            return False, f"shape changed {baseline.shape} -> {shocked.shape}"
        if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(shocked)):
            return False, "non-finite baseline or shocked value"
        denom = np.maximum(np.abs(baseline), atol)
        change = float(np.max(np.abs(shocked - baseline) / denom))
        return change <= max_relative_change, (
            f"max_relative_change={change:.6g}, limit={max_relative_change:.6g}"
        )

    return PhenoTest(f"stability<={max_relative_change:g}", check)


__all__ = [
    "CheckResult",
    "DynamicShock",
    "GenoContext",
    "GenoReport",
    "GenoTest",
    "NoiseSpec",
    "PhenoContext",
    "PhenoReport",
    "PhenoTest",
    "PhenoTrial",
    "StaticShock",
    "geno_context",
    "geno_forbid_families",
    "geno_max_depth",
    "geno_max_nodes",
    "geno_require_families",
    "pheno_finite",
    "pheno_stability",
    "run_geno_tests",
    "run_pheno_tests",
    "shock_dynamic_leaves",
    "shock_static_terminals",
]
