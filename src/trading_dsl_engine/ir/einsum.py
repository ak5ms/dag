from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import prod
from typing import Iterable, Sequence


Dimension = int | None
Label = str


class EinsumParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EinsumSpec:
    """Canonical backend-neutral NumPy-style einsum description.

    ``None`` extents denote the compile-time instrument dimension, which is
    resolved by a physical backend before contraction planning.
    """

    subscripts: str
    input_labels: tuple[tuple[Label, ...], ...]
    output_labels: tuple[Label, ...]
    label_extents: tuple[tuple[Label, Dimension], ...]
    optimize: str = "none"

    @property
    def extents(self) -> dict[Label, Dimension]:
        return dict(self.label_extents)

    @property
    def output_shape(self) -> tuple[Dimension, ...]:
        extents = self.extents
        return tuple(extents[label] for label in self.output_labels)


@dataclass(frozen=True, slots=True)
class ContractionStep:
    """One unary or binary contraction in a static contraction path."""

    operand_positions: tuple[int, ...]
    input_labels: tuple[tuple[Label, ...], ...]
    output_labels: tuple[Label, ...]
    loop_labels: tuple[Label, ...]
    loop_extents: tuple[int, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    estimated_flops: int

    @property
    def output_size(self) -> int:
        return prod(self.output_shape) if self.output_shape else 1


@dataclass(frozen=True, slots=True)
class ContractionPlan:
    steps: tuple[ContractionStep, ...]
    output_shape: tuple[int, ...]
    estimated_flops: int
    largest_intermediate: int


def normalize_optimize(value: object) -> str:
    if value is True:
        return "greedy"
    if value is False or value is None:
        return "none"
    text = str(value).lower()
    if text in {"true", "greedy", "auto", "auto-hq"}:
        return "greedy"
    if text in {"false", "none"}:
        return "none"
    if text == "optimal":
        return "optimal"
    raise EinsumParseError(f"unsupported einsum optimize value {value!r}")


def _tokenize_term(term: str, *, output: bool) -> tuple[str, ...]:
    tokens: list[str] = []
    index = 0
    seen_ellipsis = False
    while index < len(term):
        char = term[index]
        if char == ".":
            if term[index : index + 3] != "..." or seen_ellipsis:
                raise EinsumParseError(
                    "einsum ellipsis must be exactly '...' and appear at most once per term"
                )
            tokens.append("...")
            seen_ellipsis = True
            index += 3
            continue
        if not char.isalpha() or not char.isascii():
            where = "output" if output else "input"
            raise EinsumParseError(f"invalid einsum {where} subscript {char!r}")
        tokens.append(char)
        index += 1
    return tuple(tokens)


def _merge_extent(
    left: Dimension,
    right: Dimension,
    *,
    diagonal: bool = False,
    broadcast: bool = False,
) -> Dimension:
    if left == right:
        return left
    if left is None or right is None:
        # Equality involving symbolic N is checked after N is known. For an
        # ellipsis, symbolic N may also resolve against a unit broadcast axis.
        return None
    if diagonal:
        raise EinsumParseError(
            f"repeated einsum label dimensions differ: {left} != {right}"
        )
    if broadcast:
        if left == 1:
            return right
        if right == 1:
            return left
        raise EinsumParseError(
            f"einsum ellipsis dimensions are not broadcastable: {left} and {right}"
        )
    raise EinsumParseError(
        f"einsum label dimensions differ without an ellipsis: {left} != {right}"
    )


def parse_einsum(
    subscripts: str,
    operand_shapes: Sequence[Sequence[Dimension]],
    *,
    optimize: object = False,
) -> EinsumSpec:
    """Parse NumPy string-form einsum subscripts and infer symbolic output shape.

    Supported syntax matches NumPy's string form: ASCII letter labels, empty
    scalar terms, one ellipsis per term, implicit or explicit output, repeated
    labels for diagonals, and broadcasting only through ellipsis dimensions.
    """

    if not isinstance(subscripts, str):
        raise EinsumParseError("einsum subscripts must be a string")
    compact = "".join(subscripts.split())
    if compact.count("->") > 1:
        raise EinsumParseError("einsum subscripts may contain at most one '->'")
    if "->" in compact:
        input_text, output_text = compact.split("->", 1)
        explicit_output = True
    else:
        input_text, output_text = compact, ""
        explicit_output = False

    input_terms = input_text.split(",")
    if len(input_terms) != len(operand_shapes):
        raise EinsumParseError(
            f"einsum has {len(input_terms)} subscript terms for "
            f"{len(operand_shapes)} operands"
        )
    if not operand_shapes:
        raise EinsumParseError("einsum requires at least one operand")

    token_terms = tuple(_tokenize_term(term, output=False) for term in input_terms)
    ellipsis_ranks: list[int] = []
    for tokens, shape in zip(token_terms, operand_shapes):
        explicit_rank = sum(token != "..." for token in tokens)
        if "..." in tokens:
            ellipsis_rank = len(shape) - explicit_rank
            if ellipsis_rank < 0:
                raise EinsumParseError("too many einsum subscripts for operand rank")
        else:
            if explicit_rank != len(shape):
                raise EinsumParseError(
                    f"operand has rank {len(shape)} but einsum term has "
                    f"{explicit_rank} subscripts"
                )
            ellipsis_rank = 0
        ellipsis_ranks.append(ellipsis_rank)

    max_ellipsis = max(ellipsis_ranks, default=0)
    ellipsis_labels = tuple(f"@{index}" for index in range(max_ellipsis))
    expanded_inputs: list[tuple[Label, ...]] = []
    for tokens, ellipsis_rank in zip(token_terms, ellipsis_ranks):
        expanded: list[Label] = []
        for token in tokens:
            if token == "...":
                # NumPy right-aligns broadcast ellipsis dimensions.
                expanded.extend(ellipsis_labels[max_ellipsis - ellipsis_rank :])
            else:
                expanded.append(token)
        expanded_inputs.append(tuple(expanded))

    counts: dict[Label, int] = {}
    for labels in expanded_inputs:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

    if explicit_output:
        output_tokens = _tokenize_term(output_text, output=True)
        output_list: list[Label] = []
        for token in output_tokens:
            if token == "...":
                output_list.extend(ellipsis_labels)
            else:
                output_list.append(token)
        output_labels = tuple(output_list)
        if len(set(output_labels)) != len(output_labels):
            raise EinsumParseError("einsum output labels may not repeat")
        missing = [label for label in output_labels if label not in counts]
        if missing:
            raise EinsumParseError(
                f"einsum output label {missing[0]!r} does not appear in an input"
            )
    else:
        # NumPy places broadcast ellipsis first, followed by labels occurring
        # exactly once in sorted order.
        singleton_labels = sorted(
            label
            for label, count in counts.items()
            if count == 1 and label not in ellipsis_labels
        )
        output_labels = ellipsis_labels + tuple(singleton_labels)

    extents: dict[Label, Dimension] = {}
    for labels, raw_shape in zip(expanded_inputs, operand_shapes):
        shape = tuple(raw_shape)
        local: dict[Label, Dimension] = {}
        for label, extent in zip(labels, shape):
            if not (extent is None or isinstance(extent, int) and extent >= 0):
                raise EinsumParseError(f"invalid operand extent {extent!r}")
            if label in local:
                local[label] = _merge_extent(
                    local[label], extent, diagonal=True
                )
            else:
                local[label] = extent
        for label, extent in local.items():
            if label in extents:
                extents[label] = _merge_extent(
                    extents[label],
                    extent,
                    broadcast=label in ellipsis_labels,
                )
            else:
                extents[label] = extent

    return EinsumSpec(
        subscripts=compact,
        input_labels=tuple(expanded_inputs),
        output_labels=output_labels,
        label_extents=tuple(extents.items()),
        optimize=normalize_optimize(optimize),
    )


def resolve_spec(
    spec: EinsumSpec, operand_shapes: Sequence[Sequence[int]]
) -> EinsumSpec:
    """Resolve symbolic dimensions and repeat all NumPy shape checks."""

    return parse_einsum(spec.subscripts, operand_shapes, optimize=spec.optimize)


@dataclass(frozen=True, slots=True)
class _Term:
    labels: tuple[Label, ...]
    shape: tuple[int, ...]


def _ordered_union(*terms: Iterable[Label]) -> tuple[Label, ...]:
    result: list[Label] = []
    seen: set[Label] = set()
    for term in terms:
        for label in term:
            if label not in seen:
                seen.add(label)
                result.append(label)
    return tuple(result)


def _step_for_pair(
    terms: Sequence[_Term],
    left_pos: int,
    right_pos: int,
    final_output: tuple[Label, ...],
    extents: dict[Label, int],
) -> ContractionStep:
    left, right = terms[left_pos], terms[right_pos]
    other_labels: set[Label] = set()
    for position, term in enumerate(terms):
        if position not in {left_pos, right_pos}:
            other_labels.update(term.labels)

    needed = set(final_output) | other_labels
    union = _ordered_union(left.labels, right.labels)
    result_set = set(union) & needed
    result_labels = tuple(label for label in final_output if label in result_set)
    result_labels += tuple(
        label for label in union if label in result_set and label not in result_labels
    )
    reduction_labels = tuple(label for label in union if label not in result_set)
    loop_labels = result_labels + reduction_labels
    loop_extents = tuple(extents[label] for label in loop_labels)
    output_shape = tuple(extents[label] for label in result_labels)
    return ContractionStep(
        operand_positions=(left_pos, right_pos),
        input_labels=(left.labels, right.labels),
        output_labels=result_labels,
        loop_labels=loop_labels,
        loop_extents=loop_extents,
        input_shapes=(left.shape, right.shape),
        output_shape=output_shape,
        estimated_flops=prod(loop_extents) if loop_extents else 1,
    )


def _step_for_unary(
    term: _Term,
    final_output: tuple[Label, ...],
    extents: dict[Label, int],
) -> ContractionStep:
    reduction_labels = tuple(
        label for label in _ordered_union(term.labels) if label not in final_output
    )
    loop_labels = final_output + reduction_labels
    loop_extents = tuple(extents[label] for label in loop_labels)
    return ContractionStep(
        operand_positions=(0,),
        input_labels=(term.labels,),
        output_labels=final_output,
        loop_labels=loop_labels,
        loop_extents=loop_extents,
        input_shapes=(term.shape,),
        output_shape=tuple(extents[label] for label in final_output),
        estimated_flops=prod(loop_extents) if loop_extents else 1,
    )


def _pair_score(
    step: ContractionStep, left: _Term, right: _Term
) -> tuple[int, int, int]:
    input_size = prod(left.shape) + prod(right.shape)
    removed = input_size - step.output_size
    return step.estimated_flops, -removed, step.output_size


def _choose_pair_greedy(
    terms: Sequence[_Term],
    output: tuple[Label, ...],
    extents: dict[Label, int],
) -> tuple[int, int]:
    best_score: tuple[int, int, int] | None = None
    best_pair: tuple[int, int] | None = None
    for left, right in combinations(range(len(terms)), 2):
        step = _step_for_pair(terms, left, right, output, extents)
        score = _pair_score(step, terms[left], terms[right])
        if best_score is None or score < best_score:
            best_score = score
            best_pair = left, right
    assert best_pair is not None
    return best_pair


def _simulate_pair(
    terms: Sequence[_Term], left: int, right: int, step: ContractionStep
) -> list[_Term]:
    result = [
        term for position, term in enumerate(terms) if position not in {left, right}
    ]
    result.append(_Term(step.output_labels, step.output_shape))
    return result


def _optimal_path(
    terms: tuple[_Term, ...],
    output: tuple[Label, ...],
    extents: dict[Label, int],
) -> tuple[int, tuple[tuple[int, int], ...]]:
    @lru_cache(maxsize=None)
    def solve(state: tuple[_Term, ...]) -> tuple[int, tuple[tuple[int, int], ...]]:
        if len(state) <= 1:
            return 0, ()
        best_cost: int | None = None
        best_path: tuple[tuple[int, int], ...] | None = None
        for left, right in combinations(range(len(state)), 2):
            step = _step_for_pair(state, left, right, output, extents)
            next_state = tuple(_simulate_pair(state, left, right, step))
            tail_cost, tail_path = solve(next_state)
            cost = step.estimated_flops + tail_cost
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_path = ((left, right),) + tail_path
        assert best_cost is not None and best_path is not None
        return best_cost, best_path

    return solve(terms)


def build_contraction_plan(
    spec: EinsumSpec, operand_shapes: Sequence[Sequence[int]]
) -> ContractionPlan:
    """Build a static unary/binary contraction path.

    ``greedy`` minimizes estimated work and intermediate size at every step.
    ``optimal`` exhaustively searches paths for up to eight operands and falls
    back to greedy beyond that. ``none`` preserves left-to-right evaluation.
    """

    resolved = resolve_spec(spec, operand_shapes)
    extents = {label: int(extent) for label, extent in resolved.label_extents}
    terms = [
        _Term(labels, tuple(shape))
        for labels, shape in zip(resolved.input_labels, operand_shapes)
    ]
    steps: list[ContractionStep] = []

    optimal_path: list[tuple[int, int]] | None = None
    if resolved.optimize == "optimal" and 2 < len(terms) <= 8:
        _, path = _optimal_path(tuple(terms), resolved.output_labels, extents)
        optimal_path = list(path)

    while len(terms) > 1:
        if resolved.optimize == "none":
            left, right = 0, 1
        elif optimal_path is not None:
            left, right = optimal_path.pop(0)
        else:
            left, right = _choose_pair_greedy(
                terms, resolved.output_labels, extents
            )
        step = _step_for_pair(
            terms, left, right, resolved.output_labels, extents
        )
        steps.append(step)
        terms = _simulate_pair(terms, left, right, step)

    final = terms[0]
    if final.labels != resolved.output_labels or len(operand_shapes) == 1:
        step = _step_for_unary(final, resolved.output_labels, extents)
        steps.append(step)
        final = _Term(step.output_labels, step.output_shape)

    return ContractionPlan(
        steps=tuple(steps),
        output_shape=final.shape,
        estimated_flops=sum(step.estimated_flops for step in steps),
        largest_intermediate=max(
            (step.output_size for step in steps[:-1]), default=0
        ),
    )


__all__ = [
    "Dimension",
    "Label",
    "EinsumParseError",
    "EinsumSpec",
    "ContractionStep",
    "ContractionPlan",
    "normalize_optimize",
    "parse_einsum",
    "resolve_spec",
    "build_contraction_plan",
]
