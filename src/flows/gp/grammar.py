from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Literal

from deap import gp


GRAMMAR_SECTIONS: tuple[str, ...] = (
    "row.elementwise",
    "row.ewm",
    "row.rolling",
    "row.alpha",
    "row.cross_sectional",
    "row.temporal",
    "row.regression",
    "row.reductions",
    "utils.elementwise",
    "utils.cross_sectional",
    "utils.group",
    "utils.temporal",
    "tensor.terminals",
    "tensor.elementwise",
    "tensor.temporal",
    "tensor.utils.elementwise",
    "tensor.utils.temporal",
    "tensor.regression",
)


@dataclass(frozen=True)
class GrammarPolicy:
    """Top-level selection policy for the GP grammar.

    The same policy is used by every registrar. Patterns use shell-style
    matching, so ``tensor.*`` or ``group_*`` are valid selectors.

    Examples
    --------
    Disable group utilities::

        GrammarPolicy(exclude_sections=("utils.group",))

    Keep only a small row grammar::

        GrammarPolicy(include_sections=("row.elementwise", "row.ewm"))

    Remove individual operator families everywhere, including tensor overloads::

        GrammarPolicy(exclude_families=("rolling_entropy", "group_*"))
    """

    include_sections: tuple[str, ...] = ("*",)
    exclude_sections: tuple[str, ...] = ()
    include_families: tuple[str, ...] = ("*",)
    exclude_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "include_sections",
            "exclude_sections",
            "include_families",
            "exclude_families",
        ):
            values = tuple(str(value).strip() for value in getattr(self, field_name))
            if field_name.startswith("include_") and not values:
                raise ValueError(f"{field_name} cannot be empty; use ('*',) for all")
            if any(not value for value in values):
                raise ValueError(f"{field_name} cannot contain empty patterns")
            object.__setattr__(self, field_name, values)

    @staticmethod
    def _matches(value: str, patterns: tuple[str, ...]) -> bool:
        return any(fnmatchcase(value, pattern) for pattern in patterns)

    def allows_section(self, section: str) -> bool:
        return (
            self._matches(section, self.include_sections)
            and not self._matches(section, self.exclude_sections)
        )

    def allows(self, section: str, family: str) -> bool:
        return (
            self.allows_section(section)
            and self._matches(family, self.include_families)
            and not self._matches(family, self.exclude_families)
        )

    @property
    def is_default(self) -> bool:
        return (
            self.include_sections == ("*",)
            and not self.exclude_sections
            and self.include_families == ("*",)
            and not self.exclude_families
        )


def _unique_primitives(pset: gp.PrimitiveSetTyped):
    seen: set[str] = set()
    for primitives in pset.primitives.values():
        for primitive in primitives:
            if primitive.name in seen:
                continue
            seen.add(primitive.name)
            if primitive.name not in getattr(pset, "gp_primitive_family", {}):
                # Generation-only scaffolding such as static passthroughs is not
                # part of the public grammar.
                continue
            yield primitive


def grammar_rows(
    pset: gp.PrimitiveSetTyped | None = None,
    *,
    level: Literal["section", "family", "signature"] = "family",
) -> list[dict[str, object]]:
    """Return the effective grammar as machine-readable rows.

    ``section`` gives a compact architecture view, ``family`` gives one row per
    operator family, and ``signature`` gives every concrete typed overload.
    """

    if pset is None:
        from flows.gp.factory import make_pset

        pset = make_pset()

    family_of = pset.gp_primitive_family
    section_of = pset.gp_primitive_section
    primitives = list(_unique_primitives(pset))

    if level == "signature":
        return [
            {
                "section": section_of[primitive.name],
                "family": family_of[primitive.name],
                "primitive": primitive.name,
                "inputs": tuple(type_.__name__ for type_ in primitive.args),
                "output": primitive.ret.__name__,
            }
            for primitive in sorted(
                primitives,
                key=lambda p: (section_of[p.name], family_of[p.name], p.name),
            )
        ]

    grouped: dict[tuple[str, ...], list] = {}
    for primitive in primitives:
        section = section_of[primitive.name]
        family = family_of[primitive.name]
        key = (section,) if level == "section" else (section, family)
        grouped.setdefault(key, []).append(primitive)

    rows: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        section = key[0]
        families = sorted({family_of[value.name] for value in values})
        inputs = sorted(
            {
                "(" + ", ".join(type_.__name__ for type_ in value.args) + ")"
                for value in values
            }
        )
        outputs = sorted({value.ret.__name__ for value in values})
        row: dict[str, object] = {
            "section": section,
            "families": tuple(families),
            "family_count": len(families),
            "overloads": len(values),
            "inputs": tuple(inputs),
            "outputs": tuple(outputs),
        }
        if level == "family":
            row["family"] = key[1]
            row.pop("families")
            row.pop("family_count")
        rows.append(row)
    return rows


def grammar_families(
    pset: gp.PrimitiveSetTyped | None = None,
    *,
    section: str | None = None,
) -> tuple[str, ...]:
    """List active families, optionally restricted to a section pattern."""

    rows = grammar_rows(pset, level="family")
    families = {
        str(row["family"])
        for row in rows
        if section is None or fnmatchcase(str(row["section"]), section)
    }
    return tuple(sorted(families))


def format_grammar_table(
    pset: gp.PrimitiveSetTyped | None = None,
    *,
    level: Literal["section", "family", "signature"] = "section",
) -> str:
    """Format the effective grammar as a Markdown table."""

    rows = grammar_rows(pset, level=level)
    if level == "section":
        lines = [
            "| Section | Families | Overloads | Outputs |",
            "|---|---:|---:|---|",
        ]
        for row in rows:
            outputs = ", ".join(row["outputs"])
            lines.append(
                f"| {row['section']} | {row['family_count']} | {row['overloads']} | {outputs} |"
            )
        return "\n".join(lines)

    if level == "family":
        lines = [
            "| Section | Family | Overloads | Inputs | Outputs |",
            "|---|---|---:|---|---|",
        ]
        for row in rows:
            inputs = "<br>".join(row["inputs"])
            outputs = ", ".join(row["outputs"])
            lines.append(
                f"| {row['section']} | {row['family']} | {row['overloads']} | {inputs} | {outputs} |"
            )
        return "\n".join(lines)

    lines = [
        "| Section | Family | Primitive | Inputs | Output |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        inputs = ", ".join(row["inputs"])
        lines.append(
            f"| {row['section']} | {row['family']} | {row['primitive']} | ({inputs}) | {row['output']} |"
        )
    return "\n".join(lines)


__all__ = [
    "GRAMMAR_SECTIONS",
    "GrammarPolicy",
    "format_grammar_table",
    "grammar_families",
    "grammar_rows",
]
