from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one patch anchor, found {count}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1))


def patch_clarabel_native() -> None:
    path = ROOT / "src/trading_dsl_engine/cpp_stream/optimizer/clarabel_native.py"
    replace_once(
        path,
        """@dataclass(frozen=True, slots=True)\nclass FieldAlias:\n""",
        """@dataclass(frozen=True, slots=True)\nclass ConstraintValueLayout:\n    name: str\n    constraint_index: int\n    label: str | None\n    shape: tuple[int, ...]\n    size: int\n\n\n@dataclass(frozen=True, slots=True)\nclass FieldAlias:\n""",
    )
    replace_once(
        path,
        """    clarabel: ClarabelNativePaths\n    instrument_count: int | None = None\n""",
        """    clarabel: ClarabelNativePaths\n    instrument_count: int | None = None\n    constraint_values: tuple[ConstraintValueLayout, ...] = ()\n""",
    )
    replace_once(
        path,
        """            duals=self.duals,\n            aliases=self.aliases,\n""",
        """            duals=self.duals,\n            aliases=self.aliases,\n            constraint_values=self.constraint_values,\n""",
    )
    replace_once(
        path,
        """    duals: tuple[DualLayout, ...],\n    aliases: tuple[FieldAlias, ...],\n) -> FieldLayout:\n""",
        """    duals: tuple[DualLayout, ...],\n    aliases: tuple[FieldAlias, ...],\n    constraint_values: tuple[ConstraintValueLayout, ...] = (),\n) -> FieldLayout:\n""",
    )
    replace_once(
        path,
        """    for dual_index, dual in enumerate(duals):\n""",
        """    for value_index, value in enumerate(constraint_values):\n        bases = [\n            value.name,\n            f\"constraint[{value.constraint_index}].value\",\n        ]\n        if value.label is not None:\n            bases.append(f\"{value.label}.value\")\n        for base in bases:\n            index_text = _match_base_field(name, base)\n            if index_text is _NO_FIELD_MATCH:\n                continue\n            return _indexed_result_layout(\n                name,\n                kind=\"constraint_value\",\n                source_name=value.name,\n                source_index=value_index,\n                shape=value.shape,\n                size=value.size,\n                index_text=index_text,\n            )\n    for dual_index, dual in enumerate(duals):\n""",
    )
    replace_once(
        path,
        """    available.extend((\"dual[index]\", \"constraint[index].dual\"))\n""",
        """    available.extend((\"dual[index]\", \"constraint[index].dual\"))\n    available.extend(\n        f\"constraint[{value.constraint_index}].value\"\n        for value in constraint_values\n    )\n""",
    )
    replace_once(
        path,
        """    \"DualLayout\",\n    \"FieldAlias\",\n""",
        """    \"DualLayout\",\n    \"ConstraintValueLayout\",\n    \"FieldAlias\",\n""",
    )


def patch_factory() -> None:
    path = ROOT / "src/trading_dsl_engine/cpp_stream/optimizer/factory.py"
    replace_once(
        path,
        """    ClarabelNativePaths,\n    DualLayout,\n""",
        """    ClarabelNativePaths,\n    ConstraintValueLayout,\n    DualLayout,\n""",
    )
    replace_once(path, "_FACTORY_CACHE_SCHEMA = 4", "_FACTORY_CACHE_SCHEMA = 5")
    replace_once(
        path,
        """    aliases: tuple[FieldAlias, ...]\n    instrument_count: int\n""",
        """    aliases: tuple[FieldAlias, ...]\n    instrument_count: int\n    constraint_values: tuple[ConstraintValueLayout, ...] = ()\n""",
    )
    replace_once(
        path,
        """            duals=self.duals,\n            aliases=self.aliases,\n""",
        """            duals=self.duals,\n            aliases=self.aliases,\n            constraint_values=self.constraint_values,\n""",
    )
    start = "def _augment_constraint_values(cp, problem, requested_fields):\n"
    end = "\n\ndef _call_with_named_values(factory, signature, values):\n"
    text = path.read_text()
    if "def _constraint_value_layouts(" not in text:
        begin = text.index(start)
        finish = text.index(end, begin)
        replacement = """def _constraint_value_layouts(cp, problem, requested_fields):\n    requested = _requested_constraint_values(problem, requested_fields)\n    layouts = []\n    for index in requested:\n        constraint = problem.constraints[index]\n        expression = _constraint_value_expression(cp, constraint)\n        if not expression.is_affine():\n            raise ValueError(\n                f\"constraint value {index} must be affine for native \"\n                \"post-solve evaluation\"\n            )\n        layouts.append(\n            ConstraintValueLayout(\n                f\"v{index}\",\n                index,\n                _constraint_label(constraint),\n                tuple(int(extent) for extent in expression.shape),\n                int(expression.size),\n            )\n        )\n    return tuple(layouts)\n"""
        path.write_text(text[:begin] + replacement + text[finish:])
    replace_once(
        path,
        """        problem, aliases = _augment_constraint_values(\n            cp, problem, requested_fields\n        )\n""",
        """        constraint_values = _constraint_value_layouts(\n            cp, problem, requested_fields\n        )\n""",
    )
    replace_once(path, "        return problem, aliases\n", "        return problem, constraint_values\n")
    replace_once(
        path,
        """    def _prototype(self, problem, aliases, n_instruments):\n""",
        """    def _prototype(self, problem, constraint_values, n_instruments):\n""",
    )
    replace_once(
        path,
        """            tuple(\n                FieldAlias(name, primal_name)\n                for name, primal_name in sorted(aliases.items())\n            ),\n            n_instruments,\n        )\n\n    def _cache_key(self, problem, n_instruments: int) -> str:\n""",
        """            (),\n            n_instruments,\n            constraint_values,\n        )\n\n    def _cache_key(\n        self,\n        problem,\n        n_instruments: int,\n        constraint_values: tuple[ConstraintValueLayout, ...],\n    ) -> str:\n""",
    )
    replace_once(
        path,
        """            \"constraints\": [\n""",
        """            \"constraint_values\": [\n                (value.constraint_index, value.label, value.shape)\n                for value in constraint_values\n            ],\n            \"constraints\": [\n""",
    )
    replace_once(
        path,
        """        problem, aliases = self._instantiate_problem(\n""",
        """        problem, constraint_values = self._instantiate_problem(\n""",
    )
    replace_once(
        path,
        """            return self._prototype(problem, aliases, resolved_n)\n\n        cache_key = self._cache_key(problem, int(n_instruments))\n""",
        """            return self._prototype(problem, constraint_values, resolved_n)\n\n        cache_key = self._cache_key(\n            problem, int(n_instruments), constraint_values\n        )\n""",
    )
    replace_once(
        path,
        """                    field_aliases=aliases,\n                    force=force,\n""",
        """                    constraint_value_indices=tuple(\n                        value.constraint_index for value in constraint_values\n                    ),\n                    field_aliases={},\n                    force=force,\n""",
    )


if __name__ == "__main__":
    patch_clarabel_native()
    patch_factory()
