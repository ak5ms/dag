from __future__ import annotations

from deap import gp

from flows.gp.pset import make_pset


def signature_rows(pset: gp.PrimitiveSetTyped) -> list[tuple[str, str, str, str]]:
    seen: set[str] = set()
    rows: list[tuple[str, str, str, str]] = []
    for primitives in pset.primitives.values():
        for primitive in primitives:
            if primitive.name in seen:
                continue
            seen.add(primitive.name)
            family = pset.gp_primitive_family[primitive.name]
            args = ", ".join(type_.__name__ for type_ in primitive.args)
            rows.append((family, primitive.name, args, primitive.ret.__name__))
    rows.sort()
    return rows


def format_signature_table(pset: gp.PrimitiveSetTyped | None = None) -> str:
    pset = pset or make_pset()
    rows = signature_rows(pset)
    lines = [
        f"TOTAL_PRIMITIVES={len(rows)}",
        f"TOTAL_FAMILIES={len(pset.gp_operator_families)}",
        f"DSL_FAMILIES={len(pset.gp_dsl_operator_families)}",
        f"COMPOSITE_FAMILIES={len(pset.gp_composite_operator_families)}",
    ]
    lines.extend(
        f"{family}\t{name}\t({args}) -> {ret}"
        for family, name, args, ret in rows
    )
    return "\n".join(lines)


def main() -> None:
    print(format_signature_table())


if __name__ == "__main__":
    main()


__all__ = ["format_signature_table", "signature_rows"]
