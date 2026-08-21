from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from trading_dsl_engine.base.dsl import DSLFunctionRegistry
from trading_dsl_engine.cpp_stream.python.compile import (
    FormulaInput,
    compile_formula as _compile_formula,
)
from trading_dsl_engine.cpp_stream.python.runtime import CppStreamRuntime, RunResult
from trading_dsl_engine.cpp_stream.python.source_types import InputTypeSpec
from trading_dsl_engine.cpp_stream.python.sources import (
    InputSource,
    PreparedSource,
    SourceAdapter,
    SourceInfo,
    SourceValue,
    inspect_source,
    inspect_source_mapping,
    open_source,
    register_source_adapter,
    source,
)
from trading_dsl_engine.cpp_stream.python.utils import *  # noqa: F403
from trading_dsl_engine.cpp_stream.python.utils import __all__ as _utils_all
from trading_dsl_engine.cpp_stream.python.xs_gauss import xs_gauss


def infer_n_instruments(infos: Mapping[str, SourceInfo]) -> int:
    """Infer the instrument extent from source row shapes."""
    counts = Counter(
        int(info.input_type.row_shape[0])
        for info in infos.values()
        if info.input_type.row_shape
        and int(info.input_type.row_shape[0]) > 1
    )
    if not counts:
        raise ValueError(
            "could not infer n_instruments from scalar-only source metadata; "
            "pass n_instruments explicitly"
        )
    maximum = max(counts.values())
    winners = sorted(extent for extent, count in counts.items() if count == maximum)
    if len(winners) != 1:
        raise ValueError(
            "ambiguous n_instruments from source row shapes: "
            f"counts={dict(sorted(counts.items()))}; pass n_instruments explicitly"
        )
    return winners[0]


def compile_formula(
    formula: FormulaInput,
    data: Mapping[str, SourceValue] | None = None,
    *,
    n_instruments: int | None = None,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
    default_group_capacity: int = 64,
    key_cardinalities: Mapping[str, int] | None = None,
    prefetch_rows: int = 16,
    input_types: Mapping[str, InputTypeSpec] | None = None,
) -> CppStreamRuntime:
    """Compile one or many formulas, inferring N from supplied sources."""
    resolved_n = n_instruments
    if data is not None and resolved_n is None:
        resolved_n = infer_n_instruments(
            inspect_source_mapping(data, expected_types=input_types)
        )
    return _compile_formula(
        formula,
        data,
        n_instruments=resolved_n,
        dsl_registry=dsl_registry,
        column_names=column_names,
        default_group_capacity=default_group_capacity,
        key_cardinalities=key_cardinalities,
        prefetch_rows=prefetch_rows,
        input_types=input_types,
    )


__all__ = [
    "compile_formula",
    "infer_n_instruments",
    "CppStreamRuntime",
    "RunResult",
    "InputTypeSpec",
    "InputSource",
    "PreparedSource",
    "SourceAdapter",
    "SourceInfo",
    "inspect_source",
    "inspect_source_mapping",
    "open_source",
    "register_source_adapter",
    "source",
    "xs_gauss",
    *_utils_all,
]
