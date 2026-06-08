from importlib.util import module_from_spec, spec_from_file_location
from math import inf
from pathlib import Path

from trading_dsl_engine.jax_flat import ValueRange, compile_formula, field, metadata
from trading_dsl_engine.jax_flat.ops import NaryOp, OP_FACTORIES


def _session_pov_benchmark_module():
    path = Path(__file__).resolve().parents[2] / "examples" / "session_pov_benchmark.py"
    spec = spec_from_file_location("session_pov_benchmark", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_any_node(runtime, label, expected_types, expected_range=None, expected_units=None, width=None):
    matches = [node.metadata for node in runtime.get_node_metadata(label) if node.metadata.types == expected_types]
    if expected_range is not None:
        matches = [meta for meta in matches if meta.range == expected_range]
    if expected_units is not None:
        matches = [meta for meta in matches if meta.units.as_dict() == expected_units and not meta.units.is_unknown()]
    if width is not None:
        matches = [meta for meta in matches if meta.width == width]
    assert matches, label


def test_runtime_exposes_unit_exponents_and_range_for_formula():
    schema = metadata(
        {
            "close": field(units={"dollar": 1}, range="real", types=("price",)),
            "volume": field(units={"shares": 1}, range="nonnegative", types=("volume",)),
        },
        type_relations=(("price", "currency"),),
    )

    runtime = compile_formula("close * volume ** 3", metadata=schema, cpp=False)

    assert runtime.get_units().as_dict() == {"dollar": 1.0, "shares": 3.0}
    assert runtime.get_range().as_tuple() == (-inf, inf)
    assert runtime.program.metadata.input_fields["close"].types == frozenset({"price"})
    assert runtime.get_type_relations().closure(("price",)) == frozenset({"price", "currency"})
    graph = runtime.get_type_relations()
    price_idx = graph.types.index("price")
    currency_idx = graph.types.index("currency")
    assert graph.as_matrix()[price_idx][currency_idx]


def test_runtime_range_uses_interval_arithmetic_through_operators():
    runtime = compile_formula(
        "abs(ret - 0.5) / 2",
        metadata={"ret": field(range=(0, 1), types=("return",))},
        cpp=False,
    )

    assert runtime.get_units().as_dict() == {}
    assert runtime.get_range() == ValueRange(0.0, 0.25)


def test_same_variable_division_reduces_to_unit_range():
    runtime = compile_formula(
        "close / close",
        metadata={"close": field(units={"dollar": 1}, range="real", types=("price",))},
        cpp=False,
    )

    assert runtime.get_units().as_dict() == {}
    assert runtime.get_range() == ValueRange(1.0, 1.0)


def test_algebraic_reductions_cover_arithmetic_identities():
    schema = {"close": field(units={"dollar": 1}, range=(10, 20), types=("price",))}

    identity_cases = [
        ("close + 0", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("0 + close", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("close - 0", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("0 - close", {"dollar": 1.0}, ValueRange(-20.0, -10.0)),
        ("close - close", {"dollar": 1.0}, ValueRange(0.0, 0.0)),
        ("close * 1", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("1 * close", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("close * 0", {"dollar": 1.0}, ValueRange(0.0, 0.0)),
        ("close / 1", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("close ** 1", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("close ** 0", {}, ValueRange(1.0, 1.0)),
        ("close * close", {"dollar": 2.0}, ValueRange(100.0, 400.0)),
        ("1 ** close", {}, ValueRange(1.0, 1.0)),
        ("close % close", {"dollar": 1.0}, ValueRange(0.0, 0.0)),
        ("0 % close", {}, ValueRange(0.0, 0.0)),
        ("abs(close)", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("fillna(close, close)", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
    ]
    for formula, units, expected_range in identity_cases:
        runtime = compile_formula(formula, metadata=schema, cpp=False)
        assert runtime.get_units().as_dict() == units, formula
        assert runtime.get_range() == expected_range, formula


def test_algebraic_reductions_cover_conditionals_and_booleans():
    schema = {"signal": field(range=(0, 1), types=("boolean",))}

    cases = [
        ("where(1, signal, 0)", ValueRange(0.0, 1.0)),
        ("where(0, 1, signal)", ValueRange(0.0, 1.0)),
        ("where(signal, 7, 7)", ValueRange(7.0, 7.0)),
        ("signal == signal", ValueRange(1.0, 1.0)),
        ("signal != signal", ValueRange(0.0, 0.0)),
        ("signal < signal", ValueRange(0.0, 0.0)),
        ("signal ^ signal", ValueRange(0.0, 0.0)),
        ("signal & 1", ValueRange(0.0, 1.0)),
        ("signal | 0", ValueRange(0.0, 1.0)),
        ("signal & 0", ValueRange(0.0, 0.0)),
        ("signal | 1", ValueRange(1.0, 1.0)),
        ("isnan(1)", ValueRange(0.0, 0.0)),
    ]
    for formula, expected_range in cases:
        runtime = compile_formula(formula, metadata=schema, cpp=False)
        assert runtime.get_units().as_dict() == {}, formula
        assert runtime.get_range() == expected_range, formula


def test_metadata_auto_traces_nary_ops_for_ranges_and_types():
    schema = {
        "x": field(range=(-1.2, 2.7)),
        "y": field(range=(-2, 3)),
    }

    cases = [
        ("ceil(x)", ValueRange(-1.0, 3.0), frozenset()),
        ("floor(x)", ValueRange(-2.0, 2.0), frozenset()),
        ("round(x)", ValueRange(-1.0, 3.0), frozenset()),
        ("fraction(x)", ValueRange(0.0, 0.8), frozenset()),
        ("sign(x)", ValueRange(-1.0, 1.0), frozenset()),
        ("arctan(x)", ValueRange(-0.8760580505981934, 1.2160906747839564), frozenset()),
        ("x > y", ValueRange(0.0, 1.0), frozenset({"boolean"})),
    ]
    for formula, expected_range, expected_types in cases:
        runtime = compile_formula(formula, metadata=schema, cpp=False)
        assert runtime.get_range() == expected_range, formula
        assert runtime.get_types() == expected_types, formula


def test_metadata_auto_traces_new_nary_op_from_operator_implementation(monkeypatch):
    monkeypatch.setitem(OP_FACTORIES, ("double_plus_one", 1), lambda: NaryOp(lambda x: x * 2.0 + 1.0))
    monkeypatch.setitem(OP_FACTORIES, ("is_negative", 1), lambda: NaryOp(lambda x: x < 0.0))

    runtime = compile_formula("double_plus_one(x)", metadata={"x": field(range=(1, 3))}, cpp=False)
    bool_runtime = compile_formula("is_negative(x)", metadata={"x": field(range=(-1, 1))}, cpp=False)

    assert runtime.get_units().as_dict() == {}
    assert runtime.get_range() == ValueRange(3.0, 7.0)
    assert bool_runtime.get_range() == ValueRange(0.0, 1.0)
    assert bool_runtime.get_types() == frozenset({"boolean"})


def test_roll_ret_macro_traces_to_reasonable_return_metadata():
    runtime = compile_formula(
        "roll_ret(close, 1, 1)",
        metadata={"close": field(units={"dollar": 1}, range=(100, 200), types=("price",))},
        cpp=False,
    )

    assert runtime.get_units().as_dict() == {}
    assert not runtime.get_units().is_unknown()
    assert runtime.get_range() == ValueRange(-0.5, 1.0)
    assert runtime.get_types() == frozenset({"return"})


def test_session_pov_roll_formula_metadata_traces_through_root_roll():
    module = _session_pov_benchmark_module()
    base = 1.7e15
    runtime = compile_formula(
        module._formula("roll"),
        metadata={
            "ev_ts": field(range=(base, base + module.DAY_US), types=("timestamp",)),
            "session_start": field(range=(base, base + module.DAY_US), types=("timestamp",)),
            "session_end": field(range=(base + module.DAY_US, base + 2.0 * module.DAY_US), types=("timestamp",)),
            "volume": field(units="shares", range=(0, 200), types=("volume",)),
            "is_tradable0": field(range=(0, 1), types=("boolean",)),
            "is_tradable1": field(range=(0, 1), types=("boolean",)),
            "wdte": field(range=(0, 1), types=("boolean",)),
            "vwap0": field(units="dollar", range=(100, 200), types=("price",)),
            "vwap1": field(units="dollar", range=(100, 200), types=("price",)),
        },
        cpp=False,
    )

    assert runtime.get_units().as_dict() == {}
    assert not runtime.get_units().is_unknown()
    assert runtime.get_range() == ValueRange(-1.0, 2.0)
    assert runtime.get_types() == frozenset({"return"})

    _assert_any_node(runtime, "volume_for_seen_session", frozenset({"volume"}), ValueRange(0.0, 200.0), {"shares": 1.0})
    _assert_any_node(runtime, "self_", frozenset({"volume"}), ValueRange(0.0, 200.0), {"shares": 1.0})
    _assert_any_node(runtime, "cumsum", frozenset({"volume"}), ValueRange(0.0, 200.0), {"shares": 1.0})
    _assert_any_node(runtime, "groupby", frozenset({"volume"}), ValueRange(0.0, 200.0), {"shares": 1.0})
    _assert_any_node(runtime, "pct_seen_session_volume", frozenset({"ratio"}), ValueRange(0.0, 1.0), {})
    _assert_any_node(runtime, "where", frozenset({"price"}), ValueRange(100.0, 200.0), {"dollar": 1.0})
    _assert_any_node(runtime, "shift", frozenset({"price"}), ValueRange(100.0, 200.0), {"dollar": 1.0})
    _assert_any_node(runtime, "div", frozenset({"ratio"}), ValueRange(0.5, 2.0), {})
    _assert_any_node(runtime, "sub", frozenset({"return"}), ValueRange(-0.5, 1.0), {})
    _assert_any_node(runtime, "cat", frozenset({"return"}), ValueRange(-0.5, 1.0), {}, width=2)
    _assert_any_node(runtime, "einsum", frozenset({"return"}), ValueRange(-1.0, 2.0), {})


def test_unit_incompatible_addition_keeps_formula_compilable_with_unknown_units():
    runtime = compile_formula(
        "close + volume",
        metadata={
            "close": field(units={"dollar": 1}, range="real"),
            "volume": field(units={"shares": 1}, range="nonnegative"),
        },
        cpp=False,
    )

    assert runtime.get_units().is_unknown()
    assert runtime.get_units().as_dict() == {}
    assert runtime.get_range().as_tuple() == (-inf, inf)
