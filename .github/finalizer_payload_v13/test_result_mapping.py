from __future__ import annotations

from collections import OrderedDict

from trading_dsl_engine.base.result_mapping import ResultMapping, support_formula_mappings


def test_result_mapping_map_and_flatten_preserve_nested_order():
    values = ResultMapping((("signals", ResultMapping((("fast", 1), ("slow", 2)))), ("risk", 3)))
    mapped = values.map(lambda value: value * 10)
    assert mapped == {"signals": {"fast": 10, "slow": 20}, "risk": 30}
    assert list(mapped.flatten().items()) == [
        (("signals", "fast"), 10),
        (("signals", "slow"), 20),
        (("risk",), 30),
    ]


def test_formula_mapping_adapter_compiles_flat_once_and_reconstructs_load():
    captured = []

    class Result:
        def load(self):
            return ("FAST", "SLOW", "RISK")

    class Runtime:
        def run(self):
            return Result()

    def compile_flat(formulas):
        captured.append(formulas)
        return Runtime()

    compile_formula = support_formula_mappings(compile_flat)
    runtime = compile_formula(OrderedDict((("signals", OrderedDict((("fast", "f5"), ("slow", "f20")))), ("risk", "f60"))))
    loaded = runtime.run().load()
    assert captured == [["f5", "f20", "f60"]]
    assert isinstance(loaded, ResultMapping)
    assert loaded["signals"]["fast"] == "FAST"
    assert loaded["signals"]["slow"] == "SLOW"
    assert loaded["risk"] == "RISK"


def test_formula_mapping_adapter_rejects_empty_nested_mapping():
    compile_formula = support_formula_mappings(lambda formulas: object())
    try:
        compile_formula({"valid": "x", "empty": {}})
    except ValueError as exc:
        assert "cannot be empty" in str(exc)
    else:
        raise AssertionError("empty nested formula mapping was accepted")


def test_public_cpp_stream_compiler_installs_mapping_adapter():
    from trading_dsl_engine import cpp_stream

    assert getattr(cpp_stream.compile_formula, "_supports_formula_mappings", False)
