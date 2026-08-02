from trading_dsl_engine.base.dsl import cumsum, ewm, ffill, shift, var
from trading_dsl_engine.ir import compile_ir
from trading_dsl_engine.ir.types import SCALAR, VECTOR


def test_temporal_lane_operators_preserve_scalar_input_type() -> None:
    for formula in (
        cumsum(var("x")),
        ffill(var("x")),
        shift(var("x"), 2),
        ewm(var("x"), 21),
    ):
        program = compile_ir(formula, input_value_types={"x": SCALAR})
        assert program.nodes[program.output_id].value_type == SCALAR


def test_temporal_lane_operators_preserve_vector_input_type() -> None:
    for formula in (
        cumsum(var("x")),
        ffill(var("x")),
        shift(var("x"), 2),
        ewm(var("x"), 21),
    ):
        program = compile_ir(formula, input_value_types={"x": VECTOR})
        assert program.nodes[program.output_id].value_type == VECTOR
