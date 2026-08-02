from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def path(name: str) -> Path:
    return ROOT / name


def replace_once(name: str, old: str, new: str) -> None:
    target = path(name)
    text = target.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"{name}: expected exactly one match for {old[:80]!r}, found {text.count(old)}")
    target.write_text(text.replace(old, new, 1))


# ---------------------------------------------------------------------------
# Public expression and DSL API.
# ---------------------------------------------------------------------------
replace_once(
    "src/trading_dsl_engine/base/parser.py",
    "    def groupby(self, key, rhs=None, *args):\n",
    '''    def sum(self, axis=None):
        from trading_dsl_engine.base.dsl import reduction

        return reduction("sum", self, axis=axis)

    def mean(self, axis=None):
        from trading_dsl_engine.base.dsl import reduction

        return reduction("mean", self, axis=axis)

    def std(self, axis=None, ddof=0):
        from trading_dsl_engine.base.dsl import reduction

        return reduction("std", self, axis=axis, ddof=ddof)

    def emit(self, mode="last"):
        from trading_dsl_engine.base.dsl import emit

        return emit(self, mode=mode)

    def groupby(self, key, rhs=None, *args):
''',
)
replace_once(
    "src/trading_dsl_engine/base/parser.py",
    "        if isinstance(node, ast.Tuple):\n            if len(node.elts) == 0:\n                raise FormulaParseError(\"Key tuples cannot be empty\")\n            return KeyTuple(tuple(self._expr(item) for item in node.elts))\n",
    "        if isinstance(node, (ast.Tuple, ast.List)):\n"
    "            if len(node.elts) == 0:\n"
    "                raise FormulaParseError(\"Expression lists cannot be empty\")\n"
    "            return KeyTuple(tuple(self._expr(item) for item in node.elts))\n",
)

replace_once(
    "src/trading_dsl_engine/base/dsl.py",
    "    if isinstance(value, tuple):\n",
    "    if isinstance(value, (tuple, list)):\n",
)
replace_once(
    "src/trading_dsl_engine/base/dsl.py",
    "            \"xstd\",\n            \"mean\",\n            \"outer\",\n",
    "            \"xstd\",\n            \"outer\",\n",
)
replace_once(
    "src/trading_dsl_engine/base/dsl.py",
    "    \"groupby\": _dsl_signature(\"key_tuple\", \"lhs\", \"op_using_self_\"),\n",
    "    \"groupby\": _dsl_signature(\"key_tuple\", \"lhs\", \"op_using_self_\"),\n"
    "    \"sum\": _dsl_signature(\"x\", \"axis\", defaults={\"axis\": None}),\n"
    "    \"mean\": _dsl_signature(\"x\", \"axis\", defaults={\"axis\": None}),\n"
    "    \"std\": _dsl_signature(\"x\", \"axis\", \"ddof\", defaults={\"axis\": None, \"ddof\": 0}),\n"
    "    \"emit\": _dsl_signature(\"x\", \"mode\", defaults={\"mode\": \"last\"}),\n",
)
replace_once(
    "src/trading_dsl_engine/base/dsl.py",
    "def call(name: str, *args, **kwargs) -> Expr:\n    return Call(\n        name,\n        tuple(ensure_expr(a) for a in args),\n        tuple((key, ensure_expr(value)) for key, value in kwargs.items()),\n    )\n\n\nGROUPBY_VALUE_PLACEHOLDER",
    '''def call(name: str, *args, **kwargs) -> Expr:
    return Call(
        name,
        tuple(ensure_expr(a) for a in args),
        tuple((key, ensure_expr(value)) for key, value in kwargs.items()),
    )


def _axis_expr(axis) -> Expr:
    if isinstance(axis, Expr):
        return axis
    if isinstance(axis, (int, float)):
        return Number(float(axis))
    if isinstance(axis, (tuple, list)):
        if not axis:
            raise ValueError("axis cannot be empty")
        return KeyTuple(tuple(_axis_expr(item) for item in axis))
    raise TypeError("axis must be an int or a non-empty list/tuple of ints")


def reduction(name: str, x, *, axis=None, ddof=0) -> Expr:
    if name not in {"sum", "mean", "std"}:
        raise ValueError(f"unsupported reduction {name!r}")
    kwargs = []
    if axis is not None:
        kwargs.append(("axis", _axis_expr(axis)))
    if name == "std" and ddof != 0:
        kwargs.append(("ddof", ensure_expr(ddof)))
    return Call(name, (ensure_expr(x),), tuple(kwargs))


def emit(x, *, mode="last") -> Expr:
    if mode != "last":
        raise ValueError("emit currently supports only mode='last'")
    return Call("emit", (ensure_expr(x),), (("mode", String(mode)),))


GROUPBY_VALUE_PLACEHOLDER''',
)
replace_once(
    "src/trading_dsl_engine/base/dsl.py",
    "cat = op(\"cat\")\ngroupby = op(\"groupby\")\n",
    "cat = op(\"cat\")\ngroupby = op(\"groupby\")\nsum = op(\"sum\")\nstd = op(\"std\")\n",
)

# ---------------------------------------------------------------------------
# Backend-neutral IR.
# ---------------------------------------------------------------------------
replace_once(
    "src/trading_dsl_engine/ir/ops.py",
    "@dataclass(frozen=True, slots=True)\nclass CumsumOp:\n    pass\n\n\n",
    '''@dataclass(frozen=True, slots=True)
class CumsumOp:
    pass


@dataclass(frozen=True, slots=True)
class ReductionOp:
    kind: str
    axes: tuple[int, ...]
    ddof: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"sum", "mean", "std"}:
            raise ValueError(f"unsupported reduction kind {self.kind!r}")
        if self.ddof < 0:
            raise ValueError("reduction ddof must be >= 0")

    @property
    def temporal(self) -> bool:
        return 0 in self.axes


@dataclass(frozen=True, slots=True)
class EmitOp:
    mode: str = "last"

    def __post_init__(self) -> None:
        if self.mode != "last":
            raise ValueError(f"unsupported emit mode {self.mode!r}")


''',
)
replace_once(
    "src/trading_dsl_engine/ir/ops.py",
    "    | CumsumOp\n    | FFillOp\n",
    "    | CumsumOp\n    | ReductionOp\n    | EmitOp\n    | FFillOp\n",
)
replace_once(
    "src/trading_dsl_engine/ir/ops.py",
    "    \"CumsumOp\",\n    \"FFillOp\",\n",
    "    \"CumsumOp\",\n    \"ReductionOp\",\n    \"EmitOp\",\n    \"FFillOp\",\n",
)
replace_once(
    "src/trading_dsl_engine/ir/__init__.py",
    "    CumsumOp,\n    EwmOp,\n",
    "    CumsumOp,\n    EmitOp,\n    EwmOp,\n",
)
replace_once(
    "src/trading_dsl_engine/ir/__init__.py",
    "    NaryOp,\n    XsRankOp,\n",
    "    NaryOp,\n    ReductionOp,\n    XsRankOp,\n",
)
replace_once(
    "src/trading_dsl_engine/ir/__init__.py",
    "    \"CumsumOp\",\n    \"EwmOp\",\n",
    "    \"CumsumOp\",\n    \"ReductionOp\",\n    \"EmitOp\",\n    \"EwmOp\",\n",
)

# ---------------------------------------------------------------------------
# IR frontend: axes are against (time, *row_shape).
# ---------------------------------------------------------------------------
replace_once(
    "src/trading_dsl_engine/ir/frontend.py",
    "    CumsumOp,\n    CustomCallOp,\n",
    "    CumsumOp,\n    CustomCallOp,\n    EmitOp,\n",
)
replace_once(
    "src/trading_dsl_engine/ir/frontend.py",
    "    RidgeProjectionOp,\n    ShiftOp,\n",
    "    RidgeProjectionOp,\n    ReductionOp,\n    ShiftOp,\n",
)
insert_before = "def _normalize_ewm(call: Call) -> tuple[Expr, float, int, bool, bool]:\n"
frontend_helpers = '''def _reduction_arguments(call: Call) -> tuple[Expr, Expr | None, int]:
    if call.fn not in {"sum", "mean", "std"}:
        raise FormulaIRCompileError(f"invalid reduction {call.fn!r}")
    names = ("x", "axis", "ddof") if call.fn == "std" else ("x", "axis")
    values: dict[str, Expr] = {}
    explicit: set[str] = set()
    for name, value in zip(names, call.args):
        values[name] = value
        explicit.add(name)
    if len(call.args) > len(names):
        raise FormulaIRCompileError(f"{call.fn} received too many arguments")
    for name, value in call.kwargs:
        if name not in names or name in explicit:
            raise FormulaIRCompileError(f"invalid {call.fn} argument {name!r}")
        values[name] = value
        explicit.add(name)
    if "x" not in values:
        raise FormulaIRCompileError(f"{call.fn} requires x")
    ddof = _literal_int(values.get("ddof", Number(0.0)), "std ddof", 0)
    return values["x"], values.get("axis"), ddof


def _reduction_axes(axis: Expr | None, stream_rank: int) -> tuple[int, ...]:
    if stream_rank <= 0:
        raise FormulaIRCompileError("reduction stream rank must be positive")
    items = (
        tuple(range(stream_rank))
        if axis is None
        else axis.items
        if isinstance(axis, KeyTuple)
        else (axis,)
    )
    normalized: list[int] = []
    for item in items:
        value = _literal_int(item, "reduction axis")
        if value < 0:
            value += stream_rank
        if value < 0 or value >= stream_rank:
            raise FormulaIRCompileError(
                f"reduction axis {value} outside rank {stream_rank}"
            )
        if value in normalized:
            raise FormulaIRCompileError(f"duplicate reduction axis {value}")
        normalized.append(value)
    return tuple(sorted(normalized))


def _normalize_emit(call: Call) -> tuple[Expr, str]:
    if len(call.args) != 1:
        raise FormulaIRCompileError("emit expects one expression")
    values = dict(call.kwargs)
    if len(values) != len(call.kwargs) or set(values) - {"mode"}:
        raise FormulaIRCompileError("emit supports only the mode keyword")
    mode = _literal_string(values.get("mode", String("last")), "emit mode")
    if mode != "last":
        raise FormulaIRCompileError("emit currently supports only mode='last'")
    return call.args[0], mode


'''
replace_once(
    "src/trading_dsl_engine/ir/frontend.py",
    insert_before,
    frontend_helpers + insert_before,
)
replace_once(
    "src/trading_dsl_engine/ir/frontend.py",
    "        if node.fn == \"cat\":\n",
    '''        if node.fn in {"sum", "mean", "std"}:
            expression, axis, ddof = _reduction_arguments(node)
            child = self.build(expression)
            child_type = self.nodes[child].value_type
            try:
                row_shape = child_type.logical_shape
            except ValueError as exc:
                raise FormulaIRCompileError(
                    f"{node.fn} cannot reduce object values"
                ) from exc
            axes = _reduction_axes(axis, 1 + len(row_shape))
            output_shape = tuple(
                extent
                for full_axis, extent in enumerate(row_shape, start=1)
                if full_axis not in axes
            )
            return self._append(
                ReductionOp(node.fn, axes, ddof),
                (child,),
                tensor(output_shape, dtype=child_type.dtype),
            )
        if node.fn == "emit":
            expression, mode = _normalize_emit(node)
            child = self.build(expression)
            return self._append(
                EmitOp(mode), (child,), self.nodes[child].value_type
            )
        if node.fn == "cat":
''',
)
replace_once(
    "src/trading_dsl_engine/ir/frontend.py",
    "        inner_program = Program(\n            tuple(inner.nodes),\n            (inner_root,),\n            (\"__self__\",)\n            + tuple(f\"__capture_{index}__\" for index in range(len(inner.capture_ids))),\n        )\n",
    '''        inner_program = Program(
            tuple(inner.nodes),
            (inner_root,),
            ("__self__",)
            + tuple(f"__capture_{index}__" for index in range(len(inner.capture_ids))),
        )
        if any(
            isinstance(inner_node.op, (ReductionOp, EmitOp))
            for inner_node in inner_program.nodes
        ):
            raise FormulaIRCompileError(
                "reductions and emit are not yet supported inside groupby"
            )
''',
)
replace_once(
    "src/trading_dsl_engine/ir/frontend.py",
    "    root = builder.build(expression)\n    return Program(tuple(builder.nodes), (root,), tuple(builder.inputs))\n",
    '''    root = builder.build(expression)
    for node_id, node in enumerate(builder.nodes):
        terminal = isinstance(node.op, EmitOp) or (
            isinstance(node.op, ReductionOp) and node.op.temporal
        )
        if terminal and node_id != root:
            raise FormulaIRCompileError(
                "temporal reductions and emit('last') must be the terminal output"
            )
    return Program(tuple(builder.nodes), (root,), tuple(builder.inputs))
''',
)

# ---------------------------------------------------------------------------
# Physical lowering.
# ---------------------------------------------------------------------------
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/lowering.py",
    "    CumsumOp,\n    CustomCallOp,\n",
    "    CumsumOp,\n    CustomCallOp,\n    EmitOp,\n",
)
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/lowering.py",
    "    RidgeProjectionOp,\n    ShiftOp,\n",
    "    RidgeProjectionOp,\n    ReductionOp,\n    ShiftOp,\n",
)
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/lowering.py",
    "    output_row_width: int\n    output_shape: tuple[int, ...]\n",
    "    output_row_width: int\n    output_shape: tuple[int, ...]\n    output_mode: str\n",
)
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/lowering.py",
    "        CumsumOp,\n        FFillOp,\n",
    "        CumsumOp,\n        ReductionOp,\n        EmitOp,\n        FFillOp,\n",
)
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/lowering.py",
    "        if isinstance(op, InstrumentBasisMeanOp):\n",
    '''        if isinstance(op, ReductionOp):
            if op.temporal and not is_root:
                raise CppStreamLoweringError(
                    "temporal reductions must be the terminal output"
                )
            out = value_dest(is_root, node_shape)
            stages.append(
                Stage(
                    "reduce",
                    children,
                    out,
                    n_instruments,
                    output_kind=node.value_type.kind,
                    output_width=int(node.value_type.width),
                    op=op,
                )
            )
            sources[node_id] = source_from_dest(out, node_shape)
            continue

        if isinstance(op, EmitOp):
            if not is_root:
                raise CppStreamLoweringError("emit('last') must be the terminal output")
            out = value_dest(True, node_shape)
            stages.append(
                Stage(
                    "emit_last",
                    children,
                    out,
                    n_instruments,
                    output_kind=node.value_type.kind,
                    output_width=int(node.value_type.width),
                    op=op,
                )
            )
            sources[node_id] = source_from_dest(out, node_shape)
            continue

        if isinstance(op, InstrumentBasisMeanOp):
''',
)
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/lowering.py",
    "        root_shape,\n    )\n",
    '''        root_shape,
        "final"
        if isinstance(program.nodes[root].op, EmitOp)
        or (
            isinstance(program.nodes[root].op, ReductionOp)
            and program.nodes[root].op.temporal
        )
        else "rows",
    )
''',
)

# ---------------------------------------------------------------------------
# C++ nodes and code generation.
# ---------------------------------------------------------------------------
path("src/trading_dsl_engine/cpp_stream/cpp/stackdsl/ops/reduction.hpp").write_text(r'''#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "stackdsl/ops/einsum.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template <std::size_t... Axes>
struct AxisList {
    static constexpr std::size_t size = sizeof...(Axes);
    static constexpr std::array<std::size_t, size> values{Axes...};

    static constexpr bool contains(std::size_t axis) noexcept {
        return ((axis == Axes) || ... || false);
    }
};

struct SumReductionPolicy {};
struct MeanReductionPolicy {};
struct StdReductionPolicy {};

template <class Shape, class Axes>
consteval std::size_t reduced_output_size() {
    std::size_t result = 1;
    for (std::size_t axis = 0; axis < Shape::rank; ++axis) {
        if (!Axes::contains(axis)) result *= Shape::dims[axis];
    }
    return result;
}

template <class Shape, class Axes>
STACKDSL_HOT std::size_t reduced_output_index(std::size_t flat) noexcept {
    if constexpr (Axes::size == 0) {
        return flat;
    } else {
        std::array<std::size_t, Shape::rank> indexes{};
        for (std::size_t axis = Shape::rank; axis-- > 0;) {
            indexes[axis] = flat % Shape::dims[axis];
            flat /= Shape::dims[axis];
        }
        std::size_t output = 0;
        for (std::size_t axis = 0; axis < Shape::rank; ++axis) {
            if (!Axes::contains(axis)) {
                output = output * Shape::dims[axis] + indexes[axis];
            }
        }
        return output;
    }
}

template <class Policy, std::size_t Size, std::size_t Ddof>
struct ReductionState {
    alignas(64) std::array<double, Size> total{};
    alignas(64) std::array<double, Size> mean{};
    alignas(64) std::array<double, Size> m2{};
    alignas(64) std::array<std::uint64_t, Size> count{};

    STACKDSL_HOT void reset() noexcept {
        total.fill(0.0);
        mean.fill(0.0);
        m2.fill(0.0);
        count.fill(0);
    }

    STACKDSL_HOT void add(std::size_t index, double value) noexcept {
        if (!finite(value)) return;
        if constexpr (std::is_same_v<Policy, StdReductionPolicy>) {
            const std::uint64_t next_count = count[index] + 1;
            const double delta = value - mean[index];
            mean[index] += delta / static_cast<double>(next_count);
            const double delta2 = value - mean[index];
            m2[index] = std::fma(delta, delta2, m2[index]);
            count[index] = next_count;
        } else {
            total[index] += value;
            ++count[index];
        }
    }

    STACKDSL_HOT double result(std::size_t index) const noexcept {
        if (count[index] == 0) return kNaN;
        if constexpr (std::is_same_v<Policy, SumReductionPolicy>) {
            return total[index];
        } else if constexpr (std::is_same_v<Policy, MeanReductionPolicy>) {
            return total[index] / static_cast<double>(count[index]);
        } else {
            if (count[index] <= Ddof) return kNaN;
            const double denominator = static_cast<double>(count[index] - Ddof);
            return std::sqrt(std::max(0.0, m2[index] / denominator));
        }
    }
};

template <
    class Tensor,
    class Out,
    class Axes,
    class Policy,
    std::size_t Ddof,
    bool Temporal
>
struct ReductionNode {
    static constexpr std::size_t input_size = Tensor::shape::size;
    static constexpr std::size_t output_size =
        reduced_output_size<typename Tensor::shape, Axes>();
    using State = ReductionState<Policy, output_size, Ddof>;

    State state{};

    STACKDSL_HOT void setup() noexcept { state.reset(); }

    template <class Context>
    STACKDSL_HOT static void accumulate(State& target, const Context& ctx) noexcept {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 8
#endif
        for (std::size_t offset = 0; offset < input_size; ++offset) {
            target.add(
                reduced_output_index<typename Tensor::shape, Axes>(offset),
                Tensor::read_flat(ctx, offset)
            );
        }
    }

    template <class Context>
    STACKDSL_HOT static void write_result(
        const State& source, Context& ctx
    ) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        for (std::size_t index = 0; index < output_size; ++index) {
            out[index] = source.result(index);
        }
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        if constexpr (Temporal) {
            accumulate(state, ctx);
        } else {
            State row{};
            row.reset();
            accumulate(row, ctx);
            write_result(row, ctx);
        }
    }

    template <class Context>
    STACKDSL_HOT void finalize(Context& ctx) noexcept {
        write_result(state, ctx);
    }
};

template <class Tensor, class Out>
struct EmitLastNode {
    static constexpr std::size_t size = Tensor::shape::size;
    alignas(64) std::array<double, size> value{};
    bool seen = false;

    STACKDSL_HOT void setup() noexcept {
        value.fill(kNaN);
        seen = false;
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        Tensor::load_contiguous(ctx, 0, size, value.data());
        seen = true;
    }

    template <class Context>
    STACKDSL_HOT void finalize(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        for (std::size_t index = 0; index < size; ++index) {
            out[index] = seen ? value[index] : kNaN;
        }
    }
};

}  // namespace stackdsl
''')
replace_once(
    "src/trading_dsl_engine/cpp_stream/cpp/stackdsl/runtime.hpp",
    '#include "stackdsl/ops/cumsum.hpp"\n',
    '#include "stackdsl/ops/cumsum.hpp"\n#include "stackdsl/ops/reduction.hpp"\n',
)

replace_once(
    "src/trading_dsl_engine/cpp_stream/python/codegen.py",
    "    EwmOp,\n    FFillOp,\n",
    "    EmitOp,\n    EwmOp,\n    FFillOp,\n",
)
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/codegen.py",
    "    RidgeOp,\n    ShiftOp,\n",
    "    ReductionOp,\n    RidgeOp,\n    ShiftOp,\n",
)
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/codegen.py",
    "    if stage.kind == \"cat\":\n",
    '''    if stage.kind == "reduce":
        assert isinstance(stage.op, ReductionOp)
        tensor_source = _tensor_source_type(
            stage.inputs[0], n=n, input_types=input_types
        )
        row_axes = tuple(axis - 1 for axis in stage.op.axes if axis != 0)
        policy = {
            "sum": "stackdsl::SumReductionPolicy",
            "mean": "stackdsl::MeanReductionPolicy",
            "std": "stackdsl::StdReductionPolicy",
        }[stage.op.kind]
        return tmpl(
            "stackdsl::ReductionNode",
            tensor_source,
            out,
            tmpl("stackdsl::AxisList", *(IntArg(axis) for axis in row_axes)),
            Name(policy),
            IntArg(stage.op.ddof),
            BoolArg(stage.op.temporal),
        )
    if stage.kind == "emit_last":
        assert isinstance(stage.op, EmitOp)
        return tmpl(
            "stackdsl::EmitLastNode",
            _tensor_source_type(stage.inputs[0], n=n, input_types=input_types),
            out,
        )
    if stage.kind == "cat":
''',
)
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/codegen.py",
    "class StageView:\n    index: int\n    cpp_type: str\n    checked: bool = False\n",
    "class StageView:\n    index: int\n    cpp_type: str\n    checked: bool = False\n    finalizer: bool = False\n",
)
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/codegen.py",
    "                    True,\n                )\n",
    "                    True,\n                    False,\n                )\n",
)
# The normal StageView constructor has positional index/type only; add a named flag.
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/codegen.py",
    "                StageView(\n                    index,\n                    _stage_type(\n                        stage, n, direct, input_types=input_types\n                    ).render(),\n                )\n",
    '''                StageView(
                    index,
                    _stage_type(
                        stage, n, direct, input_types=input_types
                    ).render(),
                    finalizer=stage.kind in {"reduce", "emit_last"}
                    and plan.output_mode == "final",
                )
''',
)
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/codegen.py",
    "            output_row_width=plan.output_row_width,\n            prefetch_rows=prefetch_rows,\n",
    "            output_row_width=plan.output_row_width,\n"
    "            output_mode=plan.output_mode,\n"
    "            prefetch_rows=prefetch_rows,\n",
)

# Backend serial runner output policy and finalize hook.
template = path("src/trading_dsl_engine/cpp_stream/python/templates/runner.cpp.j2")
text = template.read_text()
text = text.replace(
    "    constexpr std::size_t output_row_bytes = output_row_width * sizeof(double);\n"
    "    stackdsl::MMapFile output(output_path, true, rows * output_row_bytes);\n",
    "    constexpr std::size_t output_row_bytes = output_row_width * sizeof(double);\n"
    "    const std::size_t output_rows = {% if output_mode == 'final' %}1{% else %}rows{% endif %};\n"
    "    stackdsl::MMapFile output(output_path, true, output_rows * output_row_bytes);\n",
)
text = text.replace(
    "        ctx.output = out + t * output_row_width;\n",
    "{% if output_mode == 'final' %}\n"
    "        ctx.output = nullptr;\n"
    "{% else %}\n"
    "        ctx.output = out + t * output_row_width;\n"
    "{% endif %}\n",
)
text = text.replace(
    "        if (async_writeback_bytes > 0) {\n",
    "{% if output_mode != 'final' %}\n"
    "        if (async_writeback_bytes > 0) {\n",
    1,
)
text = text.replace(
    "        }\n    }\n    const auto ended",
    "        }\n{% endif %}\n    }\n"
    "{% if output_mode == 'final' %}\n"
    "    ctx.output = out;\n"
    "{% for stage in stages %}{% if stage.finalizer %}\n"
    "    s{{ stage.index }}.finalize(ctx);\n"
    "{% endif %}{% endfor %}\n"
    "{% endif %}\n"
    "    const auto ended",
    1,
)
text = text.replace(
    "    const std::size_t output_bytes = rows * output_row_bytes;\n",
    "    const std::size_t output_bytes = output_rows * output_row_bytes;\n",
)
template.write_text(text)

# ---------------------------------------------------------------------------
# Runtime result metadata.
# ---------------------------------------------------------------------------
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/runtime.py",
    "class RunResult:\n    output_path: Path\n    rows: int\n    seconds: float\n",
    '''class RunResult:
    output_path: Path
    rows: int
    seconds: float
    output_rows: int
    output_shape: tuple[int, ...]
    output_mode: str
''',
)
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/runtime.py",
    "            return RunResult(\n                output_path=output,\n                rows=int(rows.value),\n                seconds=float(seconds.value),\n            )\n",
    '''            processed_rows = int(rows.value)
            logical_shape = (
                self.plan.output_shape
                if self.plan.output_mode == "final"
                else (processed_rows,) + self.plan.output_shape
            )
            return RunResult(
                output_path=output,
                rows=processed_rows,
                seconds=float(seconds.value),
                output_rows=1 if self.plan.output_mode == "final" else processed_rows,
                output_shape=logical_shape,
                output_mode=self.plan.output_mode,
            )
''',
)
replace_once(
    "src/trading_dsl_engine/cpp_stream/python/runtime.py",
    "            f\"scratch_slots={self.plan.scratch_slots}\",\n",
    "            f\"scratch_slots={self.plan.scratch_slots}\",\n"
    "            f\"output_mode={self.plan.output_mode}\",\n"
    "            f\"output_shape={self.plan.output_shape}\",\n",
)

# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
path("tests/trading_dsl_engine/cpp_stream/test_reductions.py").write_text(r'''from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trading_dsl_engine.base.dsl import cat, cumsum, var
from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.ir.frontend import FormulaIRCompileError


def _run(tmp_path: Path, expression, data: dict[str, np.ndarray]):
    runtime = compile_formula(expression, data, n_instruments=data["x"].shape[1])
    result = runtime.run(out_path=tmp_path / "out.bin")
    values = np.fromfile(result.output_path, dtype=np.float64)
    return runtime, result, values.reshape(result.output_shape or ())


def test_temporal_sum_writes_one_final_vector(tmp_path: Path) -> None:
    x = np.arange(60, dtype=np.float64).reshape(10, 6)
    x[3, 2] = np.nan
    runtime, result, actual = _run(tmp_path, var("x").sum(axis=0), {"x": x})
    np.testing.assert_allclose(actual, np.nansum(x, axis=0))
    assert runtime.plan.output_mode == "final"
    assert result.rows == 10
    assert result.output_rows == 1
    assert result.output_shape == (6,)
    assert result.output_path.stat().st_size == 6 * 8


def test_row_reductions_compose_and_keep_row_emission(tmp_path: Path) -> None:
    x = np.arange(48, dtype=np.float64).reshape(8, 6)
    expression = var("x").sum(axis=1) + 2.0
    runtime, result, actual = _run(tmp_path, expression, {"x": x})
    np.testing.assert_allclose(actual, np.nansum(x, axis=1) + 2.0)
    assert runtime.plan.output_mode == "rows"
    assert result.output_shape == (8,)


def test_mixed_axis_mean_and_std_stream_without_materializing_time(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(37, 5))
    y = rng.normal(size=(37, 5))
    x[4, 1] = np.nan
    features = cat(var("x"), var("y"))

    _, mean_result, mean_value = _run(
        tmp_path, features.mean(axis=[0, 1]), {"x": x, "y": y}
    )
    expected = np.nanmean(np.stack((x, y), axis=-1), axis=(0, 1))
    np.testing.assert_allclose(mean_value, expected, rtol=1e-13, atol=1e-13)
    assert mean_result.output_shape == (2,)

    _, std_result, std_value = _run(
        tmp_path, features.std(axis=[0, 1], ddof=1), {"x": x, "y": y}
    )
    expected_std = np.nanstd(
        np.stack((x, y), axis=-1), axis=(0, 1), ddof=1
    )
    np.testing.assert_allclose(std_value, expected_std, rtol=1e-12, atol=1e-12)
    assert std_result.output_shape == (2,)


def test_reduction_axis_uses_full_materialized_rank_and_negative_axes(tmp_path: Path) -> None:
    x = np.arange(36, dtype=np.float64).reshape(6, 6)
    _, result, actual = _run(tmp_path, var("x").mean(axis=-1), {"x": x})
    np.testing.assert_allclose(actual, np.mean(x, axis=1))
    assert result.output_shape == (6,)


def test_emit_last_reuses_streaming_state_without_row_output(tmp_path: Path) -> None:
    x = np.arange(42, dtype=np.float64).reshape(7, 6)
    expression = cumsum(var("x")).emit("last")
    _, result, actual = _run(tmp_path, expression, {"x": x})
    np.testing.assert_allclose(actual, np.cumsum(x, axis=0)[-1])
    assert result.output_mode == "final"
    assert result.output_path.stat().st_size == 6 * 8


def test_row_then_temporal_reduction_composes(tmp_path: Path) -> None:
    x = np.arange(60, dtype=np.float64).reshape(10, 6)
    expression = (var("x") + 1.0).sum(axis=1).mean(axis=0)
    _, result, actual = _run(tmp_path, expression, {"x": x})
    np.testing.assert_allclose(actual, np.mean(np.sum(x + 1.0, axis=1)))
    assert result.output_shape == ()
    assert result.output_path.stat().st_size == 8


def test_temporal_reduction_and_emit_must_be_terminal() -> None:
    x = var("x")
    data = {"x": np.ones((4, 3), dtype=np.float64)}
    with pytest.raises(FormulaIRCompileError, match="terminal output"):
        compile_formula(x.sum(axis=0) + 1.0, data, n_instruments=3)
    with pytest.raises(FormulaIRCompileError, match="terminal output"):
        compile_formula(x.emit("last") + 1.0, data, n_instruments=3)


def test_string_formula_accepts_list_axes(tmp_path: Path) -> None:
    x = np.arange(24, dtype=np.float64).reshape(4, 6)
    runtime = compile_formula("sum(x, axis=[0, 1])", {"x": x}, n_instruments=6)
    result = runtime.run(out_path=tmp_path / "scalar.bin")
    actual = np.fromfile(result.output_path, dtype=np.float64)
    np.testing.assert_allclose(actual, [np.sum(x)])
    assert result.output_shape == ()
''')

# ---------------------------------------------------------------------------
# Benchmarks and documentation.
# ---------------------------------------------------------------------------
path("scripts/benchmark_cpp_stream_reductions.py").write_text(r'''from __future__ import annotations

import os
from pathlib import Path
from statistics import median
from time import perf_counter
import tempfile

import numpy as np

from trading_dsl_engine.base.dsl import cat, cumsum, var
from trading_dsl_engine.cpp_stream import compile_formula


ROWS = int(os.environ.get("CPP_STREAM_REDUCTION_ROWS", "1000000"))
N = int(os.environ.get("CPP_STREAM_REDUCTION_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_REDUCTION_RUNS", "7"))
WARMUPS = int(os.environ.get("CPP_STREAM_REDUCTION_WARMUPS", "1"))
OUTPUT_ROOT = Path(os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR", tempfile.gettempdir()))


def rates(runtime, output: Path):
    for _ in range(WARMUPS):
        runtime.run(out_path=output, async_writeback_mb=0)
    results = [runtime.run(out_path=output, async_writeback_mb=0) for _ in range(RUNS)]
    return results, [result.rows_per_second for result in results]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cpp_stream_reduction_") as temporary:
        root = Path(temporary)
        rng = np.random.default_rng(42)
        paths = {name: root / f"{name}.npy" for name in ("x", "y")}
        for name in paths:
            array = np.lib.format.open_memmap(
                paths[name], mode="w+", dtype=np.float64, shape=(ROWS, N)
            )
            for start in range(0, ROWS, 131072):
                stop = min(start + 131072, ROWS)
                array[start:stop] = rng.normal(size=(stop - start, N))
            array.flush()
            del array

        x = var("x")
        y = var("y")
        computation = cat(x * 1.01 + y, x - y * 0.1, x * y)
        full = compile_formula(computation, paths, n_instruments=N)
        reduced = compile_formula(computation.sum(axis=0), paths, n_instruments=N)
        mean_runtime = compile_formula(computation.mean(axis=0), paths, n_instruments=N)
        std_runtime = compile_formula(computation.std(axis=0), paths, n_instruments=N)
        emit_runtime = compile_formula(cumsum(x).emit("last"), paths, n_instruments=N)

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        full_path = OUTPUT_ROOT / "cpp_stream_reduction_full.bin"
        reduced_path = OUTPUT_ROOT / "cpp_stream_reduction_sum.bin"
        mean_path = OUTPUT_ROOT / "cpp_stream_reduction_mean.bin"
        std_path = OUTPUT_ROOT / "cpp_stream_reduction_std.bin"
        emit_path = OUTPUT_ROOT / "cpp_stream_reduction_emit.bin"

        full_results, full_rates = rates(full, full_path)
        reduced_results, reduced_rates = rates(reduced, reduced_path)
        _, mean_rates = rates(mean_runtime, mean_path)
        _, std_rates = rates(std_runtime, std_path)
        _, emit_rates = rates(emit_runtime, emit_path)

        started = perf_counter()
        materialized = np.memmap(
            full_path, mode="r", dtype=np.float64, shape=(ROWS, N, 3)
        )
        post_sum = np.nansum(materialized, axis=0)
        post_seconds = perf_counter() - started
        native_sum = np.fromfile(reduced_path, dtype=np.float64).reshape(N, 3)
        np.testing.assert_allclose(native_sum, post_sum, rtol=1e-11, atol=1e-8)

        full_median = median(full_rates)
        reduced_median = median(reduced_rates)
        if reduced_median <= full_median:
            raise RuntimeError(
                f"streaming reduction was not faster: {reduced_median=} {full_median=}"
            )

        full_bytes = full_path.stat().st_size
        reduced_bytes = reduced_path.stat().st_size
        full_seconds = median(result.seconds for result in full_results)
        reduced_seconds = median(result.seconds for result in reduced_results)

        print(f"rows={ROWS:,} instruments={N} features=3 warmups={WARMUPS} runs={RUNS}")
        print(f"full_median={full_median/1e6:.6f} M rows/s seconds={full_seconds:.6f} bytes={full_bytes}")
        print(f"sum_axis0_median={reduced_median/1e6:.6f} M rows/s seconds={reduced_seconds:.6f} bytes={reduced_bytes}")
        print(f"native_reduction_speedup={reduced_median/full_median:.3f}x")
        print(f"full_plus_numpy_reduction_seconds={full_seconds + post_seconds:.6f}")
        print(f"native_vs_full_plus_post_speedup={(full_seconds + post_seconds)/reduced_seconds:.3f}x")
        print(f"mean_axis0_median={median(mean_rates)/1e6:.6f} M rows/s")
        print(f"std_axis0_median={median(std_rates)/1e6:.6f} M rows/s")
        print(f"cumsum_emit_last_median={median(emit_rates)/1e6:.6f} M rows/s")
        print(f"checksum={float(np.nansum(native_sum)):.12g}")

        for output in (full_path, reduced_path, mean_path, std_path, emit_path):
            output.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
''')

path("src/trading_dsl_engine/cpp_stream/REDUCTIONS.md").write_text(r'''# Streaming reductions

Reductions use the same expression API as other operations:

```python
result = compile_formula((x * y).sum(axis=0), data).run(out_path="sum.bin")
row_mean = x.mean(axis=1)
feature_std = cat(x, y).std(axis=[0, 1], ddof=1)
last_cumulative = x.cumsum().emit("last")
```

Axes are interpreted against the logical materialized result `(time, *row_shape)`.
Axis `0` is therefore time. A reduction containing axis `0` is evaluated online and
emits one final output; it never creates the intermediate time-sized result. Axes
that do not contain `0` reduce the current row and continue to emit one result per
input row, so they compose normally with subsequent operations.

Temporal reductions and `emit("last")` are terminal because they remove the streaming
time dimension. Row reductions can appear anywhere in the graph. `sum`, `mean`, and
`std` ignore non-finite observations; empty groups and standard deviations with
`count <= ddof` produce NaN. Standard deviation uses an online Welford accumulator.

`RunResult.rows` remains the number of input rows processed, preserving throughput
reporting. `RunResult.output_rows`, `output_shape`, and `output_mode` describe the
materialized output. A temporal reduction has `output_rows == 1` and stores only one
fixed-size result in the output file.
''')

# Include permanent tests and benchmark in backend CI.
workflow = path(".github/workflows/cpp-stream-backend.yml")
workflow_text = workflow.read_text()
workflow_text = workflow_text.replace(
    "          tests/trading_dsl_engine/cpp_stream/test_einsum.py\n"
    "          tests/trading_dsl_engine/cpp_stream/test_roll_rets.py\n",
    "          tests/trading_dsl_engine/cpp_stream/test_einsum.py\n"
    "          tests/trading_dsl_engine/cpp_stream/test_reductions.py\n"
    "          tests/trading_dsl_engine/cpp_stream/test_roll_rets.py\n",
)
workflow_text += '''      - name: Smoke benchmark streaming reductions
        env:
          PYTHONPATH: src
          CPP_STREAM_REDUCTION_ROWS: "500000"
          CPP_STREAM_REDUCTION_INSTRUMENTS: "9"
          CPP_STREAM_REDUCTION_RUNS: "3"
          CPP_STREAM_REDUCTION_WARMUPS: "1"
          CPP_STREAM_BENCH_OUTPUT_DIR: /dev/shm
        run: python scripts/benchmark_cpp_stream_reductions.py
'''
workflow.write_text(workflow_text)

# Document the invariant in AGENTS.md.
agents = path("src/trading_dsl_engine/cpp_stream/AGENTS.md")
agents_text = agents.read_text()
agents_text += '''

## Streaming reductions

- Reduction axes refer to `(time, *row_shape)`; axis 0 is temporal.
- A temporal reduction or `emit("last")` must not allocate or write a time-sized output.
- Row reductions remain ordinary composable stages; temporal reductions are terminal.
- Use fixed-size accumulators only. `std` uses Welford state and no hot-path allocation.
- Benchmarks must compare the fused native reduction with full materialization and
  post-hoc reduction, validate output checksums, and report output byte counts.
'''
agents.write_text(agents_text)

print("applied cpp_stream reductions patch")
