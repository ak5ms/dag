from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one patch anchor, found {count}: {old[:120]!r}"
        )
    path.write_text(text.replace(old, new, 1))


def patch_frontend() -> None:
    path = ROOT / "src/trading_dsl_engine/ir/frontend.py"
    replace_once(
        path,
        """    def _build_call(self, node: Call) -> int:\n        if node.fn in _NARY_ARITY:\n""",
        """    def _build_call(self, node: Call) -> int:\n        # A scalar `where(open, optimizer_field, NaN)` is a control-flow\n        # guard for the generated optimizer, not an eager elementwise mask.\n        # Encode the guard as a second projection child so cpp_stream can\n        # skip parameter loading, solving, and feedback-state advancement.\n        if (\n            node.fn == \"where\"\n            and not node.kwargs\n            and len(node.args) == 3\n            and isinstance(node.args[1], CvxpyFieldExpr)\n            and isinstance(node.args[2], Number)\n            and isinstance(node.args[2].value, float)\n            and math.isnan(node.args[2].value)\n        ):\n            condition = self.build(node.args[0])\n            condition_type = self.nodes[condition].value_type\n            try:\n                condition_shape = condition_type.logical_shape\n            except ValueError:\n                condition_shape = None\n            if condition_shape == ():\n                projection = node.args[1]\n                child = self.build(projection.program_expr)\n                child_op = self.nodes[child].op\n                if not isinstance(child_op, CvxpyProgramOp):\n                    raise FormulaIRCompileError(\n                        \"optimizer field projection lost its generated \"\n                        \"program object\"\n                    )\n                field = child_op.program.resolve_field(projection.field)\n                return self._append(\n                    CvxpyProjectionOp(field),\n                    (child, condition),\n                    _generated_field_value_type(\n                        child_op.program, field.logical_shape\n                    ),\n                )\n        if node.fn in _NARY_ARITY:\n""",
    )


def patch_lowering() -> None:
    path = ROOT / "src/trading_dsl_engine/cpp_stream/python/lowering_full.py"
    replace_once(
        path,
        """    clarabel_stage_by_object: dict[Source, int] = {}\n""",
        """    clarabel_stage_by_object: dict[tuple[Source, Source | None], int] = {}\n""",
    )
    replace_once(
        path,
        """        if isinstance(op, CvxpyProjectionOp):\n            object_source = children[0]\n            if object_source.kind != \"clarabel\" or not isinstance(\n                object_source.op, CvxpyProgramOp\n            ):\n                raise CppStreamLoweringError(\n                    \"optimizer projection lost its generated program object\"\n                )\n            field = op.field\n            out = value_dest(is_root, node_shape)\n            member = Stage(\n                \"clarabel\",\n                object_source.parts,\n                out,\n                1,\n                output_kind=node.value_type.kind,\n                output_width=int(node.value_type.width),\n                op=object_source.op,\n                projection=field.name,\n                final_only=final_only,\n            )\n            previous_index = clarabel_stage_by_object.get(object_source)\n            if previous_index is None:\n                clarabel_stage_by_object[object_source] = len(stages)\n                stages.append(member)\n            else:\n                previous = stages[previous_index]\n                members = (\n                    previous.members\n                    if previous.kind == \"clarabel_bundle\"\n                    else (previous,)\n                )\n                stages[previous_index] = replace(\n                    previous,\n                    kind=\"clarabel_bundle\",\n                    members=(*members, member),\n                )\n            sources[node_id] = source_from_dest(\n                out, node_shape, final_only=final_only\n            )\n            continue\n""",
        """        if isinstance(op, CvxpyProjectionOp):\n            if len(children) not in {1, 2}:\n                raise CppStreamLoweringError(\n                    \"optimizer projection expects its object and optional \"\n                    \"scalar guard\"\n                )\n            object_source = children[0]\n            guard_source = children[1] if len(children) == 2 else None\n            if object_source.kind != \"clarabel\" or not isinstance(\n                object_source.op, CvxpyProgramOp\n            ):\n                raise CppStreamLoweringError(\n                    \"optimizer projection lost its generated program object\"\n                )\n            if guard_source is not None and guard_source.shape != ():\n                raise CppStreamLoweringError(\n                    \"optimizer where guard must be scalar for the whole row\"\n                )\n            field = op.field\n            out = value_dest(is_root, node_shape)\n            stage_inputs = object_source.parts + (\n                () if guard_source is None else (guard_source,)\n            )\n            member = Stage(\n                \"clarabel\",\n                stage_inputs,\n                out,\n                1,\n                output_kind=node.value_type.kind,\n                output_width=int(node.value_type.width),\n                op=object_source.op,\n                projection=field.name,\n                final_only=final_only,\n            )\n            bundle_key = (object_source, guard_source)\n            previous_index = clarabel_stage_by_object.get(bundle_key)\n            if previous_index is None:\n                clarabel_stage_by_object[bundle_key] = len(stages)\n                stages.append(member)\n            else:\n                previous = stages[previous_index]\n                members = (\n                    previous.members\n                    if previous.kind == \"clarabel_bundle\"\n                    else (previous,)\n                )\n                stages[previous_index] = replace(\n                    previous,\n                    kind=\"clarabel_bundle\",\n                    members=(*members, member),\n                )\n            sources[node_id] = source_from_dest(\n                out, node_shape, final_only=final_only\n            )\n            continue\n""",
    )


def patch_codegen() -> None:
    path = ROOT / "src/trading_dsl_engine/cpp_stream/python/codegen.py"
    replace_once(
        path,
        """        program = physical.op.program\n        if len(physical.inputs) != len(program.parameters):\n            raise ValueError(\"generated optimizer parameter/source count mismatch\")\n        feedback_fields = physical.op.feedback_fields or (\n            (None,) * len(physical.inputs)\n        )\n        if len(feedback_fields) != len(physical.inputs):\n            raise ValueError(\"generated optimizer feedback/source count mismatch\")\n        bindings = []\n        for index, (source, feedback) in enumerate(\n            zip(physical.inputs, feedback_fields)\n        ):\n""",
        """        program = physical.op.program\n        parameter_count = len(program.parameters)\n        if len(physical.inputs) not in {parameter_count, parameter_count + 1}:\n            raise ValueError(\n                \"generated optimizer expects parameter sources and an \"\n                \"optional scalar guard\"\n            )\n        parameter_sources = physical.inputs[:parameter_count]\n        guard_source = (\n            physical.inputs[-1]\n            if len(physical.inputs) == parameter_count + 1\n            else None\n        )\n        feedback_fields = physical.op.feedback_fields or (\n            (None,) * parameter_count\n        )\n        if len(feedback_fields) != parameter_count:\n            raise ValueError(\"generated optimizer feedback/source count mismatch\")\n        bindings = []\n        for index, (source, feedback) in enumerate(\n            zip(parameter_sources, feedback_fields)\n        ):\n""",
    )
    replace_once(
        path,
        """        return tmpl(\n            \"stackdsl::ClarabelNode\",\n            Name(program.class_name),\n            tmpl(\"stackdsl::ClarabelParameterList\", *bindings),\n            tmpl(\"stackdsl::ClarabelProjectionList\", *projections),\n        )\n""",
        """        guard_type = (\n            Name(\"stackdsl::ClarabelAlwaysEnabled\")\n            if guard_source is None\n            else _source_type(guard_source, n=n, input_types=input_types)\n        )\n        return tmpl(\n            \"stackdsl::ClarabelNode\",\n            Name(program.class_name),\n            tmpl(\"stackdsl::ClarabelParameterList\", *bindings),\n            tmpl(\"stackdsl::ClarabelProjectionList\", *projections),\n            guard_type,\n        )\n""",
    )


def patch_cpp_node() -> None:
    path = ROOT / (
        "src/trading_dsl_engine/cpp_stream/cpp/stackdsl/ops/clarabel_program.hpp"
    )
    replace_once(
        path,
        """#include <cstdint>\n""",
        """#include <cstdint>\n#include <limits>\n""",
    )
    replace_once(
        path,
        """template <class... Projections>\nstruct ClarabelProjectionList {};\n\ntemplate <class Program, class Parameters, class Projections>\nclass ClarabelNode;\n\ntemplate <class Program, class... Bindings, class... Projections>\nclass ClarabelNode<\n    Program,\n    ClarabelParameterList<Bindings...>,\n    ClarabelProjectionList<Projections...>\n> {\n""",
        """template <class... Projections>\nstruct ClarabelProjectionList {};\n\nstruct ClarabelAlwaysEnabled {\n    template <class Context>\n    STACKDSL_HOT static double read_flat(\n        const Context&, std::size_t\n    ) noexcept {\n        return 1.0;\n    }\n};\n\ntemplate <\n    class Program,\n    class Parameters,\n    class Projections,\n    class Guard = ClarabelAlwaysEnabled\n>\nclass ClarabelNode;\n\ntemplate <\n    class Program,\n    class... Bindings,\n    class... Projections,\n    class Guard\n>\nclass ClarabelNode<\n    Program,\n    ClarabelParameterList<Bindings...>,\n    ClarabelProjectionList<Projections...>,\n    Guard\n> {\n""",
    )
    replace_once(
        path,
        """    template <class Projection, class Context>\n    STACKDSL_HOT void project(Context& ctx) noexcept {\n""",
        """    template <class Projection, class Context>\n    STACKDSL_HOT void project_nan(Context& ctx) noexcept {\n        auto* STACKDSL_RESTRICT out =\n            ctx.template write_ptr<typename Projection::output_type>();\n        for (std::size_t index = 0; index < Projection::count; ++index) {\n            out[index] = std::numeric_limits<double>::quiet_NaN();\n        }\n    }\n\n    template <class Projection, class Context>\n    STACKDSL_HOT void project(Context& ctx) noexcept {\n""",
    )
    replace_once(
        path,
        """    template <class Context>\n    STACKDSL_HOT void on_data(Context& ctx) {\n        bool changed = false;\n""",
        """    template <class Context>\n    STACKDSL_HOT void on_data(Context& ctx) {\n        if (Guard::read_flat(ctx, 0) == 0.0) {\n            (project_nan<Projections>(ctx), ...);\n            return;\n        }\n        bool changed = false;\n""",
    )


if __name__ == "__main__":
    patch_frontend()
    patch_lowering()
    patch_codegen()
    patch_cpp_node()
