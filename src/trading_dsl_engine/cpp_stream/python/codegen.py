from __future__ import annotations

from dataclasses import dataclass

from trading_dsl_engine.cpp_stream.python.lowering import GroupStage, Plan, Stage, op_cpp_type


@dataclass(frozen=True, slots=True)
class GeneratedSource:
    text: str


def _source_list(sources) -> str:
    return "stackdsl::SourceList<" + ", ".join(source.cpp() for source in sources) + ">"


def _inner_struct(name: str, group: GroupStage) -> str:
    decls: list[str] = []
    setup: list[str] = []
    calls: list[str] = []
    for i, stage in enumerate(group.inner.stages):
        if stage.kind == "groupby":
            raise ValueError("nested groupby is not supported")
        stage_type = op_cpp_type(stage, "N", grouped_capacity_expr="Capacity")
        decls.append(f"    {stage_type} s{i}{{}};")
        setup.append(f"        s{i}.setup();")
        calls.append(f"        s{i}.on_data(ctx);")
    return f'''template <std::size_t N, std::size_t Capacity>\nstruct {name} {{\n    stackdsl::GroupRowContext<N, {group.inner.input_count}, {group.inner.scratch_slots}> ctx{{}};\n{chr(10).join(decls)}\n\n    void setup() noexcept {{\n{chr(10).join(setup)}\n    }}\n\n    void on_data(\n        const std::array<const double*, {group.inner.input_count}>& feeds,\n        const std::array<std::uint16_t, N>& slots,\n        const std::array<std::uint16_t, N>& partitions,\n        double* out) noexcept {{\n        ctx.inputs = feeds;\n        ctx.group_slots = &slots;\n        ctx.partitions = &partitions;\n        ctx.output = out;\n{chr(10).join(calls)}\n    }}\n}};\n'''


def _group_type(group_name: str, group: GroupStage, stage: Stage, n: int) -> str:
    if not group.key_sources:
        resolver = f"stackdsl::NoKeyResolver<{n}>"
    elif group.dense_cardinality is not None:
        resolver = f"stackdsl::DenseGroupResolver<{n}, {group.dense_cardinality}, {group.dense_offset}>"
    else:
        resolver = f"stackdsl::HashGroupResolver<{n}, {len(group.key_sources)}, {group.capacity}, {group.hash_capacity}>"
    partition_args = ", ".join(str(value) for value in group.partitions)
    partitions = f"stackdsl::StaticPartitions<{n}, {partition_args}>"
    inner = f"{group_name}<{n}, {resolver}::capacity>"
    return (
        f"stackdsl::GroupByNode<{n}, {resolver}, {partitions}, {inner}, {stage.out.cpp()}, "
        f"{_source_list(group.key_sources)}, {_source_list(group.feed_sources)}>"
    )


def render_translation_unit(plan: Plan, *, n_instruments: int, prefetch_rows: int) -> GeneratedSource:
    group_names: dict[int, str] = {}
    inner_defs: list[str] = []
    for i, stage in enumerate(plan.stages):
        if stage.kind != "groupby":
            continue
        assert stage.group is not None
        name = f"CppStreamInner{i}"
        group_names[i] = name
        inner_defs.append(_inner_struct(name, stage.group))

    stage_decls: list[str] = []
    setup_calls: list[str] = []
    row_calls: list[str] = []
    for i, stage in enumerate(plan.stages):
        if stage.kind == "groupby":
            assert stage.group is not None
            stage_type = _group_type(group_names[i], stage.group, stage, n_instruments)
            stage_decls.append(f"        {stage_type} s{i}{{}};")
            setup_calls.append(f"        s{i}.setup();")
            row_calls.append(f"                if (!s{i}.on_data_checked(ctx)) return 4;")
        else:
            stage_type = op_cpp_type(stage, str(n_instruments))
            stage_decls.append(f"        {stage_type} s{i}{{}};")
            setup_calls.append(f"        s{i}.setup();")
            row_calls.append(f"                s{i}.on_data(ctx);")

    input_decls: list[str] = []
    input_validate: list[str] = []
    input_bases: list[str] = []
    row_bind: list[str] = []
    prefetch: list[str] = []
    for i in range(plan.input_count):
        input_decls.append(f"        stackdsl::MMapFile in{i}(input_paths[{i}], false);")
        input_decls.append(f"        in{i}.advise_sequential();")
        input_validate.append(f"        if (in{i}.size() % row_bytes != 0) return 2;")
        input_validate.append(f"        const std::size_t rows{i} = in{i}.size() / row_bytes;")
        if i > 0:
            input_validate.append(f"        if (rows{i} != rows0) return 3;")
        input_bases.append(f"        const auto* STACKDSL_RESTRICT base{i} = static_cast<const double*>(in{i}.data());")
        row_bind.append(f"                ctx.inputs[{i}] = base{i} + t * N;")
        if prefetch_rows > 0:
            prefetch.append(f"                if (t + {prefetch_rows} < rows) __builtin_prefetch(base{i} + (t + {prefetch_rows}) * N, 0, 1);")
    if plan.input_count == 0:
        raise ValueError("cpp_stream requires at least one file-backed input")

    source = f'''#include <array>\n#include <chrono>\n#include <cstddef>\n#include <cstdint>\n#include <exception>\n#include <string>\n\n#include "stackdsl/runtime.hpp"\n\nnamespace {{\nthread_local std::string g_last_error;\n\n{chr(10).join(inner_defs)}\n}}  // namespace\n\nextern "C" const char* cpp_stream_last_error() noexcept {{\n    return g_last_error.c_str();\n}}\n\nextern "C" int cpp_stream_run_files(\n    const char* const* input_paths,\n    std::size_t input_count,\n    const char* output_path,\n    std::size_t async_writeback_bytes,\n    std::size_t* rows_out,\n    double* seconds_out) noexcept {{\n    try {{\n        g_last_error.clear();\n        constexpr std::size_t N = {n_instruments};\n        constexpr std::size_t row_bytes = N * sizeof(double);\n        if (input_count != {plan.input_count} || input_paths == nullptr || output_path == nullptr) return 1;\n{chr(10).join(input_decls)}\n{chr(10).join(input_validate)}\n        const std::size_t rows = rows0;\n        stackdsl::MMapFile output(output_path, true, rows * row_bytes);\n        output.advise_sequential();\n        auto* STACKDSL_RESTRICT out = static_cast<double*>(output.data());\n{chr(10).join(input_bases)}\n\n        stackdsl::RowContext<N, {plan.input_count}, {plan.scratch_slots}> ctx{{}};\n{chr(10).join(stage_decls)}\n{chr(10).join(setup_calls)}\n\n        std::size_t writeback_start = 0;\n        const auto started = std::chrono::steady_clock::now();\n        for (std::size_t t = 0; t < rows; ++t) {{\n{chr(10).join(prefetch)}\n{chr(10).join(row_bind)}\n                ctx.output = out + t * N;\n{chr(10).join(row_calls)}\n                if (async_writeback_bytes > 0) {{\n                    const std::size_t completed = (t + 1) * row_bytes;\n                    if (completed - writeback_start >= async_writeback_bytes) {{\n                        output.request_writeback(writeback_start, completed - writeback_start);\n                        writeback_start = completed;\n                    }}\n                }}\n        }}\n        const auto ended = std::chrono::steady_clock::now();\n        if (rows_out) *rows_out = rows;\n        if (seconds_out) *seconds_out = std::chrono::duration<double>(ended - started).count();\n        return 0;\n    }} catch (const std::exception& exc) {{\n        g_last_error = exc.what();\n        return 100;\n    }} catch (...) {{\n        g_last_error = "unknown cpp_stream exception";\n        return 101;\n    }}\n}}\n'''
    return GeneratedSource(source)
