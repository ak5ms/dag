from pathlib import Path


path = (
    Path(__file__).resolve().parents[1]
    / "src/trading_dsl_engine/cpp_stream/cpp/stackdsl/ops/clarabel_program.hpp"
)
text = path.read_text()
old = """        if (Guard::read_flat(ctx, 0) == 0.0) {
            (project_nan<Projections>(ctx), ...);
            return;
        }
"""
new = """        if constexpr (!std::is_same_v<Guard, ClarabelAlwaysEnabled>) {
            if (ctx.template read<Guard>(0) == 0.0) {
                (project_nan<Projections>(ctx), ...);
                return;
            }
        }
"""
if new not in text:
    if text.count(old) != 1:
        raise RuntimeError("missing generated optimizer guard-read anchor")
    path.write_text(text.replace(old, new, 1))
