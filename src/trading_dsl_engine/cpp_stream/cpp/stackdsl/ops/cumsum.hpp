#pragma once

#include <array>
#include <cstddef>

#include "stackdsl/engine.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template <
    std::size_t N,
    class In,
    class Out,
    class Execution = DirectExecution<N>
>
struct CumsumNode {
    alignas(64) std::array<double, Execution::state_size> value{};

    void setup() noexcept { value.fill(0.0); }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = begin; lane < end; ++lane) {
            const double x = ctx.template read<In>(lane);
            const std::size_t index = Execution::state_index(ctx, lane);
            if (finite(x)) {
                value[index] += x;
                out[lane] = value[index];
            } else {
                out[lane] = kNaN;
            }
        }
    }
};

}  // namespace stackdsl
