#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "stackdsl/engine.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

// A pandas-compatible EWM state machine whose span is read from another DSL
// expression on every row/lane.  The fixed-span EwmNode remains unchanged and
// retains its compile-time alpha and optimized contiguous fast path.
template <
    std::size_t N,
    class In,
    class SpanIn,
    class Out,
    int MinPeriods,
    bool IgnoreNa,
    bool Adjust,
    class Execution = DirectExecution<N>
>
struct DynamicEwmNode {
    static constexpr std::size_t state_size = Execution::state_size;

    alignas(64) std::array<double, state_size> value{};
    alignas(64) std::array<double, state_size> weight{};
    alignas(64) std::array<std::int64_t, state_size> count{};
    alignas(64) std::array<std::uint8_t, state_size> initialized{};

    void setup() noexcept {
        value.fill(0.0);
        weight.fill(0.0);
        count.fill(0);
        initialized.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const double span = ctx.template read<SpanIn>(lane);
            if (!finite(span) || span <= 0.0) {
                // A bad runtime parameter does not corrupt or advance the state.
                out[lane] = kNaN;
                continue;
            }

            const double alpha = 2.0 / (span + 1.0);
            const double old_weight_factor = 1.0 - alpha;
            const double x = ctx.template read<In>(lane);
            const bool observation = finite(x);

            if constexpr (MinPeriods <= 0 && IgnoreNa && !Adjust) {
                if (observation) {
                    if (initialized[index]) {
                        value[index] = std::fma(
                            alpha,
                            x - value[index],
                            value[index]
                        );
                    } else {
                        value[index] = x;
                        initialized[index] = 1;
                    }
                }
                out[lane] = initialized[index] ? value[index] : kNaN;
                continue;
            }

            double old_weight = weight[index];
            if (initialized[index] && (observation || !IgnoreNa)) {
                old_weight *= old_weight_factor;
            }
            if (observation) {
                if (initialized[index]) {
                    double new_weight = Adjust ? 1.0 : alpha;
                    if constexpr (!Adjust) {
                        if (std::abs(alpha - 0.5) <= 1e-12) {
                            new_weight = 1.0 - old_weight;
                        }
                    }
                    if (value[index] != x) {
                        value[index] =
                            (old_weight * value[index] + new_weight * x)
                            / (old_weight + new_weight);
                    }
                    old_weight = Adjust ? old_weight + new_weight : 1.0;
                } else {
                    value[index] = x;
                    initialized[index] = 1;
                    old_weight = 1.0;
                }
                ++count[index];
            }
            weight[index] = old_weight;
            const bool enough = MinPeriods <= 0 || count[index] >= MinPeriods;
            out[lane] = initialized[index] && enough ? value[index] : kNaN;
        }
    }
};

}  // namespace stackdsl
