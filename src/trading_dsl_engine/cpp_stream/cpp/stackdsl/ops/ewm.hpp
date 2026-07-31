#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "stackdsl/utils.hpp"

namespace stackdsl {

template <std::size_t N, class In, class Out, std::uint64_t SpanBits, int MinPeriods, bool IgnoreNa, bool Adjust>
struct EwmNode {
    static constexpr double span = std::bit_cast<double>(SpanBits);
    static_assert(span > 0.0);
    static constexpr double alpha = 2.0 / (span + 1.0);
    static constexpr double old_weight_factor = 1.0 - alpha;

    alignas(64) std::array<double, N> value{};
    alignas(64) std::array<double, N> weight{};
    alignas(64) std::array<std::int64_t, N> count{};
    alignas(64) std::array<std::uint8_t, N> initialized{};

    void setup() noexcept {
        value.fill(0.0);
        weight.fill(0.0);
        count.fill(0);
        initialized.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        for (std::size_t i = 0; i < N; ++i) {
            const double x = ctx.template read<In>(i);
            const bool observation = finite(x);
            double old_weight = weight[i];
            if (initialized[i] && (observation || !IgnoreNa)) old_weight *= old_weight_factor;
            if (observation) {
                if (initialized[i]) {
                    double new_weight = Adjust ? 1.0 : alpha;
                    if constexpr (!Adjust) {
                        if (std::abs(alpha - 0.5) <= 1e-12) new_weight = 1.0 - old_weight;
                    }
                    if (value[i] != x) value[i] = (old_weight * value[i] + new_weight * x) / (old_weight + new_weight);
                    old_weight = Adjust ? old_weight + new_weight : 1.0;
                } else {
                    value[i] = x;
                    initialized[i] = 1;
                    old_weight = 1.0;
                }
                ++count[i];
            }
            weight[i] = old_weight;
            const bool enough = MinPeriods <= 0 || count[i] >= MinPeriods;
            out[i] = initialized[i] && enough ? value[i] : kNaN;
        }
    }
};

}  // namespace stackdsl
