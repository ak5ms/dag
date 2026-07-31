#pragma once

#include <array>
#include <cstddef>

#include "stackdsl/utils.hpp"

namespace stackdsl {

template <std::size_t N, class In, class Out>
struct CumsumNode {
    alignas(64) std::array<double, N> value{};

    void setup() noexcept { value.fill(0.0); }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        for (std::size_t i = 0; i < N; ++i) {
            const double x = ctx.template read<In>(i);
            if (finite(x)) {
                value[i] += x;
                out[i] = value[i];
            } else {
                out[i] = kNaN;
            }
        }
    }
};

}  // namespace stackdsl
