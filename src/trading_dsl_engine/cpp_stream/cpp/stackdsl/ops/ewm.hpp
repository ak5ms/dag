#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

#include "stackdsl/engine.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template <
    std::size_t StateSize,
    std::uint64_t SpanBits,
    int MinPeriods,
    bool IgnoreNa,
    bool Adjust
>
struct EwmState {
    static constexpr double span = std::bit_cast<double>(SpanBits);
    static_assert(span > 0.0);
    static constexpr double alpha = 2.0 / (span + 1.0);
    static constexpr double old_weight_factor = 1.0 - alpha;

    alignas(64) std::array<double, StateSize> value{};
    alignas(64) std::array<double, StateSize> weight{};
    alignas(64) std::array<std::int64_t, StateSize> count{};
    alignas(64) std::array<std::uint8_t, StateSize> initialized{};
    bool all_initialized = false;

    void setup() noexcept {
        value.fill(0.0);
        weight.fill(0.0);
        count.fill(0);
        initialized.fill(0);
        all_initialized = false;
    }

    STACKDSL_HOT double recursive_update(
        std::size_t index, double x
    ) noexcept {
        if (finite(x)) {
            if (initialized[index]) {
                value[index] = std::fma(alpha, x - value[index], value[index]);
            } else {
                value[index] = x;
                initialized[index] = 1;
            }
        }
        return initialized[index] ? value[index] : kNaN;
    }

    STACKDSL_HOT double general_update(
        std::size_t index, double x
    ) noexcept {
        const bool observation = finite(x);
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
                    value[index] = (
                        old_weight * value[index] + new_weight * x
                    ) / (old_weight + new_weight);
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
        return initialized[index] && enough ? value[index] : kNaN;
    }

    STACKDSL_HOT double update(std::size_t index, double x) noexcept {
        if constexpr (MinPeriods <= 0 && IgnoreNa && !Adjust) {
            return recursive_update(index, x);
        } else {
            return general_update(index, x);
        }
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    std::uint64_t SpanBits,
    int MinPeriods,
    bool IgnoreNa,
    bool Adjust,
    class Execution = DirectExecution<N>
>
struct EwmNode {
    using State = EwmState<
        Execution::state_size, SpanBits, MinPeriods, IgnoreNa, Adjust
    >;
    static constexpr double alpha = State::alpha;
    State state{};

    void setup() noexcept { state.setup(); }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();

        if constexpr (MinPeriods <= 0 && IgnoreNa && !Adjust) {
            if constexpr (Execution::contiguous_lanes) {
                run_recursive_contiguous(ctx, out);
            } else {
                run_recursive_indexed(ctx, out);
            }
            return;
        }

        run_general(ctx, out);
    }

private:
    template <class Context>
    STACKDSL_HOT void run_recursive_contiguous(Context& ctx, double* STACKDSL_RESTRICT out) noexcept {
        std::array<double, N> input{};
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        bool all_finite = true;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = begin; lane < end; ++lane) {
            input[lane] = ctx.template read<In>(lane);
            all_finite = all_finite && finite(input[lane]);
        }
        if (state.all_initialized && all_finite) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
            for (std::size_t lane = begin; lane < end; ++lane) {
                const double next = std::fma(
                    alpha,
                    input[lane] - state.value[lane],
                    state.value[lane]
                );
                state.value[lane] = next;
                out[lane] = next;
            }
            return;
        }

        bool now_all_initialized = true;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = begin; lane < end; ++lane) {
            out[lane] = state.recursive_update(lane, input[lane]);
            now_all_initialized = now_all_initialized && state.initialized[lane];
        }
        state.all_initialized = now_all_initialized;
    }

    template <class Context>
    STACKDSL_HOT void run_recursive_indexed(Context& ctx, double* STACKDSL_RESTRICT out) noexcept {
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            out[lane] = state.recursive_update(
                index, ctx.template read<In>(lane)
            );
        }
    }

    template <class Context>
    STACKDSL_HOT void run_general(Context& ctx, double* STACKDSL_RESTRICT out) noexcept {
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            out[lane] = state.general_update(
                index, ctx.template read<In>(lane)
            );
        }
    }
};

template <
    std::size_t N,
    class Inputs,
    class Outputs,
    std::uint64_t SpanBits,
    int MinPeriods,
    bool IgnoreNa,
    bool Adjust,
    class Execution = DirectExecution<N>
>
struct EwmBundleNode;

template <
    std::size_t N,
    std::uint64_t SpanBits,
    int MinPeriods,
    bool IgnoreNa,
    bool Adjust,
    class Execution,
    class... Inputs,
    class... Outputs
>
struct EwmBundleNode<
    N,
    TypeList<Inputs...>,
    TypeList<Outputs...>,
    SpanBits,
    MinPeriods,
    IgnoreNa,
    Adjust,
    Execution
> {
    static_assert(sizeof...(Inputs) == sizeof...(Outputs));
    static_assert(sizeof...(Inputs) > 1);
    using Kernel = EwmState<
        Execution::state_size, SpanBits, MinPeriods, IgnoreNa, Adjust
    >;
    static constexpr std::size_t BundleSize = sizeof...(Inputs);
    std::array<Kernel, sizeof...(Inputs)> states{};

    void setup() noexcept {
        for (auto& state : states) state.setup();
    }

    template <class Context, std::size_t... Indexes>
    STACKDSL_HOT void load_lane(
        Context& ctx,
        std::size_t lane,
        std::array<std::array<double, N>, BundleSize>& values,
        bool& all_finite,
        std::index_sequence<Indexes...>
    ) noexcept {
        ((values[Indexes][lane] = ctx.template read<
              std::tuple_element_t<Indexes, std::tuple<Inputs...>>
          >(lane),
          all_finite = all_finite && finite(values[Indexes][lane])), ...);
    }

    template <std::size_t Index, class Context>
    STACKDSL_HOT void run_recursive_fast(
        Context& ctx,
        const std::array<std::array<double, N>, BundleSize>& values,
        std::size_t begin,
        std::size_t end
    ) noexcept {
        auto& state = states[Index];
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<
            std::tuple_element_t<Index, std::tuple<Outputs...>>
        >();
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = begin; lane < end; ++lane) {
            const double next = std::fma(
                Kernel::alpha,
                values[Index][lane] - state.value[lane],
                state.value[lane]
            );
            state.value[lane] = next;
            out[lane] = next;
        }
    }

    template <std::size_t Index, class Context>
    STACKDSL_HOT void run_recursive_checked(
        Context& ctx,
        const std::array<std::array<double, N>, BundleSize>& values,
        std::size_t begin,
        std::size_t end
    ) noexcept {
        auto& state = states[Index];
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<
            std::tuple_element_t<Index, std::tuple<Outputs...>>
        >();
        bool now_all_initialized = true;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t state_index =
                Execution::state_index(ctx, lane);
            out[lane] = state.recursive_update(
                state_index, values[Index][lane]
            );
            now_all_initialized = now_all_initialized
                && state.initialized[state_index];
        }
        state.all_initialized = now_all_initialized;
    }

    template <std::size_t Index, class Context>
    STACKDSL_HOT void run_general(
        Context& ctx,
        const std::array<std::array<double, N>, BundleSize>& values,
        std::size_t begin,
        std::size_t end
    ) noexcept {
        auto& state = states[Index];
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<
            std::tuple_element_t<Index, std::tuple<Outputs...>>
        >();
        for (std::size_t lane = begin; lane < end; ++lane) {
            out[lane] = state.general_update(
                Execution::state_index(ctx, lane), values[Index][lane]
            );
        }
    }

    template <class Context, std::size_t... Indexes>
    STACKDSL_HOT void run_recursive_fast_all(
        Context& ctx,
        const std::array<std::array<double, N>, BundleSize>& values,
        std::size_t begin,
        std::size_t end,
        std::index_sequence<Indexes...>
    ) noexcept {
        (run_recursive_fast<Indexes>(ctx, values, begin, end), ...);
    }

    template <class Context, std::size_t... Indexes>
    STACKDSL_HOT void run_recursive_checked_all(
        Context& ctx,
        const std::array<std::array<double, N>, BundleSize>& values,
        std::size_t begin,
        std::size_t end,
        std::index_sequence<Indexes...>
    ) noexcept {
        (run_recursive_checked<Indexes>(ctx, values, begin, end), ...);
    }

    template <class Context, std::size_t... Indexes>
    STACKDSL_HOT void run_general_all(
        Context& ctx,
        const std::array<std::array<double, N>, BundleSize>& values,
        std::size_t begin,
        std::size_t end,
        std::index_sequence<Indexes...>
    ) noexcept {
        (run_general<Indexes>(ctx, values, begin, end), ...);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        std::array<std::array<double, N>, BundleSize> values{};
        bool all_finite = true;
        for (std::size_t lane = begin; lane < end; ++lane) {
            load_lane(
                ctx, lane, values, all_finite,
                std::index_sequence_for<Inputs...>{}
            );
        }
        if constexpr (MinPeriods <= 0 && IgnoreNa && !Adjust) {
            const bool all_initialized = [&]<std::size_t... Indexes>(
                std::index_sequence<Indexes...>
            ) {
                return (states[Indexes].all_initialized && ...);
            }(std::index_sequence_for<Inputs...>{});
            if constexpr (Execution::contiguous_lanes) {
                if (all_initialized && all_finite) {
                    run_recursive_fast_all(
                        ctx, values, begin, end,
                        std::index_sequence_for<Inputs...>{}
                    );
                    return;
                }
            }
            run_recursive_checked_all(
                ctx, values, begin, end,
                std::index_sequence_for<Inputs...>{}
            );
        } else {
            run_general_all(
                ctx, values, begin, end,
                std::index_sequence_for<Inputs...>{}
            );
        }
    }
};

}  // namespace stackdsl
