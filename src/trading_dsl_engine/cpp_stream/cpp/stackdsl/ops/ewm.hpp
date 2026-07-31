#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "stackdsl/utils.hpp"

namespace stackdsl {

template <std::size_t N>
struct DirectStateIndex {
    static constexpr std::size_t state_size = N;
    static constexpr bool contiguous_lanes = true;

    template <class Context>
    STACKDSL_HOT static std::size_t get(const Context&, std::size_t lane) noexcept {
        return lane;
    }
};

template <std::size_t N, std::size_t Capacity>
struct GroupedStateIndex {
    static constexpr std::size_t state_size = N * Capacity;
    static constexpr bool contiguous_lanes = false;

    template <class Context>
    STACKDSL_HOT static std::size_t get(const Context& ctx, std::size_t lane) noexcept {
        return static_cast<std::size_t>((*ctx.group_slots)[lane]) * N + lane;
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
    class StateIndex
>
struct BasicEwmNode {
    static constexpr double span = std::bit_cast<double>(SpanBits);
    static_assert(span > 0.0);
    static constexpr double alpha = 2.0 / (span + 1.0);
    static constexpr double old_weight_factor = 1.0 - alpha;
    static constexpr std::size_t state_size = StateIndex::state_size;

    alignas(64) std::array<double, state_size> value{};
    alignas(64) std::array<double, state_size> weight{};
    alignas(64) std::array<std::int64_t, state_size> count{};
    alignas(64) std::array<std::uint8_t, state_size> initialized{};
    bool all_initialized = false;

    void setup() noexcept {
        value.fill(0.0);
        weight.fill(0.0);
        count.fill(0);
        initialized.fill(0);
        all_initialized = false;
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();

        if constexpr (MinPeriods <= 0 && IgnoreNa && !Adjust) {
            if constexpr (StateIndex::contiguous_lanes) {
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
        bool all_finite = true;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = 0; lane < N; ++lane) {
            input[lane] = ctx.template read<In>(lane);
            all_finite = all_finite && finite(input[lane]);
        }
        if (all_initialized && all_finite) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
            for (std::size_t lane = 0; lane < N; ++lane) {
                const double next = std::fma(alpha, input[lane] - value[lane], value[lane]);
                value[lane] = next;
                out[lane] = next;
            }
            return;
        }

        bool now_all_initialized = true;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = 0; lane < N; ++lane) {
            if (finite(input[lane])) {
                if (initialized[lane]) value[lane] = std::fma(alpha, input[lane] - value[lane], value[lane]);
                else {
                    value[lane] = input[lane];
                    initialized[lane] = 1;
                }
            }
            out[lane] = initialized[lane] ? value[lane] : kNaN;
            now_all_initialized = now_all_initialized && initialized[lane];
        }
        all_initialized = now_all_initialized;
    }

    template <class Context>
    STACKDSL_HOT void run_recursive_indexed(Context& ctx, double* STACKDSL_RESTRICT out) noexcept {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = 0; lane < N; ++lane) {
            const std::size_t index = StateIndex::get(ctx, lane);
            const double x = ctx.template read<In>(lane);
            if (finite(x)) {
                if (initialized[index]) value[index] = std::fma(alpha, x - value[index], value[index]);
                else {
                    value[index] = x;
                    initialized[index] = 1;
                }
            }
            out[lane] = initialized[index] ? value[index] : kNaN;
        }
    }

    template <class Context>
    STACKDSL_HOT void run_general(Context& ctx, double* STACKDSL_RESTRICT out) noexcept {
        for (std::size_t lane = 0; lane < N; ++lane) {
            const std::size_t index = StateIndex::get(ctx, lane);
            const double x = ctx.template read<In>(lane);
            const bool observation = finite(x);
            double old_weight = weight[index];
            if (initialized[index] && (observation || !IgnoreNa)) old_weight *= old_weight_factor;
            if (observation) {
                if (initialized[index]) {
                    double new_weight = Adjust ? 1.0 : alpha;
                    if constexpr (!Adjust) {
                        if (std::abs(alpha - 0.5) <= 1e-12) new_weight = 1.0 - old_weight;
                    }
                    if (value[index] != x) value[index] = (old_weight * value[index] + new_weight * x) / (old_weight + new_weight);
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

template <std::size_t N, class In, class Out, std::uint64_t SpanBits, int MinPeriods, bool IgnoreNa, bool Adjust>
using EwmNode = BasicEwmNode<N, In, Out, SpanBits, MinPeriods, IgnoreNa, Adjust, DirectStateIndex<N>>;

template <std::size_t N, std::size_t Capacity, class In, class Out, std::uint64_t SpanBits, int MinPeriods, bool IgnoreNa, bool Adjust>
using GroupedEwmNode = BasicEwmNode<N, In, Out, SpanBits, MinPeriods, IgnoreNa, Adjust, GroupedStateIndex<N, Capacity>>;

}  // namespace stackdsl
