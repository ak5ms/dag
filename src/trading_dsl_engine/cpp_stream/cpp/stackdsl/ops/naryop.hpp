#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "stackdsl/engine.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

struct AddOp { static constexpr int arity = 2; STACKDSL_HOT static double apply(double a, double b) noexcept { return a + b; } };
struct SubOp { static constexpr int arity = 2; STACKDSL_HOT static double apply(double a, double b) noexcept { return a - b; } };
struct MulOp { static constexpr int arity = 2; STACKDSL_HOT static double apply(double a, double b) noexcept { return a * b; } };
struct DivOp { static constexpr int arity = 2; STACKDSL_HOT static double apply(double a, double b) noexcept { return b == 0.0 ? kNaN : a / b; } };
struct ModOp { static constexpr int arity = 2; STACKDSL_HOT static double apply(double a, double b) noexcept { return b == 0.0 ? kNaN : std::fmod(a, b); } };
struct FloorOp { static constexpr int arity = 1; STACKDSL_HOT static double apply(double a) noexcept { return std::floor(a); } };

template <
    std::size_t N,
    class Lhs,
    class Rhs,
    class Out,
    class Op,
    class Execution = DirectExecution<N>
>
struct BinaryNode {
    static_assert(Op::arity == 2);
    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        (void)sizeof(Execution);
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
        for (std::size_t lane = 0; lane < N; ++lane) {
            out[lane] = Op::apply(ctx.template read<Lhs>(lane), ctx.template read<Rhs>(lane));
        }
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    class Op,
    class Execution = DirectExecution<N>
>
struct UnaryNode {
    static_assert(Op::arity == 1);
    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        (void)sizeof(Execution);
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
        for (std::size_t lane = 0; lane < N; ++lane) {
            out[lane] = Op::apply(ctx.template read<In>(lane));
        }
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    class Execution = DirectExecution<N>
>
struct CopyNode {
    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        (void)sizeof(Execution);
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        for (std::size_t lane = 0; lane < N; ++lane) {
            out[lane] = ctx.template read<In>(lane);
        }
    }
};

struct RankItem {
    std::uint32_t group;
    double value;
    std::uint32_t lane;
};

template <
    std::size_t N,
    class In,
    class Out,
    class Execution = DirectExecution<N>
>
struct XsRankNode {
    static constexpr int arity = 1;
    RankScoreTable<N> scores{};

    void setup() noexcept { scores.setup(); }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        if constexpr (N <= 16) rank_count(ctx, out);
        else rank_sort(ctx, out);
    }

private:
    template <class Context>
    STACKDSL_HOT void rank_count(Context& ctx, double* STACKDSL_RESTRICT out) noexcept {
        std::array<double, N> values{};
        std::array<std::uint32_t, N> groups{};
        bool all_finite = true;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = 0; lane < N; ++lane) {
            values[lane] = ctx.template read<In>(lane);
            groups[lane] = Execution::rank_group(ctx, lane);
            all_finite = all_finite && finite(values[lane]);
        }

        if (all_finite) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
            for (std::size_t lane = 0; lane < N; ++lane) {
                std::size_t count = 0;
                std::size_t upper = 0;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
                for (std::size_t other = 0; other < N; ++other) {
                    const bool same = groups[other] == groups[lane];
                    count += static_cast<std::size_t>(same);
                    upper += static_cast<std::size_t>(same && values[other] <= values[lane]);
                }
                out[lane] = scores.get(count, upper - 1);
            }
            return;
        }

        std::array<std::uint8_t, N> valid{};
        for (std::size_t lane = 0; lane < N; ++lane) {
            valid[lane] = static_cast<std::uint8_t>(finite(values[lane]));
            if (!valid[lane]) out[lane] = kNaN;
        }
        for (std::size_t lane = 0; lane < N; ++lane) {
            if (!valid[lane]) continue;
            std::size_t count = 0;
            std::size_t upper = 0;
            for (std::size_t other = 0; other < N; ++other) {
                if (!valid[other] || groups[other] != groups[lane]) continue;
                ++count;
                upper += static_cast<std::size_t>(values[other] <= values[lane]);
            }
            out[lane] = scores.get(count, upper - 1);
        }
    }

    template <class Context>
    STACKDSL_HOT void rank_sort(Context& ctx, double* STACKDSL_RESTRICT out) noexcept {
        std::array<RankItem, N> items{};
        std::size_t count = 0;
        for (std::size_t lane = 0; lane < N; ++lane) {
            const double value = ctx.template read<In>(lane);
            if (finite(value)) {
                items[count++] = RankItem{
                    Execution::rank_group(ctx, lane),
                    value,
                    static_cast<std::uint32_t>(lane),
                };
            } else {
                out[lane] = kNaN;
            }
        }
        std::sort(
            items.begin(),
            items.begin() + static_cast<std::ptrdiff_t>(count),
            [](const RankItem& a, const RankItem& b) {
                return a.group < b.group || (a.group == b.group && a.value < b.value);
            }
        );
        std::size_t group_start = 0;
        while (group_start < count) {
            std::size_t group_end = group_start + 1;
            while (group_end < count && items[group_end].group == items[group_start].group) ++group_end;
            const std::size_t group_count = group_end - group_start;
            std::size_t tie_start = group_start;
            while (tie_start < group_end) {
                std::size_t upper = tie_start + 1;
                while (upper < group_end && items[upper].value == items[tie_start].value) ++upper;
                const double score = scores.get(group_count, upper - group_start - 1);
                for (std::size_t pos = tie_start; pos < upper; ++pos) out[items[pos].lane] = score;
                tie_start = upper;
            }
            group_start = group_end;
        }
    }
};

}  // namespace stackdsl
