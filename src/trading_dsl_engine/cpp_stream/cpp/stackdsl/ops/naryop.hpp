#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "stackdsl/engine.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template <class Result>
STACKDSL_HOT constexpr Result invalid_integral_result() noexcept {
    static_assert(std::is_integral_v<Result>);
    if constexpr (std::is_signed_v<Result>) return std::numeric_limits<Result>::min();
    else return std::numeric_limits<Result>::max();
}

// Policies are templated on the compiler-selected result type. Inputs are read in
// their native type; same-typed integer operations therefore never pass through
// double. Mixed-type promotion occurs only because the operation's result type
// requires it.
struct AddOp {
    static constexpr int arity = 2;
    template <class Result, class A, class B>
    STACKDSL_HOT static Result apply(A a, B b) noexcept {
        return static_cast<Result>(a) + static_cast<Result>(b);
    }
};

struct SubOp {
    static constexpr int arity = 2;
    template <class Result, class A, class B>
    STACKDSL_HOT static Result apply(A a, B b) noexcept {
        return static_cast<Result>(a) - static_cast<Result>(b);
    }
};

struct MulOp {
    static constexpr int arity = 2;
    template <class Result, class A, class B>
    STACKDSL_HOT static Result apply(A a, B b) noexcept {
        return static_cast<Result>(a) * static_cast<Result>(b);
    }
};

struct DivOp {
    static constexpr int arity = 2;
    template <class Result, class A, class B>
    STACKDSL_HOT static Result apply(A a, B b) noexcept {
        const Result lhs = static_cast<Result>(a);
        const Result rhs = static_cast<Result>(b);
        if (rhs == Result{0}) {
            if constexpr (std::is_floating_point_v<Result>) {
                return std::numeric_limits<Result>::quiet_NaN();
            } else {
                return invalid_integral_result<Result>();
            }
        }
        if constexpr (std::is_integral_v<Result>) {
            // When an explicitly integral key expression contains floor(div(...)),
            // the compiler keeps the division integral. Use mathematical floor
            // division rather than C++'s truncation-toward-zero rule.
            Result quotient = lhs / rhs;
            const Result remainder = lhs % rhs;
            if constexpr (std::is_signed_v<Result>) {
                if (remainder != Result{0} && ((remainder < Result{0}) != (rhs < Result{0}))) {
                    --quotient;
                }
            }
            return quotient;
        } else {
            return lhs / rhs;
        }
    }
};

struct ModOp {
    static constexpr int arity = 2;
    template <class Result, class A, class B>
    STACKDSL_HOT static Result apply(A a, B b) noexcept {
        const Result lhs = static_cast<Result>(a);
        const Result rhs = static_cast<Result>(b);
        if (rhs == Result{0}) {
            if constexpr (std::is_floating_point_v<Result>) {
                return std::numeric_limits<Result>::quiet_NaN();
            } else {
                return invalid_integral_result<Result>();
            }
        }
        if constexpr (std::is_integral_v<Result>) return lhs % rhs;
        else return std::fmod(lhs, rhs);
    }
};

struct FloorOp {
    static constexpr int arity = 1;
    template <class Result, class A>
    STACKDSL_HOT static Result apply(A a) noexcept {
        if constexpr (std::is_integral_v<Result>) return static_cast<Result>(a);
        else return std::floor(static_cast<Result>(a));
    }
};

template <
    std::size_t N,
    class Lhs,
    class Rhs,
    class Out,
    class Result,
    class Op,
    class Execution = DirectExecution<N>
>
struct BinaryNode {
    static_assert(Op::arity == 2);
    static_assert(std::is_same_v<Out, OutputDst> || std::is_same_v<destination_value_t<Out>, Result>);
    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        (void)sizeof(Execution);
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
        for (std::size_t lane = 0; lane < N; ++lane) {
            out[lane] = Op::template apply<Result>(
                ctx.template read_native<Lhs>(lane),
                ctx.template read_native<Rhs>(lane)
            );
        }
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    class Result,
    class Op,
    class Execution = DirectExecution<N>
>
struct UnaryNode {
    static_assert(Op::arity == 1);
    static_assert(std::is_same_v<Out, OutputDst> || std::is_same_v<destination_value_t<Out>, Result>);
    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        (void)sizeof(Execution);
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
        for (std::size_t lane = 0; lane < N; ++lane) {
            out[lane] = Op::template apply<Result>(ctx.template read_native<In>(lane));
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
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        for (std::size_t lane = 0; lane < N; ++lane) {
            out[lane] = ctx.template read_native<In>(lane);
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
