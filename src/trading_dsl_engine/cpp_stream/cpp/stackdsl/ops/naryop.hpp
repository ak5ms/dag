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

template <class Value>
STACKDSL_HOT bool is_nan_value(Value value) noexcept {
    if constexpr (std::is_floating_point_v<Value>) return std::isnan(value);
    else {
        (void)value;
        return false;
    }
}

template <class Result, class... Values>
STACKDSL_HOT Result predicate_result(bool predicate, Values... values) noexcept {
    if ((is_nan_value(values) || ...)) {
        if constexpr (std::is_floating_point_v<Result>) {
            return std::numeric_limits<Result>::quiet_NaN();
        } else {
            return invalid_integral_result<Result>();
        }
    }
    return static_cast<Result>(predicate);
}

struct AddOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        return static_cast<R>(a) + static_cast<R>(b);
    }
};
struct SubOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        return static_cast<R>(a) - static_cast<R>(b);
    }
};
struct MulOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        return static_cast<R>(a) * static_cast<R>(b);
    }
};
struct PowOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        return static_cast<R>(
            std::pow(static_cast<double>(a), static_cast<double>(b))
        );
    }
};

struct DivOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        const R lhs = static_cast<R>(a), rhs = static_cast<R>(b);
        if (rhs == R{0}) {
            if constexpr (std::is_floating_point_v<R>) {
                return std::numeric_limits<R>::quiet_NaN();
            } else {
                return invalid_integral_result<R>();
            }
        }
        if constexpr (std::is_integral_v<R>) {
            R quotient = lhs / rhs;
            const R remainder = lhs % rhs;
            if constexpr (std::is_signed_v<R>) {
                if (remainder != R{0} && ((remainder < R{0}) != (rhs < R{0}))) {
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
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        const R lhs = static_cast<R>(a), rhs = static_cast<R>(b);
        if (rhs == R{0}) {
            if constexpr (std::is_floating_point_v<R>) {
                return std::numeric_limits<R>::quiet_NaN();
            } else {
                return invalid_integral_result<R>();
            }
        }
        if constexpr (std::is_integral_v<R>) return lhs % rhs;
        else return std::fmod(lhs, rhs);
    }
};

struct FloorOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        if constexpr (std::is_integral_v<R>) return static_cast<R>(a);
        else return std::floor(static_cast<R>(a));
    }
};
struct AbsOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(std::abs(static_cast<double>(a)));
    }
};
struct CeilOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(std::ceil(static_cast<double>(a)));
    }
};
struct ExpOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(std::exp(static_cast<double>(a)));
    }
};
struct LogOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(std::log(static_cast<double>(a)));
    }
};
struct RoundOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(std::nearbyint(static_cast<double>(a)));
    }
};
struct SignOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        const double value = static_cast<double>(a);
        if (std::isnan(value)) {
            return std::numeric_limits<R>::quiet_NaN();
        }
        return static_cast<R>((value > 0.0) - (value < 0.0));
    }
};
struct FractionOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        const double value = static_cast<double>(a);
        return static_cast<R>(std::copysign(
            std::abs(value) - std::floor(std::abs(value)), value
        ));
    }
};
struct PurifyOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        const double value = static_cast<double>(a);
        return std::isfinite(value)
            ? static_cast<R>(value)
            : std::numeric_limits<R>::quiet_NaN();
    }
};
struct AtanOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(std::atan(static_cast<double>(a)));
    }
};
struct AcosOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(std::acos(static_cast<double>(a)));
    }
};
struct AsinOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(std::asin(static_cast<double>(a)));
    }
};
struct SinOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(std::sin(static_cast<double>(a)));
    }
};
struct CosOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(std::cos(static_cast<double>(a)));
    }
};
struct TanOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(std::tan(static_cast<double>(a)));
    }
};
struct TanhOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(std::tanh(static_cast<double>(a)));
    }
};
struct SqrtOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(std::sqrt(static_cast<double>(a)));
    }
};
struct IsNanOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(is_nan_value(a));
    }
};
struct IsFiniteOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        if constexpr (std::is_floating_point_v<A>) {
            return static_cast<R>(std::isfinite(a));
        }
        return R{1};
    }
};
struct LogicalNotOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return predicate_result<R>(a == A{0}, a);
    }
};
struct NormInvOp {
    static constexpr int arity = 1;
    template <class R, class A>
    STACKDSL_HOT static R apply(A a) noexcept {
        return static_cast<R>(norm_inv(static_cast<double>(a)));
    }
};
struct MinOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        if (is_nan_value(a) || is_nan_value(b)) {
            return std::numeric_limits<R>::quiet_NaN();
        }
        return std::min(static_cast<R>(a), static_cast<R>(b));
    }
};
struct MaxOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        if (is_nan_value(a) || is_nan_value(b)) {
            return std::numeric_limits<R>::quiet_NaN();
        }
        return std::max(static_cast<R>(a), static_cast<R>(b));
    }
};
struct EqOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        return predicate_result<R>(a == b, a, b);
    }
};
struct NeOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        return predicate_result<R>(a != b, a, b);
    }
};
struct LtOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        return predicate_result<R>(a < b, a, b);
    }
};
struct GtOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        return predicate_result<R>(a > b, a, b);
    }
};
struct LeOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        return predicate_result<R>(a <= b, a, b);
    }
};
struct GeOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        return predicate_result<R>(a >= b, a, b);
    }
};
struct AndOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        return predicate_result<R>((a != A{0}) && (b != B{0}), a, b);
    }
};
struct OrOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        return predicate_result<R>((a != A{0}) || (b != B{0}), a, b);
    }
};
struct XorOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        return predicate_result<R>((a != A{0}) != (b != B{0}), a, b);
    }
};
struct FillNaOp {
    static constexpr int arity = 2;
    template <class R, class A, class B>
    STACKDSL_HOT static R apply(A a, B b) noexcept {
        if constexpr (std::is_floating_point_v<A>) {
            return static_cast<R>(std::isnan(a) ? b : a);
        } else {
            return static_cast<R>(a);
        }
    }
};
struct WhereOp {
    static constexpr int arity = 3;
    template <class R, class C, class A, class B>
    STACKDSL_HOT static R apply(C c, A a, B b) noexcept {
        return static_cast<R>((c != C{0}) ? a : b);
    }
};

template <class Result, class Op, class... Inputs>
struct NaryExpressionSrc {
    static_assert(sizeof...(Inputs) == Op::arity);
    using value_type = Result;
    static constexpr std::size_t feature_width = 1;

    template <class Context>
    STACKDSL_HOT static Result read(
        const Context& ctx, std::size_t lane
    ) noexcept {
        return Op::template apply<Result>(
            ctx.template read_native<Inputs>(lane)...
        );
    }
};

template <class Result, class Op, class... Inputs>
struct expression_source_traits<
    NaryExpressionSrc<Result, Op, Inputs...>
> {
    using source = NaryExpressionSrc<Result, Op, Inputs...>;
    using nested = typename expression_sources_for<Inputs...>::type;
    using type = typename type_list_append_unique<nested, source>::type;
};

namespace nary_detail {

template <class Exponent>
consteval bool has_small_integer_exponent() {
    if constexpr (!requires { Exponent::value; }) {
        return false;
    } else {
        constexpr auto value = Exponent::value;
        if constexpr (std::is_integral_v<decltype(value)>) {
            return value >= -64 && value <= 64;
        } else {
            return value >= -64.0 && value <= 64.0
                && value == static_cast<long long>(value);
        }
    }
}

template <unsigned long long Exponent, class Result>
STACKDSL_HOT Result positive_integer_power(Result base) noexcept {
    if constexpr (Exponent == 0) return Result{1};
    else if constexpr (Exponent == 1) return base;
    else {
        const Result half = positive_integer_power<Exponent / 2>(base);
        const Result square = half * half;
        if constexpr ((Exponent & 1ULL) != 0) return square * base;
        else return square;
    }
}

template <long long Exponent, class Result>
STACKDSL_HOT Result integer_power(Result base) noexcept {
    if constexpr (Exponent < 0) {
        return Result{1} /
            positive_integer_power<static_cast<unsigned long long>(-Exponent)>(
                base
            );
    } else {
        return positive_integer_power<
            static_cast<unsigned long long>(Exponent)
        >(base);
    }
}

}  // namespace nary_detail

template <class Result, class Base, class Exponent>
struct NaryExpressionSrc<Result, PowOp, Base, Exponent> {
    using value_type = Result;
    static constexpr std::size_t feature_width = 1;

    template <class Context>
    STACKDSL_HOT static Result read(
        const Context& ctx, std::size_t lane
    ) noexcept {
        if constexpr (nary_detail::has_small_integer_exponent<Exponent>()) {
            return nary_detail::integer_power<
                static_cast<long long>(Exponent::value)
            >(static_cast<Result>(ctx.template read_native<Base>(lane)));
        } else {
            return PowOp::template apply<Result>(
                ctx.template read_native<Base>(lane),
                ctx.template read_native<Exponent>(lane)
            );
        }
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
    STACKDSL_HOT void setup() noexcept {}
    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            out[lane] = Op::template apply<Result>(
                ctx.template read_native<Lhs>(lane),
                ctx.template read_native<Rhs>(lane)
            );
        }
    }
};

template <
    std::size_t N,
    class A,
    class B,
    class C,
    class Out,
    class Result,
    class Op,
    class Execution = DirectExecution<N>
>
struct TernaryNode {
    static_assert(Op::arity == 3);
    STACKDSL_HOT void setup() noexcept {}
    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            out[lane] = Op::template apply<Result>(
                ctx.template read_native<A>(lane),
                ctx.template read_native<B>(lane),
                ctx.template read_native<C>(lane)
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
    STACKDSL_HOT void setup() noexcept {}
    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            out[lane] = Op::template apply<Result>(
                ctx.template read_native<In>(lane)
            );
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
        auto* out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            out[lane] = ctx.template read_native<In>(lane);
        }
    }
};

// Keep records compact: both rank sorting and xs_gauss move these repeatedly.
struct RankItem {
    double value;
    std::uint32_t group;
    std::uint32_t lane;
};
static_assert(sizeof(RankItem) == 16);

namespace nary_detail {

template <std::size_t N, class Less>
STACKDSL_HOT void sort_rank_items(
    std::array<RankItem, N>& items,
    std::size_t count,
    Less less
) noexcept {
    // The common futures cross section is small.  Avoid introsort setup and let
    // the compiler specialize a compact insertion sort for those fixed widths.
    if constexpr (N <= 16) {
        for (std::size_t index = 1; index < count; ++index) {
            const RankItem item = items[index];
            std::size_t position = index;
            while (position != 0 && less(item, items[position - 1])) {
                items[position] = items[position - 1];
                --position;
            }
            items[position] = item;
        }
    } else {
        std::sort(
            items.begin(),
            items.begin() + static_cast<std::ptrdiff_t>(count),
            less
        );
    }
}

struct MagnitudeLess {
    STACKDSL_HOT bool operator()(
        const RankItem& left, const RankItem& right
    ) const noexcept {
        if (left.group != right.group) return left.group < right.group;
        const double left_abs = std::abs(left.value);
        const double right_abs = std::abs(right.value);
        if (left_abs != right_abs) return left_abs < right_abs;
        return left.lane < right.lane;
    }
};

struct ValueLess {
    STACKDSL_HOT bool operator()(
        const RankItem& left, const RankItem& right
    ) const noexcept {
        if (left.group != right.group) return left.group < right.group;
        if (left.value != right.value) return left.value < right.value;
        return left.lane < right.lane;
    }
};

}  // namespace nary_detail

// The primary template is defined in cross_sectional.hpp.  A declaration here
// lets the -0.0 compile-time tag select the native xs_gauss node without a
// separate header or a runtime branch.
template <
    std::size_t N,
    class In,
    class Out,
    std::uint64_t PowerBits,
    class Execution
>
struct XsGeneralizedRankNode;

inline constexpr std::uint64_t kXsGaussPowerTag = 0x8000000000000000ULL;

template <
    std::size_t N,
    class In,
    class Out,
    class Execution = DirectExecution<N>
>
struct XsGaussNode {
    static constexpr std::size_t Groups = Execution::cross_state_size;
    static constexpr double MaxProbability = 0x1.fffffffffffffp-1;

    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t write_begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t write_end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = write_begin; lane < write_end; ++lane) {
            out[lane] = kNaN;
        }

        std::array<RankItem, N> items{};
        std::array<double, N> levels{};
        std::size_t item_count = 0;
        for (std::size_t lane = 0; lane < N; ++lane) {
            const double value = ctx.template read<In>(lane);
            if (!finite(value)) continue;
            items[item_count++] = {
                value,
                static_cast<std::uint32_t>(Execution::cross_group(ctx, lane)),
                static_cast<std::uint32_t>(lane),
            };
        }
        if (item_count == 0) return;

        // q's spacing is the cumulative sum of sorted absolute magnitudes.
        nary_detail::sort_rank_items(
            items, item_count, nary_detail::MagnitudeLess{}
        );
        std::size_t group_begin = 0;
        while (group_begin < item_count) {
            std::size_t group_end = group_begin + 1;
            while (
                group_end < item_count
                && items[group_end].group == items[group_begin].group
            ) {
                ++group_end;
            }

            const double min_abs = std::abs(items[group_begin].value);
            const double max_abs = std::abs(items[group_end - 1].value);
            if (!(max_abs > 0.0)) {
                for (std::size_t position = group_begin; position < group_end; ++position) {
                    levels[position] = 0.5;
                }
                group_begin = group_end;
                continue;
            }

            double total = 0.0;
            for (std::size_t position = group_begin; position < group_end; ++position) {
                total += std::abs(items[position].value);
            }
            double denominator = total + 0.5 * (min_abs + max_abs);
            double multiplier = 1.0;
            if (!finite(denominator)) {
                // Rare overflow fallback; common rows avoid the extra divisions.
                multiplier = 1.0 / max_abs;
                total = 0.0;
                for (std::size_t position = group_begin; position < group_end; ++position) {
                    total += std::abs(items[position].value) * multiplier;
                }
                denominator = total + 0.5 * (min_abs * multiplier + 1.0);
            }

            double cumulative = 0.0;
            for (std::size_t position = group_begin; position < group_end; ++position) {
                cumulative += std::abs(items[position].value) * multiplier;
                levels[position] = cumulative / denominator;
            }

            // Zero magnitudes otherwise create q=0 and norm_inv(q)=-inf.
            std::size_t first_positive = group_begin;
            while (
                first_positive < group_end
                && std::abs(items[first_positive].value) == 0.0
            ) {
                ++first_positive;
            }
            if (first_positive != group_begin && first_positive < group_end) {
                const double first_level = levels[first_positive];
                const std::size_t zero_count = first_positive - group_begin;
                const double spacing = first_level / static_cast<double>(zero_count + 1);
                for (std::size_t index = 0; index < zero_count; ++index) {
                    levels[group_begin + index] = spacing * static_cast<double>(index + 1);
                }
            }
            group_begin = group_end;
        }

        // Assign the magnitude-derived grid by ascending x rank.  The two sorts
        // are intrinsic: magnitude order and signed-value order are independent.
        nary_detail::sort_rank_items(
            items, item_count, nary_detail::ValueLess{}
        );
        std::array<double, Groups> mean{};
        std::array<double, Groups> m2{};
        std::array<std::uint32_t, Groups> count{};

        group_begin = 0;
        while (group_begin < item_count) {
            std::size_t group_end = group_begin + 1;
            while (
                group_end < item_count
                && items[group_end].group == items[group_begin].group
            ) {
                ++group_end;
            }
            std::size_t tie_begin = group_begin;
            while (tie_begin < group_end) {
                std::size_t tie_end = tie_begin + 1;
                while (
                    tie_end < group_end
                    && items[tie_end].value == items[tie_begin].value
                ) {
                    ++tie_end;
                }
                double probability = levels[tie_end - 1];
                probability = std::max(
                    std::numeric_limits<double>::min(),
                    std::min(probability, MaxProbability)
                );
                const double score = norm_inv(probability);
                const std::size_t group = items[tie_begin].group;
                for (std::size_t position = tie_begin; position < tie_end; ++position) {
                    // Reuse the value field for the raw score; this removes an
                    // additional N-double scratch array and another input read.
                    items[position].value = score;
                    const std::uint32_t next_count = count[group] + 1;
                    const double delta = score - mean[group];
                    mean[group] += delta / static_cast<double>(next_count);
                    m2[group] = std::fma(
                        delta, score - mean[group], m2[group]
                    );
                    count[group] = next_count;
                }
                tie_begin = tie_end;
            }
            group_begin = group_end;
        }

        std::array<double, Groups> scale{};
        for (std::size_t group = 0; group < Groups; ++group) {
            if (count[group] != 0) {
                scale[group] = std::sqrt(
                    std::max(0.0, m2[group]) / static_cast<double>(count[group])
                );
            }
        }
        for (std::size_t position = 0; position < item_count; ++position) {
            const std::size_t lane = items[position].lane;
            if (lane < write_begin || lane >= write_end) continue;
            const double stddev = scale[items[position].group];
            out[lane] = stddev > 0.0 && finite(stddev)
                ? items[position].value / stddev
                : 0.0;
        }
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    class Execution
>
struct XsGeneralizedRankNode<
    N, In, Out, kXsGaussPowerTag, Execution
> : XsGaussNode<N, In, Out, Execution> {};

template <
    std::size_t N,
    class In,
    class Out,
    class Score,
    class Execution = DirectExecution<N>
>
struct XsRankNodeImpl {
    Score scores{};
    void setup() noexcept { scores.setup(); }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* out = ctx.template write_ptr<Out>();
        if constexpr (N <= 16) rank_count(ctx, out);
        else rank_sort(ctx, out);
    }

private:
    template <class Context>
    STACKDSL_HOT void rank_count(Context& ctx, double* out) noexcept {
        std::array<double, N> values{};
        std::array<std::uint32_t, N> groups{};
        bool all_finite = true;
        for (std::size_t lane = 0; lane < N; ++lane) {
            values[lane] = ctx.template read<In>(lane);
            groups[lane] = Execution::rank_group(ctx, lane);
            all_finite &= finite(values[lane]);
        }
        for (std::size_t lane = 0; lane < N; ++lane) {
            if (!finite(values[lane])) {
                out[lane] = kNaN;
                continue;
            }
            std::size_t count = 0, upper = 0;
            for (std::size_t other = 0; other < N; ++other) {
                if (
                    (all_finite || finite(values[other])) &&
                    groups[other] == groups[lane]
                ) {
                    ++count;
                    upper += static_cast<std::size_t>(
                        values[other] <= values[lane]
                    );
                }
            }
            out[lane] = scores.get(count, upper - 1);
        }
    }

    template <class Context>
    STACKDSL_HOT void rank_sort(Context& ctx, double* out) noexcept {
        std::array<RankItem, N> items{};
        std::size_t count = 0;
        for (std::size_t lane = 0; lane < N; ++lane) {
            const double value = ctx.template read<In>(lane);
            if (finite(value)) {
                items[count++] = {
                    value,
                    Execution::rank_group(ctx, lane),
                    static_cast<std::uint32_t>(lane),
                };
            } else {
                out[lane] = kNaN;
            }
        }
        nary_detail::sort_rank_items(
            items, count, nary_detail::ValueLess{}
        );
        std::size_t start = 0;
        while (start < count) {
            std::size_t end = start + 1;
            while (end < count && items[end].group == items[start].group) ++end;
            const std::size_t n = end - start;
            std::size_t tie = start;
            while (tie < end) {
                std::size_t upper = tie + 1;
                while (
                    upper < end && items[upper].value == items[tie].value
                ) {
                    ++upper;
                }
                const double score = scores.get(n, upper - start - 1);
                for (std::size_t p = tie; p < upper; ++p) {
                    out[items[p].lane] = score;
                }
                tie = upper;
            }
            start = end;
        }
    }
};

template <std::size_t N>
struct NormalRankScore {
    RankScoreTable<N> table{};
    void setup() noexcept { table.setup(); }
    STACKDSL_HOT double get(
        std::size_t count,
        std::size_t upper_minus_one
    ) const noexcept {
        return table.get(count, upper_minus_one);
    }
};

template <std::size_t N>
struct PctRankScore {
    STACKDSL_HOT void setup() noexcept {}
    STACKDSL_HOT double get(
        std::size_t count,
        std::size_t upper_minus_one
    ) const noexcept {
        return static_cast<double>(upper_minus_one + 1) /
            static_cast<double>(count + 1);
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    class Execution = DirectExecution<N>
>
struct XsRankNode : XsRankNodeImpl<
    N, In, Out, NormalRankScore<N>, Execution
> {};

template <
    std::size_t N,
    class In,
    class Out,
    class Execution = DirectExecution<N>
>
struct XsPctRankNode : XsRankNodeImpl<
    N, In, Out, PctRankScore<N>, Execution
> {};

}  // namespace stackdsl
