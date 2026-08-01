#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if defined(__GNUC__) || defined(__clang__)
#define STACKDSL_HOT inline __attribute__((always_inline, hot))
#define STACKDSL_RESTRICT __restrict__
#else
#define STACKDSL_HOT inline
#define STACKDSL_RESTRICT
#endif

namespace stackdsl {

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

STACKDSL_HOT bool finite(double value) noexcept { return std::isfinite(value); }

STACKDSL_HOT double norm_inv(double p) noexcept {
    if (std::isnan(p)) return kNaN;
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return std::numeric_limits<double>::infinity();
    constexpr double a1=-3.969683028665376e+01,a2=2.209460984245205e+02,a3=-2.759285104469687e+02,a4=1.383577518672690e+02,a5=-3.066479806614716e+01,a6=2.506628277459239e+00;
    constexpr double b1=-5.447609879822406e+01,b2=1.615858368580409e+02,b3=-1.556989798598866e+02,b4=6.680131188771972e+01,b5=-1.328068155288572e+01;
    constexpr double c1=-7.784894002430293e-03,c2=-3.223964580411365e-01,c3=-2.400758277161838e+00,c4=-2.549732539343734e+00,c5=4.374664141464968e+00,c6=2.938163982698783e+00;
    constexpr double d1=7.784695709041462e-03,d2=3.224671290700398e-01,d3=2.445134137142996e+00,d4=3.754408661907416e+00;
    constexpr double p_low=0.02425,p_high=1.0-p_low;
    double x;
    if (p < p_low) { const double q=std::sqrt(-2.0*std::log(p)); x=(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1.0); }
    else if (p <= p_high) { const double q=p-0.5,r=q*q; x=(((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0); }
    else { const double q=std::sqrt(-2.0*std::log(1.0-p)); x=-(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1.0); }
    constexpr double inv_sqrt_2=0.707106781186547524400844362104849039;
    constexpr double sqrt_2pi=2.506628274631000502415765284811045253;
    const double error=0.5*std::erfc(-x*inv_sqrt_2)-p;
    const double u=error*sqrt_2pi*std::exp(0.5*x*x);
    return x-u/(1.0+0.5*x*u);
}

template <std::size_t Index, class ValueType = double, std::size_t RowWidth = 0>
struct InputSrc {
    static constexpr std::size_t input_index = Index;
    using value_type = ValueType;
    static constexpr std::size_t row_width = RowWidth;
};

template <std::size_t Index> struct SlotSrc { static constexpr std::size_t slot_index=Index; };
template <double Value> struct LiteralSrc { static constexpr double value=Value; };
struct OutputDst {};
template <std::size_t Index> struct SlotDst { static constexpr std::size_t slot_index=Index; };
template <class T> inline constexpr bool is_literal_source_v = requires { T::value; };

template <std::size_t N, std::size_t Inputs, std::size_t ScratchSlots>
struct alignas(64) RowContext {
    std::array<const void*, Inputs> inputs{};
    alignas(64) std::array<std::array<double, N>, ScratchSlots> scratch{};
    double* output=nullptr;

    template <class Src>
    STACKDSL_HOT double read(std::size_t lane) const noexcept {
        if constexpr (requires { Src::input_index; }) {
            using ValueType = typename Src::value_type;
            const auto* values = static_cast<const ValueType*>(inputs[Src::input_index]);
            const std::size_t offset = Src::row_width == 1 ? 0 : lane;
            return static_cast<double>(values[offset]);
        } else if constexpr (requires { Src::slot_index; }) {
            return scratch[Src::slot_index][lane];
        } else {
            return Src::value;
        }
    }

    template <class Src>
    STACKDSL_HOT const double* read_ptr() const noexcept {
        static_assert(!is_literal_source_v<Src>);
        if constexpr (requires { Src::input_index; }) {
            static_assert(std::is_same_v<typename Src::value_type, double>);
            static_assert(Src::row_width == 0 || Src::row_width == N);
            return static_cast<const double*>(inputs[Src::input_index]);
        } else {
            return scratch[Src::slot_index].data();
        }
    }

    template <class Dst>
    STACKDSL_HOT double* write_ptr() noexcept {
        if constexpr (std::is_same_v<Dst, OutputDst>) return output;
        else return scratch[Dst::slot_index].data();
    }
};

template <std::size_t N>
struct RankScoreTable {
    std::array<double,(N+1)*N> values{};
    void setup() noexcept {
        for (std::size_t count=1; count<=N; ++count) {
            const double denom=static_cast<double>(count+1);
            for (std::size_t pos=0; pos<count; ++pos) values[count*N+pos]=norm_inv(static_cast<double>(pos+1)/denom);
        }
    }
    STACKDSL_HOT double get(std::size_t count,std::size_t upper_minus_one) const noexcept { return values[count*N+upper_minus_one]; }
};

}  // namespace stackdsl
