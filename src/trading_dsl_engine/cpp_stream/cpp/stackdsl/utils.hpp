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

// Generic compile-time packs used by automatic multi-output lowering.  Keeping
// the container operator-agnostic lets EWM, reductions, and model projections
// share the same code-generation machinery.
template <class... Types>
struct TypeList {};

template <class T>
inline constexpr bool always_false_v = false;

template <class T>
STACKDSL_HOT bool finite(T value) noexcept {
    if constexpr (std::is_floating_point_v<T>) return std::isfinite(value);
    else return true;
}

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
    static constexpr std::size_t feature_width = 1;
};

template <std::size_t Index, class ValueType = double, bool RowScalar = false>
struct SlotSrc {
    static constexpr std::size_t slot_index = Index;
    using value_type = ValueType;
    static constexpr bool row_scalar = RowScalar;
    static constexpr std::size_t feature_width = 1;
};

template <std::size_t Index, std::size_t Width>
struct MatrixSlotSrc {
    static constexpr std::size_t matrix_slot_index = Index;
    static constexpr std::size_t feature_width = Width;
    using value_type = double;
};

template <std::size_t Index, std::size_t Size>
struct TensorSlotSrc {
    static constexpr std::size_t matrix_slot_index = Index;
    static constexpr std::size_t tensor_slot_index = Index;
    static constexpr std::size_t tensor_size = Size;
    static constexpr std::size_t feature_width = Size;
    using value_type = double;
};

template <auto Value>
struct LiteralSrc {
    using value_type = std::remove_cv_t<decltype(Value)>;
    static constexpr value_type value = Value;
    static constexpr std::size_t feature_width = 1;
};

struct NaNLiteralSrc { using value_type = double; static constexpr double value = std::numeric_limits<double>::quiet_NaN(); static constexpr std::size_t feature_width = 1; };
struct PositiveInfinityLiteralSrc { using value_type = double; static constexpr double value = std::numeric_limits<double>::infinity(); static constexpr std::size_t feature_width = 1; };
struct NegativeInfinityLiteralSrc { using value_type = double; static constexpr double value = -std::numeric_limits<double>::infinity(); static constexpr std::size_t feature_width = 1; };

struct OutputDst { using value_type = double; };

template <std::size_t Index, class ValueType = double>
struct SlotDst { static constexpr std::size_t slot_index = Index; using value_type = ValueType; };

template <std::size_t Index, std::size_t Width>
struct MatrixSlotDst { static constexpr std::size_t matrix_slot_index = Index; static constexpr std::size_t feature_width = Width; using value_type = double; };

template <std::size_t Index, std::size_t Size>
struct TensorSlotDst { static constexpr std::size_t matrix_slot_index = Index; static constexpr std::size_t tensor_slot_index = Index; static constexpr std::size_t tensor_size = Size; using value_type = double; };

template <class T> inline constexpr bool is_literal_source_v = requires { T::value; };
template <class Src> using source_value_t = typename Src::value_type;
template <class Dst> using destination_value_t = typename Dst::value_type;
template <class Src> inline constexpr std::size_t source_width_v = Src::feature_width;

template <std::size_t N, std::size_t Inputs, std::size_t ScratchSlots, std::size_t MatrixScratchSlots = 0, std::size_t MatrixScratchWidth = 1>
struct alignas(64) RowContext {
    std::array<const void*, Inputs> inputs{};
    alignas(64) std::array<std::array<double, N>, ScratchSlots> scratch_f64{};
    alignas(64) std::array<std::array<float, N>, ScratchSlots> scratch_f32{};
    alignas(64) std::array<std::array<std::int64_t, N>, ScratchSlots> scratch_i64{};
    alignas(64) std::array<std::array<std::uint64_t, N>, ScratchSlots> scratch_u64{};
    alignas(64) std::array<std::array<std::int32_t, N>, ScratchSlots> scratch_i32{};
    alignas(64) std::array<std::array<std::uint32_t, N>, ScratchSlots> scratch_u32{};
    alignas(64) std::array<std::array<double, N * MatrixScratchWidth>, MatrixScratchSlots> scratch_matrix_f64{};
    double* output=nullptr;
    std::size_t lane_begin=0;
    std::size_t lane_end=N;

    template <class T>
    STACKDSL_HOT auto& scratch_storage() noexcept {
        if constexpr (std::is_same_v<T, double>) return scratch_f64;
        else if constexpr (std::is_same_v<T, float>) return scratch_f32;
        else if constexpr (std::is_same_v<T, std::int64_t>) return scratch_i64;
        else if constexpr (std::is_same_v<T, std::uint64_t>) return scratch_u64;
        else if constexpr (std::is_same_v<T, std::int32_t>) return scratch_i32;
        else if constexpr (std::is_same_v<T, std::uint32_t>) return scratch_u32;
        else static_assert(always_false_v<T>, "unsupported cpp_stream scratch type");
    }

    template <class T>
    STACKDSL_HOT const auto& scratch_storage() const noexcept {
        if constexpr (std::is_same_v<T, double>) return scratch_f64;
        else if constexpr (std::is_same_v<T, float>) return scratch_f32;
        else if constexpr (std::is_same_v<T, std::int64_t>) return scratch_i64;
        else if constexpr (std::is_same_v<T, std::uint64_t>) return scratch_u64;
        else if constexpr (std::is_same_v<T, std::int32_t>) return scratch_i32;
        else if constexpr (std::is_same_v<T, std::uint32_t>) return scratch_u32;
        else static_assert(always_false_v<T>, "unsupported cpp_stream scratch type");
    }

    template <class Src>
    STACKDSL_HOT source_value_t<Src> read_native(std::size_t lane) const noexcept {
        static_assert(source_width_v<Src> == 1, "scalar read of matrix/tensor source");
        if constexpr (requires { Src::read(*this, lane); }) {
            return Src::read(*this, lane);
        } else if constexpr (requires { Src::input_index; }) {
            const auto* values = static_cast<const source_value_t<Src>*>(inputs[Src::input_index]);
            return values[Src::row_width == 1 ? 0 : lane];
        } else if constexpr (requires { Src::slot_index; }) {
            return scratch_storage<source_value_t<Src>>()[Src::slot_index][Src::row_scalar ? 0 : lane];
        } else {
            return Src::value;
        }
    }

    template <class Src>
    STACKDSL_HOT double read(std::size_t lane) const noexcept { return static_cast<double>(read_native<Src>(lane)); }

    template <class Src>
    STACKDSL_HOT double read_feature(std::size_t lane, std::size_t feature) const noexcept {
        if constexpr (requires { Src::matrix_slot_index; }) return scratch_matrix_f64[Src::matrix_slot_index][lane * Src::feature_width + feature];
        else { (void)feature; return read<Src>(lane); }
    }

    template <class Src>
    STACKDSL_HOT const double* read_ptr() const noexcept {
        static_assert(source_width_v<Src> == 1);
        static_assert(!is_literal_source_v<Src>);
        static_assert(std::is_same_v<source_value_t<Src>, double>);
        if constexpr (requires { Src::input_index; }) {
            static_assert(Src::row_width == 0 || Src::row_width == N);
            return static_cast<const double*>(inputs[Src::input_index]);
        } else {
            static_assert(!Src::row_scalar);
            return scratch_f64[Src::slot_index].data();
        }
    }

    template <class Dst>
    STACKDSL_HOT auto* write_ptr() noexcept {
        if constexpr (std::is_same_v<Dst, OutputDst>) return output;
        else if constexpr (requires { Dst::matrix_slot_index; }) return scratch_matrix_f64[Dst::matrix_slot_index].data();
        else return scratch_storage<destination_value_t<Dst>>()[Dst::slot_index].data();
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
