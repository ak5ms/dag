#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "stackdsl/ops/einsum.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template <std::size_t Index, class TensorSource>
struct ClarabelParameterBinding {
    static constexpr std::size_t index = Index;
    static constexpr bool feedback = false;
    using source_type = TensorSource;
};

template <
    std::size_t Index,
    std::size_t PrimalIndex,
    std::size_t Offset,
    std::size_t Count,
    std::size_t Stride,
    class InitialSource
>
struct ClarabelPreviousPrimalBinding {
    static constexpr std::size_t index = Index;
    static constexpr bool feedback = true;
    static constexpr std::size_t primal_index = PrimalIndex;
    static constexpr std::size_t offset = Offset;
    static constexpr std::size_t count = Count;
    static constexpr std::size_t stride = Stride;
    using source_type = InitialSource;
};

template <class... Bindings>
struct ClarabelParameterList {};

enum class ClarabelResultKind : std::uint8_t {
    Primal,
    Dual,
    ConstraintValue,
    Info,
};

template <
    ClarabelResultKind Kind,
    std::size_t SourceIndex,
    std::size_t Offset,
    std::size_t Count,
    std::size_t Stride,
    class Out
>
struct ClarabelProjection {
    static constexpr ClarabelResultKind kind = Kind;
    static constexpr std::size_t source_index = SourceIndex;
    static constexpr std::size_t offset = Offset;
    static constexpr std::size_t count = Count;
    static constexpr std::size_t stride = Stride;
    using output_type = Out;
};

template <
    std::size_t PrimalIndex,
    std::size_t Offset,
    std::size_t Count,
    std::size_t Stride,
    class Out
>
using ClarabelPrimalProjection = ClarabelProjection<
    ClarabelResultKind::Primal,
    PrimalIndex,
    Offset,
    Count,
    Stride,
    Out
>;

template <class... Projections>
struct ClarabelProjectionList {};

struct ClarabelAlwaysEnabled {
    template <class Context>
    STACKDSL_HOT static double read_flat(
        const Context&, std::size_t
    ) noexcept {
        return 1.0;
    }
};

template <
    class Program,
    class Parameters,
    class Projections,
    class Guard = ClarabelAlwaysEnabled
>
class ClarabelNode;

template <
    class Program,
    class... Bindings,
    class... Projections,
    class Guard
>
class ClarabelNode<
    Program,
    ClarabelParameterList<Bindings...>,
    ClarabelProjectionList<Projections...>,
    Guard
> {
    alignas(64) Program program_{};
    bool has_solution_{false};

    STACKDSL_HOT static bool same_value(double left, double right) noexcept {
        return std::bit_cast<std::uint64_t>(left)
            == std::bit_cast<std::uint64_t>(right);
    }

    template <class Binding, class Context>
    STACKDSL_HOT bool load_direct_parameter(const Context& ctx) {
        constexpr std::size_t index = Binding::index;
        using Source = typename Binding::source_type;
        static_assert(
            Source::shape::size == Program::template parameter_size<index>(),
            "CVXPY parameter and cpp_stream source sizes differ"
        );
        auto target = program_.template parameter_buffer<index>();
        bool changed = false;
        for (std::size_t offset = 0; offset < target.size(); ++offset) {
            if (!same_value(
                    static_cast<double>(target[offset]),
                    Source::read_flat(ctx, offset)
                )) {
                changed = true;
                break;
            }
        }
        if (!changed) return false;
        Source::load_contiguous(ctx, 0, target.size(), target.data());
        program_.template mark_parameter_dirty<index>();
        return true;
    }

    template <class Binding, class Context>
    STACKDSL_HOT bool load_feedback_parameter(const Context& ctx) {
        constexpr std::size_t index = Binding::index;
        using Initial = typename Binding::source_type;
        static_assert(
            Binding::count == Program::template parameter_size<index>(),
            "CVXPY feedback field and parameter sizes differ"
        );
        static_assert(
            Initial::shape::size == 1
                || Initial::shape::size
                    == Program::template parameter_size<index>(),
            "CVXPY feedback initializer must be scalar or parameter-shaped"
        );
        auto target = program_.template parameter_buffer<index>();
        bool changed = false;
        if (!has_solution_) {
            if constexpr (Initial::shape::size == 1) {
                const double value = Initial::read_flat(ctx, 0);
                for (std::size_t offset = 0; offset < target.size(); ++offset) {
                    if (!same_value(static_cast<double>(target[offset]), value)) {
                        changed = true;
                        break;
                    }
                }
                if (!changed) return false;
                for (auto& item : target) item = value;
            } else {
                for (std::size_t offset = 0; offset < target.size(); ++offset) {
                    if (!same_value(
                            static_cast<double>(target[offset]),
                            Initial::read_flat(ctx, offset)
                        )) {
                        changed = true;
                        break;
                    }
                }
                if (!changed) return false;
                Initial::load_contiguous(
                    ctx, 0, target.size(), target.data());
            }
        } else {
            const auto previous =
                program_.template primal<Binding::primal_index>();
            for (std::size_t offset = 0; offset < target.size(); ++offset) {
                const double value = previous[
                    Binding::offset + offset * Binding::stride];
                if (!same_value(static_cast<double>(target[offset]), value)) {
                    changed = true;
                    break;
                }
            }
            if (!changed) return false;
            for (std::size_t offset = 0; offset < target.size(); ++offset) {
                target[offset] = previous[
                    Binding::offset + offset * Binding::stride];
            }
        }
        program_.template mark_parameter_dirty<index>();
        return true;
    }

    template <class Binding, class Context>
    STACKDSL_HOT bool load_parameter(const Context& ctx) {
        if constexpr (Binding::feedback) {
            return load_feedback_parameter<Binding>(ctx);
        } else {
            return load_direct_parameter<Binding>(ctx);
        }
    }

    template <class Binding>
    STACKDSL_HOT void cache_feedback_result() noexcept {
        if constexpr (Binding::feedback) {
            // Some CVXPY inverse maps depend on parameter values. Retrieve the
            // carried primal immediately after its solve, before the following
            // row writes any new parameter buffers.
            (void)program_.template primal<Binding::primal_index>();
        }
    }

    template <class Projection, class Context>
    STACKDSL_HOT void project_nan(Context& ctx) noexcept {
        auto* STACKDSL_RESTRICT out =
            ctx.template write_ptr<typename Projection::output_type>();
        for (std::size_t index = 0; index < Projection::count; ++index) {
            out[index] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    template <class Projection, class Context>
    STACKDSL_HOT void project(Context& ctx) noexcept {
        auto* STACKDSL_RESTRICT out =
            ctx.template write_ptr<typename Projection::output_type>();
        if constexpr (Projection::kind == ClarabelResultKind::Info) {
            static_assert(Projection::count == 1);
            out[0] = program_.template info<Projection::source_index>();
        } else {
            const auto source = [&]() {
                if constexpr (
                    Projection::kind == ClarabelResultKind::Primal
                ) {
                    return program_.template primal<Projection::source_index>();
                } else if constexpr (
                    Projection::kind == ClarabelResultKind::Dual
                ) {
                    return program_.template dual<Projection::source_index>();
                } else {
                    static_assert(
                        Projection::kind == ClarabelResultKind::ConstraintValue
                    );
                    return program_.template constraint_value<
                        Projection::source_index
                    >();
                }
            }();
            for (std::size_t index = 0; index < Projection::count; ++index) {
                out[index] = source[
                    Projection::offset + index * Projection::stride
                ];
            }
        }
    }

public:
    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) {
        if constexpr (!std::is_same_v<Guard, ClarabelAlwaysEnabled>) {
            if (ctx.template read<Guard>(0) == 0.0) {
                (project_nan<Projections>(ctx), ...);
                return;
            }
        }
        bool changed = false;
        ((changed = load_parameter<Bindings>(ctx) || changed), ...);
        // Generated settings are fixed for this node, so bitwise-identical
        // parameters imply an identical problem and the cached result is valid.
        if (changed || !has_solution_) {
            program_.solve();
            has_solution_ = true;
        }
        (cache_feedback_result<Bindings>(), ...);
        (project<Projections>(ctx), ...);
    }
};

}  // namespace stackdsl
