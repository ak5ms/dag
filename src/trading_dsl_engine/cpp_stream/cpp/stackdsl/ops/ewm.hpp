#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "stackdsl/engine.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

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
    static constexpr double span = std::bit_cast<double>(SpanBits);
    static_assert(span > 0.0);
    static constexpr double alpha = 2.0 / (span + 1.0);
    static constexpr double old_weight_factor = 1.0 - alpha;
    static constexpr std::size_t state_size = Execution::state_size;

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
        if (all_initialized && all_finite) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
            for (std::size_t lane = begin; lane < end; ++lane) {
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
        for (std::size_t lane = begin; lane < end; ++lane) {
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
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
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
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
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

template <class In, class Out>
struct EwmBinding {
    using input_type = In;
    using output_type = Out;
};

struct EwmDiscardDst {};

template <class... Bindings>
struct EwmBindingList {};

template <std::size_t Component>
struct EwmComponentSrc {
    using value_type = double;
    static constexpr std::size_t feature_width = 1;
    static constexpr std::size_t component = Component;
};

template <
    class Source,
    class Out,
    std::size_t Stride,
    std::size_t Offset
>
struct EwmEpilogueBinding {
    using source_type = Source;
    using output_type = Out;
    static constexpr std::size_t stride = Stride;
    static constexpr std::size_t offset = Offset;
};

template <class... Bindings>
struct EwmEpilogueList {};

template <class Context, class Values>
class EwmEpilogueContext {
public:
    STACKDSL_HOT EwmEpilogueContext(
        const Context& context,
        const Values& values
    ) noexcept : context_(context), values_(values) {}

    template <class Source>
    STACKDSL_HOT auto read_native(std::size_t lane) const noexcept {
        if constexpr (requires { Source::component; }) {
            static_assert(Source::component < std::tuple_size_v<Values>);
            return values_[Source::component];
        } else {
            return context_.template read_native<Source>(lane);
        }
    }

    template <class Source>
    STACKDSL_HOT double read(std::size_t lane) const noexcept {
        return static_cast<double>(read_native<Source>(lane));
    }

private:
    const Context& context_;
    const Values& values_;
};

// One generic physical EWM node can advance any number of sibling scalar
// expression graphs.  The expression types expose the entire stateless graph to
// the optimizer, while this node shares pandas-style validity metadata for as
// long as the siblings observe the same missing-value pattern.  A state index
// splits permanently into per-component metadata only if those patterns diverge.
template <
    std::size_t N,
    std::uint64_t SpanBits,
    int MinPeriods,
    bool IgnoreNa,
    bool Adjust,
    class Execution,
    class Bindings,
    class Epilogues
>
struct EwmBundleNode;

template <
    std::size_t N,
    std::uint64_t SpanBits,
    int MinPeriods,
    bool IgnoreNa,
    bool Adjust,
    class Execution,
    class... Bindings,
    class... Epilogues
>
struct EwmBundleNode<
    N,
    SpanBits,
    MinPeriods,
    IgnoreNa,
    Adjust,
    Execution,
    EwmBindingList<Bindings...>,
    EwmEpilogueList<Epilogues...>
> {
    static_assert(sizeof...(Bindings) > 1);
    static constexpr std::size_t component_count = sizeof...(Bindings);
    static constexpr double span = std::bit_cast<double>(SpanBits);
    static_assert(span > 0.0);
    static constexpr double alpha = 2.0 / (span + 1.0);
    static constexpr double old_weight_factor = 1.0 - alpha;
    static constexpr std::size_t state_size = Execution::state_size;

    using ComponentValues = std::array<double, component_count>;
    using ComponentCounts = std::array<std::int64_t, component_count>;
    using ComponentFlags = std::array<std::uint8_t, component_count>;
    using ExpressionSources = typename expression_sources_for<
        typename Bindings::input_type...
    >::type;

    alignas(64) std::array<ComponentValues, state_size> value{};

    // Shared metadata is the normal path for composed moments because every raw
    // moment uses the same complete-case observation mask.
    alignas(64) std::array<double, state_size> shared_weight{};
    alignas(64) std::array<std::int64_t, state_size> shared_count{};
    alignas(64) std::array<std::uint8_t, state_size> shared_initialized{};
    alignas(64) std::array<std::uint8_t, state_size> shared_mode{};

    // Fixed-capacity fallback storage keeps arbitrary sibling EWMs semantically
    // exact without heap allocation when their observation patterns differ.
    alignas(64) std::array<ComponentValues, state_size> component_weight{};
    alignas(64) std::array<ComponentCounts, state_size> component_count_seen{};
    alignas(64) std::array<ComponentFlags, state_size> component_initialized{};
    bool all_initialized = false;

    void setup() noexcept {
        value.fill(ComponentValues{});
        shared_weight.fill(0.0);
        shared_count.fill(0);
        shared_initialized.fill(0);
        shared_mode.fill(1);
        component_weight.fill(ComponentValues{});
        component_count_seen.fill(ComponentCounts{});
        component_initialized.fill(ComponentFlags{});
        all_initialized = false;
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto outputs = output_pointers(ctx);
        if constexpr (MinPeriods <= 0 && IgnoreNa && !Adjust) {
            if constexpr (Execution::contiguous_lanes) {
                run_recursive_contiguous(ctx, outputs);
            } else {
                run_recursive_indexed(ctx, outputs);
            }
        } else {
            run_general(ctx, outputs);
        }
    }

private:
    template <class Context>
    STACKDSL_HOT static auto output_pointers(Context& ctx) noexcept {
        return std::array<double*, component_count>{
            binding_output_pointer<Bindings>(ctx)...
        };
    }

    template <class Binding, class Context>
    STACKDSL_HOT static double* binding_output_pointer(
        Context& ctx
    ) noexcept {
        if constexpr (
            std::is_same_v<typename Binding::output_type, EwmDiscardDst>
        ) {
            (void)ctx;
            return nullptr;
        } else {
            return ctx.template write_ptr<typename Binding::output_type>();
        }
    }

    template <class Context>
    STACKDSL_HOT static ComponentValues read_inputs(
        const Context& ctx, std::size_t lane
    ) noexcept {
        // Keeping all siblings in one pack expansion gives the compiler a single
        // scalar SSA region in which it can eliminate common subexpressions.
        ExpressionCacheContext<Context, ExpressionSources> cached(ctx);
        return ComponentValues{
            cached.template read<typename Bindings::input_type>(lane)...
        };
    }

    STACKDSL_HOT static bool observations_agree(
        const ComponentValues& input
    ) noexcept {
        const bool first = finite(input[0]);
        for (std::size_t component = 1; component < component_count; ++component) {
            if (finite(input[component]) != first) return false;
        }
        return true;
    }

    STACKDSL_HOT void split_metadata(std::size_t index) noexcept {
        for (std::size_t component = 0; component < component_count; ++component) {
            component_weight[index][component] = shared_weight[index];
            component_count_seen[index][component] = shared_count[index];
            component_initialized[index][component] = shared_initialized[index];
        }
        shared_mode[index] = 0;
    }

    STACKDSL_HOT bool component_is_initialized(
        std::size_t index, std::size_t component
    ) const noexcept {
        return shared_mode[index]
            ? shared_initialized[index] != 0
            : component_initialized[index][component] != 0;
    }

    STACKDSL_HOT ComponentValues visible_values(
        std::size_t index
    ) const noexcept {
        ComponentValues visible{};
        for (std::size_t component = 0; component < component_count; ++component) {
            const std::int64_t observations = shared_mode[index]
                ? shared_count[index]
                : component_count_seen[index][component];
            const bool enough = MinPeriods <= 0
                || observations >= MinPeriods;
            visible[component] =
                component_is_initialized(index, component) && enough
                ? value[index][component]
                : kNaN;
        }
        return visible;
    }

    template <class Epilogue, class Context>
    STACKDSL_HOT static void emit_epilogue(
        Context& ctx,
        const ComponentValues& visible,
        std::size_t lane
    ) noexcept {
        EwmEpilogueContext<Context, ComponentValues> epilogue_context(
            ctx, visible
        );
        using Source = typename Epilogue::source_type;
        using Sources = typename expression_sources_for<Source>::type;
        ExpressionCacheContext<decltype(epilogue_context), Sources> cached(
            epilogue_context
        );
        double* STACKDSL_RESTRICT output =
            ctx.template write_ptr<typename Epilogue::output_type>();
        output[lane * Epilogue::stride + Epilogue::offset] =
            cached.template read<Source>(lane);
    }

    template <class Context>
    STACKDSL_HOT void emit_epilogues(
        Context& ctx,
        std::size_t index,
        std::size_t lane
    ) const noexcept {
        if constexpr (sizeof...(Epilogues) != 0) {
            const ComponentValues visible = visible_values(index);
            (emit_epilogue<Epilogues>(ctx, visible, lane), ...);
        }
    }

    template <class Outputs>
    STACKDSL_HOT void update_recursive(
        std::size_t index,
        std::size_t lane,
        const ComponentValues& input,
        const Outputs& outputs
    ) noexcept {
        const bool agree = observations_agree(input);
        if (shared_mode[index] && agree) {
            const bool observation = finite(input[0]);
            if (observation) {
                if (shared_initialized[index]) {
                    for (std::size_t component = 0; component < component_count; ++component) {
                        value[index][component] = std::fma(
                            alpha,
                            input[component] - value[index][component],
                            value[index][component]
                        );
                    }
                } else {
                    value[index] = input;
                    shared_initialized[index] = 1;
                }
            }
            for (std::size_t component = 0; component < component_count; ++component) {
                if (outputs[component] != nullptr) {
                    outputs[component][lane] = shared_initialized[index]
                        ? value[index][component]
                        : kNaN;
                }
            }
            return;
        }

        if (shared_mode[index]) split_metadata(index);
        for (std::size_t component = 0; component < component_count; ++component) {
            if (finite(input[component])) {
                if (component_initialized[index][component]) {
                    value[index][component] = std::fma(
                        alpha,
                        input[component] - value[index][component],
                        value[index][component]
                    );
                } else {
                    value[index][component] = input[component];
                    component_initialized[index][component] = 1;
                }
            }
            if (outputs[component] != nullptr) {
                outputs[component][lane] = component_initialized[index][component]
                    ? value[index][component]
                    : kNaN;
            }
        }
    }

    template <class Context, class Outputs>
    STACKDSL_HOT void run_recursive_contiguous(
        Context& ctx, const Outputs& outputs
    ) noexcept {
        std::array<ComponentValues, N> input{};
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        bool all_finite = true;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = begin; lane < end; ++lane) {
            input[lane] = read_inputs(ctx, lane);
            for (double item : input[lane]) all_finite = all_finite && finite(item);
        }

        if (all_initialized && all_finite) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
            for (std::size_t lane = begin; lane < end; ++lane) {
                for (std::size_t component = 0; component < component_count; ++component) {
                    const double next = std::fma(
                        alpha,
                        input[lane][component] - value[lane][component],
                        value[lane][component]
                    );
                    value[lane][component] = next;
                    if (outputs[component] != nullptr) {
                        outputs[component][lane] = next;
                    }
                }
                emit_epilogues(ctx, lane, lane);
            }
            return;
        }

        bool now_all_initialized = true;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = begin; lane < end; ++lane) {
            update_recursive(lane, lane, input[lane], outputs);
            emit_epilogues(ctx, lane, lane);
            for (std::size_t component = 0; component < component_count; ++component) {
                now_all_initialized = now_all_initialized
                    && component_is_initialized(lane, component);
            }
        }
        all_initialized = now_all_initialized;
    }

    template <class Context, class Outputs>
    STACKDSL_HOT void run_recursive_indexed(
        Context& ctx, const Outputs& outputs
    ) noexcept {
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            update_recursive(
                index,
                lane,
                read_inputs(ctx, lane),
                outputs
            );
            emit_epilogues(ctx, index, lane);
        }
    }

    STACKDSL_HOT static void update_general_component(
        double x,
        double& current_value,
        double& current_weight,
        std::int64_t& current_count,
        std::uint8_t& initialized
    ) noexcept {
        const bool observation = finite(x);
        double old_weight = current_weight;
        if (initialized && (observation || !IgnoreNa)) {
            old_weight *= old_weight_factor;
        }
        if (observation) {
            if (initialized) {
                double new_weight = Adjust ? 1.0 : alpha;
                if constexpr (!Adjust) {
                    if (std::abs(alpha - 0.5) <= 1e-12) {
                        new_weight = 1.0 - old_weight;
                    }
                }
                if (current_value != x) {
                    current_value = (
                        old_weight * current_value + new_weight * x
                    ) / (old_weight + new_weight);
                }
                old_weight = Adjust ? old_weight + new_weight : 1.0;
            } else {
                current_value = x;
                initialized = 1;
                old_weight = 1.0;
            }
            ++current_count;
        }
        current_weight = old_weight;
    }

    template <class Context, class Outputs>
    STACKDSL_HOT void run_general(
        Context& ctx, const Outputs& outputs
    ) noexcept {
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const ComponentValues input = read_inputs(ctx, lane);
            const bool agree = observations_agree(input);
            if (shared_mode[index] && agree) {
                const bool observation = finite(input[0]);
                double old_weight = shared_weight[index];
                if (
                    shared_initialized[index]
                    && (observation || !IgnoreNa)
                ) {
                    old_weight *= old_weight_factor;
                }
                if (observation) {
                    if (shared_initialized[index]) {
                        double new_weight = Adjust ? 1.0 : alpha;
                        if constexpr (!Adjust) {
                            if (std::abs(alpha - 0.5) <= 1e-12) {
                                new_weight = 1.0 - old_weight;
                            }
                        }
                        for (std::size_t component = 0; component < component_count; ++component) {
                            if (value[index][component] != input[component]) {
                                value[index][component] = (
                                    old_weight * value[index][component]
                                    + new_weight * input[component]
                                ) / (old_weight + new_weight);
                            }
                        }
                        old_weight = Adjust ? old_weight + new_weight : 1.0;
                    } else {
                        value[index] = input;
                        shared_initialized[index] = 1;
                        old_weight = 1.0;
                    }
                    ++shared_count[index];
                }
                shared_weight[index] = old_weight;
                const bool enough = MinPeriods <= 0
                    || shared_count[index] >= MinPeriods;
                for (std::size_t component = 0; component < component_count; ++component) {
                    if (outputs[component] != nullptr) {
                        outputs[component][lane] =
                            shared_initialized[index] && enough
                            ? value[index][component]
                            : kNaN;
                    }
                }
                emit_epilogues(ctx, index, lane);
                continue;
            }

            if (shared_mode[index]) split_metadata(index);
            for (std::size_t component = 0; component < component_count; ++component) {
                update_general_component(
                    input[component],
                    value[index][component],
                    component_weight[index][component],
                    component_count_seen[index][component],
                    component_initialized[index][component]
                );
                const bool enough = MinPeriods <= 0
                    || component_count_seen[index][component] >= MinPeriods;
                if (outputs[component] != nullptr) {
                    outputs[component][lane] =
                        component_initialized[index][component] && enough
                        ? value[index][component]
                        : kNaN;
                }
            }
            emit_epilogues(ctx, index, lane);
        }
    }
};

}  // namespace stackdsl
