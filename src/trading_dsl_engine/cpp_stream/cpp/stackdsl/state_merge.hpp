#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "stackdsl/ops/reduction.hpp"

namespace stackdsl {

// The generated runner knows only how work was partitioned. It deliberately
// does not know how a particular operator combines mutable state. These two
// small value types carry the partition geometry into a compile-time
// customization point implemented below.
struct RowShardStateMerge {};

struct LaneShardStateMerge {
    std::size_t lane_begin;
    std::size_t lane_end;
};

namespace state_merge_detail {

// Stateless stages and stateful stages without a valid merge law are ignored.
// The parallel planner is responsible for selecting a strategy only when every
// required stage is mergeable. Keeping the fallback a no-op lets one generic
// tuple walk cover the complete generated plan without a Jinja type switch.
template <class Stage, class Partition>
STACKDSL_HOT void merge_builtin_state(
    Stage&,
    const Stage&,
    const Partition&
) noexcept {}

// ReductionState already owns the numerically correct block-combination law
// through merge_block(). This helper only selects which output cells belong to
// the worker being merged. In particular, std uses parallel Welford state rather
// than combining already-rounded standard deviations.
template <
    class Policy,
    std::size_t Size,
    std::size_t Ddof,
    bool IgnoreNa
>
STACKDSL_HOT void merge_reduction_state_range(
    ReductionState<Policy, Size, Ddof, IgnoreNa>& target,
    const ReductionState<Policy, Size, Ddof, IgnoreNa>& source,
    std::size_t begin,
    std::size_t end
) noexcept {
    const std::size_t bounded_end = std::min(end, Size);
    for (std::size_t index = std::min(begin, bounded_end);
         index < bounded_end;
         ++index) {
        // IgnoreNa reductions never consult invalid[]. Avoid reading that array
        // in this specialization because reset() intentionally does not touch it.
        const bool source_invalid = [&] {
            if constexpr (IgnoreNa) return false;
            else return source.invalid[index];
        }();
        target.merge_block(
            index,
            source.total[index],
            source.mean[index],
            source.m2[index],
            source.count[index],
            source_invalid
        );
    }
}

// One temporal reduction has one accumulator array. Row shards contribute to
// every output cell, while lane shards contribute only to the cells owned by
// their leading-axis interval.
template <
    class Tensor,
    class Out,
    class Axes,
    class Policy,
    std::size_t Ddof,
    bool IgnoreNa,
    bool Temporal,
    class Execution
>
STACKDSL_HOT void merge_builtin_state(
    ReductionNode<
        Tensor,
        Out,
        Axes,
        Policy,
        Ddof,
        IgnoreNa,
        Temporal,
        Execution
    >& target,
    const ReductionNode<
        Tensor,
        Out,
        Axes,
        Policy,
        Ddof,
        IgnoreNa,
        Temporal,
        Execution
    >& source,
    const RowShardStateMerge&
) noexcept {
    if constexpr (Temporal) {
        using Node = ReductionNode<
            Tensor,
            Out,
            Axes,
            Policy,
            Ddof,
            IgnoreNa,
            Temporal,
            Execution
        >;
        merge_reduction_state_range(
            target.state,
            source.state,
            0,
            Node::output_size
        );
    }
}

template <
    class Tensor,
    class Out,
    class Axes,
    class Policy,
    std::size_t Ddof,
    bool IgnoreNa,
    bool Temporal,
    class Execution
>
STACKDSL_HOT void merge_builtin_state(
    ReductionNode<
        Tensor,
        Out,
        Axes,
        Policy,
        Ddof,
        IgnoreNa,
        Temporal,
        Execution
    >& target,
    const ReductionNode<
        Tensor,
        Out,
        Axes,
        Policy,
        Ddof,
        IgnoreNa,
        Temporal,
        Execution
    >& source,
    const LaneShardStateMerge& partition
) noexcept {
    using Node = ReductionNode<
        Tensor,
        Out,
        Axes,
        Policy,
        Ddof,
        IgnoreNa,
        Temporal,
        Execution
    >;
    if constexpr (Temporal && Node::retains_leading_axis) {
        merge_reduction_state_range(
            target.state,
            source.state,
            partition.lane_begin * Node::output_lane_width,
            partition.lane_end * Node::output_lane_width
        );
    }
}

// A reduction bundle has the same merge law as a single reduction, repeated
// for each compile-time component. The component loop is ordinary C++, not
// generated code, so adding another bundled reduction does not enlarge Jinja.
template <
    class Axes,
    class Policy,
    std::size_t Ddof,
    bool IgnoreNa,
    bool Temporal,
    class Execution,
    class... Bindings
>
STACKDSL_HOT void merge_builtin_state(
    ReductionBundleNode<
        Axes,
        Policy,
        Ddof,
        IgnoreNa,
        Temporal,
        Execution,
        Bindings...
    >& target,
    const ReductionBundleNode<
        Axes,
        Policy,
        Ddof,
        IgnoreNa,
        Temporal,
        Execution,
        Bindings...
    >& source,
    const RowShardStateMerge&
) noexcept {
    using Node = ReductionBundleNode<
        Axes,
        Policy,
        Ddof,
        IgnoreNa,
        Temporal,
        Execution,
        Bindings...
    >;
    if constexpr (Temporal) {
        for (std::size_t component = 0;
             component < Node::component_count;
             ++component) {
            merge_reduction_state_range(
                target.state[component],
                source.state[component],
                0,
                Node::output_size
            );
        }
    }
}

template <
    class Axes,
    class Policy,
    std::size_t Ddof,
    bool IgnoreNa,
    bool Temporal,
    class Execution,
    class... Bindings
>
STACKDSL_HOT void merge_builtin_state(
    ReductionBundleNode<
        Axes,
        Policy,
        Ddof,
        IgnoreNa,
        Temporal,
        Execution,
        Bindings...
    >& target,
    const ReductionBundleNode<
        Axes,
        Policy,
        Ddof,
        IgnoreNa,
        Temporal,
        Execution,
        Bindings...
    >& source,
    const LaneShardStateMerge& partition
) noexcept {
    using Node = ReductionBundleNode<
        Axes,
        Policy,
        Ddof,
        IgnoreNa,
        Temporal,
        Execution,
        Bindings...
    >;
    if constexpr (Temporal && Node::retains_leading_axis) {
        for (std::size_t component = 0;
             component < Node::component_count;
             ++component) {
            merge_reduction_state_range(
                target.state[component],
                source.state[component],
                partition.lane_begin * Node::output_lane_width,
                partition.lane_end * Node::output_lane_width
            );
        }
    }
}

// emit(last) is not an arithmetic reduction. For row shards, later shards own
// later rows, so merging workers in row order intentionally overwrites the
// earlier value. For lane shards, each worker owns a disjoint slice of the final
// tensor and only that slice is copied into the owner runner.
template <class Tensor, class Out>
STACKDSL_HOT void merge_builtin_state(
    EmitLastNode<Tensor, Out>& target,
    const EmitLastNode<Tensor, Out>& source,
    const RowShardStateMerge&
) noexcept {
    if (!source.seen) return;
    target.value = source.value;
    target.seen = true;
}

template <class Tensor, class Out>
STACKDSL_HOT void merge_builtin_state(
    EmitLastNode<Tensor, Out>& target,
    const EmitLastNode<Tensor, Out>& source,
    const LaneShardStateMerge& partition
) noexcept {
    if (!source.seen) return;
    using Shape = typename Tensor::shape;
    if constexpr (Shape::rank > 0) {
        constexpr std::size_t leading_extent = Shape::dims[0];
        constexpr std::size_t lane_width = Shape::size / leading_extent;
        const std::size_t begin =
            std::min(partition.lane_begin, leading_extent) * lane_width;
        const std::size_t end =
            std::min(partition.lane_end, leading_extent) * lane_width;
        std::copy(
            source.value.begin() + static_cast<std::ptrdiff_t>(begin),
            source.value.begin() + static_cast<std::ptrdiff_t>(end),
            target.value.begin() + static_cast<std::ptrdiff_t>(begin)
        );
        target.seen = true;
    }
}

}  // namespace state_merge_detail

// Future stateful operators can keep their merge law beside the operator by
// exposing `merge_state_from(source, partition)`. The generic runner will use
// that member automatically. Existing reduction and emit nodes are adapted by
// the compile-time overloads above; no runtime virtual dispatch is introduced.
template <class Stage, class Partition>
STACKDSL_HOT void merge_stage_state(
    Stage& target,
    const Stage& source,
    const Partition& partition
) noexcept {
    if constexpr (requires {
        { target.merge_state_from(source, partition) } noexcept;
    }) {
        target.merge_state_from(source, partition);
    } else {
        state_merge_detail::merge_builtin_state(
            target,
            source,
            partition
        );
    }
}

namespace state_merge_detail {

template <
    class TargetTuple,
    class SourceTuple,
    class Partition,
    std::size_t... Indexes
>
STACKDSL_HOT void merge_stage_tuples_impl(
    TargetTuple target,
    SourceTuple source,
    const Partition& partition,
    std::index_sequence<Indexes...>
) noexcept {
    (merge_stage_state(
        std::get<Indexes>(target),
        std::get<Indexes>(source),
        partition
    ), ...);
}

}  // namespace state_merge_detail

// The generated runner supplies tuples of references to all physical stages.
// The fold expression is resolved and inlined at compile time, so the generated
// source contains one stable orchestration call regardless of operator mix.
template <class TargetTuple, class SourceTuple, class Partition>
STACKDSL_HOT void merge_stage_states(
    TargetTuple target,
    SourceTuple source,
    const Partition& partition
) noexcept {
    using Target = std::remove_cvref_t<TargetTuple>;
    using Source = std::remove_cvref_t<SourceTuple>;
    static_assert(
        std::tuple_size_v<Target> == std::tuple_size_v<Source>,
        "worker runners must have identical physical stage layouts"
    );
    state_merge_detail::merge_stage_tuples_impl(
        target,
        source,
        partition,
        std::make_index_sequence<std::tuple_size_v<Target>>{}
    );
}

}  // namespace stackdsl
