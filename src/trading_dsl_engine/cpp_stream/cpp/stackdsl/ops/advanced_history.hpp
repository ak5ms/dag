#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "stackdsl/engine.hpp"
#include "stackdsl/ops/order_tree.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template <std::uint64_t... Bits>
struct DoubleList {
    static constexpr std::size_t size = sizeof...(Bits);
    static constexpr std::array<double, size> values{
        std::bit_cast<double>(Bits)...
    };
};

template <
    std::size_t N,
    class In,
    class Out,
    class Execution = DirectExecution<N>
>
struct PeriodsSinceChangeNode {
    alignas(64) std::array<double, Execution::state_size> last{};
    alignas(64) std::array<std::uint64_t, Execution::state_size> periods{};
    alignas(64) std::array<std::uint8_t, Execution::state_size> initialized{};

    void setup() noexcept {
        last.fill(kNaN);
        periods.fill(0);
        initialized.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const double value = ctx.template read<In>(lane);
            if (!initialized[index]) {
                initialized[index] = 1;
                periods[index] = 0;
            } else {
                const bool equal = (std::isnan(value) && std::isnan(last[index]))
                    || value == last[index];
                periods[index] = equal ? periods[index] + 1 : 0;
            }
            last[index] = value;
            out[lane] = static_cast<double>(periods[index]);
        }
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    std::uint64_t ThresholdBits,
    bool Relative,
    bool MoveByThreshold,
    class Execution = DirectExecution<N>
>
struct HumpNode {
    static constexpr double threshold = std::bit_cast<double>(ThresholdBits);
    alignas(64) std::array<double, Execution::state_size> value{};
    alignas(64) std::array<std::uint8_t, Execution::state_size> initialized{};

    void setup() noexcept {
        value.fill(kNaN);
        initialized.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const double incoming = ctx.template read<In>(lane);
            if (!initialized[index]) {
                if (finite(incoming)) {
                    value[index] = incoming;
                    initialized[index] = 1;
                }
                out[lane] = initialized[index] ? value[index] : kNaN;
                continue;
            }
            if (!finite(incoming)) {
                out[lane] = value[index];
                continue;
            }
            const double difference = incoming - value[index];
            const double limit = Relative
                ? threshold * std::abs(incoming + value[index])
                : threshold;
            if (std::abs(difference) > limit) {
                value[index] = MoveByThreshold
                    ? value[index] + std::copysign(limit, difference)
                    : incoming;
            }
            out[lane] = value[index];
        }
    }
};

template <
    std::size_t N,
    class Trigger,
    class Alpha,
    class Exit,
    class Out,
    class Execution = DirectExecution<N>
>
struct TradeWhenNode {
    alignas(64) std::array<double, Execution::state_size> value{};
    alignas(64) std::array<std::uint8_t, Execution::state_size> initialized{};

    void setup() noexcept {
        value.fill(kNaN);
        initialized.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const double exit = ctx.template read<Exit>(lane);
            const double trigger = ctx.template read<Trigger>(lane);
            if (finite(exit) && exit > 0.0) {
                value[index] = kNaN;
                initialized[index] = 1;
            } else if (finite(trigger) && trigger > 0.0) {
                value[index] = ctx.template read<Alpha>(lane);
                initialized[index] = 1;
            }
            out[lane] = initialized[index] ? value[index] : kNaN;
        }
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    class FeedForward,
    class Recursive,
    class Execution = DirectExecution<N>
>
struct LinearFilterNode {
    static constexpr std::size_t H = FeedForward::size;
    static constexpr std::size_t T = Recursive::size;
    static_assert(H > 0);
    alignas(64) std::array<double, Execution::state_size * H> input{};
    alignas(64) std::array<double, Execution::state_size * T> output{};
    alignas(64) std::array<std::uint64_t, Execution::state_size> step{};

    void setup() noexcept {
        input.fill(kNaN);
        output.fill(kNaN);
        step.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const std::uint64_t current_step = step[index];
            input[index * H + current_step % H] = ctx.template read<In>(lane);
            double result = 0.0;
            bool valid = true;
            for (std::size_t lag = 0; lag < H; ++lag) {
                const double weight = FeedForward::values[lag];
                if (weight == 0.0 || current_step < lag) continue;
                const double value = input[
                    index * H + (current_step + H - lag) % H
                ];
                if (!finite(value)) valid = false;
                else result = std::fma(weight, value, result);
            }
            if constexpr (T > 0) {
                for (std::size_t lag = 0; lag < T; ++lag) {
                    const double weight = Recursive::values[lag];
                    if (weight == 0.0 || current_step <= lag) continue;
                    const double value = output[
                        index * T + (current_step + T - lag - 1) % T
                    ];
                    if (!finite(value)) valid = false;
                    else result = std::fma(weight, value, result);
                }
                output[index * T + current_step % T] = valid ? result : kNaN;
            }
            ++step[index];
            out[lane] = valid ? result : kNaN;
        }
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    std::size_t Periods,
    std::size_t MinPeriods,
    class Execution = DirectExecution<N>
>
struct RollingProductNode {
    static constexpr std::size_t StateSize = Execution::state_size;
    alignas(64) std::array<double, StateSize * Periods> ring{};
    alignas(64) std::array<double, StateSize> log_sum{};
    alignas(64) std::array<std::uint32_t, StateSize> count{};
    alignas(64) std::array<std::uint32_t, StateSize> zeros{};
    alignas(64) std::array<std::uint8_t, StateSize> negative{};
    alignas(64) std::array<std::uint64_t, StateSize> step{};

    void setup() noexcept {
        ring.fill(kNaN);
        log_sum.fill(0.0);
        count.fill(0);
        zeros.fill(0);
        negative.fill(0);
        step.fill(0);
    }

    STACKDSL_HOT void update(std::size_t index, double value, bool add) noexcept {
        const int direction = add ? 1 : -1;
        count[index] = static_cast<std::uint32_t>(
            static_cast<int>(count[index]) + direction
        );
        if (value == 0.0) {
            zeros[index] = static_cast<std::uint32_t>(
                static_cast<int>(zeros[index]) + direction
            );
        } else {
            log_sum[index] += static_cast<double>(direction) * std::log(std::abs(value));
            if (value < 0.0) negative[index] ^= 1U;
        }
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const std::size_t position = index * Periods + step[index] % Periods;
            const double outgoing = ring[position];
            if (finite(outgoing)) update(index, outgoing, false);
            const double incoming = ctx.template read<In>(lane);
            ring[position] = incoming;
            if (finite(incoming)) update(index, incoming, true);
            ++step[index];
            if (count[index] < MinPeriods || count[index] == 0) out[lane] = kNaN;
            else if (zeros[index] != 0) out[lane] = 0.0;
            else out[lane] = (negative[index] ? -1.0 : 1.0) * std::exp(log_sum[index]);
        }
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    std::size_t Periods,
    std::size_t MinPeriods,
    std::size_t K,
    bool IgnoreZero,
    class Execution = DirectExecution<N>
>
struct RollingKthNode {
    static constexpr std::size_t StateSize = Execution::state_size;
    alignas(64) std::array<FixedRecencyList<Periods>, StateSize> recency{};
    alignas(64) std::array<std::uint64_t, StateSize> step{};

    void setup() noexcept {
        for (auto& item : recency) item.setup();
        step.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const std::uint64_t current = step[index];
            const auto position = static_cast<
                typename FixedRecencyList<Periods>::Index
            >(current % Periods);
            auto& history = recency[index];
            history.erase(position);
            const double incoming = ctx.template read<In>(lane);
            if (finite(incoming) && (!IgnoreZero || incoming != 0.0)) {
                history.insert_newest(position, incoming);
            }
            ++step[index];
            out[lane] = history.count >= MinPeriods && history.count >= K
                ? history.kth_newest(K - 1)
                : kNaN;
        }
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    std::size_t Periods,
    class Execution = DirectExecution<N>
>
struct RollingPrevDiffNode {
    static constexpr std::size_t StateSize = Execution::state_size;
    static constexpr std::size_t SwitchScan = 8;
    alignas(64) std::array<double, StateSize * Periods> ring{};
    alignas(64) std::array<double, StateSize> run_value{};
    alignas(64) std::array<double, StateSize> run_candidate{};
    alignas(64) std::array<std::uint64_t, StateSize> run_candidate_step{};
    alignas(64) std::array<std::uint8_t, StateSize> run_mode{};
    alignas(64) std::array<std::uint64_t, StateSize> step{};

    void setup() noexcept {
        ring.fill(kNaN);
        run_value.fill(kNaN);
        run_candidate.fill(kNaN);
        run_candidate_step.fill(0);
        run_mode.fill(0);
        step.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const std::uint64_t current = step[index];
            const double incoming = ctx.template read<In>(lane);
            const std::size_t position = current % Periods;
            const std::size_t base = index * Periods;
            ring[base + position] = incoming;
            ++step[index];
            out[lane] = kNaN;

            if (!finite(incoming)) [[unlikely]] {
                run_mode[index] = 0;
                continue;
            }

            if (run_mode[index]) [[unlikely]] {
                if (incoming != run_value[index]) {
                    // The immediately preceding finite run is necessarily the
                    // nearest different observation.
                    out[lane] = run_value[index];
                    run_mode[index] = 0;
                } else if (
                    finite(run_candidate[index])
                    && current - run_candidate_step[index] < Periods
                ) {
                    out[lane] = run_candidate[index];
                }
                continue;
            }

            if constexpr (Periods > 1) {
                if (current != 0) {
                    const double previous = ring[
                        base + (current + Periods - 1) % Periods
                    ];
                    if (finite(previous) && previous != incoming) {
                        out[lane] = previous;
                        continue;
                    }
                }
            }

            const std::size_t available = std::min<std::uint64_t>(
                current + 1, Periods
            );
            const std::size_t scan = std::min(
                available - 1, SwitchScan
            );
            for (std::size_t age = 2; age <= scan; ++age) {
                const double value = ring[
                    base + (current + Periods - age) % Periods
                ];
                if (finite(value) && value != incoming) {
                    out[lane] = value;
                    break;
                }
            }
            if (finite(out[lane]) || available - 1 <= SwitchScan) continue;

            // A repeated/NaN-heavy prefix made the bounded fast scan expensive.
            // Find its one live predecessor once, then answer the rest of the run
            // in O(1) until the value changes.
            run_value[index] = incoming;
            run_candidate[index] = kNaN;
            for (std::size_t age = SwitchScan + 1; age < available; ++age) {
                const double value = ring[
                    base + (current + Periods - age) % Periods
                ];
                if (finite(value) && value != incoming) {
                    run_candidate[index] = value;
                    run_candidate_step[index] = current - age;
                    out[lane] = value;
                    break;
                }
            }
            run_mode[index] = 1;
        }
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    std::size_t Periods,
    std::size_t MinPeriods,
    class Execution = DirectExecution<N>
>
struct RollingLinearDecayNode {
    static constexpr std::size_t StateSize = Execution::state_size;
    alignas(64) std::array<double, StateSize * Periods> ring{};
    alignas(64) std::array<double, StateSize> sum{};
    alignas(64) std::array<double, StateSize> weighted{};
    alignas(64) std::array<double, StateSize> weight_sum{};
    alignas(64) std::array<std::uint32_t, StateSize> count{};
    alignas(64) std::array<std::uint64_t, StateSize> step{};

    void setup() noexcept {
        ring.fill(kNaN);
        sum.fill(0.0);
        weighted.fill(0.0);
        weight_sum.fill(0.0);
        count.fill(0);
        step.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const std::uint64_t current = step[index];
            const std::size_t position = index * Periods + current % Periods;
            weighted[index] -= sum[index];
            weight_sum[index] -= static_cast<double>(count[index]);
            const double outgoing = ring[position];
            if (finite(outgoing)) {
                sum[index] -= outgoing;
                --count[index];
            }
            const double incoming = ctx.template read<In>(lane);
            ring[position] = incoming;
            if (finite(incoming)) {
                sum[index] += incoming;
                weighted[index] = std::fma(
                    static_cast<double>(Periods), incoming, weighted[index]
                );
                weight_sum[index] += static_cast<double>(Periods);
                ++count[index];
            }
            ++step[index];
            out[lane] = count[index] >= MinPeriods && weight_sum[index] > 0.0
                ? weighted[index] / weight_sum[index]
                : kNaN;
        }
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    std::size_t Periods,
    std::size_t MinPeriods,
    std::size_t Buckets,
    class Execution = DirectExecution<N>
>
struct RollingEntropyNode {
    static constexpr std::size_t StateSize = Execution::state_size;
    static constexpr bool UseTree = Periods >= 64;
    using Ring = std::conditional_t<
        UseTree,
        std::array<double, 1>,
        std::array<double, StateSize * Periods>
    >;
    using Tree = std::conditional_t<
        UseTree, FixedOrderTree<Periods>, EmptyOrderTree
    >;
    alignas(64) Ring ring{};
    alignas(64) std::array<Tree, StateSize> tree{};
    alignas(64) std::array<std::uint64_t, StateSize> step{};

    void setup() noexcept {
        ring.fill(kNaN);
        for (auto& item : tree) item.setup();
        step.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const std::uint64_t current = step[index];
            const std::size_t ring_position = current % Periods;
            const double incoming = ctx.template read<In>(lane);
            if constexpr (UseTree) {
                tree[index].replace(
                    static_cast<typename FixedOrderTree<Periods>::Index>(
                        ring_position
                    ),
                    incoming
                );
            } else {
                ring[index * Periods + ring_position] = incoming;
            }
            ++step[index];
            double minimum = std::numeric_limits<double>::infinity();
            double maximum = -std::numeric_limits<double>::infinity();
            std::size_t count = 0;
            if constexpr (UseTree) {
                count = tree[index].size();
                if (count != 0) {
                    minimum = tree[index].minimum();
                    maximum = tree[index].maximum();
                }
            } else {
                for (std::size_t position = 0; position < Periods; ++position) {
                    const double value = ring[index * Periods + position];
                    if (!finite(value)) continue;
                    minimum = std::min(minimum, value);
                    maximum = std::max(maximum, value);
                    ++count;
                }
            }
            if (count < MinPeriods || count == 0) {
                out[lane] = kNaN;
                continue;
            }
            if (minimum == maximum) {
                out[lane] = 0.0;
                continue;
            }
            std::array<std::uint32_t, Buckets> counts{};
            const double scale = static_cast<double>(Buckets) / (maximum - minimum);
            auto add_bucket = [&](double value) {
                const std::size_t bucket = std::min<std::size_t>(
                    Buckets - 1,
                    static_cast<std::size_t>((value - minimum) * scale)
                );
                ++counts[bucket];
            };
            if constexpr (UseTree) {
                tree[index].for_each_active(add_bucket);
            } else {
                for (std::size_t position = 0; position < Periods; ++position) {
                    const double value = ring[index * Periods + position];
                    if (finite(value)) add_bucket(value);
                }
            }
            double entropy = 0.0;
            for (std::uint32_t bucket_count : counts) {
                if (bucket_count == 0) continue;
                const double probability = static_cast<double>(bucket_count)
                    / static_cast<double>(count);
                entropy -= probability * std::log(probability);
            }
            out[lane] = entropy;
        }
    }
};

}  // namespace stackdsl
