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
#include "stackdsl/ops/naryop.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template <class... Sources>
struct SourceList {};

template <std::size_t N, std::uint16_t... Groups>
struct StaticPartitions {
    static_assert(sizeof...(Groups) == N);
    static constexpr std::array<std::uint16_t, N> values{Groups...};
};

template <std::size_t Parts>
struct KeyBits {
    std::array<std::uint64_t, Parts> part{};
    friend STACKDSL_HOT bool operator==(const KeyBits& a, const KeyBits& b) noexcept { return a.part == b.part; }
};

STACKDSL_HOT std::uint64_t canonical_key_bits(double x) noexcept {
    if (std::isnan(x)) return 0x7ff8000000000000ULL;
    if (x == 0.0) return 0ULL;
    return std::bit_cast<std::uint64_t>(x);
}

STACKDSL_HOT std::uint64_t hash_mix(std::uint64_t value) noexcept {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    return value ^ (value >> 33);
}

template <std::size_t Parts>
STACKDSL_HOT std::uint64_t hash_key(const KeyBits<Parts>& key) noexcept {
    std::uint64_t hash = 0x9e3779b97f4a7c15ULL;
    for (std::size_t i = 0; i < Parts; ++i) {
        hash ^= hash_mix(key.part[i] + (i + 1) * 0x9e3779b97f4a7c15ULL);
        hash = (hash << 13) | (hash >> 51);
    }
    return hash_mix(hash);
}

consteval std::size_t next_power_of_two(std::size_t value) {
    std::size_t result = 1;
    while (result < value) result <<= 1;
    return result;
}

template <std::size_t Parts, std::size_t Capacity, std::size_t HashCapacityHint = 0>
struct FixedGroupTable {
    static_assert(Capacity > 0);
    static constexpr std::size_t requested_buckets = HashCapacityHint > 0 ? HashCapacityHint : Capacity * 2;
    static constexpr std::size_t bucket_count = next_power_of_two(requested_buckets < 8 ? 8 : requested_buckets);
    std::array<KeyBits<Parts>, Capacity> keys{};
    std::array<std::int32_t, bucket_count> buckets{};
    std::size_t count = 0;

    void setup() noexcept { buckets.fill(-1); count = 0; }

    STACKDSL_HOT int get_or_insert(const KeyBits<Parts>& key) noexcept {
        const std::size_t start = static_cast<std::size_t>(hash_key(key)) & (bucket_count - 1);
        for (std::size_t probe = 0; probe < bucket_count; ++probe) {
            const std::size_t bucket = (start + probe) & (bucket_count - 1);
            const std::int32_t slot = buckets[bucket];
            if (slot < 0) {
                if (count >= Capacity) return -1;
                const int inserted = static_cast<int>(count++);
                keys[static_cast<std::size_t>(inserted)] = key;
                buckets[bucket] = inserted;
                return inserted;
            }
            if (keys[static_cast<std::size_t>(slot)] == key) return slot;
        }
        return -1;
    }
};

template <std::size_t N, std::size_t Parts, std::size_t Capacity, std::size_t HashCapacityHint = 0>
struct HashGroupResolver {
    static constexpr std::size_t capacity = Capacity;
    FixedGroupTable<Parts, Capacity, HashCapacityHint> table{};
    std::array<KeyBits<Parts>, N> cached_keys{};
    std::array<std::uint16_t, N> cached_slots{};
    std::array<std::uint8_t, N> cache_valid{};

    void setup() noexcept { table.setup(); cache_valid.fill(0); }

    template <class Context, class... KeySources>
    bool resolve_all(Context& ctx, SourceList<KeySources...>, std::array<std::uint16_t, N>& slots) noexcept {
        static_assert(sizeof...(KeySources) == Parts);
        KeyBits<Parts> previous_key{};
        int previous_slot = -1;
        bool previous_valid = false;
        for (std::size_t lane = 0; lane < N; ++lane) {
            KeyBits<Parts> key{{canonical_key_bits(ctx.template read<KeySources>(lane))...}};
            int slot = -1;
            if (cache_valid[lane] && cached_keys[lane] == key) slot = cached_slots[lane];
            else if (previous_valid && previous_key == key) slot = previous_slot;
            else {
                slot = table.get_or_insert(key);
                if (slot < 0) return false;
            }
            cached_keys[lane] = key;
            cached_slots[lane] = static_cast<std::uint16_t>(slot);
            cache_valid[lane] = 1;
            slots[lane] = static_cast<std::uint16_t>(slot);
            previous_key = key;
            previous_slot = slot;
            previous_valid = true;
        }
        return true;
    }
};

template <std::size_t N>
struct NoKeyResolver {
    static constexpr std::size_t capacity = 1;
    void setup() noexcept {}
    template <class Context>
    bool resolve_all(Context&, SourceList<>, std::array<std::uint16_t, N>& slots) noexcept { slots.fill(0); return true; }
};

template <std::size_t N, std::size_t Cardinality, std::int64_t Offset = 0>
struct DenseGroupResolver {
    static constexpr std::size_t capacity = Cardinality + 1;
    void setup() noexcept {}
    template <class Context, class KeySource>
    bool resolve_all(Context& ctx, SourceList<KeySource>, std::array<std::uint16_t, N>& slots) noexcept {
        static_assert(capacity <= static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()));
        for (std::size_t lane = 0; lane < N; ++lane) {
            const double raw = ctx.template read<KeySource>(lane);
            if (std::isnan(raw)) { slots[lane] = static_cast<std::uint16_t>(Cardinality); continue; }
            if (!finite(raw)) return false;
            const double rounded = std::round(raw);
            if (std::abs(raw - rounded) > 1e-12) return false;
            const std::int64_t value = static_cast<std::int64_t>(rounded) - Offset;
            if (value < 0 || value >= static_cast<std::int64_t>(Cardinality)) return false;
            slots[lane] = static_cast<std::uint16_t>(value);
        }
        return true;
    }
};

template <std::size_t N, std::size_t Inputs, std::size_t ScratchSlots>
struct alignas(64) GroupRowContext {
    std::array<const double*, Inputs> inputs{};
    alignas(64) std::array<std::array<double, N>, ScratchSlots> scratch{};
    double* output = nullptr;
    const std::array<std::uint16_t, N>* group_slots = nullptr;
    const std::array<std::uint16_t, N>* partitions = nullptr;
    template <class Src> STACKDSL_HOT double read(std::size_t lane) const noexcept {
        if constexpr (requires { Src::input_index; }) return inputs[Src::input_index][lane];
        else if constexpr (requires { Src::slot_index; }) return scratch[Src::slot_index][lane];
        else return Src::value;
    }
    template <class Src> STACKDSL_HOT const double* read_ptr() const noexcept {
        static_assert(!is_literal_source_v<Src>);
        if constexpr (requires { Src::input_index; }) return inputs[Src::input_index];
        else return scratch[Src::slot_index].data();
    }
    template <class Dst> STACKDSL_HOT double* write_ptr() noexcept {
        if constexpr (std::is_same_v<Dst, OutputDst>) return output;
        else return scratch[Dst::slot_index].data();
    }
};

template <std::size_t N, std::size_t Capacity, class In, class Out>
struct GroupedCumsumNode {
    alignas(64) std::array<double, N * Capacity> value{};
    void setup() noexcept { value.fill(0.0); }
    template <class Context> STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        for (std::size_t lane = 0; lane < N; ++lane) {
            const double x = ctx.template read<In>(lane);
            const std::size_t index = static_cast<std::size_t>((*ctx.group_slots)[lane]) * N + lane;
            if (finite(x)) { value[index] += x; out[lane] = value[index]; }
            else out[lane] = kNaN;
        }
    }
};

template <std::size_t N, std::size_t Capacity, class In, class Out, std::uint64_t SpanBits, int MinPeriods, bool IgnoreNa, bool Adjust>
struct GroupedEwmNode {
    static constexpr double span = std::bit_cast<double>(SpanBits);
    static_assert(span > 0.0);
    static constexpr double alpha = 2.0 / (span + 1.0);
    static constexpr double old_weight_factor = 1.0 - alpha;
    static constexpr std::size_t state_size = N * Capacity;
    alignas(64) std::array<double, state_size> value{};
    alignas(64) std::array<double, state_size> weight{};
    alignas(64) std::array<std::int64_t, state_size> count{};
    alignas(64) std::array<std::uint8_t, state_size> initialized{};

    void setup() noexcept { value.fill(0.0); weight.fill(0.0); count.fill(0); initialized.fill(0); }
    template <class Context> STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        for (std::size_t lane = 0; lane < N; ++lane) {
            const std::size_t index = static_cast<std::size_t>((*ctx.group_slots)[lane]) * N + lane;
            const double x = ctx.template read<In>(lane);
            const bool observation = finite(x);
            double old_weight = weight[index];
            if (initialized[index] && (observation || !IgnoreNa)) old_weight *= old_weight_factor;
            if (observation) {
                if (initialized[index]) {
                    double new_weight = Adjust ? 1.0 : alpha;
                    if constexpr (!Adjust) if (std::abs(alpha - 0.5) <= 1e-12) new_weight = 1.0 - old_weight;
                    if (value[index] != x) value[index] = (old_weight * value[index] + new_weight * x) / (old_weight + new_weight);
                    old_weight = Adjust ? old_weight + new_weight : 1.0;
                } else { value[index] = x; initialized[index] = 1; old_weight = 1.0; }
                ++count[index];
            }
            weight[index] = old_weight;
            const bool enough = MinPeriods <= 0 || count[index] >= MinPeriods;
            out[lane] = initialized[index] && enough ? value[index] : kNaN;
        }
    }
};

struct GroupRankItem { std::uint32_t group; double value; std::uint16_t lane; };

template <std::size_t N, std::size_t Capacity, class In, class Out>
struct GroupedXsRankNode {
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
        std::array<std::uint8_t, N> valid{};
        for (std::size_t lane = 0; lane < N; ++lane) {
            values[lane] = ctx.template read<In>(lane);
            valid[lane] = static_cast<std::uint8_t>(finite(values[lane]));
            groups[lane] = static_cast<std::uint32_t>((*ctx.partitions)[lane]) * static_cast<std::uint32_t>(Capacity)
                + static_cast<std::uint32_t>((*ctx.group_slots)[lane]);
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
        std::array<GroupRankItem, N> items{};
        std::size_t count = 0;
        for (std::size_t lane = 0; lane < N; ++lane) {
            const double value = ctx.template read<In>(lane);
            if (!finite(value)) { out[lane] = kNaN; continue; }
            const std::uint32_t group = static_cast<std::uint32_t>((*ctx.partitions)[lane]) * static_cast<std::uint32_t>(Capacity) + static_cast<std::uint32_t>((*ctx.group_slots)[lane]);
            items[count++] = GroupRankItem{group, value, static_cast<std::uint16_t>(lane)};
        }
        std::sort(items.begin(), items.begin() + static_cast<std::ptrdiff_t>(count), [](const GroupRankItem& a, const GroupRankItem& b) { return a.group < b.group || (a.group == b.group && a.value < b.value); });
        std::size_t group_start = 0;
        while (group_start < count) {
            std::size_t group_end = group_start + 1;
            while (group_end < count && items[group_end].group == items[group_start].group) ++group_end;
            const std::size_t group_count = group_end - group_start;
            std::size_t tie_start = group_start;
            while (tie_start < group_end) {
                std::size_t upper = tie_start + 1;
                while (upper < group_end && items[upper].value == items[tie_start].value) ++upper;
                const std::size_t upper_rank = upper - group_start;
                const double score = scores.get(group_count, upper_rank - 1);
                for (std::size_t pos = tie_start; pos < upper; ++pos) out[items[pos].lane] = score;
                tie_start = upper;
            }
            group_start = group_end;
        }
    }
};

template <std::size_t N, class Resolver, class Partitions, class InnerPlan, class Out, class KeyList, class FeedList>
struct GroupByNode;

template <std::size_t N, class Resolver, class Partitions, class InnerPlan, class Out, class... KeySources, class... FeedSources>
struct GroupByNode<N, Resolver, Partitions, InnerPlan, Out, SourceList<KeySources...>, SourceList<FeedSources...>> {
    Resolver resolver{};
    InnerPlan inner{};
    std::array<std::uint16_t, N> group_slots{};
    void setup() noexcept { resolver.setup(); inner.setup(); }
    template <class Context> STACKDSL_HOT bool on_data_checked(Context& ctx) noexcept {
        if (!resolver.resolve_all(ctx, SourceList<KeySources...>{}, group_slots)) return false;
        std::array<const double*, sizeof...(FeedSources)> feeds{ctx.template read_ptr<FeedSources>()...};
        inner.on_data(feeds, group_slots, Partitions::values, ctx.template write_ptr<Out>());
        return true;
    }
    template <class Context> STACKDSL_HOT void on_data(Context& ctx) noexcept { (void)on_data_checked(ctx); }
};

}  // namespace stackdsl
