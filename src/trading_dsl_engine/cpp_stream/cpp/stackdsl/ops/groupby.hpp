#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "stackdsl/engine.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template <class... Sources>
struct SourceList {};

template <class... Keys>
struct KeyList {};

STACKDSL_HOT std::uint64_t canonical_key_bits(double x) noexcept;

template <class Source, std::size_t NumKeys = 0, std::int64_t Offset = 0, bool RowScalar = false>
struct KeySpec {
    using source = Source;
    static constexpr std::size_t num_keys = NumKeys;
    static constexpr std::int64_t offset = Offset;
    static constexpr bool row_scalar = RowScalar;

    template <class Context>
    STACKDSL_HOT static auto read_native(const Context& ctx, std::size_t lane) noexcept {
        return ctx.template read_native<Source>(RowScalar ? 0 : lane);
    }

    template <class Context>
    STACKDSL_HOT static std::uint64_t canonical_bits(const Context& ctx, std::size_t lane) noexcept {
        const auto raw = read_native(ctx, lane);
        using Value = std::remove_cv_t<decltype(raw)>;
        if constexpr (std::is_integral_v<Value>) return static_cast<std::uint64_t>(raw);
        else return canonical_key_bits(static_cast<double>(raw));
    }
};

template <std::size_t N, std::uint16_t... Groups>
struct StaticPartitions {
    static_assert(sizeof...(Groups) == N);
    static constexpr std::array<std::uint16_t, N> values{Groups...};
};

template <std::size_t Parts>
struct KeyBits {
    std::array<std::uint64_t, Parts> part{};
    friend STACKDSL_HOT bool operator==(const KeyBits& a, const KeyBits& b) noexcept {
        return a.part == b.part;
    }
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

template <std::size_t N, std::size_t Capacity, std::size_t HashCapacityHint, class... Keys>
struct HashGroupResolver {
    static constexpr std::size_t capacity = Capacity;
    static constexpr std::size_t parts = sizeof...(Keys);
    FixedGroupTable<parts, Capacity, HashCapacityHint> table{};
    std::array<KeyBits<parts>, N> cached_keys{};
    std::array<std::uint16_t, N> cached_slots{};
    std::array<std::uint8_t, N> cache_valid{};

    void setup() noexcept { table.setup(); cache_valid.fill(0); }

    template <class Context>
    STACKDSL_HOT static KeyBits<parts> make_key(const Context& ctx, std::size_t lane) noexcept {
        return KeyBits<parts>{{Keys::canonical_bits(ctx, lane)...}};
    }

    template <class Context>
    bool resolve_range(
        Context& ctx,
        KeyList<Keys...>,
        std::array<std::uint16_t, N>& slots,
        std::size_t begin,
        std::size_t end
    ) noexcept {
        if constexpr ((Keys::row_scalar && ...)) {
            const int slot = table.get_or_insert(make_key(ctx, 0));
            if (slot < 0) return false;
            for (std::size_t lane = begin; lane < end; ++lane) {
                slots[lane] = static_cast<std::uint16_t>(slot);
            }
            return true;
        }
        KeyBits<parts> previous_key{};
        int previous_slot = -1;
        bool previous_valid = false;
        for (std::size_t lane = begin; lane < end; ++lane) {
            const auto key = make_key(ctx, lane);
            int slot;
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

template <class... Keys>
consteval std::size_t dense_tuple_capacity() {
    std::size_t result = 1;
    ((result *= Keys::num_keys + 1), ...);
    return result;
}

template <std::size_t N, class... Keys>
struct DenseTupleGroupResolver {
    static constexpr std::size_t capacity = dense_tuple_capacity<Keys...>();
    static_assert(capacity <= static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()));
    void setup() noexcept {}

    template <class Key, class Context>
    STACKDSL_HOT static bool append_key(const Context& ctx, std::size_t lane, std::size_t& slot) noexcept {
        const auto raw = Key::read_native(ctx, lane);
        using Value = std::remove_cv_t<decltype(raw)>;
        std::size_t digit = 0;
        if constexpr (std::is_integral_v<Value>) {
            const __int128 delta = static_cast<__int128>(raw) - static_cast<__int128>(Key::offset);
            if (delta < 0 || delta >= static_cast<__int128>(Key::num_keys)) return false;
            digit = static_cast<std::size_t>(delta);
        } else {
            const double value = static_cast<double>(raw);
            if (std::isnan(value)) digit = Key::num_keys;
            else {
                if (!finite(value)) return false;
                const double rounded = std::round(value);
                if (std::abs(value - rounded) > 1e-12) return false;
                const long double delta = static_cast<long double>(rounded) - static_cast<long double>(Key::offset);
                if (delta < 0.0L || delta >= static_cast<long double>(Key::num_keys)) return false;
                digit = static_cast<std::size_t>(delta);
            }
        }
        slot = slot * (Key::num_keys + 1) + digit;
        return true;
    }

    template <class Context>
    STACKDSL_HOT static bool resolve_lane(const Context& ctx, std::size_t lane, std::uint16_t& slot_out) noexcept {
        std::size_t slot = 0;
        const bool valid = (append_key<Keys>(ctx, lane, slot) && ...);
        if (!valid) return false;
        slot_out = static_cast<std::uint16_t>(slot);
        return true;
    }

    template <class Context>
    bool resolve_range(
        Context& ctx,
        KeyList<Keys...>,
        std::array<std::uint16_t, N>& slots,
        std::size_t begin,
        std::size_t end
    ) noexcept {
        if constexpr ((Keys::row_scalar && ...)) {
            std::uint16_t slot = 0;
            if (!resolve_lane(ctx, 0, slot)) return false;
            for (std::size_t lane = begin; lane < end; ++lane) slots[lane] = slot;
            return true;
        }
        for (std::size_t lane = begin; lane < end; ++lane) {
            if (!resolve_lane(ctx, lane, slots[lane])) return false;
        }
        return true;
    }
};

template <std::size_t N>
struct NoKeyResolver {
    static constexpr std::size_t capacity = 1;
    void setup() noexcept {}
    template <class Context>
    bool resolve_range(
        Context&,
        KeyList<>,
        std::array<std::uint16_t, N>& slots,
        std::size_t begin,
        std::size_t end
    ) noexcept {
        for (std::size_t lane = begin; lane < end; ++lane) slots[lane] = 0;
        return true;
    }
};

template <
    std::size_t N,
    std::size_t Inputs,
    std::size_t ScratchSlots,
    std::size_t MatrixScratchSlots = 0,
    std::size_t MatrixScratchWidth = 1
>
struct alignas(64) GroupRowContext {
    std::array<const double*, Inputs> inputs{};
    alignas(64) std::array<std::array<double, N>, ScratchSlots> scratch{};
    alignas(64) std::array<std::array<double, N * MatrixScratchWidth>, MatrixScratchSlots> scratch_matrix{};
    double* output = nullptr;
    const std::array<std::uint16_t, N>* group_slots = nullptr;
    const std::array<std::uint16_t, N>* partitions = nullptr;
    std::size_t lane_begin = 0;
    std::size_t lane_end = N;

    template <class Src>
    STACKDSL_HOT double read_native(std::size_t lane) const noexcept {
        static_assert(source_width_v<Src> == 1);
        if constexpr (requires { Src::input_index; }) return inputs[Src::input_index][lane];
        else if constexpr (requires { Src::slot_index; }) return scratch[Src::slot_index][Src::row_scalar ? 0 : lane];
        else return Src::value;
    }
    template <class Src> STACKDSL_HOT double read(std::size_t lane) const noexcept { return read_native<Src>(lane); }
    template <class Src>
    STACKDSL_HOT double read_feature(std::size_t lane, std::size_t feature) const noexcept {
        if constexpr (requires { Src::matrix_slot_index; }) return scratch_matrix[Src::matrix_slot_index][lane * MatrixScratchWidth + feature];
        else { (void)feature; return read<Src>(lane); }
    }
    template <class Src>
    STACKDSL_HOT const double* read_ptr() const noexcept {
        static_assert(source_width_v<Src> == 1 && !is_literal_source_v<Src>);
        if constexpr (requires { Src::input_index; }) return inputs[Src::input_index];
        else return scratch[Src::slot_index].data();
    }
    template <class Dst>
    STACKDSL_HOT double* write_ptr() noexcept {
        if constexpr (std::is_same_v<Dst, OutputDst>) return output;
        else if constexpr (requires { Dst::matrix_slot_index; }) return scratch_matrix[Dst::matrix_slot_index].data();
        else return scratch[Dst::slot_index].data();
    }
};

template <std::size_t N, class Resolver, class Partitions, class InnerPlan, class Out, class Keys, class FeedList>
struct GroupByNode;

template <std::size_t N, class Resolver, class Partitions, class InnerPlan, class Out, class... Keys, class... FeedSources>
struct GroupByNode<N, Resolver, Partitions, InnerPlan, Out, KeyList<Keys...>, SourceList<FeedSources...>> {
    Resolver resolver{};
    InnerPlan inner{};
    std::array<std::uint16_t, N> group_slots{};
    void setup() noexcept { resolver.setup(); inner.setup(); }
    template <class Context>
    STACKDSL_HOT bool on_data_checked(Context& ctx) noexcept {
        const std::size_t begin = ctx.lane_begin;
        const std::size_t end = ctx.lane_end;
        if (!resolver.resolve_range(ctx, KeyList<Keys...>{}, group_slots, begin, end)) return false;
        std::array<const double*, sizeof...(FeedSources)> feeds{ctx.template read_ptr<FeedSources>()...};
        inner.on_data(
            feeds,
            group_slots,
            Partitions::values,
            ctx.template write_ptr<Out>(),
            begin,
            end
        );
        return true;
    }
    template <class Context> STACKDSL_HOT void on_data(Context& ctx) noexcept { (void)on_data_checked(ctx); }
};

}  // namespace stackdsl
