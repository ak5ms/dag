#pragma once

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace stackdsl {

#if defined(__linux__)
inline bool read_cpu_topology_value(
    unsigned cpu,
    const char* field,
    int& value
) {
    const std::string path =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
        "/topology/" + field;
    std::ifstream input(path);
    return static_cast<bool>(input >> value);
}

inline std::vector<unsigned> physical_core_first_order(
    const std::vector<unsigned>& cpus
) {
    struct CpuLocation {
        unsigned cpu;
        int package;
        int core;
    };

    std::vector<CpuLocation> locations;
    locations.reserve(cpus.size());
    for (const unsigned cpu : cpus) {
        int package = 0;
        int core = 0;
        if (
            !read_cpu_topology_value(cpu, "physical_package_id", package) ||
            !read_cpu_topology_value(cpu, "core_id", core)
        ) {
            return cpus;
        }
        locations.push_back({cpu, package, core});
    }

    std::vector<std::pair<int, int>> seen_cores;
    seen_cores.reserve(locations.size());
    std::vector<unsigned> primary_threads;
    std::vector<unsigned> sibling_threads;
    primary_threads.reserve(locations.size());
    sibling_threads.reserve(locations.size());

    for (const CpuLocation& location : locations) {
        const std::pair<int, int> key{location.package, location.core};
        if (
            std::find(seen_cores.begin(), seen_cores.end(), key) ==
            seen_cores.end()
        ) {
            seen_cores.push_back(key);
            primary_threads.push_back(location.cpu);
        } else {
            sibling_threads.push_back(location.cpu);
        }
    }

    primary_threads.insert(
        primary_threads.end(),
        sibling_threads.begin(),
        sibling_threads.end()
    );
    return primary_threads;
}
#endif

inline std::vector<unsigned> allowed_cpu_ids() {
    std::vector<unsigned> result;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &set)) result.push_back(cpu);
        }
        result = physical_core_first_order(result);
    }
#endif
    if (result.empty()) {
        const unsigned count = std::max(1u, std::thread::hardware_concurrency());
        result.reserve(count);
        for (unsigned cpu = 0; cpu < count; ++cpu) result.push_back(cpu);
    }
    return result;
}

inline std::size_t available_cpu_count() {
    return allowed_cpu_ids().size();
}

inline void pin_current_thread(std::size_t worker_index) noexcept {
#if defined(__linux__)
    const auto cpus = allowed_cpu_ids();
    if (cpus.empty()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpus[worker_index % cpus.size()], &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)worker_index;
#endif
}

inline std::size_t resolved_thread_count(
    std::size_t requested,
    std::size_t maximum_parallelism
) {
    const std::size_t available = available_cpu_count();
    const std::size_t desired = requested == 0 ? available : requested;
    return std::max<std::size_t>(
        1,
        std::min({desired, available, std::max<std::size_t>(1, maximum_parallelism)})
    );
}

}  // namespace stackdsl
