#pragma once

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace stackdsl {

inline std::vector<unsigned> allowed_cpu_ids() {
    std::vector<unsigned> result;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &set)) result.push_back(cpu);
        }
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
