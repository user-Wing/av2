#ifndef SVT_AV2_INTER_PROFILE_H
#define SVT_AV2_INTER_PROFILE_H

/* Private diagnostics only. Include before system headers so strict ISO C
 * POSIX builds expose clock_gettime when profiling is enabled. */
#ifndef SVT_AV2_INTER_PROFILE
#define SVT_AV2_INTER_PROFILE 0
#endif

#if SVT_AV2_INTER_PROFILE
#if !defined(_WIN32) && (defined(__unix__) || defined(__APPLE__)) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdint.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

#if defined(_WIN32)
#define SVT_AV2_IP_CLOCK_NAME "qpc"
static inline uint64_t svt_av2_ip_now(void) {
    LARGE_INTEGER counter;
    return QueryPerformanceCounter(&counter) && counter.QuadPart >= 0 ?
        (uint64_t)counter.QuadPart + 1u : 0u;
}
static inline uint64_t svt_av2_ip_hz(void) {
    LARGE_INTEGER freq;
    return QueryPerformanceFrequency(&freq) && freq.QuadPart > 0 ? (uint64_t)freq.QuadPart : 0u;
}
#elif defined(CLOCK_MONOTONIC)
#define SVT_AV2_IP_CLOCK_NAME "posix_monotonic"
static inline uint64_t svt_av2_ip_now(void) {
    struct timespec t;
    return clock_gettime(CLOCK_MONOTONIC, &t) == 0 && t.tv_sec >= 0 ?
        (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec + 1u : 0u;
}
static inline uint64_t svt_av2_ip_hz(void) { return UINT64_C(1000000000); }
#else
/* ISO C fallback: process CPU clock, not a wall-clock measurement. Never
 * substitute TIME_UTC, which can jump backwards or forwards mid-frame. */
#define SVT_AV2_IP_CLOCK_NAME "c_process_cpu"
static inline uint64_t svt_av2_ip_now(void) {
    const clock_t t = clock();
    return t == (clock_t)-1 ? 0u : (uint64_t)t + 1u;
}
static inline uint64_t svt_av2_ip_hz(void) { return (uint64_t)CLOCKS_PER_SEC; }
#endif

static inline uint64_t svt_av2_ip_delta(uint64_t start, uint64_t end) {
    return start && end >= start ? end - start : 0u;
}

/* Native ticks are accumulated in hot regions; conversion and formatting
 * occur once per frame. The cast's operand is bounded by UINT64_MAX. */
static inline unsigned long long svt_av2_ip_ns(uint64_t ticks, uint64_t hz) {
    if (!hz) return 0u;
    if (hz == UINT64_C(1000000000)) return (unsigned long long)ticks;
    const double ns = (double)ticks * 1e9 / (double)hz;
    return ns < 18446744073709551616.0 ? (unsigned long long)ns : UINT64_MAX;
}

#define SVT_AV2_IP_START(name) const uint64_t name = svt_av2_ip_now()
#define SVT_AV2_IP_ADD(total, start) ((total) += svt_av2_ip_delta((start), svt_av2_ip_now()))
#else
/* Elide even the timers' identifiers and storage in normal builds. */
#define SVT_AV2_IP_START(name) ((void)0)
#define SVT_AV2_IP_ADD(total, start) ((void)0)
#endif

#endif
