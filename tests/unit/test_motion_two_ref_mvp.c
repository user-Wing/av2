/* Standalone, no changes to the existing test targets. */
#include "motion_two_ref_mvp.h"
#include <stdio.h>

static unsigned checks, failures;
#define CHECK(expr) do { ++checks; if (!(expr)) { ++failures; \
    fprintf(stderr, "line %d: %s\n", __LINE__, #expr); } } while (0)

int main(void) {
    static const uint32_t inv_q14[8] = {16384,8192,5461,4096,3276,2730,2340,2048};
    int32_t v = 91;
    for (uint32_t i = 1; i <= 8u; ++i)
        for (uint32_t j = 1; j <= 8u; ++j)
            for (int32_t m = -128; m <= 128; ++m) {
                const int64_t numerator = (int64_t)m * i * inv_q14[j - 1u];
                const int32_t expected = (int32_t)((numerator + (m < 0 ? -8192 : 8192)) / 16384);
                CHECK(svt_av2_mvp_project_q3(m, i, j, &v) && v == expected);
            }
    CHECK(svt_av2_mvp_project_q3(1, 3, 2, &v) && v == 2);
    CHECK(svt_av2_mvp_project_q3(-1, 3, 2, &v) && v == -2);
    CHECK(svt_av2_mvp_project_q3(4, 6, 7, &v) && v == 3);
    CHECK(svt_av2_mvp_project_q3(-4, 6, 7, &v) && v == -3);
    /* Exact rational would round 3.5 to 4; the Q14 reciprocal is low. */
    CHECK(svt_av2_mvp_project_q3(3, 7, 6, &v) && v == 3);
    CHECK(svt_av2_mvp_project_q3(-3, 7, 6, &v) && v == -3);
    CHECK(svt_av2_mvp_project_q3(2, 3, 2, &v) && v == 3);
    CHECK(!svt_av2_mvp_project_q3(129, 3, 2, &v) && v == 3);
    CHECK(!svt_av2_mvp_project_q3(-129, 3, 2, &v) && v == 3);
    CHECK(!svt_av2_mvp_project_q3(2, 0, 2, &v) && v == 3);
    CHECK(!svt_av2_mvp_project_q3(2, 3, 9, &v) && v == 3);
    CHECK(!svt_av2_mvp_project_q3(2, 3, 2, NULL));
    CHECK(svt_av2_mvp_quarter_delta(7, 3, &v) && v == 2);
    CHECK(svt_av2_mvp_quarter_delta(-7, -3, &v) && v == -2);
    CHECK(!svt_av2_mvp_quarter_delta(8, 3, &v) && v == -2);
    CHECK(!svt_av2_mvp_quarter_delta(-8, -3, &v) && v == -2);
    CHECK(svt_av2_mvp_quarter_delta(128, -128, &v) && v == 128);
    CHECK(svt_av2_mvp_quarter_delta(-128, 128, &v) && v == -128);
    CHECK(!svt_av2_mvp_quarter_delta(-128, 130, &v) && v == -128);
    CHECK(!svt_av2_mvp_quarter_delta(-128, 2, NULL));
    uint32_t distances[2] = {42u, 43u};
    CHECK(svt_av2_mvp_two_past_distances(257u, 256u, 254u, distances) && distances[0] == 1u && distances[1] == 3u);
    CHECK(svt_av2_mvp_two_past_distances(UINT64_MAX, UINT64_MAX-8u, UINT64_MAX-1u, distances) &&
        distances[0] == 8u && distances[1] == 1u);
    CHECK(!svt_av2_mvp_two_past_distances(257u, 256u, 256u, distances));
    CHECK(!svt_av2_mvp_two_past_distances(257u, 255u, 258u, distances));
    CHECK(!svt_av2_mvp_two_past_distances(257u, 255u, 0u, distances));
    CHECK(!svt_av2_mvp_two_past_distances(257u, 255u, 250u, NULL));
    CHECK(distances[0] == 8u && distances[1] == 1u);
    printf("%u checks, %u failures\n", checks, failures);
    return failures != 0;
}
