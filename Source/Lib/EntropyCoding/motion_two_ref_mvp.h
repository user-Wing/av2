/* Projection primitives for the restricted two-past-reference MVP profile.
 * This is not a candidate selector. The caller supplies an admissible
 * neighbor after the normative direct/derived spatial scan and must apply
 * the block-specific reference clamp before coding or predicting a block.
 */
#ifndef SVT_AV2_MOTION_TWO_REF_MVP_H
#define SVT_AV2_MOTION_TWO_REF_MVP_H

#include <stdint.h>
#include <stddef.h>

/* Use expanded display indices, never subtract raw wrapped OBU order hints.
 * Validate both distances before publishing either output. */
static inline int svt_av2_mvp_two_past_distances(uint64_t current, uint64_t ref0,
    uint64_t ref1, uint32_t distance[2]) {
    if (distance == NULL || ref0 == ref1 || ref0 >= current || ref1 >= current)
        return 0;
    const uint64_t d0 = current - ref0, d1 = current - ref1;
    if (d0 > 8u || d1 > 8u)
        return 0;
    distance[0] = (uint32_t)d0;
    distance[1] = (uint32_t)d1;
    return 1;
}

/* AV2 projects eighth-luma components, even in a quarter-MVD frame.
 * Use a truncated Q14 reciprocal and round the signed product by magnitude
 * (ties away from zero), not an exact rational or quarter-grid operation.
 * Distances are bounded to eight. The caller applies the spatial clamp.
 * Return 0 without changing the output on unsupported input. */
static inline int svt_av2_motion_q3_limit(int wide) { return wide == 2 ? 504 : wide ? 248 : 128; }

static inline int svt_av2_mvp_project_q3_range(int wide, int32_t mv_q3, uint32_t target_distance,
    uint32_t neighbor_distance, int32_t* result_q3) {
    const int32_t limit = svt_av2_motion_q3_limit(wide);
    if (result_q3 == NULL || mv_q3 < -limit || mv_q3 > limit ||
        target_distance == 0u || target_distance > 8u ||
        neighbor_distance == 0u || neighbor_distance > 8u)
        return 0;
    const uint32_t magnitude = (uint32_t)(mv_q3 >= 0 ? mv_q3 : -mv_q3);
    const uint32_t product = magnitude * target_distance * (16384u / neighbor_distance);
    const int32_t rounded = (int32_t)((product + 8192u) >> 14);
    *result_q3 = mv_q3 < 0 ? -rounded : rounded;
    return 1;
}

static inline int svt_av2_mvp_project_q3(int32_t mv_q3, uint32_t target_distance,
    uint32_t neighbor_distance, int32_t* result_q3) {
    return svt_av2_mvp_project_q3_range(0, mv_q3, target_distance, neighbor_distance, result_q3);
}

/* Convert an exact eighth-luma MV difference into quarter MVD units.
 * In this range, an odd predictor can only emit an odd target. This is a
 * representability check, not a way to quantize the selected predictor. The
 * +/-128 bound is the current writer's per-component quarter MVD range. */
static inline int svt_av2_mvp_quarter_delta_range(int wide, int32_t target_q3, int32_t predictor_q3,
    int32_t* delta_q2) {
    const int32_t limit = svt_av2_motion_q3_limit(wide);
    if (delta_q2 == NULL || target_q3 < -limit || target_q3 > limit ||
        predictor_q3 < -8 * limit || predictor_q3 > 8 * limit)
        return 0;
    const int32_t delta = target_q3 - predictor_q3;
    if (delta < -2 * limit || delta > 2 * limit || delta % 2 != 0)
        return 0;
    *delta_q2 = delta / 2;
    return 1;
}

static inline int svt_av2_mvp_quarter_delta(int32_t target_q3, int32_t predictor_q3, int32_t* delta_q2) {
    return svt_av2_mvp_quarter_delta_range(0, target_q3, predictor_q3, delta_q2);
}

#endif
