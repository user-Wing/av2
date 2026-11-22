#include "motion_estimation.h"
#include <stdint.h>

#if (defined(_M_X64) && !defined(_M_ARM64EC)) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#define SVT_ME_SSE2 1
#else
#define SVT_ME_SSE2 0
#endif

static uint32_t edge_coordinate(int64_t coordinate, uint32_t extent) {
    if (coordinate < 0) return 0;
    if (coordinate >= extent) return extent - 1;
    return (uint32_t)coordinate;
}
static inline uint32_t sample_difference(int a, int b) {
    const int diff = a - b;
    return (uint32_t)(diff < 0 ? -diff : diff);
}

/* Like AVM's SAD over an edge-extended reference, but only read visible
 * samples: repeat the first/last sample around the contiguous middle span. */
static inline uint64_t sad_row_8(const uint8_t* src, const uint8_t* ref,
    uint32_t left, uint32_t middle, uint32_t right) {
    uint64_t sad = 0;
    if (left != 0) {
        const uint8_t edge = ref[0];
        for (uint32_t x = 0; x < left; ++x) sad += sample_difference(src[x], edge);
        src += left;
    }
    uint32_t x = 0u;
#if SVT_ME_SSE2
    if (middle >= 16u) {
        __m128i lanes = _mm_setzero_si128();
        const uint32_t stop = middle & ~15u;
        for (; x < stop; x += 16u) {
            const __m128i a = _mm_loadu_si128((const __m128i*)(src + x));
            const __m128i b = _mm_loadu_si128((const __m128i*)(ref + x));
            lanes = _mm_add_epi64(lanes, _mm_sad_epu8(a, b));
        }
        /* Per-lane total <= 255 * UINT32_MAX, so the reduction fits uint64_t. */
        uint64_t values[2];
        _mm_storeu_si128((__m128i*)values, lanes);
        sad += values[0] + values[1];
    }
#endif
    for (; x < middle; ++x) sad += sample_difference(src[x], ref[x]);
    if (right != 0) {
        const uint8_t edge = ref[middle != 0 ? middle - 1u : 0u];
        src += middle;
        for (uint32_t tail = 0; tail < right; ++tail) sad += sample_difference(src[tail], edge);
    }
    return sad;
}

static inline uint64_t sad_row_10(const uint8_t* src, const uint8_t* ref,
    uint32_t left, uint32_t middle, uint32_t right) {
    uint64_t sad = 0;
    if (left != 0) {
        const uint16_t edge = svt_av2_sample_load(ref, 10);
        for (uint32_t x = 0; x < left; ++x)
            sad += sample_difference(svt_av2_sample_load(src + (size_t)x * 2u, 10), edge);
        src += (size_t)left * 2u;
    }
    uint32_t x = 0u;
#if SVT_ME_SSE2
    if (middle >= 8u && middle <= 4096u) {
        __m128i lanes = _mm_setzero_si128();
        const __m128i zero = _mm_setzero_si128();
        const uint32_t stop = middle & ~7u;
        for (; x < stop; x += 8u) {
            const size_t offset = (size_t)x * 2u;
            const __m128i a = _mm_loadu_si128((const __m128i*)(src + offset));
            const __m128i b = _mm_loadu_si128((const __m128i*)(ref + offset));
            /* Unsigned absolute difference even for full-range words. */
            const __m128i diff = _mm_or_si128(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a));
            lanes = _mm_add_epi32(lanes, _mm_add_epi32(_mm_unpacklo_epi16(diff, zero),
                                                       _mm_unpackhi_epi16(diff, zero)));
        }
        /* At most 4096*65535, well below INT32_MAX. */
        lanes = _mm_add_epi32(lanes, _mm_srli_si128(lanes, 8));
        lanes = _mm_add_epi32(lanes, _mm_srli_si128(lanes, 4));
        sad += (uint32_t)_mm_cvtsi128_si32(lanes);
    }
#endif
    for (; x < middle; ++x) {
        const size_t offset = (size_t)x * 2u;
        sad += sample_difference(svt_av2_sample_load(src + offset, 10), svt_av2_sample_load(ref + offset, 10));
    }
    if (right != 0) {
        const uint16_t edge = svt_av2_sample_load(ref + (size_t)(middle != 0 ? middle - 1u : 0u) * 2u, 10);
        src += (size_t)middle * 2u;
        for (uint32_t tail = 0; tail < right; ++tail)
            sad += sample_difference(svt_av2_sample_load(src + (size_t)tail * 2u, 10), edge);
    }
    return sad;
}

static uint64_t candidate_sad(const SvtAv2FrameBuffer* source,
    const SvtAv2FrameBuffer* reference, const SvtAv2BlockView* block,
    SvtAv2MotionVector mv, uint64_t limit) {
    const int64_t rx = (int64_t)block->x + mv.mv_x;
    const int64_t ry = (int64_t)block->y + mv.mv_y;
    const int64_t right_overhang = rx + block->w - reference->visible_width;
    /* Cap each overhang at the block width: a narrow block can lie entirely
     * outside the frame. In that case ref points at the repeated edge. */
    const uint32_t left = rx < 0 ? (uint32_t)(-rx < block->w ? -rx : block->w) : 0;
    const uint32_t right = right_overhang > 0 ?
        (uint32_t)(right_overhang < block->w ? right_overhang : block->w) : 0;
    const uint32_t middle = block->w - left - right;
    const uint32_t step = source->bit_depth > 8 ? 2u : 1u;
    const uint8_t* src = source->planes[0] + (size_t)block->y * source->strides[0] + (size_t)block->x * step;
    const uint8_t* ref_column = reference->planes[0] + (size_t)edge_coordinate(rx, reference->visible_width) * step;
    uint64_t cost = 0;
    for (uint32_t y = 0; y < block->h; ++y) {
        const uint8_t* ref = ref_column + (size_t)edge_coordinate(ry + y, reference->visible_height) * reference->strides[0];
        const uint8_t* row = src + (size_t)y * source->strides[0];
        cost += step == 1u ? sad_row_8(row, ref, left, middle, right) : sad_row_10(row, ref, left, middle, right);
        /* A strictly larger partial SAD cannot win, including the tie rule. */
        if (cost > limit) break;
    }
    return cost;
}
static int magnitude(int32_t value) { return value < 0 ? -value : value; }
static int preferred(SvtAv2MotionVector candidate, SvtAv2MotionVector best) {
    const int a = magnitude(candidate.mv_x) + magnitude(candidate.mv_y);
    const int b = magnitude(best.mv_x) + magnitude(best.mv_y);
    return a < b || (a == b && (candidate.mv_y < best.mv_y ||
        (candidate.mv_y == best.mv_y && candidate.mv_x < best.mv_x)));
}
SvtAv2Error svt_av2_motion_estimate(const SvtAv2FrameBuffer* source,
    const SvtAv2FrameBuffer* reference, const SvtAv2BlockView* block,
    SvtAv2MotionSearchResult* result) {
    if (source == NULL || reference == NULL || block == NULL || result == NULL ||
        source->buffer == NULL || reference->buffer == NULL || source->planes[0] == NULL || reference->planes[0] == NULL)
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (source->visible_width != reference->visible_width || source->visible_height != reference->visible_height ||
        source->bit_depth != reference->bit_depth || source->chroma_format != reference->chroma_format ||
        block->w == 0 || block->h == 0 || block->x >= source->visible_width || block->y >= source->visible_height ||
        block->w > source->visible_width - block->x || block->h > source->visible_height - block->y)
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    SvtAv2MotionSearchResult best = { { 0, 0 }, 0, 1 };
    best.sad = candidate_sad(source, reference, block, best.mv, UINT64_MAX);
    for (int32_t y = -SVT_AV2_ME_SEARCH_RADIUS; y <= SVT_AV2_ME_SEARCH_RADIUS; ++y)
        for (int32_t x = -SVT_AV2_ME_SEARCH_RADIUS; x <= SVT_AV2_ME_SEARCH_RADIUS; ++x) {
            if (x == 0 && y == 0) continue;
            const SvtAv2MotionVector candidate = { x, y };
            const uint64_t sad = candidate_sad(source, reference, block, candidate, best.sad);
            ++best.candidates;
            if (sad < best.sad || (sad == best.sad && preferred(candidate, best.mv))) {
                best.sad = sad;
                best.mv = candidate;
            }
        }
    *result = best;
    return SVT_AV2_ERROR_NONE;
}

void svt_av2_motion_extend(const SvtAv2FrameBuffer* source, const SvtAv2FrameBuffer* reference,
    const SvtAv2BlockView* block, SvtAv2MotionSearchResult* result) {
    if (result->sad < (uint64_t)block->w * block->h * (3u << (source->bit_depth - 8u))) return;
    for (int step = 4; step; step /= 2) {
        const SvtAv2MotionVector center = step == 4 ? (SvtAv2MotionVector){0, 0} : result->mv;
        const int radius = step == 4 ? 16 : step;
        for (int dy = -radius; dy <= radius; dy += step) for (int dx = -radius; dx <= radius; dx += step) {
            const SvtAv2MotionVector mv = {center.mv_x + dx, center.mv_y + dy};
            if ((magnitude(mv.mv_x) <= 8 && magnitude(mv.mv_y) <= 8) ||
                magnitude(mv.mv_x) > 16 || magnitude(mv.mv_y) > 16) continue;
            const uint64_t sad = candidate_sad(source, reference, block, mv, result->sad);
            ++result->candidates;
            if (sad < result->sad || (sad == result->sad && preferred(mv, result->mv))) {
                result->sad = sad;
                result->mv = mv;
            }
        }
        if (magnitude(result->mv.mv_x) <= 8 && magnitude(result->mv.mv_y) <= 8) break;
    }
}

SvtAv2Error svt_av2_motion_estimate_list(const SvtAv2FrameBuffer* source,
    const SvtAv2FrameBuffer* reference, const SvtAv2BlockView* block,
    SvtAv2MotionSearchList* result) {
    if (source == NULL || reference == NULL || block == NULL || result == NULL ||
        source->buffer == NULL || reference->buffer == NULL || source->planes[0] == NULL || reference->planes[0] == NULL)
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (source->visible_width != reference->visible_width || source->visible_height != reference->visible_height ||
        source->bit_depth != reference->bit_depth || source->chroma_format != reference->chroma_format ||
        block->w == 0 || block->h == 0 || block->x >= source->visible_width || block->y >= source->visible_height ||
        block->w > source->visible_width - block->x || block->h > source->visible_height - block->y)
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    const SvtAv2MotionVector zero = {0, 0};
    const uint64_t zero_sad = candidate_sad(source, reference, block, zero, UINT64_MAX);
    result->count = 1u; result->candidates = 1u;
    result->mv[0] = zero; result->sad[0] = zero_sad;
    for (int32_t y = -SVT_AV2_ME_SEARCH_RADIUS; y <= SVT_AV2_ME_SEARCH_RADIUS; ++y)
        for (int32_t x = -SVT_AV2_ME_SEARCH_RADIUS; x <= SVT_AV2_ME_SEARCH_RADIUS; ++x) {
            if (x == 0 && y == 0) continue;
            const SvtAv2MotionVector mv = {x, y};
            const uint64_t limit = result->count < 3u ? UINT64_MAX : result->sad[2];
            const uint64_t sad = candidate_sad(source, reference, block, mv, limit);
            ++result->candidates;
            uint32_t slot = 0u;
            while (slot < result->count && (sad > result->sad[slot] ||
                (sad == result->sad[slot] && !preferred(mv, result->mv[slot])))) ++slot;
            if (slot >= 3u) continue;
            if (result->count < 3u) ++result->count;
            for (uint32_t i = result->count - 1u; i > slot; --i) {
                result->mv[i] = result->mv[i - 1u]; result->sad[i] = result->sad[i - 1u];
            }
            result->mv[slot] = mv; result->sad[slot] = sad;
        }
    int have_zero = 0;
    for (uint32_t i = 0u; i < result->count; ++i)
        if (result->mv[i].mv_x == 0 && result->mv[i].mv_y == 0) have_zero = 1;
    if (!have_zero) {
        result->mv[result->count] = zero; result->sad[result->count++] = zero_sad;
    }
    return SVT_AV2_ERROR_NONE;
}

static void keep_wide(SvtAv2MotionSearchList* result, SvtAv2MotionVector mv, uint64_t sad) {
    uint32_t slot = 0u;
    while (slot < result->count && (sad > result->sad[slot] ||
        (sad == result->sad[slot] && !preferred(mv, result->mv[slot])))) ++slot;
    if (slot >= 2u) return;
    if (result->count < 2u) ++result->count;
    for (uint32_t i = result->count - 1u; i > slot; --i) {
        result->mv[i] = result->mv[i - 1u]; result->sad[i] = result->sad[i - 1u];
    }
    result->mv[slot] = mv; result->sad[slot] = sad;
}

SvtAv2Error svt_av2_motion_estimate_wide(const SvtAv2FrameBuffer* source,
    const SvtAv2FrameBuffer* reference, const SvtAv2BlockView* block,
    SvtAv2MotionSearchList* result) {
    if (source == NULL || reference == NULL || block == NULL || result == NULL ||
        source->buffer == NULL || reference->buffer == NULL || source->planes[0] == NULL || reference->planes[0] == NULL)
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (source->visible_width != reference->visible_width || source->visible_height != reference->visible_height ||
        source->bit_depth != reference->bit_depth || source->chroma_format != reference->chroma_format ||
        block->w == 0 || block->h == 0 || block->x >= source->visible_width || block->y >= source->visible_height ||
        block->w > source->visible_width - block->x || block->h > source->visible_height - block->y)
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    uint8_t seen[33u * 33u] = {0};
    result->count = result->candidates = 0u;
    for (int32_t y = -16; y <= 16; y += 4)
        for (int32_t x = -16; x <= 16; x += 4) {
            if (x >= -8 && x <= 8 && y >= -8 && y <= 8) continue;
            const SvtAv2MotionVector mv = {x, y};
            const uint64_t limit = result->count < 2u ? UINT64_MAX : result->sad[1];
            const uint64_t sad = candidate_sad(source, reference, block, mv, limit);
            seen[(size_t)(y + 16) * 33u + (uint32_t)(x + 16)] = 1u;
            ++result->candidates; keep_wide(result, mv, sad);
        }
    const SvtAv2MotionVector center = result->mv[0];
    for (int32_t y = center.mv_y - 2; y <= center.mv_y + 2; ++y)
        for (int32_t x = center.mv_x - 2; x <= center.mv_x + 2; ++x) {
            if (x < -16 || x > 16 || y < -16 || y > 16 ||
                (x >= -8 && x <= 8 && y >= -8 && y <= 8)) continue;
            const size_t index = (size_t)(y + 16) * 33u + (uint32_t)(x + 16);
            if (seen[index]) continue;
            seen[index] = 1u;
            const SvtAv2MotionVector mv = {x, y};
            const uint64_t limit = result->count < 2u ? UINT64_MAX : result->sad[1];
            const uint64_t sad = candidate_sad(source, reference, block, mv, limit);
            ++result->candidates; keep_wide(result, mv, sad);
        }
    return SVT_AV2_ERROR_NONE;
}
