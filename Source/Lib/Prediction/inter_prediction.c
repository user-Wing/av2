#include "inter_prediction.h"

/* Private build switch for exact timing against the unchanged scalar kernels. */
#ifndef SVT_AV2_SHARP_SSE2
#define SVT_AV2_SHARP_SSE2 1
#endif
#ifndef SVT_AV2_SHARP_INTEGER_SAD
#define SVT_AV2_SHARP_INTEGER_SAD 1
#endif
#if SVT_AV2_SHARP_SSE2 && ((defined(_M_X64) && !defined(_M_ARM64EC)) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__))
#include <emmintrin.h>
#define SVT_SHARP_HAVE_SSE2 1
#else
#define SVT_SHARP_HAVE_SSE2 0
#endif

static int64_t floor_half(int64_t value) {
    return value >= 0 ? value / 2 : -((-value + 1) / 2);
}
static uint16_t reference_sample(const SvtAv2FrameBuffer* reference, uint32_t plane,
    int64_t x, int64_t y) {
    const uint32_t w = svt_av2_picture_plane_width(reference, plane);
    const uint32_t h = svt_av2_picture_plane_height(reference, plane);
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= w) x = w - 1;
    if (y >= h) y = h - 1;
    const uint32_t step = reference->bit_depth > 8 ? 2u : 1u;
    return svt_av2_sample_load(reference->planes[plane] + (size_t)y * reference->strides[plane] +
        (size_t)x * step, reference->bit_depth);
}
SvtAv2Error svt_av2_inter_predict(const SvtAv2FrameBuffer* reference, uint32_t plane,
    const SvtAv2BlockView* block, SvtAv2MotionVector mv, uint16_t* prediction,
    uint32_t prediction_stride) {
    if (reference == NULL || block == NULL || prediction == NULL || reference->buffer == NULL ||
        plane >= 3 || reference->planes[plane] == NULL || prediction_stride < block->w ||
        block->w == 0 || block->h == 0) return SVT_AV2_ERROR_INVALID_PARAMETER;
    const uint32_t width = svt_av2_picture_plane_width(reference, plane);
    const uint32_t height = svt_av2_picture_plane_height(reference, plane);
    if (block->x >= width || block->y >= height || block->w > width - block->x || block->h > height - block->y)
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    const int subsampled = plane != 0 && reference->chroma_format == SVT_AV2_CHROMA_420;
    const int64_t dx = subsampled ? floor_half(mv.mv_x) : mv.mv_x;
    const int64_t dy = subsampled ? floor_half(mv.mv_y) : mv.mv_y;
    const int half_x = subsampled && (int64_t)mv.mv_x - 2 * dx != 0;
    const int half_y = subsampled && (int64_t)mv.mv_y - 2 * dy != 0;
    /* AV2 spec 7.13.3.16 and interpolation process; AVM filter.h BILINEAR3,
     * convolve.c single-reference round0=3/round1=11. Half-phase coefficients
     * are64/64; first-stage rounding loses no bits. Preserve both axes' exact
     * sum until final rounding (rounding twice would differ for odd sums). */
    for (uint32_t y = 0; y < block->h; ++y) for (uint32_t x = 0; x < block->w; ++x) {
        const int64_t rx = (int64_t)block->x + x + dx, ry = (int64_t)block->y + y + dy;
        uint32_t sum = reference_sample(reference, plane, rx, ry);
        if (half_x) sum += reference_sample(reference, plane, rx + 1, ry);
        if (half_y) {
            sum += reference_sample(reference, plane, rx, ry + 1);
            if (half_x) sum += reference_sample(reference, plane, rx + 1, ry + 1);
        }
        const uint32_t shift = (uint32_t)(half_x + half_y);
        if (shift != 0) sum = (sum + (1u << (shift - 1))) >> shift;
        prediction[(size_t)y * prediction_stride + x] = (uint16_t)sum;
    }
    return SVT_AV2_ERROR_NONE;
}

static int64_t split_quarter_offset(int64_t offset, uint32_t* fraction) {
    const int64_t whole = floor_half(floor_half(offset));
    *fraction = (uint32_t)(offset - 4 * whole);
    return whole;
}

static uint32_t clamp_sample_pos(int64_t pos, uint32_t size) {
    return pos < 0 ? 0 : pos >= size ? size - 1 : (uint32_t)pos;
}

SvtAv2Error svt_av2_inter_predict_half(const SvtAv2FrameBuffer* reference, uint32_t plane,
    const SvtAv2BlockView* block, SvtAv2HalfPelMotionVector mv, uint16_t* prediction,
    uint32_t prediction_stride) {
    if (reference == NULL || block == NULL || prediction == NULL || reference->buffer == NULL ||
        plane >= 3 || reference->planes[plane] == NULL || prediction_stride < block->w ||
        block->w == 0 || block->h == 0) return SVT_AV2_ERROR_INVALID_PARAMETER;
    const uint32_t width = svt_av2_picture_plane_width(reference, plane);
    const uint32_t height = svt_av2_picture_plane_height(reference, plane);
    if (block->x >= width || block->y >= height || block->w > width - block->x || block->h > height - block->y)
        return SVT_AV2_ERROR_INVALID_PARAMETER;

    /* AV2 v1.0.0 section 7.13.3.17: a half-luma MV step is 8 Q4
     * units on Y/444 and 4 on 420 chroma. Convert to quarter plane samples. */
    const int quarter_scale = plane != 0 && reference->chroma_format == SVT_AV2_CHROMA_420 ? 1 : 2;
    uint32_t fx, fy;
    const int64_t base_x = (int64_t)block->x + split_quarter_offset((int64_t)mv.mv_x * quarter_scale, &fx);
    const int64_t base_y = (int64_t)block->y + split_quarter_offset((int64_t)mv.mv_y * quarter_scale, &fy);

    /* Sections 7.13.3.16-18: BILINEAR taps at these phases are (128-32*f, 32*f),
     * including 4-sample dimensions. Horizontal round0=3 is exact; vertical
     * round1=11 (postround=0) reduces to one rounding of the weighted sum
     * with denominator 16. Clip each tap to the reference, not the MV. */
    const uint32_t wx = 4 - fx, wy = 4 - fy;
    const uint32_t bd = reference->bit_depth, step = bd > 8 ? 2u : 1u;
    const uint8_t* src = reference->planes[plane];
    const uint32_t stride = reference->strides[plane];
    /* A visible contiguous horizontal footprint needs no tap clipping. Select
     * the axes once per row and keep the same single final rounding. The edge
     * loop below remains the reference for partial/fully overhanging blocks. */
    if (base_x >= 0 && base_x + block->w + (fx != 0) <= width) {
        const uint8_t* col = src + (size_t)base_x * step;
        for (uint32_t y = 0; y < block->h; ++y) {
            const int64_t ry = base_y + y;
            const uint8_t* top = col + (size_t)clamp_sample_pos(ry, height) * stride;
            uint16_t* out = prediction + (size_t)y * prediction_stride;
            if (!fy && !fx) {
                for (uint32_t x = 0; x < block->w; ++x)
                    out[x] = svt_av2_sample_load(top + (size_t)x * step, bd);
            } else if (!fy) {
                for (uint32_t x = 0; x < block->w; ++x) {
                    const uint8_t* a = top + (size_t)x * step;
                    out[x] = (uint16_t)((wx * svt_av2_sample_load(a, bd) +
                        fx * svt_av2_sample_load(a + step, bd) + 2u) >> 2);
                }
            } else {
                const uint8_t* bottom = col + (size_t)clamp_sample_pos(ry + 1, height) * stride;
                if (!fx) {
                    for (uint32_t x = 0; x < block->w; ++x) {
                        const size_t i = (size_t)x * step;
                        out[x] = (uint16_t)((wy * svt_av2_sample_load(top + i, bd) +
                            fy * svt_av2_sample_load(bottom + i, bd) + 2u) >> 2);
                    }
                } else {
                    for (uint32_t x = 0; x < block->w; ++x) {
                        const size_t i = (size_t)x * step;
                        const uint32_t a = wx * svt_av2_sample_load(top + i, bd) +
                            fx * svt_av2_sample_load(top + i + step, bd);
                        const uint32_t b = wx * svt_av2_sample_load(bottom + i, bd) +
                            fx * svt_av2_sample_load(bottom + i + step, bd);
                        out[x] = (uint16_t)((wy * a + fy * b + 8u) >> 4);
                    }
                }
            }
        }
        return SVT_AV2_ERROR_NONE;
    }
    for (uint32_t y = 0; y < block->h; ++y) {
        const int64_t ry = base_y + y;
        const uint8_t* row0 = src + (size_t)clamp_sample_pos(ry, height) * stride;
        const uint8_t* row1 = src + (size_t)clamp_sample_pos(ry + 1, height) * stride;
        for (uint32_t x = 0; x < block->w; ++x) {
            const int64_t rx = base_x + x;
            const size_t i0 = (size_t)clamp_sample_pos(rx, width) * step;
            const size_t i1 = (size_t)clamp_sample_pos(rx + 1, width) * step;
            uint32_t top = wx * svt_av2_sample_load(row0 + i0, bd);
            if (fx != 0) top += fx * svt_av2_sample_load(row0 + i1, bd);
            uint32_t sum = wy * top;
            if (fy != 0) {
                uint32_t bottom = wx * svt_av2_sample_load(row1 + i0, bd);
                if (fx != 0) bottom += fx * svt_av2_sample_load(row1 + i1, bd);
                sum += fy * bottom;
            }
            prediction[(size_t)y * prediction_stride + x] = (uint16_t)((sum + 8) >> 4);
        }
    }
    return SVT_AV2_ERROR_NONE;
}

/* AV2 v1.0.0 inter prediction filter selection and interpolation (7.13.3.16-18).
 * The normative Q7 coefficients are all even: store Q6 and reduce each
 * rounding shift by one. Four-tap rows start at -1. Half/quarter paths reach
 * only even Q4 phases; the odd rows are needed for eighth-luma 420 chroma. */
static const int16_t sharp_q6[7][8] = {
    {-1, 3, -6,62, 8,-3,  2,-1},
    {-2, 5,-11,58,19,-7,  3,-1},
    {-2, 5,-12,50,30,-10, 4,-1},
    {-2, 6,-12,40,40,-12, 6,-2},
    {-1, 4,-10,30,50,-12, 5,-2},
    {-1, 3, -7,19,58,-11, 5,-2},
    {-1, 2, -3, 8,62, -6, 3,-1},
};
static const int16_t small_regular_q6[7][4] = {
    {-4,61, 9,-2}, {-6,55,19,-4}, {-7,47,29,-5}, {-6,38,38,-6},
    {-5,29,47,-7}, {-4,19,55,-6}, {-2, 9,61,-4},
};

static const int16_t sharp_odd_q6[8][8] = {
    {-1,1, -3,63, 4, -1, 1, 0}, {-1,4, -9,60,13, -5, 3,-1},
    {-2,5,-11,54,24, -9, 4,-1}, {-2,5,-12,45,35,-11, 5,-1},
    {-1,5,-11,35,45,-12, 5,-2}, {-1,4, -9,24,54,-11, 5,-2},
    {-1,3, -5,13,60, -9, 4,-1}, { 0,1, -1, 4,63, -3, 1,-1},
};
static const int16_t small_regular_odd_q6[8][4] = {
    {-2,63, 4,-1}, {-5,58,14,-3}, {-6,51,24,-5}, {-6,42,33,-5},
    {-5,33,42,-6}, {-5,24,51,-6}, {-3,14,58,-5}, {-1, 4,63,-2},
};

/* `offset` is in sixteenth-plane-sample units. Widened input MVs make
 * subtracting fifteen for floor division safe, including INT32_MIN. */
static int64_t split_sixteenth_offset(int64_t offset, uint32_t* phase) {
    const int64_t whole = (offset >= 0 ? offset : offset - 15) / 16;
    *phase = (uint32_t)(offset - 16 * whole);
    return whole;
}

typedef struct SharpAxis {
    const int16_t* weights;
    uint32_t taps;
    int first;
} SharpAxis;

static SharpAxis sharp_axis(uint32_t phase, uint32_t span) {
    static const int16_t identity = 64;
    if (phase == 0u) return (SharpAxis){&identity, 1u, 0};
    const uint32_t index = (phase - 1u) >> 1;
    if (span <= 4u)
        return (SharpAxis){phase & 1u ? small_regular_odd_q6[index] : small_regular_q6[index], 4u, -1};
    return (SharpAxis){phase & 1u ? sharp_odd_q6[index] : sharp_q6[index], 8u, -3};
}

#define SHARP_ROW_WIDTH 64u

static void sharp_horizontal_row(const uint8_t* src, uint32_t width,
    uint32_t depth, int64_t start, uint32_t count, SharpAxis axis, int32_t* out) {
    uint16_t input[SHARP_ROW_WIDTH + 7u];
    const size_t step = depth > 8u ? 2u : 1u;
    const uint32_t span = count + axis.taps - 1u;
    start += axis.first;
    /* Clip each reference sample once, outside of the two convolution passes.
     * Never read the frame-buffer padding. The usual in-frame path is linear. */
    if (start >= 0 && start + span <= width) {
        const uint8_t* col = src + (size_t)start * step;
        for (uint32_t j = 0; j < span; ++j)
            input[j] = svt_av2_sample_load(col + (size_t)j * step, depth);
    } else {
        for (uint32_t j = 0; j < span; ++j) {
            const uint32_t x = clamp_sample_pos(start + j, width);
            input[j] = svt_av2_sample_load(src + (size_t)x * step, depth);
        }
    }
    /* A multiple of the rounding divisor keeps the right shift unsigned
     * in effect, even on negative filter lobes. Remove the corresponding
     * constant after the vertical pass (the kernel has unit DC gain). */
    const int32_t bias = 1 << (depth + 5u);
    const int16_t* f = axis.weights;
    if (axis.taps == 1u) {
        for (uint32_t x = 0; x < count; ++x) out[x] = (bias >> 2) + 16 * input[x];
    } else if (axis.taps == 4u) {
        for (uint32_t x = 0; x < count; ++x) {
            const int32_t sum = f[0]*input[x] + f[1]*input[x+1u] + f[2]*input[x+2u] + f[3]*input[x+3u];
            out[x] = (sum + bias + 2) >> 2;
        }
    } else {
        for (uint32_t x = 0; x < count; ++x) {
            const int32_t sum = f[0]*input[x] + f[1]*input[x+1u] + f[2]*input[x+2u] + f[3]*input[x+3u] +
                f[4]*input[x+4u] + f[5]*input[x+5u] + f[6]*input[x+6u] + f[7]*input[x+7u];
            out[x] = (sum + bias + 2) >> 2;
        }
    }
}

static uint16_t sharp_clip(int32_t value, int32_t limit) {
    return (uint16_t)(value < 0 ? 0 : value > limit ? limit : value);
}

static void sharp_vertical_row(const int32_t* const rows[8], uint32_t count,
    SharpAxis axis, uint32_t depth, uint16_t* out) {
    const int32_t range = 1 << depth;
    const int32_t v_origin = range + (range >> 1);
    const int32_t v_bias = (range << 10) + 512;
    const int16_t* f = axis.weights;
    if (axis.taps == 1u) {
        /* The identity vertical pass still has its own rounding. Combining it
         * with the horizontal round before this would shift threshold ties. */
        for (uint32_t x = 0; x < count; ++x)
            out[x] = sharp_clip(((rows[0][x] + 8) >> 4) - (range >> 1), range - 1);
    } else if (axis.taps == 4u) {
        for (uint32_t x = 0; x < count; ++x) {
            const int32_t sum = f[0]*rows[0][x] + f[1]*rows[1][x] + f[2]*rows[2][x] + f[3]*rows[3][x];
            out[x] = sharp_clip(((sum + v_bias) >> 10) - v_origin, range - 1);
        }
    } else {
        for (uint32_t x = 0; x < count; ++x) {
            const int32_t sum = f[0]*rows[0][x] + f[1]*rows[1][x] + f[2]*rows[2][x] + f[3]*rows[3][x] +
                f[4]*rows[4][x] + f[5]*rows[5][x] + f[6]*rows[6][x] + f[7]*rows[7][x];
            out[x] = sharp_clip(((sum + v_bias) >> 10) - v_origin, range - 1);
        }
    }
}

#if SVT_SHARP_HAVE_SSE2
typedef struct SharpVectorAxis {
    SharpAxis axis;
    __m128i pair[4];
} SharpVectorAxis;

static void sharp_vector_axis(SharpVectorAxis* vector, SharpAxis axis) {
    vector->axis = axis;
    for (uint32_t i = 0; i < axis.taps / 2u; ++i) {
        const int16_t a = axis.weights[2u * i], b = axis.weights[2u * i + 1u];
        vector->pair[i] = _mm_set_epi16(b, a, b, a, b, a, b, a);
    }
}

static inline void sharp_vector_madd(__m128i a, __m128i b, __m128i weights, __m128i* lo, __m128i* hi) {
    *lo = _mm_add_epi32(*lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), weights));
    *hi = _mm_add_epi32(*hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), weights));
}

static inline __m128i sharp_load_eight(const uint8_t* src, size_t step) {
    return step == 1u ? _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)src), _mm_setzero_si128()) :
        _mm_loadu_si128((const __m128i*)src);
}

static inline void sharp_horizontal_pair(const uint8_t* src, size_t step, __m128i center,
    __m128i weights, __m128i* lo, __m128i* hi) {
    sharp_vector_madd(_mm_xor_si128(sharp_load_eight(src, step), center),
        _mm_xor_si128(sharp_load_eight(src + step, step), center), weights, lo, hi);
}

/* Horizontal dot on arbitrary storage words remains within
 * [-28*65535,92*65535]; the Q6 bias and rounding fit signed 32-bit.
 * A separate representability check keeps vertical packing exact. */
static int sharp_narrow_tail(const int32_t* values, uint32_t first, uint32_t count) {
    for (uint32_t x = first; x < count; ++x)
        if (values[x] < INT16_MIN || values[x] > INT16_MAX) return 0;
    return 1;
}

static int sharp_horizontal_sse2(const uint8_t* src, uint32_t width, uint32_t depth, int64_t start,
    uint32_t count, const SharpVectorAxis* kernel, int check_narrow, int32_t* out) {
    const SharpAxis axis = kernel->axis;
    const size_t step = depth > 8u ? 2u : 1u;
    const int64_t first = start + axis.first;
    const int32_t bias = 1 << (depth + 5u);
    uint32_t x = 0;
    const __m128i zero = _mm_setzero_si128();
    __m128i range_bits = zero;
    if (count >= 8u && first >= 0 && first + count + axis.taps - 1u <= width) {
        const uint8_t* col = src + (size_t)first * step;
        const uint32_t stop = count & ~7u;
        const __m128i pack_bias = _mm_set1_epi32(32768);
        if (axis.taps == 1u) {
            const __m128i round_bias = _mm_set1_epi32(bias >> 2);
            for (; x < stop; x += 8u) {
                const __m128i samples = sharp_load_eight(col + (size_t)x * step, step);
                const __m128i lo = _mm_add_epi32(round_bias, _mm_slli_epi32(_mm_unpacklo_epi16(samples, zero), 4));
                const __m128i hi = _mm_add_epi32(round_bias, _mm_slli_epi32(_mm_unpackhi_epi16(samples, zero), 4));
                _mm_storeu_si128((__m128i*)(out + x), lo);
                _mm_storeu_si128((__m128i*)(out + x + 4u), hi);
                if (check_narrow) range_bits = _mm_or_si128(range_bits,
                    _mm_or_si128(_mm_add_epi32(lo, pack_bias), _mm_add_epi32(hi, pack_bias)));
            }
        } else {
            /* pmaddwd samples are signed. Center each unsigned 16-bit word
             * by 32768 and restore 64*32768 (unit DC gain) in the bias. This
             * also preserves full unsigned words outside the 10-bit range. */
            const __m128i center = _mm_set1_epi16(depth == 10u ? INT16_MIN : 0);
            const __m128i round_bias = _mm_set1_epi32(bias + 2 + (depth == 10u ? 64 * 32768 : 0));
            for (; x < stop; x += 8u) {
                const uint8_t* tap = col + (size_t)x * step;
                __m128i lo = round_bias, hi = round_bias;
                sharp_horizontal_pair(tap, step, center, kernel->pair[0], &lo, &hi);
                sharp_horizontal_pair(tap + 2u * step, step, center, kernel->pair[1], &lo, &hi);
                if (axis.taps == 8u) {
                    sharp_horizontal_pair(tap + 4u * step, step, center, kernel->pair[2], &lo, &hi);
                    sharp_horizontal_pair(tap + 6u * step, step, center, kernel->pair[3], &lo, &hi);
                }
                lo = _mm_srai_epi32(lo, 2);
                hi = _mm_srai_epi32(hi, 2);
                _mm_storeu_si128((__m128i*)(out + x), lo);
                _mm_storeu_si128((__m128i*)(out + x + 4u), hi);
                if (check_narrow) range_bits = _mm_or_si128(range_bits,
                    _mm_or_si128(_mm_add_epi32(lo, pack_bias), _mm_add_epi32(hi, pack_bias)));
            }
        }
    }
    /* Any clipped horizontal footprint stays on the old edge replication
     * loop. A SIMD load never crosses its exact last tap or touches padding. */
    if (x < count) sharp_horizontal_row(src, width, depth, start + x, count - x, axis, out + x);
    if (!check_narrow) return 1;
    return _mm_movemask_epi8(_mm_cmpeq_epi32(_mm_srli_epi32(range_bits, 16), zero)) == 0xffff &&
        sharp_narrow_tail(out, x, count);
}

static inline __m128i sharp_pack_row(const int32_t* row) {
    return _mm_packs_epi32(_mm_loadu_si128((const __m128i*)row), _mm_loadu_si128((const __m128i*)(row + 4u)));
}

static void sharp_vertical_sse2(const int32_t* const rows[8], uint32_t count,
    const SharpVectorAxis* kernel, uint32_t depth, int narrow, uint16_t* out) {
    const SharpAxis axis = kernel->axis;
    if (count < 8u || (!narrow && axis.taps != 1u)) {
        sharp_vertical_row(rows, count, axis, depth, out);
        return;
    }
    /* Across every Q6 row, sum of negative magnitudes <=28 and positive
     * sum <=92. With legal b-bit pixels (D=2^b), the horizontal output
     * ((32*D + dot + 2) >> 2) is in [263,7913] for 8b, [1031,31721]
     * for 10b (the identity falls inside). Thus signed 16-bit row packing
     * is exact; nonconforming 10b words were checked before entering here.
     * Even an arbitrary signed-16 vertical input has |dot| <=120*32768;
     * adding 1024*D+512 cannot overflow an int32. */
    const int32_t range = 1 << depth;
    const __m128i zero = _mm_setzero_si128();
    const __m128i limit = _mm_set1_epi16((int16_t)(range - 1));
    const __m128i origin = _mm_set1_epi32(axis.taps == 1u ? range >> 1 : range + (range >> 1));
    const __m128i rounding = _mm_set1_epi32(axis.taps == 1u ? 8 : (range << 10) + 512);
    uint32_t x = 0;
    const uint32_t stop = count & ~7u;
    for (; x < stop; x += 8u) {
        __m128i lo = rounding, hi = rounding;
        if (axis.taps == 1u) {
            lo = _mm_srai_epi32(_mm_add_epi32(lo, _mm_loadu_si128((const __m128i*)(rows[0] + x))), 4);
            hi = _mm_srai_epi32(_mm_add_epi32(hi, _mm_loadu_si128((const __m128i*)(rows[0] + x + 4u))), 4);
        } else {
            sharp_vector_madd(sharp_pack_row(rows[0] + x), sharp_pack_row(rows[1] + x), kernel->pair[0], &lo, &hi);
            sharp_vector_madd(sharp_pack_row(rows[2] + x), sharp_pack_row(rows[3] + x), kernel->pair[1], &lo, &hi);
            if (axis.taps == 8u) {
                sharp_vector_madd(sharp_pack_row(rows[4] + x), sharp_pack_row(rows[5] + x), kernel->pair[2], &lo, &hi);
                sharp_vector_madd(sharp_pack_row(rows[6] + x), sharp_pack_row(rows[7] + x), kernel->pair[3], &lo, &hi);
            }
            lo = _mm_srai_epi32(lo, 10);
            hi = _mm_srai_epi32(hi, 10);
        }
        /* Packing first is safe even for the wide identity result: signed
         * saturation is outside the [0,limit] interval of the final clip. */
        const __m128i pack = _mm_packs_epi32(_mm_sub_epi32(lo, origin), _mm_sub_epi32(hi, origin));
        _mm_storeu_si128((__m128i*)(out + x), _mm_min_epi16(_mm_max_epi16(pack, zero), limit));
    }
    if (x < count) {
        const int32_t* tail[8];
        for (uint32_t i = 0; i < axis.taps; ++i) tail[i] = rows[i] + x;
        sharp_vertical_row(tail, count - x, axis, depth, out + x);
    }
}
#endif

typedef struct SharpLumaSad {
    const uint8_t* source;
    uint32_t stride;
    uint64_t limit, sad;
} SharpLumaSad;

static uint64_t sharp_row_sad(const uint8_t* source, const uint16_t* predicted, uint32_t count,
    uint32_t depth) {
    const size_t step = depth > 8u ? 2u : 1u;
    uint64_t total = 0;
    uint32_t x = 0;
#if SVT_SHARP_HAVE_SSE2
    __m128i sum = _mm_setzero_si128();
    const __m128i zero = sum;
    for (; x + 8u <= count; x += 8u) {
        const __m128i a = sharp_load_eight(source + (size_t)x * step, step);
        const __m128i b = _mm_loadu_si128((const __m128i*)(predicted + x));
        const __m128i diff = _mm_or_si128(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a));
        sum = _mm_add_epi32(sum, _mm_add_epi32(_mm_unpacklo_epi16(diff, zero), _mm_unpackhi_epi16(diff, zero)));
    }
    /* count <= SHARP_ROW_WIDTH. Even arbitrary uint16 words sum below 2^22. */
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
    total = (uint32_t)_mm_cvtsi128_si32(sum);
#endif
    for (; x < count; ++x) {
        const int32_t d = (int32_t)svt_av2_sample_load(source + (size_t)x * step, depth) - predicted[x];
        total += (uint32_t)(d < 0 ? -d : d);
    }
    return total;
}

#if SVT_AV2_SHARP_INTEGER_SAD
/* The integer identity does not need a predictor store/load. Compare the
 * visible reference bytes directly; only a complete in-row vector is read. */
static uint64_t sharp_storage_row_sad(const uint8_t* src, const uint8_t* ref, uint32_t count,
    uint32_t depth) {
    const size_t step = depth > 8u ? 2u : 1u;
    uint64_t total = 0u;
    uint32_t x = 0u;
#if SVT_SHARP_HAVE_SSE2
    __m128i sum = _mm_setzero_si128();
    if (step == 1u) {
        for (; x + 16u <= count; x += 16u) {
            const __m128i a = _mm_loadu_si128((const __m128i*)(src + x));
            const __m128i b = _mm_loadu_si128((const __m128i*)(ref + x));
            sum = _mm_add_epi32(sum, _mm_sad_epu8(a, b));
        }
        total = (uint32_t)_mm_cvtsi128_si32(_mm_add_epi32(sum, _mm_srli_si128(sum, 8)));
    } else {
        const __m128i zero = sum;
        for (; x + 8u <= count; x += 8u) {
            const __m128i a = _mm_loadu_si128((const __m128i*)(src + 2u * x));
            const __m128i b = _mm_loadu_si128((const __m128i*)(ref + 2u * x));
            const __m128i diff = _mm_or_si128(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a));
            sum = _mm_add_epi32(sum, _mm_add_epi32(_mm_unpacklo_epi16(diff, zero), _mm_unpackhi_epi16(diff, zero)));
        }
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
        total = (uint32_t)_mm_cvtsi128_si32(_mm_add_epi32(sum, _mm_srli_si128(sum, 4)));
    }
#endif
    for (; x < count; ++x) {
        const int32_t d = (int32_t)svt_av2_sample_load(src + x * step, depth) -
            svt_av2_sample_load(ref + x * step, depth);
        total += (uint32_t)(d < 0 ? -d : d);
    }
    return total;
}
#endif

static int sharp_sad_sum(SharpLumaSad* screen, uint64_t row_sad) {
    screen->sad += row_sad;
    if (screen->sad <= screen->limit) return 0;
    screen->sad = screen->limit + 1u; /* a strict exceedance also proves limit is not UINT64_MAX */
    return 1;
}

static int sharp_sad_accumulate(SharpLumaSad* screen, const uint16_t* predicted, uint32_t row,
    uint32_t width, uint32_t depth) {
    return sharp_sad_sum(screen, sharp_row_sad(screen->source + (size_t)row * screen->stride,
        predicted, width, depth));
}

static SvtAv2Error predict_eighth_sharp(const SvtAv2FrameBuffer* reference, uint32_t plane,
    const SvtAv2BlockView* block, int64_t mv_x, int64_t mv_y, uint16_t* prediction,
    uint32_t prediction_stride, SharpLumaSad* screen) {
    if (reference == NULL || block == NULL || reference->buffer == NULL ||
        plane >= 3u || reference->planes[plane] == NULL ||
        (screen == NULL && (prediction == NULL || prediction_stride < block->w)) ||
        block->w == 0u || block->h == 0u || (reference->bit_depth != 8u && reference->bit_depth != 10u))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    const uint32_t w = svt_av2_picture_plane_width(reference, plane);
    const uint32_t h = svt_av2_picture_plane_height(reference, plane);
    if (block->x >= w || block->y >= h || block->w > w - block->x || block->h > h - block->y)
        return SVT_AV2_ERROR_INVALID_PARAMETER;

    const int q4_scale = plane != 0u && reference->chroma_format == SVT_AV2_CHROMA_420 ? 1 : 2;
    uint32_t fx, fy;
    const int64_t x = (int64_t)block->x + split_sixteenth_offset(mv_x * q4_scale, &fx);
    const int64_t y = (int64_t)block->y + split_sixteenth_offset(mv_y * q4_scale, &fy);
    const uint32_t depth = reference->bit_depth, stride = reference->strides[plane];
    const uint8_t* base = reference->planes[plane];
    uint16_t scratch[SHARP_ROW_WIDTH];
    if (fx == 0u && fy == 0u) {
        if (screen != NULL) {
            const size_t step = depth > 8u ? 2u : 1u;
            const int visible = x >= 0 && x + block->w <= w;
            for (uint32_t row = 0; row < block->h; ++row) {
                const uint8_t* ref_row = base + (size_t)clamp_sample_pos(y + row, h) * stride;
                if (visible) {
                    ref_row += (size_t)x * step;
#if SVT_AV2_SHARP_INTEGER_SAD
                    if (sharp_sad_sum(screen, sharp_storage_row_sad(screen->source + (size_t)row * screen->stride,
                            ref_row, block->w, depth))) break;
                    continue;
#else
                    for (uint32_t i = 0; i < block->w; ++i)
                        scratch[i] = svt_av2_sample_load(ref_row + (size_t)i * step, depth);
#endif
                } else {
                    for (uint32_t i = 0; i < block->w; ++i)
                        scratch[i] = svt_av2_sample_load(ref_row + (size_t)clamp_sample_pos(x + i, w) * step, depth);
                }
                if (sharp_sad_accumulate(screen, scratch, row, block->w, depth)) break;
            }
            return SVT_AV2_ERROR_NONE;
        }
        /* Whole-plane phases are exactly representable in half-luma units,
         * including the half-pel caller's signed endpoints. */
        const SvtAv2HalfPelMotionVector half = {(int32_t)(mv_x / 4), (int32_t)(mv_y / 4)};
        return svt_av2_inter_predict_half(reference, plane, block, half, prediction, prediction_stride);
    }

    const SharpAxis horiz = sharp_axis(fx, block->w);
    const SharpAxis vert = sharp_axis(fy, block->h);
    const int64_t top = y + vert.first;
    int32_t intermediate[8][SHARP_ROW_WIDTH];
#if SVT_SHARP_HAVE_SSE2
    SharpVectorAxis horizontal, vertical;
    int row_narrow[8];
    const int check_narrow = depth == 10u && vert.taps > 1u;
    sharp_vector_axis(&horizontal, horiz);
    sharp_vector_axis(&vertical, vert);
#endif
    /* A ring holds only the rows the vertical filter still needs. Width
     * tiling bounds stack use even for a larger block; filter choice is based
     * on the whole block, not this implementation's chunk dimensions. */
    for (uint32_t col = 0u; col < block->w;) {
        const uint32_t remaining = block->w - col;
        const uint32_t count = remaining < SHARP_ROW_WIDTH ? remaining : SHARP_ROW_WIDTH;
        for (uint32_t k = 0u; k + 1u < vert.taps; ++k) {
            const uint32_t iy = clamp_sample_pos(top + k, h);
#if SVT_SHARP_HAVE_SSE2
            row_narrow[k] = sharp_horizontal_sse2(base + (size_t)iy * stride, w, depth, x + col,
                count, &horizontal, check_narrow, intermediate[k]);
#else
            sharp_horizontal_row(base + (size_t)iy * stride, w, depth, x + col, count, horiz, intermediate[k]);
#endif
        }
        for (uint32_t row = 0u; row < block->h; ++row) {
            const uint32_t last = row + vert.taps - 1u;
            const uint32_t iy = clamp_sample_pos(top + last, h);
            const int32_t* rows[8];
            uint16_t* const dst = screen != NULL ? scratch : prediction + (size_t)row * prediction_stride + col;
#if SVT_SHARP_HAVE_SSE2
            row_narrow[last & 7u] = sharp_horizontal_sse2(base + (size_t)iy * stride, w, depth, x + col,
                count, &horizontal, check_narrow, intermediate[last & 7u]);
            int narrow = 1;
            for (uint32_t k = 0u; k < vert.taps; ++k) {
                const uint32_t index = (row + k) & 7u;
                rows[k] = intermediate[index];
                narrow &= row_narrow[index];
            }
            sharp_vertical_sse2(rows, count, &vertical, depth, narrow, dst);
#else
            sharp_horizontal_row(base + (size_t)iy * stride, w, depth, x + col, count, horiz, intermediate[last & 7u]);
            for (uint32_t k = 0u; k < vert.taps; ++k) rows[k] = intermediate[(row + k) & 7u];
            sharp_vertical_row(rows, count, vert, depth, dst);
#endif
            /* Screen mode accepts one width chunk and never observes scratch
             * after a candidate is proven worse than the incumbent. */
            if (screen != NULL && sharp_sad_accumulate(screen, scratch, row, count, depth))
                return SVT_AV2_ERROR_NONE;
        }
        col += count;
    }
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_inter_predict_half_sharp(const SvtAv2FrameBuffer* reference, uint32_t plane,
    const SvtAv2BlockView* block, SvtAv2HalfPelMotionVector mv, uint16_t* prediction,
    uint32_t prediction_stride) {
    return predict_eighth_sharp(reference, plane, block, 4 * (int64_t)mv.mv_x, 4 * (int64_t)mv.mv_y,
        prediction, prediction_stride, NULL);
}

SvtAv2Error svt_av2_inter_predict_quarter_sharp(const SvtAv2FrameBuffer* reference, uint32_t plane,
    const SvtAv2BlockView* block, SvtAv2QuarterPelMotionVector mv, uint16_t* prediction,
    uint32_t prediction_stride) {
    return predict_eighth_sharp(reference, plane, block, 2 * (int64_t)mv.mv_x, 2 * (int64_t)mv.mv_y,
        prediction, prediction_stride, NULL);
}

SvtAv2Error svt_av2_inter_predict_eighth_sharp(const SvtAv2FrameBuffer* reference, uint32_t plane,
    const SvtAv2BlockView* block, SvtAv2EighthPelMotionVector mv, uint16_t* prediction,
    uint32_t prediction_stride) {
    return predict_eighth_sharp(reference, plane, block, mv.mv_x, mv.mv_y, prediction, prediction_stride, NULL);
}

static SvtAv2Error sharp_luma_sad(const SvtAv2FrameBuffer* source,
    const SvtAv2FrameBuffer* reference, const SvtAv2BlockView* block,
    int64_t mv_x, int64_t mv_y, uint64_t limit, uint64_t* sad) {
    if (source == NULL || reference == NULL || block == NULL || sad == NULL || source->buffer == NULL ||
        source->planes[0] == NULL || source->visible_width != reference->visible_width ||
        source->visible_height != reference->visible_height || source->bit_depth != reference->bit_depth ||
        source->chroma_format != reference->chroma_format || block->w == 0u || block->w > SHARP_ROW_WIDTH ||
        block->h == 0u || block->x >= source->visible_width || block->y >= source->visible_height ||
        block->w > source->visible_width - block->x || block->h > source->visible_height - block->y)
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    const size_t step = source->bit_depth > 8u ? 2u : 1u;
    SharpLumaSad screen = {source->planes[0] + (size_t)block->y * source->strides[0] + (size_t)block->x * step,
        source->strides[0], limit, 0u};
    const SvtAv2Error error = predict_eighth_sharp(reference, 0u, block, mv_x, mv_y, NULL, 0u, &screen);
    if (error == SVT_AV2_ERROR_NONE) *sad = screen.sad;
    return error;
}

SvtAv2Error svt_av2_inter_luma_sad_quarter_sharp(const SvtAv2FrameBuffer* source,
    const SvtAv2FrameBuffer* reference, const SvtAv2BlockView* block,
    SvtAv2QuarterPelMotionVector mv, uint64_t limit, uint64_t* sad) {
    return sharp_luma_sad(source, reference, block, 2 * (int64_t)mv.mv_x, 2 * (int64_t)mv.mv_y, limit, sad);
}

SvtAv2Error svt_av2_inter_luma_sad_eighth_sharp(const SvtAv2FrameBuffer* source,
    const SvtAv2FrameBuffer* reference, const SvtAv2BlockView* block,
    SvtAv2EighthPelMotionVector mv, uint64_t limit, uint64_t* sad) {
    return sharp_luma_sad(source, reference, block, mv.mv_x, mv.mv_y, limit, sad);
}
