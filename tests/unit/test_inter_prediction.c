#include "test_util.h"
#include "inter_prediction.h"
#include <stdint.h>
#include <string.h>

static uint32_t edge_sample(const SvtAv2FrameBuffer* ref, uint32_t p, int x, int y) {
    const int width = (int)svt_av2_picture_plane_width(ref, p), height = (int)svt_av2_picture_plane_height(ref, p);
    x = x < 0 ? 0 : x >= width ? width - 1 : x;
    y = y < 0 ? 0 : y >= height ? height - 1 : y;
    const uint8_t* q = ref->planes[p] + (size_t)y * ref->strides[p] + (size_t)x * (ref->bit_depth == 8 ? 1u : 2u);
    return ref->bit_depth == 8 ? q[0] : (uint32_t)q[0] + 256u * q[1];
}
/* Independent full 8-tap separable filter with the fixed v1.0.0 BILINEAR
 * table entries for Q4 phases 0, 4, 8, 12 and AVM single-reference offsets. */
static uint16_t filter_model(const SvtAv2FrameBuffer* ref, uint32_t p, int x, int y, int mx, int my) {
    static const int filter[4][8] = {
        { 0, 0, 0, 128,  0, 0, 0, 0 },
        { 0, 0, 0,  96, 32, 0, 0, 0 },
        { 0, 0, 0,  64, 64, 0, 0, 0 },
        { 0, 0, 0,  32, 96, 0, 0, 0 },
    };
    const int q4_step = p != 0 && ref->chroma_format == SVT_AV2_CHROMA_420 ? 4 : 8;
    const int qx = 16 * x + q4_step * mx, qy = 16 * y + q4_step * my;
    const int rx = (qx < 0 ? qx - 15 : qx) / 16, ry = (qy < 0 ? qy - 15 : qy) / 16;
    const int* hfilter = filter[(qx - rx * 16) / 4];
    const int* vfilter = filter[(qy - ry * 16) / 4];
    const uint32_t offset_bits = ref->bit_depth + 11;
    int64_t vertical = (int64_t)1 << offset_bits;
    for (int j = 0; j < 8; ++j) {
        int64_t horizontal = (int64_t)1 << (ref->bit_depth + 6);
        for (int i = 0; i < 8; ++i)
            horizontal += (int64_t)hfilter[i] * edge_sample(ref, p, rx + i - 3, ry + j - 3);
        horizontal = (horizontal + 4) >> 3;
        vertical += (int64_t)vfilter[j] * horizontal;
    }
    const int64_t value = ((vertical + 1024) >> 11) -
        (((int64_t)1 << (offset_bits - 11)) + ((int64_t)1 << (offset_bits - 12)));
    return (uint16_t)value;
}

static void check_half_blocks(const SvtAv2FrameBuffer* ref, uint32_t p) {
    const uint32_t w = svt_av2_picture_plane_width(ref, p), h = svt_av2_picture_plane_height(ref, p);
    const SvtAv2BlockView blocks[] = {
        { 0, 0, 4, 4, NULL },             /* top-left edge */
        { w - 4, h - 4, 4, 4, NULL },     /* bottom-right edge */
        { 4, 4, 2, 2, NULL },            /* 420 chroma from a 4x4 luma sub-block */
        { 4, 4, 4, 8, NULL },            /* 4-tap width threshold */
        { 4, 4, 8, 4, NULL },            /* 4-tap height threshold */
    };
    const uint32_t stride = 9;
    for (size_t b = 0; b < sizeof(blocks) / sizeof(blocks[0]); ++b) {
        const SvtAv2BlockView* block = &blocks[b];
        for (int my = -7; my <= 7; ++my) for (int mx = -7; mx <= 7; ++mx) {
            const SvtAv2HalfPelMotionVector mv = { mx, my };
            uint16_t pred[8 * 9], legacy[8 * 9];
            for (size_t i = 0; i < sizeof(pred) / sizeof(pred[0]); ++i) pred[i] = 0xA5A5;
            SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_half(ref, p, block, mv, pred, stride), SVT_AV2_ERROR_NONE);
            const int is_integer_luma = mx % 2 == 0 && my % 2 == 0;
            if (is_integer_luma) {
                const SvtAv2MotionVector integer_mv = { mx / 2, my / 2 };
                SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict(ref, p, block, integer_mv, legacy, stride), SVT_AV2_ERROR_NONE);
            }
            for (uint32_t y = 0; y < block->h; ++y) {
                for (uint32_t x = 0; x < block->w; ++x) {
                    const uint16_t actual = pred[y * stride + x];
                    SVT_TEST_CHECK_EQ_INT(actual, filter_model(ref, p, (int)(block->x + x), (int)(block->y + y), mx, my));
                    if (is_integer_luma) SVT_TEST_CHECK_EQ_INT(actual, legacy[y * stride + x]);
                }
                for (uint32_t x = block->w; x < stride; ++x)
                    SVT_TEST_CHECK_EQ_INT(pred[y * stride + x], 0xA5A5);
            }
        }
    }
}

static void check_half_rounding_and_umv(void) {
    SvtAv2FrameBuffer ref;
    SVT_TEST_CHECK_EQ_INT(svt_av2_picture_alloc_ex(&ref, 8, 8, 8, SVT_AV2_CHROMA_420), SVT_AV2_ERROR_NONE);
    const SvtAv2BlockView block = { 0, 0, 4, 4, NULL };
    const SvtAv2HalfPelMotionVector half_x = { 1, 0 }, half_xy = { 1, 1 };
    uint16_t pred[16] = { 0 };
    for (uint32_t p = 0; p < 2; ++p) {
        uint8_t* plane = ref.planes[p];
        const uint32_t stride = ref.strides[p], first = stride + 1;
        /* Fractional horizontal result is exactly 1/2. Rounding it early to
         * 1 would make the diagonal prediction 1 instead of 0 on both planes. */
        plane[first + 1] = (uint8_t)(p == 0 ? 1 : 2);
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_half(&ref, p, &block, half_x, pred, 4), SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK_EQ_INT(pred[5], 1);
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_half(&ref, p, &block, half_xy, pred, 4), SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK_EQ_INT(pred[5], 0);
        /* Now make the diagonal result exactly 1/2: ties round up. */
        plane[first + stride + 1] = plane[first + 1];
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_half(&ref, p, &block, half_xy, pred, 4), SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK_EQ_INT(pred[5], 1);

        const uint32_t side = p == 0 ? 8 : 4;
        const uint32_t last_row = (side - 1) * stride;
        plane[0] = 17;
        plane[side - 1] = 41;
        plane[last_row] = 83;
        plane[last_row + side - 1] = 239;
        const SvtAv2HalfPelMotionVector corners[] = {
            { INT32_MIN, INT32_MIN }, { INT32_MAX, INT32_MIN },
            { INT32_MIN, INT32_MAX }, { INT32_MAX, INT32_MAX },
        };
        const uint16_t expected[] = { 17, 41, 83, 239 };
        for (size_t c = 0; c < sizeof(corners) / sizeof(corners[0]); ++c) {
            SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_half(&ref, p, &block, corners[c], pred, 4), SVT_AV2_ERROR_NONE);
            for (size_t i = 0; i < sizeof(pred) / sizeof(pred[0]); ++i) SVT_TEST_CHECK_EQ_INT(pred[i], expected[c]);
        }
    }
    svt_av2_picture_free(&ref);
}
/* AV2 v1.0.0 block inter prediction process, in full Q7 units. Use the
 * specification's signed Round2 directly, without implementation biases, row
 * caching or any shared offset/table representation. */
static int64_t spec_round2(int64_t value, int64_t divisor) {
    const int64_t biased = value + divisor/2;
    return (biased >= 0 ? biased : biased - (divisor - 1)) / divisor;
}
static int64_t sharp_eighth_oracle(const SvtAv2FrameBuffer* ref, uint32_t p,
    const SvtAv2BlockView* block, uint32_t x, uint32_t y, int64_t eighth_x, int64_t eighth_y) {
    static const int q7[2][8][8] = {
        {{0,0,0,128,0,0,0,0}, {-2,6,-12,124,16,-6,4,-2},
         {-4,10,-22,116,38,-14,6,-2}, {-4,10,-24,100,60,-20,8,-2},
         {-4,12,-24,80,80,-24,12,-4}, {-2,8,-20,60,100,-24,10,-4},
         {-2,6,-14,38,116,-22,10,-4}, {-2,4,-6,16,124,-12,6,-2}},
        {{0,0,0,128,0,0,0,0}, {0,0,-8,122,18,-4,0,0},
         {0,0,-12,110,38,-8,0,0}, {0,0,-14,94,58,-10,0,0},
         {0,0,-12,76,76,-12,0,0}, {0,0,-10,58,94,-14,0,0},
         {0,0,-8,38,110,-12,0,0}, {0,0,-4,18,122,-8,0,0}},
    };
    static const int q7_odd[2][8][8] = {
        {{-2,2,-6,126,8,-2,2,0}, {-2,8,-18,120,26,-10,6,-2},
         {-4,10,-22,108,48,-18,8,-2}, {-4,10,-24,90,70,-22,10,-2},
         {-2,10,-22,70,90,-24,10,-4}, {-2,8,-18,48,108,-22,10,-4},
         {-2,6,-10,26,120,-18,8,-2}, {0,2,-2,8,126,-6,2,-2}},
        {{0,0,-4,126,8,-2,0,0}, {0,0,-10,116,28,-6,0,0},
         {0,0,-12,102,48,-10,0,0}, {0,0,-12,84,66,-10,0,0},
         {0,0,-10,66,84,-12,0,0}, {0,0,-10,48,102,-12,0,0},
         {0,0,-6,28,116,-10,0,0}, {0,0,-2,8,126,-4,0,0}},
    };
    const int step = p != 0u && ref->chroma_format == SVT_AV2_CHROMA_420 ? 1 : 2;
    const int64_t qx = 16 * (int64_t)(block->x + x) + step * eighth_x;
    const int64_t qy = 16 * (int64_t)(block->y + y) + step * eighth_y;
    const int64_t rx = (qx < 0 ? qx - 15 : qx) / 16, ry = (qy < 0 ? qy - 15 : qy) / 16;
    const uint32_t fx = (uint32_t)(qx - rx*16), fy = (uint32_t)(qy - ry*16);
    const int* xf = (fx & 1u ? q7_odd : q7)[block->w <= 4u][fx/2];
    const int* yf = (fy & 1u ? q7_odd : q7)[block->h <= 4u][fy/2];
    const int width = (int)svt_av2_picture_plane_width(ref, p), height = (int)svt_av2_picture_plane_height(ref, p);
    int64_t v = 0;
    for (int j = 0; j < 8; ++j) {
        if (yf[j] == 0) continue;
        const int64_t sy = ry + j - 3;
        const int py = sy < 0 ? 0 : sy >= height ? height - 1 : (int)sy;
        int64_t h = 0;
        for (int i = 0; i < 8; ++i) {
            if (xf[i] == 0) continue;
            const int64_t sx = rx + i - 3;
            const int px = sx < 0 ? 0 : sx >= width ? width - 1 : (int)sx;
            h += (int64_t)xf[i] * edge_sample(ref, p, px, py);
        }
        v += yf[j] * spec_round2(h, 8);
    }
    return spec_round2(v, 2048);
}

static int64_t sharp_oracle(const SvtAv2FrameBuffer* ref, uint32_t p,
    const SvtAv2BlockView* block, uint32_t x, uint32_t y, int64_t quarter_x, int64_t quarter_y) {
    return sharp_eighth_oracle(ref, p, block, x, y, 2 * quarter_x, 2 * quarter_y);
}

static void check_quarter_sharp_block(const SvtAv2FrameBuffer* ref, uint32_t plane,
    const SvtAv2BlockView* block, SvtAv2QuarterPelMotionVector mv,
    uint16_t* pred, uint16_t* half, uint32_t stride) {
    const uint32_t limit = (1u << ref->bit_depth) - 1u;
    const int also_half = mv.mv_x % 2 == 0 && mv.mv_y % 2 == 0;
    for (uint32_t y = 0u; y < block->h; ++y)
        for (uint32_t x = 0u; x < stride; ++x) pred[y * stride + x] = 0xA5A5;
    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_quarter_sharp(ref, plane, block, mv, pred, stride),
        SVT_AV2_ERROR_NONE);
    if (also_half) {
        const SvtAv2HalfPelMotionVector half_mv = {mv.mv_x / 2, mv.mv_y / 2};
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_half_sharp(ref, plane, block, half_mv, half, stride),
            SVT_AV2_ERROR_NONE);
    }
    for (uint32_t y = 0u; y < block->h; ++y) {
        for (uint32_t x = 0u; x < block->w; ++x) {
            const int64_t oracle = sharp_oracle(ref, plane, block, x, y, mv.mv_x, mv.mv_y);
            const uint16_t expected = (uint16_t)(oracle < 0 ? 0 : oracle > limit ? limit : oracle);
            SVT_TEST_CHECK_EQ_INT(pred[y * stride + x], expected);
            if (also_half) SVT_TEST_CHECK_EQ_INT(pred[y * stride + x], half[y * stride + x]);
        }
        for (uint32_t x = block->w; x < stride; ++x)
            SVT_TEST_CHECK_EQ_INT(pred[y * stride + x], 0xA5A5);
    }
}

static void check_sharp(void) {
    enum { W = 144, H = 80, IN_STRIDE = W*2+5, OUT_STRIDE = 72 };
    static uint8_t storage[3][H * IN_STRIDE + 1u];
    static uint16_t pred[64 * OUT_STRIDE], identity[64 * OUT_STRIDE];
    static const SvtAv2HalfPelMotionVector moves[] = {
        {0,0}, {4,-8}, {1,0}, {-1,0}, {2,0}, {3,0}, {0,1}, {0,-1}, {0,2}, {0,3},
        {1,1}, {1,2}, {1,3}, {2,1}, {2,2}, {2,3}, {3,1}, {3,2}, {3,3},
        {-1,1}, {1,-1}, {-3,-2}, {-31,17}, {-32,-32}, {29,32},
        {INT32_MIN,INT32_MAX}, {INT32_MAX,INT32_MIN}, {INT32_MAX,INT32_MAX},
    };
    static const SvtAv2QuarterPelMotionVector quarter_edges[] = {
        {0,0}, {-8,16}, {1,-1}, {-3,3}, {5,7}, {-63,37}, {-64,-64}, {61,64},
        {INT32_MIN,INT32_MIN}, {INT32_MAX,INT32_MIN}, {INT32_MIN,INT32_MAX},
        {INT32_MAX,INT32_MAX}, {INT32_MIN + 1,INT32_MAX - 1},
    };
    uint32_t seed = 0x545a1106u;
    for (uint32_t variant = 0u; variant < 3u; ++variant) {
        const uint32_t depth = variant == 0u ? 8u : 10u;
        const SvtAv2ChromaFormat format = variant == 2u ? SVT_AV2_CHROMA_444 : SVT_AV2_CHROMA_420;
        const uint32_t limit = (1u << depth) - 1u, step = depth == 8u ? 1u : 2u;
        SvtAv2FrameBuffer ref;
        const SvtAv2Error allocated = svt_av2_picture_alloc_ex(&ref, W, H, depth, format);
        SVT_TEST_CHECK_EQ_INT(allocated, SVT_AV2_ERROR_NONE);
        if (allocated != SVT_AV2_ERROR_NONE) continue;
        for (uint32_t p = 0u; p < 3u; ++p) {
            ref.planes[p] = storage[p] + 1u;
            ref.strides[p] = IN_STRIDE; /* Odd byte alignment and poisoned padding. */
            memset(storage[p], 0xff, sizeof(storage[p]));
            const uint32_t w = svt_av2_picture_plane_width(&ref, p), h = svt_av2_picture_plane_height(&ref, p);
            for (uint32_t y = 0u; y < h; ++y) for (uint32_t x = 0u; x < w; ++x) {
                seed = seed * 1664525u + 1013904223u;
                const uint32_t value = (x < w/2u) ? (seed >> 11) & limit : (seed >> 31) * limit;
                svt_av2_sample_store(ref.planes[p] + (size_t)y * IN_STRIDE + x * step, depth, (uint16_t)value);
            }
            const SvtAv2BlockView blocks[] = {
                {0,0,4,4,NULL}, {w-4u,h-4u,4,4,NULL}, {7,9,2,2,NULL},
                {4,4,4,8,NULL}, {6,7,8,4,NULL}, {0,h-16u,16,16,NULL},
                {16,0,32,32,NULL}, {w-64u,h-(p==0u || format==SVT_AV2_CHROMA_444 ? 64u : 32u),
                    64,p==0u || format==SVT_AV2_CHROMA_444 ? 64u : 32u,NULL},
                {w-67u,8,67,9,NULL}, /* Last strip is small, but the filter is not. */
            };
            uint32_t clipped_low = 0u, clipped_high = 0u;
            for (size_t b = 0u; b < sizeof(blocks)/sizeof(blocks[0]); ++b) {
                const SvtAv2BlockView* block = &blocks[b];
                for (size_t m = 0u; m < sizeof(moves)/sizeof(moves[0]); ++m) {
                    for (size_t i = 0u; i < sizeof(pred)/sizeof(pred[0]); ++i) pred[i] = 0xA5A5;
                    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_half_sharp(&ref, p, block, moves[m], pred, OUT_STRIDE),
                        SVT_AV2_ERROR_NONE);
                    const uint32_t modulus = p && format == SVT_AV2_CHROMA_420 ? 4u : 2u;
                    const int is_identity_phase = (uint32_t)moves[m].mv_x % modulus == 0u &&
                                                  (uint32_t)moves[m].mv_y % modulus == 0u;
                    if (is_identity_phase)
                        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_half(&ref, p, block, moves[m], identity, OUT_STRIDE),
                            SVT_AV2_ERROR_NONE);
                    for (uint32_t y = 0u; y < block->h; ++y) {
                        for (uint32_t x = 0u; x < block->w; ++x) {
                            const int64_t oracle = sharp_oracle(&ref, p, block, x, y,
                                2 * (int64_t)moves[m].mv_x, 2 * (int64_t)moves[m].mv_y);
                            const uint16_t expected = (uint16_t)(oracle < 0 ? 0 : oracle > limit ? limit : oracle);
                            clipped_low += oracle < 0;
                            clipped_high += oracle > limit;
                            SVT_TEST_CHECK_EQ_INT(pred[y * OUT_STRIDE + x], expected);
                            if (is_identity_phase)
                                SVT_TEST_CHECK_EQ_INT(pred[y * OUT_STRIDE + x], identity[y * OUT_STRIDE + x]);
                        }
                        for (uint32_t x = block->w; x < OUT_STRIDE; ++x)
                            SVT_TEST_CHECK_EQ_INT(pred[y * OUT_STRIDE + x], 0xA5A5);
                    }
                }
            }
            SVT_TEST_CHECK(clipped_low > 0u && clipped_high > 0u);
            for (size_t b = 0u; b < sizeof(blocks)/sizeof(blocks[0]); ++b) {
                const SvtAv2BlockView* block = &blocks[b];
                if (block->w <= 8u && block->h <= 8u) {
                    /* Every x/y phase pair, including negative non-integers
                     * on the top-left border and both 4/8-tap orientations. */
                    for (int sy = -7; sy <= 0; ++sy) for (int sx = 0; sx <= 7; ++sx) {
                        const SvtAv2QuarterPelMotionVector mv = {b == 0u ? -sx : sx, sy};
                        check_quarter_sharp_block(&ref, p, block, mv, pred, identity, OUT_STRIDE);
                    }
                }
                for (size_t m = 0u; m < sizeof(quarter_edges)/sizeof(quarter_edges[0]); ++m)
                    check_quarter_sharp_block(&ref, p, block, quarter_edges[m], pred, identity, OUT_STRIDE);
            }
            SvtAv2BlockView bad = blocks[0];
            bad.x = w - 2u;
            SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_half_sharp(&ref, p, &bad, moves[0], pred, OUT_STRIDE),
                SVT_AV2_ERROR_INVALID_PARAMETER);
            SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_half_sharp(&ref, p, &blocks[0], moves[0], pred, 3u),
                SVT_AV2_ERROR_INVALID_PARAMETER);
            SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_quarter_sharp(&ref, p, &bad, quarter_edges[0], pred, OUT_STRIDE),
                SVT_AV2_ERROR_INVALID_PARAMETER);
            SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_quarter_sharp(&ref, p, &blocks[0], quarter_edges[0], pred, 3u),
                SVT_AV2_ERROR_INVALID_PARAMETER);
        }
        svt_av2_picture_free(&ref);
    }
}

static void check_sharp_vector_footprints(void) {
    enum { W = 96, H = 48, MAX_STRIDE = 2 * W + 1, OUT_STRIDE = 27 };
    static uint8_t storage[H * MAX_STRIDE + 1u];
    uint16_t pred[16 * OUT_STRIDE + 1u], half[16 * OUT_STRIDE];
    uint32_t seed = 0x658e5778u;
    for (uint32_t variant = 0u; variant < 4u; ++variant) {
        const uint32_t depth = variant == 0u ? 8u : 10u;
        const SvtAv2ChromaFormat format = variant < 2u ? SVT_AV2_CHROMA_420 : SVT_AV2_CHROMA_444;
        const uint32_t limit = (1u << depth) - 1u, step = depth == 8u ? 1u : 2u;
        SvtAv2FrameBuffer ref;
        const SvtAv2Error allocated = svt_av2_picture_alloc_ex(&ref, W, H, depth, format);
        SVT_TEST_CHECK_EQ_INT(allocated, SVT_AV2_ERROR_NONE);
        if (allocated != SVT_AV2_ERROR_NONE) continue;
        for (uint32_t p = 0u; p < 2u; ++p) {
            const uint32_t w = svt_av2_picture_plane_width(&ref, p), h = svt_av2_picture_plane_height(&ref, p);
            const uint32_t stride = w * step + 1u;
            ref.planes[p] = storage + 1u;
            ref.strides[p] = stride;  /* odd/almost tight rows and poisoned padding */
            memset(storage, 0xd7, sizeof(storage));
            for (uint32_t y = 0u; y < h; ++y) for (uint32_t x = 0u; x < w; ++x) {
                seed = seed * 1664525u + 1013904223u;
                /* Check old full-word semantics as well as legal 8/10-bit
                 * extremes. Full words exercise both signs in a centered
                 * pmaddwd and the non-narrow vertical fallback. */
                const uint32_t word = variant == 3u ? seed >> 16 :
                    (x < w / 2u ? (seed >> 11) & limit : (seed >> 31) * limit);
                svt_av2_sample_store(ref.planes[p] + (size_t)y * stride + x * step, depth, (uint16_t)word);
            }
            const SvtAv2BlockView blocks[] = {
                {8,8,7,9,NULL}, {8,8,8,9,NULL}, {8,8,15,11,NULL}, {8,8,16,4,NULL},
                {8,4,4,16,NULL}, {w-12u,5,8,8,NULL}, {w-17u,h-11u,17,11,NULL},
                {0,0,23,7,NULL}, {8,1,23,6,NULL},
            };
            for (size_t b = 0u; b < sizeof(blocks) / sizeof(blocks[0]); ++b) {
                const int phases = p && format == SVT_AV2_CHROMA_420 ? 8 : 4;
                for (int py = 0; py < phases; ++py) for (int px = 0; px < phases; ++px) {
                    /* The existing whole-pixel copy bypasses SHARP clipping. */
                    if (variant == 3u && px == 0 && py == 0) continue;
                    const SvtAv2QuarterPelMotionVector mv = {b == 7u ? -px : px, b == 7u ? -py : py};
                    pred[0] = 0xa5a5;
                    check_quarter_sharp_block(&ref, p, &blocks[b], mv, pred + 1u, half, OUT_STRIDE);
                    SVT_TEST_CHECK_EQ_INT(pred[0], 0xa5a5);
                }
            }
        }
        svt_av2_picture_free(&ref);
    }
}

static void check_eighth_sharp(void) {
    uint32_t seed = 0xae1785u;
    for (uint32_t variant = 0u; variant < 3u; ++variant) {
        SvtAv2FrameBuffer ref = {0};
        const SvtAv2ChromaFormat fmt = variant == 2u ? SVT_AV2_CHROMA_444 : SVT_AV2_CHROMA_420;
        const uint32_t depth = variant == 0u ? 8u : 10u, step = depth > 8u ? 2u : 1u;
        const uint32_t max_sample = (1u << depth) - 1u;
        const SvtAv2Error st = svt_av2_picture_alloc_ex(&ref, 96, 56, depth, fmt);
        SVT_TEST_CHECK_EQ_INT(st, SVT_AV2_ERROR_NONE);
        if (st != SVT_AV2_ERROR_NONE) continue;
        for (uint32_t p = 0u; p < 3u; ++p) {
            const uint32_t w = svt_av2_picture_plane_width(&ref, p), h = svt_av2_picture_plane_height(&ref, p);
            for (uint32_t y = 0u; y < h; ++y) for (uint32_t x = 0u; x < w; ++x) {
                seed = seed * 1664525u + 1013904223u;
                const uint16_t word = (uint16_t)(x & 1u ? (seed >> 31) * max_sample : (seed >> 15) & max_sample);
                svt_av2_sample_store(ref.planes[p] + (size_t)y * ref.strides[p] + x * step, depth, word);
            }
            const SvtAv2BlockView blocks[] = {{0, 0, 4, 4, NULL}, {8, 8, 8, 4, NULL},
                {8, 6, 4, 8, NULL}, {12, 8, 16, 12, NULL}, {w-9, h-7, 9, 7, NULL}};
            const int32_t phases = (p != 0u && fmt == SVT_AV2_CHROMA_420) ? 16 : 8;
            for (size_t b = 0; b < sizeof(blocks) / sizeof(blocks[0]); ++b) {
                const SvtAv2BlockView* block = &blocks[b];
                for (int32_t y = 0; y < phases; ++y) for (int32_t x = 0; x < phases; ++x) {
                    const SvtAv2EighthPelMotionVector mv = {b == 0u ? -x-9 : x, b == 0u ? -y-7 : y};
                    uint16_t pred[17 * 12];
                    for (size_t i = 0; i < sizeof(pred) / sizeof(pred[0]); ++i) pred[i] = 0xa5a5u;
                    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_eighth_sharp(&ref, p, block, mv, pred, 17u),
                        SVT_AV2_ERROR_NONE);
                    for (uint32_t yy = 0u; yy < block->h; ++yy) {
                        for (uint32_t xx = 0u; xx < block->w; ++xx) {
                            const int64_t expected = sharp_eighth_oracle(&ref, p, block, xx, yy, mv.mv_x, mv.mv_y);
                            SVT_TEST_CHECK_EQ_INT(pred[yy * 17u + xx], expected < 0 ? 0u :
                                expected > max_sample ? max_sample : (uint32_t)expected);
                        }
                        SVT_TEST_CHECK_EQ_INT(pred[yy * 17u + block->w], 0xa5a5u);
                    }
                }
            }
        }
        svt_av2_picture_free(&ref);
    }
}

static void check_sharp_bounded_sad(void) {
    const SvtAv2BlockView blocks[] = {{0,0,64,24,NULL}, {16,8,32,17,NULL}, {81,46,15,18,NULL},
        {8,7,9,4,NULL}, {4,4,4,7,NULL}, {8,12,8,8,NULL}};
    const int32_t offsets[] = {-380, -15, -4, -3, -2, -1, 0, 1, 2, 3, 4, 19, 421};
    uint32_t seed = UINT32_C(0x5b3ae310);
    uint16_t pred[64 * 64];
    for (uint32_t variant = 0u; variant < 3u; ++variant) {
        SvtAv2FrameBuffer src = {0}, ref = {0};
        const uint32_t depth = variant == 0u ? 8u : 10u, step = depth == 8u ? 1u : 2u;
        const SvtAv2ChromaFormat mode = variant < 2u ? SVT_AV2_CHROMA_420 : SVT_AV2_CHROMA_444;
        const SvtAv2Error err_ref = svt_av2_picture_alloc_ex(&ref, 96, 64, depth, mode);
        const SvtAv2Error err_src = svt_av2_picture_alloc_ex(&src, 96, 64, depth, mode);
        SVT_TEST_CHECK_EQ_INT(err_ref, SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK_EQ_INT(err_src, SVT_AV2_ERROR_NONE);
        if (err_ref != SVT_AV2_ERROR_NONE || err_src != SVT_AV2_ERROR_NONE) {
            svt_av2_picture_free(&ref);
            svt_av2_picture_free(&src);
            continue;
        }
        for (uint32_t y = 0u; y < 64u; ++y) for (uint32_t x = 0u; x < 96u; ++x) {
            const uint32_t mask = (1u << (variant == 2u ? 16u : depth)) - 1u;
            seed = seed * 1664525u + 1013904223u;
            svt_av2_sample_store(src.planes[0] + (size_t)y * src.strides[0] + x * step, depth,
                (uint16_t)((seed >> 16) & mask));
            seed = seed * 1664525u + 1013904223u;
            svt_av2_sample_store(ref.planes[0] + (size_t)y * ref.strides[0] + x * step, depth,
                (uint16_t)((seed >> 16) & mask));
        }
        for (size_t i = 0u; i < sizeof(blocks) / sizeof(blocks[0]); ++i) {
            const SvtAv2BlockView* b = &blocks[i];
            for (size_t yy = 0u; yy < sizeof(offsets) / sizeof(offsets[0]); ++yy)
                for (size_t xx = 0u; xx < sizeof(offsets) / sizeof(offsets[0]); ++xx) {
                    const SvtAv2QuarterPelMotionVector mv = {offsets[xx], offsets[yy]};
                    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_quarter_sharp(&ref, 0u, b, mv, pred, 64u),
                        SVT_AV2_ERROR_NONE);
                    uint64_t exact = 0;
                    for (uint32_t y = 0u; y < b->h; ++y) for (uint32_t x = 0u; x < b->w; ++x) {
                        const uint8_t* px = src.planes[0] + (size_t)(b->y + y) * src.strides[0] + (size_t)(b->x + x) * step;
                        const int d = (int)svt_av2_sample_load(px, depth) - pred[y * 64u + x];
                        exact += (uint32_t)(d < 0 ? -d : d);
                    }
                    const uint64_t limits[] = {0u, exact / 2u, exact ? exact - 1u : 0u,
                        exact, exact + 1u, UINT64_MAX};
                    for (size_t t = 0u; t < sizeof(limits) / sizeof(limits[0]); ++t) {
                        uint64_t got = UINT64_MAX;
                        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_luma_sad_quarter_sharp(&src, &ref, b, mv, limits[t], &got),
                            SVT_AV2_ERROR_NONE);
                        SVT_TEST_CHECK(got == (exact <= limits[t] ? exact : limits[t] + 1u));
                    }
                }
        }
        SvtAv2BlockView bad = {0, 0, 65, 8, NULL};
        const SvtAv2QuarterPelMotionVector zero = {0, 0};
        uint64_t sentinel = UINT64_MAX - 1u, got = sentinel;
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_luma_sad_quarter_sharp(&src, &ref, &bad, zero, 0, &got),
            SVT_AV2_ERROR_INVALID_PARAMETER);
        src.bit_depth = 0u;
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_luma_sad_quarter_sharp(&src, &ref, &blocks[0], zero, 0, &got),
            SVT_AV2_ERROR_INVALID_PARAMETER);
        SVT_TEST_CHECK(got == sentinel);
        svt_av2_picture_free(&src);
        svt_av2_picture_free(&ref);
    }
}

static uint64_t eighth_sad_oracle(const SvtAv2FrameBuffer* src, const SvtAv2FrameBuffer* ref,
    const SvtAv2BlockView* b, SvtAv2EighthPelMotionVector mv, uint64_t* first_row) {
    const int64_t max_sample = ((int64_t)1 << ref->bit_depth) - 1;
    uint64_t sum = 0u;
    for (uint32_t y = 0u; y < b->h; ++y) {
        for (uint32_t x = 0u; x < b->w; ++x) {
            const int sx = (int)(b->x + x), sy = (int)(b->y + y);
            int64_t value;
            if (mv.mv_x % 8 == 0 && mv.mv_y % 8 == 0) {
                /* Whole-pixel sampling intentionally preserves raw words;
                 * SHARP clips only outputs with a fractional axis. */
                value = edge_sample(ref, 0u, sx + mv.mv_x / 8, sy + mv.mv_y / 8);
            } else {
                value = sharp_eighth_oracle(ref, 0u, b, x, y, mv.mv_x, mv.mv_y);
                value = value < 0 ? 0 : value > max_sample ? max_sample : value;
            }
            const int64_t d = (int64_t)edge_sample(src, 0u, sx, sy) - value;
            sum += (uint64_t)(d < 0 ? -d : d);
        }
        if (y == 0u) *first_row = sum;
    }
    return sum;
}

static void check_eighth_sad_case(const SvtAv2FrameBuffer* src, const SvtAv2FrameBuffer* ref,
    const SvtAv2BlockView* b, SvtAv2EighthPelMotionVector mv) {
    uint64_t first_row = 0u;
    const uint64_t exact = eighth_sad_oracle(src, ref, b, mv, &first_row);
    const uint64_t limits[] = {0u, exact / 2u, first_row, exact ? exact - 1u : 0u,
        exact, exact + 1u, UINT64_MAX};
    for (size_t i = 0u; i < sizeof(limits) / sizeof(limits[0]); ++i) {
        uint64_t got = UINT64_MAX;
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_luma_sad_eighth_sharp(src, ref, b, mv, limits[i], &got),
            SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK(got == (exact <= limits[i] ? exact : limits[i] + 1u));
    }
    if (mv.mv_x % 2 == 0 && mv.mv_y % 2 == 0) {
        /* Also pin equivalence of the two public units on their shared lattice. */
        const SvtAv2QuarterPelMotionVector q = {mv.mv_x / 2, mv.mv_y / 2};
        uint64_t quarter = UINT64_MAX;
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_luma_sad_quarter_sharp(src, ref, b, q, exact, &quarter),
            SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK(quarter == exact);
    }
}

static void check_eighth_sad_invalid(const SvtAv2FrameBuffer* src, const SvtAv2FrameBuffer* ref) {
    const SvtAv2BlockView block = {0, 0, 8, 8, NULL};
    const SvtAv2EighthPelMotionVector mv = {-3, 5};
    const SvtAv2BlockView bad[] = {{0,0,0,8,NULL}, {0,0,8,0,NULL}, {0,0,65,8,NULL},
        {UINT32_MAX,0,8,8,NULL}, {0,UINT32_MAX,8,8,NULL}, {89,0,8,8,NULL}, {0,57,8,8,NULL}};
    const uint64_t sentinel = UINT64_C(0xfb338e1d2336a651);
    uint64_t got = sentinel;
    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_luma_sad_eighth_sharp(NULL, ref, &block, mv, 0u, &got),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    SVT_TEST_CHECK(got == sentinel);
    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_luma_sad_eighth_sharp(src, NULL, &block, mv, 0u, &got),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    SVT_TEST_CHECK(got == sentinel);
    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_luma_sad_eighth_sharp(src, ref, NULL, mv, 0u, &got),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    SVT_TEST_CHECK(got == sentinel);
    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_luma_sad_eighth_sharp(src, ref, &block, mv, 0u, NULL),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    for (size_t i = 0u; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_luma_sad_eighth_sharp(src, ref, &bad[i], mv, 0u, &got),
            SVT_AV2_ERROR_INVALID_PARAMETER);
        SVT_TEST_CHECK(got == sentinel);
    }
    SvtAv2FrameBuffer broken[] = {*src, *src, *src, *src, *src, *src, *ref, *ref};
    broken[0].buffer = NULL;
    broken[1].planes[0] = NULL;
    broken[2].visible_width--;
    broken[3].visible_height--;
    broken[4].bit_depth = 0u;
    broken[5].chroma_format = src->chroma_format == SVT_AV2_CHROMA_420 ? SVT_AV2_CHROMA_444 : SVT_AV2_CHROMA_420;
    broken[6].buffer = NULL;
    broken[7].planes[0] = NULL;
    for (size_t i = 0u; i < sizeof(broken) / sizeof(broken[0]); ++i) {
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_luma_sad_eighth_sharp(i < 6u ? &broken[i] : src,
            i < 6u ? ref : &broken[i], &block, mv, 0u, &got), SVT_AV2_ERROR_INVALID_PARAMETER);
        SVT_TEST_CHECK(got == sentinel);
    }
}

static void check_eighth_sharp_bounded_sad(void) {
    enum { W = 96, H = 64, STRIDE = W * 2 + 5 };
    static uint8_t src_luma[H * STRIDE + 1u], ref_luma[H * STRIDE + 1u];
    const SvtAv2BlockView interior = {23, 13, 9, 11, NULL};
    const SvtAv2BlockView blocks[] = {{0,0,64,64,NULL}, {W-15,H-18,15,18,NULL},
        {8,7,32,4,NULL}, {4,2,4,17,NULL}, {W-1,H-1,1,1,NULL}, {0,H-8,7,8,NULL}};
    const SvtAv2EighthPelMotionVector moves[] = {
        {0,0}, {8,-8}, {-127,127}, {128,-128}, {-128,-128}, {127,127},
        {1,0}, {0,-1}, {-1,5}, {7,-3}, {15,-16}, {16,-17},
        {INT32_MIN,INT32_MAX-7}, {INT32_MAX-7,INT32_MIN},
        {INT32_MIN+1,INT32_MAX}, {INT32_MAX,INT32_MIN+1}
    };
    uint32_t seed = UINT32_C(0xb748c891);
    for (uint32_t variant = 0u; variant < 3u; ++variant) {
        SvtAv2FrameBuffer src = {0}, ref = {0};
        const uint32_t depth = variant == 0u ? 8u : 10u, step = depth > 8u ? 2u : 1u;
        const SvtAv2ChromaFormat mode = variant == 2u ? SVT_AV2_CHROMA_444 : SVT_AV2_CHROMA_420;
        const uint32_t mask = (1u << (variant == 2u ? 16u : depth)) - 1u;
        const SvtAv2Error err_src = svt_av2_picture_alloc_ex(&src, W, H, depth, mode);
        const SvtAv2Error err_ref = svt_av2_picture_alloc_ex(&ref, W, H, depth, mode);
        SVT_TEST_CHECK_EQ_INT(err_src, SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK_EQ_INT(err_ref, SVT_AV2_ERROR_NONE);
        if (err_src != SVT_AV2_ERROR_NONE || err_ref != SVT_AV2_ERROR_NONE) {
            svt_av2_picture_free(&ref);
            svt_av2_picture_free(&src);
            continue;
        }
        /* Unaligned, different almost-tight strides. Padding cannot be mistaken
         * for pixels, and full words trigger unsigned subtraction and widening. */
        src.planes[0] = src_luma + 1u; src.strides[0] = W * step + 3u;
        ref.planes[0] = ref_luma + 1u; ref.strides[0] = W * step + 5u;
        memset(src_luma, 0xd7, sizeof(src_luma));
        memset(ref_luma, 0xa6, sizeof(ref_luma));
        for (uint32_t y = 0u; y < H; ++y) for (uint32_t x = 0u; x < W; ++x) {
            seed = seed * 1664525u + 1013904223u;
            const uint32_t source_word = x & 1u ? (seed >> 31) * mask : (seed >> 16) & mask;
            seed = seed * 1664525u + 1013904223u;
            const uint32_t ref_word = x & 1u ? (seed >> 16) & mask : (seed >> 31) * mask;
            svt_av2_sample_store(src.planes[0] + (size_t)y * src.strides[0] + x * step, depth, (uint16_t)source_word);
            svt_av2_sample_store(ref.planes[0] + (size_t)y * ref.strides[0] + x * step, depth, (uint16_t)ref_word);
        }
        for (int32_t y = -4; y <= 3; ++y) for (int32_t x = -4; x <= 3; ++x) {
            const SvtAv2EighthPelMotionVector mv = {x, y};
            check_eighth_sad_case(&src, &ref, &interior, mv);
        }
        for (size_t b = 0u; b < sizeof(blocks) / sizeof(blocks[0]); ++b)
            for (size_t m = 0u; m < sizeof(moves) / sizeof(moves[0]); ++m)
                check_eighth_sad_case(&src, &ref, &blocks[b], moves[m]);
        /* True equality even at a zero bound. */
        const SvtAv2EighthPelMotionVector zero = {0,0};
        check_eighth_sad_case(&ref, &ref, &interior, zero);
        check_eighth_sad_invalid(&src, &ref);
        svt_av2_picture_free(&ref);
        svt_av2_picture_free(&src);
    }
}

int main(void) {
    uint32_t seed = 0x31415926u;
    for (uint32_t f = 0; f < 3; ++f) {
        const uint32_t bd = f == 0 ? 8 : 10, step = bd == 8 ? 1u : 2u;
        const SvtAv2ChromaFormat format = f == 2 ? SVT_AV2_CHROMA_444 : SVT_AV2_CHROMA_420;
        SvtAv2FrameBuffer ref;
        SVT_TEST_CHECK_EQ_INT(svt_av2_picture_alloc_ex(&ref, 32, 32, bd, format), SVT_AV2_ERROR_NONE);
        uint8_t bytes[3][32 * 65 + 1];
        for (uint32_t p = 0; p < 3; ++p) {
            ref.planes[p] = bytes[p] + 1; ref.strides[p] = 65;
            for (uint32_t y = 0; y < svt_av2_picture_plane_height(&ref, p); ++y)
                for (uint32_t x = 0; x < svt_av2_picture_plane_width(&ref, p); ++x) {
                    seed = seed * 1664525u + 1013904223u;
                    const uint16_t value = (uint16_t)((seed >> 12) & ((1u << bd) - 1u));
                    svt_av2_sample_store(ref.planes[p] + y * 65 + x * step, bd, value);
                }
            for (int my = -8; my <= 8; ++my) for (int mx = -8; mx <= 8; ++mx) {
                const uint32_t width = svt_av2_picture_plane_width(&ref, p), height = svt_av2_picture_plane_height(&ref, p);
                const SvtAv2BlockView block = { 0, 0, width, height, NULL };
                const SvtAv2MotionVector mv = { mx, my };
                uint16_t prediction[32 * 33];
                for (uint32_t i = 0; i < 32 * 33; ++i) prediction[i] = 0xA5A5;
                SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict(&ref, p, &block, mv, prediction, 33), SVT_AV2_ERROR_NONE);
                for (uint32_t y = 0; y < height; ++y) {
                    for (uint32_t x = 0; x < width; ++x)
                        SVT_TEST_CHECK_EQ_INT(prediction[y * 33 + x], filter_model(&ref, p, (int)x, (int)y, 2 * mx, 2 * my));
                    SVT_TEST_CHECK_EQ_INT(prediction[y * 33 + width], 0xA5A5);
                }
            }
            check_half_blocks(&ref, p);
        }
        svt_av2_picture_free(&ref);
    }
    check_half_rounding_and_umv();
    check_sharp();
    check_sharp_vector_footprints();
    check_eighth_sharp();
    check_sharp_bounded_sad();
    check_eighth_sharp_bounded_sad();
    SVT_TEST_REPORT("test_inter_prediction");
}
