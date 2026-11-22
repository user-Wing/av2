#include "test_util.h"
#include "motion_estimation.h"
#include <stdint.h>
#include <stdlib.h>

static uint16_t sample(const SvtAv2FrameBuffer* p, int x, int y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= (int)p->visible_width) x = (int)p->visible_width - 1;
    if (y >= (int)p->visible_height) y = (int)p->visible_height - 1;
    const uint8_t* q = p->planes[0] + (size_t)y * p->strides[0] + (size_t)x * (p->bit_depth == 8 ? 1u : 2u);
    return p->bit_depth == 8 ? q[0] : (uint16_t)(q[0] + 256u * q[1]);
}

static void store(SvtAv2FrameBuffer* p, uint32_t x, uint32_t y, uint16_t value) {
    svt_av2_sample_store(p->planes[0] + (size_t)y * p->strides[0] + x * (p->bit_depth == 8 ? 1u : 2u),
        p->bit_depth, value);
}

/* Scalar oracle: clamp each sample independently, never stop early, then rank
 * exact scores in bulk rather than inserting as candidates are evaluated. */
typedef struct MotionScore {
    int x, y;
    uint64_t sad;
} MotionScore;

static MotionScore score(const SvtAv2FrameBuffer* src, const SvtAv2FrameBuffer* ref,
    const SvtAv2BlockView* block, int x, int y) {
    MotionScore result = {x, y, 0};
    for (uint32_t j = 0; j < block->h; ++j) for (uint32_t i = 0; i < block->w; ++i)
        result.sad += (uint64_t)abs((int)sample(src, (int)(block->x + i), (int)(block->y + j)) -
            (int)sample(ref, (int)(block->x + i) + x, (int)(block->y + j) + y));
    return result;
}

static int compare_scores(const void* pa, const void* pb) {
    const MotionScore* a = pa;
    const MotionScore* b = pb;
    if (a->sad != b->sad) return a->sad < b->sad ? -1 : 1;
    const int a_norm = abs(a->x) + abs(a->y), b_norm = abs(b->x) + abs(b->y);
    if (a_norm != b_norm) return a_norm - b_norm;
    if (a->y != b->y) return a->y - b->y;
    return a->x - b->x;
}

static void check_score(SvtAv2MotionVector mv, uint64_t sad, MotionScore expected) {
    SVT_TEST_CHECK_EQ_INT(mv.mv_x, expected.x);
    SVT_TEST_CHECK_EQ_INT(mv.mv_y, expected.y);
    SVT_TEST_CHECK_EQ_INT(sad, expected.sad);
}

static void check_square(const SvtAv2FrameBuffer* src, const SvtAv2FrameBuffer* ref,
    const SvtAv2BlockView* block) {
    MotionScore scores[17 * 17];
    size_t count = 0;
    for (int y = -8; y <= 8; ++y) for (int x = -8; x <= 8; ++x)
        scores[count++] = score(src, ref, block, x, y);
    const MotionScore zero = scores[count / 2];
    qsort(scores, count, sizeof(scores[0]), compare_scores);

    SvtAv2MotionSearchResult best = {0};
    const SvtAv2Error best_err = svt_av2_motion_estimate(src, ref, block, &best);
    SVT_TEST_CHECK_EQ_INT(best_err, SVT_AV2_ERROR_NONE);
    if (best_err == SVT_AV2_ERROR_NONE) {
        check_score(best.mv, best.sad, scores[0]);
        SVT_TEST_CHECK_EQ_INT(best.candidates, count);
    }

    SvtAv2MotionSearchList list = {0};
    const SvtAv2Error list_err = svt_av2_motion_estimate_list(src, ref, block, &list);
    SVT_TEST_CHECK_EQ_INT(list_err, SVT_AV2_ERROR_NONE);
    if (list_err != SVT_AV2_ERROR_NONE) return;
    SVT_TEST_CHECK_EQ_INT(list.candidates, count);
    uint32_t expected_count = 4;
    for (size_t i = 0; i < 3; ++i) {
        check_score(list.mv[i], list.sad[i], scores[i]);
        if (scores[i].x == 0 && scores[i].y == 0) expected_count = 3;
    }
    if (expected_count == 4) check_score(list.mv[3], list.sad[3], zero);
    SVT_TEST_CHECK_EQ_INT(list.count, expected_count);
}

static void check_wide(const SvtAv2FrameBuffer* src, const SvtAv2FrameBuffer* ref,
    const SvtAv2BlockView* block) {
    MotionScore scores[9 * 9 + 5 * 5];
    size_t count = 0;
    for (int y = -16; y <= 16; y += 4) for (int x = -16; x <= 16; x += 4)
        if (abs(y) > 8 || abs(x) > 8) scores[count++] = score(src, ref, block, x, y);
    qsort(scores, count, sizeof(scores[0]), compare_scores);
    const MotionScore center = scores[0];
    for (int y = center.y - 2; y <= center.y + 2; ++y) for (int x = center.x - 2; x <= center.x + 2; ++x)
        if (abs(y) <= 16 && abs(x) <= 16 && (abs(y) > 8 || abs(x) > 8) && (x % 4 != 0 || y % 4 != 0))
            scores[count++] = score(src, ref, block, x, y);
    qsort(scores, count, sizeof(scores[0]), compare_scores);

    SvtAv2MotionSearchList list = {0};
    const SvtAv2Error err = svt_av2_motion_estimate_wide(src, ref, block, &list);
    SVT_TEST_CHECK_EQ_INT(err, SVT_AV2_ERROR_NONE);
    if (err != SVT_AV2_ERROR_NONE) return;
    SVT_TEST_CHECK_EQ_INT(list.count, 2);
    SVT_TEST_CHECK_EQ_INT(list.candidates, count);
    for (size_t i = 0; i < 2; ++i) check_score(list.mv[i], list.sad[i], scores[i]);
}

int main(void) {
    enum { WIDTH = 56, HEIGHT = 48 };
    static const SvtAv2BlockView blocks[] = {
        {20, 16, 16, 16, NULL},                                    /* interior */
        {1, 2, 17, 11, NULL}, {WIDTH - 11, 0, 11, 7, NULL},        /* partial top corners */
        {0, HEIGHT - 8, 8, 8, NULL}, {WIDTH - 8, HEIGHT - 8, 8, 8, NULL},
        {0, 0, 1, 1, NULL}, {WIDTH - 1, HEIGHT - 1, 1, 1, NULL},   /* wholly outside for some MVs */
        {0, 16, 32, 4, NULL}, {3, HEIGHT - 3, 19, 3, NULL},
        {0, 0, WIDTH, HEIGHT, NULL},
    };
    uint32_t seed = 0x31415926u;
    for (uint32_t f = 0; f < 3; ++f) {
        const uint32_t bd = f == 0 ? 8 : 10;
        const uint16_t maximum = (uint16_t)((1u << bd) - 1u);
        SvtAv2FrameBuffer src, ref;
        const SvtAv2ChromaFormat format = f == 2 ? SVT_AV2_CHROMA_444 : SVT_AV2_CHROMA_420;
        const SvtAv2Error src_err = svt_av2_picture_alloc_ex(&src, WIDTH, HEIGHT, bd, format);
        const SvtAv2Error ref_err = svt_av2_picture_alloc_ex(&ref, WIDTH, HEIGHT, bd, format);
        SVT_TEST_CHECK_EQ_INT(src_err, SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK_EQ_INT(ref_err, SVT_AV2_ERROR_NONE);
        if (src_err != SVT_AV2_ERROR_NONE || ref_err != SVT_AV2_ERROR_NONE) {
            svt_av2_picture_free(&src); svt_av2_picture_free(&ref);
            continue;
        }
        /* Non-sample bytes deliberately differ from the extended edges. */
        memset(src.planes[0], 0x5a, (size_t)src.strides[0] * src.aligned_height[0]);
        memset(ref.planes[0], 0xa5, (size_t)ref.strides[0] * ref.aligned_height[0]);
        for (uint32_t y = 0; y < HEIGHT; ++y) for (uint32_t x = 0; x < WIDTH; ++x) {
            seed = seed * 1664525u + 1013904223u;
            store(&ref, x, y, (uint16_t)((seed >> 12) & maximum));
        }
        for (uint32_t trial = 0; trial < 100; ++trial) {
            const int radius = trial % 2 == 0 ? 8 : 16;
            const uint32_t span = 2u * (uint32_t)radius + 1u;
            const int dx = (int)(trial % span) - radius, dy = (int)((trial * 7) % span) - radius;
            for (uint32_t y = 0; y < HEIGHT; ++y) for (uint32_t x = 0; x < WIDTH; ++x)
                store(&src, x, y, sample(&ref, (int)x + dx, (int)y + dy));
            const SvtAv2BlockView* block = &blocks[trial % (sizeof(blocks) / sizeof(blocks[0]))];
            check_square(&src, &ref, block);
            check_wide(&src, &ref, block);
        }
        for (uint32_t y = 0; y < HEIGHT; ++y) for (uint32_t x = 0; x < WIDTH; ++x) {
            store(&src, x, y, maximum);
            store(&ref, x, y, maximum);
        }
        const SvtAv2BlockView flat = {0, 0, 16, 16, NULL};
        check_square(&src, &ref, &flat);
        check_wide(&src, &ref, &flat);

        /* (0,-2) wins with SAD 0. (0,-1) ties after row one only, so stopping
         * on equality instead of strictly greater would hand it the win. */
        for (uint32_t y = 0; y < HEIGHT; ++y) for (uint32_t x = 0; x < WIDTH; ++x) {
            store(&src, x, y, 0);
            store(&ref, x, y, y < 2 ? 0 : maximum);
        }
        const SvtAv2BlockView early_stop = {10, 2, 16, 2, NULL};
        check_square(&src, &ref, &early_stop);
        check_wide(&src, &ref, &early_stop);
        svt_av2_picture_free(&src); svt_av2_picture_free(&ref);
    }
    SVT_TEST_REPORT("test_motion_estimation");
}
