#include "test_util.h"
#include "reference_frame.h"
#include "reconstruction.h"
#include <string.h>

static void check_slot(const SvtAv2ReferenceFrame* slot, uint64_t index,
    uint16_t value, uint32_t depth) {
    const SvtAv2FrameBuffer* picture = svt_av2_reference_view(slot);
    SVT_TEST_CHECK(picture != NULL);
    SVT_TEST_CHECK(!slot->prepared_valid);
    if (picture == NULL) return;
    SVT_TEST_CHECK_EQ_INT(slot->frame_index, index);
    for (uint32_t plane = 0u; plane < 3u; ++plane) {
        SVT_TEST_CHECK_EQ_INT(svt_av2_sample_load(picture->planes[plane], depth), value);
        const size_t offset = (size_t)(svt_av2_picture_plane_height(picture, plane) - 1u) * picture->strides[plane] +
            (size_t)(svt_av2_picture_plane_width(picture, plane) - 1u) * (depth == 10u ? 2u : 1u);
        SVT_TEST_CHECK_EQ_INT(svt_av2_sample_load(picture->planes[plane] + offset, depth), value);
    }
}

static void test_pair(uint32_t depth, SvtAv2ChromaFormat format, uint32_t period) {
    SvtAv2ReferenceFrame last = {0}, anchor = {0};
    SvtAv2FrameBuffer recon = {0};
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_init(&last, 16u, 16u, depth, format), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_init(&anchor, 16u, 16u, depth, format), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_picture_alloc_ex(&recon, 16u, 16u, depth, format), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_recon_reset_frame(&recon, 17u), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &anchor, &recon, 0u, 0u),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &last, &recon, 0u, period),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &anchor, &recon, 1u, period),
        SVT_AV2_ERROR_INVALID_STATE);
    /* KEY stages only slot0; a malformed stage destination in slot1 must
     * not affect it. The decoder drops the initial duplicate KEY mapping. */
    anchor.prepared.visible_width = 8u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &anchor, &recon, 0u, period),
        SVT_AV2_ERROR_NONE);
    check_slot(&last, 0u, 17u, depth);
    SVT_TEST_CHECK(svt_av2_reference_view(&anchor) == NULL && !anchor.prepared_valid);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_commit(&anchor), SVT_AV2_ERROR_INVALID_STATE);
    anchor.prepared.visible_width = 16u;

    for (uint64_t i = 1u; i <= 258u; ++i) {
        const uint16_t pixel = (uint16_t)(17u + i % 227u);
        const uint64_t old = i - i % period;
        SVT_TEST_CHECK_EQ_INT(svt_av2_recon_reset_frame(&recon, pixel), SVT_AV2_ERROR_NONE);
        if (i == 1u) {
            SVT_TEST_CHECK_EQ_INT(svt_av2_reference_prepare(&last, &recon, i), SVT_AV2_ERROR_NONE);
            SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &anchor, &recon, i, period),
                SVT_AV2_ERROR_INVALID_STATE);
            SVT_TEST_CHECK(last.prepared_valid); /* pair does not discard somebody else's stage */
            last.prepared_valid = 0;
        }
        if (i % period == 0u) {
            const uint8_t* last_buffer = last.committed.buffer;
            const uint8_t* anchor_buffer = anchor.committed.buffer;
            anchor.prepared.visible_width = 8u;
            SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &anchor, &recon, i, period),
                SVT_AV2_ERROR_INVALID_PARAMETER);
            anchor.prepared.visible_width = 16u;
            SVT_TEST_CHECK(last.committed.buffer == last_buffer);
            SVT_TEST_CHECK(anchor.committed.buffer == anchor_buffer);
            check_slot(&last, i - 1u, (uint16_t)(17u + (i - 1u) % 227u), depth);
            if (i == period) {
                SVT_TEST_CHECK(svt_av2_reference_view(&anchor) == NULL && !anchor.prepared_valid);
                SVT_TEST_CHECK_EQ_INT(svt_av2_reference_commit(&last), SVT_AV2_ERROR_INVALID_STATE);
                SVT_TEST_CHECK_EQ_INT(svt_av2_reference_commit(&anchor), SVT_AV2_ERROR_INVALID_STATE);
            } else {
                check_slot(&anchor, i - period, (uint16_t)(17u + (i - period) % 227u), depth);
            }
        }
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &anchor, &recon, i, period),
            SVT_AV2_ERROR_NONE);
        check_slot(&last, i, pixel, depth);
        if (i < period) {
            SVT_TEST_CHECK(svt_av2_reference_view(&anchor) == NULL && !anchor.prepared_valid);
        } else {
            check_slot(&anchor, old, (uint16_t)(17u + old % 227u), depth);
            SVT_TEST_CHECK(last.committed.buffer != anchor.committed.buffer);
        }
    }
    svt_av2_picture_free(&recon);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_release(&last), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_release(&anchor), SVT_AV2_ERROR_NONE);
}

static void test_rotating_anchor(void) {
    SvtAv2ReferenceFrame recent = {0}, older = {0};
    SvtAv2FrameBuffer recon = {0};
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_init(&recent, 16, 16, 10, SVT_AV2_CHROMA_420), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_init(&older, 16, 16, 10, SVT_AV2_CHROMA_420), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_picture_alloc_ex(&recon, 16, 16, 10, SVT_AV2_CHROMA_420), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_recon_reset_frame(&recon, 41), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&recent, &older, &recon, 0, 0), SVT_AV2_ERROR_NONE);
    uint32_t anchor = 0;
    for (uint32_t i = 1; i < 12; ++i) {
        if (i == 8) continue;
        const int rotate = i == 1 || i == 9;
        const uint32_t held = i == 9;
        if (rotate) anchor = i - held - 1;
        SVT_TEST_CHECK_EQ_INT(svt_av2_recon_reset_frame(&recon, (uint16_t)(41+i)), SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_anchor(&recent, &older, &recon, i, held, rotate), SVT_AV2_ERROR_NONE);
        check_slot(&recent, i, (uint16_t)(41+i), 10);
        check_slot(&older, anchor, (uint16_t)(41+anchor), 10);
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_anchor(&recent, &older, &recon, i, held, rotate), SVT_AV2_ERROR_INVALID_STATE);
    }
    svt_av2_reference_release(&recent);
    svt_av2_reference_release(&older);
    svt_av2_picture_free(&recon);
}

static void test_pair_masked(uint32_t depth, SvtAv2ChromaFormat format) {
    SvtAv2ReferenceFrame last = {0}, anchor = {0};
    SvtAv2FrameBuffer recon = {0};
    const uint32_t masks[] = {0u,1u,1u,3u,1u,3u,3u,1u,1u,3u};
    const uint32_t invalid_p[] = {0u,2u,4u,7u,UINT32_MAX};
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_init(&last, 16u, 16u, depth, format), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_init(&anchor, 16u, 16u, depth, format), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_picture_alloc_ex(&recon, 16u, 16u, depth, format), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_recon_reset_frame(&recon, 20u), SVT_AV2_ERROR_NONE);
    for (uint32_t mask = 1u; mask <= 3u; ++mask)
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, &anchor, &recon, 0u, mask),
            SVT_AV2_ERROR_INVALID_PARAMETER);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(NULL, &anchor, &recon, 0u, 0u),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, NULL, &recon, 0u, 0u),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, &last, &recon, 0u, 0u),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, &anchor, NULL, 0u, 0u),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, &anchor, &recon, 1u, 3u),
        SVT_AV2_ERROR_INVALID_STATE); /* LAST has never been seeded. */
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_prepare(&anchor, &recon, 0u), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, &anchor, &recon, 0u, 0u),
        SVT_AV2_ERROR_INVALID_STATE);
    SVT_TEST_CHECK(anchor.prepared_valid && !last.valid && !anchor.valid);
    anchor.prepared_valid = 0;
    /* KEY does not inspect an idle LONG scratch buffer. */
    anchor.prepared.visible_width = 8u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, &anchor, &recon, 0u, masks[0]),
        SVT_AV2_ERROR_NONE);
    anchor.prepared.visible_width = 16u;
    check_slot(&last, 0u, 20u, depth);
    SVT_TEST_CHECK(svt_av2_reference_view(&anchor) == NULL && !anchor.prepared_valid);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, &anchor, &recon, 0u, 0u),
        SVT_AV2_ERROR_INVALID_STATE);
    for (size_t k = 0u; k < sizeof(invalid_p) / sizeof(invalid_p[0]); ++k)
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, &anchor, &recon, 1u, invalid_p[k]),
            SVT_AV2_ERROR_INVALID_PARAMETER);

    uint64_t long_index = 0u;
    for (uint64_t i = 1u; i < sizeof(masks) / sizeof(masks[0]); ++i) {
        const uint32_t mask = masks[i];
        const uint16_t pixel = (uint16_t)(20u + i);
        const uint8_t* const last_buffer = last.committed.buffer;
        const uint8_t* const long_buffer = anchor.committed.buffer;
        const uint64_t last_prepared = last.prepared_index, long_prepared = anchor.prepared_index;
        SVT_TEST_CHECK_EQ_INT(svt_av2_recon_reset_frame(&recon, pixel), SVT_AV2_ERROR_NONE);
        /* Gap and old/repeated picture IDs cannot advance either slot. */
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, &anchor, &recon, i + 1u, mask),
            SVT_AV2_ERROR_INVALID_STATE);
        if (i > 1u)
            SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, &anchor, &recon, i - 1u, mask),
                SVT_AV2_ERROR_INVALID_STATE);
        if (i == 4u) {
            /* P3 seeded LONG, so the strict period-4 wrapper must reject the
             * same prior state even though the explicit transaction accepts it. */
            SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &anchor, &recon, i, 4u),
                SVT_AV2_ERROR_INVALID_STATE);
            anchor.frame_index = i; /* A future LONG is invalid even on hold. */
            SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, &anchor, &recon, i, mask),
                SVT_AV2_ERROR_INVALID_STATE);
            anchor.frame_index = 0u; /* A valid duplicate KEY is invalid. */
            SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, &anchor, &recon, i, mask),
                SVT_AV2_ERROR_INVALID_STATE);
            anchor.frame_index = long_index;
        }
        if (i == 8u)
            SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&anchor, &last, &recon, i, mask),
                SVT_AV2_ERROR_INVALID_STATE); /* The older LONG cannot be LAST. */
        if (mask == 3u) {
            /* Slot 0 can stage successfully but cannot publish when slot 1's
             * destination copy fails (before and after LONG becomes valid). */
            anchor.prepared.visible_width = 8u;
            SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, &anchor, &recon, i, mask),
                SVT_AV2_ERROR_INVALID_PARAMETER);
            anchor.prepared.visible_width = 16u;
            SVT_TEST_CHECK_EQ_INT(svt_av2_reference_commit(&last), SVT_AV2_ERROR_INVALID_STATE);
            SVT_TEST_CHECK_EQ_INT(svt_av2_reference_commit(&anchor), SVT_AV2_ERROR_INVALID_STATE);
        }
        SVT_TEST_CHECK(last.committed.buffer == last_buffer && anchor.committed.buffer == long_buffer);
        SVT_TEST_CHECK(last.prepared_index == last_prepared && anchor.prepared_index == long_prepared);
        check_slot(&last, i - 1u, (uint16_t)(pixel - 1u), depth);
        if (long_index != 0u) check_slot(&anchor, long_index, (uint16_t)(20u + long_index), depth);
        else SVT_TEST_CHECK(svt_av2_reference_view(&anchor) == NULL && !anchor.prepared_valid);
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair_masked(&last, &anchor, &recon, i, mask),
            SVT_AV2_ERROR_NONE);
        check_slot(&last, i, pixel, depth);
        if (mask == 3u) long_index = i;
        else SVT_TEST_CHECK(anchor.committed.buffer == long_buffer);
        if (long_index != 0u) {
            check_slot(&anchor, long_index, (uint16_t)(20u + long_index), depth);
            SVT_TEST_CHECK(last.committed.buffer != anchor.committed.buffer);
        } else SVT_TEST_CHECK(svt_av2_reference_view(&anchor) == NULL && !anchor.prepared_valid);
    }
    svt_av2_picture_free(&recon);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_release(&last), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_release(&anchor), SVT_AV2_ERROR_NONE);
}

static void fill_pattern(SvtAv2FrameBuffer* frame, int32_t dx, int32_t dy, uint32_t bias) {
    const uint32_t depth = frame->bit_depth;
    const uint32_t step = depth == 10u ? 2u : 1u, scale = depth == 10u ? 4u : 1u;
    for (uint32_t y = 0u; y < frame->visible_height; ++y) {
        for (uint32_t x = 0u; x < frame->visible_width; ++x) {
            int32_t a = (int32_t)x + dx, b = (int32_t)y + dy;
            if (a < 0) a = 0;
            if (b < 0) b = 0;
            if (a >= (int32_t)frame->visible_width) a = (int32_t)frame->visible_width - 1;
            if (b >= (int32_t)frame->visible_height) b = (int32_t)frame->visible_height - 1;
            const uint32_t value = (uint32_t)(40 + a / 2 + (a * 7 + a * a * 3 + b * 11 + b * b * 13 + a * b * 17) % 79);
            svt_av2_sample_store(frame->planes[0] + (size_t)y * frame->strides[0] + (size_t)x * step,
                depth, (uint16_t)(value * scale + bias));
        }
    }
}

static void test_small_choice(uint32_t depth, SvtAv2ChromaFormat format) {
    SvtAv2ReferenceFrame last = {0}, anchor = {0};
    SvtAv2FrameBuffer picture = {0};
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_init(&last, 8u, 8u, depth, format), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_init(&anchor, 8u, 8u, depth, format), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_picture_alloc_ex(&picture, 8u, 8u, depth, format), SVT_AV2_ERROR_NONE);
    const uint16_t scale = depth == 10u ? 4u : 1u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_recon_reset_frame(&picture, (uint16_t)(76u * scale)), SVT_AV2_ERROR_NONE);
    for (uint64_t i = 0u; i <= 2u; ++i)
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &anchor, &picture, i, 2u), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_recon_reset_frame(&picture, (uint16_t)(75u * scale)), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &anchor, &picture, 3u, 2u), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_recon_reset_frame(&picture, (uint16_t)(100u * scale)), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_choose_slot(&picture, &last, &anchor), 0u); /* 1/25 < 5% */
    SVT_TEST_CHECK_EQ_INT(svt_av2_recon_reset_frame(&picture, (uint16_t)(76u * scale)), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_choose_slot(&picture, &last, &anchor), 1u);
    svt_av2_picture_free(&picture);
    svt_av2_reference_release(&last);
    svt_av2_reference_release(&anchor);
}

static void test_choice(uint32_t depth, SvtAv2ChromaFormat format) {
    SvtAv2ReferenceFrame last = {0}, anchor = {0};
    SvtAv2FrameBuffer source = {0}, recon = {0};
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_init(&last, 96u, 72u, depth, format), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_init(&anchor, 96u, 72u, depth, format), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_picture_alloc_ex(&recon, 96u, 72u, depth, format), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_picture_alloc_ex(&source, 96u, 72u, depth, format), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_recon_reset_frame(&recon, 0u), SVT_AV2_ERROR_NONE);
    fill_pattern(&recon, 0, 0, 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &anchor, &recon, 0u, 4u), SVT_AV2_ERROR_NONE);
    fill_pattern(&source, 4, -4, 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_choose_slot(&source, &last, &anchor), 0u); /* missing long */
    for (uint64_t i = 1u; i <= 4u; ++i)
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &anchor, &recon, i, 4u), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_choose_slot(&source, &last, &anchor), 0u); /* same index */
    const uint32_t bias = depth == 10u ? 12u : 3u;
    fill_pattern(&recon, 4, -4, bias);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &anchor, &recon, 5u, 4u), SVT_AV2_ERROR_NONE);
    /* Recent matches geometry without translation, while older needs a small
     * two-axis translation to reproduce the textured source exactly. */
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_choose_slot(&source, &last, &anchor), 1u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_choose_slot(&source, NULL, &anchor), 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_choose_slot(&source, &last, NULL), 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_choose_slot(NULL, &last, &anchor), 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_choose_slot(&source, &anchor, &last), 0u);
    SvtAv2FrameBuffer invalid = source;
    invalid.strides[0] = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_choose_slot(&invalid, &last, &anchor), 0u);
    invalid = source;
    invalid.visible_width -= 8u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_choose_slot(&invalid, &last, &anchor), 0u);

    fill_pattern(&source, 4, -4, bias);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_choose_slot(&source, &last, &anchor), 0u);
    fill_pattern(&recon, 0, 0, 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &anchor, &recon, 6u, 4u), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_choose_slot(&source, &last, &anchor), 0u); /* distinct-index tie */
    if (depth == 10u) {
        fill_pattern(&recon, 0, 0, 1u);
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_refresh_pair(&last, &anchor, &recon, 7u, 4u), SVT_AV2_ERROR_NONE);
        fill_pattern(&source, 0, 0, 0u);
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_choose_slot(&source, &last, &anchor), 0u); /* absolute floor */
    }
    svt_av2_picture_free(&source);
    svt_av2_picture_free(&recon);
    svt_av2_reference_release(&last);
    svt_av2_reference_release(&anchor);
}

int main(void) {
    test_rotating_anchor();
    const uint32_t depths[] = { 8, 10, 10 };
    const SvtAv2ChromaFormat formats[] = { SVT_AV2_CHROMA_420, SVT_AV2_CHROMA_420, SVT_AV2_CHROMA_444 };
    for (uint32_t f = 0; f < 3; ++f) {
        SvtAv2ReferenceFrame ref; SvtAv2FrameBuffer current;
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_init(&ref, 16, 16, depths[f], formats[f]), SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK(svt_av2_reference_view(&ref) == NULL);
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_commit(&ref), SVT_AV2_ERROR_INVALID_STATE);
        SVT_TEST_CHECK_EQ_INT(svt_av2_picture_alloc_ex(&current, 16, 16, depths[f], formats[f]), SVT_AV2_ERROR_NONE);
        for (uint64_t index = 0; index < 3; ++index) {
            const uint16_t value = (uint16_t)(17 + index * 71);
            SVT_TEST_CHECK_EQ_INT(svt_av2_recon_reset_frame(&current, value), SVT_AV2_ERROR_NONE);
            SVT_TEST_CHECK_EQ_INT(svt_av2_reference_prepare(&ref, &current, index), SVT_AV2_ERROR_NONE);
            if (index != 0) {
                SVT_TEST_CHECK_EQ_INT(ref.frame_index, index - 1);
                SVT_TEST_CHECK_EQ_INT(svt_av2_sample_load(ref.committed.planes[0], depths[f]), 17 + (index - 1) * 71);
            } else SVT_TEST_CHECK(svt_av2_reference_view(&ref) == NULL);
            SVT_TEST_CHECK(ref.prepared.buffer != current.buffer);
            SVT_TEST_CHECK(ref.prepared.buffer != ref.committed.buffer);
            SVT_TEST_CHECK_EQ_INT(svt_av2_recon_reset_frame(&current, 0), SVT_AV2_ERROR_NONE);
            SVT_TEST_CHECK_EQ_INT(svt_av2_reference_commit(&ref), SVT_AV2_ERROR_NONE);
            SVT_TEST_CHECK_EQ_INT(ref.frame_index, index);
            for (uint32_t p = 0; p < 3; ++p)
                SVT_TEST_CHECK_EQ_INT(svt_av2_sample_load(ref.committed.planes[p], depths[f]), value);
            SVT_TEST_CHECK_EQ_INT(svt_av2_reference_commit(&ref), SVT_AV2_ERROR_INVALID_STATE);
        }
        SvtAv2FrameBuffer wrong;
        SVT_TEST_CHECK_EQ_INT(svt_av2_picture_alloc_ex(&wrong, 8, 8, depths[f], formats[f]), SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_prepare(&ref, &wrong, 3), SVT_AV2_ERROR_INVALID_PARAMETER);
        SVT_TEST_CHECK_EQ_INT(ref.frame_index, 2);
        SVT_TEST_CHECK_EQ_INT(svt_av2_sample_load(ref.committed.planes[0], depths[f]), 159);
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_commit(&ref), SVT_AV2_ERROR_INVALID_STATE);
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_prepare(&ref, &current, 4), SVT_AV2_ERROR_INVALID_STATE);
        uint8_t odd[3][16 * 35 + 1];
        const uint32_t step = depths[f] == 8 ? 1u : 2u;
        for (uint32_t p = 0; p < 3; ++p) {
            current.planes[p] = odd[p] + 1; current.strides[p] = 35;
            for (uint32_t y = 0; y < svt_av2_picture_plane_height(&current, p); ++y)
                for (uint32_t x = 0; x < svt_av2_picture_plane_width(&current, p); ++x)
                    svt_av2_sample_store(current.planes[p] + y * 35 + x * step, depths[f], 100);
        }
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_prepare(&ref, &current, 3), SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_commit(&ref), SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK_EQ_INT(svt_av2_sample_load(ref.committed.planes[2], depths[f]), 100);
        svt_av2_picture_free(&wrong); svt_av2_picture_free(&current);
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_release(&ref), SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_release(&ref), SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK(svt_av2_reference_view(&ref) == NULL);
        test_pair(depths[f], formats[f], 2u << f);
        test_pair_masked(depths[f], formats[f]);
        test_choice(depths[f], formats[f]);
        test_small_choice(depths[f], formats[f]);
    }
    SVT_TEST_REPORT("test_reference_frame");
}
