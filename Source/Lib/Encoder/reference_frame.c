#include "reference_frame.h"
#include <string.h>

SvtAv2Error svt_av2_reference_init(SvtAv2ReferenceFrame* reference, uint32_t width,
    uint32_t height, uint32_t bit_depth, SvtAv2ChromaFormat chroma_format) {
    if (reference == NULL) return SVT_AV2_ERROR_INVALID_PARAMETER;
    memset(reference, 0, sizeof(*reference));
    SvtAv2Error error = svt_av2_picture_alloc_ex(&reference->committed, width, height, bit_depth, chroma_format);
    if (error != SVT_AV2_ERROR_NONE) return error;
    error = svt_av2_picture_alloc_ex(&reference->prepared, width, height, bit_depth, chroma_format);
    if (error != SVT_AV2_ERROR_NONE) svt_av2_reference_release(reference);
    return error;
}
/* Caller has validated the index/state transition. Pixel copy retains its
 * own checks and does not publish the prepared picture. */
static SvtAv2Error stage_unchecked(SvtAv2ReferenceFrame* reference,
    const SvtAv2FrameBuffer* reconstruction, uint64_t frame_index) {
    const SvtAv2Error error = svt_av2_picture_copy_in(&reference->prepared, reconstruction);
    if (error != SVT_AV2_ERROR_NONE) return error;
    reference->prepared_index = frame_index;
    reference->prepared_valid = 1;
    return SVT_AV2_ERROR_NONE;
}
SvtAv2Error svt_av2_reference_prepare(SvtAv2ReferenceFrame* reference,
    const SvtAv2FrameBuffer* reconstruction, uint64_t frame_index) {
    return svt_av2_reference_prepare_interval(reference, reconstruction, frame_index, 1u);
}
SvtAv2Error svt_av2_reference_prepare_interval(SvtAv2ReferenceFrame* reference,
    const SvtAv2FrameBuffer* reconstruction, uint64_t frame_index, uint32_t interval) {
    if (reference == NULL) return SVT_AV2_ERROR_INVALID_PARAMETER;
    reference->prepared_valid = 0;
    if (interval == 0u) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if ((reference->valid && (reference->frame_index > UINT64_MAX - interval ||
         frame_index != reference->frame_index + interval)) ||
        (!reference->valid && frame_index != 0)) return SVT_AV2_ERROR_INVALID_STATE;
    return stage_unchecked(reference, reconstruction, frame_index);
}
static void publish_prepared(SvtAv2ReferenceFrame* reference) {
    const SvtAv2FrameBuffer previous = reference->committed;
    reference->committed = reference->prepared;
    reference->prepared = previous;
    reference->frame_index = reference->prepared_index;
    reference->valid = 1;
    reference->prepared_valid = 0;
}
SvtAv2Error svt_av2_reference_commit(SvtAv2ReferenceFrame* reference) {
    if (reference == NULL) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (!reference->prepared_valid) return SVT_AV2_ERROR_INVALID_STATE;
    publish_prepared(reference);
    return SVT_AV2_ERROR_NONE;
}
/* Caller has validated both slots and ensured the staging banks are idle. */
static SvtAv2Error refresh_pair_unchecked(SvtAv2ReferenceFrame* slot0,
    SvtAv2ReferenceFrame* slot1, const SvtAv2FrameBuffer* reconstruction,
    uint64_t frame_index, int refresh_long) {
    const uint64_t old_prepared_index = slot0->prepared_index;
    SvtAv2Error error = stage_unchecked(slot0, reconstruction, frame_index);
    if (error != SVT_AV2_ERROR_NONE) return error;
    if (refresh_long) {
        error = stage_unchecked(slot1, reconstruction, frame_index);
        if (error != SVT_AV2_ERROR_NONE) {
            /* Invalidate the private scratch pixels as well as their index;
             * neither committed buffer has been touched or exchanged. */
            slot0->prepared_index = old_prepared_index;
            slot0->prepared_valid = 0;
            return error;
        }
    }
    /* From here neither swap can fail, so a packet cannot publish only one of
     * its two refreshes. */
    publish_prepared(slot0);
    if (refresh_long) publish_prepared(slot1);
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_reference_refresh_pair_masked(SvtAv2ReferenceFrame* slot0,
    SvtAv2ReferenceFrame* slot1, const SvtAv2FrameBuffer* reconstruction,
    uint64_t frame_index, uint32_t refresh_mask) {
    return svt_av2_reference_refresh_after_holds(slot0, slot1, reconstruction, frame_index, refresh_mask, 0u);
}

SvtAv2Error svt_av2_reference_refresh_after_holds(SvtAv2ReferenceFrame* slot0,
    SvtAv2ReferenceFrame* slot1, const SvtAv2FrameBuffer* reconstruction,
    uint64_t frame_index, uint32_t refresh_mask, uint32_t held_pictures) {
    if (held_pictures && held_pictures >= frame_index) return SVT_AV2_ERROR_INVALID_STATE;
    if (slot0 == NULL || slot1 == NULL || slot0 == slot1 || reconstruction == NULL ||
        (frame_index == 0u ? refresh_mask != 0u : (refresh_mask != 1u && refresh_mask != 3u)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (slot0->prepared_valid || slot1->prepared_valid) return SVT_AV2_ERROR_INVALID_STATE;
    if (frame_index == 0u) {
        if (slot0->valid || slot1->valid) return SVT_AV2_ERROR_INVALID_STATE;
    } else {
        const uint64_t previous = frame_index - 1u - held_pictures;
        if (!slot0->valid || slot0->frame_index != previous ||
            (slot1->valid && (slot1->frame_index == 0u || slot1->frame_index > previous)))
            return SVT_AV2_ERROR_INVALID_STATE;
    }
    return refresh_pair_unchecked(slot0, slot1, reconstruction, frame_index, (refresh_mask & 2u) != 0u);
}

SvtAv2Error svt_av2_reference_store(SvtAv2ReferenceFrame* slot, const SvtAv2FrameBuffer* recon, uint64_t index) {
    if (!slot || !recon || slot->prepared_valid || (slot->valid && slot->frame_index >= index))
        return SVT_AV2_ERROR_INVALID_STATE;
    SvtAv2Error err = stage_unchecked(slot, recon, index);
    if (!err) publish_prepared(slot);
    return err;
}

SvtAv2Error svt_av2_reference_refresh_anchor(SvtAv2ReferenceFrame* recent, SvtAv2ReferenceFrame* older,
    const SvtAv2FrameBuffer* reconstruction, uint64_t frame_index, uint32_t held, int rotate) {
    if (!recent || !older || !reconstruction || recent == older) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (held >= frame_index || !recent->valid || recent->frame_index != frame_index - held - 1u ||
        recent->prepared_valid || older->prepared_valid ||
        (older->valid && older->frame_index >= recent->frame_index)) return SVT_AV2_ERROR_INVALID_STATE;
    SvtAv2ReferenceFrame* dest = rotate ? older : recent;
    const SvtAv2Error error = stage_unchecked(dest, reconstruction, frame_index);
    if (error != SVT_AV2_ERROR_NONE) return error;
    publish_prepared(dest);
    if (rotate) {
        const SvtAv2ReferenceFrame saved = *recent;
        *recent = *older;
        *older = saved;
    }
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_reference_refresh_pair(SvtAv2ReferenceFrame* slot0,
    SvtAv2ReferenceFrame* slot1, const SvtAv2FrameBuffer* reconstruction,
    uint64_t frame_index, uint32_t period) {
    if (slot0 == NULL || slot1 == NULL || slot0 == slot1 || reconstruction == NULL || period == 0u)
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (slot0->prepared_valid || slot1->prepared_valid)
        return SVT_AV2_ERROR_INVALID_STATE;
    if (frame_index == 0u) {
        if (slot0->valid || slot1->valid) return SVT_AV2_ERROR_INVALID_STATE;
    } else {
        const uint64_t previous = frame_index - 1u;
        const uint64_t expected_long = previous - previous % period;
        if (!slot0->valid || slot0->frame_index != previous ||
            (expected_long == 0u ? slot1->valid :
                !slot1->valid || slot1->frame_index != expected_long))
            return SVT_AV2_ERROR_INVALID_STATE;
    }
    /* The decoder discards the KEY alias in slot 1. Only a positive
     * cadence frame creates its first distinct resident picture. */
    const int refresh_long = frame_index != 0u && frame_index % period == 0u;
    return refresh_pair_unchecked(slot0, slot1, reconstruction, frame_index, refresh_long);
}
const SvtAv2FrameBuffer* svt_av2_reference_view(const SvtAv2ReferenceFrame* reference) {
    return reference != NULL && reference->valid ? &reference->committed : NULL;
}
enum { REF_PATCH_SIDE = 8, REF_PATCH_ROWS = 4, REF_PATCH_SAMPLES = REF_PATCH_SIDE * REF_PATCH_ROWS };

static int luma_valid(const SvtAv2FrameBuffer* pic) {
    if (pic == NULL || pic->planes[0] == NULL ||
        (pic->bit_depth != 8u && pic->bit_depth != 10u) ||
        pic->visible_width < REF_PATCH_SIDE || pic->visible_height < REF_PATCH_SIDE)
        return 0;
    const uint32_t bytes = pic->bit_depth == 10u ? 2u : 1u;
    return pic->strides[0] / bytes >= pic->visible_width &&
        (size_t)pic->visible_height <= SIZE_MAX / pic->strides[0];
}

static int luma_matches(const SvtAv2FrameBuffer* source, const SvtAv2FrameBuffer* ref) {
    return luma_valid(ref) && ref->visible_width == source->visible_width &&
        ref->visible_height == source->visible_height && ref->bit_depth == source->bit_depth &&
        ref->chroma_format == source->chroma_format;
}

static uint32_t patch_sad(const uint16_t src[REF_PATCH_SAMPLES],
    const SvtAv2FrameBuffer* ref, uint32_t x, uint32_t y) {
    const uint32_t bytes = ref->bit_depth == 10u ? 2u : 1u;
    uint32_t sad = 0u;
    for (uint32_t row = 0u; row < REF_PATCH_ROWS; ++row) {
        const uint8_t* ref_row = ref->planes[0] + (size_t)(y + 2u * row) * ref->strides[0] +
            (size_t)x * bytes;
        for (uint32_t col = 0u; col < REF_PATCH_SIDE; ++col) {
            const int32_t diff = (int32_t)src[row * REF_PATCH_SIDE + col] -
                svt_av2_sample_load(ref_row + (size_t)col * bytes, ref->bit_depth);
            sad += (uint32_t)(diff < 0 ? -diff : diff);
        }
    }
    return sad;
}

static uint32_t translated_patch_sad(const uint16_t src[REF_PATCH_SAMPLES],
    const SvtAv2FrameBuffer* ref, uint32_t x, uint32_t y, int32_t rx, int32_t ry) {
    int32_t best_x = 0, best_y = 0;
    uint32_t best = patch_sad(src, ref, x, y);
    /* Always test zero shift, then descend 4, 2, 1 from the previous winner.
     * The center of each ring is already scored. A patch tests <=25 MVs. */
    for (int32_t step = 4; step != 0 && best != 0u; step /= 2) {
        const int32_t center_x = best_x, center_y = best_y;
        for (int32_t dy = -step; dy <= step; dy += step) {
            for (int32_t dx = -step; dx <= step; dx += step) {
                if (!dx && !dy) continue;
                const int32_t ix = center_x + dx, iy = center_y + dy;
                if (ix < -rx || ix > rx || iy < -ry || iy > ry) continue;
                const uint32_t cost = patch_sad(src, ref,
                    (uint32_t)((int64_t)x + ix), (uint32_t)((int64_t)y + iy));
                if (cost < best) { best = cost; best_x = ix; best_y = iy; }
            }
        }
    }
    return best;
}

static uint32_t patch_grid_count(uint32_t extent, uint32_t cap) {
    const uint32_t n = extent / 16u;
    return n == 0u ? 1u : n > cap ? cap : n;
}

uint32_t svt_av2_reference_choose_slot(const SvtAv2FrameBuffer* source,
    const SvtAv2ReferenceFrame* slot0, const SvtAv2ReferenceFrame* slot1) {
    const SvtAv2FrameBuffer* recent = svt_av2_reference_view(slot0);
    const SvtAv2FrameBuffer* older = svt_av2_reference_view(slot1);
    if (recent == NULL || older == NULL || slot1->frame_index >= slot0->frame_index ||
        !luma_valid(source) || !luma_matches(source, recent) || !luma_matches(source, older))
        return 0u;

    const uint32_t nx = patch_grid_count(source->visible_width, 8u);
    const uint32_t ny = patch_grid_count(source->visible_height, 6u);
    const uint32_t room_x = source->visible_width - REF_PATCH_SIDE;
    const uint32_t room_y = source->visible_height - REF_PATCH_SIDE;
    const uint32_t rx = room_x < 14u ? room_x / 2u : 7u;
    const uint32_t ry = room_y < 14u ? room_y / 2u : 7u;
    const uint32_t bytes = source->bit_depth == 10u ? 2u : 1u;
    uint64_t recent_sad = 0u, older_sad = 0u;
    /* <=48 uniformly spread patches, 32 luma samples each. The margin
     * makes every reference read visible, even on a small or edged picture. */
    for (uint32_t i = 0u; i < ny; ++i) {
        const uint32_t y = ry + (uint32_t)(((uint64_t)room_y - 2u * ry) * (2u * i + 1u) / (2u * ny));
        for (uint32_t j = 0u; j < nx; ++j) {
            const uint32_t x = rx + (uint32_t)(((uint64_t)room_x - 2u * rx) * (2u * j + 1u) / (2u * nx));
            uint16_t patch[REF_PATCH_SAMPLES];
            for (uint32_t row = 0u; row < REF_PATCH_ROWS; ++row) {
                const uint8_t* src = source->planes[0] + (size_t)(y + 2u * row) * source->strides[0] +
                    (size_t)x * bytes;
                for (uint32_t col = 0u; col < REF_PATCH_SIDE; ++col)
                    patch[row * REF_PATCH_SIDE + col] = svt_av2_sample_load(src + (size_t)col * bytes,
                        source->bit_depth);
            }
            recent_sad += translated_patch_sad(patch, recent, x, y, (int32_t)rx, (int32_t)ry);
            older_sad += translated_patch_sad(patch, older, x, y, (int32_t)rx, (int32_t)ry);
        }
    }
    if (older_sad >= recent_sad) return 0u;
    const uint64_t gain = recent_sad - older_sad;
    const uint32_t depth_scale = source->bit_depth == 10u ? 4u : 1u;
    /* Require >5% relative and >0.25 8-bit-equivalent levels/sample.
     * This is a sampling heuristic, not a second full RDO candidate. */
    return (uint32_t)(gain * 20u > recent_sad &&
        gain * 4u > (uint64_t)nx * ny * REF_PATCH_SAMPLES * depth_scale);
}

SvtAv2Error svt_av2_reference_release(SvtAv2ReferenceFrame* reference) {
    if (reference == NULL) return SVT_AV2_ERROR_INVALID_PARAMETER;
    svt_av2_picture_free(&reference->committed);
    svt_av2_picture_free(&reference->prepared);
    memset(reference, 0, sizeof(*reference));
    return SVT_AV2_ERROR_NONE;
}
