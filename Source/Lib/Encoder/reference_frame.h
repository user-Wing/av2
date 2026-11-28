#ifndef SVT_AV2_REFERENCE_FRAME_H
#define SVT_AV2_REFERENCE_FRAME_H
#include "picture.h"

/* Prepared pixels remain private until commit. Both buffers are allocated at init,
 * so committing a serialized packet cannot fail due to allocation. */
typedef struct SvtAv2ReferenceFrame {
    SvtAv2FrameBuffer committed;
    SvtAv2FrameBuffer prepared;
    uint64_t frame_index;
    uint64_t prepared_index;
    int valid;
    int prepared_valid;
} SvtAv2ReferenceFrame;
SvtAv2Error svt_av2_reference_init(SvtAv2ReferenceFrame* reference, uint32_t width,
    uint32_t height, uint32_t bit_depth, SvtAv2ChromaFormat chroma_format);
SvtAv2Error svt_av2_reference_prepare(SvtAv2ReferenceFrame* reference,
    const SvtAv2FrameBuffer* reconstruction, uint64_t frame_index);
/* As prepare(), but with an exact, caller-proven cadence between refreshes. */
SvtAv2Error svt_av2_reference_prepare_interval(SvtAv2ReferenceFrame* reference,
    const SvtAv2FrameBuffer* reconstruction, uint64_t frame_index, uint32_t interval);
SvtAv2Error svt_av2_reference_commit(SvtAv2ReferenceFrame* reference);
/* Refresh an independently allocated pair with no outstanding preparations.
 * Slot 0 advances each picture, including index 0. Slot 1 stays invalid
 * until the first positive period multiple, then advances at each multiple.
 * This mirrors decoder removal of duplicate initial KEY references.
 * Failed calls never change either committed picture. */
SvtAv2Error svt_av2_reference_refresh_pair(SvtAv2ReferenceFrame* slot0,
    SvtAv2ReferenceFrame* slot1, const SvtAv2FrameBuffer* reconstruction,
    uint64_t frame_index, uint32_t period);
/* Refresh the same idle pair using the already chosen packet mask. KEY
 * (index 0) takes internal mask 0, commits slot 0, and leaves slot 1 invalid.
 * P takes 1 (only LAST) or 3 (LAST and LONG); LAST must contain index-1,
 * and a valid LONG must contain a positive index no newer than LAST. No
 * cadence is inferred here. A failed stage publishes neither slot. */
SvtAv2Error svt_av2_reference_refresh_pair_masked(SvtAv2ReferenceFrame* slot0,
    SvtAv2ReferenceFrame* slot1, const SvtAv2FrameBuffer* reconstruction,
    uint64_t frame_index, uint32_t refresh_mask);
/* A display-only SEF leaves both reference slots untouched. */
SvtAv2Error svt_av2_reference_refresh_after_holds(SvtAv2ReferenceFrame* slot0,
    SvtAv2ReferenceFrame* slot1, const SvtAv2FrameBuffer* reconstruction,
    uint64_t frame_index, uint32_t refresh_mask, uint32_t held_pictures);
/* Preserve the last anchor when opening a new short-reference chain. */
SvtAv2Error svt_av2_reference_refresh_anchor(SvtAv2ReferenceFrame* recent, SvtAv2ReferenceFrame* older,
    const SvtAv2FrameBuffer* reconstruction, uint64_t frame_index, uint32_t held, int rotate);
/* Luma patch SAD with bounded local integer translation. Slot 1 must beat
 * both an absolute and a relative margin; prefer slot 0 for ties, missing or
 * incompatible pictures, or when both slots hold the same/newer index. */
uint32_t svt_av2_reference_choose_slot(const SvtAv2FrameBuffer* source,
    const SvtAv2ReferenceFrame* slot0, const SvtAv2ReferenceFrame* slot1);
const SvtAv2FrameBuffer* svt_av2_reference_view(const SvtAv2ReferenceFrame* reference);
SvtAv2Error svt_av2_reference_release(SvtAv2ReferenceFrame* reference);
#endif
