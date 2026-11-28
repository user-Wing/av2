#ifndef SVT_AV2_GDF_LUMA_H
#define SVT_AV2_GDF_LUMA_H
#include "picture.h"

/* For a single-tile KEY, filter the restored Y plane using the pre-restoration
 * guide and the two deblocked boundary lines at each internal stripe. */
SvtAv2Error svt_av2_gdf_luma(SvtAv2FrameBuffer* frame, const uint16_t* guide,
    const uint16_t* deblocked, uint32_t base_q_idx, uint32_t code, uint32_t ref_distance, uint32_t threads);
/* Four models and four strengths compete with unfiltered sampled units. */
SvtAv2Error svt_av2_gdf_select(SvtAv2FrameBuffer* recon, const SvtAv2FrameBuffer* source,
    const uint16_t* guide, uint32_t qindex, uint32_t distance, uint32_t* code);
uint16_t* svt_av2_gdf_copy_luma(const SvtAv2FrameBuffer* frame);
#endif
