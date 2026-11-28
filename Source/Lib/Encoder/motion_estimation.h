#ifndef SVT_AV2_MOTION_ESTIMATION_H
#define SVT_AV2_MOTION_ESTIMATION_H
#include "block_geometry.h"
#define SVT_AV2_ME_SEARCH_RADIUS 8
/* Absolute displacement in integer luma samples; reference=current+MV. */
typedef struct SvtAv2MotionVector { int32_t mv_x; int32_t mv_y; } SvtAv2MotionVector;
typedef struct SvtAv2MotionSearchResult {
    SvtAv2MotionVector mv;
    uint64_t sad;
    uint32_t candidates;
} SvtAv2MotionSearchResult;
/* Search-only colocated velocity, measured in eighths of a pixel per frame. */
#define SVT_AV2_ME_FIELD_COLS 32u
#define SVT_AV2_ME_FIELD_ROWS 18u
typedef struct SvtAv2MotionField {
    SvtAv2MotionVector velocity[SVT_AV2_ME_FIELD_ROWS][SVT_AV2_ME_FIELD_COLS];
} SvtAv2MotionField;
void svt_av2_motion_temporal(const SvtAv2FrameBuffer* source, const SvtAv2FrameBuffer* reference,
    const SvtAv2BlockView* block, const SvtAv2MotionField* field, int distance,
    int radius, SvtAv2MotionSearchResult* result);
/* Quarter-resolution luma is an average of each 4x4 source neighborhood. */
void svt_av2_motion_downsample4(const SvtAv2FrameBuffer* frame,uint16_t* quarter);
void svt_av2_motion_hme(const SvtAv2FrameBuffer* source,const SvtAv2FrameBuffer* reference,
    const uint16_t* quarter_source,const uint16_t* quarter_ref,const SvtAv2BlockView* block,
    int radius, SvtAv2MotionSearchResult* result);
/* Extend a successful integer search on high-residual blocks, within the
 * projected Q3 predictor's +/-16 pixel range. */
void svt_av2_motion_extend(const SvtAv2FrameBuffer* source, const SvtAv2FrameBuffer* reference,
    const SvtAv2BlockView* block, SvtAv2MotionSearchResult* result);
SvtAv2Error svt_av2_motion_estimate(const SvtAv2FrameBuffer* source,
    const SvtAv2FrameBuffer* reference, const SvtAv2BlockView* luma_block,
    SvtAv2MotionSearchResult* result);
/* Exact top-three SAD candidates followed by zero MV when not already kept.
 * Slot zero uses the same tie rule as the legacy single-winner entry point. */
typedef struct SvtAv2MotionSearchList {
    SvtAv2MotionVector mv[4];
    uint64_t sad[4];
    uint32_t count, candidates;
} SvtAv2MotionSearchList;
SvtAv2Error svt_av2_motion_estimate_list(const SvtAv2FrameBuffer* source,
    const SvtAv2FrameBuffer* reference, const SvtAv2BlockView* luma_block,
    SvtAv2MotionSearchList* result);
/* Fixed outer-grid search, then one fixed-center refinement; <=80 positions.
 * Returns the best two distinct visited MVs outside the legacy +/-8 square. */
SvtAv2Error svt_av2_motion_estimate_wide(const SvtAv2FrameBuffer* source,
    const SvtAv2FrameBuffer* reference, const SvtAv2BlockView* luma_block,
    SvtAv2MotionSearchList* result);
#endif
