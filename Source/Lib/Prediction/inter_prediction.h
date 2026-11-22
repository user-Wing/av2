#ifndef SVT_AV2_INTER_PREDICTION_H
#define SVT_AV2_INTER_PREDICTION_H
#include "motion_estimation.h"
/* Fixed AV2 BILINEAR frame filter. Destination stride is uint16 samples. */
SvtAv2Error svt_av2_inter_predict(const SvtAv2FrameBuffer* reference, uint32_t plane,
    const SvtAv2BlockView* block, SvtAv2MotionVector mv, uint16_t* prediction,
    uint32_t prediction_stride);

/* Each component is in units of 1/2 luma sample, regardless of plane. */
typedef struct SvtAv2HalfPelMotionVector {
    int32_t mv_x;
    int32_t mv_y;
} SvtAv2HalfPelMotionVector;

/* Single unscaled reference with fixed AV2 BILINEAR and edge replication.
 * The block is in plane coordinates; destination stride is in uint16 samples. */
SvtAv2Error svt_av2_inter_predict_half(const SvtAv2FrameBuffer* reference, uint32_t plane,
    const SvtAv2BlockView* block, SvtAv2HalfPelMotionVector mv, uint16_t* prediction,
    uint32_t prediction_stride);

/* Single unscaled reference with fixed AV2 MULTITAP_SHARP (4-tap REGULAR
 * on each plane axis of at most four samples). Same units as BILINEAR above. */
SvtAv2Error svt_av2_inter_predict_half_sharp(const SvtAv2FrameBuffer* reference, uint32_t plane,
    const SvtAv2BlockView* block, SvtAv2HalfPelMotionVector mv, uint16_t* prediction,
    uint32_t prediction_stride);

/* Each component is in units of 1/4 luma sample, regardless of plane. */
typedef struct SvtAv2QuarterPelMotionVector {
    int32_t mv_x;
    int32_t mv_y;
} SvtAv2QuarterPelMotionVector;

/* Same fixed filter and geometry as half_sharp; quarter-luma MV units. */
SvtAv2Error svt_av2_inter_predict_quarter_sharp(const SvtAv2FrameBuffer* reference, uint32_t plane,
    const SvtAv2BlockView* block, SvtAv2QuarterPelMotionVector mv, uint16_t* prediction,
    uint32_t prediction_stride);

/* Exact fixed SHARP predictor in eighth-luma units; can represent odd
 * spatially projected MVP components even in a quarter-precision frame.
 * This private entry does not change which MVs the bitstream writer accepts. */
typedef struct SvtAv2EighthPelMotionVector {
    int32_t mv_x, mv_y;
} SvtAv2EighthPelMotionVector;

SvtAv2Error svt_av2_inter_predict_eighth_sharp(const SvtAv2FrameBuffer* reference, uint32_t plane,
    const SvtAv2BlockView* block, SvtAv2EighthPelMotionVector mv, uint16_t* prediction,
    uint32_t prediction_stride);

/* Luma-only shortlist metric for width <= 64. Exact SHARP SAD if <= limit,
 * otherwise returns limit+1. UINT64_MAX requests the complete cost. This
 * function never writes reconstructed samples; reference extension and the
 * two convolution rounding stages are shared with the normal predictor. */
SvtAv2Error svt_av2_inter_luma_sad_quarter_sharp(const SvtAv2FrameBuffer* source,
    const SvtAv2FrameBuffer* reference, const SvtAv2BlockView* block,
    SvtAv2QuarterPelMotionVector mv, uint64_t limit, uint64_t* sad);
/* Same metric/limit contract on the actual eighth-luma lattice. On failure
 * neither entry modifies *sad. */
SvtAv2Error svt_av2_inter_luma_sad_eighth_sharp(const SvtAv2FrameBuffer* source,
    const SvtAv2FrameBuffer* reference, const SvtAv2BlockView* block,
    SvtAv2EighthPelMotionVector mv, uint64_t limit, uint64_t* sad);
#endif
