#ifndef SVT_AV2_CDEF_LUMA_H
#define SVT_AV2_CDEF_LUMA_H
#include "picture.h"

/* One frame-wide primary strength, damping 4 and CDEF on skipped blocks.
 * If requested, return the owned, tightly packed pre-CDEF plane for restoration. */
SvtAv2Error svt_av2_cdef_luma_apply(SvtAv2FrameBuffer* recon, uint32_t primary, uint16_t** deblocked);
#endif
