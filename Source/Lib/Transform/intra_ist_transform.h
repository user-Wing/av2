#ifndef SVT_AV2_INTRA_IST_TRANSFORM_H
#define SVT_AV2_INTRA_IST_TRANSFORM_H

#include "intra_ist.h"

/* Forward and transposed inverse of the AV2 48-coefficient support region.
 * Input/output are the primary transform's packed grid (up to 32 per side).
 * Callers supply its scan and validate the coded EOB before inverse. */
void svt_av2_intra_ist_transform(int32_t* coeff, const uint16_t* scan,
    uint32_t w, uint32_t h, uint32_t bit_depth, uint32_t mode,
    uint32_t primary, uint32_t code, int inverse);

#endif
