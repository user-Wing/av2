#ifndef SVT_AV2_INTRA_IST_H
#define SVT_AV2_INTRA_IST_H

#include "intra_mode_map.h"

/* AV2 intra mode numbering for the IST set and transpose tables. */
static inline uint32_t svt_av2_ist_mode(uint32_t mode) {
    static const uint8_t modes[14] = {0,1,2,12,9,10,11,12,3,8,5,4,6,7};
    const uint32_t base = svt_av2_intra_mode_base(mode);
    return base < 14u ? modes[base] : 12u;
}

static inline uint32_t svt_av2_ist_eob_limit(uint32_t w, uint32_t h, uint32_t primary) {
    if (primary != 0u && primary != 3u) return 0u;
    if (w < 8u || h < 8u) return 8u;
    return primary == 3u || (w == 8u && h == 8u) ? 20u : 32u;
}

/* IST code zero selects the identity. Otherwise each set has three kernels. */
static inline uint32_t svt_av2_ist_type(uint32_t code) { return 1u + (code - 1u) % 3u; }
static inline uint32_t svt_av2_ist_set(uint32_t code) { return (code - 1u) / 3u; }

#endif
