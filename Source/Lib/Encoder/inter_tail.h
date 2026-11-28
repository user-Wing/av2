#ifndef SVT_AV2_INTER_TAIL_H
#define SVT_AV2_INTER_TAIL_H

#include <stdint.h>

#define SVT_AV2_INTER_TAIL_MAX 4u

/* The caller provides the canonical scan permutation for one supported packed
 * DCT block. The returned offsets follow reverse scan order, not raster order.
 * Only the returned prefix of out is written; the coefficient buffer is read only. */
static inline uint32_t svt_av2_inter_tail_positions(const int32_t* qcoeff,
    const uint16_t* scan, uint32_t count, uint16_t out[SVT_AV2_INTER_TAIL_MAX]) {
    if (!qcoeff || !scan || !out || (count < 64u || count > 1024u || (count & (count - 1u))))
        return 0u;

    const uint32_t scaled_floor = count / 64u;
    const uint32_t floor = scaled_floor > 3u ? scaled_floor : 3u;
    uint32_t selected = 0u;
    for (uint32_t pos = count; pos > floor && selected < SVT_AV2_INTER_TAIL_MAX;) {
        const uint16_t offset = scan[--pos];
        const int32_t level = qcoeff[offset];
        if (!level) continue;
        if (level != 1 && level != -1) break;
        out[selected++] = offset;
    }
    return selected;
}

#endif /* SVT_AV2_INTER_TAIL_H */
