#include "intra_ist_transform.h"
#include "intra_ist_data.h"

#include <string.h>

static int32_t round_clip(int32_t value, uint32_t bit_depth) {
    const int32_t bound = 1 << (7u + bit_depth);
    const int32_t rounded = value < 0 ? -((-value + 64) >> 7) : (value + 64) >> 7;
    return rounded < -bound ? -bound : rounded >= bound ? bound - 1 : rounded;
}

void svt_av2_intra_ist_transform(int32_t* coeff, const uint16_t* scan,
    uint32_t w, uint32_t h, uint32_t bit_depth, uint32_t mode,
    uint32_t primary, uint32_t code, int inverse) {
    const uint32_t set = svt_av2_ist_set(code);
    const uint32_t type = svt_av2_ist_type(code) - 1u;
    const uint32_t rows = svt_av2_ist_eob_limit(w, h, primary);
    const int transpose = mode == 2u || mode == 6u || mode == 8u || mode == 11u;
    const uint8_t* mapping = ist_8x8_map[set][type];
    const int16_t (*kernel)[48] = ist_8x8_kernel[set][type];
    const uint32_t stride = w > 32u ? 32u : w;
    const size_t count = (size_t)stride * (h > 32u ? 32u : h);
    uint16_t positions[48];
    for (uint32_t i = 0u; i < 48u; ++i) {
        const uint32_t position = ist_8x8_scan[mapping[i]];
        const uint32_t x = position & 7u, y = position >> 3u;
        positions[i] = (uint16_t)(transpose ? x * stride + y : y * stride + x);
    }
    int32_t input[48];
    if (inverse) {
        for (uint32_t i = 0u; i < rows; ++i) input[i] = coeff[scan[i]];
        memset(coeff, 0, count * sizeof(*coeff));
        for (uint32_t i = 0u; i < 48u; ++i) {
            int32_t sum = 0;
            for (uint32_t j = 0u; j < rows; ++j) sum += kernel[j][i] * input[j];
            coeff[positions[i]] = round_clip(sum, bit_depth);
        }
    } else {
        for (uint32_t i = 0u; i < 48u; ++i) input[i] = coeff[positions[i]];
        memset(coeff, 0, count * sizeof(*coeff));
        for (uint32_t i = 0u; i < rows; ++i) {
            int32_t sum = 0;
            for (uint32_t j = 0u; j < 48u; ++j) sum += kernel[i][j] * input[j];
            coeff[scan[i]] = round_clip(sum, bit_depth);
        }
    }
}
