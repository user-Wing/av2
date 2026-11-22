/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include "cdef_luma.h"

#include <stdlib.h>

static int floor_log2(int value) {
    int bits = 0;
    while (value > 1) { value >>= 1; ++bits; }
    return bits;
}

/* Equivalent to the line-length-weighted AVM cdef_find_dir_c scores. */
static int find_direction(const uint16_t* src, int stride, int shift, int* variance) {
    int sums[8][15] = {{0}};
    uint8_t lengths[8][15] = {{0}};
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            const int value = (src[y * stride + x] >> shift) - 128;
            const int line[8] = {y+x, y+x/2, y, 3+y-x/2, 7+y-x, 3-y/2+x, x, y/2+x};
            for (int d = 0; d < 8; ++d) {
                sums[d][line[d]] += value;
                ++lengths[d][line[d]];
            }
        }
    }
    int cost[8] = {0}, best = 0;
    for (int d = 0; d < 8; ++d) {
        for (int i = 0; i < 15; ++i)
            if (lengths[d][i]) cost[d] += sums[d][i] * sums[d][i] * (840 / lengths[d][i]);
        if (cost[d] > cost[best]) best = d;
    }
    *variance = (cost[best] - cost[(best + 4) & 7]) >> 10;
    return best;
}

SvtAv2Error svt_av2_cdef_luma_apply(SvtAv2FrameBuffer* recon, uint32_t primary, uint16_t** deblocked) {
    if (!recon || !recon->planes[0] || !primary || primary > 15u ||
        (recon->bit_depth != 8u && recon->bit_depth != 10u) ||
        recon->visible_width < 8u || recon->visible_height < 8u ||
        (recon->visible_width & 7u) || (recon->visible_height & 7u) ||
        recon->visible_width > 4096u || recon->visible_height > 2160u)
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    const int width = (int)recon->visible_width, height = (int)recon->visible_height;
    const int shift = (int)recon->bit_depth - 8, step = shift ? 2 : 1;
    uint16_t* src = (uint16_t*)malloc((size_t)width * height * sizeof(*src));
    if (!src) return SVT_AV2_ERROR_OUT_OF_MEMORY;
    for (int y = 0; y < height; ++y) {
        const uint8_t* row = recon->planes[0] + (size_t)y * recon->strides[0];
        for (int x = 0; x < width; ++x) src[y * width + x] = svt_av2_sample_load(row + x * step, recon->bit_depth);
    }
    static const int dx[8][2] = {{1,2},{1,2},{1,2},{1,2},{1,2},{0,1},{0,0},{0,-1}};
    static const int dy[8][2] = {{-1,-2},{0,-1},{0,0},{0,1},{1,2},{1,2},{1,2},{1,2}};
    for (int by = 0; by < height; by += 8) for (int bx = 0; bx < width; bx += 8) {
        int variance;
        const int dir = find_direction(src + by * width + bx, width, shift, &variance);
        if (!variance) continue;
        const int scale = SVT_AV2_MIN(12, floor_log2(variance >> 6));
        const int strength = (((int)primary << shift) * (4 + scale) + 8) >> 4;
        if (!strength) continue;
        const int damp = SVT_AV2_MAX(0, 4 + shift - floor_log2(strength));
        const int odd = (strength >> shift) & 1;
        for (int y = by; y < by + 8; ++y) for (int x = bx; x < bx + 8; ++x) {
            const int center = src[y * width + x];
            int sum = 0;
            for (int k = 0; k < 2; ++k) for (int sign = -1; sign <= 1; sign += 2) {
                const int xx = x + sign * dx[dir][k], yy = y + sign * dy[dir][k];
                if (xx < 0 || xx >= width || yy < 0 || yy >= height) continue;
                const int delta = src[yy * width + xx] - center;
                const int magnitude = abs(delta);
                const int amount = SVT_AV2_MIN(magnitude, SVT_AV2_MAX(0, strength - (magnitude >> damp)));
                const int tap = k ? 2 + odd : 4 - odd;
                sum += tap * (delta < 0 ? -amount : amount);
            }
            svt_av2_sample_store(recon->planes[0] + (size_t)y * recon->strides[0] + x * step, recon->bit_depth,
                                 (uint16_t)(center + ((8 + sum - (sum < 0)) >> 4)));
        }
    }
    if (deblocked) *deblocked = src;
    else free(src);
    return SVT_AV2_ERROR_NONE;
}
