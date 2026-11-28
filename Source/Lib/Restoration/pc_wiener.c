/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. See AVM_LICENSE and
 * AVM_PATENTS in this directory.
 *
 * Scalar adaptation of fixed AVM restoration.c / convolve.c. The reference's
 * rolling 6x6 feature sums are evaluated directly at each 4x4 block. This
 * preserves the integer classifier and avoids decoder/line-buffer state.
 */
#include "pc_wiener.h"
#include "pc_wiener_tables.h"
#include "quantization.h"

#include <stdlib.h>

static int clamp_int(int x, int low, int high) {
    return x < low ? low : x > high ? high : x;
}

static int32_t round_signed(int32_t x, unsigned shift) {
    const int32_t half = (int32_t)(1u << (shift - 1u));
    return x < 0 ? -((-x + half) >> shift) : (x + half) >> shift;
}

/* Before each stripe, map its two halo rows to the decoder's saved
 * pre-CDEF lines. At frame edges the outermost post-CDEF row is repeated. */
static int sample(const uint16_t* const* lines, int x, int y, int width, int top, int bottom) {
    return lines[clamp_int(y-top+2,0,bottom-top+3)][clamp_int(x,0,width-1)];
}

int svt_av2_pcwiener_filter_y_parallel(const uint16_t* src, ptrdiff_t src_stride,
    uint16_t* dst, ptrdiff_t dst_stride, uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t base_q_idx, const uint8_t* tskip,
    ptrdiff_t tskip_stride, const uint16_t* deblocked, uint32_t threads) {
    if (!src || !dst || src == dst || !tskip || !width || !height ||
        width % 8u || height % 8u || width > 16384u || height > 16384u ||
        src_stride < (ptrdiff_t)width || dst_stride < (ptrdiff_t)width ||
        tskip_stride < (ptrdiff_t)(width / 4u) ||
        (bit_depth != 8u && bit_depth != 10u)) return -1;
    SvtAv2QuantParams quant;
    if (svt_av2_quant_params_init_tx_bd(&quant,base_q_idx,8u,8u,bit_depth) != SVT_AV2_QUANT_OK)
        return -1;
    const int set = base_q_idx < 130u ? 0 : base_q_idx < 190u ? 1 : base_q_idx < 220u ? 2 : 3;
    const int16_t (*sets[4])[NUM_PC_WIENER_TAPS_LUMA] = {
        pcwiener_filters_luma_0,pcwiener_filters_luma_1,
        pcwiener_filters_luma_2,pcwiener_filters_luma_3};
    const uint8_t* selectors[4] = {pc_wiener_sub_classify_to_64_0,
        pc_wiener_sub_classify_to_64_1,pc_wiener_sub_classify_to_64_2,
        pc_wiener_sub_classify_to_64_3};
    const int16_t (*filters)[NUM_PC_WIENER_TAPS_LUMA] = sets[set];
    const uint8_t* selector = selectors[set];
    const unsigned bd_shift = bit_depth - 8u;
    int qstep = quant.plane[0].ac;
    unsigned qshift = 5u + bd_shift + 8u;
    if (bd_shift) { qstep = round_signed(qstep,bd_shift);qshift -= bd_shift; }
    int32_t qval[253][4];
    for (int t = 0;t <= 252;++t) {
        const int shifted = t * (1 << (qshift - 8u));
        const int product = round_signed(t * qstep,8u);
        for (int k = 0;k < 4;++k) {
            const int32_t v = mode_weights[k][0] * shifted + mode_weights[k][1] * qstep +
                mode_weights[k][2] * product;
            qval[t][k] = 255 * (round_signed(v,qshift) + mode_offsets[k]);
        }
    }
    static const int dy[12] = {1,0,2,0,1,-1,2,2,1,1,3,0};
    static const int dx[12] = {0,1,0,2,1,1,1,-1,2,-2,0,3};
    const int max_sample = (1 << bit_depth) - 1;
    const int w = (int)width,h = (int)height;
    for (int y=0; y<h/4; ++y) for (int x=0; x<w/4; ++x)
        if (tskip[(ptrdiff_t)y*tskip_stride+x] > 1u) return -1;
    if (!threads) threads = 1u;
    int stripe;
#ifdef _OPENMP
#pragma omp parallel for if(threads > 1u) num_threads(threads) schedule(static)
#else
    (void)threads;
#endif
    for (stripe = -8; stripe < h; stripe += 64) {
        const int top = stripe < 0 ? 0 : stripe;
        const int bottom = stripe+64 < h ? stripe+64 : h;
        const uint16_t* lines[68];
        for (int y = top-2; y < bottom+2; ++y) {
            const int inside_frame = y >= 0 && y < h;
            const uint16_t* plane = deblocked && inside_frame && (y < top || y >= bottom) ? deblocked : src;
            lines[y-top+2] = plane + (ptrdiff_t)clamp_int(y,0,h-1) * src_stride;
        }
        for (int left = 0;left < w;left += 64) {
            const int right = left + 64 < w ? left + 64 : w;
            enum { pitch = 70 };
            uint16_t padded[pitch * pitch];
            for (int y=top-3; y<bottom+3; ++y) for (int x=left-3; x<right+3; ++x)
                padded[(y-top+3)*pitch+x-left+3]=(uint16_t)sample(lines,x,y,w,top,bottom);
            for (int by = top;by < bottom;by += 4) {
                for (int bx = left;bx < right;bx += 4) {
                    int32_t features[4] = {0,0,0,0};
                    int skipped = 0;
                    for (int fy = by - 1;fy <= by + 4;++fy) {
                        for (int fx = bx - 1;fx <= bx + 4;++fx) {
                            const uint16_t* p = padded + (fy-top+3)*pitch + (fx-left+3);
                            const int center = 2*p[0];
                            features[1] += abs(p[-pitch] + p[pitch] - center);
                            features[2] += abs(p[1-pitch] + p[pitch-1] - center);
                            features[3] += abs(p[-1-pitch] + p[pitch+1] - center);
                        }
                    }
                    /* The 6x6 window covers 1,4,1 pixels in each of three skip cells. */
                    const int col=bx/4, lo=clamp_int(col-1,left/4,right/4-1), hi=clamp_int(col+1,left/4,right/4-1);
                    for (int j=-1; j<=1; ++j) {
                        const uint8_t* row = tskip + (ptrdiff_t)clamp_int(by/4+j,top/4,bottom/4-1)*tskip_stride;
                        skipped += (j==0 ? 4 : 1) * (row[lo]+4*row[col]+row[hi]);
                    }
                    skipped *= tskip_normalizer;
                    int lut = 0;
                    for (int k = 0;k < 4;++k) {
                        int32_t value = features[k] * feature_normalizers_luma[k];
                        if (bd_shift) value = round_signed(value,bd_shift);
                        value = round_signed(value + qval[skipped][k],14u);
                        lut += pc_wiener_thresholds[k] * (clamp_int(value,0,255) >> pc_wiener_threshold_shift);
                    }
                    const int16_t* filter = filters[selector[pc_wiener_lut_to_class_index[lut]]];
                    for (int y = by;y < by + 4;++y) {
                        for (int x = bx;x < bx + 4;++x) {
                            const uint16_t* p = padded + (y-top+3)*pitch + (x-left+3);
                            int32_t v = (128 + filter[12]) * p[0];
                            for (int k = 0;k < 12;++k) {
                                const int offset = dy[k]*pitch+dx[k];
                                v += filter[k] * (p[offset] + p[-offset]);
                            }
                            dst[(ptrdiff_t)y * dst_stride + x] = (uint16_t)clamp_int(round_signed(v,7u),0,max_sample);
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int svt_av2_pcwiener_filter_y_boundaries(const uint16_t* src, ptrdiff_t src_stride,
    uint16_t* dst, ptrdiff_t dst_stride, uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t base_q_idx, const uint8_t* tskip,
    ptrdiff_t tskip_stride, const uint16_t* deblocked) {
    return svt_av2_pcwiener_filter_y_parallel(src,src_stride,dst,dst_stride,width,height,
        bit_depth,base_q_idx,tskip,tskip_stride,deblocked,1u);
}

int svt_av2_pcwiener_filter_y(const uint16_t* src, ptrdiff_t src_stride,
    uint16_t* dst, ptrdiff_t dst_stride, uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t base_q_idx, const uint8_t* tskip,
    ptrdiff_t tskip_stride) {
    return svt_av2_pcwiener_filter_y_boundaries(src,src_stride,dst,dst_stride,width,height,
        bit_depth,base_q_idx,tskip,tskip_stride,NULL);
}
