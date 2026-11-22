#ifndef M6_SATD_ANGLES
#define M6_SATD_ANGLES 0
#endif
#ifndef M6_LAMBDA_NUMERATOR
#define M6_LAMBDA_NUMERATOR 1u
#endif
#ifndef M6_LAMBDA_DENOMINATOR
#define M6_LAMBDA_DENOMINATOR 1u
#endif
#if M6_LAMBDA_NUMERATOR <= 0 || M6_LAMBDA_DENOMINATOR <= 0
#error Lambda scale must be positive
#endif
#ifndef M6_FINE16
#define M6_FINE16 0
#endif
#ifndef M6_FINE16_PAIR
#define M6_FINE16_PAIR 0
#endif
#if M6_FINE16_PAIR && !M6_FINE16
#error M6_FINE16_PAIR_requires_M6_FINE16
#endif
#ifndef M6_FINE16_RETAIN_OLD
#define M6_FINE16_RETAIN_OLD 0
#endif
#ifndef M6_FINE16_REVERSE_CHOICES
#define M6_FINE16_REVERSE_CHOICES 0
#endif
#ifndef M6_TREE_SHARED
#define M6_TREE_SHARED 0
#endif
#ifndef M6_TOOL_REPEAT
#define M6_TOOL_REPEAT 0
#endif
#ifndef M6_REPEAT_RETAIN_OLD
#define M6_REPEAT_RETAIN_OLD 0
#endif
#ifndef M6_TREE_RETAIN_OLD
#define M6_TREE_RETAIN_OLD 0
#endif
#ifndef M6_TREE_RDOQ
#define M6_TREE_RDOQ 0
#endif
#ifndef M6_RDOQ_CONTEXT
#define M6_RDOQ_CONTEXT 0
#endif

#ifndef M6_ROOT_ASYM
#define M6_ROOT_ASYM 0
#endif
#include "inter_profile.h"
#include "rd_frame.h"
#include "transform_path.h"
#include "transform.h"
#include "intra_ist_transform.h"
#include "coeff_syntax.h"
#include "inter_prediction.h"
#include "inter_tail.h"
#include "motion_two_ref_mvp.h"
#include "intra_prediction.h"
#include "motion_estimation.h"
#include "quantization.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SVT_AV2_INTER_ME_REUSE
#define SVT_AV2_INTER_ME_REUSE 1
#endif
#ifndef SVT_AV2_INTER_PIXEL_REUSE
#define SVT_AV2_INTER_PIXEL_REUSE 1
#endif
#ifndef SVT_AV2_INTER_COARSE_PREFIX
#define SVT_AV2_INTER_COARSE_PREFIX 1
#endif
#ifndef SVT_AV2_INTER_COARSE_PREFIX_VERIFY
#define SVT_AV2_INTER_COARSE_PREFIX_VERIFY 0
#endif
#ifndef SVT_AV2_INTER_COARSE_RATE_PREFIX
#define SVT_AV2_INTER_COARSE_RATE_PREFIX 1
#endif
#ifndef SVT_AV2_INTER_COARSE_RATE_PREFIX_VERIFY
#define SVT_AV2_INTER_COARSE_RATE_PREFIX_VERIFY 0
#endif
#define RD_MAX_LEAVES 64u
#ifndef M6_FINE16_GEOMETRY
#define M6_FINE16_GEOMETRY M6_FINE16
#endif
#define RD_RDOQ_LEAF (UINT32_MAX - 1u)
#define INTER_FINE_MAX_LEAVES 16u /* Four quadrants can each split into four 16x16 leaves. */
#define RD_MAX_COEFFS (3u * 64u * 64u)
#define RD_HASH_OFFSET UINT64_C(14695981039346656037)
#define RD_HASH_PRIME UINT64_C(1099511628211)

SvtAv2Error svt_av2_rd_lambda(uint32_t qindex, uint32_t bit_depth,
    uint64_t* lambda_q8, uint32_t* qstep_qtx) {
    SvtAv2QuantParams params;
    if (lambda_q8 == NULL || qstep_qtx == NULL ||
        svt_av2_quant_params_init_tx_bd(&params, qindex, 16u, 16u, bit_depth) != SVT_AV2_QUANT_OK)
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    const uint64_t q = (uint32_t)params.plane[0].ac;
    const unsigned shift = 2u + 2u * (bit_depth - 8u);
    *qstep_qtx = (uint32_t)q;
    *lambda_q8 = (q * q + (UINT64_C(1) << (shift - 1u))) >> shift;
    return SVT_AV2_ERROR_NONE;
}

static SvtAv2Error rd_lambda_ai_scaled(uint32_t qindex, uint32_t bit_depth,
    uint32_t numerator, uint32_t denominator, uint64_t* lambda_q8, uint32_t* qstep_qtx) {
    SvtAv2Error error = svt_av2_rd_lambda(qindex, bit_depth, lambda_q8, qstep_qtx);
    if (error != SVT_AV2_ERROR_NONE) return error;
    const uint64_t q = *qstep_qtx;
    uint64_t rdmult = (q * q * 80u / 32u + 32u) >> 6u;
    if (bit_depth == 10u) rdmult = (rdmult + 8u) >> 4u;
    if (rdmult == 0u) rdmult = 1u;
    *lambda_q8 = (rdmult + 1u) >> 1u;
    if (numerator != denominator) {
        *lambda_q8 = (*lambda_q8 * numerator + denominator / 2u) / denominator;
        if (*lambda_q8 == 0u) *lambda_q8 = 1u;
    }
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_rd_lambda_ai(uint32_t qindex, uint32_t bit_depth,
    uint64_t* lambda_q8, uint32_t* qstep_qtx) {
    return rd_lambda_ai_scaled(qindex, bit_depth, M6_LAMBDA_NUMERATOR,
        M6_LAMBDA_DENOMINATOR, lambda_q8, qstep_qtx);
}

SvtAv2Error svt_av2_rd_lambda_ai150(uint32_t qindex, uint32_t bit_depth,
    uint64_t* lambda_q8, uint32_t* qstep_qtx) {
    return rd_lambda_ai_scaled(qindex, bit_depth, 6u, 4u, lambda_q8, qstep_qtx);
}

SvtAv2Error svt_av2_rd_cost(uint64_t distortion, uint64_t lambda_q8,
    uint64_t rate_q12, uint64_t* cost) {
    /* Exact rounded product / 2^20 without requiring nonstandard uint128.
     * Every term is nonnegative, so overflow can be rejected before multiply. */
    const uint64_t mask = (UINT64_C(1) << 20u) - 1u;
    const uint64_t a = lambda_q8 >> 20u, b = lambda_q8 & mask;
    const uint64_t c = rate_q12 >> 20u, d = rate_q12 & mask;
    if (cost == NULL || (a != 0u && rate_q12 > UINT64_MAX / a) ||
        (b != 0u && c > UINT64_MAX / b)) return SVT_AV2_ERROR_INVALID_PARAMETER;
    const uint64_t high = a * rate_q12, middle = b * c;
    const uint64_t low = (b * d + (UINT64_C(1) << 19u)) >> 20u;
    if (high > UINT64_MAX - middle || high + middle > UINT64_MAX - low ||
        high + middle + low > UINT64_MAX - distortion) return SVT_AV2_ERROR_INVALID_PARAMETER;
    *cost = distortion + high + middle + low;
    return SVT_AV2_ERROR_NONE;
}

uint64_t svt_av2_rd_distortion(const uint64_t sse[3], uint32_t bit_depth,
    uint32_t chroma_format) {
    const uint64_t weight = chroma_format == SVT_AV2_CHROMA_420 ? 4u : 1u;
    const unsigned shift = 2u * (bit_depth - 8u);
    const uint64_t value = 4u * sse[0] + weight * (sse[1] + sse[2]);
    return shift == 0u ? value : (value + (UINT64_C(1) << (shift - 1u))) >> shift;
}

#ifndef M6_Y_BOUND
#define M6_Y_BOUND 0
#endif
#ifndef M6_Y_BOUND_VERIFY
#define M6_Y_BOUND_VERIFY 0
#endif
#ifndef M6_TX_CACHE
#define M6_TX_CACHE 0
#endif
#ifndef M6_TX_CACHE_SLOTS
#define M6_TX_CACHE_SLOTS 1024u
#endif
#ifndef M6_TX_CACHE_WAYS
#define M6_TX_CACHE_WAYS 1u
#endif
#if M6_TX_CACHE
#include <stdio.h>
#if M6_TX_CACHE_WAYS == 0 || (M6_TX_CACHE_WAYS & (M6_TX_CACHE_WAYS-1u)) != 0 || M6_TX_CACHE_WAYS > M6_TX_CACHE_SLOTS
#error M6_TX_CACHE_WAYS_must_be_a_power_of_two_not_greater_than_slots
#endif
#if M6_TX_CACHE_SLOTS == 0 || (M6_TX_CACHE_SLOTS & (M6_TX_CACHE_SLOTS - 1u)) != 0
#error M6_TX_CACHE_SLOTS_must_be_a_power_of_two
#endif
#include "rd_transform_memo.inc"
#endif

#if SVT_AV2_INTER_PIXEL_REUSE
#define RD_INTER_PIXEL_WAYS 3u
/* Shapes are the complete 64 root and four aligned 32 quadrants. Each
 * bucket is SB-local, so position and shape are implicit and never hashed. */
typedef struct RdInterPixelEntry {
    SvtAv2TileBlockMode mode;
    uint64_t sse[3];
    int32_t coeff[3][1024];
    uint8_t pixels[3][64u * 64u * 2u];
} RdInterPixelEntry;
typedef struct RdInterPixelMemo {
    uint32_t used[5], next[5];
    uint64_t hits, misses;
    RdInterPixelEntry entries[5][RD_INTER_PIXEL_WAYS];
} RdInterPixelMemo;
#endif

#if SVT_AV2_INTER_COARSE_PREFIX
/* Unique preceding binary digits (low to high in coding order), not a hash.
 * There are 15 rank contexts and only 14 reusable leaf outputs; depth-4
 * reconstructions are each consumed exactly once. Scope is one complete SB. */
typedef struct RdCoarseLeaf {
    SvtAv2TileBlockMode mode;
    SvtAv2TileCoeffBlock blocks[3];
    uint64_t sse[3];
    int32_t qcoeff[3u * 32u * 32u];
    uint8_t pixels[3u * 32u * 32u * 2u];
} RdCoarseLeaf;

typedef struct RdCoarsePrefix {
    SvtAv2BlockView geometry[4][3];
    RdCoarseLeaf leaves[14];
    uint8_t choices[15][2];
    uint16_t rank_valid, leaf_valid;
    uint64_t trees, rank_hits, leaf_hits;
    int trace;
#if SVT_AV2_INTER_COARSE_RATE_PREFIX
    SvtAv2TileInterRateCache* rate_cache; /* Frame-owned; used only for four-leaf coarse trials. */
    uint64_t rate_trees, rate_candidates;
    int rate_trace;
#endif
} RdCoarsePrefix;
#endif

/* Three plane bits per leaf index the original coefficient order. */
typedef struct RdInterTail {
    uint16_t scans[3][1024]; /* adjusted 8x8, 16x16, 32x32 (including coded 64x64) */
    uint16_t positions[INTER_FINE_MAX_LEAVES * 3u][SVT_AV2_INTER_TAIL_MAX];
    uint8_t counts[INTER_FINE_MAX_LEAVES * 3u];
    uint64_t mask, zero_mask;
    int try_sparse_planes;
    uint64_t trials, accepts, coeffs[3], leaves[3], zero_tbs, rate_saved_q12;
    int trace;
} RdInterTail;

#if SVT_AV2_INTER_PROFILE
typedef struct RdInterProfile {
    uint64_t coarse, integer_me, half, quarter, long_ref, fine, fine_screen, skip, tail, final;
    uint64_t forward_path, inverse_path, rate_trials;
    uint64_t split_rate, split_rate_setup;
} RdInterProfile;
#endif

typedef struct RdWork {
    uint8_t alternate_y[RD_MAX_LEAVES];
    uint8_t leaf_at[16][16]; /* 4x4 luma tile to causal leaf index. */
    uint8_t ist[RD_MAX_LEAVES];
    uint32_t ist_trials, ist_wins;
    int ist_all_leaves;
    uint32_t video_rect, video_rect_trials, video_rect_wins;
    int rank_intra_only;
    int detail_rd;
    int full_inter_fine;
    int wide_inter_me;
    int edge_inter;
    int late_inter_skip;
    uint32_t rect32_search, rect32_trials, rect32_wins, half_rect_wins, rect16_wins;
    int rect_rdoq;
    uint8_t* smooth_grid; /* Committed Y and UV smooth modes, one entry per 4x4 luma MI. */
    const SvtAv2FrameBuffer* source;
    SvtAv2FrameBuffer* recon;
    const SvtAv2FrameBuffer* reference;
    const SvtAv2FrameBuffer* secondary;
    const SvtAv2RdFrameConfig* config;
    SvtAv2ReconScratch* scratch;
    SvtAv2RdFrameOutput* output;
    SvtAv2RdFrameStats stats;
    SvtAv2FrameGeometry geometry;
    SvtAv2TileRateSession* rate;
    RdInterTail* tail; /* Frame-owned. All replay candidates borrow this state. */
#if SVT_AV2_EXPERIMENTAL_STILL_UV_RD
    unsigned still_uv_rd;
    uint64_t late_uv_trials, late_uv_wins, late_uv_implicit, late_uv_verified;
    uint64_t final_uv[4]; /* ordinary, explicit, implicit, MHCCP */
#endif
#if SVT_AV2_EXPERIMENTAL_CFL_PAIR_RD
    uint32_t cfl_pair_rd;
    uint64_t cfl_pair_leaves, cfl_pair_trials, cfl_pair_wins, cfl_pair_one_zero, cfl_pair_verified;
#endif
#if SVT_AV2_EXPERIMENTAL_INTRA_EOB
    uint32_t intra_eob_enabled, intra_eob_limit, intra_eob_edits;
    uint64_t intra_eob_trials, intra_eob_wins, intra_eob_levels, intra_eob_rate_saved;
#endif
#if SVT_AV2_EXPERIMENTAL_VIDEO_CFL
    uint64_t video_cfl_trials, video_cfl_wins, video_cfl_implicit, video_cfl_saved_q12;
#endif
#if SVT_AV2_INTER_COARSE_PREFIX
    RdCoarsePrefix* coarse_prefix; /* Frame-owned; validity cleared for each split tree. */
#endif
#if SVT_AV2_INTER_PROFILE
    RdInterProfile* profile; /* Same frame-owned counters in speculative RdWork copies. */
#endif
#if SVT_AV2_INTER_PIXEL_REUSE
    RdInterPixelMemo* inter_pixel; /* One frame owner; candidate snapshots borrow. */
#endif
    SvtAv2TileRatePrefix* basic_rate_prefix;
    SvtAv2TileRatePrefix* rdoq_rate_prefix; /* One frame owner; snapshots borrow. */
    SvtAv2TileCoeffBlock rdoq_rate_blocks[3];
    int32_t rdoq_rate_coeffs[3u*1024u];
    uint32_t rdoq_rate_active,rdoq_rate_ready,rdoq_rate_leaf,rdoq_use_context;
    uint32_t basic_rate_active, defer_rdoq, tree_rdoq, tool_rounds;
    struct RdBasicMemo* basic_memo;
#if M6_TX_CACHE
    RdTransformCache* transform_cache; /* Shared by finalist snapshots; never rewound. */
#endif
    uint32_t memo_prefix, memo_active;
    SvtAv2BlockView leaves[RD_MAX_LEAVES];
    SvtAv2MotionVector mvs[RD_MAX_LEAVES];
    SvtAv2MotionVector q3_seed[INTER_FINE_MAX_LEAVES]; /* Raw projected override, distinct from actual per-prefix MV. */
    uint32_t q3_seed_mask, long_mask; /* One bit per projected coarse/fine coding leaf. */
    int fine_trace, near_trace;
    uint64_t near_trials, near_wins[2], near_odd, near_empty;
    uint32_t fine_inter16[2], fine_odd16[2], fine_nonzero16[2], fine_mixed_sb;
    SvtAv2MotionSearchList mv_lists[4];
#if SVT_AV2_INTER_ME_REUSE
    struct {
        uint32_t count;
        SvtAv2BlockView leaves[4];
        SvtAv2MotionVector mvs[4];
    } integer_seed[2]; /* Immutable 64/32 LAST results for this complete SB only. */
#endif
    uint32_t wide_stage;
    uint32_t motion_policy; /* two bits per full-SB leaf select a candidate slot */
    uint32_t inter_skip_mask; /* winning or trial zero-residual leaves of a complete SB */
    uint32_t leaf_count;
    SvtAv2TileCoeffBlock blocks[RD_MAX_LEAVES * 3u];
    SvtAv2TileBlockMode modes[RD_MAX_LEAVES];
    int32_t coeffs[RD_MAX_COEFFS];
    size_t coeff_count;
    uint32_t coeff_policy; /* three bits per leaf: one tail-level decrement flag per plane */
    int32_t rdoq_coeff[RD_MAX_COEFFS];
    uint8_t rdoq_saved[RD_MAX_LEAVES * 3u];
    uint8_t rdoq_tx_types[RD_MAX_LEAVES * 3u];
    uint32_t rdoq_target; /* block to propose, UINT32_MAX for replay only */
    uint32_t joint_y_tx; /* zero on the retained original tool routes */
    uint32_t rdoq_phase; /* private bounded proposal strength: /16, /4, 1x */
    uint16_t prediction[3][64u * 64u];
    uint16_t backup[3][64u * 64u];
    uint16_t prefix_recon[3][64u * 64u];
    uint64_t prefix_sse[3];
    size_t prefix_coeff_count;
    uint32_t prefix_leaf, prefix_active;
    uint32_t uv_y_cache, uv_y_size; /* refine_basic UV: 0=off, 1=capture, 2=replay */
    uint16_t uv_y_recon[64u * 64u];
    int32_t uv_y_coeff[64u * 64u];
    uint32_t uv_y_offset[RD_MAX_LEAVES];
    uint64_t uv_y_sse[RD_MAX_LEAVES];
    SvtAv2TileCoeffBlock uv_y_blocks[RD_MAX_LEAVES];
    SvtAv2BlockView region[3];
    uint8_t ids[RD_MAX_LEAVES], best_ids[RD_MAX_LEAVES];
    uint8_t uv_ids[RD_MAX_LEAVES];
    uint8_t tx_types[RD_MAX_LEAVES];
    uint8_t cfl_explicit[RD_MAX_LEAVES], cfl_alpha_idx[RD_MAX_LEAVES], cfl_alpha_signs[RD_MAX_LEAVES];
    uint8_t mrl[RD_MAX_LEAVES];
    uint8_t mhccp[RD_MAX_LEAVES];
    uint32_t skip_mhccp_refinement;
} RdWork;

#if SVT_AV2_INTER_PROFILE
static int rd_profile_rate(RdWork* work, size_t sb, uint16_t decision,
    const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count, int commit, uint64_t* rate) {
    SVT_AV2_IP_START(start);
    const int status = svt_av2_tile_rate_trial(work->rate, sb, decision, blocks,
        block_count, modes, mode_count, commit, rate);
    SVT_AV2_IP_ADD(work->profile->rate_trials, start);
    return status;
}
#define RD_RATE(work, ...) rd_profile_rate((work), __VA_ARGS__)
#else
#define RD_RATE(work, ...) svt_av2_tile_rate_trial((work)->rate, __VA_ARGS__)
#endif

/* Node identity is an exact chosen-prefix extension; kind scores are separate. */
typedef struct RdBasicMemoNode {
    uint32_t parent,key[38],valid;
    uint64_t sse[2][3],rate[2];
} RdBasicMemoNode;
typedef struct RdBasicMemo {
    uint32_t count;
    uint16_t slots[16384];
    RdBasicMemoNode nodes[8192];
} RdBasicMemo;

static uint64_t hash_word(uint64_t hash, uint32_t value) {
    for (unsigned i = 0u; i < 4u; ++i) {
        hash = (hash ^ (value & 255u)) * RD_HASH_PRIME;
        value >>= 8u;
    }
    return hash;
}

static int collect_leaf(void* user, const SvtAv2BlockView* block) {
    RdWork* work = (RdWork*)user;
    if (work->leaf_count >= RD_MAX_LEAVES) return -1;
    work->leaves[work->leaf_count++] = *block;
    return 0;
}

static SvtAv2Error geometry_for_sb(RdWork* work, size_t sb, uint16_t decision) {
    work->output->decisions[sb] = decision;
#if SVT_AV2_INTER_ME_REUSE
    const int reuse = work->reference && work->config->frame_header.mv_half && decision < 2u &&
        work->region[0].w == 64u && work->region[0].h == 64u;
    if (reuse && work->integer_seed[decision].count) {
        work->leaf_count = work->integer_seed[decision].count;
        memcpy(work->leaves, work->integer_seed[decision].leaves, work->leaf_count * sizeof(*work->leaves));
        memcpy(work->mvs, work->integer_seed[decision].mvs, work->leaf_count * sizeof(*work->mvs));
        return SVT_AV2_ERROR_NONE;
    }
#endif
    work->leaf_count = 0u;
    SvtAv2Error error = svt_av2_partition_traverse_sb_plan(&work->geometry, sb,
        work->output->decisions, work->output->decision_count, collect_leaf, work);
    if (error != SVT_AV2_ERROR_NONE) return error;
    if (work->config->screen_trees) {
        for (uint32_t i = 0; i < work->leaf_count; ++i) {
            const SvtAv2BlockView* b = &work->leaves[i];
            for (uint32_t y = b->y; y < b->y + b->h; y += 4u)
                memset(&work->leaf_at[y % 64u / 4u][b->x % 64u / 4u], (int)i, b->w / 4u);
        }
    }
    if (work->reference != NULL) {
        SVT_AV2_IP_START(profile_me);
        for (uint32_t i = 0u; i < work->leaf_count; ++i) {
            if (work->config->motion_optimization && !work->config->frame_header.mv_half &&
                work->region[0].w == 64u && work->region[0].h == 64u) {
                if (i >= 4u) return SVT_AV2_ERROR_INVALID_STATE;
                error = svt_av2_motion_estimate_list(work->source, work->reference, &work->leaves[i], &work->mv_lists[i]);
                if (error != SVT_AV2_ERROR_NONE) return error;
                work->mvs[i] = work->mv_lists[i].mv[0];
                work->stats.motion_candidates += work->mv_lists[i].candidates;
                continue;
            }
            SvtAv2MotionSearchResult result;
            error = svt_av2_motion_estimate(work->source, work->reference, &work->leaves[i], &result);
            if (error != SVT_AV2_ERROR_NONE) return error;
            if (work->wide_inter_me && work->secondary)
                svt_av2_motion_extend(work->source, work->reference, &work->leaves[i], &result);
            work->mvs[i] = result.mv;
            /* Work and tile motion vectors are in the frame's signaled unit;
             * the integer-only ME entry points always return full pixels. */
            if (work->config->frame_header.mv_half) {
                const int32_t scale = work->config->frame_header.mv_half == 2u ? 4 : 2;
                work->mvs[i].mv_x *= scale;
                work->mvs[i].mv_y *= scale;
            }
            work->stats.motion_candidates += result.candidates;
        }
        SVT_AV2_IP_ADD(work->profile->integer_me, profile_me);
    }
#if SVT_AV2_INTER_ME_REUSE
    if (reuse && work->leaf_count <= 4u) {
        memcpy(work->integer_seed[decision].leaves, work->leaves, work->leaf_count * sizeof(*work->leaves));
        memcpy(work->integer_seed[decision].mvs, work->mvs, work->leaf_count * sizeof(*work->mvs));
        work->integer_seed[decision].count = work->leaf_count;
    }
#endif
    return SVT_AV2_ERROR_NONE;
}

static void region_copy_buffer(RdWork* work, uint16_t buffer[3][64u * 64u], int restore) {
    const uint32_t step = work->recon->bit_depth > 8u ? 2u : 1u;
    for (uint32_t plane = 0u; plane < 3u; ++plane) {
        const SvtAv2BlockView* block = &work->region[plane];
        for (uint32_t y = 0u; y < block->h; ++y) {
            uint8_t* row = work->recon->planes[plane] +
                (size_t)(block->y + y) * work->recon->strides[plane] + (size_t)block->x * step;
            uint8_t* saved = (uint8_t*)buffer[plane] + (size_t)y * block->w * step;
            if (restore) memcpy(row, saved, (size_t)block->w * step);
            else memcpy(saved, row, (size_t)block->w * step);
        }
    }
}

static void region_copy(RdWork* work, int restore) {
    region_copy_buffer(work, work->backup, restore);
}

static SvtAv2Error plane_blocks(RdWork* work, uint32_t leaf, SvtAv2BlockView blocks[3]) {
    blocks[0] = work->leaves[leaf];
    const SvtAv2Error error = svt_av2_chroma_view(&work->geometry, &blocks[0], &blocks[1]);
    blocks[2] = blocks[1];
    return error;
}

/* mode IDs are stable: DC=0,V=1,H=2,ME=3. They also define tie breaks. */
static void basic_memo_key(const RdWork* work,uint32_t leaf,uint32_t mode,
    const SvtAv2BlockView blocks[3],uint32_t key[38]) {
    for (uint32_t p=0u;p<3u;++p) {
        key[4u*p]=blocks[p].x;key[4u*p+1u]=blocks[p].y;
        key[4u*p+2u]=blocks[p].w;key[4u*p+3u]=blocks[p].h;
    }
    const uint32_t tuple[26]={mode,work->uv_ids[leaf],work->mrl[leaf],work->tx_types[leaf],
        work->cfl_explicit[leaf],work->cfl_alpha_idx[leaf],work->cfl_alpha_signs[leaf],work->mhccp[leaf],
        work->coeff_policy,work->rdoq_target,work->rdoq_saved[leaf*3u],work->rdoq_saved[leaf*3u+1u],work->rdoq_saved[leaf*3u+2u],
        work->rdoq_tx_types[leaf*3u],work->rdoq_tx_types[leaf*3u+1u],work->rdoq_tx_types[leaf*3u+2u],
        work->wide_stage,work->motion_policy,(uint32_t)work->mvs[leaf].mv_x,(uint32_t)work->mvs[leaf].mv_y,
        0u,mode,0u,work->uv_ids[leaf],0u,work->uv_ids[leaf]};
    memcpy(key+12u,tuple,sizeof(tuple));
}

static uint32_t basic_memo_node(RdBasicMemo* memo,uint32_t parent,const uint32_t key[38]) {
    uint32_t hash=parent;
    for (uint32_t i=0u;i<38u;++i) hash=(hash^key[i])*16777619u;
    for (uint32_t slot=hash&16383u;;slot=(slot+1u)&16383u) {
        const uint32_t id=memo->slots[slot];
        if (id) {
            const RdBasicMemoNode* node=&memo->nodes[id-1u];
            if (node->parent==parent && memcmp(node->key,key,sizeof(node->key))==0) return id;
        } else {
            if (memo->count==8192u) return UINT32_MAX;
            RdBasicMemoNode* node=&memo->nodes[memo->count];
            node->parent=parent;memcpy(node->key,key,sizeof(node->key));node->valid=0u;
            memo->slots[slot]=(uint16_t)++memo->count;return memo->count;
        }
    }
}

static int basic_memo_allowed(const RdWork* work,int partial) {
    if (!work->basic_memo || partial || work->reference || !work->basic_rate_active ||
        work->config->rd_mode!=3u || work->config->frame_header.m5_intra_modes<3u ||
        work->coeff_policy || work->wide_stage || work->motion_policy || work->rdoq_target!=UINT32_MAX) return 0;
    for (uint32_t i=0u;i<RD_MAX_LEAVES;++i)
        if (work->mrl[i] || work->tx_types[i] || work->cfl_explicit[i] || work->cfl_alpha_idx[i] ||
            work->cfl_alpha_signs[i] || work->mhccp[i] || work->rdoq_saved[i*3u] ||
            work->rdoq_saved[i*3u+1u] || work->rdoq_saved[i*3u+2u]) return 0;
    return 1;
}

static RdBasicMemoNode* basic_memo_trial(RdWork* work,uint32_t leaf,uint32_t mode,
    const SvtAv2BlockView blocks[3]) {
    if (!work->memo_active) return NULL;
    uint32_t key[38];basic_memo_key(work,leaf,mode,blocks,key);
    const uint32_t id=basic_memo_node(work->basic_memo,work->memo_prefix,key);
    return id==UINT32_MAX?NULL:&work->basic_memo->nodes[id-1u];
}

static SvtAv2Error basic_memo_cost(const RdWork* work,const RdBasicMemoNode* node,
    uint32_t kind,uint64_t* cost) {
    uint64_t sse[3];
    for (uint32_t p=0u;p<3u;++p) sse[p]=work->prefix_sse[p]+node->sse[kind][p];
    return svt_av2_rd_cost(svt_av2_rd_distortion(sse,work->source->bit_depth,work->source->chroma_format),
        work->stats.lambda_q8,svt_av2_tile_rate_prefix_current_rate(work->basic_rate_prefix)+node->rate[kind],cost);
}

static void basic_memo_save(const RdWork* work,RdBasicMemoNode* node,uint32_t kind,
    const uint64_t sse[3],uint64_t rate) {
    for (uint32_t p=0u;p<3u;++p) node->sse[kind][p]=sse[p]-work->prefix_sse[p];
    node->rate[kind]=rate-svt_av2_tile_rate_prefix_current_rate(work->basic_rate_prefix);
    node->valid|=1u<<kind;
}

static int legal_mode(const RdWork* work, uint32_t leaf, uint32_t mode) {
    if (mode == 0u) return 1;
    if (mode == 3u) {
        if (work->secondary != NULL) {
            const SvtAv2BlockView* b = &work->leaves[leaf];
            if (b->w != b->h || (b->w != 32u && b->w != 64u &&
                !(b->w == 16u && work->config->inter_fine) &&
                !(work->edge_inter && b->w == 8u && (work->region[0].w < 64u || work->region[0].h < 64u))) || b->x % b->w || b->y % b->h) return 0;
        }
        return work->reference != NULL;
    }
    const uint32_t w = work->leaves[leaf].w, h = work->leaves[leaf].h;
    if (work->config->rd_mode==3u && mode>=8u && w!=h &&
        !((w==8u && h==16u) || (w==16u && h==8u)) &&
        !(work->config->angle_search && w>=16u && h>=16u && w<=4u*h && h<=4u*w)) return 0;
    if (mode >= 8u && w != h && !((w == 8u && h == 16u) || (w == 16u && h == 8u)) &&
        !(work->config->angle_search && w >= 8u && h >= 8u &&
          ((w <= 4u * h && h <= 4u * w) ||
           (w == 8u && h == 64u) || (w == 64u && h == 8u)))) return 0;
    if (mode >= 14u) return mode <= 61u && work->config->angle_search && work->config->frame_header.m5_intra_modes >= 3u;
    if (mode >= 8u) return mode <= 13u && work->config->frame_header.m5_intra_modes >= 3u;
    if (mode >= 4u) return mode <= 7u && work->config->frame_header.m5_intra_modes >= 2u;
    if (!work->config->frame_header.m5_intra_modes) return 0;
    return mode == 1u ? work->leaves[leaf].y != 0u : work->leaves[leaf].x != 0u;
}

/* Availability of the extra smooth endpoint in the current raster SB tree.
 * Earlier SB rows and the already decoded left SB are committed; inside this
 * SB only preceding coding leaves may supply a sample. Coordinates are luma. */
static int reference_decoded(const RdWork* work, uint32_t leaf, int32_t x, int32_t y) {
    if (x < 0 || y < 0) return 0;
    const uint32_t px = (uint32_t)x, py = (uint32_t)y;
    const SvtAv2BlockView* sb = &work->region[0];
    if (py < sb->y) return 1;
    if (py >= sb->y + sb->h) return 0;
    if (px < sb->x) return 1;
    if (work->config->screen_trees) return px < sb->x + sb->w && work->leaf_at[py % 64u / 4u][px % 64u / 4u] < leaf;
    for (uint32_t i = 0u; i < leaf; ++i) {
        const SvtAv2BlockView* b = &work->leaves[i];
        if (px >= b->x && px < b->x + b->w && py >= b->y && py < b->y + b->h) return 1;
    }
    return 0;
}

static uint32_t rd_smooth_mode(uint32_t mode) { return mode >= 4u && mode <= 6u; }

static uint32_t rd_smooth_neighbor(const RdWork* work, uint32_t leaf,
    uint32_t plane, uint32_t x, uint32_t y) {
    const SvtAv2BlockView* sb = &work->region[0];
    if (x < sb->x || y < sb->y) {
        const size_t cols = work->geometry.visible_width / 4u;
        return (work->smooth_grid[(y / 4u) * cols + x / 4u] >> (plane != 0u)) & 1u;
    }
    if (work->config->screen_trees) {
        const uint32_t i = work->leaf_at[y % 64u / 4u][x % 64u / 4u];
        return i < leaf && rd_smooth_mode(plane ? work->uv_ids[i] : work->ids[i]);
    }
    for (uint32_t i = 0u; i < leaf; ++i) {
        const SvtAv2BlockView* b = work->leaves + i;
        if (x >= b->x && x < b->x + b->w && y >= b->y && y < b->y + b->h)
            return rd_smooth_mode(plane ? work->uv_ids[i] : work->ids[i]);
    }
    return 0u;
}

/* Only prior leaves in this candidate overlay the committed rate-session MI
 * grid. Their is_new/skip fields are immaterial to the DRL0 spatial stack. */
static int rd_motion_predict(const RdWork* work, uint32_t leaf, uint32_t rank,
    SvtAv2MotionVector* predictor) {
    if (leaf >= RD_MAX_LEAVES) return 0;
    SvtAv2MotionCell cells[RD_MAX_LEAVES];
    for (uint32_t i = 0; i < leaf; ++i) {
        const SvtAv2TileBlockMode* mode = &work->modes[i];
        cells[i] = (SvtAv2MotionCell){mode->mv_x, mode->mv_y, 1u,
            mode->is_inter ? (uint8_t)(1u + mode->reference_rank) : 0u, 0u, 0u};
    }
    const SvtAv2MotionPrefix prefix = {work->leaves, cells, leaf};
    return svt_av2_tile_rate_motion_predict(work->rate, &work->leaves[leaf], rank,
        &prefix, &predictor->mv_x, &predictor->mv_y) == 0;
}

/* Clip a search seed to the representable interval and preserve the exact
 * predictor parity (choose the lower candidate on a half-lattice tie). */
static int rd_snap_motion_component(int32_t desired, int32_t predictor, int32_t* target) {
    int32_t lo = SVT_AV2_MAX(-128, predictor - 256), hi = SVT_AV2_MIN(128, predictor + 256);
    lo += (lo - predictor) % 2 != 0;
    hi -= (hi - predictor) % 2 != 0;
    if (lo > hi) return 0;
    int32_t value = SVT_AV2_MIN(SVT_AV2_MAX(desired, lo), hi);
    value -= (value - predictor) % 2 != 0;
    *target = value;
    return 1;
}

static int rd_snap_motion(SvtAv2MotionVector desired, SvtAv2MotionVector predictor,
    SvtAv2MotionVector* target) {
    return rd_snap_motion_component(desired.mv_x, predictor.mv_x, &target->mv_x) &&
        rd_snap_motion_component(desired.mv_y, predictor.mv_y, &target->mv_y);
}

typedef enum ProjectedModeResult {
    PROJECTED_MODE_ERROR, PROJECTED_MODE_OK, PROJECTED_MODE_EMPTY
} ProjectedModeResult;

static ProjectedModeResult resolve_projected_mode(const RdWork* work, uint32_t leaf, uint32_t intra_inter,
    SvtAv2TileBlockMode* mode) {
    *mode = (SvtAv2TileBlockMode){0};
    if (intra_inter != 3u) return PROJECTED_MODE_OK;
    if (leaf >= RD_MAX_LEAVES || (leaf >= INTER_FINE_MAX_LEAVES && !work->edge_inter))
        return PROJECTED_MODE_ERROR;
    mode->is_inter = 1u;
    mode->reference_rank = leaf < INTER_FINE_MAX_LEAVES ? (work->long_mask >> leaf) & 1u : 0u;
    SvtAv2MotionVector predictor, target;
    if (!rd_motion_predict(work, leaf, mode->reference_rank, &predictor)) return PROJECTED_MODE_ERROR;
    const SvtAv2MotionVector want = leaf < INTER_FINE_MAX_LEAVES && ((work->long_mask | work->q3_seed_mask) & (1u << leaf)) ? work->q3_seed[leaf] :
        (SvtAv2MotionVector){2 * work->mvs[leaf].mv_x, 2 * work->mvs[leaf].mv_y};
    if (!rd_snap_motion(want, predictor, &target)) return PROJECTED_MODE_EMPTY;
    mode->mv_x = target.mv_x;
    mode->mv_y = target.mv_y;
    return PROJECTED_MODE_OK;
}

static int rd_projected_mode(const RdWork* work, uint32_t leaf, uint32_t intra_inter,
    SvtAv2TileBlockMode* mode) {
    return resolve_projected_mode(work, leaf, intra_inter, mode) == PROJECTED_MODE_OK;
}

/* Optional rank/geometry changes can expose a LAST->LONG spatial projection
 * beyond the target/MVD lattice at distance 4+. Reject only that proposal,
 * without reconstructing it or changing the committed MI grid. At distance
 * <=3 the projection is at most 384 Q3, whose lattice always has a target. */
static ProjectedModeResult preflight_projected_path(RdWork* work, const uint8_t* ids,
    size_t sb, uint16_t decision) {
    if (!work->long_mask || work->config->frame_header.ref_distance[1] <= 3u) return PROJECTED_MODE_OK;
    for (uint32_t leaf = 0u; leaf < work->leaf_count; ++leaf) {
        SvtAv2TileBlockMode resolved;
        const ProjectedModeResult state = resolve_projected_mode(work, leaf, ids[leaf], &resolved);
        if (state != PROJECTED_MODE_OK) {
            if (state == PROJECTED_MODE_EMPTY && work->fine_trace)
                fprintf(stderr, "projected_proposal_reject hint=%u sb=%zu plan=%u leaf=%u distance=%u\n",
                    work->config->frame_header.order_hint, sb, (unsigned)decision, leaf,
                    work->config->frame_header.ref_distance[1]);
            return state;
        }
        work->modes[leaf] = resolved;
    }
    return PROJECTED_MODE_OK;
}

static int rd_inter_mode(const RdWork* work, uint32_t leaf, uint32_t mode,
    SvtAv2TileBlockMode* resolved) {
    if (work->secondary != NULL) return rd_projected_mode(work, leaf, mode, resolved);
    *resolved = mode == 3u ? (SvtAv2TileBlockMode){1u, work->mvs[leaf].mv_x, work->mvs[leaf].mv_y, 0u} :
        (SvtAv2TileBlockMode){0};
    return 1;
}

static SvtAv2Error predict_with_motion(RdWork* work, uint32_t leaf, uint32_t mode,
    const SvtAv2BlockView blocks[3], uint64_t sad[3], int plane_selection,
    const SvtAv2TileBlockMode* resolved) {
    uint8_t bytes[64u * 64u * 2u];
    const uint16_t one = 1u;
    const int direct = work->config->rd_mode == 3u && work->source->bit_depth == 10u &&
        ((const unsigned char*)&one)[0] == 1u;
    const uint32_t step = work->source->bit_depth > 8u ? 2u : 1u;
    const uint32_t first=plane_selection==2 || (work->uv_y_cache==2u && plane_selection==0)?1u:0u;
    SvtAv2TileBlockMode local;
    const SvtAv2FrameBuffer* projected_ref = NULL;
    if (mode == 3u) {
        if (resolved == NULL) {
            if (!rd_inter_mode(work, leaf, mode, &local)) return SVT_AV2_ERROR_INVALID_STATE;
            resolved = &local;
        }
        if (work->secondary != NULL) projected_ref = resolved->reference_rank ? work->secondary : work->reference;
    }
    for (uint32_t plane=first;plane<(plane_selection==1?1u:3u);++plane) {
        const SvtAv2BlockView* block = &blocks[plane];
        SvtAv2Error error;
        if (projected_ref != NULL) {
            const SvtAv2EighthPelMotionVector mv = {resolved->mv_x, resolved->mv_y};
            error = svt_av2_inter_predict_eighth_sharp(projected_ref, plane, block, mv,
                work->prediction[plane], block->w);
        } else if (mode == 3u) {
            const SvtAv2MotionVector mv = {resolved->mv_x, resolved->mv_y};
            if (work->config->frame_header.mv_half == 2u) {
                const SvtAv2QuarterPelMotionVector quarter = {mv.mv_x, mv.mv_y};
                error = svt_av2_inter_predict_quarter_sharp(work->reference, plane, block, quarter,
                    work->prediction[plane], block->w);
            } else if (work->config->frame_header.mv_half) {
                const SvtAv2HalfPelMotionVector half = {mv.mv_x, mv.mv_y};
                error = work->config->frame_header.sharp_filter ?
                    svt_av2_inter_predict_half_sharp(work->reference, plane, block, half,
                        work->prediction[plane], block->w) :
                    svt_av2_inter_predict_half(work->reference, plane, block, half,
                        work->prediction[plane], block->w);
            } else error = svt_av2_inter_predict(work->reference, plane, block, mv,
                work->prediction[plane], block->w);
        } else {
            uint8_t* const predicted = direct ? (uint8_t*)work->prediction[plane] : bytes;
            const SvtAv2IntraBlock intra = {block->x, block->y, block->w, block->h, plane};
            SvtAv2IntraReference reference;
            if (work->config->frame_header.m5_intra_modes < 2u) {
                error = svt_av2_intra_reference_build(work->recon, &intra, &reference);
                if (error != SVT_AV2_ERROR_NONE) return error;
            }
            if (work->config->frame_header.m5_intra_modes >= 2u) {
                const int32_t scale = plane && work->source->chroma_format == SVT_AV2_CHROMA_420 ? 2 : 1;
                const int tr = reference_decoded(work, leaf, (int32_t)(block->x + block->w) * scale,
                    ((int32_t)block->y - 1) * scale);
                const int bl = reference_decoded(work, leaf, ((int32_t)block->x - 1) * scale,
                    (int32_t)(block->y + block->h) * scale);
                const int inherit = plane != 0u && mode >= 14u &&
                    svt_av2_intra_mode_base(mode) == work->uv_ids[leaf];
                if ((plane == 0u && (mode >= 8u || work->mrl[leaf])) || inherit) {
                    uint32_t tr_count = 0u, bl_count = 0u;
                    const uint32_t tr_limit = plane == 0u && work->mrl[leaf] ? block->w : block->h;
                    const uint32_t bl_limit = plane == 0u && work->mrl[leaf] ? block->h : block->w;
                    while (tr_count < tr_limit && reference_decoded(work, leaf,
                        (int32_t)(block->x + block->w + tr_count) * scale, ((int32_t)block->y - 1) * scale)) ++tr_count;
                    while (bl_count < bl_limit && reference_decoded(work, leaf,
                        ((int32_t)block->x - 1) * scale, (int32_t)(block->y + block->h + bl_count) * scale)) ++bl_count;
                    const int ibp = work->config->frame_header.enable_ibp && (block->w != 4u || block->h != 4u);
                    uint32_t top_filter = 0u, left_filter = 0u;
                    if (work->smooth_grid) {
                        const uint32_t x = blocks[0].x, y = blocks[0].y;
                        const int left = x && rd_smooth_neighbor(work, leaf, plane, x - 1u, y);
                        const int top = y && rd_smooth_neighbor(work, leaf, plane, x, y - 1u);
                        top_filter = 1u + (ibp ? top : top || left);
                        left_filter = 1u + (ibp ? left : top || left);
                    }
                    error = plane == 0u && work->mrl[leaf] ?
                        svt_av2_intra_mrls_predict(work->recon,&intra,mode,work->mrl[leaf],
                            tr_count,bl_count,predicted,block->w*step) :
                        svt_av2_intra_directional_predict_tools(work->recon, &intra,
                            mode, tr_count, bl_count, top_filter, left_filter, ibp,
                            predicted, block->w * step);
                } else error = svt_av2_intra_basic_predict(work->recon, &intra,
                    plane == 0u ? mode : work->uv_ids[leaf] == 14u ? 0u : work->uv_ids[leaf], tr, bl, predicted, block->w * step);
            } else if (plane == 0u && mode != 0u)
                error = svt_av2_intra_axis_predict(&reference, &intra,
                    mode == 1u ? SVT_AV2_INTRA_AXIS_VERTICAL : SVT_AV2_INTRA_AXIS_HORIZONTAL,
                    predicted, block->w * step);
            else error = svt_av2_intra_dc_predict(&reference, &intra, predicted, block->w * step);
            if (error == SVT_AV2_ERROR_NONE && work->config->frame_header.enable_ibp &&
                (plane == 0u ? mode == 0u : work->uv_ids[leaf] == 0u))
                svt_av2_intra_ibp_dc(work->recon, &intra, predicted, block->w * step);
            if (error == SVT_AV2_ERROR_NONE && !direct)
                for (uint32_t i = 0u; i < block->w * block->h; ++i)
                    work->prediction[plane][i] = svt_av2_sample_load(bytes + i * step, work->source->bit_depth);
        }
        if (error != SVT_AV2_ERROR_NONE) return error;
        if (sad == NULL) continue;
        sad[plane] = 0u;
        for (uint32_t y = 0u; y < block->h; ++y) {
            const uint8_t* row = work->source->planes[plane] +
                (size_t)(block->y + y) * work->source->strides[plane] + (size_t)block->x * step;
            for (uint32_t x = 0u; x < block->w; ++x) {
                const int difference = (int)svt_av2_sample_load(row + x * step, work->source->bit_depth) -
                    work->prediction[plane][(size_t)y * block->w + x];
                sad[plane] += (uint64_t)(difference < 0 ? -difference : difference);
            }
        }
    }
    return SVT_AV2_ERROR_NONE;
}

static SvtAv2Error predict(RdWork* work, uint32_t leaf, uint32_t mode,
    const SvtAv2BlockView blocks[3], uint64_t sad[3], int plane_selection) {
    return predict_with_motion(work, leaf, mode, blocks, sad, plane_selection, NULL);
}

#if M6_SATD_ANGLES
/* Hadamard cost only screens additional modes; exact R+D still decides. */
static uint64_t angle_satd(const RdWork* work,const SvtAv2BlockView* b) {
    uint64_t cost=0u;
    const uint32_t step=work->source->bit_depth>8u?2u:1u;
    for (uint32_t y=0u;y<b->h;y+=4u) for (uint32_t x=0u;x<b->w;x+=4u) {
        int32_t tmp[4][4];
        for (uint32_t j=0u;j<4u;++j) {
            int32_t d[4];
            const uint8_t* row=work->source->planes[0]+(size_t)(b->y+y+j)*work->source->strides[0]+(size_t)(b->x+x)*step;
            for (uint32_t i=0u;i<4u;++i) d[i]=(int32_t)svt_av2_sample_load(row+i*step,work->source->bit_depth)-work->prediction[0][(size_t)(y+j)*b->w+x+i];
            const int32_t a=d[0]+d[3],b0=d[1]+d[2],c=d[1]-d[2],e=d[0]-d[3];
            tmp[j][0]=a+b0;tmp[j][1]=e+c;tmp[j][2]=a-b0;tmp[j][3]=e-c;
        }
        for (uint32_t i=0u;i<4u;++i) {
            const int32_t a=tmp[0][i]+tmp[3][i],b0=tmp[1][i]+tmp[2][i],c=tmp[1][i]-tmp[2][i],e=tmp[0][i]-tmp[3][i];
            const int32_t d[4]={a+b0,e+c,a-b0,e-c};
            for (uint32_t j=0u;j<4u;++j) cost+=(uint64_t)(d[j]<0?-d[j]:d[j]);
        }
    }
    return (cost+1u)>>1u;
}
static void angle_satd_rank(const RdWork* work,const SvtAv2BlockView* b,uint32_t mode,uint8_t ranked[4],uint64_t scores[4]) {
    if (!work->config->satd_angles) return;
    const uint64_t score=angle_satd(work,b);
    for (uint32_t slot=0u;slot<4u;++slot) if (score<scores[slot] || (score==scores[slot] && mode<ranked[slot])) {
        for (uint32_t j=3u;j>slot;--j) {scores[j]=scores[j-1u];ranked[j]=ranked[j-1u];}
        scores[slot]=score;ranked[slot]=(uint8_t)mode;break;
    }
}
static void angle_satd_append(const RdWork* work,uint8_t candidates[14],uint32_t* count,const uint8_t ranked[4]) {
    if (!work->config->satd_angles) return;
    for (uint32_t slot=0u;slot<4u;++slot) {
        uint32_t i=0u;while(i<*count && candidates[i]!=ranked[slot]) ++i;
        if (i==*count) candidates[(*count)++]=ranked[slot];
    }
}
#endif

static SvtAv2Error rank_modes(RdWork* work, uint32_t leaf,
    const SvtAv2BlockView blocks[3], uint8_t ranked[4], uint32_t* count, int partial) {
    uint64_t costs[4] = {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
    uint64_t y_costs[4] = {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
    *count = 0u;
    for (uint32_t mode = 0u; mode < 4u; ++mode) {
        if ((work->rank_intra_only && mode == 3u) || !legal_mode(work, leaf, mode)) continue;
        uint64_t sad[3];
        const SvtAv2Error error = predict(work, leaf, mode, blocks, sad, 0);
        if (error != SVT_AV2_ERROR_NONE) return error;
        costs[mode] = sad[0] + sad[1] + sad[2];
        y_costs[mode] = sad[0];
        ranked[(*count)++] = (uint8_t)mode;
    }
    if (partial) {
        /* Reproduce M5: first choose intra on Y, then intra/inter on all planes. */
        uint32_t intra = 0u;
        for (uint32_t mode = 1u; mode < 3u; ++mode)
            if (y_costs[mode] < y_costs[intra]) intra = mode;
        ranked[0] = (uint8_t)(costs[3] < costs[intra] ? 3u : intra);
        *count = 1u;
        return SVT_AV2_ERROR_NONE;
    }
    for (uint32_t i = 1u; i < *count; ++i) {
        const uint8_t mode = ranked[i];
        uint32_t j = i;
        while (j != 0u && (costs[mode] < costs[ranked[j - 1u]] ||
            (costs[mode] == costs[ranked[j - 1u]] && mode < ranked[j - 1u]))) {
            ranked[j] = ranked[j - 1u]; --j;
        }
        ranked[j] = mode;
    }
    return SVT_AV2_ERROR_NONE;
}

/* Proposal cost uses the real coefficient symbol writer with the default CDF
 * and zero neighbours. It deliberately does not mutate the tile CDF session.
 * The outer block-coordinate pass accepts only a complete SB improvement with
 * actual neighbours, adaptive CDFs and inverse/clip pixel distortion. */
static SvtAv2Error rdoq_rate(const RdWork* work, uint32_t plane,
    const SvtAv2BlockView* block, const int32_t* qcoeff, uint32_t tx_type, uint32_t mode, uint64_t* rate) {
#if M6_RDOQ_CONTEXT
    if (work->rdoq_rate_active) {
        if (plane>=3u || work->rdoq_target/3u!=work->rdoq_rate_leaf ||
            work->rdoq_target%3u!=plane) return SVT_AV2_ERROR_INVALID_STATE;
        SvtAv2TileCoeffBlock trial[3];
        memcpy(trial,work->rdoq_rate_blocks,sizeof(trial));
        /* Rebind borrowed views after every RdWork copy; snapshots contain
         * no self-referential anchor pointers. */
        for (uint32_t p=0u;p<3u;++p) trial[p].qcoeff=work->rdoq_rate_coeffs+p*1024u;
#if M6_TX_CACHE
        if (work->transform_cache) ++work->transform_cache->rdoq_live_rate_calls;
#endif
        if (trial[plane].width!=block->w || trial[plane].height!=block->h)
            return SVT_AV2_ERROR_INVALID_STATE;
        trial[plane].qcoeff=qcoeff;trial[plane].tx_type=(uint8_t)tx_type;
        uint64_t total=0u;
        const uint64_t prefix=svt_av2_tile_rate_prefix_current_rate(work->rdoq_rate_prefix);
        if (svt_av2_tile_rate_prefix_trial(work->rdoq_rate_prefix,work->rdoq_rate_leaf,
            trial,0,&total)!=0 || total<prefix) return SVT_AV2_ERROR_INVALID_STATE;
        *rate=total-prefix;return SVT_AV2_ERROR_NONE;
    }
#endif
    SvtAv2TxSize tx;
    if (svt_av2_coeff_tx_size_from_dims(block->w, block->h, &tx) != SVT_AV2_COEFF_OK)
        return SVT_AV2_ERROR_INVALID_STATE;
    const uint32_t scale = plane && work->source->chroma_format == SVT_AV2_CHROMA_420 ? 2u : 1u;
    const SvtAv2CoeffPlaneInfo pi = {plane, tx,
        block->x / 4u, block->y / 4u,
        work->source->visible_width / (4u * scale),
        work->source->visible_height / (4u * scale), block->w, block->h};
    SvtAv2CoeffNeighbours neighbours;
    svt_av2_coeff_neighbours_reset(&neighbours);
    SvtAv2EntropyEncoder enc;
    svt_av2_entropy_rate_init(&enc);
    if (svt_av2_coeff_write_block_transform_ist(&enc, &pi, 0u, qcoeff, &neighbours,
        svt_av2_coeff_q_context(work->config->frame_header.base_q_idx), 0u,
        work->config->frame_header.reduced_tx_set, tx_type, mode,
        work->config->frame_header.enable_intra_ist, 0u) != 0)
        return SVT_AV2_ERROR_INVALID_STATE;
    *rate = enc.rate_q12;
    return SVT_AV2_ERROR_NONE;
}

static uint64_t rdoq_distortion(const RdWork* work, uint32_t plane,
    const SvtAv2QuantParams* params, uint64_t error) {
    /* DCT coefficient gain is 8,4,2 for tx_scale 0,1,2. For 64x64
     * the packed 32x32 inverse is upsampled, giving the same energy scale. */
    const unsigned shift = (unsigned)(6 - 2 * params->tx_scale) +
        2u * (work->source->bit_depth - 8u);
    const uint64_t weight = plane == 0u || work->source->chroma_format == SVT_AV2_CHROMA_420 ? 4u : 1u;
    return (weight * error + (UINT64_C(1) << (shift - 1u))) >> shift;
}

static uint64_t rdoq_error(int32_t coeff, int32_t dequant) {
    const int64_t delta = (int64_t)coeff - dequant;
    return (uint64_t)(delta * delta);
}

static int32_t rdoq_dequant(const SvtAv2QuantParams* params, int32_t level,
    uint32_t plane, uint32_t position) {
    const int32_t value = svt_av2_dequant_position(params, level, (int)plane, position);
    const int32_t minimum = svt_av2_dequant_clip_min(params->bit_depth);
    const int32_t maximum = svt_av2_dequant_clip_max(params->bit_depth);
    return value < minimum ? minimum : value > maximum ? maximum : value;
}

static SvtAv2Error rdoq_propose(RdWork* work, uint32_t plane,
    const SvtAv2BlockView* block, size_t count, uint32_t tx_type, uint32_t mode) {
    SvtAv2QuantParams params;
    if (svt_av2_quant_params_init_matrix(&params, work->config->frame_header.base_q_idx,
        block->w, block->h, work->source->bit_depth, work->config->frame_header.qm_level, work->config->frame_header.uv_delta_q) != SVT_AV2_QUANT_OK)
        return SVT_AV2_ERROR_INVALID_STATE;
    /* Restrict the scalar proposal to nearby conservative local changes;
     * the full-SB acceptance still uses the unchanged global lambda. */
    const uint64_t proposal_lambda = work->rdoq_phase == 0u ?
        (work->stats.lambda_q8 + 8u) >> 4u : work->rdoq_phase == 1u ?
        (work->stats.lambda_q8 + 2u) >> 2u : work->stats.lambda_q8;
    int32_t* q = work->scratch->qcoeff;
    int32_t* dq = work->scratch->dqcoeff;
    const int32_t* coeff = work->scratch->coeff;
    uint16_t scan[1024];
    SvtAv2TxSize tx;
    if (svt_av2_coeff_tx_size_from_dims(block->w, block->h, &tx) != SVT_AV2_COEFF_OK)
        return SVT_AV2_ERROR_INVALID_STATE;
    if (svt_av2_coeff_scan_dct_dct(tx, scan) != count)
        return SVT_AV2_ERROR_INVALID_STATE;
    uint64_t error = 0u, rate, best_cost;
    for (size_t c = 0u; c < count; ++c) error += rdoq_error(coeff[c], dq[c]);
    SvtAv2Error status = rdoq_rate(work, plane, block, q, tx_type, mode, &rate);
    if (status != SVT_AV2_ERROR_NONE ||
        svt_av2_rd_cost(rdoq_distortion(work, plane, &params, error), proposal_lambda,
            rate, &best_cost) != SVT_AV2_ERROR_NONE) return SVT_AV2_ERROR_INVALID_STATE;
    for (size_t index = count; index != 0u; --index) {
        const uint32_t c = scan[index - 1u];
        const int32_t entering = q[c];
        const int32_t sign = coeff[c] < 0 ? -1 : 1;
        const int32_t magnitude = entering < 0 ? -entering : entering;
        int32_t chosen = entering, chosen_dq = dq[c];
        uint64_t chosen_error = error;
        const uint64_t old_error = rdoq_error(coeff[c], dq[c]);
        const int32_t levels[2] = {magnitude == 0 ? sign : (magnitude - 1) * sign,
            (magnitude + 1) * sign};
        for (uint32_t i = 0u; i < (magnitude == 0 ? 1u : 2u); ++i) {
            const int32_t candidate = levels[i];
            if (candidate > svt_av2_coeff_max_level() || candidate < -svt_av2_coeff_max_level()) continue;
            const int32_t candidate_dq = rdoq_dequant(&params, candidate, plane, c);
            const uint64_t next_error = error - old_error + rdoq_error(coeff[c], candidate_dq);
            /* A new nonzero whose squared error cannot improve is not useful
             * as a scalar proposal; context changes are left to exact SB RD. */
            if (magnitude == 0 && next_error >= error) continue;
            q[c] = candidate;
            /* EOB<=1 infers DCT. Keep ADST coefficients in its signalled
             * domain; the separate DCT candidate retains these EOB choices. */
            if (tx_type != 0u && svt_av2_coeff_find_eob(q, scan, count) <= 1u) continue;
            /* Every syntax cost is nonnegative. A strict improvement is
             * impossible when distortion alone reaches the incumbent J. */
            const uint64_t distortion = rdoq_distortion(work, plane, &params, next_error);
            if (distortion >= best_cost) continue;
            uint64_t cost;
            status = rdoq_rate(work, plane, block, q, tx_type, mode, &rate);
            if (status != SVT_AV2_ERROR_NONE ||
                svt_av2_rd_cost(distortion, proposal_lambda, rate, &cost) != SVT_AV2_ERROR_NONE)
                return SVT_AV2_ERROR_INVALID_STATE;
            if (cost < best_cost) {
                best_cost = cost; chosen = candidate; chosen_dq = candidate_dq; chosen_error = next_error;
            }
        }
        q[c] = chosen; dq[c] = chosen_dq; error = chosen_error;
    }
    /* Test every nonzero EOB cut, including DC-only and all-zero, rather than
     * assuming that a single last-level decrement can find a useful suffix. */
    int32_t best_q[1024];
    memcpy(best_q, q, count * sizeof(*q));
    const uint32_t eob = svt_av2_coeff_find_eob(q, scan, count);
    for (uint32_t index = eob; index != 0u; --index) {
        const uint32_t c = scan[index - 1u];
        if (q[c] == 0) continue;
        error -= rdoq_error(coeff[c], dq[c]);
        error += rdoq_error(coeff[c], 0); q[c] = 0;
        if (tx_type != 0u && svt_av2_coeff_find_eob(q, scan, count) <= 1u) break;
        /* Keep cutting even when this cut cannot win: its successor may
         * lower the error. Only the provably losing rate evaluation is skipped. */
        const uint64_t distortion = rdoq_distortion(work, plane, &params, error);
        if (distortion >= best_cost) continue;
        uint64_t cost;
        status = rdoq_rate(work, plane, block, q, tx_type, mode, &rate);
        if (status != SVT_AV2_ERROR_NONE ||
            svt_av2_rd_cost(distortion, proposal_lambda, rate, &cost) != SVT_AV2_ERROR_NONE)
            return SVT_AV2_ERROR_INVALID_STATE;
        if (cost < best_cost) { best_cost = cost; memcpy(best_q, q, count * sizeof(*q)); }
    }
    memcpy(q, best_q, count * sizeof(*q));
    return svt_av2_dequantize_block(&params, q, dq, count, (int)plane) == SVT_AV2_QUANT_OK ?
        SVT_AV2_ERROR_NONE : SVT_AV2_ERROR_INVALID_STATE;
}

/* Production transform_path reads only valid residual samples, geometry,
 * plane, and immutable per-frame QP/bitdepth/quantizer state. Cache the initial
 * forward result before any RDOQ mutation. Hashing only selects a slot: all
 * scalar keys and every valid residual sample must match before reuse. */
static SvtAv2Error forward_trial(RdWork* work, uint32_t plane,
    const SvtAv2BlockView* block, uint32_t tx_type, int* all_zero) {
#if M6_TX_CACHE
    RdTransformCache* const cache = work->transform_cache;
    const uint32_t key[6] = {plane, block->w, block->h, tx_type,
        work->config->frame_header.base_q_idx, (uint32_t)work->source->bit_depth};
    uint64_t hash = RD_HASH_OFFSET;
    if (cache != NULL) {
        uint32_t area = block->w * block->h, group = 0u;
        while (area > 16u && group < 8u) { area >>= 1u; ++group; }
        ++cache->requests_by_area[group];
        for (uint32_t i = 0u; i < 6u; ++i) hash = hash_word(hash, key[i]);
        for (uint32_t y = 0u; y < block->h; ++y)
            for (uint32_t x = 0u; x < block->w; ++x)
                hash = (hash ^ (uint16_t)work->scratch->residual[
                    (size_t)y * work->scratch->stride + x]) * RD_HASH_PRIME;
        const uint32_t id = rd_tx_memo_find(cache, hash, key,
            work->scratch->residual, work->scratch->stride);
        if (id) {
            if (!rd_tx_memo_restore(cache, id, work->scratch->coeff,
                    work->scratch->qcoeff, work->scratch->dqcoeff, all_zero))
                return SVT_AV2_ERROR_INVALID_STATE;
            return SVT_AV2_ERROR_NONE;
        }
        ++cache->misses;
        ++cache->misses_by_area[group];
    }
#endif
    /* The only work counter increment is an actual callback dispatch. */
    ++work->stats.forward_calls;
    SVT_AV2_IP_START(profile_forward);
    const SvtAv2Error error = tx_type ?
        work->config->transform.forward_type(work->config->transform.user,
            plane, block->w, block->h, work->scratch->residual,
            work->scratch->stride, work->scratch->coeff, work->scratch->qcoeff,
            work->scratch->dqcoeff, all_zero, tx_type) :
        work->config->transform.forward(work->config->transform.user,
            plane, block->w, block->h, work->scratch->residual,
            work->scratch->stride, work->scratch->coeff, work->scratch->qcoeff,
            work->scratch->dqcoeff, all_zero);
    SVT_AV2_IP_ADD(work->profile->forward_path, profile_forward);
#if M6_TX_CACHE
    if (cache != NULL && error == SVT_AV2_ERROR_NONE)
        (void)rd_tx_memo_store(cache, hash, key, work->scratch->residual,
            work->scratch->stride, work->scratch->coeff,
            work->scratch->qcoeff, *all_zero);
#endif
    return error;
}

/* Magnitudes of the three orthogonal AC components in a 2x2 Haar cell.
 * Comparing their total amplitude adds a texture term to the luma RD search.
 * The 1/12 normalization bounds the penalty by the four samples' SSE. */
static int haar_ac(const uint8_t* a, size_t stride, uint32_t step, uint32_t bd) {
    const int tl = svt_av2_sample_load(a, bd), tr = svt_av2_sample_load(a + step, bd);
    const int bl = svt_av2_sample_load(a + stride, bd), br = svt_av2_sample_load(a + stride + step, bd);
    return abs(tl + tr - bl - br) + abs(tl - tr + bl - br) + abs(tl - tr - bl + br);
}

static uint64_t detail_error(const RdWork* work, const SvtAv2BlockView* block) {
    const uint32_t bd = work->source->bit_depth, step = bd > 8u ? 2u : 1u;
    const size_t ss = work->source->strides[0], rs = work->recon->strides[0];
    uint64_t error = 0u;
    for (uint32_t y = 0u; y + 1u < block->h; y += 2u) {
        const uint8_t* src = work->source->planes[0] + (size_t)(block->y + y) * ss + block->x * step;
        const uint8_t* rec = work->recon->planes[0] + (size_t)(block->y + y) * rs + block->x * step;
        for (uint32_t x = 0u; x + 1u < block->w; x += 2u) {
            const int d = haar_ac(src + x * step, ss, step, bd) - haar_ac(rec + x * step, rs, step, bd);
            error += (uint64_t)(d * d);
        }
    }
    return (error + 6u) / 12u;
}

/* The inter predictor is the final reconstruction when all three transforms
 * are skipped. Emit the normal coefficient descriptors with packed zero
 * grids, so writer/estimator derive skip_txfm themselves. */
static SvtAv2Error reconstruct_inter_zero(RdWork* work, uint32_t leaf,
    const SvtAv2BlockView blocks[3], uint64_t sse[3]) {
    const uint32_t bd = work->source->bit_depth, step = bd > 8u ? 2u : 1u;
    for (uint32_t plane = 0u; plane < 3u; ++plane) {
        const SvtAv2BlockView* block = &blocks[plane];
        const size_t count = (size_t)SVT_AV2_MIN(block->w, 32u) * SVT_AV2_MIN(block->h, 32u);
        if (count > RD_MAX_COEFFS - work->coeff_count) return SVT_AV2_ERROR_INVALID_STATE;
        SvtAv2TileCoeffBlock entry = {0};
        entry.plane = plane; entry.x = block->x; entry.y = block->y;
        entry.width = block->w; entry.height = block->h;
        entry.qcoeff = work->coeffs + work->coeff_count;
        memset(work->coeffs + work->coeff_count, 0, count * sizeof(*work->coeffs));
        work->coeff_count += count;
        work->blocks[leaf * 3u + plane] = entry;
        uint64_t sum = 0u;
        for (uint32_t y = 0u; y < block->h; ++y) {
            const size_t x_bytes = (size_t)block->x * step;
            const uint8_t* src = work->source->planes[plane] +
                (size_t)(block->y + y) * work->source->strides[plane] + x_bytes;
            uint8_t* dst = work->recon->planes[plane] +
                (size_t)(block->y + y) * work->recon->strides[plane] + x_bytes;
            const uint16_t* pred = work->prediction[plane] + (size_t)y * block->w;
            for (uint32_t x = 0u; x < block->w; ++x) {
                const int diff = (int)svt_av2_sample_load(src + x * step, bd) - pred[x];
                sum += (uint64_t)(diff * diff);
                svt_av2_sample_store(dst + x * step, bd, pred[x]);
            }
        }
        sse[plane] += sum;
        if (work->detail_rd && !plane) sse[0] += detail_error(work, block);
    }
    return SVT_AV2_ERROR_NONE;
}

static uint32_t rd_ist_mode(uint32_t mode, const SvtAv2BlockView* block, uint32_t mrl) {
    return svt_av2_ist_mode(svt_av2_intra_tx_mode(mode, block->w, block->h, mrl));
}

/* The primary-forward cache is shared with the identity arm. Keep the
 * replacement separate until its quantized EOB can actually signal IST. */
static SvtAv2Error quantize_ist(RdWork* work, const SvtAv2BlockView* b, uint32_t primary,
    uint32_t mode, uint32_t code, uint8_t* coded) {
    SvtAv2TxSize size;
    uint16_t scan[1024];
    int32_t coeff[1024], qcoeff[1024], dqcoeff[1024];
    SvtAv2QuantParams params;
    const uint32_t depth = work->source->bit_depth;
    const size_t count = (size_t)SVT_AV2_MIN(b->w, 32u) * SVT_AV2_MIN(b->h, 32u);
    if (svt_av2_coeff_tx_size_from_dims(b->w, b->h, &size) != 0 ||
        svt_av2_coeff_scan_dct_dct(size, scan) != count ||
        svt_av2_quant_params_init_matrix(&params, work->config->frame_header.base_q_idx,
            b->w, b->h, depth, work->config->frame_header.qm_level, work->config->frame_header.uv_delta_q) != SVT_AV2_QUANT_OK)
        return SVT_AV2_ERROR_INVALID_STATE;
    memcpy(coeff, work->scratch->coeff, count * sizeof(*coeff));
    svt_av2_intra_ist_transform(coeff, scan, b->w, b->h, depth, mode, primary, code, 0);
    params.avm_fp_round = work->config->avm_fp_round;
    if (svt_av2_quantize_block(&params, coeff, qcoeff, dqcoeff, count, 0,
            svt_av2_tx_log_scale(b->w, b->h)) != SVT_AV2_QUANT_OK)
        return SVT_AV2_ERROR_UNSUPPORTED;
    if (svt_av2_coeff_find_eob(qcoeff, scan, count) <= 1u) return SVT_AV2_ERROR_NONE;
    svt_av2_intra_ist_transform(dqcoeff, scan, b->w, b->h, depth, mode, primary, code, 1);
    memcpy(work->scratch->qcoeff, qcoeff, count * sizeof(*qcoeff));
    memcpy(work->scratch->dqcoeff, dqcoeff, count * sizeof(*dqcoeff));
    *coded = (uint8_t)code;
    return SVT_AV2_ERROR_NONE;
}

#if SVT_AV2_INTER_PIXEL_REUSE
static int rd_inter_pixel_bucket(const RdWork* work, uint32_t leaf, const SvtAv2BlockView* y) {
    const SvtAv2BlockView* sb = &work->region[0];
    if (sb->w != 64u || sb->h != 64u || y->w != y->h || work->uv_y_cache || work->tx_types[leaf] ||
        work->uv_ids[leaf] || work->mrl[leaf] || work->mhccp[leaf] || work->cfl_explicit[leaf] ||
        work->cfl_alpha_idx[leaf] || work->cfl_alpha_signs[leaf] || work->coeff_policy ||
        (leaf < 4u && (work->inter_skip_mask & (1u << leaf)))) return -1;
    const uint32_t x = y->x - sb->x, offset_y = y->y - sb->y;
    if (y->w == 64u && x == 0u && offset_y == 0u) return 0;
    if (y->w == 32u && (x == 0u || x == 32u) && (offset_y == 0u || offset_y == 32u))
        return 1 + (int32_t)(x / 32u + 2u * (offset_y / 32u));
    return -1;
}

static const RdInterPixelEntry* rd_inter_pixel_find(const RdInterPixelMemo* memo, uint32_t bucket,
    const SvtAv2TileBlockMode* key) {
    for (uint32_t i = 0u; i < memo->used[bucket]; ++i) {
        const RdInterPixelEntry* e = &memo->entries[bucket][i];
        if (e->mode.mv_x == key->mv_x && e->mode.mv_y == key->mv_y &&
            e->mode.reference_rank == key->reference_rank) return e;
    }
    return NULL;
}

static SvtAv2Error rd_inter_pixel_load(RdWork* work, uint32_t leaf, const SvtAv2BlockView blocks[3],
    const RdInterPixelEntry* cached, uint64_t sse[3], uint32_t step) {
    for (uint32_t p = 0u; p < 3u; ++p) {
        const SvtAv2BlockView* b = &blocks[p];
        const size_t count = (size_t)SVT_AV2_MIN(b->w, 32u) * SVT_AV2_MIN(b->h, 32u);
        if (count > RD_MAX_COEFFS - work->coeff_count) return SVT_AV2_ERROR_INVALID_STATE;
        SvtAv2TileCoeffBlock entry = {0};
        entry.plane = p; entry.x = b->x; entry.y = b->y; entry.width = b->w; entry.height = b->h;
        entry.qcoeff = work->coeffs + work->coeff_count;
        work->blocks[leaf * 3u + p] = entry;
        memcpy(work->coeffs + work->coeff_count, cached->coeff[p], count * sizeof(int32_t));
        work->coeff_count += count;
        const size_t row = (size_t)b->w * step, stride = work->recon->strides[p];
        uint8_t* dst = work->recon->planes[p] + (size_t)b->y * stride + (size_t)b->x * step;
        for (uint32_t y = 0u; y < b->h; ++y)
            memcpy(dst + (size_t)y * stride, cached->pixels[p] + y * row, row);
        sse[p] += cached->sse[p];
    }
    return SVT_AV2_ERROR_NONE;
}

static void rd_inter_pixel_store(RdWork* work, uint32_t leaf, const SvtAv2BlockView blocks[3],
    uint32_t bucket, const uint64_t before[3], const uint64_t sse[3], uint32_t step) {
    RdInterPixelMemo* memo = work->inter_pixel;
    const uint32_t slot = memo->next[bucket];
    RdInterPixelEntry* e = &memo->entries[bucket][slot];
    e->mode = work->modes[leaf];
    for (uint32_t p = 0u; p < 3u; ++p) {
        const SvtAv2BlockView* b = &blocks[p];
        const size_t count = (size_t)SVT_AV2_MIN(b->w, 32u) * SVT_AV2_MIN(b->h, 32u);
        memcpy(e->coeff[p], work->blocks[leaf * 3u + p].qcoeff, count * sizeof(int32_t));
        const size_t row = (size_t)b->w * step, stride = work->recon->strides[p];
        const uint8_t* src = work->recon->planes[p] + (size_t)b->y * stride + (size_t)b->x * step;
        for (uint32_t y = 0u; y < b->h; ++y)
            memcpy(e->pixels[p] + y * row, src + (size_t)y * stride, row);
        e->sse[p] = sse[p] - before[p];
    }
    if (memo->used[bucket] < RD_INTER_PIXEL_WAYS) ++memo->used[bucket];
    memo->next[bucket] = (slot + 1u) % RD_INTER_PIXEL_WAYS;
}
#endif

#if SVT_AV2_EXPERIMENTAL_INTRA_EOB
static uint32_t intra_weak_eob(const int32_t* qcoeff, uint32_t width, uint32_t height,
    uint32_t tx_type, uint16_t out[SVT_AV2_INTER_TAIL_MAX]) {
    /* Initial subset: packed square DCTs only. Never alter a typed transform
     * whose implicit type might change after the EOB is reduced. */
    SvtAv2TxSize size;
    if (tx_type || width != height || width < 8u || width > 64u ||
        svt_av2_coeff_tx_size_from_dims(width, height, &size) != SVT_AV2_COEFF_OK)
        return 0u;
    uint16_t scan[1024];
    const size_t count = svt_av2_coeff_scan_dct_dct(size, scan);
    return svt_av2_inter_tail_positions(qcoeff, scan, (uint32_t)count, out);
}
#endif

static SvtAv2Error reconstruct_leaf_planes(RdWork* work, uint32_t leaf, uint32_t mode,
    const SvtAv2BlockView blocks[3], uint64_t sse[3], uint32_t planes) {
    SvtAv2TileBlockMode* resolved = &work->modes[leaf];
    if (!rd_inter_mode(work, leaf, mode, resolved)) return SVT_AV2_ERROR_INVALID_STATE;
    const uint32_t step = work->source->bit_depth > 8u ? 2u : 1u;
#if SVT_AV2_INTER_PIXEL_REUSE
    int bucket = -1;
    uint64_t before[3];
    if (mode == 3u && planes == 3u && work->inter_pixel != NULL && !work->inter_skip_mask &&
        (work->tail == NULL || leaf >= INTER_FINE_MAX_LEAVES ||
         !((work->tail->mask | work->tail->zero_mask) & (UINT64_C(7) << (3u * leaf)))))
        bucket = rd_inter_pixel_bucket(work, leaf, &blocks[0]);
    if (bucket >= 0) {
        const RdInterPixelEntry* hit = rd_inter_pixel_find(work->inter_pixel, (uint32_t)bucket, resolved);
        if (hit != NULL) {
            ++work->inter_pixel->hits;
            return rd_inter_pixel_load(work, leaf, blocks, hit, sse, step);
        }
        ++work->inter_pixel->misses;
        memcpy(before, sse, sizeof(before));
    }
#endif
    SvtAv2Error error = predict_with_motion(work, leaf, mode, blocks, NULL, planes==1u, resolved);
    if (error != SVT_AV2_ERROR_NONE) return error;
    if (work->inter_skip_mask && leaf < INTER_FINE_MAX_LEAVES && (work->inter_skip_mask & (1u << leaf))) {
        if (mode != 3u || planes != 3u) return SVT_AV2_ERROR_INVALID_STATE;
        return reconstruct_inter_zero(work, leaf, blocks, sse);
    }
    for (uint32_t plane = 0u; plane < planes; ++plane) {
        const SvtAv2BlockView* block = &blocks[plane];
        const size_t count = (size_t)SVT_AV2_MIN(block->w, 32u) * SVT_AV2_MIN(block->h, 32u);
        const uint64_t old_sse = sse[plane];
        uint8_t* destination = work->recon->planes[plane] +
            (size_t)block->y * work->recon->strides[plane] + (size_t)block->x * step;
        if (plane == 0u && work->uv_y_cache == 2u) {
            SvtAv2TileCoeffBlock* entry = &work->blocks[leaf * 3u];
            *entry = work->uv_y_blocks[leaf];
#if SVT_AV2_EXPERIMENTAL_STILL_UV_RD
            /* The Y pixels and coefficients are immutable during the UV
             * pass, but these fields belong to the proposed chroma mode. */
            entry->mhccp=work->mhccp[leaf];
            entry->cfl_explicit=work->cfl_explicit[leaf];
            entry->cfl_alpha_idx=work->cfl_alpha_idx[leaf];
            entry->cfl_alpha_signs=work->cfl_alpha_signs[leaf];
#endif
            entry->qcoeff = work->coeffs + work->coeff_count;
            memcpy(work->coeffs + work->coeff_count, work->uv_y_coeff + work->uv_y_offset[leaf], count * sizeof(int32_t));
            work->coeff_count += count;
            for (uint32_t y = 0u; y < block->h; ++y)
                memcpy(destination + (size_t)y * work->recon->strides[0],
                    (const uint8_t*)work->uv_y_recon +
                    ((size_t)(block->y - work->region[0].y + y) * 64u + block->x - work->region[0].x) * step,
                    (size_t)block->w * step);
            sse[0] += work->uv_y_sse[leaf];
            continue;
        }
        if (plane && mode != 3u && work->uv_ids[leaf] == 14u) {
            /* CfL reads the candidate Y after inverse transform and clip.
             * The earlier all-plane prediction supplies only a DC placeholder. */
            const SvtAv2IntraBlock intra = {block->x, block->y, block->w, block->h, plane};
            uint8_t cfl[64u * 64u * 2u];
            if (work->mhccp[leaf]) {
                const int32_t scale = work->source->chroma_format == SVT_AV2_CHROMA_420 ? 2 : 1;
                uint32_t tr = 0u, bl = 0u;
                while (tr < block->w && reference_decoded(work,leaf,
                    (int32_t)(block->x+block->w+tr)*scale,((int32_t)block->y-1)*scale)) ++tr;
                while (bl < block->h && reference_decoded(work,leaf,
                    ((int32_t)block->x-1)*scale,(int32_t)(block->y+block->h+bl)*scale)) ++bl;
                error=svt_av2_intra_mhccp_predict(work->recon,&intra,work->mhccp[leaf]-1u,
                    tr,bl,cfl,block->w*step,NULL);
            } else if (work->cfl_explicit[leaf]) {
                const uint32_t joint=work->cfl_alpha_signs[leaf];
                const uint32_t sign=plane==1u?(joint+1u)/3u:(joint+1u)%3u;
                const uint32_t idx=plane==1u?work->cfl_alpha_idx[leaf]>>3u:work->cfl_alpha_idx[leaf]&7u;
                const int32_t alpha=sign==0u?0:sign==1u?-(int32_t)(idx+1u):(int32_t)(idx+1u);
                error=svt_av2_intra_cfl_explicit_predict(work->recon,&intra,64u,cfl,block->w*step,alpha);
            } else error = svt_av2_intra_cfl_implicit_predict(work->recon, &intra, 64u,
                cfl, block->w * step, NULL);
            if (error != SVT_AV2_ERROR_NONE) return error;
            for (uint32_t i = 0u; i < block->w * block->h; ++i)
                work->prediction[plane][i] = svt_av2_sample_load(cfl + i * step, work->source->bit_depth);
        }
        for (uint32_t y = 0u; y < block->h; ++y) {
            const uint8_t* source = work->source->planes[plane] +
                (size_t)(block->y + y) * work->source->strides[plane] + (size_t)block->x * step;
            for (uint32_t x = 0u; x < block->w; ++x) {
                const uint16_t prediction = work->prediction[plane][(size_t)y * block->w + x];
                work->scratch->residual[(size_t)y * work->scratch->stride + x] =
                    (int16_t)((int)svt_av2_sample_load(source + x * step, work->source->bit_depth) - prediction);
                svt_av2_sample_store(destination + (size_t)y * work->recon->strides[plane] + x * step,
                    work->recon->bit_depth, prediction);
            }
        }
        uint32_t tx_type = plane == 0u ? work->tx_types[leaf] : 0u;
        int all_zero = 0;
        error = forward_trial(work,plane,block,tx_type,&all_zero);
        if (error != SVT_AV2_ERROR_NONE) return error;
        if (tx_type != 0u) {
            uint16_t scan[256];
            SvtAv2TxSize size;
            if (svt_av2_coeff_tx_size_from_dims(block->w, block->h, &size) != SVT_AV2_COEFF_OK ||
                svt_av2_coeff_scan_dct_dct(size, scan) != count)
                return SVT_AV2_ERROR_INVALID_STATE;
            /* EOB<=1 never signals an intra TX type. Requantize DCT instead. */
            if (svt_av2_coeff_find_eob(work->scratch->qcoeff, scan, count) <= 1u) {
                tx_type = 0u;
                error = forward_trial(work,plane,block,0u,&all_zero);
                if (error != SVT_AV2_ERROR_NONE) return error;
            }
        }
        const uint32_t policy = leaf < 4u ? (work->coeff_policy >> (3u * leaf + plane)) & 1u : 0u;
        if (policy != 0u) {
            uint16_t scan[1024];
            uint32_t tx_size = 0u;
            for (uint32_t side = 4u; side < block->w; side *= 2u) ++tx_size;
            if (svt_av2_coeff_scan_dct_dct((SvtAv2TxSize)tx_size, scan) != count)
                return SVT_AV2_ERROR_INVALID_STATE;
            const uint32_t eob = svt_av2_coeff_find_eob(work->scratch->qcoeff, scan, count);
            if (eob > 1u) {
                int32_t* last = &work->scratch->qcoeff[scan[eob - 1u]];
                *last += *last > 0 ? -1 : 1;
                SvtAv2QuantParams params;
                if (svt_av2_quant_params_init_matrix(&params, work->config->frame_header.base_q_idx,
                    block->w, block->h, work->source->bit_depth, work->config->frame_header.qm_level, work->config->frame_header.uv_delta_q) != SVT_AV2_QUANT_OK ||
                    svt_av2_dequantize_block(&params, work->scratch->qcoeff,
                        work->scratch->dqcoeff, count, (int)plane) != SVT_AV2_QUANT_OK)
                    return SVT_AV2_ERROR_INVALID_STATE;
            }
        }
        const uint32_t rdoq_block = leaf * 3u + plane;
        if (work->config->coeff_optimization == 2u &&
            (work->rdoq_target == rdoq_block || work->rdoq_target == RD_RDOQ_LEAF)) {
#if M6_FLIP_TX
            const uint32_t tx_mode=plane==0u?svt_av2_intra_tx_mode(mode,block->w,block->h,work->mrl[leaf]):mode;
#else
            const uint32_t tx_mode=mode;
#endif
#if M6_RDOQ_CONTEXT
            const uint64_t prefix_before=work->rdoq_rate_ready?
                svt_av2_tile_rate_prefix_current_rate(work->rdoq_rate_prefix):0u;
            work->rdoq_rate_active=work->rdoq_rate_ready;
#endif
            error = rdoq_propose(work, plane, block, count, tx_type, tx_mode);
#if M6_RDOQ_CONTEXT
            work->rdoq_rate_active=0u;
            if (error==SVT_AV2_ERROR_NONE && work->rdoq_rate_ready &&
                prefix_before!=svt_av2_tile_rate_prefix_current_rate(work->rdoq_rate_prefix))
                error=SVT_AV2_ERROR_INVALID_STATE;
#endif
            if (error != SVT_AV2_ERROR_NONE) return error;
            work->rdoq_tx_types[rdoq_block] = (uint8_t)tx_type;
            memcpy(work->rdoq_coeff + work->coeff_count, work->scratch->qcoeff, count * sizeof(int32_t));
        } else if (work->rdoq_saved[rdoq_block]) {
            SvtAv2QuantParams params;
            tx_type = work->rdoq_tx_types[rdoq_block];
            memcpy(work->scratch->qcoeff, work->rdoq_coeff + work->coeff_count, count * sizeof(int32_t));
            if (svt_av2_quant_params_init_matrix(&params, work->config->frame_header.base_q_idx,
                block->w, block->h, work->source->bit_depth, work->config->frame_header.qm_level, work->config->frame_header.uv_delta_q) != SVT_AV2_QUANT_OK ||
                svt_av2_dequantize_block(&params, work->scratch->qcoeff,
                    work->scratch->dqcoeff, count, (int)plane) != SVT_AV2_QUANT_OK)
                return SVT_AV2_ERROR_INVALID_STATE;
        }
        if (work->tail && leaf < INTER_FINE_MAX_LEAVES && (work->tail->zero_mask & (UINT64_C(1) << rdoq_block))) {
            memset(work->scratch->qcoeff, 0, count * sizeof(int32_t));
            memset(work->scratch->dqcoeff, 0, count * sizeof(int32_t));
        }
        if (work->tail && work->tail->mask && leaf < INTER_FINE_MAX_LEAVES &&
            (work->tail->mask & (UINT64_C(1) << rdoq_block))) {
            if (mode != 3u || tx_type != 0u || !work->tail->counts[rdoq_block])
                return SVT_AV2_ERROR_INVALID_STATE;
            for (uint32_t i = 0u; i < work->tail->counts[rdoq_block]; ++i) {
                const uint32_t pos = work->tail->positions[rdoq_block][i];
                if (pos >= count || (work->scratch->qcoeff[pos] != 1 && work->scratch->qcoeff[pos] != -1))
                    return SVT_AV2_ERROR_INVALID_STATE;
                /* Native no-qmatrix dequant is elementwise; zero level maps
                 * to zero at the same packed offset (including 64x64). */
                work->scratch->qcoeff[pos] = 0;
                work->scratch->dqcoeff[pos] = 0;
            }
        }
#if SVT_AV2_EXPERIMENTAL_INTRA_EOB
        if (work->intra_eob_limit && !resolved->is_inter) {
            uint16_t positions[SVT_AV2_INTER_TAIL_MAX];
            const uint32_t n = intra_weak_eob(work->scratch->qcoeff, block->w, block->h,
                                             tx_type, positions);
            const uint32_t edits = SVT_AV2_MIN(n, work->intra_eob_limit);
            for (uint32_t i = 0u; i < edits; ++i) {
                work->scratch->qcoeff[positions[i]] = 0;
                work->scratch->dqcoeff[positions[i]] = 0;
            }
            work->intra_eob_edits += edits;
        }
#endif
        uint8_t ist = 0u;
        if (plane == 0u && work->ist[leaf]) {
            error = quantize_ist(work, block, tx_type, rd_ist_mode(mode, block, work->mrl[leaf]),
                work->ist[leaf], &ist);
            if (error != SVT_AV2_ERROR_NONE) return error;
        }
        SVT_AV2_IP_START(profile_inverse);
        error = tx_type ? work->config->transform.inverse_type(work->config->transform.user, plane, block->w, block->h,
            work->scratch->dqcoeff, work->scratch->stride, work->scratch->inverse_residual, tx_type) :
            work->config->transform.inverse(work->config->transform.user, plane, block->w, block->h,
            work->scratch->dqcoeff, work->scratch->stride, work->scratch->inverse_residual);
        SVT_AV2_IP_ADD(work->profile->inverse_path, profile_inverse);
        if (error != SVT_AV2_ERROR_NONE) return error;
        error = svt_av2_recon_write_block(work->recon, plane, block, destination,
            work->recon->strides[plane], work->scratch->inverse_residual, work->scratch->stride);
        if (error != SVT_AV2_ERROR_NONE) return error;
        if (count > RD_MAX_COEFFS - work->coeff_count) return SVT_AV2_ERROR_INVALID_STATE;
        SvtAv2TileCoeffBlock* entry = &work->blocks[leaf * 3u + plane];
        entry->plane = plane; entry->x = block->x; entry->y = block->y;
        entry->width = block->w; entry->height = block->h;
        entry->intra_mode = (uint8_t)(mode == 3u ? 0u : plane == 0u ? mode :
            (work->config->frame_header.m5_intra_modes >= 2u || work->config->frame_header.enable_cfl_intra) ?
            work->uv_ids[leaf] : 0u);
        entry->tx_type = (uint8_t)tx_type;
        entry->ist = ist;
        entry->mrl = (uint8_t)(plane == 0u ? work->mrl[leaf] : 0u);
        entry->mhccp = (uint8_t)(plane == 0u ? work->mhccp[leaf] : 0u);
        entry->cfl_explicit = (uint8_t)(plane == 0u ? work->cfl_explicit[leaf] : 0u);
        entry->cfl_alpha_idx = (uint8_t)(plane == 0u ? work->cfl_alpha_idx[leaf] : 0u);
        entry->cfl_alpha_signs = (uint8_t)(plane == 0u ? work->cfl_alpha_signs[leaf] : 0u);
        entry->qcoeff = work->coeffs + work->coeff_count;
        memcpy(work->coeffs + work->coeff_count, work->scratch->qcoeff, count * sizeof(int32_t));
        work->coeff_count += count;
        for (uint32_t y = 0u; y < block->h; ++y) {
            const uint8_t* src = work->source->planes[plane] +
                (size_t)(block->y + y) * work->source->strides[plane] + (size_t)block->x * step;
            const uint8_t* rec = destination + (size_t)y * work->recon->strides[plane];
            for (uint32_t x = 0u; x < block->w; ++x) {
                const int difference = (int)svt_av2_sample_load(src + x * step, work->source->bit_depth) -
                    (int)svt_av2_sample_load(rec + x * step, work->recon->bit_depth);
                sse[plane] += (uint64_t)(difference * difference);
            }
        }
        if (work->detail_rd && !plane) sse[0] += detail_error(work, block);
        if (plane == 0u && work->uv_y_cache == 1u) {
            work->uv_y_blocks[leaf] = *entry;
            work->uv_y_offset[leaf] = work->uv_y_size;
            memcpy(work->uv_y_coeff + work->uv_y_size, entry->qcoeff, count * sizeof(int32_t));
            work->uv_y_size += (uint32_t)count;
            work->uv_y_sse[leaf] = sse[0] - old_sse;
            for (uint32_t y = 0u; y < block->h; ++y)
                memcpy((uint8_t*)work->uv_y_recon +
                    ((size_t)(block->y - work->region[0].y + y) * 64u + block->x - work->region[0].x) * step,
                    destination + (size_t)y * work->recon->strides[0], (size_t)block->w * step);
        }
    }
#if SVT_AV2_INTER_PIXEL_REUSE
    if (bucket >= 0) rd_inter_pixel_store(work, leaf, blocks, (uint32_t)bucket, before, sse, step);
#endif
    return SVT_AV2_ERROR_NONE;
}

static SvtAv2Error reconstruct_leaf(RdWork* work,uint32_t leaf,uint32_t mode,
    const SvtAv2BlockView blocks[3],uint64_t sse[3]) {
    return reconstruct_leaf_planes(work,leaf,mode,blocks,sse,3u);
}

static uint64_t path_hash(const RdWork* work) {
    uint64_t hash = RD_HASH_OFFSET;
    for (uint32_t i = 0u; i < work->leaf_count; ++i) {
        hash = hash_word(hash, work->ids[i]);
        if (work->config->frame_header.enable_mrls) hash = hash_word(hash, work->mrl[i]);
        if (work->config->transform_selection) hash = hash_word(hash, work->blocks[i * 3u].tx_type);
        if (work->blocks[i * 3u].ist) hash = hash_word(hash, work->blocks[i * 3u].ist);
        if (work->config->frame_header.m5_intra_modes >= 2u || work->config->frame_header.enable_cfl_intra)
            hash = hash_word(hash, work->uv_ids[i]);
        if (work->config->mhccp_search) hash = hash_word(hash, work->mhccp[i]);
        if (work->cfl_explicit[i] && work->uv_ids[i] == 14u) {
            hash = hash_word(hash, 1u);
            hash = hash_word(hash, work->cfl_alpha_idx[i]);
            hash = hash_word(hash, work->cfl_alpha_signs[i]);
        }
        hash = hash_word(hash, (uint32_t)work->modes[i].mv_x);
        hash = hash_word(hash, (uint32_t)work->modes[i].mv_y);
        if (work->secondary != NULL) hash = hash_word(hash, work->modes[i].reference_rank);
    }
    for (size_t i = 0u; i < work->coeff_count; ++i) hash = hash_word(hash, (uint32_t)work->coeffs[i]);
    const uint32_t step = work->recon->bit_depth > 8u ? 2u : 1u;
    for (uint32_t plane = 0u; plane < 3u; ++plane) {
        const SvtAv2BlockView* block = &work->region[plane];
        for (uint32_t y = 0u; y < block->h; ++y) {
            const uint8_t* row = work->recon->planes[plane] +
                (size_t)(block->y + y) * work->recon->strides[plane] + (size_t)block->x * step;
            for (uint32_t x = 0u; x < block->w; ++x)
                hash = hash_word(hash, svt_av2_sample_load(row + x * step, work->recon->bit_depth));
        }
    }
    if (work->coeff_policy != 0u) hash = hash_word(hash, work->coeff_policy);
    if (work->motion_policy != 0u) hash = hash_word(hash, work->motion_policy);
    if (work->wide_stage != 0u) hash = hash_word(hash, work->wide_stage);
    if (work->inter_skip_mask != 0u) hash = hash_word(hash, work->inter_skip_mask);
    return hash;
}

/* ordinal digits choose mode IDs in exhaustive, ranked SAD slots in pruned.
 * saved IDs replay exactly; a candidate never reconstructs a different winner. */
static SvtAv2Error build_path(RdWork* work, uint32_t ordinal, int pruned,
    int partial, const uint8_t* saved_ids, int* legal, uint64_t* distortion, uint32_t* path_id) {
    uint64_t sse[3] = {0u, 0u, 0u};
    const uint32_t first_leaf = work->prefix_active ? work->prefix_leaf : 0u;
    if (work->prefix_active) memcpy(sse, work->prefix_sse, sizeof(sse));
    work->coeff_count = work->prefix_active ? work->prefix_coeff_count : 0u;
    *path_id = 0u; *legal = 1;
    if (!partial && work->leaf_count <= 4u)
        for (uint32_t leaf = 0u; leaf < first_leaf; ++leaf)
            *path_id |= work->ids[leaf] << ((work->config->frame_header.m5_intra_modes >= 3u ? 4u :
                work->config->frame_header.m5_intra_modes == 2u ? 3u : 2u) * leaf);
    if (!partial && !pruned && saved_ids == NULL) {
        uint32_t modes = ordinal;
        for (uint32_t leaf = first_leaf; leaf < work->leaf_count; ++leaf) {
            if (!legal_mode(work, leaf, modes % 4u)) { *legal = 0; return SVT_AV2_ERROR_NONE; }
            modes /= 4u;
        }
    }
    for (uint32_t leaf = first_leaf; leaf < work->leaf_count; ++leaf) {
        SvtAv2BlockView blocks[3];
        SvtAv2Error error = plane_blocks(work, leaf, blocks);
        if (error != SVT_AV2_ERROR_NONE) return error;
        uint32_t mode;
        if (saved_ids != NULL && !(work->rank_intra_only && saved_ids[leaf] == UINT8_MAX)) mode = saved_ids[leaf];
        else if (partial || pruned) {
            uint8_t ranked[4]; uint32_t count;
            error = rank_modes(work, leaf, blocks, ranked, &count, partial);
            if (error != SVT_AV2_ERROR_NONE) return error;
            const uint32_t digit = partial ? 0u : ordinal % 2u;
            ordinal /= 2u;
            if (digit >= count) { *legal = 0; return SVT_AV2_ERROR_NONE; }
            mode = ranked[digit];
        } else { mode = ordinal % 4u; ordinal /= 4u; }
        if (!legal_mode(work, leaf, mode)) { *legal = 0; return SVT_AV2_ERROR_NONE; }
        work->ids[leaf] = (uint8_t)mode;
        /* Only coarse full trees fit the packed ID; fine/partial use the hash. */
        if (!partial && work->leaf_count <= 4u) *path_id |= mode << ((work->config->frame_header.m5_intra_modes >= 3u ? 4u :
            work->config->frame_header.m5_intra_modes == 2u ? 3u : 2u) * leaf);
        error = reconstruct_leaf(work, leaf, mode, blocks, sse);
        if (error != SVT_AV2_ERROR_NONE) return error;
    }
    *distortion = svt_av2_rd_distortion(sse, work->source->bit_depth, work->source->chroma_format);
    return SVT_AV2_ERROR_NONE;
}

#if SVT_AV2_INTER_COARSE_PREFIX
static void coarse_prefix_pixels(RdWork* work, const SvtAv2BlockView blocks[3],
                                 uint8_t* pixels, int load) {
    const uint32_t step = work->recon->bit_depth > 8u ? 2u : 1u;
    for (uint32_t p = 0u; p < 3u; ++p) {
        const SvtAv2BlockView* b = &blocks[p];
        const size_t span = (size_t)b->w * step;
        uint8_t* current = work->recon->planes[p] +
            (size_t)b->y * work->recon->strides[p] + (size_t)b->x * step;
        for (uint32_t y = 0u; y < b->h; ++y) {
            uint8_t* row = current + (size_t)y * work->recon->strides[p];
            if (load) memcpy(row, pixels, span);
            else memcpy(pixels, row, span);
            pixels += span;
        }
    }
}

/* Ref, geometry, normative committed MI/CDF and the external pixel boundary
 * are fixed during these 16 speculative paths. At each depth the preceding
 * digits fix the original earlier inter modes/MVs and reconstructed planes;
 * untouched future pixels have the same initial backup. The chosen digit
 * extends that exact prefix. Keep full rate trials and final replay separate. */
static SvtAv2Error coarse_prefix_path(RdWork* work, uint32_t ordinal, int* legal,
                                      uint64_t* distortion, uint32_t* path_id) {
    RdCoarsePrefix* cache = work->coarse_prefix;
    uint64_t sse[3] = {0u, 0u, 0u};
    uint32_t prefix = 0u;
    const size_t chroma_area = work->source->chroma_format == SVT_AV2_CHROMA_420 ? 256u : 1024u;
    const size_t area = 1024u + 2u * chroma_area;
    work->coeff_count = 0u;
    *path_id = 0u; *legal = 1;
    for (uint32_t leaf = 0u; leaf < 4u; ++leaf) {
        const SvtAv2BlockView* g = cache->geometry[leaf];
        const uint32_t rank_index = (1u << leaf) - 1u + prefix;
        const uint16_t rank_bit = (uint16_t)(1u << rank_index);
        if (cache->rank_valid & rank_bit) ++cache->rank_hits;
        else {
            uint8_t ranked[4]; uint32_t count = 0u;
            SvtAv2Error error = rank_modes(work, leaf, g, ranked, &count, 0);
            if (error != SVT_AV2_ERROR_NONE) return error;
            /* A visible P leaf always has DC and inter, but preserve the
             * ordinal legality rule even if additional modes are disabled. */
            for (uint32_t i = 0u; i < 2u; ++i)
                cache->choices[rank_index][i] = i < count ? ranked[i] : UINT8_MAX;
            cache->rank_valid |= rank_bit;
        }
        const uint32_t digit = (ordinal >> leaf) & 1u;
        const uint32_t mode = cache->choices[rank_index][digit];
        if (mode == UINT8_MAX || !legal_mode(work, leaf, mode)) {
            *legal = 0; return SVT_AV2_ERROR_NONE;
        }
        work->ids[leaf] = (uint8_t)mode;
        *path_id |= mode << (2u * leaf);
        prefix |= digit << leaf;
        if (leaf == 3u) {
            SvtAv2Error error = reconstruct_leaf(work, leaf, mode, g, sse);
            if (error != SVT_AV2_ERROR_NONE) return error;
            continue;
        }
        const uint32_t index = (2u << leaf) - 2u + prefix;
        const uint16_t bit = (uint16_t)(1u << index);
        RdCoarseLeaf* node = &cache->leaves[index];
        SvtAv2TileCoeffBlock* block = work->blocks + 3u * leaf;
        int32_t* coeff = work->coeffs + work->coeff_count;
        if (cache->leaf_valid & bit) {
            ++cache->leaf_hits;
            work->modes[leaf] = node->mode;
            memcpy(block, node->blocks, sizeof(node->blocks));
            block[0].qcoeff = coeff;
            block[1].qcoeff = coeff + 1024u;
            block[2].qcoeff = coeff + 1024u + chroma_area;
            memcpy(coeff, node->qcoeff, area * sizeof(*coeff));
            work->coeff_count += area;
            for (uint32_t p = 0u; p < 3u; ++p) sse[p] += node->sse[p];
            coarse_prefix_pixels(work, g, node->pixels, 1);
        } else {
            uint64_t delta[3] = {0u, 0u, 0u};
            const size_t before = work->coeff_count;
            SvtAv2Error error = reconstruct_leaf(work, leaf, mode, g, delta);
            if (error != SVT_AV2_ERROR_NONE) return error;
            if (work->coeff_count - before != area) return SVT_AV2_ERROR_INVALID_STATE;
            node->mode = work->modes[leaf];
            memcpy(node->blocks, block, sizeof(node->blocks));
            for (uint32_t p = 0u; p < 3u; ++p) {
                node->blocks[p].qcoeff = NULL;
                node->sse[p] = delta[p];
                sse[p] += delta[p];
            }
            memcpy(node->qcoeff, coeff, area * sizeof(*coeff));
            coarse_prefix_pixels(work, g, node->pixels, 0);
            cache->leaf_valid |= bit;
        }
    }
    *distortion = svt_av2_rd_distortion(sse, work->source->bit_depth, work->source->chroma_format);
    return SVT_AV2_ERROR_NONE;
}

#if SVT_AV2_INTER_COARSE_PREFIX_VERIFY
static SvtAv2Error verify_coarse_prefix(RdWork* work, size_t sb, uint32_t ordinal,
                                      int legal, uint64_t distortion, uint32_t path_id) {
    uint64_t rate = 0u;
    const uint64_t hash = legal ? path_hash(work) : 0u;
    if (legal && RD_RATE(work, sb, 1u, work->blocks, 12u, work->modes, 4u, 0, &rate) != 0)
        return SVT_AV2_ERROR_INVALID_STATE;
    region_copy(work, 1);
    uint64_t expected_d = 0u, expected_rate = 0u;
    uint32_t expected_id; int expected_legal;
    SvtAv2Error error = build_path(work, ordinal, 1, 0, NULL, &expected_legal, &expected_d, &expected_id);
    if (error != SVT_AV2_ERROR_NONE) return error;
    if (legal != expected_legal || (legal && (distortion != expected_d || path_id != expected_id ||
        hash != path_hash(work) || RD_RATE(work, sb, 1u, work->blocks, 12u, work->modes, 4u, 0, &expected_rate) != 0 ||
        rate != expected_rate))) return SVT_AV2_ERROR_INVALID_STATE;
    return SVT_AV2_ERROR_NONE;
}
#endif
#endif

static SvtAv2Error append_winner(RdWork* work) {
    SvtAv2RdFrameOutput* output = work->output;
    const size_t block_count = work->leaf_count * 3u;
    if (block_count > output->block_capacity - output->block_count ||
        work->leaf_count > output->mode_capacity - output->mode_count ||
        work->coeff_count > output->coeff_capacity - output->coeff_count)
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    for (size_t i = 0u; i < block_count; ++i) {
        SvtAv2TileCoeffBlock entry = work->blocks[i];
        const size_t offset = (size_t)(entry.qcoeff - work->coeffs);
        entry.qcoeff = output->qcoeff + output->coeff_count + offset;
        output->blocks[output->block_count + i] = entry;
    }
    memcpy(output->modes + output->mode_count, work->modes, work->leaf_count * sizeof(*work->modes));
    memcpy(output->qcoeff + output->coeff_count, work->coeffs, work->coeff_count * sizeof(*work->coeffs));
    output->block_count += block_count; output->mode_count += work->leaf_count;
    output->coeff_count += work->coeff_count;
#if SVT_AV2_EXPERIMENTAL_STILL_UV_RD
    if (work->config->frame_header.m5_intra_modes==4u) {
        for (uint32_t i=0u;i<work->leaf_count;++i) {
            const SvtAv2TileCoeffBlock* b=work->blocks+3u*i;
            const unsigned bin=b[1].intra_mode!=14u?0u:b->mhccp?3u:b->cfl_explicit?1u:2u;
            ++work->final_uv[bin];
        }
    }
#endif
    return SVT_AV2_ERROR_NONE;
}

/* Extended plans keep stage payloads in their legacy low word. Existing
 * plans retain their exact ID, tie order and audit hash. */
static uint64_t plan_candidate_id(uint16_t decision, uint32_t legacy, uint32_t payload) {
    return decision < 255u ? legacy : ((uint64_t)decision << 32u) | payload;
}

typedef struct RdBasicChoice {
    uint64_t cost, distortion, rate, hash;
    uint64_t id;
} RdBasicChoice;

/* A coordinate trial cannot change earlier coding leaves. Reconstruct that
 * fixed prefix once, retaining its coefficient storage, modes and exact SSE.
 * Each trial still rebuilds the entire affected suffix and rates the full SB. */
static SvtAv2Error prepare_prefix(RdWork* work, uint32_t first_leaf) {
    region_copy(work, 1);
    work->coeff_count = 0u;
    memset(work->prefix_sse, 0, sizeof(work->prefix_sse));
    for (uint32_t leaf = 0u; leaf < first_leaf; ++leaf) {
        SvtAv2BlockView blocks[3];
        SvtAv2Error error = plane_blocks(work, leaf, blocks);
        if (error != SVT_AV2_ERROR_NONE) return error;
        work->ids[leaf] = work->best_ids[leaf];
        error = reconstruct_leaf(work, leaf, work->best_ids[leaf], blocks, work->prefix_sse);
        if (error != SVT_AV2_ERROR_NONE) return error;
    }
    work->prefix_coeff_count = work->coeff_count;
    work->prefix_leaf = first_leaf;
    region_copy_buffer(work, work->prefix_recon, 0);
    return SVT_AV2_ERROR_NONE;
}

/* Later coordinate passes fix all leaves before the next target. Extend
 * that prefix only once; trials still rebuild the suffix and rate the full SB. */
static SvtAv2Error extend_prefix(RdWork* work, uint32_t first_leaf) {
    region_copy_buffer(work, work->prefix_recon, 1);
    work->coeff_count = work->prefix_coeff_count;
    for (uint32_t leaf = work->prefix_leaf; leaf < first_leaf; ++leaf) {
        SvtAv2BlockView blocks[3];
        SvtAv2Error error = plane_blocks(work, leaf, blocks);
        if (error != SVT_AV2_ERROR_NONE) return error;
        work->ids[leaf] = work->best_ids[leaf];
        error = reconstruct_leaf(work, leaf, work->best_ids[leaf], blocks, work->prefix_sse);
        if (error != SVT_AV2_ERROR_NONE) return error;
    }
    work->prefix_coeff_count = work->coeff_count;
    work->prefix_leaf = first_leaf;
    region_copy_buffer(work, work->prefix_recon, 0);
    return SVT_AV2_ERROR_NONE;
}

static SvtAv2Error refine_basic_original(RdWork* work, size_t sb, int partial,
    uint16_t decision, RdBasicChoice* best) {
    SvtAv2Error error;
    /* One bounded raster coordinate pass, seeded by the original DC/V/H
     * winner. Each Y or shared U/V trial restores its fixed coding prefix,
     * rebuilds the affected suffix and trials the complete actual tile syntax.
     * Later TXs therefore see the candidate's normative earlier samples. */
    error = prepare_prefix(work, 0u);
    if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
    work->prefix_active = 1u;
    for (uint32_t leaf = 0u; leaf < work->leaf_count; ++leaf) {
        error = extend_prefix(work, leaf);
        if (error != SVT_AV2_ERROR_NONE) {
            work->prefix_active = 0u; region_copy(work, 1); return error;
        }
        for (uint32_t component = 0u; component < 2u; ++component) {
            uint32_t choice = component ? work->uv_ids[leaf] : work->best_ids[leaf];
            const uint32_t entering_uv = work->uv_ids[leaf];
            const uint32_t mode_count = !component && work->config->frame_header.m5_intra_modes >= 3u ? 14u :
                component && work->config->frame_header.m5_intra_modes == 4u ? 9u : 8u;
            for (uint32_t index = 0u; index < mode_count; ++index) {
                const uint32_t slot = work->config->reverse_candidates ? mode_count - 1u - index : index;
                const uint32_t mode = component && slot == 8u ? 14u : slot;
                if (mode == 3u || (mode != 14u && !legal_mode(work, leaf, mode))) continue;
                uint8_t ids[RD_MAX_LEAVES];
                memcpy(ids, work->best_ids, work->leaf_count);
                if (component) work->uv_ids[leaf] = (uint8_t)mode;
                else ids[leaf] = (uint8_t)mode;
                region_copy_buffer(work, work->prefix_recon, 1);
                uint64_t d = 0u, r = 0u, cost = 0u; uint32_t path; int valid;
                error = build_path(work, 0u, 0, partial, ids, &valid, &d, &path);
                if (error == SVT_AV2_ERROR_NONE && (!valid ||
                    RD_RATE(work, sb, decision, work->blocks,
                        work->leaf_count * 3u, NULL, 0u, 0, &r) != 0))
                    error = SVT_AV2_ERROR_INVALID_STATE;
                if (error == SVT_AV2_ERROR_NONE)
                    error = svt_av2_rd_cost(d, work->stats.lambda_q8, r, &cost);
                if (error != SVT_AV2_ERROR_NONE) {
                    work->prefix_active = 0u; region_copy(work, 1); return error;
                }
                ++work->stats.candidate_count;
                const uint32_t payload = (1u << 28u) |
                    (work->config->frame_header.m5_intra_modes >= 3u ?
                        (leaf << 5u) | (component << 4u) : (leaf << 4u) | (component << 3u)) | mode;
                const uint64_t id = plan_candidate_id(decision, payload |
                    (work->config->partition_selection >= 2u ? (uint32_t)decision <<
                    (work->config->frame_header.m5_intra_modes >= 3u ? 16u : 12u) : 0u), payload);
                if (work->config->candidate_hook != NULL)
                    work->config->candidate_hook(work->config->candidate_hook_user, sb, id, d, r, cost, 0);
                if (cost < best->cost || (cost == best->cost && mode < choice)) {
                    best->cost = cost; best->distortion = d; best->rate = r; best->id = id;
                    best->hash = path_hash(work); choice = mode;
                    if (!component) memcpy(work->best_ids, ids, work->leaf_count);
                }
                if (component) work->uv_ids[leaf] = (uint8_t)entering_uv;
            }
            if (component) work->uv_ids[leaf] = (uint8_t)choice;
        }
    }
    work->prefix_active = 0u;
    region_copy(work, 1);
    return SVT_AV2_ERROR_NONE;
}

/* The clipped candidate Y is already reconstructed. Classic CfL reads only
 * that Y and causal UV DC references; this score never selects an RD winner. */
static SvtAv2Error local_basic_cost(RdWork* work, uint32_t leaf, uint32_t mode, uint64_t* cost) {
    SvtAv2BlockView blocks[3];
    SvtAv2Error error=plane_blocks(work,leaf,blocks);
    if (error!=SVT_AV2_ERROR_NONE) return error;
    uint64_t sse[3]={0u,0u,0u},rate=0u;
    work->coeff_count=work->prefix_coeff_count;work->ids[leaf]=(uint8_t)mode;
    RdBasicMemoNode* memo=basic_memo_trial(work,leaf,mode,blocks);
    if (memo && (memo->valid&(1u<<0u))) return basic_memo_cost(work,memo,0u,cost);
    error=reconstruct_leaf(work,leaf,mode,blocks,sse);
    if (error!=SVT_AV2_ERROR_NONE) return error;
    if (work->basic_rate_active) {
        for (uint32_t plane=0u;plane<3u;++plane) sse[plane]+=work->prefix_sse[plane];
        if (svt_av2_tile_rate_prefix_trial(work->basic_rate_prefix,leaf,
            work->blocks+leaf*3u,0,&rate)!=0) return SVT_AV2_ERROR_INVALID_STATE;
    } else for (uint32_t plane=0u;plane<3u;++plane) {
        const SvtAv2TileCoeffBlock* entry=&work->blocks[leaf*3u+plane];
        uint64_t part=0u;
#if M6_FLIP_TX
            const uint32_t tx_mode=plane==0u?svt_av2_intra_tx_mode(entry->intra_mode,entry->width,entry->height,entry->mrl):entry->intra_mode;
#else
            const uint32_t tx_mode=entry->intra_mode;
#endif
            error=rdoq_rate(work,plane,blocks+plane,entry->qcoeff,entry->tx_type,tx_mode,&part);
        if (error!=SVT_AV2_ERROR_NONE) return error;
        rate+=part;
    }
    error=svt_av2_rd_cost(svt_av2_rd_distortion(sse,work->source->bit_depth,
        work->source->chroma_format),work->stats.lambda_q8,rate,cost);
    if (error==SVT_AV2_ERROR_NONE && memo) basic_memo_save(work,memo,0u,sse,rate);
    return error;
}

static SvtAv2Error local_basic_y_cost(RdWork* work,uint32_t leaf,uint32_t mode,uint64_t* cost) {
    SvtAv2BlockView blocks[3];
    SvtAv2Error error=plane_blocks(work,leaf,blocks);
    if (error!=SVT_AV2_ERROR_NONE) return error;
    uint64_t sse[3],rate=0u;memcpy(sse,work->prefix_sse,sizeof(sse));
    work->coeff_count=work->prefix_coeff_count;work->ids[leaf]=(uint8_t)mode;
    RdBasicMemoNode* memo=basic_memo_trial(work,leaf,mode,blocks);
    if (memo && (memo->valid&(1u<<1u))) return basic_memo_cost(work,memo,1u,cost);
    error=reconstruct_leaf_planes(work,leaf,mode,blocks,sse,1u);
    if (error!=SVT_AV2_ERROR_NONE) return error;
    if (svt_av2_tile_rate_prefix_trial_y(work->basic_rate_prefix,leaf,
        work->blocks+leaf*3u,&rate)!=0) return SVT_AV2_ERROR_INVALID_STATE;
    error=svt_av2_rd_cost(svt_av2_rd_distortion(sse,work->source->bit_depth,
        work->source->chroma_format),work->stats.lambda_q8,rate,cost);
    if (error==SVT_AV2_ERROR_NONE && memo) basic_memo_save(work,memo,1u,sse,rate);
    return error;
}

static SvtAv2Error refine_basic_fast(RdWork* work, size_t sb, int partial,
    uint16_t decision, RdBasicChoice* best) {
    work->uv_y_cache=0u;work->memo_active=0u;work->memo_prefix=0u;
    if (work->config->rd_mode!=3u)
        return refine_basic_original(work,sb,partial,decision,best);
    uint8_t entering_y[RD_MAX_LEAVES],entering_uv[RD_MAX_LEAVES];
    memcpy(entering_y,work->best_ids,sizeof(entering_y));
    memcpy(entering_uv,work->uv_ids,sizeof(entering_uv));
    if (work->basic_rate_prefix!=NULL) {
        size_t leaves=0u;
        if (svt_av2_tile_rate_prefix_begin(work->basic_rate_prefix,work->rate,sb,
            decision,&leaves)!=0 || leaves!=work->leaf_count) return SVT_AV2_ERROR_INVALID_STATE;
        work->basic_rate_active=1u;
    }
    SvtAv2Error error=prepare_prefix(work,0u);
    if (error!=SVT_AV2_ERROR_NONE) {work->basic_rate_active=0u;region_copy(work,1);return error;}
    work->prefix_active=1u;
    work->memo_active=(uint32_t)basic_memo_allowed(work,partial);
    for (uint32_t leaf=0u;leaf<work->leaf_count && error==SVT_AV2_ERROR_NONE;++leaf) {
        error=extend_prefix(work,leaf);
        if (error!=SVT_AV2_ERROR_NONE) break;
        if (leaf && work->basic_rate_active) {
            uint64_t previous_rate=0u;
            if (svt_av2_tile_rate_prefix_trial(work->basic_rate_prefix,leaf-1u,
                work->blocks+(leaf-1u)*3u,1,&previous_rate)!=0) {
                error=SVT_AV2_ERROR_INVALID_STATE;break;
            }
        }
        if (leaf && work->memo_active) {
            SvtAv2BlockView blocks[3];uint32_t key[38];
            error=plane_blocks(work,leaf-1u,blocks);if (error!=SVT_AV2_ERROR_NONE) break;
            basic_memo_key(work,leaf-1u,work->best_ids[leaf-1u],blocks,key);
            work->memo_prefix=basic_memo_node(work->basic_memo,work->memo_prefix,key);
            if (work->memo_prefix==UINT32_MAX) work->memo_active=0u;
        }
        uint64_t y_mask=UINT64_MAX;
        if (!partial && work->basic_rate_active && work->config->frame_header.m5_intra_modes>=3u) {
            SvtAv2BlockView blocks[3];
            error=plane_blocks(work,leaf,blocks);
            if (error!=SVT_AV2_ERROR_NONE) break;
            uint8_t ranked[4]={255u,255u,255u,255u};
            uint64_t scores[4]={UINT64_MAX,UINT64_MAX,UINT64_MAX,UINT64_MAX};
            for (uint32_t mode=0u;mode<14u;++mode) {
                if (mode==3u || !legal_mode(work,leaf,mode)) continue;
                uint64_t sad[3]={0u,0u,0u};
                error=predict(work,leaf,mode,blocks,sad,1);
                if (error!=SVT_AV2_ERROR_NONE) break;
                for (uint32_t slot=0u;slot<4u;++slot) if (sad[0]<scores[slot] ||
                    (sad[0]==scores[slot] && mode<ranked[slot])) {
                    for (uint32_t j=3u;j>slot;--j) {scores[j]=scores[j-1u];ranked[j]=ranked[j-1u];}
                    scores[slot]=sad[0];ranked[slot]=(uint8_t)mode;break;
                }
            }
            if (error!=SVT_AV2_ERROR_NONE) break;
            y_mask=UINT64_C(1)<<work->best_ids[leaf];
            for (uint32_t slot=0u;slot<4u;++slot)
                if (ranked[slot]!=255u) y_mask|=UINT64_C(1)<<ranked[slot];
        }
        for (uint32_t component=0u;component<2u && error==SVT_AV2_ERROR_NONE;++component) {
            work->uv_y_size=0u;
            if (component && work->basic_rate_active) work->uv_y_cache=1u;
            const uint32_t original_uv=work->uv_ids[leaf];
            const uint32_t incumbent=component?original_uv:work->best_ids[leaf];
            uint64_t uv_mask=UINT64_MAX;
            if (component && !partial && work->basic_rate_active && work->config->frame_header.m5_intra_modes==4u) {
                SvtAv2BlockView blocks[3];
                error=plane_blocks(work,leaf,blocks);
                uint8_t ranked[2]={255u,255u};uint64_t scores[2]={UINT64_MAX,UINT64_MAX};
                for (uint32_t mode=0u;mode<8u && error==SVT_AV2_ERROR_NONE;++mode) {
                    if (mode==3u || !legal_mode(work,leaf,mode)) continue;
                    work->uv_ids[leaf]=(uint8_t)mode;
                    uint64_t sad[3]={0u,0u,0u};
                    error=predict(work,leaf,work->best_ids[leaf],blocks,sad,2);
                    if (error!=SVT_AV2_ERROR_NONE) break;
                    const uint64_t score=sad[1]+sad[2];
                    for (uint32_t slot=0u;slot<2u;++slot) if (score<scores[slot] ||
                        (score==scores[slot] && mode<ranked[slot])) {
                        if (!slot) {scores[1]=scores[0];ranked[1]=ranked[0];}
                        scores[slot]=score;ranked[slot]=(uint8_t)mode;break;
                    }
                }
                work->uv_ids[leaf]=(uint8_t)original_uv;
                if (error!=SVT_AV2_ERROR_NONE) {work->uv_y_cache=0u;break;}
                uv_mask=(UINT64_C(1)<<incumbent)|(UINT64_C(1)<<14u);
                for (uint32_t slot=0u;slot<2u;++slot)
                    if (ranked[slot]!=255u) uv_mask|=UINT64_C(1)<<ranked[slot];
            }
            uint32_t choice=incumbent;
            uint64_t selected=UINT64_MAX, second_score=UINT64_MAX;
            uint32_t second_mode=incumbent;
            const uint32_t mode_count=!component && work->config->frame_header.m5_intra_modes>=3u?14u:
                component && work->config->frame_header.m5_intra_modes==4u?9u:8u;
            for (uint32_t scan=0u;scan<mode_count;++scan) {
                const uint32_t mode=component && scan==8u?14u:scan;
                if (mode==3u || (mode!=14u && !legal_mode(work,leaf,mode))) continue;
                if (!component && !(y_mask&(UINT64_C(1)<<mode))) continue;
                if (component && !(uv_mask&(UINT64_C(1)<<mode))) continue;
                work->uv_ids[leaf]=(uint8_t)(component?mode:original_uv);
                uint64_t score=0u;
                error=!component && !partial && work->basic_rate_active &&
                    work->config->frame_header.m5_intra_modes>=3u ?
                    local_basic_y_cost(work,leaf,mode,&score) :
                    local_basic_cost(work,leaf,component?work->best_ids[leaf]:mode,&score);
                if (error!=SVT_AV2_ERROR_NONE) break;
                if (component && work->basic_rate_active && work->uv_y_size) work->uv_y_cache=2u;
                if (score<selected || (score==selected &&
                    (mode==incumbent || (choice!=incumbent && mode<choice)))) {
                    second_score=selected; second_mode=choice;
                    selected=score;choice=mode;
                } else if (score<second_score) {
                    second_score=score; second_mode=mode;
                }
            }
            if (work->config->screen_trees && !component)
                work->alternate_y[leaf]=(uint8_t)second_mode;
            work->uv_y_cache=0u;work->uv_y_size=0u;
            work->ids[leaf]=work->best_ids[leaf];
            work->uv_ids[leaf]=(uint8_t)(component?choice:original_uv);
            if (!component) work->best_ids[leaf]=(uint8_t)choice;
            work->coeff_count=work->prefix_coeff_count;
            region_copy_buffer(work,work->prefix_recon,1);
            if (selected==UINT64_MAX && error==SVT_AV2_ERROR_NONE) error=SVT_AV2_ERROR_INVALID_STATE;
        }
    }
    if (error==SVT_AV2_ERROR_NONE) error=extend_prefix(work,work->leaf_count);
    uint64_t prefix_rate=0u;
    if (error==SVT_AV2_ERROR_NONE && work->basic_rate_active &&
        (svt_av2_tile_rate_prefix_trial(work->basic_rate_prefix,work->leaf_count-1u,
            work->blocks+(work->leaf_count-1u)*3u,1,&prefix_rate)!=0 ||
         svt_av2_tile_rate_prefix_finish(work->basic_rate_prefix,&prefix_rate)!=0))
        error=SVT_AV2_ERROR_INVALID_STATE;
    uint64_t d=0u,r=0u,cost=0u;
    if (error==SVT_AV2_ERROR_NONE) {
        d=svt_av2_rd_distortion(work->prefix_sse,work->source->bit_depth,work->source->chroma_format);
        if (RD_RATE(work,sb,decision,work->blocks,
            work->leaf_count*3u,NULL,0u,0,&r)!=0 ||
            (work->basic_rate_active && prefix_rate!=r)) error=SVT_AV2_ERROR_INVALID_STATE;
    }
    if (error==SVT_AV2_ERROR_NONE) error=svt_av2_rd_cost(d,work->stats.lambda_q8,r,&cost);
    if (error==SVT_AV2_ERROR_NONE) {
        const uint32_t payload=(1u<<28u)|((work->leaf_count-1u)<<5u)|(1u<<4u)|work->uv_ids[work->leaf_count-1u];
        const uint64_t id=plan_candidate_id(decision,payload|((uint32_t)decision<<16u),payload);
        ++work->stats.candidate_count;
        if (work->config->candidate_hook!=NULL)
            work->config->candidate_hook(work->config->candidate_hook_user,sb,id,d,r,cost,0);
        if (cost<best->cost) {
            best->cost=cost;best->distortion=d;best->rate=r;best->id=id;best->hash=path_hash(work);
        } else {
            memcpy(work->best_ids,entering_y,sizeof(entering_y));
            memcpy(work->uv_ids,entering_uv,sizeof(entering_uv));
        }
    }
    work->memo_active=0u;work->basic_rate_active=0u;work->prefix_active=0u;region_copy(work,1);
    return error;
}

static SvtAv2Error refine_basic(RdWork* work,size_t sb,int partial,
    uint16_t decision,RdBasicChoice* best) {
    return work->config->rd_mode==3u?refine_basic_fast(work,sb,partial,decision,best):
        refine_basic_original(work,sb,partial,decision,best);
}

static SvtAv2Error rank_cfl_alphas(RdWork* work, const SvtAv2BlockView* block,
    uint32_t plane, int8_t ranked[2]) {
    uint8_t predicted[64u*64u*2u];
    uint64_t scores[2]={UINT64_MAX,UINT64_MAX};
    const uint32_t step=work->source->bit_depth>8u?2u:1u;
    const SvtAv2IntraBlock intra={block->x,block->y,block->w,block->h,plane};
    for (int32_t alpha=-8;alpha<=8;++alpha) {
        const SvtAv2Error error=svt_av2_intra_cfl_explicit_predict(work->recon,&intra,64u,
            predicted,block->w*step,alpha);
        if (error!=SVT_AV2_ERROR_NONE) return error;
        uint64_t score=0u;
        for (uint32_t y=0u;y<block->h;++y) for (uint32_t x=0u;x<block->w;++x) {
            const uint8_t* sample=work->source->planes[plane]+(size_t)(block->y+y)*work->source->strides[plane]+(block->x+x)*step;
            const int difference=(int)svt_av2_sample_load(sample,work->source->bit_depth)-
                (int)svt_av2_sample_load(predicted+((size_t)y*block->w+x)*step,work->source->bit_depth);
            score+=(uint64_t)(difference*difference);
        }
        for (uint32_t slot=0u;slot<2u;++slot) if (score<scores[slot]) {
            if (!slot) {scores[1]=scores[0];ranked[1]=ranked[0];}
            scores[slot]=score;ranked[slot]=(int8_t)alpha;break;
        }
    }
    return SVT_AV2_ERROR_NONE;
}

typedef struct RdJointChoice {
    RdBasicChoice choice;
    uint32_t mode;
} RdJointChoice;

/* A finite proposal domain, with exact suffix reconstruction and full tile R.
 * DCT-J only ranks two basic Y modes; it never replaces the entering tuple.
 * The complete original tool route is separately retained by the wrapper. */
static SvtAv2Error refine_joint_y_tx(RdWork* work, size_t sb, int partial,
    uint16_t decision, RdBasicChoice* best) {
    SvtAv2Error error=prepare_prefix(work,0u);
    if (error!=SVT_AV2_ERROR_NONE) {region_copy(work,1);return error;}
    work->prefix_active=1u;
    for (uint32_t leaf=0u;leaf<work->leaf_count;++leaf) {
        const uint32_t w=work->leaves[leaf].w,h=work->leaves[leaf].h;
        if (!((w==h && w<=16u) || (w==8u && h==16u) || (w==16u && h==8u))) continue;
        error=extend_prefix(work,leaf);
        if (error!=SVT_AV2_ERROR_NONE) {work->prefix_active=0u;region_copy(work,1);return error;}
        const uint8_t old_mode=work->best_ids[leaf],old_mrl=work->mrl[leaf],old_tx=work->tx_types[leaf];
        RdJointChoice ranked[2]={{{UINT64_MAX,0u,0u,0u,UINT64_MAX},255u},
            {{UINT64_MAX,0u,0u,0u,UINT64_MAX},255u}};
        for (uint32_t index=0u;index<14u;++index) {
            const uint32_t mode=work->config->reverse_candidates?13u-index:index;
            if (mode==3u || !legal_mode(work,leaf,mode)) continue;
            RdBasicChoice trial=*best;
            if (mode!=old_mode || old_mrl || old_tx) {
                work->best_ids[leaf]=(uint8_t)mode;work->mrl[leaf]=0u;work->tx_types[leaf]=0u;
                region_copy_buffer(work,work->prefix_recon,1);
                uint32_t path;int valid;
                error=build_path(work,0u,0,partial,work->best_ids,&valid,&trial.distortion,&path);
                if (error==SVT_AV2_ERROR_NONE && (!valid || RD_RATE(work,sb,decision,
                    work->blocks,work->leaf_count*3u,NULL,0u,0,&trial.rate)!=0)) error=SVT_AV2_ERROR_INVALID_STATE;
                if (error==SVT_AV2_ERROR_NONE) error=svt_av2_rd_cost(trial.distortion,work->stats.lambda_q8,trial.rate,&trial.cost);
                if (error!=SVT_AV2_ERROR_NONE) {work->prefix_active=0u;region_copy(work,1);return error;}
                trial.hash=path_hash(work);
                const uint32_t payload=((1u<<31u)|(1u<<23u))|(leaf<<8u)|mode;
                trial.id=plan_candidate_id(decision,payload,payload);
                ++work->stats.candidate_count;
                if (work->config->candidate_hook!=NULL) work->config->candidate_hook(work->config->candidate_hook_user,
                    sb,trial.id,trial.distortion,trial.rate,trial.cost,0);
            }
            for (uint32_t slot=0u;slot<2u;++slot) if (trial.cost<ranked[slot].choice.cost ||
                (trial.cost==ranked[slot].choice.cost && mode<ranked[slot].mode)) {
                if (!slot) ranked[1]=ranked[0];
                ranked[slot].choice=trial;ranked[slot].mode=mode;break;
            }
            work->best_ids[leaf]=old_mode;work->mrl[leaf]=old_mrl;work->tx_types[leaf]=old_tx;
        }
        RdBasicChoice chosen=*best,coordinate=*best;
        uint32_t chosen_code=UINT32_MAX;
        uint8_t chosen_mode=old_mode,chosen_mrl=old_mrl,chosen_tx=old_tx;
        for (uint32_t index=0u;index<8u;++index) {
            const uint32_t pair=work->config->reverse_candidates?7u-index:index;
            const uint32_t rank=pair/4u,type=pair%4u,mode=ranked[rank].mode;
            if (mode==255u || (type==1u && w==4u && mode==7u)) continue;
            work->best_ids[leaf]=(uint8_t)mode;work->mrl[leaf]=0u;work->tx_types[leaf]=(uint8_t)type;
            RdBasicChoice trial=ranked[rank].choice;
            if (type) {
                region_copy_buffer(work,work->prefix_recon,1);
                uint32_t path;int valid;
                error=build_path(work,0u,0,partial,work->best_ids,&valid,&trial.distortion,&path);
                if (error==SVT_AV2_ERROR_NONE && (!valid || RD_RATE(work,sb,decision,
                    work->blocks,work->leaf_count*3u,NULL,0u,0,&trial.rate)!=0)) error=SVT_AV2_ERROR_INVALID_STATE;
                if (error==SVT_AV2_ERROR_NONE) error=svt_av2_rd_cost(trial.distortion,work->stats.lambda_q8,trial.rate,&trial.cost);
                if (error!=SVT_AV2_ERROR_NONE) {work->prefix_active=0u;region_copy(work,1);return error;}
                trial.hash=path_hash(work);++work->stats.candidate_count;
            }
            const uint32_t code=mode*4u+type,payload=((1u<<31u)|(1u<<23u))|(1u<<20u)|(leaf<<8u)|code;
            trial.id=plan_candidate_id(decision,payload,payload);
            if (work->config->candidate_hook!=NULL) work->config->candidate_hook(work->config->candidate_hook_user,
                sb,trial.id,trial.distortion,trial.rate,trial.cost,0);
            if (!rank && trial.cost<coordinate.cost) coordinate=trial;
            if (trial.cost<chosen.cost || (trial.cost==chosen.cost && chosen_code!=UINT32_MAX && code<chosen_code)) {
                chosen=trial;chosen_code=code;chosen_mode=(uint8_t)mode;chosen_mrl=0u;chosen_tx=(uint8_t)type;
            }
        }
        if (work->config->candidate_hook!=NULL) {
            const uint32_t payload=((1u<<31u)|(1u<<23u))|(2u<<20u)|(leaf<<8u);
            work->config->candidate_hook(work->config->candidate_hook_user,sb,plan_candidate_id(decision,payload,payload),
                coordinate.distortion,coordinate.rate,coordinate.cost,0);
            work->config->candidate_hook(work->config->candidate_hook_user,sb,plan_candidate_id(decision,payload|1u,payload|1u),
                chosen.distortion,chosen.rate,chosen.cost,0);
        }
        work->best_ids[leaf]=chosen_mode;work->mrl[leaf]=chosen_mrl;work->tx_types[leaf]=chosen_tx;
        *best=chosen;
    }
    work->prefix_active=0u;region_copy(work,1);
    return SVT_AV2_ERROR_NONE;
}

#if M6_RDOQ_CONTEXT
static SvtAv2Error prepare_rdoq_rate_context(RdWork* work,size_t sb,uint16_t decision,
    uint32_t leaf) {
    work->rdoq_rate_active=0u;work->rdoq_rate_ready=0u;
    if (!work->rdoq_use_context || work->config->frame_header.m5_intra_modes<3u)
        return SVT_AV2_ERROR_NONE;
    if (!work->rdoq_rate_prefix || work->rdoq_target!=UINT32_MAX)
        return SVT_AV2_ERROR_INVALID_STATE;
    /* extend_prefix has rebuilt every prior leaf from the selected coefficients.
     * Rebuild this leaf without a new proposal; stale rejected suffixes are not
     * valid anchors. Other planes are fixed only for proposing levels. */
    region_copy_buffer(work,work->prefix_recon,1);
    SvtAv2BlockView blocks[3];
    SvtAv2Error error=plane_blocks(work,leaf,blocks);
    if (error!=SVT_AV2_ERROR_NONE) return error;
    work->coeff_count=work->prefix_coeff_count;work->ids[leaf]=work->best_ids[leaf];
    work->uv_y_cache=0u;
    uint64_t sse[3]={0u,0u,0u};
    error=reconstruct_leaf(work,leaf,work->best_ids[leaf],blocks,sse);
    if (error!=SVT_AV2_ERROR_NONE) return error;
    for (uint32_t p=0u;p<3u;++p) {
        work->rdoq_rate_blocks[p]=work->blocks[leaf*3u+p];
        const size_t count=(size_t)SVT_AV2_MIN(blocks[p].w,32u)*SVT_AV2_MIN(blocks[p].h,32u);
        memcpy(work->rdoq_rate_coeffs+p*1024u,work->blocks[leaf*3u+p].qcoeff,count*sizeof(int32_t));
        work->rdoq_rate_blocks[p].qcoeff=NULL;
    }
    size_t leaves=0u;
    if (svt_av2_tile_rate_prefix_begin(work->rdoq_rate_prefix,work->rate,sb,decision,&leaves)!=0 ||
        leaves!=work->leaf_count) return SVT_AV2_ERROR_INVALID_STATE;
    for (uint32_t i=0u;i<leaf;++i) {
        uint64_t rate=0u;
        if (svt_av2_tile_rate_prefix_trial(work->rdoq_rate_prefix,i,
            work->blocks+i*3u,1,&rate)!=0) return SVT_AV2_ERROR_INVALID_STATE;
    }
    work->rdoq_rate_leaf=leaf;work->rdoq_rate_ready=1u;
    region_copy_buffer(work,work->prefix_recon,1);
    return SVT_AV2_ERROR_NONE;
}
#endif

#ifndef M6_RDOQ_WIDE
#define M6_RDOQ_WIDE 0
#endif
static int rdoq_leaf_supported(const RdWork* work, const SvtAv2BlockView* leaf) {
    return leaf->w == leaf->h || (leaf->w <= 16u && leaf->h <= 16u) ||
        (work->rect_rdoq && ((leaf->w == 32u && leaf->h == 16u) ||
                             (leaf->w == 16u && leaf->h == 32u)));
}

/* Choose scalar or optimized coefficients at each causal leaf. Replay the
 * complete winning SB through the ordinary encoder and entropy checks. */
static SvtAv2Error refine_rdoq_leaves(RdWork* work, size_t sb, uint16_t decision, RdBasicChoice* best) {
    size_t leaves;
    if (svt_av2_tile_rate_prefix_begin(work->basic_rate_prefix, work->rate, sb, decision, &leaves) ||
        leaves != work->leaf_count) return SVT_AV2_ERROR_INVALID_STATE;
    region_copy(work, 1);
    work->coeff_count = 0;
    work->rdoq_phase = 0u;
    uint64_t total_sse[3] = {0}, rate = 0;
    for (uint32_t leaf = 0; leaf < leaves; ++leaf) {
        SvtAv2BlockView blocks[3];
        SvtAv2Error error = plane_blocks(work, leaf, blocks);
        if (error != SVT_AV2_ERROR_NONE) return error;
        const size_t offset = work->coeff_count;
        uint64_t scalar_cost = UINT64_MAX, sse[3];
        work->ids[leaf] = work->best_ids[leaf];
        for (unsigned trial = 0; ; ++trial) {
            memset(sse, 0, sizeof(sse));
            work->coeff_count = offset;
            work->rdoq_target = trial == 1 ? RD_RDOQ_LEAF : UINT32_MAX;
            error = reconstruct_leaf(work, leaf, work->ids[leaf], blocks, sse);
            if (error != SVT_AV2_ERROR_NONE) return error;
            if (trial == 2 || !rdoq_leaf_supported(work, blocks)) break;
            uint64_t cost;
            if (svt_av2_tile_rate_prefix_trial(work->basic_rate_prefix, leaf, work->blocks + leaf * 3u, 0, &rate))
                return SVT_AV2_ERROR_INVALID_STATE;
            error = svt_av2_rd_cost(svt_av2_rd_distortion(sse, work->source->bit_depth,
                work->source->chroma_format), work->stats.lambda_q8, rate, &cost);
            if (error != SVT_AV2_ERROR_NONE) return error;
            if (!trial) scalar_cost = cost;
            else {
                ++work->stats.coeff_trials;
                if (cost < scalar_cost) {
                    memset(work->rdoq_saved + leaf * 3u, 1, 3u);
                    ++work->stats.coeff_accepts;
                    break;
                }
            }
        }
        if (svt_av2_tile_rate_prefix_trial(work->basic_rate_prefix, leaf, work->blocks + leaf * 3u, 1, &rate))
            return SVT_AV2_ERROR_INVALID_STATE;
        for (unsigned p = 0; p < 3; ++p) total_sse[p] += sse[p];
    }
    work->rdoq_phase = 0;
    work->rdoq_target = UINT32_MAX;
    if (svt_av2_tile_rate_prefix_finish(work->basic_rate_prefix, &rate)) return SVT_AV2_ERROR_INVALID_STATE;
    uint64_t cost;
    const uint64_t d = svt_av2_rd_distortion(total_sse, work->source->bit_depth, work->source->chroma_format);
    SvtAv2Error error = svt_av2_rd_cost(d, work->stats.lambda_q8, rate, &cost);
    if (error != SVT_AV2_ERROR_NONE) return error;
    const uint64_t id = plan_candidate_id(decision, 1u << 26u, 1u << 26u);
    ++work->stats.candidate_count;
    if (work->config->candidate_hook)
        work->config->candidate_hook(work->config->candidate_hook_user, sb, id, d, rate, cost, 0);
    if (cost < best->cost) *best = (RdBasicChoice){cost, d, rate, path_hash(work), id};
    else memset(work->rdoq_saved, 0, sizeof(work->rdoq_saved));
    region_copy(work, 1);
    return SVT_AV2_ERROR_NONE;
}

static SvtAv2Error refine_rdoq_single(RdWork* work, size_t sb, int partial,
    uint16_t best_decision, RdBasicChoice* best) {
    SvtAv2Error error;
    work->rdoq_phase=0u;
    uint64_t best_cost=best->cost, best_d=best->distortion, best_r=best->rate, best_hash=best->hash;
    uint64_t best_id=best->id;
    if (work->config->coeff_optimization == 2u) {
        error = prepare_prefix(work, 0u);
        if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
        work->prefix_active = 1u;
        /* One proposal per leaf/plane. Previously accepted coefficient arrays
         * are replayed; later leaves see the candidate's real reconstruction.
         * No approximate coefficient cost can directly become a winner. */
        for (uint32_t block = 0u; block < work->leaf_count * 3u; ++block) {
            const SvtAv2BlockView* leaf = &work->leaves[block / 3u];
            if (!rdoq_leaf_supported(work, leaf)) continue;
            work->rdoq_target = UINT32_MAX;
            error = extend_prefix(work, block / 3u);
            if (error != SVT_AV2_ERROR_NONE) {
                work->prefix_active = 0u; region_copy(work, 1); return error;
            }
            work->rdoq_target = block;
            region_copy_buffer(work, work->prefix_recon, 1);
            uint64_t d = 0u, r = 0u, cost = 0u; uint32_t path; int valid;
            error = build_path(work, 0u, 0, partial, work->best_ids, &valid, &d, &path);
            if (error == SVT_AV2_ERROR_NONE && (!valid ||
                RD_RATE(work, sb, best_decision, work->blocks,
                    work->leaf_count * 3u, NULL, 0u, 0, &r) != 0)) error = SVT_AV2_ERROR_INVALID_STATE;
            if (error == SVT_AV2_ERROR_NONE) error = svt_av2_rd_cost(d, work->stats.lambda_q8, r, &cost);
            if (error != SVT_AV2_ERROR_NONE) { work->prefix_active = 0u; region_copy(work, 1); return error; }
            const uint32_t payload = (1u << 26u) | block;
            const uint64_t id = plan_candidate_id(best_decision,payload,payload);
            ++work->stats.candidate_count; ++work->stats.coeff_trials;
            if (work->config->candidate_hook != NULL)
                work->config->candidate_hook(work->config->candidate_hook_user, sb, id, d, r, cost, 0);
            if (cost < best_cost) {
                best_cost = cost; best_d = d; best_r = r; best_id = id;
                best_hash = path_hash(work); work->rdoq_saved[block] = 1u;
                ++work->stats.coeff_accepts;
            }
        }
        work->rdoq_target = UINT32_MAX;
        work->prefix_active = 0u; region_copy(work, 1);
    }
    best->cost=best_cost; best->distortion=best_d; best->rate=best_r;
    best->hash=best_hash; best->id=best_id;
    return SVT_AV2_ERROR_NONE;
}

#if M6_RDOQ_WIDE
static SvtAv2Error refine_rdoq_multi(RdWork* work, size_t sb, int partial,
    uint16_t best_decision, RdBasicChoice* best) {
    SvtAv2Error error;
    work->rdoq_phase=0u;
    uint64_t best_cost=best->cost, best_d=best->distortion, best_r=best->rate, best_hash=best->hash;
    uint64_t best_id=best->id;
    if (work->config->coeff_optimization == 2u) {
        error = prepare_prefix(work, 0u);
        if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
        work->prefix_active = 1u;
        /* Bounded strengths start from the same scalar coefficients. Every
         * proposal rebuilds the suffix and rates the complete actual SB.
         * No approximate coefficient cost can directly become a winner. */
        for (uint32_t block = 0u; block < work->leaf_count * 3u; ++block) {
            const SvtAv2BlockView* leaf=&work->leaves[block/3u];
            if (!rdoq_leaf_supported(work, leaf)) continue;
            work->rdoq_target = UINT32_MAX;
            error = extend_prefix(work, block / 3u);
            if (error != SVT_AV2_ERROR_NONE) {
                work->prefix_active = 0u; region_copy(work, 1); return error;
            }
#if M6_RDOQ_CONTEXT
            error=prepare_rdoq_rate_context(work,sb,best_decision,block/3u);
            if (error!=SVT_AV2_ERROR_NONE) {
                work->prefix_active=0u;region_copy(work,1);return error;
            }
#endif
            int32_t chosen_coeff[1024];
            size_t chosen_offset=0u,chosen_count=0u;
            uint32_t chosen_phase=3u;uint8_t chosen_type=0u;
            for (uint32_t index=0u;index<3u;++index) {
                const uint32_t phase=work->config->reverse_candidates?2u-index:index;
                work->rdoq_phase=phase;
                work->rdoq_target = block;
                region_copy_buffer(work, work->prefix_recon, 1);
                uint64_t d = 0u, r = 0u, cost = 0u; uint32_t path; int valid;
                error = build_path(work, 0u, 0, partial, work->best_ids, &valid, &d, &path);
                if (error == SVT_AV2_ERROR_NONE && (!valid ||
                    RD_RATE(work, sb, best_decision, work->blocks,
                        work->leaf_count * 3u, NULL, 0u, 0, &r) != 0)) error = SVT_AV2_ERROR_INVALID_STATE;
                if (error == SVT_AV2_ERROR_NONE) error = svt_av2_rd_cost(d, work->stats.lambda_q8, r, &cost);
                if (error != SVT_AV2_ERROR_NONE) { work->prefix_active = 0u; region_copy(work, 1); return error; }
                const uint32_t payload = (1u << (phase ? 29u : 26u)) | (phase << 8u) | block;
                const uint64_t id = plan_candidate_id(best_decision,payload,payload);
                ++work->stats.candidate_count; ++work->stats.coeff_trials;
                if (work->config->candidate_hook != NULL)
                    work->config->candidate_hook(work->config->candidate_hook_user, sb, id, d, r, cost, 0);
                if (cost < best_cost || (cost == best_cost && chosen_phase < 3u && phase < chosen_phase)) {
                    best_cost = cost; best_d = d; best_r = r; best_id = id;
                    best_hash = path_hash(work); work->rdoq_saved[block] = 1u;
                    const SvtAv2TileCoeffBlock* entry=work->blocks+block;
                    chosen_offset=(size_t)(entry->qcoeff-work->coeffs);
                    chosen_count=(size_t)SVT_AV2_MIN(entry->width,32u)*SVT_AV2_MIN(entry->height,32u);
                    memcpy(chosen_coeff,entry->qcoeff,chosen_count*sizeof(*chosen_coeff));
                    chosen_type=work->rdoq_tx_types[block];chosen_phase=phase;
                    ++work->stats.coeff_accepts;
                }
            }
            if (chosen_phase < 3u) {
                memcpy(work->rdoq_coeff+chosen_offset,chosen_coeff,chosen_count*sizeof(*chosen_coeff));
                work->rdoq_tx_types[block]=chosen_type;
            }
        }
        work->rdoq_rate_active=0u;
        work->rdoq_target = UINT32_MAX;
        work->rdoq_phase = 0u;
        work->prefix_active = 0u; region_copy(work, 1);
    }
    best->cost=best_cost; best->distortion=best_d; best->rate=best_r;
    best->hash=best_hash; best->id=best_id;
    return SVT_AV2_ERROR_NONE;
}

#endif
static SvtAv2Error refine_rdoq_fast(RdWork* work,size_t sb,int partial,
    uint16_t best_decision,RdBasicChoice* best) {
#if M6_RDOQ_WIDE
    if (work->config->coeff_optimization!=2u)
        return refine_rdoq_single(work,sb,partial,best_decision,best);
    RdWork* states=(RdWork*)malloc(2u*sizeof(*states));
    if (states==NULL) return SVT_AV2_ERROR_OUT_OF_MEMORY;
    states[0]=*work;
    const RdBasicChoice entry=*best;
    RdBasicChoice winner={UINT64_MAX,0u,0u,0u,UINT64_MAX};
    uint32_t winner_route=0u;
    SvtAv2RdFrameStats aggregate=work->stats;
    SvtAv2Error error=SVT_AV2_ERROR_NONE;
    const uint32_t routes=M6_RDOQ_CONTEXT?3u:2u;
    for (uint32_t route=0u;route<routes;++route) {
        *work=states[0];work->stats=aggregate;
        work->rdoq_use_context=route==2u;work->rdoq_rate_active=0u;work->rdoq_rate_ready=0u;
        region_copy(work,1);
        RdBasicChoice trial=entry;
#if M6_TX_CACHE
        if (work->transform_cache) ++work->transform_cache->rdoq_route_trials[route];
#endif
        error=route?refine_rdoq_multi(work,sb,partial,best_decision,&trial):
            refine_rdoq_single(work,sb,partial,best_decision,&trial);
        aggregate=work->stats;
        if (error!=SVT_AV2_ERROR_NONE) break;
        if (trial.cost<winner.cost) {winner=trial;winner_route=route;states[1]=*work;}
    }
    *work=states[error==SVT_AV2_ERROR_NONE?1u:0u];work->stats=aggregate;
    work->rdoq_rate_active=0u;work->rdoq_rate_ready=0u;work->rdoq_use_context=0u;
#if M6_TX_CACHE
    if (error==SVT_AV2_ERROR_NONE && work->transform_cache)
        ++work->transform_cache->rdoq_route_wins[winner_route];
#else
    (void)winner_route;
#endif
    region_copy(work,1);free(states);
    if (error==SVT_AV2_ERROR_NONE) *best=winner;
    return error;
#else
    return refine_rdoq_single(work,sb,partial,best_decision,best);
#endif
}

typedef struct RdLeafTools {
    uint8_t y,uv,mrl,tx,explicit_cfl,signs,idx,mh,effective_tx;
} RdLeafTools;

static RdLeafTools leaf_tools_get(const RdWork* work,uint32_t leaf) {
    const RdLeafTools tuple={work->best_ids[leaf],work->uv_ids[leaf],work->mrl[leaf],
        work->tx_types[leaf],work->cfl_explicit[leaf],work->cfl_alpha_signs[leaf],
        work->cfl_alpha_idx[leaf],work->mhccp[leaf],work->blocks[leaf*3u].tx_type};
    return tuple;
}
static void leaf_tools_set(RdWork* work,uint32_t leaf,const RdLeafTools* tuple) {
    work->best_ids[leaf]=tuple->y;work->uv_ids[leaf]=tuple->uv;
    work->mrl[leaf]=tuple->mrl;work->tx_types[leaf]=tuple->tx;
    work->cfl_explicit[leaf]=tuple->explicit_cfl;work->cfl_alpha_signs[leaf]=tuple->signs;
    work->cfl_alpha_idx[leaf]=tuple->idx;work->mhccp[leaf]=tuple->mh;
    work->blocks[leaf*3u].tx_type=tuple->effective_tx;
}
static uint64_t leaf_tools_key(const RdLeafTools* t) {
    return ((uint64_t)t->y<<56u)|((uint64_t)t->uv<<48u)|((uint64_t)t->mrl<<40u)|
        ((uint64_t)t->tx<<32u)|((uint64_t)t->explicit_cfl<<24u)|
        ((uint64_t)t->signs<<16u)|((uint64_t)t->idx<<8u)|t->mh;
}
/* Partial-prefix scores never masquerade as complete-SB candidate hooks. */
static SvtAv2Error linear_leaf_trial(RdWork* work,uint32_t leaf,RdLeafTools proposal,
    RdLeafTools* best,uint64_t* cost,uint64_t entering_cost,int* accepted) {
    leaf_tools_set(work,leaf,&proposal);
    region_copy_buffer(work,work->prefix_recon,1);
    uint64_t score=0u;
#if M6_Y_BOUND && M6_TX_CACHE
    uint64_t lower=0u,y_rate=0u;
    int bound_active=0;
#if M6_Y_BOUND_VERIFY
    uint8_t y_probe[64u*64u*2u];
#endif
    /* This stage precedes RDOQ. The exact production Y reconstruction is the
     * same in the one-plane and full path. Omit only nonnegative UV error/rate.
     * Strict > preserves all possible cost ties and the existing tuple order. */
    if (work->transform_cache && work->basic_rate_active && !work->memo_active &&
        !work->uv_y_cache && work->rdoq_target==UINT32_MAX &&
        !work->rdoq_saved[leaf*3u]) {
        SvtAv2BlockView blocks[3];
        SvtAv2Error pre=plane_blocks(work,leaf,blocks);
        uint64_t y_sse[3]={0u,0u,0u};
        work->coeff_count=work->prefix_coeff_count;work->ids[leaf]=proposal.y;
        if (pre==SVT_AV2_ERROR_NONE)
            pre=reconstruct_leaf_planes(work,leaf,proposal.y,blocks,y_sse,1u);
        if (pre!=SVT_AV2_ERROR_NONE) { *accepted=0;leaf_tools_set(work,leaf,best);return pre; }
        uint64_t partial_sse[3]={work->prefix_sse[0]+y_sse[0],work->prefix_sse[1],work->prefix_sse[2]};
        if (svt_av2_tile_rate_prefix_trial_y(work->basic_rate_prefix,leaf,
            work->blocks+leaf*3u,&y_rate)!=0)
            return SVT_AV2_ERROR_INVALID_STATE;
        pre=svt_av2_rd_cost(svt_av2_rd_distortion(partial_sse,work->source->bit_depth,
            work->source->chroma_format),work->stats.lambda_q8,y_rate,&lower);
        if (pre!=SVT_AV2_ERROR_NONE) return pre;
        ++work->transform_cache->bound_trials;bound_active=1;
#if M6_Y_BOUND_VERIFY
        const SvtAv2BlockView* yb=&blocks[0];
        const uint32_t bytes=work->recon->bit_depth>8u?2u:1u;
        for (uint32_t y=0u;y<yb->h;++y)
            memcpy(y_probe+(size_t)y*yb->w*bytes,work->recon->planes[0]+
                (size_t)(yb->y+y)*work->recon->strides[0]+yb->x*bytes,(size_t)yb->w*bytes);
#else
        if (lower>*cost) {
            ++work->transform_cache->bound_pruned;
            *accepted=0;leaf_tools_set(work,leaf,best);
            region_copy_buffer(work,work->prefix_recon,1);
            work->coeff_count=work->prefix_coeff_count;
            return SVT_AV2_ERROR_NONE;
        }
#endif
        region_copy_buffer(work,work->prefix_recon,1);
    }
#endif
    const SvtAv2Error error=local_basic_cost(work,leaf,proposal.y,&score);
#if M6_Y_BOUND && M6_TX_CACHE
    if (error==SVT_AV2_ERROR_NONE && bound_active && lower>score)
        return SVT_AV2_ERROR_INVALID_STATE;
#if M6_Y_BOUND_VERIFY
    if (error==SVT_AV2_ERROR_NONE && bound_active) {
        uint64_t full_rate=0u;
        if (svt_av2_tile_rate_prefix_trial(work->basic_rate_prefix,leaf,
            work->blocks+leaf*3u,0,&full_rate)!=0 || y_rate>full_rate)
            return SVT_AV2_ERROR_INVALID_STATE;
        const SvtAv2BlockView* yb=&work->leaves[leaf];
        const uint32_t bytes=work->recon->bit_depth>8u?2u:1u;
        for (uint32_t y=0u;y<yb->h;++y)
            if (memcmp(y_probe+(size_t)y*yb->w*bytes,work->recon->planes[0]+
                (size_t)(yb->y+y)*work->recon->strides[0]+yb->x*bytes,(size_t)yb->w*bytes)!=0)
                return SVT_AV2_ERROR_INVALID_STATE;
    }
#endif
#endif
    *accepted=0;
    if (error==SVT_AV2_ERROR_NONE && (score<*cost ||
        (score==*cost && score<entering_cost && leaf_tools_key(&proposal)<leaf_tools_key(best)))) {
        *best=leaf_tools_get(work,leaf);*cost=score;*accepted=1;
    }
    leaf_tools_set(work,leaf,best);
    return error;
}

#if SVT_AV2_EXPERIMENTAL_STILL_UV_RD
/* Re-evaluate a compact set of ordinary UV modes against implicit CfL on the
 * finalized Y/angle/MRL/TX tuple. Earlier basic UV competed on another Y.
 * SAD only shortlists ordinary modes; every admitted arm uses joint YUV Q12. */
static SvtAv2Error refine_late_uv(RdWork* work,uint32_t leaf,RdLeafTools* best,uint64_t* cost) {
    const RdLeafTools entering=*best;
    const uint64_t anchor=*cost;
    SvtAv2BlockView blocks[3];
    SvtAv2Error error=plane_blocks(work,leaf,blocks);
    uint8_t ranked[2]={255u,255u};
    uint64_t scores[2]={UINT64_MAX,UINT64_MAX};
    region_copy_buffer(work,work->prefix_recon,1);
    for (uint32_t uv=0u;uv<8u && error==SVT_AV2_ERROR_NONE;++uv) {
        if (uv==3u || !legal_mode(work,leaf,uv)) continue;
        work->uv_ids[leaf]=(uint8_t)uv;
        uint64_t sad[3]={0u,0u,0u};
        error=predict(work,leaf,entering.y,blocks,sad,2);
        const uint64_t s=sad[1]+sad[2];
        for (uint32_t p=0u;p<2u;++p) if (s<scores[p] || (s==scores[p] && uv<ranked[p])) {
            if (p==0u) {scores[1]=scores[0];ranked[1]=ranked[0];}
            scores[p]=s;ranked[p]=(uint8_t)uv;break;
        }
    }
    leaf_tools_set(work,leaf,&entering);
    if (error!=SVT_AV2_ERROR_NONE) return error;
    uint32_t mask=(1u<<14u)|1u;
    for (uint32_t i=0u;i<2u;++i) if (ranked[i]!=255u) mask|=1u<<ranked[i];
    if (entering.uv!=14u || (!entering.explicit_cfl && !entering.mh)) mask&=~(1u<<entering.uv);
    const uint64_t prefix_rate=svt_av2_tile_rate_prefix_current_rate(work->basic_rate_prefix);
    work->uv_y_cache=1u;work->uv_y_size=0u;
    work->coeff_count=work->prefix_coeff_count;work->ids[leaf]=entering.y;
    uint64_t y_sse[3]={0u,0u,0u};
    error=reconstruct_leaf_planes(work,leaf,entering.y,blocks,y_sse,1u);
    work->uv_y_cache=2u;
    for (uint32_t i=0u;i<15u && error==SVT_AV2_ERROR_NONE;++i) {
        const uint32_t m=work->config->reverse_candidates?14u-i:i;
        if (!(mask&(1u<<m))) continue;
        RdLeafTools proposal=entering;
        proposal.uv=(uint8_t)m;
        proposal.explicit_cfl=proposal.mh=proposal.signs=proposal.idx=0u;
        int accepted=0;
        error=linear_leaf_trial(work,leaf,proposal,best,cost,anchor,&accepted);
        ++work->late_uv_trials;
    }
    work->uv_y_cache=0u;work->uv_y_size=0u;
    if (error!=SVT_AV2_ERROR_NONE) return error;
    if (svt_av2_tile_rate_prefix_current_rate(work->basic_rate_prefix)!=prefix_rate)
        return SVT_AV2_ERROR_INVALID_STATE;
    if (*cost<anchor) {
        uint64_t cached=0u,independent=0u,replay=0u;
        /* The last candidate need not be the winner. Rebuild it once from
         * the cached Y, then independently via the uncached production path. */
        leaf_tools_set(work,leaf,best);
        region_copy_buffer(work,work->prefix_recon,1);
        work->uv_y_cache=2u;
        error=local_basic_cost(work,leaf,best->y,&replay);
        work->uv_y_cache=0u;
        if (error!=SVT_AV2_ERROR_NONE) return error;
        if (replay!=*cost || svt_av2_tile_rate_prefix_trial(work->basic_rate_prefix,leaf,
                work->blocks+leaf*3u,0,&cached)!=0) return SVT_AV2_ERROR_INVALID_STATE;
        region_copy_buffer(work,work->prefix_recon,1);
        error=local_basic_cost(work,leaf,best->y,&replay);
        if (error!=SVT_AV2_ERROR_NONE) return error;
        if (replay!=*cost || svt_av2_tile_rate_prefix_trial(work->basic_rate_prefix,leaf,
                work->blocks+leaf*3u,0,&independent)!=0 || independent!=cached)
            return SVT_AV2_ERROR_INVALID_STATE;
        *best=leaf_tools_get(work,leaf);
        ++work->late_uv_wins;++work->late_uv_verified;
        work->late_uv_implicit+=best->uv==14u;
    }
    return SVT_AV2_ERROR_NONE;
}
#endif

#if SVT_AV2_EXPERIMENTAL_CFL_PAIR_RD || SVT_AV2_EXPERIMENTAL_VIDEO_CFL
static void cfl_set_explicit_pair(RdLeafTools* t,int32_t u,int32_t v) {
    const uint32_t us=u<0?1u:u>0?2u:0u, vs=v<0?1u:v>0?2u:0u;
    const uint32_t ui=u?(uint32_t)(u<0?-u:u)-1u:0u, vi=v?(uint32_t)(v<0?-v:v)-1u:0u;
    t->uv=14u;t->explicit_cfl=1u;t->mh=0u;
    t->signs=(uint8_t)(us*3u+vs-1u);t->idx=(uint8_t)((ui<<3u)|vi);
}
#endif

#if SVT_AV2_EXPERIMENTAL_CFL_PAIR_RD
/* The predictor SSE provides a cheap seed, not an alpha decision. Keep both
 * legacy seeds, zero, weak signed values and the nearest options to the best
 * predictor. Quantize each plane/alpha once, then price JOINT pairs from the
 * live leaf checkpoint: V syntax depends on U's levels and adaptive CDFs. */
#define RD_CFL_PLANE_OFFERS 7u
typedef struct RdCflPlaneOffer {
    int8_t alpha;
    uint64_t sse;
    int32_t qcoeff[1024];
} RdCflPlaneOffer;

static uint32_t cfl_offer_list(const int8_t seed[2], RdCflPlaneOffer out[RD_CFL_PLANE_OFFERS]) {
    const int values[RD_CFL_PLANE_OFFERS]={seed[0],seed[1],0,-1,1,seed[0]-1,seed[0]+1};
    uint32_t n=0u;
    for (uint32_t i=0u;i<RD_CFL_PLANE_OFFERS;++i) {
        if (values[i]<-8 || values[i]>8) continue;
        uint32_t j=0u;
        while (j<n && out[j].alpha!=values[i]) ++j;
        if (j==n) out[n++].alpha=(int8_t)values[i];
    }
    return n;
}

/* Does not write candidate pixels to the causal reconstruction. Production
 * inverse and sample clipping are used when measuring the candidate SSE. */
static SvtAv2Error cfl_quantized_offer(RdWork* work,const SvtAv2BlockView* b,
    uint32_t plane,RdCflPlaneOffer* offer) {
    const uint32_t bd=work->source->bit_depth, step=bd>8u?2u:1u;
    const uint32_t stride=work->scratch->stride;
    const SvtAv2IntraBlock intra={b->x,b->y,b->w,b->h,plane};
    uint8_t pred[64u*64u*2u];
    SvtAv2Error error=svt_av2_intra_cfl_explicit_predict(work->recon,&intra,64u,pred,b->w*step,offer->alpha);
    if (error!=SVT_AV2_ERROR_NONE) return error;
    for (uint32_t y=0u;y<b->h;++y) {
        const uint8_t* src=work->source->planes[plane]+(size_t)(b->y+y)*work->source->strides[plane]+b->x*step;
        for (uint32_t x=0u;x<b->w;++x)
            work->scratch->residual[(size_t)y*stride+x]=(int16_t)((int)svt_av2_sample_load(src+x*step,bd)-
                (int)svt_av2_sample_load(pred+((size_t)y*b->w+x)*step,bd));
    }
    int all_zero=0;
    error=forward_trial(work,plane,b,0u,&all_zero);
    if (error!=SVT_AV2_ERROR_NONE) return error;
    const size_t count=(size_t)SVT_AV2_MIN(b->w,32u)*SVT_AV2_MIN(b->h,32u);
    memcpy(offer->qcoeff,work->scratch->qcoeff,count*sizeof(*offer->qcoeff));
    error=work->config->transform.inverse(work->config->transform.user,plane,b->w,b->h,
        work->scratch->dqcoeff,stride,work->scratch->inverse_residual);
    if (error!=SVT_AV2_ERROR_NONE) return error;
    const int max=(1<<(int)bd)-1;
    offer->sse=0u;
    for (uint32_t y=0u;y<b->h;++y) {
        const uint8_t* src=work->source->planes[plane]+(size_t)(b->y+y)*work->source->strides[plane]+b->x*step;
        for (uint32_t x=0u;x<b->w;++x) {
            const int value=(int)svt_av2_sample_load(pred+((size_t)y*b->w+x)*step,bd)+
                work->scratch->inverse_residual[(size_t)y*stride+x];
            const int d=(int)svt_av2_sample_load(src+x*step,bd)-SVT_AV2_MAX(0,SVT_AV2_MIN(value,max));
            offer->sse+=(uint64_t)(d*d);
        }
    }
    return SVT_AV2_ERROR_NONE;
}

static SvtAv2Error refine_cfl_pair_rd(RdWork* work,uint32_t leaf,const SvtAv2BlockView blocks[3],
    int8_t seed[2][2],RdLeafTools* best,uint64_t* cost) {
    RdCflPlaneOffer u[RD_CFL_PLANE_OFFERS],v[RD_CFL_PLANE_OFFERS];
    const uint32_t nu=cfl_offer_list(seed[0],u),nv=cfl_offer_list(seed[1],v);
    SvtAv2TileCoeffBlock trial[3];
    memcpy(trial,work->blocks+leaf*3u,sizeof(trial));
    uint64_t sse[3];memcpy(sse,work->prefix_sse,sizeof(sse));
    const SvtAv2BlockView* yb=blocks;
    const uint32_t bd=work->source->bit_depth,step=bd>8u?2u:1u;
    for (uint32_t y=0u;y<yb->h;++y) {
        const uint8_t* src=work->source->planes[0]+(size_t)(yb->y+y)*work->source->strides[0]+yb->x*step;
        const uint8_t* rec=work->recon->planes[0]+(size_t)(yb->y+y)*work->recon->strides[0]+yb->x*step;
        for (uint32_t x=0u;x<yb->w;++x) {
            const int d=(int)svt_av2_sample_load(src+x*step,bd)-(int)svt_av2_sample_load(rec+x*step,bd);
            sse[0]+=(uint64_t)(d*d);
        }
    }
    if (work->detail_rd) sse[0] += detail_error(work, yb);
    for (uint32_t p=1u;p<3u;++p) for (uint32_t i=0u;i<(p==1u?nu:nv);++i) {
        const SvtAv2Error error=cfl_quantized_offer(work,blocks+p,p,(p==1u?u:v)+i);
        if (error!=SVT_AV2_ERROR_NONE) return error;
    }
    const uint64_t start=*cost;
    uint64_t winning_rate=0u;
    unsigned one_zero=0u;
    const uint64_t prefix=svt_av2_tile_rate_prefix_current_rate(work->basic_rate_prefix);
    ++work->cfl_pair_leaves;
    trial[0].mhccp=0u;trial[0].cfl_explicit=1u;
    trial[1].intra_mode=trial[2].intra_mode=14u;
    for (uint32_t i=0u;i<nu*nv;++i) {
        const uint32_t idx=work->config->reverse_candidates?nu*nv-1u-i:i;
        const RdCflPlaneOffer* a=u+idx/nv;
        const RdCflPlaneOffer* b=v+idx%nv;
        if (a->alpha==0 && b->alpha==0) continue;
        RdLeafTools proposal=*best;
        cfl_set_explicit_pair(&proposal,a->alpha,b->alpha);
        trial[0].cfl_alpha_signs=proposal.signs;trial[0].cfl_alpha_idx=proposal.idx;
        trial[1].qcoeff=a->qcoeff;trial[2].qcoeff=b->qcoeff;
        sse[1]=work->prefix_sse[1]+a->sse;sse[2]=work->prefix_sse[2]+b->sse;
        uint64_t rate=0u,j=0u;
        if (svt_av2_tile_rate_prefix_trial(work->basic_rate_prefix,leaf,trial,0,&rate)!=0 || rate<prefix)
            return SVT_AV2_ERROR_INVALID_STATE;
        SvtAv2Error error=svt_av2_rd_cost(svt_av2_rd_distortion(sse,bd,work->source->chroma_format),
            work->stats.lambda_q8,rate,&j);
        if (error!=SVT_AV2_ERROR_NONE) return error;
        ++work->cfl_pair_trials;++work->stats.cfl_trials;
        if (j<*cost || (j==*cost && j<start && leaf_tools_key(&proposal)<leaf_tools_key(best))) {
            *best=proposal;*cost=j;winning_rate=rate;
            one_zero=a->alpha==0 || b->alpha==0;
        }
    }
    if (svt_av2_tile_rate_prefix_current_rate(work->basic_rate_prefix)!=prefix)
        return SVT_AV2_ERROR_INVALID_STATE;
    if (*cost<start) {
        uint64_t replay_cost=0u,replay_rate=0u;
        leaf_tools_set(work,leaf,best);
        region_copy_buffer(work,work->prefix_recon,1);
        const SvtAv2Error error=local_basic_cost(work,leaf,best->y,&replay_cost);
        if (error!=SVT_AV2_ERROR_NONE) return error;
        if (replay_cost!=*cost || svt_av2_tile_rate_prefix_trial(work->basic_rate_prefix,leaf,
                work->blocks+leaf*3u,0,&replay_rate)!=0 || replay_rate!=winning_rate)
            return SVT_AV2_ERROR_INVALID_STATE;
        *best=leaf_tools_get(work,leaf);
        ++work->cfl_pair_verified;++work->cfl_pair_wins;++work->stats.cfl_accepts;
        work->cfl_pair_one_zero+=one_zero;
    }
    /* These arms may have lost earlier on the seed luma, before Y/angle/TX
     * refinement. Compete again from the same prefix with final candidate Y. */
    for (uint32_t arm=0u;arm<2u;++arm) {
        RdLeafTools proposal=*best;
        proposal.uv=arm?14u:0u;
        proposal.explicit_cfl=proposal.mh=proposal.signs=proposal.idx=0u;
        if (leaf_tools_key(&proposal)==leaf_tools_key(best)) continue;
        int accepted=0;
        const SvtAv2Error error=linear_leaf_trial(work,leaf,proposal,best,cost,start,&accepted);
        if (error!=SVT_AV2_ERROR_NONE) return error;
    }
    return SVT_AV2_ERROR_NONE;
}
#endif

#if M6_TOOL_REPEAT
/* A changed direction or MRL may not support the inherited primary TX.
 * Keep every legal inherited pair and independently offer the default TX.
 * Reject known illegal syntax before touching candidate state. */
static SvtAv2Error repeat_pair_trial(RdWork* work,uint32_t leaf,RdLeafTools proposal,
    RdLeafTools* best,uint64_t* cost,uint64_t anchor,int mrl_trial) {
    const uint32_t w=work->leaves[leaf].w,h=work->leaves[leaf].h;
    const uint8_t inherited=proposal.tx;
    for (uint32_t arm=0u;arm<2u;++arm) {
        if (arm && !inherited) continue;
        proposal.tx=arm?0u:inherited;proposal.effective_tx=0u;
        if (proposal.tx && svt_av2_intra_tx_symbol(
            svt_av2_intra_tx_mode(proposal.y,w,h,proposal.mrl),w,h,proposal.tx)>=7u) {
#if M6_TX_CACHE
            if (work->transform_cache) ++work->transform_cache->repeat_pair_illegal;
#endif
            continue;
        }
        int accepted=0;
        const SvtAv2Error error=linear_leaf_trial(work,leaf,proposal,best,cost,anchor,&accepted);
        if (error!=SVT_AV2_ERROR_NONE) return error;
        if (mrl_trial) {++work->stats.mrl_trials;work->stats.mrl_accepts+=(uint64_t)accepted;}
        else {++work->stats.angle_trials;work->stats.angle_accepts+=(uint64_t)accepted;}
#if M6_TX_CACHE
        if (work->transform_cache) {
            ++work->transform_cache->repeat_pair_trials;
            work->transform_cache->repeat_pair_default+=(uint64_t)arm;
        }
#endif
    }
    return SVT_AV2_ERROR_NONE;
}
#endif

static SvtAv2Error refine_tools_linear(RdWork* work,size_t sb,int partial,
    uint16_t decision,RdBasicChoice* best) {
    RdLeafTools entering[RD_MAX_LEAVES];
    for (uint32_t leaf=0u;leaf<work->leaf_count;++leaf) entering[leaf]=leaf_tools_get(work,leaf);
    size_t leaves=0u;
    SvtAv2Error error=SVT_AV2_ERROR_NONE;
    if (svt_av2_tile_rate_prefix_begin(work->basic_rate_prefix,work->rate,sb,decision,&leaves)!=0 ||
        leaves!=work->leaf_count) return SVT_AV2_ERROR_INVALID_STATE;
    work->basic_rate_active=1u;
    error=prepare_prefix(work,0u);
    if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
    work->prefix_active=1u;
    for (uint32_t leaf=0u;leaf<work->leaf_count;++leaf) {
        error=extend_prefix(work,leaf);
        if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
        uint64_t previous=0u;
        if (leaf && svt_av2_tile_rate_prefix_trial(work->basic_rate_prefix,leaf-1u,
            work->blocks+(leaf-1u)*3u,1,&previous)!=0) {error=SVT_AV2_ERROR_INVALID_STATE;goto cleanup;}
        RdLeafTools selected=leaf_tools_get(work,leaf);
        region_copy_buffer(work,work->prefix_recon,1);
        uint64_t cost=0u;
        error=local_basic_cost(work,leaf,selected.y,&cost);
        if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
        selected=leaf_tools_get(work,leaf);
        const uint32_t rounds=work->tool_rounds?work->tool_rounds:1u;
#if M6_TOOL_REPEAT && M6_TX_CACHE
        int first_joint_changed=0;
#endif
        for (uint32_t tool_round=0u;tool_round<rounds;++tool_round) {
            if (tool_round) {
                work->uv_y_cache=0u;work->uv_y_size=0u;
                leaf_tools_set(work,leaf,&selected);
                region_copy_buffer(work,work->prefix_recon,1);
                error=local_basic_cost(work,leaf,selected.y,&cost);
                if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
                selected=leaf_tools_get(work,leaf);
            }
#if M6_TOOL_REPEAT && M6_TX_CACHE
            const uint64_t round_start=leaf_tools_key(&selected);
#endif
        if (work->config->angle_search && legal_mode(work,leaf,14u)) {
            SvtAv2BlockView blocks[3];
            error=plane_blocks(work,leaf,blocks);
            if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
            uint8_t ranked[4]={255u,255u,255u,255u};
            uint64_t scores[4]={UINT64_MAX,UINT64_MAX,UINT64_MAX,UINT64_MAX};
#if M6_SATD_ANGLES
            uint8_t satd_ranked[4]={255u,255u,255u,255u};
            uint64_t satd_scores[4]={UINT64_MAX,UINT64_MAX,UINT64_MAX,UINT64_MAX};
#endif
            for (uint32_t mode=14u;mode<=61u;++mode) {
                uint64_t sad[3]={0u,0u,0u};
                error=predict(work,leaf,mode,blocks,sad,1);
                if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
#if M6_SATD_ANGLES
                angle_satd_rank(work,&blocks[0],mode,satd_ranked,satd_scores);
#endif
                for (uint32_t slot=0u;slot<4u;++slot) if (sad[0]<scores[slot] ||
                    (sad[0]==scores[slot] && mode<ranked[slot])) {
                    for (uint32_t j=3u;j>slot;--j) {scores[j]=scores[j-1u];ranked[j]=ranked[j-1u];}
                    scores[slot]=sad[0];ranked[slot]=(uint8_t)mode;break;
                }
            }
            uint8_t candidates[14];uint32_t count=4u;
            memcpy(candidates,ranked,sizeof(ranked));
            const uint32_t joint=svt_av2_intra_mode_joint(svt_av2_intra_mode_base(selected.y));
            if (joint>=5u) for (uint32_t slot=0u;slot<6u;++slot) {
                const uint8_t mode=(uint8_t)(14u+((joint-5u)/7u)*6u+slot);
                uint32_t i=0u;while (i<count && candidates[i]!=mode) ++i;
                if (i==count) candidates[count++]=mode;
            }
#if M6_SATD_ANGLES
            angle_satd_append(work,candidates,&count,satd_ranked);
#endif
            const uint64_t anchor=cost;
            for (uint32_t index=0u;index<count;++index) {
                RdLeafTools proposal=selected;
                proposal.y=candidates[work->config->reverse_candidates?count-1u-index:index];
#if M6_TOOL_REPEAT
                if (tool_round) {
                    error=repeat_pair_trial(work,leaf,proposal,&selected,&cost,anchor,0);
                    if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
                    continue;
                }
#endif
                int accepted=0;
                error=linear_leaf_trial(work,leaf,proposal,&selected,&cost,anchor,&accepted);
                if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
                ++work->stats.angle_trials;work->stats.angle_accepts+=(uint64_t)accepted;
            }
        }
        const uint32_t w=work->leaves[leaf].w,h=work->leaves[leaf].h;
        if (work->config->frame_header.enable_mrls && w>=8u && h>=8u &&
            (w==h || w==2u*h || h==2u*w || (w==16u && h==64u) || (w==64u && h==16u)) &&
            svt_av2_intra_mode_joint(selected.y)>=5u) {
            static const uint8_t candidates[6]={1u,2u,3u,5u,6u,7u};
            const uint64_t anchor=cost;
            for (uint32_t index=0u;index<6u;++index) {
                RdLeafTools proposal=selected;
                proposal.mrl=candidates[work->config->reverse_candidates?5u-index:index];
#if M6_TOOL_REPEAT
                if (tool_round) {
                    error=repeat_pair_trial(work,leaf,proposal,&selected,&cost,anchor,1);
                    if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
                    continue;
                }
#endif
                int accepted=0;
                error=linear_leaf_trial(work,leaf,proposal,&selected,&cost,anchor,&accepted);
                if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
                ++work->stats.mrl_trials;work->stats.mrl_accepts+=(uint64_t)accepted;
            }
        }
        if (work->config->transform_selection && ((w==h && w<=16u) ||
            (work->config->transform_selection==2u && ((w==8u && h==16u) || (w==16u && h==8u))))) {
            const uint64_t anchor=cost;
            const uint32_t count=work->config->transform_selection==2u?(M6_FLIP_TX?8u:3u):1u;
            for (uint32_t index=0u;index<count;++index) {
                RdLeafTools proposal=selected;
                proposal.tx=(uint8_t)(count==1u?3u:work->config->reverse_candidates?count-index:1u+index);
                if (proposal.tx==1u && w==4u && selected.y==7u) continue;
#if M6_FLIP_TX
                if (work->config->transform_selection==2u &&
                    svt_av2_intra_tx_symbol(svt_av2_intra_tx_mode(proposal.y,w,h,proposal.mrl),w,h,proposal.tx)>=7u) continue;
#endif
                int accepted=0;
                error=linear_leaf_trial(work,leaf,proposal,&selected,&cost,anchor,&accepted);
                if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
                ++work->stats.transform_trials;work->stats.transform_accepts+=(uint64_t)accepted;
            }
        }

#if M6_TOOL_REPEAT && M6_TX_CACHE
        const uint8_t before_joint_y=selected.y;
#endif
#if M6_JOINT_Y_TX && M6_FLIP_TX
        /* The DCT-selected prediction need not win after primary TX search.
         * Test every legal basic prediction/TX pair from one fixed entering
         * tuple and the same causal prefix; retain the original selected tuple.
         * Subsequent UV search and final complete-SB J gate are unchanged. */
        if (work->config->transform_selection==2u &&
            ((w==h && w<=16u) || (w==8u && h==16u) || (w==16u && h==8u))) {
            static const uint8_t modes[13]={0u,1u,2u,4u,5u,6u,7u,8u,9u,10u,11u,12u,13u};
            const RdLeafTools anchor_tuple=selected;
            const uint64_t anchor_cost=cost;
            const uint32_t pairs=work->config->screen_trees?9u:13u*9u;
            for (uint32_t index=0u;index<pairs;++index) {
                const uint32_t pair=work->config->reverse_candidates?pairs-1u-index:index;
                RdLeafTools proposal=anchor_tuple;
                proposal.y=work->config->screen_trees ? work->alternate_y[leaf] : modes[pair/9u];proposal.tx=(uint8_t)(pair%9u);
                proposal.mrl=0u;proposal.effective_tx=0u;
                if (!legal_mode(work,leaf,proposal.y) ||
                    svt_av2_intra_tx_symbol(svt_av2_intra_tx_mode(proposal.y,w,h,proposal.mrl),w,h,proposal.tx)>=7u ||
                    leaf_tools_key(&proposal)==leaf_tools_key(&anchor_tuple)) continue;
                int accepted=0;
                error=linear_leaf_trial(work,leaf,proposal,&selected,&cost,anchor_cost,&accepted);
                if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
                ++work->stats.transform_trials;work->stats.transform_accepts+=(uint64_t)accepted;
            }
        }
#endif
#if M6_TOOL_REPEAT && M6_TX_CACHE
        if (!tool_round && selected.y!=before_joint_y) first_joint_changed=1;
#endif
        if (work->config->cfl_search) {
            leaf_tools_set(work,leaf,&selected);
            region_copy_buffer(work,work->prefix_recon,1);
            error=local_basic_cost(work,leaf,selected.y,&cost);
            if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
            SvtAv2BlockView blocks[3];
            error=plane_blocks(work,leaf,blocks);
            int8_t alphas[2][2]={{0,0},{0,0}};
            if (error==SVT_AV2_ERROR_NONE) error=rank_cfl_alphas(work,blocks+1u,1u,alphas[0]);
            if (error==SVT_AV2_ERROR_NONE) error=rank_cfl_alphas(work,blocks+2u,2u,alphas[1]);
            if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
#if SVT_AV2_EXPERIMENTAL_CFL_PAIR_RD
            if (work->cfl_pair_rd && !partial && work->rdoq_target==UINT32_MAX && !work->coeff_policy &&
                !work->memo_active && !work->uv_y_cache && !work->rdoq_saved[leaf*3u+1u] &&
                !work->rdoq_saved[leaf*3u+2u] && work->source->chroma_format==SVT_AV2_CHROMA_420) {
                error=refine_cfl_pair_rd(work,leaf,blocks,alphas,&selected,&cost);
                if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
            } else
#endif
            {
            const uint64_t anchor=cost;
            for (uint32_t index=0u;index<4u;++index) {
                const uint32_t pair=work->config->reverse_candidates?3u-index:index;
                const int32_t u=alphas[0][pair/2u],v=alphas[1][pair%2u];
                if (!u && !v) continue;
                const uint32_t usign=u<0?1u:u>0?2u:0u,vsign=v<0?1u:v>0?2u:0u;
                const uint32_t ui=u?(uint32_t)(u<0?-u:u)-1u:0u,vi=v?(uint32_t)(v<0?-v:v)-1u:0u;
                RdLeafTools proposal=selected;
                proposal.uv=14u;proposal.explicit_cfl=1u;proposal.mh=0u;
                proposal.signs=(uint8_t)(usign*3u+vsign-1u);proposal.idx=(uint8_t)((ui<<3u)|vi);
                int accepted=0;
                error=linear_leaf_trial(work,leaf,proposal,&selected,&cost,anchor,&accepted);
                if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
                ++work->stats.cfl_trials;work->stats.cfl_accepts+=(uint64_t)accepted;
            }
            }
        }
        if (work->config->mhccp_search &&
            svt_av2_intra_mhccp_allowed(w,h,(uint32_t)work->source->chroma_format)) {
            const uint64_t anchor=cost;
            for (uint32_t index=0u;index<3u;++index) {
                RdLeafTools proposal=selected;
                proposal.uv=14u;proposal.explicit_cfl=0u;proposal.idx=proposal.signs=0u;
                proposal.mh=(uint8_t)(work->config->reverse_candidates?3u-index:1u+index);
                int accepted=0;
                error=linear_leaf_trial(work,leaf,proposal,&selected,&cost,anchor,&accepted);
                if (error!=SVT_AV2_ERROR_NONE) goto cleanup;
                ++work->stats.mhccp_trials;work->stats.mhccp_accepts+=(uint64_t)accepted;
            }
        }

            leaf_tools_set(work,leaf,&selected);
#if M6_TOOL_REPEAT && M6_TX_CACHE
            if (tool_round && work->transform_cache) {
                ++work->transform_cache->repeat_rounds;
                work->transform_cache->repeat_changes+=(uint64_t)(leaf_tools_key(&selected)!=round_start);
                work->transform_cache->repeat_after_joint+=(uint64_t)first_joint_changed;
            }
#endif
            if (rounds>1u) {work->uv_y_cache=0u;work->uv_y_size=0u;}
        }
#if SVT_AV2_EXPERIMENTAL_STILL_UV_RD
        if (work->still_uv_rd && !partial && work->config->frame_header.m5_intra_modes == 4u &&
            work->source->chroma_format == SVT_AV2_CHROMA_420) {
            error = refine_late_uv(work, leaf, &selected, &cost);
            if (error != SVT_AV2_ERROR_NONE) goto cleanup;
        }
#endif
        leaf_tools_set(work,leaf,&selected);
    }
    error=extend_prefix(work,work->leaf_count);
    if (error==SVT_AV2_ERROR_NONE) {
        uint64_t rate=0u,whole=0u,cost=0u;
        if (svt_av2_tile_rate_prefix_trial(work->basic_rate_prefix,work->leaf_count-1u,
            work->blocks+(work->leaf_count-1u)*3u,1,&rate)!=0 ||
            svt_av2_tile_rate_prefix_finish(work->basic_rate_prefix,&rate)!=0 ||
            RD_RATE(work,sb,decision,work->blocks,
                work->leaf_count*3u,NULL,0u,0,&whole)!=0 || rate!=whole)
            error=SVT_AV2_ERROR_INVALID_STATE;
        const uint64_t d=svt_av2_rd_distortion(work->prefix_sse,work->source->bit_depth,work->source->chroma_format);
        if (error==SVT_AV2_ERROR_NONE) error=svt_av2_rd_cost(d,work->stats.lambda_q8,rate,&cost);
        if (error==SVT_AV2_ERROR_NONE) {
            const uint64_t id=plan_candidate_id(decision,(1u<<29u)|decision,1u<<29u);
            ++work->stats.candidate_count;
            if (work->config->candidate_hook!=NULL)
                work->config->candidate_hook(work->config->candidate_hook_user,sb,id,d,rate,cost,0);
            if (cost<best->cost) {
                best->cost=cost;best->distortion=d;best->rate=rate;best->hash=path_hash(work);best->id=id;
            } else for (uint32_t leaf=0u;leaf<work->leaf_count;++leaf) leaf_tools_set(work,leaf,entering+leaf);
        }
    }
cleanup:
    if (error!=SVT_AV2_ERROR_NONE)
        for (uint32_t leaf=0u;leaf<work->leaf_count;++leaf) leaf_tools_set(work,leaf,entering+leaf);
    work->basic_rate_active=0u;work->prefix_active=0u;region_copy(work,1);
    if (error == SVT_AV2_ERROR_NONE && work->config->screen_trees)
        return refine_rdoq_leaves(work, sb, decision, best);
    return error==SVT_AV2_ERROR_NONE && !work->defer_rdoq?refine_rdoq_fast(work,sb,partial,decision,best):error;
}

static SvtAv2Error refine_tools_once(RdWork* work, size_t sb, int partial,
    uint16_t best_decision, RdBasicChoice* best) {
    SvtAv2Error error;
    uint64_t best_cost=best->cost, best_d=best->distortion, best_r=best->rate, best_hash=best->hash;
    uint64_t best_id=best->id;
    if (work->config->angle_search) {
        error = prepare_prefix(work, 0u);
        if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
        work->prefix_active = 1u;
        /* Retain the selected tree and all old candidates. Screen all 48 new
         * angles by causal Y SAD, then exact-test the best four plus up to six
         * offsets around the selected directional base. UV inherits a V/H
         * angle where required; only the final R+D trials compute UV. */
        for (uint32_t leaf = 0u; leaf < work->leaf_count; ++leaf) {
            if (!legal_mode(work, leaf, 14u)) continue;
            error = extend_prefix(work, leaf);
            if (error != SVT_AV2_ERROR_NONE) {
                work->prefix_active = 0u; region_copy(work, 1); return error;
            }
            uint64_t seed_d; uint32_t seed_path; int seed_valid;
            error = build_path(work, 0u, 0, partial, work->best_ids, &seed_valid, &seed_d, &seed_path);
            if (error != SVT_AV2_ERROR_NONE || !seed_valid) {
                work->prefix_active = 0u; region_copy(work, 1); return SVT_AV2_ERROR_INVALID_STATE;
            }
            SvtAv2BlockView blocks[3];
            error = plane_blocks(work, leaf, blocks);
            if (error != SVT_AV2_ERROR_NONE) { work->prefix_active = 0u; region_copy(work, 1); return error; }
            uint8_t ranked[4] = {255u,255u,255u,255u};
            uint64_t scores[4] = {UINT64_MAX,UINT64_MAX,UINT64_MAX,UINT64_MAX};
#if M6_SATD_ANGLES
            uint8_t satd_ranked[4]={255u,255u,255u,255u};
            uint64_t satd_scores[4]={UINT64_MAX,UINT64_MAX,UINT64_MAX,UINT64_MAX};
#endif
            for (uint32_t mode = 14u; mode <= 61u; ++mode) {
                uint64_t sad[3] = {0u,0u,0u};
                error = predict(work, leaf, mode, blocks, sad, 1);
                if (error != SVT_AV2_ERROR_NONE) { work->prefix_active = 0u; region_copy(work, 1); return error; }
#if M6_SATD_ANGLES
                angle_satd_rank(work,&blocks[0],mode,satd_ranked,satd_scores);
#endif
                for (uint32_t slot = 0u; slot < 4u; ++slot) {
                    if (sad[0] < scores[slot] || (sad[0] == scores[slot] && mode < ranked[slot])) {
                        for (uint32_t j = 3u; j > slot; --j) { scores[j] = scores[j - 1u]; ranked[j] = ranked[j - 1u]; }
                        scores[slot] = sad[0]; ranked[slot] = (uint8_t)mode; break;
                    }
                }
            }
            uint8_t candidates[14]; uint32_t count = 4u;
            memcpy(candidates, ranked, sizeof(ranked));
            const uint32_t base = svt_av2_intra_mode_base(work->best_ids[leaf]);
            const uint32_t joint = svt_av2_intra_mode_joint(base);
            if (joint >= 5u) {
                for (uint32_t slot = 0u; slot < 6u; ++slot) {
                    const uint8_t mode = (uint8_t)(14u + ((joint - 5u) / 7u) * 6u + slot);
                    uint32_t i = 0u;
                    while (i < count && candidates[i] != mode) ++i;
                    if (i == count) candidates[count++] = mode;
                }
            }
#if M6_SATD_ANGLES
            angle_satd_append(work,candidates,&count,satd_ranked);
#endif
            uint32_t choice = work->best_ids[leaf];
            for (uint32_t index = 0u; index < count; ++index) {
                const uint32_t mode = candidates[work->config->reverse_candidates ? count - 1u - index : index];
                uint8_t ids[RD_MAX_LEAVES];
                memcpy(ids, work->best_ids, work->leaf_count); ids[leaf] = (uint8_t)mode;
                region_copy_buffer(work, work->prefix_recon, 1);
                uint64_t d = 0u, r = 0u, cost = 0u;
                uint32_t path; int valid;
                error = build_path(work, 0u, 0, partial, ids, &valid, &d, &path);
                if (error == SVT_AV2_ERROR_NONE && (!valid ||
                    RD_RATE(work, sb, best_decision, work->blocks,
                        work->leaf_count * 3u, NULL, 0u, 0, &r) != 0)) error = SVT_AV2_ERROR_INVALID_STATE;
                if (error == SVT_AV2_ERROR_NONE) error = svt_av2_rd_cost(d, work->stats.lambda_q8, r, &cost);
                if (error != SVT_AV2_ERROR_NONE) { work->prefix_active = 0u; region_copy(work, 1); return error; }
                ++work->stats.candidate_count; ++work->stats.angle_trials;
                const uint32_t payload = (1u << 26u) | (leaf << 6u) | mode;
                const uint64_t id = plan_candidate_id(best_decision, payload, payload);
                if (work->config->candidate_hook != NULL)
                    work->config->candidate_hook(work->config->candidate_hook_user, sb, id, d, r, cost, 0);
                if (cost < best_cost || (cost == best_cost && mode < choice)) {
                    best_cost = cost; best_d = d; best_r = r; best_hash = path_hash(work); best_id = id;
                    memcpy(work->best_ids, ids, work->leaf_count); choice = mode;
                    ++work->stats.angle_accepts;
                }
            }
        }
        work->prefix_active = 0u; region_copy(work, 1);
    }
    if (work->config->frame_header.enable_mrls) {
        /* Refine the selected direction before TX/RDOQ. Every candidate
         * restores and rebuilds its complete causal SB, including UV CfL. */
        static const uint8_t candidates[6] = {1u,2u,3u,5u,6u,7u};
        error=prepare_prefix(work,0u);
        if (error!=SVT_AV2_ERROR_NONE) {region_copy(work,1);return error;}
        work->prefix_active=1u;
        for (uint32_t leaf=0u;leaf<work->leaf_count;++leaf) {
            const uint32_t w=work->leaves[leaf].w,h=work->leaves[leaf].h;
            if ((w!=h && w!=2u*h && h!=2u*w &&
                !((w==16u && h==64u) || (w==64u && h==16u))) || w<8u || h<8u ||
                svt_av2_intra_mode_joint(work->best_ids[leaf])<5u) continue;
            error=extend_prefix(work,leaf);
            if (error!=SVT_AV2_ERROR_NONE) {work->prefix_active=0u;region_copy(work,1);return error;}
            uint32_t choice=0u;
            for (uint32_t index=0u;index<6u;++index) {
                const uint32_t selection=candidates[work->config->reverse_candidates?5u-index:index];
                work->mrl[leaf]=(uint8_t)selection;
                region_copy_buffer(work,work->prefix_recon,1);
                uint64_t d=0u,r=0u,cost=0u; uint32_t path; int valid;
                error=build_path(work,0u,0,partial,work->best_ids,&valid,&d,&path);
                if (error==SVT_AV2_ERROR_NONE && (!valid ||
                    RD_RATE(work,sb,best_decision,work->blocks,
                        work->leaf_count*3u,NULL,0u,0,&r)!=0)) error=SVT_AV2_ERROR_INVALID_STATE;
                if (error==SVT_AV2_ERROR_NONE) error=svt_av2_rd_cost(d,work->stats.lambda_q8,r,&cost);
                if (error!=SVT_AV2_ERROR_NONE) {work->prefix_active=0u;region_copy(work,1);return error;}
                ++work->stats.candidate_count; ++work->stats.mrl_trials;
                const uint32_t payload=(1u<<24u)|(leaf<<3u)|selection;
                const uint64_t id=plan_candidate_id(best_decision,payload,payload);
                if (work->config->candidate_hook!=NULL)
                    work->config->candidate_hook(work->config->candidate_hook_user,sb,id,d,r,cost,0);
                if (cost<best_cost || (cost==best_cost && selection<choice)) {
                    best_cost=cost;best_d=d;best_r=r;best_hash=path_hash(work);best_id=id;choice=selection;
                    ++work->stats.mrl_accepts;
                }
            }
            work->mrl[leaf]=(uint8_t)choice;
        }
        work->prefix_active=0u;region_copy(work,1);
    }
    if (work->config->transform_selection) {
        error = prepare_prefix(work, 0u);
        if (error != SVT_AV2_ERROR_NONE) { work->prefix_active = 0u; region_copy(work, 1); return error; }
        work->prefix_active = 1u;
        /* Keep the entering DCT candidate; each primary candidate rebuilds the
         * full SB from the same reconstruction and committed CDF prefix. */
        for (uint32_t leaf = 0u; leaf < work->leaf_count; ++leaf) {
            const uint32_t w = work->leaves[leaf].w, h = work->leaves[leaf].h;
            if (!((w == h && w <= 16u) || (work->config->transform_selection == 2u &&
                ((w == 8u && h == 16u) || (w == 16u && h == 8u))))) continue;
            error = extend_prefix(work, leaf);
            if (error != SVT_AV2_ERROR_NONE) {
                work->prefix_active = 0u; region_copy(work, 1); return error;
            }
            uint32_t choice = work->tx_types[leaf];
            const uint32_t count = work->config->transform_selection == 2u ? 3u : 1u;
            for (uint32_t index = 0u; index < count; ++index) {
                const uint32_t type = count == 1u ? 3u :
                    work->config->reverse_candidates ? 3u - index : 1u + index;
                /* av2_md_idx2type[size_class=0][Paeth] excludes ADST_DCT. */
                if (type == 1u && work->leaves[leaf].w == 4u && work->best_ids[leaf] == 7u) continue;
                work->tx_types[leaf] = (uint8_t)type;
                region_copy_buffer(work, work->prefix_recon, 1);
                uint64_t d = 0u, r = 0u, cost = 0u;
                uint32_t path; int valid;
                error = build_path(work, 0u, 0, partial, work->best_ids, &valid, &d, &path);
                if (error == SVT_AV2_ERROR_NONE && (!valid ||
                    RD_RATE(work, sb, best_decision, work->blocks,
                        work->leaf_count * 3u, NULL, 0u, 0, &r) != 0)) error = SVT_AV2_ERROR_INVALID_STATE;
                if (error == SVT_AV2_ERROR_NONE) error = svt_av2_rd_cost(d, work->stats.lambda_q8, r, &cost);
                if (error != SVT_AV2_ERROR_NONE) { work->prefix_active = 0u; region_copy(work, 1); return error; }
                ++work->stats.candidate_count; ++work->stats.transform_trials;
                const uint32_t payload = (1u << 27u) | (leaf << 2u) | type;
                const uint64_t id = plan_candidate_id(best_decision,payload,payload);
                if (work->config->candidate_hook != NULL)
                    work->config->candidate_hook(work->config->candidate_hook_user, sb, id, d, r, cost, 0);
                if (cost < best_cost || (count > 1u && cost == best_cost && type < choice)) {
                    best_cost = cost; best_d = d; best_r = r; best_hash = path_hash(work); best_id = id;
                    choice = type; ++work->stats.transform_accepts;
                }
                work->tx_types[leaf] = (uint8_t)choice;
            }
        }
        work->prefix_active = 0u; region_copy(work, 1);
    }
    if (work->joint_y_tx) {
        RdBasicChoice joint={best_cost,best_d,best_r,best_hash,best_id};
        error=refine_joint_y_tx(work,sb,partial,best_decision,&joint);
        if (error!=SVT_AV2_ERROR_NONE) return error;
        best_cost=joint.cost;best_d=joint.distortion;best_r=joint.rate;best_hash=joint.hash;best_id=joint.id;
    }
    if (work->config->cfl_search) {
        error=prepare_prefix(work,0u);
        if (error!=SVT_AV2_ERROR_NONE) {region_copy(work,1);return error;}
        work->prefix_active=1u;
        for (uint32_t leaf=0u;leaf<work->leaf_count;++leaf) {
            error=extend_prefix(work,leaf);
            if (error!=SVT_AV2_ERROR_NONE) {work->prefix_active=0u;region_copy(work,1);return error;}
            uint64_t seed_d;uint32_t seed_path;int valid;
            error=build_path(work,0u,0,partial,work->best_ids,&valid,&seed_d,&seed_path);
            if (error!=SVT_AV2_ERROR_NONE || !valid) {work->prefix_active=0u;region_copy(work,1);return SVT_AV2_ERROR_INVALID_STATE;}
            SvtAv2BlockView blocks[3];
            error=plane_blocks(work,leaf,blocks);
            int8_t alphas[2][2]={{0,0},{0,0}};
            if (error==SVT_AV2_ERROR_NONE) error=rank_cfl_alphas(work,blocks+1u,1u,alphas[0]);
            if (error==SVT_AV2_ERROR_NONE) error=rank_cfl_alphas(work,blocks+2u,2u,alphas[1]);
            if (error!=SVT_AV2_ERROR_NONE) {work->prefix_active=0u;region_copy(work,1);return error;}
            uint8_t chosen_uv=work->uv_ids[leaf], chosen_explicit=work->cfl_explicit[leaf];
            uint8_t chosen_idx=work->cfl_alpha_idx[leaf], chosen_signs=work->cfl_alpha_signs[leaf];
            for (uint32_t index=0u;index<4u;++index) {
                const uint32_t pair=work->config->reverse_candidates?3u-index:index;
                const int32_t u=alphas[0][pair/2u],v=alphas[1][pair%2u];
                if (!u && !v) continue;
                const uint32_t usign=u<0?1u:u>0?2u:0u, vsign=v<0?1u:v>0?2u:0u;
                const uint32_t signs=usign*3u+vsign-1u;
                const uint32_t ui=u?(uint32_t)(u<0?-u:u)-1u:0u, vi=v?(uint32_t)(v<0?-v:v)-1u:0u;
                const uint32_t idx=(ui<<3u)|vi, code=(signs<<6u)|idx;
                work->uv_ids[leaf]=14u;work->cfl_explicit[leaf]=1u;
                work->cfl_alpha_idx[leaf]=(uint8_t)idx;work->cfl_alpha_signs[leaf]=(uint8_t)signs;
                region_copy_buffer(work,work->prefix_recon,1);
                uint64_t d=0u,r=0u,cost=0u;uint32_t path;
                error=build_path(work,0u,0,partial,work->best_ids,&valid,&d,&path);
                if (error==SVT_AV2_ERROR_NONE && (!valid || RD_RATE(work,sb,best_decision,
                    work->blocks,work->leaf_count*3u,NULL,0u,0,&r)!=0)) error=SVT_AV2_ERROR_INVALID_STATE;
                if (error==SVT_AV2_ERROR_NONE) error=svt_av2_rd_cost(d,work->stats.lambda_q8,r,&cost);
                if (error!=SVT_AV2_ERROR_NONE) {work->prefix_active=0u;region_copy(work,1);return error;}
                ++work->stats.candidate_count;++work->stats.cfl_trials;
                const uint32_t payload=(1u<<25u)|(leaf<<9u)|code;
                const uint64_t id=plan_candidate_id(best_decision,payload|((uint32_t)best_decision<<16u),payload);
                if (work->config->candidate_hook!=NULL)
                    work->config->candidate_hook(work->config->candidate_hook_user,sb,id,d,r,cost,0);
                if (cost<best_cost || (cost==best_cost && chosen_explicit && code<((uint32_t)chosen_signs<<6u|chosen_idx))) {
                    best_cost=cost;best_d=d;best_r=r;best_hash=path_hash(work);best_id=id;
                    chosen_uv=14u;chosen_explicit=1u;chosen_idx=(uint8_t)idx;chosen_signs=(uint8_t)signs;
                    ++work->stats.cfl_accepts;
                }
                work->uv_ids[leaf]=chosen_uv;work->cfl_explicit[leaf]=chosen_explicit;
                work->cfl_alpha_idx[leaf]=chosen_idx;work->cfl_alpha_signs[leaf]=chosen_signs;
            }
        }
        work->prefix_active=0u;region_copy(work,1);
    }
    if (work->config->mhccp_search && !work->skip_mhccp_refinement) {
        error=prepare_prefix(work,0u);
        if (error!=SVT_AV2_ERROR_NONE) {region_copy(work,1);return error;}
        work->prefix_active=1u;
        for (uint32_t leaf=0u;leaf<work->leaf_count;++leaf) {
            if (!svt_av2_intra_mhccp_allowed(work->leaves[leaf].w,work->leaves[leaf].h,
                (uint32_t)work->source->chroma_format)) continue;
            error=extend_prefix(work,leaf);
            if (error!=SVT_AV2_ERROR_NONE) {work->prefix_active=0u;region_copy(work,1);return error;}
            const uint8_t old_uv=work->uv_ids[leaf],old_explicit=work->cfl_explicit[leaf];
            uint32_t choice=0u;
            for (uint32_t index=0u;index<3u;++index) {
                const uint32_t selection=work->config->reverse_candidates?3u-index:1u+index;
                work->uv_ids[leaf]=14u;work->cfl_explicit[leaf]=0u;work->mhccp[leaf]=(uint8_t)selection;
                region_copy_buffer(work,work->prefix_recon,1);
                uint64_t d=0u,r=0u,cost=0u;uint32_t path;int valid;
                error=build_path(work,0u,0,partial,work->best_ids,&valid,&d,&path);
                if (error==SVT_AV2_ERROR_NONE && (!valid || RD_RATE(work,sb,best_decision,
                    work->blocks,work->leaf_count*3u,NULL,0u,0,&r)!=0)) error=SVT_AV2_ERROR_INVALID_STATE;
                if (error==SVT_AV2_ERROR_NONE) error=svt_av2_rd_cost(d,work->stats.lambda_q8,r,&cost);
                if (error!=SVT_AV2_ERROR_NONE) {work->prefix_active=0u;region_copy(work,1);return error;}
                ++work->stats.candidate_count;++work->stats.mhccp_trials;
                const uint32_t payload=(UINT32_C(1)<<30u)|(leaf<<2u)|selection;
                const uint64_t id=plan_candidate_id(best_decision,payload|((uint32_t)best_decision<<16u),payload);
                if (work->config->candidate_hook!=NULL)
                    work->config->candidate_hook(work->config->candidate_hook_user,sb,id,d,r,cost,0);
                if (cost<best_cost || (cost==best_cost && choice && selection<choice)) {
                    best_cost=cost;best_d=d;best_r=r;best_hash=path_hash(work);best_id=id;choice=selection;
                    ++work->stats.mhccp_accepts;
                }
                work->mhccp[leaf]=(uint8_t)choice;
                work->uv_ids[leaf]=choice?14u:old_uv;
                work->cfl_explicit[leaf]=choice?0u:old_explicit;
            }
        }
        work->prefix_active=0u;region_copy(work,1);
    }
    if (!partial && work->config->coeff_optimization == 1u) {
        const uint64_t mode_id = best_id;
        uint32_t best_policy = 0u;
        /* One raster coordinate pass over the original winning tree/modes.
         * Every trial rebuilds all leaves from the identical SB entry state. */
        for (uint32_t leaf = 0u; leaf < work->leaf_count; ++leaf) {
            for (uint32_t plane = 0u; plane < 3u; ++plane) {
                work->coeff_policy = best_policy | (1u << (3u * leaf + plane));
                region_copy(work, 1);
                uint64_t trial_d = 0u, trial_r = 0u, trial_cost = 0u;
                uint32_t trial_path; int trial_legal;
                error = build_path(work, 0u, 0, 0, work->best_ids, &trial_legal, &trial_d, &trial_path);
                if (error == SVT_AV2_ERROR_NONE && (!trial_legal ||
                    RD_RATE(work, sb, best_decision, work->blocks,
                        work->leaf_count * 3u, work->reference == NULL ? NULL : work->modes,
                        work->reference == NULL ? 0u : work->leaf_count, 0, &trial_r) != 0))
                    error = SVT_AV2_ERROR_INVALID_STATE;
                if (error == SVT_AV2_ERROR_NONE)
                    error = svt_av2_rd_cost(trial_d, work->stats.lambda_q8, trial_r, &trial_cost);
                if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
                ++work->stats.candidate_count; ++work->stats.coeff_trials;
                const uint64_t trial_id = mode_id | (work->coeff_policy << 9u);
                if (work->config->candidate_hook != NULL)
                    work->config->candidate_hook(work->config->candidate_hook_user, sb, trial_id,
                        trial_d, trial_r, trial_cost, 0);
                if (trial_cost < best_cost || (trial_cost == best_cost && trial_id < best_id)) {
                    best_cost = trial_cost; best_d = trial_d; best_r = trial_r; best_id = trial_id;
                    best_policy = work->coeff_policy; best_hash = path_hash(work);
                    ++work->stats.coeff_accepts;
                }
            }
        }
        work->coeff_policy = best_policy;
        region_copy(work, 1);
    }
    if (work->config->coeff_optimization == 2u) {
        error = prepare_prefix(work, 0u);
        if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
        work->prefix_active = 1u;
        /* Bounded strengths start from the same scalar coefficients. Every
         * proposal rebuilds the suffix and rates the complete actual SB.
         * No approximate coefficient cost can directly become a winner. */
        for (uint32_t block = 0u; block < work->leaf_count * 3u; ++block) {
            work->rdoq_target = UINT32_MAX;
            error = extend_prefix(work, block / 3u);
            if (error != SVT_AV2_ERROR_NONE) {
                work->prefix_active = 0u; region_copy(work, 1); return error;
            }
            int32_t chosen_coeff[1024];
            size_t chosen_offset=0u,chosen_count=0u;
            uint32_t chosen_phase=3u;uint8_t chosen_type=0u;
            for (uint32_t index=0u;index<3u;++index) {
                const uint32_t phase=work->config->reverse_candidates?2u-index:index;
                work->rdoq_phase=phase;
                work->rdoq_target = block;
                region_copy_buffer(work, work->prefix_recon, 1);
                uint64_t d = 0u, r = 0u, cost = 0u; uint32_t path; int valid;
                error = build_path(work, 0u, 0, partial, work->best_ids, &valid, &d, &path);
                if (error == SVT_AV2_ERROR_NONE && (!valid ||
                    RD_RATE(work, sb, best_decision, work->blocks,
                        work->leaf_count * 3u, NULL, 0u, 0, &r) != 0)) error = SVT_AV2_ERROR_INVALID_STATE;
                if (error == SVT_AV2_ERROR_NONE) error = svt_av2_rd_cost(d, work->stats.lambda_q8, r, &cost);
                if (error != SVT_AV2_ERROR_NONE) { work->prefix_active = 0u; region_copy(work, 1); return error; }
                const uint32_t payload = (1u << (phase ? 29u : 26u)) | (phase << 8u) | block |
                    (work->rdoq_use_context?(1u<<27u):0u);
                const uint64_t id = plan_candidate_id(best_decision,payload,payload);
                ++work->stats.candidate_count; ++work->stats.coeff_trials;
                if (work->config->candidate_hook != NULL)
                    work->config->candidate_hook(work->config->candidate_hook_user, sb, id, d, r, cost, 0);
                if (cost < best_cost || (cost == best_cost && chosen_phase < 3u && phase < chosen_phase)) {
                    best_cost = cost; best_d = d; best_r = r; best_id = id;
                    best_hash = path_hash(work); work->rdoq_saved[block] = 1u;
                    const SvtAv2TileCoeffBlock* entry=work->blocks+block;
                    chosen_offset=(size_t)(entry->qcoeff-work->coeffs);
                    chosen_count=(size_t)SVT_AV2_MIN(entry->width,32u)*SVT_AV2_MIN(entry->height,32u);
                    memcpy(chosen_coeff,entry->qcoeff,chosen_count*sizeof(*chosen_coeff));
                    chosen_type=work->rdoq_tx_types[block];chosen_phase=phase;
                    ++work->stats.coeff_accepts;
                }
            }
            if (chosen_phase < 3u) {
                memcpy(work->rdoq_coeff+chosen_offset,chosen_coeff,chosen_count*sizeof(*chosen_coeff));
                work->rdoq_tx_types[block]=chosen_type;
            }
        }
        work->rdoq_target = UINT32_MAX;
        work->rdoq_phase = 0u;
        work->prefix_active = 0u; region_copy(work, 1);
    }
    best->cost=best_cost; best->distortion=best_d; best->rate=best_r;
    best->hash=best_hash; best->id=best_id;
    return SVT_AV2_ERROR_NONE;
}

/* Retain the complete original routes before comparing joint Y/TX routes.
 * Every branch starts at the same basic seed with identical syntax/CDFs.
 * With joint search off, the existing MH two-route order and tie break remain. */
static SvtAv2Error refine_tools(RdWork* work, size_t sb, int partial,
    uint16_t decision, RdBasicChoice* best) {
    if (work->config->rd_mode==3u)
        return refine_tools_linear(work,sb,partial,decision,best);
    const uint32_t mh_count=work->config->mhccp_search &&
        work->config->coeff_optimization==2u?2u:1u;
    const uint32_t joint_count=work->config->intra_joint?2u:1u;
    const uint32_t count=mh_count*joint_count;
    if (count==1u) return refine_tools_once(work,sb,partial,decision,best);
    RdWork* states=(RdWork*)malloc(2u*sizeof(*states));
    if (states==NULL) return SVT_AV2_ERROR_OUT_OF_MEMORY;
    states[0]=*work;
    SvtAv2RdFrameStats aggregate=work->stats;
    RdBasicChoice winner={UINT64_MAX,0u,0u,0u,UINT64_MAX};
    RdBasicChoice groups[2]={{UINT64_MAX,0u,0u,0u,UINT64_MAX},
        {UINT64_MAX,0u,0u,0u,UINT64_MAX}};
    uint32_t winner_group=UINT32_MAX;
    SvtAv2Error error=SVT_AV2_ERROR_NONE;
    for (uint32_t i=0u;i<count;++i) {
        const uint32_t branch=work->config->reverse_candidates?count-1u-i:i;
        const uint32_t joint=branch/mh_count;
        *work=states[0];work->stats=aggregate;
        work->joint_y_tx=joint;
        work->skip_mhccp_refinement=mh_count==2u && branch%mh_count==0u;
        region_copy(work,1);
        RdBasicChoice trial=*best;
        error=refine_tools_once(work,sb,partial,decision,&trial);
        aggregate=work->stats;
        if (error!=SVT_AV2_ERROR_NONE) break;
        if (trial.cost<groups[joint].cost || (trial.cost==groups[joint].cost &&
            (trial.hash<groups[joint].hash || (trial.hash==groups[joint].hash && trial.id<groups[joint].id))))
            groups[joint]=trial;
        if (trial.cost<winner.cost || (trial.cost==winner.cost &&
            (joint<winner_group || (joint==winner_group &&
            (trial.hash<winner.hash || (trial.hash==winner.hash && trial.id<winner.id)))))) {
            winner=trial;winner_group=joint;states[1]=*work;
        }
    }
    *work=states[error==SVT_AV2_ERROR_NONE?1u:0u];
    work->stats=aggregate;work->skip_mhccp_refinement=0u;work->joint_y_tx=0u;
    if (error==SVT_AV2_ERROR_NONE) {
        *best=winner;
        if (joint_count==2u && work->config->candidate_hook!=NULL) {
            const uint32_t payload=(UINT32_C(1)<<31u)|(1u<<23u)|(3u<<20u);
            for (uint32_t group=0u;group<2u;++group)
                work->config->candidate_hook(work->config->candidate_hook_user,sb,
                    plan_candidate_id(decision,payload|group,payload|group),
                    groups[group].distortion,groups[group].rate,groups[group].cost,0);
        }
    }
    region_copy(work,1);
    free(states);
    return error;
}

typedef struct RdTreeSeed {
    RdBasicChoice choice;
    uint16_t decision;
    uint32_t leaf_count;
    uint8_t ids[RD_MAX_LEAVES], uv[RD_MAX_LEAVES];
} RdTreeSeed;

static void offer_tree(const RdWork* work, uint16_t decision, const RdBasicChoice* choice,
    RdTreeSeed seeds[4], uint32_t* count) {
    const uint32_t limit=work->config->tree_finalists>=8u?4u:2u;
    uint32_t slot=0u;
    while (slot<*count && seeds[slot].decision!=decision) ++slot;
    if (slot<*count) {
        if (choice->cost>=seeds[slot].choice.cost) return;
        for (uint32_t j=slot+1u;j<*count;++j) seeds[j-1u]=seeds[j];
        --*count;
    }
    slot=0u;
    while (slot<*count && (seeds[slot].choice.cost<choice->cost ||
        (seeds[slot].choice.cost==choice->cost && seeds[slot].decision<decision))) ++slot;
    if (slot>=limit) return;
    if (*count<limit) ++*count;
    for (uint32_t j=*count-1u;j>slot;--j) seeds[j]=seeds[j-1u];
    seeds[slot].choice=*choice;seeds[slot].decision=decision;seeds[slot].leaf_count=work->leaf_count;
    memcpy(seeds[slot].ids,work->best_ids,work->leaf_count);
    memcpy(seeds[slot].uv,work->uv_ids,work->leaf_count);
}

/* Each complete tree starts at the same SB entry. Only the final replay is
 * appended and committed; the rate session keeps its original committed CDF. */
static SvtAv2Error refine_tree_finalists_once(RdWork* work, size_t sb, int partial,
    const RdTreeSeed seeds[10], uint32_t count, uint16_t* decision, RdBasicChoice* best) {
    if (!count || count>10u) return SVT_AV2_ERROR_INVALID_PARAMETER;
    RdWork* saved=(RdWork*)malloc(sizeof(*saved));
    if (saved==NULL) return SVT_AV2_ERROR_OUT_OF_MEMORY;
    *saved=*work;
    const uint16_t incumbent=*decision;
    const uint32_t defer=!work->tree_rdoq && work->config->rd_mode==3u && count>1u && !partial &&
        work->config->coeff_optimization==2u;
    uint16_t selected=incumbent;
    RdBasicChoice winner={UINT64_MAX,0u,0u,0u,UINT64_MAX};
    SvtAv2Error error=SVT_AV2_ERROR_NONE;
    for (uint32_t i=0u; i<count; ++i) {
        const RdTreeSeed* seed=&seeds[work->config->reverse_candidates ? count-1u-i : i];
        region_copy(work,1);
        work->prefix_active=0u; work->prefix_leaf=0u; work->prefix_coeff_count=0u;
        work->coeff_policy=work->motion_policy=work->wide_stage=0u;
        memset(work->mrl,0,sizeof(work->mrl));
        memset(work->mhccp,0,sizeof(work->mhccp));
        memset(work->tx_types,0,sizeof(work->tx_types));
        memset(work->cfl_explicit,0,sizeof(work->cfl_explicit));
        memset(work->cfl_alpha_idx,0,sizeof(work->cfl_alpha_idx));
        memset(work->cfl_alpha_signs,0,sizeof(work->cfl_alpha_signs));
        memset(work->rdoq_saved,0,sizeof(work->rdoq_saved));
        work->rdoq_target=UINT32_MAX;
        error=geometry_for_sb(work,sb,seed->decision);
        if (error!=SVT_AV2_ERROR_NONE || work->leaf_count!=seed->leaf_count) {
            error=SVT_AV2_ERROR_INVALID_STATE; break;
        }
        memcpy(work->best_ids,seed->ids,work->leaf_count);
        memcpy(work->uv_ids,seed->uv,work->leaf_count);
        RdBasicChoice trial=seed->choice;
        uint64_t d=0u,r=0u; uint32_t path; int valid;
        error=build_path(work,0u,0,partial,work->best_ids,&valid,&d,&path);
        if (error!=SVT_AV2_ERROR_NONE || !valid || d!=trial.distortion || path_hash(work)!=trial.hash ||
            RD_RATE(work,sb,seed->decision,work->blocks,
                work->leaf_count*3u,NULL,0u,0,&r)!=0 || r!=trial.rate) {
            error=SVT_AV2_ERROR_INVALID_STATE; break;
        }
        region_copy(work,1);
        work->defer_rdoq=defer;
        error=refine_tools(work,sb,partial,seed->decision,&trial);
#if M6_TREE_RDOQ && M6_TX_CACHE
        if (work->tree_rdoq && work->transform_cache) ++work->transform_cache->treeq_finalists;
#endif
        work->defer_rdoq=0u;
        if (error!=SVT_AV2_ERROR_NONE) break;
        error=build_path(work,0u,0,partial,work->best_ids,&valid,&d,&path);
        if (error!=SVT_AV2_ERROR_NONE || !valid || d!=trial.distortion || path_hash(work)!=trial.hash ||
            RD_RATE(work,sb,seed->decision,work->blocks,
                work->leaf_count*3u,NULL,0u,0,&r)!=0 || r!=trial.rate) {
            error=SVT_AV2_ERROR_INVALID_STATE; break;
        }
        ++work->stats.tree_finalist_trials;
        if (work->config->candidate_hook!=NULL)
            work->config->candidate_hook(work->config->candidate_hook_user,sb,
                plan_candidate_id(seed->decision,(UINT32_C(1)<<31u)|seed->decision,UINT32_C(1)<<31u),
                trial.distortion,trial.rate,trial.cost,0);
        const int tie=trial.cost==winner.cost &&
            ((seed->decision==incumbent && selected!=incumbent) ||
             (seed->decision!=incumbent && selected!=incumbent && seed->decision<selected) ||
             (seed->decision==selected && trial.hash<winner.hash));
        if (trial.cost<winner.cost || tie) {
            winner=trial; selected=seed->decision; *saved=*work;
        }
    }
    if (error==SVT_AV2_ERROR_NONE) {
        const SvtAv2RdFrameStats aggregate=work->stats;
        *work=*saved; work->stats=aggregate;
        if (selected!=incumbent) ++work->stats.tree_finalist_accepts;
        *decision=selected; *best=winner;
        error=geometry_for_sb(work,sb,selected);
        if (error==SVT_AV2_ERROR_NONE && defer)
            error=refine_rdoq_fast(work,sb,partial,selected,best);
    }
    work->defer_rdoq=0u;
    region_copy(work,1);
    free(saved);
    return error;
}

#if M6_TREE_RDOQ
static SvtAv2Error treeq_verify(RdWork* work,size_t sb,int partial,
    uint16_t decision,const RdBasicChoice* expected) {
    region_copy(work,1);
    uint64_t d=0u,r=0u,j=0u;uint32_t path=0u;int valid=0;
    SvtAv2Error error=build_path(work,0u,0,partial,work->best_ids,&valid,&d,&path);
    if (error==SVT_AV2_ERROR_NONE && (!valid || d!=expected->distortion ||
        path_hash(work)!=expected->hash || RD_RATE(work,sb,decision,
        work->blocks,work->leaf_count*3u,NULL,0u,0,&r)!=0 || r!=expected->rate))
        error=SVT_AV2_ERROR_INVALID_STATE;
    if (error==SVT_AV2_ERROR_NONE) error=svt_av2_rd_cost(d,work->stats.lambda_q8,r,&j);
    if (error==SVT_AV2_ERROR_NONE && j!=expected->cost) error=SVT_AV2_ERROR_INVALID_STATE;
    region_copy(work,1);return error;
}
#endif
#if M6_TREE_RDOQ && M6_TREE_SHARED
/* F is the unchanged linear tool search with final RDOQ deferred. G is the
 * entire original RDOQ wrapper. The old winner is selected by F, the new one
 * by G; both preserve the original incumbent/plan/hash tie ordering. */
static int shared_tree_better(const RdBasicChoice* trial,uint16_t plan,
    const RdBasicChoice* winner,uint16_t selected,uint16_t incumbent) {
    return trial->cost<winner->cost || (trial->cost==winner->cost &&
        ((plan==incumbent && selected!=incumbent) ||
         (plan!=incumbent && selected!=incumbent && plan<selected) ||
         (plan==selected && trial->hash<winner->hash)));
}
static SvtAv2Error refine_tree_finalists_shared(RdWork* work,size_t sb,int partial,
    const RdTreeSeed seeds[10],uint32_t count,uint16_t* decision,RdBasicChoice* best) {
    if (!count || count>10u) return SVT_AV2_ERROR_INVALID_PARAMETER;
    RdWork* states=(RdWork*)malloc(3u*sizeof(*states));
    if (!states) return SVT_AV2_ERROR_OUT_OF_MEMORY;
    states[0]=*work;
    const uint16_t incumbent=*decision;
    uint16_t old_plan=incumbent,new_plan=incumbent;
    RdBasicChoice old_scalar={UINT64_MAX,0u,0u,0u,UINT64_MAX};
    RdBasicChoice old_final=old_scalar,new_final=old_scalar;
    SvtAv2Error error=SVT_AV2_ERROR_NONE;
    for (uint32_t i=0u;i<count;++i) {
        const SvtAv2RdFrameStats aggregate=work->stats;
        *work=states[0];work->stats=aggregate;
        work->tree_rdoq=0u;region_copy(work,1);
        const RdTreeSeed* seed=&seeds[work->config->reverse_candidates?count-1u-i:i];
        work->prefix_active=0u; work->prefix_leaf=0u; work->prefix_coeff_count=0u;
        work->coeff_policy=work->motion_policy=work->wide_stage=0u;
        memset(work->mrl,0,sizeof(work->mrl));
        memset(work->mhccp,0,sizeof(work->mhccp));
        memset(work->tx_types,0,sizeof(work->tx_types));
        memset(work->cfl_explicit,0,sizeof(work->cfl_explicit));
        memset(work->cfl_alpha_idx,0,sizeof(work->cfl_alpha_idx));
        memset(work->cfl_alpha_signs,0,sizeof(work->cfl_alpha_signs));
        memset(work->rdoq_saved,0,sizeof(work->rdoq_saved));
        work->rdoq_target=UINT32_MAX;
        error=geometry_for_sb(work,sb,seed->decision);
        if (error!=SVT_AV2_ERROR_NONE || work->leaf_count!=seed->leaf_count) {
            error=SVT_AV2_ERROR_INVALID_STATE; break;
        }
        memcpy(work->best_ids,seed->ids,work->leaf_count);
        memcpy(work->uv_ids,seed->uv,work->leaf_count);
        RdBasicChoice trial=seed->choice;
        uint64_t d=0u,r=0u; uint32_t path; int valid;
        error=build_path(work,0u,0,partial,work->best_ids,&valid,&d,&path);
        if (error!=SVT_AV2_ERROR_NONE || !valid || d!=trial.distortion || path_hash(work)!=trial.hash ||
            RD_RATE(work,sb,seed->decision,work->blocks,
                work->leaf_count*3u,NULL,0u,0,&r)!=0 || r!=trial.rate) {
            error=SVT_AV2_ERROR_INVALID_STATE; break;
        }
        region_copy(work,1);work->defer_rdoq=1u;
#if M6_TX_CACHE
        if (work->transform_cache) ++work->transform_cache->shared_f;
#endif
        error=refine_tools(work,sb,partial,seed->decision,&trial);
        work->defer_rdoq=0u;
        if (error!=SVT_AV2_ERROR_NONE) break;
        const RdBasicChoice scalar=trial; /* F hash/cost, never overwritten by G. */
        error=treeq_verify(work,sb,partial,seed->decision,&scalar);
        if (error!=SVT_AV2_ERROR_NONE) break;
        const int select_old=shared_tree_better(&scalar,seed->decision,
            &old_scalar,old_plan,incumbent);
#if M6_TX_CACHE
        if (work->transform_cache) {
            ++work->transform_cache->shared_g;
            ++work->transform_cache->treeq_finalists;
        }
#endif
        error=refine_rdoq_fast(work,sb,partial,seed->decision,&trial);
        if (error==SVT_AV2_ERROR_NONE) error=treeq_verify(work,sb,partial,seed->decision,&trial);
        if (error!=SVT_AV2_ERROR_NONE) break;
        ++work->stats.tree_finalist_trials;
        if (work->config->candidate_hook!=NULL)
            work->config->candidate_hook(work->config->candidate_hook_user,sb,
                plan_candidate_id(seed->decision,(UINT32_C(1)<<31u)|seed->decision,UINT32_C(1)<<31u),
                trial.distortion,trial.rate,trial.cost,0);
        if (select_old) {
            old_scalar=scalar;old_final=trial;old_plan=seed->decision;states[1]=*work;
        }
        if (shared_tree_better(&trial,seed->decision,&new_final,new_plan,incumbent)) {
            new_final=trial;new_plan=seed->decision;states[2]=*work;
        }
    }
    const SvtAv2RdFrameStats aggregate=work->stats;
    RdBasicChoice winner=*best;uint16_t winner_plan=incumbent;
    if (error==SVT_AV2_ERROR_NONE) {
        const int new_wins=new_final.cost<old_final.cost;
#if M6_TX_CACHE
        if (work->transform_cache) {
            ++work->transform_cache->treeq_trials;
            work->transform_cache->treeq_wins+=(uint64_t)new_wins;
            work->transform_cache->treeq_changes+=(uint64_t)(new_wins && new_plan!=old_plan);
        }
#endif
        const int take_new=new_wins && !M6_TREE_RETAIN_OLD;
        winner=take_new?new_final:old_final;winner_plan=take_new?new_plan:old_plan;
        *work=states[take_new?2u:1u];
    } else *work=states[0];
    work->stats=aggregate;work->tree_rdoq=0u;work->defer_rdoq=0u;
    if (error==SVT_AV2_ERROR_NONE) {
        if (winner_plan!=incumbent) ++work->stats.tree_finalist_accepts;
        const uint32_t selected_leaves=work->leaf_count;
#if M6_TX_CACHE
        if (work->transform_cache) {
            ++work->transform_cache->treeq_plan_commits;
            work->transform_cache->treeq_plan_repairs+=
                (uint64_t)(work->output->decisions[sb]!=winner_plan);
        }
#endif
        error=geometry_for_sb(work,sb,winner_plan);
        if (error==SVT_AV2_ERROR_NONE && work->leaf_count!=selected_leaves)
            error=SVT_AV2_ERROR_INVALID_STATE;
        if (error==SVT_AV2_ERROR_NONE) error=treeq_verify(work,sb,partial,winner_plan,&winner);
    }
    region_copy(work,1);free(states);
    if (error==SVT_AV2_ERROR_NONE) {*best=winner;*decision=winner_plan;}
    return error;
}
#endif

static SvtAv2Error refine_tree_finalists(RdWork* work,size_t sb,int partial,
    const RdTreeSeed seeds[10],uint32_t count,uint16_t* decision,RdBasicChoice* best) {
#if M6_TREE_RDOQ
    if (work->config->rd_mode!=3u || count<2u || partial || work->config->coeff_optimization!=2u)
        return refine_tree_finalists_once(work,sb,partial,seeds,count,decision,best);
    RdWork* states=(RdWork*)malloc(2u*sizeof(*states));
    if (!states) return SVT_AV2_ERROR_OUT_OF_MEMORY;
    states[0]=*work;const RdBasicChoice entry=*best;const uint16_t entry_decision=*decision;
    work->tree_rdoq=0u;
    RdBasicChoice winner=entry;uint16_t winner_decision=entry_decision;
#if M6_TREE_SHARED
    SvtAv2Error error=refine_tree_finalists_shared(work,sb,partial,seeds,count,&winner_decision,&winner);
    SvtAv2RdFrameStats aggregate=work->stats;
#else
    SvtAv2Error error=refine_tree_finalists_once(work,sb,partial,seeds,count,&winner_decision,&winner);
    if (error==SVT_AV2_ERROR_NONE) error=treeq_verify(work,sb,partial,winner_decision,&winner);
    SvtAv2RdFrameStats aggregate=work->stats;
    if (error==SVT_AV2_ERROR_NONE) {
        states[1]=*work;
        *work=states[0];work->stats=aggregate;work->tree_rdoq=1u;region_copy(work,1);
        RdBasicChoice trial=entry;uint16_t trial_decision=entry_decision;
        error=refine_tree_finalists_once(work,sb,partial,seeds,count,&trial_decision,&trial);
        if (error==SVT_AV2_ERROR_NONE) error=treeq_verify(work,sb,partial,trial_decision,&trial);
        aggregate=work->stats;
#if M6_TX_CACHE
        if (work->transform_cache) {
            ++work->transform_cache->treeq_trials;
            work->transform_cache->treeq_wins+=(uint64_t)(error==SVT_AV2_ERROR_NONE && trial.cost<winner.cost);
            work->transform_cache->treeq_changes+=(uint64_t)(error==SVT_AV2_ERROR_NONE && trial.cost<winner.cost && trial_decision!=winner_decision);
        }
#endif
#if M6_TREE_RETAIN_OLD
        *work=states[1];
#else
        if (error==SVT_AV2_ERROR_NONE && trial.cost<winner.cost) {
            winner=trial;winner_decision=trial_decision;
        } else *work=states[1];
#endif
    }
#endif
#if M6_TOOL_REPEAT
    if (error==SVT_AV2_ERROR_NONE) {
        states[1]=*work;
        *work=states[0];work->stats=aggregate;work->tree_rdoq=1u;work->tool_rounds=2u;
        region_copy(work,1);
        RdBasicChoice trial=entry;uint16_t trial_decision=entry_decision;
        error=refine_tree_finalists_once(work,sb,partial,seeds,count,&trial_decision,&trial);
        if (error==SVT_AV2_ERROR_NONE) error=treeq_verify(work,sb,partial,trial_decision,&trial);
        aggregate=work->stats;
#if M6_TX_CACHE
        if (work->transform_cache) {
            ++work->transform_cache->repeat_trials;
            work->transform_cache->repeat_wins+=(uint64_t)(error==SVT_AV2_ERROR_NONE && trial.cost<winner.cost);
        }
#endif
#if M6_REPEAT_RETAIN_OLD
        *work=states[1];
#else
        if (error==SVT_AV2_ERROR_NONE && trial.cost<winner.cost) {
            winner=trial;winner_decision=trial_decision;
        } else *work=states[1];
#endif
    }
#endif
    if (error!=SVT_AV2_ERROR_NONE) *work=states[0];
    work->stats=aggregate;work->tree_rdoq=0u;work->defer_rdoq=0u;work->tool_rounds=0u;
    /* RdWork snapshots borrow output->decisions. Rebind that shared plan to
     * the selected state before final replay and the eventual packet writer. */
    if (error==SVT_AV2_ERROR_NONE) {
        const uint32_t selected_leaves=work->leaf_count;
#if M6_TX_CACHE
        if (work->transform_cache) {
            ++work->transform_cache->treeq_plan_commits;
            work->transform_cache->treeq_plan_repairs+=
                (uint64_t)(work->output->decisions[sb]!=winner_decision);
        }
#endif
        error=geometry_for_sb(work,sb,winner_decision);
        if (error==SVT_AV2_ERROR_NONE && work->leaf_count!=selected_leaves)
            error=SVT_AV2_ERROR_INVALID_STATE;
    }
    region_copy(work,1);free(states);
    if (error==SVT_AV2_ERROR_NONE) {*best=winner;*decision=winner_decision;}
    return error;
#else
    return refine_tree_finalists_once(work,sb,partial,seeds,count,decision,best);
#endif
}

 #include "fine16_partition_search.inc"
 #include "fine16_pair_search.inc"

/* Keep equivalent old geometries in their established plan namespace. The
 * coordinate search retains its independent base5 code even after this map. */
static uint16_t mixed8_plan(uint32_t code) {
    static const uint16_t uniform[5]={2u,3u,85u,86u,1u};
    uint32_t digits=code, square_code=4u, power=1u, first=code%5u;
    int squares=1, same=1;
    for (uint32_t q=0u;q<4u;++q) {
        const uint32_t shape=digits%5u;digits/=5u;
        const uint32_t depth=shape==4u?1u:shape==0u?2u:3u;
        if (shape==2u || shape==3u) squares=0;
        if (shape!=first) same=0;
        square_code+=(depth-1u)*power;power*=3u;
    }
    return same?uniform[first]:(uint16_t)(squares?square_code:255u+code);
}

#ifndef SVT_AV2_INTER_BOUNDED_SAD
#define SVT_AV2_INTER_BOUNDED_SAD 1
#endif
#ifndef SVT_AV2_INTER_BOUNDED_Q3
#define SVT_AV2_INTER_BOUNDED_Q3 1
#endif

static SvtAv2Error quarter_mv_sad(RdWork* work, const SvtAv2BlockView* block,
    SvtAv2MotionVector mv, uint64_t limit, uint64_t* sad);

/* Test the eight half-pel neighbors of each integer winner. Luma SAD is
 * only a shortlist, never the final decision. The two finalists use the
 * same three-plane reconstruction and syntax transaction as the incumbent. */
static SvtAv2Error refine_halfpel(RdWork* work, size_t sb, uint16_t decision,
    RdBasicChoice* best) {
    uint32_t policy = 0u;
    for (uint32_t leaf = 0u; leaf < work->leaf_count; ++leaf) {
        SvtAv2BlockView blocks[3];
        SvtAv2Error error = plane_blocks(work, leaf, blocks);
        if (error != SVT_AV2_ERROR_NONE) return error;
        const SvtAv2MotionVector center = work->mvs[leaf];
        SvtAv2MotionVector shortlist[2] = {{0, 0}, {0, 0}};
        uint64_t order[2] = {UINT64_MAX, UINT64_MAX};
        const int quarter = work->config->frame_header.mv_half == 2u;
        SvtAv2MotionVector anchor = center;
        uint64_t anchor_sad = UINT64_MAX;
        if (quarter) {
            error = quarter_mv_sad(work, &blocks[0], center, UINT64_MAX, &anchor_sad);
            if (error != SVT_AV2_ERROR_NONE) return error;
        }
        for (uint32_t level = 0u; level < (quarter ? 2u : 1u); ++level) {
            const SvtAv2MotionVector origin = level == 0u ? center : anchor;
            const int32_t step = quarter && level == 0u ? 2 : 1;
            for (int32_t dy = -1; dy <= 1; ++dy) {
                for (int32_t dx = -1; dx <= 1; ++dx) {
                    if (!dx && !dy) continue;
                    const SvtAv2MotionVector mv = {origin.mv_x + dx * step, origin.mv_y + dy * step};
                    uint64_t luma_sad;
                    if (quarter) {
                        /* A larger SAD than the runner-up cannot enter either
                         * slot: every positional tie-break here is below 32.
                         * Keep exact SAD on equality for the tie-break itself. */
                        const uint64_t bound = order[1] == UINT64_MAX ? UINT64_MAX : order[1] / 32u;
                        error = quarter_mv_sad(work, &blocks[0], mv, bound, &luma_sad);
                    } else {
                        uint64_t sad[3];
                        work->mvs[leaf] = mv;
                        error = predict(work, leaf, 3u, blocks, sad, 1);
                        luma_sad = error == SVT_AV2_ERROR_NONE ? sad[0] : 0u;
                        ++work->stats.motion_candidates;
                    }
                    if (error != SVT_AV2_ERROR_NONE) { work->mvs[leaf] = center; return error; }
                    /* One half-lattice anchor, then a quarter ring. Rank both rings
                     * by luma SAD; tie order for the original half path is unchanged. */
                    const uint64_t key = luma_sad * 32u + (dx && dy ? 16u : 0u) + 6u * level +
                        (uint32_t)((dy + 1) * 3 + dx + 1);
                    if (key < order[0]) {
                        order[1] = order[0]; shortlist[1] = shortlist[0];
                        order[0] = key; shortlist[0] = mv;
                    } else if (key < order[1]) {
                        order[1] = key; shortlist[1] = mv;
                    }
                }
            }
            if (level == 0u && quarter && order[0] / 32u < anchor_sad) anchor = shortlist[0];
        }
        SvtAv2MotionVector winner = center;
        uint32_t winner_policy = policy;
        for (uint32_t which = 0u; which < 2u; ++which) {
            const uint32_t rank = work->config->reverse_candidates ? 1u - which : which;
            const uint32_t trial_policy = policy | ((rank + 1u) << (leaf * 2u));
            uint8_t ids[4];
            memcpy(ids, work->best_ids, work->leaf_count);
            ids[leaf] = 3u;
            work->mvs[leaf] = shortlist[rank];
            region_copy(work, 1);
            uint64_t distortion = 0u, rate = 0u, cost = 0u;
            uint32_t path_id;
            int legal;
            error = build_path(work, 0u, 0, 0, ids, &legal, &distortion, &path_id);
            if (error == SVT_AV2_ERROR_NONE && (!legal ||
                RD_RATE(work, sb, decision, work->blocks,
                    work->leaf_count * 3u, work->modes, work->leaf_count, 0, &rate) != 0))
                error = SVT_AV2_ERROR_INVALID_STATE;
            if (error == SVT_AV2_ERROR_NONE)
                error = svt_av2_rd_cost(distortion, work->stats.lambda_q8, rate, &cost);
            if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
            ++work->stats.candidate_count;
            ++work->stats.motion_trials;
            const uint64_t id = (UINT64_C(1) << 40u) | ((uint64_t)trial_policy << 32u) |
                ((uint64_t)decision << 16u) | path_id;
            if (work->config->candidate_hook)
                work->config->candidate_hook(work->config->candidate_hook_user, sb, id,
                    distortion, rate, cost, 0);
            if (cost < best->cost || (cost == best->cost && id < best->id)) {
                *best = (RdBasicChoice){cost, distortion, rate, path_hash(work), id};
                winner = shortlist[rank];
                winner_policy = trial_policy;
                memcpy(work->best_ids, ids, work->leaf_count);
            }
        }
        if (winner_policy != policy) ++work->stats.motion_accepts;
        policy = winner_policy;
        work->mvs[leaf] = winner;
    }
    region_copy(work, 1);
    return SVT_AV2_ERROR_NONE;
}

static SvtAv2Error quarter_mv_sad(RdWork* work, const SvtAv2BlockView* block,
    SvtAv2MotionVector mv, uint64_t limit, uint64_t* sad) {
    const SvtAv2QuarterPelMotionVector quarter = {mv.mv_x, mv.mv_y};
#if SVT_AV2_INTER_BOUNDED_SAD
    const SvtAv2Error error = svt_av2_inter_luma_sad_quarter_sharp(work->source, work->reference,
        block, quarter, limit, sad);
    if (error == SVT_AV2_ERROR_NONE) ++work->stats.motion_candidates;
    return error;
#else
    (void)limit;
    const SvtAv2Error error = svt_av2_inter_predict_quarter_sharp(work->reference, 0u,
        block, quarter, work->prediction[0], block->w);
    if (error != SVT_AV2_ERROR_NONE) return error;
    ++work->stats.motion_candidates;
    const uint32_t step = work->source->bit_depth > 8u ? 2u : 1u;
    *sad = 0u;
    for (uint32_t y = 0u; y < block->h; ++y) {
        const uint8_t* row = work->source->planes[0] +
            (size_t)(block->y + y) * work->source->strides[0] + (size_t)block->x * step;
        for (uint32_t x = 0u; x < block->w; ++x) {
            const int d = (int)svt_av2_sample_load(row + x * step, work->source->bit_depth) -
                work->prediction[0][(size_t)y * block->w + x];
            *sad += (uint64_t)(d < 0 ? -d : d);
        }
    }
    return SVT_AV2_ERROR_NONE;
#endif
}

static SvtAv2Error projected_luma_sad(RdWork* work, const SvtAv2FrameBuffer* ref,
    const SvtAv2BlockView* b, SvtAv2MotionVector mv, uint64_t limit, uint64_t* sad) {
    ++work->stats.motion_candidates;
    const SvtAv2EighthPelMotionVector q3 = {mv.mv_x, mv.mv_y};
#if SVT_AV2_INTER_BOUNDED_SAD && SVT_AV2_INTER_BOUNDED_Q3
    return svt_av2_inter_luma_sad_eighth_sharp(work->source, ref, b, q3, limit, sad);
#else
#if SVT_AV2_INTER_BOUNDED_SAD
    if (mv.mv_x % 2 == 0 && mv.mv_y % 2 == 0) {
        const SvtAv2QuarterPelMotionVector q2 = {mv.mv_x / 2, mv.mv_y / 2};
        return svt_av2_inter_luma_sad_quarter_sharp(work->source, ref, b, q2, limit, sad);
    }
#endif
    const SvtAv2Error error = svt_av2_inter_predict_eighth_sharp(ref, 0u, b, q3,
        work->prediction[0], b->w);
    if (error != SVT_AV2_ERROR_NONE) return error;
    const uint32_t step = work->source->bit_depth > 8u ? 2u : 1u;
    uint64_t total = 0u;
    for (uint32_t y = 0u; y < b->h; ++y) {
        const uint8_t* src = work->source->planes[0] +
            (size_t)(b->y + y) * work->source->strides[0] + (size_t)b->x * step;
        const uint16_t* pred = work->prediction[0] + (size_t)y * b->w;
        for (uint32_t x = 0u; x < b->w; ++x) {
            const int d = (int)svt_av2_sample_load(src + x * step, work->source->bit_depth) - pred[x];
            total += (uint32_t)(d < 0 ? -d : d);
        }
        if (total > limit) { *sad = limit + 1u; return SVT_AV2_ERROR_NONE; }
    }
    *sad = total;
    return SVT_AV2_ERROR_NONE;
#endif
}

/* One optional LONG RD contender per coarse leaf. The luma screen uses a
 * co-located, DRL0 and time-scaled LAST proposal; it cannot change the winner
 * by itself. Only exact adaptive whole-SB replay can promote a candidate. */
static SvtAv2Error refine_long_ref(RdWork* work, size_t sb, uint16_t decision,
    RdBasicChoice* best) {
    if (work->leaf_count > 4u) return SVT_AV2_ERROR_INVALID_STATE;
    uint64_t d = 0u;
    uint32_t path = 0u;
    int legal = 0;
    SvtAv2Error error;
    SvtAv2TileBlockMode accepted[4];
    /* The incumbent already has exact SSE and syntax costs. Only its modes
     * are needed for spatial lookups here; defer pixels until a true RD trial
     * or final replay. */
    for (uint32_t leaf = 0u; leaf < work->leaf_count; ++leaf) {
        if (!rd_projected_mode(work, leaf, work->best_ids[leaf], &accepted[leaf]))
            return SVT_AV2_ERROR_INVALID_STATE;
        work->modes[leaf] = accepted[leaf];
    }
    for (uint32_t leaf = 0u; leaf < work->leaf_count; ++leaf) {
        const SvtAv2BlockView* b = &work->leaves[leaf];
        memcpy(work->modes, accepted, leaf * sizeof(*accepted));
        SvtAv2MotionVector last_pred, long_pred, last;
        if (!rd_motion_predict(work, leaf, 0u, &last_pred) || !rd_motion_predict(work, leaf, 1u, &long_pred) ||
            !rd_snap_motion((SvtAv2MotionVector){2 * work->mvs[leaf].mv_x, 2 * work->mvs[leaf].mv_y}, last_pred, &last))
            return SVT_AV2_ERROR_INVALID_STATE;
        SvtAv2MotionVector seed[3] = {{0,0}, long_pred, {0,0}};
        if (!rd_snap_motion(seed[0], long_pred, &seed[0])) continue; /* Empty MVD lattice intersection. */
        const uint32_t* dist = work->config->frame_header.ref_distance;
        if (!svt_av2_mvp_project_q3(last.mv_x, dist[1], dist[0], &seed[2].mv_x) ||
            !svt_av2_mvp_project_q3(last.mv_y, dist[1], dist[0], &seed[2].mv_y))
            return SVT_AV2_ERROR_INVALID_STATE;
        uint64_t base = 0u;
        error = projected_luma_sad(work, work->reference, b, last, UINT64_MAX, &base);
        if (error != SVT_AV2_ERROR_NONE) return error;
        /* A small slack leaves a chance for chroma/rate improvements. */
        uint64_t best_sad = base + (base >> 4u) + 1u;
        int have = 0;
        SvtAv2MotionVector selected = {0,0};
        for (uint32_t i = 0u; i < 3u; ++i) {
            if (!rd_snap_motion(seed[i], long_pred, &seed[i])) return SVT_AV2_ERROR_INVALID_STATE;
            int duplicate = 0;
            for (uint32_t j = 0u; j < i; ++j)
                duplicate |= seed[i].mv_x == seed[j].mv_x && seed[i].mv_y == seed[j].mv_y;
            if (duplicate) continue;
            uint64_t sad = 0u;
            error = projected_luma_sad(work, work->secondary, b, seed[i], best_sad - 1u, &sad);
            if (error != SVT_AV2_ERROR_NONE) return error;
            if (sad < best_sad) { selected = seed[i]; best_sad = sad; have = 1; }
            if (!best_sad) break;
        }
        if (!have) continue;
        /* Quarter cross on the exact MVD lattice. Opposite subpel steps
         * remain representable even when the actual predictor is odd. */
        const SvtAv2MotionVector anchor = selected;
        static const int8_t dx[4] = {-2,2,0,0}, dy[4] = {0,0,-2,2};
        for (uint32_t i = 0; i < 4u && best_sad; ++i) {
            const SvtAv2MotionVector desired = {anchor.mv_x + dx[i], anchor.mv_y + dy[i]};
            SvtAv2MotionVector q3;
            if (!rd_snap_motion(desired, long_pred, &q3)) return SVT_AV2_ERROR_INVALID_STATE;
            if (q3.mv_x == anchor.mv_x && q3.mv_y == anchor.mv_y) continue;
            uint64_t sad = 0u;
            error = projected_luma_sad(work, work->secondary, b, q3, best_sad - 1u, &sad);
            if (error != SVT_AV2_ERROR_NONE) return error;
            if (sad < best_sad) { selected = q3; best_sad = sad; }
        }
        const uint32_t incumbent = work->long_mask;
        work->long_mask |= 1u << leaf;
        work->q3_seed[leaf] = selected;
        uint8_t ids[4];
        memcpy(ids, work->best_ids, work->leaf_count);
        ids[leaf] = 3u;
        const ProjectedModeResult viable = preflight_projected_path(work, ids, sb, decision);
        if (viable == PROJECTED_MODE_EMPTY) { work->long_mask = incumbent; continue; }
        if (viable == PROJECTED_MODE_ERROR) { region_copy(work, 1); return SVT_AV2_ERROR_INVALID_STATE; }
        region_copy(work, 1);
        uint64_t r = 0u, cost = 0u;
        error = build_path(work, 0u, 0, 0, ids, &legal, &d, &path);
        if (error == SVT_AV2_ERROR_NONE && (!legal || RD_RATE(work, sb,
            decision, work->blocks, work->leaf_count * 3u, work->modes, work->leaf_count, 0, &r) != 0))
            error = SVT_AV2_ERROR_INVALID_STATE;
        if (error == SVT_AV2_ERROR_NONE) error = svt_av2_rd_cost(d, work->stats.lambda_q8, r, &cost);
        if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
        ++work->stats.candidate_count;
        const uint64_t id = (UINT64_C(1) << 41u) | ((uint64_t)work->long_mask << 32u) |
            ((uint64_t)decision << 16u) | path;
        if (work->config->candidate_hook)
            work->config->candidate_hook(work->config->candidate_hook_user, sb, id, d, r, cost, 0);
        if (cost < best->cost) {
            *best = (RdBasicChoice){cost, d, r, path_hash(work), id};
            memcpy(work->best_ids, ids, work->leaf_count);
            memcpy(accepted, work->modes, work->leaf_count * sizeof(*accepted));
        } else work->long_mask = incumbent;
    }
    region_copy(work, 1);
    return SVT_AV2_ERROR_NONE;
}

/* Bounded inter-only 32->16 proposal. Measure the coarse winner's MV on
 * each 32 quadrant, then spend independent ME only on the two worst. */
static SvtAv2Error fine_search_mv(RdWork* work, const SvtAv2BlockView* block,
    SvtAv2MotionVector parent, SvtAv2MotionVector* mv, uint64_t* sad) {
    SvtAv2MotionSearchResult integer;
    SvtAv2Error error = svt_av2_motion_estimate(work->source, work->reference, block, &integer);
    if (error != SVT_AV2_ERROR_NONE) return error;
    work->stats.motion_candidates += integer.candidates;
    SvtAv2MotionVector anchor = {4 * integer.mv.mv_x, 4 * integer.mv.mv_y};
    uint64_t anchor_sad = integer.sad;
    for (int32_t step = 2; step > 0; --step) {
        const SvtAv2MotionVector origin = anchor;
        for (int32_t y = -1; y <= 1; ++y) for (int32_t x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) continue;
            const SvtAv2MotionVector candidate = {origin.mv_x + step * x, origin.mv_y + step * y};
            uint64_t trial_sad;
            error = quarter_mv_sad(work, block, candidate, anchor_sad, &trial_sad);
            if (error != SVT_AV2_ERROR_NONE) return error;
            if (trial_sad < anchor_sad) { anchor_sad = trial_sad; anchor = candidate; }
        }
    }
    uint64_t parent_sad;
    error = quarter_mv_sad(work, block, parent, anchor_sad, &parent_sad);
    if (error != SVT_AV2_ERROR_NONE) return error;
    *mv = anchor_sad < parent_sad ? anchor : parent;
    *sad = SVT_AV2_MIN(anchor_sad, parent_sad);
    return SVT_AV2_ERROR_NONE;
}

typedef struct InterFinePath {
    uint32_t count, long_mask, q3_seed_mask;
    uint16_t decision;
    uint8_t ids[INTER_FINE_MAX_LEAVES];
    SvtAv2MotionVector mvs[INTER_FINE_MAX_LEAVES];
    SvtAv2MotionVector q3_seed[INTER_FINE_MAX_LEAVES];
    SvtAv2BlockView leaves[INTER_FINE_MAX_LEAVES];
} InterFinePath;

static void fine_save(InterFinePath* path, const RdWork* work, uint16_t decision, const uint8_t* ids) {
    path->count = work->leaf_count;
    path->decision = decision;
    path->long_mask = work->long_mask;
    path->q3_seed_mask = work->q3_seed_mask;
    memcpy(path->ids, ids, path->count);
    memcpy(path->mvs, work->mvs, path->count * sizeof(*path->mvs));
    memcpy(path->q3_seed, work->q3_seed, path->count * sizeof(*path->q3_seed));
    memcpy(path->leaves, work->leaves, path->count * sizeof(*path->leaves));
}

static void fine_restore(RdWork* work, const InterFinePath* path) {
    work->leaf_count = path->count;
    work->long_mask = path->long_mask;
    work->q3_seed_mask = path->q3_seed_mask;
    memcpy(work->best_ids, path->ids, path->count);
    memcpy(work->mvs, path->mvs, path->count * sizeof(*path->mvs));
    memcpy(work->q3_seed, path->q3_seed, path->count * sizeof(*path->q3_seed));
    memcpy(work->leaves, path->leaves, path->count * sizeof(*path->leaves));
}

/* Screen on the incumbent parent's actual Q3 lattice and selected reference.
 * Exact replay later resnaps every raw proposal against its own fine prefix. */
static SvtAv2Error fine_search_projected(RdWork* work, const SvtAv2FrameBuffer* ref,
    const SvtAv2BlockView* block, SvtAv2MotionVector parent, SvtAv2MotionVector* mv, uint64_t* sad) {
    SvtAv2MotionSearchResult integer;
    SvtAv2Error error = svt_av2_motion_estimate(work->source, ref, block, &integer);
    if (error != SVT_AV2_ERROR_NONE) return error;
    if (work->wide_inter_me) svt_av2_motion_extend(work->source, ref, block, &integer);
    work->stats.motion_candidates += integer.candidates;
    *mv = parent;
    error = projected_luma_sad(work, ref, block, parent, UINT64_MAX, sad);
    if (error != SVT_AV2_ERROR_NONE || *sad == 0u) return error;
    SvtAv2MotionVector anchor;
    const SvtAv2MotionVector raw = {8 * integer.mv.mv_x, 8 * integer.mv.mv_y};
    if (!rd_snap_motion(raw, parent, &anchor)) return SVT_AV2_ERROR_INVALID_STATE;
    uint64_t cost;
    error = projected_luma_sad(work, ref, block, anchor, UINT64_MAX, &cost);
    if (error != SVT_AV2_ERROR_NONE) return error;
    for (int32_t step = 4; step >= 2 && cost > 0u; step /= 2) {
        const SvtAv2MotionVector origin = anchor;
        for (int32_t y = -1; y <= 1 && cost > 0u; ++y) for (int32_t x = -1; x <= 1 && cost > 0u; ++x) {
            if (!x && !y) continue;
            const SvtAv2MotionVector offset = {origin.mv_x + step * x, origin.mv_y + step * y};
            SvtAv2MotionVector trial;
            if (!rd_snap_motion(offset, parent, &trial)) return SVT_AV2_ERROR_INVALID_STATE;
            if (trial.mv_x == anchor.mv_x && trial.mv_y == anchor.mv_y) continue;
            uint64_t v;
            error = projected_luma_sad(work, ref, block, trial, cost - 1u, &v);
            if (error != SVT_AV2_ERROR_NONE) return error;
            if (v < cost) { cost = v; anchor = trial; }
        }
    }
    if (cost < *sad) { *sad = cost; *mv = anchor; }
    return SVT_AV2_ERROR_NONE;
}

static SvtAv2Error refine_inter_fine(RdWork* work, size_t sb, uint16_t* decision,
    RdBasicChoice* best) {
    SVT_AV2_IP_START(profile_fine_screen);
    if (*decision > 1u || work->leaf_count != (*decision ? 4u : 1u))
        return SVT_AV2_ERROR_INVALID_STATE;
    InterFinePath winner = {0};
    fine_save(&winner, work, *decision, work->best_ids);
    const int projected = work->secondary != NULL;
    SvtAv2TileBlockMode parent_motion[4];
    if (projected) for (uint32_t i = 0u; i < work->leaf_count; ++i) {
        if (!rd_projected_mode(work, i, 3u, &parent_motion[i])) return SVT_AV2_ERROR_INVALID_STATE;
        /* When scoring intra as a parent, use its virtual LAST seed for the
         * inter screen, but keep the intra cell in downstream predictors. */
        work->modes[i] = work->best_ids[i] == 3u ? parent_motion[i] : (SvtAv2TileBlockMode){0};
    }
    const InterFinePath coarse = winner;
    SvtAv2MotionVector parent_mv[4], parent_actual[4], child_mv[4][4];
    const SvtAv2FrameBuffer* parent_ref[4] = {0};
    uint8_t parent_id[4];
    SvtAv2BlockView quadrant[4];
    uint64_t coarse_sad[4];
    const uint32_t x0 = work->region[0].x, y0 = work->region[0].y;
    for (uint32_t q = 0u; q < 4u; ++q) {
        const uint32_t parent = *decision ? q : 0u;
        parent_mv[q] = work->mvs[parent]; parent_id[q] = work->best_ids[parent];
        quadrant[q] = work->region[0];
        quadrant[q].x = x0 + (q % 2u) * 32u; quadrant[q].y = y0 + (q / 2u) * 32u;
        quadrant[q].w = quadrant[q].h = 32u;
        if (projected) {
            parent_actual[q] = (SvtAv2MotionVector){parent_motion[parent].mv_x, parent_motion[parent].mv_y};
            parent_ref[q] = parent_motion[parent].reference_rank ? work->secondary : work->reference;
        }
        SvtAv2Error error = projected ? projected_luma_sad(work, parent_ref[q], &quadrant[q],
            parent_actual[q], UINT64_MAX, &coarse_sad[q]) :
            quarter_mv_sad(work, &quadrant[q], parent_mv[q], UINT64_MAX, &coarse_sad[q]);
        if (error != SVT_AV2_ERROR_NONE) return error;
    }
    uint32_t masks[4], mask = 0u, mask_count = 0u;
    const uint64_t min_sad = UINT64_C(32) * 32u * (2u << (work->source->bit_depth - 8u));
    for (uint32_t pass = 0u; pass < (work->full_inter_fine ? 4u : 2u); ++pass) {
        uint32_t q = 0u;
        for (uint32_t i = 1u; i < 4u; ++i) if (coarse_sad[i] > coarse_sad[q]) q = i;
        if (coarse_sad[q] < min_sad) break;
        uint64_t refined_sad = 0u;
        for (uint32_t i = 0u; i < 4u; ++i) {
            SvtAv2BlockView child = quadrant[q];
            child.x += (i % 2u) * 16u; child.y += (i / 2u) * 16u;
            child.w = child.h = 16u;
            uint64_t sad;
            SvtAv2Error error = projected ? fine_search_projected(work, parent_ref[q], &child,
                parent_actual[q], &child_mv[q][i], &sad) :
                fine_search_mv(work, &child, parent_mv[q], &child_mv[q][i], &sad);
            if (error != SVT_AV2_ERROR_NONE) return error;
            refined_sad += sad;
        }
        if (refined_sad < coarse_sad[q] || work->full_inter_fine) {
            mask |= 1u << q;
            masks[mask_count++] = mask;
        }
        coarse_sad[q] = 0u;
    }
    SVT_AV2_IP_ADD(work->profile->fine_screen, profile_fine_screen);
    for (uint32_t k = 0u; k < mask_count; ++k) {
        const uint32_t m = masks[work->config->reverse_candidates ? mask_count - 1u - k : k];
        /* Ternary square plan: digits 0/1 select 32x32/16x16. */
        uint16_t plan = 4u;
        static const uint16_t power[4] = {1u, 3u, 9u, 27u};
        for (uint32_t q = 0u; q < 4u; ++q) if (m & (1u << q)) plan += power[q];
        work->output->decisions[sb] = plan;
        work->leaf_count = 0u;
        SvtAv2Error error = svt_av2_partition_traverse_sb_plan(&work->geometry, sb,
            work->output->decisions, work->output->decision_count, collect_leaf, work);
        if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
        uint32_t expected_leaves = 4u;
        for (uint32_t q = 0u; q < 4u; ++q) if (m & (1u << q)) expected_leaves += 3u;
        if (work->leaf_count != expected_leaves) {
            region_copy(work, 1);
            return SVT_AV2_ERROR_INVALID_STATE;
        }
        uint8_t ids[INTER_FINE_MAX_LEAVES];
        work->long_mask = work->q3_seed_mask = 0u;
        for (uint32_t i = 0u; i < work->leaf_count; ++i) {
            const uint32_t x = work->leaves[i].x - x0, y = work->leaves[i].y - y0;
            const uint32_t q = (y / 32u) * 2u + x / 32u;
            const uint32_t c = (y % 32u / 16u) * 2u + x % 32u / 16u;
            const int split = (m & (1u << q)) != 0u;
            ids[i] = split ? 3u : parent_id[q];
            work->mvs[i] = !projected && split ? child_mv[q][c] : parent_mv[q];
            if (projected && ids[i] == 3u) {
                const uint32_t parent = coarse.decision ? q : 0u;
                const uint32_t bit = 1u << i;
                if (parent_motion[parent].reference_rank) work->long_mask |= bit;
                if (split || ((coarse.long_mask | coarse.q3_seed_mask) & (1u << parent))) {
                    work->q3_seed_mask |= bit;
                    work->q3_seed[i] = split ? child_mv[q][c] : coarse.q3_seed[parent];
                }
            }
        }
        if (projected) {
            const ProjectedModeResult viable = preflight_projected_path(work, ids, sb, plan);
            if (viable == PROJECTED_MODE_EMPTY) continue;
            if (viable == PROJECTED_MODE_ERROR) { region_copy(work, 1); return SVT_AV2_ERROR_INVALID_STATE; }
        }
        region_copy(work, 1);
        uint64_t d = 0u, r = 0u, cost = 0u;
        uint32_t ignored_path;
        int legal;
        error = build_path(work, 0u, 0, 0, ids, &legal, &d, &ignored_path);
        if (error == SVT_AV2_ERROR_NONE && (!legal || RD_RATE(work, sb, plan,
                work->blocks, work->leaf_count * 3u, work->modes, work->leaf_count, 0, &r) != 0))
            error = SVT_AV2_ERROR_INVALID_STATE;
        if (error == SVT_AV2_ERROR_NONE) error = svt_av2_rd_cost(d, work->stats.lambda_q8, r, &cost);
        if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
        ++work->stats.candidate_count; ++work->stats.inter_fine_trials;
        const uint64_t id = (UINT64_C(1) << 42u) | plan;
        if (work->config->candidate_hook)
            work->config->candidate_hook(work->config->candidate_hook_user, sb, id, d, r, cost, 0);
        if (cost < best->cost || (cost == best->cost && id < best->id)) {
            *best = (RdBasicChoice){cost, d, r, path_hash(work), id};
            fine_save(&winner, work, plan, ids);
        }
    }
    if (mask_count != 0u) {
        region_copy(work, 1);
        if (winner.decision != *decision) ++work->stats.inter_fine_accepts;
        *decision = winner.decision;
        work->output->decisions[sb] = winner.decision;
        fine_restore(work, &winner);
    }
    return SVT_AV2_ERROR_NONE;
}

/* Give the canonical DRL0 predictor one chance beyond the SAD shortlist.
 * Earlier positions in the raster-coordinate pass are fixed. A win can
 * propagate an odd Q3 lattice to later leaves, so their targets must be
 * resolved afresh from raw seeds; never reuse a speculative entropy state. */
static SvtAv2Error refine_inter_near(RdWork* work, size_t sb, uint16_t decision,
    RdBasicChoice* best) {
    enum { max_trials = 2 };
    const int projected = work->secondary != NULL;
    const int32_t limit = projected ? 128 : 64;
    uint32_t tried = 0u, won = 0u;
    for (uint32_t leaf = 0u; leaf < work->leaf_count && tried < max_trials; ++leaf) {
        SvtAv2TileBlockMode incumbent;
        if (!rd_inter_mode(work, leaf, work->best_ids[leaf], &incumbent))
            return SVT_AV2_ERROR_INVALID_STATE;
        work->modes[leaf] = incumbent;
        const SvtAv2BlockView* b = &work->leaves[leaf];
        if (!incumbent.is_inter || b->w != b->h ||
            (b->w != 64u && b->w != 32u && b->w != 16u) || b->x % b->w || b->y % b->h)
            continue;
        SvtAv2MotionVector pred;
        if (!rd_motion_predict(work, leaf, incumbent.reference_rank, &pred))
            return SVT_AV2_ERROR_INVALID_STATE;
        if ((pred.mv_x == incumbent.mv_x && pred.mv_y == incumbent.mv_y) ||
            pred.mv_x < -limit || pred.mv_x > limit || pred.mv_y < -limit || pred.mv_y > limit)
            continue;
        ++tried;
        const uint32_t bit = 1u << leaf;
        const uint32_t saved_mask = work->q3_seed_mask;
        SvtAv2MotionVector saved;
        if (projected) {
            saved = work->q3_seed[leaf];
            work->q3_seed[leaf] = pred;
            work->q3_seed_mask |= bit;
        } else {
            saved = work->mvs[leaf];
            work->mvs[leaf] = pred;
        }
        const ProjectedModeResult valid = projected ?
            preflight_projected_path(work, work->best_ids, sb, decision) : PROJECTED_MODE_OK;
        if (valid == PROJECTED_MODE_ERROR) return SVT_AV2_ERROR_INVALID_STATE;
        if (valid == PROJECTED_MODE_EMPTY) ++work->near_empty;
        else {
            region_copy(work, 1);
            uint64_t distortion, rate, cost;
            uint32_t path_id;
            int legal;
            SvtAv2Error error = build_path(work, 0u, 0, 0, work->best_ids,
                &legal, &distortion, &path_id);
            if (error != SVT_AV2_ERROR_NONE) return error;
            const SvtAv2TileBlockMode* mode = &work->modes[leaf];
            if (!legal || !mode->is_inter || mode->reference_rank != incumbent.reference_rank ||
                mode->mv_x != pred.mv_x || mode->mv_y != pred.mv_y ||
                RD_RATE(work, sb, decision, work->blocks, work->leaf_count * 3u,
                    work->modes, work->leaf_count, 0, &rate) != 0)
                return SVT_AV2_ERROR_INVALID_STATE;
            error = svt_av2_rd_cost(distortion, work->stats.lambda_q8, rate, &cost);
            if (error != SVT_AV2_ERROR_NONE) return error;
            ++work->near_trials;
            ++work->stats.candidate_count;
            const uint64_t id = (UINT64_C(1) << 44u) | ((uint64_t)(won | bit) << 24u) |
                ((uint64_t)decision << 8u) | leaf;
            if (work->config->candidate_hook)
                work->config->candidate_hook(work->config->candidate_hook_user, sb, id, distortion, rate, cost, 0);
            /* Keep the incumbent on equality; this raster pass has one target per leaf. */
            if (cost < best->cost) {
                *best = (RdBasicChoice){cost, distortion, rate, path_hash(work), id};
                won |= bit;
                ++work->near_wins[incumbent.reference_rank];
                if (projected && (pred.mv_x % 2 || pred.mv_y % 2)) ++work->near_odd;
                continue;
            }
        }
        if (projected) {
            work->q3_seed[leaf] = saved;
            work->q3_seed_mask = saved_mask;
        } else work->mvs[leaf] = saved;
        work->modes[leaf] = incumbent;
    }
    region_copy(work, 1);
    return SVT_AV2_ERROR_NONE;
}

/* One last raster-coordinate pass, with the winning tree and motion seeds
 * fixed. A zero-residual inter leaf may replace either normal inter or intra.
 * The exact prefix is retained, but its causal suffix must be reconstructed
 * and the whole SB rated: later intra predictors and CDFs can change. */
static SvtAv2Error refine_inter_skip(RdWork* work, size_t sb, uint16_t decision,
    RdBasicChoice* best) {
    SvtAv2Error error = prepare_prefix(work, 0u);
    if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
    work->prefix_active = 1u;
    uint32_t winner_mask = 0u;
    for (uint32_t leaf = 0u; leaf < work->leaf_count; ++leaf) {
        error = extend_prefix(work, leaf);
        if (error != SVT_AV2_ERROR_NONE) break;
        if (work->late_inter_skip && work->best_ids[leaf] != 3u) continue;
        uint8_t ids[INTER_FINE_MAX_LEAVES];
        memcpy(ids, work->best_ids, work->leaf_count);
        ids[leaf] = 3u;
        work->inter_skip_mask = winner_mask | (1u << leaf);
        uint64_t d = 0u, r = 0u, cost = 0u;
        uint32_t path_id;
        int legal;
        error = build_path(work, 0u, 0, 0, ids, &legal, &d, &path_id);
        if (error == SVT_AV2_ERROR_NONE && (!legal ||
            RD_RATE(work, sb, decision, work->blocks, work->leaf_count * 3u,
                work->modes, work->leaf_count, 0, &r) != 0))
            error = SVT_AV2_ERROR_INVALID_STATE;
        if (error == SVT_AV2_ERROR_NONE) error = svt_av2_rd_cost(d, work->stats.lambda_q8, r, &cost);
        if (error != SVT_AV2_ERROR_NONE) break;
        ++work->stats.candidate_count;
        ++work->stats.inter_skip_trials;
        const uint64_t id = (UINT64_C(1) << 41u) | ((uint64_t)work->inter_skip_mask << 32u) |
            ((uint64_t)decision << 16u) | path_id;
        if (work->config->candidate_hook != NULL)
            work->config->candidate_hook(work->config->candidate_hook_user, sb, id, d, r, cost, 0);
        /* IDs of strict extensions increase, so ties retain the incumbent. */
        if (cost < best->cost || (cost == best->cost && id < best->id)) {
            *best = (RdBasicChoice){cost, d, r, path_hash(work), id};
            memcpy(work->best_ids, ids, work->leaf_count);
            winner_mask = work->inter_skip_mask;
            ++work->stats.inter_skip_accepts;
        }
        work->inter_skip_mask = winner_mask;
    }
    work->prefix_active = 0u;
    work->inter_skip_mask = winner_mask;
    region_copy(work, 1);
    return error;
}

#if SVT_AV2_EXPERIMENTAL_INTRA_EOB
static SvtAv2Error replay_intra_eob(RdWork* work, size_t sb, uint16_t decision,
    uint32_t limit, uint64_t* d, uint64_t* r) {
    region_copy(work, 1);
    work->intra_eob_limit = limit;
    work->intra_eob_edits = 0u;
    uint32_t path_id;
    int legal;
    const SvtAv2Error error = build_path(work, 0u, 0, 0, work->best_ids, &legal, d, &path_id);
    work->intra_eob_limit = 0u;
    if (error != SVT_AV2_ERROR_NONE) return error;
    if (!legal || RD_RATE(work, sb, decision, work->blocks, work->leaf_count * 3u,
            work->reference ? work->modes : NULL, work->reference ? work->leaf_count : 0u,
            0, r) != 0)
        return SVT_AV2_ERROR_INVALID_STATE;
    return SVT_AV2_ERROR_NONE;
}

/* A bounded AVM-style EOB proposal, not a port of its coefficient trellis.
 * Rebuild from the same SB checkpoint so later intra prediction (including
 * CfL) sees the changed pixels. Only the ordinary tile transaction commits. */
static SvtAv2Error refine_intra_eob(RdWork* work, size_t sb, uint16_t decision,
    RdBasicChoice* best, uint64_t* rate) {
    if (work->prefix_active || work->memo_active || work->uv_y_cache)
        return SVT_AV2_ERROR_INVALID_STATE;
    int eligible = 0;
    for (uint32_t leaf = 0u; !eligible && leaf < work->leaf_count; ++leaf) {
        if (work->modes[leaf].is_inter) continue;
        for (uint32_t p = 0u; !eligible && p < 3u; ++p) {
            const SvtAv2TileCoeffBlock* b = &work->blocks[3u * leaf + p];
            uint16_t positions[SVT_AV2_INTER_TAIL_MAX];
            eligible = intra_weak_eob(b->qcoeff, b->width, b->height, b->tx_type, positions) != 0u;
        }
    }
    if (!eligible) return SVT_AV2_ERROR_NONE;

    const uint64_t orig_r = best->rate;
    const uint64_t lambda = (work->stats.lambda_q8 + 2u) >> 2u;
    uint64_t best_j;
    SvtAv2Error error = svt_av2_rd_cost(best->distortion, lambda, best->rate, &best_j);
    if (error != SVT_AV2_ERROR_NONE) return error;
    const uint32_t trials[] = {1u, SVT_AV2_INTER_TAIL_MAX};
    uint32_t winning_limit = 0u, winning_edits = 0u;
    for (size_t i = 0u; i < sizeof(trials) / sizeof(*trials); ++i) {
        uint64_t d = 0u, r = 0u, j;
        error = replay_intra_eob(work, sb, decision, trials[i], &d, &r);
        if (error != SVT_AV2_ERROR_NONE) return error;
        ++work->intra_eob_trials;
        ++work->stats.candidate_count;
        if (r >= orig_r) continue;
        error = svt_av2_rd_cost(d, lambda, r, &j);
        if (error != SVT_AV2_ERROR_NONE) return error;
        if (j >= best_j) continue;
        error = svt_av2_rd_cost(d, work->stats.lambda_q8, r, &best->cost);
        if (error != SVT_AV2_ERROR_NONE) return error;
        best_j = j;
        best->distortion = d;
        best->rate = r;
        best->hash = path_hash(work);
        winning_limit = trials[i];
        winning_edits = work->intra_eob_edits;
    }
    uint64_t replay_d = 0u;
    error = replay_intra_eob(work, sb, decision, winning_limit, &replay_d, rate);
    if (error != SVT_AV2_ERROR_NONE) return error;
    if (replay_d != best->distortion || *rate != best->rate || path_hash(work) != best->hash ||
        work->intra_eob_edits != winning_edits)
        return SVT_AV2_ERROR_INVALID_STATE;
    if (winning_limit) {
        ++work->intra_eob_wins;
        work->intra_eob_levels += winning_edits;
        work->intra_eob_rate_saved += orig_r - best->rate;
    }
    return SVT_AV2_ERROR_NONE;
}
#endif

static void trace_fine_winner(RdWork* work) {
    uint32_t inter = 0u, intra = 0u;
    for (uint32_t i = 0u; i < work->leaf_count; ++i) {
        const SvtAv2TileBlockMode* mode = &work->modes[i];
        if (!mode->is_inter) { ++intra; continue; }
        const SvtAv2BlockView* leaf = &work->leaves[i];
        if (leaf->w != 16u || leaf->h != 16u) continue;
        const uint32_t rank = mode->reference_rank;
        ++inter;
        ++work->fine_inter16[rank];
        work->fine_odd16[rank] += (mode->mv_x % 2 != 0 || mode->mv_y % 2 != 0);
        int nonzero = 0;
        for (uint32_t p = 0u; p < 3u && !nonzero; ++p) {
            const SvtAv2TileCoeffBlock* block = &work->blocks[3u * i + p];
            for (uint32_t k = 0u; k < block->width * block->height && !nonzero; ++k)
                nonzero = block->qcoeff[k] != 0;
        }
        work->fine_nonzero16[rank] += (uint32_t)nonzero;
    }
    if (inter && intra) ++work->fine_mixed_sb;
}

/* The independently rebuilt incumbent is in work->blocks. Propose at most
 * two nested paths, luma and YUV (or chroma alone), then independently replay
 * the best. Intra suffix samples and arithmetic/CDF state must be recomputed
 * causally; a local coefficient-rate estimate cannot authorize the edit. */
static SvtAv2Error refine_inter_tail(RdWork* work, size_t sb, uint16_t decision,
    RdBasicChoice* best, uint64_t* final_rate) {
    RdInterTail* t = work->tail;
    if (work->leaf_count > INTER_FINE_MAX_LEAVES || t->mask || work->prefix_active)
        return SVT_AV2_ERROR_INVALID_STATE;
    uint64_t sparse_planes[2] = {0, 0};
    uint8_t nonzero[INTER_FINE_MAX_LEAVES * 3u] = {0};
    uint64_t masks[2] = {0, 0};
    for (uint32_t leaf = 0u; leaf < work->leaf_count; ++leaf) {
        if (work->best_ids[leaf] != 3u) continue;
        for (uint32_t plane = 0u; plane < 3u; ++plane) {
            const uint32_t k = 3u * leaf + plane;
            const SvtAv2TileCoeffBlock* b = &work->blocks[k];
            if (b->tx_type || b->width != b->height || b->width < 8u || b->width > 64u) continue;
            const uint32_t side = b->width >= 32u ? 32u : b->width;
            if (side != 8u && side != 16u && side != 32u) continue;
            const uint32_t s = side == 8u ? 0u : side == 16u ? 1u : 2u;
            if (t->try_sparse_planes) {
                const uint32_t cap = 1u + side * side / 64u;
                unsigned energy = 0u;
                for (uint32_t i = 0; i < side * side && energy <= cap; ++i) {
                    const int32_t level = b->qcoeff[i];
                    energy += (unsigned)abs(level);
                    nonzero[k] += level != 0;
                }
                if (energy && energy <= cap) sparse_planes[plane != 0u] |= UINT64_C(1) << k;
            }
            const uint32_t n = svt_av2_inter_tail_positions(b->qcoeff, t->scans[s], side * side, t->positions[k]);
            if (!n) continue;
            t->counts[k] = (uint8_t)n;
            masks[plane != 0u] |= UINT64_C(1) << k;
        }
    }
    const uint64_t candidates[2] = {masks[0], masks[0] | masks[1]};
    if (!candidates[1] && !sparse_planes[0] && !sparse_planes[1]) return SVT_AV2_ERROR_NONE;
    const uint64_t original_rate = best->rate;
    const uint64_t lambda = (work->stats.lambda_q8 + 2u) >> 2u;
    uint64_t best_j;
    SvtAv2Error error = svt_av2_rd_cost(best->distortion, lambda, best->rate, &best_j);
    if (error != SVT_AV2_ERROR_NONE) return error;
    uint64_t winner = 0u, zero_winner = 0u;
    for (uint32_t c = 0u; c < (t->try_sparse_planes ? 4u : 2u); ++c) {
        const uint64_t zero = c < 2u ? 0u : zero_winner | sparse_planes[c - 2u];
        const uint64_t trim = c < 2u ? candidates[c] : winner & ~zero;
        if ((trim == winner && zero == zero_winner) || (c == 1u && candidates[1] == candidates[0])) continue;
        region_copy(work, 1);
        t->mask = trim;
        t->zero_mask = zero;
        uint64_t d = 0u, r = 0u, j = 0u;
        uint32_t path_id;
        int legal;
        error = build_path(work, 0u, 0, 0, work->best_ids, &legal, &d, &path_id);
        if (error != SVT_AV2_ERROR_NONE) return error;
        if (!legal || RD_RATE(work, sb, decision, work->blocks,
            work->leaf_count * 3u, work->modes, work->leaf_count, 0, &r) != 0)
            return SVT_AV2_ERROR_INVALID_STATE;
        ++t->trials;
        ++work->stats.candidate_count;
        error = svt_av2_rd_cost(d, lambda, r, &j);
        if (error != SVT_AV2_ERROR_NONE) return error;
        /* Compare both proposals at the conservative lambda. Require an
         * actual reduction in the adaptive coder's total Q12 cost. */
        if (r < original_rate && j < best_j) {
            error = svt_av2_rd_cost(d, work->stats.lambda_q8, r, &best->cost);
            if (error != SVT_AV2_ERROR_NONE) return error;
            best->distortion = d; best->rate = r; best->hash = path_hash(work);
            best_j = j;
            winner = trim;
            zero_winner = zero;
        }
    }
    region_copy(work, 1);
    t->mask = winner;
    t->zero_mask = zero_winner;
    uint64_t replay_d = 0u;
    uint32_t replay_path;
    int replay_legal;
    error = build_path(work, 0u, 0, 0, work->best_ids, &replay_legal, &replay_d, &replay_path);
    if (error != SVT_AV2_ERROR_NONE) return error;
    if (!replay_legal || replay_d != best->distortion || path_hash(work) != best->hash ||
        RD_RATE(work, sb, decision, work->blocks,
            work->leaf_count * 3u, work->modes, work->leaf_count, 0, final_rate) != 0 || *final_rate != best->rate)
        return SVT_AV2_ERROR_INVALID_STATE;
    if (winner || zero_winner) {
        ++t->accepts;
        t->rate_saved_q12 += original_rate - best->rate;
        for (uint32_t leaf = 0u; leaf < work->leaf_count; ++leaf) {
            if (!((winner | zero_winner) & (UINT64_C(7) << (3u * leaf)))) continue;
            const uint32_t side = work->leaves[leaf].w;
            ++t->leaves[side == 16u ? 0u : side == 32u ? 1u : 2u];
            for (uint32_t p = 0u; p < 3u; ++p) {
                const uint32_t k = leaf * 3u + p;
                if (!((winner | zero_winner) & (UINT64_C(1) << k))) continue;
                t->coeffs[p] += (zero_winner & (UINT64_C(1) << k)) ? nonzero[k] : t->counts[k];
                const SvtAv2TileCoeffBlock* b = &work->blocks[k];
                const size_t n = (size_t)SVT_AV2_MIN(b->width, 32u) * SVT_AV2_MIN(b->height, 32u);
                size_t i = 0u;
                while (i < n && !b->qcoeff[i]) ++i;
                t->zero_tbs += i == n;
            }
        }
    }
    return SVT_AV2_ERROR_NONE;
}

static SvtAv2Error replay_sb_choice(RdWork* work,size_t sb,uint16_t decision,RdBasicChoice* trial) {
    uint32_t path;
    int legal;
    region_copy(work,1);
    SvtAv2Error error=build_path(work,0u,0,0,work->best_ids,&legal,&trial->distortion,&path);
    if (error!=SVT_AV2_ERROR_NONE) return error;
    if (!legal || RD_RATE(work,sb,decision,work->blocks,work->leaf_count*3u,
        work->reference?work->modes:NULL,work->reference?work->leaf_count:0u,0,&trial->rate)!=0)
        return SVT_AV2_ERROR_INVALID_STATE;
    trial->hash=path_hash(work);
    return svt_av2_rd_cost(trial->distortion,work->stats.lambda_q8,trial->rate,&trial->cost);
}

static SvtAv2Error refine_ist_leaf(RdWork* work, size_t sb, uint16_t decision,
    uint32_t leaf, uint32_t primary, RdBasicChoice* best) {
    const uint32_t first = primary == 3u ? 22u : 1u;
    const uint32_t count = primary == 3u ? 12u : 21u;
    const uint64_t entering_cost = best->cost, entering_rate = best->rate;
    uint8_t winner = 0u;
    for (uint32_t i = 0u; i < count; ++i) {
        const uint32_t code = first + (work->config->reverse_candidates ? count - 1u - i : i);
        work->ist[leaf] = (uint8_t)code;
        RdBasicChoice trial = *best;
        SvtAv2Error error = replay_sb_choice(work, sb, decision, &trial);
        if (error != SVT_AV2_ERROR_NONE) return error;
        ++work->ist_trials;
        if (!work->blocks[leaf * 3u].ist) continue;
        trial.id = (UINT64_C(1) << 48u) | ((uint64_t)decision << 16u) | (leaf << 6u) | code;
        ++work->stats.candidate_count;
        if (work->config->candidate_hook)
            work->config->candidate_hook(work->config->candidate_hook_user, sb, trial.id,
                trial.distortion, trial.rate, trial.cost, 0);
        if ((work->ist_all_leaves || trial.rate < entering_rate) && trial.cost < entering_cost &&
            (trial.cost < best->cost || (trial.cost == best->cost && code < winner))) {
            winner = (uint8_t)code;
            *best = trial;
        }
    }
    work->ist[leaf] = winner;
    RdBasicChoice replay = *best;
    SvtAv2Error error = replay_sb_choice(work, sb, decision, &replay);
    if (error != SVT_AV2_ERROR_NONE) return error;
    if (replay.rate != best->rate || replay.distortion != best->distortion ||
        replay.hash != best->hash || replay.cost != best->cost ||
        work->blocks[leaf * 3u].ist != winner)
        return SVT_AV2_ERROR_INVALID_STATE;
    work->ist_wins += winner != 0u;
    return SVT_AV2_ERROR_NONE;
}

static SvtAv2Error refine_ist(RdWork* work, size_t sb, uint16_t decision, RdBasicChoice* best) {
    uint32_t leaf = UINT32_MAX, primary = 0u;
    uint64_t strength = 0u;
    for (uint32_t i = 0u; i < work->leaf_count; ++i) {
        const SvtAv2TileCoeffBlock* b = work->blocks + i * 3u;
        const uint32_t limit = svt_av2_ist_eob_limit(b->width, b->height, b->tx_type);
        if (work->modes[i].is_inter || b->width < 8u || b->height < 8u || !limit ||
            rd_ist_mode(work->ids[i], &work->leaves[i], work->mrl[i]) >= 12u)
            continue;
        SvtAv2TxSize size;
        uint16_t scan[1024];
        if (svt_av2_coeff_tx_size_from_dims(b->width, b->height, &size) != 0)
            return SVT_AV2_ERROR_INVALID_STATE;
        const size_t count = svt_av2_coeff_scan_dct_dct(size, scan);
        const uint32_t eob = svt_av2_coeff_find_eob(b->qcoeff, scan, count);
        if (eob < 2u || eob > 2u * limit) continue;
        uint64_t score = 0u;
        for (uint32_t j = 1u; j < eob; ++j) {
            const int32_t level = b->qcoeff[scan[j]];
            score += (uint64_t)(level < 0 ? -(int64_t)level : level);
        }
        if (work->ist_all_leaves) {
            if (!score) continue;
            SvtAv2Error error = refine_ist_leaf(work, sb, decision, i, b->tx_type, best);
            if (error != SVT_AV2_ERROR_NONE) return error;
        } else if (score > strength) {
            strength = score;
            leaf = i;
            primary = b->tx_type;
        }
    }
    if (leaf == UINT32_MAX) return SVT_AV2_ERROR_NONE;

    return refine_ist_leaf(work, sb, decision, leaf, primary, best);
}

#if SVT_AV2_EXPERIMENTAL_VIDEO_CFL
/* Bounded P/KEY probe: never changes luma mode, partition, motion or refs.
 * Rebuild the entire causal SB for each proposal, including prior M74 edits.
 * This does not invoke the still-only prefix or the coarse ordinal cache. */
static SvtAv2Error refine_video_cfl(RdWork* work,size_t sb,uint16_t decision,
    RdBasicChoice* best,uint64_t* rate) {
    uint32_t leaf=UINT32_MAX;
    uint64_t strength=0u;
    for (uint32_t i=0u;i<work->leaf_count;++i) {
        if (work->modes[i].is_inter) continue;
        uint64_t score=0u;
        for (uint32_t p=1u;p<3u;++p) {
            const SvtAv2TileCoeffBlock* b=work->blocks+i*3u+p;
            const size_t n=(size_t)SVT_AV2_MIN(b->width,32u)*SVT_AV2_MIN(b->height,32u);
            for (size_t k=0u;k<n;++k) {
                const int32_t q=b->qcoeff[k];
                score+=(uint64_t)(q<0?-(int64_t)q:q);
            }
        }
        if (score>strength) {strength=score;leaf=i;}
    }
    if (leaf==UINT32_MAX) return SVT_AV2_ERROR_NONE;
    SvtAv2BlockView blocks[3];
    SvtAv2Error error=plane_blocks(work,leaf,blocks);
    int8_t seed[2][2]={{0,0},{0,0}};
    if (error==SVT_AV2_ERROR_NONE) error=rank_cfl_alphas(work,blocks+1u,1u,seed[0]);
    if (error==SVT_AV2_ERROR_NONE) error=rank_cfl_alphas(work,blocks+2u,2u,seed[1]);
    if (error!=SVT_AV2_ERROR_NONE) return error;
    const RdLeafTools entering=leaf_tools_get(work,leaf);
    RdLeafTools winner=entering;
    const uint64_t orig_rate=best->rate;
    const int8_t u[7]={0,seed[0][0],seed[0][0],seed[0][1],seed[0][1],0,seed[0][0]};
    const int8_t v[7]={0,seed[1][0],seed[1][1],seed[1][0],seed[1][1],seed[1][0],0};
    uint8_t visited[17][17]={{0}};
    for (uint32_t i=0u;i<7u;++i) {
        const uint32_t k=work->config->reverse_candidates?6u-i:i;
        RdLeafTools proposal=entering;
        if (!k) {proposal.uv=14u;proposal.explicit_cfl=proposal.mh=proposal.signs=proposal.idx=0u;}
        else {
            if ((!u[k] && !v[k]) || visited[u[k]+8][v[k]+8]) continue;
            visited[u[k]+8][v[k]+8]=1u;
            cfl_set_explicit_pair(&proposal,u[k],v[k]);
        }
        leaf_tools_set(work,leaf,&proposal);
        RdBasicChoice trial=*best;
        error=replay_sb_choice(work,sb,decision,&trial);
        if (error!=SVT_AV2_ERROR_NONE) return error;
        trial.id=(UINT64_C(1)<<47u)|((uint64_t)decision<<24u)|((uint64_t)leaf<<10u)|
            ((uint64_t)proposal.explicit_cfl<<9u)|((uint64_t)proposal.signs<<6u)|proposal.idx;
        ++work->stats.candidate_count;++work->video_cfl_trials;
        if (work->config->candidate_hook)
            work->config->candidate_hook(work->config->candidate_hook_user,sb,trial.id,
                trial.distortion,trial.rate,trial.cost,0);
        if (trial.rate<orig_rate && (trial.cost<best->cost ||
            (trial.cost==best->cost && trial.id<best->id))) {
            *best=trial;winner=leaf_tools_get(work,leaf);
        }
    }
    leaf_tools_set(work,leaf,&winner);
    RdBasicChoice replay=*best;
    error=replay_sb_choice(work,sb,decision,&replay);
    if (error!=SVT_AV2_ERROR_NONE) return error;
    if (replay.rate!=best->rate || replay.distortion!=best->distortion || replay.hash!=best->hash ||
        replay.cost!=best->cost) return SVT_AV2_ERROR_INVALID_STATE;
    *rate=replay.rate;
    if (winner.uv==14u) {
        ++work->video_cfl_wins;work->video_cfl_implicit+=!winner.explicit_cfl;
        work->video_cfl_saved_q12+=orig_rate-best->rate;
    }
    return SVT_AV2_ERROR_NONE;
}
#endif

/* Try the still binary trees in video. Preserve the other three quadrants
 * and project the parent motion seed onto eligible square children. New
 * rectangular leaves rank their intra modes against the causal pixels. */
static SvtAv2Error refine_video_rect(RdWork* work, size_t sb, uint16_t* decision, RdBasicChoice* best) {
    const uint16_t original = *decision;
    if (work->leaf_count > INTER_FINE_MAX_LEAVES || work->inter_skip_mask) return SVT_AV2_ERROR_NONE;
    InterFinePath incumbent, winner;
    fine_save(&incumbent, work, original, work->best_ids);
    winner = incumbent;
    uint32_t inter_count = 0u, parent[4] = {0};
    for (uint32_t i = 0u; i < incumbent.count; ++i) {
        inter_count += incumbent.ids[i] == 3u;
        const SvtAv2BlockView* b = &incumbent.leaves[i];
        parent[(b->y % 64u / 32u) * 2u + b->x % 64u / 32u] = i;
    }
    if (inter_count && original != 1u) return SVT_AV2_ERROR_NONE;
    const uint32_t others = 4u * (1u + 7u + 49u); /* Three intact 32x32 quadrants. */
    SvtAv2Error error = SVT_AV2_ERROR_NONE;
    work->rank_intra_only = 1;
    for (uint32_t index = 0u; index < 50u; ++index) {
        const uint32_t arm = work->config->reverse_candidates ? 49u - index : index;
        if (inter_count && arm < 2u) continue;
        const uint16_t plan = (uint16_t)(arm < 2u ? 87u + arm : arm < 18u ?
            SVT_AV2_HALF_RECT_BASE + (arm - 2u) * 343u + others :
            SVT_AV2_RECT16_BASE + (arm - 18u) * 343u + others);
        work->output->decisions[sb] = plan;
        work->leaf_count = 0u;
        error = svt_av2_partition_traverse_sb_plan(&work->geometry, sb, work->output->decisions,
            work->output->decision_count, collect_leaf, work);
        if (error != SVT_AV2_ERROR_NONE) break;
        uint8_t ids[INTER_FINE_MAX_LEAVES];
        if (inter_count) {
            work->long_mask = work->q3_seed_mask = 0u;
            for (uint32_t i = 0u; i < work->leaf_count; ++i) {
                const SvtAv2BlockView* b = &work->leaves[i];
                const uint32_t old = parent[(b->y % 64u / 32u) * 2u + b->x % 64u / 32u];
                const int intact = b->w == 32u && b->h == 32u;
                const int motion_child = work->config->inter_fine && b->w == 16u && b->h == 16u &&
                    incumbent.ids[old] == 3u;
                ids[i] = intact || motion_child ? incumbent.ids[old] : UINT8_MAX;
                work->mvs[i] = incumbent.mvs[old];
                work->q3_seed[i] = incumbent.q3_seed[old];
                if (ids[i] == 3u) {
                    work->long_mask |= ((incumbent.long_mask >> old) & 1u) << i;
                    work->q3_seed_mask |= ((incumbent.q3_seed_mask >> old) & 1u) << i;
                }
            }
            const ProjectedModeResult state = preflight_projected_path(work, ids, sb, plan);
            if (state == PROJECTED_MODE_EMPTY) continue;
            if (state == PROJECTED_MODE_ERROR) { error = SVT_AV2_ERROR_INVALID_STATE; break; }
        }
        region_copy(work, 1);
        uint64_t d = 0u, r = 0u, j = 0u;
        uint32_t path;
        int legal;
        error = build_path(work, 0u, 1, 1, inter_count ? ids : NULL, &legal, &d, &path);
        if (error != SVT_AV2_ERROR_NONE) break;
        if (!legal || RD_RATE(work, sb, plan, work->blocks, work->leaf_count * 3u,
            work->reference ? work->modes : NULL, work->reference ? work->leaf_count : 0u, 0, &r)) {
            error = SVT_AV2_ERROR_INVALID_STATE;
            break;
        }
        error = svt_av2_rd_cost(d, work->stats.lambda_q8, r, &j);
        if (error != SVT_AV2_ERROR_NONE) break;
        ++work->video_rect_trials;
        ++work->stats.candidate_count;
        if (j < best->cost || (j == best->cost && *decision != original && plan < *decision)) {
            *best = (RdBasicChoice){j, d, r, path_hash(work), plan_candidate_id(plan, 0u, 0u)};
            *decision = plan;
            fine_save(&winner, work, plan, work->ids);
        }
    }
    work->rank_intra_only = 0;
    region_copy(work, 1);
    work->output->decisions[sb] = *decision;
    fine_restore(work, &winner);
    work->video_rect_wins += *decision != original;
    return error;
}

static double ssim_luma_variance(const SvtAv2FrameBuffer* src, uint32_t bx, uint32_t by) {
    const uint32_t bd = src->bit_depth, step = bd == 8u ? 1u : 2u;
    uint32_t sum = 0u, squares = 0u, n = 0u;
    for (uint32_t y = by; y < SVT_AV2_MIN(by + 8u, src->visible_height); ++y) {
        const uint8_t* row = src->planes[0] + (size_t)y * src->strides[0];
        for (uint32_t x = bx; x < SVT_AV2_MIN(bx + 8u, src->visible_width); ++x) {
            const uint32_t sample = svt_av2_sample_load(row + x * step, bd);
            sum += sample;
            squares += sample * sample;
            ++n;
        }
    }
    const double mean = (double)sum / n;
    return floor(((double)squares / n - mean * mean) / (1u << (2u * (bd - 8u))) + 0.5);
}

/* AVM SSIM tuning fits a scale per 16x16 from the 8x8 variances. SVT
 * shares an RD multiplier across every candidate of the same superblock. */
static void ssim_sb_scales(const SvtAv2FrameBuffer* src, uint32_t cols, uint32_t rows, double* scales) {
    double frame_log = 0.0;
    uint32_t frame_cells = 0u;
    for (uint32_t sy = 0u; sy < rows; ++sy) for (uint32_t sx = 0u; sx < cols; ++sx) {
        double sb_log = 0.0;
        uint32_t cells = 0u;
        const uint32_t end_x = SVT_AV2_MIN(64u * (sx + 1u), src->visible_width);
        const uint32_t end_y = SVT_AV2_MIN(64u * (sy + 1u), src->visible_height);
        for (uint32_t y = sy * 64u; y < end_y; y += 16u) for (uint32_t x = sx * 64u; x < end_x; x += 16u) {
            double variance = 0.0;
            uint32_t blocks = 0u;
            for (uint32_t by = y; by < SVT_AV2_MIN(end_y, y + 16u); by += 8u)
                for (uint32_t bx = x; bx < SVT_AV2_MIN(end_x, x + 16u); bx += 8u) {
                    variance += ssim_luma_variance(src, bx, by);
                    ++blocks;
                }
            sb_log += log(67.035434 * (1.0 - exp(-0.0021489 * variance / blocks)) + 17.492222);
            ++cells;
        }
        scales[(size_t)sy * cols + sx] = sb_log / cells;
        frame_log += sb_log;
        frame_cells += cells;
    }
    frame_log /= frame_cells;
    for (size_t sb = 0u; sb < (size_t)cols * rows; ++sb) scales[sb] = exp(scales[sb] - frame_log);
}

/* Score seven uniform quadrant shapes, then replay their best local choices
 * together so that the mixed tree pays its actual causal prediction and rate. */
static SvtAv2Error screen_still_trees(RdWork* work, size_t sb, uint16_t* decision, RdBasicChoice* best) {
    uint64_t quadrant_cost[4] = {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
    uint32_t shape[4] = {0};
    RdTreeSeed seeds[4];
    uint32_t seed_count = 0;
    uint64_t cell_cost[2][4][4] = {0};
    uint64_t square_cost[4] = {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
    uint32_t square_shape[4] = {0};
    static const uint16_t long_plans[] = {87u, 88u, 91u, 92u};
    for (unsigned trial = 0; trial < 10u + SVT_AV2_ARRAY_SIZE(long_plans); ++trial) {
        uint16_t plan = 0;
        if (trial > 0 && trial < 8)
            plan = (uint16_t)(SVT_AV2_MIXED_RECT32_BASE + (trial - 1) * 400u);
        if (trial == 8) {
            plan = (uint16_t)(SVT_AV2_MIXED_RECT32_BASE + shape[0] + 7u * shape[1] +
                49u * shape[2] + 343u * shape[3]);
            if (shape[0] == shape[1] && shape[1] == shape[2] && shape[2] == shape[3]) continue;
        }
        if (trial == 9) {
#if M6_FINE16_GEOMETRY
            uint64_t gain = 0;
            for (unsigned q = 0; q < 4; ++q) {
                uint64_t mixed = 0;
                uint32_t mask = 0;
                for (unsigned c = 0; c < 4; ++c) {
                    const unsigned split = cell_cost[1][q][c] < cell_cost[0][q][c];
                    mask |= split << c;
                    mixed += cell_cost[split][q][c];
                }
                if (!mask || mask == 15u || mixed >= quadrant_cost[q]) continue;
                uint64_t saving = quadrant_cost[q] - mixed;
                uint32_t others = 0;
                for (unsigned k = 4; k-- > 0;) if (k != q) {
                    others = others * 5u + square_shape[k];
                    const uint64_t penalty = square_cost[k] - quadrant_cost[k];
                    saving -= SVT_AV2_MIN(saving, penalty);
                }
                if (saving > gain) {
                    gain = saving;
                    plan = (uint16_t)(4096u + (q * 14u + mask - 1u) * 125u + others);
                }
            }
            if (!gain) continue;
#else
            continue;
#endif
        }
        if (trial >= 10) plan = long_plans[trial - 10];
        SvtAv2Error error = geometry_for_sb(work, sb, plan);
        if (error != SVT_AV2_ERROR_NONE) return error;
        size_t count;
        if (svt_av2_tile_rate_prefix_begin(work->basic_rate_prefix, work->rate, sb, plan, &count) ||
            count != work->leaf_count) return SVT_AV2_ERROR_INVALID_STATE;
        region_copy(work, 1);
        work->coeff_count = 0;
        uint64_t total_sse[3] = {0}, local_cost[4] = {0}, rate = 0;
        for (uint32_t leaf = 0; leaf < count; ++leaf) {
            SvtAv2BlockView blocks[3];
            error = plane_blocks(work, leaf, blocks);
            if (error != SVT_AV2_ERROR_NONE) return error;
            uint64_t lowest = UINT64_MAX;
            uint32_t mode = 0;
            for (uint32_t m = 0; m < 14; ++m) {
                if (!legal_mode(work, leaf, m)) continue;
                uint64_t sad[3];
                error = predict(work, leaf, m, blocks, sad, 1);
                if (error != SVT_AV2_ERROR_NONE) return error;
                if (sad[0] < lowest) { lowest = sad[0]; mode = m; }
            }
            work->ids[leaf] = (uint8_t)mode;
            uint64_t sse[3] = {0};
            error = reconstruct_leaf(work, leaf, mode, blocks, sse);
            if (error != SVT_AV2_ERROR_NONE) return error;
            uint64_t previous_rate = rate;
            if (svt_av2_tile_rate_prefix_trial(work->basic_rate_prefix, leaf,
                work->blocks + 3u * leaf, 1, &rate)) return SVT_AV2_ERROR_INVALID_STATE;
            for (unsigned p = 0; p < 3; ++p) total_sse[p] += sse[p];
            uint64_t cost;
            error = svt_av2_rd_cost(svt_av2_rd_distortion(sse, work->source->bit_depth,
                work->source->chroma_format), work->stats.lambda_q8, rate - previous_rate, &cost);
            if (error != SVT_AV2_ERROR_NONE) return error;
            const unsigned q = (blocks[0].y % 64u / 32u) * 2u + blocks[0].x % 64u / 32u;
            local_cost[q] += cost;
            if (trial == 1 || trial == 2) {
                const unsigned cell = (blocks[0].y % 32u / 16u) * 2u + blocks[0].x % 32u / 16u;
                cell_cost[trial - 1][q][cell] += cost;
            }
        }
        if (svt_av2_tile_rate_prefix_finish(work->basic_rate_prefix, &rate))
            return SVT_AV2_ERROR_INVALID_STATE;
        if (trial > 0 && trial < 8) {
            for (unsigned i = 0; i < 4; ++i) {
                if (trial <= 5 && local_cost[i] < square_cost[i]) {
                    square_cost[i] = local_cost[i];
                    square_shape[i] = trial - 1;
                }
                if (local_cost[i] < quadrant_cost[i]) {
                    shape[i] = trial - 1;
                    quadrant_cost[i] = local_cost[i];
                }
            }
        }
        const uint64_t d = svt_av2_rd_distortion(total_sse, work->source->bit_depth, work->source->chroma_format);
        uint64_t cost;
        error = svt_av2_rd_cost(d, work->stats.lambda_q8, rate, &cost);
        if (error != SVT_AV2_ERROR_NONE) return error;
        ++work->stats.candidate_count;
        if (work->config->candidate_hook)
            work->config->candidate_hook(work->config->candidate_hook_user, sb, plan, d, rate, cost, 0);
        if (seed_count < 2 || cost <= seeds[1].choice.cost) {
            const RdBasicChoice choice = {cost, d, rate, path_hash(work), plan};
            memcpy(work->best_ids, work->ids, count);
            offer_tree(work, plan, &choice, seeds, &seed_count);
        }
        /* At less than four coded bytes there is little rate to recover by splitting. */
        if (trial == 0 && rate < 32u * 4096u) break;
    }
    RdTreeSeed winner = seeds[0];
    uint8_t alternate[RD_MAX_LEAVES];
    const uint64_t cutoff = winner.choice.cost + winner.choice.cost / 5u;
    for (uint32_t i = 0; i < seed_count; ++i) {
        RdTreeSeed* seed = &seeds[i];
        if (seed->choice.cost > cutoff) break;
        SvtAv2Error error = geometry_for_sb(work, sb, seed->decision);
        if (error != SVT_AV2_ERROR_NONE) return error;
        memcpy(work->best_ids, seed->ids, seed->leaf_count);
        memcpy(work->uv_ids, seed->uv, seed->leaf_count);
        error = refine_basic(work, sb, 0, seed->decision, &seed->choice);
        if (error != SVT_AV2_ERROR_NONE) return error;
        if (!i || seed->choice.cost < winner.choice.cost) {
            winner = *seed;
            memcpy(winner.ids, work->best_ids, seed->leaf_count);
            memcpy(winner.uv, work->uv_ids, seed->leaf_count);
            memcpy(alternate, work->alternate_y, seed->leaf_count);
        }
    }
    *decision = winner.decision;
    *best = winner.choice;
    memcpy(work->best_ids, winner.ids, winner.leaf_count);
    memcpy(work->uv_ids, winner.uv, winner.leaf_count);
    memcpy(work->alternate_y, alternate, winner.leaf_count);
    return SVT_AV2_ERROR_NONE;
}

static SvtAv2Error encode_sb(RdWork* work, size_t sb) {
    SVT_AV2_IP_START(profile_coarse);
    memset(work->ist, 0, sizeof(work->ist));
/* Transform input is position-independent: preserve exact cache across SBs.
     * A new frame owns a fresh zero-initialized cache with immutable transform state. */
    if (work->basic_memo) {memset(work->basic_memo->slots,0,sizeof(work->basic_memo->slots));work->basic_memo->count=0u;}
    work->coeff_policy = 0u; work->motion_policy = 0u; work->wide_stage = 0u;
    work->inter_skip_mask = 0u;
    if (work->tail) work->tail->mask = work->tail->zero_mask = 0u;
    work->long_mask = work->q3_seed_mask = 0u;
#if SVT_AV2_INTER_PIXEL_REUSE
    if (work->inter_pixel != NULL) {
        memset(work->inter_pixel->used, 0, sizeof(work->inter_pixel->used));
        memset(work->inter_pixel->next, 0, sizeof(work->inter_pixel->next));
    }
#endif
#if SVT_AV2_INTER_ME_REUSE
    work->integer_seed[0].count = work->integer_seed[1].count = 0u;
#endif
    memset(work->mrl,0,sizeof(work->mrl));
    memset(work->mhccp,0,sizeof(work->mhccp));
    memset(work->rdoq_saved, 0, sizeof(work->rdoq_saved));
    work->rdoq_target = UINT32_MAX;
    memset(work->uv_ids, 0, sizeof(work->uv_ids));
    memset(work->tx_types, 0, sizeof(work->tx_types));
    memset(work->cfl_explicit,0,sizeof(work->cfl_explicit));
    memset(work->cfl_alpha_idx,0,sizeof(work->cfl_alpha_idx));
    memset(work->cfl_alpha_signs,0,sizeof(work->cfl_alpha_signs));
    SvtAv2Error error = svt_av2_block_from_sb(&work->geometry,
        (uint32_t)(sb % work->geometry.sb_cols), (uint32_t)(sb / work->geometry.sb_cols), &work->region[0]);
    if (error != SVT_AV2_ERROR_NONE) return error;
    /* This is a visible backup rectangle, not a coded TX shape. Partial SB
     * dimensions need direct subsampling rather than the coded-shape table. */
    work->region[1] = work->region[0];
    if (work->source->chroma_format == SVT_AV2_CHROMA_420) {
        work->region[1].x /= 2u; work->region[1].y /= 2u;
        work->region[1].w /= 2u; work->region[1].h /= 2u;
    }
    work->region[2] = work->region[1];
    const int partial = work->region[0].w != 64u || work->region[0].h != 64u;
    if (partial) ++work->stats.partial_superblocks; else ++work->stats.full_superblocks;
    region_copy(work, 0);
    uint64_t best_cost = UINT64_MAX, best_d = 0u, best_r = 0u, best_hash = 0u;
    uint64_t best_id = UINT64_MAX; uint16_t best_decision = 0u;
    RdTreeSeed finalists[10], reserved[2], before_thin[2], thin64[2], rect32_seed, half_rect_seed, rect16_seed;
    uint32_t finalist_count=0u, reserved_count=0u, before_thin_count=0u, thin64_count=0u;
    rect32_seed.choice.cost = UINT64_MAX;
    half_rect_seed.choice.cost = rect16_seed.choice.cost = UINT64_MAX;
#if M6_ROOT_ASYM
    RdTreeSeed before_asym[2];
    uint32_t before_asym_count=0u;
#endif
    const int screen = work->config->screen_trees && !partial;
    if (screen) {
        RdBasicChoice best = {UINT64_MAX, 0, 0, 0, 0};
        error = screen_still_trees(work, sb, &best_decision, &best);
        if (error != SVT_AV2_ERROR_NONE) return error;
        best_cost = best.cost; best_d = best.distortion; best_r = best.rate;
        best_hash = best.hash; best_id = best.id;
    }
    const uint32_t tree_count = screen ? 0u : !partial && work->config->partition_selection ? 2u : 1u;
    for (uint32_t decision = 0u; decision < tree_count; ++decision) {
        error = geometry_for_sb(work, sb, (uint16_t)decision);
        if (error != SVT_AV2_ERROR_NONE) break;
        const int pruned = decision != 0u && work->config->rd_mode >= 2u;
        const uint32_t paths = partial ? 1u : 1u << ((pruned ? 1u : 2u) * work->leaf_count);
#if SVT_AV2_INTER_COARSE_PREFIX
        const int use_coarse_prefix = decision == 1u && pruned && !partial && work->leaf_count == 4u &&
            work->coarse_prefix != NULL;
        if (use_coarse_prefix) {
            RdCoarsePrefix* cache = work->coarse_prefix;
            cache->rank_valid = cache->leaf_valid = 0u;
            ++cache->trees;
#if SVT_AV2_INTER_COARSE_RATE_PREFIX
            if (cache->rate_cache != NULL) {
                SVT_AV2_IP_START(rate_setup);
                if (svt_av2_tile_inter_rate_cache_begin(cache->rate_cache, work->rate, sb) != 0)
                    return SVT_AV2_ERROR_INVALID_STATE;
                SVT_AV2_IP_ADD(work->profile->split_rate_setup, rate_setup);
                ++cache->rate_trees;
            }
#endif
            for (uint32_t leaf = 0u; leaf < 4u; ++leaf) {
                error = plane_blocks(work, leaf, cache->geometry[leaf]);
                if (error != SVT_AV2_ERROR_NONE) return error;
                for (uint32_t p = 0u; p < 3u; ++p) {
                    const SvtAv2BlockView* b = &cache->geometry[leaf][p];
                    const uint32_t side = p && work->source->chroma_format == SVT_AV2_CHROMA_420 ? 16u : 32u;
                    if (b->w != side || b->h != side) return SVT_AV2_ERROR_INVALID_STATE;
                }
            }
        }
#endif
        for (uint32_t index = 0u; index < paths; ++index) {
            const uint32_t ordinal = work->config->reverse_candidates ? paths - 1u - index : index;
            region_copy(work, 1);
            uint64_t distortion = 0u, rate = 0u, cost; uint32_t path_id; int legal;
#if SVT_AV2_INTER_COARSE_PREFIX
            if (use_coarse_prefix) {
                error = coarse_prefix_path(work, ordinal, &legal, &distortion, &path_id);
#if SVT_AV2_INTER_COARSE_PREFIX_VERIFY
                if (error == SVT_AV2_ERROR_NONE)
                    error = verify_coarse_prefix(work, sb, ordinal, legal, distortion, path_id);
#endif
            } else
#endif
                error = build_path(work, ordinal, pruned, partial, NULL, &legal, &distortion, &path_id);
            if (error != SVT_AV2_ERROR_NONE) break;
            if (!legal) continue;
            int rate_status;
#if SVT_AV2_INTER_COARSE_PREFIX
            if (use_coarse_prefix) {
                SVT_AV2_IP_START(split_rate);
#if SVT_AV2_INTER_COARSE_RATE_PREFIX
                if (work->coarse_prefix->rate_cache != NULL) {
                    rate_status = svt_av2_tile_inter_rate_cache_trial(work->coarse_prefix->rate_cache,
                        ordinal, work->blocks, work->modes, &rate);
                    SVT_AV2_IP_ADD(work->profile->rate_trials, split_rate);
                    if (rate_status == 0) ++work->coarse_prefix->rate_candidates;
                } else
#endif
                    rate_status = RD_RATE(work, sb, 1u, work->blocks, 12u, work->modes, 4u, 0, &rate);
                SVT_AV2_IP_ADD(work->profile->split_rate, split_rate);
#if SVT_AV2_INTER_COARSE_RATE_PREFIX && SVT_AV2_INTER_COARSE_RATE_PREFIX_VERIFY
                if (work->coarse_prefix->rate_cache != NULL) {
                    uint64_t expected = 0u;
                    const int oracle = svt_av2_tile_rate_trial(work->rate, sb, 1u, work->blocks,
                        12u, work->modes, 4u, 0, &expected);
                    if (rate_status != oracle || rate != expected) {
                        fprintf(stderr, "inter_rate_prefix_mismatch hint=%u sb=%zu ordinal=%u cache=%llu oracle=%llu status=%d/%d\n",
                            work->config->frame_header.order_hint, sb, ordinal, (unsigned long long)rate,
                            (unsigned long long)expected, rate_status, oracle);
                        rate_status = SVT_AV2_ENTROPY_ERROR_RANGE;
                    }
                }
#endif
            } else
#endif
                rate_status = RD_RATE(work, sb, (uint16_t)decision, work->blocks,
                    work->leaf_count * 3u, work->reference == NULL ? NULL : work->modes,
                    work->reference == NULL ? 0u : work->leaf_count, 0, &rate);
            if (rate_status != 0) { error = SVT_AV2_ERROR_INVALID_STATE; break; }
            error = svt_av2_rd_cost(distortion, work->stats.lambda_q8, rate, &cost);
            if (error != SVT_AV2_ERROR_NONE) break;
            ++work->stats.candidate_count;
            const uint64_t candidate_id = decision *
                (work->config->frame_header.m5_intra_modes >= 3u ? 65536u :
                 work->config->frame_header.m5_intra_modes == 2u ? 4096u : 256u) + path_id;
            if (work->config->candidate_hook != NULL)
                work->config->candidate_hook(work->config->candidate_hook_user, sb, candidate_id,
                    distortion, rate, cost, 0);
            if (cost < best_cost || (cost == best_cost && candidate_id < best_id)) {
                best_cost = cost; best_d = distortion; best_r = rate; best_id = candidate_id;
                best_decision = (uint16_t)decision; best_hash = path_hash(work);
                memcpy(work->best_ids, work->ids, work->leaf_count);
            }
        }
        if (error != SVT_AV2_ERROR_NONE) break;
    }
    region_copy(work, 1);
    if (error != SVT_AV2_ERROR_NONE || best_id == UINT64_MAX)
        return error != SVT_AV2_ERROR_NONE ? error : SVT_AV2_ERROR_INVALID_STATE;
    error = geometry_for_sb(work, sb, best_decision);
    if (error != SVT_AV2_ERROR_NONE) return error;
    if (!partial && work->reference != NULL && work->config->motion_optimization &&
        !work->config->frame_header.mv_half) {
        uint32_t best_policy = 0u;
        for (uint32_t leaf = 0u; leaf < work->leaf_count; ++leaf) {
            const uint8_t entering_mode = work->best_ids[leaf];
            const uint32_t entering_slot = (best_policy >> (2u * leaf)) & 3u;
            for (uint32_t index = 0u; index < work->mv_lists[leaf].count; ++index) {
                const uint32_t slot = work->config->reverse_candidates ? work->mv_lists[leaf].count - 1u - index : index;
                if (entering_mode == 3u && entering_slot == slot) continue;
                work->motion_policy = (best_policy & ~(3u << (2u * leaf))) | (slot << (2u * leaf));
                for (uint32_t i = 0u; i < work->leaf_count; ++i)
                    work->mvs[i] = work->mv_lists[i].mv[(work->motion_policy >> (2u * i)) & 3u];
                uint8_t trial_ids[4]; memcpy(trial_ids, work->best_ids, work->leaf_count);
                trial_ids[leaf] = 3u;
                region_copy(work, 1);
                uint64_t trial_d = 0u, trial_r = 0u, trial_cost = 0u;
                uint32_t trial_path; int trial_legal;
                error = build_path(work, 0u, 0, 0, trial_ids, &trial_legal, &trial_d, &trial_path);
                if (error == SVT_AV2_ERROR_NONE && (!trial_legal ||
                    RD_RATE(work, sb, best_decision, work->blocks,
                        work->leaf_count * 3u, work->modes, work->leaf_count, 0, &trial_r) != 0))
                    error = SVT_AV2_ERROR_INVALID_STATE;
                if (error == SVT_AV2_ERROR_NONE)
                    error = svt_av2_rd_cost(trial_d, work->stats.lambda_q8, trial_r, &trial_cost);
                if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
                ++work->stats.candidate_count; ++work->stats.motion_trials;
                const uint64_t trial_id = (uint32_t)best_decision * 256u + trial_path + (work->motion_policy << 21u);
                if (work->config->candidate_hook != NULL)
                    work->config->candidate_hook(work->config->candidate_hook_user, sb, trial_id,
                        trial_d, trial_r, trial_cost, 0);
                if (trial_cost < best_cost || (trial_cost == best_cost && trial_id < best_id)) {
                    best_cost = trial_cost; best_d = trial_d; best_r = trial_r; best_id = trial_id;
                    best_policy = work->motion_policy; best_hash = path_hash(work);
                    memcpy(work->best_ids, trial_ids, work->leaf_count);
                    ++work->stats.motion_accepts;
                }
            }
        }
        work->motion_policy = best_policy;
        for (uint32_t i = 0u; i < work->leaf_count; ++i)
            work->mvs[i] = work->mv_lists[i].mv[(best_policy >> (2u * i)) & 3u];
        region_copy(work, 1);
    }
    if (!partial && work->reference != NULL && work->config->motion_optimization == 2u) {
        SvtAv2MotionSearchList wide[4];
        SvtAv2MotionVector incumbent[4];
        memcpy(incumbent, work->mvs, work->leaf_count * sizeof(*incumbent));
        const uint32_t incumbent_policy = work->motion_policy;
        uint32_t best_policy = 0u; int accepted = 0;
        for (uint32_t i = 0u; i < work->leaf_count; ++i) {
            error = svt_av2_motion_estimate_wide(work->source, work->reference, &work->leaves[i], &wide[i]);
            if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
            work->stats.motion_candidates += wide[i].candidates;
        }
        for (uint32_t leaf = 0u; leaf < work->leaf_count; ++leaf) {
            for (uint32_t index = 0u; index < wide[leaf].count; ++index) {
                const uint32_t slot = 1u + (work->config->reverse_candidates ? wide[leaf].count - 1u - index : index);
                work->motion_policy = (best_policy & ~(3u << (2u * leaf))) | (slot << (2u * leaf));
                work->wide_stage = 1u;
                for (uint32_t i = 0u; i < work->leaf_count; ++i) {
                    const uint32_t choice = (work->motion_policy >> (2u * i)) & 3u;
                    work->mvs[i] = choice == 0u ? incumbent[i] : wide[i].mv[choice - 1u];
                }
                uint8_t trial_ids[4]; memcpy(trial_ids, work->best_ids, work->leaf_count);
                trial_ids[leaf] = 3u;
                region_copy(work, 1);
                uint64_t trial_d = 0u, trial_r = 0u, trial_cost = 0u;
                uint32_t trial_path; int trial_legal;
                error = build_path(work, 0u, 0, 0, trial_ids, &trial_legal, &trial_d, &trial_path);
                if (error == SVT_AV2_ERROR_NONE && (!trial_legal ||
                    RD_RATE(work, sb, best_decision, work->blocks,
                        work->leaf_count * 3u, work->modes, work->leaf_count, 0, &trial_r) != 0))
                    error = SVT_AV2_ERROR_INVALID_STATE;
                if (error == SVT_AV2_ERROR_NONE)
                    error = svt_av2_rd_cost(trial_d, work->stats.lambda_q8, trial_r, &trial_cost);
                if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
                ++work->stats.candidate_count; ++work->stats.wide_trials;
                const uint64_t trial_id = (1u << 29u) | ((uint32_t)best_decision * 256u + trial_path) | (work->motion_policy << 21u);
                if (work->config->candidate_hook != NULL)
                    work->config->candidate_hook(work->config->candidate_hook_user, sb, trial_id,
                        trial_d, trial_r, trial_cost, 0);
                if (trial_cost < best_cost || (trial_cost == best_cost && trial_id < best_id)) {
                    best_cost = trial_cost; best_d = trial_d; best_r = trial_r; best_id = trial_id;
                    best_policy = work->motion_policy; best_hash = path_hash(work); accepted = 1;
                    memcpy(work->best_ids, trial_ids, work->leaf_count); ++work->stats.wide_accepts;
                }
            }
        }
        work->wide_stage = accepted ? 1u : 0u;
        work->motion_policy = accepted ? best_policy : incumbent_policy;
        for (uint32_t i = 0u; i < work->leaf_count; ++i) {
            const uint32_t choice = (best_policy >> (2u * i)) & 3u;
            work->mvs[i] = choice == 0u ? incumbent[i] : wide[i].mv[choice - 1u];
        }
        region_copy(work, 1);
    }
    if (!screen && work->config->frame_header.m5_intra_modes >= 2u) {
        RdBasicChoice best = {best_cost, best_d, best_r, best_hash, best_id};
        error = refine_basic(work, sb, partial, best_decision, &best);
        if (error != SVT_AV2_ERROR_NONE) return error;
        best_cost = best.cost; best_d = best.distortion; best_r = best.rate;
        best_hash = best.hash; best_id = best.id;
        if (work->config->tree_finalists && !partial)
            offer_tree(work,best_decision,&best,finalists,&finalist_count);
    }
    if (!screen && !partial && work->config->partition_selection >= 2u) {
        uint8_t saved_ids[RD_MAX_LEAVES], saved_uv[RD_MAX_LEAVES];
        const uint16_t initial_decision = best_decision;
        const uint64_t initial_cost = best_cost;
        memcpy(saved_ids, work->best_ids, sizeof(saved_ids));
        memcpy(saved_uv, work->uv_ids, sizeof(saved_uv));
        uint32_t mixed_codes[4] = {0u, 0u, 0u, 0u};
        uint64_t mixed_costs[4] = {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
        const uint32_t tree_trials = work->config->partition_selection == 10u ? 68u :
            work->config->partition_selection == 9u ? 66u :
            work->config->partition_selection == 8u ? 58u :
            work->config->partition_selection == 7u ? (M6_ROOT_ASYM ? 60u : 56u) :
            work->config->partition_selection == 6u ? 40u :
            work->config->partition_selection == 5u ? (M6_ROOT_ASYM ? 28u : 24u) :
            work->config->partition_selection == 4u ? 18u :
            work->config->partition_selection == 3u ? 16u : 3u;
        /* Start the rectangular quadrant sweep from the best mixed8 tree. */
        const uint32_t rect_trials = work->rect32_search == 3u ? 56u : work->rect32_search == 2u ? 24u : work->rect32_search ? 8u : 0u;
        uint32_t rect_code = 0u;
        uint64_t rect_cost = UINT64_MAX;
        for (uint32_t index = 0u; index < tree_trials + rect_trials; ++index) {
#if M6_ROOT_ASYM
            if ((work->config->partition_selection==5u || work->config->partition_selection==7u) &&
                index==(work->config->partition_selection==7u?56u:24u)) {
                before_asym_count=SVT_AV2_MIN(finalist_count,2u);
                memcpy(before_asym,finalists,before_asym_count*sizeof(*before_asym));
            }
#endif
            if (index==18u && work->config->tree_finalists>=4u) {
                reserved_count=SVT_AV2_MIN(finalist_count,2u);
                memcpy(reserved,finalists,reserved_count*sizeof(*reserved));
            }
            if (index==56u && work->config->tree_finalists>=8u) {
                before_thin_count=SVT_AV2_MIN(finalist_count,2u);
                memcpy(before_thin,finalists,before_thin_count*sizeof(*before_thin));
            }
            if (index==56u) {
                /* Seed the thin-shape pass from the completed old mixed8 domain. */
                uint32_t digits=mixed_codes[2],power=1u;
                for (uint32_t q=0u;q<4u;++q) {
                    mixed_codes[3]+=(digits%5u)*power;digits/=5u;power*=7u;
                }
                mixed_costs[3]=mixed_costs[2];
            }
            uint16_t decision = (uint16_t)index;
            uint32_t family = 0u, mixed_code = 0u;
            const int rect_trial = index >= tree_trials;
            const int rect16_trial = index >= tree_trials + 24u;
            const int half_rect_trial = index >= tree_trials + 8u;
            if (rect_trial) {
                static const uint32_t powers[4] = {1u,7u,49u,343u};
                if (index == tree_trials) {
                    uint32_t digits = mixed_codes[2];
                    for (uint32_t q = 0u; q < 4u; ++q, digits /= 5u)
                        rect_code += (digits % 5u) * powers[q];
                    rect_cost = mixed_costs[2];
                }
                if (half_rect_trial) {
                    const uint32_t variant = index - tree_trials - (rect16_trial ? 24u : 8u);
                    const uint32_t power = powers[variant / (rect16_trial ? 8u : 4u)];
                    const uint32_t others = rect_code % power + rect_code / (7u * power) * power;
                    decision = (uint16_t)((rect16_trial ? SVT_AV2_RECT16_BASE : SVT_AV2_HALF_RECT_BASE) + variant * 343u + others);
                } else {
                    const uint32_t q = (index - tree_trials) / 2u;
                    const uint32_t shape = 5u + (index - tree_trials) % 2u;
                    const uint32_t power = powers[q];
                    mixed_code = rect_code - (rect_code / power % 7u) * power + shape * power;
                    decision = (uint16_t)(SVT_AV2_MIXED_RECT32_BASE + mixed_code);
                }
#if M6_ROOT_ASYM
            } else if ((work->config->partition_selection==5u || work->config->partition_selection==7u) &&
                index>=(work->config->partition_selection==7u?56u:24u)) {
                decision=(uint16_t)(3285u+index-(work->config->partition_selection==7u?56u:24u));
#endif
            } else if (index >= 66u) {
                decision=(uint16_t)(3283u+index-66u);
            } else if (index >= 58u) {
                static const uint32_t powers[4] = {1u,7u,49u,343u};
                family=3u;
                const uint32_t quadrant=(index-58u)/2u,shape=(index-58u)%2u+5u;
                const uint32_t power=powers[quadrant];
                mixed_code=mixed_codes[family]-mixed_codes[family]/power%7u*power+shape*power;
                decision=svt_av2_partition_mixed7_plan(mixed_code);
                if (mixed_code==mixed_codes[family]) continue;
            } else if (index >= 56u) {
                family=3u;mixed_code=(index-56u+5u)*400u;
                decision=(uint16_t)(880u+index-56u);
            } else if (index >= 40u) {
                static const uint32_t powers[4] = {1u,5u,25u,125u};
                family=2u;
                const uint32_t quadrant=(index-40u)/4u, shape=(index-40u)%4u+1u;
                const uint32_t power=powers[quadrant];
                mixed_code=mixed_codes[family]-mixed_codes[family]/power%5u*power+shape*power;
                decision=mixed8_plan(mixed_code);
                if (mixed_code==mixed_codes[family]) continue;
            } else if (index >= 24u) {
                static const uint32_t powers[4] = {1u, 3u, 9u, 27u};
                family = (index - 24u) / 8u;
                const uint32_t quadrant = (index - 24u) % 8u / 2u;
                const uint32_t shape = (index - 24u) % 2u + 1u;
                const uint32_t power = powers[quadrant];
                mixed_code = mixed_codes[family] - mixed_codes[family] / power % 3u * power + shape * power;
                decision = (uint16_t)((family ? 174u : 93u) + mixed_code);
                if (mixed_code == mixed_codes[family]) continue;
            } else if (index >= 16u) decision = (uint16_t)(85u + index - 16u);
            else if (index >= 4u) {
                const uint32_t quadrant = (index - 4u) / 3u;
                const uint32_t depth = (index - 4u) % 3u + 1u;
                uint32_t code = 4u, power = 1u, uniform = 0u;
                for (uint32_t q = 0u; q < 4u; ++q) {
                    const uint32_t d = q == quadrant ? depth :
                        (best_decision ? svt_av2_partition_plan_depth(best_decision, q) : 1u);
                    code += (d - 1u) * power; power *= 3u;
                    if (q == 0u) uniform = d;
                    else if (d != uniform) uniform = 0u;
                }
                decision = (uint16_t)(uniform ? uniform : code);
            }
            if (decision == initial_decision || decision == best_decision) {
                const uint64_t known=decision == initial_decision ? initial_cost : best_cost;
                if (decision == 1u || decision == 2u) mixed_costs[decision - 1u] = known;
                if (decision == 2u && index<40u) mixed_costs[2]=known;
                if (index>=40u && index<56u && known<mixed_costs[2]) {mixed_costs[2]=known;mixed_codes[2]=mixed_code;}
                if (index>=56u && index<66u && known<mixed_costs[3]) {mixed_costs[3]=known;mixed_codes[3]=mixed_code;}
                continue;
            }
            error = geometry_for_sb(work, sb, decision);
            if (error != SVT_AV2_ERROR_NONE) return error;
            memset(work->uv_ids, 0, sizeof(work->uv_ids));
            RdBasicChoice trial = {UINT64_MAX, 0u, 0u, 0u, UINT64_MAX};
            /* Coarse seeds retain the original candidate domain. The fine
             * tree starts from one causal DC/V/H SAD path, then runs the same
             * true-rate seven-mode Y/UV coordinate pass as each coarse tree. */
            const int greedy = decision >= 2u;
            const int pruned = decision != 0u && work->config->rd_mode >= 2u;
            const uint32_t paths = greedy ? 1u : 1u << ((pruned ? 1u : 2u) * work->leaf_count);
            for (uint32_t seed_index = 0u; seed_index < paths; ++seed_index) {
                const uint32_t ordinal = work->config->reverse_candidates ? paths - 1u - seed_index : seed_index;
                region_copy(work, 1);
                uint64_t d, r, cost; uint32_t path; int valid;
                error = build_path(work, ordinal, pruned, greedy, NULL, &valid, &d, &path);
                if (error != SVT_AV2_ERROR_NONE) return error;
                if (!valid) continue;
                if (RD_RATE(work, sb, decision, work->blocks,
                    work->leaf_count * 3u, NULL, 0u, 0, &r) != 0)
                    return SVT_AV2_ERROR_INVALID_STATE;
                error = svt_av2_rd_cost(d, work->stats.lambda_q8, r, &cost);
                if (error != SVT_AV2_ERROR_NONE) return error;
                const uint64_t id = plan_candidate_id(decision, (uint32_t)decision *
                    (work->config->frame_header.m5_intra_modes >= 3u ? 65536u : 4096u) + path, path);
                ++work->stats.candidate_count;
                if (work->config->candidate_hook != NULL)
                    work->config->candidate_hook(work->config->candidate_hook_user, sb, id, d, r, cost, 0);
                if (cost < trial.cost || (cost == trial.cost && id < trial.id)) {
                    trial.cost = cost; trial.distortion = d; trial.rate = r;
                    trial.hash = path_hash(work); trial.id = id;
                    memcpy(work->best_ids, work->ids, work->leaf_count);
                }
            }
            if (trial.id == UINT64_MAX) return SVT_AV2_ERROR_INVALID_STATE;
            error = refine_basic(work, sb, 0, decision, &trial);
            if (error != SVT_AV2_ERROR_NONE) return error;
            if (decision == 1u || decision == 2u) mixed_costs[decision - 1u] = trial.cost;
            if (decision == 2u && index<40u) mixed_costs[2]=trial.cost;
            if (!rect_trial && index >= 24u && index < 66u && trial.cost < mixed_costs[family]) {
                mixed_costs[family] = trial.cost;
                mixed_codes[family] = mixed_code;
            }
            if (rect_trial) {
                ++work->rect32_trials;
                if (!half_rect_trial && trial.cost < rect_cost) {
                    rect_code = mixed_code;
                    rect_cost = trial.cost;
                }
                RdTreeSeed* seed = rect16_trial ? &rect16_seed : half_rect_trial ? &half_rect_seed : &rect32_seed;
                if (trial.cost < seed->choice.cost) {
                    seed->choice = trial; seed->decision = decision;
                    seed->leaf_count = work->leaf_count;
                    memcpy(seed->ids, work->best_ids, work->leaf_count);
                    memcpy(seed->uv, work->uv_ids, work->leaf_count);
                }
            } else if (work->config->tree_finalists==10u && index>=66u) {
                /* Keep both new shapes without evicting any of the old eight seeds. */
                RdTreeSeed* seed=&thin64[thin64_count++];
                seed->choice=trial;seed->decision=decision;seed->leaf_count=work->leaf_count;
                memcpy(seed->ids,work->best_ids,work->leaf_count);
                memcpy(seed->uv,work->uv_ids,work->leaf_count);
            } else if (work->config->tree_finalists)
                offer_tree(work,decision,&trial,finalists,&finalist_count);
            if (trial.cost < best_cost || (trial.cost == best_cost && decision < best_decision)) {
                best_cost = trial.cost; best_d = trial.distortion; best_r = trial.rate;
                best_hash = trial.hash; best_id = trial.id; best_decision = decision;
                memcpy(saved_ids, work->best_ids, sizeof(saved_ids));
                memcpy(saved_uv, work->uv_ids, sizeof(saved_uv));
            }
        }
        memcpy(work->best_ids, saved_ids, sizeof(saved_ids));
        memcpy(work->uv_ids, saved_uv, sizeof(saved_uv));
        error = geometry_for_sb(work, sb, best_decision);
        if (error != SVT_AV2_ERROR_NONE) return error;
        region_copy(work, 1);
    }
    /* Preserve both seeds from the original square/small-rectangle domain.
     * Newly added trees otherwise may evict them before full tool refinement. */
#if M6_ROOT_ASYM
    for (uint32_t i=0u;i<before_asym_count;++i) {
        uint32_t j=0u;
        while (j<finalist_count && (finalists[j].decision!=before_asym[i].decision ||
            finalists[j].choice.hash!=before_asym[i].choice.hash)) ++j;
        if (j==finalist_count) finalists[finalist_count++]=before_asym[i];
    }
#endif
    for (uint32_t i=0u; i<reserved_count; ++i) {
        uint32_t j=0u;
        while (j<finalist_count && (finalists[j].decision!=reserved[i].decision ||
            finalists[j].choice.hash!=reserved[i].choice.hash)) ++j;
        if (j==finalist_count) finalists[finalist_count++]=reserved[i];
    }
    for (uint32_t i=0u;i<before_thin_count;++i) {
        uint32_t j=0u;
        while (j<finalist_count && (finalists[j].decision!=before_thin[i].decision ||
            finalists[j].choice.hash!=before_thin[i].choice.hash)) ++j;
        if (j==finalist_count) finalists[finalist_count++]=before_thin[i];
    }
    for (uint32_t i=0u;i<thin64_count;++i) finalists[finalist_count++]=thin64[i];
    if (work->config->tree_finalists && rect32_seed.choice.cost != UINT64_MAX)
        finalists[finalist_count++] = rect32_seed;
    if (work->config->tree_finalists && half_rect_seed.choice.cost != UINT64_MAX)
        finalists[finalist_count++] = half_rect_seed;
    if (work->config->tree_finalists && rect16_seed.choice.cost != UINT64_MAX)
        finalists[finalist_count++] = rect16_seed;
    RdBasicChoice tool_best={best_cost,best_d,best_r,best_hash,best_id};
    if (work->config->tree_finalists && !partial && finalist_count>1u)
        error=refine_tree_finalists(work,sb,partial,finalists,finalist_count,&best_decision,&tool_best);
    else error=refine_tools(work,sb,partial,best_decision,&tool_best);
    if (error!=SVT_AV2_ERROR_NONE) { region_copy(work,1); return error; }
#if M6_FINE16
    if (!partial && work->config->rd_mode==3u && work->config->partition_selection>=2u &&
        work->config->tree_finalists) {
        RdTreeSeed fine16_seeds[4];uint32_t fine16_count=0u;
        error=fine16_prepare(work,sb,best_decision,fine16_seeds,&fine16_count);
        if (error==SVT_AV2_ERROR_NONE && fine16_count)
            error=fine16_refine(work,sb,fine16_seeds,fine16_count,&best_decision,&tool_best);
#if M6_FINE16_PAIR
        if (error==SVT_AV2_ERROR_NONE && fine16_count) {
            RdTreeSeed pair_seeds[4];uint32_t pair_count=0u;
            error=fine16_pair_prepare(work,sb,best_decision,fine16_seeds[0].decision,pair_seeds,&pair_count);
            if (error==SVT_AV2_ERROR_NONE && pair_count)
                error=fine16_pair_refine(work,sb,pair_seeds,pair_count,&best_decision,&tool_best);
        }
#endif
        if (error!=SVT_AV2_ERROR_NONE) {region_copy(work,1);return error;}
    }
#endif
    if (work->output->decisions[sb]!=best_decision) {
        region_copy(work,1);return SVT_AV2_ERROR_INVALID_STATE;
    }
    SVT_AV2_IP_ADD(work->profile->coarse, profile_coarse);
    if (!partial && work->config->frame_header.mv_half) {
        SVT_AV2_IP_START(profile_subpel);
        /* The opt-in half-pel profile is limited to the original 64/32 trees. */
        if (work->leaf_count > 4u) return SVT_AV2_ERROR_INVALID_STATE;
        error = refine_halfpel(work, sb, best_decision, &tool_best);
#if SVT_AV2_INTER_PROFILE
        if (work->config->frame_header.mv_half == 2u)
            SVT_AV2_IP_ADD(work->profile->quarter, profile_subpel);
        else SVT_AV2_IP_ADD(work->profile->half, profile_subpel);
#endif
        if (error != SVT_AV2_ERROR_NONE) return error;
    }
    if (!partial && work->config->inter_skip_search) {
        SVT_AV2_IP_START(profile_skip);
        if (work->leaf_count > 4u) return SVT_AV2_ERROR_INVALID_STATE;
        error = refine_inter_skip(work, sb, best_decision, &tool_best);
        SVT_AV2_IP_ADD(work->profile->skip, profile_skip);
        if (error != SVT_AV2_ERROR_NONE) return error;
    }
    if (!partial && work->secondary != NULL) {
        SVT_AV2_IP_START(profile_long);
        error = refine_long_ref(work, sb, best_decision, &tool_best);
        SVT_AV2_IP_ADD(work->profile->long_ref, profile_long);
        if (error != SVT_AV2_ERROR_NONE) { if(work->fine_trace) fprintf(stderr,"refine_long_error sb=%zu\n",sb); region_copy(work, 1); return error; }
    }
    if (!partial && work->config->inter_fine) {
        SVT_AV2_IP_START(profile_fine);
        error = refine_inter_fine(work, sb, &best_decision, &tool_best);
        SVT_AV2_IP_ADD(work->profile->fine, profile_fine);
        if (error != SVT_AV2_ERROR_NONE) { if(work->fine_trace) fprintf(stderr,"refine_fine_error sb=%zu\n",sb); region_copy(work, 1); return error; }
    }
    if (!partial && work->config->inter_near) {
        error = refine_inter_near(work, sb, best_decision, &tool_best);
        if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
    }
    if (!partial && work->video_rect) {
        error = refine_video_rect(work, sb, &best_decision, &tool_best);
        if (error != SVT_AV2_ERROR_NONE) return error;
    }
    if (work->late_inter_skip && (!partial || (work->edge_inter && work->leaf_count <= INTER_FINE_MAX_LEAVES))) {
        error = refine_inter_skip(work, sb, best_decision, &tool_best);
        if (error != SVT_AV2_ERROR_NONE) return error;
    }
    SVT_AV2_IP_START(profile_replay);
    best_cost=tool_best.cost; best_d=tool_best.distortion; best_r=tool_best.rate;
    best_hash=tool_best.hash; best_id=tool_best.id;
    uint64_t distortion = 0u, rate = 0u; uint32_t path_id; int legal;
    error = build_path(work, 0u, 0, partial, work->best_ids, &legal, &distortion, &path_id);
    if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
    if (!legal || distortion != best_d || path_hash(work) != best_hash ||
        RD_RATE(work, sb, best_decision, work->blocks, work->leaf_count * 3u,
            work->reference == NULL ? NULL : work->modes, work->reference == NULL ? 0u : work->leaf_count, 0, &rate) != 0 || rate != best_r) {
        region_copy(work, 1); return SVT_AV2_ERROR_INVALID_STATE;
    }
    SVT_AV2_IP_ADD(work->profile->final, profile_replay);
    if (work->tail && !partial) {
        SVT_AV2_IP_START(profile_tail);
        error = refine_inter_tail(work, sb, best_decision, &tool_best, &rate);
        SVT_AV2_IP_ADD(work->profile->tail, profile_tail);
        if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
        best_cost = tool_best.cost; best_d = tool_best.distortion; best_r = tool_best.rate; best_hash = tool_best.hash;
    }
#if SVT_AV2_EXPERIMENTAL_VIDEO_CFL
    if (work->config->video_cfl_search && !partial) {
        error = refine_video_cfl(work,sb,best_decision,&tool_best,&rate);
        if (error!=SVT_AV2_ERROR_NONE) {region_copy(work,1);return error;}
        best_cost=tool_best.cost;best_d=tool_best.distortion;best_r=tool_best.rate;
        best_hash=tool_best.hash;best_id=tool_best.id;
    }
#endif
#if SVT_AV2_EXPERIMENTAL_INTRA_EOB
    if (work->intra_eob_enabled && !partial) {
        error = refine_intra_eob(work, sb, best_decision, &tool_best, &rate);
        if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
        best_cost = tool_best.cost; best_d = tool_best.distortion;
        best_r = tool_best.rate; best_hash = tool_best.hash;
    }
#endif
    if (work->config->intra_ist_search && !partial) {
        error = refine_ist(work, sb, best_decision, &tool_best);
        if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
        best_cost = tool_best.cost; best_d = tool_best.distortion; best_r = tool_best.rate;
        best_hash = tool_best.hash; best_id = tool_best.id;
    }
    SVT_AV2_IP_START(profile_commit);
    ++work->stats.replay_checks;
    error = append_winner(work);
    if (error != SVT_AV2_ERROR_NONE) { region_copy(work, 1); return error; }
    if (RD_RATE(work, sb, best_decision, work->blocks, work->leaf_count * 3u,
        work->reference == NULL ? NULL : work->modes, work->reference == NULL ? 0u : work->leaf_count, 1, &rate) != 0 || rate != best_r) {
        region_copy(work, 1); return SVT_AV2_ERROR_INVALID_STATE;
    }
    work->rect32_wins += svt_av2_partition_is_mixed_rect32(best_decision) != 0;
    work->rect16_wins += svt_av2_partition_is_rect16(best_decision) != 0;
    work->half_rect_wins += svt_av2_partition_is_half_rect(best_decision) != 0;
    if (work->smooth_grid) {
        const size_t cols = work->geometry.visible_width / 4u;
        for (uint32_t i = 0u; i < work->leaf_count; ++i) {
            const SvtAv2BlockView* b = work->leaves + i;
            const uint32_t smooth = rd_smooth_mode(work->ids[i]) | (rd_smooth_mode(work->uv_ids[i]) << 1u);
            for (uint32_t y = b->y / 4u; y < (b->y + b->h) / 4u; ++y)
                memset(work->smooth_grid + y * cols + b->x / 4u, (int)smooth, b->w / 4u);
        }
    }
    if (work->fine_trace && !partial && work->leaf_count > 4u) trace_fine_winner(work);
    work->stats.distortion += best_d; work->stats.rate_q12 += best_r; work->stats.cost += best_cost;
    if (work->config->candidate_hook != NULL)
        work->config->candidate_hook(work->config->candidate_hook_user, sb, best_id,
            best_d, best_r, best_cost, 1);
    if (best_id >> 32u) work->stats.winner_hash = hash_word(work->stats.winner_hash, (uint32_t)(best_id >> 32u));
    work->stats.winner_hash = hash_word(hash_word(work->stats.winner_hash, (uint32_t)best_id), (uint32_t)best_hash);
    SVT_AV2_IP_ADD(work->profile->final, profile_commit);
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_rd_frame_encode(const SvtAv2FrameBuffer* source,
    SvtAv2FrameBuffer* recon, const SvtAv2ReferenceFrame* reference,
    const SvtAv2RdFrameConfig* config, SvtAv2ReconScratch* scratch,
    SvtAv2RdFrameOutput* output, SvtAv2RdFrameStats* stats) {
    if (stats != NULL) memset(stats, 0, sizeof(*stats));
    if (output != NULL) output->block_count = output->mode_count = output->coeff_count = output->decision_count = 0u;
    if (source == NULL || recon == NULL || config == NULL || scratch == NULL || output == NULL ||
        source->buffer == NULL || recon->buffer == NULL || source->buffer == recon->buffer ||
        source->visible_width != recon->visible_width || source->visible_height != recon->visible_height ||
        source->bit_depth != recon->bit_depth || source->chroma_format != recon->chroma_format ||
        config->rd_mode < 1u || config->rd_mode > 3u || config->partition_selection > 10u ||
        (config->rd_mode==3u && (reference!=NULL || config->frame_header.multi_frame ||
            config->frame_header.m5_intra_modes<3u || config->partition_selection>7u ||
            config->intra_joint || config->coeff_optimization==1u || config->motion_optimization ||
            config->tree_finalists>4u)) ||
        config->rd_lambda_model > 2u || (config->rd_lambda_model && (reference != NULL ||
            config->frame_header.multi_frame) && !(config->rd_mode == 2u && config->frame_header.multi_frame)) ||
        (config->rd_lambda_scale && (config->rd_lambda_scale < 50u || config->rd_lambda_scale > 250u || !config->rd_lambda_model)) ||
        (config->partition_selection >= 2u && config->frame_header.m5_intra_modes < 2u) ||
        config->frame_header.enable_mrls > 1u || (config->frame_header.enable_mrls &&
            (reference != NULL || config->frame_header.multi_frame || config->frame_header.m5_intra_modes<3u)) ||
        (config->tree_finalists!=0u && config->tree_finalists!=2u && config->tree_finalists!=4u &&
         config->tree_finalists!=8u && config->tree_finalists!=10u) ||
        (config->tree_finalists && (reference!=NULL || config->frame_header.multi_frame ||
            config->partition_selection<2u || config->frame_header.m5_intra_modes<2u)) ||
        config->intra_joint > 1u || (config->intra_joint && (reference != NULL ||
            config->frame_header.multi_frame || config->frame_header.m5_intra_modes < 3u ||
            config->transform_selection != 2u)) ||
        config->mhccp_search > 1u || config->frame_header.enable_mhccp != config->mhccp_search ||
        (config->mhccp_search && (reference != NULL || config->frame_header.multi_frame ||
            config->frame_header.m5_intra_modes != 4u)) ||
        config->cfl_search > 1u || (config->cfl_search && (reference != NULL ||
            config->frame_header.multi_frame || config->frame_header.m5_intra_modes != 4u)) ||
#if SVT_AV2_EXPERIMENTAL_VIDEO_CFL
        !svt_av2_frame_header_video_cfl_valid(&config->frame_header) || config->video_cfl_search>1u ||
        (config->video_cfl_search && (!config->frame_header.enable_cfl_intra || config->rd_mode!=2u ||
            config->inter_skip_search || config->transform_selection || config->coeff_optimization ||
            source->bit_depth!=10u || source->chroma_format!=SVT_AV2_CHROMA_420)) ||
#endif
        config->angle_search > 1u || (config->angle_search && (reference != NULL ||
            config->frame_header.multi_frame || config->frame_header.m5_intra_modes < 3u)) ||
        config->transform_selection > 2u ||
        (config->transform_selection && (reference != NULL || config->frame_header.multi_frame ||
            config->frame_header.reduced_tx_set != (config->transform_selection == 2u ? 0u : 1u) ||
            config->frame_header.m5_intra_modes < 2u ||
            config->transform.forward_type == NULL || config->transform.inverse_type == NULL)) ||
        (!config->transform_selection && config->frame_header.reduced_tx_set == 1u) ||
        config->coeff_optimization > 2u || config->motion_optimization > 5u ||
        (config->frame_header.mv_half != (config->motion_optimization == 5u ? 2u :
            config->motion_optimization >= 3u ? 1u : 0u)) ||
        (config->frame_header.sharp_filter != (config->motion_optimization >= 4u ? 1u : 0u)) ||
        (config->frame_header.mv_half && (reference == NULL || config->rd_mode != 2u ||
            config->partition_selection > 1u)) ||
        config->inter_skip_search > 1u || (config->inter_skip_search &&
            (reference == NULL || config->rd_mode != 2u || config->partition_selection > 1u ||
             config->coeff_optimization)) ||
        config->inter_fine > 1u || (config->inter_fine &&
            (reference == NULL || !config->frame_header.multi_frame || config->rd_mode != 2u ||
             config->partition_selection != 1u || config->frame_header.m5_intra_modes > 1u ||
             config->motion_optimization != 5u || config->inter_skip_search)) ||
        config->inter_near > 1u || (config->inter_near &&
             (reference == NULL || !config->frame_header.multi_frame || config->rd_mode != 2u ||
              config->partition_selection > 1u || config->frame_header.m5_intra_modes > 1u ||
              config->motion_optimization != 5u || config->inter_skip_search || config->coeff_optimization ||
              config->transform_selection)) ||
         config->inter_tail > 1u || (config->inter_tail &&
            (reference == NULL || !config->frame_header.multi_frame || config->rd_mode != 2u ||
             config->partition_selection > 1u || config->frame_header.m5_intra_modes > 1u ||
             config->motion_optimization != 5u || config->inter_skip_search || config->coeff_optimization ||
             config->transform_selection || !svt_av2_transform_path_cacheable(&config->transform,
                 config->frame_header.base_q_idx, config->frame_header.bit_depth))) ||
        (config->coeff_optimization == 2u && config->frame_header.m5_intra_modes < 2u) ||
        (config->coeff_optimization && config->motion_optimization) ||
        (config->frame_header.entropy_adaptation &&
            (config->frame_header.multi_frame || reference != NULL) &&
            (config->rd_mode != 2u || config->frame_header.m5_intra_modes > 1u ||
             config->partition_selection > 1u || config->coeff_optimization)) ||
        config->frame_header.m5_intra_modes > 4u ||
        (config->frame_header.m5_intra_modes >= 2u && (reference != NULL ||
            config->frame_header.multi_frame || config->coeff_optimization == 1u || config->motion_optimization)) ||
        config->transform.forward == NULL || config->transform.inverse == NULL ||
        output->blocks == NULL || output->modes == NULL || output->qcoeff == NULL || output->decisions == NULL ||
        scratch->residual == NULL || scratch->inverse_residual == NULL || scratch->coeff == NULL ||
        scratch->qcoeff == NULL || scratch->dqcoeff == NULL || scratch->stride < 64u ||
        scratch->capacity < 4096u || scratch->coeff_capacity < 1024u)
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    const SvtAv2FrameHeaderConfig* header = &config->frame_header;
    if (header->frame_width != source->visible_width || header->frame_height != source->visible_height ||
        header->bit_depth != (uint32_t)source->bit_depth || header->chroma_format != (uint32_t)source->chroma_format ||
        (header->inter_frame != 0u ? 1u : 0u) != (reference != NULL ? 1u : 0u)) return SVT_AV2_ERROR_INVALID_PARAMETER;
    const SvtAv2FrameBuffer* ref = reference == NULL ? NULL : svt_av2_reference_view(reference);
    if (reference != NULL && (ref == NULL || ref->buffer == source->buffer || ref->buffer == recon->buffer ||
        ref->visible_width != source->visible_width || ref->visible_height != source->visible_height ||
        ref->bit_depth != source->bit_depth || ref->chroma_format != source->chroma_format))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    const SvtAv2FrameBuffer* second = config->secondary_reference == NULL ? NULL :
        svt_av2_reference_view(config->secondary_reference);
    if (config->secondary_reference != NULL && (second == NULL || ref == NULL ||
        second->buffer == ref->buffer || second->buffer == source->buffer || second->buffer == recon->buffer ||
        second->visible_width != source->visible_width || second->visible_height != source->visible_height ||
        second->bit_depth != source->bit_depth || second->chroma_format != source->chroma_format ||
        config->rd_mode != 2u || config->motion_optimization != 5u || config->partition_selection > 1u ||
        config->inter_skip_search || header->logical_ref_count != 2u ||
        header->ref_distance[0] != 1u || header->ref_distance[1] < 2u || header->ref_distance[1] > 8u))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (second == NULL && (header->logical_ref_count == 2u || header->ref_distance[0] || header->ref_distance[1]))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (header->enable_intra_edge_filter && (header->multi_frame || config->rd_mode != 3u ||
        source->chroma_format != SVT_AV2_CHROMA_420))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    SVT_AV2_IP_START(profile_setup);
    RdWork* work = (RdWork*)calloc(1u, sizeof(*work));
    if (work == NULL) return SVT_AV2_ERROR_OUT_OF_MEMORY;
#if SVT_AV2_INTER_PROFILE
    RdInterProfile profile = {0};
    work->profile = &profile;
#endif
    work->source = source; work->recon = recon; work->reference = ref; work->secondary = second;
    work->config = config; work->scratch = scratch; work->output = output;
    const char* ist_search = getenv("SVT_AV2_IST_SEARCH");
    work->ist_all_leaves = ist_search && !strcmp(ist_search, "full");
    const char* edge = getenv("SVT_AV2_EDGE_INTER");
    work->edge_inter = edge && !strcmp(edge, "1");
    const char* wide_me = getenv("SVT_AV2_INTER_ME_WIDE");
    work->wide_inter_me = wide_me && !strcmp(wide_me, "1");
    const char* fine = getenv("SVT_AV2_INTER_FINE_FULL");
    work->full_inter_fine = fine && !strcmp(fine, "1");
    const char* late_skip = getenv("SVT_AV2_FINAL_SKIP");
    work->late_inter_skip = ref && config->inter_tail && late_skip && !strcmp(late_skip, "1");
    const char* detail = getenv("SVT_AV2_DETAIL_RD");
    work->detail_rd = detail && (!strcmp(detail, "all") || !strcmp(detail, header->inter_frame ? "p" : "key"));
    const char* video_rect = getenv("SVT_AV2_VIDEO_RECT");
    work->video_rect = header->multi_frame && header->m5_intra_modes == 1u &&
        config->rd_mode == 2u && !config->coeff_optimization && !config->transform_selection &&
        video_rect && (!strcmp(video_rect, "all") || !strcmp(video_rect, header->inter_frame ? "p" : "key"));
    const char* rect32 = getenv("SVT_AV2_STILL_RECT_RD");
    const char* rect_rdoq = getenv("SVT_AV2_RECT_RDOQ");
    work->rect_rdoq = rect_rdoq && !strcmp(rect_rdoq, "1");
    if (rect32 && config->rd_mode == 3u && config->partition_selection == 7u)
        work->rect32_search = !strcmp(rect32, "deep") ? 3u : !strcmp(rect32, "half") ? 2u : !strcmp(rect32, "1") ? 1u : 0u;
#if SVT_AV2_EXPERIMENTAL_STILL_UV_RD
    const char* late_uv = getenv("SVT_AV2_STILL_UV_RD");
    work->still_uv_rd = late_uv && !strcmp(late_uv, "1");
#endif
#if SVT_AV2_EXPERIMENTAL_CFL_PAIR_RD
    const char* cfl_pair_rd = getenv("SVT_AV2_CFL_PAIR_RD");
    work->cfl_pair_rd = cfl_pair_rd && !strcmp(cfl_pair_rd,"1");
#endif
#if SVT_AV2_EXPERIMENTAL_INTRA_EOB
    /* Private research control: use the same binary with identical KEY
     * reconstruction to distinguish a direct P effect from a changed seed. */
    const char* eob_scope = getenv("SVT_AV2_INTRA_EOB");
    work->intra_eob_enabled = eob_scope && (!strcmp(eob_scope, "all") ||
        (ref ? !strcmp(eob_scope, "p") : !strcmp(eob_scope, "key")));
#endif
#if SVT_AV2_INTER_COARSE_PREFIX
    if (ref && config->rd_mode == 2u && config->partition_selection == 1u && config->motion_optimization == 5u &&
        header->m5_intra_modes <= 1u && !config->coeff_optimization && !config->transform_selection &&
        !config->inter_skip_search && (source->chroma_format == SVT_AV2_CHROMA_420 ||
        source->chroma_format == SVT_AV2_CHROMA_444) &&
        svt_av2_transform_path_cacheable(&config->transform, header->base_q_idx, source->bit_depth)) {
        /* Failure falls back to the original search without changing any candidate. */
        work->coarse_prefix = (RdCoarsePrefix*)malloc(sizeof(*work->coarse_prefix));
        if (work->coarse_prefix != NULL) {
            const char* trace = getenv("SVT_AV2_INTER_COARSE_TRACE");
            work->coarse_prefix->trace = trace && trace[0] == '1' && trace[1] == '\0';
            work->coarse_prefix->trees = work->coarse_prefix->rank_hits = work->coarse_prefix->leaf_hits = 0u;
#if SVT_AV2_INTER_COARSE_RATE_PREFIX
            /* Allocation failure uses the unchanged full SB rate writer. */
            (void)svt_av2_tile_inter_rate_cache_create(&work->coarse_prefix->rate_cache);
            work->coarse_prefix->rate_trees = work->coarse_prefix->rate_candidates = 0u;
            const char* rate_trace = getenv("SVT_AV2_INTER_COARSE_RATE_TRACE");
            work->coarse_prefix->rate_trace = rate_trace && rate_trace[0] == '1' && rate_trace[1] == '\0';
#endif
        }
    }
#endif
    if (config->inter_near) {
        const char* trace = getenv("SVT_AV2_INTER_NEAR_TRACE");
        work->near_trace = trace && trace[0] == '1' && trace[1] == '\0';
    }
    if (second != NULL && config->inter_fine) {
        const char* trace = getenv("SVT_AV2_INTER_FINE_TRACE");
        work->fine_trace = trace && trace[0] == '1' && trace[1] == '\0';
    }
#if SVT_AV2_INTER_PIXEL_REUSE
    if (ref != NULL && header->mv_half == 2u && header->sharp_filter && config->rd_mode == 2u &&
        config->partition_selection <= 1u && !config->coeff_optimization && !config->transform_selection &&
        svt_av2_transform_path_cacheable(&config->transform, header->base_q_idx, source->bit_depth)) {
        /* Allocation failure is an ordinary miss for the entire frame. Tests
         * using instrumented transform callbacks don't enter this profile. */
        work->inter_pixel = (RdInterPixelMemo*)malloc(sizeof(*work->inter_pixel));
        if (work->inter_pixel != NULL) work->inter_pixel->hits = work->inter_pixel->misses = 0u;
    }
#endif
#if M6_TX_CACHE
    /* Allocation failure safely falls back to the original forward path. */
    if (config->rd_mode==3u && svt_av2_transform_path_cacheable(&config->transform,
        header->base_q_idx,header->bit_depth))
        work->transform_cache=rd_tx_memo_create();
    if (work->transform_cache) {
        work->transform_cache->qm_level = header->qm_level;
        work->transform_cache->uv_delta_q = header->uv_delta_q;
    }
#endif
    work->stats.winner_hash = RD_HASH_OFFSET;
    uint32_t qstep;
    SvtAv2Error error = config->rd_lambda_model == 2u ?
        svt_av2_rd_lambda_ai150(header->base_q_idx, source->bit_depth, &work->stats.lambda_q8, &qstep) :
        config->rd_lambda_model ?
        svt_av2_rd_lambda_ai(header->base_q_idx, source->bit_depth, &work->stats.lambda_q8, &qstep) :
        svt_av2_rd_lambda(header->base_q_idx, source->bit_depth, &work->stats.lambda_q8, &qstep);
    if (error == SVT_AV2_ERROR_NONE && config->rd_lambda_scale) {
        work->stats.lambda_q8 = SVT_AV2_MAX(UINT64_C(1), (work->stats.lambda_q8 * config->rd_lambda_scale + 50u) / 100u);
    }
    if (error == SVT_AV2_ERROR_NONE && config->inter_tail) {
        work->tail = (RdInterTail*)calloc(1u, sizeof(*work->tail));
        const char* zero = getenv("SVT_AV2_SPARSE_PLANES");
        if (work->tail) work->tail->try_sparse_planes = zero && !strcmp(zero, "1");
        if (work->tail == NULL) error = SVT_AV2_ERROR_OUT_OF_MEMORY;
        else {
            const char* trace = getenv("SVT_AV2_INTER_TAIL_TRACE");
            work->tail->trace = trace && trace[0] == '1' && trace[1] == '\0';
            for (uint32_t i = 0u; i < 3u; ++i) {
                const uint32_t side = 8u << i;
                SvtAv2TxSize tx;
                if (svt_av2_coeff_tx_size_from_dims(side, side, &tx) != SVT_AV2_COEFF_OK ||
                    svt_av2_coeff_scan_dct_dct(tx, work->tail->scans[i]) != side * side) {
                    error = SVT_AV2_ERROR_INVALID_STATE;
                    break;
                }
            }
        }
    }
    if (error == SVT_AV2_ERROR_NONE)
        error = svt_av2_frame_geometry_init_ex(&work->geometry, source->visible_width,
            source->visible_height, 64u, source->chroma_format);
    if (error == SVT_AV2_ERROR_NONE) {
        const size_t count = (size_t)work->geometry.sb_cols * work->geometry.sb_rows;
        if (output->decision_capacity < count) error = SVT_AV2_ERROR_INVALID_PARAMETER;
        else { output->decision_count = count; memset(output->decisions, 0, count * sizeof(*output->decisions)); }
    }
    if (error == SVT_AV2_ERROR_NONE && svt_av2_tile_rate_create(header, &work->rate) != 0)
        error = SVT_AV2_ERROR_INVALID_STATE;
    if (error == SVT_AV2_ERROR_NONE && header->enable_intra_edge_filter) {
        work->smooth_grid = (uint8_t*)calloc((size_t)(source->visible_width / 4u) *
            (source->visible_height / 4u), sizeof(*work->smooth_grid));
        if (work->smooth_grid == NULL) error = SVT_AV2_ERROR_OUT_OF_MEMORY;
    }
    if (error==SVT_AV2_ERROR_NONE && config->rd_mode==3u &&
        svt_av2_tile_rate_prefix_create(&work->basic_rate_prefix)!=0)
        error=SVT_AV2_ERROR_OUT_OF_MEMORY;
#if M6_RDOQ_CONTEXT
    if (error==SVT_AV2_ERROR_NONE && config->rd_mode==3u && config->coeff_optimization==2u &&
        svt_av2_tile_rate_prefix_create(&work->rdoq_rate_prefix)!=0)
        error=SVT_AV2_ERROR_OUT_OF_MEMORY;
#endif
    if (error==SVT_AV2_ERROR_NONE && work->basic_rate_prefix)
        work->basic_memo=(RdBasicMemo*)calloc(1u,sizeof(*work->basic_memo));
    if (error == SVT_AV2_ERROR_NONE)
        error = svt_av2_recon_reset_frame(recon, (uint16_t)(1u << (source->bit_depth - 1u)));
    const uint64_t frame_lambda = work->stats.lambda_q8;
    double* ssim_scales = NULL;
    const char* ssim = getenv("SVT_AV2_SSIM_RD");
    if (error == SVT_AV2_ERROR_NONE && ssim &&
        (!strcmp(ssim, "all") || !strcmp(ssim, header->inter_frame ? "p" : "key"))) {
        ssim_scales = (double*)malloc(output->decision_count * sizeof(*ssim_scales));
        if (!ssim_scales) error = SVT_AV2_ERROR_OUT_OF_MEMORY;
        else ssim_sb_scales(source, work->geometry.sb_cols, work->geometry.sb_rows, ssim_scales);
    }
    SVT_AV2_IP_START(profile_sb_loop);
    if (error == SVT_AV2_ERROR_NONE)
        for (size_t sb = 0u; sb < output->decision_count; ++sb) {
            if (ssim_scales)
                work->stats.lambda_q8 = (uint64_t)((double)frame_lambda * ssim_scales[sb] + 0.5);
            error = encode_sb(work, sb);
            if (error != SVT_AV2_ERROR_NONE) {
                if (work->fine_trace) fprintf(stderr, "rd_failed hint=%u sb=%zu err=%d\n", header->order_hint, sb, (int)error);
                break;
            }
        }
    free(ssim_scales);
    work->stats.lambda_q8 = frame_lambda;
#if SVT_AV2_INTER_PROFILE
    const uint64_t profile_end = svt_av2_ip_now(), hz = svt_av2_ip_hz();
    if (error == SVT_AV2_ERROR_NONE && hz && profile_setup && profile_sb_loop && profile_end) {
        /* integer_me/fine_screen are nested; callback/rate timings are another
         * view over the loop, not added to the partitioned SB phase totals. */
        fprintf(stderr, "inter_profile_rd kind=%s hint=%u clock=%s units=ns setup=%llu sb_loop=%llu coarse=%llu integer_me=%llu half=%llu quarter=%llu long_ref=%llu fine=%llu fine_screen=%llu skip=%llu tail=%llu final=%llu forward_path=%llu inverse_path=%llu rate_trials=%llu\n",
            ref ? "P" : header->multi_frame ? "KEY" : "INTRA", header->order_hint, SVT_AV2_IP_CLOCK_NAME,
            svt_av2_ip_ns(svt_av2_ip_delta(profile_setup,profile_sb_loop),hz),
            svt_av2_ip_ns(svt_av2_ip_delta(profile_sb_loop,profile_end),hz),
            svt_av2_ip_ns(profile.coarse,hz), svt_av2_ip_ns(profile.integer_me,hz),
            svt_av2_ip_ns(profile.half,hz), svt_av2_ip_ns(profile.quarter,hz),
            svt_av2_ip_ns(profile.long_ref,hz), svt_av2_ip_ns(profile.fine,hz),
            svt_av2_ip_ns(profile.fine_screen,hz), svt_av2_ip_ns(profile.skip,hz),
            svt_av2_ip_ns(profile.tail,hz), svt_av2_ip_ns(profile.final,hz),
            svt_av2_ip_ns(profile.forward_path,hz), svt_av2_ip_ns(profile.inverse_path,hz),
            svt_av2_ip_ns(profile.rate_trials,hz));
        if (ref)
            fprintf(stderr, "inter_profile_coarse_rate hint=%u setup_ns=%llu trial_ns=%llu\n",
                header->order_hint, svt_av2_ip_ns(profile.split_rate_setup,hz),
                svt_av2_ip_ns(profile.split_rate,hz));
    }
#endif
    if (error == SVT_AV2_ERROR_NONE && config->end_models != NULL &&
        svt_av2_tile_rate_end_models(work->rate, config->end_models) != 0)
        error = SVT_AV2_ERROR_INVALID_STATE;
    if (error == SVT_AV2_ERROR_NONE && stats != NULL) *stats = work->stats;
    if (error == SVT_AV2_ERROR_NONE && work->near_trace)
        fprintf(stderr, "inter_near hint=%u trials=%llu accepts=%llu rank0=%llu rank1=%llu odd=%llu empty=%llu\n",
            header->order_hint, (unsigned long long)work->near_trials,
            (unsigned long long)(work->near_wins[0] + work->near_wins[1]),
            (unsigned long long)work->near_wins[0], (unsigned long long)work->near_wins[1],
            (unsigned long long)work->near_odd, (unsigned long long)work->near_empty);
    if (error == SVT_AV2_ERROR_NONE && work->fine_trace)
        fprintf(stderr, "inter_fine_projected hint=%u rank0=%u rank1=%u odd0=%u odd1=%u nz0=%u nz1=%u mixed=%u\n",
            header->order_hint, work->fine_inter16[0], work->fine_inter16[1], work->fine_odd16[0],
            work->fine_odd16[1], work->fine_nonzero16[0], work->fine_nonzero16[1], work->fine_mixed_sb);
#if M6_ROOT_ASYM
    if (error == SVT_AV2_ERROR_NONE) {
        uint32_t counts[4]={0u,0u,0u,0u};
        for (size_t sb=0u;sb<output->decision_count;++sb)
            if (output->decisions[sb]>=3285u && output->decisions[sb]<=3288u)
                ++counts[output->decisions[sb]-3285u];
        fprintf(stderr,"root_asym winners=%u,%u,%u,%u\n",counts[0],counts[1],counts[2],counts[3]);
    }
#endif
    if (error != SVT_AV2_ERROR_NONE)
        output->block_count = output->mode_count = output->coeff_count = output->decision_count = 0u;
    svt_av2_tile_rate_prefix_free(work->rdoq_rate_prefix);
    svt_av2_tile_rate_prefix_free(work->basic_rate_prefix);
    free(work->basic_memo);
#if SVT_AV2_INTER_PIXEL_REUSE
    if (work->inter_pixel != NULL) {
        const char* trace = getenv("SVT_AV2_INTER_PIXEL_TRACE");
        if (trace && trace[0] == '1' && trace[1] == '\0' && error == SVT_AV2_ERROR_NONE)
            fprintf(stderr, "inter_pixel hint=%u hits=%llu misses=%llu bytes=%zu\n", header->order_hint,
                (unsigned long long)work->inter_pixel->hits, (unsigned long long)work->inter_pixel->misses,
                sizeof(*work->inter_pixel));
        free(work->inter_pixel);
    }
#endif
#if M6_TX_CACHE
    if (work->transform_cache != NULL)
        rd_tx_memo_report(work->transform_cache);
#if M6_Y_BOUND
    if (work->transform_cache) fprintf(stderr,"ybound trials=%llu pruned=%llu verify=%u\n",
        (unsigned long long)work->transform_cache->bound_trials,
        (unsigned long long)work->transform_cache->bound_pruned,(unsigned)M6_Y_BOUND_VERIFY);
#endif
#if M6_TREE_RDOQ
    if (work->transform_cache) fprintf(stderr,"treeq trials=%llu wins=%llu changes=%llu full_finalists=%llu retained_old=%u\n",
        (unsigned long long)work->transform_cache->treeq_trials,
        (unsigned long long)work->transform_cache->treeq_wins,
        (unsigned long long)work->transform_cache->treeq_changes,
        (unsigned long long)work->transform_cache->treeq_finalists,(unsigned)M6_TREE_RETAIN_OLD);
#endif
#if M6_TOOL_REPEAT
    if (work->transform_cache) fprintf(stderr,"repeat trials=%llu wins=%llu rounds=%llu changes=%llu after_joint=%llu retained_old=%u\n",
        (unsigned long long)work->transform_cache->repeat_trials,
        (unsigned long long)work->transform_cache->repeat_wins,
        (unsigned long long)work->transform_cache->repeat_rounds,
        (unsigned long long)work->transform_cache->repeat_changes,
        (unsigned long long)work->transform_cache->repeat_after_joint,(unsigned)M6_REPEAT_RETAIN_OLD);
#endif
#if M6_TOOL_REPEAT
    if (work->transform_cache) fprintf(stderr,"repeat_pair trials=%llu illegal=%llu default=%llu\n",
        (unsigned long long)work->transform_cache->repeat_pair_trials,
        (unsigned long long)work->transform_cache->repeat_pair_illegal,
        (unsigned long long)work->transform_cache->repeat_pair_default);
#endif
#if M6_TREE_RDOQ
    if (work->transform_cache) fprintf(stderr,"treeq_plan commits=%llu repairs=%llu\n",
        (unsigned long long)work->transform_cache->treeq_plan_commits,
        (unsigned long long)work->transform_cache->treeq_plan_repairs);
#endif
#if M6_TREE_RDOQ && M6_TREE_SHARED
    if (work->transform_cache) fprintf(stderr,"shared_tree F=%llu G=%llu\n",
        (unsigned long long)work->transform_cache->shared_f,
        (unsigned long long)work->transform_cache->shared_g);
#endif
#if M6_RDOQ_CONTEXT
    if (work->transform_cache) fprintf(stderr,"rdoqcontext trials=%llu,%llu,%llu wins=%llu,%llu,%llu live_rate_calls=%llu\n",
        (unsigned long long)work->transform_cache->rdoq_route_trials[0],
        (unsigned long long)work->transform_cache->rdoq_route_trials[1],
        (unsigned long long)work->transform_cache->rdoq_route_trials[2],
        (unsigned long long)work->transform_cache->rdoq_route_wins[0],
        (unsigned long long)work->transform_cache->rdoq_route_wins[1],
        (unsigned long long)work->transform_cache->rdoq_route_wins[2],
        (unsigned long long)work->transform_cache->rdoq_live_rate_calls);
#endif
    free(work->transform_cache);
#endif
    svt_av2_tile_rate_free(work->rate);
    if (work->tail && work->tail->trace && error == SVT_AV2_ERROR_NONE) {
        const RdInterTail* t = work->tail;
        fprintf(stderr, "inter_tail hint=%u trials=%llu accepts=%llu y=%llu u=%llu v=%llu leaf16=%llu leaf32=%llu leaf64=%llu zero=%llu local_rate_saved_q12=%llu\n",
            header->order_hint, (unsigned long long)t->trials, (unsigned long long)t->accepts,
            (unsigned long long)t->coeffs[0], (unsigned long long)t->coeffs[1], (unsigned long long)t->coeffs[2],
            (unsigned long long)t->leaves[0], (unsigned long long)t->leaves[1], (unsigned long long)t->leaves[2],
            (unsigned long long)t->zero_tbs, (unsigned long long)t->rate_saved_q12);
    }
    free(work->tail);
#if SVT_AV2_EXPERIMENTAL_STILL_UV_RD
    if (error == SVT_AV2_ERROR_NONE && work->still_uv_rd && getenv("SVT_AV2_STILL_UV_TRACE"))
        fprintf(stderr, "still_uv_rd trials=%llu wins=%llu implicit_wins=%llu verified=%llu final_uv=%llu/%llu/%llu/%llu\n",
            (unsigned long long)work->late_uv_trials, (unsigned long long)work->late_uv_wins,
            (unsigned long long)work->late_uv_implicit, (unsigned long long)work->late_uv_verified,
            (unsigned long long)work->final_uv[0], (unsigned long long)work->final_uv[1],
            (unsigned long long)work->final_uv[2], (unsigned long long)work->final_uv[3]);
#endif
    if (error == SVT_AV2_ERROR_NONE && work->video_rect && getenv("SVT_AV2_STILL_RECT_TRACE"))
        fprintf(stderr, "video_rect kind=%s hint=%u trials=%u wins=%u\n", header->inter_frame ? "P" : "KEY",
            header->order_hint, work->video_rect_trials, work->video_rect_wins);
    if (error == SVT_AV2_ERROR_NONE && work->rect32_search && getenv("SVT_AV2_STILL_RECT_TRACE"))
        fprintf(stderr, "still_rect_rd trials=%u wins=%u half_wins=%u rect16_wins=%u\n",
            work->rect32_trials, work->rect32_wins, work->half_rect_wins, work->rect16_wins);
    if (config->intra_ist_search && getenv("SVT_AV2_INTRA_IST_TRACE"))
        fprintf(stderr, "intra_ist kind=%s hint=%u trials=%u wins=%u\n",
            ref ? "P" : "KEY", header->order_hint, work->ist_trials, work->ist_wins);
#if SVT_AV2_EXPERIMENTAL_VIDEO_CFL
    const char* video_cfl_trace=getenv("SVT_AV2_VIDEO_CFL_TRACE");
    if (config->video_cfl_search && video_cfl_trace && !strcmp(video_cfl_trace,"1") && error==SVT_AV2_ERROR_NONE)
        fprintf(stderr,"video_cfl kind=%s hint=%u trials=%llu wins=%llu implicit=%llu saved_q12=%llu\n",
            ref?"P":"KEY",header->order_hint,(unsigned long long)work->video_cfl_trials,
            (unsigned long long)work->video_cfl_wins,(unsigned long long)work->video_cfl_implicit,
            (unsigned long long)work->video_cfl_saved_q12);
#endif
#if SVT_AV2_EXPERIMENTAL_CFL_PAIR_RD
    const char* cfl_pair_trace=getenv("SVT_AV2_CFL_PAIR_TRACE");
    if (work->cfl_pair_rd && cfl_pair_trace && !strcmp(cfl_pair_trace,"1") && error==SVT_AV2_ERROR_NONE)
        fprintf(stderr,"cfl_pair leaves=%llu trials=%llu wins=%llu one_zero=%llu verified=%llu\n",
            (unsigned long long)work->cfl_pair_leaves,(unsigned long long)work->cfl_pair_trials,
            (unsigned long long)work->cfl_pair_wins,(unsigned long long)work->cfl_pair_one_zero,
            (unsigned long long)work->cfl_pair_verified);
#endif
#if SVT_AV2_EXPERIMENTAL_INTRA_EOB
    const char* eob_trace = getenv("SVT_AV2_INTRA_EOB_TRACE");
    if (work->intra_eob_enabled && eob_trace && strcmp(eob_trace, "1") == 0 && error == SVT_AV2_ERROR_NONE)
        fprintf(stderr, "intra_eob kind=%s hint=%u trials=%llu accepts=%llu levels=%llu saved_q12=%llu\n",
            ref ? "P" : header->multi_frame ? "KEY" : "STILL", header->order_hint,
            (unsigned long long)work->intra_eob_trials, (unsigned long long)work->intra_eob_wins,
            (unsigned long long)work->intra_eob_levels, (unsigned long long)work->intra_eob_rate_saved);
#endif
#if SVT_AV2_INTER_COARSE_PREFIX
    if (work->coarse_prefix != NULL) {
        const RdCoarsePrefix* cache = work->coarse_prefix;
        if (cache->trace && error == SVT_AV2_ERROR_NONE)
            fprintf(stderr, "inter_coarse_prefix hint=%u trees=%llu rank_hits=%llu leaf_hits=%llu bytes=%zu\n",
                header->order_hint, (unsigned long long)cache->trees, (unsigned long long)cache->rank_hits,
                (unsigned long long)cache->leaf_hits, sizeof(*cache));
#if SVT_AV2_INTER_COARSE_RATE_PREFIX
        if (cache->rate_trace && cache->rate_cache != NULL && error == SVT_AV2_ERROR_NONE) {
            uint64_t leaves = 0u, events = 0u;
            size_t memory = 0u;
            svt_av2_tile_inter_rate_cache_stats(cache->rate_cache, &leaves, &events, &memory);
            fprintf(stderr, "inter_coarse_rate hint=%u trees=%llu candidates=%llu leaves=%llu events=%llu bytes=%zu verified=%u\n",
                header->order_hint, (unsigned long long)cache->rate_trees, (unsigned long long)cache->rate_candidates,
                (unsigned long long)leaves, (unsigned long long)events, memory,
                (unsigned)SVT_AV2_INTER_COARSE_RATE_PREFIX_VERIFY);
        }
        svt_av2_tile_inter_rate_cache_free(cache->rate_cache);
#endif
        free(work->coarse_prefix);
    }
#endif
    free(work->smooth_grid);
    free(work);
    return error;
}
