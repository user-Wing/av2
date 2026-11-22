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
#ifndef M6_ROOT_ASYM
#define M6_ROOT_ASYM 0
#endif

/*
 * SVT-AV2 - M1 closed loop key frame writer (T005).
 * See tile_writer.h for the bit layout and the spec references.
 *
 * Layout note: the tile group payload is a header region written with the raw
 * BitWriter (is_first_tile_group + frame_header, then byte_alignment) followed
 * by the arithmetic-coded symbol bytes. The two regions are built in separate
 * buffers and concatenated because they use different coders; the boundary is
 * byte aligned by construction (byte_alignment() forces it).
 */
#include "tile_writer.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "bit_writer.h"
#include "block_writer.h"
#include "entropy_encoder.h"
#include "entropy_models.h"
#include "coeff_syntax.h"
#include "cdf_tables.h"
#include "block_geometry.h"
#include "intra_mode_map.h"
#include "intra_prediction.h"
#include <string.h>

/* Forced BO4 profile can need 65 header bits before byte_alignment: 9 bytes. */
#define SVT_AV2_TILE_HEADER_MAX_BYTES 16u

/* Forward boundaries from AVM entropy_inits_modes.h, shared-plane rows. */
static const uint16_t split_cdf[64] = {
    28084,23755,23634,19368,24961,14941,16154,5905,
    21934,10440,11984,3474,20492,6963,8099,1529,
    24117,7871,23604,8429,27356,22441,8897,6811,
    17592,5648,5339,1082,26143,11379,20142,7401,
    26235,23674,12441,10482,20663,4192,5274,713,
    28255,27370,23527,20990,26727,21187,25324,17838,
    26136,16591,19838,10605,22914,12609,11341,4556,
    24218,13059,15378,5858,21644,7767,8309,1687
};
static const uint16_t rect_cdf[64] = {
    14644,10173,18529,16071,20263,12813,26612,23277,
    10594,7000,20002,12889,13854,10750,18380,17505,
    14430,11554,20078,19097,15278,10137,21921,14621,
    19330,15921,26218,24318,16384,16384,16384,16384,
    16384,16384,16384,16384,16066,9225,22849,14817,
    16384,16384,16384,16384,16384,16384,16384,16384,
    18543,13210,24367,18417,24701,18911,29590,27778,
    3400,935,10365,1723,16384,16384,16384,16384
};

/* One report for a successful formal P tile only. RD trial trees always have
 * a null trace pointer and never query the environment or snapshot the coder. */
typedef struct TileInterTrace {
    uint64_t intra_leaves, inter_leaves, skip_txfm_leaves, newmv_leaves;
    uint64_t zero_mv_leaves, fractional_mv_leaves;
    uint64_t coeff_slots[3], nonzero[3], inter_nonzero[3];
    uint64_t partition_q12, motion_q12, intra_mode_q12, coeff_q12[3];
} TileInterTrace;

/* A range-width proxy, not the RD estimator or a per-syntax byte count.
 * In ent_normalize(), each shift/bypass bit increments offs*8 + cnt (an
 * emitted byte increments offs and subtracts 8 from cnt). The normalized
 * range is [32768,65535]; cost_q12(round(rng/2)) - 4096 approximates the
 * remaining -log2(rng/32768) correction. The initial coder's tell is zero.
 * Differences before finalization count what each span consumes from the
 * live range, including the actual coder's adapted probabilities and bypass. */
static uint64_t tile_trace_tell_q12(const SvtAv2EntropyEncoder* enc) {
    const int64_t bits = (int64_t)enc->offs * 8 + enc->cnt + 9;
    uint32_t frac = 0u;
    (void)svt_av2_entropy_probability_cost_q12((enc->rng + 1u) >> 1u, &frac);
    return (uint64_t)(bits * 4096 + frac - 4096);
}

static void tile_trace_print(const TileInterTrace* t, const SvtAv2FrameHeaderConfig* cfg,
    uint64_t range_q12, size_t header_size, size_t coded_size) {
    fprintf(stderr, "SVT_AV2_INTER_TRACE {\"schema\":1,\"order_hint\":%" PRIu32
        ",\"width\":%" PRIu32 ",\"height\":%" PRIu32 ",\"bit_depth\":%" PRIu32
        ",\"chroma_format\":%" PRIu32 ",\"base_q_idx\":%" PRIu32
        ",\"mv_half\":%" PRIu32 ",\"entropy_adaptation\":%" PRIu32
        ",\"leaf_counts\":{\"intra\":%" PRIu64 ",\"inter\":%" PRIu64
        ",\"skip_txfm\":%" PRIu64 ",\"newmv\":%" PRIu64 ",\"zero_mv\":%" PRIu64 ",\"fractional_mv\":%" PRIu64 "}"
        ",\"coeff_slots_yuv\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]"
        ",\"nonzero_yuv\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]"
        ",\"inter_nonzero_yuv\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]"
        ",\"range_proxy_q12\":{\"partition\":%" PRIu64 ",\"motion\":%" PRIu64
        ",\"intra_mode\":%" PRIu64 ",\"coeff_yuv\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64 "],\"total\":%" PRIu64 "}"
        ",\"tile_header_bytes\":%zu,\"tile_entropy_bytes\":%zu,\"tile_payload_bytes\":%zu}\n",
        cfg->order_hint, cfg->frame_width, cfg->frame_height, cfg->bit_depth, cfg->chroma_format,
        cfg->base_q_idx, cfg->mv_half, cfg->entropy_adaptation, t->intra_leaves, t->inter_leaves,
        t->skip_txfm_leaves, t->newmv_leaves, t->zero_mv_leaves, t->fractional_mv_leaves,
        t->coeff_slots[0], t->coeff_slots[1], t->coeff_slots[2], t->nonzero[0], t->nonzero[1], t->nonzero[2],
        t->inter_nonzero[0], t->inter_nonzero[1], t->inter_nonzero[2], t->partition_q12, t->motion_q12,
        t->intra_mode_q12, t->coeff_q12[0], t->coeff_q12[1], t->coeff_q12[2], range_q12,
        header_size, coded_size, header_size + coded_size);
}

typedef struct TileTree {
    SvtAv2EntropyEncoder* enc;
    TileInterTrace* trace; /* Non-NULL only in the optional formal P writer. */
    SvtAv2FrameGeometry geometry;
    const SvtAv2TileCoeffBlock* blocks;
    size_t count, next;
    uint32_t chroma_shift, q_ctx;
    const SvtAv2TileBlockMode* modes;
    size_t mode_count;
    const uint16_t* partition_decisions;
    size_t partition_decision_count;
    uint32_t m5_intra_modes;
    uint32_t reduced_tx_set;
    uint32_t enable_mrls;
    uint32_t enable_cfl_intra;
    uint32_t enable_intra_ist;
    uint32_t enable_mhccp;
    const SvtAv2Ccso64Plan* ccso64;
    uint32_t enable_pc_wiener;
    uint32_t pc_wiener_unit;
    const uint64_t* pc_wiener_none_mask;
    uint16_t pc_wiener_cdf[6];
    uint16_t ccso_cdf[2][4][6]; /* Canonical inverse boundary, terminal, count, AVM_PARA2. */
    uint8_t mrl_above[SVT_AV2_COEFF_CTX_UNITS];
    uint8_t mrl_left[SVT_AV2_COEFF_CTX_UNITS];
    uint8_t m5_above_mode[SVT_AV2_COEFF_CTX_UNITS];
    uint8_t m5_left_mode[SVT_AV2_COEFF_CTX_UNITS];
    uint8_t cfl_above[SVT_AV2_COEFF_CTX_UNITS];
    uint8_t cfl_left[SVT_AV2_COEFF_CTX_UNITS];
    SvtAv2MotionSyntax motion;
    uint8_t above_partition[SVT_AV2_COEFF_CTX_UNITS];
    uint8_t left_partition[SVT_AV2_COEFF_CTX_UNITS];
    SvtAv2CoeffNeighbours neighbours[3];
} TileTree;

/* M5: the v1.0.0 decoder's joint-mode ordering, reduced to modes 0/1/2
 * that the current selector may emit (DC/V/H, angle delta zero).
 * See fixed reconintra.h: default_mode_list_y and reconintra.c:391. */
static const uint8_t m5_default_mode_list[56] = {
    17,45,3,10,24,31,38,52,15,19,43,47,1,5,8,12,
    22,26,29,33,36,40,50,54,16,18,44,46,2,4,9,11,
    23,25,30,32,37,39,51,53,14,20,42,48,0,6,7,13,
    21,27,28,34,35,41,49,55
};
static const uint16_t m5_y_cdf[3][7] = {
    {15175,20075,21728,24098,26405,27655,28860},
    {10114,14957,16815,19127,20147,25583,27169},
    {5636,9004,10456,12122,12744,20325,25607}
};
static const uint16_t m5_y_offset_cdf[3][5] = {
    {12743,18172,20194,23648,26419},
    {8976,16084,20827,24595,28496},
    {8784,14556,19710,24903,28724}
};
static void m5_add_joint(uint8_t* list, uint8_t* present, uint32_t* size,
                         uint32_t joint) {
    if (!present[joint]) {
        list[(*size)++] = (uint8_t)joint;
        present[joint] = 1u;
    }
}
static int m5_write_intra_y_mode(TileTree* tree, const SvtAv2BlockView* b,
                               uint32_t mode, uint32_t uv_mode) {
    const int is_cfl = tree->enable_cfl_intra && uv_mode == 14u;
    if (mode > (tree->m5_intra_modes >= 3u ? 61u : 7u) || mode == 3u ||
        (uv_mode > 7u && !is_cfl) || uv_mode == 3u ||
        (tree->m5_intra_modes < 2u && (mode > 2u || (uv_mode != 0u && !is_cfl))))
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    const uint32_t x = b->x / 4u, y = b->y / 4u;
    const uint32_t w = b->w / 4u, h = b->h / 4u;
    if (w == 0u || h == 0u || x + w > SVT_AV2_COEFF_CTX_UNITS ||
        y + h > SVT_AV2_COEFF_CTX_UNITS)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    const uint32_t left_mode = b->x ? tree->m5_left_mode[y + h - 1u] : 0u;
    const uint32_t above_mode = b->y ? tree->m5_above_mode[x + w - 1u] : 0u;
    const uint32_t left = svt_av2_intra_mode_joint(left_mode);
    const uint32_t above = svt_av2_intra_mode_joint(above_mode);
    const uint32_t context = (left >= 5u) + (above >= 5u);
    uint32_t directional[2] = {left, above};
    uint32_t count = context;
    if (count == 2u && left == above) count = 1u;
    if (count == 1u && left < 5u) directional[0] = above;
    uint8_t list[61] = {0}, present[61] = {0};
    uint32_t list_size = 0u;
    for (uint32_t j = 0; j < 5u; ++j)
        m5_add_joint(list, present, &list_size, j);
    if (b->w >= 8u && b->h >= 8u) {
        for (uint32_t j = 0; j < count; ++j)
            m5_add_joint(list, present, &list_size, directional[j]);
        if (b->w * b->h > 64u) {
            for (uint32_t offset = 0; offset < 4u; ++offset)
                for (uint32_t j = 0; j < count; ++j) {
                    const uint32_t n = directional[j];
                    m5_add_joint(list, present, &list_size,
                                 (n - offset + 50u) % 56u + 5u);
                    m5_add_joint(list, present, &list_size,
                                 (n + offset - 4u) % 56u + 5u);
                }
        }
    }
    for (uint32_t j = 0; j < 56u; ++j)
        m5_add_joint(list, present, &list_size, (uint32_t)m5_default_mode_list[j] + 5u);
    const uint32_t goal = svt_av2_intra_mode_joint(mode);
    uint32_t index = 0u;
    while (index < list_size && list[index] != goal) ++index;
    if (index >= list_size) return SVT_AV2_ENTROPY_ERROR_RANGE;

    const uint32_t set = index < 13u ? 0u : 1u + (index - 13u) / 16u;
    int st = svt_av2_entropy_encode_model(tree->enc, SVT_AV2_MODEL_Y_SET, svt_av2_cdf_y_mode_set, 4u, set);
    if (st != 0) return st;
    if (set == 0u) {
        const uint32_t symbol = index < 7u ? index : 7u;
        st = svt_av2_entropy_encode_model(tree->enc, SVT_AV2_MODEL_Y_IDX + context, m5_y_cdf[context], 8u, symbol);
        if (st == 0 && symbol == 7u)
            st = svt_av2_entropy_encode_model(tree->enc, SVT_AV2_MODEL_Y_OFFSET + context, m5_y_offset_cdf[context],
                                                6u, index - 7u);
    } else {
        st = svt_av2_entropy_encode_literal(tree->enc, (index - 13u) % 16u, 4);
    }
    const uint32_t mrl = tree->enable_mrls ? tree->blocks[tree->next].mrl : 0u;
    const uint32_t directional_mode = svt_av2_intra_mode_joint(mode) >= 5u;
    if (mrl > 7u || mrl == 4u || (mrl && (!tree->enable_mrls || !directional_mode ||
        (b->w!=b->h && b->w!=2u*b->h && b->h!=2u*b->w &&
            !((b->w==16u && b->h==64u) || (b->w==64u && b->h==16u))) ||
        b->w<8u || b->h<8u))) return SVT_AV2_ENTROPY_ERROR_RANGE;
    if (tree->enable_mrls && directional_mode) {
        static const uint16_t mrl_cdf[3][3] = {
            {29573u,31193u,32023u},{21812u,27066u,30279u},{16076u,23806u,28762u}};
        static const uint16_t multi_cdf[3] = {19678u,12287u,9574u};
        /* fetch_spatial_neighbors selects BL, AR, L, A and excludes above
         * at the SB top. With only one side it can count that side twice. */
        uint32_t neighbors[2] = {0u,0u}, neighbor_count = 0u;
        if (b->x) neighbors[neighbor_count++] = tree->mrl_left[y+h-1u];
        if (b->y % 64u && neighbor_count < 2u) neighbors[neighbor_count++] = tree->mrl_above[x+w-1u];
        if (b->x && neighbor_count < 2u) neighbors[neighbor_count++] = tree->mrl_left[y];
        if (b->y % 64u && neighbor_count < 2u) neighbors[neighbor_count++] = tree->mrl_above[x];
        const uint32_t ctx = ((neighbors[0]&3u)!=0u) + ((neighbors[1]&3u)!=0u);
        st = svt_av2_entropy_encode_model(tree->enc,SVT_AV2_MODEL_MRL+ctx,mrl_cdf[ctx],4u,mrl&3u);
        if (st == 0 && (mrl&3u)) {
            const uint32_t multi_ctx = (neighbors[0]>>2u) + (neighbors[1]>>2u);
            st = svt_av2_entropy_encode_model(tree->enc,SVT_AV2_MODEL_MULTI_MRL+multi_ctx,
                multi_cdf+multi_ctx,2u,mrl>>2u);
        }
        if (st != 0) return st;
    }
    memset(tree->mrl_above+x,(int)mrl,w);
    memset(tree->mrl_left+y,(int)mrl,h);
    return st;
}

static int m5_write_intra_uv_mode(TileTree* tree,const SvtAv2BlockView* b,
    uint32_t mode,uint32_t uv_mode) {
    const uint32_t x=b->x/4u,y=b->y/4u,w=b->w/4u,h=b->h/4u;
    int st=0;
    /* AVM get_uv_intra_mode_set: directional co-located Y first, followed by
     * DC/SMOOTH/SMOOTH_V/SMOOTH_H/PAETH and the remaining V/H/... modes. */
    const uint32_t base_mode = svt_av2_intra_mode_base(mode);
    const uint32_t uv_context = svt_av2_intra_mode_joint(mode) >= 5u;
    uint32_t uv_index = uv_mode == 0u ? 0u : uv_mode >= 4u ? uv_mode - 3u : uv_mode + 4u;
    if (uv_context) {
        if (uv_mode == base_mode) uv_index = 0u;
        else if (uv_mode == 2u && base_mode == 1u) uv_index = 6u;
        else ++uv_index;
    }
    if (st == 0 && tree->enable_cfl_intra) {
        /* Chroma uses the above/left at the start of the shared-tree leaf,
         * unlike the end-of-span luma mode context. */
        static const uint16_t cfl_cdf[3] = {20441u,11610u,4643u};
        const uint32_t cfl_context = (b->x ? tree->cfl_left[y] : 0u) + (b->y ? tree->cfl_above[x] : 0u);
        st = svt_av2_entropy_encode_model(tree->enc, SVT_AV2_MODEL_CFL + cfl_context,
            cfl_cdf + cfl_context, 2u, uv_mode == 14u);
        if (st == 0 && uv_mode == 14u) {
            static const uint16_t cfl_index_cdf[1] = {12507u};
            const SvtAv2TileCoeffBlock* input = tree->blocks + tree->next;
            const int mhccp_allowed = tree->enable_mhccp &&
                svt_av2_intra_mhccp_allowed(b->w,b->h,tree->geometry.chroma_format);
            if (input->cfl_explicit > 1u || input->mhccp > 3u ||
                (input->mhccp && (!mhccp_allowed || input->cfl_explicit))) return SVT_AV2_ENTROPY_ERROR_RANGE;
            if (mhccp_allowed) {
                static const uint16_t mhccp_cdf[1] = {15499u};
                st = svt_av2_entropy_encode_model(tree->enc,SVT_AV2_MODEL_MHCCP,mhccp_cdf,2u,input->mhccp != 0u);
            }
            if (st == 0 && input->mhccp) {
                static const uint16_t dir_cdf[4][2] = {{10923u,21845u},{8795u,15105u},{10433u,15974u},{17085u,21689u}};
                const uint32_t group = svt_av2_intra_mhccp_size_group(b->w,b->h);
                st = svt_av2_entropy_encode_model(tree->enc,SVT_AV2_MODEL_MHCCP_DIR+group,dir_cdf[group],3u,input->mhccp-1u);
            } else if (st == 0) st = svt_av2_entropy_encode_model(tree->enc, SVT_AV2_MODEL_CFL_INDEX,
                cfl_index_cdf, 2u, input->cfl_explicit ? 0u : 1u);
            if (st == 0 && input->cfl_explicit) {
                static const uint16_t sign_cdf[7] = {2421u,4332u,11256u,12766u,21386u,28725u,32087u};
                static const uint16_t alpha_cdf[6][7] = {
                    {21679u,25305u,30646u,31512u,32537u,32646u,32696u},
                    {8262u,16302u,24082u,29422u,31398u,32286u,32525u},
                    {17235u,26166u,30378u,31305u,32373u,32549u,32668u},
                    {17618u,25732u,27865u,30338u,31125u,31522u,32238u},
                    {17542u,23066u,27907u,28728u,30702u,31165u,31435u},
                    {17675u,24802u,30468u,30783u,31841u,32264u,32422u}
                };
                const uint32_t joint=input->cfl_alpha_signs, idx=input->cfl_alpha_idx;
                if (joint >= 8u || idx >= 64u) return SVT_AV2_ENTROPY_ERROR_RANGE;
                const uint32_t usign=(joint+1u)/3u, vsign=(joint+1u)%3u;
                if ((!usign && (idx>>3u)) || (!vsign && (idx&7u)))
                    return SVT_AV2_ENTROPY_ERROR_RANGE;
                st=svt_av2_entropy_encode_model(tree->enc,SVT_AV2_MODEL_CFL_SIGN,sign_cdf,8u,joint);
                if (st==0 && usign) {
                    const uint32_t ctx=joint+1u-3u;
                    st=svt_av2_entropy_encode_model(tree->enc,SVT_AV2_MODEL_CFL_ALPHA+ctx,alpha_cdf[ctx],8u,idx>>3u);
                }
                if (st==0 && vsign) {
                    const uint32_t ctx=vsign*3u+usign-3u;
                    st=svt_av2_entropy_encode_model(tree->enc,SVT_AV2_MODEL_CFL_ALPHA+ctx,alpha_cdf[ctx],8u,idx&7u);
                }
            }
        }
    }
    if (st == 0 && uv_mode != 14u)
        st = svt_av2_entropy_encode_model(tree->enc, SVT_AV2_MODEL_UV + uv_context,
            uv_context ? svt_av2_cdf_uv_mode_cfl_not_allowed_1 :
                          svt_av2_cdf_uv_mode_cfl_not_allowed_0, 8u, uv_index);
    if (st == 0 && uv_mode != 14u && uv_index == 7u) st = svt_av2_entropy_encode_literal(tree->enc, 0u, 3);
    if (st == 0) {
        memset(tree->m5_above_mode + x, (int)mode, w);
        memset(tree->m5_left_mode + y, (int)mode, h);
        if (tree->enable_cfl_intra) {
            memset(tree->cfl_above + x, uv_mode == 14u, w);
            memset(tree->cfl_left + y, uv_mode == 14u, h);
        }
    }
    return st;
}

static int m5_write_intra_mode(TileTree* tree,const SvtAv2BlockView* b,
    uint32_t mode,uint32_t uv_mode) {
    const int st=m5_write_intra_y_mode(tree,b,mode,uv_mode);
    return st==0?m5_write_intra_uv_mode(tree,b,mode,uv_mode):st;
}

static uint32_t log2_dim(uint32_t d) {
    uint32_t n = 0;
    while ((1u << n) < d) ++n;
    return n;
}

static int tile_partition_write(TileTree* tree, const SvtAv2BlockView* b,
    SvtAv2PartitionType part, int implied) {
    const uint32_t lw = log2_dim(b->w), lh = log2_dim(b->h);
    const uint32_t above = (tree->above_partition[b->x / 4u] >> (lw - 3u)) & 1u;
    const uint32_t left = (tree->left_partition[b->y / 4u] >> (lh - 3u)) & 1u;
    const uint32_t raw = 2u * left + above;
    int st = 0;
    if (!implied) {
        const int aspect4 = b->w == 4u * b->h || b->h == 4u * b->w;
        const uint32_t size_map = aspect4 ?
            10u + 2u * ((lw < lh ? lw : lh) - 2u) + (b->w > b->h) : (lw > lh ? lw : lh) - 3u;
        /* Missing half-block rows/columns forbid NONE. In that case do_split
         * is implied even when the binary orientation itself is signaled. */
        if (b->x + b->w / 2u < tree->geometry.visible_width &&
            b->y + b->h / 2u < tree->geometry.visible_height) {
            st = svt_av2_entropy_encode_model(tree->enc, SVT_AV2_MODEL_SPLIT + size_map * 4u + raw, split_cdf + size_map * 4u + raw,
                                              2u, part != SVT_AV2_PARTITION_NONE);
        }
        if (st != 0) return st;
        if (part != SVT_AV2_PARTITION_NONE) {
            uint32_t rect_map;
            if (aspect4) rect_map = b->h > b->w ? 13u : 14u;
            else if (b->w == b->h) rect_map = b->w == 64u ? 6u : (b->w == 32u ? 3u : 0u);
            else if (b->h > b->w) rect_map = b->w == 32u ? 4u : 1u;
            else rect_map = b->h == 32u ? 5u : 2u;
            st = svt_av2_entropy_encode_model(tree->enc, SVT_AV2_MODEL_RECT + rect_map * 4u + raw, rect_cdf + rect_map * 4u + raw,
                                              2u, part == SVT_AV2_PARTITION_VERT);
            if (st != 0) return st;
        }
    }
    return 0;
}

static int tile_coeff_plane_write(TileTree* tree,const SvtAv2BlockView* b,
    uint32_t p,uint32_t is_inter,uint32_t skip_txfm,uint32_t m5_mode,uint8_t* u_nonzero) {
    int st=0;
    const SvtAv2TileCoeffBlock* input = tree->blocks + tree->next++;
    const uint32_t width = b->w >> (p == 0u ? 0u : tree->chroma_shift);
    const uint32_t height = b->h >> (p == 0u ? 0u : tree->chroma_shift);
    const uint32_t x = b->x >> (p == 0u ? 0u : tree->chroma_shift);
    const uint32_t y = b->y >> (p == 0u ? 0u : tree->chroma_shift);
    if (input->plane != p || input->x != x || input->y != y ||
        input->width != width || input->height != height || input->qcoeff == NULL)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    SvtAv2TxSize tx_size;
    if (svt_av2_coeff_tx_size_from_dims(width,height,&tx_size) != 0)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    SvtAv2CoeffPlaneInfo pi = {p, tx_size, x / 4u, y / 4u,
        tree->geometry.visible_width / (4u << (p == 0u ? 0u : tree->chroma_shift)),
        tree->geometry.visible_height / (4u << (p == 0u ? 0u : tree->chroma_shift)), width, height};
#if M6_FLIP_TX
    if (p==0u) m5_mode=svt_av2_intra_tx_mode(input->intra_mode,width,height,input->mrl);
#else
    if (p==0u && tree->enable_intra_ist)
        m5_mode=svt_av2_intra_tx_mode(input->intra_mode,width,height,input->mrl);
#endif
    if (skip_txfm) {
        memset(tree->neighbours[p].above_level + x/4u, 0, width/4u);
        memset(tree->neighbours[p].above_dc + x/4u, 0, width/4u);
        memset(tree->neighbours[p].left_level + y/4u, 0, height/4u);
        memset(tree->neighbours[p].left_dc + y/4u, 0, height/4u);
    } else st = svt_av2_coeff_write_block_transform_ist(tree->enc, &pi, *u_nonzero,
        input->qcoeff, tree->neighbours + p, tree->q_ctx, is_inter, tree->reduced_tx_set,
        tree->reduced_tx_set <= 1u ? input->tx_type : 0u, m5_mode, tree->enable_intra_ist,
        tree->enable_intra_ist ? input->ist : 0u);
    if (st != SVT_AV2_COEFF_OK) {
        st = st == SVT_AV2_COEFF_ERR_NO_SPACE ? SVT_AV2_ENTROPY_ERROR_NO_SPACE
                                             : SVT_AV2_ENTROPY_ERROR_RANGE;
    }
    if (p == 1u) {
        const size_t count = svt_av2_coeff_coeff_count(pi.tx_size);
        for (size_t c = 0; c < count; ++c) *u_nonzero |= input->qcoeff[c] != 0;
    }
    return st;
}

/* Inline rows follow every TileTree trial/commit; failed trials retain no updates. */
static void tile_ccso64_init(TileTree* tree, const SvtAv2Ccso64Plan* plan) {
    tree->ccso64=plan;
    if (plan) svt_av2_ccso64_cdf_init(tree->ccso_cdf);
}

static int tile_ccso64_write(TileTree* tree, const SvtAv2BlockView* b) {
    if (tree->ccso64 == NULL || (b->x & 63u) || (b->y & 63u)) return 0;
    const size_t unit = (size_t)(b->y / 64u) * tree->geometry.sb_cols + b->x / 64u;
    const SvtAv2Ccso64Plan* plan = tree->ccso64;
    const uint8_t* flags = plan->unit_flags ? plan->unit_flags[unit] : plan->block_flag;
    for (uint32_t p = 0u; p < 2u; ++p) if (plan->enabled[p]) {
        const uint32_t context=svt_av2_ccso64_context(plan,unit,tree->geometry.sb_cols,p);
        const int st=svt_av2_ccso64_symbol(tree->enc,tree->ccso_cdf[p][context],flags[p],
            tree->enc->models!=NULL);
        if (st != 0) return st;
    }
    return 0;
}

static int tile_leaf_write(TileTree* tree, const SvtAv2BlockView* b) {
    int st=0;
    if (tree->next + 3u > tree->count) return SVT_AV2_ENTROPY_ERROR_RANGE;
    TileInterTrace* const trace = tree->trace;
    uint64_t tell = trace ? tile_trace_tell_q12(tree->enc) : 0u;
    uint32_t is_inter = 0, skip_txfm = 0;
    if (tree->modes) {
        const size_t idx = tree->next / 3u;
        if (idx >= tree->mode_count) return SVT_AV2_ENTROPY_ERROR_RANGE;
        const SvtAv2TileBlockMode* mode = tree->modes + idx;
        is_inter = mode->is_inter;
        skip_txfm = is_inter;
        for (uint32_t p = 0; p < 3u; ++p) {
            const SvtAv2TileCoeffBlock* in = tree->blocks + tree->next + p;
            if (in->qcoeff == NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
            const uint32_t dim = in->width < 32u ? in->width : 32u;
            const uint32_t height = in->height < 32u ? in->height : 32u;
            if (trace) {
                /* Same coded transform domain as skip and coeff syntax. */
                uint32_t nz = 0u;
                for (uint32_t c = 0u; c < dim * height; ++c) nz += in->qcoeff[c] != 0;
                trace->coeff_slots[p] += dim * height;
                trace->nonzero[p] += nz;
                if (is_inter) trace->inter_nonzero[p] += nz;
                if (nz != 0u) skip_txfm = 0u;
            } else {
                for (uint32_t c = 0; c < dim * height; ++c) if (in->qcoeff[c] != 0) skip_txfm = 0;
            }
        }
        st = svt_av2_motion_write_block(&tree->motion, tree->enc, b, mode, skip_txfm);
        if (trace && st == 0) {
            const uint64_t after = tile_trace_tell_q12(tree->enc);
            trace->motion_q12 += after - tell;
            tell = after;
            if (is_inter) {
                const SvtAv2MotionCell* written = tree->motion.cells +
                    (size_t)(b->y / 4u) * tree->motion.cols + b->x / 4u;
                ++trace->inter_leaves;
                trace->skip_txfm_leaves += skip_txfm;
                trace->newmv_leaves += written->is_new;
                trace->zero_mv_leaves += mode->mv_x == 0 && mode->mv_y == 0;
                const uint32_t frac_mask = (1u << tree->motion.mv_half) - 1u;
                trace->fractional_mv_leaves +=
                    (((uint32_t)mode->mv_x | (uint32_t)mode->mv_y) & frac_mask) != 0u;
            } else ++trace->intra_leaves;
        }
    }
    /* Fixed AVM blockd.c av2_get_joint_mode: an inter neighbour contributes
     * DC to intra mode ordering, replacing any earlier directional block. */
    if (is_inter && tree->m5_intra_modes && st == 0) {
        memset(tree->m5_above_mode + b->x / 4u, 0, b->w / 4u);
        memset(tree->m5_left_mode + b->y / 4u, 0, b->h / 4u);
        if (tree->enable_cfl_intra) {
            /* The decoder initializes inter UV mode to UV_DC_PRED. */
            memset(tree->cfl_above + b->x / 4u, 0, b->w / 4u);
            memset(tree->cfl_left + b->y / 4u, 0, b->h / 4u);
        }
    }
    if (st == 0) st = tile_ccso64_write(tree, b);
    uint32_t m5_mode = 0u;
    if (!is_inter && tree->m5_intra_modes) {
        /* M5 syntax is enabled explicitly by tile tree configuration. */
        m5_mode = tree->blocks[tree->next].intra_mode;
        if (m5_mode > (tree->m5_intra_modes >= 3u ? 61u : 7u) || m5_mode == 3u || (m5_mode > 2u && tree->m5_intra_modes < 2u))
            return SVT_AV2_ENTROPY_ERROR_RANGE;
    }
    const int coded_uv = tree->m5_intra_modes >= 2u || tree->enable_cfl_intra;
    if (!is_inter && coded_uv &&
        tree->blocks[tree->next + 1u].intra_mode != tree->blocks[tree->next + 2u].intra_mode)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    if (!is_inter && tree->m5_intra_modes && st == 0)
        st = m5_write_intra_mode(tree, b, m5_mode,
            coded_uv ? tree->blocks[tree->next + 1u].intra_mode : 0u);
    if (!is_inter && !tree->m5_intra_modes && st == 0)
        st = svt_av2_entropy_encode_model(tree->enc, SVT_AV2_MODEL_Y_SET, svt_av2_cdf_y_mode_set, 4u, 0u);
    /* AVM alphabets have eight symbols; the M1 seven-symbol approximation is
     * retained only in block_write_minimal_intra's frozen regression path. */
    if (!is_inter && !tree->m5_intra_modes && st == 0)
        st = svt_av2_entropy_encode_model(tree->enc, SVT_AV2_MODEL_Y_IDX, svt_av2_cdf_y_mode_index_0, 8u, 0u);
    if (!is_inter && !tree->m5_intra_modes && st == 0) st = svt_av2_entropy_encode_model(tree->enc, SVT_AV2_MODEL_UV + (m5_mode != 0u),
        m5_mode ? svt_av2_cdf_uv_mode_cfl_not_allowed_1 : svt_av2_cdf_uv_mode_cfl_not_allowed_0,
        8u, m5_mode ? 1u : 0u);
    if (!is_inter && tree->modes && m5_mode == 0u && st == 0) {
        const uint16_t palette_cdf = 30045u;
        st = svt_av2_entropy_encode_model(tree->enc, SVT_AV2_MODEL_PALETTE, &palette_cdf, 2u, 0u);
    }
    uint8_t u_nonzero = 0u;
    if (trace && st == 0) {
        const uint64_t after_mode = tile_trace_tell_q12(tree->enc);
        trace->intra_mode_q12 += after_mode - tell;
        tell = after_mode;
        for (uint32_t p = 0; p < 3u && st == 0; ++p) {
            st=tile_coeff_plane_write(tree,b,p,is_inter,skip_txfm,m5_mode,&u_nonzero);
            if (st == 0) {
                const uint64_t after = tile_trace_tell_q12(tree->enc);
                trace->coeff_q12[p] += after - tell;
                tell = after;
            }
        }
    } else {
        for (uint32_t p = 0; p < 3u && st == 0; ++p)
            st=tile_coeff_plane_write(tree,b,p,is_inter,skip_txfm,m5_mode,&u_nonzero);
    }
    memset(tree->above_partition + b->x / 4u, (int)(64u - b->w / 4u), b->w / 4u);
    memset(tree->left_partition + b->y / 4u, (int)(64u - b->h / 4u), b->h / 4u);
    return st;
}

static int tile_pc_wiener_write(TileTree* tree, const SvtAv2BlockView* b) {
    const uint32_t ru = tree->pc_wiener_unit ? tree->pc_wiener_unit : 256u;
    if (tree->enable_pc_wiener && b->w == 64u && b->h == 64u &&
        b->x % ru == 0u && b->y % ru == 0u) {
        const uint32_t cols = (tree->geometry.visible_width + ru / 2u) / ru;
        const uint32_t rows = (tree->geometry.visible_height + ru / 2u) / ru;
        if (b->x / ru < (cols ? cols : 1u) && b->y / ru < (rows ? rows : 1u)) {
            const uint32_t unit = (b->y / ru) * (cols ? cols : 1u) + b->x / ru;
            if (unit / 64u >= SVT_AV2_PC_WIENER_MASK_WORDS) return SVT_AV2_ENTROPY_ERROR_RANGE;
            const int status = svt_av2_ccso64_symbol(tree->enc, tree->pc_wiener_cdf,
                1u ^ (uint32_t)((tree->pc_wiener_none_mask[unit / 64u] >> (unit & 63u)) & 1u),
                tree->enc->models != NULL);
            if (status != 0) return status;
        }
    }
    return 0;
}

static int tile_tree_write(TileTree* tree, const SvtAv2BlockView* b) {
    if (b->x >= tree->geometry.visible_width || b->y >= tree->geometry.visible_height) return 0;
    const uint64_t tell = tree->trace ? tile_trace_tell_q12(tree->enc) : 0u;
    const int restoration_status = tile_pc_wiener_write(tree, b);
    if (restoration_status != 0) return restoration_status;
    SvtAv2PartitionType part;
    int implied;
    if (svt_av2_partition_coded_type_plan_validated(&tree->geometry, b,
        tree->partition_decisions, &part, &implied) != 0)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    int st=tile_partition_write(tree,b,part,implied);
    if (st!=0) return st;
    /* In this P profile there is no restoration syntax (or leaf CCSO). */
    if (tree->trace) tree->trace->partition_q12 += tile_trace_tell_q12(tree->enc) - tell;
    if (part != SVT_AV2_PARTITION_NONE) {
        SvtAv2BlockView child = *b;
        if (part == SVT_AV2_PARTITION_HORZ) child.h /= 2u;
        else if (part == SVT_AV2_PARTITION_VERT) child.w /= 2u;
        else return SVT_AV2_ENTROPY_ERROR_RANGE;
        st = tile_tree_write(tree, &child);
        if (st != 0) return st;
        if (part == SVT_AV2_PARTITION_HORZ) child.y += child.h;
        else child.x += child.w;
        return tile_tree_write(tree, &child);
    }
    return tile_leaf_write(tree,b);
}

static int tile_mv_precision_valid(const SvtAv2FrameHeaderConfig* cfg) {
    if (cfg->logical_ref_count > 2u || (cfg->logical_ref_count == 2u &&
        (cfg->dual_ref != 1u || cfg->inter_frame != 1u || !cfg->multi_frame ||
         cfg->reference_slot || cfg->retain_reference))) return 0;
    const uint32_t d0 = cfg->ref_distance[0], d1 = cfg->ref_distance[1];
    if ((d0 || d1) && (cfg->logical_ref_count != 2u || d0 < 1u || d0 > 8u || d1 < 1u || d1 > 8u ||
        d0 == d1 || cfg->mv_half != 2u || !cfg->sharp_filter)) return 0;
    return cfg->mv_half == 0u || (cfg->mv_half <= 2u && cfg->inter_frame == 1u &&
        cfg->multi_frame && cfg->sb_shift == SVT_AV2_FRAME_SB_SHIFT_64);
}

static int tile_motion_init(SvtAv2MotionSyntax* motion, const SvtAv2FrameHeaderConfig* cfg) {
    const int st = svt_av2_motion_syntax_init_precision(motion, cfg->frame_width,
         cfg->frame_height, cfg->mv_half);
    if (st == 0) {
        motion->two_refs = cfg->logical_ref_count == 2u;
        motion->ref_distance[0] = cfg->ref_distance[0];
        motion->ref_distance[1] = cfg->ref_distance[1];
    }
    return st;
}

static int tile_write_coeffs(SvtAv2EntropyEncoder* enc, const SvtAv2FrameHeaderConfig* cfg,
                            const SvtAv2TileCoeffBlock* blocks, size_t count,
                            const SvtAv2TileBlockMode* modes, size_t mode_count,
                            TileInterTrace* trace) {
    if (!tile_mv_precision_valid(cfg) || !svt_av2_frame_header_video_cfl_valid(cfg) ||
        ((cfg->mv_half || cfg->logical_ref_count == 2u) && modes == NULL))
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    if (svt_av2_ccso64_validate(cfg->ccso64, cfg->frame_width, cfg->frame_height,
        cfg->bit_depth, cfg->chroma_format) != 0 ||
        (cfg->ccso64 && (cfg->multi_frame || cfg->inter_frame || cfg->sb_shift != 4u)))
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    if ((cfg->base_q_idx < (cfg->bit_depth == 10u ? 49u : 1u) || cfg->base_q_idx > (cfg->bit_depth == 10u ? 303u : 255u)) || cfg->frame_width > 4096u || cfg->frame_height > 2160u)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    TileTree tree;
    memset(&tree, 0, sizeof(tree));
    tree.enc = enc; tree.trace = trace; tree.blocks = blocks; tree.count = count;
    tree.modes = modes; tree.mode_count = mode_count;
    tree.partition_decisions = cfg->partition_decisions;
    tree.partition_decision_count = cfg->partition_decision_count;
    tree.m5_intra_modes = cfg->m5_intra_modes;
    tree.reduced_tx_set = cfg->reduced_tx_set;
    tree.enable_mrls = cfg->enable_mrls;
    tree.enable_cfl_intra = svt_av2_frame_header_use_cfl(cfg);
    tree.enable_intra_ist = cfg->enable_intra_ist;
    tree.enable_mhccp = cfg->enable_mhccp;
    tile_ccso64_init(&tree, cfg->ccso64);
    tree.enable_pc_wiener = cfg->enable_pc_wiener;
    tree.pc_wiener_unit = cfg->pc_wiener_unit;
    tree.pc_wiener_none_mask = cfg->pc_wiener_none_mask;
    { const uint16_t initial[6] = {19969, 0, 0, 1, 3, 4};
      memcpy(tree.pc_wiener_cdf, initial, sizeof(initial)); }
    tree.chroma_shift = cfg->chroma_format == 3u ? 0u : 1u;
    tree.q_ctx = svt_av2_coeff_q_context(cfg->base_q_idx);
    if (svt_av2_frame_geometry_init_ex(&tree.geometry, cfg->frame_width, cfg->frame_height, 64u,
        (SvtAv2ChromaFormat)cfg->chroma_format) != 0)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    if (svt_av2_partition_plan_validate(&tree.geometry,
        tree.partition_decisions, tree.partition_decision_count) != 0)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    if (modes) {
        const int st = tile_motion_init(&tree.motion, cfg);
        if (st != 0) return st;
    }
    for (uint32_t y = 0; y < cfg->frame_height; y += 64u) {
        for (uint32_t x = 0; x < cfg->frame_width; x += 64u) {
            SvtAv2BlockView sb = {0};
            sb.x = x; sb.y = y; sb.w = sb.h = 64u;
            const int st = tile_tree_write(&tree, &sb);
            if (st != 0) { svt_av2_motion_syntax_free(&tree.motion); return st; }
        }
    }
    svt_av2_motion_syntax_free(&tree.motion);
    return tree.next == count ? 0 : SVT_AV2_ENTROPY_ERROR_RANGE;
}

struct SvtAv2TileRateSession {
    TileTree tree;
    uint64_t pc_wiener_mask[SVT_AV2_PC_WIENER_MASK_WORDS];
    uint16_t* plan;
    size_t cursor, sb_count;
    uint32_t inter_frame;
    SvtAv2MotionCell saved_motion[16u * 16u];
    uint32_t adaptive;
    SvtAv2EntropyModels committed_models, scratch_models;
};

static int tile_models_seed(const SvtAv2FrameHeaderConfig* cfg, SvtAv2EntropyModels* models) {
    if (cfg->primary_models != NULL) {
        if (!cfg->inter_frame || !cfg->multi_frame || cfg->entropy_adaptation != 1u)
            return SVT_AV2_ENTROPY_ERROR_RANGE;
        *models = *cfg->primary_models;
        return 0;
    }
    return svt_av2_entropy_models_init(models, svt_av2_coeff_q_context(cfg->base_q_idx));
}

int svt_av2_tile_rate_create(const SvtAv2FrameHeaderConfig* cfg,
                             SvtAv2TileRateSession** session) {
    if (session == NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
    *session = NULL;
    if (cfg == NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
    if (!tile_mv_precision_valid(cfg) || !svt_av2_frame_header_video_cfl_valid(cfg))
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    if (svt_av2_ccso64_validate(cfg->ccso64, cfg->frame_width, cfg->frame_height,
        cfg->bit_depth, cfg->chroma_format) != 0 ||
        (cfg->ccso64 && (cfg->multi_frame || cfg->inter_frame || cfg->sb_shift != 4u)))
        return SVT_AV2_ENTROPY_ERROR_RANGE;

    if (cfg->entropy_adaptation > 1u) return SVT_AV2_ENTROPY_ERROR_RANGE;
    const uint32_t qmin = cfg->bit_depth == 10u ? 49u : 1u;
    const uint32_t qmax = cfg->bit_depth == 10u ? 303u : 255u;
    if (cfg->base_q_idx < qmin || cfg->base_q_idx > qmax ||
        cfg->frame_width > 4096u || cfg->frame_height > 2160u ||
        (cfg->bit_depth != 8u && cfg->bit_depth != 10u) ||
        (cfg->chroma_format != 1u && cfg->chroma_format != 3u) ||
        (cfg->bit_depth == 8u && cfg->chroma_format == 3u) || cfg->inter_frame > 1u)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    SvtAv2TileRateSession* s = (SvtAv2TileRateSession*)calloc(1u, sizeof(*s));
    if (s == NULL) return SVT_AV2_ENTROPY_ERROR_NO_SPACE;
    if (svt_av2_frame_geometry_init_ex(&s->tree.geometry, cfg->frame_width,
        cfg->frame_height, 64u, (SvtAv2ChromaFormat)cfg->chroma_format) != 0) {
        free(s); return SVT_AV2_ENTROPY_ERROR_RANGE;
    }
    s->sb_count = (size_t)s->tree.geometry.sb_cols * s->tree.geometry.sb_rows;
    s->plan = (uint16_t*)calloc(s->sb_count, sizeof(*s->plan));
    if (s->plan == NULL) { free(s); return SVT_AV2_ENTROPY_ERROR_NO_SPACE; }
    s->tree.partition_decisions = s->plan;
    s->tree.partition_decision_count = s->sb_count;
    s->tree.m5_intra_modes = cfg->m5_intra_modes;
    s->tree.reduced_tx_set = cfg->reduced_tx_set;
    s->tree.enable_mrls = cfg->enable_mrls;
    s->tree.enable_cfl_intra = svt_av2_frame_header_use_cfl(cfg);
    s->tree.enable_intra_ist = cfg->enable_intra_ist;
    s->tree.enable_mhccp = cfg->enable_mhccp;
    tile_ccso64_init(&s->tree, cfg->ccso64);
    s->tree.enable_pc_wiener = cfg->enable_pc_wiener;
    s->tree.pc_wiener_unit = cfg->pc_wiener_unit;
    if (cfg->enable_pc_wiener) {
        memcpy(s->pc_wiener_mask, cfg->pc_wiener_none_mask, sizeof(s->pc_wiener_mask));
        s->tree.pc_wiener_none_mask = s->pc_wiener_mask;
    }
    { const uint16_t initial[6] = {19969, 0, 0, 1, 3, 4};
      memcpy(s->tree.pc_wiener_cdf, initial, sizeof(initial)); }
    s->tree.chroma_shift = cfg->chroma_format == 3u ? 0u : 1u;
    s->tree.q_ctx = svt_av2_coeff_q_context(cfg->base_q_idx);
    s->inter_frame = cfg->inter_frame;
    s->adaptive = cfg->entropy_adaptation;
    if ((s->adaptive || cfg->primary_models) && tile_models_seed(cfg, &s->committed_models) != 0) {
        svt_av2_tile_rate_free(s); return SVT_AV2_ENTROPY_ERROR_RANGE;
    }
    if (s->inter_frame) {
        const int st = tile_motion_init(&s->tree.motion, cfg);
        if (st != 0) { svt_av2_tile_rate_free(s); return st; }
    }
    *session = s;
    return 0;
}

void svt_av2_tile_rate_free(SvtAv2TileRateSession* session) {
    if (session == NULL) return;
    svt_av2_motion_syntax_free(&session->tree.motion);
    free(session->plan);
    free(session);
}

int svt_av2_tile_rate_end_models(const SvtAv2TileRateSession* s, SvtAv2EntropyModels* models) {
    if (s == NULL || models == NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
    if (!s->adaptive || s->cursor != s->sb_count) return SVT_AV2_ENTROPY_ERROR_RANGE;
    *models = s->committed_models;
    return 0;
}

int svt_av2_tile_rate_motion_predict(const SvtAv2TileRateSession* s,
    const SvtAv2BlockView* block, uint32_t rank, const SvtAv2MotionPrefix* prefix,
    int32_t* mv_x, int32_t* mv_y) {
    if (s == NULL || s->cursor >= s->sb_count || !s->inter_frame)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    return svt_av2_motion_predict(&s->tree.motion, block, rank, prefix, mv_x, mv_y);
}

int svt_av2_tile_rate_trial(SvtAv2TileRateSession* s, size_t sb_index,
    uint16_t partition, const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count, int commit,
    uint64_t* rate_q12) {
    if (s == NULL || blocks == NULL || rate_q12 == NULL)
        return SVT_AV2_ENTROPY_ERROR_NULL;
    *rate_q12 = 0u;
    if (sb_index != s->cursor || sb_index >= s->sb_count ||
        (commit != 0 && commit != 1) || block_count == 0u || block_count % 3u ||
        (s->inter_frame ? modes == NULL || mode_count != block_count / 3u :
                         modes != NULL || mode_count != 0u))
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    if (svt_av2_partition_plan_validate_sb(&s->tree.geometry, sb_index, partition) != 0)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    const uint16_t old_plan = s->plan[sb_index];
    s->plan[sb_index] = partition;
    SvtAv2BlockView sb = {0};
    sb.x = (uint32_t)(sb_index % s->tree.geometry.sb_cols) * 64u;
    sb.y = (uint32_t)(sb_index / s->tree.geometry.sb_cols) * 64u;
    sb.w = sb.h = 64u;
    const uint32_t x4 = sb.x / 4u, y4 = sb.y / 4u;
    uint32_t width4 = 0u, height4 = 0u;
    if (s->inter_frame) {
        width4 = s->tree.motion.cols - x4;
        height4 = s->tree.motion.rows - y4;
        if (width4 > 16u) width4 = 16u;
        if (height4 > 16u) height4 = 16u;
        for (uint32_t row = 0u; row < height4; ++row)
            memcpy(s->saved_motion + row * 16u,
                s->tree.motion.cells + (size_t)(y4 + row) * s->tree.motion.cols + x4,
                width4 * sizeof(*s->saved_motion));
    }
    TileTree trial = s->tree;
    SvtAv2EntropyEncoder enc;
    svt_av2_entropy_rate_init(&enc);
    if (s->adaptive) {
        /* Every candidate starts from the committed prefix. Failed and losing
         * trials cannot mutate it, including equal raw CDF boundaries. */
        s->scratch_models = s->committed_models;
        enc.models = &s->scratch_models;
    }
    trial.enc = &enc;
    trial.blocks = blocks; trial.count = block_count; trial.next = 0u;
    trial.modes = modes; trial.mode_count = mode_count;
    int st = tile_tree_write(&trial, &sb);
    if (st == 0 && trial.next != block_count) st = SVT_AV2_ENTROPY_ERROR_RANGE;
    if (st == 0) *rate_q12 = enc.rate_q12;
    if (st == 0 && commit) {
        if (s->adaptive) s->committed_models = s->scratch_models;
        s->tree = trial;
        s->tree.enc = NULL; s->tree.blocks = NULL; s->tree.modes = NULL;
        ++s->cursor;
    } else {
        s->plan[sb_index] = old_plan;
        for (uint32_t row = 0u; row < height4; ++row)
            memcpy(s->tree.motion.cells + (size_t)(y4 + row) * s->tree.motion.cols + x4,
                s->saved_motion + row * 16u, width4 * sizeof(*s->saved_motion));
    }
    return st;
}

/* Rate-only intra transaction. Partition symbols are consumed in the original
 * preorder; partition neighbours change only after a complete coding leaf. */
#define TILE_PREFIX_NODES 127u
#define TILE_PREFIX_SBS (64u * 34u)
typedef struct TilePrefixEvent {
    SvtAv2BlockView block;
    SvtAv2PartitionType part;
    int implied;
} TilePrefixEvent;

struct SvtAv2TileRatePrefix {
    TileTree tree, trial;
    SvtAv2EntropyEncoder enc;
    SvtAv2EntropyModels models, scratch_models;
    TilePrefixEvent events[TILE_PREFIX_NODES], pending[TILE_PREFIX_NODES];
    uint16_t plan[TILE_PREFIX_SBS];
    size_t count, pos, leaf, leaves;
    uint32_t adaptive, active;
};

static int tile_prefix_flatten(const SvtAv2FrameGeometry* geometry,
    const uint16_t* plan, const SvtAv2BlockView* b, TilePrefixEvent* events,
    size_t capacity, size_t* count, size_t* leaves) {
    if (b->x>=geometry->visible_width || b->y>=geometry->visible_height) return 0;
    if (*count>=capacity) return SVT_AV2_ENTROPY_ERROR_RANGE;
    TilePrefixEvent* event=events+*count;
    event->block=*b;
    if (svt_av2_partition_coded_type_plan_validated(geometry,b,plan,
        &event->part,&event->implied)!=0) return SVT_AV2_ENTROPY_ERROR_RANGE;
    ++*count;
    if (event->part==SVT_AV2_PARTITION_NONE) {++*leaves;return 0;}
    SvtAv2BlockView child=*b;
    if (event->part==SVT_AV2_PARTITION_HORZ) child.h/=2u;
    else if (event->part==SVT_AV2_PARTITION_VERT) child.w/=2u;
    else return SVT_AV2_ENTROPY_ERROR_RANGE;
    int st=tile_prefix_flatten(geometry,plan,&child,events,capacity,count,leaves);
    if (st!=0) return st;
    if (event->part==SVT_AV2_PARTITION_HORZ) child.y+=child.h;
    else child.x+=child.w;
    return tile_prefix_flatten(geometry,plan,&child,events,capacity,count,leaves);
}

static int tile_prefix_advance(TileTree* tree, const TilePrefixEvent* events,
    size_t count, size_t* pos) {
    while (*pos<count) {
        const TilePrefixEvent* event=events+(*pos)++;
        const int st=tile_partition_write(tree,&event->block,event->part,event->implied);
        if (st!=0) return st;
        if (event->part==SVT_AV2_PARTITION_NONE) break;
    }
    return 0;
}

/* Inter's decision-1 rate trie follows the original ordinal loop rather than
 * changing evaluation order. Four leaves imply fifteen nonterminal states
 * (including the root after its first leaf-partition event). */
#define TILE_INTER_RATE_EVENTS 7u
#define TILE_INTER_RATE_NODES 15u
#define TILE_INTER_RATE_SB_MI 16u
#define TILE_INTER_RATE_MI_COUNT (TILE_INTER_RATE_SB_MI * TILE_INTER_RATE_SB_MI)

typedef struct TileInterRateCtxPair {
    uint8_t above[TILE_INTER_RATE_SB_MI], left[TILE_INTER_RATE_SB_MI];
} TileInterRateCtxPair;

typedef struct TileInterRateNode {
    SvtAv2EntropyModels models;
    SvtAv2MotionCell motion[TILE_INTER_RATE_MI_COUNT];
    TileInterRateCtxPair part, y_mode, mrl, cfl, level[3], dc[3];
    uint64_t rate_q12, symbols, literal_bits;
    size_t pos;
} TileInterRateNode;

struct SvtAv2TileInterRateCache {
    SvtAv2TileRateSession* session;
    TileTree tree;
    SvtAv2EntropyModels models;
    TileInterRateNode nodes[TILE_INTER_RATE_NODES];
    SvtAv2MotionCell entry_motion[TILE_INTER_RATE_MI_COUNT];
    TilePrefixEvent events[TILE_INTER_RATE_EVENTS];
    size_t sb_index;
    uint32_t valid, x4, y4;
    uint64_t leaf_writes, event_writes;
};

static void tile_inter_rate_motion_copy(SvtAv2TileInterRateCache* c,
    SvtAv2MotionCell* patch, int restore) {
    const size_t stride = c->tree.motion.cols;
    SvtAv2MotionCell* const base = c->tree.motion.cells + (size_t)c->y4 * stride + c->x4;
    for (uint32_t row = 0u; row < TILE_INTER_RATE_SB_MI; ++row) {
        SvtAv2MotionCell* const cells = base + row * stride;
        SvtAv2MotionCell* const saved = patch + row * TILE_INTER_RATE_SB_MI;
        if (restore) memcpy(cells, saved, TILE_INTER_RATE_SB_MI * sizeof(*cells));
        else memcpy(saved, cells, TILE_INTER_RATE_SB_MI * sizeof(*cells));
    }
}

static void tile_inter_rate_copy_pair(TileInterRateCtxPair* pair,
    uint8_t* above, uint8_t* left, size_t n, int restore) {
    if (restore) {
        memcpy(above, pair->above, n);
        memcpy(left, pair->left, n);
    } else {
        memcpy(pair->above, above, n);
        memcpy(pair->left, left, n);
    }
}

/* Only these spans are written by the four leaves. Their external neighbours
 * stay committed in the working tree. UV is 8 MI per SB side in 4:2:0. */
static void tile_inter_rate_context_copy(SvtAv2TileInterRateCache* c,
    TileInterRateNode* node, int restore) {
    TileTree* const tree = &c->tree;
    const uint32_t x = c->x4, y = c->y4;
    const size_t n = TILE_INTER_RATE_SB_MI;
    tile_inter_rate_copy_pair(&node->part, tree->above_partition + x,
        tree->left_partition + y, n, restore);
    if (tree->m5_intra_modes)
        tile_inter_rate_copy_pair(&node->y_mode, tree->m5_above_mode + x,
            tree->m5_left_mode + y, n, restore);
    if (tree->m5_intra_modes >= 3u && tree->enable_mrls)
        tile_inter_rate_copy_pair(&node->mrl, tree->mrl_above + x,
            tree->mrl_left + y, n, restore);
    if (tree->enable_cfl_intra)
        tile_inter_rate_copy_pair(&node->cfl, tree->cfl_above + x,
            tree->cfl_left + y, n, restore);
    for (uint32_t p = 0u; p < 3u; ++p) {
        const uint32_t shift = p ? tree->chroma_shift : 0u;
        SvtAv2CoeffNeighbours* const nb = &tree->neighbours[p];
        tile_inter_rate_copy_pair(&node->level[p], nb->above_level + (x >> shift),
            nb->left_level + (y >> shift), n >> shift, restore);
        tile_inter_rate_copy_pair(&node->dc[p], nb->above_dc + (x >> shift),
            nb->left_dc + (y >> shift), n >> shift, restore);
    }
}

static int tile_inter_rate_advance(SvtAv2TileInterRateCache* c, size_t* pos) {
    const size_t before = *pos;
    const int st = tile_prefix_advance(&c->tree, c->events, TILE_INTER_RATE_EVENTS, pos);
    c->event_writes += *pos - before - (st != 0 ? 1u : 0u);
    return st;
}

static void tile_inter_rate_save(SvtAv2TileInterRateCache* c, TileInterRateNode* node,
    const SvtAv2EntropyEncoder* enc, size_t pos, int include_motion) {
    tile_inter_rate_context_copy(c, node, 0);
    if (include_motion) tile_inter_rate_motion_copy(c, node->motion, 0);
    if (enc->models) node->models = c->models;
    node->rate_q12 = enc->rate_q12;
    node->symbols = enc->rate_symbols;
    node->literal_bits = enc->rate_literal_bits;
    node->pos = pos;
}

int svt_av2_tile_inter_rate_cache_create(SvtAv2TileInterRateCache** cache) {
    if (cache == NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
    *cache = (SvtAv2TileInterRateCache*)calloc(1u, sizeof(**cache));
    return *cache != NULL ? 0 : SVT_AV2_ENTROPY_ERROR_NO_SPACE;
}

void svt_av2_tile_inter_rate_cache_free(SvtAv2TileInterRateCache* cache) { free(cache); }

void svt_av2_tile_inter_rate_cache_stats(const SvtAv2TileInterRateCache* c,
    uint64_t* leaf_writes, uint64_t* event_writes, size_t* scratch_bytes) {
    if (leaf_writes) *leaf_writes = c ? c->leaf_writes : 0u;
    if (event_writes) *event_writes = c ? c->event_writes : 0u;
    if (scratch_bytes) *scratch_bytes = c ? sizeof(*c) : 0u;
}

int svt_av2_tile_inter_rate_cache_begin(SvtAv2TileInterRateCache* c,
    SvtAv2TileRateSession* s, size_t sb_index) {
    if (c == NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
    c->session = NULL;
    c->valid = 0u;
    if (s == NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
    const SvtAv2FrameGeometry* const g = &s->tree.geometry;
    if (!s->inter_frame || s->tree.motion.cells == NULL || s->tree.ccso64 != NULL ||
        sb_index != s->cursor || sb_index >= s->sb_count || s->sb_count > TILE_PREFIX_SBS ||
        svt_av2_partition_plan_validate_sb(g, sb_index, 1u) != 0)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    SvtAv2BlockView sb = {0};
    sb.x = (uint32_t)(sb_index % g->sb_cols) * 64u;
    sb.y = (uint32_t)(sb_index / g->sb_cols) * 64u;
    sb.w = sb.h = 64u;
    const uint32_t x4 = sb.x / 4u, y4 = sb.y / 4u;
    if (s->tree.motion.cols < x4 + TILE_INTER_RATE_SB_MI ||
        s->tree.motion.rows < y4 + TILE_INTER_RATE_SB_MI)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    /* Flatten a private plan. This is not a partial write to the live plan. */
    uint16_t plan[TILE_PREFIX_SBS];
    memcpy(plan, s->plan, s->sb_count * sizeof(*plan));
    plan[sb_index] = 1u;
    size_t count = 0u, leaves = 0u, pos = 0u;
    int st = tile_prefix_flatten(g, plan, &sb, c->events, TILE_INTER_RATE_EVENTS, &count, &leaves);
    if (st != 0) return st;
    if (count != TILE_INTER_RATE_EVENTS || leaves != 4u) return SVT_AV2_ENTROPY_ERROR_RANGE;
    c->tree = s->tree;
    c->tree.trace = NULL;
    c->tree.blocks = NULL;
    c->tree.modes = NULL;
    c->tree.partition_decisions = NULL;
    c->tree.partition_decision_count = 0u;
    c->x4 = x4;
    c->y4 = y4;
    SvtAv2EntropyEncoder enc;
    svt_av2_entropy_rate_init(&enc);
    if (s->adaptive) {
        c->models = s->committed_models;
        enc.models = &c->models;
    }
    c->tree.enc = &enc;
    /* The only possible per-SB restoration symbol precedes the root event;
     * descendants in this tree are strictly less than 64x64. */
    st = tile_pc_wiener_write(&c->tree, &sb);
    if (st == 0) st = tile_inter_rate_advance(c, &pos);
    c->tree.enc = NULL;
    if (st != 0) return st;
    tile_inter_rate_save(c, &c->nodes[0], &enc, pos, 0);
    tile_inter_rate_motion_copy(c, c->entry_motion, 0);
    c->sb_index = sb_index;
    c->session = s;
    c->valid = 1u;
    return 0;
}

static uint32_t tile_inter_rate_node_index(uint32_t ordinal, uint32_t depth) {
    const uint32_t mask = (1u << depth) - 1u;
    return mask + (ordinal & mask);
}

int svt_av2_tile_inter_rate_cache_trial(SvtAv2TileInterRateCache* c, uint32_t ordinal,
    const SvtAv2TileCoeffBlock blocks[12], const SvtAv2TileBlockMode modes[4], uint64_t* rate_q12) {
    if (rate_q12 == NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
    *rate_q12 = 0u;
    if (c == NULL || blocks == NULL || modes == NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
    if (c->session == NULL || c->session->cursor != c->sb_index || ordinal >= 16u)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    uint32_t depth = 3u;
    while (!(c->valid & (1u << tile_inter_rate_node_index(ordinal, depth)))) --depth;
    TileInterRateNode* const node = &c->nodes[tile_inter_rate_node_index(ordinal, depth)];
    TileTree* const tree = &c->tree;
    SvtAv2EntropyEncoder enc;
    svt_av2_entropy_rate_init(&enc);
    if (c->session->adaptive) {
        c->models = node->models;
        enc.models = &c->models;
    }
    enc.rate_q12 = node->rate_q12;
    enc.rate_symbols = node->symbols;
    enc.rate_literal_bits = node->literal_bits;
    size_t pos = node->pos;
    tile_inter_rate_context_copy(c, node, 1);
    /* The entry grid is already live at depth 0. For other depths borrow its
     * current SB rectangle, not any neighbour outside it. Every path below
     * funnels through restoration, before the caller can query the live MI. */
    if (depth) tile_inter_rate_motion_copy(c, node->motion, 1);
    tree->enc = &enc;
    tree->blocks = blocks;
    tree->count = 12u;
    tree->next = 3u * depth;
    tree->modes = modes;
    tree->mode_count = 4u;
    uint32_t pending = 0u, leaf_writes = 0u;
    int st = 0;
    for (; depth < 4u && st == 0; ++depth) {
        st = tile_leaf_write(tree, &c->events[pos - 1u].block);
        if (st != 0) break;
        ++leaf_writes;
        st = tile_inter_rate_advance(c, &pos);
        if (st == 0 && depth < 3u) {
            const uint32_t idx = tile_inter_rate_node_index(ordinal, depth + 1u);
            tile_inter_rate_save(c, &c->nodes[idx], &enc, pos, 1);
            pending |= 1u << idx;
        }
    }
    c->leaf_writes += leaf_writes;
    if (st == 0 && (tree->next != 12u || pos != TILE_INTER_RATE_EVENTS))
        st = SVT_AV2_ENTROPY_ERROR_RANGE;
    tile_inter_rate_motion_copy(c, c->entry_motion, 1);
    tree->enc = NULL;
    tree->blocks = NULL;
    tree->modes = NULL;
    if (st == 0) {
        /* Failed candidates cannot establish an identity for any of their
         * newly evaluated prefixes. Already-valid parents are read-only. */
        c->valid |= pending;
        *rate_q12 = enc.rate_q12;
    }
    return st;
}

int svt_av2_tile_rate_prefix_create(SvtAv2TileRatePrefix** prefix) {
    if (prefix==NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
    *prefix=(SvtAv2TileRatePrefix*)calloc(1u,sizeof(**prefix));
    return *prefix!=NULL?0:SVT_AV2_ENTROPY_ERROR_NO_SPACE;
}

void svt_av2_tile_rate_prefix_free(SvtAv2TileRatePrefix* prefix) {free(prefix);}

int svt_av2_tile_rate_prefix_begin(SvtAv2TileRatePrefix* p,
    const SvtAv2TileRateSession* s, size_t sb_index, uint16_t partition,
    size_t* leaf_count) {
    if (p==NULL || s==NULL || leaf_count==NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
    *leaf_count=0u;
    if (s->inter_frame || sb_index!=s->cursor || sb_index>=s->sb_count ||
        s->sb_count>TILE_PREFIX_SBS ||
        svt_av2_partition_plan_validate_sb(&s->tree.geometry,sb_index,partition)!=0)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    memcpy(p->plan,s->plan,s->sb_count*sizeof(*p->plan));
    p->plan[sb_index]=partition;
    SvtAv2BlockView sb={0};
    sb.x=(uint32_t)(sb_index%s->tree.geometry.sb_cols)*64u;
    sb.y=(uint32_t)(sb_index/s->tree.geometry.sb_cols)*64u;sb.w=sb.h=64u;
    size_t count=0u,leaves=0u,pos=0u;
    int st=tile_prefix_flatten(&s->tree.geometry,p->plan,&sb,p->pending,TILE_PREFIX_NODES,&count,&leaves);
    if (st!=0 || leaves==0u) return SVT_AV2_ENTROPY_ERROR_RANGE;
    p->trial=s->tree;
    p->trial.partition_decisions=NULL;p->trial.partition_decision_count=0u;
    p->trial.blocks=NULL;p->trial.count=p->trial.next=0u;
    p->trial.modes=NULL;p->trial.mode_count=0u;
    SvtAv2EntropyEncoder enc;
    svt_av2_entropy_rate_init(&enc);
    if (s->adaptive) {p->scratch_models=s->committed_models;enc.models=&p->scratch_models;}
    p->trial.enc=&enc;
    st=tile_pc_wiener_write(&p->trial,&sb);
    if (st==0) st=tile_prefix_advance(&p->trial,p->pending,count,&pos);
    if (st!=0) return st;
    p->tree=p->trial;p->tree.enc=NULL;
    p->enc=enc;p->adaptive=s->adaptive;
    if (p->adaptive) {p->models=p->scratch_models;p->enc.models=&p->models;}
    memcpy(p->events,p->pending,count*sizeof(*p->events));
    p->count=count;p->pos=pos;p->leaf=0u;p->leaves=leaves;p->active=1u;
    *leaf_count=leaves;
    return 0;
}

int svt_av2_tile_rate_prefix_trial(SvtAv2TileRatePrefix* p, size_t leaf,
    const SvtAv2TileCoeffBlock blocks[3], int commit, uint64_t* rate_q12) {
    if (p==NULL || blocks==NULL || rate_q12==NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
    *rate_q12=0u;
    if (!p->active || leaf!=p->leaf || leaf>=p->leaves || p->pos==0u ||
        (commit!=0 && commit!=1)) return SVT_AV2_ENTROPY_ERROR_RANGE;
    p->trial=p->tree;
    SvtAv2EntropyEncoder enc=p->enc;
    if (p->adaptive) {p->scratch_models=p->models;enc.models=&p->scratch_models;}
    p->trial.enc=&enc;p->trial.blocks=blocks;p->trial.count=3u;p->trial.next=0u;
    int st=tile_leaf_write(&p->trial,&p->events[p->pos-1u].block);
    if (st==0 && p->trial.next!=3u) st=SVT_AV2_ENTROPY_ERROR_RANGE;
    const uint64_t candidate_rate=enc.rate_q12;
    size_t pos=p->pos;
    if (st==0 && commit) st=tile_prefix_advance(&p->trial,p->events,p->count,&pos);
    if (st!=0) return st;
    if (commit) {
        p->tree=p->trial;p->tree.enc=NULL;p->tree.blocks=NULL;
        p->tree.count=p->tree.next=0u;
        p->enc=enc;
        if (p->adaptive) {p->models=p->scratch_models;p->enc.models=&p->models;}
        p->pos=pos;++p->leaf;
    }
    *rate_q12=candidate_rate;
    return 0;
}

/* Rate-only luma subset. UV syntax uses disjoint model rows, so omitting
 * it cannot alter Y coefficient costs. Never commit or advance this subset. */
int svt_av2_tile_rate_prefix_trial_y(SvtAv2TileRatePrefix* p,size_t leaf,
    const SvtAv2TileCoeffBlock* y,uint64_t* rate_q12) {
    if (p==NULL || y==NULL || rate_q12==NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
    *rate_q12=0u;
    if (!p->active || leaf!=p->leaf || leaf>=p->leaves || p->pos==0u ||
        p->tree.m5_intra_modes<3u) return SVT_AV2_ENTROPY_ERROR_RANGE;
    p->trial=p->tree;
    SvtAv2EntropyEncoder enc=p->enc;
    if (p->adaptive) {p->scratch_models=p->models;enc.models=&p->scratch_models;}
    p->trial.enc=&enc;p->trial.blocks=y;p->trial.count=1u;p->trial.next=0u;
    const SvtAv2BlockView* b=&p->events[p->pos-1u].block;
    int st=tile_ccso64_write(&p->trial,b);
    if (st==0) st=m5_write_intra_y_mode(&p->trial,b,y->intra_mode,0u);
    uint8_t unused=0u;
    if (st==0) st=tile_coeff_plane_write(&p->trial,b,0u,0u,0u,y->intra_mode,&unused);
    if (st==0 && p->trial.next!=1u) st=SVT_AV2_ENTROPY_ERROR_RANGE;
    if (st==0) *rate_q12=enc.rate_q12;
    return st;
}

int svt_av2_tile_rate_prefix_finish(const SvtAv2TileRatePrefix* p, uint64_t* rate_q12) {
    if (p==NULL || rate_q12==NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
    *rate_q12=0u;
    if (!p->active || p->leaf!=p->leaves || p->pos!=p->count)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    *rate_q12=p->enc.rate_q12;
    return 0;
}

int svt_av2_tile_rate_frame(const SvtAv2FrameHeaderConfig* cfg,
    const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count, uint64_t* rate_q12) {
    if (cfg == NULL || blocks == NULL || rate_q12 == NULL)
        return SVT_AV2_ENTROPY_ERROR_NULL;
    if (cfg->entropy_adaptation > 1u) return SVT_AV2_ENTROPY_ERROR_RANGE;
    if (cfg->inter_frame ? modes == NULL || mode_count * 3u != block_count :
                           modes != NULL || mode_count != 0u)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    SvtAv2EntropyEncoder enc;
    svt_av2_entropy_rate_init(&enc);
    SvtAv2EntropyModels models;
    if (cfg->entropy_adaptation || cfg->primary_models) {
        const int status = tile_models_seed(cfg, &models);
        if (status != 0) return status;
        enc.models = &models;
    }
    const int st = tile_write_coeffs(&enc, cfg, blocks, block_count, modes, mode_count, NULL);
    *rate_q12 = st == 0 ? enc.rate_q12 : 0u;
    return st;
}

/* One payload byte per luma sample plus the header allowance: a safe bound for
 * this configuration. Returned to callers so they never size by hand. */
size_t svt_av2_tile_writer_max_payload_bytes(uint32_t width, uint32_t height) {
    size_t samples = (size_t)width * (size_t)height;
    return SVT_AV2_TILE_HEADER_MAX_BYTES + samples + 2u;
}

/* Shared implementation. `scratch` may be NULL, in which case the entropy
 * scratch is derived from `out_cap` (the caller's real payload capacity) rather
 * than from a fixed stack bound, so a frame is never capped by a hidden
 * constant. */
static int tile_writer_write(const SvtAv2FrameHeaderConfig* cfg, uint8_t* out,
                             size_t out_cap, size_t* out_size,
                             const SvtAv2TileWriterScratch* scratch,
                             const SvtAv2TileCoeffBlock* blocks, size_t block_count,
                             const SvtAv2TileBlockMode* modes, size_t mode_count,
                             SvtAv2EntropyModels* end_models) {
    if (cfg == NULL || out == NULL || out_size == NULL) {
        return SVT_AV2_BW_ERROR_NULL;
    }

    if (cfg->entropy_adaptation > 1u || (cfg->entropy_adaptation && blocks == NULL) ||
        (end_models != NULL && (!cfg->multi_frame || cfg->entropy_adaptation != 1u ||
         end_models == cfg->primary_models)) ||
        ((cfg->mv_half || cfg->logical_ref_count == 2u) && (blocks == NULL || modes == NULL)))
        return SVT_AV2_ENTROPY_ERROR_RANGE;

    /* --- header region: is_first_tile_group + frame_header + alignment --- */
    uint8_t         header[SVT_AV2_TILE_HEADER_MAX_BYTES];
    SvtAv2BitWriter bw;
    svt_av2_bit_writer_init(&bw, header, sizeof(header));

    int st = svt_av2_bit_writer_write_bit(&bw, 1u); /* is_first_tile_group */
    if (st != SVT_AV2_BW_OK) {
        return st;
    }
    st = svt_av2_frame_header_write(&bw, cfg);
    if (st != SVT_AV2_BW_OK) {
        return st;
    }
    st = svt_av2_bit_writer_byte_alignment(&bw);
    if (st != SVT_AV2_BW_OK) {
        return st;
    }
    size_t header_size = svt_av2_bit_writer_bytes_written(&bw);
    if (header_size > out_cap) {
        *out_size = header_size;
        return SVT_AV2_BW_ERROR_NO_SPACE;
    }

    /* --- entropy scratch -------------------------------------------------
     * The coded bytes are produced into `coded` and copied into `out` at the
     * end, because the final length is only known once the coder is done. The
     * coded region can therefore be at most `out_cap - header_size` bytes; the
     * precarry needs one uint16_t per coded byte. When the caller does not
     * supply scratch we allocate exactly that much, so the effective bound is
     * the caller's own capacity and never a fixed constant. */
    uint8_t*  coded_owned    = NULL;
    uint16_t* precarry_owned = NULL;
    uint8_t*  coded          = NULL;
    size_t    coded_cap      = 0u;
    uint16_t* precarry       = NULL;
    size_t    precarry_cap   = 0u;

    if (scratch != NULL) {
        coded        = scratch->coded;
        coded_cap    = scratch->coded_cap;
        precarry     = scratch->precarry;
        precarry_cap = scratch->precarry_cap;
        if (coded == NULL || precarry == NULL) {
            return SVT_AV2_ENTROPY_ERROR_NULL;
        }
    } else {
        coded_cap = out_cap - header_size;
        if (coded_cap == 0u) {
            *out_size = header_size + 1u;
            return SVT_AV2_BW_ERROR_NO_SPACE;
        }
        coded_owned = (uint8_t*)malloc(coded_cap);
        /* One uint16_t per coded byte plus the coder's carry slack (see
         * entropy_encoder.h). */
        precarry_cap = coded_cap + 2u;
        precarry_owned = (uint16_t*)malloc(precarry_cap * sizeof(uint16_t));
        if (coded_owned == NULL || precarry_owned == NULL) {
            free(coded_owned);
            free(precarry_owned);
            return SVT_AV2_BW_ERROR_NO_SPACE;
        }
        coded    = coded_owned;
        precarry = precarry_owned;
    }

    /* --- symbol region: the entropy-coded tile payload --- */
    SvtAv2EntropyEncoder enc;
    svt_av2_entropy_encoder_init(&enc, coded, coded_cap, precarry,
                                 precarry_cap);

    SvtAv2EntropyModels models;
    if (cfg->entropy_adaptation) {
        st = tile_models_seed(cfg, &models);
        if (st != 0) { free(coded_owned); free(precarry_owned); return st; }
        enc.models = &models;
    }
    TileInterTrace stats = {0};
    TileInterTrace* trace = NULL;
    if (cfg->inter_frame && modes != NULL) {
        const char* value = getenv("SVT_AV2_INTER_TRACE");
        if (value && strcmp(value, "1") == 0) trace = &stats;
    }
    if (blocks == NULL) {
        st = svt_av2_block_write_minimal_intra(&enc, cfg->base_q_idx);
    } else {
        st = tile_write_coeffs(&enc, cfg, blocks, block_count, modes, mode_count, trace);
    }
    if (st != SVT_AV2_ENTROPY_OK) {
        if (st == SVT_AV2_ENTROPY_ERROR_NO_SPACE) {
            *out_size = svt_av2_tile_writer_max_payload_bytes(cfg->frame_width, cfg->frame_height);
            st = SVT_AV2_BW_ERROR_NO_SPACE;
        }
        free(coded_owned);
        free(precarry_owned);
        return st;
    }
    const uint64_t range_q12 = trace ? tile_trace_tell_q12(&enc) : 0u;
    uint8_t* coded_bytes = NULL;
    size_t   coded_size  = 0u;
    st = svt_av2_entropy_encoder_done(&enc, &coded_bytes, &coded_size);
    if (st != SVT_AV2_ENTROPY_OK) {
        /* The entropy scratch was too small for the payload this block
         * produced. The exact byte count is not available from the coder, so
         * report the bound for the geometry the caller must retry with. */
        *out_size = svt_av2_tile_writer_max_payload_bytes(cfg->frame_width,
                                                          cfg->frame_height);
        free(coded_owned);
        free(precarry_owned);
        return (st == SVT_AV2_ENTROPY_ERROR_NO_SPACE)
                   ? SVT_AV2_BW_ERROR_NO_SPACE
                   : st;
    }
    if (header_size + coded_size > out_cap) {
        /* Exact requirement; nothing has been written to out yet. */
        *out_size = header_size + coded_size;
        free(coded_owned);
        free(precarry_owned);
        return SVT_AV2_BW_ERROR_NO_SPACE;
    }

    /* --- concatenate --- */
    for (size_t i = 0; i < header_size; ++i) {
        out[i] = header[i];
    }
    for (size_t i = 0; i < coded_size; ++i) {
        out[header_size + i] = coded_bytes[i];
    }
    *out_size = header_size + coded_size;
    if (end_models != NULL) *end_models = models;
    if (trace) tile_trace_print(trace, cfg, range_q12, header_size, coded_size);
    free(coded_owned);
    free(precarry_owned);
    return 0;
}

int svt_av2_tile_writer_write_closed_loop_key(const SvtAv2FrameHeaderConfig* cfg,
                                              uint8_t* out,
                                              size_t   out_cap,
                                              size_t*  out_size) {
    return tile_writer_write(cfg, out, out_cap, out_size, NULL, NULL, 0u, NULL, 0u, NULL);
}

int svt_av2_tile_writer_write_closed_loop_key_scratch(
    const SvtAv2FrameHeaderConfig* cfg, uint8_t* out, size_t out_cap,
    size_t* out_size, const SvtAv2TileWriterScratch* scratch) {
    return tile_writer_write(cfg, out, out_cap, out_size, scratch, NULL, 0u, NULL, 0u, NULL);
}

int svt_av2_tile_writer_write_closed_loop_key_coeffs(
    const SvtAv2FrameHeaderConfig* cfg, const SvtAv2TileCoeffBlock* blocks,
    size_t block_count, uint8_t* out, size_t out_cap, size_t* out_size,
    const SvtAv2TileWriterScratch* scratch) {
    if (blocks == NULL) return SVT_AV2_BW_ERROR_NULL;
    return tile_writer_write(cfg, out, out_cap, out_size, scratch, blocks, block_count, NULL, 0u, NULL);
}

int svt_av2_tile_writer_write_inter_coeffs(
    const SvtAv2FrameHeaderConfig* cfg, const SvtAv2TileCoeffBlock* blocks,
    size_t block_count, const SvtAv2TileBlockMode* modes, size_t mode_count,
    uint8_t* out, size_t out_cap, size_t* out_size) {
    if (blocks == NULL || modes == NULL) return SVT_AV2_BW_ERROR_NULL;
    if (!cfg || !cfg->inter_frame || mode_count * 3u != block_count) return SVT_AV2_BW_ERROR_RANGE;
    return tile_writer_write(cfg, out, out_cap, out_size, NULL, blocks, block_count, modes, mode_count, NULL);
}

int svt_av2_tile_writer_write_video_models(const SvtAv2FrameHeaderConfig* cfg,
    const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count,
    uint8_t* out, size_t out_cap, size_t* out_size, SvtAv2EntropyModels* end_models) {
    if (cfg == NULL || blocks == NULL || end_models == NULL) return SVT_AV2_ENTROPY_ERROR_NULL;
    if (cfg->inter_frame ? modes == NULL || block_count % 3u || mode_count != block_count / 3u :
                          modes != NULL || mode_count != 0u) return SVT_AV2_ENTROPY_ERROR_RANGE;
    return tile_writer_write(cfg, out, out_cap, out_size, NULL, blocks, block_count, modes, mode_count, end_models);
}


uint64_t svt_av2_tile_rate_prefix_current_rate(const SvtAv2TileRatePrefix* prefix) {
    return prefix->enc.rate_q12;
}
