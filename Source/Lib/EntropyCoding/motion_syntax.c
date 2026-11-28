#include "motion_syntax.h"
#include "entropy_models.h"
#include "motion_two_ref_mvp.h"
#include <stdlib.h>
#include <string.h>

/* Fixed AVM entropy_inits_modes.h and entropy_inits_mv.h forward boundaries.
 * These are the static CDFs; the optional PRIMARY_REF_NONE adaptive bank uses
 * the same defaults and is updated independently for each frame. */
static const uint16_t mode_cdf[5][2] = {
    {10043,11100},{21561,21758},{25411,25714},{14117,14341},{18288,18577}};
static const uint16_t inter_cdf[4] = {1522,14381,10455,27796};
static const uint16_t drl_cdf[5] = {15721,21115,19567,17602,13319};
static int symbol(SvtAv2EntropyEncoder* e, uint16_t cdf, uint32_t model, uint32_t v) {
    return svt_av2_entropy_encode_model(e, model, &cdf, 2u, v);
}
static int literal(SvtAv2EntropyEncoder* e, uint32_t value, uint32_t bits) {
    return svt_av2_entropy_encode_literal(e, value, (int)bits);
}
static int quniform(SvtAv2EntropyEncoder* e, uint32_t n, uint32_t value) {
    uint32_t l = 0;
    while ((1u << l) < n) ++l;
    if (l == 0) return 0;
    const uint32_t m = (1u << l) - n;
    if (value < m) return literal(e, value, l - 1u);
    int st = literal(e, m + ((value - m) >> 1), l - 1u);
    return st != 0 ? st : literal(e, (value - m) & 1u, 1u);
}
static int motion_write_difference(SvtAv2EntropyEncoder* e, int32_t dx, int32_t dy,
    uint32_t mv_half, int wide) {
    const int32_t limit = mv_half == 2u ? svt_av2_motion_q3_limit(wide) : mv_half ? 64 : 32;
    if (dx < -limit || dx > limit || dy < -limit || dy > limit)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    const uint32_t x = (uint32_t)abs(dx), y = (uint32_t)abs(dy), shell = x + y;
    uint32_t cls = 0;
    while ((2u << cls) <= shell) ++cls;
    const uint32_t offset = shell - (cls == 0u ? 0u : (1u << cls));
    static const uint16_t class_cdf[6] = {8680,13723,18208,22686,26722,30020};
    /* AVM v1.0.0 default_nmv_context, precision HALF_PEL = 4. */
    static const uint16_t half_class_0[6] = {4324,15300,23690,28697,31282,32359};
    static const uint16_t half_class_1[7] = {19707,28414,31240,31648,32692,32717,32721};
    /* QTR_PEL = 5: 16 classes split 8+8, rather than HALF_PEL's 7+8. */
    static const uint16_t quarter_class_0[7] = {7497,17301,23848,27438,29395,30879,32003};
    static const uint16_t quarter_class_1[7] = {18469,27427,31562,32652,32724,32728,32732};
    static const uint16_t low[2] = {14587,20966};
    static const uint16_t bits[9] = {17943,18934,18928,18696,19044,20362,20426,22563,22190};
    const uint32_t first_class_in_set1 = mv_half == 2u ? 8u : 7u;
    const uint32_t set = cls >= first_class_in_set1;
    int st = symbol(e, 31579, SVT_AV2_MODEL_MV_SET, set);
    if (st == 0) {
        if (mv_half == 2u) st = set ? svt_av2_entropy_encode_model(e, SVT_AV2_MODEL_MV_QUARTER_CLASS1,
                                      quarter_class_1, 8u, cls - first_class_in_set1) :
                                      svt_av2_entropy_encode_model(e, SVT_AV2_MODEL_MV_QUARTER_CLASS0, quarter_class_0, 8u, cls);
        else if (mv_half) st = set ? svt_av2_entropy_encode_model(e, SVT_AV2_MODEL_MV_HALF_CLASS1, half_class_1, 8u, cls - 7u) :
                                     svt_av2_entropy_encode_model(e, SVT_AV2_MODEL_MV_HALF_CLASS0, half_class_0, 7u, cls);
        else st = svt_av2_entropy_encode_model(e, SVT_AV2_MODEL_MV_CLASS, class_cdf, 7u, cls);
    }
    if (cls < 2u) {
        if (st == 0) st = symbol(e, low[cls], SVT_AV2_MODEL_MV_LOW + cls, offset);
    } else if (cls == 2u) {
        for (uint32_t i = 0; i < 3u && st == 0; ++i) {
            st = i == 0u ? symbol(e, 13189, SVT_AV2_MODEL_MV_CLASS2, offset != i) : literal(e, offset != i, 1u);
            if (offset == i) break;
        }
    } else {
        for (uint32_t i = 0; i < cls && st == 0; ++i) {
            /* [0][i] is shared across precisions, including the legacy
             * HALF_BIT6 ID. Class 8 consumes bit 7 even at shell 256. */
            const uint32_t model = i < 6u ? SVT_AV2_MODEL_MV_BITS + i :
                i == 6u ? SVT_AV2_MODEL_MV_HALF_BIT6 : i == 7u ? SVT_AV2_MODEL_MV_QUARTER_BIT7 : SVT_AV2_MODEL_MV_QUARTER_BIT8;
            st = symbol(e, bits[i], model, (offset >> i) & 1u);
        }
    }
    const uint32_t maximum = shell >> 1, pair = x < shell - x ? x : shell - x;
    static const uint16_t greater[2] = {5663,4856};
    for (uint32_t i = 0; i < maximum && i < 2u && st == 0; ++i) {
        st = symbol(e, greater[i], SVT_AV2_MODEL_MV_GREATER + i, pair != i);
        if (pair == i) break;
    }
    if (st == 0 && maximum > 2u && pair >= 2u) st = quniform(e, maximum - 1u, pair - 2u);
    if (st == 0 && shell > 0u && !(pair == maximum && (shell & 1u) == 0u)) {
        static const uint16_t col[4] = {13445,13541,14045,12888};
        st = symbol(e, col[cls < 3u ? cls : 3u], SVT_AV2_MODEL_MV_COL + (cls < 3u ? cls : 3u), x > maximum);
    }
    if (st == 0 && y != 0u) st = literal(e, dy < 0, 1u);
    if (st == 0 && x != 0u) st = literal(e, dx < 0, 1u);
    return st;
}
int svt_av2_motion_write_difference(SvtAv2EntropyEncoder* e, int32_t dx, int32_t dy) {
    return motion_write_difference(e, dx, dy, 0u, 0);
}
int svt_av2_motion_syntax_init(SvtAv2MotionSyntax* s, uint32_t w, uint32_t h) {
    return svt_av2_motion_syntax_init_precision(s, w, h, 0u);
}
int svt_av2_motion_syntax_init_precision(SvtAv2MotionSyntax* s, uint32_t w, uint32_t h,
    uint32_t mv_half) {
    memset(s, 0, sizeof(*s));
    if (mv_half > 2u) return SVT_AV2_ENTROPY_ERROR_RANGE;
    s->cols = w / 4u; s->rows = h / 4u;
    s->mv_half = mv_half;
    s->cells = (SvtAv2MotionCell*)calloc((size_t)s->cols * s->rows, sizeof(*s->cells));
    return s->cells == NULL ? SVT_AV2_ENTROPY_ERROR_NO_SPACE : 0;
}
void svt_av2_motion_syntax_free(SvtAv2MotionSyntax* s) {
    free(s->cells); memset(s, 0, sizeof(*s));
}
static const SvtAv2MotionCell* cell(const SvtAv2MotionSyntax* s, int32_t x, int32_t y) {
    if (x < 0 || y < 0 || (uint32_t)x >= s->cols || (uint32_t)y >= s->rows) return NULL;
    const SvtAv2MotionCell* c = s->cells + (size_t)y * s->cols + (uint32_t)x;
    return c->valid ? c : NULL;
}
/* Speculative leaves belong to the current SB and are visible only to this
 * query. An intra match hides any backing cell and cannot supply a vector. */
static const SvtAv2MotionCell* mvp_cell(const SvtAv2MotionSyntax* s,
    const SvtAv2MotionPrefix* prefix, int32_t x, int32_t y) {
    if (prefix && x >= 0 && y >= 0) {
        const uint32_t px = (uint32_t)x * 4u, py = (uint32_t)y * 4u;
        for (size_t i = 0; i < prefix->count; ++i) {
            const SvtAv2BlockView* b = &prefix->blocks[i];
            if (px >= b->x && px < b->x + b->w && py >= b->y && py < b->y + b->h)
                return &prefix->cells[i];
        }
    }
    return cell(s,x,y);
}

int svt_av2_motion_predict(const SvtAv2MotionSyntax* s, const SvtAv2BlockView* b,
    uint32_t rank, const SvtAv2MotionPrefix* prefix, int32_t* out_x, int32_t* out_y) {
    if (!s || !b || !out_x || !out_y || rank > s->two_refs)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    const int projected = s->two_refs && s->ref_distance[0] != 0u;
    const uint8_t reference_plus_one = (uint8_t)(1u + rank);
    const int32_t x = (int32_t)(b->x / 4u), y = (int32_t)(b->y / 4u);
    const int32_t w = (int32_t)(b->w / 4u), h = (int32_t)(b->h / 4u);
    const SvtAv2MotionCell* spatial[4] = {mvp_cell(s,prefix,x-1,y+h-1),mvp_cell(s,prefix,x+w-1,y-1),
        mvp_cell(s,prefix,x-1,y),mvp_cell(s,prefix,x,y-1)};
    /* DRL reordering and temporal/ref banks are disabled. Retain the legacy
     * integer scan; HALF and QUARTER follow AVM's scan at the SB row boundary,
     * where above MVP samples (unlike mode-context samples) use an 8x8 grid. */
    const SvtAv2MotionCell* candidates[9] = {spatial[0],spatial[1],spatial[2],spatial[3],
        mvp_cell(s, prefix,x-1,y+h),mvp_cell(s, prefix,x+w,y-1),mvp_cell(s, prefix,x-1,y-1),mvp_cell(s, prefix,x-3,y+h-1),mvp_cell(s, prefix,x-3,y)};
    if (s->mv_half) {
        const int32_t sb_mi = 16; /* 64x64 SB, 4x4 MI */
        const int sb_row_boundary = y % sb_mi == 0;
        const int32_t aligned_x = x & ~1;
        const int32_t above_end = sb_row_boundary ? aligned_x + (w > 2 ? w - 2 : 0) : x + w - 1;
        const int32_t above_right = sb_row_boundary ? aligned_x + (w > 2 ? w : 2) : x + w;
        const int32_t above_left = sb_row_boundary ? aligned_x - 2 : x - 1;
        const int32_t outer_left = x - 3 + (w == 1 ? x & 1 : 0);
        candidates[1] = mvp_cell(s, prefix, above_end, y - 1);
        candidates[2] = h > 1 ? spatial[2] : NULL;
        candidates[3] = w > (sb_row_boundary ? 2 : 1) ? mvp_cell(s, prefix, sb_row_boundary ? aligned_x : x, y - 1) : NULL;
        candidates[4] = y % sb_mi + h < sb_mi ? candidates[4] : NULL;
        candidates[5] = sb_row_boundary || x % sb_mi + w < sb_mi ? mvp_cell(s, prefix, above_right, y - 1) : NULL;
        candidates[6] = mvp_cell(s, prefix, above_left, y - 1);
        candidates[7] = mvp_cell(s, prefix, outer_left, y + h - 1);
        candidates[8] = h > 1 ? mvp_cell(s, prefix, outer_left, y) : NULL;
    }
    int32_t px = 0, py = 0;
    const SvtAv2MotionCell* first = NULL;
    /* Direct entries win over derived ones even when found later in the
     * spatial scan. DRL0, no reorder, no temporal/reference MV banks. */
    for (uint32_t i = 0u; i < 9u; ++i) {
        const SvtAv2MotionCell* candidate = candidates[i];
        if (candidate == NULL || !candidate->is_inter) continue;
        if (!projected || candidate->is_inter == 3u || candidate->is_inter == reference_plus_one) { first = candidate; break; }
        /* AVM derives spatial predictors only from references on the same temporal side. */
        if (!s->opposite_directions && first == NULL) first = candidate;
    }
    if (first != NULL) {
        const int second = first->is_inter == 3u && rank == 1u;
        px = second ? first->second_x : first->mv_x;
        py = second ? first->second_y : first->mv_y;
        if (projected && first->is_inter != 3u && first->is_inter != reference_plus_one) {
            const uint32_t from = s->ref_distance[first->is_inter - 1u], to = s->ref_distance[rank];
            if (!svt_av2_mvp_project_q3_range(s->wide_motion, px, to, from, &px) || !svt_av2_mvp_project_q3_range(s->wide_motion, py, to, from, &py))
                return SVT_AV2_ENTROPY_ERROR_RANGE;
        }
    }
    /* AVM clamps the spatial predictor before coding the difference, even
     * for one reference. The MI grid is four luma pixels per unit. */
    const int32_t unit = projected ? 32 : s->mv_half == 2u ? 16 : s->mv_half ? 8 : 4;
    const int32_t left = -(x+w+4)*unit, top = -(y+h+4)*unit;
    const int32_t right = ((int32_t)s->cols-x+4)*unit, bottom = ((int32_t)s->rows-y+4)*unit;
    px = px<left ? left : px>right ? right : px;
    py = py<top ? top : py>bottom ? bottom : py;
    *out_x = px;
    *out_y = py;
    return 0;
}

int svt_av2_motion_write_block_delta(SvtAv2MotionSyntax* s, SvtAv2EntropyEncoder* e,
    const SvtAv2BlockView* b, const SvtAv2TileBlockMode* m, uint32_t skip_txfm, const int* delta) {
    const int projected = s->two_refs && s->ref_distance[0] != 0u;
    const int32_t q3_limit = svt_av2_motion_q3_limit(s->wide_motion);
    const int32_t limit = projected ? q3_limit : s->mv_half == 2u ? q3_limit / 2 : s->mv_half ? 32 : 16;
    if (m->is_inter > 1u || m->mv_x < -limit || m->mv_x > limit || m->mv_y < -limit || m->mv_y > limit)
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    const uint32_t rank = s->two_refs ? m->reference_rank : 0u;
    const int compound = m->is_inter && s->compound_average && rank == 2u;
    if (s->two_refs && ((!compound && rank > m->is_inter) || ((!projected || !m->is_inter) && (m->mv_x || m->mv_y))))
        return SVT_AV2_ENTROPY_ERROR_RANGE;
    if (projected && m->is_inter) {
        const int square = b->w == b->h &&
            (b->w == 8u || b->w == 16u || b->w == 32u || b->w == 64u);
        const int rect = (b->w == 32u && b->h == 16u) || (b->w == 16u && b->h == 32u) ||
            (b->w == 64u && b->h == 32u) || (b->w == 32u && b->h == 64u);
        if (s->mv_half != 2u || !(square || rect) || b->x % b->w || b->y % b->h)
            return SVT_AV2_ENTROPY_ERROR_RANGE;
    }
    const uint8_t reference_plus_one = m->is_inter ? (uint8_t)(1u + rank) : 0u;
    const int32_t x = (int32_t)(b->x / 4u), y = (int32_t)(b->y / 4u);
    const int32_t w = (int32_t)(b->w / 4u), h = (int32_t)(b->h / 4u);
    /* av2_get_intra_inter_context uses the first two available spatial entries. */
    const SvtAv2MotionCell* neigh[4] = {cell(s,x-1,y+h-1),cell(s,x+w-1,y-1),cell(s,x-1,y),cell(s,x,y-1)};
    uint32_t count = 0, intra = 0;
    for (uint32_t i = 0; i < 4u && count < 2u; ++i)
        if (neigh[i]) { ++count; intra += !neigh[i]->is_inter; }
    const uint32_t ictx = count == 2u ? (intra == 2u ? 3u : intra) : (intra ? 2u : 0u);
    const SvtAv2MotionCell* spatial[4] = {cell(s,x-1,y+h-1),cell(s,x+w-1,y-1),cell(s,x-1,y),cell(s,x,y-1)};
    uint32_t left = 0, above = 0, any_new = 0;
    for (uint32_t i = 0; i < 4u; ++i) if (spatial[i] && spatial[i]->is_inter &&
        (!s->two_refs || spatial[i]->is_inter == reference_plus_one || (!compound && spatial[i]->is_inter == 3u))) {
        if (i == 0u || i == 2u) left = 1u; else above = 1u;
        /* NEAR_NEWMV contributes NEW only to its second single-reference stack. */
        if (compound || spatial[i]->is_inter != 3u || rank == 1u)
            any_new |= spatial[i]->is_new;
    }
    const uint32_t ctx = left + above + 2u * any_new;
    int32_t dx = 0, dy = 0;
    if (m->is_inter && !compound) {
        int32_t px, py;
        const int pred = svt_av2_motion_predict(s, b, rank, NULL, &px, &py);
        if (pred != 0) return pred;
        if (projected) {
            if (!svt_av2_mvp_quarter_delta_range(s->wide_motion, m->mv_x, px, &dx) || !svt_av2_mvp_quarter_delta_range(s->wide_motion, m->mv_y, py, &dy))
                return SVT_AV2_ENTROPY_ERROR_RANGE;
        } else {
            dx = m->mv_x - px; dy = m->mv_y - py;
        }
    }
    uint32_t is_new = m->is_inter && (dx != 0 || dy != 0);
    int st = symbol(e, inter_cdf[ictx], SVT_AV2_MODEL_INTRA_INTER + ictx, m->is_inter);
    uint32_t skip_ctx = 0;
    count = 0;
    for (uint32_t i = 0; i < 4u && count < 2u; ++i)
        if (neigh[i]) { ++count; skip_ctx += neigh[i]->skip_txfm; }
    static const uint16_t skip_cdf[3] = {25865,14316,4598};
    if (st == 0 && m->is_inter) st = symbol(e, skip_cdf[skip_ctx], SVT_AV2_MODEL_SKIP + skip_ctx, skip_txfm);
    if (!st && delta) {
        static const uint16_t delta_cdf[7] = {16594,23325,26424,28225,29358,30099,30613};
        const uint32_t dq = (uint32_t)abs(*delta)/4u;
        if (*delta%4 || dq>=7u) return SVT_AV2_ENTROPY_ERROR_RANGE;
        st = svt_av2_entropy_encode_model(e, SVT_AV2_MODEL_VIDEO_DELTA_Q, delta_cdf, 8u, dq);
        if (!st && dq) st=svt_av2_entropy_encode_bit(e, *delta<0);
    }
    if (st == 0 && m->is_inter && s->compound_average) {
        unsigned type[2] = {0,0}, n = 0;
        for (unsigned i=0; i<4 && n<2; ++i) if (neigh[i]) type[n++]=neigh[i]->is_inter;
        /* Rank 0 points to the future. A compound neighbour has both ranks. */
        unsigned cc;
        if (!n) cc=1u;
        else if (n==1u) cc=type[0]==3u ? 3u : type[0]==1u;
        else if (type[0]!=3u && type[1]!=3u) cc=(type[0]==1u) ^ (type[1]==1u);
        else if (type[0]==3u && type[1]==3u) cc=4u;
        else cc=2u+(type[type[0]==3u] != 2u);
        static const uint16_t comp[5] = {26924,25000,17949,13581,7034};
        st = symbol(e, comp[cc], SVT_AV2_MODEL_COMP_INTER + cc, (unsigned)compound);
    }
    if (!st && compound) {
        static const uint16_t types[5][4] = {
            {12177,20001,23193,24448},{21192,25117,27806,27948},{26779,28724,30192,30249},
            {12506,17871,21295,21389},{16948,20335,22582,22617}};
        const int near = m->mv_x || m->mv_y || m->second_x || m->second_y;
        if (ctx >= 5u) return SVT_AV2_ENTROPY_ERROR_RANGE;
        if (near) {
            int32_t x0, y0, x1, y1;
            if (svt_av2_motion_predict(s,b,0,NULL,&x0,&y0) || svt_av2_motion_predict(s,b,1,NULL,&x1,&y1) ||
                x0 != m->mv_x || y0 != m->mv_y ||
                !svt_av2_mvp_quarter_delta_range(s->wide_motion,m->second_x,x1,&dx) ||
                !svt_av2_mvp_quarter_delta_range(s->wide_motion,m->second_y,y1,&dy))
                return SVT_AV2_ENTROPY_ERROR_RANGE;
            is_new = dx != 0 || dy != 0;
        }
        if (!st) st = symbol(e, 24720, SVT_AV2_MODEL_COMP_JOINT, 0u);
        if (!st) st = svt_av2_entropy_encode_model(e, SVT_AV2_MODEL_COMP_MODE+ctx, types[ctx], 5u, near ? is_new : 3u);
        /* Distinct references in NEAR_NEARMV and NEAR_NEWMV each select spatial DRL index 0. */
        for (unsigned i = 0; near && i < 2u && !st; ++i)
            st = symbol(e, drl_cdf[ctx], SVT_AV2_MODEL_DRL + ctx, 0u);
    } else if (st == 0 && m->is_inter && s->two_refs) {
        /* AV2 single-reference rank 0: count the first two line neighbours,
         * counting intra as available but not as a reference. */
        uint32_t rank0 = 0, rank1 = 0;
        count = 0;
        for (uint32_t i = 0; i < 4u && count < 2u; ++i) if (neigh[i]) {
            ++count;
            rank0 += neigh[i]->is_inter == 1u || neigh[i]->is_inter == 3u;
            rank1 += neigh[i]->is_inter >= 2u;
        }
        const uint32_t ref_ctx = rank0 > rank1 ? 2u : rank0 == rank1 ? 1u : 0u;
        static const uint16_t ref_cdf[3] = {26469,13631,2599};
        /* The three contexts do not share training counters. Each model is
         * part of the ordinary frame-local bank and so follows trial rollback. */
        st = symbol(e, ref_cdf[ref_ctx], SVT_AV2_MODEL_SINGLE_REF + ref_ctx, rank == 0u);
    }
    if (st == 0 && m->is_inter && !compound) st = svt_av2_entropy_encode_model(e, SVT_AV2_MODEL_INTER_MODE + ctx, mode_cdf[ctx], 3u, is_new ? 2u : 0u);
    if (st == 0 && m->is_inter && !compound) st = symbol(e, drl_cdf[ctx], SVT_AV2_MODEL_DRL + ctx, 0u);
    if (st == 0 && is_new)
        st = motion_write_difference(e, dx, dy, s->mv_half, s->wide_motion);
    if (st == 0) {
        SvtAv2MotionCell value = {m->mv_x,m->mv_y,1u,reference_plus_one,(uint8_t)is_new,(uint8_t)skip_txfm,
            compound ? m->second_x : 0,compound ? m->second_y : 0};
        for (int32_t yy = y; yy < y+h; ++yy)
            for (int32_t xx = x; xx < x+w; ++xx)
                s->cells[(size_t)yy * s->cols + (uint32_t)xx] = value;
    }
    return st;
}

int svt_av2_motion_write_block(SvtAv2MotionSyntax* s, SvtAv2EntropyEncoder* e,
    const SvtAv2BlockView* b, const SvtAv2TileBlockMode* m, uint32_t skip_txfm) {
    return svt_av2_motion_write_block_delta(s,e,b,m,skip_txfm,NULL);
}
