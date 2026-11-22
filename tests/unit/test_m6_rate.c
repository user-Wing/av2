#include "tile_writer.h"
#include "entropy_encoder.h"
#include "coeff_syntax.h"
#include "inter_tail.h"
#include "test_util.h"
#include <math.h>
#include <string.h>

/* Preserve every Q12 value of the original integer logarithm. */
static uint32_t reference_probability_cost(uint32_t probability) {
    uint32_t exponent = 0u, fraction = 0u;
    for (uint32_t p = probability; p > 1u; p >>= 1) ++exponent;
    uint64_t normalized = (uint64_t)probability << (31u - exponent);
    for (uint32_t bit = 0u; bit < 12u; ++bit) {
        normalized = (normalized * normalized) >> 31;
        fraction <<= 1;
        if (normalized >= ((uint64_t)1u << 32)) {
            normalized >>= 1;
            fraction |= 1u;
        }
    }
    return (15u - exponent) * 4096u - fraction;
}

static void probability_cost(void) {
    uint32_t cost = 0u, previous = UINT32_MAX;
    for (uint32_t p = 1u; p <= 32768u; ++p) {
        SVT_TEST_CHECK(svt_av2_entropy_probability_cost_q12(p, &cost) == 0);
        SVT_TEST_CHECK_EQ_INT(cost, reference_probability_cost(p));
        const double exact = -log((double)p / 32768.0) / log(2.0) * 4096.0;
        SVT_TEST_CHECK((double)cost >= exact - 0.00001 && (double)cost - exact < 1.001);
        SVT_TEST_CHECK(cost <= previous);
        previous = cost;
    }
    SVT_TEST_CHECK(cost == 0u);
    SVT_TEST_CHECK(svt_av2_entropy_probability_cost_q12(0u, &cost) != 0);
    SVT_TEST_CHECK(svt_av2_entropy_probability_cost_q12(32769u, &cost) != 0);
    SvtAv2EntropyEncoder enc;
    svt_av2_entropy_rate_init(&enc);
    uint16_t cdf = 16384u;
    SVT_TEST_CHECK(svt_av2_entropy_encode_symbol(&enc, &cdf, 2u, 0u) == 0);
    SVT_TEST_CHECK(svt_av2_entropy_encode_symbol(&enc, &cdf, 2u, 1u) == 0);
    SVT_TEST_CHECK(svt_av2_entropy_encode_literal(&enc, 65535u, 16) == 0);
    SVT_TEST_CHECK(enc.rate_q12 == 18u * 4096u);
    SVT_TEST_CHECK(enc.rate_symbols == 2u && enc.rate_literal_bits == 16u);
    cdf = 0u;
    SVT_TEST_CHECK(svt_av2_entropy_encode_symbol(&enc, &cdf, 2u, 0u) != 0);
    SVT_TEST_CHECK(enc.rate_q12 == 18u * 4096u);
}

/* Two neighbouring SBs, four distinct coeff/mode paths and an error after
 * motion writing. Compare SB trial/commit with independently replayed frame. */
static void transactions(uint32_t bit_depth, uint32_t format, uint32_t inter,
                         uint32_t adaptive, uint32_t base_q_idx) {
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, 128u, 64u);
    cfg.bit_depth = bit_depth; cfg.chroma_format = format;
    cfg.base_q_idx = base_q_idx;
    cfg.m5_intra_modes = 1u; cfg.inter_frame = inter; cfg.multi_frame = inter;
    cfg.entropy_adaptation = adaptive;
    cfg.reduced_tx_set = 2u;
    int32_t coeffs[6][1024] = {{0}};
    SvtAv2TileCoeffBlock blocks[6] = {{0}};
    SvtAv2TileBlockMode modes[2] = {{1u, 3, -2,0u}, {1u, -4, 2,0u}};
    for (uint32_t sb = 0u; sb < 2u; ++sb) {
        for (uint32_t p = 0u; p < 3u; ++p) {
            const uint32_t i = sb * 3u + p;
            const uint32_t shift = p && format == 1u ? 1u : 0u;
            blocks[i].plane = p; blocks[i].x = (sb * 64u) >> shift;
            blocks[i].y = 0u;
            blocks[i].width = blocks[i].height = 64u >> shift;
            blocks[i].intra_mode = (uint8_t)sb;
            blocks[i].qcoeff = coeffs[i];
            coeffs[i][0] = sb ? -12 : 13107;
            coeffs[i][17] = sb ? 3 : -2;
        }
    }
    SvtAv2TileRateSession *session = NULL, *clean = NULL;
    SVT_TEST_CHECK(svt_av2_tile_rate_create(&cfg, &session) == 0);
    SVT_TEST_CHECK(svt_av2_tile_rate_create(&cfg, &clean) == 0);
    if (!session || !clean) {
        svt_av2_tile_rate_free(clean); svt_av2_tile_rate_free(session);
        return;
    }
    uint64_t first = 0u, second = 0u, repeat = 0u, expected = 0u, sum = 0u;
    for (uint32_t sb = 0u; sb < 2u; ++sb) {
        const SvtAv2TileBlockMode* mode = inter ? modes + sb : NULL;
        const size_t count = inter ? 1u : 0u;
        SVT_TEST_CHECK(svt_av2_tile_rate_trial(session,sb,0u,blocks+sb*3u,3u,mode,count,0,&first) == 0);
        const int32_t saved = coeffs[sb*3u][0];
        coeffs[sb*3u][0] = 0;
        SVT_TEST_CHECK(svt_av2_tile_rate_trial(session,sb,0u,blocks+sb*3u,3u,mode,count,0,&second) == 0);
        /* The clean session visits the alternatives in the reverse order. */
        SVT_TEST_CHECK(svt_av2_tile_rate_trial(clean,sb,0u,blocks+sb*3u,3u,mode,count,0,&repeat) == 0);
        SVT_TEST_CHECK(repeat == second);
        coeffs[sb*3u][0] = saved;
        SVT_TEST_CHECK(first != second);
        /* Mismatched U geometry fails after Y/context/motion were consumed. */
        ++blocks[sb*3u+1u].x;
        SVT_TEST_CHECK(svt_av2_tile_rate_trial(session,sb,0u,blocks+sb*3u,3u,mode,count,1,&repeat) != 0);
        --blocks[sb*3u+1u].x;
        if (inter) {
            SvtAv2TileBlockMode invalid = *mode;
            invalid.mv_x = 17;
            SVT_TEST_CHECK(svt_av2_tile_rate_trial(session,sb,0u,blocks+sb*3u,3u,&invalid,count,1,&repeat) != 0);
        }
        SVT_TEST_CHECK(svt_av2_tile_rate_trial(session,sb,0u,blocks+sb*3u,3u,mode,count,0,&repeat) == 0);
        SVT_TEST_CHECK(repeat == first);
        SVT_TEST_CHECK(svt_av2_tile_rate_trial(clean,sb,0u,blocks+sb*3u,3u,mode,count,1,&expected) == 0);
        SVT_TEST_CHECK(first == expected);
        SVT_TEST_CHECK(svt_av2_tile_rate_trial(session,sb,0u,blocks+sb*3u,3u,mode,count,1,&repeat) == 0);
        SVT_TEST_CHECK(repeat == first);
        sum += first;
        SVT_TEST_CHECK(svt_av2_tile_rate_trial(session,sb,0u,blocks+sb*3u,3u,mode,count,0,&repeat) != 0);
    }
    SVT_TEST_CHECK(svt_av2_tile_rate_frame(&cfg,blocks,6u,inter?modes:NULL,inter?2u:0u,&expected) == 0);
    SVT_TEST_CHECK(sum == expected);
    svt_av2_tile_rate_free(clean); svt_av2_tile_rate_free(session);
}

typedef struct Path {
    SvtAv2TileCoeffBlock blocks[96];
    SvtAv2TileBlockMode modes[32];
    int32_t coeff[96][1024];
    size_t count;
    uint32_t shift;
} Path;

static int fill_path(void* user, const SvtAv2BlockView* b) {
    Path* path = (Path*)user;
    if (path->count >= 32u) return -1;
    const size_t leaf = path->count++;
    path->modes[leaf].is_inter = (uint32_t)(leaf % 2u);
    path->modes[leaf].mv_x = (int32_t)(leaf % 7u) - 3;
    path->modes[leaf].mv_y = (int32_t)(leaf % 5u) - 2;
    for (uint32_t p = 0u; p < 3u; ++p) {
        const size_t i = leaf * 3u + p;
        const uint32_t shift = p ? path->shift : 0u;
        SvtAv2TileCoeffBlock* block = path->blocks + i;
        block->plane = p; block->x = b->x >> shift; block->y = b->y >> shift;
        block->width = b->w >> shift; block->height = b->h >> shift;
        block->intra_mode = (uint8_t)(leaf % 3u);
        block->qcoeff = path->coeff[i];
        path->coeff[i][0] = (int32_t)(leaf + 1u);
    }
    return 0;
}

static void split_and_partial(uint32_t width, uint32_t height, uint32_t format,
                              uint16_t partition, uint32_t inter, uint32_t adaptive) {
    /* This object exceeds MSVC's recommended stack frame size. */
    static Path path;
    memset(&path, 0, sizeof(path));
    path.shift = format == 1u ? 1u : 0u;
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, width, height);
    cfg.bit_depth = 10u; cfg.chroma_format = format; cfg.base_q_idx = 49u;
    cfg.m5_intra_modes = 1u; cfg.inter_frame = inter; cfg.multi_frame = inter;
    cfg.entropy_adaptation = adaptive;
    cfg.partition_decisions = &partition; cfg.partition_decision_count = 1u;
    SvtAv2FrameGeometry geometry;
    SVT_TEST_CHECK(svt_av2_frame_geometry_init_ex(&geometry,width,height,64u,
        (SvtAv2ChromaFormat)format) == 0);
    SVT_TEST_CHECK(svt_av2_partition_traverse_coded_plan(&geometry,&partition,1u,fill_path,&path) == 0);
    SVT_TEST_CHECK(path.count > 1u);
    SvtAv2TileRateSession* session = NULL;
    SVT_TEST_CHECK(svt_av2_tile_rate_create(&cfg,&session) == 0);
    if (!session) return;
    uint64_t trial, commit, whole;
    SVT_TEST_CHECK(svt_av2_tile_rate_trial(session,0u,partition,path.blocks,path.count*3u,
        inter?path.modes:NULL,inter?path.count:0u,0,&trial) == 0);
    SVT_TEST_CHECK(svt_av2_tile_rate_trial(session,0u,partition,path.blocks,path.count*3u,
        inter?path.modes:NULL,inter?path.count:0u,1,&commit) == 0);
    SVT_TEST_CHECK(svt_av2_tile_rate_frame(&cfg,path.blocks,path.count*3u,
        inter?path.modes:NULL,inter?path.count:0u,&whole) == 0);
    SVT_TEST_CHECK(trial == commit && trial == whole);
    svt_av2_tile_rate_free(session);
}

static void inter_tail_selector(void) {
    const uint16_t untouched[SVT_AV2_INTER_TAIL_MAX] = {0xffffu, 0xffffu, 0xffffu, 0xffffu};
    uint16_t scan[1024], out[SVT_AV2_INTER_TAIL_MAX];
    int32_t q[1024] = {0}, before[1024];
    const SvtAv2TxSize sizes[] = {SVT_AV2_TX_8X8, SVT_AV2_TX_16X16, SVT_AV2_TX_32X32, SVT_AV2_TX_64X64};
    const uint32_t counts[] = {64u, 256u, 1024u, 1024u};
    const uint32_t floors[] = {3u, 4u, 16u, 16u};
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
        const uint32_t n = counts[s], f = floors[s];
        const size_t scan_count = svt_av2_coeff_scan_dct_dct(sizes[s], scan);
        SVT_TEST_CHECK_EQ_INT(scan_count, n);
        if (scan_count != n) continue;
        memset(q, 0, sizeof(q));
        memcpy(out, untouched, sizeof(out));
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_tail_positions(q, scan, n, out), 0u);
        SVT_TEST_CHECK(memcmp(out, untouched, sizeof(out)) == 0);

        /* Skipping zeros and descending scan positions is not a raster scan.
         * All six coefficients are unit magnitude but the result caps at four. */
        for (uint32_t i = 0; i < 6u; ++i) q[scan[n - 1u - 3u * i]] = i & 1u ? -1 : 1;
        memcpy(before, q, sizeof(q));
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_tail_positions(q, scan, n, out), SVT_AV2_INTER_TAIL_MAX);
        for (uint32_t i = 0; i < SVT_AV2_INTER_TAIL_MAX; ++i)
            SVT_TEST_CHECK_EQ_INT(out[i], scan[n - 1u - 3u * i]);
        SVT_TEST_CHECK(memcmp(before, q, sizeof(q)) == 0);

        /* A strong level cuts the search; even a negative extreme is not
         * negated or mistaken for a unit coefficient. */
        q[scan[n - 7u]] = INT32_MIN;
        memcpy(before, q, sizeof(q));
        memcpy(out, untouched, sizeof(out));
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_tail_positions(q, scan, n, out), 2u);
        SVT_TEST_CHECK_EQ_INT(out[0], scan[n - 1u]);
        SVT_TEST_CHECK_EQ_INT(out[1], scan[n - 4u]);
        SVT_TEST_CHECK(memcmp(out + 2u, untouched + 2u, 2u * sizeof(*out)) == 0);
        SVT_TEST_CHECK(memcmp(before, q, sizeof(q)) == 0);

        /* The floor is inclusive and protects the lower scan position, not
         * lower raster offsets. Both floors are tested in the same block. */
        memset(q, 0, sizeof(q));
        q[scan[f - 1u]] = -1;
        q[scan[f]] = 1;
        memcpy(out, untouched, sizeof(out));
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_tail_positions(q, scan, n, out), 1u);
        SVT_TEST_CHECK_EQ_INT(out[0], scan[f]);
        SVT_TEST_CHECK_EQ_INT(out[1], untouched[1]);
        q[scan[f - 1u]] = 0;
        q[scan[2u * f]] = -1;
        SVT_TEST_CHECK_EQ_INT(svt_av2_inter_tail_positions(q, scan, n, out), 2u);
        SVT_TEST_CHECK_EQ_INT(out[0], scan[2u * f]);
        SVT_TEST_CHECK_EQ_INT(out[1], scan[f]);
        /* Clearing the two returned tail positions may empty TXB; use the
         * asserted oracle offsets so a failed assertion cannot index q. */
        q[scan[2u * f]] = q[scan[f]] = 0;
        SVT_TEST_CHECK_EQ_INT(svt_av2_coeff_find_eob(q, scan, n), 0u);
    }

    /* For nominal 64x64, scan offsets address the packed 32x32 region. */
    uint16_t packed_scan[1024];
    const size_t packed_count = svt_av2_coeff_scan_dct_dct(SVT_AV2_TX_32X32, packed_scan);
    const size_t nominal_count = svt_av2_coeff_scan_dct_dct(SVT_AV2_TX_64X64, scan);
    SVT_TEST_CHECK_EQ_INT(packed_count, 1024u);
    SVT_TEST_CHECK_EQ_INT(nominal_count, packed_count);
    if (nominal_count != 1024u || packed_count != 1024u) return;
    memset(q, 0, sizeof(q));
    q[packed_scan[32]] = -1;
    SVT_TEST_CHECK(packed_scan[32] != 32u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_tail_positions(q, scan, 1024u, out), 1u);
    SVT_TEST_CHECK_EQ_INT(out[0], packed_scan[32]);

    memcpy(out, untouched, sizeof(out));
    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_tail_positions(NULL, scan, 1024u, out), 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_tail_positions(q, NULL, 1024u, out), 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_tail_positions(q, scan, 1024u, NULL), 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_tail_positions(q, scan, 0u, out), 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_tail_positions(q, scan, 16u, out), 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_tail_positions(q, scan, 8u * 16u, out), 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_inter_tail_positions(q, scan, 64u * 64u, out), 0u);
    SVT_TEST_CHECK(memcmp(out, untouched, sizeof(out)) == 0);
}

int main(void) {
    probability_cost();
    inter_tail_selector();
    for (uint32_t inter = 0u; inter < 2u; ++inter) {
        transactions(8u,1u,inter,0u,100u);
        transactions(10u,1u,inter,0u,148u);
        transactions(10u,3u,inter,0u,148u);
        split_and_partial(64u,64u,1u,1u,inter,0u);
        split_and_partial(64u,64u,3u,1u,inter,0u);
        split_and_partial(56u,40u,1u,0u,inter,0u);
        split_and_partial(56u,40u,3u,0u,inter,0u);
    }
    transactions(8u,1u,0u,1u,100u);
    transactions(10u,1u,0u,1u,148u);
    transactions(10u,3u,0u,1u,148u);
    /* AVM base_qindex thresholds are 90, 140, and 190 for both depths. */
    const uint32_t base_q_idx[] = {80u, 100u, 150u, 210u};
    for (size_t i = 0u; i < sizeof(base_q_idx) / sizeof(base_q_idx[0]); ++i) {
        transactions(8u,1u,1u,1u,base_q_idx[i]);
        transactions(10u,1u,1u,1u,base_q_idx[i]);
        transactions(10u,3u,1u,1u,base_q_idx[i]);
    }
    split_and_partial(64u,64u,1u,1u,1u,1u);
    split_and_partial(64u,64u,3u,1u,1u,1u);
    split_and_partial(56u,40u,1u,0u,1u,1u);
    split_and_partial(56u,40u,3u,0u,1u,1u);
    split_and_partial(64u,64u,1u,2u,0u,0u);
    split_and_partial(64u,64u,3u,2u,0u,0u);
    split_and_partial(64u,64u,1u,2u,0u,1u);
    split_and_partial(64u,64u,3u,2u,0u,1u);
    split_and_partial(64u,64u,1u,1u,0u,1u);
    split_and_partial(64u,64u,3u,1u,0u,1u);
    split_and_partial(64u,64u,1u,SVT_AV2_MIXED_RECT32_BASE + 5u + 6u * 7u + 4u * 343u,0u,1u);
    split_and_partial(64u,64u,1u,SVT_AV2_RECT16_BASE + 5u * 343u + 4u * 57u,0u,1u);
    split_and_partial(64u,64u,1u,SVT_AV2_HALF_RECT_BASE + 3u * 343u + 4u + 7u * 5u + 49u * 6u,0u,1u);
    split_and_partial(56u,40u,1u,0u,0u,1u);
    split_and_partial(56u,40u,3u,0u,0u,1u);
    SVT_TEST_REPORT("m6_rate");
}
