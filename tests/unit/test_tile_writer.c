/*
 * T005 unit tests: minimal CLOSED_LOOP_KEY tile writer and its entropy payload.
 *
 * The expected tile payload bytes were produced by an independent model: the
 * AV2 v1.0.0 spec syntax was traced by hand, the seven CDFs were read directly
 * from the spec attachment tables, and the resulting symbol stream was coded
 * with a port of the reference range coder (avm/avm_dsp/entenc.c). The same
 * bytes are what the read-only AVM oracle (avmdec.exe) accepts as a one-frame
 * 64x64 CLOSED_LOOP_KEY picture, so this test pins the exact bitstream rather
 * than only re-checking internal consistency.
 *
 * Expected tile payload (4 bytes): E6 C8 00 12
 *   E6 C8 00 : is_first_tile_group(1) + 22-bit frame header, byte aligned
 *   12       : range-coded 7 symbols
 *             do_split=0, y_mode_set=0, y_mode_index=0, uv_mode=0,
 *             all_zero(luma)=1, all_zero(U)=1, all_zero(V)=1
 */
#include "bit_writer.h"
#include "block_writer.h"
#include "entropy_models.h"
#include "frame_header.h"
#include "keyframe.h"
#include "tile_writer.h"

#include "test_util.h"

#include <stdint.h>
#include <string.h>

/* The frozen M1 frame header descriptor length, in bits. */
#define SVT_AV2_M1_FRAME_HEADER_BITS SVT_AV2_FRAME_HEADER_M1_BITS

static int bytes_eq(const uint8_t* got, const uint8_t* want, size_t n) {
    return memcmp(got, want, n) == 0;
}

static void test_frame_header_bit_length(void) {
    /* The header alone must consume exactly SVT_AV2_M1_FRAME_HEADER_BITS bits,
     * with no trailing padding. Checked via the bit writer's position. */
    uint8_t             buf[8];
    SvtAv2BitWriter     bw;
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_bit_writer_init(&bw, buf, sizeof(buf));
    svt_av2_frame_header_config_default(&cfg, 64, 64);
    SVT_TEST_CHECK(svt_av2_frame_header_write(&bw, &cfg) == SVT_AV2_BW_OK);
    SVT_TEST_CHECK(svt_av2_bit_writer_bits_written(&bw) ==
                   SVT_AV2_M1_FRAME_HEADER_BITS);
}

static void test_tile_payload_exact_bytes(void) {
    /* Full tile_group_obu() payload for the M1 64x64 CLOSED_LOOP_KEY picture. */
    static const uint8_t expect[4] = {0xE6, 0xC8, 0x00, 0x12};

    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, 64, 64);

    uint8_t tile[SVT_AV2_TILE_PAYLOAD_MAX_BYTES];
    size_t  tile_size = 0;
    int st = svt_av2_tile_writer_write_closed_loop_key(&cfg, tile, sizeof(tile),
                                                       &tile_size);
    SVT_TEST_CHECK(st == 0);
    SVT_TEST_CHECK(tile_size == sizeof(expect));
    SVT_TEST_CHECK(bytes_eq(tile, expect, sizeof(expect)));
}

static void test_tile_payload_no_space_fails_closed(void) {
    /* A buffer one byte short must fail rather than truncate silently. */
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, 64, 64);

    uint8_t tile[3];
    size_t  tile_size = 123; /* poison: must not be read on failure */
    int st = svt_av2_tile_writer_write_closed_loop_key(&cfg, tile, sizeof(tile),
                                                       &tile_size);
    SVT_TEST_CHECK(st != 0);
}

static void test_block_symbols_independent_roundtrip(void) {
    /* The policy layer writes exactly SVT_AV2_BLOCK_MIN_SYMBOLS symbols and the
     * result must match the reference byte 0x12 (the coded form of the seven
     * symbols with the real spec CDFs). */
    /* The M1 block is a fixed 7-symbol payload: three all_zero symbols plus the
     * four do_split/seg/skip/DC symbols, which code to a single byte. Size the
     * scratch explicitly rather than from SVT_AV2_TILE_PAYLOAD_MAX_BYTES, which
     * is documented as the M1 payload bound and not as coder scratch. */
    uint8_t             coded[16];
    uint16_t            precarry[18];
    SvtAv2EntropyEncoder enc;
    svt_av2_entropy_encoder_init(&enc, coded, sizeof(coded), precarry,
                                 sizeof(precarry) / sizeof(precarry[0]));
    SVT_TEST_CHECK(svt_av2_block_write_minimal_intra(&enc, 100) ==
                   SVT_AV2_ENTROPY_OK);

    uint8_t* out = NULL;
    size_t   n   = 0;
    SVT_TEST_CHECK(svt_av2_entropy_encoder_done(&enc, &out, &n) ==
                   SVT_AV2_ENTROPY_OK);
    SVT_TEST_CHECK(n == 1u);
    SVT_TEST_CHECK(n == 1u && out[0] == 0x12u);
}

static void test_block_rejects_wrong_quantizer(void) {
    /* The tabulated all_zero CDFs are for COEFF_CDF_Q_CTXS index 1 (base_q_idx
     * in (90,140]). Any other quantizer must be rejected, not coded with a
     * mismatched CDF. */
    uint8_t             coded[8];
    uint16_t            precarry[8];
    SvtAv2EntropyEncoder enc;
    svt_av2_entropy_encoder_init(&enc, coded, sizeof(coded), precarry, 8);
    SVT_TEST_CHECK(svt_av2_block_write_minimal_intra(&enc, 40) ==
                   SVT_AV2_ENTROPY_ERROR_RANGE);
}

static void test_m5_p_modes_and_inter_metadata(void) {
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, 192u, 64u);
    cfg.multi_frame = 1u;
    cfg.inter_frame = 1u;
    cfg.m5_intra_modes = 1u;
    cfg.reduced_tx_set = 2u;
    const int32_t zero[1024] = {0};
    SvtAv2TileCoeffBlock blocks[9];
    memset(blocks, 0, sizeof(blocks));
    for (uint32_t b = 0; b < 3u; ++b) {
        for (uint32_t p = 0; p < 3u; ++p) {
            SvtAv2TileCoeffBlock* block = blocks + b*3u + p;
            block->plane = p;
            block->x = b * (p ? 32u : 64u);
            block->width = block->height = p ? 32u : 64u;
            block->qcoeff = zero;
            block->intra_mode = b == 0u ? 1u : 2u;
        }
    }
    const SvtAv2TileBlockMode modes[3] = {{0u,0,0,0u},{1u,0,0,0u},{0u,0,0,0u}};
    uint8_t first[1024], second[1024];
    size_t first_size = 0u, second_size = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_writer_write_inter_coeffs(&cfg,
        blocks, 9u, modes, 3u, first, sizeof(first), &first_size), 0);
    /* An inter block's intra metadata cannot leak into the following H block. */
    blocks[3].intra_mode = 255u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_writer_write_inter_coeffs(&cfg,
        blocks, 9u, modes, 3u, second, sizeof(second), &second_size), 0);
    SVT_TEST_CHECK(first_size == second_size &&
        memcmp(first, second, first_size) == 0);
    blocks[6].intra_mode = 3u;
    SVT_TEST_CHECK(svt_av2_tile_writer_write_inter_coeffs(&cfg,
        blocks, 9u, modes, 3u, second, sizeof(second), &second_size) != 0);
    /* The disabled M4 path never consumes any intra-mode metadata. */
    cfg.m5_intra_modes = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_writer_write_inter_coeffs(&cfg,
        blocks, 9u, modes, 3u, second, sizeof(second), &second_size), 0);
}

static void test_m52_partition_plan_controls_coeff_order(void) {
    const int32_t zero[1024] = {0};
    const SvtAv2TileBlockMode modes[4] = {{0u,0,0,0u},{1u,0,0,0u},{0u,0,0,0u},{1u,0,0,0u}};
    for (uint32_t format = 0; format < 3u; ++format) {
        SvtAv2FrameHeaderConfig cfg;
        svt_av2_frame_header_config_default(&cfg, 64u, 64u);
        cfg.bit_depth = format ? 10u : 8u;
        cfg.chroma_format = format == 2u ? 3u : 1u;
        cfg.base_q_idx = format ? 148u : 100u;
        cfg.multi_frame = 1u;
        cfg.reduced_tx_set = 2u;
        uint16_t decision = 1u;
        cfg.partition_decisions = &decision;
        cfg.partition_decision_count = 1u;
        SvtAv2TileCoeffBlock blocks[12];
        memset(blocks, 0, sizeof(blocks));
        for (uint32_t q = 0; q < 4u; ++q) {
            for (uint32_t p = 0; p < 3u; ++p) {
                SvtAv2TileCoeffBlock* block = blocks + q*3u + p;
                const uint32_t size = p && format != 2u ? 16u : 32u;
                block->plane = p;
                block->x = (q & 1u) * size;
                block->y = (q >> 1u) * size;
                block->width = block->height = size;
                block->qcoeff = zero;
                block->intra_mode = q & 1u ? 2u : 1u;
            }
        }
        uint8_t out[2048];
        size_t count = 0u;
        for (uint32_t adaptive_mode = 0; adaptive_mode < 2u; ++adaptive_mode) {
            cfg.m5_intra_modes = adaptive_mode;
            cfg.inter_frame = 0u;
            SVT_TEST_CHECK_EQ_INT(svt_av2_tile_writer_write_closed_loop_key_coeffs(
                &cfg, blocks, 12u, out, sizeof(out), &count, NULL), 0);
            cfg.inter_frame = 1u;
            SVT_TEST_CHECK_EQ_INT(svt_av2_tile_writer_write_inter_coeffs(
                &cfg, blocks, 12u, modes, 4u, out, sizeof(out), &count), 0);
        }
        /* Writer must consume the shared plan, never infer a split from coeffs. */
        decision = 0u;
        SVT_TEST_CHECK(svt_av2_tile_writer_write_inter_coeffs(
            &cfg, blocks, 12u, modes, 4u, out, sizeof(out), &count) != 0);
        decision = 2u;
        SVT_TEST_CHECK(svt_av2_tile_writer_write_inter_coeffs(
            &cfg, blocks, 12u, modes, 4u, out, sizeof(out), &count) != 0);
        cfg.partition_decisions = NULL;
        SVT_TEST_CHECK(svt_av2_tile_writer_write_inter_coeffs(
            &cfg, blocks, 12u, modes, 4u, out, sizeof(out), &count) != 0);
    }
}

static void test_m53_supported_qindex_range(void) {
    const int32_t zero[1024] = {0};
    for (uint32_t format = 0; format < 3u; ++format) {
        SvtAv2FrameHeaderConfig cfg;
        svt_av2_frame_header_config_default(&cfg,64u,64u);
        cfg.bit_depth = format ? 10u : 8u;
        cfg.chroma_format = format == 2u ? 3u : 1u;
        cfg.multi_frame = 1u;
        cfg.reduced_tx_set = 2u;
        SvtAv2TileCoeffBlock blocks[3];
        memset(blocks,0,sizeof(blocks));
        for (uint32_t p = 0; p < 3u; ++p) {
            blocks[p].plane = p;
            blocks[p].width = blocks[p].height = p && format != 2u ? 32u : 64u;
            blocks[p].qcoeff = zero;
        }
        const uint32_t valid[] = {format ? 49u : 1u,90u,91u,140u,141u,190u,191u,format ? 303u : 255u};
        uint8_t out[256];
        size_t count = 0;
        for (size_t i = 0; i < sizeof(valid)/sizeof(valid[0]); ++i) {
            cfg.base_q_idx = valid[i];
            SVT_TEST_CHECK_EQ_INT(svt_av2_tile_writer_write_closed_loop_key_coeffs(
                &cfg,blocks,3u,out,sizeof(out),&count,NULL),0);
        }
        cfg.base_q_idx = format ? 48u : 0u;
        SVT_TEST_CHECK(svt_av2_tile_writer_write_closed_loop_key_coeffs(
            &cfg,blocks,3u,out,sizeof(out),&count,NULL) != 0);
        cfg.base_q_idx = format ? 304u : 256u;
        SVT_TEST_CHECK(svt_av2_tile_writer_write_closed_loop_key_coeffs(
            &cfg,blocks,3u,out,sizeof(out),&count,NULL) != 0);
    }
}

static void test_two_logical_ref_rate_rollback(void) {
    static const int32_t zero[1024] = {0};
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, 128u, 64u);
    cfg.multi_frame = cfg.inter_frame = cfg.dual_ref = 1u;
    cfg.reduced_tx_set = 2u;
    cfg.logical_ref_count = 2u;
    SvtAv2TileCoeffBlock blocks[6] = {0};
    SvtAv2TileBlockMode modes[2] = {{1u,0,0,0u},{1u,0,0,1u}};
    for (uint32_t i = 0; i < 6u; ++i) {
        const uint32_t p = i % 3u;
        blocks[i].plane = p;
        blocks[i].width = blocks[i].height = p ? 32u : 64u;
        blocks[i].x = (i / 3u) * blocks[i].width;
        blocks[i].qcoeff = zero;
    }
    uint64_t whole = 0u, left = 0u, right = 0u, scratch = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_frame(&cfg, blocks, 6u, modes, 2u, &whole), 0);
    SvtAv2TileRateSession* session = NULL;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_create(&cfg, &session), 0);
    if (session == NULL) return;
    modes[0].reference_rank = 1u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(session, 0u, 0u, blocks, 3u, modes, 1u, 0, &scratch), 0);
    modes[0].reference_rank = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(session, 0u, 0u, blocks, 3u, modes, 1u, 1, &left), 0);
    SVT_TEST_CHECK(scratch != left); /* Same block, different arithmetic rank. */
    modes[1].reference_rank = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(session, 1u, 0u, blocks + 3u, 3u, modes + 1u, 1u, 0, &scratch), 0);
    modes[1].reference_rank = 1u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(session, 1u, 0u, blocks + 3u, 3u, modes + 1u, 1u, 1, &right), 0);
    SVT_TEST_CHECK(scratch != right);
    SVT_TEST_CHECK_EQ_INT(left + right, whole);
    svt_av2_tile_rate_free(session);
}

static void test_adaptive_ref_rate_rollback(int inherited) {
    static const int32_t zero[1024] = {0};
    enum { COLS = 4, ROWS = 3, SBS = COLS * ROWS, LEAVES = SBS * 4 };
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, COLS * 64u, ROWS * 64u);
    cfg.multi_frame = cfg.inter_frame = cfg.dual_ref = cfg.entropy_adaptation = 1u;
    cfg.reduced_tx_set = cfg.logical_ref_count = cfg.mv_half = 2u;
    cfg.sharp_filter = 1u;
    uint16_t plan[SBS];
    SvtAv2TileCoeffBlock blocks[LEAVES * 3] = {0};
    SvtAv2TileBlockMode modes[LEAVES] = {0};
    for (uint32_t sb = 0u; sb < SBS; ++sb) {
        plan[sb] = 1u;
        for (uint32_t j = 0u; j < 4u; ++j) {
            const uint32_t x = (sb % COLS) * 64u + (j & 1u) * 32u;
            const uint32_t y = (sb / COLS) * 64u + (j >> 1u) * 32u;
            const uint32_t leaf = sb * 4u + j;
            modes[leaf].is_inter = 1u;
            modes[leaf].reference_rank = (x >= 128u) ^ (y >= 96u);
            for (uint32_t p = 0; p < 3u; ++p) {
                const uint32_t shift = p != 0u;
                SvtAv2TileCoeffBlock* b = &blocks[leaf * 3u + p];
                b->plane = p;
                b->width = b->height = 32u >> shift;
                b->x = x >> shift;
                b->y = y >> shift;
                b->qcoeff = zero;
            }
        }
    }
    cfg.partition_decisions = plan;
    cfg.partition_decision_count = SBS;
    cfg.order_hint = 9u;
    SvtAv2EntropyModels seed, original;
    if (inherited) {
        SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_models_init(&seed, 3u), 0);
        for (uint32_t i = 0u; i < 32u; ++i)
            SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_model_update(seed.cdf[SVT_AV2_MODEL_SINGLE_REF], 2u, 1u), 0);
        svt_av2_entropy_models_decay(&seed);
        original = seed;
        cfg.primary_models = &seed;
    }
    SvtAv2TileRateSession *fresh = NULL, *with_trials = NULL;
    const int a = svt_av2_tile_rate_create(&cfg, &fresh);
    const int b = svt_av2_tile_rate_create(&cfg, &with_trials);
    SVT_TEST_CHECK_EQ_INT(a, 0);
    SVT_TEST_CHECK_EQ_INT(b, 0);
    if (a || b) { svt_av2_tile_rate_free(fresh); svt_av2_tile_rate_free(with_trials); return; }
    uint64_t whole = 0u, sum = 0u;
    uint32_t differences = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_frame(&cfg, blocks, LEAVES * 3u, modes, LEAVES, &whole), 0);
    for (size_t sb = 0u; sb < SBS; ++sb) {
        const SvtAv2TileBlockMode* winner = modes + sb * 4u;
        const SvtAv2TileCoeffBlock* coeff = blocks + sb * 12u;
        SvtAv2TileBlockMode losing[4];
        memcpy(losing, winner, sizeof(losing));
        for (uint32_t j = 0u; j < 4u; ++j) losing[j].reference_rank ^= 1u;
        uint64_t losing_rate = 0u, failure = 99u, committed = 0u, control = 0u;
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(with_trials, sb, plan[sb], coeff, 12u, losing, 4u, 0, &losing_rate), 0);
        losing[3].mv_y = -1; /* Last leaf rejects after prior ranks were trained. */
        SVT_TEST_CHECK(svt_av2_tile_rate_trial(with_trials, sb, plan[sb], coeff, 12u, losing, 4u, 1, &failure) != 0);
        SVT_TEST_CHECK_EQ_INT(failure, 0u);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(with_trials, sb, plan[sb], coeff, 12u, winner, 4u, 1, &committed), 0);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(fresh, sb, plan[sb], coeff, 12u, winner, 4u, 1, &control), 0);
        SVT_TEST_CHECK_EQ_INT(committed, control);
        differences += losing_rate != committed;
        sum += committed;
    }
    SVT_TEST_CHECK_EQ_INT(sum, whole);
    SVT_TEST_CHECK(differences > 6u);
    SvtAv2EntropyModels trial_end, fresh_end, formal_end;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_end_models(with_trials, &trial_end), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_end_models(fresh, &fresh_end), 0);
    SVT_TEST_CHECK(memcmp(&trial_end, &fresh_end, sizeof(trial_end)) == 0);
    memset(&formal_end, 0x1d, sizeof(formal_end));
    const SvtAv2EntropyModels poison = formal_end;
    uint8_t packet[4096]; size_t written = SIZE_MAX;
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_video_models(&cfg, 9u, blocks, LEAVES * 3u,
        modes, LEAVES, packet, 20u, &written, &formal_end), SVT_AV2_BW_ERROR_NO_SPACE);
    SVT_TEST_CHECK_EQ_INT(written, 0u);
    SVT_TEST_CHECK(memcmp(&formal_end, &poison, sizeof(formal_end)) == 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_video_models(&cfg, 9u, blocks, LEAVES * 3u,
        modes, LEAVES, packet, sizeof(packet), &written, &formal_end), 0);
    SVT_TEST_CHECK(written > 20u);
    SVT_TEST_CHECK(memcmp(&formal_end, &fresh_end, sizeof(formal_end)) == 0);
    if (inherited) SVT_TEST_CHECK(memcmp(&seed, &original, sizeof(seed)) == 0);
    svt_av2_tile_rate_free(with_trials);
    svt_av2_tile_rate_free(fresh);
}

/* Stable binary choices in coding order: a path's lower bits determine all
 * motion/metadata/coefficient inputs in that prefix, including on reverse
 * traversal. The projected vectors are multiples of eight Q3 units, keeping
 * their quarter-delta lattice legal across 1:4 cross-reference prediction. */
static void inter_cache_candidate(uint32_t sb_x, uint32_t sb_y, uint32_t ordinal,
    const SvtAv2FrameHeaderConfig* cfg, SvtAv2TileCoeffBlock blocks[12], SvtAv2TileBlockMode modes[4]) {
    static const int32_t zero[1024] = {0};
    static const int32_t dc[1024] = {-5};
    static const int32_t ac[1024] = {0,3,-1,2};
    static const int32_t mixed[1024] = {3,-2,0,0,1};
    const int32_t* const residuals[4] = {zero, dc, ac, mixed};
    const uint32_t sb = sb_x / 64u + sb_y / 64u;
    memset(blocks, 0, 12u * sizeof(*blocks));
    memset(modes, 0, 4u * sizeof(*modes));
    for (uint32_t leaf = 0u; leaf < 4u; ++leaf) {
        const uint32_t inter = (ordinal >> leaf) & 1u;
        const uint32_t x = sb_x + (leaf & 1u) * 32u;
        const uint32_t y = sb_y + (leaf >> 1u) * 32u;
        modes[leaf].is_inter = inter;
        if (inter) {
            modes[leaf].reference_rank = cfg->dual_ref ? (leaf + sb) & 1u : 0u;
            modes[leaf].mv_x = 8 * (int32_t)(leaf + 1u);
            modes[leaf].mv_y = -8 * (int32_t)((sb + leaf) % 4u + 1u);
        }
        for (uint32_t p = 0u; p < 3u; ++p) {
            SvtAv2TileCoeffBlock* b = blocks + leaf * 3u + p;
            const uint32_t shift = p != 0u && cfg->chroma_format != 3u;
            b->plane = p;
            b->width = b->height = 32u >> shift;
            b->x = x >> shift;
            b->y = y >> shift;
            b->qcoeff = residuals[(sb + leaf + p + 2u * inter) & 3u];
            if (inter && (sb + leaf) % 4u == 0u) b->qcoeff = zero;
            if (!inter && !p) b->intra_mode = (uint8_t)((y && !(leaf & 1u)) ? 1u : x ? 2u : 0u);
        }
    }
}

/* Queries do not supply a candidate prefix. If a bad/losing trial leaked its
 * MI rectangle into the live session, its nonzero vectors can be observed in
 * the subsequent leaves even when there is no committed current-SB predictor. */
static void inter_cache_check_live_motion(const SvtAv2TileRateSession* cached,
    const SvtAv2TileRateSession* fresh, uint32_t x, uint32_t y, uint32_t rank_count) {
    for (uint32_t leaf = 0u; leaf < 4u; ++leaf) {
        SvtAv2BlockView b = {0};
        b.x = x + (leaf & 1u) * 32u;
        b.y = y + (leaf >> 1u) * 32u;
        b.w = b.h = 32u;
        for (uint32_t rank = 0u; rank < rank_count; ++rank) {
            int32_t cx = 0, cy = 0, fx = 0, fy = 0;
            SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_motion_predict(cached, &b, rank, NULL, &cx, &cy), 0);
            SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_motion_predict(fresh, &b, rank, NULL, &fx, &fy), 0);
            SVT_TEST_CHECK(cx == fx && cy == fy);
        }
    }
}

static void test_inter_cache_exact(uint32_t depth, uint32_t chroma, uint32_t adaptation,
    uint32_t projected, uint32_t reverse) {
    enum { COLS = 3, ROWS = 2, SBS = COLS * ROWS };
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, 64u * COLS, 64u * ROWS);
    cfg.multi_frame = cfg.inter_frame = cfg.m5_intra_modes = cfg.sharp_filter = 1u;
    cfg.reduced_tx_set = cfg.mv_half = 2u;
    cfg.bit_depth = depth;
    cfg.chroma_format = chroma;
    cfg.base_q_idx = depth == 10u ? 148u : 100u;
    cfg.entropy_adaptation = adaptation != 0u;
    cfg.dual_ref = projected;
    cfg.logical_ref_count = projected ? 2u : 1u;
    cfg.ref_distance[0] = projected;
    cfg.ref_distance[1] = 4u * projected;
    cfg.order_hint = 9u;
    uint16_t plan[SBS];
    for (size_t sb = 0u; sb < SBS; ++sb) plan[sb] = 1u;
    cfg.partition_decisions = plan;
    cfg.partition_decision_count = SBS;
    SvtAv2EntropyModels seed, original;
    if (adaptation == 2u) {
        SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_models_init(&seed, 3u), 0);
        for (uint32_t i = 0u; i < 32u; ++i) {
            SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_model_update(seed.cdf[SVT_AV2_MODEL_SINGLE_REF], 2u, 1u), 0);
            SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_model_update(seed.cdf[SVT_AV2_MODEL_INTER_MODE], 3u, 2u), 0);
        }
        svt_av2_entropy_models_decay(&seed);
        original = seed;
        cfg.primary_models = &seed;
    }
    SvtAv2TileRateSession *cached = NULL, *fresh = NULL;
    SvtAv2TileInterRateCache* cache = NULL;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_create(&cfg, &cached), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_create(&cfg, &fresh), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_inter_rate_cache_create(&cache), 0);
    if (!cached || !fresh || !cache) goto cleanup;
    SvtAv2TileCoeffBlock winners[SBS * 12];
    SvtAv2TileBlockMode winner_modes[SBS * 4];
    uint64_t sum = 0u, whole = 0u;
    for (uint32_t sb = 0u; sb < SBS; ++sb) {
        const uint32_t x = (sb % COLS) * 64u, y = (sb / COLS) * 64u;
        SvtAv2TileCoeffBlock blocks[12];
        SvtAv2TileBlockMode modes[4];
        uint64_t rate, control;
        SVT_TEST_CHECK(svt_av2_tile_inter_rate_cache_begin(cache, cached, sb + 1u) < 0);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_inter_rate_cache_begin(cache, cached, sb), 0);
        inter_cache_candidate(x, y, 15u, &cfg, blocks, modes);
        rate = 99u;
        SVT_TEST_CHECK(svt_av2_tile_inter_rate_cache_trial(cache, 16u, blocks, modes, &rate) < 0);
        SVT_TEST_CHECK_EQ_INT(rate, 0u);
        /* Cold miss: deliberately alter the first MV, then fail at the final
         * V plane, after motion and both earlier planes have been visited.
         * Retry with the normal first MV; failed prefixes must be unpublished. */
        static const int32_t changed_prefix[1024] = {27, -18, 13, -9, 6, -4};
        modes[0].mv_x = -16;
        blocks[0].qcoeff = changed_prefix;
        uint64_t changed_rate = 0u;
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(fresh, sb, 1u, blocks, 12u, modes, 4u, 0, &changed_rate), 0);
        blocks[11].plane = 1u;
        rate = 99u;
        SVT_TEST_CHECK(svt_av2_tile_inter_rate_cache_trial(cache, 15u, blocks, modes, &rate) < 0);
        SVT_TEST_CHECK_EQ_INT(rate, 0u);
        inter_cache_check_live_motion(cached, fresh, x, y, cfg.logical_ref_count);
        inter_cache_candidate(x, y, 15u, &cfg, blocks, modes);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_inter_rate_cache_trial(cache, 15u, blocks, modes, &rate), 0);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(fresh, sb, 1u, blocks, 12u, modes, 4u, 0, &control), 0);
        SVT_TEST_CHECK_EQ_INT(rate, control);
        SVT_TEST_CHECK(changed_rate != control);

        /* Reset to a fresh per-SB cache for an uncontaminated 16-path count. */
        uint64_t leaves_before = 0u, events_before = 0u, leaves_after = 0u, events_after = 0u;
        svt_av2_tile_inter_rate_cache_stats(cache, &leaves_before, &events_before, NULL);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_inter_rate_cache_begin(cache, cached, sb), 0);
        uint64_t first = 0u;
        uint32_t differing = 0u;
        for (uint32_t i = 0u; i < 16u; ++i) {
            const uint32_t ordinal = reverse ? 15u - i : i;
            inter_cache_candidate(x, y, ordinal, &cfg, blocks, modes);
            SVT_TEST_CHECK_EQ_INT(svt_av2_tile_inter_rate_cache_trial(cache, ordinal, blocks, modes, &rate), 0);
            inter_cache_check_live_motion(cached, fresh, x, y, cfg.logical_ref_count);
            SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(fresh, sb, 1u, blocks, 12u, modes, 4u, 0, &control), 0);
            SVT_TEST_CHECK_EQ_INT(rate, control);
            SVT_TEST_CHECK(rate > 0u);
            if (!i) first = rate;
            differing += rate != first;
        }
        SVT_TEST_CHECK(differing >= 12u);
        svt_av2_tile_inter_rate_cache_stats(cache, &leaves_after, &events_after, NULL);
        SVT_TEST_CHECK_EQ_INT(leaves_after - leaves_before, 30u); /* 14 nonterminal + 16 terminal. */
        SVT_TEST_CHECK_EQ_INT(events_after - events_before, 21u);

        /* Warm hit: an invalid terminal plane cannot poison its already valid
         * prefix. Replay and commit a varied winner separately in both sessions. */
        const uint32_t selected = sb == 0u ? 15u : (7u + 5u * sb) & 15u;
        SvtAv2TileCoeffBlock* chosen = winners + 12u * sb;
        SvtAv2TileBlockMode* chosen_modes = winner_modes + 4u * sb;
        inter_cache_candidate(x, y, selected, &cfg, chosen, chosen_modes);
        chosen[11].qcoeff = NULL;
        rate = 99u;
        SVT_TEST_CHECK(svt_av2_tile_inter_rate_cache_trial(cache, selected, chosen, chosen_modes, &rate) < 0);
        SVT_TEST_CHECK_EQ_INT(rate, 0u);
        inter_cache_check_live_motion(cached, fresh, x, y, cfg.logical_ref_count);
        inter_cache_candidate(x, y, selected, &cfg, chosen, chosen_modes);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_inter_rate_cache_trial(cache, selected, chosen, chosen_modes, &rate), 0);
        uint64_t committed = 0u;
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(cached, sb, 1u, chosen, 12u, chosen_modes, 4u, 1, &committed), 0);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(fresh, sb, 1u, chosen, 12u, chosen_modes, 4u, 1, &control), 0);
        SVT_TEST_CHECK_EQ_INT(committed, rate);
        SVT_TEST_CHECK_EQ_INT(committed, control);
        sum += committed;
        rate = 99u;
        SVT_TEST_CHECK(svt_av2_tile_inter_rate_cache_trial(cache, selected, chosen, chosen_modes, &rate) < 0);
        SVT_TEST_CHECK_EQ_INT(rate, 0u);
    }
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_frame(&cfg, winners, SBS * 12u, winner_modes, SBS * 4u, &whole), 0);
    SVT_TEST_CHECK_EQ_INT(sum, whole);
    if (adaptation) {
        SvtAv2EntropyModels cached_end, fresh_end, formal_end;
        uint8_t tile[8192];
        size_t written = 0u;
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_end_models(cached, &cached_end), 0);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_end_models(fresh, &fresh_end), 0);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_writer_write_video_models(&cfg, winners, SBS * 12u,
            winner_modes, SBS * 4u, tile, sizeof(tile), &written, &formal_end), 0);
        SVT_TEST_CHECK(written > 0u);
        SVT_TEST_CHECK(memcmp(&cached_end, &fresh_end, sizeof(cached_end)) == 0);
        SVT_TEST_CHECK(memcmp(&cached_end, &formal_end, sizeof(cached_end)) == 0);
    }
    if (adaptation == 2u) SVT_TEST_CHECK(memcmp(&seed, &original, sizeof(seed)) == 0);
cleanup:
    svt_av2_tile_inter_rate_cache_free(cache);
    svt_av2_tile_rate_free(cached);
    svt_av2_tile_rate_free(fresh);
}

static void test_inter_cache_rejects_partial_superblock(void) {
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, 72u, 64u);
    cfg.multi_frame = cfg.inter_frame = cfg.sharp_filter = 1u;
    cfg.mv_half = cfg.reduced_tx_set = 2u;
    SvtAv2TileRateSession* session = NULL;
    SvtAv2TileInterRateCache* cache = NULL;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_create(&cfg, &session), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_inter_rate_cache_create(&cache), 0);
    if (session && cache) {
        SvtAv2TileCoeffBlock blocks[12];
        SvtAv2TileBlockMode modes[4];
        uint64_t rate = 0u;
        inter_cache_candidate(0u, 0u, 15u, &cfg, blocks, modes);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_inter_rate_cache_begin(cache, session, 0u), 0);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(session, 0u, 1u, blocks, 12u, modes, 4u, 1, &rate), 0);
        SVT_TEST_CHECK(svt_av2_tile_inter_rate_cache_begin(cache, session, 1u) < 0);
    }
    svt_av2_tile_inter_rate_cache_free(cache);
    svt_av2_tile_rate_free(session);
}

int main(void) {
    test_inter_cache_rejects_partial_superblock();
    for (uint32_t reverse = 0u; reverse < 2u; ++reverse) {
        test_inter_cache_exact(8u, 1u, 0u, 0u, reverse); /* Frozen single-ref 420. */
        test_inter_cache_exact(10u, 1u, 1u, 1u, reverse); /* Fresh adaptive projected 420. */
        test_inter_cache_exact(10u, 3u, 0u, 1u, reverse); /* Frozen projected 444. */
        test_inter_cache_exact(10u, 3u, 2u, 1u, reverse); /* Inherited projected 444. */
    }
    test_two_logical_ref_rate_rollback();
    test_adaptive_ref_rate_rollback(0);
    test_adaptive_ref_rate_rollback(1);
    test_m53_supported_qindex_range();
    test_m52_partition_plan_controls_coeff_order();
    test_m5_p_modes_and_inter_metadata();
    test_frame_header_bit_length();
    test_tile_payload_exact_bytes();
    test_tile_payload_no_space_fails_closed();
    test_block_symbols_independent_roundtrip();
    test_block_rejects_wrong_quantizer();

    SVT_TEST_REPORT("test_tile_writer");
}
