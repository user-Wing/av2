/*
 * T005 unit tests: minimal AV2 frame header writer.
 *
 * The expected bit pattern is derived from the AV2 v1.0.0 spec syntax and
 * semantics (frame_header_info() for single_picture_header_flag = 1 and
 * obu_type = OBU_CLOSED_LOOP_KEY), not from frame_header.c, so the two must
 * agree bit-for-bit.
 *
 * 22-bit header for base_q_idx = 100:
 *   uvlc cur_mfh_id                 0      -> "1"
 *   uvlc seq_header_id_in_frame_header 0   -> "1"
 *   allow_screen_content_tools      0
 *   allow_intrabc                   0
 *   disable_cdf_update              1
 *   uniform_tile_spacing_flag       1
 *   base_q_idx                     100     -> 0110 0100
 *   segmentation_enabled            0
 *   using_qmatrix                   0
 *   delta_q_present                 0
 *   apply_deblocking_filter[0]      0
 *   apply_deblocking_filter[1]      0
 *   tx_mode_select                  0
 *   reduced_tx_set                  0 (2 bits)
 *
 *   bits: 1 1 0 0 1 1 01100100 0 0 0 0 0 0 00
 *   finalize a 4-byte buffer with trailing bits -> trailing 1 then zeros:
 *   byte0 = 1100 1101 = 0xCD
 *   byte1 = 1001 0000 = 0x90
 *   byte2 = 0000 0001 = 0x01  (last data bit region + trailing "1")
 */
#include "bit_writer.h"
#include "entropy_models.h"
#include "frame_header.h"
#include "keyframe.h"

#include "test_util.h"

#include <stdint.h>
#include <string.h>

static void test_header_length_bits(void) {
    uint8_t                 buf[8];
    SvtAv2BitWriter         bw;
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_bit_writer_init(&bw, buf, sizeof(buf));
    svt_av2_frame_header_config_default(&cfg, 64, 64);
    SVT_TEST_CHECK(svt_av2_frame_header_write(&bw, &cfg) == SVT_AV2_BW_OK);
    SVT_TEST_CHECK(svt_av2_bit_writer_bits_written(&bw) ==
                   SVT_AV2_FRAME_HEADER_M1_BITS);
}

static void test_header_defaults(void) {
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, 64, 64);
    SVT_TEST_CHECK(cfg.base_q_idx == 100u); /* non-lossless */
    SVT_TEST_CHECK(cfg.frame_width == 64u);
    SVT_TEST_CHECK(cfg.frame_height == 64u);
}

static void test_header_bytes(void) {
    /* Build a 4-byte-aligned container: the 22-bit header, then byte_alignment,
     * then trailing_bits to make the whole payload 4 bytes. */
    uint8_t                 buf[4] = {0, 0, 0, 0};
    SvtAv2BitWriter         bw;
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_bit_writer_init(&bw, buf, sizeof(buf));
    svt_av2_frame_header_config_default(&cfg, 64, 64);
    SVT_TEST_CHECK(svt_av2_frame_header_write(&bw, &cfg) == SVT_AV2_BW_OK);
    SVT_TEST_CHECK(svt_av2_bit_writer_byte_alignment(&bw) == SVT_AV2_BW_OK);

    /* bits 0..21 hold the header; header bit 21 (tx/reduced 2-bit field) is 0.
     * After byte_alignment the first 3 bytes are: 0xCD 0x90 0x00. */
    SVT_TEST_CHECK(buf[0] == 0xCDu);
    SVT_TEST_CHECK(buf[1] == 0x90u);
    SVT_TEST_CHECK(buf[2] == 0x00u);
}

static void test_header_rejects_zero_quantizer(void) {
    /* base_q_idx = 0 would be lossless, which the M1 subset excludes. */
    uint8_t                 buf[8];
    SvtAv2BitWriter         bw;
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_bit_writer_init(&bw, buf, sizeof(buf));
    svt_av2_frame_header_config_default(&cfg, 64, 64);
    cfg.base_q_idx = 0u;
    SVT_TEST_CHECK(svt_av2_frame_header_write(&bw, &cfg) ==
                   SVT_AV2_BW_ERROR_RANGE);
}

static void test_header_null_args(void) {
    uint8_t          buf[8];
    SvtAv2BitWriter  bw;
    svt_av2_bit_writer_init(&bw, buf, sizeof(buf));
    SVT_TEST_CHECK(svt_av2_frame_header_write(&bw, NULL) ==
                   SVT_AV2_BW_ERROR_NULL);
    SVT_TEST_CHECK(svt_av2_frame_header_write(NULL, NULL) ==
                   SVT_AV2_BW_ERROR_NULL);
}

static void test_m3_quantizer_width(void) {
    uint8_t out[8];
    SvtAv2BitWriter bw;
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, 64u, 64u);
    cfg.bit_depth = 10u;
    cfg.base_q_idx = 148u;
    svt_av2_bit_writer_init(&bw, out, sizeof(out));
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_OK);
    SVT_TEST_CHECK_EQ_INT(svt_av2_bit_writer_bits_written(&bw), SVT_AV2_FRAME_HEADER_M1_BITS + 1u);
    cfg.base_q_idx = 512u;
    svt_av2_bit_writer_init(&bw, out, sizeof(out));
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_ERROR_RANGE);
    cfg.bit_depth = 8u; cfg.chroma_format = 3u; cfg.base_q_idx = 100u;
    svt_av2_bit_writer_init(&bw, out, sizeof(out));
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_ERROR_UNSUPPORTED);
}

static void test_m4_inter_rejects_invalid_context(void) {
    uint8_t out[8];
    SvtAv2BitWriter bw;
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, 64u, 64u);
    cfg.inter_frame = 1u;
    svt_av2_bit_writer_init(&bw, out, sizeof(out));
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_ERROR_UNSUPPORTED);
    SVT_TEST_CHECK_EQ_INT(svt_av2_bit_writer_bits_written(&bw), 0u);
    cfg.multi_frame = 1u;
    cfg.inter_frame = 2u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_ERROR_UNSUPPORTED);
    SVT_TEST_CHECK_EQ_INT(svt_av2_bit_writer_bits_written(&bw), 0u);
}

static uint32_t read_bits(const uint8_t* data, size_t pos, size_t count) {
    uint32_t value = 0u;
    for (size_t i = pos; i < pos + count; ++i)
        value = (value << 1u) | ((data[i >> 3u] >> (7u - (i & 7u))) & 1u);
    return value;
}

static void test_dual_ref_headers(void) {
    uint8_t legacy[16], dual[16];
    SvtAv2BitWriter bw;
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, 64u, 64u);
    SVT_TEST_CHECK_EQ_INT(cfg.dual_ref | cfg.refresh_mask | cfg.reference_slot, 0u);
    cfg.multi_frame = 1u;
    svt_av2_bit_writer_init(&bw, legacy, sizeof(legacy));
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_OK);
    const size_t key_bits = svt_av2_bit_writer_bits_written(&bw);
    cfg.dual_ref = 1u;
    svt_av2_bit_writer_init(&bw, dual, sizeof(dual));
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_OK);
    SVT_TEST_CHECK_EQ_INT(svt_av2_bit_writer_bits_written(&bw), key_bits);
    SVT_TEST_CHECK(memcmp(legacy, dual, (key_bits + 7u) / 8u) == 0);
    svt_av2_bit_writer_init(&bw, dual, sizeof(dual));
    cfg.refresh_mask = 3u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_ERROR_UNSUPPORTED);
    cfg.refresh_mask = 0u;
    cfg.reference_slot = 1u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_ERROR_UNSUPPORTED);
    cfg.reference_slot = 0u;
    cfg.multi_frame = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_ERROR_UNSUPPORTED);
    SVT_TEST_CHECK_EQ_INT(svt_av2_bit_writer_bits_written(&bw), 0u);

    cfg.multi_frame = cfg.inter_frame = 1u;
    cfg.order_hint = 255u;
    cfg.mv_half = 2u;
    cfg.sharp_filter = cfg.entropy_adaptation = 1u;
    cfg.dual_ref = 0u;
    svt_av2_bit_writer_init(&bw, legacy, sizeof(legacy));
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_OK);
    const size_t p_bits = svt_av2_bit_writer_bits_written(&bw);
    cfg.dual_ref = 1u;
    for (uint32_t mask = 0u; mask < 4u; ++mask) for (uint32_t slot = 0u; slot < 2u; ++slot) {
        cfg.refresh_mask = mask;
        cfg.reference_slot = slot;
        svt_av2_bit_writer_init(&bw, dual, sizeof(dual));
        SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_OK);
        SVT_TEST_CHECK_EQ_INT(svt_av2_bit_writer_bits_written(&bw), p_bits + 2u);
        SVT_TEST_CHECK_EQ_INT(read_bits(dual, 0u, 13u), read_bits(legacy, 0u, 13u));
        SVT_TEST_CHECK_EQ_INT(read_bits(dual, 13u, 5u), 31u); /* PRIMARY_REF_NONE */
        SVT_TEST_CHECK_EQ_INT(read_bits(dual, 18u, 2u), mask);
        SVT_TEST_CHECK_EQ_INT(read_bits(dual, 20u, 1u), 1u); /* explicit map */
        SVT_TEST_CHECK_EQ_INT(read_bits(dual, 21u, 3u), 1u); /* one logical reference */
        SVT_TEST_CHECK_EQ_INT(read_bits(dual, 24u, 1u), slot);
        for (size_t i = 23u; i < p_bits; ++i)
            SVT_TEST_CHECK_EQ_INT(read_bits(dual, i + 2u, 1u), read_bits(legacy, i, 1u));
    }
    svt_av2_bit_writer_init(&bw, dual, sizeof(dual));
    cfg.logical_ref_count = 2u;
    for (uint32_t slot = 0; slot < 2u; ++slot) {
        cfg.reference_slot = slot;
        svt_av2_bit_writer_init(&bw, dual, sizeof(dual));
        SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_OK);
        SVT_TEST_CHECK_EQ_INT(read_bits(dual, 24u, 2u), slot ? 2u : 1u);
    }
    for (uint32_t role = SVT_AV2_HIDDEN_ANCHOR; role <= SVT_AV2_LAST_MIDPOINT; ++role) {
        cfg.paired_role = role;
        cfg.refresh_mask = role == SVT_AV2_HIDDEN_ANCHOR ? 2u : 0u;
        svt_av2_bit_writer_init(&bw, dual, sizeof(dual));
        SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_OK);
        SVT_TEST_CHECK_EQ_INT(read_bits(dual, 3u, 1u), role != SVT_AV2_HIDDEN_ANCHOR);
        cfg.refresh_mask ^= 2u;
        svt_av2_bit_writer_init(&bw, dual, sizeof(dual));
        SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_ERROR_RANGE);
    }
    cfg.paired_role = 0u;
    cfg.logical_ref_count = 0u;
    svt_av2_bit_writer_init(&bw, dual, sizeof(dual));
    cfg.refresh_mask = 4u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_ERROR_RANGE);
    cfg.refresh_mask = 1u;
    cfg.reference_slot = 2u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_ERROR_RANGE);
    cfg.reference_slot = 0u;
    cfg.retain_reference = 1u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_ERROR_UNSUPPORTED);
    cfg.retain_reference = cfg.dual_ref = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_ERROR_UNSUPPORTED);
    SVT_TEST_CHECK_EQ_INT(svt_av2_bit_writer_bits_written(&bw), 0u);
}

static void test_half_header(void) {
    uint8_t integer[8], half[8];
    SvtAv2BitWriter bw;
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, 64u, 64u);
    SVT_TEST_CHECK_EQ_INT(cfg.mv_half, 0u);
    cfg.multi_frame = cfg.inter_frame = 1u;
    svt_av2_bit_writer_init(&bw, integer, sizeof(integer));
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_OK);
    const size_t integer_bits = svt_av2_bit_writer_bits_written(&bw);
    cfg.mv_half = 1u;
    svt_av2_bit_writer_init(&bw, half, sizeof(half));
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_OK);
    SVT_TEST_CHECK_EQ_INT(svt_av2_bit_writer_bits_written(&bw), integer_bits + 2u);
    /* AVM v1.0.0 screen / integer / IBC / [precision] / fixed BILINEAR / static CDF. */
    SVT_TEST_CHECK_EQ_INT(read_bits(integer, 23u, 7u), 0x67u);
    SVT_TEST_CHECK_EQ_INT(read_bits(half, 23u, 9u), 0x107u);
    SVT_TEST_CHECK_EQ_INT(read_bits(half, 0u, 23u), read_bits(integer, 0u, 23u));
    SVT_TEST_CHECK_EQ_INT(read_bits(half, 32u, integer_bits - 30u),
        read_bits(integer, 30u, integer_bits - 30u));
    svt_av2_bit_writer_init(&bw, half, sizeof(half));
    cfg.entropy_adaptation = 1u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_OK);
    SVT_TEST_CHECK_EQ_INT(read_bits(half, 23u, 9u), 0x106u);
    svt_av2_bit_writer_init(&bw, half, sizeof(half));
    SVT_TEST_CHECK_EQ_INT(cfg.retain_reference, 0u);
    uint8_t refreshing[8];
    svt_av2_bit_writer_init(&bw, refreshing, sizeof(refreshing));
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_OK);
    const size_t refreshing_bits = svt_av2_bit_writer_bits_written(&bw);
    cfg.retain_reference = 1u;
    svt_av2_bit_writer_init(&bw, half, sizeof(half));
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_OK);
    SVT_TEST_CHECK_EQ_INT(svt_av2_bit_writer_bits_written(&bw), refreshing_bits);
    for (size_t bit = 0; bit < refreshing_bits; ++bit)
        SVT_TEST_CHECK_EQ_INT(read_bits(half, bit, 1u) ^ read_bits(refreshing, bit, 1u), bit == 18u);
    SVT_TEST_CHECK_EQ_INT(read_bits(half, 18u, 1u), 0u);
    cfg.sharp_filter = 1u;
    svt_av2_bit_writer_init(&bw, half, sizeof(half));
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_OK);
    SVT_TEST_CHECK_EQ_INT(read_bits(half, 29u, 2u), 2u);
    cfg.mv_half = 2u;
    svt_av2_bit_writer_init(&bw, half, sizeof(half));
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_OK);
    SVT_TEST_CHECK_EQ_INT(svt_av2_bit_writer_bits_written(&bw) + 1u, refreshing_bits);
    SVT_TEST_CHECK_EQ_INT(read_bits(half, 26u, 1u), 1u); /* quarter precision selector */
    SVT_TEST_CHECK_EQ_INT(read_bits(half, 28u, 2u), 2u);
    cfg.sharp_filter = 0u;
    svt_av2_bit_writer_init(&bw, half, sizeof(half));
    cfg.entropy_adaptation = 0u;
    cfg.inter_frame = cfg.mv_half = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_ERROR_UNSUPPORTED);
    cfg.mv_half = 3u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_frame_header_write(&bw, &cfg), SVT_AV2_BW_ERROR_RANGE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_bit_writer_bits_written(&bw), 0u);
}

static void test_half_class7_and_predictor(void) {
    SvtAv2MotionSyntax motion;
    SvtAv2EntropyEncoder actual, expected;
    SvtAv2BlockView b = {0};
    SvtAv2TileBlockMode m = {1u, -32, -32,0u};
    b.w = b.h = 64u;
    const int init = svt_av2_motion_syntax_init_precision(&motion, 192u, 64u, 1u);
    SVT_TEST_CHECK_EQ_INT(init, 0);
    if (init != 0) return;
    svt_av2_entropy_rate_init(&actual);
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&motion, &actual, &b, &m, 1u), 0);
    b.x = 64u;
    m.mv_x = m.mv_y = 32;
    svt_av2_entropy_rate_init(&actual);
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&motion, &actual, &b, &m, 1u), 0);

    /* From (-32,-32) to (+32,+32): delta (64,64), shell 128, class7,
     * offset zero (seven CDF bits), diagonal pair 64 (no column symbol). */
    /* Both bottom-left and left entries name the already coded 64x64 SB. */
    static const uint16_t inter = 1522u, skip = 4598u, mode[] = {14117u,14341u};
    static const uint16_t drl = 17602u, set = 31579u;
    static const uint16_t class1[] = {19707u,28414u,31240u,31648u,32692u,32717u,32721u};
    static const uint16_t offset[] = {17943u,18934u,18928u,18696u,19044u,20362u,20426u};
    static const uint16_t greater[] = {5663u,4856u};
    svt_av2_entropy_rate_init(&expected);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_symbol(&expected, &inter, 2u, 1u), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_symbol(&expected, &skip, 2u, 1u), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_symbol(&expected, mode, 3u, 2u), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_symbol(&expected, &drl, 2u, 0u), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_symbol(&expected, &set, 2u, 1u), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_symbol(&expected, class1, 8u, 0u), 0);
    for (size_t i = 0u; i < sizeof(offset) / sizeof(offset[0]); ++i)
        SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_symbol(&expected, offset + i, 2u, 0u), 0);
    for (size_t i = 0u; i < sizeof(greater) / sizeof(greater[0]); ++i)
        SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_symbol(&expected, greater + i, 2u, 1u), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_literal(&expected, 63u, 6), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_literal(&expected, 0u, 2), 0);
    SVT_TEST_CHECK_EQ_INT(actual.rate_q12, expected.rate_q12);
    SVT_TEST_CHECK_EQ_INT(actual.rate_symbols, expected.rate_symbols);
    SVT_TEST_CHECK_EQ_INT(actual.rate_literal_bits, expected.rate_literal_bits);
    b.x = 128u;
    m.mv_x = -33;
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&motion, &actual, &b, &m, 1u), SVT_AV2_ENTROPY_ERROR_RANGE);
    svt_av2_motion_syntax_free(&motion);

    /* At an SB row boundary the top-left MVP is aligned to x=0 rather than
     * x=4 for an 8x8 block at x=8. Its signed odd half units must stay exact. */
    const int grid = svt_av2_motion_syntax_init_precision(&motion, 64u, 128u, 1u);
    SVT_TEST_CHECK_EQ_INT(grid, 0);
    if (grid != 0) return;
    SvtAv2MotionCell* above = motion.cells + 15u * motion.cols;
    above[0] = (SvtAv2MotionCell){-1, 3, 1u, 1u, 1u, 1u};
    above[1] = (SvtAv2MotionCell){3, -1, 1u, 1u, 1u, 1u};
    b.x = b.w = b.h = 8u;
    b.y = 64u;
    m.mv_x = -1; m.mv_y = 3;
    svt_av2_entropy_rate_init(&actual);
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&motion, &actual, &b, &m, 1u), 0);
    SVT_TEST_CHECK_EQ_INT(actual.rate_symbols, 4u); /* NEARMV: no difference. */
    SVT_TEST_CHECK_EQ_INT(motion.cells[16u * motion.cols + 2u].is_new, 0u);
    svt_av2_motion_syntax_free(&motion);
}

static void test_half_adaptive_trajectory(void) {
    static SvtAv2EntropyModels written_models, trial_models;
    const SvtAv2TileBlockMode modes[] = {{1u,-32,-32,0u},{1u,32,32,0u},{1u,32,32,0u},{1u,31,29,0u}};
    uint8_t bytes[128];
    uint16_t carry[130];
    for (uint32_t qctx = 0u; qctx < 4u; ++qctx) {
        SvtAv2MotionSyntax written_motion, trial_motion;
        const int a = svt_av2_motion_syntax_init_precision(&written_motion, 256u, 64u, 1u);
        const int b = svt_av2_motion_syntax_init_precision(&trial_motion, 256u, 64u, 1u);
        SVT_TEST_CHECK_EQ_INT(a, 0);
        SVT_TEST_CHECK_EQ_INT(b, 0);
        if (!a && !b) {
            SvtAv2EntropyEncoder formal, trial;
            SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_models_init(&written_models, qctx), 0);
            trial_models = written_models;
            svt_av2_entropy_encoder_init(&formal, bytes, sizeof(bytes), carry, 130u);
            svt_av2_entropy_rate_init(&trial);
            formal.models = &written_models;
            trial.models = &trial_models;
            for (uint32_t sb = 0u; sb < 4u; ++sb) {
                SvtAv2BlockView view = {0};
                view.x = sb * 64u; view.w = view.h = 64u;
                SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&written_motion, &formal, &view, modes + sb, 1u), 0);
                SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&trial_motion, &trial, &view, modes + sb, 1u), 0);
                SVT_TEST_CHECK(memcmp(&written_models, &trial_models, sizeof(written_models)) == 0);
            }
            /* Classes 6/2, class 7, and offset bit6 must adapt separate rows.
             * The unchanged NEAR in SB2 consumes no MV difference models. */
            SVT_TEST_CHECK_EQ_INT(written_models.cdf[SVT_AV2_MODEL_MV_HALF_CLASS0][7], 2u);
            SVT_TEST_CHECK_EQ_INT(written_models.cdf[SVT_AV2_MODEL_MV_HALF_CLASS1][8], 1u);
            SVT_TEST_CHECK_EQ_INT(written_models.cdf[SVT_AV2_MODEL_MV_HALF_BIT6][2], 1u);
            SVT_TEST_CHECK_EQ_INT(written_models.cdf[SVT_AV2_MODEL_MV_CLASS][7], 0u);
        }
        if (!a) svt_av2_motion_syntax_free(&written_motion);
        if (!b) svt_av2_motion_syntax_free(&trial_motion);
    }
}

static void test_half_rate_packet(void) {
    SvtAv2FrameHeaderConfig cfg;
    SvtAv2TileRateSession* session = NULL;
    SvtAv2TileCoeffBlock blocks[6] = {0};
    static const int32_t zero_coeff[1024] = {0};
    const SvtAv2TileBlockMode modes[] = {{1u,-32,-32,0u}, {1u,32,32,0u}};
    uint8_t packet[128];
    size_t bytes = 0u;
    svt_av2_frame_header_config_default(&cfg, 128u, 64u);
    cfg.mv_half = cfg.multi_frame = cfg.inter_frame = 1u;
    cfg.reduced_tx_set = 2u;
    for (uint32_t sb = 0u; sb < 2u; ++sb) for (uint32_t p = 0u; p < 3u; ++p) {
        SvtAv2TileCoeffBlock* b = blocks + sb * 3u + p;
        b->plane = p;
        b->width = b->height = p ? 32u : 64u;
        b->x = b->width * sb;
        b->qcoeff = zero_coeff;
    }
    for (uint32_t adapt = 0u; adapt < 2u; ++adapt) {
        cfg.entropy_adaptation = adapt;
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_create(&cfg, &session), 0);
        if (!session) return;
        uint64_t first = 0u, second = 0u, retry = 0u, full = 0u;
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(session, 0u, 0u, blocks, 3u, modes, 1u, 1, &first), 0);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(session, 1u, 0u, blocks + 3u, 3u, modes + 1u, 1u, 0, &second), 0);
        ++blocks[4].x; /* Fail after motion was consumed, then retry. */
        SVT_TEST_CHECK(svt_av2_tile_rate_trial(session, 1u, 0u, blocks + 3u, 3u, modes + 1u, 1u, 1, &retry) != 0);
        --blocks[4].x;
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_trial(session, 1u, 0u, blocks + 3u, 3u, modes + 1u, 1u, 1, &retry), 0);
        SVT_TEST_CHECK_EQ_INT(retry, second);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_frame(&cfg, blocks, 6u, modes, 2u, &full), 0);
        SVT_TEST_CHECK_EQ_INT(full, first + second);
        svt_av2_tile_rate_free(session);
        SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_tools_half_entropy(128u, 64u, 8u, 1u, 100u,
            1u, blocks, 6u, modes, 2u, 1u, 0u, NULL, 0u, adapt, packet, sizeof(packet), &bytes), 0);
        SVT_TEST_CHECK(bytes > 0u);
    }
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_tools_half(128u, 64u, 8u, 1u, 100u,
        0u, blocks, 6u, modes, 2u, 1u, 0u, NULL, 0u, packet, sizeof(packet), &bytes),
        SVT_AV2_BW_ERROR_UNSUPPORTED);
    SVT_TEST_CHECK_EQ_INT(bytes, 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_tools_half(128u, 64u, 8u, 1u, 100u,
        1u, blocks, 6u, modes, 2u, 1u, 0u, NULL, 0u, packet, sizeof(packet), &bytes), 0);
    SVT_TEST_CHECK(bytes > 0u);
    SVT_TEST_CHECK(svt_av2_encode_packet_tools(128u, 64u, 8u, 1u, 100u,
        1u, blocks, 6u, modes, 2u, 1u, 0u, NULL, 0u, packet, sizeof(packet), &bytes) != 0);
}

static void test_dual_ref_packets(void) {
    static const int32_t zero_coeff[1024] = {0};
    SvtAv2TileCoeffBlock blocks[3] = {0};
    const SvtAv2TileBlockMode mode = {1u, 1, -3,0u};
    uint8_t legacy[128], dual[128], retry[128];
    size_t legacy_bytes = 0u, dual_bytes = 0u, retry_bytes = 0u;
    for (uint32_t p = 0u; p < 3u; ++p) {
        blocks[p].plane = p;
        blocks[p].width = blocks[p].height = p ? 32u : 64u;
        blocks[p].qcoeff = zero_coeff;
    }
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_low_delay_packet(64u, 64u, 8u, 1u, 100u,
        0u, blocks, 3u, NULL, 0u, legacy, sizeof(legacy), &legacy_bytes), SVT_AV2_BW_OK);
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_dual_ref(64u, 64u, 8u, 1u, 100u,
        0u, blocks, 3u, NULL, 0u, 0u, NULL, 0u, 0u, 0u, 0u, 0u, 0u,
        dual, sizeof(dual), &dual_bytes), SVT_AV2_BW_OK);
    SVT_TEST_CHECK_EQ_INT(dual_bytes, legacy_bytes);
    /* The tiny test packet uses one-byte OBU lengths. The KEY, including all
     * alignment and its coded payload, must be identical. Only sequence bit 81
     * (count=1) changes, after the two-byte TD and the two-byte sequence prefix. */
    const size_t seq_count_lsb = 4u * 8u + 81u;
    SVT_TEST_CHECK(dual_bytes * 8u > seq_count_lsb);
    if (dual_bytes == legacy_bytes) for (size_t i = 0u; i < dual_bytes * 8u; ++i)
        SVT_TEST_CHECK_EQ_INT(read_bits(dual, i, 1u), read_bits(legacy, i, 1u) ^ (uint32_t)(i == seq_count_lsb));
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_dual_ref(64u, 64u, 8u, 1u, 100u,
        0u, blocks, 3u, NULL, 0u, 0u, NULL, 0u, 0u, 0u, 0u, 3u, 0u,
        retry, sizeof(retry), &retry_bytes), SVT_AV2_BW_ERROR_UNSUPPORTED);
    SVT_TEST_CHECK_EQ_INT(retry_bytes, 0u);

    /* P at order-hint wrap: outer is_first_tile_group=1 plus the two-slot
     * frame header; no OBU beyond the delimiter and the regular tile group. */
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_dual_ref(64u, 64u, 8u, 1u, 100u,
        256u, blocks, 3u, &mode, 1u, 0u, NULL, 0u, 1u, 2u, 1u, 2u, 1u,
        dual, sizeof(dual), &dual_bytes), SVT_AV2_BW_OK);
    SVT_TEST_CHECK(dual_bytes > 8u);
    if (dual_bytes > 8u) {
        const uint8_t* hdr = dual + 4u;
        SVT_TEST_CHECK_EQ_INT(read_bits(hdr, 0u, 1u), 1u);
        SVT_TEST_CHECK_EQ_INT(read_bits(hdr, 6u, 8u), 0u); /* 256 & 255 */
        SVT_TEST_CHECK_EQ_INT(read_bits(hdr, 14u, 5u), 31u); /* PRIMARY_REF_NONE */
        SVT_TEST_CHECK_EQ_INT(read_bits(hdr, 19u, 2u), 2u); /* refresh slot 1 only */
        SVT_TEST_CHECK_EQ_INT(read_bits(hdr, 21u, 5u), 19u); /* explicit 1, count 001, physical slot 1 */
    }
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_dual_ref(64u, 64u, 8u, 1u, 100u,
        256u, blocks, 3u, &mode, 1u, 0u, NULL, 0u, 1u, 2u, 1u, 2u, 1u,
        retry, 4u, &retry_bytes), SVT_AV2_BW_ERROR_NO_SPACE);
    SVT_TEST_CHECK_EQ_INT(retry_bytes, 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_dual_ref(64u, 64u, 8u, 1u, 100u,
        256u, blocks, 3u, &mode, 1u, 0u, NULL, 0u, 1u, 2u, 1u, 2u, 1u,
        retry, sizeof(retry), &retry_bytes), SVT_AV2_BW_OK);
    SVT_TEST_CHECK_EQ_INT(retry_bytes, dual_bytes);
    if (retry_bytes == dual_bytes) SVT_TEST_CHECK(memcmp(dual, retry, dual_bytes) == 0);
}

static void test_two_logical_ref_packets(void) {
    static const int32_t zero[1024] = {0};
    SvtAv2TileCoeffBlock blocks[3] = {0};
    SvtAv2TileBlockMode mode = {1u, 0, 0, 1u};
    for (uint32_t p = 0u; p < 3u; ++p) {
        blocks[p].plane = p;
        blocks[p].width = blocks[p].height = p ? 32u : 64u;
        blocks[p].qcoeff = zero;
    }
    uint8_t out[128], retry[128];
    size_t bytes = 0u, rebuilt = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_two_refs(64u, 64u, 8u, 1u, 100u,
        256u, blocks, 3u, &mode, 1u, 0u, NULL, 0u, 2u, 1u, 3u, out, sizeof(out), &bytes), 0);
    SVT_TEST_CHECK(bytes > 8u);
    if (bytes > 8u) {
        const uint8_t* h = out + 4u;
        SVT_TEST_CHECK_EQ_INT(read_bits(h, 6u, 8u), 0u); /* wrapped order hint */
        SVT_TEST_CHECK_EQ_INT(read_bits(h, 14u, 5u), 31u); /* PRIMARY_REF_NONE */
        SVT_TEST_CHECK_EQ_INT(read_bits(h, 19u, 2u), 3u);
        SVT_TEST_CHECK_EQ_INT(read_bits(h, 21u, 6u), 41u); /* 1 010 0 1 */
    }
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_two_refs(64u, 64u, 8u, 1u, 100u,
        256u, blocks, 3u, &mode, 1u, 0u, NULL, 0u, 2u, 1u, 3u, retry, 4u, &rebuilt), SVT_AV2_BW_ERROR_NO_SPACE);
    SVT_TEST_CHECK_EQ_INT(rebuilt, 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_two_refs(64u, 64u, 8u, 1u, 100u,
        256u, blocks, 3u, &mode, 1u, 0u, NULL, 0u, 2u, 1u, 3u, retry, sizeof(retry), &rebuilt), 0);
    SVT_TEST_CHECK(rebuilt == bytes && memcmp(retry, out, bytes) == 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_two_refs_entropy(64u, 64u, 8u, 1u, 100u,
        256u, blocks, 3u, &mode, 1u, 0u, NULL, 0u, 0u, 2u, 1u, 3u, retry, sizeof(retry), &rebuilt), 0);
    SVT_TEST_CHECK(rebuilt == bytes && memcmp(retry, out, bytes) == 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_two_refs_entropy(64u, 64u, 8u, 1u, 100u,
        256u, blocks, 3u, &mode, 1u, 0u, NULL, 0u, 1u, 2u, 1u, 3u, retry, sizeof(retry), &rebuilt), 0);
    if (rebuilt > 8u && bytes > 8u) {
        SVT_TEST_CHECK_EQ_INT(read_bits(out + 4u, 34u, 1u), 1u); /* static */
        SVT_TEST_CHECK_EQ_INT(read_bits(retry + 4u, 34u, 1u), 0u); /* adaptive */
        uint8_t again[128];
        size_t retry_bytes = 7u;
        SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_two_refs_entropy(64u, 64u, 8u, 1u, 100u,
            256u, blocks, 3u, &mode, 1u, 0u, NULL, 0u, 1u, 2u, 1u, 3u, again, 4u, &retry_bytes),
            SVT_AV2_BW_ERROR_NO_SPACE);
        SVT_TEST_CHECK_EQ_INT(retry_bytes, 0u);
        SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_two_refs_entropy(64u, 64u, 8u, 1u, 100u,
            256u, blocks, 3u, &mode, 1u, 0u, NULL, 0u, 1u, 2u, 1u, 3u, again, sizeof(again), &retry_bytes), 0);
        SVT_TEST_CHECK(retry_bytes == rebuilt && memcmp(again, retry, rebuilt) == 0);
    }
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_two_refs_entropy(64u, 64u, 8u, 1u, 100u,
        256u, blocks, 3u, &mode, 1u, 0u, NULL, 0u, 2u, 2u, 1u, 3u, retry, sizeof(retry), &rebuilt),
        SVT_AV2_BW_ERROR_RANGE);
    SVT_TEST_CHECK_EQ_INT(rebuilt, 0u);
    mode.reference_rank = 2u;
    SVT_TEST_CHECK(svt_av2_encode_packet_two_refs(64u, 64u, 8u, 1u, 100u,
        4u, blocks, 3u, &mode, 1u, 0u, NULL, 0u, 2u, 1u, 3u, out, sizeof(out), &bytes) != 0);
    SVT_TEST_CHECK_EQ_INT(bytes, 0u);
    mode.reference_rank = 0u;
    mode.mv_x = 1;
    SVT_TEST_CHECK(svt_av2_encode_packet_two_refs(64u, 64u, 8u, 1u, 100u,
        4u, blocks, 3u, &mode, 1u, 0u, NULL, 0u, 2u, 1u, 3u, out, sizeof(out), &bytes) != 0);
    SVT_TEST_CHECK_EQ_INT(bytes, 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_encode_packet_two_refs(64u, 64u, 8u, 1u, 100u,
        0u, blocks, 3u, NULL, 0u, 0u, NULL, 0u, 0u, 0u, 0u, out, sizeof(out), &bytes), SVT_AV2_BW_ERROR_UNSUPPORTED);
}

int main(void) {
    test_two_logical_ref_packets();
    test_dual_ref_headers();
    test_dual_ref_packets();
    test_half_header();
    test_half_class7_and_predictor();
    test_half_adaptive_trajectory();
    test_half_rate_packet();
    test_m4_inter_rejects_invalid_context();
    test_m3_quantizer_width();
    test_header_length_bits();
    test_header_defaults();
    test_header_bytes();
    test_header_rejects_zero_quantizer();
    test_header_null_args();

    SVT_TEST_REPORT("test_frame_header");
}
