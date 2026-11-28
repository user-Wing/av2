#include "entropy_models.h"
#include "motion_syntax.h"
#include "tile_writer.h"
#include "test_util.h"
#include <string.h>

/* Independent scalar expression of the fixed inverse-CDF update. The external
 * extractor/update oracle also compiles the actual fixed AVM prob.h. */
static void reference_update(uint16_t* cdf, uint32_t n, uint32_t s) {
    const uint32_t interval = cdf[n] < 16u ? 0u : cdf[n] < 32u ? 1u : 2u;
    const uint32_t divisor = 1u << (cdf[n + 1u + interval] + 2u);
    for (uint32_t i = 0; i < n - 1u; ++i) {
        if (i >= s) cdf[i] = (uint16_t)(cdf[i] - cdf[i] / divisor);
        else cdf[i] = (uint16_t)(cdf[i] + (32768u - cdf[i]) / divisor);
    }
    if (cdf[n] != 32u) ++cdf[n];
}

static void updates(void) {
    uint32_t equal = 0u;
    for (uint32_t q = 0u; q < 4u; ++q) {
        for (uint32_t id = 0u; id < SVT_AV2_MODEL_COUNT; ++id) {
            const uint32_t n = svt_av2_model_nsyms[id];
            uint16_t actual[12], expected[12];
            memcpy(actual, svt_av2_model_defaults[q][id], sizeof(actual));
            memcpy(expected, actual, sizeof(expected));
            for (uint32_t step = 0u; step < 512u; ++step) {
                const uint32_t s = step < 300u ? n / 2u : (step * 17u + id) % n;
                reference_update(expected, n, s);
                SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_model_update(actual, n, s), 0);
                SVT_TEST_CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
                for (uint32_t j = 1u; j + 1u < n; ++j) equal += actual[j - 1u] == actual[j];
            }
        }
    }
    SVT_TEST_CHECK(equal > 0u);
}

static void decay(void) {
    static SvtAv2EntropyModels models, expected;
    for (uint32_t q = 0u; q < 4u; ++q) {
        SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_models_init(&models, q), 0);
        svt_av2_entropy_models_decay(&models);
        SVT_TEST_CHECK(memcmp(models.cdf, svt_av2_model_defaults[q], sizeof(models.cdf)) == 0);

        /* Train every semantic row, then exercise every legal count in each
         * row. The whole-bank comparison includes probabilities, terminal,
         * training offsets and the padding after smaller alphabets. */
        for (uint32_t id = 0u; id < SVT_AV2_MODEL_COUNT; ++id) {
            const uint32_t n = svt_av2_model_nsyms[id];
            for (uint32_t i = 0u; i < 32u; ++i)
                SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_model_update(models.cdf[id], n, (id + i) % n), 0);
        }
        for (uint32_t step = 0u; step <= 32u; ++step) {
            expected = models;
            for (uint32_t id = 0u; id < SVT_AV2_MODEL_COUNT; ++id) {
                const uint32_t n = svt_av2_model_nsyms[id];
                const uint32_t count = (step + id) % 33u;
                models.cdf[id][n] = (uint16_t)count;
                expected.cdf[id][n] = (uint16_t)(count - (count + 3u) / 4u);
            }
            svt_av2_entropy_models_decay(&models);
            SVT_TEST_CHECK(memcmp(&models, &expected, sizeof(models)) == 0);
        }
    }
}

static void isolation(void) {
    SvtAv2EntropyModels a, b, before;
    SvtAv2EntropyEncoder ea, eb;
    uint8_t bytes_a[65536], bytes_b[65536];
    uint16_t carry_a[65538], carry_b[65538];
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_models_init(&a, 1u), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_models_init(&b, 1u), 0);
    before = b;
    svt_av2_entropy_encoder_init(&ea, bytes_a, sizeof(bytes_a), carry_a, 65538u);
    svt_av2_entropy_encoder_init(&eb, bytes_b, sizeof(bytes_b), carry_b, 65538u);
    ea.models = &a; eb.models = &b;
    const uint32_t id = SVT_AV2_MODEL_Y_IDX;
    for (uint32_t s = 0u; s < 1200u; ++s) {
        SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_model(&ea, id, NULL, 8u, s < 1100u ? 4u : s % 8u), 0);
    }
    SVT_TEST_CHECK(memcmp(&b, &before, sizeof(b)) == 0);
    /* A separate equal-initialized semantic model was not modified. */
    SVT_TEST_CHECK(memcmp(a.cdf[SVT_AV2_MODEL_INTER_TX], before.cdf[SVT_AV2_MODEL_INTER_TX], 24u) == 0);
    for (uint32_t s = 0u; s < 1200u; ++s)
        SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_model(&eb, id, NULL, 8u, s < 1100u ? 4u : s % 8u), 0);
    uint8_t *out_a = NULL, *out_b = NULL;
    size_t size_a = 0, size_b = 0;
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encoder_done(&ea, &out_a, &size_a), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encoder_done(&eb, &out_b, &size_b), 0);
    SVT_TEST_CHECK(size_a == size_b && memcmp(out_a, out_b, size_a) == 0);
    SVT_TEST_CHECK(memcmp(&a, &b, sizeof(a)) == 0);
    before = a;
    SVT_TEST_CHECK(svt_av2_entropy_encode_model(&ea, SVT_AV2_MODEL_COUNT, NULL, 2u, 0u) != 0);
    SVT_TEST_CHECK(svt_av2_entropy_encode_model(&ea, id, NULL, 7u, 0u) != 0);
    SVT_TEST_CHECK(memcmp(&a, &before, sizeof(a)) == 0);
    ea.error = 1;
    SVT_TEST_CHECK(svt_av2_entropy_encode_model(&ea, id, NULL, 8u, 0u) != 0);
    SVT_TEST_CHECK(memcmp(&a, &before, sizeof(a)) == 0);
    svt_av2_entropy_rate_init(&ea); ea.models = &a;
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_model(&ea, id, NULL, 8u, 0u), 0);
    SVT_TEST_CHECK(memcmp(&a, &before, sizeof(a)) != 0);
    SVT_TEST_CHECK(ea.rate_q12 > 0u && ea.rate_symbols == 1u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_models_init(&a, 1u), 0);
    SVT_TEST_CHECK(memcmp(a.cdf, svt_av2_model_defaults[1], sizeof(a.cdf)) == 0);
}

static void fresh_writer(uint32_t depth, uint32_t format, uint32_t inter) {
    SvtAv2FrameHeaderConfig cfg;
    svt_av2_frame_header_config_default(&cfg, 128u, 64u);
    cfg.bit_depth = depth; cfg.chroma_format = format;
    cfg.base_q_idx = depth == 10u ? 148u : 100u;
    cfg.m5_intra_modes = 1u; cfg.inter_frame = inter; cfg.multi_frame = 1u;
    cfg.reduced_tx_set = 2u; cfg.entropy_adaptation = 1u;
    int32_t coeffs[6][1024] = {{0}};
    SvtAv2TileCoeffBlock blocks[6];
    SvtAv2TileBlockMode modes[2] = {{1u, 3, -2,0u, 0, 0}, {0u, 0, 0,0u, 0, 0}};
    for (uint32_t sb = 0u; sb < 2u; ++sb) {
        for (uint32_t p = 0u; p < 3u; ++p) {
            const uint32_t i = sb * 3u + p;
            const uint32_t shift = p && format == 1u ? 1u : 0u;
            blocks[i].plane = p; blocks[i].x = (sb * 64u) >> shift;
            blocks[i].y = 0u; blocks[i].width = blocks[i].height = 64u >> shift;
            blocks[i].intra_mode = (uint8_t)sb; blocks[i].qcoeff = coeffs[i];
            for (uint32_t c = 0u; c < 100u; ++c) coeffs[i][c] = (int32_t)((c + i) % 17u) - 8;
        }
    }
    int32_t saved[6][1024]; memcpy(saved, coeffs, sizeof(saved));
    uint8_t clean[32768], retry[32768], tiny[16];
    memset(tiny, 0xa7, sizeof(tiny));
    size_t first = 0u, second = 0u, need = 0u;
    SvtAv2TileRateSession* session = NULL;
    uint64_t rate = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_create(&cfg, &session), 0);
    SVT_TEST_CHECK(session != NULL);
    svt_av2_tile_rate_free(session);
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_frame(&cfg, blocks, 6u,
        inter ? modes : NULL, inter ? 2u : 0u, &rate), 0);
    SVT_TEST_CHECK(rate > 0u);
    SVT_TEST_CHECK(svt_av2_tile_writer_write_closed_loop_key(&cfg, tiny, sizeof(tiny), &need) != 0);
    if (inter) {
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_writer_write_inter_coeffs(&cfg, blocks, 6u, modes, 2u, clean, sizeof(clean), &first), 0);
        SVT_TEST_CHECK(svt_av2_tile_writer_write_inter_coeffs(&cfg, blocks, 6u, modes, 2u, tiny, sizeof(tiny), &need) != 0);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_writer_write_inter_coeffs(&cfg, blocks, 6u, modes, 2u, retry, sizeof(retry), &second), 0);
    } else {
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_writer_write_closed_loop_key_coeffs(&cfg, blocks, 6u, clean, sizeof(clean), &first, NULL), 0);
        SVT_TEST_CHECK(svt_av2_tile_writer_write_closed_loop_key_coeffs(&cfg, blocks, 6u, tiny, sizeof(tiny), &need, NULL) != 0);
        SVT_TEST_CHECK_EQ_INT(svt_av2_tile_writer_write_closed_loop_key_coeffs(&cfg, blocks, 6u, retry, sizeof(retry), &second, NULL), 0);
    }
    SVT_TEST_CHECK(first > sizeof(tiny) && first == second && memcmp(clean, retry, first) == 0);
    for (size_t i = 0u; i < sizeof(tiny); ++i) SVT_TEST_CHECK(tiny[i] == 0xa7);
    SVT_TEST_CHECK(memcmp(coeffs, saved, sizeof(saved)) == 0);
}

/* Hand-derived symbol trace for fixed AVM QTR_PEL (5), all skip_txfm.
 * The moving 64x64 leaves exercise shell127, shell255 (class7 still in set0),
 * shell256 (class8/set1), NEAR reuse, and one quarter unit without rounding. */
static int quarter_bit(SvtAv2EntropyEncoder* s, uint32_t id, uint16_t p, uint32_t value) {
    return svt_av2_entropy_encode_model(s, id, &p, 2u, value);
}

static void quarter_expected(SvtAv2EntropyEncoder* e, uint32_t sb) {
    static const uint16_t mode[][2] = {{10043,11100},{21561,21758},{14117,14341}};
    static const uint16_t drl[] = {15721,21115,17602};
    static const uint16_t class0[] = {7497,17301,23848,27438,29395,30879,32003};
    static const uint16_t class1[] = {18469,27427,31562,32652,32724,32728,32732};
    static const uint16_t offset[] = {17943,18934,18928,18696,19044,20362,20426,22563};
    static const uint32_t count[] = {6u,7u,8u,0u,0u};
    static const uint32_t ids[] = {SVT_AV2_MODEL_MV_BITS,SVT_AV2_MODEL_MV_BITS + 1u,
        SVT_AV2_MODEL_MV_BITS + 2u,SVT_AV2_MODEL_MV_BITS + 3u,SVT_AV2_MODEL_MV_BITS + 4u,
        SVT_AV2_MODEL_MV_BITS + 5u,SVT_AV2_MODEL_MV_HALF_BIT6,SVT_AV2_MODEL_MV_QUARTER_BIT7};
    const uint32_t ctx = sb == 0u ? 0u : sb == 4u ? 1u : 3u;
    const uint32_t row = sb == 0u ? 0u : sb == 4u ? 1u : 2u;
    SVT_TEST_CHECK_EQ_INT(quarter_bit(e, SVT_AV2_MODEL_INTRA_INTER, 1522u, 1u), 0);
    SVT_TEST_CHECK_EQ_INT(quarter_bit(e, SVT_AV2_MODEL_SKIP + (sb ? 2u : 0u), sb ? 4598u : 25865u, 1u), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_model(e, SVT_AV2_MODEL_INTER_MODE + ctx, mode[row], 3u,
        sb == 3u ? 0u : 2u), 0);
    SVT_TEST_CHECK_EQ_INT(quarter_bit(e, SVT_AV2_MODEL_DRL + ctx, drl[row], 0u), 0);
    if (sb == 3u) return; /* NEAR has no MVD. */
    SVT_TEST_CHECK_EQ_INT(quarter_bit(e, SVT_AV2_MODEL_MV_SET, 31579u, sb == 2u), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_model(e,
        sb == 2u ? SVT_AV2_MODEL_MV_QUARTER_CLASS1 : SVT_AV2_MODEL_MV_QUARTER_CLASS0,
        sb == 2u ? class1 : class0, 8u, sb < 2u ? count[sb] : 0u), 0);
    for (uint32_t i = 0; i < count[sb]; ++i)
        SVT_TEST_CHECK_EQ_INT(quarter_bit(e, ids[i], offset[i], sb < 2u), 0);
    if (sb == 4u) {
        SVT_TEST_CHECK_EQ_INT(quarter_bit(e, SVT_AV2_MODEL_MV_LOW, 14587u, 1u), 0);
        SVT_TEST_CHECK_EQ_INT(quarter_bit(e, SVT_AV2_MODEL_MV_COL, 13445u, 1u), 0);
        SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_literal(e, 0u, 1), 0); /* x positive; no y sign. */
        return;
    }
    SVT_TEST_CHECK_EQ_INT(quarter_bit(e, SVT_AV2_MODEL_MV_GREATER, 5663u, 1u), 0);
    SVT_TEST_CHECK_EQ_INT(quarter_bit(e, SVT_AV2_MODEL_MV_GREATER + 1u, 4856u, 1u), 0);
    const int bits = sb == 0u ? 6 : 7;
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_literal(e, (1u << bits) - 1u, bits), 0);
    if (sb != 2u) /* In odd shells the larger column owns the unpaired unit. */
        SVT_TEST_CHECK_EQ_INT(quarter_bit(e, SVT_AV2_MODEL_MV_COL + 3u, 12888u, 1u), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encode_literal(e, sb == 1u ? 0u : 3u, 2), 0);
}

static void quarter_motion(void) {
    static SvtAv2EntropyModels actual_models, expected_models, rate_models;
    static const SvtAv2TileBlockMode moves[5] = {{1u,-64,-63,0u, 0, 0},{1u,64,64,0u, 0, 0},{1u,-64,-64,0u, 0, 0},
        {1u,-64,-64,0u, 0, 0},{1u,-63,-64,0u, 0, 0}};
    static const SvtAv2TileBlockMode invalid[] = {{1u,65,0,0u, 0, 0},{1u,0,-65,0u, 0, 0}};
    SvtAv2MotionSyntax reject;
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_syntax_init_precision(&reject, 320u, 64u, 3u), SVT_AV2_ENTROPY_ERROR_RANGE);
    SVT_TEST_CHECK(reject.cells == NULL);
    const int32_t limit[2] = {16,32};
    for (uint32_t precision = 0; precision < 2u; ++precision) {
        SvtAv2MotionSyntax old;
        const int rc = svt_av2_motion_syntax_init_precision(&old, 64u, 64u, precision);
        SVT_TEST_CHECK_EQ_INT(rc, 0);
        if (!rc) {
            SvtAv2BlockView view = {0}; view.w = view.h = 64u;
            const SvtAv2TileBlockMode out_of_range = {1u, 0, limit[precision] + 1,0u, 0, 0};
            SvtAv2EntropyEncoder e; svt_av2_entropy_rate_init(&e);
            SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&old, &e, &view, &out_of_range, 1u), SVT_AV2_ENTROPY_ERROR_RANGE);
            SVT_TEST_CHECK(e.rate_symbols == 0u && !old.cells[0].valid);
            svt_av2_motion_syntax_free(&old);
        }
    }
    for (uint32_t adapt = 0; adapt <= 1u; ++adapt) {
        SvtAv2MotionSyntax wire, estimator;
        const int a = svt_av2_motion_syntax_init_precision(&wire, 320u, 64u, 2u);
        const int b = svt_av2_motion_syntax_init_precision(&estimator, 320u, 64u, 2u);
        SVT_TEST_CHECK_EQ_INT(a, 0); SVT_TEST_CHECK_EQ_INT(b, 0);
        if (!a && !b) {
            uint8_t bytes[256], ref_bytes[256], *actual_out = NULL, *expected_out = NULL;
            uint16_t carry[258], ref_carry[258];
            size_t actual_len = 0, expected_len = 0;
            SvtAv2EntropyEncoder actual, expected, rate;
            svt_av2_entropy_encoder_init(&actual, bytes, sizeof(bytes), carry, 258u);
            svt_av2_entropy_encoder_init(&expected, ref_bytes, sizeof(ref_bytes), ref_carry, 258u);
            svt_av2_entropy_rate_init(&rate);
            if (adapt) {
                SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_models_init(&actual_models, 2u), 0);
                expected_models = rate_models = actual_models;
                actual.models = &actual_models;
                expected.models = &expected_models;
                rate.models = &rate_models;
            }
            for (uint32_t i = 0; i < 5u; ++i) {
                SvtAv2BlockView view = {0}; view.x = i * 64u; view.w = view.h = 64u;
                if (i == 3u) {
                    const uint64_t rate_before = rate.rate_q12;
                    for (uint32_t j = 0; j < 2u; ++j)
                        SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&estimator, &rate, &view, invalid + j, 1u),
                            SVT_AV2_ENTROPY_ERROR_RANGE);
                    SVT_TEST_CHECK(rate.rate_q12 == rate_before && !estimator.cells[48].valid);
                }
                quarter_expected(&expected, i);
                SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&wire, &actual, &view, moves + i, 1u), 0);
                SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&estimator, &rate, &view, moves + i, 1u), 0);
                if (adapt) {
                    SVT_TEST_CHECK(memcmp(&actual_models, &expected_models, sizeof(actual_models)) == 0);
                    SVT_TEST_CHECK(memcmp(&actual_models, &rate_models, sizeof(actual_models)) == 0);
                }
                SVT_TEST_CHECK_EQ_INT(wire.cells[i * 16u].is_new, i != 3u);
                SVT_TEST_CHECK_EQ_INT(wire.cells[i * 16u].mv_x, moves[i].mv_x);
                SVT_TEST_CHECK_EQ_INT(wire.cells[i * 16u].mv_y, moves[i].mv_y);
            }
            SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encoder_done(&actual, &actual_out, &actual_len), 0);
            SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_encoder_done(&expected, &expected_out, &expected_len), 0);
            SVT_TEST_CHECK(actual_len > 0 && actual_len == expected_len && memcmp(actual_out, expected_out, actual_len) == 0);
            if (adapt) {
                SVT_TEST_CHECK_EQ_INT(actual_models.cdf[SVT_AV2_MODEL_MV_QUARTER_CLASS0][8], 3u);
                SVT_TEST_CHECK_EQ_INT(actual_models.cdf[SVT_AV2_MODEL_MV_QUARTER_CLASS1][8], 1u);
                SVT_TEST_CHECK_EQ_INT(actual_models.cdf[SVT_AV2_MODEL_MV_HALF_BIT6][2], 2u);
                SVT_TEST_CHECK_EQ_INT(actual_models.cdf[SVT_AV2_MODEL_MV_QUARTER_BIT7][2], 1u);
            }
        }
        if (!a) svt_av2_motion_syntax_free(&wire);
        if (!b) svt_av2_motion_syntax_free(&estimator);
    }
}

static void wide_quarter_shell(void) {
    SvtAv2MotionSyntax motion;
    SvtAv2EntropyModels models;
    SvtAv2EntropyEncoder rate;
    svt_av2_entropy_rate_init(&rate);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_models_init(&models, 2u), 0);
    rate.models = &models;
    const int status = svt_av2_motion_syntax_init_precision(&motion, 128u, 64u, 2u);
    SVT_TEST_CHECK_EQ_INT(status, 0);
    if (status) return;
    motion.wide_motion = 2u;
    for (unsigned i = 0; i < 2; ++i) {
        SvtAv2BlockView view = {0};
        view.x = i*64u;
        view.w = view.h = 64u;
        const int mv = i ? -252 : 252;
        const SvtAv2TileBlockMode mode = {1u,mv,mv,0u,0,0};
        SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&motion, &rate, &view, &mode, 1u), 0);
    }
    SVT_TEST_CHECK_EQ_INT(models.cdf[SVT_AV2_MODEL_MV_QUARTER_BIT8][2], 1u);
    svt_av2_motion_syntax_free(&motion);
}

static void independent_ref_contexts(void) {
    static const uint16_t initial[3][12] = {
        {6299, 0, 0, 2, 3, 4}, {19137, 0, 0, 2, 2, 3}, {30169, 0, 0, 2, 3, 4}
    };
    for (uint32_t q = 0u; q < 4u; ++q) {
        for (uint32_t ctx = 0u; ctx < 3u; ++ctx) {
            const uint32_t id = SVT_AV2_MODEL_SINGLE_REF + ctx;
            SVT_TEST_CHECK_EQ_INT(svt_av2_model_nsyms[id], 2u);
            SVT_TEST_CHECK(memcmp(svt_av2_model_defaults[q][id], initial[ctx], sizeof(initial[ctx])) == 0);
        }
    }
    SvtAv2EntropyModels coded_models, rate_models;
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_models_init(&coded_models, 1u), 0);
    rate_models = coded_models;
    SvtAv2EntropyEncoder wire, rate;
    uint8_t out[4096];
    uint16_t carry[4098];
    svt_av2_entropy_encoder_init(&wire, out, sizeof(out), carry, sizeof(carry) / sizeof(carry[0]));
    svt_av2_entropy_rate_init(&rate);
    wire.models = &coded_models;
    rate.models = &rate_models;
    SvtAv2MotionSyntax written, estimated;
    const int a = svt_av2_motion_syntax_init_precision(&written, 512u, 512u, 2u);
    const int b = svt_av2_motion_syntax_init_precision(&estimated, 512u, 512u, 2u);
    SVT_TEST_CHECK_EQ_INT(a, 0);
    SVT_TEST_CHECK_EQ_INT(b, 0);
    if (a || b) {
        svt_av2_motion_syntax_free(&written);
        svt_av2_motion_syntax_free(&estimated);
        return;
    }
    written.two_refs = estimated.two_refs = 1u;
    uint16_t expected[3][12];
    uint32_t visits[3] = {0};
    uint32_t visible[16][16] = {{0}}; /* 0=absent/intra, 1=recent, 2=long */
    memcpy(expected, initial, sizeof(expected));
    for (uint32_t i = 0u; i < 256u; ++i) {
        /* Four 32² leaves per 64² SB, in spatial traversal order. */
        const uint32_t x = 2u * ((i / 4u) % 8u) + (i & 1u);
        const uint32_t y = 2u * (i / 32u) + ((i >> 1u) & 1u);
        const uint32_t inter = i % 19u != 10u;
        const uint32_t rank = inter ? (x >= 8u) ^ (y >= 8u) : 0u;
        const SvtAv2TileBlockMode mode = {inter, 0, 0, rank, 0, 0};
        const SvtAv2BlockView block = {x * 32u, y * 32u, 32u, 32u, NULL};
        const uint32_t left = x ? visible[y][x - 1u] : 0u;
        const uint32_t above = y ? visible[y - 1u][x] : 0u;
        const uint32_t n0 = (left == 1u) + (above == 1u), n1 = (left == 2u) + (above == 2u);
        const uint32_t context = n0 < n1 ? 0u : n0 == n1 ? 1u : 2u;
        if (inter) {
            ++visits[context];
            reference_update(expected[context], 2u, 1u - rank);
        }
        visible[y][x] = inter ? 1u + rank : 0u;
        SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&written, &wire, &block, &mode, 1u), 0);
        SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&estimated, &rate, &block, &mode, 1u), 0);
        for (uint32_t c = 0u; c < 3u; ++c) {
            const uint32_t id = SVT_AV2_MODEL_SINGLE_REF + c;
            SVT_TEST_CHECK(memcmp(coded_models.cdf[id], expected[c], sizeof(expected[c])) == 0);
            SVT_TEST_CHECK(memcmp(rate_models.cdf[id], expected[c], sizeof(expected[c])) == 0);
        }
    }
    SVT_TEST_CHECK(visits[0] > 32u && visits[1] > 16u && visits[2] > 32u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_entropy_models_init(&rate_models, 3u), 0);
    const SvtAv2TileBlockMode old_mode = {1u, 0, 0, 9u, 0, 0}; /* legacy ignores field */
    const SvtAv2BlockView old_block = {0u, 0u, 32u, 32u, NULL};
    estimated.two_refs = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&estimated, &rate, &old_block, &old_mode, 1u), 0);
    for (uint32_t c = 0u; c < 3u; ++c)
        SVT_TEST_CHECK(memcmp(rate_models.cdf[SVT_AV2_MODEL_SINGLE_REF + c], initial[c], sizeof(initial[c])) == 0);
    svt_av2_motion_syntax_free(&written);
    svt_av2_motion_syntax_free(&estimated);
}

int main(void) {
    updates(); decay(); isolation(); quarter_motion(); wide_quarter_shell(); independent_ref_contexts();
    for (uint32_t inter = 0u; inter < 2u; ++inter) {
        fresh_writer(8u, 1u, inter); fresh_writer(10u, 1u, inter); fresh_writer(10u, 3u, inter);
    }
    SVT_TEST_REPORT("m7_entropy");
}
