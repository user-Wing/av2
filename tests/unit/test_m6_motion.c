#include "rd_frame.h"
#include "transform_path.h"
#include "test_util.h"
#include "motion_estimation.h"
#include "inter_prediction.h"

#include <stdlib.h>
#include <string.h>

static uint64_t total_accepts;

typedef struct Audit {
    uint64_t minimum[16], seen[16], committed[16];
    uint64_t minimum_id[16];
    uint64_t lambda;
    uint64_t first_sb_costs[512];
    uint8_t first_sb_seen[512];
    const struct Audit* exhaustive;
    SvtAv2RdFrameOutput* output;
    const SvtAv2FrameBuffer* source;
    const SvtAv2FrameBuffer* reference;
    size_t previous_modes;
    uint32_t width, height;
} Audit;

static void observe(void* user, size_t sb, uint64_t id, uint64_t d,
    uint64_t r, uint64_t cost, int committed) {
    Audit* audit = (Audit*)user;
    SVT_TEST_CHECK(sb < 16u);
    if (sb >= 16u) return;
    /* Small fixture range: independent ordinary multiplication is safe here. */
    SVT_TEST_CHECK(r < UINT64_MAX / audit->lambda);
    SVT_TEST_CHECK_EQ_INT(cost, d + (audit->lambda * r + 524288u) / 1048576u);
    if (!committed) {
        if (sb == 0u && id < 512u) {
            audit->first_sb_costs[id] = cost; audit->first_sb_seen[id] = 1u;
            if (audit->exhaustive != NULL) {
                SVT_TEST_CHECK(audit->exhaustive->first_sb_seen[id] != 0u);
                SVT_TEST_CHECK_EQ_INT(cost, audit->exhaustive->first_sb_costs[id]);
            }
        }
        if (audit->seen[sb] == 0u || cost < audit->minimum[sb] ||
            (cost == audit->minimum[sb] && id < audit->minimum_id[sb])) {
            audit->minimum[sb] = cost; audit->minimum_id[sb] = id;
        }
        ++audit->seen[sb];
    } else {
        SVT_TEST_CHECK_EQ_INT(cost, audit->minimum[sb]);
        SVT_TEST_CHECK_EQ_INT(id, audit->minimum_id[sb]);
        if (audit->reference != NULL && audit->output != NULL) {
            const size_t count = audit->output->mode_count - audit->previous_modes;
            const uint32_t cols = (audit->width + 63u) / 64u;
            const int full = audit->width - (uint32_t)(sb % cols) * 64u >= 64u &&
                audit->height - (uint32_t)(sb / cols) * 64u >= 64u;
            if (full) {
                SVT_TEST_CHECK(count <= 4u);
                for (size_t leaf = 0u; leaf < count; ++leaf) {
                    const SvtAv2TileCoeffBlock* block = &audit->output->blocks[(audit->previous_modes + leaf) * 3u];
                    SvtAv2BlockView view = {0}; view.x = block->x; view.y = block->y;
                    view.w = block->width; view.h = block->height;
                    SvtAv2MotionSearchList list;
                    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_estimate_list(audit->source, audit->reference, &view, &list), SVT_AV2_ERROR_NONE);
                    const uint32_t slot = (id >> (21u + 2u * (uint32_t)leaf)) & 3u;
                    const SvtAv2TileBlockMode* mode = &audit->output->modes[audit->previous_modes + leaf];
                    SVT_TEST_CHECK_EQ_INT(mode->is_inter, ((id >> (2u * (uint32_t)leaf)) & 3u) == 3u);
                    if (mode->is_inter) {
                        SVT_TEST_CHECK(slot < list.count);
                        SVT_TEST_CHECK_EQ_INT(mode->mv_x, list.mv[slot].mv_x);
                        SVT_TEST_CHECK_EQ_INT(mode->mv_y, list.mv[slot].mv_y);
                    }
                }
            }
            audit->previous_modes = audit->output->mode_count;
        }
        ++audit->committed[sb];
    }
}

static void fill(SvtAv2FrameBuffer* picture) {
    const uint32_t step = picture->bit_depth > 8u ? 2u : 1u;
    for (uint32_t p = 0u; p < 3u; ++p)
        for (uint32_t y = 0u; y < svt_av2_picture_plane_height(picture, p); ++y)
            for (uint32_t x = 0u; x < svt_av2_picture_plane_width(picture, p); ++x) {
                const uint32_t value = ((x * 11u + y * 7u + (x * y % 31u) + p * 27u) & 255u)
                    << (picture->bit_depth - 8u);
                svt_av2_sample_store(picture->planes[p] + (size_t)y * picture->strides[p] + x * step,
                    picture->bit_depth, (uint16_t)value);
            }
}

static uint64_t picture_hash(const SvtAv2FrameBuffer* picture) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint32_t p = 0u; p < 3u; ++p)
        for (size_t i = 0u; i < (size_t)picture->strides[p] * picture->aligned_height[p]; ++i)
            hash = (hash ^ picture->planes[p][i]) * UINT64_C(1099511628211);
    return hash;
}

static uint64_t independent_distortion(const SvtAv2FrameBuffer* source,
    const SvtAv2FrameBuffer* recon) {
    uint64_t sums[16] = {0u};
    const uint32_t cols = (source->visible_width + 63u) / 64u;
    const uint32_t step = source->bit_depth > 8u ? 2u : 1u;
    for (uint32_t p = 0u; p < 3u; ++p) {
        const uint32_t scale = p != 0u && source->chroma_format == SVT_AV2_CHROMA_420 ? 2u : 1u;
        const uint64_t weight = p == 0u || scale == 2u ? 4u : 1u;
        for (uint32_t y = 0u; y < svt_av2_picture_plane_height(source, p); ++y)
            for (uint32_t x = 0u; x < svt_av2_picture_plane_width(source, p); ++x) {
                const int a = svt_av2_sample_load(source->planes[p] + (size_t)y * source->strides[p] + x * step, source->bit_depth);
                const int b = svt_av2_sample_load(recon->planes[p] + (size_t)y * recon->strides[p] + x * step, recon->bit_depth);
                const size_t sb = (size_t)(y * scale / 64u) * cols + x * scale / 64u;
                sums[sb] += weight * (uint64_t)((a - b) * (a - b));
            }
    }
    uint64_t result = 0u;
    for (size_t i = 0u; i < 16u; ++i)
        result += source->bit_depth == 10u ? (sums[i] + 8u) / 16u : sums[i];
    return result;
}

typedef struct FaultPath { SvtAv2IntraTransformPath real; uint32_t calls, fail_at; } FaultPath;
static SvtAv2Error fault_forward(void* user, uint32_t p, uint32_t w, uint32_t h,
    const int16_t* residual, uint32_t stride, int32_t* coeff, int32_t* qcoeff,
    int32_t* dqcoeff, int* zero) {
    FaultPath* fault = (FaultPath*)user;
    const SvtAv2Error error = fault->real.forward(fault->real.user, p, w, h,
        residual, stride, coeff, qcoeff, dqcoeff, zero);
    return ++fault->calls == fault->fail_at ? SVT_AV2_ERROR_INVALID_STATE : error;
}
static SvtAv2Error fault_inverse(void* user, uint32_t p, uint32_t w, uint32_t h,
    const int32_t* dqcoeff, uint32_t stride, int16_t* residual) {
    FaultPath* fault = (FaultPath*)user;
    return fault->real.inverse(fault->real.user, p, w, h, dqcoeff, stride, residual);
}

static void frame_test(uint32_t width, uint32_t height, uint32_t bd,
    SvtAv2ChromaFormat format, int inter) {
    SvtAv2FrameBuffer source = {0}, recon = {0};
    SvtAv2ReferenceFrame reference = {0}; SvtAv2ReconScratch scratch = {0};
    SVT_TEST_CHECK_EQ_INT(svt_av2_picture_alloc_ex(&source, width, height, bd, format), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_picture_alloc_ex(&recon, width, height, bd, format), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_recon_pool_init(&scratch, 64u, 64u), SVT_AV2_ERROR_NONE);
    fill(&source);
    if (inter) {
        const uint32_t step = bd > 8u ? 2u : 1u;
        for (uint32_t y = 0u; y < height; ++y)
            for (uint32_t x = 0u; x < width; ++x)
                svt_av2_sample_store(source.planes[0] + (size_t)y * source.strides[0] + x * step,
                    bd, (uint16_t)(128u << (bd - 8u)));
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_init(&reference, width, height, bd, format), SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_prepare(&reference, &source, 0u), SVT_AV2_ERROR_NONE);
        SVT_TEST_CHECK_EQ_INT(svt_av2_reference_commit(&reference), SVT_AV2_ERROR_NONE);
        uint16_t* prediction = (uint16_t*)malloc((size_t)width * height * sizeof(uint16_t));
        const SvtAv2MotionVector mv = {-1, 0};
        for (uint32_t p = 1u; p < 3u; ++p) {
            SvtAv2BlockView view = {0};
            view.w = svt_av2_picture_plane_width(&source, p); view.h = svt_av2_picture_plane_height(&source, p);
            SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict(&reference.committed, p, &view, mv, prediction, view.w), SVT_AV2_ERROR_NONE);
            for (uint32_t y = 0u; y < view.h; ++y)
                for (uint32_t x = 0u; x < view.w; ++x)
                    svt_av2_sample_store(source.planes[p] + (size_t)y * source.strides[p] + x * step,
                        bd, prediction[(size_t)y * view.w + x]);
        }
        free(prediction);
    }
    const uint64_t source_hash = picture_hash(&source);
    const uint64_t reference_hash = inter ? picture_hash(&reference.committed) : 0u;
    const size_t count = (size_t)width * height;
    SvtAv2RdFrameOutput output = {0};
    output.block_capacity = count / 64u * 3u; output.mode_capacity = count / 64u;
    output.coeff_capacity = count * 3u; output.decision_capacity = ((width + 63u) / 64u) * ((height + 63u) / 64u);
    output.blocks = (SvtAv2TileCoeffBlock*)calloc(output.block_capacity, sizeof(*output.blocks));
    output.modes = (SvtAv2TileBlockMode*)calloc(output.mode_capacity, sizeof(*output.modes));
    output.qcoeff = (int32_t*)calloc(output.coeff_capacity, sizeof(*output.qcoeff));
    output.decisions = (uint16_t*)calloc(output.decision_capacity, sizeof(*output.decisions));
    SVT_TEST_CHECK(output.blocks != NULL && output.modes != NULL && output.qcoeff != NULL && output.decisions != NULL);
    SvtAv2RdFrameConfig config = {0};
    svt_av2_frame_header_config_default(&config.frame_header, width, height);
    config.frame_header.bit_depth = bd; config.frame_header.chroma_format = (uint32_t)format;
    config.frame_header.base_q_idx = 100u + (bd == 10u ? 48u : 0u);
    config.frame_header.reduced_tx_set = 2u; config.frame_header.multi_frame = 1u;
    config.frame_header.m5_intra_modes = 1u; config.frame_header.inter_frame = inter != 0;
    config.rd_mode = 1u; config.partition_selection = 1u;
    SvtAv2TransformPathState transform;
    SVT_TEST_CHECK_EQ_INT(svt_av2_transform_path_init_bd(&transform, &config.transform,
        config.frame_header.base_q_idx, bd), SVT_AV2_ERROR_NONE);
    Audit audit = {0}; uint32_t qstep;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_lambda(config.frame_header.base_q_idx, bd, &audit.lambda, &qstep), SVT_AV2_ERROR_NONE);
    config.candidate_hook = observe; config.candidate_hook_user = &audit;
    SvtAv2RdFrameStats baseline = {0}, first = {0}, reversed = {0}, pruned = {0};
    config.candidate_hook = NULL;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, inter ? &reference : NULL,
        &config, &scratch, &output, &baseline), SVT_AV2_ERROR_NONE);
    const size_t baseline_modes = output.mode_count;
    config.motion_optimization = 1u;
    config.coeff_optimization = 1u;
    SvtAv2RdFrameStats rejected = {0};
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, inter ? &reference : NULL,
        &config, &scratch, &output, &rejected), SVT_AV2_ERROR_INVALID_PARAMETER);
    SVT_TEST_CHECK_EQ_INT(output.block_count, 0u);
    config.coeff_optimization = 0u;
    config.candidate_hook = observe;
    audit.output = &output; audit.source = &source; audit.reference = inter ? &reference.committed : NULL;
    audit.width = width; audit.height = height;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, inter ? &reference : NULL,
        &config, &scratch, &output, &first), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(first.replay_checks, output.decision_count);
    SVT_TEST_CHECK_EQ_INT(first.distortion, independent_distortion(&source, &recon));
    for (size_t i = 0u; i < output.decision_count; ++i) SVT_TEST_CHECK_EQ_INT(audit.committed[i], 1u);
    SVT_TEST_CHECK(first.motion_trials <= first.full_superblocks * 16u);
    SVT_TEST_CHECK(first.motion_accepts <= first.motion_trials);
    total_accepts += first.motion_accepts;
    SVT_TEST_CHECK_EQ_INT(first.candidate_count - baseline.candidate_count, first.motion_trials);
    if (width == 64u && height == 64u) SVT_TEST_CHECK(first.cost <= baseline.cost);
    if (!inter || (width < 64u && height < 64u)) {
        SVT_TEST_CHECK_EQ_INT(first.motion_trials, 0u);
        SVT_TEST_CHECK_EQ_INT(first.winner_hash, baseline.winner_hash);
    }
    config.frame_header.partition_decisions = output.decisions;
    config.frame_header.partition_decision_count = output.decision_count;
    uint64_t replay_rate = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_frame(&config.frame_header, output.blocks, output.block_count,
        inter ? output.modes : NULL, inter ? output.mode_count : 0u, &replay_rate), 0);
    SVT_TEST_CHECK_EQ_INT(replay_rate, first.rate_q12);
    size_t recon_bytes = 0u;
    for (uint32_t p = 0u; p < 3u; ++p) recon_bytes += (size_t)recon.strides[p] * recon.aligned_height[p];
    uint8_t* saved_recon = (uint8_t*)malloc(recon_bytes);
    int32_t* saved_coeff = (int32_t*)malloc(output.coeff_count * sizeof(int32_t));
    const size_t saved_count = output.coeff_count;
    memcpy(saved_recon, recon.buffer, recon_bytes);
    memcpy(saved_coeff, output.qcoeff, saved_count * sizeof(int32_t));
    config.frame_header.partition_decisions = NULL; config.frame_header.partition_decision_count = 0u;
    config.reverse_candidates = 1u; memset(&audit, 0, sizeof(audit));
    svt_av2_rd_lambda(config.frame_header.base_q_idx, bd, &audit.lambda, &qstep);
    audit.output = &output; audit.source = &source; audit.reference = inter ? &reference.committed : NULL;
    audit.width = width; audit.height = height;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, inter ? &reference : NULL,
        &config, &scratch, &output, &reversed), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(first.winner_hash, reversed.winner_hash);
    SVT_TEST_CHECK_EQ_INT(first.cost, reversed.cost);
    SVT_TEST_CHECK_EQ_INT(first.rate_q12, reversed.rate_q12);
    SVT_TEST_CHECK_EQ_INT(output.coeff_count, saved_count);
    SVT_TEST_CHECK(memcmp(saved_coeff, output.qcoeff, saved_count * sizeof(int32_t)) == 0);
    SVT_TEST_CHECK(memcmp(saved_recon, recon.buffer, recon_bytes) == 0);
    config.reverse_candidates = 0u; config.rd_mode = 2u; config.candidate_hook = NULL;
    /* Reverse enumeration collected the complete original domain. On the first
     * SB, identical entry state permits exact per-candidate subset comparison. */
    Audit pruned_audit = {0}; pruned_audit.lambda = audit.lambda;
    pruned_audit.exhaustive = &audit;
    pruned_audit.output = &output; pruned_audit.source = &source;
    pruned_audit.reference = inter ? &reference.committed : NULL;
    pruned_audit.width = width; pruned_audit.height = height;
    config.candidate_hook = observe; config.candidate_hook_user = &pruned_audit;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, inter ? &reference : NULL,
        &config, &scratch, &output, &pruned), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK(pruned.candidate_count <= first.candidate_count);
    /* A single SB shares its entry state, so the pruned subset cannot beat full. */
    if (width <= 64u && height <= 64u) SVT_TEST_CHECK(pruned.cost >= first.cost);
    const size_t capacity = output.coeff_capacity; output.coeff_capacity = 0u;
    config.candidate_hook = NULL;
    SVT_TEST_CHECK(svt_av2_rd_frame_encode(&source, &recon, inter ? &reference : NULL,
        &config, &scratch, &output, &pruned) != SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(output.block_count, 0u); output.coeff_capacity = capacity;
    config.rd_mode = 1u;
    FaultPath fault = {config.transform, 0u, 5u};
    if (inter && width == 64u && height == 64u)
        fault.fail_at = (uint32_t)(baseline.forward_calls - baseline_modes * 3u + 1u);
    config.transform.user = &fault; config.transform.forward = fault_forward;
    config.transform.inverse = fault_inverse;
    SVT_TEST_CHECK(svt_av2_rd_frame_encode(&source, &recon, inter ? &reference : NULL,
        &config, &scratch, &output, &pruned) != SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(output.block_count, 0u);
    config.transform = fault.real;
    config.rd_mode = 1u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, inter ? &reference : NULL,
        &config, &scratch, &output, &reversed), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(first.winner_hash, reversed.winner_hash);
    SVT_TEST_CHECK(memcmp(saved_recon, recon.buffer, recon_bytes) == 0);
    SVT_TEST_CHECK_EQ_INT(picture_hash(&source), source_hash);
    if (inter) {
        SVT_TEST_CHECK_EQ_INT(picture_hash(&reference.committed), reference_hash);
        SVT_TEST_CHECK_EQ_INT(reference.frame_index, 0u);
    }
    free(saved_recon); free(saved_coeff); free(output.blocks); free(output.modes);
    free(output.qcoeff); free(output.decisions);
    svt_av2_reference_release(&reference); svt_av2_recon_pool_free(&scratch);
    svt_av2_picture_free(&source); svt_av2_picture_free(&recon);
}

typedef struct SkipAudit {
    Audit common;
    uint64_t skip_trials, skip_winners, full_leaves, split_sbs, fractional_winners;
    const SvtAv2FrameBuffer* recon;
    const SvtAv2FrameBuffer* reference;
    uint32_t half;
} SkipAudit;

static void observe_skip(void* user, size_t sb, uint64_t id, uint64_t d,
    uint64_t r, uint64_t cost, int committed) {
    SkipAudit* audit = (SkipAudit*)user;
    observe(&audit->common, sb, id, d, r, cost, committed);
    const uint32_t cols = (audit->common.width + 63u) / 64u;
    const int full = audit->common.width - (uint32_t)(sb % cols) * 64u >= 64u &&
        audit->common.height - (uint32_t)(sb / cols) * 64u >= 64u;
    const size_t start = audit->common.previous_modes, end = audit->common.output->mode_count;
    if (committed) {
        audit->common.previous_modes = end;
        if (full) { audit->full_leaves += end - start; audit->split_sbs += end - start > 1u; }
    }
    if (!((id >> 41u) & 1u)) return;
    SVT_TEST_CHECK(full);
    if (!committed) { ++audit->skip_trials; return; }
    const uint32_t mask = (uint32_t)((id >> 32u) & 15u);
    const uint32_t bd = audit->recon->bit_depth, step = bd > 8u ? 2u : 1u;
    SVT_TEST_CHECK(mask > 0u && end > start && end - start <= 4u && (mask >> (end - start)) == 0u);
    for (size_t i = start; i < end; ++i) {
        if (!(mask & (1u << (i - start)))) continue;
        ++audit->skip_winners;
        const SvtAv2TileBlockMode* mode = audit->common.output->modes + i;
        SVT_TEST_CHECK_EQ_INT(mode->is_inter, 1u);
        if (audit->half && (mode->mv_x % 2 != 0 || mode->mv_y % 2 != 0)) ++audit->fractional_winners;
        for (uint32_t p = 0u; p < 3u; ++p) {
            const SvtAv2TileCoeffBlock* block = audit->common.output->blocks + i * 3u + p;
            const size_t coeffs = (size_t)(block->width < 32u ? block->width : 32u) *
                (block->height < 32u ? block->height : 32u);
            for (size_t k = 0u; k < coeffs; ++k) SVT_TEST_CHECK_EQ_INT(block->qcoeff[k], 0);
            SvtAv2BlockView view = {0};
            view.x = block->x; view.y = block->y; view.w = block->width; view.h = block->height;
            uint16_t pred[4096];
            const SvtAv2MotionVector mv = {mode->mv_x, mode->mv_y};
            const SvtAv2HalfPelMotionVector half = {mode->mv_x, mode->mv_y};
            const SvtAv2Error e = audit->half ? svt_av2_inter_predict_half(audit->reference,
                p, &view, half, pred, view.w) : svt_av2_inter_predict(audit->reference, p, &view, mv, pred, view.w);
            SVT_TEST_CHECK_EQ_INT(e, SVT_AV2_ERROR_NONE);
            if (e != SVT_AV2_ERROR_NONE) continue;
            for (uint32_t y = 0u; y < view.h; ++y)
                for (uint32_t x = 0u; x < view.w; ++x)
                    SVT_TEST_CHECK_EQ_INT(pred[(size_t)y * view.w + x], svt_av2_sample_load(audit->recon->planes[p] +
                        (size_t)(view.y + y) * audit->recon->strides[p] + (view.x + x) * step, bd));
        }
    }
}

static void skip_frame_test(uint32_t width, uint32_t height, uint32_t bd, SvtAv2ChromaFormat format,
    uint32_t motion, uint32_t adaptive, uint32_t partition) {
    SvtAv2FrameBuffer source = {0}, recon = {0};
    SvtAv2ReferenceFrame reference = {0};
    SvtAv2ReconScratch scratch = {0};
    SvtAv2RdFrameOutput output = {0};
    const size_t count = (size_t)width * height;
    const uint32_t step = bd > 8u ? 2u : 1u;
    output.block_capacity = count / 64u * 3u; output.mode_capacity = count / 64u;
    output.coeff_capacity = count * 3u; output.decision_capacity = ((width + 63u) / 64u) * ((height + 63u) / 64u);
    output.blocks = calloc(output.block_capacity, sizeof(*output.blocks));
    output.modes = calloc(output.mode_capacity, sizeof(*output.modes));
    output.qcoeff = calloc(output.coeff_capacity, sizeof(*output.qcoeff));
    output.decisions = calloc(output.decision_capacity, sizeof(*output.decisions));
    const SvtAv2Error a = svt_av2_picture_alloc_ex(&source, width, height, bd, format);
    const SvtAv2Error b = svt_av2_picture_alloc_ex(&recon, width, height, bd, format);
    const SvtAv2Error c = svt_av2_recon_pool_init(&scratch, 64u, 64u);
    const SvtAv2Error f = svt_av2_reference_init(&reference, width, height, bd, format);
    const int allocated = !a && !b && !c && !f && output.blocks && output.modes && output.qcoeff && output.decisions;
    SVT_TEST_CHECK(allocated);
    if (!allocated) goto done;
    fill(&source);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_prepare(&reference, &source, 0u), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_commit(&reference), SVT_AV2_ERROR_NONE);
    for (uint32_t p = 0u; p < 3u; ++p) {
        const uint32_t uv420 = p && format == SVT_AV2_CHROMA_420;
        const uint32_t pix_w = svt_av2_picture_plane_width(&source, p);
        const uint32_t pix_h = svt_av2_picture_plane_height(&source, p);
        for (uint32_t y = 0u; y < pix_h; ++y) {
            const uint8_t* ref = reference.committed.planes[p] + (size_t)y * reference.committed.strides[p];
            for (uint32_t x = 0u; x < pix_w; ++x) {
                uint8_t* pixel = source.planes[p] + (size_t)y * source.strides[p] + x * step;
                uint16_t value = svt_av2_sample_load(ref + x * step, bd);
                /* Opposing 3.5-pel shifts make a single SB MV genuinely
                 * inferior to separate quadrant MVs. Compute edge-extended
                 * fractional samples independently, including UV420. */
                if (motion == 3u) {
                    const uint32_t parity = (y / (uv420 ? 16u : 32u) + x / (uv420 ? 16u : 32u)) % 2u;
                    const int32_t denom = uv420 ? 4 : 2, units = parity ? -7 : 7;
                    const int32_t integral = units >= 0 ? units / denom : (units - denom + 1) / denom;
                    const uint32_t fraction = (uint32_t)(units - integral * denom);
                    const int32_t x0 = (int32_t)x + integral, x1 = x0 + 1;
                    const uint32_t a_pos = x0 < 0 ? 0u : (uint32_t)x0 >= pix_w ? pix_w - 1u : (uint32_t)x0;
                    const uint32_t b_pos = x1 < 0 ? 0u : (uint32_t)x1 >= pix_w ? pix_w - 1u : (uint32_t)x1;
                    const uint16_t aa = svt_av2_sample_load(ref + a_pos * step, bd);
                    const uint16_t bb = svt_av2_sample_load(ref + b_pos * step, bd);
                    value = (uint16_t)((((uint32_t)denom - fraction) * aa + fraction * bb +
                        (uint32_t)denom / 2u) / (uint32_t)denom);
                }
                const uint16_t delta = (uint16_t)(2u << (bd - 8u));
                if ((uint32_t)value + delta < (1u << bd)) value = (uint16_t)(value + delta);
                svt_av2_sample_store(pixel, bd, value);
            }
        }
    }
    const uint64_t src_hash = picture_hash(&source), ref_hash = picture_hash(&reference.committed);
    SvtAv2RdFrameConfig cfg = {0};
    svt_av2_frame_header_config_default(&cfg.frame_header, width, height);
    cfg.frame_header.bit_depth = bd; cfg.frame_header.chroma_format = (uint32_t)format;
    cfg.frame_header.base_q_idx = 145u + (bd == 10u ? 48u : 0u);
    cfg.frame_header.reduced_tx_set = 2u; cfg.frame_header.multi_frame = cfg.frame_header.inter_frame = 1u;
    cfg.frame_header.m5_intra_modes = 1u; cfg.frame_header.mv_half = motion == 3u ? 1u : 0u;
    cfg.frame_header.entropy_adaptation = adaptive;
    cfg.rd_mode = 2u; cfg.partition_selection = partition; cfg.motion_optimization = motion;
    SvtAv2TransformPathState transform;
    SVT_TEST_CHECK_EQ_INT(svt_av2_transform_path_init_bd(&transform, &cfg.transform,
        cfg.frame_header.base_q_idx, bd), SVT_AV2_ERROR_NONE);
    SvtAv2RdFrameStats baseline = {0}, first = {0}, reverse = {0};
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, &reference, &cfg, &scratch, &output, &baseline), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(baseline.inter_skip_trials + baseline.inter_skip_accepts, 0u);
    SkipAudit audit = {0};
    audit.recon = &recon; audit.reference = &reference.committed;
    audit.half = cfg.frame_header.mv_half;
    audit.common.width = width; audit.common.height = height;
    audit.common.output = &output;
    audit.common.lambda = baseline.lambda_q8;
    cfg.inter_skip_search = 1u;
    cfg.candidate_hook = observe_skip;
    cfg.candidate_hook_user = &audit;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, &reference, &cfg, &scratch, &output, &first), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(first.replay_checks, output.decision_count);
    SVT_TEST_CHECK_EQ_INT(first.distortion, independent_distortion(&source, &recon));
    SVT_TEST_CHECK_EQ_INT(first.inter_skip_trials, audit.skip_trials);
    SVT_TEST_CHECK_EQ_INT(first.inter_skip_trials, audit.full_leaves);
    SVT_TEST_CHECK_EQ_INT(first.inter_skip_accepts, audit.skip_winners);
    SVT_TEST_CHECK(first.inter_skip_trials >= first.full_superblocks && first.inter_skip_trials <= first.full_superblocks * 4u);
    SVT_TEST_CHECK(first.inter_skip_accepts > 0u);
    if (partition) SVT_TEST_CHECK(audit.split_sbs > 0u);
    if (motion == 3u) SVT_TEST_CHECK(audit.fractional_winners > 0u);
    if (width == 64u && height == 64u && partition == 0u) {
        SVT_TEST_CHECK_EQ_INT(first.inter_skip_trials, 1u);
        SVT_TEST_CHECK(first.cost < baseline.cost);
        SVT_TEST_CHECK_EQ_INT(first.forward_calls + 3u, baseline.forward_calls);
    }
    cfg.frame_header.partition_decisions = output.decisions;
    cfg.frame_header.partition_decision_count = output.decision_count;
    uint64_t replay_rate = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_frame(&cfg.frame_header, output.blocks, output.block_count,
        output.modes, output.mode_count, &replay_rate), 0);
    SVT_TEST_CHECK_EQ_INT(replay_rate, first.rate_q12);
    cfg.frame_header.partition_decisions = NULL; cfg.frame_header.partition_decision_count = 0u;
    cfg.candidate_hook = NULL;
    cfg.reverse_candidates = 1u;
    const uint64_t recon_hash = picture_hash(&recon);
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, &reference, &cfg, &scratch, &output, &reverse), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(reverse.winner_hash, first.winner_hash);
    SVT_TEST_CHECK_EQ_INT(reverse.rate_q12, first.rate_q12);
    SVT_TEST_CHECK_EQ_INT(reverse.inter_skip_accepts, first.inter_skip_accepts);
    SVT_TEST_CHECK_EQ_INT(recon_hash, picture_hash(&recon));
    cfg.inter_skip_search = 2u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, &reference, &cfg, &scratch, &output, &reverse),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    cfg.inter_skip_search = 1u;
    cfg.rd_mode = 1u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, &reference, &cfg, &scratch, &output, &reverse),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    SVT_TEST_CHECK_EQ_INT(src_hash, picture_hash(&source));
    SVT_TEST_CHECK_EQ_INT(ref_hash, picture_hash(&reference.committed));

done:
    free(output.blocks); free(output.modes); free(output.qcoeff); free(output.decisions);
    svt_av2_reference_release(&reference); svt_av2_recon_pool_free(&scratch);
    svt_av2_picture_free(&source); svt_av2_picture_free(&recon);
}

static void fine_frame_test(uint32_t width, uint32_t height, uint32_t bd,
    SvtAv2ChromaFormat chroma, uint32_t adaptive) {
    SvtAv2FrameBuffer source = {0}, recon = {0};
    SvtAv2ReferenceFrame ref = {0};
    SvtAv2ReconScratch scratch = {0};
    SvtAv2RdFrameOutput output = {0};
    const size_t pixels = (size_t)width * height;
    output.block_capacity = 3u * pixels / 64u; output.mode_capacity = pixels / 64u;
    output.coeff_capacity = pixels * 3u;
    output.decision_capacity = ((width + 63u) / 64u) * ((height + 63u) / 64u);
    output.blocks = (SvtAv2TileCoeffBlock*)calloc(output.block_capacity, sizeof(*output.blocks));
    output.modes = (SvtAv2TileBlockMode*)calloc(output.mode_capacity, sizeof(*output.modes));
    output.qcoeff = (int32_t*)calloc(output.coeff_capacity, sizeof(*output.qcoeff));
    output.decisions = (uint16_t*)calloc(output.decision_capacity, sizeof(*output.decisions));
    SVT_TEST_CHECK(output.blocks && output.modes && output.qcoeff && output.decisions);
    if (!output.blocks || !output.modes || !output.qcoeff || !output.decisions) goto done;
    const SvtAv2Error a = svt_av2_picture_alloc_ex(&source, width, height, bd, chroma);
    const SvtAv2Error b = svt_av2_picture_alloc_ex(&recon, width, height, bd, chroma);
    const SvtAv2Error c = svt_av2_reference_init(&ref, width, height, bd, chroma);
    const SvtAv2Error d = svt_av2_recon_pool_init(&scratch, 64u, 64u);
    SVT_TEST_CHECK(a == 0 && b == 0 && c == 0 && d == 0);
    if (a || b || c || d) goto done;
    fill(&source);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_prepare(&ref, &source, 0u), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(svt_av2_reference_commit(&ref), SVT_AV2_ERROR_NONE);
    /* Abrupt opposing odd-quarter shifts within one 32 quadrant; remaining
     * quadrants use one vector. Includes both completed SBs and clipped edges. */
    const uint32_t step = bd > 8u ? 2u : 1u;
    for (uint32_t p = 0u; p < 3u; ++p) {
        const uint32_t side = p != 0u && chroma == SVT_AV2_CHROMA_420 ? 8u : 16u;
        const uint32_t w = svt_av2_picture_plane_width(&source, p);
        const uint32_t h = svt_av2_picture_plane_height(&source, p);
        for (uint32_t y = 0u; y < h; y += side) for (uint32_t x = 0u; x < w; x += side) {
            if (w - x < side || h - y < side) continue;
            const SvtAv2BlockView leaf = {x, y, side, side, NULL};
            const uint32_t q = (y / (2u * side) % 2u) * 2u + x / (2u * side) % 2u;
            const uint32_t sb = x / (4u * side) + 3u * (y / (4u * side));
            const uint32_t cell = (y / side % 2u) * 2u + x / side % 2u;
            static const SvtAv2QuarterPelMotionVector shifts[4] = {{-29, 13}, {27, -13}, {29, 9}, {-27, -9}};
            const SvtAv2QuarterPelMotionVector ambient = {3, -1};
            const SvtAv2QuarterPelMotionVector mv = q == sb % 4u ? shifts[cell] : ambient;
            uint16_t prediction[16u * 16u];
            SVT_TEST_CHECK_EQ_INT(svt_av2_inter_predict_quarter_sharp(&ref.committed, p,
                &leaf, mv, prediction, side), SVT_AV2_ERROR_NONE);
            for (uint32_t j = 0u; j < side; ++j) for (uint32_t i = 0u; i < side; ++i)
                svt_av2_sample_store(source.planes[p] + (size_t)(y + j) * source.strides[p] + (x + i) * step,
                    bd, prediction[(size_t)j * side + i]);
        }
    }
    const uint64_t src_hash = picture_hash(&source), ref_hash = picture_hash(&ref.committed);
    SvtAv2RdFrameConfig cfg = {0};
    svt_av2_frame_header_config_default(&cfg.frame_header, width, height);
    cfg.frame_header.bit_depth = bd; cfg.frame_header.chroma_format = (uint32_t)chroma;
    cfg.frame_header.base_q_idx = 116u + (bd == 10u ? 48u : 0u);
    cfg.frame_header.reduced_tx_set = 2u; cfg.frame_header.multi_frame = cfg.frame_header.inter_frame = 1u;
    cfg.frame_header.m5_intra_modes = 1u; cfg.frame_header.mv_half = 2u; cfg.frame_header.sharp_filter = 1u;
    cfg.frame_header.entropy_adaptation = adaptive;
    cfg.rd_mode = 2u; cfg.partition_selection = 1u; cfg.motion_optimization = 5u;
    SvtAv2TransformPathState transform;
    SVT_TEST_CHECK_EQ_INT(svt_av2_transform_path_init_bd(&transform, &cfg.transform,
        cfg.frame_header.base_q_idx, bd), SVT_AV2_ERROR_NONE);
    SvtAv2RdFrameStats coarse = {0}, fine = {0}, reverse = {0};
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, &ref, &cfg, &scratch, &output, &coarse), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(coarse.inter_fine_trials + coarse.inter_fine_accepts, 0u);
    const uint64_t coarse_recon = picture_hash(&recon);
    Audit audit = {0}; audit.lambda = coarse.lambda_q8;
    cfg.inter_fine = 1u; cfg.candidate_hook = observe; cfg.candidate_hook_user = &audit;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, &ref, &cfg, &scratch, &output, &fine), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(fine.replay_checks, output.decision_count);
    SVT_TEST_CHECK_EQ_INT(fine.distortion, independent_distortion(&source, &recon));
    SVT_TEST_CHECK(fine.inter_fine_trials <= 2u * fine.full_superblocks);
    SVT_TEST_CHECK(fine.inter_fine_accepts > 0u && fine.inter_fine_accepts <= fine.inter_fine_trials);
    unsigned split_sbs = 0u, fractional_fine = 0u;
    for (size_t i = 0u; i < output.decision_count; ++i) {
        const uint32_t cols = (width + 63u) / 64u;
        SVT_TEST_CHECK_EQ_INT(audit.committed[i], 1u);
        if (height - (uint32_t)(i / cols) * 64u < 64u || width - (uint32_t)(i % cols) * 64u < 64u) {
            SVT_TEST_CHECK_EQ_INT(output.decisions[i], 0u);
            continue;
        }
        if (output.decisions[i] > 1u) {
            ++split_sbs;
            SVT_TEST_CHECK_EQ_INT(audit.minimum_id[i], (UINT64_C(1) << 42u) | output.decisions[i]);
            unsigned splits = 0u;
            for (uint32_t q = 0u; q < 4u; ++q)
                splits += svt_av2_partition_plan_depth(output.decisions[i], q) == 2u;
            SVT_TEST_CHECK(splits == 1u || splits == 2u);
        }
    }
    SVT_TEST_CHECK_EQ_INT(split_sbs, fine.inter_fine_accepts);
    for (size_t i = 0u; i < output.mode_count; ++i) {
        const SvtAv2TileCoeffBlock* y = &output.blocks[3u * i];
        const SvtAv2TileBlockMode* m = &output.modes[i];
        if (y->width == 16u && y->height == 16u && m->is_inter && (m->mv_x % 2 || m->mv_y % 2))
            ++fractional_fine;
    }
    SVT_TEST_CHECK(fractional_fine > 0u);
    if (width == 64u && height == 64u) SVT_TEST_CHECK(fine.cost < coarse.cost);
    cfg.frame_header.partition_decisions = output.decisions;
    cfg.frame_header.partition_decision_count = output.decision_count;
    uint64_t rate = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_tile_rate_frame(&cfg.frame_header, output.blocks, output.block_count,
        output.modes, output.mode_count, &rate), 0);
    SVT_TEST_CHECK_EQ_INT(rate, fine.rate_q12);
    const uint64_t recon_hash = picture_hash(&recon);
    cfg.frame_header.partition_decisions = NULL; cfg.frame_header.partition_decision_count = 0u;
    cfg.candidate_hook = NULL; cfg.reverse_candidates = 1u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, &ref, &cfg, &scratch, &output, &reverse), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(reverse.winner_hash, fine.winner_hash);
    SVT_TEST_CHECK_EQ_INT(reverse.inter_fine_trials, fine.inter_fine_trials);
    SVT_TEST_CHECK_EQ_INT(reverse.inter_fine_accepts, fine.inter_fine_accepts);
    SVT_TEST_CHECK_EQ_INT(reverse.rate_q12, fine.rate_q12);
    SVT_TEST_CHECK_EQ_INT(picture_hash(&recon), recon_hash);
    cfg.inter_skip_search = 1u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, &ref, &cfg, &scratch, &output, &reverse),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    cfg.inter_skip_search = 0u; cfg.partition_selection = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, &ref, &cfg, &scratch, &output, &reverse),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    cfg.partition_selection = 1u; cfg.inter_fine = 2u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, &ref, &cfg, &scratch, &output, &reverse),
        SVT_AV2_ERROR_INVALID_PARAMETER);
    cfg.inter_fine = 0u;
    SVT_TEST_CHECK_EQ_INT(svt_av2_rd_frame_encode(&source, &recon, &ref, &cfg, &scratch, &output, &reverse), SVT_AV2_ERROR_NONE);
    SVT_TEST_CHECK_EQ_INT(reverse.winner_hash, coarse.winner_hash);
    SVT_TEST_CHECK_EQ_INT(picture_hash(&recon), coarse_recon);
    SVT_TEST_CHECK_EQ_INT(picture_hash(&source), src_hash);
    SVT_TEST_CHECK_EQ_INT(picture_hash(&ref.committed), ref_hash);

done:
    free(output.blocks); free(output.modes); free(output.qcoeff); free(output.decisions);
    svt_av2_reference_release(&ref); svt_av2_recon_pool_free(&scratch);
    svt_av2_picture_free(&source); svt_av2_picture_free(&recon);
}

typedef struct FullCandidate { int32_t x, y; uint64_t sad; } FullCandidate;
static int full_order(const void* av, const void* bv) {
    const FullCandidate* a = (const FullCandidate*)av;
    const FullCandidate* b = (const FullCandidate*)bv;
    if (a->sad != b->sad) return a->sad < b->sad ? -1 : 1;
    const int am = abs(a->x) + abs(a->y), bm = abs(b->x) + abs(b->y);
    if (am != bm) return am < bm ? -1 : 1;
    if (a->y != b->y) return a->y < b->y ? -1 : 1;
    return a->x < b->x ? -1 : a->x != b->x;
}
static uint64_t full_sad(const SvtAv2FrameBuffer* source, const SvtAv2FrameBuffer* reference,
    const SvtAv2BlockView* block, int32_t dx, int32_t dy) {
    uint64_t sum = 0u;
    const uint32_t step = source->bit_depth > 8u ? 2u : 1u;
    for (uint32_t y = 0u; y < block->h; ++y)
        for (uint32_t x = 0u; x < block->w; ++x) {
            int32_t rx = (int32_t)(block->x + x) + dx, ry = (int32_t)(block->y + y) + dy;
            if (rx < 0) rx = 0;
            if (ry < 0) ry = 0;
            if (rx >= (int32_t)reference->visible_width) rx = (int32_t)reference->visible_width - 1;
            if (ry >= (int32_t)reference->visible_height) ry = (int32_t)reference->visible_height - 1;
            const int a = svt_av2_sample_load(source->planes[0] + (size_t)(block->y + y) * source->strides[0] + (block->x + x) * step, source->bit_depth);
            const int b = svt_av2_sample_load(reference->planes[0] + (size_t)ry * reference->strides[0] + (size_t)rx * step, reference->bit_depth);
            sum += (uint64_t)abs(a - b);
        }
    return sum;
}
static void list_tests(void) {
    unsigned appended = 0u;
    for (unsigned format = 0u; format < 3u; ++format) {
        const uint32_t bd = format == 0u ? 8u : 10u, step = bd == 8u ? 1u : 2u;
        const SvtAv2ChromaFormat chroma = format == 2u ? SVT_AV2_CHROMA_444 : SVT_AV2_CHROMA_420;
        SvtAv2FrameBuffer source = {0}, reference = {0};
        svt_av2_picture_alloc_ex(&source, 32u, 32u, bd, chroma);
        svt_av2_picture_alloc_ex(&reference, 32u, 32u, bd, chroma);
        for (unsigned pattern = 0u; pattern < 3u; ++pattern) {
            fill(&reference);
            for (uint32_t y = 0u; y < 32u; ++y)
                for (uint32_t x = 0u; x < 32u; ++x) {
                    const uint32_t rx = x + 5u < 32u ? x + 5u : 31u, ry = y > 3u ? y - 3u : 0u;
                    uint16_t value = pattern == 0u ? 0u : pattern == 1u ?
                        (uint16_t)(((x * 19u + y * 23u) & 255u) << (bd - 8u)) :
                        svt_av2_sample_load(reference.planes[0] + (size_t)ry * reference.strides[0] + rx * step, bd);
                    svt_av2_sample_store(source.planes[0] + (size_t)y * source.strides[0] + x * step, bd, value);
                    if (pattern == 0u)
                        svt_av2_sample_store(reference.planes[0] + (size_t)y * reference.strides[0] + x * step, bd, 0u);
                }
            for (unsigned position = 0u; position < 3u; ++position) {
                SvtAv2BlockView block = {0}; block.x = position * 8u; block.y = position * 8u;
                block.w = block.h = 16u;
                if (pattern == 2u && position == 1u) {
                    SvtAv2MotionField field = {0};
                    field.velocity[SVT_AV2_ME_FIELD_ROWS / 2u][SVT_AV2_ME_FIELD_COLS / 2u] =
                        (SvtAv2MotionVector){20, -12};
                    SvtAv2MotionSearchResult result = {{0, 0}, UINT64_MAX, 0u};
                    svt_av2_motion_temporal(&source, &reference, &block, &field, 9u, 16, &result);
                    SVT_TEST_CHECK_EQ_INT(result.candidates, 0u);
                    svt_av2_motion_temporal(&source, &reference, &block, &field, 2u, 16, &result);
                    SVT_TEST_CHECK_EQ_INT(result.sad, 0u);
                    SVT_TEST_CHECK_EQ_INT(result.mv.mv_x, 5);
                    SVT_TEST_CHECK_EQ_INT(result.mv.mv_y, -3);
                    field.velocity[SVT_AV2_ME_FIELD_ROWS / 2u][SVT_AV2_ME_FIELD_COLS / 2u] =
                        (SvtAv2MotionVector){-20, 12};
                    result = (SvtAv2MotionSearchResult){{0, 0}, UINT64_MAX, 0u};
                    svt_av2_motion_temporal(&source, &reference, &block, &field, -2, 16, &result);
                    SVT_TEST_CHECK_EQ_INT(result.sad, 0u);
                    SVT_TEST_CHECK_EQ_INT(result.mv.mv_x, 5);
                    SVT_TEST_CHECK_EQ_INT(result.mv.mv_y, -3);
                }
                FullCandidate oracle[289]; size_t n = 0u;
                for (int32_t y = -8; y <= 8; ++y)
                    for (int32_t x = -8; x <= 8; ++x) {
                        oracle[n].x = x; oracle[n].y = y;
                        oracle[n++].sad = full_sad(&source, &reference, &block, x, y);
                    }
                qsort(oracle, n, sizeof(oracle[0]), full_order);
                SvtAv2MotionSearchList list; SvtAv2MotionSearchResult legacy;
                SVT_TEST_CHECK_EQ_INT(svt_av2_motion_estimate_list(&source, &reference, &block, &list), SVT_AV2_ERROR_NONE);
                SVT_TEST_CHECK_EQ_INT(svt_av2_motion_estimate(&source, &reference, &block, &legacy), SVT_AV2_ERROR_NONE);
                SVT_TEST_CHECK_EQ_INT(list.candidates, 289u);
                SVT_TEST_CHECK_EQ_INT(list.candidates, legacy.candidates);
                SVT_TEST_CHECK_EQ_INT(list.mv[0].mv_x, legacy.mv.mv_x);
                SVT_TEST_CHECK_EQ_INT(list.mv[0].mv_y, legacy.mv.mv_y);
                SVT_TEST_CHECK_EQ_INT(list.sad[0], legacy.sad);
                int top_zero = 0;
                for (size_t i = 0u; i < 3u; ++i) {
                    SVT_TEST_CHECK_EQ_INT(list.mv[i].mv_x, oracle[i].x);
                    SVT_TEST_CHECK_EQ_INT(list.mv[i].mv_y, oracle[i].y);
                    SVT_TEST_CHECK_EQ_INT(list.sad[i], oracle[i].sad);
                    top_zero |= oracle[i].x == 0 && oracle[i].y == 0;
                }
                SVT_TEST_CHECK_EQ_INT(list.count, top_zero ? 3u : 4u);
                if (!top_zero) {
                    ++appended;
                    SVT_TEST_CHECK_EQ_INT(list.mv[3].mv_x, 0);
                    SVT_TEST_CHECK_EQ_INT(list.mv[3].mv_y, 0);
                    SVT_TEST_CHECK_EQ_INT(list.sad[3], full_sad(&source, &reference, &block, 0, 0));
                }
                for (uint32_t i = 0u; i < list.count; ++i)
                    for (uint32_t j = i + 1u; j < list.count; ++j)
                        SVT_TEST_CHECK(list.mv[i].mv_x != list.mv[j].mv_x || list.mv[i].mv_y != list.mv[j].mv_y);
            }
        }
        svt_av2_picture_free(&source); svt_av2_picture_free(&reference);
    }
    SVT_TEST_CHECK(appended > 0u);
}

static void check_projected_motion_parity(void) {
    SvtAv2MotionSyntax motion;
    const int init = svt_av2_motion_syntax_init_precision(&motion, 128u, 64u, 2u);
    SVT_TEST_CHECK_EQ_INT(init, 0);
    if (init != 0) return;
    motion.two_refs = 1u;
    motion.ref_distance[0] = 2u;
    motion.ref_distance[1] = 3u;
    SvtAv2EntropyEncoder enc;
    svt_av2_entropy_rate_init(&enc);
    const SvtAv2BlockView first = {0,0,64,64,NULL}, second = {64,0,16,16,NULL};
    const SvtAv2TileBlockMode old = {1u,-2,2,1u, 0, 0}, bad = {1u,0,0,0u, 0, 0}, exact = {1u,-1,1,0u, 0, 0};
    const SvtAv2MotionCell speculative = {-2,2,1u,2u,1u,1u, 0, 0};
    const SvtAv2MotionCell intra = {0,0,1u,0u,0u,1u, 0, 0};
    SvtAv2MotionPrefix prefix = {&first, &speculative, 1u};
    int32_t px = 77, py = 88;
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_predict(&motion, &second, 0u, NULL, &px, &py), 0);
    SVT_TEST_CHECK(px == 0 && py == 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_predict(&motion, &second, 0u, &prefix, &px, &py), 0);
    SVT_TEST_CHECK(px == -1 && py == 1 && !motion.cells[0].valid && enc.rate_q12 == 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&motion, &enc, &first, &old, 1u), 0);
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_predict(&motion, &second, 0u, NULL, &px, &py), 0);
    SVT_TEST_CHECK(px == -1 && py == 1);
    prefix.cells = &intra;
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_predict(&motion, &second, 0u, &prefix, &px, &py), 0);
    SVT_TEST_CHECK(px == 0 && py == 0 && motion.cells[0].is_inter == 2u);
    const uint64_t before = enc.rate_q12, symbols = enc.rate_symbols;
    const unsigned right = second.x / 4u;
    SVT_TEST_CHECK_EQ_INT(motion.cells[right].valid, 0u);
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&motion, &enc, &second, &bad, 1u),
        SVT_AV2_ENTROPY_ERROR_RANGE);
    SVT_TEST_CHECK(enc.rate_q12 == before && enc.rate_symbols == symbols && !motion.cells[right].valid);
    const SvtAv2BlockView unsupported[] = {{64,0,4,4,NULL}, {68,0,16,16,NULL}, {64,16,16,32,NULL}, {64,0,32,8,NULL}};
    for (uint32_t i = 0u; i < sizeof(unsupported) / sizeof(unsupported[0]); ++i) {
        SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&motion, &enc, &unsupported[i], &exact, 1u),
            SVT_AV2_ENTROPY_ERROR_RANGE);
        SVT_TEST_CHECK(enc.rate_q12 == before && enc.rate_symbols == symbols && !motion.cells[right + 1u].valid);
    }
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&motion, &enc, &second, &exact, 1u), 0);
    SVT_TEST_CHECK_EQ_INT(motion.cells[right].is_inter, 1u);
    SVT_TEST_CHECK_EQ_INT(motion.cells[right].is_new, 0u);
    SVT_TEST_CHECK_EQ_INT(motion.cells[right].mv_x, -1);
    SVT_TEST_CHECK_EQ_INT(motion.cells[right].mv_y, 1);
    SVT_TEST_CHECK(enc.rate_q12 > before && enc.rate_symbols > symbols);
    const SvtAv2BlockView edge = {80,0,8,8,NULL};
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_predict(&motion, &edge, 0u, NULL, &px, &py), 0);
    const SvtAv2TileBlockMode edge_mode = {1u,px,py,0u, 0, 0};
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&motion, &enc, &edge, &edge_mode, 1u), 0);
    SVT_TEST_CHECK(motion.cells[edge.x / 4u].valid && !motion.cells[edge.x / 4u].is_new);
    const SvtAv2BlockView rectangles[] = {{64,32,16,32,NULL}, {96,48,32,16,NULL}};
    for (unsigned i = 0; i < 2u; ++i) {
        const SvtAv2BlockView* b = &rectangles[i];
        SVT_TEST_CHECK_EQ_INT(svt_av2_motion_predict(&motion, b, i, NULL, &px, &py), 0);
        const SvtAv2TileBlockMode mode = {1u,px,py,i, 0, 0};
        SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&motion, &enc, b, &mode, 1u), 0);
        for (unsigned y = b->y; y < b->y + b->h; y += 4u)
            for (unsigned x = b->x; x < b->x + b->w; x += 4u) {
                const SvtAv2MotionCell* cell = &motion.cells[(y / 4u) * motion.cols + x / 4u];
                SVT_TEST_CHECK(cell->valid && cell->is_inter == i + 1u && cell->mv_x == px && cell->mv_y == py);
            }
    }
    motion.wide_motion = 1u;
    const SvtAv2BlockView wide_block = {96,0,16,16,NULL};
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_predict(&motion, &wide_block, 0u, NULL, &px, &py), 0);
    const SvtAv2TileBlockMode wide_mode = {1u, 240 + abs(px % 2), -240 + abs(py % 2), 0u, 0, 0};
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_write_block(&motion, &enc, &wide_block, &wide_mode, 0), 0);
    SVT_TEST_CHECK_EQ_INT(motion.cells[wide_block.x / 4u].mv_x, wide_mode.mv_x);
    motion.two_refs = 0;
    motion.wide_motion = 2;
    const SvtAv2BlockView bottom = {64,56,8,8,NULL};
    motion.cells[15u*motion.cols+15u] = (SvtAv2MotionCell){0,252,1,1,0,0,0,0};
    SVT_TEST_CHECK_EQ_INT(svt_av2_motion_predict(&motion,&bottom,0,NULL,&px,&py),0);
    SVT_TEST_CHECK_EQ_INT(py,96); /* Eight visible pixels plus the 16-pixel border, in Q2. */
    svt_av2_motion_syntax_free(&motion);
}

int main(void) {
    check_projected_motion_parity();
    list_tests();
    frame_test(56u, 56u, 10u, SVT_AV2_CHROMA_420, 1);
    frame_test(64u, 64u, 8u, SVT_AV2_CHROMA_420, 0);
    frame_test(64u, 64u, 8u, SVT_AV2_CHROMA_420, 1);
    frame_test(128u, 128u, 8u, SVT_AV2_CHROMA_420, 0);
    frame_test(128u, 128u, 8u, SVT_AV2_CHROMA_420, 1);
    frame_test(136u, 72u, 10u, SVT_AV2_CHROMA_420, 0);
    frame_test(120u, 120u, 10u, SVT_AV2_CHROMA_444, 1);
    skip_frame_test(64u, 64u, 8u, SVT_AV2_CHROMA_420, 0u, 0u, 0u);
    skip_frame_test(136u, 72u, 10u, SVT_AV2_CHROMA_420, 3u, 1u, 1u);
    skip_frame_test(64u, 64u, 10u, SVT_AV2_CHROMA_444, 2u, 1u, 0u);
    fine_frame_test(64u, 64u, 8u, SVT_AV2_CHROMA_420, 0u);
    fine_frame_test(136u, 128u, 10u, SVT_AV2_CHROMA_420, 1u);
    fine_frame_test(128u, 72u, 10u, SVT_AV2_CHROMA_444, 0u);
    SVT_TEST_CHECK(total_accepts > 0u);
    SVT_TEST_REPORT("test_m6_motion");
}
