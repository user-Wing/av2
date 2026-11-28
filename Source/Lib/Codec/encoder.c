#ifndef M6_SATD_ANGLES
#define M6_SATD_ANGLES 0
#endif
#ifndef M6_DEBLOCK
#define M6_DEBLOCK 0
#endif
#ifndef M6_DEBLOCK_TEST
#define M6_DEBLOCK_TEST 0
#endif
#ifndef M6_PC_WIENER
#define M6_PC_WIENER 0
#endif
#ifndef M6_CONTENT_GUARD
#define M6_CONTENT_GUARD 0
#endif
#ifndef M6_PC_WIENER_TEST
#define M6_PC_WIENER_TEST 0
#endif

/*
 * SVT-AV2 encoder API implementation.
 *
 * M0 implemented only the handle lifecycle, parameter storage and the state
 * machine. M1 onwards (this file, owned by T013) additionally runs the real
 * encode path: send_picture() validates and copies the caller's planes,
 * get_packet() returns the AV2 elementary stream produced by Bitstream/
 * keyframe.c. The CLI does not have a second encode path any more; it drives
 * this API (contract section 7).
 *
 * Encoding is synchronous. Still-picture workers join before packet publication.
 */

#include "inter_profile.h"
#include "codec.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "keyframe.h"
#include "entropy_models.h"
#include "motion_two_ref_mvp.h"
#include "rd_frame.h"
#include "motion_estimation.h"
#include "deblock_luma.h"
#include "cdef_luma.h"
#include "gdf_luma.h"
#if M6_PC_WIENER
#include "../Restoration/pc_wiener.h"
#if M6_CONTENT_GUARD
#include "content_filter_guard.h"
#endif
#endif
#include "inter_frame.h"
#include "pyramid8.h"
#include "obu_writer.h"
#include "bit_writer.h" /* SVT_AV2_BW_ERROR_NO_SPACE */
#include "intra_frame.h" /* svt_av2_intra_frame_encode (T012), M2 recon */

#define SVT_AV2_CONTEXT_MAGIC 0x53564132u /* "SVA2" */

/* Freeze the physical map and refresh decision before encoding, so RD and
 * the formal writer see the same logical references and expanded distances. */
typedef struct SvtAv2VideoCdfState {
    SvtAv2ReferenceFrame extra_ref[4]; /* Interior reference levels beyond the two end anchors. */
    SvtAv2EntropyModels slots[6]; /* Decayed, committed banks for the physical DPB. */
    SvtAv2EntropyModels rd_end, writer_end; /* Detached raw banks for exact comparison. */
    SvtAv2FrameHeaderConfig frame_header;
    SvtAv2MotionField motion_field;
    uint16_t* video_qindex;
    size_t video_qindex_count;
    int use_motion_field;
    int64_t slot_frame[6];
    int trace;
#if SVT_AV2_EXPERIMENTAL_VIDEO_CFL
    uint8_t cfl_sequence, cfl_key, cfl_p;
#endif
} SvtAv2VideoCdfState;

typedef struct SvtAv2PictureReference {
    const SvtAv2ReferenceFrame* predictor;
    const SvtAv2ReferenceFrame* secondary_reference;
    uint32_t dual_ref, retain_reference, refresh_mask, reference_slot;
    uint32_t slot_mask, rotate_anchor;
    uint32_t secondary_slot, refresh_slot, paired_role;
    uint32_t ref_distance[2]; /* Populated only with a distinct [LAST, LONG]. */
} SvtAv2PictureReference;

typedef struct SvtAv2AnchorProbe {
    uint16_t current[SVT_AV2_ANCHOR_GRID_COUNT];
    uint32_t age, reason, source_q8, last_q8, long_q8;
} SvtAv2AnchorProbe;

static SvtAv2Error encode_reconstruction(SvtAv2EncoderContext* context,
                                         uint32_t width,
                                         uint32_t height,
                                         uint32_t base_q_idx,
                                         const SvtAv2PictureReference* refs);

/* QP / base_q_idx validation range.
 *
 * M0 used [0,63] as a placeholder. The M0-M1 contract section 6 (frozen by
 * T004 against AV2 v1.0.0) records base_q_idx for 8-bit as f(8), i.e. the legal
 * range is [0,255] (MaxQ = MAXQ_8_BITS), NOT AV1's [0,63]. The API now accepts
 * the full legal range. This only widens what is accepted; it does not decide
 * any encoding behaviour - the M1 fixed QP value is chosen by T005. */
#define SVT_AV2_QP_MIN 0
#define SVT_AV2_QP_MAX 255

/* --- defaults ------------------------------------------------------------- */

void svt_av2_config_set_defaults(SvtAv2EncoderConfig* config) {
    if (config == NULL) {
        return;
    }
    memset(config, 0, sizeof(*config));
    config->chroma_format = SVT_AV2_CHROMA_420;
    config->bit_depth = SVT_AV2_BIT_DEPTH_8;
    config->frame_rate_num = 25;
    config->frame_rate_den = 1;
    config->qp = 100;
    config->preset = SVT_AV2_PRESET_SLOW;
    config->threads = 1;
    config->deterministic = 1;
    config->output_path = NULL;
    config->prediction_structure = SVT_AV2_PRED_ALL_INTRA;
    config->intra_mode_selection = 0u;
    config->partition_selection = 0u;
    config->rd_mode = 0u;
    config->coeff_optimization = 0u;
    config->motion_optimization = 0u;
    config->entropy_adaptation = 0u;
    config->still_picture = 0u;
    config->transform_selection = 0u;
    config->angle_search = 0u;
    config->mrl_search = 0u;
    config->tree_refinement = 0u;
    config->rd_lambda_model = 0u;
    config->cfl_search = 0u;
    config->mhccp_search = 0u;
    config->intra_joint = 0u;
}

/* --- validation ----------------------------------------------------------- */

static int is_valid_chroma_format(SvtAv2ChromaFormat format) {
    return format == SVT_AV2_CHROMA_400 || format == SVT_AV2_CHROMA_420 || format == SVT_AV2_CHROMA_422 ||
           format == SVT_AV2_CHROMA_444;
}

static int is_valid_bit_depth(SvtAv2BitDepth bit_depth) {
    return bit_depth == SVT_AV2_BIT_DEPTH_8 || bit_depth == SVT_AV2_BIT_DEPTH_10 || bit_depth == SVT_AV2_BIT_DEPTH_12;
}

static int is_valid_preset(SvtAv2Preset preset) {
    return preset == SVT_AV2_PRESET_FASTEST || preset == SVT_AV2_PRESET_FAST || preset == SVT_AV2_PRESET_SLOW ||
           preset == SVT_AV2_PRESET_PLACEBO;
}

SvtAv2Error svt_av2_config_validate(const SvtAv2EncoderConfig* config) {
    if (config == NULL) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    if (config->width == 0 || config->height == 0) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    if (!is_valid_chroma_format(config->chroma_format)) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    if (!is_valid_bit_depth(config->bit_depth)) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    if (config->frame_rate_num == 0 || config->frame_rate_den == 0) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    if (config->qp < SVT_AV2_QP_MIN || config->qp > SVT_AV2_QP_MAX) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    if (!is_valid_preset(config->preset)) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    if (config->output_path == NULL || config->output_path[0] == '\0') {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }

    if (config->intra_mode_selection > 4u) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->intra_mode_selection >= 2u && (!config->still_picture || !config->rd_mode ||
        config->coeff_optimization == 1u || config->motion_optimization))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->partition_selection > 10u || (config->partition_selection >= 2u &&
        config->intra_mode_selection < 2u)) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->rd_mode > 3u) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->rd_mode==3u && (!config->still_picture || config->intra_mode_selection<3u ||
        config->partition_selection>7u || config->intra_joint || config->coeff_optimization==1u ||
        config->motion_optimization)) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->rd_lambda_model > 2u || (config->rd_lambda_model &&
        (!config->rd_mode || (!config->still_picture && (config->rd_mode != 2u ||
         config->prediction_structure != SVT_AV2_PRED_LOW_DELAY_P))))) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->cfl_search > 1u || (config->cfl_search &&
        (!config->still_picture || !config->rd_mode || config->intra_mode_selection != 4u)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->mhccp_search > 1u || (config->mhccp_search &&
        (!config->still_picture || !config->rd_mode || config->intra_mode_selection != 4u)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->intra_joint > 1u || (config->intra_joint &&
        (config->intra_mode_selection < 3u || config->transform_selection != 2u)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->angle_search > 1u || (config->angle_search && (!config->still_picture ||
        !config->rd_mode || config->intra_mode_selection < 3u || config->motion_optimization)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->tree_refinement > 3u || (config->tree_refinement && (!config->still_picture ||
        !config->rd_mode || config->intra_mode_selection < 2u || config->partition_selection < 2u)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->tree_refinement == 3u && (config->partition_selection != 7u || config->rd_mode != 3u ||
        config->intra_mode_selection != 4u || config->coeff_optimization != 2u ||
        config->chroma_format != SVT_AV2_CHROMA_420))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->mrl_search > 1u || (config->mrl_search && (!config->still_picture ||
        !config->rd_mode || config->intra_mode_selection < 3u || config->motion_optimization)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->transform_selection > 2u || (config->transform_selection &&
        (!config->still_picture || !config->rd_mode || config->intra_mode_selection < 2u ||
            config->coeff_optimization == 1u || config->motion_optimization)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->coeff_optimization > 2u || (config->coeff_optimization && !config->rd_mode) ||
        (config->coeff_optimization == 2u && (!config->still_picture || config->intra_mode_selection < 2u)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->motion_optimization > 5u ||
        (config->motion_optimization && (!config->rd_mode || config->coeff_optimization)) ||
        (config->motion_optimization >= 3u &&
            (config->prediction_structure != SVT_AV2_PRED_LOW_DELAY_P || config->rd_mode != 2u ||
             config->partition_selection > 1u)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    /* Video RD uses the same frame-local bank as formal coding, restricted
     * to the low-delay P profile. Still-picture support is unchanged. */
    if (config->entropy_adaptation > 1u) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->entropy_adaptation && config->rd_mode) {
        if (config->still_picture) {
            if (config->coeff_optimization == 1u || config->motion_optimization)
                return SVT_AV2_ERROR_INVALID_PARAMETER;
        } else if (config->prediction_structure != SVT_AV2_PRED_LOW_DELAY_P ||
                   config->rd_mode != 2u || config->intra_mode_selection > 1u ||
                   config->partition_selection > 1u || config->coeff_optimization) {
            return SVT_AV2_ERROR_INVALID_PARAMETER;
        }
    }
    if (config->still_picture > 1u || (config->still_picture &&
        config->prediction_structure != SVT_AV2_PRED_ALL_INTRA))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (config->prediction_structure != SVT_AV2_PRED_ALL_INTRA &&
        config->prediction_structure != SVT_AV2_PRED_LOW_DELAY_P)
        return SVT_AV2_ERROR_INVALID_PARAMETER;

    /* M0 guard rails: reject geometry that would overflow the sample counter or
     * that no realistic source provides. Keeps later milestones honest. */
    uint32_t lw = 0;
    uint32_t lh = 0;
    uint32_t cw = 0;
    uint32_t ch = 0;
    if (svt_av2_compute_plane_dimensions(config->width, config->height, config->chroma_format, &lw, &lh, &cw, &ch) !=
        SVT_AV2_ERROR_NONE) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    size_t samples = 0;
    if (svt_av2_compute_frame_sample_count(config->width, config->height, config->chroma_format, &samples) !=
        SVT_AV2_ERROR_NONE) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    return SVT_AV2_ERROR_NONE;
}

/* --- pipeline buffers (T013) ---------------------------------------------- */

void svt_av2_context_release_pipeline(SvtAv2EncoderContext* context) {
    if (context == NULL) {
        return;
    }
    /* The pending source picture is an allocated SvtAv2FrameBuffer; its free
     * leaves the struct zeroed and safe for a later alloc. */
    svt_av2_picture_free(&context->frame_source);
    context->frame_width = 0;
    context->frame_height = 0;
    context->frame_pts = 0;
    context->frame_pending = 0;
    context->eos = 0;
    context->eos_reported = 0;
    memset(&context->block_ref_stats, 0, sizeof(context->block_ref_stats));
    context->anchor_source_valid = 0u;
    context->latest_slot = 0u;
    context->anchor_display = context->previous_anchor_display = 0u;
    memset(context->slot_display,0,sizeof(context->slot_display));
    context->pyramid_span = context->tree_full = 0u;
    context->held_pictures = 0u;
    context->previous_motion_q8 = UINT32_MAX;
    context->quiet_frames = 0u;
    context->last_clean_frame = 0u;

    svt_av2_reference_release(&context->reference);
    svt_av2_reference_release(&context->long_reference);
    if (context->video_cdf && context->motion_q == 20u)
        for (unsigned i = 0; i < 4; ++i) svt_av2_reference_release(&context->video_cdf->extra_ref[i]);
    if (context->video_cdf) free(context->video_cdf->video_qindex);
    free(context->video_cdf);
    context->video_cdf = NULL;
    free(context->picture_weights); context->picture_weights = NULL;
    free(context->partition_decisions); context->partition_decisions = NULL;
    context->partition_decision_capacity = context->partition_decision_count = 0;
    free(context->coded_modes); context->coded_modes = NULL;
    free(context->coded_blocks);
    free(context->coded_coeffs);
    context->coded_blocks = NULL;
    context->coded_coeffs = NULL;
    context->coded_block_capacity = context->coded_coeff_capacity = 0;
    free(context->packet);
    context->packet = NULL;
    context->packet_size = 0;
    context->packet_capacity = 0;
    context->packet_pending = 0;

    /* The recon picture and the block pool are owned by the context. Both
     * teardown calls are safe on a zeroed (never-initialised) member. */
    if (context->recon_ready) {
        svt_av2_recon_pool_free(&context->scratch);
        svt_av2_picture_free(&context->recon);
        context->recon_ready = 0;
    }
}

SvtAv2Error svt_av2_context_store_packet(SvtAv2EncoderContext* context,
                                         const uint8_t* stream,
                                         size_t stream_size,
                                         int32_t pts) {
    (void)pts;
    if (context == NULL || stream == NULL || stream_size == 0) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }

    /* Grow the packet buffer geometrically so a multi-frame run does not
     * reallocate on every frame, while never trusting a fixed capacity. */
    if (context->packet_capacity < stream_size) {
        uint8_t* grown = (uint8_t*)realloc(context->packet, stream_size);
        if (grown == NULL) {
            return SVT_AV2_ERROR_OUT_OF_MEMORY;
        }
        context->packet = grown;
        context->packet_capacity = stream_size;
    }
    memcpy(context->packet, stream, stream_size);
    context->packet_size = stream_size;
    context->packet_pending = 1;
    return SVT_AV2_ERROR_NONE;
}

/* --- input picture validation (contract section 7) ------------------------ */

/* Validates a submitted picture against the frozen M1 configuration.
 *
 * This does NOT accept an arbitrary void* and pretend to check it. It verifies
 * every field the contract requires (section 7): the picture metadata must
 * match the configured stream, the planes must be non-NULL and their strides
 * must be able to hold the visible plane width. Stride is in BYTES per row.
 *
 * The returned state describes which of Y/U/V are required so the caller can
 * report the exact plane that failed. */
static SvtAv2Error validate_picture(const SvtAv2EncoderContext* context,
                                    const SvtAv2Picture* picture,
                                    uint32_t* chroma_width,
                                    uint32_t* chroma_height) {
    const SvtAv2EncoderConfig* config = &context->config;

    if (picture->width == 0 || picture->height == 0) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    /* The picture must describe the same stream the handle was configured for.
     * M1 is fixed-format, so a mismatch is a request for something unsupported,
     * not something to silently convert. */
    if (picture->width != config->width || picture->height != config->height) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    if (picture->chroma_format != config->chroma_format) {
        return SVT_AV2_ERROR_UNSUPPORTED;
    }
    if (picture->bit_depth != config->bit_depth) {
        return SVT_AV2_ERROR_UNSUPPORTED;
    }

    if (!((picture->bit_depth == SVT_AV2_BIT_DEPTH_8 && picture->chroma_format == SVT_AV2_CHROMA_420) ||
          (picture->bit_depth == SVT_AV2_BIT_DEPTH_10 &&
           (picture->chroma_format == SVT_AV2_CHROMA_420 || picture->chroma_format == SVT_AV2_CHROMA_444))))
        return SVT_AV2_ERROR_UNSUPPORTED;

    SvtAv2ChromaFormat format = picture->chroma_format;
    uint32_t           lw = 0, lh = 0, cw = 0, ch = 0;
    if (svt_av2_compute_plane_dimensions(picture->width, picture->height, format, &lw, &lh, &cw, &ch) !=
        SVT_AV2_ERROR_NONE) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    (void)lw;
    (void)lh;
    *chroma_width = cw;
    *chroma_height = ch;

    /* Every plane must be present and readable for its visible width. A stride
     * smaller than the visible width would make the encoder read outside the
     * caller's rows; the contract forbids "detecting" capacity, so anything we
     * cannot prove is rejected here. */
    const uint32_t bytes = picture->bit_depth == SVT_AV2_BIT_DEPTH_10 ? 2u : 1u;
    if (picture->planes[0] == NULL || picture->strides[0] < picture->width * bytes) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    if (picture->planes[1] == NULL || picture->strides[1] < cw * bytes) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    if (picture->planes[2] == NULL || picture->strides[2] < cw * bytes) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }

    if (bytes == 2u) {
        for (uint32_t plane = 0; plane < 3u; ++plane) {
            const uint32_t w = plane == 0u ? picture->width : cw;
            const uint32_t h = plane == 0u ? picture->height : ch;
            for (uint32_t y = 0; y < h; ++y) {
                const uint8_t* row = picture->planes[plane] + (size_t)y * picture->strides[plane];
                for (uint32_t x = 0; x < w; ++x)
                    if (((uint32_t)row[2u*x] | ((uint32_t)row[2u*x+1u] << 8u)) > 1023u)
                        return SVT_AV2_ERROR_INVALID_PARAMETER;
            }
        }
    }

    /* Reject geometry whose tightly packed size would overflow. */
    size_t samples = 0;
    if (svt_av2_compute_frame_sample_count(picture->width, picture->height, format, &samples) !=
        SVT_AV2_ERROR_NONE) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    return SVT_AV2_ERROR_NONE;
}

/* --- error strings -------------------------------------------------------- */

const char* svt_av2_error_to_string(SvtAv2Error error) {
    switch (error) {
    case SVT_AV2_ERROR_NONE: return "SVT_AV2_ERROR_NONE";
    case SVT_AV2_ERROR_INVALID_HANDLE: return "SVT_AV2_ERROR_INVALID_HANDLE";
    case SVT_AV2_ERROR_INVALID_PARAMETER: return "SVT_AV2_ERROR_INVALID_PARAMETER";
    case SVT_AV2_ERROR_INVALID_STATE: return "SVT_AV2_ERROR_INVALID_STATE";
    case SVT_AV2_ERROR_NOT_IMPLEMENTED: return "SVT_AV2_ERROR_NOT_IMPLEMENTED";
    case SVT_AV2_ERROR_OUT_OF_MEMORY: return "SVT_AV2_ERROR_OUT_OF_MEMORY";
    case SVT_AV2_ERROR_IO: return "SVT_AV2_ERROR_IO";
    case SVT_AV2_ERROR_UNSUPPORTED: return "SVT_AV2_ERROR_UNSUPPORTED";
    case SVT_AV2_ERROR_INTERNAL: return "SVT_AV2_ERROR_INTERNAL";
    default: return "SVT_AV2_ERROR_UNKNOWN";
    }
}

/* --- handle helpers ------------------------------------------------------- */

static SvtAv2EncoderContext* context_from_handle(SvtAv2Handle handle) {
    if (handle == NULL) {
        return NULL;
    }
    if (handle->magic != SVT_AV2_CONTEXT_MAGIC) {
        return NULL;
    }
    if (handle->state == SVT_AV2_STATE_CLOSED) {
        return NULL;
    }
    return handle;
}

static void copy_string(char* destination, size_t capacity, const char* source) {
    if (capacity == 0) {
        return;
    }
    if (source == NULL) {
        destination[0] = '\0';
        return;
    }
    const size_t length = strlen(source);
    const size_t copy = (length < capacity - 1) ? length : capacity - 1;
    memcpy(destination, source, copy);
    destination[copy] = '\0';
}

/* --- API ------------------------------------------------------------------ */

SvtAv2Error svt_av2_enc_init_handle(SvtAv2Handle* handle, const SvtAv2EncoderInfo** info) {
    if (handle == NULL) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    *handle = NULL;

    SvtAv2EncoderContext* context = (SvtAv2EncoderContext*)calloc(1, sizeof(SvtAv2EncoderContext));
    if (context == NULL) {
        return SVT_AV2_ERROR_OUT_OF_MEMORY;
    }
    context->magic = SVT_AV2_CONTEXT_MAGIC;
    context->state = SVT_AV2_STATE_UNINITIALIZED;
    context->inter_ref_period = 1u;
    context->pictures_sent = 0;
    context->pictures_encoded = 0;
    context->fatal_error = 0;
    context->frame_source = (SvtAv2FrameBuffer){0}; /* zeroed; alloc'd on demand */
    context->frame_width = 0;
    context->frame_height = 0;
    context->frame_pts = 0;
    context->frame_pending = 0;
    context->eos = 0;
    context->eos_reported = 0;
    context->packet = NULL;
    context->packet_size = 0;
    context->packet_capacity = 0;
    context->packet_pending = 0;
    svt_av2_config_set_defaults(&context->config);

    if (info != NULL) {
        *info = svt_av2_enc_get_build_info();
    }
    *handle = context;
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_enc_set_parameter(SvtAv2Handle handle, SvtAv2Parameter param, const void* value) {
    SvtAv2EncoderContext* context = context_from_handle(handle);
    if (context == NULL) {
        return SVT_AV2_ERROR_INVALID_HANDLE;
    }
    if (context->state != SVT_AV2_STATE_UNINITIALIZED) {
        return SVT_AV2_ERROR_INVALID_STATE;
    }
    if (value == NULL) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    if ((int)param < 0 || param >= SVT_AV2_PARAM_COUNT) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }

    switch (param) {
    case SVT_AV2_PARAM_WIDTH: context->config.width = *(const uint32_t*)value; break;
    case SVT_AV2_PARAM_HEIGHT: context->config.height = *(const uint32_t*)value; break;
    case SVT_AV2_PARAM_CHROMA_FORMAT: context->config.chroma_format = *(const SvtAv2ChromaFormat*)value; break;
    case SVT_AV2_PARAM_BIT_DEPTH: context->config.bit_depth = *(const SvtAv2BitDepth*)value; break;
    case SVT_AV2_PARAM_FRAME_RATE_NUM: context->config.frame_rate_num = *(const uint32_t*)value; break;
    case SVT_AV2_PARAM_FRAME_RATE_DEN: context->config.frame_rate_den = *(const uint32_t*)value; break;
    case SVT_AV2_PARAM_QP: context->config.qp = *(const int32_t*)value; break;
    case SVT_AV2_PARAM_PRESET: context->config.preset = *(const SvtAv2Preset*)value; break;
    case SVT_AV2_PARAM_TILE_COLUMNS: {
        uint32_t count = *(const uint32_t*)value;
        if (count != 1u && count != 2u && count != 4u && count != 8u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->tile_columns = count;
        break;
    }
    case SVT_AV2_PARAM_THREADS: context->config.threads = *(const uint32_t*)value; break;
    case SVT_AV2_PARAM_DETERMINISTIC: context->config.deterministic = *(const int32_t*)value; break;
    case SVT_AV2_PARAM_OUTPUT_PATH: {
        const char* path = *(const char* const*)value;
        if (path == NULL || path[0] == '\0') {
            return SVT_AV2_ERROR_INVALID_PARAMETER;
        }
        const size_t length = strlen(path);
        char*        copy = (char*)malloc(length + 1);
        if (copy == NULL) {
            return SVT_AV2_ERROR_OUT_OF_MEMORY;
        }
        memcpy(copy, path, length + 1);
        if (context->config.output_path != NULL) {
            free((void*)context->config.output_path);
        }
        context->config.output_path = copy;
        break;
    }
    case SVT_AV2_PARAM_PREDICTION_STRUCTURE: context->config.prediction_structure = *(const SvtAv2PredictionStructure*)value; break;
    case SVT_AV2_PARAM_INTRA_MODE_SELECTION:
        if (*(const uint32_t*)value > 4u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.intra_mode_selection = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_PARTITION_SELECTION:
        if (*(const uint32_t*)value > 10u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.partition_selection = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_RD_MODE:
        if (*(const uint32_t*)value > 3u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.rd_mode = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_COEFF_OPTIMIZATION:
        if (*(const uint32_t*)value > 2u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.coeff_optimization = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_MOTION_OPTIMIZATION:
        if (*(const uint32_t*)value > 5u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.motion_optimization = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_ENTROPY_ADAPTATION:
        if (*(const uint32_t*)value > 1u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.entropy_adaptation = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_STILL_PICTURE:
        if (*(const uint32_t*)value > 1u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.still_picture = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_TRANSFORM_SELECTION:
        if (*(const uint32_t*)value > 2u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.transform_selection = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_TREE_REFINEMENT:
        if (*(const uint32_t*)value > 3u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.tree_refinement = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_RD_LAMBDA_MODEL:
        if (*(const uint32_t*)value > 2u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.rd_lambda_model = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_RD_LAMBDA_SCALE: {
        const uint32_t scale = *(const uint32_t*)value;
        if (scale && (scale < 50u || scale > 250u)) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->rd_lambda_scale = scale;
        break;
    }
    case SVT_AV2_PARAM_CFL_SEARCH:
        if (*(const uint32_t*)value > 1u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.cfl_search = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_MHCCP_SEARCH:
        if (*(const uint32_t*)value > 1u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.mhccp_search = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_INTRA_JOINT:
        if (*(const uint32_t*)value > 1u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.intra_joint = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_MRL_SEARCH:
        if (*(const uint32_t*)value > 1u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.mrl_search = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_ANGLE_SEARCH:
        if (*(const uint32_t*)value > 1u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->config.angle_search = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_CONTENT_TYPE: {
        const SvtAv2ContentType type=*(const SvtAv2ContentType*)value;
        if (type!=SVT_AV2_CONTENT_NONE && type!=SVT_AV2_CONTENT_ANIMATION &&
            type!=SVT_AV2_CONTENT_REAL) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->content_type=type;
        break;
    }
    case SVT_AV2_PARAM_INTER_SKIP_SEARCH:
        if (*(const uint32_t*)value > 1u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->inter_skip_search = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_INTER_REF_PERIOD: {
        const uint32_t period = *(const uint32_t*)value;
        if (period != 1u && period != 2u && period != 4u && period != 8u)
            return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->inter_ref_period = period;
        break;
    }
    case SVT_AV2_PARAM_VIDEO_KEY_QP: {
        const int32_t qp = *(const int32_t*)value;
        if (qp < SVT_AV2_QP_MIN || qp > SVT_AV2_QP_MAX) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->video_key_qp = qp;
        break;
    }
    case SVT_AV2_PARAM_VIDEO_ANCHOR_QP: {
        const int32_t qp = *(const int32_t*)value;
        if (qp < SVT_AV2_QP_MIN || qp > SVT_AV2_QP_MAX) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->video_anchor_qp = qp;
        break;
    }
    case SVT_AV2_PARAM_INTER_FINE:
        if (*(const uint32_t*)value > 1u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->inter_fine = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_DUAL_REF_PERIOD: {
        const uint32_t period = *(const uint32_t*)value;
        if (period != 0u && period != 2u && period != 4u && period != 8u)
            return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->dual_ref_period = period;
        break;
    }
    case SVT_AV2_PARAM_DUAL_REF_POLICY:
        if (*(const uint32_t*)value > 3u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->dual_ref_policy = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_DUAL_REF_CADENCE:
        if (*(const uint32_t*)value > 1u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->dual_ref_cadence = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_INTER_TAIL:
        if (*(const uint32_t*)value > 1u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->inter_tail = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_INTER_NEAR:
        if (*(const uint32_t*)value > 1u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->inter_near = *(const uint32_t*)value;
        break;
    case SVT_AV2_PARAM_CDF_INHERIT:
        if (*(const uint32_t*)value > 1u) return SVT_AV2_ERROR_INVALID_PARAMETER;
        context->cdf_inherit = *(const uint32_t*)value;
        break;
    default: return SVT_AV2_ERROR_INVALID_PARAMETER;
    }

    return SVT_AV2_ERROR_NONE;
}

/* Match zenrav1e's 10-bit luma/chroma matrix schedules to the
 * nearest AV1 AC step, including AV2's three fractional quantizer bits. */
static void zen_matrix_levels(int q, int uv_delta, uint8_t* luma, uint8_t* chroma) {
    static const int y_transition[] = {1284, 2008, 3052, 4880, 5848, 6920, 9840, 20632};
    static const int uv_transition[] = {1284, 3560, 9840, 20632, 30136, 44240};
    const int y = svt_av2_quant_qlookup(q);
    *luma = 10u;
    for (unsigned i=0; i<SVT_AV2_ARRAY_SIZE(y_transition); ++i) *luma -= y > y_transition[i];
    if (!*chroma) {
        const int uv = svt_av2_quant_qlookup(SVT_AV2_MIN(303, SVT_AV2_MAX(1, q+uv_delta)));
        *chroma = 10u;
        for (unsigned i=0; i<SVT_AV2_ARRAY_SIZE(uv_transition); ++i) *chroma -= uv > uv_transition[i];
    }
}

static int ccso_config_supported(const SvtAv2EncoderConfig* q) {
    return q->still_picture && q->prediction_structure==SVT_AV2_PRED_ALL_INTRA && q->rd_mode==3u &&
        q->intra_mode_selection==4u && (q->partition_selection==5u || q->partition_selection==7u) && q->tree_refinement>=2u &&
        q->entropy_adaptation==1u && (q->rd_lambda_model==1u || q->rd_lambda_model==2u) && q->transform_selection==2u &&
        q->coeff_optimization==2u && q->angle_search==1u && q->mrl_search==1u &&
        q->cfl_search==1u && q->mhccp_search==1u && !q->intra_joint && !q->motion_optimization &&
        q->chroma_format==SVT_AV2_CHROMA_420 && (q->bit_depth==8u || q->bit_depth==10u);
}

SvtAv2Error svt_av2_enc_init(SvtAv2Handle handle) {
    SvtAv2EncoderContext* context = context_from_handle(handle);
    if (context == NULL) {
        return SVT_AV2_ERROR_INVALID_HANDLE;
    }
    if (context->state != SVT_AV2_STATE_UNINITIALIZED) {
        return SVT_AV2_ERROR_INVALID_STATE;
    }

    const SvtAv2Error err = svt_av2_config_validate(&context->config);
    if (err != SVT_AV2_ERROR_NONE) {
        return err;
    }
    if (context->rd_lambda_scale && !context->config.rd_lambda_model) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (((context->dual_ref_policy || context->video_anchor_qp) && !context->dual_ref_period) ||
        (context->dual_ref_period && (context->config.still_picture ||
         context->config.prediction_structure != SVT_AV2_PRED_LOW_DELAY_P ||
         context->config.motion_optimization != 5u || context->config.rd_mode != 2u ||
         context->config.partition_selection > 1u || context->inter_ref_period != 1u)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if ((context->dual_ref_policy == 3u && context->inter_skip_search) ||
        (context->dual_ref_cadence && (context->dual_ref_policy != 3u || context->dual_ref_period != 4u)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (context->video_key_qp && (context->config.still_picture ||
        context->config.prediction_structure != SVT_AV2_PRED_LOW_DELAY_P))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (context->inter_fine && (context->config.prediction_structure != SVT_AV2_PRED_LOW_DELAY_P ||
        context->config.still_picture || context->config.rd_mode != 2u ||
        context->config.motion_optimization != 5u || context->config.partition_selection != 1u ||
        context->config.intra_mode_selection > 1u || context->inter_skip_search))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if ((context->inter_tail || context->inter_near) && (context->config.still_picture || context->inter_skip_search ||
        context->config.prediction_structure != SVT_AV2_PRED_LOW_DELAY_P || context->config.rd_mode != 2u ||
        context->config.motion_optimization != 5u || context->config.partition_selection > 1u ||
        context->config.intra_mode_selection > 1u || context->config.coeff_optimization ||
        context->config.transform_selection || (context->inter_tail && context->dual_ref_period &&
         context->dual_ref_policy != 3u)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (context->cdf_inherit && (context->config.still_picture ||
         context->config.prediction_structure != SVT_AV2_PRED_LOW_DELAY_P ||
         context->config.rd_mode != 2u || context->config.entropy_adaptation != 1u))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if ((context->inter_skip_search || context->inter_ref_period != 1u) &&
        (context->config.prediction_structure != SVT_AV2_PRED_LOW_DELAY_P ||
         context->config.rd_mode != 2u || context->config.partition_selection > 1u ||
         context->config.coeff_optimization || context->config.still_picture))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (context->config.width % 8u != 0 || context->config.height % 8u != 0 ||
        context->config.width > 4096u || context->config.height > 2160u ||
        !((context->config.bit_depth == SVT_AV2_BIT_DEPTH_8 && context->config.chroma_format == SVT_AV2_CHROMA_420) ||
          (context->config.bit_depth == SVT_AV2_BIT_DEPTH_10 &&
           (context->config.chroma_format == SVT_AV2_CHROMA_420 || context->config.chroma_format == SVT_AV2_CHROMA_444))) ||
        context->config.qp == 0)
        return SVT_AV2_ERROR_UNSUPPORTED;
    const char* aq=getenv("SVT_AV2_STILL_AQ");
    context->still_aq=aq && (!strcmp(aq,"variance") || !strcmp(aq,"octile") || !strcmp(aq,"contour"));
    const char* block_gdf=getenv("SVT_AV2_TILE_GDF");
    context->block_gdf=block_gdf && !strcmp(block_gdf,"blocks");
    if ((context->still_aq || context->block_gdf) && !ccso_config_supported(&context->config)) return SVT_AV2_ERROR_UNSUPPORTED;
    if (context->tile_columns > 1u && (!ccso_config_supported(&context->config) || context->config.width < context->tile_columns * 128u ||
        context->config.height <= 64u)) return SVT_AV2_ERROR_UNSUPPORTED;
    if (context->config.threads > 1) {
#ifdef _OPENMP
        if (!context->config.still_picture || !context->config.angle_search ||
            context->config.rd_mode != 3u || context->config.threads > 64u)
#endif
            return SVT_AV2_ERROR_UNSUPPORTED;
    }

#if SVT_AV2_EXPERIMENTAL_VIDEO_CFL
    /* Warm control keeps the same SEQ/KEY when comparing a P-only search.
     * The distinct off control measures the overhead of the sequence tool. */
    const char* cfl = getenv("SVT_AV2_VIDEO_CFL");
    const int cfl_seq = cfl && (!strcmp(cfl,"signal") || !strcmp(cfl,"p") ||
        !strcmp(cfl,"key") || !strcmp(cfl,"all"));
    if (cfl_seq && (!context->cdf_inherit || context->config.bit_depth != SVT_AV2_BIT_DEPTH_10 ||
        context->config.chroma_format != SVT_AV2_CHROMA_420 || context->config.intra_mode_selection != 1u ||
        context->config.transform_selection || context->config.coeff_optimization || context->inter_skip_search))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
#endif
    const char* ist = getenv("SVT_AV2_INTRA_IST");
    uint8_t ist_scope = 0u;
    if (ist && strcmp(ist, "off")) {
        ist_scope = !strcmp(ist, "signal") ? 1u : !strcmp(ist, "key") ? 2u :
            !strcmp(ist, "p") ? 4u : !strcmp(ist, "all") ? 6u : 0u;
        if (!ist_scope || !context->config.entropy_adaptation ||
            context->config.chroma_format != SVT_AV2_CHROMA_420 ||
            !(context->config.still_picture ? context->config.rd_mode == 3u : context->cdf_inherit))
            return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    const char* qm12 = getenv("SVT_AV2_QM12");
    const char* delta = getenv("SVT_AV2_UV_DELTA_Q");
    long uv_delta = 0;
    if (delta) {
        char* end;
        uv_delta = strtol(delta, &end, 10);
        if (end == delta || *end || uv_delta < -23 || uv_delta > 8 ||
            !(context->config.still_picture ? context->config.rd_mode == 3u : context->cdf_inherit))
            return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    const char* level = getenv("SVT_AV2_QM_LEVEL");
    const int curve = level && !strcmp(level, "zen-curve");
    if (curve && (!context->config.still_picture || context->config.bit_depth != SVT_AV2_BIT_DEPTH_10))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    const int inter_curve = level && !strcmp(level,"inter-curve");
    if (inter_curve && (context->config.prediction_structure != SVT_AV2_PRED_LOW_DELAY_P ||
        context->config.bit_depth != SVT_AV2_BIT_DEPTH_10 || !context->cdf_inherit))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    uint8_t qm_level = (curve || inter_curve) ? 8u : !level || !strcmp(level, "12") ? 12u :
        !strcmp(level, "8") ? 8u : !strcmp(level, "4") ? 4u : 0u;
    if (!qm_level) return SVT_AV2_ERROR_INVALID_PARAMETER;
    const uint8_t qm_scope = ((!qm12 && level) || (qm12 && !strcmp(qm12, "all"))) ? 3u :
        qm12 && !strcmp(qm12, "key") ? 1u : qm12 && !strcmp(qm12, "p") ? 2u : 0u;
    if (qm_scope && (context->config.chroma_format != SVT_AV2_CHROMA_420 ||
        !(context->config.still_picture ? context->config.rd_mode == 3u : context->cdf_inherit)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    const char* uv_level = getenv("SVT_AV2_QM_UV_LEVEL");
    uint8_t qm_uv_level = !uv_level ? 0u : !strcmp(uv_level,"4") ? 4u :
        !strcmp(uv_level,"8") ? 8u : !strcmp(uv_level,"12") ? 12u : 0u;
    if (uv_level && (!qm_uv_level || !qm_scope)) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (curve) zen_matrix_levels(context->config.qp+48,(int)uv_delta,&qm_level,&qm_uv_level);
    const char* edge = getenv("SVT_AV2_STILL_EDGE_FILTER");
    const char* ibp = getenv("SVT_AV2_IBP");
    const int enable_ibp = ibp && !strcmp(ibp, "1");
    const int edge_filter = edge && !strcmp(edge, "1");
    if (enable_ibp && (context->config.chroma_format != SVT_AV2_CHROMA_420 ||
        !(context->config.still_picture ? context->config.rd_mode == 3u : context->cdf_inherit)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (edge_filter && (!context->config.still_picture || context->config.rd_mode != 3u ||
        context->config.chroma_format != SVT_AV2_CHROMA_420))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    const char* gdf = getenv("SVT_AV2_GDF");
    const int quality_gdf = gdf && !strcmp(gdf, "still-quality");
    if (quality_gdf && !context->config.still_picture) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (gdf && !quality_gdf && strcmp(gdf,"off") && strcmp(gdf,"key") && strcmp(gdf,"all"))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    const int gdf_key = quality_gdf || (gdf && (!strcmp(gdf,"key") || !strcmp(gdf,"all")));
    if (gdf_key && (context->config.chroma_format != SVT_AV2_CHROMA_420 ||
        (!context->config.still_picture && !context->cdf_inherit)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    const char* motion_q = getenv("SVT_AV2_MOTION_Q");
    const int full_window = motion_q && !strcmp(motion_q,"pyramid-window");
    const int motion_layers = motion_q && !strcmp(motion_q, "pyramid-motion");
    const int drawing_cadence = motion_q && (!strcmp(motion_q, "drawing-rd") || !strcmp(motion_q, "filtered-drawing"));
    const int adaptive_q = motion_q && !strcmp(motion_q, "adaptive") ? 1 :
        motion_q && !strcmp(motion_q, "hold") ? 2 :
        motion_q && !strcmp(motion_q, "reuse") ? 3 :
        motion_q && !strcmp(motion_q, "reuse-motion") ? 4 :
        motion_q && !strcmp(motion_q, "rd-reuse") ? 5 :
        motion_q && !strcmp(motion_q, "layered") ? 6 :
        motion_q && !strcmp(motion_q, "temporal-layers") ? 7 :
        motion_q && !strcmp(motion_q, "temporal8") ? 8 :
        motion_q && !strcmp(motion_q, "motion-refresh") ? 9 :
        motion_q && !strcmp(motion_q, "qstep-layers") ? 10 :
        (drawing_cadence || (motion_q && !strcmp(motion_q, "rotating-anchor"))) ? 11 :
        motion_q && !strcmp(motion_q, "hierarchical-anchor") ? 12 :
        motion_q && !strcmp(motion_q, "paired-anchor") ? 13 :
        motion_q && !strcmp(motion_q, "paired-refs") ? 14 :
        motion_q && !strcmp(motion_q, "quartet-refs") ? 15 :
        motion_q && !strcmp(motion_q, "adaptive-gop") ? 16 :
        motion_q && !strcmp(motion_q, "overlay-rd") ? 17 :
        motion_q && !strcmp(motion_q, "tree-rd") ? 18 :
        motion_q && !strcmp(motion_q, "tree-overlay") ? 19 :
        (motion_layers || full_window || (motion_q && (!strcmp(motion_q, "pyramid8") || !strcmp(motion_q, "pyramid8-blend") || !strcmp(motion_q, "pyramid8-rd") || !strcmp(motion_q, "pyramid-tail") || !strcmp(motion_q, "pyramid-compound") || !strcmp(motion_q, "pyramid16")))) ? 20 : 0;
    if (adaptive_q >= 11 && (context->dual_ref_policy != 3u || !context->cdf_inherit))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (adaptive_q && (!context->dual_ref_cadence || !context->video_anchor_qp ||
        (adaptive_q >= 3 && context->config.chroma_format != SVT_AV2_CHROMA_420)))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    svt_av2_context_release_pipeline(context);
    context->motion_q = (uint8_t)adaptive_q;
    const char* display_refs = getenv("SVT_AV2_DISPLAY_REFS");
    const char* runs = getenv("SVT_AV2_DRAWING_RUNS");
    context->drawing_runs = adaptive_q == 20 && runs && !strcmp(runs,"cadence");
    context->display_all_slots = adaptive_q == 20 && display_refs && !strcmp(display_refs,"all");
    context->motion_layers = (uint8_t)motion_layers;
    context->adaptive_pyramid = full_window ? 4u : motion_q && !strcmp(motion_q, "pyramid16") ? 3u :
        (motion_layers || (motion_q && (!strcmp(motion_q, "pyramid-tail") || !strcmp(motion_q, "pyramid-compound")))) ? 2u :
        motion_q && !strcmp(motion_q, "pyramid8-rd");
    context->drawing_cadence = (uint8_t)drawing_cadence;
    context->intra_ist_scope = ist_scope;
    context->qm_scope = qm_scope;
    context->inter_qm_curve = (uint8_t)inter_curve;
    context->qm_level = qm_level;
    context->qm_uv_level = qm_uv_level;
    context->uv_delta_q = (int8_t)uv_delta;
    context->enable_ibp = (uint8_t)enable_ibp;
    context->still_edge_filter = (uint8_t)edge_filter;
    const char* compensation = getenv("SVT_AV2_LUMA_COMPENSATION");
    context->luma_compensation_scope = compensation && !strcmp(compensation, "all") ? 3u :
        compensation && !strcmp(compensation, "key") ? 1u :
        compensation && !strcmp(compensation, "p") ? 2u : 0u;
    context->gdf_key = (uint8_t)gdf_key;
    context->gdf_still_quality = (uint8_t)quality_gdf;
    context->gdf_inter = gdf && !strcmp(gdf, "all");
    const char* source_cdef = getenv("SVT_AV2_SOURCE_CDEF");
    context->source_cdef_scope = source_cdef && !strcmp(source_cdef, "all") ? 3u :
        source_cdef && !strcmp(source_cdef, "key") ? 1u :
        source_cdef && !strcmp(source_cdef, "p") ? 2u : 0u;
    if (context->cdf_inherit) {
        SvtAv2VideoCdfState* cdf = (SvtAv2VideoCdfState*)malloc(sizeof(*cdf));
        if (cdf == NULL) return SVT_AV2_ERROR_OUT_OF_MEMORY;
        const char* field = getenv("SVT_AV2_TEMPORAL_ME");
        cdf->use_motion_field = context->dual_ref_policy == 3u && field && !strcmp(field, "1");
        if (cdf->use_motion_field) memset(&cdf->motion_field, 0, sizeof(cdf->motion_field));
        cdf->video_qindex = NULL;
        cdf->video_qindex_count = 0;
        memset(cdf->extra_ref, 0, sizeof(cdf->extra_ref));
        for (unsigned i = 0; i < 6; ++i) cdf->slot_frame[i] = -1;
        const char* trace = getenv("SVT_AV2_CDF_INHERIT_TRACE");
        cdf->trace = trace && trace[0] == '1' && trace[1] == '\0';
#if SVT_AV2_EXPERIMENTAL_VIDEO_CFL
        cdf->cfl_sequence = (uint8_t)cfl_seq;
        cdf->cfl_key = (uint8_t)(cfl_seq && (!strcmp(cfl,"key") || !strcmp(cfl,"all")));
        cdf->cfl_p = (uint8_t)(cfl_seq && (!strcmp(cfl,"p") || !strcmp(cfl,"all")));
#endif
        context->video_cdf = cdf;
    }
    context->state = SVT_AV2_STATE_INITIALIZED;
    context->pictures_sent = 0;
    context->pictures_encoded = 0;
    return SVT_AV2_ERROR_NONE;
}

/* True when the owned source picture is already allocated at this geometry and
 * can be reused for the next frame without reallocation. */
static int frame_source_fits(const SvtAv2EncoderContext* context,
                             uint32_t width,
                             uint32_t height) {
    return context->frame_source.visible_width == width &&
           context->frame_source.visible_height == height &&
           context->frame_source.planes[0] != NULL;
}

/* Copies the caller's picture into the encoder-owned allocated source picture,
 * so the caller may free or reuse its planes as soon as send_picture() returns
 * (contract section 1.2).
 *
 * The copy goes through T007's adaptation entry: svt_av2_picture_view() builds a
 * borrowed view over the caller's planes (which are NOT an allocation) and
 * svt_av2_picture_copy_from_view() copies it into the allocated destination,
 * stripping row padding and retaining no caller pointer. This is the single
 * sanctioned bridge from the public SvtAv2Picture to an internal frame buffer;
 * no buffer pointer is faked. */
static SvtAv2Error copy_picture(const SvtAv2Picture* picture,
                                         SvtAv2FrameBuffer* destination) {
    SvtAv2BorrowedView view;
    SvtAv2Error        status = svt_av2_picture_view_ex(&view,
                                                     picture->planes,
                                                     picture->strides,
                                                     picture->width,
                                                     picture->height,
                                                     (uint32_t)picture->bit_depth, picture->chroma_format);
    if (status != SVT_AV2_ERROR_NONE) {
        return status;
    }
    return svt_av2_picture_copy_from_view(destination, &view);
}

/* Luma-only unsharp proposal based on AVM tune_vmaf's Gaussian kernel.
 * Bound the correction both in sample units and by the local extrema to
 * protect line art. Read the caller's immutable plane, including its stride. */
static void compensate_luma(const SvtAv2Picture* input, SvtAv2FrameBuffer* dst) {
    static const int weights[5] = {4, 15, 26, 15, 4};
    const int bd = (int)input->bit_depth, step = bd > 8 ? 2 : 1;
    const int limit = 4 << (bd - 8);
    for (uint32_t y = 2u; y + 2u < input->height; ++y) {
        for (uint32_t x = 2u; x + 2u < input->width; ++x) {
            const uint8_t* center = input->planes[0] + (size_t)y * input->strides[0] + x * step;
            const int sample = svt_av2_sample_load(center, bd);
            int sum = 0, lo = sample, hi = sample;
            for (int j = 0; j < 5; ++j) {
                const uint8_t* row = input->planes[0] + (size_t)(y - 2u + j) * input->strides[0] + (x - 2u) * step;
                int horizontal = 0;
                for (int i = 0; i < 5; ++i) {
                    const int value = svt_av2_sample_load(row + i * step, bd);
                    horizontal += weights[i] * value;
                    if (i >= 1 && i <= 3 && j >= 1 && j <= 3) {
                        if (value < lo) lo = value;
                        if (value > hi) hi = value;
                    }
                }
                sum += weights[j] * horizontal;
            }
            const int diff = sample - ((sum + 2048) >> 12);
            const int correction = diff >= 0 ? (diff + 1) / 2 : -((1 - diff) / 2);
            const int value = sample + SVT_AV2_MIN(limit, SVT_AV2_MAX(-limit, correction));
            svt_av2_sample_store(dst->planes[0] + (size_t)y * dst->strides[0] + x * step,
                bd, (uint16_t)SVT_AV2_MIN(hi, SVT_AV2_MAX(lo, value)));
        }
    }
}

/* Reuses the frame reconstruction and 64x64 block scratch across frames. */
static SvtAv2Error ensure_recon_buffers(SvtAv2EncoderContext* context,
                                        uint32_t width,
                                        uint32_t height) {
    if (context->recon_ready && context->recon.visible_width == width &&
        context->recon.visible_height == height) {
        return SVT_AV2_ERROR_NONE;
    }

    if (context->recon_ready) {
        svt_av2_recon_pool_free(&context->scratch);
        svt_av2_picture_free(&context->recon);
        context->recon_ready = 0;
    }

    SvtAv2Error status = svt_av2_picture_alloc_ex(&context->recon, width, height,
        (uint32_t)context->config.bit_depth, context->config.chroma_format);
    if (status != SVT_AV2_ERROR_NONE) {
        return status;
    }
    /* The pool must cover the largest block the traversal can hand the core:
     * one superblock, 64x64 for the frozen M2 configuration. */
    status = svt_av2_recon_pool_init(&context->scratch, 64u, 64u);
    if (status != SVT_AV2_ERROR_NONE) {
        svt_av2_picture_free(&context->recon);
        return status;
    }
    if (context->config.prediction_structure == SVT_AV2_PRED_LOW_DELAY_P) {
        svt_av2_reference_release(&context->reference);
        svt_av2_reference_release(&context->long_reference);
        status = svt_av2_reference_init(&context->reference, width, height,
            (uint32_t)context->config.bit_depth, context->config.chroma_format);
        if (status == SVT_AV2_ERROR_NONE && context->dual_ref_period)
            status = svt_av2_reference_init(&context->long_reference, width, height,
                (uint32_t)context->config.bit_depth, context->config.chroma_format);
        if (context->motion_q == 20u) {
            for (unsigned i = 0; i < (context->adaptive_pyramid > 2u ? context->adaptive_pyramid : 2u) && !status; ++i) {
                svt_av2_reference_release(&context->video_cdf->extra_ref[i]);
                status = svt_av2_reference_init(&context->video_cdf->extra_ref[i], width, height,
                    (uint32_t)context->config.bit_depth, context->config.chroma_format);
            }
        }
        if (status != SVT_AV2_ERROR_NONE) {
            svt_av2_reference_release(&context->reference);
            svt_av2_reference_release(&context->long_reference);
            svt_av2_recon_pool_free(&context->scratch);
            svt_av2_picture_free(&context->recon);
            return status;
        }
    }
    context->recon_ready = 1;
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_enc_send_picture(SvtAv2Handle handle, const void* picture) {
    SvtAv2EncoderContext* context = context_from_handle(handle);
    if (context == NULL) {
        return SVT_AV2_ERROR_INVALID_HANDLE;
    }
    if (context->state != SVT_AV2_STATE_INITIALIZED) {
        return SVT_AV2_ERROR_INVALID_STATE;
    }

    /* send_picture(NULL) is EOS / flush (contract section 7). It is only
     * meaningful once the pipeline is drained; asking for EOS while a packet is
     * still untaken is a state error, not a silent discard. */
    if (picture == NULL) {
        if (context->frame_pending || context->packet_pending) {
            return SVT_AV2_ERROR_INVALID_STATE;
        }
        /* An encoded hidden anchor still needs its midpoint and display command. */
        if (context->motion_q >= 13u && context->pictures_encoded &&
            context->anchor_display >= context->pictures_encoded)
            return SVT_AV2_ERROR_INVALID_STATE;
        context->eos = 1;
        return SVT_AV2_ERROR_NONE;
    }

    /* At most one picture may be in flight: the encoder is synchronous and the
     * packet is returned by the very next get_packet() call. A second picture
     * before that would silently drop the first frame's stream. */
    if (context->frame_pending || context->packet_pending) {
        return SVT_AV2_ERROR_INVALID_STATE;
    }
    if (context->eos) {
        /* EOS already signalled: no further pictures are accepted. */
        return SVT_AV2_ERROR_INVALID_STATE;
    }
    if (context->config.still_picture && context->pictures_sent != 0u)
        return SVT_AV2_ERROR_INVALID_STATE;

    const SvtAv2Picture* input = (const SvtAv2Picture*)picture;
    if (context->motion_q == 20u) {
        const unsigned i = context->pictures_sent % SVT_AV2_LOOKAHEAD_KEY_INTERVAL;
        const unsigned base = context->pictures_sent - i;
        unsigned offset = i;
        if (i && i <= context->anchor_display) {
            const unsigned begin = context->anchor_display - context->pyramid_span;
            offset = begin + svt_av2_pyramid_offset(context->pyramid_span, i-begin-1u);
        } else if (i && input->pts > (int64_t)(base+i)) {
            const unsigned span = (unsigned)input->pts - base - i + 1u;
            const int variable = context->adaptive_pyramid == 4u && span >= 2u && span <= 32u && !(span & 1u);
            if (!variable && span != 8u && !(context->adaptive_pyramid == 3u && span == 16u) &&
                (context->adaptive_pyramid < 2u || (span != 2u && span != 4u)))
                return SVT_AV2_ERROR_INVALID_PARAMETER;
            if (i != context->anchor_display+1u || (!variable && (i-1u) % span) || i+span > SVT_AV2_LOOKAHEAD_KEY_INTERVAL) return SVT_AV2_ERROR_INVALID_PARAMETER;
            offset = i+span-1u;
        }
        if (input->pts < 0 || (unsigned)input->pts != base+offset) return SVT_AV2_ERROR_INVALID_PARAMETER;
    } else if (context->motion_q >= 13u) {
        const uint32_t i = context->pictures_sent % SVT_AV2_LOOKAHEAD_KEY_INTERVAL;
        const uint32_t pts = (uint32_t)input->pts % SVT_AV2_LOOKAHEAD_KEY_INTERVAL;
        const uint32_t lookahead = context->motion_q >= 15u ? 3u : 1u;
        unsigned expected = i - 1u;
        if (context->motion_q >= 18u && context->tree_full && i <= context->anchor_display) {
            const unsigned remaining = context->anchor_display - i;
            if (remaining == 2u) expected = i;
            else if (remaining == 1u) expected = i - 2u;
        }
        if (input->pts < 0 || (uint32_t)input->pts / SVT_AV2_LOOKAHEAD_KEY_INTERVAL !=
            context->pictures_sent / SVT_AV2_LOOKAHEAD_KEY_INTERVAL || (!i && pts) ||
            (i && i <= context->anchor_display && pts != expected) ||
            (i > context->anchor_display && ((context->motion_q < 16u && i != context->anchor_display + 1u) ||
                pts < i || pts - i > lookahead || (context->motion_q >= 18u && pts > i && i != context->anchor_display+1u))))
            return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    uint32_t             chroma_width = 0;
    uint32_t             chroma_height = 0;
    const SvtAv2Error    validation = validate_picture(context, input, &chroma_width, &chroma_height);
    if (validation != SVT_AV2_ERROR_NONE) {
        return validation;
    }

    size_t samples = 0;
    if (svt_av2_compute_frame_sample_count(input->width, input->height, input->chroma_format, &samples) !=
        SVT_AV2_ERROR_NONE) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    (void)samples; /* the allocated layout is sized by geometry, not by this count */

    /* Allocate (or reuse) the source picture in the allocated layout the M2
     * encoder core consumes. svt_av2_picture_alloc() computes the padded rows
     * and rejects a size change, so a later frame of a different size is a
     * clean reallocation rather than an overrun. */
    if (!frame_source_fits(context, input->width, input->height)) {
        svt_av2_picture_free(&context->frame_source);
        const SvtAv2Error alloc_err =
            svt_av2_picture_alloc_ex(&context->frame_source, input->width, input->height,
                (uint32_t)input->bit_depth, input->chroma_format);
        if (alloc_err != SVT_AV2_ERROR_NONE) {
            return alloc_err;
        }
    }

    const SvtAv2Error copy_err = copy_picture(input, &context->frame_source);
    if (copy_err != SVT_AV2_ERROR_NONE) {
        return copy_err;
    }
    const int new_lookahead_gop = context->motion_q >= 13u &&
        context->pictures_encoded == SVT_AV2_LOOKAHEAD_KEY_INTERVAL;
    const int is_p = !new_lookahead_gop && context->config.prediction_structure == SVT_AV2_PRED_LOW_DELAY_P &&
        context->pictures_encoded != 0u;
    if (context->luma_compensation_scope & (is_p ? 2u : 1u))
        compensate_luma(input, &context->frame_source);
    /* Prefilter the owned input before prediction and quantization. No CDEF
     * syntax is required, and reference frames still use decoder reconstruction. */
    if ((context->source_cdef_scope & (is_p ? 2u : 1u)) &&
        !(input->width & 7u) && !(input->height & 7u) &&
        input->width <= 4096u && input->height <= 2160u) {
        const SvtAv2Error error = svt_av2_cdef_luma_apply(&context->frame_source, 4u, NULL);
        if (error != SVT_AV2_ERROR_NONE) return error;
    }

    /* Source, reconstruction and scratch remain distinct owned buffers. */
    const SvtAv2Error recon_err = ensure_recon_buffers(context, input->width, input->height);
    if (recon_err != SVT_AV2_ERROR_NONE) {
        return recon_err;
    }

    context->frame_width = input->width;
    context->frame_height = input->height;
    if (new_lookahead_gop) {
        /* A hidden reference does not advance the decoder's display-order
         * clock. Start a closed GOP before the eight-bit order hint wraps. */
        context->pictures_encoded = 0u;
        context->reference.valid = context->long_reference.valid = 0;
        context->anchor_display = context->previous_anchor_display = 0u;
        memset(context->slot_display,0,sizeof(context->slot_display));
        context->latest_slot = context->held_pictures = context->tree_full = 0u;
        context->anchor_source_valid = 0u;
        for (unsigned i = 0; i < 6; ++i) context->video_cdf->slot_frame[i] = -1;
        if (context->motion_q == 20u)
            for (unsigned i = 0; i < 4; ++i) context->video_cdf->extra_ref[i].valid = 0;
        if (context->video_cdf->use_motion_field)
            memset(&context->video_cdf->motion_field, 0, sizeof(context->video_cdf->motion_field));
    }
    free(context->picture_weights);
    context->picture_weights = NULL;
    context->picture_qp = 0;
    context->frame_pts = input->pts;
    context->frame_pending = 1;
    context->pictures_sent++;
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_enc_send_picture_at_qp(SvtAv2Handle handle, const SvtAv2Picture* picture, int32_t qp) {
    if (!context_from_handle(handle)) return SVT_AV2_ERROR_INVALID_HANDLE;
    if (!picture || qp < 1 || qp > 255) return SVT_AV2_ERROR_INVALID_PARAMETER;
    SvtAv2Error err = svt_av2_enc_send_picture(handle,picture);
    if (err == SVT_AV2_ERROR_NONE) handle->picture_qp = qp;
    return err;
}

SvtAv2Error svt_av2_enc_send_picture_weighted(SvtAv2Handle handle, const SvtAv2Picture* picture,
        int32_t qp, const uint16_t* weights, size_t count) {
    SvtAv2EncoderContext* c=context_from_handle(handle);
    if (!c) return SVT_AV2_ERROR_INVALID_HANDLE;
    if (c->state!=SVT_AV2_STATE_INITIALIZED || c->packet_pending || c->frame_pending)
        return SVT_AV2_ERROR_INVALID_STATE;
    if (!picture || !weights || qp<1 || qp>255 ||
        picture->width!=c->config.width || picture->height!=c->config.height ||
        count!=(uint64_t)((c->config.width+UINT64_C(63))/64u)*((c->config.height+UINT64_C(63))/64u) ||
        !count || count>SIZE_MAX/sizeof(*weights))
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (c->config.prediction_structure!=SVT_AV2_PRED_LOW_DELAY_P || !c->config.rd_mode || c->tile_columns>1u)
        return SVT_AV2_ERROR_UNSUPPORTED;
    for (size_t i=0;i<count;++i) if (!weights[i]) return SVT_AV2_ERROR_INVALID_PARAMETER;
    uint16_t* copy=(uint16_t*)malloc(count*sizeof(*copy));
    if (!copy) return SVT_AV2_ERROR_OUT_OF_MEMORY;
    memcpy(copy,weights,count*sizeof(*copy));
    SvtAv2Error err=svt_av2_enc_send_picture_at_qp(handle,picture,qp);
    if (err) free(copy);
    else c->picture_weights=copy;
    return err;
}

/* Encodes the pending frame into context->packet, if a frame is waiting and no
 * packet has been produced for it yet. This is the single encode path used by
 * both get_packet() entry points; the CLI reaches it through the same API, so
 * there is no second, divergent encoder (contract section 7).
 *
 * Returns SVT_AV2_ERROR_NONE when a packet is ready or when there is simply
 * nothing to do; a real failure is returned and latched in fatal_error. */

/* Private experiment: unchanged r25 RD, then one bounded BO4 packet proposal. */
#define CCSO_AUTO_UNITS (64u * 34u)
typedef struct CcsoAuto {
    uint64_t table[CCSO_AUTO_UNITS][2][4][3],raw[3],post[3],flag_rate[2],estimate;
    uint8_t map[CCSO_AUTO_UNITS][2],best_flags[2][CCSO_AUTO_UNITS];
    uint32_t cols,units,lut[2],mask,writers,rewrites,no_space;
    SvtAv2Ccso64Plan plan;
    SvtAv2FrameHeaderConfig header;
    SvtAv2Ccso64HeaderSize headers[4];
    clock_t scan_ticks,select_ticks,pack_ticks,apply_ticks;
} CcsoAuto;
static void ccso_auto_lut(uint32_t id,int8_t offsets[4]) {
    for (uint32_t b=0u;b<4u;++b) {const uint32_t digit=id%3u;id/=3u;offsets[b]=(int8_t)(digit==1u?-1:digit==2u?1:0);}
}
static SvtAv2Error ccso_auto_cost(const SvtAv2EncoderContext* c,const uint64_t raw[3],uint64_t rate,uint64_t* cost) {
    return svt_av2_rd_cost(svt_av2_rd_distortion(raw,(uint32_t)c->config.bit_depth,SVT_AV2_CHROMA_420),c->rd_stats.lambda_q8,rate,cost);
}
static uint64_t ccso_auto_hash(const uint8_t* data,size_t size) {
    uint64_t hash=UINT64_C(14695981039346656037);
    for (size_t i=0u;i<size;++i) hash=(hash^data[i])*UINT64_C(1099511628211);
    return hash; /* Diagnostic checksum; external closure also compares full SHA/bytes. */
}
static void ccso_auto_subset(CcsoAuto* w,uint32_t mask) {
    for (uint32_t p=0u;p<2u;++p) {
        w->plan.enabled[p]=(uint8_t)((mask>>p)&1u);ccso_auto_lut(w->lut[p],w->plan.offset[p]);
        for (uint32_t u=0u;u<w->units;++u) w->map[u][p]=w->plan.enabled[p]?w->best_flags[p][u]:0u;
    }
    w->header.ccso64=mask?&w->plan:NULL;
}
static SvtAv2Error ccso_auto_prepare(SvtAv2EncoderContext* c,CcsoAuto** output,int reverse,uint32_t db,int delta,int delta_h,uint32_t cdef_primary) {
    *output=NULL;
    CcsoAuto* w=(CcsoAuto*)calloc(1u,sizeof(*w));
    if (!w) return SVT_AV2_ERROR_OUT_OF_MEMORY;
    const uint32_t bd=(uint32_t)c->config.bit_depth,step=bd==10u?2u:1u,width=c->frame_width,height=c->frame_height;
    const int maximum=(1<<bd)-1;
    w->cols=(width+63u)/64u;w->units=w->cols*((height+63u)/64u);
    w->plan.unit_flags=(const uint8_t (*)[2])w->map;w->plan.unit_count=w->units;
    svt_av2_frame_header_config_default(&w->header,width,height);
    w->header.cdef_primary=cdef_primary;
    w->header.deblock_y=db;w->header.deblock_delta=delta;w->header.deblock_delta_h=delta_h;
    w->header.base_q_idx=(uint32_t)c->config.qp+(bd==10u?48u:0u);
    w->header.bit_depth=bd;w->header.chroma_format=1u;w->header.m5_intra_modes=4u;
    w->header.entropy_adaptation=1u;w->header.reduced_tx_set=0u;w->header.enable_mrls=w->header.enable_mhccp=1u;
    w->header.enable_intra_ist=c->intra_ist_scope!=0u;
    w->header.enable_ibp=c->enable_ibp;
    w->header.uv_delta_q=c->uv_delta_q;
    w->header.qm_level=c->path_state.qm_level;
    w->header.qm_uv_level=c->path_state.qm_uv_level;
    w->header.enable_intra_edge_filter=c->still_edge_filter;
    w->header.partition_decisions=c->partition_decisions;w->header.partition_decision_count=c->partition_decision_count;
    clock_t start=clock();
    for (uint32_t p=0u;p<3u;++p) for (uint32_t y=0u;y<(height>>(p!=0u));++y)
        for (uint32_t x=0u;x<(width>>(p!=0u));++x) {
            const int src=svt_av2_sample_load(c->frame_source.planes[p]+(size_t)y*c->frame_source.strides[p]+x*step,bd);
            const int rec=svt_av2_sample_load(c->recon.planes[p]+(size_t)y*c->recon.strides[p]+x*step,bd);
            if (!p) {const int d=src-rec;w->raw[0]+=(uint64_t)(d*d);continue;}
            const uint32_t luma=svt_av2_sample_load(c->recon.planes[0]+(size_t)(2u*y)*c->recon.strides[0]+2u*x*step,bd);
            const uint32_t band=luma>>(bd-2u),unit=(y/32u)*w->cols+x/32u;
            for (uint32_t k=0u;k<3u;++k) {
                int v=rec+(k==1u?-1:k==2u?1:0);v=v<0?0:v>maximum?maximum:v;
                const int d=src-v;w->table[unit][p-1u][band][k]+=(uint64_t)(d*d);
            }
        }
    for (uint32_t u=0u;u<w->units;++u) for (uint32_t p=0u;p<2u;++p)
        for (uint32_t b=0u;b<4u;++b) w->raw[p+1u]+=w->table[u][p][b][0];
    w->scan_ticks=clock()-start;start=clock();
    uint64_t plane_sse[2]={0u},best_cost[2]={UINT64_MAX,UINT64_MAX};
    SvtAv2Error error=SVT_AV2_ERROR_INVALID_STATE;
    for (uint32_t p=0u;p<2u;++p) for (uint32_t n=0u;n<81u;++n) {
        const uint32_t id=reverse?80u-n:n;
        memset(w->map,0,sizeof(w->map));w->plan.enabled[0]=(uint8_t)(p==0u);w->plan.enabled[1]=(uint8_t)(p==1u);
        ccso_auto_lut(id,w->plan.offset[p]);w->header.ccso64=&w->plan;
        SvtAv2Ccso64HeaderSize h;
        if (svt_av2_ccso64_header_size(&w->header,&h)!=0) goto fail;
        const uint64_t base_rate=c->rd_stats.rate_q12+(uint64_t)(h.sequence_bytes+h.frame_bytes)*32768u;
        uint64_t raw[3]={w->raw[0],w->raw[1],w->raw[2]},rate=0u,cost=0u;
        uint16_t cdf[2][4][6];svt_av2_ccso64_cdf_init(cdf);
        for (uint32_t u=0u;u<w->units;++u) {
            uint64_t off=0u,on=0u;
            for (uint32_t b=0u;b<4u;++b) {
                const int offset=w->plan.offset[p][b];off+=w->table[u][p][b][0];
                on+=w->table[u][p][b][offset<0?1u:offset>0?2u:0u];
            }
            const uint32_t ctx=svt_av2_ccso64_context(&w->plan,u,w->cols,p);
            uint64_t chosen=UINT64_MAX,chosen_rate=0u,chosen_sse=0u;uint16_t chosen_row[6];uint32_t flag=0u;
            for (uint32_t f=0u;f<2u;++f) {
                uint16_t row[6];memcpy(row,cdf[p][ctx],sizeof(row));
                SvtAv2EntropyEncoder enc;svt_av2_entropy_rate_init(&enc);
                if (svt_av2_ccso64_symbol(&enc,row,f,1)!=0) goto fail;
                uint64_t candidate[3]={raw[0],raw[1],raw[2]};candidate[p+1u]=raw[p+1u]-off+(f?on:off);
                if (ccso_auto_cost(c,candidate,base_rate+rate+enc.rate_q12,&cost)!=0) goto fail;
                if (f==0u || cost<chosen) {chosen=cost;chosen_rate=enc.rate_q12;chosen_sse=candidate[p+1u];flag=f;memcpy(chosen_row,row,sizeof(row));}
            }
            w->map[u][p]=(uint8_t)flag;raw[p+1u]=chosen_sse;rate+=chosen_rate;
            memcpy(cdf[p][ctx],chosen_row,sizeof(chosen_row));
        }
        if (ccso_auto_cost(c,raw,base_rate+rate,&cost)!=0) goto fail;
        if (cost<best_cost[p] || (cost==best_cost[p] && id<w->lut[p])) {
            best_cost[p]=cost;w->lut[p]=id;w->flag_rate[p]=rate;plane_sse[p]=raw[p+1u];
            for (uint32_t u=0u;u<w->units;++u) w->best_flags[p][u]=w->map[u][p];
        }
    }
    w->estimate=UINT64_MAX;
    for (uint32_t mask=0u;mask<4u;++mask) {
        ccso_auto_subset(w,mask);
        if (svt_av2_ccso64_header_size(&w->header,&w->headers[mask])!=0) goto fail;
        const uint64_t raw[3]={w->raw[0],mask&1u?plane_sse[0]:w->raw[1],mask&2u?plane_sse[1]:w->raw[2]};
        const uint64_t rate=c->rd_stats.rate_q12+(uint64_t)(w->headers[mask].sequence_bytes+w->headers[mask].frame_bytes)*32768u+
            (mask&1u?w->flag_rate[0]:0u)+(mask&2u?w->flag_rate[1]:0u);
        uint64_t cost;if (ccso_auto_cost(c,raw,rate,&cost)!=0) goto fail;
        if (cost<w->estimate) {w->estimate=cost;w->mask=mask;memcpy(w->post,raw,sizeof(raw));}
    }
    ccso_auto_subset(w,w->mask);w->select_ticks=clock()-start;*output=w;return SVT_AV2_ERROR_NONE;
fail:
    free(w);return error;
}
static SvtAv2Error ccso_auto_packet(SvtAv2EncoderContext* c,CcsoAuto* w,int on,size_t* bytes) {
    size_t capacity=svt_av2_tile_writer_max_payload_bytes(c->frame_width,c->frame_height)+256u;
    if (capacity<4096u) capacity=4096u;
    if (capacity<c->packet_capacity) capacity=c->packet_capacity;
    for (;;) {
        if (c->packet_capacity<capacity) {
            uint8_t* grown=(uint8_t*)realloc(c->packet,capacity);
            if (!grown) return SVT_AV2_ERROR_OUT_OF_MEMORY;
            c->packet=grown;c->packet_capacity=capacity;
        }
        ++w->writers;
        SvtAv2FrameHeaderConfig header=w->header;
        if (!on) header.ccso64=NULL;
        const int st=svt_av2_encode_still_picture_ccso64_packet(&header,c->coded_blocks,
            c->coded_block_count,c->packet,c->packet_capacity,bytes);
        if (st!=SVT_AV2_BW_ERROR_NO_SPACE) return st==0 && *bytes?SVT_AV2_ERROR_NONE:SVT_AV2_ERROR_INTERNAL;
        ++w->no_space;if (capacity>SIZE_MAX/2u) return SVT_AV2_ERROR_OUT_OF_MEMORY;capacity*=2u;
    }
}
#if M6_PC_WIENER
#if M6_CONTENT_GUARD
static int content_guard_domain(const SvtAv2EncoderContext* c) {
    return c->content_type==SVT_AV2_CONTENT_ANIMATION &&
        c->config.bit_depth==SVT_AV2_BIT_DEPTH_10 &&
        c->config.chroma_format==SVT_AV2_CHROMA_420 &&
        c->config.still_picture==1u &&
        c->config.prediction_structure==SVT_AV2_PRED_ALL_INTRA &&
        ((c->frame_width==1920u && c->frame_height==1080u) ||
         (c->frame_width==3840u && c->frame_height==2160u));
}
#endif
static void pc_wiener_skip_map(uint8_t* skip,const SvtAv2TileCoeffBlock* blocks,size_t count,
    uint32_t cols,uint32_t height) {
    for (size_t i=0u;i<count;++i) {
        const SvtAv2TileCoeffBlock* b=blocks+i;
        if (b->plane) continue;
        uint8_t zero=1u;
        const uint32_t cw=b->width<32u?b->width:32u,ch=b->height<32u?b->height:32u;
        for (uint32_t j=0u;j<cw*ch;++j) if (b->qcoeff[j]) {zero=0u;break;}
        for (uint32_t y=b->y/4u;y<(b->y+b->height)/4u && y<height/4u;++y)
            for (uint32_t x=b->x/4u;x<(b->x+b->width)/4u && x<cols;++x)
                skip[(size_t)y*cols+x]=zero;
    }
}
static SvtAv2Error pc_wiener_try(SvtAv2EncoderContext* c,CcsoAuto* w,int ccso_on,size_t* bytes,const uint16_t* deblocked) {
    const uint32_t width=c->frame_width,height=c->frame_height,cols=width/4u;
    const uint32_t bd=(uint32_t)c->config.bit_depth,step=bd==10u?2u:1u;
    const size_t pixels=(size_t)width*height,off_bytes=*bytes;
    uint16_t* src=(uint16_t*)malloc(pixels*sizeof(*src));
    uint16_t* dst=(uint16_t*)malloc(pixels*sizeof(*dst));
    uint8_t* skip=(uint8_t*)calloc(pixels/16u,1u);
    uint8_t* packet=(uint8_t*)malloc(off_bytes);
    SvtAv2Error error=SVT_AV2_ERROR_OUT_OF_MEMORY;
    if (!src || !dst || !skip || !packet) goto done;
    memcpy(packet,c->packet,off_bytes);
    pc_wiener_skip_map(skip,c->coded_blocks,c->coded_block_count,cols,height);
    for (uint32_t y=0u;y<height;++y) for (uint32_t x=0u;x<width;++x)
        src[(size_t)y*width+x]=svt_av2_sample_load(c->recon.planes[0]+(size_t)y*c->recon.strides[0]+x*step,bd);
    error=SVT_AV2_ERROR_INVALID_STATE;
    if (svt_av2_pcwiener_filter_y_parallel(src,width,dst,width,width,height,bd,
        w->header.base_q_idx,skip,cols,deblocked,SVT_AV2_MAX(1u,c->config.threads))!=0) goto done;
    uint64_t raw[3];memcpy(raw,ccso_on?w->post:w->raw,sizeof(raw));
    const uint64_t old_y=raw[0];raw[0]=0u;
    int forced=-1;
#if M6_PC_WIENER_TEST
    const char* value=getenv("SVT_AV2_TEST_PC_WIENER");
    if (value) {
        if (strcmp(value,"0") && strcmp(value,"1") && strcmp(value,"2")) {error=SVT_AV2_ERROR_INVALID_PARAMETER;goto done;}
        forced=value[0]-'0';
    }
#endif
    const uint32_t ru=w->header.pc_wiener_unit ? w->header.pc_wiener_unit : 256u;
    const uint32_t ru_cols=SVT_AV2_MAX((width+ru/2u)/ru,1u);
    const uint32_t ru_rows=SVT_AV2_MAX((height+ru/2u)/ru,1u);
    uint16_t cdf[6]={19969,0,0,1,3,4};
    uint64_t none_mask[SVT_AV2_ARRAY_SIZE(w->header.pc_wiener_none_mask)]={0},check_y=0u;
#if M6_CONTENT_GUARD
    /* Existing test-only forced PCW behavior stays separate from automatic
     * selection. Production activation is exclusively this API hint/domain. */
    const uint32_t content_guard_active=(uint32_t)(content_guard_domain(c) && forced<0);
    uint32_t content_guard_considered=0u,content_guard_vetoes=0u;
    uint64_t content_edge_samples=0u,content_flat_samples=0u;
#endif
    for (uint32_t r=0u;r<ru_rows;++r) for (uint32_t col=0u;col<ru_cols;++col) {
        const uint32_t left=col*ru,right=col+1u==ru_cols?width:left+ru;
        const uint32_t top=r?r*ru-8u:0u,bottom=r+1u==ru_rows?height:(r+1u)*ru-8u;
        uint64_t sse[2]={0u,0u},cost[2]={0u,0u};
        uint16_t states[2][6];
        for (uint32_t y=top;y<bottom;++y) for (uint32_t x=left;x<right;++x) {
            const int target=(int)svt_av2_sample_load(c->frame_source.planes[0]+(size_t)y*c->frame_source.strides[0]+x*step,bd);
            const int d0=target-(int)src[(size_t)y*width+x],d1=target-(int)dst[(size_t)y*width+x];
            sse[0]+=(uint64_t)(d0*d0);sse[1]+=(uint64_t)(d1*d1);
        }
        for (uint32_t flag=0u;flag<2u;++flag) {
            SvtAv2EntropyEncoder rate;svt_av2_entropy_rate_init(&rate);
            memcpy(states[flag],cdf,sizeof(cdf));
            if (svt_av2_ccso64_symbol(&rate,states[flag],flag,1)!=0) goto done;
            const uint64_t region[3]={sse[flag],0u,0u};
            if (ccso_auto_cost(c,region,rate.rate_q12,&cost[flag])!=0) goto done;
        }
        const uint32_t unit=r*ru_cols+col;
        uint32_t on=forced>=0 ? (forced==2 ? (r+col)&1u : 1u) : cost[1]<cost[0];
#if M6_CONTENT_GUARD
        if (content_guard_active && on) {
            if (right<=left || bottom<=top || right>width || bottom>height ||
                right-left>384u || bottom-top>384u ||
                c->frame_source.strides[0]<(size_t)width*step) {
                error=SVT_AV2_ERROR_INVALID_STATE;goto done;
            }
            M6ContentFilterGuardStats guard_stats;
            ++content_guard_considered;
            const int allow=m6_content_filter_accept(c->frame_source.planes[0],
                c->frame_source.strides[0],bd,src,dst,width,height,left,top,
                right-left,bottom-top,&guard_stats);
            content_edge_samples+=guard_stats.edge_samples;
            content_flat_samples+=guard_stats.flat_samples;
            if (!allow) {on=0u;++content_guard_vetoes;}
        }
#endif
        memcpy(cdf,states[on],sizeof(cdf));
        check_y+=sse[0];raw[0]+=sse[on];
        if (!on) {
            none_mask[unit/64u]|=UINT64_C(1)<<(unit&63u);
            for (uint32_t y=top;y<bottom;++y)
                memcpy(dst+(size_t)y*width+left,src+(size_t)y*width+left,(right-left)*sizeof(*dst));
        }
    }
    if (deblocked) {
        /* The CCSO search measured the luma before CDEF. PC-Wiener's
         * bypass arm uses the newly filtered samples. */
        w->raw[0]=w->post[0]=check_y;
    } else if (check_y!=old_y) goto done;
    uint64_t old_j=0u,new_j=0u;
    if (ccso_auto_cost(c,ccso_on?w->post:w->raw,(uint64_t)off_bytes*32768u,&old_j)!=0) goto done;
    w->header.enable_pc_wiener=1u;
    memcpy(w->header.pc_wiener_none_mask,none_mask,sizeof(none_mask));
    size_t new_bytes=0u;
    error=ccso_auto_packet(c,w,ccso_on,&new_bytes);
    w->header.enable_pc_wiener=0u;
    memset(w->header.pc_wiener_none_mask,0,sizeof(w->header.pc_wiener_none_mask));
    if (error!=0) goto done;
    error=ccso_auto_cost(c,raw,(uint64_t)new_bytes*32768u,&new_j);
    if (error!=0) goto done;
    int accepted=new_j<old_j;
    if (forced>=0) accepted=forced!=0;
    if (accepted) {
        *bytes=new_bytes;
        for (uint32_t y=0u;y<height;++y) for (uint32_t x=0u;x<width;++x)
            svt_av2_sample_store(c->recon.planes[0]+(size_t)y*c->recon.strides[0]+x*step,bd,dst[(size_t)y*width+x]);
    } else memcpy(c->packet,packet,off_bytes);
    fprintf(stderr,"PC_WIENER accepted=%d bytes=%zu/%zu SSE=%llu/%llu J=%llu/%llu none_mask=%llu none_mask_hi=%llu\n",
        accepted,off_bytes,new_bytes,(unsigned long long)check_y,(unsigned long long)raw[0],
        (unsigned long long)old_j,(unsigned long long)new_j,(unsigned long long)none_mask[0],(unsigned long long)none_mask[1]);
#if M6_CONTENT_GUARD
    fprintf(stderr,"CONTENT_GUARD type=%s active=%u considered=%u vetoes=%u edge_samples=%llu flat_samples=%llu forced=%d accepted=%d\n",
        c->content_type==SVT_AV2_CONTENT_ANIMATION?"animation":c->content_type==SVT_AV2_CONTENT_REAL?"real":"none",
        content_guard_active,content_guard_considered,content_guard_vetoes,
        (unsigned long long)content_edge_samples,(unsigned long long)content_flat_samples,forced,accepted);
#endif
done:
    free(src);free(dst);free(skip);free(packet);return error;
}
#endif

static SvtAv2Error ccso_auto_encode_base(SvtAv2EncoderContext* c,size_t* bytes,uint32_t db,int delta,int delta_h,uint32_t ru,uint32_t cdef_primary,int stack_restoration,uint32_t gdf_code) {
    CcsoAuto* w=NULL;SvtAv2Error error=ccso_auto_prepare(c,&w,0,db,delta,delta_h,cdef_primary);if (error!=0) return error;
    w->header.pc_wiener_unit=ru;
    w->header.enable_gdf=gdf_code!=0u;w->header.gdf_code=gdf_code;
    const clock_t start=clock();size_t off_bytes=0u,on_bytes=0u;uint64_t off_j=0u,on_j=0u;
    int accepted=0;error=ccso_auto_packet(c,w,0,&off_bytes);if (error!=0) goto done;
    const uint64_t off_hash=ccso_auto_hash(c->packet,off_bytes);
    if (off_bytes>UINT64_MAX/32768u || ccso_auto_cost(c,w->raw,(uint64_t)off_bytes*32768u,&off_j)!=0) {error=SVT_AV2_ERROR_INVALID_STATE;goto done;}
    *bytes=off_bytes;
    if (w->mask) {
        error=ccso_auto_packet(c,w,1,&on_bytes);if (error!=0) goto done;
        if (on_bytes>UINT64_MAX/32768u || ccso_auto_cost(c,w->post,(uint64_t)on_bytes*32768u,&on_j)!=0) {error=SVT_AV2_ERROR_INVALID_STATE;goto done;}
        accepted=on_j<off_j;
        if (accepted) *bytes=on_bytes;
        else {
            ++w->rewrites;error=ccso_auto_packet(c,w,0,bytes);if (error!=0) goto done;
            if (*bytes!=off_bytes || ccso_auto_hash(c->packet,*bytes)!=off_hash) {error=SVT_AV2_ERROR_INVALID_STATE;goto done;}
        }
    }
    w->pack_ticks=clock()-start;
    if (accepted) {
        const clock_t apply=clock();svt_av2_ccso64_apply(&w->plan,&c->recon);
        uint64_t actual[3]={w->raw[0],0u,0u};const uint32_t bd=(uint32_t)c->config.bit_depth,step=bd==10u?2u:1u;
        for (uint32_t p=1u;p<3u;++p) for (uint32_t y=0u;y<c->frame_height/2u;++y) for (uint32_t x=0u;x<c->frame_width/2u;++x) {
            const int d=(int)svt_av2_sample_load(c->frame_source.planes[p]+(size_t)y*c->frame_source.strides[p]+x*step,bd)-
                (int)svt_av2_sample_load(c->recon.planes[p]+(size_t)y*c->recon.strides[p]+x*step,bd);actual[p]+=(uint64_t)(d*d);
        }
        if (memcmp(actual,w->post,sizeof(actual))!=0) {error=SVT_AV2_ERROR_INVALID_STATE;goto done;}
        w->apply_ticks=clock()-apply;
    }
    fprintf(stderr,"CCSO_AUTO,mask=%u,accepted=%d,units=%u,lutU=%u,lutV=%u,writers=%u,rewrite=%u,no_space=%u,work_bytes=%zu,table_bytes=%zu,off_bytes=%zu,on_bytes=%zu,Joff=%llu,Jon=%llu,estimate=%llu,lambda=%llu\n",
        w->mask,accepted,w->units,w->lut[0],w->lut[1],w->writers,w->rewrites,w->no_space,sizeof(*w),sizeof(w->table),off_bytes,on_bytes,
        (unsigned long long)off_j,(unsigned long long)on_j,(unsigned long long)w->estimate,(unsigned long long)c->rd_stats.lambda_q8);
    fprintf(stderr,"CCSO_AUTO_SSE,pre=%llu/%llu/%llu,proposal=%llu/%llu/%llu,final=%llu/%llu/%llu,flagR=%llu/%llu,seq=%zu,frame_bits=%zu,frame_bytes=%zu,ticks=%lld/%lld/%lld/%lld,hz=%ld\n",
        (unsigned long long)w->raw[0],(unsigned long long)w->raw[1],(unsigned long long)w->raw[2],
        (unsigned long long)w->post[0],(unsigned long long)w->post[1],(unsigned long long)w->post[2],
        (unsigned long long)(accepted?w->post[0]:w->raw[0]),(unsigned long long)(accepted?w->post[1]:w->raw[1]),(unsigned long long)(accepted?w->post[2]:w->raw[2]),
        (unsigned long long)w->flag_rate[0],(unsigned long long)w->flag_rate[1],w->headers[w->mask].sequence_bytes,
        w->headers[w->mask].frame_bits,w->headers[w->mask].frame_bytes,(long long)w->scan_ticks,(long long)w->select_ticks,
        (long long)w->pack_ticks,(long long)w->apply_ticks,(long)CLOCKS_PER_SEC);
    fprintf(stderr,"CCSO_AUTO_MAP,");for (uint32_t u=0u;u<w->units;++u) fputc('0'+w->map[u][0]+2*w->map[u][1],stderr);fputc('\n',stderr);
    /* CCSO chroma classification reads the deblocked luma before CDEF. */
    uint16_t* deblocked=NULL;
    if (cdef_primary)
        error=svt_av2_cdef_luma_apply(&c->recon,cdef_primary,stack_restoration || gdf_code ? &deblocked : NULL);
    uint16_t* guide = NULL;
    if (gdf_code && error == SVT_AV2_ERROR_NONE) {
        guide = svt_av2_gdf_copy_luma(&c->recon);
        if (!guide) error = SVT_AV2_ERROR_OUT_OF_MEMORY;
    }
#if M6_PC_WIENER
    if (error==SVT_AV2_ERROR_NONE && (!cdef_primary || stack_restoration))
        error=pc_wiener_try(c,w,accepted,bytes,deblocked);
#endif
    if (guide && error == SVT_AV2_ERROR_NONE)
        error = svt_av2_gdf_luma(&c->recon, guide, deblocked, w->header.base_q_idx, gdf_code, 0u, c->config.threads ? c->config.threads : 1u);
    free(guide);
    free(deblocked);
done:
    free(w);return error;
}

#if M6_DEBLOCK
static void db_recon_copy(SvtAv2EncoderContext* c,uint8_t* raw,int restore) {
    size_t offset=0u;
    const uint32_t step=c->config.bit_depth>8u?2u:1u;
    for (uint32_t p=0u;p<3u;++p) {
        const uint32_t w=c->frame_width>>(p!=0u),h=c->frame_height>>(p!=0u);
        for (uint32_t y=0u;y<h;++y) {
            uint8_t* row=c->recon.planes[p]+(size_t)y*c->recon.strides[p];
            if (restore) memcpy(row,raw+offset,(size_t)w*step);
            else memcpy(raw+offset,row,(size_t)w*step);
            offset+=(size_t)w*step;
        }
    }
}
static SvtAv2Error db_current_cost(SvtAv2EncoderContext* c,size_t bytes,int gradient_rd,uint64_t* cost) {
    uint64_t sse[3]={0u,0u,0u};
    const uint32_t bd=(uint32_t)c->config.bit_depth,step=bd>8u?2u:1u;
    int16_t above[4096];
    uint64_t gradient_sse=0u;
    for (uint32_t p=0u;p<3u;++p) {
        const uint32_t w=c->frame_width>>(p!=0u),h=c->frame_height>>(p!=0u);
        for (uint32_t y=0u;y<h;++y) {
            int left=0;
            for (uint32_t x=0u;x<w;++x) {
                const int d=(int)svt_av2_sample_load(c->frame_source.planes[p]+(size_t)y*c->frame_source.strides[p]+x*step,bd)-
                    (int)svt_av2_sample_load(c->recon.planes[p]+(size_t)y*c->recon.strides[p]+x*step,bd);
                sse[p]+=(uint64_t)(d*d);
                if (gradient_rd && p==0u) {
                    /* The difference of two adjacent reconstruction errors equals
                     * the error of the corresponding source gradient. */
                    if (x) gradient_sse+=(uint64_t)((d-left)*(d-left));
                    if (y) {
                        const int delta=d-above[x];
                        gradient_sse+=(uint64_t)(delta*delta);
                    }
                    left=d;
                    above[x]=(int16_t)d;
                }
            }
        }
    }
    sse[0]+=(gradient_sse+2u)/4u;
    if (bytes>UINT64_MAX/32768u) return SVT_AV2_ERROR_INVALID_STATE;
    return ccso_auto_cost(c,sse,(uint64_t)bytes*32768u,cost);
}
#endif
static uint32_t gdf_key_code(const SvtAv2EncoderContext* c) {
    if (c->gdf_still_quality) return c->frame_height >= 2160u ? 16u : 15u;
    return c->gdf_key ? (c->frame_height >= 2160u ? 14u : 12u) : 0u;
}

static SvtAv2Error ccso_auto_encode(SvtAv2EncoderContext* c,size_t* bytes) {
#if M6_DEBLOCK
    if (c->config.tree_refinement == 3u) {
        uint64_t edges = 0, changed = 0;
        const uint32_t q = (uint32_t)c->config.qp + (c->config.bit_depth == 10u ? 48u : 0u);
        SvtAv2Error error = svt_av2_deblock_luma_apply(&c->recon, c->coded_blocks, c->coded_block_count,
            3u, 1, 1, q, &edges, &changed);
        if (error != SVT_AV2_ERROR_NONE) return error;
        return ccso_auto_encode_base(c, bytes, 3u, 1, 1, 256u, 0u, 0, gdf_key_code(c));
    }
    const size_t raw_size=(size_t)c->frame_width*c->frame_height*3u/2u*(c->config.bit_depth>8u?2u:1u);
    uint8_t* raw=(uint8_t*)malloc(raw_size);
    uint8_t* best_recon=(uint8_t*)malloc(raw_size);
    uint8_t* best_packet=NULL;
    if (!raw || !best_recon) {free(raw);free(best_recon);return SVT_AV2_ERROR_OUT_OF_MEMORY;}
    db_recon_copy(c,raw,0);
    SvtAv2Error error=SVT_AV2_ERROR_NONE;
    uint64_t best_cost=UINT64_MAX,off_cost=0u,best_hash=0u;
    size_t best_bytes=0u;uint32_t winner=0u;
    int forced=-1;
#if M6_DEBLOCK_TEST
    const char* value=getenv("SVT_AV2_TEST_DEBLOCK_INDEX");
    if (value) {char* end=NULL;const long parsed=strtol(value,&end,10);
        if (!end || *end || parsed<0 || parsed>24) {error=SVT_AV2_ERROR_INVALID_PARAMETER;goto done;}
        forced=(int)parsed;
    }
#endif
    static const int deltas[4]={0,-1,1,-2};
    const char* fine_ru=getenv("SVT_AV2_PCWIENER_RU");
    const int try_fine=forced<0 && fine_ru && !strcmp(fine_ru,"rd");
    const char* cdef=getenv("SVT_AV2_CDEF");
    const uint32_t cdef_start=try_fine ? 27u : 25u;
    static const uint32_t primary_strengths[]={1u,2u,4u,8u};
    const int gradient_rd=cdef && !strcmp(cdef,"gradient");
    const int stack_restoration=gradient_rd || (cdef && !strcmp(cdef,"stack"));
    const int try_cdef=forced<0 && cdef && (stack_restoration || !strcmp(cdef,"rd"));
    uint32_t best_ru=256u,best_cdef=0u;
    for (uint32_t trial=0u;trial<cdef_start+(try_cdef ? SVT_AV2_ARRAY_SIZE(primary_strengths) : 0u);++trial) {
        const uint32_t primary=trial<cdef_start ? 0u : primary_strengths[trial-cdef_start];
        const uint32_t index=trial<25u ? trial : winner;
        const uint32_t ru=trial<25u || primary ? 256u : trial==25u ? 128u : 64u;
        if (forced>=0 && index && index!=(uint32_t)forced) continue;
        const uint32_t mode=!index?0u:index<=4u?1u:index<=8u?2u:3u;
        const int delta=mode==1u?deltas[index-1u]:mode==3u?deltas[(index-9u)/4u]:0;
        const int delta_h=mode==2u?deltas[index-5u]:mode==3u?deltas[(index-9u)%4u]:0;
        db_recon_copy(c,raw,1);
        uint64_t edges=0u,changed=0u;
        if (mode) error=svt_av2_deblock_luma_apply(&c->recon,c->coded_blocks,c->coded_block_count,
            mode,delta,delta_h,(uint32_t)c->config.qp+(c->config.bit_depth==10u?48u:0u),&edges,&changed);
        if (error!=SVT_AV2_ERROR_NONE) goto done;
        size_t trial_bytes=0u;
        error=ccso_auto_encode_base(c,&trial_bytes,mode,delta,delta_h,ru,primary,stack_restoration,0u);
        if (error!=SVT_AV2_ERROR_NONE) goto done;
        uint64_t cost=0u;error=db_current_cost(c,trial_bytes,gradient_rd,&cost);
        if (error!=SVT_AV2_ERROR_NONE) goto done;
        if (primary) fprintf(stderr,"CDEF_TRIAL primary=%u bytes=%zu J=%llu\n",primary,trial_bytes,(unsigned long long)cost);
        if (!trial) off_cost=cost;
        fprintf(stderr,"DEBLOCK_TRIAL index=%u mode=%u delta=%d delta_h=%d edges=%llu changed=%llu bytes=%zu J=%llu\n",
            index,mode,delta,delta_h,(unsigned long long)edges,(unsigned long long)changed,trial_bytes,(unsigned long long)cost);
        if (cost<best_cost || (forced>=0 && index==(uint32_t)forced)) {
            /* In forced tests the selected arm wins irrespective of RD. */
            if (forced>=0 && index && index!=(uint32_t)forced) continue;
            uint8_t* grown=(uint8_t*)realloc(best_packet,trial_bytes);
            if (!grown) {error=SVT_AV2_ERROR_OUT_OF_MEMORY;goto done;}
            best_packet=grown;memcpy(best_packet,c->packet,trial_bytes);
            best_bytes=trial_bytes;best_cost=cost;winner=index;best_ru=ru;best_cdef=primary;
            best_hash=ccso_auto_hash(best_packet,best_bytes);db_recon_copy(c,best_recon,0);
        }
    }
    if (try_cdef) fprintf(stderr,"CDEF_FINAL primary=%u db=%u bytes=%zu\n",best_cdef,winner,best_bytes);
    if (try_fine) fprintf(stderr,"PC_WIENER_RU winner=%u db=%u bytes=%zu\n",best_ru,winner,best_bytes);
    if (!best_packet || c->packet_capacity<best_bytes) {error=SVT_AV2_ERROR_INVALID_STATE;goto done;}
    memcpy(c->packet,best_packet,best_bytes);db_recon_copy(c,best_recon,1);*bytes=best_bytes;
    uint64_t final_cost=0u;error=db_current_cost(c,best_bytes,gradient_rd,&final_cost);
    if (error==SVT_AV2_ERROR_NONE && (final_cost!=best_cost || ccso_auto_hash(c->packet,best_bytes)!=best_hash))
        error=SVT_AV2_ERROR_INVALID_STATE;
    fprintf(stderr,"DEBLOCK_FINAL winner=%u forced=%d bytes=%zu Joff=%llu Jfinal=%llu\n",
        winner,forced,best_bytes,(unsigned long long)off_cost,(unsigned long long)best_cost);
    if (error == SVT_AV2_ERROR_NONE && c->gdf_key) {
        const uint32_t mode = !winner ? 0u : winner <= 4u ? 1u : winner <= 8u ? 2u : 3u;
        const int dv = mode == 1u ? deltas[winner-1u] : mode == 3u ? deltas[(winner-9u)/4u] : 0;
        const int dh = mode == 2u ? deltas[winner-5u] : mode == 3u ? deltas[(winner-9u)%4u] : 0;
        uint64_t edges = 0, changed = 0;
        db_recon_copy(c,raw,1);
        if (mode) error = svt_av2_deblock_luma_apply(&c->recon,c->coded_blocks,c->coded_block_count,
            mode,dv,dh,(uint32_t)c->config.qp+(c->config.bit_depth==10u?48u:0u),&edges,&changed);
        if (error == SVT_AV2_ERROR_NONE)
            error = ccso_auto_encode_base(c,bytes,mode,dv,dh,best_ru,best_cdef,stack_restoration,gdf_key_code(c));
    }
done:
    free(raw);free(best_recon);free(best_packet);return error;
#else
    return ccso_auto_encode_base(c,bytes,0u,0,0,256u,0u,0,gdf_key_code(c));
#endif
}

#include "satd_packet.inc"

static uint32_t motion_precision(uint32_t motion_optimization) {
    return motion_optimization == 5u ? 2u : motion_optimization >= 3u ? 1u : 0u;
}

static inline uint32_t anchor_difference(uint16_t a, uint16_t b) {
    return a >= b ? (uint32_t)(a - b) : (uint32_t)(b - a);
}

/* Use only the current source, one previously committed source signature,
 * and actual reconstructed LAST/LONG. Q8 output is in 8-bit sample units;
 * no signal is added to the bitstream apart from the existing refresh mask. */
static uint32_t anchor_probe(const SvtAv2EncoderContext* context, SvtAv2AnchorProbe* probe) {
    const SvtAv2FrameBuffer* src = &context->frame_source;
    const SvtAv2FrameBuffer* last = svt_av2_reference_view(&context->reference);
    const SvtAv2FrameBuffer* older = svt_av2_reference_view(&context->long_reference);
    const uint32_t bd = src->bit_depth, step = bd > 8u ? 2u : 1u;
    const uint32_t divisor = SVT_AV2_ANCHOR_GRID_COUNT << (bd - 8u);
    uint32_t src_sad = 0u, last_sad = 0u, long_sad = 0u;
    for (uint32_t y = 0u, i = 0u; y < SVT_AV2_ANCHOR_GRID_Y; ++y) {
        const uint32_t sy = ((2u * y + 1u) * src->visible_height) / (2u * SVT_AV2_ANCHOR_GRID_Y);
        const uint8_t* in = src->planes[0] + (size_t)sy * src->strides[0];
        const uint8_t* recent = last ? last->planes[0] + (size_t)sy * last->strides[0] : NULL;
        const uint8_t* held = older ? older->planes[0] + (size_t)sy * older->strides[0] : NULL;
        for (uint32_t x = 0u; x < SVT_AV2_ANCHOR_GRID_X; ++x, ++i) {
            const uint32_t offset = step * ((2u * x + 1u) * src->visible_width / (2u * SVT_AV2_ANCHOR_GRID_X));
            const uint16_t sample = svt_av2_sample_load(in + offset, bd);
            probe->current[i] = sample;
            if (context->anchor_source_valid) src_sad += anchor_difference(sample, context->anchor_source[i]);
            if (recent) last_sad += anchor_difference(sample, svt_av2_sample_load(recent + offset, bd));
            if (held) long_sad += anchor_difference(sample, svt_av2_sample_load(held + offset, bd));
        }
    }
    probe->source_q8 = (src_sad * 256u) / divisor;
    probe->last_q8 = (last_sad * 256u) / divisor;
    probe->long_q8 = (long_sad * 256u) / divisor;
    probe->age = older ? (uint32_t)(context->pictures_encoded - context->long_reference.frame_index) : 0u;
    if (context->pictures_encoded == 0u) { probe->reason = 0u; return 0u; }
    if (!older) {
        const int seed = context->pictures_encoded == 4u;
        probe->reason = seed ? 2u : 1u;
        return seed ? 3u : 1u;
    }
    const int stale = long_sad >= last_sad + divisor;
    if (probe->age >= 8u) { probe->reason = 6u; return 3u; }
    if (probe->age >= 2u && src_sad >= 3u * divisor && long_sad >= 4u * divisor && stale) {
        probe->reason = 4u; return 3u;
    }
    if (probe->age >= 4u && (2u * src_sad >= divisor || stale)) {
        probe->reason = 5u; return 3u;
    }
    probe->reason = 3u;
    return 1u;
}

static uint32_t lookahead_display(const SvtAv2EncoderContext* context) {
    return (uint32_t)context->frame_pts % SVT_AV2_LOOKAHEAD_KEY_INTERVAL;
}

/* Screen the zero-motion compound candidate against both single predictors
 * locally. The stricter margin reserves moving, textured frames for ME. */
static int compare_reference_pair(const SvtAv2FrameBuffer* src, const SvtAv2PictureReference* refs, uint64_t* cost) {
    const SvtAv2FrameBuffer* a = svt_av2_reference_view(refs->predictor);
    const SvtAv2FrameBuffer* b = svt_av2_reference_view(refs->secondary_reference);
    const unsigned step = src->bit_depth > 8u ? 2u : 1u;
    uint64_t single = 0, blend = 0, selected = 0;
    for (unsigned by = 0; by < src->visible_height; by += 32u) for (unsigned bx = 0; bx < src->visible_width; bx += 32u) {
        unsigned da = 0, db = 0, mixed = 0;
        for (unsigned y = by; y < by+32u && y < src->visible_height; y += 8u) {
            for (unsigned x = bx; x < bx+32u && x < src->visible_width; x += 8u) {
                const int v = svt_av2_sample_load(src->planes[0]+(size_t)y*src->strides[0]+x*step, src->bit_depth);
                const int l = svt_av2_sample_load(a->planes[0]+(size_t)y*a->strides[0]+x*step, src->bit_depth);
                const int r = svt_av2_sample_load(b->planes[0]+(size_t)y*b->strides[0]+x*step, src->bit_depth);
                da += (unsigned)abs(v-l);
                db += (unsigned)abs(v-r);
                mixed += (unsigned)abs(v-((l+r+1)/2));
            }
        }
        const unsigned best = SVT_AV2_MIN(da,db);
        single += best;
        blend += mixed;
        selected += refs->ref_distance[0]==refs->ref_distance[1] ? SVT_AV2_MIN(best,mixed) : best;
    }
    if (cost) *cost=selected;
    return 10u*blend < 9u*single;
}

static SvtAv2ReferenceFrame* pyramid_slot(SvtAv2EncoderContext* c, unsigned slot) {
    if (!slot) return &c->reference;
    if (slot == 1u) return &c->long_reference;
    return &c->video_cdf->extra_ref[slot-2u];
}

static SvtAv2PictureReference picture_reference(SvtAv2EncoderContext* context, SvtAv2AnchorProbe* probe) {
    SvtAv2PictureReference refs = {0};
    if (context->config.prediction_structure != SVT_AV2_PRED_LOW_DELAY_P) return refs;
    refs.dual_ref = context->dual_ref_period != 0u;
    if (context->dual_ref_cadence) refs.refresh_mask = anchor_probe(context, probe);
    if (context->pictures_encoded == 0u) return refs; /* CLK refreshes all slots implicitly. */
    refs.predictor = &context->reference;
    if (context->motion_q == 20u) {
        const unsigned i = context->pictures_encoded, pts = lookahead_display(context);
        const int nested = i <= context->anchor_display;
        const int outer = !nested && pts > i;
        refs.reference_slot = context->latest_slot;
        refs.refresh_slot = (pts & 7u) ? (context->latest_slot ^ 1u) : svt_av2_pyramid8_slot(pts);
        refs.refresh_mask = 1u;
        refs.ref_distance[0] = pts - context->anchor_display;
        if (context->drawing_runs && !nested && !outer) {
            refs.refresh_slot = context->latest_slot ^ 1u;
            uint32_t distances[2];
            SvtAv2ReferenceFrame* older = pyramid_slot(context, refs.refresh_slot);
            if (older->valid && svt_av2_mvp_two_past_distances(pts, context->anchor_display,
                    context->previous_anchor_display, distances)) {
                refs.secondary_slot = refs.refresh_slot;
                refs.secondary_reference = older;
                memcpy(refs.ref_distance, distances, sizeof(distances));
            }
        }
        if (outer) {
            refs.paired_role = SVT_AV2_HIDDEN_ANCHOR;
            refs.refresh_slot = context->latest_slot ^ 1u;
        }
        if (nested) {
            const unsigned span = context->pyramid_span;
            const unsigned begin = context->anchor_display-span, pos = pts-begin;
            const unsigned prev = svt_av2_pyramid_offset(span,i-begin-2u);
            const SvtAv2PyramidNode node = svt_av2_pyramid_node(span,i-begin-1u,context->latest_slot);
            refs.reference_slot = node.right_slot;
            refs.secondary_slot = node.left_slot;
            refs.secondary_reference = pyramid_slot(context,refs.secondary_slot);
            refs.ref_distance[0] = node.right-pos;
            refs.ref_distance[1] = pos-node.left;
            refs.refresh_slot = node.slot;
            refs.refresh_mask = !(pos & 1u);
            refs.paired_role = refs.refresh_mask ? (prev & 1u ? SVT_AV2_HIDDEN_BRANCH : SVT_AV2_HIDDEN_MIDPOINT) :
                (prev & 1u ? SVT_AV2_LAST_MIDPOINT : SVT_AV2_ONLY_MIDPOINT);
        }
        refs.predictor = pyramid_slot(context, refs.reference_slot);
        const char* bank = getenv("SVT_AV2_REF_SEARCH");
        if (nested && bank && !strcmp(bank,"past-bank")) {
            uint64_t best;
            compare_reference_pair(&context->frame_source,&refs,&best);
            for (unsigned slot=0;slot<context->adaptive_pyramid+2u;++slot) {
                SvtAv2ReferenceFrame* ref=pyramid_slot(context,slot);
                if (!ref->valid || slot==refs.secondary_slot || context->slot_display[slot]>=pts ||
                    pts-context->slot_display[slot]>32u) continue;
                SvtAv2PictureReference trial=refs;
                trial.secondary_slot=slot;
                trial.secondary_reference=ref;
                trial.ref_distance[1]=pts-context->slot_display[slot];
                uint64_t cost;
                compare_reference_pair(&context->frame_source,&trial,&cost);
                if (cost*16u<best*15u) { refs=trial; best=cost; }
            }
        }
        refs.slot_mask = refs.refresh_mask ? 1u << refs.refresh_slot : 0u;
        return refs;
    }
    if (context->motion_q >= 13u) {
        const uint32_t pts = lookahead_display(context);
        const uint32_t anchor_pts = context->anchor_display;
        if (context->motion_q >= 18u && context->tree_full &&
            context->pictures_encoded <= anchor_pts && anchor_pts - context->pictures_encoded <= 2u) {
            const uint32_t remaining = anchor_pts - context->pictures_encoded;
            refs.reference_slot = context->latest_slot;
            refs.refresh_mask = remaining == 2u;
            refs.slot_mask = refs.refresh_mask ? 1u << (refs.reference_slot ^ 1u) : 0u;
            if (remaining == 1u) {
                refs.reference_slot ^= 1u;
                refs.predictor = &context->long_reference;
                refs.ref_distance[0] = 1u;
            } else {
                refs.secondary_reference = &context->long_reference;
                refs.ref_distance[0] = remaining == 2u ? 2u : 1u;
                refs.ref_distance[1] = refs.ref_distance[0];
            }
            return refs;
        }
        refs.ref_distance[0] = pts > anchor_pts ? pts-anchor_pts : anchor_pts-pts;
        refs.slot_mask = refs.refresh_mask = pts > context->pictures_encoded ||
            (context->motion_q >= 16u && pts == context->pictures_encoded);
        if (context->motion_q >= 14u) {
            refs.reference_slot = context->latest_slot;
            refs.rotate_anchor = refs.refresh_mask;
            refs.slot_mask = refs.rotate_anchor ? 1u << (1u ^ refs.reference_slot) : 0u;
            if (pts < context->pictures_encoded && pts - context->previous_anchor_display <= 8u) {
                refs.secondary_reference = &context->long_reference;
                refs.ref_distance[1] = pts - context->previous_anchor_display;
            }
        }
        return refs;
    }
    if (context->motion_q == 7u)
        refs.refresh_mask = (context->pictures_encoded & 1u) ? 0u :
            (context->pictures_encoded & 3u) ? 1u : 3u;
    if (context->motion_q >= 8u) {
        /* Keep the eighth picture as a long reference. The middle and
         * quarter points refresh LAST, odd pictures are display-only. */
        const unsigned phase = (unsigned)context->pictures_encoded & 7u;
        refs.refresh_mask = (phase & 1u) ? 0u : phase ? 1u : 3u;
        /* Retain the newest predictor when an odd picture contains motion.
         * Use the same source-grid scale as adaptive anchor selection. */
        if (context->motion_q >= 9u && !refs.refresh_mask &&
            probe->source_q8 >= (context->motion_q == 12u ? 8u*256u : 384u))
            refs.refresh_mask = 1u;
    }
    if (context->drawing_cadence) refs.refresh_mask = 1u;
    if (refs.dual_ref) {
        if (!context->dual_ref_cadence)
            refs.refresh_mask = context->pictures_encoded % context->dual_ref_period ? 1u : 3u;
        /* CLK's mask covers both slots but the decoder deduplicates the KEY
         * buffer to slot 0. Only an explicit periodic P refresh seeds slot 1. */
        if (context->dual_ref_policy == 3u) {
            uint32_t distance[2];
            /* A periodic refresh temporarily makes both slots aliases. Test
             * before refresh and never map an alias or an empty slot to rank 1. */
            if (context->reference.valid && context->long_reference.valid &&
                svt_av2_mvp_two_past_distances(context->pictures_encoded, context->reference.frame_index,
                    context->long_reference.frame_index, distance) && distance[0] < distance[1] &&
                (distance[0] == 1u || context->motion_q >= 7u)) {
                refs.secondary_reference = &context->long_reference;
                refs.ref_distance[0] = distance[0];
                refs.ref_distance[1] = distance[1];
            }
        } else {
            if (context->long_reference.valid)
                refs.reference_slot = context->dual_ref_policy ? context->dual_ref_policy - 1u :
                    svt_av2_reference_choose_slot(&context->frame_source, &context->reference, &context->long_reference);
            if (refs.reference_slot) refs.predictor = &context->long_reference;
        }
    } else {
        refs.retain_reference = context->inter_ref_period > 1u &&
            context->pictures_encoded % context->inter_ref_period != 0u;
    }
    refs.slot_mask = refs.refresh_mask;
    if (context->motion_q >= 11u) {
        refs.reference_slot = context->latest_slot;
        refs.rotate_anchor = refs.refresh_mask && !(context->reference.frame_index & 7u);
        if (context->drawing_cadence)
            refs.rotate_anchor = !context->long_reference.valid || probe->age >= 8u;
        refs.slot_mask = refs.refresh_mask ? 1u << (refs.reference_slot ^ refs.rotate_anchor) : 0u;
    }
    if (!refs.secondary_reference)
        refs.ref_distance[0] = (uint32_t)(context->pictures_encoded - refs.predictor->frame_index);
    return refs;
}

static void record_block_ref_stats(SvtAv2EncoderContext* context, const SvtAv2PictureReference* refs) {
    SvtAv2BlockRefStats stats = {0};
    stats.refresh_mask = refs->slot_mask;
    if (refs->predictor != NULL) {
        const int two = refs->secondary_reference != NULL;
        stats.logical_ref_count = two ? 2u : 1u;
        stats.ref_distance[0] = refs->ref_distance[0];
        stats.ref_distance[1] = refs->ref_distance[1];
        for (size_t i = 0u; i < context->coded_block_count / 3u; ++i) {
            const SvtAv2TileBlockMode* mode = &context->coded_modes[i];
            if (!mode->is_inter) continue;
            /* The formal writer has already validated both the bitstream and rank. */
            const uint32_t rank = two ? mode->reference_rank : 0u;
            if (rank == 2u) { ++stats.rank_leaves[0]; ++stats.rank_leaves[1]; continue; }
            ++stats.rank_leaves[rank];
            if (two && (mode->mv_x % 2 != 0 || mode->mv_y % 2 != 0))
                ++stats.odd_mv_leaves[rank];
        }
    }
    context->block_ref_stats = stats;
}

/* Compare against the last coded source so gradual drawing changes accumulate.
 * Guard local luma changes and test chroma against the actual displayed pixels. */
static int can_show_last(const SvtAv2EncoderContext* c, const SvtAv2AnchorProbe* probe) {
    if (c->pictures_encoded < 5u || c->held_pictures >= 8u || probe->source_q8 >= 192u)
        return 0;
    const SvtAv2FrameBuffer* src = &c->frame_source;
    const uint32_t scale = 1u << (src->bit_depth - 8u);
    uint32_t changed = 0u;
    for (uint32_t i = 0; i < SVT_AV2_ANCHOR_GRID_COUNT; ++i) {
        if (anchor_difference(probe->current[i], c->anchor_source[i]) > 6u * scale && ++changed > 9u)
            return 0;
    }
    for (uint32_t p = 1u; p < 3u; ++p) {
        uint32_t sad = 0u;
        const uint32_t width = svt_av2_picture_plane_width(src, p);
        const uint32_t height = svt_av2_picture_plane_height(src, p);
        for (uint32_t y = 0; y < 18u; ++y) for (uint32_t x = 0; x < 32u; ++x) {
            const uint32_t row = (2u * y + 1u) * height / 36u;
            const size_t col = (2u * x + 1u) * width / 64u * (src->bit_depth > 8u ? 2u : 1u);
            const uint32_t d = anchor_difference(
                svt_av2_sample_load(src->planes[p] + row * (size_t)src->strides[p] + col, src->bit_depth),
                svt_av2_sample_load(c->recon.planes[p] + row * (size_t)c->recon.strides[p] + col, src->bit_depth));
            if (d > 16u * scale) return 0;
            sad += d;
        }
        if (sad > 2u * scale * 18u * 32u) return 0;
    }
    return 1;
}

static SvtAv2Error show_reference(SvtAv2EncoderContext* c, uint32_t slot, uint32_t pts, int first, int after_slot) {
    /* MFH and sequence IDs 0 (two uvlc(0)); physical LAST slot, explicit 8-bit hint. */
    uint8_t payload[2];
    SvtAv2BitWriter header, packet;
    svt_av2_bit_writer_init(&header, payload, sizeof(payload));
    svt_av2_bit_writer_init(&packet, c->packet, c->packet_capacity);
    const unsigned bits = c->adaptive_pyramid >= 3u ? 3u : c->motion_q == 20u ? 2u : 1u;
    if (svt_av2_bit_writer_write_bits(&header, (6u << bits) | (slot << 1u), bits+3u) ||
        svt_av2_bit_writer_write_bits(&header, pts & 255u, 8u) ||
        svt_av2_obu_payload_finalize(&header) ||
        (!first && svt_av2_obu_write(&packet, SVT_AV2_OBU_TEMPORAL_DELIMITER, 0u, NULL, 0u)) ||
        svt_av2_obu_write(&packet, SVT_AV2_OBU_REGULAR_SEF, 0u, payload, sizeof(payload)))
        return SVT_AV2_ERROR_INTERNAL;
    if (after_slot >= 0) {
        const uint8_t overlay = (uint8_t)(0xc0u | ((unsigned)after_slot << (6u-bits)) | (3u << (4u-bits)));
        if (svt_av2_obu_write(&packet, SVT_AV2_OBU_TEMPORAL_DELIMITER, 0u, NULL, 0u) ||
            svt_av2_obu_write(&packet, SVT_AV2_OBU_REGULAR_SEF, 0u, &overlay, 1u))
            return SVT_AV2_ERROR_INTERNAL;
    }
    memset(&c->block_ref_stats, 0, sizeof(c->block_ref_stats));
    memset(&c->rd_stats, 0, sizeof(c->rd_stats));
    c->inter_skip_trials = c->inter_skip_accepts = 0;
    c->inter_fine_trials = c->inter_fine_accepts = 0;
    c->packet_size = svt_av2_bit_writer_bytes_written(&packet);
    c->recon_frame_index = c->pictures_sent - 1u;
    ++c->pictures_encoded;
    c->packet_pending = 1;
    c->frame_pending = 0;
    ++c->held_pictures;
    return SVT_AV2_ERROR_NONE;
}

/* Compare the actual coded P packet against a 6-byte show-existing packet
 * before publishing its pixels, reference slots or inherited entropy bank. */
static SvtAv2Error display_cost(const SvtAv2EncoderContext* c, const SvtAv2FrameBuffer* pixels,
        size_t bytes, uint64_t* cost) {
    const uint32_t depth = c->frame_source.bit_depth, step = depth > 8u ? 2u : 1u;
    uint64_t sse[3] = {0};
    const char* detail=getenv("SVT_AV2_DETAIL_RD");
    const int contrast=detail && !strcmp(detail,"inter-contrast");
    if (contrast) {
        SvtAv2BlockView view={0,0,pixels->visible_width,pixels->visible_height,NULL};
        sse[0]=svt_av2_contrast_distortion(&c->frame_source,pixels,&view);
    }
    for (uint32_t p = (unsigned)contrast; p < 3; ++p) {
        const uint32_t width = svt_av2_picture_plane_width(pixels, p);
        const uint32_t height = svt_av2_picture_plane_height(pixels, p);
        for (uint32_t y = 0; y < height; ++y) {
            const uint8_t* ref = pixels->planes[p] + (size_t)y * pixels->strides[p];
            const uint8_t* src = c->frame_source.planes[p] + (size_t)y * c->frame_source.strides[p];
            for (uint32_t x = 0; x < width; ++x) {
                const int delta = (int)svt_av2_sample_load(src + x * step, depth) -
                    (int)svt_av2_sample_load(ref + x * step, depth);
                sse[p] += (uint64_t)(delta * delta);
            }
        }
    }
    return ccso_auto_cost(c, sse, (uint64_t)bytes * 32768u, cost);
}

/* A hidden anchor and its first midpoint share one temporal unit. Only the
 * last midpoint appends a second unit to display the anchor at its own PTS. */
static SvtAv2Error select_reference_display(SvtAv2EncoderContext* c, size_t bytes, int midpoint, const SvtAv2PictureReference* refs, int* used) {
    const unsigned role = midpoint ? c->video_cdf->frame_header.paired_role : 0u;
    const int first = role == SVT_AV2_FIRST_MIDPOINT || role == SVT_AV2_ONLY_MIDPOINT;
    const uint32_t pts = c->motion_q >= 13u ? lookahead_display(c) : c->pictures_encoded;
    const int last = role == SVT_AV2_LAST_MIDPOINT || role == SVT_AV2_ONLY_MIDPOINT;
    const size_t overlay_bytes = 6u - 2u * first + 5u * last;
    uint64_t best;
    SvtAv2Error err = display_cost(c, &c->recon, bytes, &best);
    if (err) return err;
    const SvtAv2FrameBuffer* selected = NULL;
    uint32_t slot = c->latest_slot;
    const unsigned count = c->display_all_slots ? c->video_cdf->frame_header.pyramid_slots : 1u+(unsigned)midpoint;
    for (unsigned rank = 0; rank < count; ++rank) {
        const SvtAv2ReferenceFrame* stored = c->display_all_slots ? pyramid_slot(c,rank) : c->motion_q == 20u ? (rank ? refs->secondary_reference : refs->predictor) :
            (rank ? &c->long_reference : &c->reference);
        const SvtAv2FrameBuffer* ref = stored ? svt_av2_reference_view(stored) : NULL;
        if (!ref) continue;
        uint64_t cost;
        err = display_cost(c, ref, overlay_bytes, &cost);
        if (err) return err;
        if (cost <= best) {
            best = cost;
            selected = ref;
            slot = c->display_all_slots ? rank : c->motion_q == 20u ? (rank ? refs->secondary_slot : refs->reference_slot) : c->latest_slot ^ rank;
        }
    }
    *used = selected != NULL;
    if (!selected) return SVT_AV2_ERROR_NONE;
    err = svt_av2_picture_copy_in(&c->recon, selected);
    return err ? err : show_reference(c, slot, pts, first, last ? (int)c->video_cdf->frame_header.reference_slot : -1);
}

#include "still_columns.inc"

static SvtAv2Error encode_pending_frame(SvtAv2EncoderContext* context) {
    if (context->packet_pending || !context->frame_pending) {
        return SVT_AV2_ERROR_NONE;
    }

    if (context->fatal_error) return SVT_AV2_ERROR_INTERNAL;
    SVT_AV2_IP_START(profile_start);
    const uint32_t width = context->frame_width, height = context->frame_height;
    SvtAv2AnchorProbe anchor;
    const SvtAv2PictureReference refs = picture_reference(context, &anchor);
    const int disposable = refs.dual_ref && refs.predictor && !refs.refresh_mask;
    const uint32_t retain_reference = refs.retain_reference || disposable;
    int32_t qp = context->config.qp;
    if (context->pictures_encoded == 0u && context->video_key_qp)
        qp = context->video_key_qp;
    else if ((refs.refresh_mask & 2u) && context->video_anchor_qp)
        qp = context->video_anchor_qp;
    /* Allocate fewer bits to the intermediate P pictures between long
     * references. Keep the first picture after a static hold at anchor Q. */
    if (context->motion_q == 7u && disposable) qp = SVT_AV2_MIN(255, qp + 16);
    if (context->motion_q >= 8u && refs.predictor && !(refs.refresh_mask & 2u)) {
        const unsigned phase = (unsigned)context->pictures_encoded & 7u;
        qp = disposable ? SVT_AV2_MIN(255, qp + 12) :
            phase == 4u ? (qp + context->video_anchor_qp) / 2 : qp;
    }
    if (context->motion_q == 10u) {
        /* Scale the AC step at each temporal level, then map it back to the
         * native quantizer. A motion-refreshed odd P has a short lifetime. */
        const unsigned phase = context->pictures_encoded & 7u;
        const unsigned percent = !refs.predictor ? 24u : phase == 0u ? 40u :
            phase == 4u ? 70u : !(phase & 1u) ? 85u : disposable ? 100u : 92u;
        const int offset = context->config.bit_depth == SVT_AV2_BIT_DEPTH_10 ? 48 : 0;
        const int ceiling = context->config.qp;
        const int target = (svt_av2_quant_qlookup(ceiling + offset) * (int)percent + 50) / 100;
        qp = ceiling;
        while (qp > 1 && svt_av2_quant_qlookup(qp - 1 + offset) >= target) --qp;
    }
    if (context->motion_q == 6u && refs.predictor && !(refs.refresh_mask & 2u)) {
        if (context->held_pictures && anchor.source_q8 >= 384u)
            qp = context->video_anchor_qp;
        else
            qp = SVT_AV2_MIN(255, qp + (context->pictures_encoded & 1u ? 12 : 4));
    }
    if (context->drawing_cadence && refs.predictor) {
        /* Give a new drawing following a hold reference quality. Nearly unchanged
         * inputs still compete with show-existing using full-frame distortion;
         * a small moving outline can be invisible to the source grid. */
        qp = anchor.source_q8 < 48u ? SVT_AV2_MIN(255, context->config.qp + 12) :
            context->held_pictures ? (context->config.qp + context->video_anchor_qp) / 2 :
            refs.rotate_anchor ? context->video_anchor_qp : context->config.qp;
    }
    int clean = 0;
    if ((context->motion_q == 3u || context->motion_q == 4u) && refs.predictor) {
        const int mask_motion = context->motion_q == 4u;
        const int can_hold = can_show_last(context, &anchor);
        clean = mask_motion && can_hold && context->previous_motion_q8 >= 5u * 256u &&
            anchor.last_q8 >= 384u && context->pictures_encoded - context->last_clean_frame >= 8u;
        if (can_hold && !clean) return show_reference(context, context->latest_slot, context->pictures_encoded, 0, -1);
        if (mask_motion && anchor.source_q8 >= 5u * 256u)
            qp = context->config.qp;
        else if (clean || context->held_pictures || anchor.source_q8 >= 384u)
            qp = context->video_anchor_qp;
    }
    if (context->motion_q == 1u && refs.predictor) {
        /* Spend the first quiet frame after motion on a clean prediction anchor.
         * Do not spend the same anchor budget on every refresh in a held shot. */
        const uint32_t motion = anchor.source_q8;
        if (motion < 384u && context->previous_motion_q8 >= 384u)
            qp = SVT_AV2_MAX(1, context->video_anchor_qp - 18);
        else if (motion < 192u)
            qp = SVT_AV2_MIN(255, context->config.qp + 12);
        else if (motion >= 5u * 256u)
            qp = SVT_AV2_MIN(255, context->config.qp + 18);
    }
    if (context->motion_q == 2u && refs.predictor) {
        const uint32_t motion = anchor.source_q8;
        /* A changed animation drawing may be followed by several holds. Spend
         * on the change itself, and avoid buying a second expensive anchor
         * for the next, nearly identical drawing. */
        clean = motion < 384u && context->quiet_frames == 3u &&
            context->pictures_encoded - context->last_clean_frame >= 16u;
        if (clean)
            qp = SVT_AV2_MAX(1, context->video_anchor_qp - 12);
        else if (motion < 192u)
            qp = SVT_AV2_MIN(255, context->config.qp + 16);
        else if (motion >= 384u && context->previous_motion_q8 < 384u)
            qp = context->video_anchor_qp;
        else if (motion >= 5u * 256u)
            qp = context->config.qp;
    }
    if (context->motion_q >= 13u && refs.predictor) {
        const unsigned phase = lookahead_display(context) & 7u;
        qp = !phase ? context->video_anchor_qp : phase == 4u ?
            (context->config.qp + context->video_anchor_qp)/2 :
            (phase & 1u) ? SVT_AV2_MIN(255, context->config.qp+12) : context->config.qp;
    }
    if (context->motion_q >= 18u && refs.predictor)
        qp = disposable ? SVT_AV2_MIN(255, context->config.qp + 12) :
            lookahead_display(context) > context->pictures_encoded ? context->video_anchor_qp : context->config.qp;
    if (context->motion_q == 20u && refs.predictor) {
        const unsigned pts = lookahead_display(context);
        const int delta = !refs.paired_role ? -6 : !(pts & 7u) ? -28 : !(pts & 3u) ? -16 : !(pts & 1u) ? -4 : 12;
        qp = SVT_AV2_MAX(1, SVT_AV2_MIN(255, context->config.qp + delta));
        if (context->adaptive_pyramid == 3u && refs.paired_role) {
            const int shift = !(pts & 15u) ? -36 : !(pts & 7u) ? -24 : !(pts & 3u) ? -12 : !(pts & 1u) ? -3 : 12;
            qp = SVT_AV2_MAX(1, SVT_AV2_MIN(255, context->config.qp+shift));
        }
        if (context->adaptive_pyramid == 4u && refs.paired_role) {
            const unsigned reach = refs.paired_role == SVT_AV2_HIDDEN_ANCHOR ?
                pts-context->pictures_encoded+1u : SVT_AV2_MAX(refs.ref_distance[0],refs.ref_distance[1]);
            const int shift = reach >= 16u ? -36 : reach >= 8u ? -24 : reach >= 4u ? -12 : reach >= 2u ? -3 : 12;
            qp = SVT_AV2_MAX(1,SVT_AV2_MIN(255,context->config.qp+shift));
        }
        if (context->motion_layers && refs.paired_role) {
            const unsigned span = refs.paired_role == SVT_AV2_HIDDEN_ANCHOR ?
                pts-context->pictures_encoded+1u : refs.ref_distance[0];
            const unsigned percent = span >= 8u ? 46u : span >= 4u ? 62u : span >= 2u ? 80u : 100u;
            const int offset = context->config.bit_depth == SVT_AV2_BIT_DEPTH_10 ? 48 : 0;
            qp = SVT_AV2_MIN(255, context->config.qp + 16);
            const int target = (svt_av2_quant_qlookup(qp+offset)*(int)percent+50)/100;
            while (qp > 1 && svt_av2_quant_qlookup(qp-1+offset) >= target) --qp;
        }
        if (context->adaptive_pyramid && !refs.paired_role)
            qp = anchor.source_q8 < 48u ? SVT_AV2_MIN(255, context->config.qp+12) :
                context->held_pictures ? (context->config.qp + context->video_anchor_qp)/2 : context->config.qp;
    }
    if (context->motion_q == 17u && refs.predictor)
        qp = disposable ? SVT_AV2_MIN(255, context->config.qp + 12) : (context->config.qp + context->video_anchor_qp) / 2;
    if (context->picture_qp) qp = context->picture_qp;
    const uint32_t base_q_idx = (uint32_t)qp + (context->config.bit_depth == SVT_AV2_BIT_DEPTH_10 ? 48u : 0u);
    const int use_ccso_search = ccso_config_supported(&context->config) && width >= 8u && width <= 4096u &&
        height >= 8u && height <= 2160u && !(width & 7u) && !(height & 7u);
    if (context->tile_columns > 1u || context->still_aq || context->block_gdf) {
        if (!use_ccso_search || width < context->tile_columns * 128u || height <= 64u) return SVT_AV2_ERROR_UNSUPPORTED;
        size_t size = 0;
        SvtAv2Error e = encode_still_columns(context, base_q_idx, &size);
        if (e != SVT_AV2_ERROR_NONE) { context->fatal_error = 1; return e; }
        context->recon_frame_index = context->pictures_encoded++;
        context->packet_size = size;
        context->packet_pending = 1;
        context->frame_pending = 0;
        return SVT_AV2_ERROR_NONE;
    }
    if (context->gdf_key && !use_ccso_search && !context->video_cdf)
        return SVT_AV2_ERROR_UNSUPPORTED;
    SVT_AV2_IP_START(profile_recon);
    const SvtAv2Error recon_err = encode_reconstruction(context, width, height, base_q_idx, &refs);
    SVT_AV2_IP_START(profile_writer);
    if (recon_err != SVT_AV2_ERROR_NONE) {
        context->fatal_error = 1;
        return recon_err;
    }
    if (context->video_cdf && context->video_cdf->frame_header.deblock_y) {
        const SvtAv2FrameHeaderConfig* h=&context->video_cdf->frame_header;
        uint64_t edges,changed;
        const SvtAv2Error filtered=svt_av2_deblock_luma_apply(&context->recon,context->coded_blocks,
            context->coded_block_count,h->deblock_y,h->deblock_delta,h->deblock_delta_h,base_q_idx,&edges,&changed);
        if (filtered) { context->fatal_error=1; return filtered; }
    }
    if (context->video_cdf && context->video_cdf->frame_header.gdf_code) {
        uint16_t* guide = svt_av2_gdf_copy_luma(&context->recon);
        const uint32_t distance = SVT_AV2_MAX(refs.ref_distance[0], refs.ref_distance[1]);
        uint32_t* code=&context->video_cdf->frame_header.gdf_code;
        const char* search=getenv("SVT_AV2_GDF_SEARCH");
        SvtAv2Error filtered=guide ? SVT_AV2_ERROR_NONE : SVT_AV2_ERROR_OUT_OF_MEMORY;
        if (!filtered && search && !strcmp(search,"sampled"))
            filtered=svt_av2_gdf_select(&context->recon,&context->frame_source,guide,base_q_idx,distance,code);
        if (!filtered && *code)
            filtered=svt_av2_gdf_luma(&context->recon,guide,guide,base_q_idx,*code,distance,1u);
        free(guide);
        if (filtered != SVT_AV2_ERROR_NONE) { context->fatal_error = 1; return filtered; }
    }
    size_t stream_size = 0;
    if (use_ccso_search) {
        const SvtAv2Error error=ccso_auto_encode(context,&stream_size);
        if (error!=SVT_AV2_ERROR_NONE) {context->fatal_error=1;return error;}
#if M6_SATD_ANGLES
        const SvtAv2Error satd_error=satd_packet(context,&stream_size);
        if (satd_error!=SVT_AV2_ERROR_NONE) {context->fatal_error=1;return satd_error;}
#endif
    } else {
    size_t capacity = svt_av2_tile_writer_max_payload_bytes(width, height) + 256u;
    if (capacity < 4096u) capacity = 4096u;
    int status;
    for (;;) {
        if (context->packet_capacity < capacity) {
            uint8_t* grown = (uint8_t*)realloc(context->packet, capacity);
            if (grown == NULL) { context->fatal_error = 1; return SVT_AV2_ERROR_OUT_OF_MEMORY; }
            context->packet = grown;
            context->packet_capacity = capacity;
        }
        if (context->video_cdf != NULL) {
            SvtAv2VideoCdfState* cdf = context->video_cdf;
            const SvtAv2TileBlockMode* modes = refs.predictor ? context->coded_modes : NULL;
            status = svt_av2_encode_packet_video_models(&cdf->frame_header,
                context->motion_q >= 13u ? lookahead_display(context) : context->pictures_encoded,
                context->coded_blocks, context->coded_block_count, modes,
                modes ? context->coded_block_count / 3u : 0u,
                context->packet, context->packet_capacity, &stream_size, &cdf->writer_end);
        } else if (context->config.still_picture && (context->intra_ist_scope || context->still_edge_filter || context->enable_ibp || context->uv_delta_q || context->path_state.qm_level)) {
            status = svt_av2_encode_still_picture_ccso64_packet(&context->still_tools_header,
                context->coded_blocks, context->coded_block_count, context->packet,
                context->packet_capacity, &stream_size);
        } else if (context->config.still_picture) {
            status = svt_av2_encode_still_picture_mhccp(width, height,
                (uint32_t)context->config.bit_depth, (uint32_t)context->config.chroma_format,
                base_q_idx,
                context->coded_blocks, context->coded_block_count, context->config.intra_mode_selection,
                context->partition_decision_count ? context->partition_decisions : NULL,
                context->partition_decision_count, context->config.entropy_adaptation,
                context->config.transform_selection, context->config.mrl_search,
                context->config.mhccp_search, context->packet, context->packet_capacity, &stream_size);
        } else if (refs.secondary_reference != NULL) {
            status = svt_av2_encode_packet_two_refs_projected(width, height,
                (uint32_t)context->config.bit_depth, (uint32_t)context->config.chroma_format,
                base_q_idx, context->pictures_encoded, context->coded_blocks, context->coded_block_count,
                context->coded_modes, context->coded_block_count / 3u, context->config.intra_mode_selection,
                context->config.partition_selection ? context->partition_decisions : NULL,
                context->config.partition_selection ? context->partition_decision_count : 0u,
                context->config.entropy_adaptation, motion_precision(context->config.motion_optimization),
                1u, refs.refresh_mask, refs.ref_distance,
                context->packet, context->packet_capacity, &stream_size);
        } else if (refs.dual_ref) {
            const int inter = refs.predictor != NULL;
            status = svt_av2_encode_packet_dual_ref(width, height,
                (uint32_t)context->config.bit_depth, (uint32_t)context->config.chroma_format,
                base_q_idx, context->pictures_encoded, context->coded_blocks, context->coded_block_count,
                inter ? context->coded_modes : NULL, inter ? context->coded_block_count / 3u : 0u,
                context->config.intra_mode_selection,
                context->config.partition_selection ? context->partition_decisions : NULL,
                context->config.partition_selection ? context->partition_decision_count : 0u,
                context->config.entropy_adaptation, inter ? motion_precision(context->config.motion_optimization) : 0u,
                inter ? 1u : 0u, refs.refresh_mask, refs.reference_slot,
                context->packet, context->packet_capacity, &stream_size);
        } else if (retain_reference || (context->config.motion_optimization >= 4u && context->pictures_encoded != 0u)) {
            status = svt_av2_encode_packet_inter_profile(width, height,
                (uint32_t)context->config.bit_depth, (uint32_t)context->config.chroma_format,
                base_q_idx,
                context->pictures_encoded, context->coded_blocks, context->coded_block_count,
                context->coded_modes, context->coded_block_count / 3u, context->config.intra_mode_selection,
                context->config.partition_selection ? context->partition_decisions : NULL,
                context->config.partition_selection ? context->partition_decision_count : 0u,
                context->config.entropy_adaptation, motion_precision(context->config.motion_optimization),
                retain_reference, context->config.motion_optimization >= 4u ? 1u : 0u,
                context->packet, context->packet_capacity, &stream_size);
        } else if (context->config.motion_optimization == 3u && context->pictures_encoded != 0u) {
            status = svt_av2_encode_packet_tools_half_entropy(width, height,
                (uint32_t)context->config.bit_depth, (uint32_t)context->config.chroma_format,
                base_q_idx,
                context->pictures_encoded, context->coded_blocks, context->coded_block_count,
                context->coded_modes, context->coded_block_count / 3u, 1u, context->config.intra_mode_selection,
                context->config.partition_selection ? context->partition_decisions : NULL,
                context->config.partition_selection ? context->partition_decision_count : 0u,
                context->config.entropy_adaptation, context->packet, context->packet_capacity, &stream_size);
        } else if (context->config.entropy_adaptation) {
            const uint32_t low_delay = context->config.prediction_structure == SVT_AV2_PRED_LOW_DELAY_P;
            status = svt_av2_encode_packet_tools_entropy(width, height,
                (uint32_t)context->config.bit_depth, (uint32_t)context->config.chroma_format,
                base_q_idx,
                context->pictures_encoded, context->coded_blocks, context->coded_block_count,
                low_delay ? context->coded_modes : NULL, low_delay ? context->coded_block_count / 3u : 0u,
                low_delay, context->config.intra_mode_selection,
                context->config.partition_selection ? context->partition_decisions : NULL,
                context->config.partition_selection ? context->partition_decision_count : 0u,
                context->config.entropy_adaptation, context->packet, context->packet_capacity, &stream_size);
        } else if (context->config.partition_selection) {
            const uint32_t low_delay = context->config.prediction_structure == SVT_AV2_PRED_LOW_DELAY_P;
            status = svt_av2_encode_packet_tools(width, height,
                (uint32_t)context->config.bit_depth, (uint32_t)context->config.chroma_format,
                base_q_idx,
                context->pictures_encoded, context->coded_blocks, context->coded_block_count,
                low_delay ? context->coded_modes : NULL, low_delay ? context->coded_block_count / 3u : 0u,
                low_delay, context->config.intra_mode_selection, context->partition_decisions,
                context->partition_decision_count, context->packet, context->packet_capacity, &stream_size);
        } else if (context->config.prediction_structure == SVT_AV2_PRED_LOW_DELAY_P) {
            status = svt_av2_encode_low_delay_packet_modes(width, height,
                (uint32_t)context->config.bit_depth, (uint32_t)context->config.chroma_format,
                base_q_idx,
                context->pictures_encoded, context->coded_blocks, context->coded_block_count,
                context->coded_modes, context->coded_block_count / 3u,
                context->config.intra_mode_selection,
                context->packet, context->packet_capacity, &stream_size);
        } else if (context->config.intra_mode_selection) {
            status = svt_av2_encode_intra_packet_modes(width, height,
                (uint32_t)context->config.bit_depth, (uint32_t)context->config.chroma_format,
                base_q_idx,
                context->pictures_encoded, context->coded_blocks, context->coded_block_count,
                context->packet, context->packet_capacity, &stream_size);
        } else status = svt_av2_encode_intra_packet_ex(width, height,
            (uint32_t)context->config.bit_depth, (uint32_t)context->config.chroma_format,
            base_q_idx,
            context->pictures_encoded, context->coded_blocks, context->coded_block_count,
            context->packet, context->packet_capacity, &stream_size);
        if (status != SVT_AV2_BW_ERROR_NO_SPACE) break;
        if (capacity > SIZE_MAX / 2u) { context->fatal_error = 1; return SVT_AV2_ERROR_OUT_OF_MEMORY; }
        capacity *= 2u;
    }
    if (status != 0 || stream_size == 0) {
        context->fatal_error = 1;
        return SVT_AV2_ERROR_INTERNAL;
    }

    }
    SvtAv2VideoCdfState* cdf = context->video_cdf;
    int64_t cdf_seed_frame = -1;
    if (cdf != NULL) {
        if (memcmp(&cdf->rd_end, &cdf->writer_end, sizeof(cdf->rd_end)) != 0) {
            context->fatal_error = 1;
            return SVT_AV2_ERROR_INVALID_STATE;
        }
        if (refs.predictor) cdf_seed_frame = (int64_t)refs.predictor->frame_index;
        svt_av2_entropy_models_decay(&cdf->writer_end);
    }
    const int overlay_midpoint = disposable && (context->adaptive_pyramid ||
        ((context->motion_q == 17u || context->motion_q == 19u) && lookahead_display(context) < context->pictures_encoded));
    const int sequential_hold = context->adaptive_pyramid && !refs.paired_role && refs.predictor &&
        (lookahead_display(context) & (context->display_all_slots && context->adaptive_pyramid == 3u ? 15u : 7u)) &&
        (context->display_all_slots || anchor.source_q8 < 2u*256u);
    const int displayed_anchor = context->motion_q >= 16u && context->motion_q != 20u && context->pictures_encoded > context->anchor_display &&
        lookahead_display(context) == context->pictures_encoded;
    if (sequential_hold || overlay_midpoint || (((context->motion_q >= 5u && context->motion_q < 13u) || displayed_anchor) && refs.predictor &&
        context->held_pictures < (context->motion_q >= 8u ? 31u : 8u) &&
        (context->motion_q >= 8u || context->pictures_encoded != 4u) && anchor.source_q8 < 2u * 256u)) {
        int reuse=0;
        SvtAv2Error e=select_reference_display(context,stream_size,overlay_midpoint,&refs,&reuse);
        if (e) { context->fatal_error=1; return e; }
        if (reuse) return SVT_AV2_ERROR_NONE;
    }
    SVT_AV2_IP_START(profile_dpb);
    SvtAv2Error ref_status = SVT_AV2_ERROR_NONE;
    const int inner_anchor = context->motion_q >= 18u && refs.predictor &&
        context->pictures_encoded < context->anchor_display && lookahead_display(context) == context->pictures_encoded;
    if (context->motion_q == 20u && refs.predictor) {
        if (!disposable) {
            ref_status = svt_av2_reference_store(pyramid_slot(context, refs.refresh_slot),
                &context->recon, context->pictures_encoded);
            context->slot_display[refs.refresh_slot]=lookahead_display(context);
        }
        if (refs.paired_role == SVT_AV2_HIDDEN_ANCHOR || !refs.paired_role) {
            context->previous_anchor_display = context->anchor_display;
            context->latest_slot = (uint8_t)refs.refresh_slot;
            context->pyramid_span = (uint8_t)(lookahead_display(context)-context->pictures_encoded+1u);
            context->anchor_display = lookahead_display(context);
        }
    } else if (context->motion_q >= 18u && refs.predictor && !disposable) {
        ref_status = svt_av2_reference_store(&context->long_reference, &context->recon, context->pictures_encoded);
        if (!ref_status && !inner_anchor) {
            const SvtAv2ReferenceFrame swap = context->reference;
            context->reference = context->long_reference;
            context->long_reference = swap;
            context->latest_slot ^= 1u;
        }
    } else if ((context->motion_q == 11u || context->motion_q == 12u || context->motion_q >= 14u) && refs.predictor && !disposable) {
        ref_status = svt_av2_reference_refresh_anchor(&context->reference, &context->long_reference,
            &context->recon, context->pictures_encoded, context->held_pictures, (int)refs.rotate_anchor);
        if (ref_status == SVT_AV2_ERROR_NONE) context->latest_slot ^= (uint8_t)refs.rotate_anchor;
    } else if (refs.dual_ref && !disposable) {
        ref_status = context->dual_ref_cadence ?
            svt_av2_reference_refresh_after_holds(&context->reference, &context->long_reference,
                &context->recon, context->pictures_encoded, refs.refresh_mask, context->held_pictures) :
            svt_av2_reference_refresh_pair(&context->reference, &context->long_reference,
                &context->recon, context->pictures_encoded, context->dual_ref_period);
    } else if (context->config.prediction_structure == SVT_AV2_PRED_LOW_DELAY_P && !retain_reference) {
        ref_status = svt_av2_reference_prepare_interval(&context->reference, &context->recon,
            context->pictures_encoded, context->inter_ref_period);
        if (ref_status == SVT_AV2_ERROR_NONE) ref_status = svt_av2_reference_commit(&context->reference);
    }
    SVT_AV2_IP_START(profile_publish);
    if (ref_status != SVT_AV2_ERROR_NONE) { context->fatal_error = 1; return ref_status; }
    if (context->motion_q >= 13u && context->motion_q != 20u && refs.predictor && !disposable && !inner_anchor) {
        context->previous_anchor_display = context->anchor_display;
        context->tree_full = lookahead_display(context) == context->pictures_encoded+3u;
        context->anchor_display = lookahead_display(context);
    }
    if (cdf != NULL) {
        /* The existing pixel transaction has succeeded. A held slot keeps
         * both its pixels and CDF; the initial KEY has only native slot 0. */
        const uint32_t frame = context->pictures_encoded;
        if (!retain_reference && (!refs.dual_ref || !refs.predictor || (refs.slot_mask & 1u))) {
            cdf->slots[0] = cdf->writer_end;
            cdf->slot_frame[0] = frame;
        }
        if (refs.dual_ref && (refs.slot_mask & 2u)) {
            cdf->slots[1] = cdf->writer_end;
            cdf->slot_frame[1] = frame;
        }
        if (context->motion_q == 20u && refs.refresh_mask && refs.refresh_slot >= 2u) {
            cdf->slots[refs.refresh_slot] = cdf->writer_end;
            cdf->slot_frame[refs.refresh_slot] = frame;
        }
        if (cdf->trace)
            fprintf(stderr, "inter_cdf frame=%u seed=%lld slot0=%lld slot1=%lld\n", frame,
                (long long)cdf_seed_frame, (long long)cdf->slot_frame[0], (long long)cdf->slot_frame[1]);
    }
    if (disposable) {
        /* Published for display only. The next anchor shares both reference
         * slots and entropy contexts with the preceding reference picture. */
        ++context->held_pictures;
    } else if (context->dual_ref_cadence) {
        memcpy(context->anchor_source, anchor.current, sizeof(context->anchor_source));
        context->anchor_source_valid = 1u;
        context->held_pictures = 0u;
        context->previous_motion_q8 = refs.predictor ? anchor.source_q8 : UINT32_MAX;
        if (context->motion_q == 2u) {
            context->quiet_frames = refs.predictor && anchor.source_q8 < 384u ?
                (uint8_t)SVT_AV2_MIN(4u, context->quiet_frames + 1u) : 0u;
        }
        if (clean) context->last_clean_frame = context->pictures_encoded;
        const char* trace = getenv("SVT_AV2_ANCHOR_TRACE");
        if (trace && trace[0] == '1' && trace[1] == '\0')
            fprintf(stderr, "anchor_cadence frame=%u age=%u reason=%u mask=%u src_q8=%u last_q8=%u long_q8=%u qp=%d\n",
                context->pictures_encoded, anchor.age, anchor.reason, refs.refresh_mask,
                anchor.source_q8, anchor.last_q8, anchor.long_q8, qp);
    }
    if (cdf && cdf->use_motion_field && refs.predictor && !retain_reference &&
        cdf->frame_header.mv_half == 2u) {
        /* Store the winner at each grid sample only after the DPB transaction.
         * Single-reference vectors use Q2 units, projected two-reference vectors Q3. */
        const int scale = refs.secondary_reference ? 1 : 2;
        for (size_t i = 0; i < context->coded_block_count / 3u; ++i) {
            const SvtAv2TileCoeffBlock* b = &context->coded_blocks[i * 3u];
            const SvtAv2TileBlockMode* mode = &context->coded_modes[i];
            const unsigned rank = refs.secondary_reference ? mode->reference_rank : 0u;
            const int distance = mode->is_inter && rank < 2u ? (int)refs.ref_distance[rank] : 0;
            SvtAv2MotionVector velocity = {0, 0};
            if (distance) {
                velocity.mv_x = (mode->mv_x * scale + (mode->mv_x < 0 ? -distance : distance) / 2) / distance;
                velocity.mv_y = (mode->mv_y * scale + (mode->mv_y < 0 ? -distance : distance) / 2) / distance;
                if (context->motion_layers && !rank && refs.paired_role >= SVT_AV2_ONLY_MIDPOINT) {
                    velocity.mv_x = -velocity.mv_x;
                    velocity.mv_y = -velocity.mv_y;
                }
            }
            const uint32_t x0 = (b->x * SVT_AV2_ME_FIELD_COLS + width - 1u) / width;
            const uint32_t y0 = (b->y * SVT_AV2_ME_FIELD_ROWS + height - 1u) / height;
            const uint32_t x1 = (SVT_AV2_MIN(width, b->x + b->width) * SVT_AV2_ME_FIELD_COLS + width - 1u) / width;
            const uint32_t y1 = (SVT_AV2_MIN(height, b->y + b->height) * SVT_AV2_ME_FIELD_ROWS + height - 1u) / height;
            for (uint32_t y = y0; y < y1; ++y) for (uint32_t x = x0; x < x1; ++x)
                cdf->motion_field.velocity[y][x] = velocity;
        }
    }
    if (context->dual_ref_policy == 3u) record_block_ref_stats(context, &refs);
    context->recon_frame_index = context->pictures_sent - 1u;

    context->packet_size = stream_size;
    context->packet_pending = 1;
    context->frame_pending = 0;
    context->pictures_encoded++;
#if SVT_AV2_INTER_PROFILE
    /* The outer phases partition this synchronous API, not source I/O. The
     * formal writer phase includes packet growth/retries (and, for special
     * still profiles only, alternate-packet/restoration trials). RD has its
     * own nested report; do not add its times to these outer phases. */
    const uint64_t profile_end = svt_av2_ip_now(), hz = svt_av2_ip_hz();
    if (hz && profile_start && profile_end)
        fprintf(stderr, "inter_profile_frame kind=%s frame=%u clock=%s units=ns total=%llu refs_qp=%llu recon_rd=%llu formal_writer=%llu dpb=%llu publish=%llu\n",
            context->config.still_picture ? "STILL" : refs.predictor ? "P" : "KEY",
            context->pictures_encoded - 1u, SVT_AV2_IP_CLOCK_NAME,
            svt_av2_ip_ns(svt_av2_ip_delta(profile_start, profile_end), hz),
            svt_av2_ip_ns(svt_av2_ip_delta(profile_start, profile_recon), hz),
            svt_av2_ip_ns(svt_av2_ip_delta(profile_recon, profile_writer), hz),
            svt_av2_ip_ns(svt_av2_ip_delta(profile_writer, profile_dpb), hz),
            svt_av2_ip_ns(svt_av2_ip_delta(profile_dpb, profile_publish), hz),
            svt_av2_ip_ns(svt_av2_ip_delta(profile_publish, profile_end), hz));
#endif
    return SVT_AV2_ERROR_NONE;
}

static int collect_coefficients(void* user, uint32_t plane,
    const SvtAv2BlockView* block, const SvtAv2ReconScratch* scratch, int all_zero) {
    SvtAv2EncoderContext* context = (SvtAv2EncoderContext*)user;
    (void)all_zero;
    size_t count = (size_t)SVT_AV2_MIN(block->w, 32u) * SVT_AV2_MIN(block->h, 32u);
    if (context->coded_block_count >= context->coded_block_capacity ||
        count > context->coded_coeff_capacity - context->coded_coeff_count) return -1;
    SvtAv2TileCoeffBlock* entry = &context->coded_blocks[context->coded_block_count++];
    entry->plane = plane; entry->x = block->x; entry->y = block->y;
    entry->width = block->w; entry->height = block->h;
    entry->qcoeff = context->coded_coeffs + context->coded_coeff_count;
    entry->tx_type = 0u;
    entry->mrl = 0u;
    entry->intra_mode = plane == 0u ? (uint8_t)context->m5_last_intra_mode : 0u;
    memcpy(context->coded_coeffs + context->coded_coeff_count, scratch->qcoeff, count * sizeof(int32_t));
    context->coded_coeff_count += count;
    return 0;
}

static int collect_inter_coefficients(void* user, const SvtAv2InterBlockResult* result,
    const SvtAv2ReconScratch* scratch) {
    SvtAv2EncoderContext* c = (SvtAv2EncoderContext*)user;
    if (!c->reference.valid || result->reference_frame_index != c->reference.frame_index ||
        result->reference_frame_index + 1u != c->pictures_encoded) return -1;
    if (result->plane == 0u) {
        c->m5_last_intra_mode = result->intra_mode;
        SvtAv2TileBlockMode* m = c->coded_modes + c->coded_block_count / 3u;
        m->is_inter = result->mode == SVT_AV2_BLOCK_SINGLE_INTER;
        m->mv_x = result->mv.mv_x; m->mv_y = result->mv.mv_y;
    }
    return collect_coefficients(user, result->plane, &result->block, scratch, result->all_zero);
}

static int m5_collect_intra_mode(void* user, uint32_t mode) {
    ((SvtAv2EncoderContext*)user)->m5_last_intra_mode = mode;
    return 0;
}

/* The hook retains exactly the qcoeff used by this reconstruction. */
static SvtAv2Error encode_reconstruction(SvtAv2EncoderContext* context,
                                         uint32_t width,
                                         uint32_t height,
                                         uint32_t base_q_idx,
                                         const SvtAv2PictureReference* refs) {
    if (!context->recon_ready) {
        return SVT_AV2_ERROR_INVALID_STATE;
    }

    size_t samples = (size_t)width * height * 3u /
        (context->config.chroma_format == SVT_AV2_CHROMA_444 ? 1u : 2u);
    size_t blocks = (size_t)width * height / 64u * 3u;
    if (context->coded_block_capacity < blocks) {
        SvtAv2TileCoeffBlock* grown = (SvtAv2TileCoeffBlock*)realloc(context->coded_blocks,
            blocks * sizeof(*grown));
        if (grown == NULL) return SVT_AV2_ERROR_OUT_OF_MEMORY;
        context->coded_blocks = grown;
        SvtAv2TileBlockMode* modes = (SvtAv2TileBlockMode*)realloc(context->coded_modes,
            (blocks / 3u) * sizeof(*modes));
        if (modes == NULL) return SVT_AV2_ERROR_OUT_OF_MEMORY;
        context->coded_modes = modes; context->coded_block_capacity = blocks;
    }
    if (context->coded_coeff_capacity < samples) {
        int32_t* grown = (int32_t*)realloc(context->coded_coeffs, samples * sizeof(*grown));
        if (grown == NULL) return SVT_AV2_ERROR_OUT_OF_MEMORY;
        context->coded_coeffs = grown; context->coded_coeff_capacity = samples;
    }
    context->coded_block_count = context->coded_coeff_count = 0;
    context->partition_decision_count = 0;
    memset(&context->rd_stats, 0, sizeof(context->rd_stats));
    context->inter_skip_trials = context->inter_skip_accepts = 0u;
    context->inter_fine_trials = context->inter_fine_accepts = 0u;
    if (context->config.partition_selection || context->config.rd_mode) {
        SvtAv2FrameGeometry geometry;
        SvtAv2Error error = svt_av2_frame_geometry_init_ex(&geometry, width, height, 64u,
            context->config.chroma_format);
        if (error != SVT_AV2_ERROR_NONE) return error;
        const size_t count = (size_t)geometry.sb_cols * geometry.sb_rows;
        if (context->partition_decision_capacity < count) {
            uint16_t* grown = (uint16_t*)realloc(context->partition_decisions, count * sizeof(*grown));
            if (grown == NULL) return SVT_AV2_ERROR_OUT_OF_MEMORY;
            context->partition_decisions = grown;
            context->partition_decision_capacity = count;
        }
        if (!context->config.rd_mode) {
            error = svt_av2_partition_select_quads(&context->frame_source, &geometry,
                context->partition_decisions, context->partition_decision_capacity);
            if (error != SVT_AV2_ERROR_NONE) return error;
        } else memset(context->partition_decisions, 0, count * sizeof(*context->partition_decisions));
        context->partition_decision_count = count;
    }
    SvtAv2IntraFrameConfig config;
    memset(&config, 0, sizeof(config));
    config.base_q_idx = base_q_idx;
    config.sb_size = 64u;
    config.frame_reset_value = (uint16_t)(1u << ((uint32_t)context->config.bit_depth - 1u));
    config.block_hook = collect_coefficients;
    config.m5_select_intra = context->config.intra_mode_selection;
    config.m5_mode_hook = config.m5_select_intra ? m5_collect_intra_mode : NULL;
    config.block_hook_user = context;
    config.partition_decisions = context->partition_decision_count ? context->partition_decisions : NULL;
    config.partition_decision_count = context->partition_decision_count;

    const SvtAv2Error path_err =
        svt_av2_transform_path_init_bd(&context->path_state, &config.transform, config.base_q_idx,
            (uint32_t)context->config.bit_depth);
    if (path_err != SVT_AV2_ERROR_NONE) {
        return path_err;
    }
    const int qm_is_p = context->config.prediction_structure == SVT_AV2_PRED_LOW_DELAY_P &&
        context->pictures_encoded != 0u;
    const char* prune = getenv("SVT_AV2_AC_PRUNE");
    context->path_state.ac_prune = prune && !strcmp(prune, "strong") ? 3u :
        prune && !strcmp(prune, "medium") ? 2u : prune && !strcmp(prune, "mild") ? 1u : 0u;
    const char* fp_round = getenv("SVT_AV2_FP_ROUND");
    context->path_state.matrix_rounding = fp_round ?
        (!strcmp(fp_round, "scan") ? 2u : !strcmp(fp_round, "matrix")) : 0u;
    context->path_state.avm_fp_round = fp_round &&
        (!strcmp(fp_round, "all") || !strcmp(fp_round, qm_is_p ? "p" : "key"));
    context->path_state.uv_delta_q = context->uv_delta_q;
    context->path_state.qm_level = (context->qm_scope & (qm_is_p ? 2u : 1u)) ? context->qm_level : 0u;
    context->path_state.qm_uv_level = context->path_state.qm_level ? context->qm_uv_level : 0u;
    if (qm_is_p && context->inter_qm_curve && context->path_state.qm_level)
        zen_matrix_levels((int)base_q_idx,context->uv_delta_q,&context->path_state.qm_level,
                          &context->path_state.qm_uv_level);
#if SVT_AV2_EXPERIMENTAL_DCT64_MATCH
    const char* dct64_scope = getenv("SVT_AV2_DCT64_MATCH");
    const int dct64_p = context->config.prediction_structure == SVT_AV2_PRED_LOW_DELAY_P &&
        context->pictures_encoded != 0u;
    context->path_state.match_dct64 = dct64_scope && (strcmp(dct64_scope, "all") == 0 ||
        strcmp(dct64_scope, dct64_p ? "p" : "key") == 0);
#endif

    /* All-Intra has no inter reference; the intra neighbour state is per frame.
     * The core overwrites every visited sample, but resetting makes the initial
     * state explicit rather than depending on the previous frame. */
    SvtAv2Error status = svt_av2_recon_reset_frame(&context->recon, config.frame_reset_value);
    if (status != SVT_AV2_ERROR_NONE) {
        return status;
    }

    if (context->config.rd_mode) {
        SvtAv2RdFrameConfig rd;
        memset(&rd, 0, sizeof(rd));
        rd.transform = config.transform;
        rd.sb_qindex=context->column_qindex;
        rd.sb_lambda_weights=context->picture_weights;
        rd.matrix_rounding = context->path_state.matrix_rounding;
        rd.avm_fp_round = context->path_state.avm_fp_round;
        rd.rd_mode = context->config.rd_mode;
        rd.coeff_optimization = context->config.coeff_optimization;
        rd.motion_optimization = context->config.motion_optimization >= 3u &&
            context->pictures_encoded == 0u ? 0u : context->config.motion_optimization;
        rd.transform_selection = context->config.transform_selection;
        rd.tree_finalists = context->config.tree_refinement == 2u ?
            (context->config.partition_selection >= 10u ? 10u :
             context->config.partition_selection >= 8u ? 8u : 4u) :
            context->config.tree_refinement ? 2u : 0u;
        const char* still_search = getenv("SVT_AV2_STILL_SEARCH");
        rd.screen_trees = context->config.tree_refinement == 3u;
        if (rd.screen_trees && still_search && (!strcmp(still_search, "rect") || !strcmp(still_search, "adaptive-rect"))) {
            /* Search rectangles before the screen profile's single filter pass. */
            rd.screen_trees = 0;
            rd.tree_finalists = 4u;
        }
        rd.subrect_search=rd.screen_trees && still_search &&
            (!strcmp(still_search,"subrect") || !strcmp(still_search,"subrect-fast"));
        rd.shortlist_trees = rd.screen_trees && still_search && !strcmp(still_search, "shortlist");
        rd.subrect_search |= rd.shortlist_trees;
        rd.thin_still_search=rd.screen_trees && still_search && !strcmp(still_search,"thin");
        rd.balanced_still_search = context->config.still_picture && rd.screen_trees &&
            still_search && (!strcmp(still_search, "balanced") || !strcmp(still_search, "directional") ||
                             !strcmp(still_search, "subrect-fast") || !strcmp(still_search, "cells"));
        if (rd.balanced_still_search && !strcmp(still_search, "directional"))
            rd.balanced_still_search=2;
        if (rd.balanced_still_search && !strcmp(still_search, "cells"))
            rd.balanced_still_search=3;
        rd.sparse_angle_search = rd.balanced_still_search &&
            !(context->content_type == SVT_AV2_CONTENT_ANIMATION &&
              (context->full_picture_width ? context->full_picture_width : context->frame_width) == 1920u);
        if (rd.screen_trees) rd.tree_finalists = 0u;
        rd.rd_lambda_model = context->config.rd_lambda_model;
        rd.rd_lambda_scale = context->rd_lambda_scale;
        rd.cfl_search = context->config.cfl_search;
        rd.mhccp_search = context->config.mhccp_search;
        rd.intra_joint = context->config.intra_joint;
        rd.angle_search = context->config.angle_search;
        rd.threads = context->config.threads;
        rd.satd_angles = context->satd_angles;
        rd.partition_selection = context->config.partition_selection;
        svt_av2_frame_header_config_default(&rd.frame_header, width, height);
        rd.frame_header.base_q_idx = config.base_q_idx;
        rd.frame_header.multi_frame = !context->config.still_picture;
        rd.frame_header.inter_frame = context->config.prediction_structure == SVT_AV2_PRED_LOW_DELAY_P &&
            context->pictures_encoded != 0u;
        rd.frame_header.mv_half = motion_precision(rd.motion_optimization);
        rd.frame_header.sharp_filter = rd.frame_header.inter_frame && rd.motion_optimization >= 4u;
        const char* range = getenv("SVT_AV2_ME_RANGE");
        rd.frame_header.wide_motion = context->video_cdf && rd.frame_header.mv_half == 2u && range ?
            (!strcmp(range, "63") ? 2u : (unsigned)!strcmp(range, "31")) : 0u;
        const char* filter = getenv("SVT_AV2_VIDEO_INTERPOLATION");
        rd.frame_header.regular_filter = context->video_cdf && rd.frame_header.mv_half == 2u &&
            filter && !strcmp(filter, "regular");
        rd.inter_skip_search = rd.frame_header.inter_frame ? context->inter_skip_search : 0u;
        rd.inter_fine = rd.frame_header.inter_frame ? context->inter_fine : 0u;
        rd.inter_tail = rd.frame_header.inter_frame ? context->inter_tail : 0u;
        rd.inter_near = rd.frame_header.inter_frame ? context->inter_near : 0u;
        rd.frame_header.retain_reference = refs->retain_reference;
        rd.frame_header.dual_ref = refs->dual_ref;
        rd.frame_header.refresh_mask = refs->slot_mask;
        rd.frame_header.reference_slot = refs->reference_slot;
        const char* inter_intra = getenv("SVT_AV2_VIDEO_INTRA_REFINE");
        rd.frame_header.inter_intra_search = rd.frame_header.inter_frame && context->video_cdf &&
            inter_intra && !strcmp(inter_intra,"directional");
        rd.frame_header.pyramid_slots = context->motion_q == 20u ? (context->adaptive_pyramid > 2u ? context->adaptive_pyramid+2u : 4u) : 0u;
        rd.frame_header.secondary_slot = refs->secondary_slot;
        const char* aq = getenv("SVT_AV2_VIDEO_AQ");
        if (refs->predictor && context->video_cdf && aq && !strcmp(aq,"variance")) {
            SvtAv2VideoCdfState* bank = context->video_cdf;
            const size_t count = ((width+63u)/64u)*(size_t)((height+63u)/64u);
            if (bank->video_qindex_count != count) {
                uint16_t* map = (uint16_t*)realloc(bank->video_qindex,count*sizeof(*map));
                if (!map) return SVT_AV2_ERROR_OUT_OF_MEMORY;
                bank->video_qindex = map;
                bank->video_qindex_count = count;
            }
            rd.sb_qindex = bank->video_qindex;
            rd.frame_header.sb_qindex = rd.sb_qindex;
        }
        rd.secondary_reference = refs->secondary_reference;
        const char* motion_profile = getenv("SVT_AV2_MOTION_Q");
        const char* compound_refs = getenv("SVT_AV2_COMPOUND_REFS");
        const int bidirectional_anchor = compound_refs && !strcmp(compound_refs,"anchors") &&
            refs->paired_role >= SVT_AV2_HIDDEN_MIDPOINT && refs->ref_distance[0] == refs->ref_distance[1];
        const int equidistant = refs->ref_distance[0] == refs->ref_distance[1];
        if (refs->secondary_reference && equidistant && (!refs->refresh_mask || bidirectional_anchor) && (context->adaptive_pyramid == 4u || context->motion_layers || (motion_profile && (!strcmp(motion_profile, "pyramid-compound") || !strcmp(motion_profile, "pyramid16"))))) {
            rd.frame_header.compound_average = 2u;
        } else if (refs->secondary_reference && equidistant && !refs->refresh_mask && ((motion_profile && !strcmp(motion_profile, "pyramid8-blend")) ||
                (context->adaptive_pyramid && compare_reference_pair(&context->frame_source, refs, NULL)))) {
            rd.frame_header.compound_average = 1;
            rd.inter_near = rd.inter_fine = 0;
        }
        if (refs->secondary_reference != NULL) {
            rd.frame_header.logical_ref_count = 2u;
            rd.frame_header.ref_distance[0] = refs->ref_distance[0];
            rd.frame_header.ref_distance[1] = refs->ref_distance[1];
        }
        rd.frame_header.order_hint = context->pictures_encoded & 255u;
        if (context->motion_q >= 13u) {
            const uint32_t pts = lookahead_display(context);
            rd.frame_header.order_hint = pts & 255u;
            if (context->motion_q == 20u) rd.frame_header.paired_role = refs->paired_role;
            else if (context->motion_q >= 18u && context->tree_full &&
                    context->pictures_encoded && context->pictures_encoded <= context->anchor_display) {
                const unsigned remaining = context->anchor_display - context->pictures_encoded;
                rd.frame_header.paired_role = remaining == 2u ? SVT_AV2_HIDDEN_MIDPOINT :
                    remaining == 1u ? SVT_AV2_ONLY_MIDPOINT : SVT_AV2_LAST_MIDPOINT;
            } else if (pts > context->pictures_encoded) rd.frame_header.paired_role = SVT_AV2_HIDDEN_ANCHOR;
            else if (pts < context->pictures_encoded) {
                const int first = context->pictures_encoded == context->reference.frame_index + 1u;
                const int last = pts + 1u == context->anchor_display;
                rd.frame_header.paired_role = first && last ? SVT_AV2_ONLY_MIDPOINT :
                    first ? SVT_AV2_FIRST_MIDPOINT : last ? SVT_AV2_LAST_MIDPOINT : SVT_AV2_INNER_MIDPOINT;
            }
        }
        rd.frame_header.reduced_tx_set = context->config.transform_selection == 2u ? 0u : context->config.transform_selection ? 1u : 2u;
        rd.frame_header.enable_mrls = context->config.mrl_search;
        rd.frame_header.enable_mhccp = context->config.mhccp_search;
        rd.frame_header.bit_depth = (uint32_t)context->config.bit_depth;
        rd.frame_header.chroma_format = (uint32_t)context->config.chroma_format;
        rd.frame_header.m5_intra_modes = context->config.intra_mode_selection;
        rd.frame_header.entropy_adaptation = context->config.entropy_adaptation;
        rd.frame_header.enable_intra_ist = context->intra_ist_scope != 0u;
        rd.frame_header.uv_delta_q = context->uv_delta_q;
        rd.frame_header.qm_level = context->path_state.qm_level;
        rd.frame_header.qm_uv_level = context->path_state.qm_uv_level;
        rd.frame_header.enable_ibp = context->enable_ibp;
        if (context->video_cdf && context->gdf_key) {
            rd.frame_header.enable_gdf = 1u;
            rd.frame_header.gdf_code = rd.frame_header.inter_frame ? (context->gdf_inter ? 8u : 0u) : gdf_key_code(context);
        }
        rd.frame_header.enable_intra_edge_filter = context->still_edge_filter;
        rd.intra_ist_search = (context->intra_ist_scope &
            (rd.frame_header.inter_frame ? 4u : 2u)) != 0u;
        SvtAv2VideoCdfState* cdf = context->video_cdf;
        if (cdf != NULL) {
            const char* deblock=getenv("SVT_AV2_VIDEO_DEBLOCK");
            if (deblock && !strcmp(deblock,"all") && rd.frame_header.chroma_format==SVT_AV2_CHROMA_420 &&
                !(width&7u) && !(height&7u) && width<=4096u && height<=2160u)
                rd.frame_header.deblock_y=3u;
            if (cdf->use_motion_field && refs->predictor && rd.frame_header.mv_half == 2u) {
                rd.temporal_motion = &cdf->motion_field;
                memcpy(rd.motion_distance, refs->ref_distance, sizeof(rd.motion_distance));
            }
            rd.end_models = &cdf->rd_end;
#if SVT_AV2_EXPERIMENTAL_VIDEO_CFL
            rd.frame_header.enable_cfl_intra = cdf->cfl_sequence;
            rd.video_cfl_search = rd.frame_header.inter_frame ? cdf->cfl_p : cdf->cfl_key;
#endif
            if (refs->predictor != NULL) {
                const uint32_t slot = refs->reference_slot; /* Physical backing of rank 0. */
                if (!refs->predictor->valid || cdf->slot_frame[slot] != (int64_t)refs->predictor->frame_index)
                    return SVT_AV2_ERROR_INVALID_STATE;
                rd.frame_header.primary_models = &cdf->slots[slot];
            }
        }
        const char* key_search=getenv("SVT_AV2_VIDEO_KEY_SEARCH");
        const int deep_key = key_search && !strcmp(key_search,"deep");
        const int full_key = deep_key || (key_search && !strcmp(key_search,"full"));
        if (cdf && !qm_is_p && key_search && (full_key || !strcmp(key_search,"directional")) &&
            rd.frame_header.bit_depth==10u &&
            rd.frame_header.chroma_format==SVT_AV2_CHROMA_420) {
            /* The first reference pays the complete intra mode and partition
             * rate and seeds the P-frame CDF bank through the same writer. */
            rd.rd_mode=3u;
            rd.frame_header.m5_intra_modes=rd.frame_header.enable_cfl_intra ? 4u : 3u;
            rd.cfl_search=rd.frame_header.enable_cfl_intra;
            /* m5 mode 4 signals CfL in the sequence and the intra tile. P
             * pictures continue using the video CfL selector for that bit. */
            rd.frame_header.enable_cfl_intra=0;
#if SVT_AV2_EXPERIMENTAL_VIDEO_CFL
            rd.video_cfl_search=0;
#endif
            if (full_key) {
                rd.transform_selection=2u;
                rd.coeff_optimization=2u;
                rd.frame_header.reduced_tx_set=0u;
            }
            rd.partition_selection=7u;
            rd.angle_search=1u;
            rd.screen_trees=!deep_key;
            rd.tree_finalists=deep_key ? 4u : 0u;
        }
        SvtAv2RdFrameOutput output;
        memset(&output, 0, sizeof(output));
        output.blocks = context->coded_blocks; output.block_capacity = context->coded_block_capacity;
        output.modes = context->coded_modes; output.mode_capacity = context->coded_block_capacity / 3u;
        output.qcoeff = context->coded_coeffs; output.coeff_capacity = context->coded_coeff_capacity;
        output.decisions = context->partition_decisions; output.decision_capacity = context->partition_decision_capacity;
        SvtAv2RdFrameStats stats;
        memset(&stats, 0, sizeof(stats));
        status = svt_av2_rd_frame_encode(&context->frame_source, &context->recon,
            refs->predictor, &rd, &context->scratch, &output, &stats);
        if (status != SVT_AV2_ERROR_NONE) return status;
        if (rd.partition_selection) {
            rd.frame_header.partition_decisions = output.decisions;
            rd.frame_header.partition_decision_count = output.decision_count;
        }
        rd.frame_header.sb_qindex=rd.sb_qindex;
        if (cdf != NULL) cdf->frame_header = rd.frame_header;
        else if (context->full_picture_width || context->intra_ist_scope || context->still_edge_filter || context->enable_ibp || context->uv_delta_q || context->path_state.qm_level)
            context->still_tools_header = rd.frame_header;
        context->coded_block_count = output.block_count;
        context->coded_coeff_count = output.coeff_count;
        context->partition_decision_count = output.decision_count;
        context->rd_stats.candidate_count = stats.candidate_count;
        context->rd_stats.forward_calls = stats.forward_calls;
        context->rd_stats.replay_checks = stats.replay_checks;
        context->rd_stats.distortion = stats.distortion;
        context->rd_stats.rate_q12 = stats.rate_q12;
        context->rd_stats.cost = stats.cost;
        context->rd_stats.lambda_q8 = stats.lambda_q8;
        context->rd_stats.full_superblocks = stats.full_superblocks;
        context->rd_stats.partial_superblocks = stats.partial_superblocks;
        context->rd_stats.motion_candidates = stats.motion_candidates;
        context->rd_stats.winner_hash = stats.winner_hash;
        context->rd_stats.coeff_trials = stats.coeff_trials;
        context->rd_stats.coeff_accepts = stats.coeff_accepts;
        context->rd_stats.motion_trials = stats.motion_trials;
        context->rd_stats.motion_accepts = stats.motion_accepts;
        context->rd_stats.wide_trials = stats.wide_trials;
        context->rd_stats.wide_accepts = stats.wide_accepts;
        context->inter_skip_trials = stats.inter_skip_trials;
        context->inter_skip_accepts = stats.inter_skip_accepts;
        context->inter_fine_trials = stats.inter_fine_trials;
        context->inter_fine_accepts = stats.inter_fine_accepts;
        return SVT_AV2_ERROR_NONE;
    }
    if (context->config.prediction_structure == SVT_AV2_PRED_LOW_DELAY_P && context->pictures_encoded != 0u) {
        SvtAv2InterFrameConfig inter;
        memset(&inter, 0, sizeof(inter));
        inter.transform = config.transform;
        inter.block_hook = collect_inter_coefficients;
        inter.block_hook_user = context;
        inter.m5_select_intra = context->config.intra_mode_selection;
        inter.partition_decisions = config.partition_decisions;
        inter.partition_decision_count = config.partition_decision_count;
        SvtAv2InterFrameStats stats;
        memset(&stats, 0, sizeof(stats));
        return svt_av2_inter_frame_encode(&context->frame_source, &context->recon,
            &context->reference, &inter, &context->scratch, &stats);
    }
    SvtAv2IntraFrameStats stats;
    memset(&stats, 0, sizeof(stats));
    (void)width;
    (void)height;
    return svt_av2_intra_frame_encode(&context->frame_source,
                                      &context->recon,
                                      &config,
                                      &context->scratch,
                                      &stats);
}

/* Exports the visible samples of the frame most recently taken by
 * svt_av2_enc_get_packet_ex(). Mirrors SvtAv2Packet: caller-owned buffer,
 * capacity as input, and a too-small buffer reports the required size without
 * writing. The bytes come from T012's single packing primitive. */
SvtAv2Error svt_av2_enc_get_rd_stats(SvtAv2Handle handle, SvtAv2RdStats* stats) {
    SvtAv2EncoderContext* context = context_from_handle(handle);
    if (context == NULL) return SVT_AV2_ERROR_INVALID_HANDLE;
    if (stats == NULL) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (context->state != SVT_AV2_STATE_INITIALIZED || context->pictures_encoded == 0u)
        return SVT_AV2_ERROR_INVALID_STATE;
    if (context->fatal_error) return SVT_AV2_ERROR_INTERNAL;
    *stats = context->rd_stats;
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_enc_get_inter_skip_stats(SvtAv2Handle handle, uint64_t* trials, uint64_t* accepts) {
    const SvtAv2EncoderContext* context = context_from_handle(handle);
    if (context == NULL) return SVT_AV2_ERROR_INVALID_HANDLE;
    if (trials == NULL || accepts == NULL || trials == accepts) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (context->state != SVT_AV2_STATE_INITIALIZED || context->pictures_encoded == 0u)
        return SVT_AV2_ERROR_INVALID_STATE;
    if (context->fatal_error) return SVT_AV2_ERROR_INTERNAL;
    *trials = context->inter_skip_trials;
    *accepts = context->inter_skip_accepts;
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_enc_get_inter_fine_stats(SvtAv2Handle handle, uint64_t* trials, uint64_t* accepts) {
    const SvtAv2EncoderContext* context = context_from_handle(handle);
    if (context == NULL) return SVT_AV2_ERROR_INVALID_HANDLE;
    if (trials == NULL || accepts == NULL || trials == accepts) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (context->state != SVT_AV2_STATE_INITIALIZED || context->pictures_encoded == 0u)
        return SVT_AV2_ERROR_INVALID_STATE;
    if (context->fatal_error) return SVT_AV2_ERROR_INTERNAL;
    *trials = context->inter_fine_trials;
    *accepts = context->inter_fine_accepts;
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_enc_get_block_ref_stats(SvtAv2Handle handle, SvtAv2BlockRefStats* stats) {
    const SvtAv2EncoderContext* context = context_from_handle(handle);
    if (context == NULL) return SVT_AV2_ERROR_INVALID_HANDLE;
    if (stats == NULL) return SVT_AV2_ERROR_INVALID_PARAMETER;
    if (context->state != SVT_AV2_STATE_INITIALIZED || context->pictures_encoded == 0u)
        return SVT_AV2_ERROR_INVALID_STATE;
    if (context->fatal_error) return SVT_AV2_ERROR_INTERNAL;
    *stats = context->block_ref_stats;
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_enc_get_recon(SvtAv2Handle handle, SvtAv2ReconFrame* recon) {
    SvtAv2EncoderContext* context = context_from_handle(handle);
    if (context == NULL) {
        return SVT_AV2_ERROR_INVALID_HANDLE;
    }
    if (context->state != SVT_AV2_STATE_INITIALIZED) {
        return SVT_AV2_ERROR_INVALID_STATE;
    }
    if (recon == NULL || recon->data == NULL) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    if (context->fatal_error) return SVT_AV2_ERROR_INTERNAL;
    if (!context->recon_ready || context->pictures_encoded == 0 || context->recon.planes[0] == NULL) {
        return SVT_AV2_ERROR_INVALID_STATE;
    }

    SvtAv2ReconExport export_meta;
    memset(&export_meta, 0, sizeof(export_meta));
    SvtAv2Error status = svt_av2_recon_export_frame(&context->recon,
                                                    context->recon_frame_index,
                                                    (uint8_t*)recon->data,
                                                    recon->capacity,
                                                    &export_meta);
    if (status == SVT_AV2_ERROR_INVALID_PARAMETER) {
        /* Distinguish "insufficient capacity" from "bad inputs" so the caller
         * can grow and retry, exactly as svt_av2_enc_get_packet_ex() does. The
         * required size is the raw size of the recon picture. */
        size_t required = 0;
        if (svt_av2_picture_raw_size(&context->recon, &required) == SVT_AV2_ERROR_NONE &&
            recon->capacity < required) {
            recon->byte_count = required;
            return SVT_AV2_ERROR_OUT_OF_MEMORY;
        }
    }
    if (status != SVT_AV2_ERROR_NONE) {
        return status;
    }

    recon->frame_index = export_meta.frame_index;
    recon->visible_width = export_meta.visible_width;
    recon->visible_height = export_meta.visible_height;
    recon->chroma_width = export_meta.chroma_width;
    recon->chroma_height = export_meta.chroma_height;
    recon->byte_count = export_meta.byte_count;
    recon->bit_depth = context->config.bit_depth;
    recon->chroma_format = context->config.chroma_format;
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_enc_get_packet(SvtAv2Handle handle, void* packet, int32_t* has_packet) {
    SvtAv2EncoderContext* context = context_from_handle(handle);
    if (context == NULL) {
        return SVT_AV2_ERROR_INVALID_HANDLE;
    }
    if (context->state != SVT_AV2_STATE_INITIALIZED) {
        return SVT_AV2_ERROR_INVALID_STATE;
    }
    if (packet == NULL || has_packet == NULL) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }

    /* This legacy entry point has no capacity parameter, so it cannot deliver
     * bytes: it reports availability only. Callers that want the stream use
     * svt_av2_enc_get_packet_ex() below. */
    *has_packet = 0;
    const SvtAv2Error err = encode_pending_frame(context);
    if (err != SVT_AV2_ERROR_NONE) {
        return err;
    }
    if (context->packet_pending) {
        *has_packet = (context->packet_size > 0) ? 1 : 0;
    }
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_enc_get_packet_ex(SvtAv2Handle handle, SvtAv2Packet* packet) {
    SvtAv2EncoderContext* context = context_from_handle(handle);
    if (context == NULL) {
        return SVT_AV2_ERROR_INVALID_HANDLE;
    }
    if (context->state != SVT_AV2_STATE_INITIALIZED) {
        return SVT_AV2_ERROR_INVALID_STATE;
    }
    if (packet == NULL) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }

    packet->size = 0;
    packet->pts = 0;

    /* A previously produced packet that the caller did not take is handed out
     * now rather than being silently replaced. */
    const SvtAv2Error err = encode_pending_frame(context);
    if (err != SVT_AV2_ERROR_NONE) {
        return err;
    }
    if (!context->packet_pending) {
        /* Nothing encoded yet, or the stream has ended. Both are "no packet",
         * distinguishable from failure by the NONE return code. */
        return SVT_AV2_ERROR_NONE;
    }
    if (packet->data == NULL) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    /* The caller sizes the buffer; a stream that does not fit is reported (with
     * the required size) instead of being truncated. */
    if (packet->capacity < context->packet_size) {
        packet->size = context->packet_size;
        return SVT_AV2_ERROR_OUT_OF_MEMORY;
    }

    memcpy(packet->data, context->packet, context->packet_size);
    packet->size = context->packet_size;
    packet->pts = context->frame_pts;
    context->packet_pending = 0;
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_enc_deinit(SvtAv2Handle* handle) {
    if (handle == NULL || *handle == NULL) {
        return SVT_AV2_ERROR_INVALID_HANDLE;
    }
    SvtAv2EncoderContext* context = *handle;
    if (context->magic != SVT_AV2_CONTEXT_MAGIC) {
        return SVT_AV2_ERROR_INVALID_HANDLE;
    }

    context->state = SVT_AV2_STATE_CLOSED;
    context->magic = 0;
    svt_av2_context_release_pipeline(context);
    if (context->config.output_path != NULL) {
        free((void*)context->config.output_path);
        context->config.output_path = NULL;
    }
    free(context);
    *handle = NULL;
    return SVT_AV2_ERROR_NONE;
}

SvtAv2Error svt_av2_enc_get_config(SvtAv2Handle handle, SvtAv2EncoderConfig* config) {
    SvtAv2EncoderContext* context = context_from_handle(handle);
    if (context == NULL) {
        return SVT_AV2_ERROR_INVALID_HANDLE;
    }
    if (config == NULL) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }
    *config = context->config;
    return SVT_AV2_ERROR_NONE;
}

const SvtAv2EncoderInfo* svt_av2_enc_get_build_info(void) {
    static SvtAv2EncoderInfo info;
    static int              initialized = 0;
    if (!initialized) {
        const SvtAv2BuildInfo* build = svt_av2_get_build_info();
        memset(&info, 0, sizeof(info));
        info.version_major = SVT_AV2_VERSION_MAJOR;
        info.version_minor = SVT_AV2_VERSION_MINOR;
        info.version_patch = SVT_AV2_VERSION_PATCH;
        copy_string(info.version_string, sizeof(info.version_string), build->version_string);
        copy_string(info.build_type, sizeof(info.build_type), build->build_type);
        copy_string(info.compiler, sizeof(info.compiler), build->compiler);
        copy_string(info.milestone, sizeof(info.milestone), build->milestone);
        initialized = 1;
    }
    return &info;
}

/* --- M0 placeholder payload ---------------------------------------------- */

static void store_u16_le(uint8_t* buffer, uint16_t value) {
    buffer[0] = (uint8_t)(value & 0xFFu);
    buffer[1] = (uint8_t)((value >> 8) & 0xFFu);
}

static void store_u32_le(uint8_t* buffer, uint32_t value) {
    buffer[0] = (uint8_t)(value & 0xFFu);
    buffer[1] = (uint8_t)((value >> 8) & 0xFFu);
    buffer[2] = (uint8_t)((value >> 16) & 0xFFu);
    buffer[3] = (uint8_t)((value >> 24) & 0xFFu);
}

SvtAv2Error svt_av2_m0_build_placeholder_payload(const SvtAv2EncoderConfig* config,
                                                 uint32_t frame_index,
                                                 uint8_t* out_buffer,
                                                 size_t out_capacity,
                                                 size_t* out_size) {
    if (config == NULL || out_buffer == NULL || out_size == NULL) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }

    /* Deterministic payload length derived from the frame index only, so that
     * two runs with the same input always produce identical bytes. */
    const uint16_t payload_length = (uint16_t)(64u + (frame_index % 64u));
    const size_t   total = SVT_AV2_M0_PLACEHOLDER_HEADER_SIZE + (size_t)payload_length;
    if (out_capacity < total) {
        return SVT_AV2_ERROR_INVALID_PARAMETER;
    }

    memcpy(out_buffer, SVT_AV2_M0_PLACEHOLDER_MAGIC, SVT_AV2_M0_PLACEHOLDER_MAGIC_SIZE);
    out_buffer[6] = (uint8_t)SVT_AV2_VERSION_MAJOR;
    out_buffer[7] = (uint8_t)SVT_AV2_VERSION_MINOR;
    store_u32_le(out_buffer + 8, frame_index);
    store_u32_le(out_buffer + 12, config->width);
    store_u32_le(out_buffer + 16, config->height);
    out_buffer[20] = (uint8_t)config->chroma_format;
    out_buffer[21] = (uint8_t)config->bit_depth;
    store_u16_le(out_buffer + 22, payload_length);

    for (uint16_t i = 0; i < payload_length; ++i) {
        out_buffer[SVT_AV2_M0_PLACEHOLDER_HEADER_SIZE + i] = (uint8_t)((frame_index + i) & 0xFFu);
    }

    *out_size = total;
    return SVT_AV2_ERROR_NONE;
}
