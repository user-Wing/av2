/*
 * SVT-AV2 - M1 keyframe encoder entry point (T005).
 *
 * Scope: the single, complete "encode one 64x64 8-bit 4:2:0 key frame to an
 * Annex B AV2 elementary stream" operation agreed with 00 and T001. It packages
 * the T005 building blocks (T004 sequence header, T005 frame header, tile
 * group, entropy coder) behind one call so the application never touches OBU
 * assembly or bitstream syntax.
 *
 * The M1 encoder is an ALL-INTRA, DC_PRED, zero-residual encoder: it produces a
 * conforming single-frame bitstream whose reconstruction is the DC prediction
 * (mid-grey, 128) for every sample. The input planes are therefore accepted for
 * API stability but are not yet used to code residual data; this is documented
 * behaviour, not a silent shortcut (see docs/bitstream-frame-minimal.md).
 *
 * Output: the complete Annex B stream
 *     OBU_TEMPORAL_DELIMITER
 *     OBU_SEQUENCE_HEADER            (64x64, level 0, 8-bit 4:2:0)
 *     OBU_CLOSED_LOOP_KEY            (tile_group_obu, one 64x64 superblock)
 *
 * Spec references (AV2 Bitstream & Decoding Process Specification v1.0.0):
 *   - 5.2.6 tile_group_obu(), Annex B.2 bitstream()
 *   - 6.17 frame_header_info(), 5.4.1 sequence_header_obu()
 */
#ifndef SVT_AV2_KEYFRAME_H
#define SVT_AV2_KEYFRAME_H

#include <stddef.h>
#include <stdint.h>
#include "tile_writer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Normal adaptive low-delay KEY/P. Returns the raw end bank only on full
 * packet success; caller may then verify, decay, and publish it. */
int svt_av2_encode_packet_video_models(const SvtAv2FrameHeaderConfig* cfg, uint32_t frame_index,
    const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count,
    uint8_t* out, size_t out_cap, size_t* out_size, struct SvtAv2EntropyModels* end_models);

/* Complete single-picture-header stream, using the production coefficient path. */
int svt_av2_encode_still_picture(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    const SvtAv2TileCoeffBlock* blocks, size_t block_count, uint32_t intra_modes,
    const uint16_t* partition_decisions, size_t decision_count, uint32_t entropy_adaptation,
    uint8_t* out, size_t out_cap, size_t* out_size);

/* Encodes one 64x64 8-bit 4:2:0 key frame into an Annex B AV2 elementary
 * stream written to `out`.
 *
 *   y, u, v          : plane pointers. For every plane the address of the
 *                      sample at (row, col) is `plane + row * stride + col`.
 *                      These are currently ignored by the M1 encoder (DC_PRED,
 *                      zero residual); they are accepted so callers do not have
 *                      to change once residual coding lands. May be NULL.
 *   y_stride, ...    : byte stride of each plane. Ignored in M1.
 *   base_q_idx       : frame quantizer index, must be non-zero (non-lossless).
 *                      The M1 all-zero CDFs are tabulated for the
 *                      COEFF_CDF_Q_CTXS index selected by this value; only the
 *                      range that maps to index 1 (base_q_idx in (90,140]) is
 *                      accepted, so a request outside it fails rather than
 *                      emitting a mismatched stream.
 *   out, out_cap     : caller-owned output buffer and its capacity in bytes.
 *                      There is no hidden fixed-size bound on the coded
 *                      picture: the tile-group payload is sized from out_cap and
 *                      a stream that does not fit fails with
 *                      SVT_AV2_BW_ERROR_NO_SPACE instead of being truncated.
 *                      The minimum accepted capacity is documented below.
 *   out_size         : receives the number of bytes written on success, and is
 *                      set to 0 on every failure so a caller can never mistake
 *                      a stale length for output.
 *
 * Returns 0 on success, or a negative status (see bit_writer.h
 * SvtAv2BitWriterStatus / entropy_encoder.h SvtAv2EntropyStatus); a truncated
 * stream is never produced. Passing NULL for out/out_size, or a base_q_idx
 * outside the supported range, returns a negative status.
 */
#define SVT_AV2_KEYFRAME_MIN_OUT_CAP 16u

int svt_av2_encode_keyframe_64x64(const uint8_t* y, size_t y_stride,
                                  const uint8_t* u, size_t u_stride,
                                  const uint8_t* v, size_t v_stride,
                                  uint32_t base_q_idx, uint8_t* out,
                                  size_t out_cap, size_t* out_size);

/* --- multi-frame sequence (S4, T013) ---------------------------------------
 *
 * A legal multi-frame AV2 sequence is NOT a concatenation of single-picture
 * streams: with single_picture_header_flag == 0 the decoder keeps one frame
 * buffer list across the whole temporal unit. The sequence therefore has the
 * shape
 *
 *     temporal_delimiter
 *     sequence_header_obu (single_picture_header_flag = 0)
 *     for each frame: closed_loop_key_obu   (no repeated delimiter/sequence)
 *
 * The two entry points below produce exactly those pieces; the caller (the
 * encoder context) owns the frame loop, the ordering and the end of stream.
 * They reuse the same OBU/frame-header/entropy writers as the single-picture
 * path, so the two can never disagree about the bit layout. */

/* Writes the temporal delimiter and the multi-frame sequence header into out.
 * `width` x `height` must be a size a valid sequence level admits; the caller
 * checks svt_av2_sequence_header_min_level_for_size() first (a size with no
 * level must be refused, not written with a bogus level).
 *
 * Returns 0 on success, or a negative SvtAv2BitWriterStatus. out_size is set to
 * 0 on every failure. */
int svt_av2_encode_sequence_header_multi(uint32_t width, uint32_t height,
                                         uint8_t* out, size_t out_cap,
                                         size_t* out_size);

/* Writes one closed-loop-key OBU for a multi-frame sequence (single_picture
 * header flag 0). Same plane/base_q_idx contract as
 * svt_av2_encode_keyframe_64x64(); the planes are still accepted-but-unused by
 * this All-Intra/DC_PRED profile. There is no temporal delimiter per frame:
 * the sequence's single leading delimiter covers them.
 *
 * Returns 0 on success, or a negative status; out_size is 0 on every failure. */
int svt_av2_encode_keyframe_64x64_multi(const uint8_t* y, size_t y_stride,
                                        const uint8_t* u, size_t u_stride,
                                        const uint8_t* v, size_t v_stride,
                                        uint32_t base_q_idx, uint8_t* out,
                                        size_t out_cap, size_t* out_size);

int svt_av2_encode_low_delay_packet(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    uint32_t frame_index, const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count,
    uint8_t* out, size_t out_cap, size_t* out_size);

int svt_av2_encode_intra_packet_modes(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    uint32_t frame_index, const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    uint8_t* out, size_t out_cap, size_t* out_size);

int svt_av2_encode_intra_packet_ex(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    uint32_t frame_index, const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    uint8_t* out, size_t out_cap, size_t* out_size);

/* Production M2: one temporal delimiter per packet, sequence header on frame
 * zero only, eight-bit order hint, DCT_DCT-only coefficient tile. The caller
 * owns block storage through the call and obtains no partial packet on error. */
/* M5: also permit DC/V/H intra blocks in the low-delay P structure. */
int svt_av2_encode_low_delay_packet_modes(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    uint32_t frame_index, const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count, uint32_t intra_modes,
    uint8_t* out, size_t out_cap, size_t* out_size);

int svt_av2_encode_packet_tools(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    uint32_t frame_index, const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count, uint32_t low_delay_p,
    uint32_t intra_modes, const uint16_t* partition_decisions, size_t decision_count,
    uint8_t* out, size_t out_cap, size_t* out_size);

/* Low-delay P only (frame_index > 0, low_delay_p == 1).
 * Identical packet/plan contract to svt_av2_encode_packet_tools, but inter MV
 * components are in half-luma-pel units. The first entry uses static CDFs;
 * the entropy entry can opt into frame-local adaptation. Both use the old key path. */
int svt_av2_encode_packet_tools_half(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    uint32_t frame_index, const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count, uint32_t low_delay_p,
    uint32_t intra_modes, const uint16_t* partition_decisions, size_t decision_count,
    uint8_t* out, size_t out_cap, size_t* out_size);

int svt_av2_encode_packet_tools_half_entropy(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    uint32_t frame_index, const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count, uint32_t low_delay_p,
    uint32_t intra_modes, const uint16_t* partition_decisions, size_t decision_count,
    uint32_t entropy_adaptation, uint8_t* out, size_t out_cap, size_t* out_size);

/* Immediate-output P: optional retention of slot 0 and fixed sharp interpolation. */
int svt_av2_encode_packet_inter_profile(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    uint32_t frame_index, const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count, uint32_t intra_modes,
    const uint16_t* partition_decisions, size_t decision_count, uint32_t entropy_adaptation,
    uint32_t mv_half, uint32_t retain_reference, uint32_t sharp_filter,
    uint8_t* out, size_t out_cap, size_t* out_size);

/* Experimental P-only packet with explicit physical map [0,1]. Both prior
 * slots must already be occupied by distinct frames; KEY and bootstrap write
 * through svt_av2_encode_packet_dual_ref. Only zero motion is accepted on
 * coded leaves pending MVP coverage. This wrapper uses static CDFs. */
int svt_av2_encode_packet_two_refs(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    uint32_t frame_index, const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count, uint32_t intra_modes,
    const uint16_t* partition_decisions, size_t decision_count,
    uint32_t mv_half, uint32_t sharp_filter, uint32_t refresh_mask,
    uint8_t* out, size_t out_cap, size_t* out_size);

/* Same zero-MV gate, with independently trained per-frame reference CDFs
 * when entropy_adaptation is 1. Entropy adaptation 0 is byte-equivalent to
 * svt_av2_encode_packet_two_refs. */
int svt_av2_encode_packet_two_refs_entropy(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    uint32_t frame_index, const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count, uint32_t intra_modes,
    const uint16_t* partition_decisions, size_t decision_count, uint32_t entropy_adaptation,
    uint32_t mv_half, uint32_t sharp_filter, uint32_t refresh_mask,
    uint8_t* out, size_t out_cap, size_t* out_size);

/* Low-delay two-slot stream; P publishes exactly one logical reference from
 * reference_slot 0/1 and refreshes the physical slots in refresh_mask (0..3).
 * For frame 0 use modes=NULL/0 and mv_half, sharp_filter, refresh_mask and
 * reference_slot all zero; CLOSED_LOOP_KEY has an implicit all-slots mask,
 * but only populates slot 0. P may select slot 1 after a P refresh writes it. */
/* Same logical map, with optional checked spatial MVP in eighth-luma units.
 * Non-null distance gives the two distinct expanded prior display distances
 * (1..8) in physical order; caller must agree with actual DPB residency.
 * Quarter SHARP, single-reference/DRL0, full aligned 32/64 inter leaves only.
 * A target on the wrong MVD parity lattice is rejected, never rounded. */
int svt_av2_encode_packet_two_refs_projected(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    uint32_t frame_index, const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count, uint32_t intra_modes,
    const uint16_t* partition_decisions, size_t decision_count, uint32_t entropy_adaptation,
    uint32_t mv_half, uint32_t sharp_filter, uint32_t refresh_mask,
    const uint32_t distance[2], uint8_t* out, size_t out_cap, size_t* out_size);

int svt_av2_encode_packet_dual_ref(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    uint32_t frame_index, const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count, uint32_t intra_modes,
    const uint16_t* partition_decisions, size_t decision_count, uint32_t entropy_adaptation,
    uint32_t mv_half, uint32_t sharp_filter, uint32_t refresh_mask, uint32_t reference_slot,
    uint8_t* out, size_t out_cap, size_t* out_size);

int svt_av2_encode_packet_tools_entropy(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    uint32_t frame_index, const SvtAv2TileCoeffBlock* blocks, size_t block_count,
    const SvtAv2TileBlockMode* modes, size_t mode_count, uint32_t low_delay_p,
    uint32_t intra_modes, const uint16_t* partition_decisions, size_t decision_count,
    uint32_t entropy_adaptation, uint8_t* out, size_t out_cap, size_t* out_size);

int svt_av2_encode_still_picture_tx(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    const SvtAv2TileCoeffBlock* blocks, size_t block_count, uint32_t intra_modes,
    const uint16_t* partition_decisions, size_t decision_count, uint32_t entropy_adaptation,
    uint32_t transform_selection, uint8_t* out, size_t out_cap, size_t* out_size);

int svt_av2_encode_intra_packet(uint32_t width, uint32_t height,
    uint32_t base_q_idx, uint32_t frame_index, const SvtAv2TileCoeffBlock* blocks,
    size_t block_count, uint8_t* out, size_t out_cap, size_t* out_size);

int svt_av2_encode_still_picture_mrls(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    const SvtAv2TileCoeffBlock* blocks, size_t block_count, uint32_t intra_modes,
    const uint16_t* partition_decisions, size_t decision_count, uint32_t entropy_adaptation,
    uint32_t transform_selection, uint32_t enable_mrls,
    uint8_t* out, size_t out_cap, size_t* out_size);

int svt_av2_encode_still_picture_mhccp(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    const SvtAv2TileCoeffBlock* blocks, size_t block_count, uint32_t intra_modes,
    const uint16_t* partition_decisions, size_t decision_count, uint32_t entropy_adaptation,
    uint32_t transform_selection, uint32_t enable_mrls, uint32_t enable_mhccp,
    uint8_t* out, size_t out_cap, size_t* out_size);

/* Complete sequence prefix and aligned tile/frame header; excludes tile OBU
 * payload length and entropy finalization, which require the final real packet. */
typedef struct SvtAv2Ccso64HeaderSize {size_t sequence_bytes,frame_bits,frame_bytes;} SvtAv2Ccso64HeaderSize;
int svt_av2_ccso64_header_size(const SvtAv2FrameHeaderConfig* cfg,SvtAv2Ccso64HeaderSize* size);
/* Packet only: never applies the filter or mutates reconstruction. */
int svt_av2_encode_still_picture_ccso64_packet(const SvtAv2FrameHeaderConfig* cfg,
    const SvtAv2TileCoeffBlock* blocks,size_t count,uint8_t* out,size_t capacity,size_t* bytes);

/* Internal forced 64-unit entry; recon must contain the complete unfiltered frame. */
int svt_av2_encode_still_picture_ccso64(uint32_t width, uint32_t height,
    uint32_t bit_depth, uint32_t chroma_format, uint32_t base_q_idx,
    const SvtAv2TileCoeffBlock* blocks, size_t block_count, uint32_t intra_modes,
    const uint16_t* decisions, size_t decision_count, uint32_t entropy,
    uint32_t transform, uint32_t mrl, uint32_t mh,
    const SvtAv2Ccso64Plan* ccso64, SvtAv2FrameBuffer* recon,
    uint8_t* out, size_t out_cap, size_t* out_size);

#ifdef __cplusplus
}
#endif

#endif /* SVT_AV2_KEYFRAME_H */
