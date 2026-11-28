#ifndef SVT_AV2_MOTION_SYNTAX_H
#define SVT_AV2_MOTION_SYNTAX_H
#include "entropy_encoder.h"
#include "block_geometry.h"
/* One entry per Y/U/V group; signed MV components normally use the frame's
 * native luma units: integer (±16), half (±32), or quarter (±64). The
 * projected two-reference profile uses exact eighth-luma components (±128)
 * despite coding their differences on the frame's quarter lattice. */
typedef struct SvtAv2TileBlockMode {
    uint32_t is_inter;
    int32_t mv_x, mv_y;
    uint32_t reference_rank; /* Read only in two-logical-reference tiles. */
} SvtAv2TileBlockMode;
typedef struct SvtAv2MotionCell {
    int32_t mv_x, mv_y;
    uint8_t valid, is_inter, is_new, skip_txfm; /* is_inter: 0 intra, 1 + inter rank. */
} SvtAv2MotionCell;
typedef struct SvtAv2MotionSyntax {
    SvtAv2MotionCell* cells;
    uint32_t cols, rows;
    uint32_t mv_half; /* Historical name: 0=integer, 1=half, 2=quarter. */
    uint32_t wide_motion;
    uint32_t two_refs; /* Set by the tile writer from its verified frame header. */
    uint32_t opposite_directions; /* Rank 0 is future and rank 1 is past at a paired midpoint. */
    uint32_t ref_distance[2]; /* Both nonzero only for supported two-past projection. */
} SvtAv2MotionSyntax;
typedef struct SvtAv2MotionPrefix {
    const SvtAv2BlockView* blocks;
    const SvtAv2MotionCell* cells;
    size_t count;
} SvtAv2MotionPrefix;
/* Exact DRL0 predictor. The optional speculative earlier leaves shadow only
 * their own coordinates; the committed grid and CDF are never mutated. */
int svt_av2_motion_predict(const SvtAv2MotionSyntax* state, const SvtAv2BlockView* block,
    uint32_t rank, const SvtAv2MotionPrefix* prefix, int32_t* mv_x, int32_t* mv_y);
int svt_av2_motion_syntax_init(SvtAv2MotionSyntax* state, uint32_t width, uint32_t height);
int svt_av2_motion_syntax_init_precision(SvtAv2MotionSyntax* state, uint32_t width,
    uint32_t height, uint32_t mv_half);
void svt_av2_motion_syntax_free(SvtAv2MotionSyntax* state);
int svt_av2_motion_write_block(SvtAv2MotionSyntax* state, SvtAv2EntropyEncoder* enc,
    const SvtAv2BlockView* block, const SvtAv2TileBlockMode* mode, uint32_t skip_txfm);
int svt_av2_motion_write_difference(SvtAv2EntropyEncoder* enc, int32_t dx, int32_t dy);
#endif
