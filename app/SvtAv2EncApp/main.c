/*
 * SvtAv2EncApp - command line front end.
 *
 * Responsibilities:
 *   * read Y4M stream metadata and report it
 *   * drive the public encoder API through a full call sequence
 *
 * Encode path (M1, T013): the app is a pure client of the public API. For each
 * frame it fills a SvtAv2Picture view over the Y4M frame buffer, calls
 * svt_av2_enc_send_picture(), then takes the coded bytes with
 * svt_av2_enc_get_packet_ex() and writes them out. It does NOT call the internal
 * keyframe entry point and does not assemble OBUs (contract section 7: the CLI
 * and the library share one encode path, so a stream written by the app is
 * exactly what the library produces).
 *
 * Unsupported input exits 14; an encoder failure exits non-zero.
 *
 * The M1 encoder is ALL-INTRA/DC_PRED with zero residual: the input pixels are
 * accepted for API stability but not yet coded, so the reconstruction is uniform
 * mid-grey (128). This is honest documented behaviour, not a silent shortcut.
 */

#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "svt_av2.h"
#include "y4m.h"
#include "pyramid8.h"

/* M4: single-threaded 8-bit420 / 10-bit420,444, DC prediction, fixed user QP. */
#define SVT_AV2_M1_DEFAULT_QP 100
#define SVT_AV2_APP_PACKET_INITIAL_BYTES 4096u
#define SVT_AV2_APP_EXIT_UNSUPPORTED_INPUT 8

#include <inttypes.h>

typedef struct AppOptions {
    const char* input_path;
    const char* output_path;
    const char* recon_path; /* NULL = do not export the reconstruction */
    int32_t     qp;
    int32_t     video_key_qp;
    int32_t     video_anchor_qp;
    int32_t     preset;
    int32_t     threads;
    int32_t     tile_columns;
    int32_t     frames; /* -1 = all frames in the source */
    int32_t     deterministic;
    SvtAv2PredictionStructure prediction_structure;
    uint32_t intra_mode_selection;
    uint32_t partition_selection;
    uint32_t transform_selection;
    uint32_t angle_search;
    uint32_t mrl_search;
    uint32_t tree_refinement;
    uint32_t rd_lambda_model;
    uint32_t rd_lambda_scale;
    uint32_t cfl_search;
    uint32_t mhccp_search;
    uint32_t intra_joint;
    uint32_t rd_mode;
    uint32_t coeff_optimization;
    uint32_t motion_optimization;
    uint32_t inter_skip_search;
    uint32_t inter_ref_period;
    uint32_t inter_fine;
    uint32_t inter_near;
    uint32_t inter_tail;
    uint32_t dual_ref_period;
    uint32_t dual_ref_policy;
    uint32_t dual_ref_cadence;
    uint32_t entropy_adaptation;
    uint32_t cdf_inherit;
    uint32_t still_picture;
    SvtAv2ContentType content_type; /* NONE unless supplied explicitly */
    int         verbose;
} AppOptions;

static void app_options_set_defaults(AppOptions* options) {
    memset(options, 0, sizeof(*options));
    options->qp = SVT_AV2_M1_DEFAULT_QP;
    options->preset = SVT_AV2_PRESET_SLOW;
    options->threads = 1;
    options->tile_columns = 1;
    options->frames = -1;
    options->deterministic = 1;
    options->inter_ref_period = 1u;
}

static void print_usage(const char* program) {
    printf("SvtAv2EncApp - SVT-AV2 reference front end\n"
           "\n"
           "Usage:\n"
           "  %s --input <file.y4m> [options]\n"
           "\n"
           "Options:\n"
           "  --input <path>        Y4M input file (required)\n"
           "  --output <path>       Encode to <path> (M4: 8-bit 420 / 10-bit 420,444).\n"
           "                        Omit to only report input metadata.\n"
           "  --frames <n>          Encode n frames: 0 = none.\n"
           "  --still-picture       Single-picture header; encode exactly one frame\n"
           "                        Default: all available frames.\n"
           "  --export-recon <path> Also write the encoder's own reconstruction\n"
           "                        (tight planar raw; 10-bit uses u16le, one frame after\n"
           "                        another) to <path>. Additive output: it does\n"
           "                        not change the encoded stream.\n"
           "  --pred-structure <all-intra|low-delay-p> Prediction structure (default: all-intra)\n"
           "  --intra-select <dc|adaptive|basic|directional|cfl>  extended single-picture Y/UV RD\n"
           "  --partition-select <fixed|adaptive|fine|mixed|rect|longrect|mixedrect|mixed8|thinrect|mixedthin|thin64>  fine=64/32/16; mixed adds 8 and quadrant splits; rect adds 16x8/8x16; longrect adds six larger rectangles; mixedrect adds mixed rectangular quadrants; mixed8 adds mixed small rectangles; thinrect adds uniform 8x32/32x8; mixedthin adds mixed thin quadrants; thin64 adds uniform 8x64/64x8 (still + basic/directional + RD)\n"
           "  --rd-mode <off|exhaustive|pruned|fast>  M6 bounded RD (default: off)\n"
           "  --tree-refine <off|rd|wide|screen>  screen=bounded mixed8 still search; rd=best two trees; wide preserves seeds before long rectangles; thinrect/mixedthin also preserve pre-thin seeds with up to eight finalists; thin64 preserves those eight plus both new shapes (still + fine or later + RD)\n"
           "  --rd-lambda <original|avm-ai|avm-ai150>  RD lambda model (still RD or pruned low-delay P; default original)\n"
           "  --rd-lambda-scale <50..250>  Percent of selected AVM lambda (default 100)\n"
           "  --content-type <animation|real> Externally supplied content hint (default: none)\n"
           "                        Animation guard: compiled on + 10-bit 420 still + 1920x1080/3840x2160\n"
           "  --cfl-search <off|rd>  Also search classic explicit UV alpha (still + cfl + RD)\n"
           "  --mhccp-search <off|rd>  Also search causal MHCCP UV prediction (still + cfl + RD)\n"
           "  --intra-joint <off|rd>  Bounded Y mode/TX joint search (still + directional/cfl + RD + full TX; default off)\n"
           "  --mrl-search <off|rd>  Bounded square/rectangle Y multiple reference lines (still + directional/CfL + RD)\n"
           "  --angle-search <off|rd>  Bounded nonzero Y angles (still + directional/CfL + RD)\n"
           "  --transform-select <dct|primary|full>  Y 4/8/16 DCT/ADST subset; full also supports 8x16/16x8 (still + basic/directional + RD)\n"
           "  fast: still intra directional/cfl, partition <= mixed8, joint off\n"
           "  --coeff-opt <off|rd|rdoq>  rdoq=bounded level/EOB search (still + RD)\n"
           "  --motion-opt <off|rd|wide|half|sharp|quarter> Fractional modes require low-delay-p, pruned RD and fixed/adaptive partitions\n"
           "  --inter-skip <off|rd>  Explicit predictor-only inter candidate (default: off)\n"
           "  --ref-period <1|2|4|8>  Refresh the single reference every n pictures (default: 1)\n"
           "                        Both require low-delay-p, pruned RD, fixed/64-32 partition, coeff off\n"
           "  --entropy-adapt <off|on> Frame-local CDF updates (default: off)\n"
           "                        With RD: still (motion off, coeff off/rdoq), or low-delay-p\n"
           "                        (pruned, DC or DC/V/H, fixed or 64/32 partition, coeff off)\n"
           "  --cdf-inherit <off|on> Seed P entropy from logical reference 0 (default: off)\n"
           "                        Requires low-delay-p, pruned RD and entropy adaptation on\n"
           "  --inter-fine <off|rd>  Local 32-to-16 search (quarter, adaptive partitions, skip RD off)\n"
           "  --inter-near <off|rd>  Test the exact DRL0 motion predictor in RD (default: off)\n"
           "                        Requires pruned low-delay-p, quarter, fixed/adaptive partitions, skip RD off\n"
           "  --inter-tail <off|rd>  Experimental inter residual tail pruning (default: off)\n"
           "                        Requires pruned low-delay-p, quarter, fixed/adaptive partitions, skip RD off\n"
           "                        With dual references, requires --dual-ref-policy block\n"
           "  --dual-ref-period <0|2|4|8>  Optional second DPB slot (quarter, --ref-period 1)\n"
           "  --dual-ref-policy <auto|last|long|block>  Reference selection (default: auto)\n"
           "                        block: experimental per-block LAST/LONG (skip RD off)\n"
           "  --dual-ref-cadence <fixed|adaptive>  Long-slot refresh timing (default: fixed)\n"
           "                        adaptive requires --dual-ref-policy block and --dual-ref-period 4\n"
           "  --qp <1..255>         CQP (10-bit actual qindex=QP+48) (default: 100)\n"
           "  --key-qp <1..255>     Override the first KEY QP in low-delay P only (default: inherit --qp)\n"
           "  --anchor-qp <1..255>  Override QP on P frames refreshing the long slot (requires dual ref)\n"
           "  --preset <0|4|8|12>   Speed preset placeholder (default: 4)\n"
           "  --tile-columns <n>    Still-picture fast RD uniform tile columns: 1, 2, 4, 8 (default: 1)\n"
           "  --threads <n>         Threads for still-picture fast RD angle search and GDF (default: 1)\n"
           "  --deterministic       Enable deterministic mode (default)\n"
           "  --no-deterministic    Disable deterministic mode\n"
           "  --verbose             Print per frame progress\n"
           "  --version             Print build information and exit\n"
           "  --help                Print this message and exit\n"
           "\n"
           "With --output the app writes an AV2 elementary stream for the supported\n"
           "M4 formats: 8-bit 420, 10-bit 420 and 10-bit 444. All-Intra is the default.\n"
           "LOW_DELAY_P uses one previously reconstructed frame. --export-recon additionally writes the encoder's own\n"
           "reconstruction so a decoder run can be compared against it.\n",
           program);
}

static void print_version(void) {
    const SvtAv2EncoderInfo* info = svt_av2_enc_get_build_info();
    const SvtAv2BuildInfo*   build = svt_av2_get_build_info();
    printf("SvtAv2EncApp %s (milestone %s)\n", info->version_string, info->milestone);
    printf("  build type : %s\n", info->build_type);
    printf("  compiler   : %s\n", info->compiler);
    printf("  avm oracle : %s\n", (build->oracle_dir[0] != '\0') ? build->oracle_dir : "(not configured)");
}

static int parse_int32(const char* text, int32_t* out) {
    if (text == NULL || *text == '\0' || out == NULL) {
        return -1;
    }
    char*              end = NULL;
    const long int     value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < -2147483647L - 1L || value > 2147483647L) {
        return -1;
    }
    *out = (int32_t)value;
    return 0;
}

/* Returns 0 on success, 1 when the program should exit successfully (help /
 * version), -1 on argument error. */
static int app_parse_args(int argc, char** argv, AppOptions* options) {
    app_options_set_defaults(options);

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const int   has_next = (i + 1) < argc;

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return 1;
        }
        if (strcmp(arg, "--version") == 0) {
            print_version();
            return 1;
        }
        if (strcmp(arg, "--verbose") == 0) {
            options->verbose = 1;
            continue;
        }
        if (strcmp(arg, "--deterministic") == 0) {
            options->deterministic = 1;
            continue;
        }
        if (strcmp(arg, "--no-deterministic") == 0) {
            options->deterministic = 0;
            continue;
        }

        const char* value = has_next ? argv[i + 1] : NULL;

        if (strcmp(arg, "--input") == 0 || strcmp(arg, "-i") == 0) {
            if (!has_next) {
                fprintf(stderr, "error: %s requires a value\n", arg);
                return -1;
            }
            options->input_path = value;
            ++i;
            continue;
        }
        if (strcmp(arg, "--output") == 0 || strcmp(arg, "-o") == 0) {
            if (!has_next) {
                fprintf(stderr, "error: %s requires a value\n", arg);
                return -1;
            }
            options->output_path = value;
            ++i;
            continue;
        }
        if (strcmp(arg, "--still-picture") == 0) {
            options->still_picture = 1u;
            continue;
        }
        if (strcmp(arg, "--frames") == 0) {
            if (!has_next || parse_int32(value, &options->frames) != 0 || options->frames < 0) {
                fprintf(stderr, "error: --frames requires a non-negative integer\n");
                return -1;
            }
            ++i;
            continue;
        }
        if (strcmp(arg, "--export-recon") == 0) {
            if (!has_next) {
                fprintf(stderr, "error: %s requires a value\n", arg);
                return -1;
            }
            options->recon_path = value;
            ++i;
            continue;
        }
        if (strcmp(arg, "--pred-structure") == 0) {
            if (!has_next || (strcmp(value, "all-intra") != 0 && strcmp(value, "low-delay-p") != 0)) {
                fprintf(stderr, "error: --pred-structure requires all-intra or low-delay-p\n");
                return -1;
            }
            options->prediction_structure = strcmp(value, "all-intra") == 0 ? SVT_AV2_PRED_ALL_INTRA : SVT_AV2_PRED_LOW_DELAY_P;
            ++i;
            continue;
        }
        if (strcmp(arg, "--rd-mode") == 0) {
            if (!has_next || (strcmp(value, "off") != 0 && strcmp(value, "exhaustive") != 0 && strcmp(value, "pruned") != 0 && strcmp(value, "fast") != 0)) {
                fprintf(stderr, "error: --rd-mode requires off, exhaustive, pruned or fast\n");
                return -1;
            }
            options->rd_mode = strcmp(value, "off") == 0 ? 0u : (strcmp(value, "exhaustive") == 0 ? 1u : strcmp(value, "pruned") == 0 ? 2u : 3u);
            ++i;
            continue;
        }
        if (strcmp(arg, "--coeff-opt") == 0) {
            if (!has_next || (strcmp(value, "off") != 0 && strcmp(value, "rd") != 0 && strcmp(value, "rdoq") != 0)) {
                fprintf(stderr, "error: --coeff-opt requires off, rd or rdoq\n");
                return -1;
            }
            options->coeff_optimization = strcmp(value, "rdoq") == 0 ? 2u : (uint32_t)(strcmp(value, "rd") == 0);
            ++i;
            continue;
        }
        if (strcmp(arg, "--entropy-adapt") == 0) {
            if (!has_next || (strcmp(value, "off") != 0 && strcmp(value, "on") != 0)) {
                fprintf(stderr, "error: --entropy-adapt requires off or on\n");
                return -1;
            }
            options->entropy_adaptation = strcmp(value, "on") == 0;
            ++i;
            continue;
        }
        if (strcmp(arg, "--cdf-inherit") == 0) {
            if (!has_next || (strcmp(value, "off") != 0 && strcmp(value, "on") != 0)) {
                fprintf(stderr, "error: --cdf-inherit requires off or on\n");
                return -1;
            }
            options->cdf_inherit = strcmp(value, "on") == 0 ? 1u : 0u;
            ++i;
            continue;
        }
        if (strcmp(arg, "--inter-fine") == 0) {
            if (!has_next || (strcmp(value, "off") && strcmp(value, "rd"))) {
                fprintf(stderr, "error: --inter-fine requires off or rd\n");
                return -1;
            }
            options->inter_fine = strcmp(value, "rd") == 0 ? 1u : 0u;
            ++i;
            continue;
        }
        if (strcmp(arg, "--inter-near") == 0) {
            if (value == NULL || (strcmp(value, "off") != 0 && strcmp(value, "rd") != 0)) {
                fprintf(stderr, "error: --inter-near requires off or rd\n");
                return -1;
            }
            options->inter_near = strcmp(value, "rd") == 0 ? 1u : 0u;
            ++i;
            continue;
        }
        if (strcmp(arg, "--inter-tail") == 0) {
            if (!has_next || (strcmp(value, "off") && strcmp(value, "rd"))) {
                fprintf(stderr, "error: --inter-tail requires off or rd\n");
                return -1;
            }
            options->inter_tail = strcmp(value, "rd") == 0 ? 1u : 0u;
            ++i;
            continue;
        }
        if (strcmp(arg, "--inter-skip") == 0) {
            if (!has_next || (strcmp(value, "off") && strcmp(value, "rd"))) {
                fprintf(stderr, "error: --inter-skip requires off or rd\n");
                return -1;
            }
            options->inter_skip_search = strcmp(value, "rd") == 0 ? 1u : 0u;
            ++i;
            continue;
        }
        if (strcmp(arg, "--dual-ref-period") == 0) {
            int32_t period;
            if (!has_next || parse_int32(value, &period) != 0 ||
                (period != 0 && period != 2 && period != 4 && period != 8)) {
                fprintf(stderr, "error: --dual-ref-period requires 0, 2, 4 or 8\n");
                return -1;
            }
            options->dual_ref_period = (uint32_t)period;
            ++i;
            continue;
        }
        if (strcmp(arg, "--dual-ref-policy") == 0) {
            if (!has_next || (strcmp(value, "auto") && strcmp(value, "last") && strcmp(value, "long") &&
                strcmp(value, "block"))) {
                fprintf(stderr, "error: --dual-ref-policy requires auto, last, long or block\n");
                return -1;
            }
            options->dual_ref_policy = strcmp(value, "auto") == 0 ? 0u : strcmp(value, "last") == 0 ? 1u :
                strcmp(value, "long") == 0 ? 2u : 3u;
            ++i;
            continue;
        }
        if (strcmp(arg, "--dual-ref-cadence") == 0) {
            if (!has_next || (strcmp(value, "fixed") && strcmp(value, "adaptive"))) {
                fprintf(stderr, "error: --dual-ref-cadence requires fixed or adaptive\n");
                return -1;
            }
            options->dual_ref_cadence = strcmp(value, "adaptive") == 0 ? 1u : 0u;
            ++i;
            continue;
        }
        if (strcmp(arg, "--ref-period") == 0) {
            int32_t period;
            if (!has_next || parse_int32(value, &period) != 0 ||
                (period != 1 && period != 2 && period != 4 && period != 8)) {
                fprintf(stderr, "error: --ref-period requires 1, 2, 4 or 8\n");
                return -1;
            }
            options->inter_ref_period = (uint32_t)period;
            ++i;
            continue;
        }
        if (strcmp(arg, "--motion-opt") == 0) {
            if (!has_next || (strcmp(value, "off") != 0 && strcmp(value, "rd") != 0 &&
                strcmp(value, "wide") != 0 && strcmp(value, "half") != 0 && strcmp(value, "sharp") != 0 &&
                strcmp(value, "quarter") != 0)) {
                fprintf(stderr, "error: --motion-opt requires off, rd, wide, half, sharp or quarter\n");
                return -1;
            }
            options->motion_optimization = strcmp(value, "off") == 0 ? 0u :
                strcmp(value, "rd") == 0 ? 1u : strcmp(value, "wide") == 0 ? 2u :
                strcmp(value, "half") == 0 ? 3u : strcmp(value, "sharp") == 0 ? 4u : 5u;
            ++i;
            continue;
        }
        if (strcmp(arg, "--angle-search") == 0) {
            if (!has_next || (strcmp(value, "off") != 0 && strcmp(value, "rd") != 0)) {
                fprintf(stderr, "error: --angle-search requires off or rd\n");
                return -1;
            }
            options->angle_search = (uint32_t)(strcmp(value, "rd") == 0);
            ++i; continue;
        }
        if (strcmp(arg, "--tree-refine") == 0) {
            if (!has_next || (strcmp(value, "off") != 0 && strcmp(value, "rd") != 0 && strcmp(value, "wide") != 0 && strcmp(value, "screen") != 0)) {
                fprintf(stderr, "error: --tree-refine requires off, rd, wide or screen\n");
                return -1;
            }
            options->tree_refinement = strcmp(value, "screen") == 0 ? 3u : strcmp(value, "wide") == 0 ? 2u : (uint32_t)(strcmp(value, "rd") == 0);
            ++i; continue;
        }
        if (strcmp(arg, "--content-type") == 0) {
            if (!has_next || (strcmp(value,"animation")!=0 && strcmp(value,"real")!=0)) {
                fprintf(stderr,"error: --content-type requires animation or real\n");
                return -1;
            }
            options->content_type=strcmp(value,"animation")==0 ?
                SVT_AV2_CONTENT_ANIMATION : SVT_AV2_CONTENT_REAL;
            ++i;continue;
        }
        if (strcmp(arg, "--rd-lambda") == 0) {
            if (!has_next || (strcmp(value, "original") != 0 && strcmp(value, "avm-ai") != 0 && strcmp(value, "avm-ai150") != 0)) {
                fprintf(stderr, "error: --rd-lambda requires original, avm-ai or avm-ai150\n");
                return -1;
            }
            options->rd_lambda_model = strcmp(value, "avm-ai150") == 0 ? 2u : (uint32_t)(strcmp(value, "avm-ai") == 0);
            ++i; continue;
        }
        if (strcmp(arg, "--rd-lambda-scale") == 0) {
            char* end = NULL;
            long sc = value ? strtol(value, &end, 10) : 0;
            if (value == NULL || end == value || *end != '\0' || sc < 50 || sc > 250) {
                fprintf(stderr, "error: --rd-lambda-scale requires an integer in 50..250\n");
                return -1;
            }
            options->rd_lambda_scale = (uint32_t)sc;
            ++i; continue;
        }
        if (strcmp(arg, "--mrl-search") == 0) {
            if (!has_next || (strcmp(value, "off") != 0 && strcmp(value, "rd") != 0)) {
                fprintf(stderr, "error: --mrl-search requires off or rd\n");
                return -1;
            }
            options->mrl_search = (uint32_t)(strcmp(value, "rd") == 0);
            ++i; continue;
        }
        if (strcmp(arg, "--cfl-search") == 0) {
            if (!has_next || (strcmp(value, "off") != 0 && strcmp(value, "rd") != 0)) {
                fprintf(stderr, "error: --cfl-search requires off or rd\n");
                return -1;
            }
            options->cfl_search = (uint32_t)(strcmp(value, "rd") == 0);
            ++i; continue;
        }
        if (strcmp(arg, "--mhccp-search") == 0) {
            if (!has_next || (strcmp(value, "off") != 0 && strcmp(value, "rd") != 0)) {
                fprintf(stderr, "error: --mhccp-search requires off or rd\n");
                return -1;
            }
            options->mhccp_search = (uint32_t)(strcmp(value, "rd") == 0);
            ++i; continue;
        }
        if (strcmp(arg, "--intra-joint") == 0) {
            if (!has_next || (strcmp(value, "off") != 0 && strcmp(value, "rd") != 0)) {
                fprintf(stderr, "error: --intra-joint requires off or rd\n");
                return -1;
            }
            options->intra_joint = (uint32_t)(strcmp(value, "rd") == 0);
            ++i; continue;
        }
        if (strcmp(arg, "--transform-select") == 0) {
            if (!has_next || (strcmp(value, "dct") != 0 && strcmp(value, "primary") != 0 && strcmp(value, "full") != 0)) {
                fprintf(stderr, "error: --transform-select requires dct, primary or full\n");
                return -1;
            }
            options->transform_selection = strcmp(value, "full") == 0 ? 2u : (uint32_t)(strcmp(value, "primary") == 0);
            ++i; continue;
        }
        if (strcmp(arg, "--partition-select") == 0) {
            if (!has_next || (strcmp(value, "fixed") != 0 && strcmp(value, "adaptive") != 0 && strcmp(value, "fine") != 0 && strcmp(value, "mixed") != 0 && strcmp(value, "rect") != 0 && strcmp(value, "longrect") != 0 && strcmp(value, "mixedrect") != 0 && strcmp(value, "mixed8") != 0 && strcmp(value, "thinrect") != 0 && strcmp(value, "mixedthin") != 0 && strcmp(value, "thin64") != 0)) {
                fprintf(stderr, "error: --partition-select requires fixed, adaptive, fine, mixed, rect, longrect, mixedrect, mixed8, thinrect, mixedthin or thin64\n");
                return -1;
            }
            options->partition_selection = strcmp(value, "thin64") == 0 ? 10u : strcmp(value, "mixedthin") == 0 ? 9u : strcmp(value, "thinrect") == 0 ? 8u : strcmp(value, "mixed8") == 0 ? 7u : strcmp(value, "mixedrect") == 0 ? 6u : strcmp(value, "longrect") == 0 ? 5u : strcmp(value, "rect") == 0 ? 4u : strcmp(value, "mixed") == 0 ? 3u :
                strcmp(value, "fine") == 0 ? 2u : (uint32_t)(strcmp(value, "adaptive") == 0);
            ++i;
            continue;
        }
        if (strcmp(arg, "--intra-select") == 0) {
            if (!has_next || (strcmp(value, "dc") != 0 && strcmp(value, "adaptive") != 0 && strcmp(value, "basic") != 0 && strcmp(value, "directional") != 0 && strcmp(value, "cfl") != 0)) {
                fprintf(stderr, "error: --intra-select requires dc, adaptive, basic, directional or cfl\n");
                return -1;
            }
            options->intra_mode_selection = strcmp(value, "cfl") == 0 ? 4u : strcmp(value, "directional") == 0 ? 3u : strcmp(value, "basic") == 0 ? 2u : (uint32_t)(strcmp(value, "adaptive") == 0);
            ++i;
            continue;
        }
        if (strcmp(arg, "--key-qp") == 0 || strcmp(arg, "--anchor-qp") == 0) {
            int32_t* qp = strcmp(arg, "--key-qp") == 0 ? &options->video_key_qp : &options->video_anchor_qp;
            if (!has_next || parse_int32(value, qp) != 0 || *qp < 1 || *qp > 255) {
                fprintf(stderr, "error: %s requires an integer in [1, 255]\n", arg);
                return -1;
            }
            ++i;
            continue;
        }
        if (strcmp(arg, "--qp") == 0) {
            if (!has_next || parse_int32(value, &options->qp) != 0 || options->qp < 1 || options->qp > 255) {
                fprintf(stderr, "error: --qp requires an integer in [1, 255]\n");
                return -1;
            }
            ++i;
            continue;
        }
        if (strcmp(arg, "--preset") == 0) {
            if (!has_next || parse_int32(value, &options->preset) != 0) {
                fprintf(stderr, "error: --preset requires an integer\n");
                return -1;
            }
            ++i;
            continue;
        }
        if (strcmp(arg, "--tile-columns") == 0) {
            if (!has_next || parse_int32(value, &options->tile_columns) != 0 ||
                (options->tile_columns != 1 && options->tile_columns != 2 &&
                 options->tile_columns != 4 && options->tile_columns != 8)) {
                fprintf(stderr, "error: --tile-columns requires 1, 2, 4, or 8\n");
                return -1;
            }
            ++i;
            continue;
        }
        if (strcmp(arg, "--threads") == 0) {
            if (!has_next || parse_int32(value, &options->threads) != 0 || options->threads < 1) {
                fprintf(stderr, "error: --threads requires a positive integer\n");
                return -1;
            }
            ++i;
            continue;
        }

        fprintf(stderr, "error: unknown argument '%s'\n", arg);
        return -1;
    }

    if (options->input_path == NULL) {
        fprintf(stderr, "error: --input is required\n");
        return -1;
    }
    if (options->cdf_inherit && (options->still_picture ||
        options->prediction_structure != SVT_AV2_PRED_LOW_DELAY_P || options->rd_mode != 2u ||
        !options->entropy_adaptation)) {
        fprintf(stderr, "error: --cdf-inherit on requires non-still low-delay-p, --rd-mode pruned "
                        "and --entropy-adapt on\n");
        return -1;
    }
    if (options->dual_ref_cadence && (options->dual_ref_policy != 3u ||
        options->dual_ref_period != 4u || options->inter_skip_search)) {
        fprintf(stderr, "error: adaptive cadence requires --dual-ref-policy block, "
                        "--dual-ref-period 4 and --inter-skip off\n");
        return -1;
    }
    if (options->inter_near && (options->still_picture ||
        options->prediction_structure != SVT_AV2_PRED_LOW_DELAY_P || options->rd_mode != 2u ||
        options->motion_optimization != 5u || options->partition_selection > 1u ||
        options->inter_skip_search || options->intra_mode_selection > 1u)) {
        fprintf(stderr, "error: --inter-near rd requires non-still low-delay-p, --rd-mode pruned, "
                        "--motion-opt quarter, fixed/adaptive partitions, simple intra, and --inter-skip off\n");
        return -1;
    }
    if (options->inter_tail && (options->still_picture ||
        options->prediction_structure != SVT_AV2_PRED_LOW_DELAY_P || options->rd_mode != 2u ||
        options->motion_optimization != 5u || options->partition_selection > 1u || options->inter_skip_search ||
        options->intra_mode_selection > 1u || options->coeff_optimization || options->transform_selection ||
        (options->dual_ref_period ? options->dual_ref_policy != 3u : options->dual_ref_policy != 0u))) {
        fprintf(stderr, "error: --inter-tail rd requires non-still low-delay-p, --rd-mode pruned, "
                        "--motion-opt quarter, fixed/adaptive partitions, simple intra, default coefficients and transforms, "
                        "--inter-skip off, and block policy if dual references are enabled\n");
        return -1;
    }
    if (options->still_picture && (options->frames == 0 || options->frames > 1 ||
        options->prediction_structure != SVT_AV2_PRED_ALL_INTRA)) {
        fprintf(stderr, "error: --still-picture requires one frame and all-intra\n");
        return -1;
    }
    return 0;
}

static void app_print_y4m_info(const char* path, const SvtAv2Y4mInfo* info) {
    printf("input       : %s\n", path);
    printf("resolution  : %ux%u\n", info->width, info->height);
    printf("frame rate  : %u/%u (%.4f fps)\n",
           info->frame_rate_num,
           info->frame_rate_den,
           (double)info->frame_rate_num / (double)info->frame_rate_den);
    printf("chroma      : %s\n", svt_av2_chroma_format_name(info->chroma_format));
    printf("bit depth   : %u\n", (unsigned)info->bit_depth);
    printf("colorspace  : %s\n", (info->color_space[0] != '\0') ? info->color_space : "(default)");
    printf("interlace   : %s\n", info->interlaced ? "interlaced" : "progressive");

    size_t samples = 0;
    if (svt_av2_compute_frame_sample_count(info->width, info->height, info->chroma_format, &samples) ==
        SVT_AV2_ERROR_NONE) {
        const size_t bytes = samples * svt_av2_bytes_per_sample(info->bit_depth);
        printf("frame size  : %zu samples / %zu bytes (unpadded)\n", samples, bytes);
    }
}

/* --- M1 input profile ------------------------------------------------------ */

/* The M1 encoder path accepts exactly one profile: 64x64, 8-bit, 4:2:0,
 * progressive. Returns 0 when the input is supported, -1 otherwise and prints
 * the reason to stderr. */
static int app_check_m1_input(const SvtAv2Y4mInfo* info) {
    if (info->width % 8u != 0 || info->height % 8u != 0) {
        fprintf(stderr, "error: M4 requires dimensions divisible by 8\n");
        return -1;
    }
    if (info->bit_depth != SVT_AV2_BIT_DEPTH_8 && info->bit_depth != SVT_AV2_BIT_DEPTH_10) {
        fprintf(stderr, "error: M4 supports only 8-bit or 10-bit input, got %u-bit\n", (unsigned)info->bit_depth);
        return -1;
    }
    if (info->chroma_format != SVT_AV2_CHROMA_420 &&
        !(info->bit_depth == SVT_AV2_BIT_DEPTH_10 && info->chroma_format == SVT_AV2_CHROMA_444)) {
        fprintf(stderr,
                "error: M4 supports 8-bit 420, 10-bit 420 or 10-bit 444 input, got %s\n",
                svt_av2_chroma_format_name(info->chroma_format));
        return -1;
    }
    if (info->interlaced) {
        fprintf(stderr, "error: M4 supports only progressive input\n");
        return -1;
    }
    return 0;
}

/* --- frame iteration ------------------------------------------------------ */

/* Reads the next frame tag. Returns 1 when a frame tag was found, 0 at clean
 * end of stream, -1 on malformed input. The stream is left positioned at the
 * first payload byte. */
static int app_read_frame_tag(FILE* stream) {
    char   tag[8];
    size_t length = 0;

    while (length < sizeof(tag) - 1) {
        const int c = fgetc(stream);
        if (c == EOF) {
            return (length == 0) ? 0 : -1;
        }
        if (c == '\n') {
            break;
        }
        tag[length++] = (char)c;
    }
    tag[length] = '\0';

    if (length < 5 || memcmp(tag, "FRAME", 5) != 0) {
        return -1;
    }
    return 1;
}

/* Reads exactly `size` bytes of frame payload into buffer. Returns 0 on
 * success, -1 on truncated input. */
static int app_read_frame_payload(FILE* stream, uint8_t* buffer, size_t size) {
    if (size == 0) {
        return 0;
    }
    return (fread(buffer, 1, size, stream) == size) ? 0 : -1;
}

/* Compare source luma in display order before deciding whether to read ahead.
 * The grid and 8-bit scale match the encoder's anchor activity probe. */
static unsigned app_luma8(const uint8_t* frame, size_t position, unsigned depth) {
    if (depth == 8u) return frame[position];
    return (frame[2u*position] | ((unsigned)frame[2u*position+1u]<<8u)) >> (depth-8u);
}

/* Estimate what fraction of the current drawing each parent can predict.
 * Coarse motion compensation within each 64x64 region distinguishes moving
 * texture from genuinely new detail. All costs use original luma samples. */
static void app_parent_fractions(const uint8_t* source, const uint8_t* parents[2],
    unsigned count, int w, int h, unsigned depth, double fractions[2]) {
    unsigned regions = 0;
    for (int by=0; by<h; by+=64) for (int bx=0; bx<w; bx+=64) {
        unsigned values[16], sum=0, n=0, intra=0, sad[2]={0,0};
        for (int y=by; y<h && y<by+64; y+=16) for (int x=bx; x<w && x<bx+64; x+=16) {
            values[n] = app_luma8(source,(size_t)y*w+x,depth);
            sum += values[n++];
        }
        const int mean = (int)((sum+n/2u)/n);
        for (unsigned i=0;i<n;++i) intra += (unsigned)abs((int)values[i]-mean);
        for (unsigned r=0;r<count;++r) {
            sad[r] = UINT32_MAX;
            for (int dy=-4;dy<=4;dy+=4) for (int dx=-4;dx<=4;dx+=4) {
                unsigned err=0,i=0;
                for (int y=by;y<h && y<by+64;y+=16) for (int x=bx;x<w && x<bx+64;x+=16) {
                    const int xx=x+dx<0?0:x+dx>=w?w-1:x+dx;
                    const int yy=y+dy<0?0:y+dy>=h?h-1:y+dy;
                    err += (unsigned)abs((int)values[i++]-(int)app_luma8(parents[r],(size_t)yy*w+xx,depth));
                }
                if (err < sad[r]) sad[r] = err;
            }
        }
        const unsigned better = count == 2u && sad[1] < sad[0];
        const unsigned baseline = intra+n;
        if (sad[better] < baseline) {
            const double gain = 1.0 - (double)sad[better]/baseline;
            if (count == 2u && sad[0] == sad[1]) { fractions[0] += 0.5*gain; fractions[1] += 0.5*gain; }
            else fractions[better] += gain;
        }
        ++regions;
    }
    fractions[0] /= regions;
    fractions[1] /= regions;
}

static void app_dependency_qp(const uint8_t* group, const uint8_t* previous, size_t bytes,
    unsigned count, const unsigned order[32], int w, int h, unsigned depth, int qp, int out[32]) {
    double weight[33];
    unsigned left[33], right[33];
    for (unsigned i=0; i<=count; ++i) weight[i]=1.0;
    for (unsigned start=0;start<count;) {
        const unsigned span = order[start]+1u-start;
        for (unsigned k=0;k<span;++k) {
            const SvtAv2PyramidNode node = svt_av2_pyramid_node(span,k,0);
            const unsigned pos = start+node.position;
            left[pos] = start+node.left;
            right[pos] = k ? start+node.right : left[pos];
        }
        start += span;
    }
    for (unsigned k=count; k--;) {
        const unsigned pos=order[k]+1u, a=left[pos], b=right[pos];
        const uint8_t* parents[2] = {a ? group+(a-1u)*bytes : previous, b ? group+(b-1u)*bytes : previous};
        double fraction[2]={0,0};
        app_parent_fractions(group+(pos-1u)*bytes,parents,1u+(a!=b),w,h,depth,fraction);
        weight[a] += weight[pos]*fraction[0];
        weight[b] += weight[pos]*fraction[1];
        const int q = qp+14-(int)lround(16.0*log2(weight[pos]));
        out[k] = q<1 ? 1 : q>255 ? 255 : q;
    }
}

static unsigned app_drawing_sad(const uint8_t* a, const uint8_t* b,
        unsigned width, unsigned height, unsigned depth) {
    unsigned sad = 0;
    for (unsigned y=0; y<18u; ++y) for (unsigned x=0; x<32u; ++x) {
        const size_t pos = (size_t)((2u*y+1u)*height/36u)*width + (2u*x+1u)*width/64u;
        sad += (unsigned)abs((int)app_luma8(a,pos,depth)-(int)app_luma8(b,pos,depth));
    }
    return sad;
}

/* A small new stroke can be invisible in a whole-frame average.
 * Screen 8x8 cells on all planes before treating the drawing as a hold. */
static int app_same_local_drawing(const uint8_t* a, const uint8_t* b,
        size_t bytes, unsigned width, unsigned height, unsigned depth) {
    const unsigned step = depth==8u ? 1u : 2u;
    const unsigned shift = bytes == (size_t)width*height*step*3u ? 0u : 1u;
    for (unsigned p=0; p<3; ++p) {
        const unsigned w = p ? width>>shift : width, h = p ? height>>shift : height;
        for (unsigned by=0; by<h; by+=8u) for (unsigned bx=0; bx<w; bx+=8u) {
            unsigned sad=0, n=0;
            for (unsigned y=by; y<h && y<by+8u; y+=2u) for (unsigned x=bx; x<w && x<bx+8u; x+=2u) {
                const size_t pos=(size_t)y*w+x;
                sad+=(unsigned)abs((int)app_luma8(a,pos,depth)-(int)app_luma8(b,pos,depth));
                ++n;
            }
            if (sad>4u*n) return 0;
        }
        a+=(size_t)w*h*step;
        b+=(size_t)w*h*step;
    }
    return 1;
}

/* Average only consecutive exposures of one drawing. Reject local outliers
 * even in an otherwise still frame, to keep changing line art and subtitles. */
static int app_filter_drawing(const uint8_t* frames, unsigned count, unsigned pos,
        size_t bytes, unsigned width, unsigned height, unsigned depth, uint8_t* dst) {
    const uint8_t* center = frames + pos*bytes;
    unsigned first = pos, end = pos+1u;
    while (first && pos-first<8u && app_drawing_sad(center,frames+(first-1u)*bytes,width,height,depth)<=576u) --first;
    while (end<count && end-pos<=8u && app_drawing_sad(center,frames+end*bytes,width,height,depth)<=576u) ++end;
    if (end-first<3u) return 0;
    const size_t samples = bytes/(depth==8u ? 1u : 2u);
    const int limit = 4 << (depth-8u);
    for (size_t i=0; i<samples; ++i) {
        const int value = depth==8u ? center[i] : ((const uint16_t*)center)[i];
        int sum = 0, weight = 0;
        for (unsigned j=first; j<end; ++j) {
            const uint8_t* frame = frames + j*bytes;
            const int other = depth==8u ? frame[i] : ((const uint16_t*)frame)[i];
            const int w = limit-abs(other-value);
            if (w>0) { sum+=w*other; weight+=w; }
        }
        const int result = (sum+weight/2)/weight; /* center always votes */
        if (depth==8u) dst[i]=(uint8_t)result;
        else ((uint16_t*)dst)[i]=(uint16_t)result;
    }
    return 1;
}

/* Count the effective number of displays that can reuse the root drawing.
 * A one-level mean luma change (on the 8-bit scale) halves its direct credit.
 * Changed regions still contribute through the existing motion-based TPL. */
static double app_direct_displays(const uint8_t* frames, size_t bytes, unsigned center,
        unsigned first, unsigned end, unsigned width, unsigned height, unsigned depth) {
    const uint8_t* reference = frames + center*bytes;
    double displays = 1.0;
    for (unsigned j = first; j < end; ++j) {
        if (j == center) continue;
        const uint8_t* frame = frames + j*bytes;
        const double difference = app_drawing_sad(frame,reference,width,height,depth) / 576.0;
        displays += 1.0 / (1.0 + difference*difference);
    }
    return displays;
}

/* Propagate future reconstruction value through each 32x32 region of the
 * actual coding tree. A moving region credits its matched parent region. */
static int app_tpl_group(const uint8_t* frames, const uint8_t* previous, size_t bytes,
        unsigned count, const unsigned order[32], int key, int w, int h, unsigned depth,
        int qp, int out[32], uint16_t* weights, int root_policy, int reuse_qp) {
    const unsigned cols = ((unsigned)w+31u)/32u, rows = ((unsigned)h+31u)/32u;
    const size_t cells = (size_t)cols*rows;
    float* credit = calloc(cells*(count+1u),sizeof(*credit));
    const unsigned sb_cols=(cols+1u)/2u;
    const size_t superblocks=(size_t)sb_cols*((rows+1u)/2u);
    double* local=weights ? calloc(2u*superblocks,sizeof(*local)) : NULL;
    if (!credit || (weights && !local)) { free(local); free(credit); return -1; }
    unsigned left[32]={0}, right[32]={0};
    for (unsigned start=(unsigned)key; start<count;) {
        const unsigned span = order[start]+1u-start;
        for (unsigned k=0;k<span;++k) {
            const SvtAv2PyramidNode node = svt_av2_pyramid_node(span,k,0);
            left[start+k] = start+node.left;
            right[start+k] = k ? start+node.right : start+node.left;
        }
        start += span;
    }
    for (unsigned k=count;k--;) {
        const unsigned pos=order[k]+1u, parent[2]={left[k],right[k]};
        const uint8_t* src=frames+(pos-1u)*bytes;
        double incoming=0, own=0;
        if (local) memset(local,0,2u*superblocks*sizeof(*local));
        for (unsigned cy=0;cy<rows;++cy) for (unsigned cx=0;cx<cols;++cx) {
            const int bx=(int)cx*32, by=(int)cy*32;
            const int bw=w-bx<32?w-bx:32, bh=h-by<32?h-by:32;
            int values[16], n=0, sum=0, intra=0;
            for (int y=0;y<bh;y+=8) for (int x=0;x<bw;x+=8) {
                values[n]=(int)app_luma8(src,(size_t)(by+y)*w+bx+x,depth);
                sum+=values[n++];
            }
            for (int i=0;i<n;++i) intra+=abs(values[i]-(sum+n/2)/n);
            const size_t cell=(size_t)cy*cols+cx;
            const double base=n+intra;
            const double value=base+credit[pos*cells+cell];
            own+=base;
            incoming+=value;
            if (local) {
                const size_t sb=(size_t)(cy/2u)*sb_cols+cx/2u;
                local[2u*sb]+=base;
                local[2u*sb+1u]+=value;
            }
            if (!k && key) continue;
            unsigned best[2]={UINT32_MAX,UINT32_MAX};
            size_t target[2]={cell,cell};
            const unsigned refs=1u+(parent[0]!=parent[1]);
            for (unsigned r=0;r<refs;++r) {
                const uint8_t* ref=parent[r] ? frames+(parent[r]-1u)*bytes : previous;
                int dx=0,dy=0;
                for (int step=16;step;step/=2) {
                    const int ox=dx,oy=dy;
                    for (int j=-1;j<=1;++j) for (int i=-1;i<=1;++i) {
                        const int tx=ox+i*step, ty=oy+j*step;
                        if (bx+tx<0 || by+ty<0 || bx+tx+bw>w || by+ty+bh>h) continue;
                        unsigned sad=0,t=0;
                        for (int y=0;y<bh;y+=8) for (int x=0;x<bw;x+=8)
                            sad+=(unsigned)abs(values[t++]-(int)app_luma8(ref,(size_t)(by+ty+y)*w+bx+tx+x,depth));
                        if (sad<best[r] || (sad==best[r] && abs(tx)+abs(ty)<abs(dx)+abs(dy))) {
                            best[r]=sad; dx=tx; dy=ty;
                        }
                    }
                    if (!best[r]) break;
                }
                target[r]=(size_t)((by+bh/2+dy)/32)*cols+(bx+bw/2+dx)/32;
            }
            const unsigned r=refs==2u && best[1]<best[0];
            if (best[r]<base) {
                const float gain=(float)(value*(1.0-best[r]/base));
                if (refs==2u && best[0]==best[1]) {
                    credit[parent[0]*cells+target[0]]+=0.5f*gain;
                    credit[parent[1]*cells+target[1]]+=0.5f*gain;
                } else credit[parent[r]*cells+target[r]]+=gain;
            }
        }
        /* The AV2 quantizer step doubles every 24 indices. Scale it by
         * the inverse square root of the estimated reconstruction value. */
        const int is_key = key && !k;
        const int root = root_policy && (is_key || (parent[0]==parent[1] && pos-parent[0]>=4u));
        double importance = incoming/own;
        if (root) {
            const double direct = app_direct_displays(frames,bytes,pos-1u,is_key ? 0u : parent[0],
                is_key ? count : pos,(unsigned)w,(unsigned)h,depth);
            if (direct < importance) {
                const double propagation = sqrt(importance/direct);
                /* Direct holds need little residual. A moving group can
                 * instead protect detail reused through motion compensation. */
                importance = root_policy == 1 ? importance/propagation : importance*propagation;
            }
        }
        int q=qp+10-(int)lround(12.0*log2(importance));
        if (root && q>qp) q=qp;
        else if (!root && is_key && q>qp-18) q=qp-18;
        if (reuse_qp && !is_key) {
            /* An already coded parent carries this drawing. Keep the usual
             * quality for new drawings and let the reused one mostly skip. */
            const unsigned refs = 1u+(parent[0]!=parent[1]);
            int local_change = 0, matched = 0;
            for (unsigned r=0; r<refs; ++r) {
                const uint8_t* reference = parent[r] ? frames+(parent[r]-1u)*bytes : previous;
                if (app_drawing_sad(src,reference,(unsigned)w,(unsigned)h,depth)>576u) continue;
                if (reuse_qp == 1 || app_same_local_drawing(src,reference,bytes,(unsigned)w,(unsigned)h,depth)) {
                    matched = 1;
                    break;
                }
                local_change = 1;
            }
            if (matched) q=qp+36;
            else if (local_change && importance>1.5 && q>qp-12) q=qp-12;
        }
        out[k]=q<1?1:q>255?255:q;
        if (local) for (size_t sb=0;sb<superblocks;++sb) {
            const double ratio=(incoming/own)*local[2u*sb]/local[2u*sb+1u];
            const long scale=lround(256.0*sqrt(ratio));
            weights[k*superblocks+sb]=(uint16_t)(scale<128?128:scale>512?512:scale);
        }
    }
    free(local);
    free(credit);
    return 0;
}

/* Compare legal mini-GOP spans on source luma in display order. Cache the
 * motion-compensated prediction error per region, then price every tree in
 * coding order. Only the chosen spans reach the encoder. */
static int app_choose_spans(const uint8_t* group, const uint8_t* previous, size_t bytes,
        unsigned count, unsigned w, unsigned h, unsigned depth, unsigned span_at[32]) {
    enum { W=64,H=36,COLS=8,ROWS=6,REGIONS=COLS*ROWS };
    const unsigned even=count&~1u, frames=even+1u;
    if (!even) return 0;
    uint8_t* grid=malloc((size_t)frames*W*H);
    uint16_t* errors=malloc((size_t)frames*frames*REGIONS*sizeof(*errors));
    if (!grid || !errors) { free(grid); free(errors); return -1; }
    for (unsigned t=0;t<frames;++t) {
        const uint8_t* src=t ? group+(t-1u)*bytes : previous;
        for (unsigned y=0;y<H;++y) for (unsigned x=0;x<W;++x)
            grid[(t*H+y)*W+x]=(uint8_t)app_luma8(src,(size_t)((2u*y+1u)*h/(2u*H))*w+(2u*x+1u)*w/(2u*W),depth);
    }
    for (unsigned t=0;t<frames;++t) for (unsigned r=0;r<frames;++r) {
        uint16_t* cost=errors+(t*frames+r)*REGIONS;
        for (int by=0;by<ROWS;++by) for (int bx=0;bx<COLS;++bx) {
            unsigned best=UINT16_MAX;
            for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx) {
                unsigned sad=0;
                for (int y=by*6;y<by*6+6 && sad<best;++y) for (int x=bx*8;x<bx*8+8;++x) {
                    const int yy=y+dy<0?0:y+dy>=H?H-1:y+dy;
                    const int xx=x+dx<0?0:x+dx>=W?W-1:x+dx;
                    sad+=(unsigned)abs(grid[(t*H+y)*W+x]-grid[(r*H+yy)*W+xx]);
                }
                if (sad<best) best=sad;
            }
            cost[by*COLS+bx]=(uint16_t)best;
        }
    }
    double suffix[33]={0};
    for (int start=(int)even-2;start>=0;start-=2) {
        double best=1e100;
        for (unsigned span=2;span<=even-(unsigned)start;span+=2) {
            double score=suffix[start+span]+REGIONS;
            for (unsigned k=0;k<span;++k) {
                const SvtAv2PyramidNode node=svt_av2_pyramid_node(span,k,0);
                const unsigned t=start+node.position;
                const uint16_t* l=errors+(t*frames+start+node.left)*REGIONS;
                const uint16_t* r=k ? errors+(t*frames+start+node.right)*REGIONS : l;
                const unsigned reach=k ? node.position-node.left : span;
                const double rate=reach>=8u ? 1.5 : reach>=4u ? 1.25 : reach>=2u ? 1.1 : 0.8;
                for (unsigned cell=0;cell<REGIONS;++cell)
                    score+=rate*log2(1.0+(l[cell]<r[cell]?l[cell]:r[cell])/48.0);
            }
            if (score<best) { best=score; span_at[start]=span; }
        }
        suffix[start]=best;
    }
    free(errors);
    free(grid);
    return 0;
}

static int app_drawing_changed(const uint8_t* frame, uint32_t w, uint32_t h, unsigned bd, uint16_t grid[576], unsigned threshold_q8) {
    uint32_t sad = 0u;
    for (unsigned y = 0; y < 18u; ++y) for (unsigned x = 0; x < 32u; ++x) {
        const size_t pos = (size_t)((2u*y+1u)*h/36u)*w + (2u*x+1u)*w/64u;
        const uint16_t value = bd > 8u ? (uint16_t)(frame[2u*pos] | (frame[2u*pos+1u] << 8u)) : frame[pos];
        const unsigned index = y*32u+x;
        sad += (uint32_t)abs((int)value - grid[index]);
        grid[index] = value;
    }
    return sad * 256u >= threshold_q8 * 576u * (1u << (bd-8u));
}

/* Match each original neighbour independently. A block mismatch or a large
 * pixel residual reduces its vote, especially across cuts and occlusions. */
static void app_temporal_luma(const uint8_t* center, const uint8_t* past, const uint8_t* future,
                              uint8_t* dst, int width, int height, unsigned depth) {
    const unsigned scale = 1u << (depth - 8u);
    const uint8_t* neighbours[2] = {past, future};
    for (int by = 0; by < height; by += 16) for (int bx = 0; bx < width; bx += 16) {
        const int h = height - by < 16 ? height - by : 16;
        const int w = width - bx < 16 ? width - bx : 16;
        const unsigned samples = (unsigned)(((h + 1) / 2) * ((w + 1) / 2));
        ptrdiff_t offsets[2] = {0, 0};
        unsigned weights[2] = {0, 0};
        for (unsigned n = 0; n < 2; ++n) {
            if (!neighbours[n]) continue;
            unsigned best = UINT32_MAX;
            int dx = 0, dy = 0;
            for (int step = 8; step && best; step /= 2) {
                const int ox = dx, oy = dy;
                static const int dirs[5][2] = {{0,0},{-1,0},{1,0},{0,-1},{0,1}};
                for (int k = step == 8 ? 0 : 1; k < 5; ++k) {
                    const int tx = ox + step * dirs[k][0], ty = oy + step * dirs[k][1];
                    if (bx+tx < 0 || by+ty < 0 || bx+tx+w > width || by+ty+h > height) continue;
                    unsigned sad = 0;
                    for (int y = 0; y < h && sad < best; y += 2) for (int x = 0; x < w; x += 2) {
                        const size_t i = (size_t)(by+y)*width + bx+x;
                        const size_t j = (size_t)(by+y+ty)*width + bx+x+tx;
                        const int a = depth == 8 ? center[i] : ((const uint16_t*)center)[i];
                        const int b = depth == 8 ? neighbours[n][j] : ((const uint16_t*)neighbours[n])[j];
                        sad += (unsigned)abs(a-b);
                    }
                    if (sad < best) { best = sad; dx = tx; dy = ty; }
                }
            }
            offsets[n] = (ptrdiff_t)dy * width + dx;
            const unsigned mean = best / (samples * scale);
            weights[n] = mean < 12u ? 12u - mean : 0u;
        }
        if (!weights[0] && !weights[1]) continue;
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const ptrdiff_t i = (ptrdiff_t)(by+y)*width + bx+x;
            const unsigned a = depth == 8 ? center[i] : ((const uint16_t*)center)[i];
            unsigned sum = 144u*a, count = 144u;
            for (unsigned n = 0; n < 2; ++n) if (weights[n]) {
                const ptrdiff_t j = i + offsets[n];
                const unsigned b = depth == 8 ? neighbours[n][j] : ((const uint16_t*)neighbours[n])[j];
                const unsigned delta = (unsigned)abs((int)a-(int)b) / scale;
                const unsigned weight = delta < 16u ? weights[n]*(16u-delta) : 0u;
                sum += weight*b;
                count += weight;
            }
            const unsigned value = (sum + count/2u) / count;
            if (depth == 8) dst[i] = (uint8_t)value;
            else ((uint16_t*)dst)[i] = (uint16_t)value;
        }
    }
}

/* Fills a SvtAv2Picture view over tightly packed supported YUV data held in a
 * single Y-then-U-then-V buffer (the Y4M layout). planes[] aliases into
 * `buffer`; strides are the plane widths in bytes. */
static void app_fill_picture(SvtAv2Picture* picture, uint8_t* buffer,
                                     uint32_t width, uint32_t height, SvtAv2BitDepth bit_depth,
                                     SvtAv2ChromaFormat chroma_format) {
    const uint32_t bytes = bit_depth == SVT_AV2_BIT_DEPTH_10 ? 2u : 1u;
    const uint32_t shift = chroma_format == SVT_AV2_CHROMA_444 ? 0u : 1u;
    const uint32_t chroma_width = width >> shift;
    const uint32_t chroma_height = height >> shift;
    uint8_t* const luma = buffer;
    uint8_t* const cb = luma + (size_t)width * height * bytes;
    uint8_t* const cr = cb + (size_t)chroma_width * chroma_height * bytes;

    memset(picture, 0, sizeof(*picture));
    picture->planes[0] = luma;
    picture->planes[1] = cb;
    picture->planes[2] = cr;
    picture->strides[0] = width * bytes;
    picture->strides[1] = chroma_width * bytes;
    picture->strides[2] = chroma_width * bytes;
    picture->width = width;
    picture->height = height;
    picture->chroma_format = chroma_format;
    picture->bit_depth = bit_depth;
    picture->pts = 0;
}

/* --- main ----------------------------------------------------------------- */

int main(int argc, char** argv) {
    AppOptions options;
    const int  parse_result = app_parse_args(argc, argv, &options);
    if (parse_result == 1) {
        return 0;
    }
    if (parse_result != 0) {
        print_usage(argv[0]);
        return 2;
    }

    SvtAv2Y4mInfo y4m;
    SvtAv2Error   err = svt_av2_y4m_read_header_from_path(options.input_path, &y4m);
    if (err != SVT_AV2_ERROR_NONE) {
        fprintf(stderr,
                "error: cannot read Y4M header from '%s' (%s)\n",
                options.input_path,
                svt_av2_error_to_string(err));
        return 3;
    }

    app_print_y4m_info(options.input_path, &y4m);

    if (options.output_path == NULL) {
        printf("mode        : metadata only (no --output given)\n");
        return 0;
    }

    /* The M1 encoder path exists for exactly one profile; unsupported input is
     * rejected here rather than producing a stream that could be mistaken for a
     * successful encode. */
    if (app_check_m1_input(&y4m) != 0) {
        return SVT_AV2_APP_EXIT_UNSUPPORTED_INPUT;
    }

    SvtAv2Handle             handle = NULL;
    const SvtAv2EncoderInfo* info = NULL;
    err = svt_av2_enc_init_handle(&handle, &info);
    if (err != SVT_AV2_ERROR_NONE) {
        fprintf(stderr, "error: svt_av2_enc_init_handle failed (%s)\n", svt_av2_error_to_string(err));
        return 4;
    }

    const SvtAv2ChromaFormat chroma = y4m.chroma_format;
    const SvtAv2BitDepth     depth = y4m.bit_depth;
    const SvtAv2Preset       preset = (SvtAv2Preset)options.preset;
    const uint32_t           threads = (uint32_t)options.threads;
    const int32_t            deterministic = options.deterministic;

    /* Parameters must be set before svt_av2_enc_init(). */
    err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_WIDTH, &y4m.width);
    if (err == SVT_AV2_ERROR_NONE) {
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_HEIGHT, &y4m.height);
    }
    if (err == SVT_AV2_ERROR_NONE) {
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_CHROMA_FORMAT, &chroma);
    }
    if (err == SVT_AV2_ERROR_NONE) {
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_BIT_DEPTH, &depth);
    }
    if (err == SVT_AV2_ERROR_NONE) {
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_FRAME_RATE_NUM, &y4m.frame_rate_num);
    }
    if (err == SVT_AV2_ERROR_NONE) {
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_FRAME_RATE_DEN, &y4m.frame_rate_den);
    }
    if (err == SVT_AV2_ERROR_NONE) {
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_QP, &options.qp);
    }
    if (err == SVT_AV2_ERROR_NONE) {
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_PRESET, &preset);
    }
    if (err == SVT_AV2_ERROR_NONE) {
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_THREADS, &threads);
    }
    if (err == SVT_AV2_ERROR_NONE) {
        const uint32_t cols=(uint32_t)options.tile_columns;
        err=svt_av2_enc_set_parameter(handle,SVT_AV2_PARAM_TILE_COLUMNS,&cols);
    }
    if (err == SVT_AV2_ERROR_NONE) {
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_DETERMINISTIC, &deterministic);
    }
    if (err == SVT_AV2_ERROR_NONE) {
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_OUTPUT_PATH, &options.output_path);
    }
    if (err != SVT_AV2_ERROR_NONE) {
        fprintf(stderr, "error: setting encoder parameters failed (%s)\n", svt_av2_error_to_string(err));
        svt_av2_enc_deinit(&handle);
        return 5;
    }

    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_PREDICTION_STRUCTURE, &options.prediction_structure);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_INTRA_MODE_SELECTION, &options.intra_mode_selection);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_PARTITION_SELECTION, &options.partition_selection);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_TRANSFORM_SELECTION, &options.transform_selection);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_ANGLE_SEARCH, &options.angle_search);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_MRL_SEARCH, &options.mrl_search);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_TREE_REFINEMENT, &options.tree_refinement);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_RD_LAMBDA_MODEL, &options.rd_lambda_model);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_RD_LAMBDA_SCALE, &options.rd_lambda_scale);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_CFL_SEARCH, &options.cfl_search);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_MHCCP_SEARCH, &options.mhccp_search);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_INTRA_JOINT, &options.intra_joint);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_RD_MODE, &options.rd_mode);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_COEFF_OPTIMIZATION, &options.coeff_optimization);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_MOTION_OPTIMIZATION, &options.motion_optimization);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_ENTROPY_ADAPTATION, &options.entropy_adaptation);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_set_parameter(handle,
        SVT_AV2_PARAM_STILL_PICTURE, &options.still_picture);
    /* Omission preserves the historical setter sequence and NONE default. */
    if (err == SVT_AV2_ERROR_NONE && options.content_type!=SVT_AV2_CONTENT_NONE)
        err = svt_av2_enc_set_parameter(handle,SVT_AV2_PARAM_CONTENT_TYPE,&options.content_type);
    if (err == SVT_AV2_ERROR_NONE && options.inter_skip_search)
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_INTER_SKIP_SEARCH, &options.inter_skip_search);
    if (err == SVT_AV2_ERROR_NONE && options.inter_ref_period != 1u)
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_INTER_REF_PERIOD, &options.inter_ref_period);
    if (err == SVT_AV2_ERROR_NONE && options.video_key_qp)
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_VIDEO_KEY_QP, &options.video_key_qp);
    if (err == SVT_AV2_ERROR_NONE && options.inter_fine)
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_INTER_FINE, &options.inter_fine);
    if (err == SVT_AV2_ERROR_NONE && options.video_anchor_qp)
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_VIDEO_ANCHOR_QP, &options.video_anchor_qp);
    if (err == SVT_AV2_ERROR_NONE && options.dual_ref_period)
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_DUAL_REF_PERIOD, &options.dual_ref_period);
    if (err == SVT_AV2_ERROR_NONE && options.dual_ref_policy)
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_DUAL_REF_POLICY, &options.dual_ref_policy);
    if (err == SVT_AV2_ERROR_NONE && options.dual_ref_cadence)
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_DUAL_REF_CADENCE, &options.dual_ref_cadence);
    if (err == SVT_AV2_ERROR_NONE && options.inter_near)
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_INTER_NEAR, &options.inter_near);
    if (err == SVT_AV2_ERROR_NONE && options.inter_tail)
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_INTER_TAIL, &options.inter_tail);
    if (err == SVT_AV2_ERROR_NONE && options.cdf_inherit)
        err = svt_av2_enc_set_parameter(handle, SVT_AV2_PARAM_CDF_INHERIT, &options.cdf_inherit);
    if (err == SVT_AV2_ERROR_NONE) err = svt_av2_enc_init(handle);
    if (err != SVT_AV2_ERROR_NONE) {
        fprintf(stderr, "error: svt_av2_enc_init failed (%s)\n", svt_av2_error_to_string(err));
        svt_av2_enc_deinit(&handle);
        return 6;
    }

    printf("encoder     : %s (%s, %s)\n", info->version_string, info->milestone, info->build_type);
    printf("config      : qp=%d preset=%d threads=%u deterministic=%d\n",
           options.qp,
           options.preset,
           threads,
           deterministic);
    printf("output      : %s\n", options.output_path);

    FILE* input = fopen(options.input_path, "rb");
    if (input == NULL) {
        fprintf(stderr, "error: cannot open '%s'\n", options.input_path);
        svt_av2_enc_deinit(&handle);
        return 3;
    }

    /* Re-read the header so the stream is positioned on the first frame tag. */
    if (svt_av2_y4m_read_header(input, &y4m) != SVT_AV2_ERROR_NONE) {
        fprintf(stderr, "error: re-reading Y4M header failed\n");
        fclose(input);
        svt_av2_enc_deinit(&handle);
        return 3;
    }

    size_t frame_bytes = 0;
    if (svt_av2_compute_frame_sample_count(y4m.width, y4m.height, y4m.chroma_format, &frame_bytes) !=
        SVT_AV2_ERROR_NONE) {
        fprintf(stderr, "error: cannot compute frame size\n");
        fclose(input);
        svt_av2_enc_deinit(&handle);
        return 7;
    }
    frame_bytes *= svt_av2_bytes_per_sample(y4m.bit_depth);

    /* One input frame buffer, reused for every frame: the API copies the pixels
     * during send_picture(), so the buffer is free to be overwritten immediately
     * (contract section 1.2). */
    uint8_t* frame_buffer = (uint8_t*)malloc(frame_bytes);
    if (frame_buffer == NULL) {
        fprintf(stderr, "error: cannot allocate %zu byte frame buffer\n", frame_bytes);
        fclose(input);
        svt_av2_enc_deinit(&handle);
        return 7;
    }

    FILE* output = fopen(options.output_path, "wb");
    if (output == NULL) {
        fprintf(stderr, "error: cannot create '%s'\n", options.output_path);
        free(frame_buffer);
        fclose(input);
        svt_av2_enc_deinit(&handle);
        return 3;
    }

    /* --frames 0 emits nothing; positive values bound the loop; the default
     * drains the whole source. Each frame uses the public API and optional
     * reconstruction export before the next picture is submitted. */
    const int32_t frame_limit = options.frames; /* -1 = all */

    size_t   packet_capacity = SVT_AV2_APP_PACKET_INITIAL_BYTES;
    uint8_t* packet = (uint8_t*)malloc(packet_capacity);
    if (packet == NULL) {
        fprintf(stderr, "error: cannot allocate %zu byte packet buffer\n", packet_capacity);
        fclose(output);
        free(frame_buffer);
        fclose(input);
        svt_av2_enc_deinit(&handle);
        return 7;
    }

    uint32_t frames_written = 0;
    int      encode_failed = 0;
    int      exit_code = 0;
    const int32_t effective_limit = options.still_picture ? 1 : frame_limit;

    /* --- optional reconstruction export -------------------------------------
     *
     * The reconstruction is the encoder's own picture, not a decode of the
     * stream. It is obtained ONLY through the public accessor
     * svt_av2_enc_get_recon(), which in turn calls the project's single packing
     * primitive; the app never packs a frame itself. The FILE* below is the one
     * and only I/O point for this feature (contract section 7.2). */
    FILE*    recon_output = NULL;
    uint8_t* recon_buffer = NULL;
    size_t   recon_capacity = 0;
    if (options.recon_path != NULL) {
        /* `data` is a required, caller-owned buffer (svt_av2.h:199): the
         * accessor reports "too small" only once it has a buffer to measure
         * against. Start with the same generous capacity the packet path uses
         * and let the grow-and-retry branch below take over on OUT_OF_MEMORY. */
        recon_capacity = SVT_AV2_APP_PACKET_INITIAL_BYTES;
        recon_buffer = (uint8_t*)malloc(recon_capacity);
        if (recon_buffer == NULL) {
            fprintf(stderr, "error: cannot allocate %zu byte recon buffer\n", recon_capacity);
            free(packet);
            free(frame_buffer);
            fclose(input);
            fclose(output);
            svt_av2_enc_deinit(&handle);
            return 7;
        }

        recon_output = fopen(options.recon_path, "wb");
        if (recon_output == NULL) {
            fprintf(stderr, "error: cannot create '%s'\n", options.recon_path);
            free(recon_buffer);
            free(packet);
            free(frame_buffer);
            fclose(input);
            fclose(output);
            svt_av2_enc_deinit(&handle);
            return 3;
        }
    }

    const char* motion_profile = getenv("SVT_AV2_MOTION_Q");
    unsigned lookahead = 0;
    const int adaptive_group = motion_profile && !strcmp(motion_profile, "adaptive-gop");
    const int tree_group = motion_profile && (!strcmp(motion_profile, "tree-rd") || !strcmp(motion_profile, "tree-overlay"));
    const int overlay_group = motion_profile && !strcmp(motion_profile, "overlay-rd");
    uint16_t previous_grid[576] = {0};
    if (tree_group || overlay_group || adaptive_group || (motion_profile && !strcmp(motion_profile, "quartet-refs"))) lookahead = 3u;
    else if (motion_profile && (!strcmp(motion_profile, "paired-anchor") || !strcmp(motion_profile, "paired-refs"))) lookahead = 1u;
    const char* dependency_mode = getenv("SVT_AV2_TEMPORAL_ALLOC");
    const int dependencies = dependency_mode && !strcmp(dependency_mode,"propagate");
    const int full_window = motion_profile && !strcmp(motion_profile,"pyramid-window");
    const unsigned max_group = full_window ? 32u : motion_profile && !strcmp(motion_profile, "pyramid16") ? 16u : 8u;
    int root_policy = 0;
    if (max_group == 16u && dependency_mode) {
        if (!strcmp(dependency_mode,"direct-roots")) root_policy = 1;
        if (!strcmp(dependency_mode,"protected-roots")) root_policy = 2;
    }
    const int reuse_qp = max_group!=16u || !dependency_mode ? 0 :
        !strcmp(dependency_mode,"local-reuse") ? 2 : !strcmp(dependency_mode,"reuse-drawing");
    const int spatial_tpl = reuse_qp || root_policy || (max_group == 16u && dependency_mode && !strcmp(dependency_mode,"spatial"));
    const int tpl = spatial_tpl || (max_group == 16u && dependency_mode && !strcmp(dependency_mode,"regions"));
    const unsigned capacity = max_group+(unsigned)tpl;
    const char* gop = getenv("SVT_AV2_GOP_SEARCH");
    const int search_spans = full_window && gop && !strcmp(gop,"regions");
    const char* runs = getenv("SVT_AV2_DRAWING_RUNS");
    const int drawing_runs = max_group == 16u && runs && !strcmp(runs,"cadence");
    const int motion_layers = motion_profile && !strcmp(motion_profile, "pyramid-motion");
    const int pyramid_tail = motion_layers || max_group >= 16u || (motion_profile && (!strcmp(motion_profile, "pyramid-tail") || !strcmp(motion_profile, "pyramid-compound")));
    const int adaptive_pyramid = pyramid_tail || (motion_profile && !strcmp(motion_profile, "pyramid8-rd"));
    const int pyramid = adaptive_pyramid || (motion_profile && (!strcmp(motion_profile, "pyramid8") || !strcmp(motion_profile, "pyramid8-blend")));
    const int paired = lookahead != 0u;
    uint8_t* pending_input = paired && frame_bytes <= SIZE_MAX / lookahead ? (uint8_t*)malloc(frame_bytes * lookahead) : NULL;
    uint8_t* pending_recon = paired && recon_output && frame_bytes <= SIZE_MAX / (tree_group ? 2u : 1u) ?
        (uint8_t*)malloc(frame_bytes * (tree_group ? 2u : 1u)) : NULL;
    const char* filter = getenv("SVT_AV2_TEMPORAL_FILTER");
    const int filter_anchors = pyramid && filter && !strcmp(filter,"hidden");
    const int filter_drawings = tpl && filter && !strcmp(filter,"held-drawings");
    const int temporal_filter = motion_profile && !strcmp(motion_profile, "filtered-drawing");
    uint8_t* originals = temporal_filter && frame_bytes <= SIZE_MAX / 3u ? (uint8_t*)malloc(3u * frame_bytes) : NULL;
    unsigned group_count = 0u, group_index = 0u;
    unsigned group_order[32] = {0};
    uint8_t group_hidden[32] = {0};
    int group_qp[32] = {0};
    uint8_t* group_input = pyramid && frame_bytes <= SIZE_MAX/capacity ? (uint8_t*)malloc(capacity*frame_bytes) : NULL;
    uint8_t* group_recon = pyramid && recon_output && group_input ? (uint8_t*)malloc(capacity*frame_bytes) : NULL;
    const size_t superblocks=((size_t)y4m.width+63u)/64u*((y4m.height+63u)/64u);
    uint16_t* group_weights=spatial_tpl ? malloc(capacity*superblocks*sizeof(*group_weights)) : NULL;
    if (spatial_tpl && !group_weights) exit_code=7;
    if (pyramid && (!group_input || (recon_output && !group_recon))) exit_code = 7;
    uint32_t originals_read = 0;
    int input_ended = 0;
    if (temporal_filter && !originals) exit_code = 7;
    unsigned next_midpoint = 0u, midpoints = 0u;
    if (paired && (!pending_input || (recon_output && !pending_recon))) exit_code = 7;
    while (!exit_code) {
        if (effective_limit >= 0 && (int32_t)frames_written >= effective_limit) {
            break;
        }

        uint32_t display = frames_written;
        uint8_t* current = frame_buffer;
        if (pyramid) {
            if (group_index == group_count) {
                if ((tpl || dependencies || search_spans) && group_count)
                    memcpy(frame_buffer,group_input+(group_count-1u)*frame_bytes,frame_bytes);
                group_index = group_count = 0;
                const unsigned i = frames_written % SVT_AV2_LOOKAHEAD_KEY_INTERVAL;
                const unsigned limit = !i ? (tpl ? capacity : 1u) : max_group <= SVT_AV2_LOOKAHEAD_KEY_INTERVAL-i ? max_group :
                    pyramid_tail ? SVT_AV2_LOOKAHEAD_KEY_INTERVAL-i : 1u;
                while (group_count < limit &&
                       (effective_limit < 0 || frames_written+group_count < (unsigned)effective_limit)) {
                    const int tag = app_read_frame_tag(input);
                    if (!tag) break;
                    if (tag < 0 || app_read_frame_payload(input, group_input + group_count*frame_bytes, frame_bytes)) {
                        fprintf(stderr, "error: malformed or truncated input frame %u\n", frames_written+group_count);
                        exit_code = 3;
                        break;
                    }
                    ++group_count;
                }
                if (exit_code || !group_count) break;
                unsigned changed_at[32] = {0};
                int sparse = 0;
                int reordered = group_count == max_group || (pyramid_tail && group_count > 1u);
                if (adaptive_pyramid && reordered) {
                    unsigned changes = 0;
                    for (unsigned j = 0; j < group_count; ++j) {
                        const int changed = app_drawing_changed(group_input + j*frame_bytes, y4m.width, y4m.height,
                            y4m.bit_depth, previous_grid, 128u);
                        changed_at[j] = changed != 0;
                        if (j) changes += changed != 0;
                    }
                    /* Repeated drawings at 8/12 fps benefit from keeping the last displayed
                     * drawing as the predictor. Dense motion can use both endpoints. */
                    sparse = drawing_runs && 2u*changes < group_count;
                    reordered = sparse ? 0 : motion_layers ? 2u*changes >= group_count :
                        pyramid_tail ? changes != 0u : changes >= 4u;
                }
                unsigned span_at[32]={0};
                if (search_spans && reordered && i && app_choose_spans(group_input,frame_buffer,frame_bytes,
                        group_count,y4m.width,y4m.height,y4m.bit_depth,span_at)) { exit_code=7; break; }
                const int buffered_key = tpl && !i;
                group_order[0]=0;
                group_hidden[0]=0;
                for (unsigned j = (unsigned)buffered_key; j < group_count;) {
                    unsigned span = 1;
                    if (span_at[j]) span=span_at[j];
                    else if (full_window && reordered && group_count-j >= 2u) span = (group_count-j) & ~1u;
                    else if (reordered) while (span < max_group && span*2u <= group_count-j) span *= 2u;
                    for (unsigned k=0; k<span; ++k) {
                        const unsigned pos = svt_av2_pyramid_offset(span,k);
                        group_order[j+k] = j+pos-1u;
                        group_hidden[j+k] = span>1u && !(pos & 1u);
                    }
                    j += span;
                }
                if (tpl) {
                    if (app_tpl_group(group_input,frame_buffer,frame_bytes,group_count,group_order,
                            buffered_key,(int)y4m.width,(int)y4m.height,y4m.bit_depth,options.qp,group_qp,group_weights,root_policy,reuse_qp)) {
                        exit_code=7;
                        break;
                    }
                } else if (dependencies && i) app_dependency_qp(group_input,frame_buffer,frame_bytes,
                    group_count,group_order,(int)y4m.width,(int)y4m.height,y4m.bit_depth,options.qp,group_qp);
                else if (sparse) {
                    for (unsigned j=0; j<group_count; ++j) {
                        unsigned duration = 1;
                        while (j+duration<group_count && !changed_at[j+duration]) ++duration;
                        int qp = changed_at[j] ? options.qp - (int)lround(9.0*log2(duration)) : options.qp+18;
                        group_qp[j] = qp<1 ? 1 : qp>255 ? 255 : qp;
                    }
                } else memset(group_qp,0,sizeof(group_qp));
            }
            const unsigned offset = group_order[group_index];
            display = frames_written - group_index + offset;
            current = group_input + frame_bytes*offset;
            if (filter_drawings && app_filter_drawing(group_input,group_count,offset,frame_bytes,
                    y4m.width,y4m.height,y4m.bit_depth,frame_buffer)) {
                current = frame_buffer;
            } else if (filter_anchors && group_hidden[group_index]) {
                memcpy(frame_buffer, current, frame_bytes);
                app_temporal_luma(current, offset ? current-frame_bytes : NULL,
                    offset+1u < group_count ? current+frame_bytes : NULL, frame_buffer,
                    (int)y4m.width, (int)y4m.height, y4m.bit_depth);
                current = frame_buffer;
            }
            ++group_index;
        } else if (temporal_filter) {
            while (!input_ended && originals_read - frames_written < 2u &&
                   (effective_limit < 0 || originals_read < (uint32_t)effective_limit)) {
                const int tag = app_read_frame_tag(input);
                if (!tag) { input_ended = 1; break; }
                uint8_t* next = originals + (originals_read % 3u) * frame_bytes;
                if (tag < 0 || app_read_frame_payload(input, next, frame_bytes)) {
                    fprintf(stderr, "error: malformed or truncated input frame %u\n", originals_read);
                    exit_code = 3;
                    break;
                }
                ++originals_read;
            }
            if (exit_code || originals_read == frames_written) break;
            const uint8_t* center = originals + (frames_written % 3u) * frame_bytes;
            memcpy(frame_buffer, center, frame_bytes);
            if (frames_written)
                app_temporal_luma(center, originals + ((frames_written-1u) % 3u) * frame_bytes,
                    originals_read - frames_written > 1u ? originals + ((frames_written+1u) % 3u) * frame_bytes : NULL,
                    frame_buffer, (int)y4m.width, (int)y4m.height, y4m.bit_depth);
        } else if (next_midpoint < midpoints) {
            unsigned slot = next_midpoint++;
            if (tree_group && midpoints == 3u && slot < 2u) slot = 1u - slot;
            current = pending_input + frame_bytes * slot;
            display = frames_written - next_midpoint + slot;
        } else {
            const int tag = app_read_frame_tag(input);
            if (!tag) break;
            if (tag < 0) {
                fprintf(stderr, "error: malformed frame tag in '%s'\n", options.input_path);
                exit_code = 3;
                break;
            }
            if (app_read_frame_payload(input, frame_buffer, frame_bytes) != 0) {
                fprintf(stderr, "error: truncated frame %u in '%s'\n", frames_written, options.input_path);
                exit_code = 3;
                break;
            }
            next_midpoint = midpoints = 0u;
            const uint32_t gop_index = frames_written % SVT_AV2_LOOKAHEAD_KEY_INTERVAL;
            const int changed = (!adaptive_group && !overlay_group) || app_drawing_changed(frame_buffer,
                y4m.width, y4m.height, y4m.bit_depth, previous_grid, overlay_group ? 64u : 512u);
            const unsigned limit = overlay_group && changed ? 1u : lookahead;
            while ((overlay_group || changed) && gop_index && midpoints < limit && gop_index + midpoints + 1u < SVT_AV2_LOOKAHEAD_KEY_INTERVAL &&
                   (effective_limit < 0 || frames_written + midpoints + 1u < (uint32_t)effective_limit)) {
                const int next = app_read_frame_tag(input);
                if (!next) break;
                memcpy(pending_input + frame_bytes * midpoints, frame_buffer, frame_bytes);
                if (next < 0 || app_read_frame_payload(input, frame_buffer, frame_bytes) != 0) {
                    fprintf(stderr, "error: malformed or truncated future anchor in '%s'\n", options.input_path);
                    exit_code = 3;
                    break;
                }
                ++midpoints;
                if (overlay_group && app_drawing_changed(frame_buffer, y4m.width, y4m.height,
                        y4m.bit_depth, previous_grid, 64u)) break;
            }
            if (exit_code) break;
            display += midpoints;
            if (adaptive_group && midpoints)
                app_drawing_changed(frame_buffer, y4m.width, y4m.height, y4m.bit_depth, previous_grid, 512u);
        }

        SvtAv2Picture picture;
        app_fill_picture(&picture, current, y4m.width, y4m.height, y4m.bit_depth, y4m.chroma_format);
        picture.pts = (int32_t)display;

        if (group_weights)
            err=svt_av2_enc_send_picture_weighted(handle,&picture,group_qp[group_index-1u],
                group_weights+(group_index-1u)*superblocks,superblocks);
        else err = pyramid && (dependencies || drawing_runs || tpl) && group_qp[group_index-1u] ?
            svt_av2_enc_send_picture_at_qp(handle,&picture,group_qp[group_index-1u]) :
            svt_av2_enc_send_picture(handle,&picture);
        if (err != SVT_AV2_ERROR_NONE) {
            fprintf(stderr,
                    "error: svt_av2_enc_send_picture failed for frame %u (%s)\n",
                    frames_written,
                    svt_av2_error_to_string(err));
            exit_code = 9;
            break;
        }

        SvtAv2Packet out;
        out.data = packet;
        out.capacity = packet_capacity;
        out.size = 0;
        out.pts = 0;
        err = svt_av2_enc_get_packet_ex(handle, &out);
        if (err == SVT_AV2_ERROR_OUT_OF_MEMORY) {
            /* The API reports the size it needs in `size`; grow and retry. This
             * is why the app never needs a hard-coded 4096-byte buffer. */
            const size_t needed = (out.size > 0) ? out.size : packet_capacity * 2u;
            uint8_t*     grown = (uint8_t*)realloc(packet, needed);
            if (grown == NULL) {
                fprintf(stderr, "error: cannot grow packet buffer to %zu bytes\n", needed);
                exit_code = 7;
                break;
            }
            packet = grown;
            packet_capacity = needed;

            out.data = packet;
            out.capacity = packet_capacity;
            out.size = 0;
            err = svt_av2_enc_get_packet_ex(handle, &out);
        }
        if (err != SVT_AV2_ERROR_NONE) {
            fprintf(stderr,
                    "error: svt_av2_enc_get_packet_ex failed for frame %u (%s)\n",
                    frames_written,
                    svt_av2_error_to_string(err));
            exit_code = 11;
            break;
        }
        if (out.size == 0) {
            fprintf(stderr, "error: encoder produced no packet for frame %u\n", frames_written);
            exit_code = 11;
            break;
        }

        if (fwrite(packet, 1, out.size, output) != out.size) {
            fprintf(stderr, "error: writing frame %u to '%s' failed\n", frames_written, options.output_path);
            exit_code = 12;
            break;
        }
        if (options.verbose) {
            printf("frame %u    : %zu bytes (pts %d)\n", frames_written, out.size, out.pts);
        }

        /* Additive output: the encoded stream above is unaffected by this
         * block. The recon is written only when the user asked for it, and a
         * failure here is a real failure (contract section 7: no fake
         * success). */
        if (recon_output != NULL) {
            SvtAv2ReconFrame recon;
            recon.data = recon_buffer;
            recon.capacity = recon_capacity;
            recon.byte_count = 0;
            recon.frame_index = 0;
            recon.visible_width = 0;
            recon.visible_height = 0;
            recon.chroma_width = 0;
            recon.chroma_height = 0;
            SvtAv2Error rerr = svt_av2_enc_get_recon(handle, &recon);
            if (rerr == SVT_AV2_ERROR_OUT_OF_MEMORY) {
                /* The accessor reports the required size in `byte_count`; grow
                 * and retry, exactly as the packet path does. */
                const size_t needed = (recon.byte_count > 0) ? recon.byte_count
                                                             : (recon_capacity > 0 ? recon_capacity * 2u : 1u);
                uint8_t* grown = (uint8_t*)realloc(recon_buffer, needed);
                if (grown == NULL) {
                    fprintf(stderr, "error: cannot grow recon buffer to %zu bytes\n", needed);
                    exit_code = 7;
                    break;
                }
                recon_buffer = grown;
                recon_capacity = needed;

                recon.data = recon_buffer;
                recon.capacity = recon_capacity;
                recon.byte_count = 0;
                rerr = svt_av2_enc_get_recon(handle, &recon);
            }
            if (rerr != SVT_AV2_ERROR_NONE) {
                fprintf(stderr,
                        "error: svt_av2_enc_get_recon failed for frame %u (%s)\n",
                        frames_written,
                        svt_av2_error_to_string(rerr));
                exit_code = 11;
                break;
            }
            if (paired && recon.byte_count != frame_bytes) { exit_code = 11; break; }
            if (pyramid) {
                if (recon.byte_count != frame_bytes) { exit_code = 11; break; }
                const unsigned offset = display - (frames_written+1u-group_index);
                memcpy(group_recon + frame_bytes*offset, recon_buffer, frame_bytes);
                if (group_index == group_count && fwrite(group_recon, frame_bytes, group_count, recon_output) != group_count) {
                    exit_code = 12;
                    break;
                }
            } else if (tree_group && midpoints == 3u && next_midpoint == 1u)
                memcpy(pending_recon + frame_bytes, recon_buffer, recon.byte_count);
            else if (display > frames_written) memcpy(pending_recon, recon_buffer, recon.byte_count);
            else if (!recon.byte_count || fwrite(recon_buffer, 1, recon.byte_count, recon_output) != recon.byte_count ||
                (tree_group && midpoints == 3u && next_midpoint == 2u &&
                 fwrite(pending_recon + frame_bytes, 1, recon.byte_count, recon_output) != recon.byte_count) ||
                (display < frames_written && next_midpoint == midpoints && fwrite(pending_recon, 1, recon.byte_count, recon_output) != recon.byte_count)) {
                fprintf(stderr,
                        "error: writing reconstruction for frame %u to '%s' failed\n",
                        frames_written,
                        options.recon_path);
                exit_code = 12;
                break;
            }
            if (options.verbose) {
                printf("recon %u    : %zu bytes (%ux%u, frame_index %u)\n",
                       frames_written,
                       recon.byte_count,
                       recon.visible_width,
                       recon.visible_height,
                       recon.frame_index);
            }
        }
        if (options.rd_mode) {
            SvtAv2RdStats rd;
            const SvtAv2Error rd_error = svt_av2_enc_get_rd_stats(handle, &rd);
            if (rd_error != SVT_AV2_ERROR_NONE) {
                fprintf(stderr, "error: reading RD statistics failed (%s)\n", svt_av2_error_to_string(rd_error));
                exit_code = 11;
                break;
            }
            printf("rd frame=%u mode=%u candidates=%" PRIu64 " forward=%" PRIu64
                   " replay=%" PRIu64 " D=%" PRIu64 " Rq12=%" PRIu64 " J=%" PRIu64
                   " lambdaq8=%" PRIu64 " full=%" PRIu64 " partial=%" PRIu64
                   " winner=%" PRIu64 "\n", frames_written, options.rd_mode,
                   rd.candidate_count, rd.forward_calls, rd.replay_checks, rd.distortion,
                   rd.rate_q12, rd.cost, rd.lambda_q8, rd.full_superblocks,
                   rd.partial_superblocks, rd.winner_hash);
            if (options.coeff_optimization)
                printf("coeff frame=%u trials=%" PRIu64 " accepts=%" PRIu64 "\n",
                    frames_written, rd.coeff_trials, rd.coeff_accepts);
            if (options.motion_optimization)
                printf("%s frame=%u trials=%" PRIu64 " accepts=%" PRIu64 "\n",
                    options.motion_optimization == 5u ? "quarter" : options.motion_optimization == 4u ? "sharp" :
                    options.motion_optimization == 3u ? "half" : "motion",
                    frames_written, rd.motion_trials, rd.motion_accepts);
            if (options.motion_optimization == 2u)
                printf("wide frame=%u trials=%" PRIu64 " accepts=%" PRIu64 "\n",
                    frames_written, rd.wide_trials, rd.wide_accepts);
            if (options.dual_ref_policy == 3u) {
                SvtAv2BlockRefStats refs;
                const SvtAv2Error ref_error = svt_av2_enc_get_block_ref_stats(handle, &refs);
                if (ref_error != SVT_AV2_ERROR_NONE) {
                    fprintf(stderr, "error: reading block reference statistics failed (%s)\n", svt_av2_error_to_string(ref_error));
                    exit_code = 11;
                    break;
                }
                printf("dual-ref frame=%u logical=%u d0=%u d1=%u refresh=%u rank0=%u rank1=%u odd_rank0=%u odd_rank1=%u\n",
                    frames_written, refs.logical_ref_count, refs.ref_distance[0], refs.ref_distance[1],
                    refs.refresh_mask, refs.rank_leaves[0], refs.rank_leaves[1], refs.odd_mv_leaves[0], refs.odd_mv_leaves[1]);
            }
            if (options.inter_skip_search) {
                uint64_t trials, accepts;
                const SvtAv2Error skip_error = svt_av2_enc_get_inter_skip_stats(handle, &trials, &accepts);
                if (skip_error != SVT_AV2_ERROR_NONE) {
                    fprintf(stderr, "error: reading inter skip statistics failed (%s)\n",
                        svt_av2_error_to_string(skip_error));
                    exit_code = 11;
                    break;
                }
                printf("inter-skip frame=%u trials=%" PRIu64 " accepts=%" PRIu64 "\n",
                    frames_written, trials, accepts);
            }
            if (options.inter_fine) {
                uint64_t trials, accepts;
                const SvtAv2Error fine_error = svt_av2_enc_get_inter_fine_stats(handle, &trials, &accepts);
                if (fine_error != SVT_AV2_ERROR_NONE) {
                    fprintf(stderr, "error: reading inter fine statistics failed (%s)\n",
                        svt_av2_error_to_string(fine_error));
                    exit_code = 11;
                    break;
                }
                printf("inter-fine frame=%u trials=%" PRIu64 " accepts=%" PRIu64 "\n",
                    frames_written, trials, accepts);
            }
        }
        frames_written++;
    }

    /* Signal end of stream through the public API so the flush path is
     * exercised, then close. A failure to close must not be reported as a
     * successful encode (contract section 7: no fake success). */
    if (exit_code == 0 && !encode_failed) {
        const SvtAv2Error eos_err = svt_av2_enc_send_picture(handle, NULL);
        if (eos_err != SVT_AV2_ERROR_NONE) {
            fprintf(stderr, "error: EOS (send_picture(NULL)) failed (%s)\n", svt_av2_error_to_string(eos_err));
            exit_code = 9;
        }
    }

    free(packet);
    free(group_input);
    free(group_weights);
    free(group_recon);
    free(originals);
    free(pending_input);
    free(pending_recon);
    free(recon_buffer);
    free(frame_buffer);
    fclose(input);

    /* Closing the output is where a buffered write can still fail; a successful
     * fwrite loop is not proof that the file is complete. */
    if (fclose(output) != 0 && exit_code == 0) {
        fprintf(stderr, "error: closing '%s' failed\n", options.output_path);
        exit_code = 12;
    }
    if (recon_output != NULL) {
        /* Same rule for the additive reconstruction file. */
        if (fclose(recon_output) != 0 && exit_code == 0) {
            fprintf(stderr, "error: closing '%s' failed\n", options.recon_path);
            exit_code = 12;
        }
    }

    const SvtAv2Error deinit_err = svt_av2_enc_deinit(&handle);
    if (exit_code != 0) {
        return exit_code;
    }
    if (deinit_err != SVT_AV2_ERROR_NONE) {
        fprintf(stderr, "error: svt_av2_enc_deinit failed (%s)\n", svt_av2_error_to_string(deinit_err));
        return 13;
    }

    printf("mode        : AV2 elementary stream (Annex B)\n");
    printf("output      : %s\n", options.output_path);
    if (options.recon_path != NULL) {
        printf("recon       : %s (%u frame%s, tight planar raw; 10-bit uses u16le)\n",
               options.recon_path,
               frames_written,
               (frames_written == 1u) ? "" : "s");
    }
    printf("frames      : %u encoded\n", frames_written);
    printf("note        : %s, coded transform residuals\n",
           options.prediction_structure == SVT_AV2_PRED_LOW_DELAY_P
               ? (lookahead == 3u ? "four-picture group, three-picture lookahead" : paired ? "paired future anchors, one-picture lookahead" : options.dual_ref_policy == 3u ? "LOW_DELAY_P, block-selected LAST/LONG" :
                    "LOW_DELAY_P, single logical reconstructed reference") :
                 options.intra_mode_selection == 4u ? "SINGLE_PICTURE/directional Y + implicit CfL UV RD" :
                 options.intra_mode_selection == 3u ? "SINGLE_PICTURE/directional Y + basic UV RD" :
                 options.intra_mode_selection == 2u ? "SINGLE_PICTURE/basic Y/UV RD" : "ALL_INTRA/DC_PRED");
    return 0;
}
