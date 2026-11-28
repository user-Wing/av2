"""Encode a 10-bit 4:2:0 Y4M sequence with a measured AV2 video profile."""
import argparse
import os
import subprocess
from pathlib import Path


# Dispatch each experiment through the same executable used by ordinary CLI callers.
_SEARCH = {"SVT_AV2_COMPOUND_MV": "search"}
_DIRECTIONAL = {"SVT_AV2_VIDEO_INTRA_REFINE": "directional"}
_TRAINED = {**_SEARCH, **_DIRECTIONAL, "SVT_AV2_SPARSE_PLANES": "trained-levels"}
_HOLD = {**_TRAINED, "SVT_AV2_DISPLAY_REFS": "all"}
_ANCHORS = {**_HOLD, "SVT_AV2_COMPOUND_MV": "block-rd", "SVT_AV2_COMPOUND_REFS": "anchors",
            "SVT_AV2_INTER_SEARCH": "hme-cost-long", "SVT_AV2_ME_RANGE": "31"}
_WIDE = {**_ANCHORS, "SVT_AV2_ME_RANGE": "63"}
_SPATIAL_RECT = {**_WIDE, "SVT_AV2_VIDEO_RECT": "root-motion", "SVT_AV2_TEMPORAL_ALLOC": "spatial"}
# Native motion profile, key QP offset, additional search settings.
_PYRAMID = {
    "pyramid8": ("pyramid8", 40, {}),
    "pyramid8-rd": ("pyramid8-rd", 22, {}),
    "pyramid-tail": ("pyramid-tail", 26, {}),
    "pyramid-compound": ("pyramid-compound", 26, {}),
    "pyramid-motion": ("pyramid-motion", 30, {"SVT_AV2_INTER_SEARCH": "hme-cost-long", "SVT_AV2_ME_RANGE": "31"}),
    "pyramid16": ("pyramid16", 30, {}),
    "pyramid-window": ("pyramid-window", 30, {}),
    "pyramid-dependency": ("pyramid-window", 30, {"SVT_AV2_TEMPORAL_ALLOC": "propagate"}),
    "pyramid-near": ("pyramid16", 30, {"SVT_AV2_COMPOUND_MV": "near"}),
    "pyramid-search": ("pyramid16", 30, _SEARCH),
    "pyramid-intra": ("pyramid16", 30, _DIRECTIONAL),
    "pyramid-levels": ("pyramid16", 30, {**_SEARCH, "SVT_AV2_SPARSE_PLANES": "video-levels"}),
    "pyramid-trained": ("pyramid16", 30, _TRAINED),
    "pyramid-hold": ("pyramid16", 30, _HOLD),
    "pyramid-block-rd": ("pyramid16", 30, {**_HOLD, "SVT_AV2_COMPOUND_MV": "block-rd"}),
    "pyramid-anchor-rd": ("pyramid16", 30, _ANCHORS),
    "pyramid-cadence": ("pyramid16", 30, {**_ANCHORS, "SVT_AV2_DRAWING_RUNS": "cadence"}),
    "pyramid-wide": ("pyramid16", 30, _WIDE),
    "pyramid-altref": ("pyramid16", 30, {**_ANCHORS, "SVT_AV2_TEMPORAL_FILTER": "hidden"}),
    "pyramid-multi-rd": ("pyramid16", 30, {**_WIDE, "SVT_AV2_COMPOUND_MV": "multi-rd"}),
    "pyramid-joint-rd": ("pyramid16", 30, {**_WIDE, "SVT_AV2_COMPOUND_MV": "joint-rd"}),
    "pyramid-short-rd": ("pyramid-compound", 30, {**_WIDE, "SVT_AV2_COMPOUND_MV": "multi-rd"}),
    "pyramid-tpl": ("pyramid16", 30, {**_WIDE, "SVT_AV2_TEMPORAL_ALLOC": "regions"}),
    "pyramid-root-rect": ("pyramid16", 30, {**_WIDE, "SVT_AV2_VIDEO_RECT": "root-motion"}),
    "pyramid-tpl-sb": ("pyramid16", 30, {**_WIDE, "SVT_AV2_TEMPORAL_ALLOC": "spatial"}),
    "pyramid-bank": ("pyramid16", 30, {**_WIDE, "SVT_AV2_REF_SEARCH": "past-bank"}),
    "pyramid-contrast": ("pyramid16", 30, {**_ANCHORS, "SVT_AV2_DETAIL_RD": "inter-contrast"}),
    "pyramid-span-rd": ("pyramid-window", 30, {**_WIDE, "SVT_AV2_GOP_SEARCH": "regions"}),
    "pyramid-quality": ("pyramid16", 30, {**_SPATIAL_RECT, "SVT_AV2_COMPOUND_MV": "multi-rd"}),
    "pyramid-quality-joint": ("pyramid16", 30, {**_SPATIAL_RECT, "SVT_AV2_COMPOUND_MV": "joint-rd"}),
    "pyramid-direct-roots": ("pyramid16", 30, {**_SPATIAL_RECT, "SVT_AV2_COMPOUND_MV": "joint-rd",
                                               "SVT_AV2_TEMPORAL_ALLOC": "direct-roots"}),
    "pyramid-protected-roots": ("pyramid16", 30, {**_SPATIAL_RECT, "SVT_AV2_COMPOUND_MV": "joint-rd",
                                                  "SVT_AV2_TEMPORAL_ALLOC": "protected-roots"}),
    "pyramid-inter-matrix": ("pyramid16", 30, {**_SPATIAL_RECT, "SVT_AV2_COMPOUND_MV": "joint-rd",
                                               "SVT_AV2_QM_LEVEL": "inter-curve"}),
    "pyramid-held-drawings": ("pyramid16", 30, {**_SPATIAL_RECT, "SVT_AV2_COMPOUND_MV": "joint-rd",
                                                "SVT_AV2_TEMPORAL_FILTER": "held-drawings"}),
    "pyramid-reuse-drawing": ("pyramid16", 30, {**_SPATIAL_RECT, "SVT_AV2_COMPOUND_MV": "joint-rd",
                                                "SVT_AV2_TEMPORAL_ALLOC": "reuse-drawing"}),
    "pyramid-local-reuse": ("pyramid16", 30, {**_SPATIAL_RECT, "SVT_AV2_COMPOUND_MV": "joint-rd",
                                              "SVT_AV2_TEMPORAL_ALLOC": "local-reuse"}),
    "pyramid-perceptual": ("pyramid16", 30, {**_HOLD, "SVT_AV2_SSIM_RD": "p", "SVT_AV2_QM_TRELLIS": "luma"}),
}


def joint_motion(encoder, source, output, qp, frames=None, recon=None):
    settings = dict(
        SVT_AV2_MOTION_Q="motion-refresh", SVT_AV2_FINAL_SKIP="1", SVT_AV2_INTER_ME_WIDE="1",
        SVT_AV2_INTER_FINE_FULL="1", SVT_AV2_INTER_SEARCH="hme-cost", SVT_AV2_EDGE_INTER="1",
        SVT_AV2_VIDEO_RECT="dual-motion", SVT_AV2_VIDEO_CFL="all", SVT_AV2_GDF="all",
        SVT_AV2_QM12="all", SVT_AV2_QM_LEVEL="8", SVT_AV2_FP_ROUND="matrix", SVT_AV2_IBP="1",
        SVT_AV2_UV_DELTA_Q="-8", SVT_AV2_DCT64_MATCH="all", SVT_AV2_VIDEO_KEY_SEARCH="full",
        SVT_AV2_VIDEO_INTERPOLATION="regular", SVT_AV2_SPARSE_PLANES="levels",
    )
    command = [
        str(encoder), "--input", str(source), "--output", str(output), "--qp", str(qp),
        "--threads", "1", "--pred-structure", "low-delay-p", "--intra-select", "adaptive",
        "--partition-select", "adaptive", "--rd-mode", "pruned", "--deterministic",
        "--motion-opt", "quarter", "--preset", "4", "--entropy-adapt", "on",
        "--cdf-inherit", "on", "--dual-ref-period", "4", "--dual-ref-policy", "block",
        "--inter-fine", "rd", "--inter-near", "rd", "--inter-tail", "rd",
        "--key-qp", str(max(1, qp - 40)), "--anchor-qp", str(max(1, qp - 20)),
        "--dual-ref-cadence", "adaptive", "--rd-lambda", "avm-ai150",
    ]
    if frames is not None:
        command.extend(["--frames", str(frames)])
    if recon is not None:
        command.extend(["--export-recon", str(recon)])
    return command, settings


def video_profile(profile, encoder, source, output, qp, frames=None, recon=None):
    command, settings = joint_motion(encoder, source, output, qp, frames, recon)
    if profile in ("anchor-gdf", "anchor-deep-key", "drawing-rd"):
        settings.update(SVT_AV2_MOTION_Q="rotating-anchor", SVT_AV2_TEMPORAL_ME="1", SVT_AV2_GDF_SEARCH="sampled")
        if profile in ("anchor-deep-key", "drawing-rd"):
            settings.update(SVT_AV2_VIDEO_KEY_SEARCH="deep", SVT_AV2_STILL_RECT_RD="deep", SVT_AV2_RECT_RDOQ="1")
        if profile == "drawing-rd":
            settings["SVT_AV2_MOTION_Q"] = "drawing-rd"
    elif profile == "paired-refs":
        settings.update(SVT_AV2_MOTION_Q="paired-refs", SVT_AV2_GDF="key", SVT_AV2_GDF_SEARCH="sampled")
    elif profile in ("quartet-refs", "adaptive-gop"):
        settings.update(SVT_AV2_MOTION_Q=profile, SVT_AV2_TEMPORAL_ME="1", SVT_AV2_GDF_SEARCH="sampled")
    elif profile in _PYRAMID or profile == "block-motion":
        motion, offset, extra = _PYRAMID.get(profile, ("drawing-rd", 40, {"SVT_AV2_INTER_FINE_FULL": "local"}))
        settings.update(SVT_AV2_MOTION_Q=motion, SVT_AV2_TEMPORAL_ME="1", SVT_AV2_GDF_SEARCH="sampled",
                        SVT_AV2_VIDEO_KEY_SEARCH="deep", SVT_AV2_STILL_RECT_RD="deep", SVT_AV2_RECT_RDOQ="1")
        settings.update(extra)
        command[command.index("--key-qp") + 1] = str(max(1, qp - offset))
    return command, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qp", required=True, type=int)
    parser.add_argument("--frames", type=int, help="Encode up to this many frames (default: until EOF)")
    parser.add_argument("--export-recon", type=Path)
    parser.add_argument("--profile", choices=("anchor-gdf", "anchor-deep-key", "drawing-rd", "joint-motion",
                                              "paired-refs", "quartet-refs", "adaptive-gop", "block-motion", *_PYRAMID),
                        default="joint-motion", help="Video coding and motion search profile. Hierarchical profiles "
                        "buffer up to 8, 16 or 32 pictures depending on the prediction structure.")
    args = parser.parse_args()
    paths = [p.resolve() for p in (args.encoder, args.input, args.output, args.export_recon) if p]
    if len(set(paths)) != len(paths):
        parser.error("Input, encoder, output and reconstruction must have distinct paths")
    command, settings = video_profile(args.profile, args.encoder.resolve(), args.input, args.output,
                                      args.qp, args.frames, args.export_recon)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SVT_AV2_")}
    return subprocess.run(command, env=dict(env, **settings)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
