"""Encode a 10-bit 4:2:0 Y4M sequence with a measured AV2 video profile."""
import argparse
import os
import subprocess
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qp", required=True, type=int)
    parser.add_argument("--frames", type=int, help="Encode up to this many frames (default: until EOF)")
    parser.add_argument("--export-recon", type=Path)
    parser.add_argument("--profile", choices=("anchor-gdf", "anchor-deep-key", "drawing-rd", "joint-motion", "paired-refs", "quartet-refs", "adaptive-gop"),
                        default="joint-motion", help="joint-motion, anchor-gdf, anchor-deep-key and drawing-rd use low delay. paired-refs uses one frame of lookahead, quartet-refs and adaptive-gop up to three.")
    args = parser.parse_args()
    paths = [p.resolve() for p in (args.encoder, args.input, args.output, args.export_recon) if p]
    if len(set(paths)) != len(paths):
        parser.error("Input, encoder, output and reconstruction must have distinct paths")
    command, settings = joint_motion(args.encoder.resolve(), args.input, args.output,
                                     args.qp, args.frames, args.export_recon)
    if args.profile in ("anchor-gdf", "anchor-deep-key", "drawing-rd"):
        settings.update(SVT_AV2_MOTION_Q="rotating-anchor", SVT_AV2_TEMPORAL_ME="1", SVT_AV2_GDF_SEARCH="sampled")
        if args.profile in ("anchor-deep-key", "drawing-rd"):
            settings.update(SVT_AV2_VIDEO_KEY_SEARCH="deep", SVT_AV2_STILL_RECT_RD="deep", SVT_AV2_RECT_RDOQ="1")
        if args.profile == "drawing-rd":
            settings["SVT_AV2_MOTION_Q"] = "drawing-rd"
    elif args.profile == "paired-refs":
        settings.update(SVT_AV2_MOTION_Q="paired-refs", SVT_AV2_GDF="key", SVT_AV2_GDF_SEARCH="sampled")
    elif args.profile in ("quartet-refs", "adaptive-gop"):
        settings.update(SVT_AV2_MOTION_Q=args.profile, SVT_AV2_TEMPORAL_ME="1", SVT_AV2_GDF_SEARCH="sampled")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SVT_AV2_")}
    return subprocess.run(command, env=dict(env, **settings)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
