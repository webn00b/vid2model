from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import convert_video_to_bvh
from .writers import write_bvh, write_csv, write_json, write_npz, write_trc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert video to BVH/JSON/CSV/NPZ/TRC using MediaPipe Pose.")
    parser.add_argument("--input", required=True, help="Path to input video file (mp4/webm/mov/etc)")
    parser.add_argument("--output", help="Legacy output path for BVH (same as --output-bvh).")
    parser.add_argument("--output-bvh", help="Path to output BVH file.")
    parser.add_argument("--output-json", help="Path to output JSON file.")
    parser.add_argument("--output-csv", help="Path to output CSV file (frame channels).")
    parser.add_argument("--output-npz", help="Path to output NPZ file (compressed arrays).")
    parser.add_argument("--output-trc", help="Path to output TRC file (marker trajectories).")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_bvh = Path(args.output_bvh).expanduser().resolve() if args.output_bvh else None
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else None
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else None
    output_npz = Path(args.output_npz).expanduser().resolve() if args.output_npz else None
    output_trc = Path(args.output_trc).expanduser().resolve() if args.output_trc else None

    if args.output:
        if output_bvh is not None:
            raise ValueError("Use either --output or --output-bvh, not both.")
        output_bvh = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if (
        output_bvh is None
        and output_json is None
        and output_csv is None
        and output_npz is None
        and output_trc is None
    ):
        raise ValueError(
            "Specify at least one target: --output-bvh, --output-json, --output-csv, --output-npz, --output-trc (or --output)."
        )

    fps, rest_offsets, motion_values, ref_root, frames_pts = convert_video_to_bvh(
        input_path=input_path,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    if output_bvh is not None:
        write_bvh(output_bvh, fps, rest_offsets, motion_values)
        print(f"Saved BVH: {output_bvh}")
    if output_json is not None:
        write_json(output_json, input_path, fps, rest_offsets, motion_values, ref_root)
        print(f"Saved JSON: {output_json}")
    if output_csv is not None:
        write_csv(output_csv, motion_values)
        print(f"Saved CSV: {output_csv}")
    if output_npz is not None:
        write_npz(output_npz, input_path, fps, rest_offsets, motion_values, ref_root)
        print(f"Saved NPZ: {output_npz}")
    if output_trc is not None:
        write_trc(output_trc, input_path, fps, frames_pts, ref_root)
        print(f"Saved TRC: {output_trc}")

    return 0
