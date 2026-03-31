#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vid2model_lib.auto_pose_dataset import append_examples_jsonl, build_auto_pose_example
from vid2model_lib.pipeline import collect_detected_pose_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a labeled auto-pose dataset from videos.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input video path. Repeat for multiple videos.",
    )
    parser.add_argument("--label", required=True, help="Label for all provided inputs.")
    parser.add_argument(
        "--output",
        default="output/auto_pose_dataset.jsonl",
        help="Output JSONL dataset path.",
    )
    parser.add_argument("--model-complexity", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--opencv-enhance",
        default="off",
        choices=("off", "light", "strong"),
        help="Optional preprocessing before pose detection.",
    )
    parser.add_argument("--max-frame-side", type=int, default=0)
    parser.add_argument(
        "--roi-crop",
        default="off",
        choices=("off", "auto"),
        help="Adaptive ROI crop between frames.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.output).expanduser()

    examples = []
    for raw_input in args.input:
        input_path = Path(raw_input).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input video does not exist: {input_path}")

        fps, _frames_pts_raw, detected_samples, stats = collect_detected_pose_samples(
            input_path=input_path,
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            progress_every=args.progress_every,
            opencv_enhance=args.opencv_enhance,
            max_frame_side=args.max_frame_side,
            roi_crop=args.roi_crop,
        )
        if not detected_samples:
            raise RuntimeError(f"No pose samples detected in {input_path}")

        example = build_auto_pose_example(
            samples=detected_samples,
            label=args.label,
            source=str(input_path),
            meta={
                "fps": float(fps),
                "detected_frames": int(stats["detected"]),
                "source_frames": int(stats["frames"]),
                "roi_used": int(stats["roi_used"]),
                "roi_fallback": int(stats["roi_fallback"]),
                "roi_resets": int(stats["roi_resets"]),
            },
        )
        examples.append(example)
        print(
            f"[vid2model] dataset example label={args.label} source={input_path} "
            f"samples={len(detected_samples)}"
        )

    written = append_examples_jsonl(out_path, examples)
    print(f"[vid2model] wrote {written} examples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
