from __future__ import annotations

import argparse
import json
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from .pipeline import build_pose_correction_profile, convert_video_to_bvh
from .writers import write_bvh, write_csv, write_json, write_npz, write_trc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert video to BVH/JSON/CSV/NPZ/TRC using MediaPipe Pose.")
    parser.add_argument("--config", help="Path to config file (.json/.yaml/.yml)")
    parser.add_argument("--input", help="Path to input video file (mp4/webm/mov/etc)")
    parser.add_argument("--output", help="Legacy output path for BVH (same as --output-bvh).")
    parser.add_argument("--output-bvh", help="Path to output BVH file.")
    parser.add_argument("--output-json", help="Path to output JSON file.")
    parser.add_argument("--output-csv", help="Path to output CSV file (frame channels).")
    parser.add_argument("--output-npz", help="Path to output NPZ file (compressed arrays).")
    parser.add_argument("--output-trc", help="Path to output TRC file (marker trajectories).")
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2])
    parser.add_argument("--min-detection-confidence", type=float)
    parser.add_argument("--min-tracking-confidence", type=float)
    parser.add_argument("--max-gap-interpolate", type=int, help="Interpolate missing detections for gaps up to N frames.")
    parser.add_argument(
        "--opencv-enhance",
        choices=["off", "light", "strong"],
        help="Optional OpenCV preprocessing before pose detection.",
    )
    parser.add_argument(
        "--max-frame-side",
        type=int,
        help="Resize frame before detection so longest side is <= N pixels (0 disables).",
    )
    parser.add_argument(
        "--roi-crop",
        choices=["off", "auto"],
        help="Adaptive ROI crop around detected person between frames.",
    )
    parser.add_argument("--progress-every", type=int, help="Print progress every N frames (0 disables).")
    parser.add_argument("--root-yaw-offset-deg", type=float, help="Extra source root yaw offset in degrees.")
    parser.add_argument("--check-tools", action="store_true", help="Validate local toolchain and exit.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "YAML config requires PyYAML. Install it or use .json config."
            ) from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError("Unsupported config format. Use .json, .yaml or .yml")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be an object/map.")
    return data


def check_tools() -> int:
    py_ok = (3, 10) <= sys.version_info[:2] <= (3, 12)
    cv2_ok = importlib.util.find_spec("cv2") is not None
    mediapipe_ok = importlib.util.find_spec("mediapipe") is not None
    blender_bin = shutil.which("blender")
    if blender_bin is None and Path("/Applications/Blender.app/Contents/MacOS/Blender").exists():
        blender_bin = "/Applications/Blender.app/Contents/MacOS/Blender"

    print(f"python: {sys.version.split()[0]} ({'ok' if py_ok else 'warn: recommended 3.10-3.12'})")
    print(f"opencv(cv2): {'ok' if cv2_ok else 'missing'}")
    print(f"mediapipe: {'ok' if mediapipe_ok else 'missing'}")
    print(f"blender: {blender_bin if blender_bin else 'not found (optional for FBX)'}")

    return 0 if (cv2_ok and mediapipe_ok) else 1


def main() -> int:
    args = parse_args()
    if args.check_tools:
        return check_tools()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(Path(args.config).expanduser().resolve())

    def merged(name: str, default: Any) -> Any:
        cli_value = getattr(args, name)
        if cli_value is not None:
            return cli_value
        if name in cfg and cfg[name] is not None:
            return cfg[name]
        return default

    input_value = merged("input", None)
    output_value = merged("output", None)
    output_bvh_value = merged("output_bvh", None)
    output_json_value = merged("output_json", None)
    output_csv_value = merged("output_csv", None)
    output_npz_value = merged("output_npz", None)
    output_trc_value = merged("output_trc", None)
    model_complexity = int(merged("model_complexity", 1))
    min_detection_confidence = float(merged("min_detection_confidence", 0.5))
    min_tracking_confidence = float(merged("min_tracking_confidence", 0.5))
    max_gap_interpolate = int(merged("max_gap_interpolate", 8))
    opencv_enhance = str(merged("opencv_enhance", "off")).strip().lower()
    max_frame_side = int(merged("max_frame_side", 0))
    roi_crop = str(merged("roi_crop", "off")).strip().lower()
    progress_every = int(merged("progress_every", 100))
    root_yaw_offset_deg = float(merged("root_yaw_offset_deg", 0.0))
    pose_corrections = build_pose_correction_profile(cfg.get("pose_corrections"))

    if model_complexity not in (0, 1, 2):
        raise ValueError("model_complexity must be one of: 0, 1, 2")
    if not (0.0 <= min_detection_confidence <= 1.0):
        raise ValueError("min_detection_confidence must be in range [0.0, 1.0]")
    if not (0.0 <= min_tracking_confidence <= 1.0):
        raise ValueError("min_tracking_confidence must be in range [0.0, 1.0]")
    if max_gap_interpolate < 0:
        raise ValueError("max_gap_interpolate must be >= 0")
    if opencv_enhance not in ("off", "light", "strong"):
        raise ValueError("opencv_enhance must be one of: off, light, strong")
    if max_frame_side < 0:
        raise ValueError("max_frame_side must be >= 0")
    if roi_crop not in ("off", "auto"):
        raise ValueError("roi_crop must be one of: off, auto")
    if progress_every < 0:
        raise ValueError("progress_every must be >= 0")

    if not input_value:
        raise ValueError("Specify --input (or use --check-tools).")

    input_path = Path(input_value).expanduser().resolve()
    output_bvh = Path(output_bvh_value).expanduser().resolve() if output_bvh_value else None
    output_json = Path(output_json_value).expanduser().resolve() if output_json_value else None
    output_csv = Path(output_csv_value).expanduser().resolve() if output_csv_value else None
    output_npz = Path(output_npz_value).expanduser().resolve() if output_npz_value else None
    output_trc = Path(output_trc_value).expanduser().resolve() if output_trc_value else None

    if output_value:
        if output_bvh is not None:
            raise ValueError("Use either --output or --output-bvh, not both.")
        output_bvh = Path(output_value).expanduser().resolve()

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
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        max_gap_interpolate=max_gap_interpolate,
        opencv_enhance=opencv_enhance,
        max_frame_side=max_frame_side,
        roi_crop=roi_crop,
        progress_every=progress_every,
        pose_corrections=pose_corrections,
        root_yaw_offset_deg=root_yaw_offset_deg,
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
