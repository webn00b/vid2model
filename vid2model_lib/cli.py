from __future__ import annotations

import argparse
import json
import importlib.util
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict

from .pipeline import build_pose_correction_profile, convert_video_to_bvh
from .writers import write_bvh, write_csv, write_diagnostic_json, write_json, write_npz, write_trc


@dataclass(frozen=True)
class CliOptions:
    input_path: Path
    output_bvh: Path | None
    output_json: Path | None
    output_csv: Path | None
    output_npz: Path | None
    output_trc: Path | None
    output_diag_json: Path | None
    skeleton_profile: Dict[str, Any] | None
    model_complexity: int
    min_detection_confidence: float
    min_tracking_confidence: float
    max_gap_interpolate: int
    opencv_enhance: str
    max_frame_side: int
    roi_crop: str
    progress_every: int
    pose_corrections: Any
    root_yaw_offset_deg: float
    lower_body_rotation_mode: str
    loop_mode: str
    preset: str


OUTPUT_WRITERS: tuple[tuple[str, str, str], ...] = (
    ("output_bvh", "write_bvh", "Saved BVH"),
    ("output_json", "write_json", "Saved JSON"),
    ("output_csv", "write_csv", "Saved CSV"),
    ("output_npz", "write_npz", "Saved NPZ"),
    ("output_trc", "write_trc", "Saved TRC"),
    ("output_diag_json", "write_diagnostic_json", "Saved diagnostic JSON"),
)


PRESET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "idle": {
        "opencv_enhance": "light",
        "roi_crop": "auto",
        "max_gap_interpolate": 12,
        "upper_body_rotation_scale": 0.75,
        "arm_rotation_scale": 0.8,
    },
    "walk": {
        "opencv_enhance": "light",
        "roi_crop": "auto",
        "max_gap_interpolate": 6,
        "loop_mode": "auto",
        "upper_body_rotation_scale": 0.9,
        "arm_rotation_scale": 0.95,
    },
    "run": {
        "model_complexity": 2,
        "opencv_enhance": "light",
        "roi_crop": "auto",
        "max_gap_interpolate": 4,
        "loop_mode": "auto",
    },
    "dance": {
        "model_complexity": 2,
        "opencv_enhance": "strong",
        "roi_crop": "auto",
        "max_gap_interpolate": 4,
        "loop_mode": "off",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert video to BVH/JSON/CSV/NPZ/TRC using MediaPipe Pose.")
    parser.add_argument("--config", help="Path to config file (.json/.yaml/.yml)")
    parser.add_argument(
        "--preset",
        choices=["idle", "walk", "run", "dance"],
        help="Optional motion preset that adjusts defaults for cleanup and detection.",
    )
    parser.add_argument("--input", help="Path to input video file (mp4/webm/mov/etc)")
    parser.add_argument("--output", help="Legacy output path for BVH (same as --output-bvh).")
    parser.add_argument("--output-bvh", help="Path to output BVH file.")
    parser.add_argument("--output-json", help="Path to output JSON file.")
    parser.add_argument("--output-csv", help="Path to output CSV file (frame channels).")
    parser.add_argument("--output-npz", help="Path to output NPZ file (compressed arrays).")
    parser.add_argument("--output-trc", help="Path to output TRC file (marker trajectories).")
    parser.add_argument("--output-diag-json", help="Path to output diagnostic JSON file.")
    parser.add_argument("--skeleton-profile-json", help="Path to model skeleton profile JSON used to override BVH rest offsets.")
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
    parser.add_argument("--upper-rotation-offset-deg", type=float, help="Extra Y rotation offset for upper body in degrees.")
    parser.add_argument("--upper-body-rotation-scale", type=float, help="Scale upper-body BVH rotations (e.g. 0.35 keeps 35% of torso/arm rotation).")
    parser.add_argument("--arm-rotation-scale", type=float, help="Scale arm and hand BVH rotations separately from torso (e.g. 0.15).")
    parser.add_argument("--root-yaw-offset-deg", type=float, help="Extra source root yaw offset in degrees.")
    parser.add_argument(
        "--lower-body-rotation-mode",
        choices=["off", "invert", "yaw180"],
        help="Optional lower-body source rotation correction.",
    )
    parser.add_argument(
        "--loop-mode",
        choices=["off", "auto", "force"],
        help="Optional loop extraction for cyclic clips.",
    )
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


def _merge_config_value(args: argparse.Namespace, cfg: Dict[str, Any], name: str, default: Any) -> Any:
    cli_value = getattr(args, name)
    if cli_value is not None:
        return cli_value
    if name in cfg and cfg[name] is not None:
        return cfg[name]
    return default


def _resolve_preset_name(args: argparse.Namespace, cfg: Dict[str, Any]) -> str:
    preset = _merge_config_value(args, cfg, "preset", "default")
    normalized = str(preset or "default").strip().lower()
    if normalized not in PRESET_DEFAULTS:
        raise ValueError(f"preset must be one of: {', '.join(PRESET_DEFAULTS)}")
    return normalized


def _resolve_path(value: Any) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _load_json_object(path: Path, description: str) -> Dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{description} root must be an object/map.")
    return loaded


def _validate_choice(value: str, name: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(choices)}")


def _validate_minimum(value: int, name: str, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _require_probability(value: float, name: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in range [0.0, 1.0]")


def _build_cli_options(args: argparse.Namespace, cfg: Dict[str, Any]) -> CliOptions:
    preset = _resolve_preset_name(args, cfg)
    preset_defaults = PRESET_DEFAULTS[preset]
    input_value = _merge_config_value(args, cfg, "input", None)
    output_value = _merge_config_value(args, cfg, "output", None)
    output_bvh = _resolve_path(_merge_config_value(args, cfg, "output_bvh", None))
    output_json = _resolve_path(_merge_config_value(args, cfg, "output_json", None))
    output_csv = _resolve_path(_merge_config_value(args, cfg, "output_csv", None))
    output_npz = _resolve_path(_merge_config_value(args, cfg, "output_npz", None))
    output_trc = _resolve_path(_merge_config_value(args, cfg, "output_trc", None))
    output_diag_json = _resolve_path(_merge_config_value(args, cfg, "output_diag_json", None))
    skeleton_profile_json = _resolve_path(_merge_config_value(args, cfg, "skeleton_profile_json", None))

    model_complexity = int(_merge_config_value(args, cfg, "model_complexity", preset_defaults.get("model_complexity", 1)))
    min_detection_confidence = float(_merge_config_value(args, cfg, "min_detection_confidence", 0.5))
    min_tracking_confidence = float(_merge_config_value(args, cfg, "min_tracking_confidence", 0.5))
    max_gap_interpolate = int(_merge_config_value(args, cfg, "max_gap_interpolate", preset_defaults.get("max_gap_interpolate", 8)))
    opencv_enhance = str(_merge_config_value(args, cfg, "opencv_enhance", preset_defaults.get("opencv_enhance", "off"))).strip().lower()
    max_frame_side = int(_merge_config_value(args, cfg, "max_frame_side", 0))
    roi_crop = str(_merge_config_value(args, cfg, "roi_crop", preset_defaults.get("roi_crop", "off"))).strip().lower()
    progress_every = int(_merge_config_value(args, cfg, "progress_every", 100))
    root_yaw_offset_deg = float(_merge_config_value(args, cfg, "root_yaw_offset_deg", 0.0))
    lower_body_rotation_mode = str(_merge_config_value(args, cfg, "lower_body_rotation_mode", "off")).strip().lower()
    loop_mode = str(_merge_config_value(args, cfg, "loop_mode", preset_defaults.get("loop_mode", "off"))).strip().lower()

    pose_corrections = build_pose_correction_profile(cfg.get("pose_corrections"))
    pose_corrections = replace(
        pose_corrections,
        upper_rotation_offset_deg=float(
            _merge_config_value(args, cfg, "upper_rotation_offset_deg", pose_corrections.upper_rotation_offset_deg)
        ),
        upper_body_rotation_scale=float(
            _merge_config_value(
                args,
                cfg,
                "upper_body_rotation_scale",
                preset_defaults.get("upper_body_rotation_scale", pose_corrections.upper_body_rotation_scale),
            )
        ),
        arm_rotation_scale=float(
            _merge_config_value(
                args,
                cfg,
                "arm_rotation_scale",
                preset_defaults.get("arm_rotation_scale", pose_corrections.arm_rotation_scale),
            )
        ),
    )

    if model_complexity not in (0, 1, 2):
        raise ValueError("model_complexity must be one of: 0, 1, 2")
    _require_probability(min_detection_confidence, "min_detection_confidence")
    _require_probability(min_tracking_confidence, "min_tracking_confidence")
    _validate_minimum(max_gap_interpolate, "max_gap_interpolate", 0)
    _validate_choice(opencv_enhance, "opencv_enhance", ("off", "light", "strong"))
    _validate_minimum(max_frame_side, "max_frame_side", 0)
    _validate_choice(roi_crop, "roi_crop", ("off", "auto"))
    _validate_minimum(progress_every, "progress_every", 0)
    _validate_choice(lower_body_rotation_mode, "lower_body_rotation_mode", ("off", "invert", "yaw180"))
    _validate_choice(loop_mode, "loop_mode", ("off", "auto", "force"))

    if not input_value:
        raise ValueError("Specify --input (or use --check-tools).")

    input_path = Path(input_value).expanduser().resolve()
    if output_value:
        if output_bvh is not None:
            raise ValueError("Use either --output or --output-bvh, not both.")
        output_bvh = Path(output_value).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if skeleton_profile_json is not None and not skeleton_profile_json.exists():
        raise FileNotFoundError(f"Skeleton profile JSON not found: {skeleton_profile_json}")

    if all(path is None for path in (output_bvh, output_json, output_csv, output_npz, output_trc, output_diag_json)):
        raise ValueError(
            "Specify at least one target: --output-bvh, --output-json, --output-csv, --output-npz, --output-trc, --output-diag-json (or --output)."
        )

    skeleton_profile = None
    if skeleton_profile_json is not None:
        skeleton_profile = _load_json_object(skeleton_profile_json, "Skeleton profile JSON")

    return CliOptions(
        input_path=input_path,
        output_bvh=output_bvh,
        output_json=output_json,
        output_csv=output_csv,
        output_npz=output_npz,
        output_trc=output_trc,
        output_diag_json=output_diag_json,
        skeleton_profile=skeleton_profile,
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
        lower_body_rotation_mode=lower_body_rotation_mode,
        loop_mode=loop_mode,
        preset=preset,
    )


def main() -> int:
    args = parse_args()
    if args.check_tools:
        return check_tools()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(Path(args.config).expanduser().resolve())
    options = _build_cli_options(args, cfg)

    fps, rest_offsets, motion_values, ref_root, frames_pts, diagnostics = convert_video_to_bvh(
        input_path=options.input_path,
        model_complexity=options.model_complexity,
        min_detection_confidence=options.min_detection_confidence,
        min_tracking_confidence=options.min_tracking_confidence,
        max_gap_interpolate=options.max_gap_interpolate,
        opencv_enhance=options.opencv_enhance,
        max_frame_side=options.max_frame_side,
        roi_crop=options.roi_crop,
        progress_every=options.progress_every,
        pose_corrections=options.pose_corrections,
        skeleton_profile=options.skeleton_profile,
        root_yaw_offset_deg=options.root_yaw_offset_deg,
        lower_body_rotation_mode=options.lower_body_rotation_mode,
        loop_mode=options.loop_mode,
        include_source_stage_diagnostics=options.output_diag_json is not None,
    )
    quality = diagnostics.get("quality")
    if isinstance(quality, dict):
        rating = str(quality.get("rating", "unknown"))
        score = float(quality.get("score", 0.0))
        reasons = quality.get("reasons") or []
        reason_text = ", ".join(str(item) for item in reasons[:4]) if isinstance(reasons, list) and reasons else "none"
        print(
            f"[vid2model] quality rating={rating} score={score:.3f} preset={options.preset} reasons={reason_text}",
            file=sys.stderr,
        )

    writer_inputs: Dict[str, tuple[Any, ...]] = {
        "output_bvh": (fps, rest_offsets, motion_values),
        "output_json": (options.input_path, fps, rest_offsets, motion_values, ref_root),
        "output_csv": (motion_values,),
        "output_npz": (options.input_path, fps, rest_offsets, motion_values, ref_root),
        "output_trc": (options.input_path, fps, frames_pts, ref_root),
        "output_diag_json": (diagnostics,),
    }
    for attr_name, writer_name, message in OUTPUT_WRITERS:
        output_path = getattr(options, attr_name)
        if output_path is None:
            continue
        writer = globals()[writer_name]
        writer(output_path, *writer_inputs[attr_name])
        print(f"{message}: {output_path}")

    return 0
