#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-}"
MIN_PYTHON_MINOR=10
MAX_PYTHON_MINOR=10
OPENCV_ENHANCE="${OPENCV_ENHANCE:-off}"
MAX_FRAME_SIDE="${MAX_FRAME_SIDE:-0}"
ROI_CROP="${ROI_CROP:-off}"
UPPER_ROTATION_OFFSET_DEG="${UPPER_ROTATION_OFFSET_DEG:-0}"
UPPER_BODY_ROTATION_SCALE="${UPPER_BODY_ROTATION_SCALE:-1}"
ARM_ROTATION_SCALE="${ARM_ROTATION_SCALE:-1}"
ROOT_YAW_OFFSET_DEG="${ROOT_YAW_OFFSET_DEG:-0}"
LOWER_BODY_ROTATION_MODE="${LOWER_BODY_ROTATION_MODE:-off}"
LOOP_MODE="${LOOP_MODE:-off}"
SKELETON_PROFILE_JSON="${SKELETON_PROFILE_JSON:-}"

python_minor() {
  "$1" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null
}

venv_python_supported() {
  local py="$1"
  local minor
  minor="$("$py" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || true)"
  [[ -n "$minor" ]] || return 1
  [[ "$minor" -ge "$MIN_PYTHON_MINOR" && "$minor" -le "$MAX_PYTHON_MINOR" ]]
}

is_supported_python() {
  local bin="$1"
  local minor
  minor="$(python_minor "$bin" || true)"
  [[ -n "$minor" ]] || return 1
  [[ "$minor" -ge "$MIN_PYTHON_MINOR" && "$minor" -le "$MAX_PYTHON_MINOR" ]]
}

select_python_bin() {
  local candidates=()
  if [[ -n "$PYTHON_BIN" ]]; then
    candidates+=("$PYTHON_BIN")
  fi
  candidates+=(python3.10 python3.12 python3.11 python3)

  local candidate
  for candidate in "${candidates[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1 && is_supported_python "$candidate"; then
      PYTHON_BIN="$candidate"
      return 0
    fi
  done
  return 1
}

if ! select_python_bin; then
  echo "No supported Python found (need Python 3.10 for the pinned mediapipe wheel)." >&2
  echo "Install Python 3.10 or set PYTHON_BIN explicitly to a supported interpreter." >&2
  exit 2
fi

recreate_venv() {
  if [[ -d "$VENV_DIR" ]]; then
    local backup_dir="${VENV_DIR}.broken.$(date +%s)"
    mv "$VENV_DIR" "$backup_dir"
    echo "Existing venv looks broken, moved to: $backup_dir"
  fi
  echo "Creating venv with $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
}

print_usage() {
  cat <<'USAGE'
Usage:
  ./convert.sh <input_video> <output_bvh> [output_json] [output_csv] [output_npz] [output_trc] [output_fbx]
  ./convert.sh --auto <input_video> [--all] [--fbx]

Auto mode:
  Writes to output/<input_stem>.* (e.g. think.mp4 -> output/think.bvh).
  --all  Also write json/csv/npz/trc.
  --fbx  Also write fbx via Blender.

Environment knobs:
  OPENCV_ENHANCE=off|light|strong   OpenCV pre-processing before pose detection.
  MAX_FRAME_SIDE=0|N                Resize frame so longest side <= N (0 disables).
  ROI_CROP=off|auto                 Adaptive person ROI crop between frames.
  UPPER_ROTATION_OFFSET_DEG=0|180   Extra Y rotation offset for upper body only.
  UPPER_BODY_ROTATION_SCALE=1|0.35  Keep only a fraction of torso/arm rotation.
  ARM_ROTATION_SCALE=1|0.15         Keep only a fraction of arm/hand rotation.
  ROOT_YAW_OFFSET_DEG=0|180|-90     Extra source root yaw offset in degrees.
  LOWER_BODY_ROTATION_MODE=off|invert|yaw180  Extra lower-body source rotation correction.
  LOOP_MODE=off|auto|force          Extract a cyclic loop window from the cleaned motion.
  SKELETON_PROFILE_JSON=path        Override BVH rest offsets using a model skeleton profile JSON.
USAGE
}

if [[ $# -lt 1 ]]; then
  print_usage
  exit 1
fi

INPUT=""
OUTPUT_BVH=""
OUTPUT_JSON=""
OUTPUT_CSV=""
OUTPUT_NPZ=""
OUTPUT_TRC=""
OUTPUT_FBX=""
OUTPUT_DIAG_JSON=""

if [[ "$1" == "--auto" ]]; then
  shift
  if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
  fi
  INPUT="$1"
  shift

  AUTO_ALL=0
  AUTO_FBX=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --all) AUTO_ALL=1 ;;
      --fbx) AUTO_FBX=1 ;;
      -h|--help)
        print_usage
        exit 0
        ;;
      *)
        echo "Unknown option in --auto mode: $1" >&2
        print_usage
        exit 2
        ;;
    esac
    shift
  done

  stem="$(basename "$INPUT")"
  stem="${stem%.*}"
  out_dir="$ROOT_DIR/output"
  mkdir -p "$out_dir"

  OUTPUT_BVH="$out_dir/$stem.bvh"
  if [[ "$AUTO_ALL" -eq 1 ]]; then
    OUTPUT_JSON="$out_dir/$stem.json"
    OUTPUT_CSV="$out_dir/$stem.csv"
    OUTPUT_NPZ="$out_dir/$stem.npz"
    OUTPUT_TRC="$out_dir/$stem.trc"
  fi
  OUTPUT_DIAG_JSON="$out_dir/$stem.diag.json"
  if [[ "$AUTO_FBX" -eq 1 ]]; then
    OUTPUT_FBX="$out_dir/$stem.fbx"
  fi
else
  if [[ $# -lt 2 ]]; then
    print_usage
    exit 1
  fi
  INPUT="$1"
  OUTPUT_BVH="$2"
  OUTPUT_JSON="${3:-}"
  OUTPUT_CSV="${4:-}"
  OUTPUT_NPZ="${5:-}"
  OUTPUT_TRC="${6:-}"
  OUTPUT_FBX="${7:-}"
  if [[ -n "$OUTPUT_BVH" ]]; then
    OUTPUT_DIAG_JSON="${OUTPUT_BVH%.*}.diag.json"
  fi
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  recreate_venv
elif ! venv_python_supported "$VENV_DIR/bin/python"; then
  recreate_venv
elif ! "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
  recreate_venv
fi

source "$VENV_DIR/bin/activate"
"$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"

CMD=("$VENV_DIR/bin/python" "$ROOT_DIR/convert_video_to_bvh.py" --input "$INPUT" --output-bvh "$OUTPUT_BVH")

if [[ -n "$OUTPUT_JSON" ]]; then
  CMD+=(--output-json "$OUTPUT_JSON")
fi
if [[ -n "$OUTPUT_CSV" ]]; then
  CMD+=(--output-csv "$OUTPUT_CSV")
fi
if [[ -n "$OUTPUT_NPZ" ]]; then
  CMD+=(--output-npz "$OUTPUT_NPZ")
fi
if [[ -n "$OUTPUT_TRC" ]]; then
  CMD+=(--output-trc "$OUTPUT_TRC")
fi
if [[ -n "$OUTPUT_DIAG_JSON" ]]; then
  CMD+=(--output-diag-json "$OUTPUT_DIAG_JSON")
fi
if [[ -n "$OPENCV_ENHANCE" ]]; then
  CMD+=(--opencv-enhance "$OPENCV_ENHANCE")
fi
if [[ -n "$MAX_FRAME_SIDE" ]]; then
  CMD+=(--max-frame-side "$MAX_FRAME_SIDE")
fi
if [[ -n "$ROI_CROP" ]]; then
  CMD+=(--roi-crop "$ROI_CROP")
fi
if [[ -n "$ROOT_YAW_OFFSET_DEG" ]]; then
  CMD+=(--root-yaw-offset-deg "$ROOT_YAW_OFFSET_DEG")
fi
if [[ -n "$UPPER_ROTATION_OFFSET_DEG" ]]; then
  CMD+=(--upper-rotation-offset-deg "$UPPER_ROTATION_OFFSET_DEG")
fi
if [[ -n "$UPPER_BODY_ROTATION_SCALE" ]]; then
  CMD+=(--upper-body-rotation-scale "$UPPER_BODY_ROTATION_SCALE")
fi
if [[ -n "$ARM_ROTATION_SCALE" ]]; then
  CMD+=(--arm-rotation-scale "$ARM_ROTATION_SCALE")
fi
if [[ -n "$LOWER_BODY_ROTATION_MODE" ]]; then
  CMD+=(--lower-body-rotation-mode "$LOWER_BODY_ROTATION_MODE")
fi
if [[ -n "$LOOP_MODE" ]]; then
  CMD+=(--loop-mode "$LOOP_MODE")
fi
if [[ -n "$SKELETON_PROFILE_JSON" ]]; then
  CMD+=(--skeleton-profile-json "$SKELETON_PROFILE_JSON")
fi

"${CMD[@]}"

if [[ -n "$OUTPUT_FBX" ]]; then
  "$ROOT_DIR/bvh_to_fbx.sh" "$OUTPUT_BVH" "$OUTPUT_FBX"
fi
