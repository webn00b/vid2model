#!/usr/bin/env bash
# Auto-detect video FPS and generate BVH with correct timing
# Usage: ./convert_auto_fps.sh <input_video> [output_bvh]

set -euo pipefail

if [[ $# -lt 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
  echo "Usage: $0 <input_video> [output_bvh] [extra_args...]"
  echo ""
  echo "Auto-detects video FPS and generates BVH with correct timing."
  echo ""
  echo "Examples:"
  echo "  $0 video.mp4"
  echo "  $0 video.mp4 output/video.bvh"
  echo "  $0 video.mp4 output/video.bvh --all --fbx"
  echo ""
  echo "Extra args are passed to convert.sh (--all, --fbx, --vrm, etc.)"
  exit 1
fi

INPUT="$1"
# Generate default output name from input filename
INPUT_STEM=$(basename "$INPUT")
INPUT_STEM="${INPUT_STEM%.*}"
OUTPUT="${2:-output/${INPUT_STEM}.bvh}"
shift 2 || true

if [[ ! -f "$INPUT" ]]; then
  echo "Error: Input file not found: $INPUT" >&2
  exit 1
fi

# Try to detect FPS from video using ffprobe
detect_fps() {
  local video="$1"

  # Try ffprobe first
  if command -v ffprobe >/dev/null 2>&1; then
    local fps_frac
    fps_frac=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
      -of default=noprint_wrappers=1:nokey=1 "$video" 2>/dev/null || echo "")

    if [[ -n "$fps_frac" ]]; then
      # Handle fraction format (e.g., "60/1" -> "60")
      if [[ "$fps_frac" == *"/"* ]]; then
        local numerator="${fps_frac%/*}"
        local denominator="${fps_frac#*/}"
        echo "scale=2; $numerator / $denominator" | bc 2>/dev/null || echo "$numerator"
      else
        echo "$fps_frac"
      fi
      return 0
    fi
  fi

  # Fallback: try ffmpeg info
  if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -hide_banner -i "$video" 2>&1 | grep -oP '(?<=, )[0-9.]+(?= fps)' | head -1
    return 0
  fi

  # If all else fails, return empty (will use default 30)
  return 1
}

echo "[convert_auto_fps] Detecting FPS from: $INPUT"
DETECTED_FPS=$(detect_fps "$INPUT" || echo "")

if [[ -z "$DETECTED_FPS" ]]; then
  echo "[convert_auto_fps] Warning: Could not detect FPS, using default 30" >&2
  DETECTED_FPS="30"
else
  # Round to nearest integer for cleaner display
  DETECTED_FPS=$(printf "%.0f" "$DETECTED_FPS")
fi

echo "[convert_auto_fps] Using FPS: $DETECTED_FPS"
echo "[convert_auto_fps] Output: $OUTPUT"
echo ""

# Call convert.sh with detected FPS
OVERRIDE_FPS="$DETECTED_FPS" bash "$(dirname "$0")/convert.sh" "$INPUT" "$OUTPUT" "$@"
