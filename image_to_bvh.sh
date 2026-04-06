#!/usr/bin/env bash
# Convert a single image (pose) to BVH animation
# Usage: ./image_to_bvh.sh image.jpg output.bvh [duration_seconds]

set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'USAGE'
Convert a single image pose to static BVH animation.

Usage:
  ./image_to_bvh.sh <image> <output_bvh> [duration_seconds]

Examples:
  ./image_to_bvh.sh poklon.jpg output/poklon.bvh
  ./image_to_bvh.sh poklon.jpg output/poklon.bvh 3

The resulting BVH will contain a static pose repeated for the duration.
Default duration: 1 second (30 frames at default 30fps)

Requirements:
  - ffmpeg (for image to video conversion)

USAGE
  exit 1
fi

IMAGE="$1"
OUTPUT_BVH="$2"
DURATION="${3:-1}"

if [[ ! -f "$IMAGE" ]]; then
  echo "Error: Image file not found: $IMAGE" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg is required but not found" >&2
  echo "Install with: brew install ffmpeg" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMP_VIDEO="/tmp/$(basename "$IMAGE" | sed 's/\.[^.]*$/.mp4/')"
IMAGE_STEM=$(basename "$IMAGE" | sed 's/\.[^.]*$//')

echo "[image_to_bvh] Converting image to video..."
echo "  Image: $IMAGE"
echo "  Duration: ${DURATION}s at 30fps = $((30 * DURATION)) frames"
echo "  Temp video: $TEMP_VIDEO"

# Create video from image
# -loop 1: loop the image
# -t: duration in seconds
# -c:v libx264: video codec
# -pix_fmt yuv420p: pixel format (for compatibility)
ffmpeg -f lavfi -i color=c=black:s=1280x720:d="${DURATION}" \
  -i "$IMAGE" \
  -filter_complex "[1]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[img];[0][img]overlay=format=auto" \
  -c:v libx264 \
  -preset medium \
  -crf 23 \
  -pix_fmt yuv420p \
  -r 30 \
  -y \
  "$TEMP_VIDEO" 2>&1 | grep -v "^frame=" || true

echo ""
echo "[image_to_bvh] Running pose detection on video..."

# Run normal conversion pipeline
bash "$ROOT_DIR/convert_auto_fps.sh" "$TEMP_VIDEO" "$OUTPUT_BVH"

echo ""
echo "[image_to_bvh] Cleaning up..."
rm -f "$TEMP_VIDEO"

echo ""
echo "[image_to_bvh] Done!"
echo "  Static pose BVH: $OUTPUT_BVH"
echo ""
echo "To use in viewer:"
echo "  1. Open viewer at http://localhost:8080/viewer"
echo "  2. Load BVH: $OUTPUT_BVH"
echo "  3. Load VRM/GLB model"
echo "  4. Retarget animation"
