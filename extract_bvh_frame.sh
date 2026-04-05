#!/usr/bin/env bash
# Extract a single frame from BVH and create static BVH animation
# Usage: ./extract_bvh_frame.sh input.bvh output.bvh [frame_index] [duration_seconds]

set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'USAGE'
Extract a frame from BVH motion and create static pose animation.

Usage:
  ./extract_bvh_frame.sh <input_bvh> <output_bvh> [frame_index] [duration_seconds]

Examples:
  ./extract_bvh_frame.sh motion.bvh poklon.bvh                    # last frame
  ./extract_bvh_frame.sh motion.bvh poklon.bvh -1                 # last frame (explicit)
  ./extract_bvh_frame.sh motion.bvh poklon.bvh 120                # frame 120
  ./extract_bvh_frame.sh motion.bvh poklon.bvh -1 2               # last frame, 2 second duration

Default: extracts last frame as 1 second static animation

USAGE
  exit 1
fi

INPUT_BVH="$1"
OUTPUT_BVH="$2"
FRAME_IDX="${3:--1}"      # -1 means last frame
DURATION="${4:-1}"        # seconds

if [[ ! -f "$INPUT_BVH" ]]; then
  echo "Error: Input BVH not found: $INPUT_BVH" >&2
  exit 1
fi

# Extract header (skeleton definition)
echo "[extract_bvh_frame] Extracting frame $FRAME_IDX from: $INPUT_BVH"

# Count total frames
TOTAL_FRAMES=$(grep "^Frames:" "$INPUT_BVH" | awk '{print $2}')
if [[ -z "$TOTAL_FRAMES" ]]; then
  echo "Error: Could not parse frame count from BVH" >&2
  exit 1
fi

echo "  Total frames in input: $TOTAL_FRAMES"

# Resolve frame index
if [[ "$FRAME_IDX" == "-1" ]] || [[ "$FRAME_IDX" == "last" ]]; then
  FRAME_IDX=$((TOTAL_FRAMES - 1))
  echo "  Using last frame: $FRAME_IDX"
elif [[ "$FRAME_IDX" -lt 0 ]]; then
  FRAME_IDX=$((TOTAL_FRAMES + FRAME_IDX))
  echo "  Resolved frame index: $FRAME_IDX"
fi

if [[ $FRAME_IDX -lt 0 ]] || [[ $FRAME_IDX -ge $TOTAL_FRAMES ]]; then
  echo "Error: Frame index $FRAME_IDX out of range [0, $((TOTAL_FRAMES - 1))]" >&2
  exit 1
fi

# Get FPS from input BVH
FRAME_TIME=$(grep "^Frame Time:" "$INPUT_BVH" | awk '{print $3}')
if [[ -z "$FRAME_TIME" ]]; then
  echo "Error: Could not parse Frame Time from BVH" >&2
  exit 1
fi

# Calculate FPS (1 / frame_time) and repeat frames
# Use bc for floating point arithmetic
FPS=$(echo "scale=1; 1 / $FRAME_TIME" | bc)
REPEAT_FRAMES=$(echo "$FPS * $DURATION" | bc | cut -d. -f1)
if [[ -z "$REPEAT_FRAMES" ]] || [[ $REPEAT_FRAMES -lt 1 ]]; then
  REPEAT_FRAMES=1
fi

echo "  Frame time: $FRAME_TIME"
echo "  FPS: $FPS"
echo "  Output: $REPEAT_FRAMES frames = ${DURATION}s"

# Extract header and target frame
python3 << PYTHON_EOF
import sys
input_file = "$INPUT_BVH"
output_file = "$OUTPUT_BVH"
frame_idx = $FRAME_IDX
repeat_count = $REPEAT_FRAMES

with open(input_file, 'r') as f:
    lines = f.readlines()

# Find MOTION section
motion_idx = None
frames_idx = None
for i, line in enumerate(lines):
    if line.startswith('MOTION'):
        motion_idx = i
    elif line.startswith('Frames:'):
        frames_idx = i
        break

if motion_idx is None or frames_idx is None:
    print(f"Error: Could not find MOTION section in {input_file}", file=sys.stderr)
    sys.exit(1)

# Extract skeleton header
header_end = frames_idx + 2

# Extract the target frame
frame_data_start = header_end
target_frame_line = lines[frame_data_start + frame_idx]

# Build output BVH
output_lines = []
output_lines.extend(lines[:header_end])
output_lines[frames_idx] = f"Frames: {repeat_count}\n"

for _ in range(repeat_count):
    output_lines.append(target_frame_line)

with open(output_file, 'w') as f:
    f.writelines(output_lines)

print(f"[extract_bvh_frame] Wrote {repeat_count} frames to {output_file}")
PYTHON_EOF

echo ""
echo "[extract_bvh_frame] Done!"
echo "  Extracted frame: $FRAME_IDX"
echo "  Output BVH: $OUTPUT_BVH"
