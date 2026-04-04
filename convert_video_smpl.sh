#!/usr/bin/env bash
set -euo pipefail

# End-to-end: Video → SMPL (4D-Humans) → BVH → optional VRM retarget.
#
# Usage:
#   ./convert_video_smpl.sh video.mp4 output.bvh
#   ./convert_video_smpl.sh video.mp4 output.bvh --vrm model.vrm output.vrm
#
# First run: ./setup_smpl_backend.sh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SMPL_VENV="$ROOT_DIR/.venv-smpl"
MAIN_VENV="$ROOT_DIR/.venv"

if [[ $# -lt 2 ]]; then
  echo "Usage: ./convert_video_smpl.sh <video> <output.bvh> [--vrm <model.vrm> <output.vrm>]"
  exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_BVH="$2"
shift 2

VRM_MODEL=""
OUTPUT_VRM=""
if [[ $# -ge 3 && "$1" == "--vrm" ]]; then
  VRM_MODEL="$2"
  OUTPUT_VRM="$3"
fi

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Video not found: $INPUT_VIDEO" >&2
  exit 1
fi

# --- Step 1: Video → SMPL params via 4D-Humans ---
SMPL_OUTPUT="${OUTPUT_BVH%.bvh}.smpl.npz"

if [[ ! -d "$SMPL_VENV" ]]; then
  echo "SMPL backend not set up. Run ./setup_smpl_backend.sh first."
  exit 2
fi

echo "=== Step 1: Video → SMPL (4D-Humans) ==="
"$SMPL_VENV/bin/python" "$ROOT_DIR/extract_smpl_from_video.py" \
  --input "$INPUT_VIDEO" \
  --output "$SMPL_OUTPUT"

# --- Step 2: SMPL → BVH ---
echo "=== Step 2: SMPL → BVH ==="

# Use main venv (has numpy/scipy) or smpl venv
PYTHON_BIN="$SMPL_VENV/bin/python"
if [[ -x "$MAIN_VENV/bin/python" ]]; then
  PYTHON_BIN="$MAIN_VENV/bin/python"
fi

"$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '$ROOT_DIR')
from vid2model_lib.smpl_to_bvh import load_smpl_output, smpl_poses_to_bvh_channels
from vid2model_lib.writers import write_bvh
from pathlib import Path
import numpy as np

poses, trans = load_smpl_output('$SMPL_OUTPUT')
print(f'Loaded SMPL: {poses.shape[0]} frames, {poses.shape[1]} values/frame')

motion, rest_offsets, fps = smpl_poses_to_bvh_channels(poses, trans)
print(f'Converted to BVH: {len(motion)} frames, {len(motion[0])} channels/frame')

write_bvh(Path('$OUTPUT_BVH'), fps, rest_offsets, motion)
print(f'Saved BVH: $OUTPUT_BVH')
"

# --- Step 3 (optional): BVH → VRM retarget ---
if [[ -n "$VRM_MODEL" && -n "$OUTPUT_VRM" ]]; then
  echo "=== Step 3: BVH → VRM retarget ==="
  "$ROOT_DIR/bvh_to_vrm.sh" "$OUTPUT_BVH" "$VRM_MODEL" "$OUTPUT_VRM"
fi

echo "=== Done ==="
