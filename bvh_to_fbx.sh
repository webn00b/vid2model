#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: ./bvh_to_fbx.sh <input_bvh> <output_fbx>"
  exit 1
fi

INPUT_BVH="$1"
OUTPUT_FBX="$2"

BLENDER="${BLENDER_BIN:-}"
if [[ -z "$BLENDER" ]]; then
  if command -v blender >/dev/null 2>&1; then
    BLENDER="$(command -v blender)"
  elif [[ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]]; then
    BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
  fi
fi

if [[ -z "$BLENDER" ]]; then
  echo "Blender not found. Install Blender or set BLENDER_BIN to blender executable path."
  echo "Example: BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender ./bvh_to_fbx.sh input.bvh output.fbx"
  exit 2
fi

"$BLENDER" -b --python "$ROOT_DIR/export_bvh_to_fbx_blender.py" -- "$INPUT_BVH" "$OUTPUT_FBX"
