#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 3 ]]; then
  echo "Usage: ./bvh_to_vrm.sh <input_bvh> <model.vrm> <output.vrm>"
  echo ""
  echo "Retargets BVH animation onto a VRM model using Blender + VRM Add-on."
  echo "Requires: Blender with VRM Add-on for Blender installed."
  exit 1
fi

INPUT_BVH="$1"
INPUT_VRM="$2"
OUTPUT_VRM="$3"

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
  echo "Example: BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender ./bvh_to_vrm.sh input.bvh model.vrm output.vrm"
  exit 2
fi

# Verify VRM addon is available
ADDON_CHECK=$("$BLENDER" -b --python-expr "
import addon_utils
vrm_addons = [m for m in addon_utils.modules() if 'vrm' in m.__name__.lower()]
print('VRM_ADDON_FOUND' if vrm_addons else 'VRM_ADDON_MISSING')
" 2>&1 | grep -o 'VRM_ADDON_\(FOUND\|MISSING\)' | tail -1)

if [[ "$ADDON_CHECK" == "VRM_ADDON_MISSING" ]]; then
  echo "VRM Add-on for Blender not found."
  echo "Install it from: https://vrm-addon-for-blender.info/"
  echo "Or via Blender: Edit -> Preferences -> Add-ons -> Install"
  exit 2
fi

"$BLENDER" -b --python "$ROOT_DIR/retarget_bvh_to_vrm_blender.py" -- "$INPUT_BVH" "$INPUT_VRM" "$OUTPUT_VRM"
