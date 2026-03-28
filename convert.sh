#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
  else
    PYTHON_BIN="python3"
  fi
fi

recreate_venv() {
  if [[ -d "$VENV_DIR" ]]; then
    local backup_dir="${VENV_DIR}.broken.$(date +%s)"
    mv "$VENV_DIR" "$backup_dir"
    echo "Existing venv looks broken, moved to: $backup_dir"
  fi
  "$PYTHON_BIN" -m venv "$VENV_DIR"
}

if [[ $# -lt 2 ]]; then
  echo "Usage: ./convert.sh <input_video> <output_bvh> [output_json] [output_csv] [output_npz] [output_trc] [output_fbx]"
  exit 1
fi

INPUT="$1"
OUTPUT_BVH="$2"
OUTPUT_JSON="${3:-}"
OUTPUT_CSV="${4:-}"
OUTPUT_NPZ="${5:-}"
OUTPUT_TRC="${6:-}"
OUTPUT_FBX="${7:-}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
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

"${CMD[@]}"

if [[ -n "$OUTPUT_FBX" ]]; then
  "$ROOT_DIR/bvh_to_fbx.sh" "$OUTPUT_BVH" "$OUTPUT_FBX"
fi
