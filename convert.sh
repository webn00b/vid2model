#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-}"
MIN_PYTHON_MINOR=10
MAX_PYTHON_MINOR=12

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
  candidates+=(python3.12 python3.11 python3.10 python3)

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
  echo "No supported Python found (need 3.${MIN_PYTHON_MINOR}-3.${MAX_PYTHON_MINOR} for stable mediapipe wheels)." >&2
  echo "Install Python 3.10/3.11/3.12 or set PYTHON_BIN explicitly to a supported interpreter." >&2
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

"${CMD[@]}"

if [[ -n "$OUTPUT_FBX" ]]; then
  "$ROOT_DIR/bvh_to_fbx.sh" "$OUTPUT_BVH" "$OUTPUT_FBX"
fi
