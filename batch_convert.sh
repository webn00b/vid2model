#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  ./batch_convert.sh <input_dir> <output_dir> [options]

Options:
  --with-json        Also write .json
  --with-csv         Also write .csv
  --with-npz         Also write .npz
  --with-trc         Also write .trc
  --with-fbx         Also write .fbx (requires Blender)
  --all-formats      Shortcut for: --with-json --with-csv --with-npz --with-trc
  --recursive        Scan input_dir recursively
  -h, --help         Show this help

Supported video extensions:
  .mp4 .mov .webm .avi .mkv .m4v (case-insensitive)
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

INPUT_DIR_RAW="$1"
OUTPUT_DIR="$2"
shift 2

WITH_JSON=0
WITH_CSV=0
WITH_NPZ=0
WITH_TRC=0
WITH_FBX=0
RECURSIVE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-json) WITH_JSON=1 ;;
    --with-csv) WITH_CSV=1 ;;
    --with-npz) WITH_NPZ=1 ;;
    --with-trc) WITH_TRC=1 ;;
    --with-fbx) WITH_FBX=1 ;;
    --all-formats)
      WITH_JSON=1
      WITH_CSV=1
      WITH_NPZ=1
      WITH_TRC=1
      ;;
    --recursive) RECURSIVE=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

if [[ ! -d "$INPUT_DIR_RAW" ]]; then
  echo "Input directory not found: $INPUT_DIR_RAW" >&2
  exit 2
fi
INPUT_DIR="$(cd "$INPUT_DIR_RAW" && pwd)"

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

if [[ "$RECURSIVE" -eq 1 ]]; then
  FIND_CMD=(find "$INPUT_DIR" -type f)
else
  FIND_CMD=(find "$INPUT_DIR" -maxdepth 1 -type f)
fi

VIDEO_FILES=()
while IFS= read -r -d '' file; do
  VIDEO_FILES+=("$file")
done < <(
  "${FIND_CMD[@]}" \( \
    -iname '*.mp4' -o \
    -iname '*.mov' -o \
    -iname '*.webm' -o \
    -iname '*.avi' -o \
    -iname '*.mkv' -o \
    -iname '*.m4v' \
  \) -print0 | sort -z
)

if [[ "${#VIDEO_FILES[@]}" -eq 0 ]]; then
  echo "No supported videos found in: $INPUT_DIR"
  exit 0
fi

SUCCESS=0
FAILED=0

for video in "${VIDEO_FILES[@]}"; do
  rel_path="${video#$INPUT_DIR/}"
  stem="${rel_path%.*}"
  out_base="$OUTPUT_DIR/$stem"
  out_dir="$(dirname "$out_base")"
  mkdir -p "$out_dir"

  out_bvh="$out_base.bvh"
  out_json=""
  out_csv=""
  out_npz=""
  out_trc=""
  out_fbx=""

  if [[ "$WITH_JSON" -eq 1 ]]; then out_json="$out_base.json"; fi
  if [[ "$WITH_CSV" -eq 1 ]]; then out_csv="$out_base.csv"; fi
  if [[ "$WITH_NPZ" -eq 1 ]]; then out_npz="$out_base.npz"; fi
  if [[ "$WITH_TRC" -eq 1 ]]; then out_trc="$out_base.trc"; fi
  if [[ "$WITH_FBX" -eq 1 ]]; then out_fbx="$out_base.fbx"; fi

  echo "[batch] Converting: $rel_path"
  if "$ROOT_DIR/convert.sh" "$video" "$out_bvh" "$out_json" "$out_csv" "$out_npz" "$out_trc" "$out_fbx"; then
    SUCCESS=$((SUCCESS + 1))
  else
    FAILED=$((FAILED + 1))
    echo "[batch] Failed: $rel_path" >&2
  fi
done

echo "[batch] Done. success=$SUCCESS failed=$FAILED total=${#VIDEO_FILES[@]}"
if [[ "$FAILED" -gt 0 ]]; then
  exit 3
fi
