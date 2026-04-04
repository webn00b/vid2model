#!/usr/bin/env bash
set -euo pipefail

# Sets up a separate Python environment for SMPL-based pose estimation.
# Uses 4D-Humans (HMR2.0) for accurate video → SMPL body parameters.
#
# Requirements:
#   - Python 3.10 (same as MediaPipe requirement)
#   - ~2GB disk for model weights
#
# After setup, use:
#   ./convert_video_smpl.sh video.mp4 output.bvh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SMPL_VENV="$ROOT_DIR/.venv-smpl"

echo "=== Setting up SMPL backend (4D-Humans / HMR2.0) ==="

# --- Find Python 3.10 ---
PYTHON_BIN=""
for candidate in python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    minor=$("$candidate" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || true)
    if [[ "$minor" == "10" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.10 required but not found." >&2
  echo "4D-Humans and its dependencies need Python 3.10." >&2
  echo "Install Python 3.10 or set PYTHON_BIN=python3.10" >&2
  exit 2
fi
echo "Using: $PYTHON_BIN ($(${PYTHON_BIN} --version))"

# --- Create venv ---
if [[ -d "$SMPL_VENV" ]]; then
  venv_minor=$("$SMPL_VENV/bin/python" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || true)
  if [[ "$venv_minor" != "10" ]]; then
    echo "Existing venv is Python 3.$venv_minor, recreating with 3.10 ..."
    rm -rf "$SMPL_VENV"
  fi
fi

if [[ ! -d "$SMPL_VENV" ]]; then
  echo "Creating venv at $SMPL_VENV ..."
  "$PYTHON_BIN" -m venv "$SMPL_VENV"
fi

source "$SMPL_VENV/bin/activate"
pip install --upgrade pip

# --- Install PyTorch (CPU/MPS for macOS) ---
echo "Installing PyTorch ..."
pip install torch torchvision

# --- Install 4D-Humans dependencies (skip chumpy, it's broken on modern Python) ---
echo "Installing 4D-Humans (HMR2.0) ..."
# Install chumpy first with workaround
pip install git+https://github.com/mattloper/chumpy.git || {
  echo "Warning: chumpy failed to install (optional, used for SMPL mesh rendering)"
  echo "Continuing without it..."
}
pip install git+https://github.com/shubham-goel/4D-Humans.git || {
  echo ""
  echo "4D-Humans install failed. Installing core dependencies manually..."
  pip install smplx==0.1.28 einops timm yacs pytorch-lightning
}

# --- Install additional dependencies ---
echo "Installing additional dependencies ..."
pip install numpy scipy opencv-python-headless

# --- Download SMPL model if not present ---
SMPL_DIR="$ROOT_DIR/models/smpl"
if [[ ! -f "$SMPL_DIR/SMPL_NEUTRAL.pkl" ]]; then
  echo ""
  echo "============================================"
  echo "  SMPL model files needed!"
  echo "============================================"
  echo ""
  echo "You need to download SMPL model files manually:"
  echo "  1. Go to https://smpl.is.tue.mpg.de/"
  echo "  2. Register and download 'SMPL for Python'"
  echo "  3. Extract and copy SMPL_NEUTRAL.pkl to:"
  echo "     $SMPL_DIR/SMPL_NEUTRAL.pkl"
  echo ""
  echo "After that, run this script again to verify."
  mkdir -p "$SMPL_DIR"
fi

echo ""
echo "=== Setup complete ==="
echo "SMPL venv: $SMPL_VENV"
echo ""
echo "Usage:"
echo "  ./convert_video_smpl.sh video.mp4 output.bvh"
