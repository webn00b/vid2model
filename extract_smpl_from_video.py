#!/usr/bin/env python3
"""Extract SMPL body parameters from video using HMR2.0 (4D-Humans).

Outputs an NPZ file with:
    - smpl_poses: (N, 72) axis-angle rotations (24 joints x 3)
    - smpl_trans: (N, 3) root translation in meters
    - fps: float

Usage:
    python extract_smpl_from_video.py --input video.mp4 --output video.smpl.npz
"""
from __future__ import annotations

import argparse
import sys
import types

# Mock pyrender — not needed for inference, doesn't work headless on macOS.
_mock = types.ModuleType("pyrender")
class _FC:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return None
    def __getattr__(self, name): return _FC()
for _n in ["Node", "Mesh", "Scene", "Renderer", "OffscreenRenderer",
           "DirectionalLight", "MetallicRoughnessMaterial", "RenderFlags",
           "IntrinsicsCamera", "PerspectiveCamera", "PointLight", "SpotLight",
           "Light", "Viewer", "TextAlign", "GLTF", "trackball", "constants"]:
    setattr(_mock, _n, _FC)
_mock.RenderFlags = type("RenderFlags", (), {"RGBA": 0, "NONE": 0, "SHADOWS_DIRECTIONAL": 0, "ALL": 0})
sys.modules["pyrender"] = _mock

from pathlib import Path

import cv2
import numpy as np
import torch

# PyTorch 2.6+ changed weights_only default to True, which breaks omegaconf checkpoints.
# Monkey-patch torch.load to force weights_only=False for HMR2 checkpoint loading.
# HMR2 checkpoints come from the official 4D-Humans release — trusted source.
import torch as _torch
_orig_torch_load = _torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
_torch.load = _patched_torch_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SMPL from video via HMR2.0")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using Apple Silicon MPS acceleration")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


def rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix (3x3) to axis-angle (3,)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(rotmat).as_rotvec()


def extract(video_path: str, output_path: str) -> None:
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT

    device = get_device()

    print("Loading HMR2.0 model ...")
    model, cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device)
    model.eval()

    img_size = cfg.MODEL.IMAGE_SIZE  # 256
    focal_length = cfg.EXTRA.FOCAL_LENGTH  # 5000

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames at {fps:.1f} fps")

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    all_poses = []
    all_trans = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == 1:
            print(f"  Frame {frame_idx}/{total_frames}")

        # Preprocess: center crop to square, resize to img_size
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        img = img[y0:y0+side, x0:x0+side]
        img = cv2.resize(img, (img_size, img_size))

        tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model({"img": tensor})

        params = out["pred_smpl_params"]

        # global_orient: (1, 1, 3, 3) rotation matrix
        # body_pose: (1, 23, 3, 3) rotation matrices
        go = params["global_orient"][0, 0].cpu().numpy()  # (3, 3)
        bp = params["body_pose"][0].cpu().numpy()          # (23, 3, 3)

        # Convert rotation matrices to axis-angle
        go_aa = rotmat_to_axis_angle(go)  # (3,)
        bp_aa = np.array([rotmat_to_axis_angle(bp[j]) for j in range(23)])  # (23, 3)

        full_pose = np.concatenate([go_aa, bp_aa.flatten()])  # (72,)
        all_poses.append(full_pose)

        # Weak-perspective camera → translation
        pred_cam = out.get("pred_cam", torch.zeros(1, 3)).cpu().numpy()[0]
        s, tx, ty = pred_cam[0], pred_cam[1], pred_cam[2]
        tz = 2.0 * focal_length / (img_size * max(abs(s), 0.01))
        all_trans.append([tx, ty, tz])

    cap.release()

    poses = np.array(all_poses, dtype=np.float64)
    trans = np.array(all_trans, dtype=np.float64)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), smpl_poses=poses, smpl_trans=trans, fps=np.float64(fps))
    print(f"Saved: {out_path} ({poses.shape[0]} frames)")


def main() -> int:
    args = parse_args()
    if not Path(args.input).exists():
        print(f"Video not found: {args.input}", file=sys.stderr)
        return 1
    extract(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
