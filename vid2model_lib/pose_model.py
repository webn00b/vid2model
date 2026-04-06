from __future__ import annotations

import urllib.request
from pathlib import Path


MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


MODEL_FILE_NAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_MODEL_FILE_NAME = "hand_landmarker.task"


def ensure_pose_model(model_complexity: int) -> Path:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILE_NAMES[model_complexity]

    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    tmp_path = model_path.with_suffix(model_path.suffix + ".download")
    urllib.request.urlretrieve(MODEL_URLS[model_complexity], tmp_path)
    tmp_path.replace(model_path)
    return model_path


def ensure_hand_model() -> Path:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / HAND_MODEL_FILE_NAME

    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    tmp_path = model_path.with_suffix(model_path.suffix + ".download")
    urllib.request.urlretrieve(HAND_MODEL_URL, tmp_path)
    tmp_path.replace(model_path)
    return model_path
