from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


@dataclass(slots=True)
class AutoPoseTrainingData:
    labels: np.ndarray
    features: np.ndarray
    sources: List[str]
    records: List[Dict[str, Any]]


def load_auto_pose_jsonl(path: Path) -> AutoPoseTrainingData:
    labels: List[str] = []
    features: List[List[float]] = []
    sources: List[str] = []
    records: List[Dict[str, Any]] = []

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        record = json.loads(text)
        if record.get("schema") != "vid2model.auto_pose_example.v1":
            raise ValueError(f"Unsupported schema on line {line_no}: {record.get('schema')!r}")
        labels.append(str(record["label"]))
        features.append([float(v) for v in record["features"]])
        sources.append(str(record.get("source", "")))
        records.append(record)

    if not records:
        raise ValueError("No examples found in dataset")

    feature_lengths = {len(row) for row in features}
    if len(feature_lengths) != 1:
        raise ValueError("All examples must have the same feature dimension")

    return AutoPoseTrainingData(
        labels=np.asarray(labels, dtype=object),
        features=np.asarray(features, dtype=np.float64),
        sources=sources,
        records=records,
    )


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0, ddof=0)
    return np.where(std < 1e-8, 1.0, std)


def train_auto_pose_centroid_model(data: AutoPoseTrainingData) -> Dict[str, np.ndarray]:
    x = np.asarray(data.features, dtype=np.float64)
    labels = np.asarray(data.labels, dtype=object)
    classes = sorted({str(label) for label in labels})
    if len(classes) < 2:
        raise ValueError("Need at least two classes to train a classifier")

    feature_mean = x.mean(axis=0)
    feature_scale = _safe_std(x)
    z = (x - feature_mean) / feature_scale

    class_counts = Counter(str(label) for label in labels)
    rows: List[np.ndarray] = []
    biases: List[float] = []
    for class_name in classes:
        mask = labels == class_name
        class_points = z[mask]
        centroid = class_points.mean(axis=0)
        rows.append(centroid)
        prior = class_counts[class_name] / float(len(labels))
        biases.append(float(-0.5 * np.dot(centroid, centroid) + np.log(prior)))

    model = {
        "classes": np.asarray(classes, dtype=object),
        "feature_mean": np.asarray(feature_mean, dtype=np.float64),
        "feature_scale": np.asarray(feature_scale, dtype=np.float64),
        "W": np.asarray(rows, dtype=np.float64),
        "b": np.asarray(biases, dtype=np.float64),
        "train_count": np.asarray(len(labels), dtype=np.int32),
        "class_counts": np.asarray([class_counts[c] for c in classes], dtype=np.int32),
    }
    return model


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def train_auto_pose_mlp_model(
    data: AutoPoseTrainingData,
    hidden_size: int = 16,
    epochs: int = 800,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    x = np.asarray(data.features, dtype=np.float64)
    labels = np.asarray(data.labels, dtype=object)
    classes = sorted({str(label) for label in labels})
    if len(classes) < 2:
        raise ValueError("Need at least two classes to train a classifier")
    if hidden_size < 2:
        raise ValueError("hidden_size must be >= 2")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if l2 < 0.0:
        raise ValueError("l2 must be >= 0")

    feature_mean = x.mean(axis=0)
    feature_scale = _safe_std(x)
    z = (x - feature_mean) / feature_scale

    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    y = np.asarray([class_to_idx[str(label)] for label in labels], dtype=np.int64)
    n_samples, n_features = z.shape
    n_classes = len(classes)
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, 0.05, size=(hidden_size, n_features))
    b1 = np.zeros(hidden_size, dtype=np.float64)
    W2 = rng.normal(0.0, 0.05, size=(n_classes, hidden_size))
    b2 = np.zeros(n_classes, dtype=np.float64)

    y_onehot = np.eye(n_classes, dtype=np.float64)[y]
    for _ in range(epochs):
        z1 = z @ W1.T + b1
        h = np.tanh(z1)
        logits = h @ W2.T + b2
        probs = _softmax(logits)

        grad_logits = (probs - y_onehot) / float(n_samples)
        grad_W2 = grad_logits.T @ h + l2 * W2
        grad_b2 = grad_logits.sum(axis=0)
        grad_h = grad_logits @ W2
        grad_z1 = grad_h * (1.0 - h * h)
        grad_W1 = grad_z1.T @ z + l2 * W1
        grad_b1 = grad_z1.sum(axis=0)

        W2 -= learning_rate * grad_W2
        b2 -= learning_rate * grad_b2
        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1

    class_counts = Counter(str(label) for label in labels)
    model = {
        "classes": np.asarray(classes, dtype=object),
        "feature_mean": np.asarray(feature_mean, dtype=np.float64),
        "feature_scale": np.asarray(feature_scale, dtype=np.float64),
        "W1": np.asarray(W1, dtype=np.float64),
        "b1": np.asarray(b1, dtype=np.float64),
        "W2": np.asarray(W2, dtype=np.float64),
        "b2": np.asarray(b2, dtype=np.float64),
        "train_count": np.asarray(len(labels), dtype=np.int32),
        "class_counts": np.asarray([class_counts[c] for c in classes], dtype=np.int32),
    }
    return model


def train_auto_pose_model(
    data: AutoPoseTrainingData,
    model_type: str = "mlp",
    hidden_size: int = 16,
    epochs: int = 800,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    model_type = str(model_type).strip().lower()
    if model_type == "centroid":
        return train_auto_pose_centroid_model(data)
    if model_type == "mlp":
        return train_auto_pose_mlp_model(
            data,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )
    raise ValueError("model_type must be one of: centroid, mlp")


def save_auto_pose_model(output_path: Path, model: Dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **model)


def summarize_auto_pose_model(model: Dict[str, np.ndarray]) -> Dict[str, Any]:
    classes = [str(v) for v in np.asarray(model["classes"], dtype=object).tolist()]
    counts = [int(v) for v in np.asarray(model["class_counts"], dtype=np.int32).tolist()]
    feature_dim = int(np.asarray(model["feature_mean"]).shape[0]) if "feature_mean" in model else int(np.asarray(model["W"]).shape[1])
    return {
        "classes": classes,
        "class_counts": counts,
        "train_count": int(np.asarray(model["train_count"]).item()),
        "feature_dim": feature_dim,
        "model_type": "mlp" if "W1" in model else "centroid",
    }
