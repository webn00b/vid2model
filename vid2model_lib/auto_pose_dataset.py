from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .pipeline import _auto_feature_vector

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}


@dataclass(slots=True)
class AutoPoseExample:
    label: str
    source: str
    sample_count: int
    features: List[float]
    summary: Dict[str, float]
    meta: Dict[str, Any]


def build_auto_pose_example(
    samples: List[Dict[str, np.ndarray]],
    label: str,
    source: str,
    meta: Dict[str, Any] | None = None,
) -> AutoPoseExample:
    if not samples:
        raise ValueError("samples must not be empty")

    features, summary = _auto_feature_vector(samples)
    payload_meta = dict(meta or {})
    payload_meta.setdefault("feature_dim", int(features.size))
    return AutoPoseExample(
        label=str(label),
        source=str(source),
        sample_count=len(samples),
        features=[float(v) for v in features.tolist()],
        summary={k: float(v) for k, v in summary.items()},
        meta=payload_meta,
    )


def example_to_record(example: AutoPoseExample) -> Dict[str, Any]:
    record = asdict(example)
    record["schema"] = "vid2model.auto_pose_example.v1"
    return record


def append_examples_jsonl(output_path: Path, examples: Iterable[AutoPoseExample]) -> int:
    records = [json.dumps(example_to_record(example), ensure_ascii=False) for example in examples]
    if not records:
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        for line in records:
            fh.write(line)
            fh.write("\n")
    return len(records)


def discover_labeled_video_inputs(dataset_dir: Path, recursive: bool = True) -> List[Tuple[Path, str]]:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")

    items: List[Tuple[Path, str]] = []
    if recursive:
        for path in sorted(dataset_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            try:
                rel = path.relative_to(dataset_dir)
            except ValueError:
                continue
            if len(rel.parts) < 2:
                continue
            label = rel.parts[0]
            items.append((path, label))
    else:
        for label_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            for path in sorted(label_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                    items.append((path, label_dir.name))
    return items
