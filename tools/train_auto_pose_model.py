#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vid2model_lib.auto_pose_trainer import (
    load_auto_pose_jsonl,
    save_auto_pose_model,
    summarize_auto_pose_model,
    train_auto_pose_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact auto-pose classifier from JSONL examples.")
    parser.add_argument("--input", required=True, help="Path to auto_pose_dataset.jsonl")
    parser.add_argument(
        "--model-type",
        default="mlp",
        choices=("mlp", "centroid"),
        help="Classifier architecture to train.",
    )
    parser.add_argument("--hidden-size", type=int, default=16, help="Hidden layer size for mlp.")
    parser.add_argument("--epochs", type=int, default=800, help="Training epochs for mlp.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for mlp.")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization for mlp.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for mlp initialization.")
    parser.add_argument(
        "--output",
        default="models/auto_pose_model.npz",
        help="Output .npz model path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    data = load_auto_pose_jsonl(input_path)
    model = train_auto_pose_model(
        data,
        model_type=args.model_type,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    save_auto_pose_model(Path(args.output).expanduser(), model)
    summary = summarize_auto_pose_model(model)
    print(
        f"[vid2model] trained auto model type={summary['model_type']} "
        f"classes={summary['classes']} counts={summary['class_counts']} feature_dim={summary['feature_dim']}"
    )
    print(f"[vid2model] wrote model to {Path(args.output).expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
