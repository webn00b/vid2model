#!/usr/bin/env python3
"""Process Slovo gesture videos with real finger tracking."""

import argparse
import csv
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process SLOVO sign language videos through the vid2model pipeline."
    )
    parser.add_argument(
        "--sign",
        default=None,
        help="Exact sign text to process (e.g. 'Привет!'). Default: process all signs.",
    )
    parser.add_argument(
        "--sign-pattern",
        default=None,
        help="Regex pattern to filter signs (e.g. '[АБВ]', 'Привет.*'). Matched against sign text.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Max number of videos to process (default: 3, 0 = unlimited).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for BVH files (default: output/slovo_<sign_slug>).",
    )
    parser.add_argument(
        "--slovo-dir",
        default="slovo-all/slovo",
        help="Path to the SLOVO dataset directory (default: slovo-all/slovo).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1). Use 4+ for batch processing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process videos whose .bvh output already exists (default: skip them).",
    )
    parser.add_argument(
        "--min-hand-coverage",
        type=float,
        default=0.3,
        help=(
            "Skip videos where detected hand frames / total frames < this ratio (default: 0.3). "
            "Requires slovo_mediapipe.json in --slovo-dir parent."
        ),
    )
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable quality screening via slovo_mediapipe.json.",
    )
    parser.add_argument(
        "--upper-body-rotation-scale",
        default="0.3",
        help="UPPER_BODY_ROTATION_SCALE env var (default: 0.3).",
    )
    parser.add_argument(
        "--arm-rotation-scale",
        default="1.0",
        help="ARM_ROTATION_SCALE env var (default: 1.0).",
    )
    parser.add_argument(
        "--root-yaw-offset-deg",
        default="90",
        help="ROOT_YAW_OFFSET_DEG env var (default: 90).",
    )
    parser.add_argument(
        "--max-frame-side",
        default="640",
        help="MAX_FRAME_SIDE for resize (default: 640, 0 = disabled).",
    )
    parser.add_argument(
        "--no-gesture-preset",
        action="store_true",
        help="Disable gesture-optimized defaults (lower-body off, ROI crop, resize).",
    )
    return parser.parse_args()


def _sign_to_slug(text: str) -> str:
    slug = re.sub(r"[^\w\u0400-\u04ff]", "_", text).strip("_").lower()
    return slug or "sign"


def _load_mediapipe_index(slovo_dir: Path) -> dict | None:
    """Load slovo_mediapipe.json. Returns dict {video_id: frames} or None if not found."""
    json_path = slovo_dir.parent / "slovo_mediapipe.json"
    if not json_path.exists():
        return None
    print(f"Loading quality index from {json_path} …", flush=True)
    with open(json_path) as f:
        return json.load(f)


def _hand_coverage(frames: list) -> float:
    """Fraction of frames that have at least one detected hand."""
    if not frames:
        return 0.0
    detected = sum(1 for frame in frames if frame)
    return detected / len(frames)


def _passes_quality(video_id: str, mp_index: dict | None, min_coverage: float) -> bool:
    if mp_index is None:
        return True
    frames = mp_index.get(video_id)
    if frames is None:
        return True
    if len(frames) == 0:
        return False
    return _hand_coverage(frames) >= min_coverage


class Manifest:
    """Thread-safe manifest writer — appends one JSON entry per video to output_dir/manifest.json."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._entries: list[dict] = []
        # Load existing entries so resume preserves history
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._entries = data
            except Exception:
                pass

    def record(self, entry: dict) -> None:
        with self._lock:
            self._entries.append(entry)
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)
            tmp.replace(self._path)

    def ids(self) -> set[str]:
        """Return set of video IDs already in manifest."""
        with self._lock:
            return {e["id"] for e in self._entries}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _process_video(
    item: dict,
    output_dir: Path,
    env_vars: dict,
    idx: int,
    total: int,
    manifest: Manifest,
    force: bool,
) -> tuple[bool, str]:
    """Run convert.sh on a single video. Returns (success, label)."""
    video_path = item["video_path"]
    sign_slug = _sign_to_slug(item["text"])
    sign_dir = output_dir / sign_slug
    sign_dir.mkdir(parents=True, exist_ok=True)
    output_path = sign_dir / f"{item['id'][:8]}.bvh"
    label = f"[{idx}/{total}] {item['text']!r} {item['id'][:8]}"

    if not video_path.exists():
        manifest.record({"id": item["id"], "sign": item["text"], "status": "missing", "bvh": None})
        return False, f"{label}  ⚠ video not found"

    # Resume: skip if output already exists and --force not set
    if not force and output_path.exists() and output_path.stat().st_size > 0:
        manifest.record({
            "id": item["id"], "sign": item["text"],
            "status": "skipped", "bvh": str(output_path),
        })
        return True, f"{label}  ↩ already done"

    cmd = ["env"] + [f"{k}={v}" for k, v in env_vars.items()] + [
        "./convert.sh",
        str(video_path),
        str(output_path),
        "--hand-tracking", "auto",
    ]

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        manifest.record({
            "id": item["id"], "sign": item["text"],
            "status": "failed", "bvh": None, "elapsed_s": round(elapsed, 1),
        })
        return False, f"{label}  ✗ failed ({elapsed:.0f}s)"

    bvh_size = output_path.stat().st_size if output_path.exists() else 0
    manifest.record({
        "id": item["id"], "sign": item["text"],
        "status": "ok", "bvh": str(output_path),
        "bvh_bytes": bvh_size, "elapsed_s": round(elapsed, 1),
    })
    return True, f"{label}  ✓ {output_path.name} ({elapsed:.0f}s)"


def main():
    args = parse_args()

    slovo_dir = Path(args.slovo_dir)
    annotations_file = slovo_dir / "annotations.csv"
    train_dir = slovo_dir / "train"
    test_dir = slovo_dir / "test"

    if not annotations_file.exists():
        print(f"Error: {annotations_file} not found", file=sys.stderr)
        sys.exit(1)

    # Load quality index
    mp_index = None
    if not args.no_quality_filter:
        mp_index = _load_mediapipe_index(slovo_dir)
        if mp_index is None:
            print("Quality index not found — skipping quality filter.", flush=True)

    # Compile pattern
    pattern = None
    if args.sign_pattern:
        try:
            pattern = re.compile(args.sign_pattern, re.UNICODE)
        except re.error as e:
            print(f"Error: invalid --sign-pattern: {e}", file=sys.stderr)
            sys.exit(1)

    # Parse annotations and filter
    candidates = []
    skipped_quality = 0

    with open(annotations_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row["text"]
            is_train = row["train"] == "True"

            if args.split == "train" and not is_train:
                continue
            if args.split == "test" and is_train:
                continue
            if args.sign and text != args.sign:
                continue
            if pattern and not pattern.search(text):
                continue

            video_id = row["attachment_id"]

            if not _passes_quality(video_id, mp_index, args.min_hand_coverage):
                skipped_quality += 1
                continue

            videos_dir = train_dir if is_train else test_dir
            candidates.append({
                "id": video_id,
                "text": text,
                "length": int(float(row["length"])),
                "video_path": videos_dir / f"{video_id}.mp4",
            })

    if skipped_quality:
        print(f"Quality filter: skipped {skipped_quality} low-coverage videos.")

    if not candidates:
        print("No matching videos found.")
        sys.exit(0)

    limit = args.count if args.count > 0 else len(candidates)
    selected = candidates[:limit]

    # Determine output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.sign:
            slug = _sign_to_slug(args.sign)
        elif args.sign_pattern:
            slug = _sign_to_slug(args.sign_pattern)
        else:
            slug = "all"
        output_dir = Path("output") / f"slovo_{slug}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest — used for resume detection
    manifest = Manifest(output_dir / "manifest.json")
    already_done = manifest.ids()

    # Apply resume filter if not forced
    if not args.force and already_done:
        before = len(selected)
        selected = [v for v in selected if v["id"] not in already_done]
        skipped_resume = before - len(selected)
        if skipped_resume:
            print(f"Resume: skipping {skipped_resume} already-manifest videos.")

    signs_in_selected = sorted({v["text"] for v in selected})
    print(f"Found {len(candidates)} candidates, processing {len(selected)}")
    if len(signs_in_selected) <= 10:
        print(f"Signs: {', '.join(signs_in_selected)}")
    print(f"Output: {output_dir}  workers={args.workers}  manifest: {output_dir}/manifest.json\n")

    # Build env overrides
    env_vars = {
        "OPENCV_ENHANCE": "light",
        "UPPER_BODY_ROTATION_SCALE": args.upper_body_rotation_scale,
        "ARM_ROTATION_SCALE": args.arm_rotation_scale,
        "ROOT_YAW_OFFSET_DEG": args.root_yaw_offset_deg,
    }
    if not args.no_gesture_preset:
        env_vars["LOWER_BODY_ROTATION_MODE"] = "off"
        env_vars["ROI_CROP"] = "auto"
        env_vars["MAX_FRAME_SIDE"] = args.max_frame_side
        env_vars["LOOP_MODE"] = "off"

    total = len(selected)
    success = 0
    skipped = 0
    failed = 0

    def _handle_result(ok: bool, msg: str) -> None:
        nonlocal success, skipped, failed
        print(msg, flush=True)
        if "↩" in msg:
            skipped += 1
        elif ok:
            success += 1
        else:
            failed += 1

    if args.workers <= 1:
        for i, item in enumerate(selected, 1):
            ok, msg = _process_video(item, output_dir, env_vars, i, total, manifest, args.force)
            _handle_result(ok, msg)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_process_video, item, output_dir, env_vars, i, total, manifest, args.force): i
                for i, item in enumerate(selected, 1)
            }
            for fut in as_completed(futures):
                ok, msg = fut.result()
                _handle_result(ok, msg)

    print(
        f"\nDone: {success} converted, {skipped} skipped (resume), {failed} failed"
        f"  →  {output_dir}/manifest.json  ({len(manifest)} total entries)"
    )


if __name__ == "__main__":
    main()
