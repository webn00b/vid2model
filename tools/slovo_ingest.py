#!/usr/bin/env python3
"""Process Slovo gesture videos with real finger tracking."""

import csv
import subprocess
import sys
from pathlib import Path

def main():
    slovo_dir = Path("slovo-all/slovo")
    annotations_file = slovo_dir / "annotations.csv"
    videos_dir = slovo_dir / "train"
    output_dir = Path("output/slovo_greetings")

    if not annotations_file.exists():
        print(f"Error: {annotations_file} not found", file=sys.stderr)
        sys.exit(1)

    if not videos_dir.exists():
        print(f"Error: {videos_dir} not found", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse annotations and find greeting videos
    greetings = []
    with open(annotations_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['text'] == 'Привет!':
                greetings.append({
                    'id': row['attachment_id'],
                    'user': row['user_id'],
                    'length': int(float(row['length'])),
                    'train': row['train'] == 'True'
                })

    print(f"Found {len(greetings)} greeting videos")

    # Process first 3
    for i, greeting in enumerate(greetings[:3]):
        video_id = greeting['id']
        video_path = videos_dir / f"{video_id}.mp4"
        output_path = output_dir / f"privet_{i+1:02d}.bvh"

        if not video_path.exists():
            print(f"⚠ Video not found: {video_path}")
            continue

        print(f"\n[{i+1}/3] Processing: {video_id}")
        print(f"  Input: {video_path}")
        print(f"  Output: {output_path}")

        cmd = [
            "./convert.sh",
            str(video_path),
            str(output_path),
            "--hand-tracking", "auto"
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"✗ Failed to process {video_id}", file=sys.stderr)
        else:
            print(f"✓ Saved: {output_path}")

    print(f"\n✓ Done! Check output/{output_dir}")

if __name__ == "__main__":
    main()
