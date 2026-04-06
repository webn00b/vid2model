#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


MANIFEST_FORMAT = "vid2model.rig-profile-manifest.v1"
PROFILE_FORMAT = "vid2model.rig-profile.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register an exported rig profile in viewer/rig-profiles/index.json."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to vid2model.rig-profile.v1 JSON file exported from the viewer.",
    )
    parser.add_argument(
        "--rig-profiles-dir",
        default="viewer/rig-profiles",
        help="Target directory that contains index.json.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Optional output filename override, e.g. moon-girl.body.rig-profile.json",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_exported_rig_profile(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    if payload.get("format") != PROFILE_FORMAT:
        raise ValueError(f"Unsupported rig profile format: {payload.get('format')!r}")
    profile = payload.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("Rig profile export must contain object field 'profile'")
    model_fingerprint = str(payload.get("modelFingerprint") or profile.get("modelFingerprint") or "").strip()
    stage = str(payload.get("stage") or profile.get("stage") or "").strip().lower()
    if not model_fingerprint:
        raise ValueError("Rig profile export is missing modelFingerprint")
    if not stage:
        raise ValueError("Rig profile export is missing stage")
    return payload


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"format": MANIFEST_FORMAT, "profiles": []}
    payload = _load_json(path)
    if payload.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"Unsupported manifest format: {payload.get('format')!r}")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("Manifest field 'profiles' must be an array")
    return payload


def build_output_filename(payload: Dict[str, Any], override: str = "") -> str:
    if override:
        return Path(override).name
    model_label = str(payload.get("modelLabel") or "model").strip()
    stage = str(payload.get("stage") or payload.get("profile", {}).get("stage") or "body").strip().lower()
    base = model_label.replace(" ", "-").replace("/", "-").replace("\\", "-")
    if "." in base:
        base = base.rsplit(".", 1)[0]
    base = "".join(ch for ch in base if ch.isalnum() or ch in {"-", "_"}).strip("-_") or "model"
    return f"{base}.{stage}.rig-profile.json"


def register_rig_profile(
    *,
    input_path: Path,
    rig_profiles_dir: Path,
    output_name: str = "",
) -> Tuple[Path, Path, Dict[str, Any]]:
    payload = load_exported_rig_profile(input_path)
    rig_profiles_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = rig_profiles_dir / "index.json"
    manifest = load_manifest(manifest_path)

    output_filename = build_output_filename(payload, output_name)
    output_path = rig_profiles_dir / output_filename
    shutil.copyfile(input_path, output_path)

    model_fingerprint = str(payload.get("modelFingerprint") or payload["profile"].get("modelFingerprint") or "").strip()
    stage = str(payload.get("stage") or payload["profile"].get("stage") or "").strip().lower()
    model_label = str(payload.get("modelLabel") or payload["profile"].get("modelLabel") or "").strip()
    entry = {
        "modelFingerprint": model_fingerprint,
        "modelLabel": model_label,
        "stage": stage,
        "path": f"./{output_filename}",
    }

    existing = [
        item
        for item in manifest["profiles"]
        if not (
            str(item.get("modelFingerprint") or "").strip() == model_fingerprint
            and str(item.get("stage") or "").strip().lower() == stage
        )
    ]
    existing.append(entry)
    existing.sort(key=lambda item: (str(item.get("modelLabel") or ""), str(item.get("stage") or ""), str(item.get("modelFingerprint") or "")))
    manifest["profiles"] = existing
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path, manifest_path, entry


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    rig_profiles_dir = Path(args.rig_profiles_dir).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Rig profile export not found: {input_path}")
    output_path, manifest_path, entry = register_rig_profile(
        input_path=input_path,
        rig_profiles_dir=rig_profiles_dir,
        output_name=args.name,
    )
    print(
        f"[vid2model] rig profile registered model={entry['modelFingerprint']} stage={entry['stage']} "
        f"profile={output_path} manifest={manifest_path}",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
