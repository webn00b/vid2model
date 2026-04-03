import json
import tempfile
import unittest
from pathlib import Path

from tools.register_rig_profile import (
    PROFILE_FORMAT,
    build_output_filename,
    load_exported_rig_profile,
    register_rig_profile,
)


class RegisterRigProfileTests(unittest.TestCase):
    def _write_export(self, root: Path, *, model_label: str = "MoonGirl.vrm", model_fingerprint: str = "rig:abcd1234", stage: str = "body") -> Path:
        path = root / "export.json"
        path.write_text(
            json.dumps(
                {
                    "format": PROFILE_FORMAT,
                    "modelLabel": model_label,
                    "modelFingerprint": model_fingerprint,
                    "stage": stage,
                    "profile": {
                        "modelLabel": model_label,
                        "modelFingerprint": model_fingerprint,
                        "stage": stage,
                        "validationStatus": "validated",
                        "namesTargetToSource": {"hips": "hips"},
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_build_output_filename_uses_model_label_and_stage(self) -> None:
        filename = build_output_filename(
            {
                "modelLabel": "MoonGirl.vrm",
                "stage": "body",
            }
        )
        self.assertEqual(filename, "MoonGirl.body.rig-profile.json")

    def test_load_exported_rig_profile_validates_format(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            export = self._write_export(root)
            payload = load_exported_rig_profile(export)
            self.assertEqual(payload["modelFingerprint"], "rig:abcd1234")

    def test_register_rig_profile_copies_file_and_updates_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            export = self._write_export(root)
            rig_profiles_dir = root / "viewer" / "rig-profiles"

            output_path, manifest_path, entry = register_rig_profile(
                input_path=export,
                rig_profiles_dir=rig_profiles_dir,
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["profiles"]), 1)
            self.assertEqual(manifest["profiles"][0]["modelFingerprint"], "rig:abcd1234")
            self.assertEqual(entry["path"], "./MoonGirl.body.rig-profile.json")

    def test_register_rig_profile_replaces_existing_entry_for_same_model_and_stage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rig_profiles_dir = root / "viewer" / "rig-profiles"
            export_a = self._write_export(root, model_label="MoonGirl.vrm", model_fingerprint="rig:abcd1234", stage="body")
            export_b = self._write_export(root, model_label="MoonGirl.vrm", model_fingerprint="rig:abcd1234", stage="body")

            register_rig_profile(input_path=export_a, rig_profiles_dir=rig_profiles_dir, output_name="first.json")
            register_rig_profile(input_path=export_b, rig_profiles_dir=rig_profiles_dir, output_name="second.json")

            manifest = json.loads((rig_profiles_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["profiles"]), 1)
            self.assertEqual(manifest["profiles"][0]["path"], "./second.json")


if __name__ == "__main__":
    unittest.main()
