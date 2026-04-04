from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class RegressionScenario:
    name: str
    description: str
    signals: tuple[str, ...]
    command: tuple[str, ...]


REGRESSION_SCENARIOS: tuple[RegressionScenario, ...] = (
    RegressionScenario(
        name="source_pipeline_diagnostics",
        description="Source-stage diagnostics distinguish risky cleanup/finalize behavior from healthy source motion.",
        signals=(
            "diagnostics.source_stages.pose.flags.source_pipeline_risk",
            "diagnostics.source_stages.motion.flags.source_motion_risk",
            "quality.retarget_risk",
        ),
        command=(
            "python3",
            "-m",
            "pytest",
            "tests/test_pipeline_modules.py",
            "-q",
            "-k",
            "quality_summary_flags_risky_tracking or reports_source_stages_in_diagnostics or "
            "source_motion_stage_diagnostics_flags_finalize_spikes or "
            "source_motion_stage_diagnostics_ignores_wraparound_steps",
        ),
    ),
    RegressionScenario(
        name="root_yaw_contract",
        description="Viewer root-yaw candidate policy must not reintroduce large source flips for clips already centered by Python export.",
        signals=(
            "viewer.retarget-root-yaw.sourceYawCandidatePolicy.allowSourceFlipCandidates",
            "viewer.retarget-root-yaw.sourceClipYawSummary.looksCentered",
        ),
        command=("node", "--test", "tests/root-yaw-contract.test.mjs"),
    ),
    RegressionScenario(
        name="atypical_model_mapping",
        description="Atypical rigs should prefer primary bones over helper/socket duplicates when choosing canonical matches.",
        signals=(
            "viewer.mapping.canonicalBonePreferenceScore",
            "viewer.mapping.buildCanonicalBoneMap",
            "viewer.mapping.topologyFallback",
        ),
        command=("node", "--test", "tests/atypical-rig-mapping.test.mjs"),
    ),
    RegressionScenario(
        name="headless_retarget_validation",
        description="Headless GLB/VRM validation should emit stable machine-readable retarget selection plus canonical-motion comparison diagnostics outside the browser viewer.",
        signals=(
            "format=vid2model.headless-retarget.v1",
            "selection.selectedModeLabel",
            "diagnostics.events.retarget-summary",
            "canonicalComparison.summary.poseError",
        ),
        command=("node", "--test", "tests/headless-retarget-validation.test.mjs"),
    ),
)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the focused vid2model regression pass for known problematic source and retarget scenarios."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Scenario name to run. Repeat to run multiple specific scenarios.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args(list(argv))


def _select_scenarios(names: Sequence[str] | None) -> list[RegressionScenario]:
    if not names:
        return list(REGRESSION_SCENARIOS)
    requested = {str(name).strip() for name in names if str(name).strip()}
    selected = [scenario for scenario in REGRESSION_SCENARIOS if scenario.name in requested]
    missing = sorted(requested.difference({scenario.name for scenario in selected}))
    if missing:
        raise ValueError(f"Unknown regression scenario(s): {', '.join(missing)}")
    return selected


def _print_scenarios(scenarios: Sequence[RegressionScenario]) -> None:
    for scenario in scenarios:
        print(f"{scenario.name}: {scenario.description}")
        print(f"  signals: {', '.join(scenario.signals)}")
        print(f"  command: {' '.join(scenario.command)}")


def run_regression_checks(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or ())
    scenarios = _select_scenarios(args.scenarios)

    if args.list:
        _print_scenarios(scenarios)
        return 0

    root = Path(__file__).resolve().parent.parent
    failures = 0
    for scenario in scenarios:
        print(f"[regression] {scenario.name}")
        print(f"  description: {scenario.description}")
        print(f"  signals: {', '.join(scenario.signals)}")
        print(f"  command: {' '.join(scenario.command)}")
        if args.dry_run:
            continue
        completed = subprocess.run(scenario.command, cwd=root)
        if completed.returncode != 0:
            failures += 1
            print(f"[regression] failed: {scenario.name}", file=sys.stderr)

    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_regression_checks(argv)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
