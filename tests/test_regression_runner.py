import io
import unittest
from contextlib import redirect_stdout

from tools.run_regression_checks import REGRESSION_SCENARIOS, main, run_regression_checks


class RegressionRunnerTests(unittest.TestCase):
    def test_regression_scenarios_cover_source_yaw_and_atypical_mapping(self) -> None:
        scenario_names = {scenario.name for scenario in REGRESSION_SCENARIOS}
        self.assertEqual(
            scenario_names,
            {
                "source_pipeline_diagnostics",
                "root_yaw_contract",
                "atypical_model_mapping",
                "headless_retarget_validation",
            },
        )

    def test_list_mode_prints_scenario_descriptions_and_signals(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            rc = run_regression_checks(["--list"])

        text = stream.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("source_pipeline_diagnostics", text)
        self.assertIn("root_yaw_contract", text)
        self.assertIn("atypical_model_mapping", text)
        self.assertIn("headless_retarget_validation", text)
        self.assertIn("signals:", text)
        self.assertIn("command:", text)

    def test_dry_run_accepts_single_scenario_filter(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            rc = run_regression_checks(["--scenario", "root_yaw_contract", "--dry-run"])

        text = stream.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("[regression] root_yaw_contract", text)
        self.assertNotIn("[regression] source_pipeline_diagnostics", text)
        self.assertIn("node --test tests/root-yaw-contract.test.mjs", text)

    def test_unknown_scenario_returns_error(self) -> None:
        rc = main(["--scenario", "missing_case", "--dry-run"])
        self.assertEqual(rc, 2)
