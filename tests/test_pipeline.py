import unittest

import numpy as np

from vid2model_lib.pipeline import fill_pose_gaps


def pts(v: float) -> dict[str, np.ndarray]:
    return {"mid_hip": np.array([v, 0.0, 0.0], dtype=np.float64)}


class PipelineGapFillingTests(unittest.TestCase):
    def test_interpolates_short_gap(self) -> None:
        frames = [pts(0.0), None, pts(2.0)]
        filled, interpolated, carried = fill_pose_gaps(frames, max_gap_interpolate=2)
        self.assertEqual(len(filled), 3)
        self.assertEqual(interpolated, 1)
        self.assertEqual(carried, 0)
        self.assertAlmostEqual(float(filled[1]["mid_hip"][0]), 1.0, places=6)

    def test_carries_long_gap(self) -> None:
        frames = [pts(0.0), None, None, pts(3.0)]
        filled, interpolated, carried = fill_pose_gaps(frames, max_gap_interpolate=1)
        self.assertEqual(len(filled), 4)
        self.assertEqual(interpolated, 0)
        self.assertEqual(carried, 2)
        self.assertAlmostEqual(float(filled[1]["mid_hip"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(filled[2]["mid_hip"][0]), 0.0, places=6)

    def test_fills_leading_and_trailing_gaps(self) -> None:
        frames = [None, pts(1.0), None]
        filled, interpolated, carried = fill_pose_gaps(frames, max_gap_interpolate=2)
        self.assertEqual(len(filled), 3)
        self.assertEqual(interpolated, 0)
        self.assertEqual(carried, 2)
        self.assertAlmostEqual(float(filled[0]["mid_hip"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(filled[2]["mid_hip"][0]), 1.0, places=6)

    def test_raises_when_no_detections(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No detectable human pose frames found"):
            fill_pose_gaps([None, None], max_gap_interpolate=3)


if __name__ == "__main__":
    unittest.main()
