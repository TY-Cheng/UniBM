from __future__ import annotations

import unittest
import warnings

import numpy as np

from scripts.unibm.summaries import estimate_sample_mode, summarize_block_maxima


class UniBmSummariesTests(unittest.TestCase):
    def test_estimate_sample_mode_warns_and_excludes_nonpositive_values(self) -> None:
        sample = np.array([-1.0, 0.0, 1.0, 1.1, 1.2, 2.5], dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mode = estimate_sample_mode(sample, warn=True)
        self.assertTrue(np.isfinite(mode))
        self.assertEqual(len(caught), 1)
        self.assertIn("excluded 2 non-positive observations", str(caught[0].message))

    def test_estimate_sample_mode_handles_empty_and_singleton_inputs(self) -> None:
        self.assertTrue(np.isnan(estimate_sample_mode([], warn=False)))
        self.assertEqual(estimate_sample_mode([3.5], warn=False), 3.5)

    def test_summarize_block_maxima_supports_all_targets(self) -> None:
        maxima = np.array([1.0, 2.0, 4.0, 8.0], dtype=float)
        self.assertAlmostEqual(
            summarize_block_maxima(maxima, target="quantile", quantile=0.5),
            float(np.quantile(maxima, 0.5, method="median_unbiased")),
        )
        self.assertAlmostEqual(summarize_block_maxima(maxima, target="mean"), 3.75)
        self.assertTrue(np.isfinite(summarize_block_maxima(maxima, target="mode")))

    def test_summarize_block_maxima_rejects_unknown_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported target"):
            summarize_block_maxima([1.0, 2.0], target="median-ish")


if __name__ == "__main__":
    unittest.main()
