from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.workflows.application_screening import screen_extreme_series


class ApplicationScreeningTests(unittest.TestCase):
    def test_screen_extreme_series_accepts_lightweight_bootstrap_override(self) -> None:
        rs = np.random.default_rng(19)
        index = pd.date_range("2000-01-01", periods=365 * 25, freq="D")
        values = pd.Series(rs.pareto(2.2, index.size) + 1.0, index=index, dtype=float)

        review = screen_extreme_series(
            values,
            name="synthetic_screening",
            bootstrap_reps=8,
        )

        self.assertGreater(review.n_obs, 0)
        self.assertTrue(np.isfinite(review.xi_hat))
        self.assertGreaterEqual(review.plateau_points, 0)


if __name__ == "__main__":
    unittest.main()
