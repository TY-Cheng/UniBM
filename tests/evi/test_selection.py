from __future__ import annotations

import unittest

import numpy as np

from unibm.evi._regression import _fit_linear_model
from unibm.evi.models import PlateauWindow
from unibm.evi.selection import select_penultimate_window


def _baseline_select_penultimate_window(
    log_block_sizes: np.ndarray,
    log_values: np.ndarray,
    *,
    min_points: int = 5,
    trim_fraction: float = 0.15,
    curvature_penalty: float = 2.0,
) -> PlateauWindow:
    x = np.asarray(log_block_sizes, dtype=float)
    y = np.asarray(log_values, dtype=float)
    n = x.size
    if n < min_points:
        raise ValueError("Not enough positive block summaries to select a plateau.")
    lo = int(np.floor(n * trim_fraction))
    hi = n - lo
    lo = min(lo, max(n - min_points, 0))
    if hi - lo < min_points:
        lo = 0
        hi = n
    best: tuple[float, int, int] | None = None
    for start in range(lo, hi - min_points + 1):
        for stop in range(start + min_points, hi + 1):
            model = _fit_linear_model(x[start:stop], y[start:stop])
            resid = y[start:stop] - model["fitted"]
            mse = float(np.mean(resid**2))
            slopes = np.diff(y[start:stop]) / np.diff(x[start:stop])
            curvature = float(np.mean(np.abs(np.diff(slopes)))) if slopes.size > 1 else 0.0
            score = (mse + float(curvature_penalty) * curvature) / np.sqrt(stop - start)
            if best is None or score < best[0]:
                best = (score, start, stop)
    assert best is not None
    _, start, stop = best
    mask = np.zeros(n, dtype=bool)
    mask[start:stop] = True
    return PlateauWindow(
        start=start,
        stop=stop,
        score=float(best[0]),
        mask=mask,
        x=x[start:stop],
        y=y[start:stop],
    )


class EviSelectionTests(unittest.TestCase):
    def test_select_penultimate_window_matches_baseline(self) -> None:
        x = np.log(np.array([4.0, 8.0, 16.0, 32.0, 64.0]))
        y = np.log(np.array([2.0, 2.5, 3.2, 4.0, 5.0]))
        plateau = select_penultimate_window(x, y, min_points=3)
        baseline = _baseline_select_penultimate_window(x, y, min_points=3)
        self.assertEqual((plateau.start, plateau.stop), (baseline.start, baseline.stop))
        self.assertAlmostEqual(plateau.score, baseline.score)
        np.testing.assert_array_equal(plateau.mask, baseline.mask)

        reset_plateau = select_penultimate_window(x, y, min_points=5, trim_fraction=0.49)
        self.assertEqual(reset_plateau.start, 0)
        self.assertEqual(reset_plateau.stop, 5)

    def test_select_penultimate_window_validates_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "Not enough positive block summaries"):
            select_penultimate_window(np.array([1.0, 2.0]), np.array([1.0, 2.0]), min_points=3)


if __name__ == "__main__":
    unittest.main()
