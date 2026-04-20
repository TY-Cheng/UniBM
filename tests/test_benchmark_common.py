from __future__ import annotations
# ruff: noqa: E402

import unittest

import numpy as np
import pandas as pd

try:
    from . import _path_setup as test_paths
except ImportError:  # pragma: no cover
    import _path_setup as test_paths

test_paths.ensure_repo_import_paths()

from benchmark.common import (
    add_wilson_bounds,
    interval_contains,
    panel_metric_ylim,
    round_up_metric_upper,
    wilson_interval,
)


class BenchmarkCommonTests(unittest.TestCase):
    def test_interval_contains_uses_closed_bounds(self) -> None:
        self.assertTrue(interval_contains((1.0, 2.0), 1.0))
        self.assertTrue(interval_contains((1.0, 2.0), 2.0))
        self.assertFalse(interval_contains((1.0, 2.0), 2.1))

    def test_add_wilson_bounds_matches_scalar_helper(self) -> None:
        frame = pd.DataFrame({"n_cover": [0, 5, 9], "n_rep": [0, 10, 10]})
        observed = add_wilson_bounds(frame.copy(), success_col="n_cover", total_col="n_rep")
        expected = np.asarray(
            [wilson_interval(row.n_cover, row.n_rep) for row in frame.itertuples(index=False)],
            dtype=float,
        )
        np.testing.assert_allclose(
            observed[["coverage_lo", "coverage_hi"]].to_numpy(dtype=float),
            expected,
            equal_nan=True,
        )

    def test_round_up_metric_upper_uses_display_steps(self) -> None:
        steps = {"ape": (1.05, 1.25, 1.5)}
        self.assertEqual(round_up_metric_upper("ape", 1.02, upper_steps=steps), 1.05)
        self.assertEqual(round_up_metric_upper("ape", 1.24, upper_steps=steps), 1.5)

    def test_panel_metric_ylim_uses_upper_column_when_available(self) -> None:
        frame = pd.DataFrame(
            {
                "method": ["baseline", "headline"],
                "ape_median": [0.80, 0.95],
                "ape_q25": [0.70, 0.85],
                "ape_q75": [1.01, 1.20],
            }
        )
        ylim = panel_metric_ylim(
            frame,
            metric="ape",
            methods=["headline"],
            metric_columns={"ape": ("ape_median", "ape_q25", "ape_q75")},
            upper_steps={"ape": (1.05, 1.25, 1.5)},
        )
        self.assertEqual(ylim, (0.0, 1.25))

    def test_panel_metric_ylim_falls_back_to_center_column(self) -> None:
        frame = pd.DataFrame({"method": ["headline"], "mape": [0.91]})
        ylim = panel_metric_ylim(
            frame,
            metric="mape",
            methods=["headline"],
            metric_columns={"mape": ("mape", None, None)},
            upper_steps={"mape": (0.5, 1.0, 1.5)},
        )
        self.assertEqual(ylim, (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
