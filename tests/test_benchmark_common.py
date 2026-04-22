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
    render_latex_table,
    render_grouped_latex_table,
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

    def test_render_grouped_latex_table_renders_group_headers_and_compact_pairs(self) -> None:
        table = pd.DataFrame(
            {
                "method": ["Northrop-sliding-FGLS", "BB-sliding-FGLS"],
                "frechet_001": ["0.29 / 0.18", "0.28 / 0.19"],
                "moving_10": ["0.27 / 0.16", "0.26 / 0.15"],
            }
        )
        latex = render_grouped_latex_table(
            table,
            row_label="method",
            groups=[
                ("Fréchet max-AR", [("frechet_001", "0.01")]),
                ("Moving Maxima (q=99)", [("moving_10", "1.0")]),
            ],
            second_header_row_label=r"true $\xi$",
            second_header_row_label_raw=True,
            caption="Grouped EI summary",
            label="tab:test-grouped",
            fit_to_width=r"\textwidth",
            pair_medians_only=True,
        )

        self.assertIn(r"\caption{Grouped EI summary}", latex)
        self.assertIn(r"\label{tab:test-grouped}", latex)
        self.assertIn(r"\multicolumn{1}{c}{Fréchet max-AR}", latex)
        self.assertIn(r"\multicolumn{1}{c}{Moving Maxima (q=99)}", latex)
        self.assertIn(r"true $\xi$ & 0.01 & 1.0 \\", latex)
        self.assertIn(r"\shortstack[l]{Northrop- \\ sliding-FGLS}", latex)
        self.assertIn(r"\shortstack[c]{0.29 \\ 0.18}", latex)
        self.assertIn(r"\caption{Grouped EI summary}", latex)

    def test_render_latex_table_supports_raw_caption_and_header_latex(self) -> None:
        table = pd.DataFrame({"Application": ["Texas streamflow"], "$\\xi$ [range]": ["0.65 [0.59, 0.65]"]})

        latex = render_latex_table(
            table,
            caption=r"Appendix \(\xi\) summary",
            label="tab:test-raw",
            header_latex={"$\\xi$ [range]": r"$\xi$ [range]"},
            caption_raw=True,
        )

        self.assertIn(r"\caption{Appendix \(\xi\) summary}", latex)
        self.assertIn(r"Application & $\xi$ [range] \\", latex)
        self.assertNotIn(r"\$\textbackslash{}xi\$", latex)

    def test_render_grouped_latex_table_supports_raw_caption(self) -> None:
        table = pd.DataFrame({"method": ["A"], "scenario": ["0.10 / 0.20"]})

        latex = render_grouped_latex_table(
            table,
            row_label="method",
            groups=[("Group", [("scenario", "0.10")])],
            caption=r"Summary with \(\theta\) and \(\xi\)",
            label="tab:test-grouped-raw",
            pair_medians_only=True,
            caption_raw=True,
        )

        self.assertIn(r"\caption{Summary with \(\theta\) and \(\xi\)}", latex)


if __name__ == "__main__":
    unittest.main()
