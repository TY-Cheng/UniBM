from __future__ import annotations
# ruff: noqa: E402

import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from data_prep.ghcn import PreparedSeries
from application.build import (
    ApplicationPreparedInputs,
    ApplicationSpec,
    build_application_bundle,
)
from application.outputs import (
    _draw_ei_ax,
    _draw_return_levels_ax,
    _seasonal_adjusted_ei_method_rows,
    application_ei_method_rows,
    application_method_rows,
    seasonal_monthly_pit_unit_frechet,
    write_application_figures,
)
from benchmark.design import fit_methods_for_series


def _make_prepared(series: pd.Series, *, name: str, provider: str, role: str) -> PreparedSeries:
    annual_maxima = series.groupby(series.index.year).max()
    return PreparedSeries(
        name=name,
        value_name="value",
        series=series,
        annual_maxima=annual_maxima,
        metadata={"provider": provider, "series_role": role},
    )


def _make_standard_bundle() -> object:
    rng = np.random.default_rng(7)
    index = pd.date_range("2000-01-01", periods=365 * 24, freq="D")
    values = pd.Series(rng.pareto(2.5, index.size) + 1.0, index=index, dtype=float)
    prepared = _make_prepared(values, name="synthetic", provider="ghcn", role="shared")
    inputs = ApplicationPreparedInputs(display=prepared, evi=prepared, ei=prepared)
    spec = ApplicationSpec(
        key="synthetic",
        provider="ghcn",
        label="Synthetic",
        figure_stem="synthetic",
        raw_key="none",
        ylabel="value",
        time_series_title="Synthetic series",
        scaling_title="Synthetic scaling",
        scaling_ylabel="log median block maximum",
        target_stability_title="Synthetic target stability",
    )
    return build_application_bundle(spec, inputs)


def _make_evi_only_bundle() -> object:
    rng = np.random.default_rng(13)
    index = pd.date_range("2001-01-01", periods=365 * 16, freq="D")
    values = pd.Series(rng.gamma(shape=2.2, scale=3.0, size=index.size), index=index, dtype=float)
    prepared = _make_prepared(values, name="evi_only", provider="ghcn", role="shared")
    inputs = ApplicationPreparedInputs(display=prepared, evi=prepared, ei=prepared)
    spec = ApplicationSpec(
        key="synthetic_evi_only",
        provider="ghcn",
        label="Synthetic EVI-only",
        figure_stem="synthetic_evi_only",
        raw_key="none",
        ylabel="value",
        time_series_title="Synthetic EVI-only series",
        scaling_title="Synthetic EVI-only scaling",
        scaling_ylabel="log median block maximum",
        target_stability_title="Synthetic EVI-only target stability",
        formal_ei=False,
    )
    return build_application_bundle(spec, inputs)


def _make_nfip_bundle() -> object:
    rng = np.random.default_rng(11)
    index = pd.date_range("2005-01-01", periods=365 * 20, freq="D")
    zero_filled = pd.Series(0.0, index=index, dtype=float)
    active_mask = rng.random(index.size) < 0.07
    zero_filled.loc[active_mask] = (rng.pareto(1.8, active_mask.sum()) + 1.0) * 1000.0
    positive_only = zero_filled[zero_filled > 0.0]
    inputs = ApplicationPreparedInputs(
        display=_make_prepared(zero_filled, name="NFIP display", provider="fema", role="display"),
        evi=_make_prepared(
            positive_only,
            name="NFIP active-day payouts",
            provider="fema",
            role="evi",
        ),
        ei=_make_prepared(zero_filled, name="NFIP payout waves", provider="fema", role="ei"),
    )
    spec = ApplicationSpec(
        key="synthetic_nfip",
        provider="fema",
        label="Synthetic NFIP",
        figure_stem="synthetic_nfip",
        raw_key="none",
        ylabel="usd",
        time_series_title="Synthetic NFIP",
        scaling_title="Synthetic NFIP scaling",
        scaling_ylabel="log median positive payouts",
        return_level_basis="claim_active_day",
        return_level_label="claim-active-day return period (years)",
        target_stability_title="Synthetic NFIP target stability",
        ei_allow_zeros=True,
    )
    return build_application_bundle(spec, inputs)


class ApplicationOutputTests(unittest.TestCase):
    def test_application_method_rows_default_to_headline_median_only(self) -> None:
        bundle = _make_standard_bundle()

        rows = application_method_rows(bundle)

        self.assertEqual([row["method"] for row in rows], ["sliding_median_fgls"])

    def test_fit_methods_for_series_can_restrict_to_headline_method(self) -> None:
        bundle = _make_standard_bundle()

        fits = fit_methods_for_series(
            bundle.prepared.evi.series.values,
            quantile=bundle.spec.quantile,
            random_state=7,
            method_ids=("sliding_median_fgls",),
        )

        self.assertEqual(list(fits), ["sliding_median_fgls"])

    def test_application_ei_method_rows_include_four_methods(self) -> None:
        bundle = _make_standard_bundle()

        rows = application_ei_method_rows(bundle)

        self.assertEqual(
            {row["method"] for row in rows},
            {
                "bb_sliding_fgls",
                "northrop_sliding_fgls",
                "k_gaps",
                "ferro_segers",
            },
        )

    def test_application_ei_method_rows_skip_evi_only_applications(self) -> None:
        bundle = _make_evi_only_bundle()

        self.assertEqual(application_ei_method_rows(bundle), [])

    def test_seasonal_monthly_pit_unit_frechet_is_deterministic_and_positive(self) -> None:
        index = pd.date_range("2000-01-01", periods=365 * 4, freq="D")
        values = pd.Series(0.0, index=index, dtype=float)
        values.iloc[::5] = 2.0
        values.iloc[1::11] = 7.0

        transformed_a = seasonal_monthly_pit_unit_frechet(values)
        transformed_b = seasonal_monthly_pit_unit_frechet(values)

        self.assertTrue(transformed_a.index.equals(values.index))
        self.assertTrue(
            np.allclose(transformed_a.to_numpy(), transformed_b.to_numpy(), equal_nan=True)
        )
        finite = transformed_a.to_numpy(dtype=float)[
            np.isfinite(transformed_a.to_numpy(dtype=float))
        ]
        self.assertTrue(np.all(finite > 0.0))

    def test_seasonal_adjusted_rows_include_four_methods(self) -> None:
        bundle = _make_standard_bundle()

        rows = _seasonal_adjusted_ei_method_rows(bundle)

        self.assertEqual(
            {row["method"] for row in rows},
            {
                "bb_sliding_fgls",
                "northrop_sliding_fgls",
                "k_gaps",
                "ferro_segers",
            },
        )
        self.assertTrue(all(row["transform"] == "monthly_pit_unit_frechet" for row in rows))

    def test_seasonal_adjusted_rows_skip_evi_only_applications(self) -> None:
        bundle = _make_evi_only_bundle()

        self.assertEqual(_seasonal_adjusted_ei_method_rows(bundle), [])

    def test_write_application_figures_writes_composite_pdf(self) -> None:
        bundle = _make_standard_bundle()

        with tempfile.TemporaryDirectory() as tmpdir:
            fig_dir = Path(tmpdir)
            write_application_figures(bundle, fig_dir)
            composite_path = fig_dir / "application_composite_synthetic.pdf"
            self.assertTrue(composite_path.exists())
            self.assertGreater(composite_path.stat().st_size, 0)

    def test_write_application_figures_omits_ei_pdf_for_evi_only_applications(self) -> None:
        bundle = _make_evi_only_bundle()

        with tempfile.TemporaryDirectory() as tmpdir:
            fig_dir = Path(tmpdir)
            write_application_figures(bundle, fig_dir)
            self.assertFalse((fig_dir / "application_ei_synthetic_evi_only.pdf").exists())
            self.assertTrue((fig_dir / "application_composite_synthetic_evi_only.pdf").exists())

    def test_ei_panel_labels_identical_windows_separately(self) -> None:
        bundle = _make_standard_bundle()
        fig, ax = plt.subplots()
        try:
            _draw_ei_ax(ax, bundle)
            _, labels = ax.get_legend_handles_labels()
            self.assertIn("BB stable window", labels)
            self.assertIn("Northrop stable window", labels)
            self.assertNotIn("shared BB/Northrop stable window", labels)
        finally:
            plt.close(fig)

    def test_nfip_return_level_panel_omits_ei_adjusted_curve(self) -> None:
        bundle = _make_nfip_bundle()
        fig, ax = plt.subplots()
        try:
            _draw_return_levels_ax(ax, bundle)
            labels = [line.get_label() for line in ax.lines]
            self.assertEqual(labels, ["UniBM return level"])
        finally:
            plt.close(fig)

    def test_evi_only_return_level_panel_omits_ei_adjusted_curve(self) -> None:
        bundle = _make_evi_only_bundle()
        fig, ax = plt.subplots()
        try:
            _draw_return_levels_ax(ax, bundle)
            labels = [line.get_label() for line in ax.lines]
            self.assertEqual(labels, ["UniBM return level"])
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
