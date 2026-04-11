from __future__ import annotations
# ruff: noqa: E402

import sys
import tempfile
import unittest
from unittest import mock
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_prep.ghcn import PreparedSeries
from application.build import (
    ApplicationPreparedInputs,
    ApplicationSpec,
    build_application_bundle,
)
from application.outputs import (
    _application_observations_per_year,
    _application_design_life_level_rows,
    _plot_daily_and_annual,
    _tau_scaling_views_for_fit,
    _draw_ei_ax,
    _draw_design_life_levels_ax,
    _seasonal_adjusted_ei_method_rows,
    application_case_audit_table,
    application_ei_method_rows,
    application_design_life_level_table,
    application_method_rows,
    application_selection_sensitivity_table,
    seasonal_monthly_pit_unit_frechet,
    write_application_figures,
)
from application.specs import APPLICATION_DESIGN_LIFE_TAUS, APPLICATION_RANDOM_STATE
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
        time_series_annual_max_yscale="log",
        design_life_level_basis="claim_active_day",
        design_life_level_label="claim-active-day design-life (years)",
        target_stability_title="Synthetic NFIP target stability",
        ei_allow_zeros=True,
    )
    return build_application_bundle(spec, inputs)


class ApplicationOutputTests(unittest.TestCase):
    def test_application_method_rows_default_to_headline_median_only(self) -> None:
        bundle = _make_standard_bundle()

        rows = application_method_rows(bundle)

        self.assertEqual({row["method"] for row in rows}, {"sliding_median_fgls"})
        self.assertEqual({row["tau"] for row in rows}, set(APPLICATION_DESIGN_LIFE_TAUS))
        self.assertIn("one_year_design_life_level", rows[0])
        self.assertIn("ten_year_design_life_level", rows[0])
        self.assertIn("design_life_level_basis", rows[0])
        self.assertTrue(all(row["shared_xi"] for row in rows))
        self.assertTrue(any(row["is_headline_tau"] for row in rows))

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

    def test_application_case_audit_table_tracks_curated_subset(self) -> None:
        base_stream_bundle = _make_standard_bundle()
        stream_bundle = replace(
            base_stream_bundle,
            spec=replace(base_stream_bundle.spec, key="tx_streamflow", label="Texas streamflow"),
        )
        base_nfip_bundle = _make_nfip_bundle()
        nfip_bundle = replace(
            base_nfip_bundle,
            spec=replace(base_nfip_bundle.spec, key="tx_nfip_claims", label="Texas NFIP claims"),
        )

        table = application_case_audit_table([nfip_bundle, stream_bundle])

        self.assertEqual(
            table.columns.tolist(),
            [
                "Application",
                "Observation clock",
                "Record span",
                "Preprocessing",
                "Normalization",
                "Stationarity caveat",
            ],
        )
        self.assertEqual(table["Application"].tolist(), ["Texas streamflow", "Texas NFIP claims"])
        self.assertIn("calendar-day", table.iloc[0]["Observation clock"])
        self.assertIn("not exposure-normalized portfolio risk", table.iloc[1]["Normalization"])

    def test_application_selection_sensitivity_table_formats_headline_ranges(self) -> None:
        base_nfip_bundle = _make_nfip_bundle()
        nfip_bundle = replace(
            base_nfip_bundle,
            spec=replace(base_nfip_bundle.spec, key="tx_nfip_claims", label="Texas NFIP claims"),
        )
        fake_ei_variants = [(nfip_bundle.ei_bb_sliding_fgls, 0.1)] * 3
        with (
            mock.patch(
                "application.outputs._fit_evi_window_variants",
                return_value=[nfip_bundle.evi_fit, nfip_bundle.evi_fit, nfip_bundle.evi_fit],
            ),
            mock.patch(
                "application.outputs._fit_ei_window_variants",
                return_value=fake_ei_variants,
            ),
        ):
            table = application_selection_sensitivity_table([nfip_bundle])

        self.assertEqual(table.shape[0], 1)
        self.assertEqual(table.iloc[0]["Application"], "Texas NFIP claims")
        self.assertIn("[", table.iloc[0]["$\\xi$ headline [range]"])
        self.assertIn("[", table.iloc[0]["$\\theta$ headline [range]"])
        self.assertIn("10y median DLL", table.columns.tolist()[3])

    def test_tau_scaling_views_for_fit(self) -> None:
        bundle = _make_standard_bundle()

        views = _tau_scaling_views_for_fit(bundle.prepared.evi.series, bundle.evi_fit)

        self.assertEqual([view.tau for view in views], list(APPLICATION_DESIGN_LIFE_TAUS))
        slopes = [view.slope for view in views]
        self.assertTrue(all(np.isclose(slope, bundle.evi_fit.slope) for slope in slopes))
        headline = next(view for view in views if view.headline)
        self.assertTrue(np.isclose(headline.intercept, bundle.evi_fit.intercept))
        self.assertTrue(
            any(not np.isclose(view.intercept, headline.intercept) for view in views[1:])
        )

    def test_application_tau_design_life_levels_are_monotone_in_tau(self) -> None:
        bundle = _make_standard_bundle()

        rows = pd.DataFrame(_application_design_life_level_rows(bundle))
        for years in sorted(rows["design_life_years"].unique()):
            sub = rows.loc[rows["design_life_years"] == years].sort_values("tau")
            values = sub["design_life_level"].to_numpy(dtype=float)
            self.assertTrue(np.all(np.diff(values) > 0.0))

    def test_application_design_life_level_table_contains_multiple_tau_rows(self) -> None:
        bundle = _make_standard_bundle()

        table = application_design_life_level_table([bundle])

        self.assertEqual(
            table["$\\tau$"].tolist(), [f"{tau:.2f}" for tau in APPLICATION_DESIGN_LIFE_TAUS]
        )
        self.assertEqual(table.shape[0], len(APPLICATION_DESIGN_LIFE_TAUS))

    def test_application_method_rows_stay_aligned_when_two_quantile_methods_are_enabled(
        self,
    ) -> None:
        bundle = _make_standard_bundle()

        with mock.patch(
            "application.outputs.APPLICATION_EVI_METHOD_IDS",
            ("sliding_median_fgls", "disjoint_median_ols"),
        ):
            rows = application_method_rows(bundle)

        frame = pd.DataFrame(rows)
        observations_per_year = _application_observations_per_year(bundle)
        self.assertEqual(
            set(frame["method"]),
            {"sliding_median_fgls", "disjoint_median_ols"},
        )
        for method in ("sliding_median_fgls", "disjoint_median_ols"):
            method_rows = frame.loc[frame["method"] == method].sort_values("tau")
            self.assertEqual(set(method_rows["tau"]), set(APPLICATION_DESIGN_LIFE_TAUS))
            fit = fit_methods_for_series(
                bundle.prepared.evi.series.values,
                quantile=bundle.spec.quantile,
                random_state=APPLICATION_RANDOM_STATE,
                method_ids=(method,),
                reuse_fits={"sliding_median_fgls": bundle.evi_fit},
            )[method]
            expected_views = _tau_scaling_views_for_fit(bundle.prepared.evi.series, fit)
            for view in expected_views:
                row = method_rows.loc[np.isclose(method_rows["tau"], view.tau)].iloc[0]
                expected_one_year, expected_ten_year = view.design_life_levels(
                    np.asarray([1.0, 10.0]),
                    observations_per_year=observations_per_year,
                )
                self.assertAlmostEqual(row["xi_hat"], fit.slope)
                self.assertAlmostEqual(row["one_year_design_life_level"], expected_one_year)
                self.assertAlmostEqual(row["ten_year_design_life_level"], expected_ten_year)

    def test_application_method_rows_degrade_nonquantile_methods_to_nan_tau(self) -> None:
        bundle = _make_standard_bundle()

        with mock.patch(
            "application.outputs.APPLICATION_EVI_METHOD_IDS",
            ("sliding_median_fgls", "sliding_mean_fgls"),
        ):
            rows = application_method_rows(bundle)

        frame = pd.DataFrame(rows)
        quantile_rows = frame.loc[frame["method"] == "sliding_median_fgls"]
        mean_rows = frame.loc[frame["method"] == "sliding_mean_fgls"]
        self.assertEqual(set(quantile_rows["tau"]), set(APPLICATION_DESIGN_LIFE_TAUS))
        self.assertEqual(len(mean_rows), 1)
        self.assertTrue(pd.isna(mean_rows.iloc[0]["tau"]))
        self.assertFalse(bool(mean_rows.iloc[0]["shared_xi"]))
        self.assertTrue(np.isnan(mean_rows.iloc[0]["one_year_design_life_level"]))
        self.assertTrue(np.isnan(mean_rows.iloc[0]["ten_year_design_life_level"]))

    def test_write_application_figures_writes_composite_pdf(self) -> None:
        bundle = _make_standard_bundle()

        with tempfile.TemporaryDirectory() as tmpdir:
            fig_dir = Path(tmpdir)
            write_application_figures(bundle, fig_dir)
            composite_path = fig_dir / "application_composite_synthetic.pdf"
            self.assertTrue(composite_path.exists())
            self.assertGreater(composite_path.stat().st_size, 0)

    def test_write_application_figures_use_design_life_filename(self) -> None:
        bundle = _make_standard_bundle()

        with tempfile.TemporaryDirectory() as tmpdir:
            fig_dir = Path(tmpdir)
            write_application_figures(bundle, fig_dir)
            design_life_path = fig_dir / "application_design_life_synthetic.pdf"
            legacy_rl_path = fig_dir / "application_rl_synthetic.pdf"
            legacy_design_life_alias_path = fig_dir / design_life_path.name.replace(
                "design_life",
                "h" + "q",
            )
            self.assertTrue(design_life_path.exists())
            self.assertGreater(design_life_path.stat().st_size, 0)
            self.assertFalse(legacy_rl_path.exists())
            self.assertFalse(legacy_design_life_alias_path.exists())

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
            legend = ax.get_legend()
            assert legend is not None
            labels = [text.get_text() for text in legend.get_texts()]
            self.assertIn("BB stable window", labels)
            self.assertIn("Northrop stable window", labels)
            self.assertNotIn("shared BB/Northrop stable window", labels)
            self.assertEqual(len(ax.collections), 2)
            self.assertTrue(all(patch.get_linewidth() == 0.0 for patch in ax.patches))
        finally:
            plt.close(fig)

    def test_nfip_design_life_level_panel_uses_new_label(self) -> None:
        bundle = _make_nfip_bundle()
        fig, ax = plt.subplots()
        try:
            _draw_design_life_levels_ax(ax, bundle)
            labels = [line.get_label() for line in ax.lines]
            self.assertEqual(
                labels,
                [
                    "design-life level (tau=0.50)",
                    "design-life level (tau=0.90)",
                    "design-life level (tau=0.95)",
                    "design-life level (tau=0.99)",
                ],
            )
            self.assertEqual(ax.get_title(), "Synthetic NFIP design-life levels")
            self.assertTrue(np.allclose(ax.get_xticks(), [1.0, 2.0, 5.0, 10.0, 25.0, 50.0]))
        finally:
            plt.close(fig)

    def test_evi_only_design_life_level_panel_uses_new_label(self) -> None:
        bundle = _make_evi_only_bundle()
        fig, ax = plt.subplots()
        try:
            _draw_design_life_levels_ax(ax, bundle)
            labels = [line.get_label() for line in ax.lines]
            self.assertEqual(
                labels,
                [
                    "design-life level (tau=0.50)",
                    "design-life level (tau=0.90)",
                    "design-life level (tau=0.95)",
                    "design-life level (tau=0.99)",
                ],
            )
            self.assertEqual(ax.get_title(), "Synthetic EVI-only design-life levels")
        finally:
            plt.close(fig)

    def test_weather_application_specs_use_log_design_life_scale(self) -> None:
        from application.specs import spec_by_key

        registry = spec_by_key()
        self.assertEqual(
            registry["houston_hobby_precipitation"].design_life_level_yscale,
            "log",
        )
        self.assertEqual(
            registry["phoenix_hot_dry_severity"].design_life_level_yscale,
            "log",
        )

    def test_nfip_application_specs_use_log_annual_max_time_series_scale(self) -> None:
        from application.specs import spec_by_key

        registry = spec_by_key()
        self.assertEqual(registry["tx_nfip_claims"].time_series_display_yscale, "linear")
        self.assertEqual(registry["tx_nfip_claims"].time_series_annual_max_yscale, "log")
        self.assertEqual(registry["fl_nfip_claims"].time_series_display_yscale, "linear")
        self.assertEqual(registry["fl_nfip_claims"].time_series_annual_max_yscale, "log")

    def test_nfip_time_series_plot_keeps_daily_linear_and_annual_max_log(self) -> None:
        bundle = _make_nfip_bundle()

        _plot_daily_and_annual(
            bundle.prepared.display,
            ylabel=bundle.spec.ylabel,
            title=bundle.spec.time_series_title,
            display_yscale=bundle.spec.time_series_display_yscale,
            annual_max_yscale=bundle.spec.time_series_annual_max_yscale,
            close=False,
        )
        fig = plt.gcf()
        try:
            axes = fig.axes
            self.assertEqual(len(axes), 2)
            self.assertEqual(axes[0].get_yscale(), "linear")
            self.assertEqual(axes[1].get_yscale(), "log")
            self.assertEqual(axes[0].get_ylabel(), "usd")
            self.assertEqual(axes[1].get_ylabel(), "annual max usd")
        finally:
            plt.close(fig)

    def test_wrapped_axis_label_splits_long_parenthetical_units(self) -> None:
        from application.outputs import _wrapped_axis_label

        self.assertEqual(
            _wrapped_axis_label("building payouts (inflation-adjusted 2025 USD)"),
            "building payouts\n(inflation-adjusted 2025 USD)",
        )
        self.assertEqual(
            _wrapped_axis_label(
                "building payouts (inflation-adjusted 2025 USD)",
                prefix="annual max",
            ),
            "annual max building payouts\n(inflation-adjusted 2025 USD)",
        )


if __name__ == "__main__":
    unittest.main()
