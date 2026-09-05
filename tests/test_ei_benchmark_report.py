from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from benchmark.design import default_ei_simulation_configs
from benchmark import ei_eval
from benchmark.ei_eval import _ei_result_row, run_ei_benchmark, summarize_ei_benchmark
from benchmark.ei_report import (
    EI_SHRINKAGE_GRID,
    EI_SHRINKAGE_METHODS,
    build_ei_shrinkage_sensitivity_summary,
    write_ei_benchmark_manuscript_artifacts,
)
from unibm.ei import ExtremalIndexEstimate, extract_stable_path_window


def _bootstrap_bundle_with_zero_northrop_sliding(
    *args: object,
    bundle: object,
    **kwargs: object,
) -> dict:
    results = {}
    for key, path in bundle.paths.items():
        levels = extract_stable_path_window(path)[0]
        covariance = np.eye(levels.size)
        if key == ("northrop", True):
            covariance = np.zeros_like(covariance)
        results[key] = {
            "block_sizes": levels,
            "samples": np.zeros((2, levels.size)),
            "covariance": covariance,
            "base_path": key[0],
            "sliding": key[1],
        }
    return results


class EiBenchmarkReportTests(unittest.TestCase):
    def test_adaptive_bypasses_fixed_cache_and_retains_degenerate_failure(self) -> None:
        values = np.random.default_rng(71).pareto(2.3, 365) + 1.0
        bundle = ei_eval.prepare_ei_bundle(values, allow_zeros=False)
        with (
            mock.patch.object(
                ei_eval, "_load_cached_ei_bootstrap_bundle", side_effect=AssertionError("fixed")
            ),
            mock.patch.object(
                ei_eval,
                "bootstrap_bm_ei_path",
                side_effect=ValueError("EI bootstrap covariance must have positive scale."),
            ),
        ):
            results = ei_eval._load_or_compute_ei_bootstrap_bundle(
                values,
                bundle=bundle,
                cache_dir=None,
                cache_key="test",
                reps="adaptive",
                random_state=0,
            )
        self.assertEqual(len(results), 4)
        for result in results.values():
            self.assertEqual(result["failure_reason"], "degenerate_adaptive_covariance")
            self.assertTrue(ei_eval._unavailable_bootstrap_covariance(result))

    def test_cached_ei_bootstrap_restores_estimator_identity(self) -> None:
        key = ("bb", True)
        levels = np.array([4, 8], dtype=int)
        bundles = {
            key: {
                "block_sizes": levels,
                "samples": np.ones((2, 2)),
                "covariance": np.eye(2),
                "base_path": key[0],
                "sliding": key[1],
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            ei_eval._save_cached_ei_bootstrap_bundle(
                cache_dir=Path(tmpdir),
                cache_key="identity",
                reps=2,
                bundles=bundles,
            )
            loaded = ei_eval._load_cached_ei_bootstrap_bundle(
                cache_dir=Path(tmpdir),
                cache_key="identity",
                selected_levels_by_key={key: levels},
                reps=2,
            )

        assert loaded is not None
        self.assertEqual(loaded[key]["base_path"], "bb")
        self.assertIs(loaded[key]["sliding"], True)

    def test_ei_detail_row_exposes_regression_scale_standard_error(self) -> None:
        cfg = SimpleNamespace(
            benchmark_set="universal",
            family="frechet_max_ar",
            scenario="example",
            n_obs=365,
            xi_true=0.5,
            theta_true=0.5,
            phi=0.0,
        )
        estimate = ExtremalIndexEstimate(
            method="bb_sliding_fgls",
            theta_hat=0.6,
            confidence_interval=(0.4, 0.8),
            standard_error=0.06,
            z_standard_error=0.1,
        )

        row = _ei_result_row(cfg, 0, estimate)

        self.assertEqual(row["z_standard_error"], 0.1)

    def test_ei_benchmark_retains_degenerate_fgls_as_failed_attempt(self) -> None:
        cfg = SimpleNamespace(
            benchmark_set="test",
            family="test",
            scenario="test",
            n_obs=365,
            xi_true=0.5,
            theta_true=1.0,
            phi=0.0,
        )
        vec = np.random.default_rng(123).pareto(2.0, 365) + 1.0

        with (
            mock.patch.object(
                ei_eval,
                "load_or_simulate_series_bank",
                return_value=np.asarray([vec]),
            ),
            mock.patch.object(
                ei_eval,
                "_load_or_compute_ei_bootstrap_bundle",
                side_effect=_bootstrap_bundle_with_zero_northrop_sliding,
            ),
        ):
            internal, _ = ei_eval.evaluate_ei_config(cfg)

        self.assertEqual(len(internal), 8)
        failed = internal.loc[internal["method"] == "northrop_sliding_fgls"].iloc[0]
        self.assertFalse(failed["fit_succeeded"])
        self.assertEqual(failed["failure_reason"], "zero_bootstrap_covariance")
        self.assertTrue(np.isnan(failed["theta_hat"]))
        self.assertFalse(failed["covered"])

        summary = summarize_ei_benchmark(internal)
        failed_summary = summary.loc[summary["method"] == "northrop_sliding_fgls"].iloc[0]
        self.assertEqual(failed_summary["n_rep"], 1)
        self.assertEqual(failed_summary["n_success"], 0)
        self.assertEqual(failed_summary["n_failed"], 1)
        self.assertEqual(failed_summary["failure_rate"], 1.0)
        self.assertEqual(failed_summary["coverage"], 0.0)
        self.assertTrue(np.isnan(failed_summary["theta_hat_se"]))

    def test_ei_shrinkage_retains_degenerate_fgls_as_failed_attempt(self) -> None:
        configs = default_ei_simulation_configs(
            xi_values=(0.50,),
            theta_values=(1.0,),
            families=("frechet_max_ar",),
            n_obs=365,
            reps=1,
        )
        vec = np.random.default_rng(123).pareto(2.0, 365) + 1.0

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "benchmark.ei_report.load_or_simulate_series_bank",
                return_value=np.asarray([vec]),
            ),
            mock.patch(
                "benchmark.ei_report._load_or_compute_ei_bootstrap_bundle",
                side_effect=_bootstrap_bundle_with_zero_northrop_sliding,
            ),
        ):
            summary, _ = build_ei_shrinkage_sensitivity_summary(
                root=tmpdir,
                configs=configs,
                deltas=(0.35,),
                methods=("northrop_sliding_fgls",),
                force=True,
            )

        row = summary.iloc[0]
        self.assertEqual(row["n_rep"], 1)
        self.assertEqual(row["n_success"], 0)
        self.assertEqual(row["n_failed"], 1)
        self.assertEqual(row["failure_rate"], 1.0)
        self.assertEqual(row["median_coverage"], 0.0)

    def test_build_ei_shrinkage_sensitivity_summary_emits_expected_grid_and_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = default_ei_simulation_configs(
                xi_values=(0.50,),
                theta_values=(0.25,),
                families=("frechet_max_ar",),
                n_obs=64,
                reps=1,
            )
            summary, output_path = build_ei_shrinkage_sensitivity_summary(
                root=tmpdir,
                configs=configs,
                force=True,
            )
            self.assertEqual(
                sorted(summary["delta"].dropna().unique().tolist()),
                list(EI_SHRINKAGE_GRID),
            )
            self.assertEqual(
                summary["method"].drop_duplicates().tolist(),
                list(EI_SHRINKAGE_METHODS),
            )
            self.assertEqual(summary["family"].drop_duplicates().tolist(), ["frechet_max_ar"])
            self.assertIn("median_ape", summary.columns)
            self.assertIn("median_coverage", summary.columns)
            self.assertIn("mean_interval_score", summary.columns)
            self.assertTrue(output_path.exists())
            persisted = pd.read_csv(output_path)
            self.assertIn("delta", persisted.columns)
            self.assertIn("method", persisted.columns)
            self.assertIn("median_ape", persisted.columns)
            self.assertIn("median_coverage", persisted.columns)
            self.assertIn("mean_interval_score", persisted.columns)

    def test_write_ei_benchmark_manuscript_artifacts_materializes_expected_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = default_ei_simulation_configs(
                xi_values=(0.50,),
                theta_values=(0.25,),
                families=("frechet_max_ar",),
                n_obs=64,
                reps=1,
            )
            _, internal_summary, _, external_summary = run_ei_benchmark(
                configs=configs,
                cache_dir=Path(tmpdir) / "cache",
                max_workers=1,
            )
            fig_dir = Path(tmpdir) / "figures"
            table_dir = Path(tmpdir) / "tables"
            web_dir = Path(tmpdir) / "web"
            fig_dir.mkdir(parents=True, exist_ok=True)
            table_dir.mkdir(parents=True, exist_ok=True)
            with (
                mock.patch("benchmark.ei_report.plot_ei_core_panels") as core_panels,
                mock.patch("benchmark.ei_report.plot_ei_targets_panels") as target_panels,
                mock.patch(
                    "benchmark.ei_report.plot_ei_interval_sharpness_scatter"
                ) as sharpness_scatter,
                mock.patch("benchmark.ei_report.plot_ei_overview_panels") as overview_panels,
                mock.patch(
                    "benchmark.ei_report.plot_ei_shrinkage_sensitivity"
                ) as shrinkage_panels,
            ):
                write_ei_benchmark_manuscript_artifacts(
                    internal_summary,
                    external_summary,
                    shrinkage_sensitivity_summary=None,
                    fig_dir=fig_dir,
                    table_dir=table_dir,
                    web_dir=web_dir,
                )

            summary_path = table_dir / "benchmark_ei_summary_main.tex"
            interval_path = table_dir / "benchmark_ei_interval_main.tex"
            overview_path = table_dir / "benchmark_ei_overview_main.tex"
            self.assertTrue(summary_path.exists())
            self.assertTrue(interval_path.exists())
            self.assertTrue(overview_path.exists())
            web_summary = pd.read_csv(web_dir / "ei_benchmark.csv")
            self.assertEqual(len(web_summary), len(internal_summary) + len(external_summary))
            self.assertEqual(
                set(web_summary["method"]),
                set(internal_summary["method"]) | set(external_summary["method"]),
            )
            self.assertIn("interval_score_mean", web_summary)
            self.assertIn("interval_score_mcse", web_summary)

            summary_tex = summary_path.read_text()
            self.assertIn(r"\label{tab:benchmark-ei-summary-main}", summary_tex)
            self.assertIn(r"\multicolumn{1}{c}{Fréchet max-AR}", summary_tex)
            self.assertIn(r"true $\xi$", summary_tex)
            self.assertIn(r"\shortstack[l]{Northrop-", summary_tex)

            interval_tex = interval_path.read_text()
            self.assertIn(r"\label{tab:benchmark-ei-interval-main}", interval_tex)
            self.assertIn("EI interval sharpness-versus-calibration summary", interval_tex)
            self.assertIn(r"mean\_interval\_score", interval_tex)

            overview_tex = overview_path.read_text()
            self.assertIn(r"\label{tab:benchmark-ei-overview-main}", overview_tex)
            self.assertIn("Appendix full EI benchmark overview", overview_tex)

            core_panels.assert_called_once_with(
                internal_summary,
                title="",
                file_path=fig_dir / "benchmark_ei_summary.pdf",
                web_path=web_dir / "ei_benchmark.png",
                save=True,
            )
            target_panels.assert_called_once_with(
                internal_summary,
                external_summary,
                title="",
                file_path=fig_dir / "benchmark_ei_targets.pdf",
                save=True,
            )
            sharpness_scatter.assert_called_once_with(
                internal_summary,
                external_summary,
                file_path=fig_dir / "benchmark_ei_interval_sharpness.pdf",
                save=True,
            )
            overview_panels.assert_called_once_with(
                internal_summary,
                external_summary,
                file_path=fig_dir / "benchmark_ei_overview.pdf",
                save=True,
            )
            shrinkage_panels.assert_not_called()
