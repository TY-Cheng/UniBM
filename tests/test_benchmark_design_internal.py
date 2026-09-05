from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from benchmark import design as benchmark_design
from benchmark.design import fit_methods_for_series
from unibm.evi import (
    block_summary_curve,
    circular_block_summary_bootstrap,
    estimate_target_scaling,
    generate_block_sizes,
    select_penultimate_window,
)


def _baseline_fit_methods_for_series(
    vec: np.ndarray,
    *,
    quantile: float,
    random_state: int,
) -> dict[str, object]:
    values = np.asarray(vec, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    block_sizes = generate_block_sizes(n_obs=values.size)
    fits: dict[str, object] = {}
    for sliding in (True, False):
        scheme_name = "sliding" if sliding else "disjoint"
        for summary_target, internal_target in (
            ("median", "quantile"),
            ("mean", "mean"),
            ("mode", "mode"),
        ):
            curve = block_summary_curve(
                values,
                block_sizes,
                sliding=sliding,
                quantile=quantile,
                target=internal_target,
            )
            plateau = select_penultimate_window(
                curve.log_block_sizes,
                curve.log_values,
                min_points=5,
                trim_fraction=0.15,
            )
            ols_id = f"{scheme_name}_{summary_target}_ols"
            fits[ols_id] = estimate_target_scaling(
                values,
                regression="OLS",
                target=internal_target,
                quantile=quantile,
                sliding=sliding,
                bootstrap_reps=0,
                random_state=random_state,
                curve=curve,
                plateau=plateau,
            )
            bootstrap_result = circular_block_summary_bootstrap(
                values,
                block_sizes,
                target=internal_target,
                quantile=quantile,
                sliding=sliding,
                reps=120,
                random_state=random_state,
            )
            fgls_id = f"{scheme_name}_{summary_target}_fgls"
            fits[fgls_id] = estimate_target_scaling(
                values,
                regression="FGLS",
                target=internal_target,
                quantile=quantile,
                sliding=sliding,
                random_state=random_state,
                curve=curve,
                plateau=plateau,
                bootstrap_result=bootstrap_result,
            )
    return fits


class BenchmarkDesignInternalTests(unittest.TestCase):
    def test_default_adaptive_bypasses_fixed_bootstrap_cache(self) -> None:
        values = np.random.default_rng(71).pareto(2.3, 365) + 1.0
        with mock.patch.object(
            benchmark_design, "_scheme_bootstrap_results", side_effect=AssertionError("fixed")
        ):
            fit = fit_methods_for_series(
                values, quantile=0.5, random_state=19, method_ids=["sliding_median_fgls"]
            )["sliding_median_fgls"]
        self.assertEqual(fit.bootstrap_reps_policy, "adaptive")
        self.assertIn(fit.bootstrap_reps_used, (128, 256, 512, 768, 1024))
        self.assertEqual(fit.covariance_shrinkage, 0.37)

    def test_cached_evi_bootstrap_restores_estimator_identity(self) -> None:
        block_sizes = np.array([4, 8], dtype=int)
        bundle = {
            "sliding": {
                "quantile": {
                    "block_sizes": block_sizes,
                    "samples": np.ones((2, 2)),
                    "covariance": np.eye(2),
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "bootstrap.npz"
            benchmark_design._save_bootstrap_results_bundle(cache_file, bundle)
            loaded = benchmark_design._load_bootstrap_results_bundle(
                cache_file,
                quantile=0.9,
            )

        result = loaded["sliding"]["quantile"]
        self.assertEqual(result["target"], "quantile")
        self.assertEqual(result["quantile"], 0.9)
        self.assertIs(result["sliding"], True)

    def test_fit_methods_for_series_matches_exact_baseline(self) -> None:
        rs = np.random.default_rng(71)
        values = rs.pareto(2.3, 365) + 1.0
        observed = fit_methods_for_series(
            values,
            quantile=0.5,
            random_state=19,
            allow_scheme_bootstrap=True,
            bootstrap_reps=120,
        )
        expected = _baseline_fit_methods_for_series(
            values,
            quantile=0.5,
            random_state=19,
        )
        self.assertEqual(set(observed), set(expected))
        for method_id in observed:
            obs_fit = observed[method_id]
            exp_fit = expected[method_id]
            self.assertAlmostEqual(obs_fit.slope, exp_fit.slope)
            np.testing.assert_allclose(obs_fit.confidence_interval, exp_fit.confidence_interval)
            self.assertEqual(
                (obs_fit.plateau.start, obs_fit.plateau.stop),
                (exp_fit.plateau.start, exp_fit.plateau.stop),
            )
