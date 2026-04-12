from __future__ import annotations
# ruff: noqa: E402

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.design import FGLS_BOOTSTRAP_REPS, fit_methods_for_series
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
                reps=FGLS_BOOTSTRAP_REPS,
                random_state=random_state,
            )
            fgls_id = f"{scheme_name}_{summary_target}_fgls"
            fits[fgls_id] = estimate_target_scaling(
                values,
                target=internal_target,
                quantile=quantile,
                sliding=sliding,
                bootstrap_reps=FGLS_BOOTSTRAP_REPS,
                random_state=random_state,
                curve=curve,
                plateau=plateau,
                bootstrap_result=bootstrap_result,
            )
    return fits


class BenchmarkDesignInternalTests(unittest.TestCase):
    def test_fit_methods_for_series_matches_exact_baseline(self) -> None:
        rs = np.random.default_rng(71)
        values = rs.pareto(2.3, 365) + 1.0
        observed = fit_methods_for_series(
            values,
            quantile=0.5,
            random_state=19,
            allow_scheme_bootstrap=True,
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


if __name__ == "__main__":
    unittest.main()
