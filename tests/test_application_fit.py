from __future__ import annotations
# ruff: noqa: E402

import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

try:
    from . import _path_setup as test_paths
except ImportError:  # pragma: no cover
    import _path_setup as test_paths

test_paths.ensure_repo_import_paths()

from application.fit import (
    _application_worker_count,
    _bootstrap_ei_path_draws,
    _materialize_ei_bootstrap_result,
    build_application_bundles_from_inputs,
    fit_application_ei_estimates,
)
from application.specs import (
    APPLICATION_EI_BOOTSTRAP_REPS,
    APPLICATION_RANDOM_STATE,
    ApplicationPreparedInputs,
    ApplicationSpec,
)
from data_prep.ghcn import PreparedSeries
from unibm._bootstrap_sampling import draw_circular_block_bootstrap_samples
from unibm.ei import (
    estimate_ferro_segers,
    estimate_k_gaps,
    estimate_pooled_bm_ei,
    prepare_ei_bundle,
)
from unibm.ei.bootstrap import bootstrap_bm_ei_path_draws


def _make_prepared(series: pd.Series, *, name: str) -> PreparedSeries:
    annual_maxima = series.groupby(series.index.year).max()
    return PreparedSeries(
        name=name,
        value_name="value",
        series=series,
        annual_maxima=annual_maxima,
        metadata={"provider": "synthetic", "series_role": "shared"},
    )


def _baseline_fit_application_ei_estimates(
    series: pd.Series,
    *,
    allow_zeros: bool,
) -> tuple[object, dict[str, object]]:
    bundle = prepare_ei_bundle(series.values, allow_zeros=allow_zeros)
    sample_bank = draw_circular_block_bootstrap_samples(
        series.to_numpy(dtype=float),
        reps=APPLICATION_EI_BOOTSTRAP_REPS,
        random_state=APPLICATION_RANDOM_STATE,
    )
    path_draws = bootstrap_bm_ei_path_draws(
        sample_bank.samples,
        block_sizes=bundle.block_sizes,
        allow_zeros=allow_zeros,
    )
    bb_bootstrap_result = _materialize_ei_bootstrap_result(
        bundle,
        path_draws,
        base_path="bb",
        sliding=True,
    )
    northrop_bootstrap_result = _materialize_ei_bootstrap_result(
        bundle,
        path_draws,
        base_path="northrop",
        sliding=True,
    )
    return bundle, {
        "bb_sliding_fgls": estimate_pooled_bm_ei(
            bundle,
            base_path="bb",
            sliding=True,
            regression="FGLS",
            bootstrap_result=bb_bootstrap_result,
        ),
        "northrop_sliding_fgls": estimate_pooled_bm_ei(
            bundle,
            base_path="northrop",
            sliding=True,
            regression="FGLS",
            bootstrap_result=northrop_bootstrap_result,
        ),
        "k_gaps": estimate_k_gaps(bundle),
        "ferro_segers": estimate_ferro_segers(bundle),
    }


class ApplicationFitTests(unittest.TestCase):
    @staticmethod
    def _series(*, seed: int = 41, allow_zeros: bool = False) -> pd.Series:
        rs = np.random.default_rng(seed)
        index = pd.date_range("2001-01-01", periods=365 * 20, freq="D")
        if allow_zeros:
            values = np.zeros(index.size, dtype=float)
            active = rs.random(index.size) < 0.15
            values[active] = rs.gamma(shape=2.5, scale=4.0, size=int(active.sum()))
        else:
            values = rs.pareto(2.4, index.size) + 1.0
        return pd.Series(values, index=index, dtype=float)

    def test_application_ei_bootstrap_draws_only_requested_keys(self) -> None:
        series = self._series(seed=51, allow_zeros=True)
        bundle = prepare_ei_bundle(series.values, allow_zeros=True)
        draws = _bootstrap_ei_path_draws(
            series,
            ei_bundle=bundle,
            allow_zeros=True,
            reps=8,
            random_state=APPLICATION_RANDOM_STATE,
        )
        self.assertEqual(set(draws), {("bb", True), ("northrop", True)})

    def test_fit_application_ei_estimates_matches_full_baseline(self) -> None:
        series = self._series(seed=53, allow_zeros=True)
        bundle, estimates = fit_application_ei_estimates(series, allow_zeros=True)
        baseline_bundle, baseline_estimates = _baseline_fit_application_ei_estimates(
            series,
            allow_zeros=True,
        )
        np.testing.assert_array_equal(bundle.block_sizes, baseline_bundle.block_sizes)
        for key in ("bb_sliding_fgls", "northrop_sliding_fgls", "k_gaps", "ferro_segers"):
            observed = estimates[key]
            expected = baseline_estimates[key]
            self.assertAlmostEqual(observed.theta_hat, expected.theta_hat)
            np.testing.assert_allclose(observed.confidence_interval, expected.confidence_interval)
            self.assertEqual(observed.stable_window, expected.stable_window)

    def test_parallel_application_bundle_builder_sets_blas_thread_caps(self) -> None:
        series = self._series(seed=59, allow_zeros=False)
        prepared = _make_prepared(series, name="synthetic")
        inputs = {
            "a": ApplicationPreparedInputs(display=prepared, evi=prepared, ei=prepared),
            "b": ApplicationPreparedInputs(display=prepared, evi=prepared, ei=prepared),
        }
        specs = (
            ApplicationSpec(
                key="a",
                provider="synthetic",
                label="A",
                figure_stem="a",
                raw_key="none",
                ylabel="value",
                time_series_title="A",
                scaling_title="A",
                scaling_ylabel="log median block maximum",
                formal_ei=False,
            ),
            ApplicationSpec(
                key="b",
                provider="synthetic",
                label="B",
                figure_stem="b",
                raw_key="none",
                ylabel="value",
                time_series_title="B",
                scaling_title="B",
                scaling_ylabel="log median block maximum",
                formal_ei=False,
            ),
        )
        worker_count = _application_worker_count(len(specs))
        if worker_count <= 1:
            self.skipTest("parallel path unavailable in this environment")
        with mock.patch.dict(
            os.environ,
            {
                key: value
                for key, value in os.environ.items()
                if key not in {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}
            },
            clear=True,
        ):
            with mock.patch("application.fit.ProcessPoolExecutor") as executor:
                executor.return_value.__enter__.return_value.map.return_value = []
                build_application_bundles_from_inputs(inputs, specs=specs)
                self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "1")
                self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), "1")
                self.assertEqual(os.environ.get("MKL_NUM_THREADS"), "1")


if __name__ == "__main__":
    unittest.main()
