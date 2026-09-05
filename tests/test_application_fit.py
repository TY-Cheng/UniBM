from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from application.fit import (
    _application_worker_count,
    build_application_bundles_from_inputs,
    fit_application_ei_estimates,
)
from application.specs import (
    APPLICATION_EI_THRESHOLD_QUANTILES,
    ApplicationPreparedInputs,
    ApplicationSpec,
)
from data_prep.ghcn import PreparedSeries


def _make_prepared(series: pd.Series, *, name: str) -> PreparedSeries:
    annual_maxima = series.groupby(series.index.year).max()
    return PreparedSeries(
        name=name,
        value_name="value",
        series=series,
        annual_maxima=annual_maxima,
        metadata={"provider": "synthetic", "series_role": "shared"},
    )


class ApplicationFitTests(unittest.TestCase):
    @staticmethod
    def _series(*, seed: int = 41, allow_zeros: bool = False) -> pd.Series:
        rs = np.random.default_rng(seed)
        index = pd.date_range("2001-01-01", periods=365 * 20, freq="D")
        if allow_zeros:
            values = np.zeros(index.size, dtype=float)
            cluster_starts = rs.random(index.size) < 0.05
            active = np.zeros(index.size, dtype=bool)
            for lag in range(5):
                active[lag:] |= cluster_starts[: index.size - lag]
            values[active] = rs.gamma(shape=2.5, scale=4.0, size=int(active.sum()))
        else:
            values = rs.pareto(2.4, index.size) + 1.0
        return pd.Series(values, index=index, dtype=float)

    def test_fit_application_ei_estimates_uses_fgls_covariance(self) -> None:
        series = self._series(seed=53, allow_zeros=True)
        bundle, estimates = fit_application_ei_estimates(series, allow_zeros=True)

        self.assertEqual(tuple(bundle.threshold_candidates), APPLICATION_EI_THRESHOLD_QUANTILES)
        self.assertEqual(
            set(estimates),
            {"bb_sliding_fgls", "northrop_sliding_fgls", "k_gaps", "ferro_segers"},
        )
        for key in ("bb_sliding_fgls", "northrop_sliding_fgls"):
            self.assertEqual(estimates[key].regression, "FGLS")
            self.assertEqual(estimates[key].ci_variant, "bootstrap_cov")
            self.assertEqual(estimates[key].bootstrap_reps_policy, "adaptive")
            self.assertIn(estimates[key].bootstrap_reps_used, (128, 256, 512, 768, 1024))
            self.assertEqual(estimates[key].covariance_shrinkage, 0.37)

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
