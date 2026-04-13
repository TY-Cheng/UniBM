from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np

import unibm
from unibm._window_ops import circular_sliding_window_maximum, sliding_window_extreme_valid
from unibm.ei import (
    EiPathBundle,
    EiStableWindow,
    ExtremalIndexEstimate,
    plot_ei_fit,
    plot_ei_path,
)
from unibm.evi import (
    BlockSummaryCurve,
    PlateauWindow,
    ScalingFit,
    plot_scaling_fit,
)
from unibm.evi.plotting import (
    _resolved_file_path,
    _should_close_figure,
)
from unibm._runtime import (
    _env_path_is_writable,
    _runtime_cache_suffix,
    prepare_matplotlib_env,
)


def _make_scaling_fit() -> ScalingFit:
    block_sizes = np.array([4, 8, 16], dtype=int)
    curve = BlockSummaryCurve(
        block_sizes=block_sizes,
        counts=np.array([128, 64, 32], dtype=int),
        values=np.array([2.0, 3.0, 4.5], dtype=float),
        positive_mask=np.array([True, True, True], dtype=bool),
    )
    plateau = PlateauWindow(
        start=1,
        stop=3,
        score=0.1,
        mask=np.array([False, True, True], dtype=bool),
        x=np.log(block_sizes[1:]),
        y=np.log(curve.values[1:]),
    )
    return ScalingFit(
        target="quantile",
        quantile=0.5,
        sliding=True,
        intercept=0.1,
        slope=0.8,
        standard_error=0.05,
        confidence_interval=(0.7, 0.9),
        curve=curve,
        plateau=plateau,
        cov_beta=np.eye(2),
        bootstrap=None,
    )


def _make_ei_path() -> EiPathBundle:
    block_sizes = np.array([4, 8, 16, 32], dtype=int)
    theta_path = np.array([0.62, 0.58, 0.57, 0.59], dtype=float)
    return EiPathBundle(
        base_path="bb",
        sliding=True,
        block_sizes=block_sizes,
        theta_path=theta_path,
        eir_path=1.0 / theta_path,
        z_path=np.log(1.0 / theta_path),
        sample_counts=np.array([120, 60, 30, 15], dtype=int),
        sample_statistics={int(level): np.linspace(0.1, 0.2, 4) for level in block_sizes},
        stable_window=EiStableWindow(8, 32),
        selected_level=16,
    )


def _make_ei_bm_fit() -> ExtremalIndexEstimate:
    return ExtremalIndexEstimate(
        method="bb_sliding_fgls",
        theta_hat=0.58,
        confidence_interval=(0.51, 0.65),
        standard_error=0.035,
        ci_method="log_wald",
        ci_variant="bootstrap_cov",
        tuning_axis="b",
        selected_level=16,
        stable_window=EiStableWindow(8, 32),
        path_level=(4, 8, 16, 32),
        path_theta=(0.62, 0.58, 0.57, 0.59),
        path_eir=tuple(1.0 / np.array([0.62, 0.58, 0.57, 0.59], dtype=float)),
        block_scheme="sliding",
        base_path="bb",
        regression="FGLS",
    )


def _make_ei_threshold_fit() -> ExtremalIndexEstimate:
    return ExtremalIndexEstimate(
        method="k_gaps",
        theta_hat=0.61,
        confidence_interval=(0.49, 0.73),
        standard_error=0.06,
        ci_method="profile",
        ci_variant="default",
        tuning_axis="u",
        selected_threshold_quantile=0.95,
        selected_threshold_value=4.2,
        selected_run_k=2,
    )


class UniBmPlottingAndApiTests(unittest.TestCase):
    def test_runtime_helpers_and_window_ops_cover_remaining_branches(self) -> None:
        fake_os = types.SimpleNamespace(environ={"USERNAME": "alice"})
        with mock.patch("unibm._runtime.os", fake_os):
            self.assertEqual(_runtime_cache_suffix(), "alice")

        self.assertFalse(_env_path_is_writable(None))
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertTrue(_env_path_is_writable(tmpdir))
            with mock.patch("pathlib.Path.unlink", side_effect=[OSError("boom"), None]):
                self.assertFalse(_env_path_is_writable(tmpdir))
            prepare_matplotlib_env(cache_tag="unibm-test")
            self.assertIn("MPLCONFIGDIR", __import__("os").environ)
            self.assertIn("XDG_CACHE_HOME", __import__("os").environ)

        self.assertIsNone(_resolved_file_path(None))
        self.assertTrue(_should_close_figure(True))
        with mock.patch.dict("sys.modules", {"ipykernel": object()}, clear=False):
            self.assertFalse(_should_close_figure(None))

        np.testing.assert_allclose(
            sliding_window_extreme_valid([1.0, 2.0, 3.0], 1, reducer="max"),
            np.array([], dtype=float),
        )
        np.testing.assert_allclose(
            circular_sliding_window_maximum([1.0, 4.0, 2.0], 2, use_fast_path=True),
            np.array([4.0, 4.0, 2.0]),
        )
        np.testing.assert_allclose(
            circular_sliding_window_maximum([1.0], 2),
            np.array([], dtype=float),
        )

    def test_plot_scaling_fit_saves_pdf_and_allows_open_figure(self) -> None:
        fit = _make_scaling_fit()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "scaling.pdf"
            plot_scaling_fit(fit, file_path=out, save=True, close=False, title="Scaling")
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)
            self.assertGreater(len(plt.get_fignums()), 0)
            plot_scaling_fit(fit, save=False, close=True)
            plt.close("all")

    def test_plot_ei_path_and_fit_support_path_and_threshold_views(self) -> None:
        path = _make_ei_path()
        bm_fit = _make_ei_bm_fit()
        threshold_fit = _make_ei_threshold_fit()
        with tempfile.TemporaryDirectory() as tmpdir:
            path_out = Path(tmpdir) / "ei-path.pdf"
            fit_out = Path(tmpdir) / "ei-fit.pdf"
            threshold_out = Path(tmpdir) / "ei-threshold.pdf"
            plot_ei_path(path, file_path=path_out, save=True, close=False, title="EI Path")
            plot_ei_fit(bm_fit, file_path=fit_out, save=True, close=False, title="EI Fit")
            plot_ei_fit(
                threshold_fit,
                file_path=threshold_out,
                save=True,
                close=True,
                title="Threshold EI",
            )
            self.assertTrue(path_out.exists())
            self.assertTrue(fit_out.exists())
            self.assertTrue(threshold_out.exists())
            self.assertGreater(path_out.stat().st_size, 0)
            self.assertGreater(fit_out.stat().st_size, 0)
            self.assertGreater(threshold_out.stat().st_size, 0)
            self.assertGreater(len(plt.get_fignums()), 0)
            plt.close("all")

    def test_public_api_exposes_only_slim_facade(self) -> None:
        self.assertEqual(
            set(unibm.__all__),
            {"__version__", "ei", "evi", "estimate_design_life_level", "estimate_evi_quantile"},
        )
        self.assertIs(unibm.evi.estimate_evi_quantile, unibm.estimate_evi_quantile)
        self.assertEqual(unibm.ei.__name__, "unibm.ei")
        self.assertTrue(hasattr(unibm.ei, "plot_ei_fit"))
        self.assertTrue(hasattr(unibm.ei, "plot_ei_path"))
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            getattr(unibm, "plot_scaling_fit")
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            getattr(unibm, "definitely_missing")


if __name__ == "__main__":
    unittest.main()
