from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np

import unibm
from unibm.models import (
    BlockSummaryCurve,
    ExtremalIndexReciprocalFit,
    PlateauWindow,
    ScalingFit,
)
from unibm.plotting import (
    _resolved_file_path,
    _save_figure_outputs,
    _should_close_figure,
    plot_extremal_index_reciprocal,
    plot_scaling_fit,
)
from unibm._runtime import (
    _env_path_is_writable,
    _runtime_cache_suffix,
    prepare_matplotlib_env,
)
from unibm.window_ops import circular_sliding_window_maximum, sliding_window_extreme_valid


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


def _make_ei_fit() -> ExtremalIndexReciprocalFit:
    block_sizes = np.array([4, 8, 16], dtype=int)
    return ExtremalIndexReciprocalFit(
        block_sizes=block_sizes,
        log_block_sizes=np.log(block_sizes),
        northrop_values=np.array([1.5, 1.6, 1.7], dtype=float),
        northrop_standard_deviations=np.array([0.3, 0.2, 0.25], dtype=float),
        bb_values=np.array([1.4, 1.45, 1.5], dtype=float),
        bb_standard_deviations=np.array([0.25, 0.21, 0.22], dtype=float),
        northrop_block_size=8,
        northrop_estimate=1.6,
        bb_block_size=8,
        bb_estimate=1.45,
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

    def test_plot_extremal_index_reciprocal_saves_pdf(self) -> None:
        fit = _make_ei_fit()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "ei.pdf"
            plot_extremal_index_reciprocal(fit, file_path=out, save=True, close=True, title="EI")
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)
            fig, ax = plt.subplots()
            scratch = Path(tmpdir) / "scratch.pdf"
            _save_figure_outputs(fig, scratch)
            self.assertTrue(scratch.exists())
            plt.close(fig)

    def test_public_api_reexports_lazy_plotting_helpers(self) -> None:
        expected = {
            "estimate_evi_quantile",
            "estimate_hill_evi",
            "estimate_extremal_index_reciprocal",
            "plot_scaling_fit",
            "plot_extremal_index_reciprocal",
        }
        self.assertTrue(expected.issubset(set(unibm.__all__)))
        self.assertIs(unibm.plot_scaling_fit, plot_scaling_fit)
        self.assertIs(unibm.plot_extremal_index_reciprocal, plot_extremal_index_reciprocal)
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            getattr(unibm, "definitely_missing")


if __name__ == "__main__":
    unittest.main()
