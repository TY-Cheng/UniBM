"""Cross-target comparison helpers for EVI block-summary curves."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .blocks import block_summary_curve


def _quantile_summary_label(quantile: float) -> str:
    """Label a near-median quantile as ``median``; otherwise format tau to two decimals."""
    if np.isclose(float(quantile), 0.5):
        return "median"
    return f"quantile_tau_{float(quantile):.2f}"


def target_stability_summary(
    vec: np.ndarray | list[float],
    block_sizes: np.ndarray,
    *,
    sliding: bool = True,
    quantile: float = 0.5,
) -> pd.DataFrame:
    """Return a DataFrame comparing three summaries on a shared block-size grid.

    Rows follow ``block_sizes``; columns are ``block_size``, the quantile
    label (``median`` near 0.5), ``mean``, and ``mode``. Values are on the
    original response scale and may be NaN. This table reports curves;
    it does not select or test a stable regression window.
    """
    out = {"block_size": np.asarray(block_sizes, dtype=int)}
    for target in ["quantile", "mean", "mode"]:
        curve = block_summary_curve(
            vec,
            block_sizes,
            sliding=sliding,
            quantile=quantile,
            target=target,
        )
        key = _quantile_summary_label(quantile) if target == "quantile" else target
        out[key] = curve.values
    return pd.DataFrame(out)


__all__ = ["_quantile_summary_label", "target_stability_summary"]
