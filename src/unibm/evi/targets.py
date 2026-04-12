"""Cross-target comparison helpers for EVI block-summary curves."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .blocks import block_summary_curve


def _quantile_summary_label(quantile: float) -> str:
    """Return a human-readable label for a quantile target column."""
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
    """Compare quantile, mean, and mode block-maxima summaries on the same grid."""
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
