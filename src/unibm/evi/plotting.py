"""Plotting helpers for EVI model objects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .._runtime import prepare_matplotlib_env

import numpy as np

from .models import ScalingFit


def _pyplot():
    """Prepare writable Matplotlib caches and lazily import pyplot on first plotting use."""
    prepare_matplotlib_env()
    import matplotlib.pyplot as plt

    return plt


def _resolved_file_path(file_path: Path | str | None) -> Path | None:
    """Convert an optional output path to Path without resolving or creating it."""
    if file_path is None:
        return None
    return Path(file_path)


def _save_figure_outputs(fig, file_path: Path) -> None:
    """Create parent directories and save the figure, overwriting an existing file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path)


def plot_scaling_fit(
    fit: ScalingFit,
    *,
    file_path: Path | str | None = None,
    dpi: int = 150,
    title: str | None = None,
    save: bool = False,
    close: bool = False,
    xlabel: str = "log(block size)",
    ylabel: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot an EVI scaling fit on the log-log block-size scale.

    Return ``(fig, ax)`` for customization. No file is saved by default; set
    ``save=True`` and ``file_path`` to save. Use ``close=True`` for batch jobs.
    Both axes show natural-log coordinates. Highlight the selected plateau
    and draw its fitted line; points outside it are shown for context.
    Saving creates parent directories and replaces an existing output file.
    A missing ``file_path`` skips saving even when ``save=True``.
    """
    plt = _pyplot()
    if ylabel is None:
        if fit.target == "quantile":
            ylabel = f"log block quantile (tau={fit.quantile:.2f})"
        else:
            ylabel = f"log block {fit.target}"
    x = np.asarray(fit.log_block_sizes, dtype=float)
    y = np.asarray(fit.log_values, dtype=float)
    plateau_mask = np.asarray(fit.plateau_mask, dtype=bool)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4), dpi=dpi)
    ax.scatter(x=x, y=y, s=18, alpha=0.7, color="tab:blue", label="log block summary")
    ax.scatter(
        x=x[plateau_mask],
        y=y[plateau_mask],
        s=28,
        alpha=0.9,
        color="tab:red",
        label="selected plateau",
    )
    fitted = fit.intercept + fit.slope * x[plateau_mask]
    ax.plot(
        x[plateau_mask],
        fitted,
        color="black",
        linestyle="--",
        lw=1.2,
        label=f"slope = {fit.slope:.3f}",
    )
    ax.axvline(np.log(fit.plateau_bounds[0]), color="grey", linestyle=":", lw=1)
    ax.axvline(np.log(fit.plateau_bounds[1]), color="grey", linestyle=":", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    file_path = _resolved_file_path(file_path)
    if save and file_path is not None:
        _save_figure_outputs(fig, file_path)
    if close:
        plt.close(fig)
    return fig, ax


__all__ = ["plot_scaling_fit"]
