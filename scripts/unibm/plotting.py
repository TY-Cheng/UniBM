"""Plotting helpers for UniBM model objects."""

from __future__ import annotations

from pathlib import Path
import sys

from ._runtime import prepare_matplotlib_env

prepare_matplotlib_env()
# isort: off
import matplotlib.pyplot as plt  # noqa: E402

# isort: on
import numpy as np

from .models import ExtremalIndexReciprocalFit, ScalingFit


def _resolved_file_path(file_path: Path | str | None) -> Path | None:
    """Coerce optional output paths to `Path` objects."""
    if file_path is None:
        return None
    return Path(file_path)


def _should_close_figure(close: bool | None) -> bool:
    """Close figures automatically in non-notebook batch usage by default."""
    if close is not None:
        return bool(close)
    return "ipykernel" not in sys.modules


def plot_scaling_fit(
    fit: ScalingFit,
    *,
    file_path: Path | str | None = None,
    dpi: int = 1200,
    title: str | None = None,
    save: bool = False,
    close: bool | None = None,
    xlabel: str = "log(block size)",
    ylabel: str | None = None,
) -> None:
    """Visualize the fitted scaling relation over the selected block-size plateau."""
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
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_path)
    if _should_close_figure(close):
        plt.close(fig)


def plot_extremal_index_reciprocal(
    fit: ExtremalIndexReciprocalFit,
    *,
    file_path: Path | str | None = None,
    dpi: int = 1200,
    title: str | None = None,
    save: bool = False,
    close: bool | None = None,
    xlabel: str = "log(block size)",
    ylabel: str = "1 / extremal index",
) -> None:
    """Visualize reciprocal extremal-index diagnostics over candidate block sizes."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=dpi)
    ax.scatter(
        fit.log_block_sizes,
        fit.northrop_values,
        label=f"Northrop = {fit.northrop_estimate:.3f}",
        s=12,
        facecolors="none",
        color="tab:blue",
    )
    ax.scatter(
        fit.log_block_sizes,
        fit.northrop_standard_deviations,
        label="sd Northrop",
        s=10,
        facecolors="none",
        color="tab:cyan",
    )
    ax.scatter(
        fit.log_block_sizes,
        fit.bb_values,
        label=f"BB = {fit.bb_estimate:.3f}",
        s=12,
        facecolors="none",
        color="tab:red",
    )
    ax.scatter(
        fit.log_block_sizes,
        fit.bb_standard_deviations,
        label="sd BB",
        s=10,
        facecolors="none",
        color="tab:orange",
    )
    ax.axvline(np.log(fit.northrop_block_size), color="tab:blue", linestyle=":", lw=1)
    ax.axvline(np.log(fit.bb_block_size), color="tab:red", linestyle=":", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    file_path = _resolved_file_path(file_path)
    if save and file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_path)
    if _should_close_figure(close):
        plt.close(fig)
