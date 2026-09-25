"""Plotting helpers for EI path and fit objects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .._runtime import prepare_matplotlib_env

import numpy as np

from .models import EiPathBundle, ExtremalIndexEstimate


def _pyplot():
    """Prepare writable Matplotlib cache locations, then import pyplot on demand."""
    prepare_matplotlib_env()
    import matplotlib.pyplot as plt

    return plt


def _save_figure_outputs(fig, file_path: Path | str) -> None:
    """Create missing parent directories and save using Matplotlib's path-based format.

    An existing destination may be overwritten; filesystem and format errors
    propagate to the caller.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path)


def _draw_path_window(ax, *, lo: int, hi: int, color: str = "tab:orange") -> None:
    """Shade one selected stable EI window on the log-block-size axis."""
    ax.axvspan(np.log(float(lo)), np.log(float(hi)), color=color, alpha=0.08, lw=0.0)


def _finite_path_arrays(
    block_sizes: np.ndarray,
    theta_path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter aligned block-size and theta arrays using finite theta values.

    Block sizes are assumed to have been validated by the path constructor.
    """
    levels = np.asarray(block_sizes, dtype=int)
    theta = np.asarray(theta_path, dtype=float)
    mask = np.isfinite(theta)
    return levels[mask], theta[mask]


def _default_path_title(path: EiPathBundle) -> str:
    """Return a concise default title for one EI path."""
    block_scheme = "sliding" if path.sliding else "disjoint"
    return f"{path.base_path} {block_scheme} EI path"


def plot_ei_path(
    path: EiPathBundle,
    *,
    file_path: Path | str | None = None,
    dpi: int = 150,
    title: str | None = None,
    close: bool = False,
    xlabel: str = "log(block size)",
    ylabel: str = "extremal index",
) -> tuple[Figure, Axes]:
    """Plot one observed EI path and any selected stable window.

    Plot finite theta values against the natural log of block size, shade the
    stored stable window when present, and mark the native estimator's level. Raise
    ``ValueError`` if the path has no finite theta values.

    Return ``(fig, ax)`` for customization at the requested ``dpi``. Saving
    requires a non-None ``file_path``; parent directories
    are created and an existing file may be overwritten. ``close=True`` closes
    the figure in pyplot after drawing/saving but still returns its objects.
    """
    plt = _pyplot()
    levels, theta = _finite_path_arrays(path.block_sizes, path.theta_path)
    if levels.size == 0:
        raise ValueError("EI path contains no finite theta values to plot.")
    x = np.log(levels.astype(float))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4), dpi=dpi)
    ax.plot(x, theta, color="tab:blue", marker="o", ms=3.2, lw=1.1, label="observed path")
    if path.stable_window is not None:
        _draw_path_window(ax, lo=path.stable_window.lo, hi=path.stable_window.hi)
    ax.axvline(
        np.log(float(path.selected_level)),
        color="tab:red",
        linestyle="--",
        lw=1.0,
        label=f"selected level = {path.selected_level}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or _default_path_title(path))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if file_path is not None:
        _save_figure_outputs(fig, file_path)
    if close:
        plt.close(fig)
    return fig, ax


def _default_fit_title(fit: ExtremalIndexEstimate) -> str:
    """Return a concise default title for one EI estimate."""
    return str(fit.method).replace("_", "-")


def _plot_path_aware_fit(ax, fit: ExtremalIndexEstimate) -> None:
    """Draw retained theta values, tuning markers, and a horizontal estimate/CI.

    Mutate the supplied axes; require at least one finite path value and show
    the confidence band only when both interval endpoints are finite.
    """
    levels, theta = _finite_path_arrays(np.asarray(fit.path_level, dtype=int), fit.path_theta)
    if levels.size == 0:
        raise ValueError("Path-aware EI plotting requires retained finite path values.")
    x = np.log(levels.astype(float))
    ax.plot(x, theta, color="tab:blue", marker="o", ms=3.0, lw=1.0, label="retained path")
    if fit.stable_window is not None:
        _draw_path_window(ax, lo=fit.stable_window.lo, hi=fit.stable_window.hi)
    if fit.selected_level is not None:
        ax.axvline(
            np.log(float(fit.selected_level)),
            color="tab:red",
            linestyle="--",
            lw=1.0,
            label=f"selected level = {fit.selected_level}",
        )
    ax.axhline(fit.theta_hat, color="black", lw=1.2, linestyle="-", label="theta_hat")
    lo, hi = fit.confidence_interval
    if np.isfinite(lo) and np.isfinite(hi):
        ax.axhspan(lo, hi, color="tab:green", alpha=0.1, lw=0.0, label="confidence interval")
    ax.set_xlabel("log(block size)")
    ax.set_ylabel("extremal index")
    ax.grid(alpha=0.3)
    ax.legend()


def _threshold_fit_label(fit: ExtremalIndexEstimate) -> str:
    """Return a compact x-axis label for one threshold-side estimate."""
    pieces = []
    if fit.selected_threshold_quantile is not None:
        pieces.append(f"u={fit.selected_threshold_quantile:.2f}")
    if fit.selected_run_k is not None:
        pieces.append(f"K={fit.selected_run_k}")
    return ", ".join(pieces) if pieces else str(fit.method)


def _plot_threshold_fit(ax, fit: ExtremalIndexEstimate) -> None:
    """Draw theta with its finite CI, or a point alone when endpoints are unavailable.

    Mutate the supplied axes and label the chosen threshold quantile/run length.
    Valid CI endpoints are assumed to enclose the point estimate.
    """
    lo, hi = fit.confidence_interval
    if np.isfinite(lo) and np.isfinite(hi):
        ax.errorbar(
            [0.0],
            [fit.theta_hat],
            yerr=[[fit.theta_hat - lo], [hi - fit.theta_hat]],
            fmt="o",
            color="tab:blue",
            capsize=4,
            label=str(fit.method).replace("_", "-"),
        )
    else:
        ax.scatter([0.0], [fit.theta_hat], color="tab:blue", s=24, label=str(fit.method))
    ax.set_xlim(-0.8, 0.8)
    ax.set_xticks([0.0], [_threshold_fit_label(fit)])
    ax.set_ylabel("extremal index")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()


def plot_ei_fit(
    fit: ExtremalIndexEstimate,
    *,
    file_path: Path | str | None = None,
    dpi: int = 150,
    title: str | None = None,
    close: bool = False,
) -> tuple[Figure, Axes]:
    """Plot one EI fit either as a retained path view or a threshold summary.

    Fits with retained path levels and theta values use a log-block-size view;
    other fits use a single point with the stored interval when finite. The
    plotted interval is supplied by the estimator, not recomputed by this helper.

    Return ``(fig, ax)`` at the requested ``dpi``. Saving requires a non-None
    ``file_path``; missing parent directories are
    created and existing files may be overwritten. ``close=True`` closes the
    pyplot figure after drawing/saving while still returning its objects.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4), dpi=dpi)
    if fit.path_level and fit.path_theta:
        _plot_path_aware_fit(ax, fit)
    else:
        _plot_threshold_fit(ax, fit)
    ax.set_title(title or _default_fit_title(fit))
    fig.tight_layout()
    if file_path is not None:
        _save_figure_outputs(fig, file_path)
    if close:
        plt.close(fig)
    return fig, ax


__all__ = ["plot_ei_fit", "plot_ei_path"]
