"""Plotting helpers for EI path and fit objects."""
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import sys

from .._runtime import prepare_matplotlib_env

prepare_matplotlib_env()
import matplotlib.pyplot as plt
import numpy as np

from .models import EiPathBundle, ExtremalIndexEstimate


def _resolved_file_path(file_path: Path | str | None) -> Path | None:
    """Coerce optional output paths to ``Path`` objects."""
    if file_path is None:
        return None
    return Path(file_path)


def _should_close_figure(close: bool | None) -> bool:
    """Close figures automatically in non-notebook batch usage by default."""
    if close is not None:
        return bool(close)
    return "ipykernel" not in sys.modules


def _save_figure_outputs(fig, file_path: Path) -> None:
    """Save the requested figure to disk."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path)


def _draw_path_window(ax, *, lo: int, hi: int, color: str = "tab:orange") -> None:
    """Shade one selected stable EI window on the log-block-size axis."""
    ax.axvspan(np.log(float(lo)), np.log(float(hi)), color=color, alpha=0.08, lw=0.0)


def _finite_path_arrays(
    block_sizes: np.ndarray,
    theta_path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned finite block-size and theta-path arrays."""
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
    dpi: int = 1200,
    title: str | None = None,
    save: bool = False,
    close: bool | None = None,
    xlabel: str = "log(block size)",
    ylabel: str = "extremal index",
) -> None:
    """Plot one observed EI path together with its selected stable window."""
    levels, theta = _finite_path_arrays(path.block_sizes, path.theta_path)
    if levels.size == 0:
        raise ValueError("EI path contains no finite theta values to plot.")
    x = np.log(levels.astype(float))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4), dpi=dpi)
    ax.plot(x, theta, color="tab:blue", marker="o", ms=3.2, lw=1.1, label="observed path")
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
    resolved = _resolved_file_path(file_path)
    if save and resolved is not None:
        _save_figure_outputs(fig, resolved)
    if _should_close_figure(close):
        plt.close(fig)


def _default_fit_title(fit: ExtremalIndexEstimate) -> str:
    """Return a concise default title for one EI estimate."""
    return str(fit.method).replace("_", "-")


def _plot_path_aware_fit(ax, fit: ExtremalIndexEstimate) -> None:
    """Draw one path-aware EI estimate from retained path metadata."""
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
    """Draw one threshold-side EI estimate as a point-and-interval summary."""
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
    dpi: int = 1200,
    title: str | None = None,
    save: bool = False,
    close: bool | None = None,
) -> None:
    """Plot one EI fit either as a retained path view or a threshold summary."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4), dpi=dpi)
    if fit.path_level and fit.path_theta:
        _plot_path_aware_fit(ax, fit)
    else:
        _plot_threshold_fit(ax, fit)
    ax.set_title(title or _default_fit_title(fit))
    fig.tight_layout()
    resolved = _resolved_file_path(file_path)
    if save and resolved is not None:
        _save_figure_outputs(fig, resolved)
    if _should_close_figure(close):
        plt.close(fig)


__all__ = ["plot_ei_fit", "plot_ei_path"]
