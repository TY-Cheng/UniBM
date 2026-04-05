"""Application-side main entrypoint for screening, fitting, and figure export.

This module keeps the real-data application workflow separate from the
synthetic benchmark code. The manuscript workflow can then orchestrate both
without mixing benchmark and application logic in one large file.

Responsibilities here are intentionally application-only:

- ensure the required raw GHCN station files exist locally;
- materialize derived application series;
- fit the manuscript application analyses on those real series;
- write application CSV outputs and manuscript-facing application figures.
"""
# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from urllib.request import urlretrieve
from typing import Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unibm._runtime import prepare_matplotlib_env

prepare_matplotlib_env("unibm-application")
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

from config import resolve_repo_dirs
from data_prep.ghcn import (
    PreparedSeries,
    materialize_derived_series,
    prepare_hot_dry_series,
    prepare_precipitation_series,
)
import numpy as np
import pandas as pd
from workflows.benchmark_design import METHOD_LOOKUP, fit_methods_for_series
from workflows.application_screening import screen_extreme_series, screening_dataframe
from unibm.core import estimate_evi_quantile, estimate_return_level
from unibm.diagnostics import estimate_extremal_index_reciprocal, target_stability_summary
from unibm.models import ExtremalIndexReciprocalFit, ScalingFit
from unibm.plotting import plot_scaling_fit


@dataclass(frozen=True)
class ApplicationSpec:
    """Static configuration for one manuscript-facing application."""

    key: str
    raw_filename: str
    source_url: str
    loader: Callable[[Path], PreparedSeries]
    observations_per_year: float
    figure_stem: str
    ylabel: str
    time_series_title: str
    scaling_title: str
    scaling_ylabel: str
    quantile: float = 0.5
    target_stability_title: str | None = None


@dataclass(frozen=True)
class ApplicationBundle:
    """Prepared series plus fitted diagnostics used across application outputs."""

    spec: ApplicationSpec
    data: PreparedSeries
    fit: ScalingFit
    extremal_index: ExtremalIndexReciprocalFit


APPLICATIONS = (
    ApplicationSpec(
        key="houston_hobby_precipitation",
        raw_filename="USW00012918.csv.gz",
        source_url="https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USW00012918.csv.gz",
        loader=prepare_precipitation_series,
        observations_per_year=183.0,
        figure_stem="houston_precipitation",
        ylabel="precipitation (mm)",
        time_series_title="Houston wet-season daily precipitation and annual maxima",
        scaling_title="Houston sliding-block quantile scaling",
        scaling_ylabel="log median block maximum",
        target_stability_title="Houston target stability across block sizes",
    ),
    ApplicationSpec(
        key="phoenix_hot_dry_severity",
        raw_filename="USW00023183.csv.gz",
        source_url="https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USW00023183.csv.gz",
        loader=prepare_hot_dry_series,
        observations_per_year=214.0,
        figure_stem="phoenix_hotdry",
        ylabel="hot-dry severity",
        time_series_title="Phoenix warm-season hot-dry severity and annual maxima",
        scaling_title="Phoenix sliding-block quantile scaling",
        scaling_ylabel="log median block maximum",
    ),
)


def ensure_application_raw_data(raw_dir: Path) -> dict[str, Path]:
    """Ensure the manuscript application raw station files exist locally.

    Existing files are reused as-is. Missing files are downloaded directly from
    the public NOAA GHCN by-station endpoint.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for spec in APPLICATIONS:
        raw_path = raw_dir / spec.raw_filename
        if not raw_path.exists():
            temp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
            try:
                urlretrieve(spec.source_url, temp_path)
                temp_path.replace(raw_path)
            except (
                Exception
            ) as exc:  # pragma: no cover - network failures are environment-specific
                temp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Failed to download application raw data for {spec.key} from {spec.source_url}."
                ) from exc
        resolved[spec.key] = raw_path
    return resolved


def _plot_daily_and_annual(
    prepared: PreparedSeries,
    *,
    ylabel: str,
    title: str,
    file_path: Path,
) -> None:
    """Write a two-panel time-series/annual-maxima figure for one application."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.2, 5.2), sharex=False, dpi=600)
    axes[0].plot(
        prepared.series.index, prepared.series.values, color="tab:blue", lw=0.6, alpha=0.85
    )
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        prepared.annual_maxima.index,
        prepared.annual_maxima.values,
        marker="o",
        ms=2.4,
        lw=0.8,
        color="tab:red",
    )
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel(f"annual max {ylabel}")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path)
    plt.close(fig)


def _plot_target_stability(
    bundle: ApplicationBundle,
    *,
    title: str,
    file_path: Path,
) -> None:
    """Compare median/mean/mode block summaries on the fitted block-size grid."""
    import matplotlib.pyplot as plt

    summary = target_stability_summary(
        bundle.data.series.values,
        block_sizes=bundle.fit.block_sizes,
        sliding=True,
        quantile=bundle.spec.quantile,
    )
    quantile_column = (
        "median"
        if np.isclose(bundle.spec.quantile, 0.5)
        else f"quantile_tau_{bundle.spec.quantile:.2f}"
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=600)
    ax.plot(
        summary["block_size"],
        summary[quantile_column],
        label="median block quantile"
        if quantile_column == "median"
        else f"block quantile (tau={bundle.spec.quantile:.2f})",
        color="tab:blue",
        lw=1.2,
    )
    ax.plot(summary["block_size"], summary["mean"], label="block mean", color="tab:orange", lw=1.0)
    ax.plot(summary["block_size"], summary["mode"], label="block mode", color="tab:green", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("block size")
    ax.set_ylabel("block-maxima summary")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path)
    plt.close(fig)


def build_application_bundle(spec: ApplicationSpec, raw_dir: Path) -> ApplicationBundle:
    """Prepare one application series and fit the main sliding-median-FGLS model."""
    prepared = spec.loader(raw_dir / spec.raw_filename)
    fit = estimate_evi_quantile(
        prepared.series.values,
        quantile=spec.quantile,
        sliding=True,
        bootstrap_reps=120,
        random_state=7,
    )
    extremal_index = estimate_extremal_index_reciprocal(prepared.series)
    return ApplicationBundle(spec=spec, data=prepared, fit=fit, extremal_index=extremal_index)


def build_application_bundles(raw_dir: Path) -> list[ApplicationBundle]:
    """Build every configured application bundle."""
    return [build_application_bundle(spec, raw_dir) for spec in APPLICATIONS]


def application_summary_record(bundle: ApplicationBundle) -> dict[str, object]:
    """Summarize the main fitted application outputs for CSV/JSON export."""
    return_levels = estimate_return_level(
        bundle.fit,
        years=np.array([1.0, 10.0, 25.0, 50.0]),
        observations_per_year=bundle.spec.observations_per_year,
    )
    return {
        "name": bundle.spec.key,
        "n_obs": int(bundle.data.series.size),
        "start": str(bundle.data.series.index.min().date()),
        "end": str(bundle.data.series.index.max().date()),
        "xi_hat": float(bundle.fit.slope),
        "xi_lo": float(bundle.fit.confidence_interval[0]),
        "xi_hi": float(bundle.fit.confidence_interval[1]),
        "plateau_lo": int(bundle.fit.plateau_bounds[0]),
        "plateau_hi": int(bundle.fit.plateau_bounds[1]),
        "return_level_1y": float(return_levels[0]),
        "return_level_10y": float(return_levels[1]),
        "return_level_25y": float(return_levels[2]),
        "return_level_50y": float(return_levels[3]),
        "eir_northrop": float(bundle.extremal_index.northrop_estimate),
        "eir_bb": float(bundle.extremal_index.bb_estimate),
    }


def application_method_fits(bundle: ApplicationBundle) -> dict[str, ScalingFit]:
    """Fit the benchmark-style method grid to one real application series."""
    return fit_methods_for_series(
        bundle.data.series.values,
        quantile=bundle.spec.quantile,
        random_state=7,
        reuse_fits={"sliding_median_fgls": bundle.fit},
    )


def application_method_rows(bundle: ApplicationBundle) -> list[dict[str, object]]:
    """Create the application-by-method comparison table used in the notebook."""
    rows: list[dict[str, object]] = []
    for method, fit in application_method_fits(bundle).items():
        spec = METHOD_LOOKUP[method]
        if fit.target == "quantile":
            one_year, ten_year = estimate_return_level(
                fit,
                years=np.array([1.0, 10.0]),
                observations_per_year=bundle.spec.observations_per_year,
            )
        else:
            one_year, ten_year = float("nan"), float("nan")
        rows.append(
            {
                "application": bundle.spec.key,
                "method": method,
                "method_label": f"{spec.block_scheme} {spec.summary_target} ({spec.regression})",
                "summary_target": spec.summary_target,
                "block_scheme": spec.block_scheme,
                "regression": spec.regression,
                "xi_hat": float(fit.slope),
                "xi_lo": float(fit.confidence_interval[0]),
                "xi_hi": float(fit.confidence_interval[1]),
                "plateau_lo": int(fit.plateau_bounds[0]),
                "plateau_hi": int(fit.plateau_bounds[1]),
                "one_year_level": float(one_year),
                "ten_year_level": float(ten_year),
            }
        )
    return rows


def write_application_figures(bundle: ApplicationBundle, fig_dir: Path) -> None:
    """Write all manuscript figures associated with one application bundle."""
    _plot_daily_and_annual(
        bundle.data,
        ylabel=bundle.spec.ylabel,
        title=bundle.spec.time_series_title,
        file_path=fig_dir / f"application_ts_{bundle.spec.figure_stem}.pdf",
    )
    plot_scaling_fit(
        bundle.fit,
        file_path=fig_dir / f"application_evi_{bundle.spec.figure_stem}.pdf",
        save=True,
        title=bundle.spec.scaling_title,
        ylabel=bundle.spec.scaling_ylabel,
    )
    if bundle.spec.target_stability_title is not None:
        _plot_target_stability(
            bundle,
            title=bundle.spec.target_stability_title,
            file_path=fig_dir / f"application_target_{bundle.spec.figure_stem}.pdf",
        )


def build_application_outputs(root: Path | str = ".") -> dict[str, Path]:
    """Materialize all application-side CSVs and figures."""
    dirs = resolve_repo_dirs(root)
    raw_dir = dirs["DIR_DATA_RAW_GHCN"]
    derived_dir = dirs["DIR_DATA_DERIVED"]
    metadata_dir = dirs["DIR_DATA_METADATA"]
    out_dir = dirs["DIR_OUT_APPLICATIONS"]
    fig_dir = dirs["DIR_MANUSCRIPT_FIGURE"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw_files = ensure_application_raw_data(raw_dir)
    materialize_derived_series(
        houston_path=raw_files["houston_hobby_precipitation"],
        phoenix_path=raw_files["phoenix_hot_dry_severity"],
        output_dir=derived_dir,
        metadata_dir=metadata_dir,
    )
    bundles = build_application_bundles(raw_dir)

    reviews = screening_dataframe(
        [screen_extreme_series(bundle.data.series, name=bundle.spec.key) for bundle in bundles]
    )
    reviews.to_csv(out_dir / "application_screening.csv", index=False)

    summaries = [application_summary_record(bundle) for bundle in bundles]
    pd.DataFrame(summaries).to_csv(out_dir / "application_summary.csv", index=False)
    with (out_dir / "application_summary.json").open("w") as fh:
        json.dump(summaries, fh, indent=2)

    method_rows = [row for bundle in bundles for row in application_method_rows(bundle)]
    pd.DataFrame(method_rows).to_csv(out_dir / "application_methods.csv", index=False)

    for bundle in bundles:
        write_application_figures(bundle, fig_dir)

    return {
        "application_screening": out_dir / "application_screening.csv",
        "application_summary": out_dir / "application_summary.csv",
        "application_methods": out_dir / "application_methods.csv",
    }


__all__ = [
    "APPLICATIONS",
    "ApplicationBundle",
    "ApplicationSpec",
    "application_method_rows",
    "application_summary_record",
    "build_application_bundle",
    "build_application_bundles",
    "build_application_outputs",
    "ensure_application_raw_data",
    "write_application_figures",
]


def main() -> None:
    outputs = build_application_outputs()
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
