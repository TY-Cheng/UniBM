"""Application-side workflow for manuscript-ready real-data analyses."""
# ruff: noqa: E402

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from urllib.request import urlretrieve

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unibm._runtime import prepare_matplotlib_env

prepare_matplotlib_env("unibm-application")
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

from config import resolve_repo_dirs
from data_prep.fema import download_nfip_claims_state, prepare_nfip_claim_series
from data_prep.ghcn import (
    PreparedSeries,
    materialize_derived_series,
    prepare_hot_dry_series,
    prepare_precipitation_series,
)
from data_prep.usgs import (
    download_usgs_daily_discharge,
    prepare_usgs_streamflow_series,
    usgs_daily_discharge_needs_refresh,
)
import numpy as np
import pandas as pd
from workflows.application_screening import screen_extreme_series, screen_extremal_index_series
from workflows.application_metadata import ensure_application_metadata
from workflows.benchmark_design import METHOD_LABELS, METHOD_LOOKUP, fit_methods_for_series
from unibm.bootstrap import draw_circular_block_bootstrap_samples
from unibm.core import estimate_evi_quantile, estimate_return_level
from unibm.extremal_index import (
    EI_ALPHA,
    EiPreparedBundle,
    ExtremalIndexEstimate,
    bootstrap_bm_ei_path_draws,
    estimate_k_gaps,
    estimate_pooled_bm_ei,
    extract_stable_path_window,
    prepare_ei_bundle,
)
from unibm.models import ScalingFit
from unibm.plotting import plot_scaling_fit


APPLICATION_RANDOM_STATE = 7
APPLICATION_EI_BOOTSTRAP_REPS = 120
RETURN_LEVEL_HORIZONS = np.asarray([1.0, 10.0, 25.0, 50.0], dtype=float)


def _status(message: str) -> None:
    print(f"[application] {message}", flush=True)


@dataclass(frozen=True)
class ApplicationPreparedInputs:
    """Role-specific prepared series used by one manuscript application."""

    display: PreparedSeries
    evi: PreparedSeries
    ei: PreparedSeries


@dataclass(frozen=True)
class ApplicationSpec:
    """Static configuration for one application-facing case study."""

    key: str
    provider: str
    label: str
    figure_stem: str
    raw_key: str
    ylabel: str
    time_series_title: str
    scaling_title: str
    scaling_ylabel: str
    quantile: float = 0.5
    observations_per_year: float | None = None
    return_level_basis: str = "calendar_year"
    return_level_label: str = "return period (years)"
    target_stability_title: str | None = None
    secondary_case: bool = False


@dataclass(frozen=True)
class ApplicationBundle:
    """Prepared series and fitted results for one application."""

    spec: ApplicationSpec
    prepared: ApplicationPreparedInputs
    evi_fit: ScalingFit
    ei_bundle: EiPreparedBundle
    ei_primary: ExtremalIndexEstimate
    ei_comparator: ExtremalIndexEstimate


APPLICATIONS = (
    ApplicationSpec(
        key="houston_hobby_precipitation",
        provider="ghcn",
        label="Houston precipitation",
        figure_stem="houston_precipitation",
        raw_key="USW00012918.csv.gz",
        ylabel="precipitation (mm)",
        time_series_title="Houston wet-season daily precipitation and annual maxima",
        scaling_title="Houston sliding-block quantile scaling",
        scaling_ylabel="log median block maximum",
        observations_per_year=183.0,
        target_stability_title="Houston target stability across block sizes",
    ),
    ApplicationSpec(
        key="phoenix_hot_dry_severity",
        provider="ghcn",
        label="Phoenix hot-dry severity",
        figure_stem="phoenix_hotdry",
        raw_key="USW00023183.csv.gz",
        ylabel="hot-dry severity",
        time_series_title="Phoenix warm-season hot-dry severity and annual maxima",
        scaling_title="Phoenix sliding-block quantile scaling",
        scaling_ylabel="log median block maximum",
        observations_per_year=214.0,
        secondary_case=True,
        target_stability_title="Phoenix target stability across block sizes",
    ),
    ApplicationSpec(
        key="tx_streamflow",
        provider="usgs",
        label="Texas streamflow",
        figure_stem="tx_streamflow",
        raw_key="TX",
        ylabel="discharge (cfs)",
        time_series_title="Texas daily discharge and annual maxima",
        scaling_title="Texas streamflow sliding-block quantile scaling",
        scaling_ylabel="log median block maximum",
        observations_per_year=365.25,
        target_stability_title="Texas streamflow target stability across block sizes",
    ),
    ApplicationSpec(
        key="fl_streamflow",
        provider="usgs",
        label="Florida streamflow",
        figure_stem="fl_streamflow",
        raw_key="FL",
        ylabel="discharge (cfs)",
        time_series_title="Florida daily discharge and annual maxima",
        scaling_title="Florida streamflow sliding-block quantile scaling",
        scaling_ylabel="log median block maximum",
        observations_per_year=365.25,
        target_stability_title="Florida streamflow target stability across block sizes",
    ),
    ApplicationSpec(
        key="tx_nfip_claims",
        provider="fema",
        label="Texas NFIP claims",
        figure_stem="tx_nfip_claims",
        raw_key="TX",
        ylabel="building payouts (2025 USD)",
        time_series_title="Texas NFIP daily building payouts and annual maxima",
        scaling_title="Texas NFIP active-day sliding-block quantile scaling",
        scaling_ylabel="log median block maximum (positive payout days)",
        return_level_basis="claim_active_day",
        return_level_label="claim-active-day return period (years)",
    ),
    ApplicationSpec(
        key="fl_nfip_claims",
        provider="fema",
        label="Florida NFIP claims",
        figure_stem="fl_nfip_claims",
        raw_key="FL",
        ylabel="building payouts (2025 USD)",
        time_series_title="Florida NFIP daily building payouts and annual maxima",
        scaling_title="Florida NFIP active-day sliding-block quantile scaling",
        scaling_ylabel="log median block maximum (positive payout days)",
        return_level_basis="claim_active_day",
        return_level_label="claim-active-day return period (years)",
    ),
)


def _spec_by_key() -> dict[str, ApplicationSpec]:
    return {spec.key: spec for spec in APPLICATIONS}


def _load_json(path: Path) -> dict[str, object]:
    with path.open() as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return loaded


def ensure_ghcn_raw_data(raw_dir: Path) -> dict[str, Path]:
    """Ensure the manuscript GHCN station files exist locally."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for spec in APPLICATIONS:
        if spec.provider != "ghcn":
            continue
        raw_path = raw_dir / spec.raw_key
        if not raw_path.exists():
            temp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
            source_url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{spec.raw_key}"
            _status(f"downloading GHCN raw series for {spec.label}")
            try:
                urlretrieve(source_url, temp_path)
                temp_path.replace(raw_path)
            except (
                Exception
            ) as exc:  # pragma: no cover - network failures are environment-specific
                temp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Failed to download application raw data for {spec.key} from {source_url}."
                ) from exc
        else:
            _status(f"reusing GHCN raw series for {spec.label}")
        resolved[spec.key] = raw_path
    return resolved


def load_usgs_frozen_sites(metadata_dir: Path) -> dict[str, dict[str, str]]:
    """Load the frozen Texas and Florida streamgage selections."""
    ensure_application_metadata(metadata_dir)
    raw = _load_json(metadata_dir / "usgs_frozen_sites.json")
    return {
        str(state).upper(): {
            "site_no": str(record["site_no"]),
            "station_name": str(record["station_name"]),
            "state_code": str(record["state_code"]).upper(),
        }
        for state, record in raw.items()
        if isinstance(record, dict)
    }


def ensure_usgs_raw_data(raw_dir: Path, *, metadata_dir: Path) -> dict[str, Path]:
    """Ensure the frozen USGS streamgages exist locally."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    frozen_sites = load_usgs_frozen_sites(metadata_dir)
    resolved: dict[str, Path] = {}
    for state_code, record in frozen_sites.items():
        site_no = record["site_no"]
        raw_path = raw_dir / f"usgs_{site_no}.csv.gz"
        if usgs_daily_discharge_needs_refresh(raw_path):
            _status(f"refreshing USGS raw series for {record['station_name']} ({site_no})")
            download_usgs_daily_discharge(site_no, raw_path)
        else:
            _status(f"reusing USGS raw series for {record['station_name']} ({site_no})")
        key = "tx_streamflow" if state_code == "TX" else "fl_streamflow"
        resolved[key] = raw_path
    return resolved


def ensure_nfip_raw_data(raw_dir: Path) -> dict[str, Path]:
    """Ensure the Texas and Florida NFIP extracts exist locally."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for state_code in ("TX", "FL"):
        raw_path = raw_dir / f"nfip_claims_{state_code.lower()}.csv.gz"
        if not raw_path.exists():
            _status(f"downloading NFIP raw claims for {state_code}")
            download_nfip_claims_state(state_code, raw_path)
        else:
            _status(f"reusing NFIP raw claims for {state_code}")
        key = "tx_nfip_claims" if state_code == "TX" else "fl_nfip_claims"
        resolved[key] = raw_path
    return resolved


def _shared_prepared_series(prepared: PreparedSeries) -> ApplicationPreparedInputs:
    """Use the same prepared series for display, EVI, and EI."""
    return ApplicationPreparedInputs(display=prepared, evi=prepared, ei=prepared)


def build_application_inputs(
    dirs: dict[str, Path],
    *,
    raw_paths: dict[str, Path] | None = None,
) -> dict[str, ApplicationPreparedInputs]:
    """Build role-specific prepared series for all configured applications."""
    if raw_paths is None:
        ensure_application_metadata(dirs["DIR_DATA_METADATA_APPLICATION"])
        _status("ensuring application raw inputs")
        raw_paths = {
            **ensure_ghcn_raw_data(dirs["DIR_DATA_RAW_GHCN"]),
            **ensure_usgs_raw_data(
                dirs["DIR_DATA_RAW_USGS"],
                metadata_dir=dirs["DIR_DATA_METADATA_APPLICATION"],
            ),
            **ensure_nfip_raw_data(dirs["DIR_DATA_RAW_FEMA"]),
        }
    _status("materializing shared GHCN-derived series")
    materialize_derived_series(
        houston_path=raw_paths["houston_hobby_precipitation"],
        phoenix_path=raw_paths["phoenix_hot_dry_severity"],
        output_dir=dirs["DIR_DATA_DERIVED"],
        metadata_dir=dirs["DIR_DATA_METADATA"],
    )
    frozen_sites = load_usgs_frozen_sites(dirs["DIR_DATA_METADATA_APPLICATION"])
    cpi_path = dirs["DIR_DATA_METADATA_APPLICATION"] / "cpi_u_calendar_2025_base.csv"
    inputs: dict[str, ApplicationPreparedInputs] = {}
    _status("preparing Houston precipitation inputs")
    inputs["houston_hobby_precipitation"] = _shared_prepared_series(
        prepare_precipitation_series(raw_paths["houston_hobby_precipitation"])
    )
    _status("preparing Phoenix hot-dry inputs")
    inputs["phoenix_hot_dry_severity"] = _shared_prepared_series(
        prepare_hot_dry_series(raw_paths["phoenix_hot_dry_severity"])
    )
    _status("preparing Texas streamflow inputs")
    inputs["tx_streamflow"] = _shared_prepared_series(
        prepare_usgs_streamflow_series(
            raw_paths["tx_streamflow"],
            state_code="TX",
            site_no=frozen_sites["TX"]["site_no"],
            station_name=frozen_sites["TX"]["station_name"],
        )
    )
    _status("preparing Florida streamflow inputs")
    inputs["fl_streamflow"] = _shared_prepared_series(
        prepare_usgs_streamflow_series(
            raw_paths["fl_streamflow"],
            state_code="FL",
            site_no=frozen_sites["FL"]["site_no"],
            station_name=frozen_sites["FL"]["station_name"],
        )
    )
    _status("preparing Texas NFIP inputs")
    inputs["tx_nfip_claims"] = ApplicationPreparedInputs(
        **prepare_nfip_claim_series(
            raw_paths["tx_nfip_claims"],
            state_code="TX",
            cpi_table_path=cpi_path,
        )
    )
    _status("preparing Florida NFIP inputs")
    inputs["fl_nfip_claims"] = ApplicationPreparedInputs(
        **prepare_nfip_claim_series(
            raw_paths["fl_nfip_claims"],
            state_code="FL",
            cpi_table_path=cpi_path,
        )
    )
    return inputs


def _application_observations_per_year(bundle: ApplicationBundle) -> float:
    """Return the effective observation rate used in return-level mapping."""
    if bundle.spec.observations_per_year is not None:
        return float(bundle.spec.observations_per_year)
    series = bundle.prepared.evi.series
    n_years = max((series.index.max() - series.index.min()).days / 365.25, 1.0)
    return float(series.size / n_years)


def _application_worker_count(n_tasks: int) -> int:
    """Resolve the application worker count from the environment."""
    requested = os.environ.get("UNIBM_APPLICATION_WORKERS")
    if requested is None:
        requested = os.environ.get("UNIBM_BENCHMARK_WORKERS")
    if requested is not None:
        try:
            workers = max(int(requested), 1)
        except ValueError:
            workers = 1
    else:
        cpu_count = os.cpu_count() or 1
        workers = max(min(cpu_count - 1, n_tasks), 1)
    return int(min(max(workers, 1), max(n_tasks, 1)))


def _build_application_bundle_worker(
    task: tuple[ApplicationSpec, ApplicationPreparedInputs],
) -> ApplicationBundle:
    """Worker wrapper so application bundles can be built in subprocesses."""
    spec, inputs = task
    return build_application_bundle(spec, inputs)


def _materialize_ei_bootstrap_result(
    series: pd.Series,
    *,
    ei_bundle: EiPreparedBundle,
    allow_zeros: bool,
    reps: int = APPLICATION_EI_BOOTSTRAP_REPS,
    random_state: int = APPLICATION_RANDOM_STATE,
) -> dict[str, np.ndarray | None]:
    """Build the pooled BB-sliding FGLS covariance summary for one real series."""
    sample_bank = draw_circular_block_bootstrap_samples(
        series.to_numpy(dtype=float),
        reps=reps,
        random_state=random_state,
    )
    full_draws = bootstrap_bm_ei_path_draws(
        sample_bank.samples,
        block_sizes=ei_bundle.block_sizes,
        allow_zeros=allow_zeros,
    )
    selected_levels, _ = extract_stable_path_window(ei_bundle.paths[("bb", True)])
    full_levels = np.asarray(ei_bundle.block_sizes, dtype=int)
    selected_idx = [int(np.flatnonzero(full_levels == level)[0]) for level in selected_levels]
    selected_draws = full_draws[("bb", True)][:, selected_idx]
    valid_draws = selected_draws[np.all(np.isfinite(selected_draws), axis=1)]
    covariance = None
    if valid_draws.shape[0] >= 2:
        covariance = np.atleast_2d(np.cov(valid_draws, rowvar=False))
    return {
        "block_sizes": selected_levels,
        "samples": valid_draws,
        "covariance": covariance,
    }


def build_application_bundle(
    spec: ApplicationSpec,
    inputs: ApplicationPreparedInputs,
) -> ApplicationBundle:
    """Fit the primary EVI and EI application estimators for one case."""
    _status(f"fitting EVI for {spec.label}")
    evi_fit = estimate_evi_quantile(
        inputs.evi.series.values,
        quantile=spec.quantile,
        sliding=True,
        bootstrap_reps=120,
        random_state=APPLICATION_RANDOM_STATE,
    )
    allow_zeros = spec.provider == "fema"
    _status(f"preparing EI bundle for {spec.label}")
    ei_bundle = prepare_ei_bundle(inputs.ei.series.values, allow_zeros=allow_zeros)
    _status(f"bootstrapping EI covariance for {spec.label}")
    bootstrap_result = _materialize_ei_bootstrap_result(
        inputs.ei.series,
        ei_bundle=ei_bundle,
        allow_zeros=allow_zeros,
        reps=APPLICATION_EI_BOOTSTRAP_REPS,
        random_state=APPLICATION_RANDOM_STATE,
    )
    _status(f"fitting BB-sliding-FGLS EI for {spec.label}")
    ei_primary = estimate_pooled_bm_ei(
        ei_bundle,
        base_path="bb",
        sliding=True,
        regression="FGLS",
        bootstrap_result=bootstrap_result,
    )
    _status(f"fitting K-gaps EI comparator for {spec.label}")
    ei_comparator = estimate_k_gaps(ei_bundle)
    return ApplicationBundle(
        spec=spec,
        prepared=inputs,
        evi_fit=evi_fit,
        ei_bundle=ei_bundle,
        ei_primary=ei_primary,
        ei_comparator=ei_comparator,
    )


def build_application_bundles(
    dirs: dict[str, Path],
    *,
    raw_paths: dict[str, Path] | None = None,
) -> list[ApplicationBundle]:
    """Build every configured application bundle."""
    inputs = build_application_inputs(dirs, raw_paths=raw_paths)
    tasks = [(spec, inputs[spec.key]) for spec in APPLICATIONS]
    workers = _application_worker_count(len(tasks))
    if workers <= 1:
        _status(f"building {len(tasks)} application bundles sequentially")
        return [build_application_bundle(spec, inputs[spec.key]) for spec in APPLICATIONS]
    _status(f"building {len(tasks)} application bundles with {workers} worker processes")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_build_application_bundle_worker, tasks, chunksize=1))


def _role_series_rows(
    bundle: ApplicationBundle,
    *,
    derived_dir: Path,
) -> list[dict[str, object]]:
    """Write and register the role-specific series behind one application."""
    rows: list[dict[str, object]] = []
    series_map = {
        "display": bundle.prepared.display,
        "evi": bundle.prepared.evi,
        "ei": bundle.prepared.ei,
    }
    for role, prepared in series_map.items():
        file_path = derived_dir / "applications" / f"{bundle.spec.key}__{role}.csv.gz"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_frame().to_csv(file_path, compression="gzip")
        rows.append(
            {
                "application": bundle.spec.key,
                "label": bundle.spec.label,
                "provider": bundle.spec.provider,
                "secondary_case": bundle.spec.secondary_case,
                "role": role,
                "series_name": prepared.name,
                "value_name": prepared.value_name,
                "n_obs": int(prepared.series.size),
                "start": str(prepared.series.index.min().date()),
                "end": str(prepared.series.index.max().date()),
                "derived_file": str(file_path),
                **prepared.metadata,
            }
        )
    return rows


def _application_return_level_rows(bundle: ApplicationBundle) -> list[dict[str, object]]:
    """Return long-form return-level summaries for one application."""
    observations_per_year = _application_observations_per_year(bundle)
    horizons = RETURN_LEVEL_HORIZONS
    unconditional = estimate_return_level(
        bundle.evi_fit,
        horizons,
        observations_per_year=observations_per_year,
    )
    adjusted = None
    if bundle.spec.provider != "fema":
        adjusted = estimate_return_level(
            bundle.evi_fit,
            horizons,
            observations_per_year=observations_per_year,
            extremal_index=bundle.ei_primary.theta_hat,
        )
    rows: list[dict[str, object]] = []
    for idx, horizon in enumerate(horizons):
        rows.append(
            {
                "application": bundle.spec.key,
                "label": bundle.spec.label,
                "provider": bundle.spec.provider,
                "return_level_basis": bundle.spec.return_level_basis,
                "horizon_years": float(horizon),
                "return_level": float(unconditional[idx]),
                "return_level_ei_adjusted": (
                    float("nan") if adjusted is None else float(adjusted[idx])
                ),
                "theta_hat": float(bundle.ei_primary.theta_hat),
            }
        )
    return rows


def application_summary_record(bundle: ApplicationBundle) -> dict[str, object]:
    """Summarize the primary EVI/EI application outputs for CSV/JSON export."""
    return {
        "application": bundle.spec.key,
        "label": bundle.spec.label,
        "provider": bundle.spec.provider,
        "secondary_case": bundle.spec.secondary_case,
        "n_display_obs": int(bundle.prepared.display.series.size),
        "n_evi_obs": int(bundle.prepared.evi.series.size),
        "n_ei_obs": int(bundle.prepared.ei.series.size),
        "start": str(bundle.prepared.display.series.index.min().date()),
        "end": str(bundle.prepared.display.series.index.max().date()),
        "xi_hat": float(bundle.evi_fit.slope),
        "xi_lo": float(bundle.evi_fit.confidence_interval[0]),
        "xi_hi": float(bundle.evi_fit.confidence_interval[1]),
        "plateau_lo": int(bundle.evi_fit.plateau_bounds[0]),
        "plateau_hi": int(bundle.evi_fit.plateau_bounds[1]),
        "theta_hat_bb_sliding_fgls": float(bundle.ei_primary.theta_hat),
        "theta_lo_bb_sliding_fgls": float(bundle.ei_primary.confidence_interval[0]),
        "theta_hi_bb_sliding_fgls": float(bundle.ei_primary.confidence_interval[1]),
        "theta_hat_k_gaps": float(bundle.ei_comparator.theta_hat),
        "theta_lo_k_gaps": float(bundle.ei_comparator.confidence_interval[0]),
        "theta_hi_k_gaps": float(bundle.ei_comparator.confidence_interval[1]),
        "mean_cluster_size": float(1.0 / bundle.ei_primary.theta_hat),
        "ei_stable_level_lo": (
            np.nan
            if bundle.ei_primary.stable_window is None
            else float(bundle.ei_primary.stable_window.lo)
        ),
        "ei_stable_level_hi": (
            np.nan
            if bundle.ei_primary.stable_window is None
            else float(bundle.ei_primary.stable_window.hi)
        ),
        "return_level_basis": bundle.spec.return_level_basis,
        "observations_per_year": _application_observations_per_year(bundle),
    }


def application_method_rows(bundle: ApplicationBundle) -> list[dict[str, object]]:
    """Create the EVI method-comparison table used in the notebook/appendix."""
    rows: list[dict[str, object]] = []
    observations_per_year = _application_observations_per_year(bundle)
    fits = fit_methods_for_series(
        bundle.prepared.evi.series.values,
        quantile=bundle.spec.quantile,
        random_state=APPLICATION_RANDOM_STATE,
        reuse_fits={"sliding_median_fgls": bundle.evi_fit},
    )
    for method, fit in fits.items():
        spec = METHOD_LOOKUP[method]
        if fit.target == "quantile":
            one_year, ten_year = estimate_return_level(
                fit,
                years=np.asarray([1.0, 10.0]),
                observations_per_year=observations_per_year,
            )
        else:
            one_year, ten_year = float("nan"), float("nan")
        rows.append(
            {
                "application": bundle.spec.key,
                "provider": bundle.spec.provider,
                "return_level_basis": bundle.spec.return_level_basis,
                "method": method,
                "method_label": METHOD_LABELS[method],
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


def application_ei_method_rows(bundle: ApplicationBundle) -> list[dict[str, object]]:
    """Create the primary formal-EI comparison table for one application."""
    primary = bundle.ei_primary
    comparator = bundle.ei_comparator
    return [
        {
            "application": bundle.spec.key,
            "provider": bundle.spec.provider,
            "method": "bb_sliding_fgls",
            "theta_hat": float(primary.theta_hat),
            "theta_lo": float(primary.confidence_interval[0]),
            "theta_hi": float(primary.confidence_interval[1]),
            "standard_error": float(primary.standard_error),
            "stable_level_lo": (
                np.nan if primary.stable_window is None else float(primary.stable_window.lo)
            ),
            "stable_level_hi": (
                np.nan if primary.stable_window is None else float(primary.stable_window.hi)
            ),
            "mean_cluster_size": float(1.0 / primary.theta_hat),
            "ci_method": primary.ci_method,
            "ci_variant": primary.ci_variant,
        },
        {
            "application": bundle.spec.key,
            "provider": bundle.spec.provider,
            "method": "k_gaps",
            "theta_hat": float(comparator.theta_hat),
            "theta_lo": float(comparator.confidence_interval[0]),
            "theta_hi": float(comparator.confidence_interval[1]),
            "standard_error": float(comparator.standard_error),
            "stable_level_lo": np.nan,
            "stable_level_hi": np.nan,
            "mean_cluster_size": float(1.0 / comparator.theta_hat),
            "ci_method": comparator.ci_method,
            "ci_variant": comparator.ci_variant,
        },
    ]


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
        prepared.series.index,
        prepared.series.values,
        color="tab:blue",
        lw=0.6,
        alpha=0.85,
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

    from unibm.diagnostics import target_stability_summary

    summary = target_stability_summary(
        bundle.prepared.evi.series.values,
        block_sizes=bundle.evi_fit.block_sizes,
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


def _plot_ei_fit(bundle: ApplicationBundle, *, file_path: Path) -> None:
    """Plot the BB sliding EI path with pooled and K-gaps estimates overlaid."""
    import matplotlib.pyplot as plt

    path = bundle.ei_bundle.paths[("bb", True)]
    finite_mask = np.isfinite(path.theta_path)
    levels = path.block_sizes[finite_mask].astype(float)
    theta_path = path.theta_path[finite_mask].astype(float)
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=600)
    ax.plot(
        np.log(levels),
        theta_path,
        color="tab:red",
        marker="D",
        ms=3.5,
        lw=1.2,
        label="BB sliding path",
    )
    if bundle.ei_primary.stable_window is not None:
        lo = float(bundle.ei_primary.stable_window.lo)
        hi = float(bundle.ei_primary.stable_window.hi)
        ax.axvspan(np.log(lo), np.log(hi), color="tab:red", alpha=0.12, label="stable window")
    ax.axhline(
        bundle.ei_primary.theta_hat,
        color="tab:red",
        lw=1.2,
        linestyle="-",
        label="BB-sliding-FGLS",
    )
    ax.axhline(
        bundle.ei_comparator.theta_hat,
        color="tab:green",
        lw=1.2,
        linestyle="--",
        label="K-gaps",
    )
    ax.set_xlabel("log(block size)")
    ax.set_ylabel("extremal index")
    ax.set_title(f"{bundle.spec.label} extremal-index path")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path)
    plt.close(fig)


def _plot_return_levels(bundle: ApplicationBundle, *, file_path: Path) -> None:
    """Plot return levels for one application."""
    import matplotlib.pyplot as plt

    rows = pd.DataFrame(_application_return_level_rows(bundle))
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=600)
    ax.plot(
        rows["horizon_years"],
        rows["return_level"],
        marker="o",
        color="tab:blue",
        lw=1.2,
        label="UniBM return level",
    )
    adjusted = rows["return_level_ei_adjusted"].to_numpy(dtype=float)
    if np.any(np.isfinite(adjusted)):
        ax.plot(
            rows["horizon_years"],
            adjusted,
            marker="s",
            color="tab:red",
            lw=1.2,
            label="EI-adjusted return level",
        )
    ax.set_xscale("log")
    ax.set_xlabel(bundle.spec.return_level_label)
    ax.set_ylabel(bundle.spec.ylabel)
    ax.set_title(f"{bundle.spec.label} return levels")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path)
    plt.close(fig)


def _plot_application_overview(
    bundles: list[ApplicationBundle],
    *,
    file_path: Path,
) -> None:
    """Plot cross-application comparisons of xi, theta, and mean cluster size."""
    import matplotlib.pyplot as plt

    labels = [bundle.spec.label for bundle in bundles]
    y = np.arange(len(bundles))
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10.5, 4.8), sharey=True, dpi=600)
    xi = np.asarray([bundle.evi_fit.slope for bundle in bundles], dtype=float)
    xi_lo = np.asarray([bundle.evi_fit.confidence_interval[0] for bundle in bundles], dtype=float)
    xi_hi = np.asarray([bundle.evi_fit.confidence_interval[1] for bundle in bundles], dtype=float)
    theta = np.asarray([bundle.ei_primary.theta_hat for bundle in bundles], dtype=float)
    theta_lo = np.asarray(
        [bundle.ei_primary.confidence_interval[0] for bundle in bundles],
        dtype=float,
    )
    theta_hi = np.asarray(
        [bundle.ei_primary.confidence_interval[1] for bundle in bundles],
        dtype=float,
    )
    cluster = 1.0 / theta
    axes[0].errorbar(
        xi,
        y,
        xerr=np.vstack([xi - xi_lo, xi_hi - xi]),
        fmt="o",
        color="tab:blue",
        capsize=2,
    )
    axes[1].errorbar(
        theta,
        y,
        xerr=np.vstack([theta - theta_lo, theta_hi - theta]),
        fmt="o",
        color="tab:red",
        capsize=2,
    )
    axes[2].scatter(cluster, y, color="tab:purple", s=18)
    axes[0].set_xlabel("xi")
    axes[1].set_xlabel("theta")
    axes[2].set_xlabel("1 / theta")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Application overview: tail severity and clustering")
    fig.tight_layout()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path)
    plt.close(fig)


def write_application_figures(bundle: ApplicationBundle, fig_dir: Path) -> None:
    """Write manuscript-ready application figures for one bundle."""
    _plot_daily_and_annual(
        bundle.prepared.display,
        ylabel=bundle.spec.ylabel,
        title=bundle.spec.time_series_title,
        file_path=fig_dir / f"application_ts_{bundle.spec.figure_stem}.pdf",
    )
    plot_scaling_fit(
        bundle.evi_fit,
        file_path=fig_dir / f"application_evi_{bundle.spec.figure_stem}.pdf",
        save=True,
        title=bundle.spec.scaling_title,
        ylabel=bundle.spec.scaling_ylabel,
    )
    if bundle.spec.target_stability_title is not None and bundle.spec.provider != "fema":
        _plot_target_stability(
            bundle,
            title=bundle.spec.target_stability_title,
            file_path=fig_dir / f"application_target_{bundle.spec.figure_stem}.pdf",
        )
    _plot_ei_fit(bundle, file_path=fig_dir / f"application_ei_{bundle.spec.figure_stem}.pdf")
    _plot_return_levels(
        bundle, file_path=fig_dir / f"application_rl_{bundle.spec.figure_stem}.pdf"
    )


def _provider_metadata_rows(
    bundle: ApplicationBundle,
    *,
    raw_path: Path | None = None,
) -> list[dict[str, object]]:
    """Return provider metadata rows for JSON sidecars."""
    rows: list[dict[str, object]] = []
    for role, prepared in {
        "display": bundle.prepared.display,
        "evi": bundle.prepared.evi,
        "ei": bundle.prepared.ei,
    }.items():
        rows.append(
            {
                "application": bundle.spec.key,
                "provider": bundle.spec.provider,
                "role": role,
                "raw_file": None if raw_path is None else str(raw_path),
                **prepared.metadata,
            }
        )
    return rows


def _usgs_site_audit_frame(metadata_dir: Path) -> pd.DataFrame:
    """Build a lightweight shortlist audit from the candidate and frozen registries."""
    candidate_map = _load_json(metadata_dir / "usgs_candidate_sites.json")
    frozen_map = load_usgs_frozen_sites(metadata_dir)
    rows: list[dict[str, object]] = []
    for state_code, candidates in candidate_map.items():
        frozen_site = frozen_map.get(str(state_code).upper(), {}).get("site_no")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            rows.append(
                {
                    "state_code": str(state_code).upper(),
                    "site_no": str(candidate.get("site_no", "")),
                    "station_name": str(candidate.get("station_name", "")),
                    "selected": str(candidate.get("site_no", "")) == str(frozen_site),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["state_code", "selected", "site_no"], ascending=[True, False, True]
    )


def build_application_outputs(root: Path | str = ".") -> dict[str, Path]:
    """Materialize all application-side CSVs, metadata, and figures."""
    dirs = resolve_repo_dirs(root)
    metadata_app_dir = dirs["DIR_DATA_METADATA_APPLICATION"]
    derived_dir = dirs["DIR_DATA_DERIVED"]
    out_dir = dirs["DIR_OUT_APPLICATIONS"]
    fig_dir = dirs["DIR_MANUSCRIPT_FIGURE"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    metadata_app_dir.mkdir(parents=True, exist_ok=True)
    ensure_application_metadata(metadata_app_dir)

    _status("ensuring raw inputs")
    raw_paths = {
        **ensure_ghcn_raw_data(dirs["DIR_DATA_RAW_GHCN"]),
        **ensure_usgs_raw_data(
            dirs["DIR_DATA_RAW_USGS"],
            metadata_dir=metadata_app_dir,
        ),
        **ensure_nfip_raw_data(dirs["DIR_DATA_RAW_FEMA"]),
    }
    _status("building application bundles")
    bundles = build_application_bundles(dirs, raw_paths=raw_paths)
    series_registry_rows: list[dict[str, object]] = []
    screening_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    return_level_rows: list[dict[str, object]] = []
    method_rows: list[dict[str, object]] = []
    ei_method_rows: list[dict[str, object]] = []
    provider_metadata: dict[str, list[dict[str, object]]] = {"ghcn": [], "usgs": [], "fema": []}

    for bundle in bundles:
        _status(f"collecting outputs for {bundle.spec.label}")
        series_registry_rows.extend(_role_series_rows(bundle, derived_dir=derived_dir))
        provider_metadata[bundle.spec.provider].extend(
            _provider_metadata_rows(bundle, raw_path=raw_paths.get(bundle.spec.key))
        )
        evi_review = screen_extreme_series(
            bundle.prepared.evi.series, name=bundle.spec.key
        ).to_record()
        evi_review["analysis_type"] = "evi"
        screening_rows.append(evi_review)
        ei_review = screen_extremal_index_series(
            bundle.prepared.ei.series,
            name=bundle.spec.key,
            allow_zeros=(bundle.spec.provider == "fema"),
        ).to_record()
        screening_rows.append(ei_review)
        summary_rows.append(application_summary_record(bundle))
        return_level_rows.extend(_application_return_level_rows(bundle))
        method_rows.extend(application_method_rows(bundle))
        ei_method_rows.extend(application_ei_method_rows(bundle))
        _status(f"writing figures for {bundle.spec.label}")
        write_application_figures(bundle, fig_dir)

    _status("writing application tables and metadata")
    series_registry = pd.DataFrame(series_registry_rows).sort_values(["application", "role"])
    series_registry.to_csv(out_dir / "application_series_registry.csv", index=False)

    screening = pd.DataFrame(screening_rows).sort_values(["name", "analysis_type"])
    screening.to_csv(out_dir / "application_screening.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values(["provider", "application"])
    summary.to_csv(out_dir / "application_summary.csv", index=False)
    with (out_dir / "application_summary.json").open("w") as fh:
        json.dump(summary_rows, fh, indent=2)

    pd.DataFrame(return_level_rows).sort_values(["application", "horizon_years"]).to_csv(
        out_dir / "application_return_levels.csv",
        index=False,
    )
    _usgs_site_audit_frame(metadata_app_dir).to_csv(
        out_dir / "application_usgs_site_screening.csv",
        index=False,
    )
    pd.DataFrame(method_rows).sort_values(["application", "method"]).to_csv(
        out_dir / "application_methods.csv",
        index=False,
    )
    pd.DataFrame(ei_method_rows).sort_values(["application", "method"]).to_csv(
        out_dir / "application_ei_methods.csv",
        index=False,
    )
    for provider, rows in provider_metadata.items():
        with (metadata_app_dir / f"{provider}_sources.json").open("w") as fh:
            json.dump(rows, fh, indent=2)

    _status("writing cross-application overview figure")
    _plot_application_overview(bundles, file_path=fig_dir / "application_overview.pdf")
    return {
        "application_series_registry": out_dir / "application_series_registry.csv",
        "application_screening": out_dir / "application_screening.csv",
        "application_summary": out_dir / "application_summary.csv",
        "application_return_levels": out_dir / "application_return_levels.csv",
        "application_methods": out_dir / "application_methods.csv",
        "application_ei_methods": out_dir / "application_ei_methods.csv",
        "application_usgs_site_screening": out_dir / "application_usgs_site_screening.csv",
    }


__all__ = [
    "APPLICATIONS",
    "ApplicationBundle",
    "ApplicationPreparedInputs",
    "ApplicationSpec",
    "application_ei_method_rows",
    "application_method_rows",
    "application_summary_record",
    "build_application_bundle",
    "build_application_bundles",
    "build_application_inputs",
    "build_application_outputs",
    "ensure_ghcn_raw_data",
    "ensure_nfip_raw_data",
    "ensure_usgs_raw_data",
    "load_usgs_frozen_sites",
    "write_application_figures",
]


def main() -> None:
    outputs = build_application_outputs()
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
