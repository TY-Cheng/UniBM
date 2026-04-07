"""Raw-input and prepared-series assembly for application workflows."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

from data_prep.fema import (
    download_nfip_claims_state,
    nfip_claims_needs_refresh,
    prepare_nfip_claim_series,
)
from data_prep.ghcn import (
    PreparedSeries,
    ghcn_station_data_needs_refresh,
    materialize_derived_series,
    prepare_hot_dry_series,
    prepare_precipitation_series,
)
from data_prep.usgs import (
    download_usgs_daily_discharge,
    prepare_usgs_streamflow_series,
    usgs_daily_discharge_needs_refresh,
)
from application.metadata import ensure_application_metadata
from application.specs import APPLICATIONS, ApplicationPreparedInputs
from shared.runtime import resolve_bool_env, status


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
    force_refresh = resolve_bool_env("UNIBM_FORCE_REFRESH_APPLICATION_DATA", default=False)
    for spec in APPLICATIONS:
        if spec.provider != "ghcn":
            continue
        raw_path = raw_dir / spec.raw_key
        required_elements = (
            ("PRCP",) if spec.key == "houston_hobby_precipitation" else ("PRCP", "TMAX")
        )
        needs_refresh = force_refresh or ghcn_station_data_needs_refresh(
            raw_path,
            required_elements=required_elements,
        )
        if needs_refresh:
            temp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
            source_url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{spec.raw_key}"
            action = "refreshing" if raw_path.exists() else "downloading"
            if force_refresh and raw_path.exists():
                action = "force-refreshing"
            status("application", f"{action} GHCN raw series for {spec.label}")
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
            status("application", f"reusing GHCN raw series for {spec.label}")
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
    force_refresh = resolve_bool_env("UNIBM_FORCE_REFRESH_APPLICATION_DATA", default=False)
    for state_code, record in frozen_sites.items():
        site_no = record["site_no"]
        raw_path = raw_dir / f"usgs_{site_no}.csv.gz"
        if force_refresh or usgs_daily_discharge_needs_refresh(raw_path):
            status(
                "application",
                f"{'force-refreshing' if force_refresh and raw_path.exists() else 'refreshing'} "
                f"USGS raw series for {record['station_name']} ({site_no})",
            )
            download_usgs_daily_discharge(site_no, raw_path)
        else:
            status(
                "application", f"reusing USGS raw series for {record['station_name']} ({site_no})"
            )
        key = "tx_streamflow" if state_code == "TX" else "fl_streamflow"
        resolved[key] = raw_path
    return resolved


def ensure_nfip_raw_data(raw_dir: Path) -> dict[str, Path]:
    """Ensure the Texas and Florida NFIP extracts exist locally."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    force_refresh = resolve_bool_env("UNIBM_FORCE_REFRESH_APPLICATION_DATA", default=False)
    for state_code in ("TX", "FL"):
        raw_path = raw_dir / f"nfip_claims_{state_code.lower()}.csv.gz"
        if force_refresh or nfip_claims_needs_refresh(raw_path, state_code=state_code):
            action = "refreshing" if raw_path.exists() else "downloading"
            if force_refresh and raw_path.exists():
                action = "force-refreshing"
            status("application", f"{action} NFIP raw claims for {state_code}")
            download_nfip_claims_state(state_code, raw_path, force_refresh=force_refresh)
        else:
            status("application", f"reusing NFIP raw claims for {state_code}")
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
        status("application", "ensuring application raw inputs")
        raw_paths = {
            **ensure_ghcn_raw_data(dirs["DIR_DATA_RAW_GHCN"]),
            **ensure_usgs_raw_data(
                dirs["DIR_DATA_RAW_USGS"],
                metadata_dir=dirs["DIR_DATA_METADATA_APPLICATION"],
            ),
            **ensure_nfip_raw_data(dirs["DIR_DATA_RAW_FEMA"]),
        }
    status("application", "materializing shared GHCN-derived series")
    materialize_derived_series(
        houston_path=raw_paths["houston_hobby_precipitation"],
        phoenix_path=raw_paths["phoenix_hot_dry_severity"],
        output_dir=dirs["DIR_DATA_DERIVED"],
        metadata_dir=dirs["DIR_DATA_METADATA"],
    )
    frozen_sites = load_usgs_frozen_sites(dirs["DIR_DATA_METADATA_APPLICATION"])
    cpi_path = dirs["DIR_DATA_METADATA_APPLICATION"] / "cpi_u_calendar_2025_base.csv"
    inputs: dict[str, ApplicationPreparedInputs] = {}
    status("application", "preparing Houston precipitation inputs")
    inputs["houston_hobby_precipitation"] = _shared_prepared_series(
        prepare_precipitation_series(raw_paths["houston_hobby_precipitation"])
    )
    status("application", "preparing Phoenix hot-dry inputs")
    inputs["phoenix_hot_dry_severity"] = _shared_prepared_series(
        prepare_hot_dry_series(raw_paths["phoenix_hot_dry_severity"])
    )
    status("application", "preparing Texas streamflow inputs")
    inputs["tx_streamflow"] = _shared_prepared_series(
        prepare_usgs_streamflow_series(
            raw_paths["tx_streamflow"],
            state_code="TX",
            site_no=frozen_sites["TX"]["site_no"],
            station_name=frozen_sites["TX"]["station_name"],
        )
    )
    status("application", "preparing Florida streamflow inputs")
    inputs["fl_streamflow"] = _shared_prepared_series(
        prepare_usgs_streamflow_series(
            raw_paths["fl_streamflow"],
            state_code="FL",
            site_no=frozen_sites["FL"]["site_no"],
            station_name=frozen_sites["FL"]["station_name"],
        )
    )
    status("application", "preparing Texas NFIP inputs")
    inputs["tx_nfip_claims"] = ApplicationPreparedInputs(
        **prepare_nfip_claim_series(
            raw_paths["tx_nfip_claims"],
            state_code="TX",
            cpi_table_path=cpi_path,
        )
    )
    status("application", "preparing Florida NFIP inputs")
    inputs["fl_nfip_claims"] = ApplicationPreparedInputs(
        **prepare_nfip_claim_series(
            raw_paths["fl_nfip_claims"],
            state_code="FL",
            cpi_table_path=cpi_path,
        )
    )
    return inputs


__all__ = [
    "build_application_inputs",
    "ensure_ghcn_raw_data",
    "ensure_nfip_raw_data",
    "ensure_usgs_raw_data",
    "load_usgs_frozen_sites",
]
