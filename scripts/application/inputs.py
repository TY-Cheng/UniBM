"""Raw-input and prepared-series assembly for application workflows."""

from __future__ import annotations

import json
from pathlib import Path

from data_prep.fema import (
    nfip_claims_needs_refresh,
    prepare_nfip_claim_series,
)
from data_prep.ghcn import (
    PreparedSeries,
    ghcn_station_data_needs_refresh,
    prepare_hot_dry_series,
    prepare_precipitation_series,
)
from data_prep.usgs import (
    prepare_usgs_streamflow_series,
    usgs_daily_discharge_needs_refresh,
)
from application.metadata import ensure_application_metadata
from application.specs import (
    CLIMATE_APPLICATIONS,
    PAPER_APPLICATIONS,
    ApplicationPreparedInputs,
    ApplicationSpec,
)
from shared.runtime import status


def _load_json(path: Path) -> dict[str, object]:
    with path.open() as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return loaded


def ensure_ghcn_raw_data(raw_dir: Path) -> dict[str, Path]:
    """Validate the canonical GHCN station files without using the network."""
    resolved: dict[str, Path] = {}
    for spec in CLIMATE_APPLICATIONS:
        raw_path = raw_dir / spec.raw_key
        required_elements = (
            ("PRCP",) if spec.key == "houston_hobby_precipitation" else ("PRCP", "TMAX")
        )
        if ghcn_station_data_needs_refresh(
            raw_path,
            required_elements=required_elements,
            expected_station_id=spec.raw_key.removesuffix(".csv.gz"),
        ):
            raise FileNotFoundError(
                f"Canonical GHCN input is missing or invalid: {raw_path}. Run `just refresh-data`."
            )
        status("application", f"using canonical GHCN raw series for {spec.label}")
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
    """Validate the frozen USGS streamgages without using the network."""
    frozen_sites = load_usgs_frozen_sites(metadata_dir)
    resolved: dict[str, Path] = {}
    for state_code, record in frozen_sites.items():
        site_no = record["site_no"]
        raw_path = raw_dir / f"usgs_{site_no}.csv.gz"
        if usgs_daily_discharge_needs_refresh(raw_path):
            raise FileNotFoundError(
                f"Canonical USGS input is missing or invalid: {raw_path}. Run `just refresh-data`."
            )
        status("application", f"using canonical USGS raw series for {record['station_name']}")
        key = "tx_streamflow" if state_code == "TX" else "fl_streamflow"
        resolved[key] = raw_path
    return resolved


def ensure_nfip_raw_data(raw_dir: Path) -> dict[str, Path]:
    """Validate the Texas and Florida NFIP extracts without using the network."""
    resolved: dict[str, Path] = {}
    for state_code in ("TX", "FL"):
        raw_path = raw_dir / f"nfip_claims_{state_code.lower()}.csv.gz"
        if nfip_claims_needs_refresh(raw_path, state_code=state_code):
            raise FileNotFoundError(
                f"Canonical NFIP input is missing or invalid: {raw_path}. Run `just refresh-data`."
            )
        status("application", f"using canonical NFIP raw claims for {state_code}")
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
    specs: tuple[ApplicationSpec, ...] = PAPER_APPLICATIONS,
) -> dict[str, ApplicationPreparedInputs]:
    """Build role-specific prepared series for the requested applications."""
    keys = {spec.key for spec in specs}
    providers = {spec.provider for spec in specs}
    if raw_paths is None:
        status("application", "validating canonical application inputs")
        raw_paths = {}
        if "ghcn" in providers:
            raw_paths.update(ensure_ghcn_raw_data(dirs["DIR_DATA_RAW_GHCN"]))
        if "usgs" in providers:
            raw_paths.update(
                ensure_usgs_raw_data(
                    dirs["DIR_DATA_RAW_USGS"],
                    metadata_dir=dirs["DIR_DATA_METADATA_APPLICATION"],
                )
            )
        if "fema" in providers:
            raw_paths.update(ensure_nfip_raw_data(dirs["DIR_DATA_RAW_FEMA"]))
    inputs: dict[str, ApplicationPreparedInputs] = {}
    if "houston_hobby_precipitation" in keys:
        status("application", "preparing Houston precipitation inputs")
        inputs["houston_hobby_precipitation"] = _shared_prepared_series(
            prepare_precipitation_series(raw_paths["houston_hobby_precipitation"])
        )
    if "phoenix_hot_dry_severity" in keys:
        status("application", "preparing Phoenix hot-dry inputs")
        inputs["phoenix_hot_dry_severity"] = _shared_prepared_series(
            prepare_hot_dry_series(raw_paths["phoenix_hot_dry_severity"])
        )
    if keys.intersection({"tx_streamflow", "fl_streamflow"}):
        frozen_sites = load_usgs_frozen_sites(dirs["DIR_DATA_METADATA_APPLICATION"])
        for state_code, key in (("TX", "tx_streamflow"), ("FL", "fl_streamflow")):
            if key not in keys:
                continue
            status("application", f"preparing {state_code} streamflow inputs")
            inputs[key] = _shared_prepared_series(
                prepare_usgs_streamflow_series(
                    raw_paths[key],
                    state_code=state_code,
                    site_no=frozen_sites[state_code]["site_no"],
                    station_name=frozen_sites[state_code]["station_name"],
                )
            )
    if keys.intersection({"tx_nfip_claims", "fl_nfip_claims"}):
        cpi_path = dirs["DIR_DATA_RAW_CPI"] / "cpi_u_monthly.csv"
        if not cpi_path.is_file():
            raise FileNotFoundError(
                f"Canonical CPI input is missing: {cpi_path}. Run `just refresh-data`."
            )
        for state_code, key in (("TX", "tx_nfip_claims"), ("FL", "fl_nfip_claims")):
            if key not in keys:
                continue
            status("application", f"preparing {state_code} NFIP inputs")
            inputs[key] = ApplicationPreparedInputs(
                **prepare_nfip_claim_series(
                    raw_paths[key],
                    state_code=state_code,
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
