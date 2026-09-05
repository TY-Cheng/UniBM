"""Explicit network refresh for the canonical application inputs."""
# ruff: noqa: E402

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

if __package__ in {None, ""}:
    import importlib.util

    _helper_path = Path(__file__).resolve().parents[1] / "shared" / "import_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("_shared_import_bootstrap", _helper_path)
    if _spec is None or _spec.loader is None:  # pragma: no cover
        raise ImportError(f"Could not load import bootstrap helper from {_helper_path}.")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    _module.ensure_scripts_on_path_from_entry(__file__)

from application.freeze_usgs import _load_candidate_sites
from application.metadata import ensure_application_metadata
from application.specs import CLIMATE_APPLICATIONS
from config import resolve_repo_dirs
from data_prep.constants import ANALYSIS_END_DATE
from data_prep.cpi import BLS_API_ENDPOINT, CPI_U_SERIES_ID, download_monthly_cpi
from data_prep.fema import (
    DEFAULT_NFIP_START_DATE,
    OPENFEMA_NFIP_CLAIMS_ENDPOINT,
    download_nfip_claims_state,
)
from data_prep.ghcn import GHCN_BY_STATION_ENDPOINT, download_ghcn_station
from data_prep.usgs import USGS_DV_ENDPOINT, download_usgs_daily_discharge
from shared.runtime import status


def _assert_data_tree_clean(repo_root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", "data"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "The data tree is dirty; review or revert it before `just refresh-data`."
        )


def refresh_application_data(root: Path | str = ".") -> dict[str, Path]:
    """Refresh all provider extracts through the fixed analysis cutoff."""
    dirs = resolve_repo_dirs(root)
    _assert_data_tree_clean(dirs["DIR_WORK"])
    metadata = ensure_application_metadata(dirs["DIR_DATA_METADATA_APPLICATION"])

    outputs: dict[str, Path] = {}
    for spec in CLIMATE_APPLICATIONS:
        path = dirs["DIR_DATA_RAW_GHCN"] / spec.raw_key
        status("refresh-data", f"downloading GHCN {spec.raw_key}")
        outputs[spec.key] = download_ghcn_station(spec.raw_key, path)

    candidates = _load_candidate_sites(metadata["candidate_sites"])
    for records in candidates.values():
        for record in records:
            site_no = str(record["site_no"])
            path = dirs["DIR_DATA_RAW_USGS"] / f"usgs_{site_no}.csv.gz"
            status("refresh-data", f"downloading USGS {site_no}")
            outputs[f"usgs_{site_no}"] = download_usgs_daily_discharge(site_no, path)

    for state_code in ("TX", "FL"):
        path = dirs["DIR_DATA_RAW_FEMA"] / f"nfip_claims_{state_code.lower()}.csv.gz"
        status("refresh-data", f"downloading NFIP {state_code}")
        outputs[f"nfip_{state_code.lower()}"] = download_nfip_claims_state(
            state_code,
            path,
            force_refresh=True,
        )

    cpi_path = dirs["DIR_DATA_RAW_CPI"] / "cpi_u_monthly.csv"
    status("refresh-data", f"downloading BLS {CPI_U_SERIES_ID}")
    outputs["cpi_u_monthly"] = download_monthly_cpi(cpi_path)

    sources_path = dirs["DIR_DATA_METADATA"] / "sources.json"
    sources_path.parent.mkdir(parents=True, exist_ok=True)
    sources_path.write_text(
        json.dumps(
            {
                "analysis_end_date": ANALYSIS_END_DATE,
                "nfip_start_date": DEFAULT_NFIP_START_DATE,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "sources": {
                    "ghcn": GHCN_BY_STATION_ENDPOINT,
                    "usgs": USGS_DV_ENDPOINT,
                    "nfip": OPENFEMA_NFIP_CLAIMS_ENDPOINT,
                    "cpi_u": BLS_API_ENDPOINT,
                },
                "cpi_series_id": CPI_U_SERIES_ID,
                "cpi_2025_10": "geometric mean of 2025-09 and 2025-11; BLS unavailable",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["sources"] = sources_path
    return outputs


def main() -> None:
    for name, path in refresh_application_data().items():
        status("refresh-data", f"{name}: {path}")


if __name__ == "__main__":
    main()
