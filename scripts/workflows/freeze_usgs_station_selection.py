"""Freeze the manuscript USGS streamflow sites from a curated candidate pool."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    from import_bootstrap import ensure_scripts_on_path_from_entry

    ensure_scripts_on_path_from_entry(__file__)

from config import resolve_repo_dirs
from data_prep.usgs import (
    download_usgs_daily_discharge,
    prepare_usgs_streamflow_series,
    usgs_daily_discharge_needs_refresh,
)
from workflows.application_metadata import ensure_application_metadata
from workflows.application_screening import screen_extreme_series
from workflows.workflow_runtime import resolve_bool_env, status


def _load_candidate_sites(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("USGS candidate-site registry must be a mapping by state code.")
    return {
        str(state).upper(): [dict(record) for record in records]
        for state, records in data.items()
        if isinstance(records, list)
    }


def _failed_screening_row(
    *,
    state_code: str,
    site_no: str,
    station_name: str,
    raw_file: Path,
    error: Exception,
) -> dict[str, object]:
    """Return a deterministic fallback row when one candidate cannot be screened."""
    return {
        "state_code": state_code,
        "site_no": site_no,
        "station_name": station_name,
        "raw_file": str(raw_file),
        "name": f"{state_code}_{site_no}",
        "n_obs": 0,
        "n_years": 0.0,
        "start": "",
        "end": "",
        "daily_positive_share": np.nan,
        "maxima_positive_share": np.nan,
        "seasonality_strength": np.nan,
        "xi_hat": np.nan,
        "xi_lower": -999.0,
        "xi_upper": np.nan,
        "plateau_bounds": (-1, -1),
        "plateau_points": 0,
        "supports_frechet_working_model": False,
        "recommended": False,
        "screen_error": str(error),
    }


def _candidate_screening_rows(
    *,
    state_code: str,
    candidates: list[dict[str, str]],
    raw_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    force_refresh = resolve_bool_env("UNIBM_FORCE_REFRESH_APPLICATION_DATA", default=False)
    for candidate in candidates:
        site_no = str(candidate["site_no"])
        station_name = str(candidate["station_name"])
        raw_file = raw_dir / f"usgs_{site_no}.csv.gz"
        try:
            if force_refresh or usgs_daily_discharge_needs_refresh(raw_file):
                verb = "force-refreshing" if force_refresh and raw_file.exists() else "refreshing"
                status("freeze_usgs", f"{verb} USGS raw series {site_no} ({station_name})")
                download_usgs_daily_discharge(site_no, raw_file)
            else:
                status("freeze_usgs", f"reusing USGS raw series {site_no} ({station_name})")
            prepared = prepare_usgs_streamflow_series(
                raw_file,
                state_code=state_code,
                site_no=site_no,
                station_name=station_name,
            )
            review = screen_extreme_series(prepared.series, name=f"{state_code}_{site_no}")
            status(
                "freeze_usgs",
                f"screened {state_code} {site_no}: recommended={review.recommended}, "
                f"plateau_points={review.plateau_points}, xi_lo={review.xi_lower:.3f}",
            )
            rows.append(
                {
                    "state_code": state_code,
                    "site_no": site_no,
                    "station_name": station_name,
                    "raw_file": str(raw_file),
                    **review.to_record(),
                }
            )
        except Exception as exc:
            status("freeze_usgs", f"skipped {state_code} {site_no} due to error: {exc}")
            rows.append(
                _failed_screening_row(
                    state_code=state_code,
                    site_no=site_no,
                    station_name=station_name,
                    raw_file=raw_file,
                    error=exc,
                )
            )
    return rows


def _rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["site_no_numeric"] = pd.to_numeric(ranked["site_no"], errors="coerce")
    return ranked.sort_values(
        [
            "state_code",
            "recommended",
            "supports_frechet_working_model",
            "plateau_points",
            "n_years",
            "xi_lower",
            "site_no_numeric",
        ],
        ascending=[True, False, False, False, False, False, True],
    ).reset_index(drop=True)


def freeze_usgs_station_selection(root: Path | str = ".") -> dict[str, Path]:
    """Screen curated USGS candidates and freeze the top-ranked TX/FL sites."""
    dirs = resolve_repo_dirs(root)
    raw_dir = dirs["DIR_DATA_RAW_USGS"]
    metadata_dir = dirs["DIR_DATA_METADATA_APPLICATION"]
    out_dir = dirs["DIR_OUT_APPLICATIONS"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_application_metadata(metadata_dir)

    candidate_path = metadata_dir / "usgs_candidate_sites.json"
    frozen_path = metadata_dir / "usgs_frozen_sites.json"
    status("freeze_usgs", f"loading candidate sites from {candidate_path}")
    candidates = _load_candidate_sites(candidate_path)
    rows = [
        row
        for state_code, state_candidates in candidates.items()
        for row in _candidate_screening_rows(
            state_code=state_code,
            candidates=state_candidates,
            raw_dir=raw_dir,
        )
    ]
    ranked = _rank_candidates(pd.DataFrame(rows))
    status("freeze_usgs", "writing ranked USGS site screening table")
    ranked.to_csv(out_dir / "application_usgs_site_screening.csv", index=False)

    frozen: dict[str, dict[str, object]] = {}
    for state_code, group in ranked.groupby("state_code", sort=True):
        winner = group.iloc[0]
        status(
            "freeze_usgs",
            f"selected {state_code} site {winner['site_no']} ({winner['station_name']})",
        )
        frozen[state_code] = {
            "site_no": str(winner["site_no"]),
            "station_name": str(winner["station_name"]),
            "state_code": str(state_code),
            "screening_source": "freeze_usgs_station_selection",
        }
    with frozen_path.open("w") as fh:
        json.dump(frozen, fh, indent=2)
    status("freeze_usgs", f"wrote frozen site registry to {frozen_path}")
    return {
        "screening": out_dir / "application_usgs_site_screening.csv",
        "frozen_sites": frozen_path,
    }


def main() -> None:
    outputs = freeze_usgs_station_selection()
    for name, path in outputs.items():
        status("freeze_usgs", f"{name}: {path}")


if __name__ == "__main__":
    main()
