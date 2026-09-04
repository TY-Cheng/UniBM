"""Freeze the manuscript USGS streamflow sites from a curated candidate pool."""
# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    import importlib.util

    _helper_path = Path(__file__).resolve().parents[1] / "shared" / "import_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("_shared_import_bootstrap", _helper_path)
    if _spec is None or _spec.loader is None:  # pragma: no cover - import bootstrap failure
        raise ImportError(f"Could not load import bootstrap helper from {_helper_path}.")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    _module.ensure_scripts_on_path_from_entry(__file__)

from config import resolve_repo_dirs
from data_prep.usgs import (
    prepare_usgs_streamflow_series,
    usgs_daily_discharge_needs_refresh,
)
from application.metadata import ensure_application_metadata
from application.screening import screen_extreme_series
from shared.runtime import status


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


def _candidate_screening_rows(
    *,
    state_code: str,
    candidates: list[dict[str, str]],
    raw_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        site_no = str(candidate["site_no"])
        station_name = str(candidate["station_name"])
        raw_file = raw_dir / f"usgs_{site_no}.csv.gz"
        if usgs_daily_discharge_needs_refresh(raw_file):
            raise FileNotFoundError(
                f"Canonical USGS input is missing or invalid: {raw_file}. Run `just refresh-data`."
            )
        status("freeze_usgs", f"using canonical USGS raw series {site_no} ({station_name})")
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
        eligible = group[group["recommended"].eq(True)]
        if eligible.empty:
            raise ValueError(f"{state_code} has no recommended USGS candidate.")
        winner = eligible.iloc[0]
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
