"""Validation for tracked application metadata."""

from __future__ import annotations

import json
from pathlib import Path


def _read_object(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"Canonical application metadata must be a JSON object: {path}.")
    return loaded


def ensure_application_metadata(metadata_dir: Path | str) -> dict[str, Path]:
    """Validate the tracked USGS candidate and frozen-site registries."""
    metadata_dir = Path(metadata_dir)
    outputs = {
        "candidate_sites": metadata_dir / "usgs_candidate_sites.json",
        "frozen_sites": metadata_dir / "usgs_frozen_sites.json",
    }
    for path in outputs.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Canonical application metadata is missing: {path}. Restore the tracked file from Git."
            )
    candidates = _read_object(outputs["candidate_sites"])
    frozen = _read_object(outputs["frozen_sites"])
    states = {"TX", "FL"}
    if set(candidates) != states or set(frozen) != states:
        raise ValueError("Canonical USGS metadata must contain exactly TX and FL.")
    candidate_sites: dict[str, set[tuple[str, str]]] = {}
    for state in states:
        records = candidates[state]
        if not isinstance(records, list) or not records:
            raise ValueError(f"Canonical USGS candidates for {state} must be a nonempty list.")
        pairs: set[tuple[str, str]] = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Canonical USGS candidate for {state} must be an object.")
            site_no = record.get("site_no")
            station_name = record.get("station_name")
            if not all(
                isinstance(value, str) and value.strip() for value in (site_no, station_name)
            ):
                raise ValueError(f"Canonical USGS candidate for {state} has invalid fields.")
            pairs.add((site_no, station_name))
        candidate_sites[state] = pairs
    for state in states:
        record = frozen[state]
        if not isinstance(record, dict):
            raise ValueError(f"Canonical frozen USGS site for {state} must be an object.")
        values = tuple(record.get(field) for field in ("site_no", "station_name", "state_code"))
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError(f"Canonical frozen USGS site for {state} has invalid fields.")
        site_no, station_name, state_code = values
        if state_code != state or (site_no, station_name) not in candidate_sites[state]:
            raise ValueError(
                f"Canonical frozen USGS site for {state} must match its candidate list."
            )
    return outputs


__all__ = [
    "ensure_application_metadata",
]
