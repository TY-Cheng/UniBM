"""Default application metadata assets for the SERRA-facing workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEFAULT_USGS_CANDIDATE_SITES: dict[str, list[dict[str, str]]] = {
    "TX": [
        {"site_no": "08114000", "station_name": "Brazos River at Richmond, TX"},
        {"site_no": "08066500", "station_name": "Trinity River at Romayor, TX"},
        {"site_no": "08158000", "station_name": "Colorado River at Austin, TX"},
    ],
    "FL": [
        {"site_no": "02236000", "station_name": "St. Johns River near Deland, FL"},
        {"site_no": "02320500", "station_name": "Suwannee River at Branford, FL"},
        {"site_no": "02366500", "station_name": "Choctawhatchee River near Bruce, FL"},
    ],
}

DEFAULT_USGS_FROZEN_SITES: dict[str, dict[str, str]] = {
    "TX": {
        "site_no": "08066500",
        "station_name": "Trinity River at Romayor, TX",
        "state_code": "TX",
        "screening_source": "default_application_metadata",
    },
    "FL": {
        "site_no": "02366500",
        "station_name": "Choctawhatchee River near Bruce, FL",
        "state_code": "FL",
        "screening_source": "default_application_metadata",
    },
}

DEFAULT_CPI_2025_BASE_ROWS: list[tuple[int, float]] = [
    (1978, 20.252),
    (1979, 22.550),
    (1980, 25.596),
    (1981, 28.237),
    (1982, 29.978),
    (1983, 30.941),
    (1984, 32.266),
    (1985, 33.414),
    (1986, 34.035),
    (1987, 35.277),
    (1988, 36.736),
    (1989, 38.514),
    (1990, 40.597),
    (1991, 42.305),
    (1992, 43.579),
    (1993, 44.883),
    (1994, 46.026),
    (1995, 47.327),
    (1996, 48.723),
    (1997, 49.841),
    (1998, 50.618),
    (1999, 51.734),
    (2000, 53.470),
    (2001, 54.991),
    (2002, 55.860),
    (2003, 57.137),
    (2004, 58.657),
    (2005, 60.641),
    (2006, 62.594),
    (2007, 64.379),
    (2008, 66.853),
    (2009, 66.615),
    (2010, 67.707),
    (2011, 69.837),
    (2012, 71.287),
    (2013, 72.342),
    (2014, 73.511),
    (2015, 73.598),
    (2016, 74.527),
    (2017, 76.114),
    (2018, 77.973),
    (2019, 79.387),
    (2020, 80.366),
    (2021, 84.142),
    (2022, 90.956),
    (2023, 94.700),
    (2024, 97.493),
    (2025, 100.000),
]


def ensure_application_metadata(metadata_dir: Path | str) -> dict[str, Path]:
    """Create default application metadata files when they are missing."""
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = metadata_dir / "usgs_candidate_sites.json"
    frozen_path = metadata_dir / "usgs_frozen_sites.json"
    cpi_path = metadata_dir / "cpi_u_calendar_2025_base.csv"

    if not candidate_path.exists():
        with candidate_path.open("w") as fh:
            json.dump(DEFAULT_USGS_CANDIDATE_SITES, fh, indent=2)

    if not frozen_path.exists():
        with frozen_path.open("w") as fh:
            json.dump(DEFAULT_USGS_FROZEN_SITES, fh, indent=2)

    if not cpi_path.exists():
        pd.DataFrame(
            DEFAULT_CPI_2025_BASE_ROWS,
            columns=["year", "cpi_2025_base"],
        ).to_csv(cpi_path, index=False)

    return {
        "candidate_sites": candidate_path,
        "frozen_sites": frozen_path,
        "cpi_2025_base": cpi_path,
    }


__all__ = [
    "DEFAULT_CPI_2025_BASE_ROWS",
    "DEFAULT_USGS_CANDIDATE_SITES",
    "DEFAULT_USGS_FROZEN_SITES",
    "ensure_application_metadata",
]
