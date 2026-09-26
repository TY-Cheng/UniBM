"""GHCN acquisition, quality-controlled observations, and prepared-series records.

Full-calendar climate construction lives in application.normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from .constants import ANALYSIS_END_DATE
from ._io import write_csv_gz_atomic


GHCN_COLUMNS = ["station_id", "date", "element", "value", "mflag", "qflag", "sflag", "obstime"]
GHCN_BY_STATION_ENDPOINT = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station"


@dataclass(frozen=True)
class PreparedSeries:
    """A prepared univariate series plus comparison maxima and provenance metadata."""

    name: str
    value_name: str
    series: pd.Series
    annual_maxima: pd.Series
    metadata: dict[str, Any]

    def to_frame(self) -> pd.DataFrame:
        return self.series.rename(self.value_name).to_frame()


def read_ghcn_station_csv(path: Path | str) -> pd.DataFrame:
    """Read a by-station GHCN CSV or CSV.GZ file."""
    df = pd.read_csv(
        path,
        header=None,
        names=GHCN_COLUMNS,
        usecols=["station_id", "date", "element", "value", "qflag"],
        dtype={
            "station_id": "string",
            "element": "category",
            "qflag": "string",
        },
        low_memory=False,
    )
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["qflag"] = df["qflag"].replace({"": np.nan})
    return df


def download_ghcn_station(station_file: str, output_path: Path | str) -> Path:
    """Download one GHCN station extract truncated at the shared cutoff."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urlretrieve(f"{GHCN_BY_STATION_ENDPOINT}/{station_file}", tmp_path)
        frame = pd.read_csv(tmp_path, header=None, names=GHCN_COLUMNS, low_memory=False)
        dates = pd.to_numeric(frame["date"], errors="coerce")
        cutoff = int(ANALYSIS_END_DATE.replace("-", ""))
        frame = frame.loc[dates <= cutoff]
        if frame.empty:
            raise ValueError(f"GHCN station {station_file} has no data through the cutoff.")
        write_csv_gz_atomic(frame, output_path, header=False)
    finally:
        tmp_path.unlink(missing_ok=True)
    return output_path


def ghcn_station_data_needs_refresh(
    path: Path | str,
    *,
    required_elements: tuple[str, ...] = (),
    expected_station_id: str | None = None,
    min_rows: int = 365,
    min_span_days: int = 365 * 5,
) -> bool:
    """Return whether an on-disk GHCN station extract looks unusable.

    This is a lightweight integrity guard for cached application inputs. It is
    intentionally conservative: existing files are reused unless they are
    missing, unreadable, empty, obviously too short, or do not contain the
    required GHCN elements for the downstream application.
    """
    path = Path(path)
    if not path.exists():
        return True
    try:
        df = read_ghcn_station_csv(path)
    except Exception:
        return True
    if df.empty or "date" not in df.columns or "element" not in df.columns:
        return True
    if int(df.shape[0]) < int(min_rows):
        return True
    date_index = pd.DatetimeIndex(df["date"]).dropna()
    if date_index.empty:
        return True
    span_days = int((date_index.max() - date_index.min()).days)
    if span_days < int(min_span_days):
        return True
    if required_elements:
        available = {str(element) for element in df["element"].dropna().astype(str).unique()}
        if not set(required_elements).issubset(available):
            return True
    if expected_station_id is not None:
        station_ids = set(df["station_id"].dropna().astype(str).unique())
        if station_ids != {expected_station_id}:
            return True
    return False


def _extract_ghcn_element(df: pd.DataFrame, element: str, *, scale: float) -> pd.Series:
    """Extract one quality-controlled GHCN element as a dated numeric series."""
    sub = df.loc[df["element"] == element].copy()
    sub = sub[sub["qflag"].isna()]
    values = pd.to_numeric(sub["value"], errors="coerce").replace(-9999, np.nan) / scale
    series = pd.Series(
        values.to_numpy(),
        index=pd.DatetimeIndex(sub["date"]),
    )
    return series[~series.index.duplicated(keep="last")].sort_index()
