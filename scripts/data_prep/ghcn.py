"""GHCN daily data preparation for the current manuscript-facing applications.

This module deliberately keeps the application engineering separate from the
UniBM methods package. The Houston and Phoenix series are both derived from
daily GHCN station files, but they represent different environmental use cases:

- native precipitation extremes for a flagship hydroclimatic application;
- a derived hot-dry severity index as a secondary scalarized hazard example.
"""

from __future__ import annotations

import json
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


def _season_index(year: int, months: tuple[int, ...]) -> pd.DatetimeIndex:
    """Return every calendar day in one declared within-year season."""
    if (
        not months
        or tuple(sorted(set(months))) != months
        or any(month < 1 or month > 12 for month in months)
    ):
        raise ValueError(
            "season months must be unique, strictly increasing integers from 1 to 12."
        )
    year_index = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return year_index[year_index.month.isin(months)]


def _complete_season_suffix_years(
    requirements: tuple[tuple[pd.Series, int], ...],
    months: tuple[int, ...],
) -> tuple[int, ...]:
    """Return the consecutive complete-season suffix ending in the cutoff year."""
    if not requirements or any(series.empty for series, _ in requirements):
        raise ValueError("GHCN inputs must contain the required elements.")
    cutoff_year = pd.Timestamp(ANALYSIS_END_DATE).year
    earliest_year = max(int(series.index.min().year) for series, _ in requirements)
    complete: list[int] = []
    for year in range(cutoff_year, earliest_year - 1, -1):
        season = _season_index(year, months)
        valid = True
        for series, lookback_days in requirements:
            required = (
                pd.date_range(season.min() - pd.Timedelta(days=lookback_days), season.max())
                if lookback_days
                else season
            )
            values = series.reindex(required)
            if values.isna().any() or not np.all(np.isfinite(values.to_numpy(dtype=float))):
                valid = False
                break
        if not valid:
            if complete:
                break
            raise ValueError(f"GHCN season {cutoff_year} is incomplete at the analysis cutoff.")
        complete.append(year)
    if not complete:
        raise ValueError("GHCN inputs contain no complete seasons.")
    return tuple(reversed(complete))


def prepare_precipitation_series(
    path: Path | str,
    *,
    wet_season_months: tuple[int, ...] = (6, 7, 8, 9, 10, 11),
    min_wet_day_mm: float = 0.0,
) -> PreparedSeries:
    """Prepare a wet-season precipitation series from daily GHCN data."""
    df = read_ghcn_station_csv(path)
    df = df[df["date"] <= ANALYSIS_END_DATE]
    precipitation = _extract_ghcn_element(df, "PRCP", scale=10.0)
    years = _complete_season_suffix_years(((precipitation, 0),), wet_season_months)
    precipitation = precipitation[
        precipitation.index.year.isin(years) & precipitation.index.month.isin(wet_season_months)
    ]
    if min_wet_day_mm > 0:
        precipitation = precipitation.clip(lower=min_wet_day_mm)
    annual_maxima = precipitation.groupby(precipitation.index.year).max()
    return PreparedSeries(
        name="wet-season daily precipitation",
        value_name="precipitation_mm",
        series=precipitation,
        annual_maxima=annual_maxima,
        metadata={
            "station_id": str(df["station_id"].iloc[0]),
            "station_name": "Houston William P Hobby AP",
            "unit": "mm",
            "wet_season_months": wet_season_months,
            "analysis_end_date": ANALYSIS_END_DATE,
            "season_start_year": years[0],
            "season_end_year": years[-1],
            "maxima_period": "season",
        },
    )


def prepare_hot_dry_series(
    path: Path | str,
    *,
    warm_season_months: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10),
    rolling_days: int = 30,
) -> PreparedSeries:
    """Construct a scalar hot-dry severity index from daily temperature and precipitation."""
    df = read_ghcn_station_csv(path)
    df = df[df["date"] <= ANALYSIS_END_DATE]
    tmax = _extract_ghcn_element(df, "TMAX", scale=10.0)
    precipitation = _extract_ghcn_element(df, "PRCP", scale=10.0)
    years = _complete_season_suffix_years(
        ((tmax, 0), (precipitation, rolling_days - 1)),
        warm_season_months,
    )
    first_season = _season_index(years[0], warm_season_months)
    last_season = _season_index(years[-1], warm_season_months)
    date_index = pd.date_range(
        first_season.min() - pd.Timedelta(days=rolling_days - 1),
        last_season.max(),
        freq="D",
    )
    tmax = tmax.reindex(date_index)
    precipitation = precipitation.reindex(date_index)
    rolling_precipitation = precipitation.rolling(
        rolling_days,
        min_periods=rolling_days,
    ).sum()
    daily = pd.DataFrame(
        {
            "tmax_c": tmax,
            "prcp_mm": precipitation,
            "prcp_roll_mm": rolling_precipitation,
        },
        index=date_index,
    )
    daily = daily[daily.index.year.isin(years) & daily.index.month.isin(warm_season_months)]
    daily["doy"] = daily.index.dayofyear
    tmax_mean = daily.groupby("doy")["tmax_c"].transform("mean")
    tmax_sd = daily.groupby("doy")["tmax_c"].transform("std").replace(0, np.nan)
    roll_mean = daily.groupby("doy")["prcp_roll_mm"].transform("mean")
    roll_sd = daily.groupby("doy")["prcp_roll_mm"].transform("std").replace(0, np.nan)
    hot_anomaly = ((daily["tmax_c"] - tmax_mean) / tmax_sd).replace([np.inf, -np.inf], np.nan)
    dry_anomaly = ((roll_mean - daily["prcp_roll_mm"]) / roll_sd).replace(
        [np.inf, -np.inf], np.nan
    )
    severity = hot_anomaly.clip(lower=0).fillna(0) + dry_anomaly.clip(lower=0).fillna(0)
    annual_maxima = severity.groupby(severity.index.year).max()
    return PreparedSeries(
        name="warm-season hot-dry severity",
        value_name="hot_dry_severity",
        series=severity,
        annual_maxima=annual_maxima,
        metadata={
            "station_id": str(df["station_id"].iloc[0]),
            "station_name": "Phoenix AP",
            "unit": "dimensionless severity",
            "warm_season_months": warm_season_months,
            "rolling_days": rolling_days,
            "rolling_min_periods": rolling_days,
            "analysis_end_date": ANALYSIS_END_DATE,
            "season_start_year": years[0],
            "season_end_year": years[-1],
            "maxima_period": "season",
        },
    )


def materialize_derived_series(
    *,
    houston_path: Path | str,
    phoenix_path: Path | str,
    output_dir: Path | str,
    metadata_dir: Path | str,
) -> dict[str, Path]:
    """Create derived application-ready series and a metadata registry."""
    output_dir = Path(output_dir)
    metadata_dir = Path(metadata_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    houston = prepare_precipitation_series(houston_path)
    phoenix = prepare_hot_dry_series(phoenix_path)
    houston_file = output_dir / "houston_hobby_precipitation.csv.gz"
    phoenix_file = output_dir / "phoenix_hot_dry_severity.csv.gz"
    write_csv_gz_atomic(houston.to_frame().reset_index(), houston_file)
    write_csv_gz_atomic(phoenix.to_frame().reset_index(), phoenix_file)
    metadata = {
        "houston_hobby_precipitation": {
            **houston.metadata,
            "raw_file": str(houston_path),
            "derived_file": str(houston_file),
            "source_url": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USW00012918.csv.gz",
        },
        "phoenix_hot_dry_severity": {
            **phoenix.metadata,
            "raw_file": str(phoenix_path),
            "derived_file": str(phoenix_file),
            "source_url": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USW00023183.csv.gz",
        },
    }
    with (metadata_dir / "weather_derived_sources.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)
    return {
        "houston_hobby_precipitation": houston_file,
        "phoenix_hot_dry_severity": phoenix_file,
    }
