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
from typing import Any

import numpy as np
import pandas as pd


GHCN_COLUMNS = ["station_id", "date", "element", "value", "mflag", "qflag", "sflag", "obstime"]


@dataclass(frozen=True)
class PreparedSeries:
    """A prepared univariate series plus annual maxima and provenance metadata."""

    name: str
    value_name: str
    series: pd.Series
    annual_maxima: pd.Series
    metadata: dict[str, Any]

    def to_frame(self) -> pd.DataFrame:
        return self.series.rename(self.value_name).to_frame()


def read_ghcn_station_csv(path: Path | str) -> pd.DataFrame:
    """Read a by-station GHCN CSV or CSV.GZ file."""
    df = pd.read_csv(path, header=None, names=GHCN_COLUMNS, low_memory=False)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    for col in ["mflag", "qflag", "sflag", "obstime"]:
        df[col] = df[col].replace({"": np.nan})
    return df


def _extract_ghcn_element(df: pd.DataFrame, element: str, *, scale: float) -> pd.Series:
    """Extract one quality-controlled GHCN element as a dated numeric series."""
    sub = df.loc[df["element"] == element].copy()
    sub = sub[sub["qflag"].isna()]
    series = pd.Series(
        sub["value"].astype(float).to_numpy() / scale,
        index=pd.DatetimeIndex(sub["date"]),
    )
    return series[~series.index.duplicated(keep="last")].sort_index()


def _drop_partial_terminal_year(series: pd.Series, *, min_fraction: float = 0.9) -> pd.Series:
    """Drop the last calendar year if it appears substantially incomplete."""
    if series.empty:
        return series
    last_year = int(series.index.year.max())
    mask_last = series.index.year == last_year
    if mask_last.sum() < min_fraction * 365:
        return series.loc[~mask_last]
    return series


def prepare_precipitation_series(
    path: Path | str,
    *,
    wet_season_months: tuple[int, ...] = (6, 7, 8, 9, 10, 11),
    min_wet_day_mm: float = 0.0,
) -> PreparedSeries:
    """Prepare a wet-season precipitation series from daily GHCN data."""
    df = read_ghcn_station_csv(path)
    precipitation = _extract_ghcn_element(df, "PRCP", scale=10.0)
    precipitation = _drop_partial_terminal_year(precipitation)
    precipitation = precipitation[precipitation.index.month.isin(wet_season_months)]
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
    tmax = _extract_ghcn_element(df, "TMAX", scale=10.0)
    precipitation = _extract_ghcn_element(df, "PRCP", scale=10.0)
    date_index = pd.date_range(
        max(tmax.index.min(), precipitation.index.min()),
        min(tmax.index.max(), precipitation.index.max()),
        freq="D",
    )
    tmax = tmax.reindex(date_index)
    precipitation = precipitation.reindex(date_index).fillna(0.0)
    rolling_precipitation = precipitation.rolling(
        rolling_days,
        min_periods=max(7, rolling_days // 3),
    ).sum()
    daily = pd.DataFrame(
        {
            "tmax_c": tmax,
            "prcp_mm": precipitation,
            "prcp_roll_mm": rolling_precipitation,
        },
        index=date_index,
    )
    daily = daily[daily.index.month.isin(warm_season_months)]
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
    severity = severity[severity > 0]
    severity = _drop_partial_terminal_year(severity)
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
    houston.to_frame().to_csv(houston_file, compression="gzip")
    phoenix.to_frame().to_csv(phoenix_file, compression="gzip")
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
    with (metadata_dir / "sources.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)
    return {
        "houston_hobby_precipitation": houston_file,
        "phoenix_hot_dry_severity": phoenix_file,
    }
