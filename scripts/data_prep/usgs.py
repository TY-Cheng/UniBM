"""USGS daily-discharge preparation for manuscript applications."""

from __future__ import annotations

from json import JSONDecodeError
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .ghcn import PreparedSeries

USGS_DV_ENDPOINT = "https://waterservices.usgs.gov/nwis/dv/"
DEFAULT_USGS_START_DATE = "1900-01-01"
MIN_USGS_HISTORY_ROWS = 365 * 5
MIN_USGS_HISTORY_SPAN_DAYS = 365 * 5
USGS_TIMEOUT_SECONDS = 30
USGS_MAX_RETRIES = 3
USGS_RETRY_WAIT_SECONDS = 2.0
USGS_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


def _normalize_site_no(site_no: object) -> str:
    """Normalize a USGS site id without discarding significant leading zeros."""
    text = str(site_no).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _timeseries_stat_code(item: dict[str, object]) -> str | None:
    """Extract the USGS daily statistic code from one timeSeries payload item."""
    name = item.get("name")
    if isinstance(name, str):
        parts = name.split(":")
        if len(parts) >= 4 and parts[-1]:
            return str(parts[-1])
    variable = item.get("variable")
    if isinstance(variable, dict):
        options = variable.get("options")
        if isinstance(options, dict):
            option = options.get("option")
            if isinstance(option, list):
                for entry in option:
                    if isinstance(entry, dict) and entry.get("name") == "Statistic":
                        code = entry.get("optionCode")
                        if code is not None:
                            return str(code)
            elif isinstance(option, dict) and option.get("name") == "Statistic":
                code = option.get("optionCode")
                if code is not None:
                    return str(code)
    return None


def _drop_partial_terminal_year(series: pd.Series, *, min_fraction: float = 0.9) -> pd.Series:
    """Drop the final year when it looks materially incomplete."""
    if series.empty:
        return series
    last_year = int(series.index.year.max())
    mask_last = series.index.year == last_year
    if mask_last.sum() < min_fraction * 365:
        return series.loc[~mask_last]
    return series


def _extract_usgs_daily_series(payload: dict[str, object]) -> tuple[pd.Series, str]:
    """Parse the USGS daily-values JSON payload for parameter 00060."""
    time_series = payload.get("value", {}).get("timeSeries", [])  # type: ignore[union-attr]
    if not isinstance(time_series, list) or not time_series:
        raise ValueError("USGS daily-values payload did not contain any timeSeries records.")
    mean_daily = [
        item
        for item in time_series
        if isinstance(item, dict) and _timeseries_stat_code(item) == "00003"
    ]
    item = mean_daily[0] if mean_daily else time_series[0]
    if not isinstance(item, dict):
        raise ValueError("Malformed USGS daily-values payload.")
    source_info = item.get("sourceInfo", {})
    station_name = (
        str(source_info.get("siteName"))
        if isinstance(source_info, dict) and source_info.get("siteName") is not None
        else "USGS streamgage"
    )
    values_block = item.get("values", [])
    if not isinstance(values_block, list) or not values_block:
        raise ValueError("USGS daily-values payload did not contain any values blocks.")
    records = values_block[0].get("value", []) if isinstance(values_block[0], dict) else []
    if not isinstance(records, list):
        raise ValueError("USGS daily-values payload did not contain a record list.")
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        date_time = record.get("dateTime")
        value = record.get("value")
        if date_time is None or value in {None, ""}:
            continue
        try:
            dates.append(pd.Timestamp(str(date_time)).normalize())
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not dates:
        raise ValueError("USGS daily-values payload did not contain any finite discharge values.")
    series = pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float)
    series = series[np.isfinite(series) & (series >= 0)]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series, station_name


def _open_usgs_json_with_retries(url: str, *, site_no: str) -> dict[str, object]:
    """Fetch one USGS daily-values payload with lightweight retries."""
    request = Request(url, headers={"User-Agent": "UniBM/0.0.0"})
    last_error: Exception | None = None
    for attempt in range(1, USGS_MAX_RETRIES + 1):
        try:
            with urlopen(request, timeout=USGS_TIMEOUT_SECONDS) as response:  # noqa: S310
                loaded = json.load(response)
            if not isinstance(loaded, dict):
                raise ValueError("USGS payload was not a JSON object.")
            return loaded
        except HTTPError as exc:
            last_error = exc
            if exc.code not in USGS_RETRY_STATUS_CODES or attempt == USGS_MAX_RETRIES:
                break
        except (URLError, JSONDecodeError) as exc:
            last_error = exc
            if attempt == USGS_MAX_RETRIES:
                break
        print(
            f"[usgs] transient network error for site {site_no} "
            f"(attempt {attempt}/{USGS_MAX_RETRIES}); retrying in {USGS_RETRY_WAIT_SECONDS:.0f}s",
            flush=True,
        )
        time.sleep(USGS_RETRY_WAIT_SECONDS)
    raise RuntimeError(
        f"USGS fetch failed after {USGS_MAX_RETRIES} retries: {url}"
    ) from last_error


def download_usgs_daily_discharge(
    site_no: str,
    output_path: Path | str,
    *,
    start_date: str | None = DEFAULT_USGS_START_DATE,
    end_date: str | None = None,
) -> Path:
    """Download one USGS daily-discharge series and save it as a compact CSV.GZ file."""
    params = {
        "format": "json",
        "sites": str(site_no),
        "parameterCd": "00060",
        "statCd": "00003",
        "siteStatus": "all",
    }
    if start_date is not None:
        params["startDT"] = start_date
    if end_date is not None:
        params["endDT"] = end_date
    url = f"{USGS_DV_ENDPOINT}?{urlencode(params)}"
    payload = _open_usgs_json_with_retries(url, site_no=str(site_no))
    series, station_name = _extract_usgs_daily_series(payload)
    frame = pd.DataFrame(
        {
            "date": series.index.strftime("%Y-%m-%d"),
            "discharge_cfs": series.to_numpy(dtype=float),
            "site_no": str(site_no),
            "station_name": station_name,
        }
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f"{output_path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        frame.to_csv(tmp_path, index=False, compression="gzip")
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
    return output_path


def read_usgs_daily_discharge_csv(path: Path | str) -> pd.DataFrame:
    """Read a compact USGS daily-discharge CSV.GZ file."""
    return pd.read_csv(
        path,
        parse_dates=["date"],
        dtype={"site_no": "string", "station_name": "string"},
    )


def usgs_daily_discharge_needs_refresh(
    path: Path | str,
    *,
    min_rows: int = MIN_USGS_HISTORY_ROWS,
    min_span_days: int = MIN_USGS_HISTORY_SPAN_DAYS,
) -> bool:
    """Return whether an on-disk USGS raw extract is too short to be useful."""
    path = Path(path)
    if not path.exists():
        return True
    try:
        df = read_usgs_daily_discharge_csv(path)
    except Exception:
        return True
    if df.empty or "date" not in df.columns:
        return True
    date_index = pd.DatetimeIndex(df["date"]).dropna()
    if date_index.empty:
        return True
    span_days = int((date_index.max() - date_index.min()).days)
    return int(df.shape[0]) < int(min_rows) or span_days < int(min_span_days)


def prepare_usgs_streamflow_series(
    path: Path | str,
    *,
    state_code: str,
    site_no: str | None = None,
    station_name: str | None = None,
) -> PreparedSeries:
    """Prepare one full-year daily-discharge application series."""
    df = read_usgs_daily_discharge_csv(path)
    if df.empty:
        raise ValueError("USGS streamflow file did not contain any rows.")
    if site_no is not None:
        target_site = _normalize_site_no(site_no)
        normalized_site = df["site_no"].astype(str).map(_normalize_site_no)
        df = df[normalized_site == target_site]
    if df.empty:
        raise ValueError("USGS streamflow file did not contain any rows for the requested site.")
    series = pd.Series(
        df["discharge_cfs"].astype(float).to_numpy(),
        index=pd.DatetimeIndex(df["date"]),
        dtype=float,
    )
    series = series[np.isfinite(series) & (series >= 0)]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series = _drop_partial_terminal_year(series)
    annual_maxima = series.groupby(series.index.year).max()
    resolved_site = str(site_no) if site_no is not None else str(df["site_no"].iloc[0])
    resolved_name = (
        str(station_name)
        if station_name is not None
        else str(df.get("station_name", pd.Series(["USGS streamgage"])).iloc[0])
    )
    return PreparedSeries(
        name=f"daily discharge at {resolved_name}",
        value_name="streamflow_cfs",
        series=series,
        annual_maxima=annual_maxima,
        metadata={
            "provider": "usgs",
            "site_no": resolved_site,
            "station_name": resolved_name,
            "state_code": str(state_code).upper(),
            "unit": "cfs",
        },
    )


__all__ = [
    "DEFAULT_USGS_START_DATE",
    "MIN_USGS_HISTORY_ROWS",
    "MIN_USGS_HISTORY_SPAN_DAYS",
    "USGS_DV_ENDPOINT",
    "download_usgs_daily_discharge",
    "prepare_usgs_streamflow_series",
    "read_usgs_daily_discharge_csv",
    "usgs_daily_discharge_needs_refresh",
]
