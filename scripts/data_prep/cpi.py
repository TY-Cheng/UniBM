"""Monthly CPI-U retrieval for NFIP inflation adjustment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .constants import ANALYSIS_END_DATE


BLS_API_ENDPOINT = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
CPI_U_SERIES_ID = "CUUR0000SA0"
CPI_2025_ANNUAL_AVERAGE = 321.943


def _fetch_bls_window(start_year: int, end_year: int) -> list[dict[str, object]]:
    """Fetch one keyless BLS API window."""
    request = Request(
        BLS_API_ENDPOINT,
        data=json.dumps(
            {
                "seriesid": [CPI_U_SERIES_ID],
                "startyear": str(start_year),
                "endyear": str(end_year),
            }
        ).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "UniBM/0.0.0"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310
        payload = json.load(response)
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS CPI request failed: {payload.get('message', [])}")
    series = payload.get("Results", {}).get("series", [])
    if not series or not isinstance(series[0], dict):
        raise ValueError("BLS CPI response did not contain the requested series.")
    records = series[0].get("data", [])
    if not isinstance(records, list):
        raise ValueError("BLS CPI response did not contain monthly observations.")
    return [record for record in records if isinstance(record, dict)]


def download_monthly_cpi(
    output_path: Path | str,
    *,
    start_year: int = 1978,
    end_year: int = int(ANALYSIS_END_DATE[:4]),
) -> Path:
    """Download monthly NSA CPI-U and explicitly impute missing October 2025."""
    observations: dict[pd.Period, float] = {}
    for window_start in range(start_year, end_year + 1, 10):
        window_end = min(window_start + 9, end_year)
        for record in _fetch_bls_window(window_start, window_end):
            period = str(record.get("period", ""))
            if len(period) != 3 or not period.startswith("M") or not period[1:].isdigit():
                continue
            month = int(period[1:])
            if not 1 <= month <= 12:
                continue
            key = pd.Period(f"{int(record['year']):04d}-{month:02d}", freq="M")
            try:
                value = float(record["value"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0:
                observations[key] = value

    expected = pd.period_range(f"{start_year}-01", f"{end_year}-12", freq="M")
    missing = set(expected).difference(observations)
    imputed_month = pd.Period("2025-10", freq="M")
    imputed = False
    if missing == {imputed_month}:
        september = observations[pd.Period("2025-09", freq="M")]
        november = observations[pd.Period("2025-11", freq="M")]
        observations[imputed_month] = float(np.sqrt(september * november))
        imputed = True
    elif missing:
        missing_text = ", ".join(str(month) for month in sorted(missing))
        raise ValueError(f"BLS CPI response is missing months: {missing_text}.")

    frame = pd.DataFrame(
        {
            "date": [month.to_timestamp().strftime("%Y-%m-%d") for month in expected],
            "cpi_u": [observations[month] for month in expected],
            "is_imputed": [imputed and month == imputed_month for month in expected],
            "note": [
                "geometric mean of 2025-09 and 2025-11; BLS observation unavailable"
                if imputed and month == imputed_month
                else ""
                for month in expected
            ],
        }
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        frame.to_csv(tmp_path, index=False, float_format="%.6f")
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return output_path


__all__ = [
    "BLS_API_ENDPOINT",
    "CPI_2025_ANNUAL_AVERAGE",
    "CPI_U_SERIES_ID",
    "download_monthly_cpi",
]
