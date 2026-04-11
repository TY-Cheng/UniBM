"""OpenFEMA NFIP claims preparation for manuscript applications."""

from __future__ import annotations

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

OPENFEMA_NFIP_CLAIMS_ENDPOINT = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
OPENFEMA_PAGE_SIZE = 5_000
OPENFEMA_TIMEOUT_SECONDS = 60
OPENFEMA_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
OPENFEMA_MAX_RETRIES = 6
OPENFEMA_RETRY_BASE_SECONDS = 2.0
DEFAULT_NFIP_END_DATE = "2025-12-31"


def _extract_openfema_records(payload: dict[str, object]) -> list[dict[str, object]]:
    """Return the first list-valued payload entry as the record list."""
    for key in ("FimaNfipClaims", "fimaNfipClaims", "results", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    for value in payload.values():
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    raise ValueError("OpenFEMA payload did not contain a recognizable record list.")


def _nfip_chunk_dir(output_path: Path | str) -> Path:
    output_path = Path(output_path)
    return output_path.parent / "chunks" / output_path.name.replace(".csv.gz", "")


def _nfip_chunk_path(output_path: Path | str, *, state_code: str, year: int) -> Path:
    return _nfip_chunk_dir(output_path) / f"nfip_claims_{state_code.lower()}_{year}.csv.gz"


def _year_bounds(start_date: str, end_date: str) -> list[tuple[int, str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date.")
    bounds: list[tuple[int, str, str]] = []
    for year in range(start.year, end.year + 1):
        lower = max(start, pd.Timestamp(year=year, month=1, day=1))
        upper = min(end, pd.Timestamp(year=year, month=12, day=31))
        bounds.append((year, lower.strftime("%Y-%m-%d"), upper.strftime("%Y-%m-%d")))
    return bounds


def _openfema_json_with_retries(url: str) -> dict[str, object]:
    """Fetch one OpenFEMA JSON payload with bounded retries for transient failures."""
    request = Request(url, headers={"User-Agent": "UniBM/0.0.0"})
    last_error: Exception | None = None
    for attempt in range(1, OPENFEMA_MAX_RETRIES + 1):
        try:
            with urlopen(request, timeout=OPENFEMA_TIMEOUT_SECONDS) as response:  # noqa: S310
                loaded = json.load(response)
            if not isinstance(loaded, dict):
                raise ValueError("OpenFEMA payload was not a JSON object.")
            return loaded
        except HTTPError as exc:
            last_error = exc
            if exc.code not in OPENFEMA_RETRY_STATUS_CODES or attempt == OPENFEMA_MAX_RETRIES:
                break
        except URLError as exc:
            last_error = exc
            if attempt == OPENFEMA_MAX_RETRIES:
                break
        wait_seconds = OPENFEMA_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
        print(
            f"[fema] transient download failure (attempt {attempt}/{OPENFEMA_MAX_RETRIES}); "
            f"retrying in {wait_seconds:.0f}s",
            flush=True,
        )
        time.sleep(wait_seconds)
    raise RuntimeError("OpenFEMA request failed after repeated retries.") from last_error


def _download_nfip_claims_window(
    state_code: str,
    *,
    start_date: str,
    end_date: str,
    page_size: int,
) -> pd.DataFrame:
    """Download one bounded NFIP date window into a dataframe."""
    records: list[dict[str, object]] = []
    skip = 0
    page_number = 0
    while True:
        params = {
            "$filter": (
                f"state eq '{state_code}' and dateOfLoss ge '{start_date}' "
                f"and dateOfLoss le '{end_date}' and amountPaidOnBuildingClaim ge 0"
            ),
            "$orderby": "dateOfLoss",
            "$select": "state,dateOfLoss,amountPaidOnBuildingClaim",
            "$skip": str(skip),
            "$top": str(page_size),
        }
        url = f"{OPENFEMA_NFIP_CLAIMS_ENDPOINT}?{urlencode(params)}"
        payload = _openfema_json_with_retries(url)
        page = _extract_openfema_records(payload)
        if not page:
            break
        records.extend(page)
        page_number += 1
        print(
            f"[fema] {state_code} {start_date[:4]}: fetched page {page_number} "
            f"({len(page)} rows, total={len(records)})",
            flush=True,
        )
        if len(page) < page_size:
            break
        skip += len(page)
        time.sleep(1.0)
    if not records:
        return pd.DataFrame(columns=["state", "dateOfLoss", "amountPaidOnBuildingClaim"])
    return pd.DataFrame.from_records(records)


def _load_nfip_chunk_if_valid(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path, parse_dates=["dateOfLoss"])
    except Exception:
        return None
    required = {"state", "dateOfLoss", "amountPaidOnBuildingClaim"}
    if not required.issubset(frame.columns):
        return None
    return frame


def _write_csv_gz_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    """Atomically write one compressed CSV."""
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


def nfip_claims_needs_refresh(
    path: Path | str,
    *,
    state_code: str,
    min_size_bytes: int = 32,
    sample_rows: int = 128,
) -> bool:
    """Return whether one cached NFIP state extract looks unusable.

    The check is intentionally cheap because the state-level combined extract can
    be large. If the top-level file looks broken, the downloader can rebuild it
    from cached yearly chunks without starting from scratch.
    """
    path = Path(path)
    if not path.exists():
        return True
    try:
        if path.stat().st_size < int(min_size_bytes):
            return True
    except OSError:
        return True
    try:
        frame = pd.read_csv(path, parse_dates=["dateOfLoss"], nrows=int(sample_rows))
    except Exception:
        return True
    required = {"state", "dateOfLoss", "amountPaidOnBuildingClaim"}
    if frame.empty or not required.issubset(frame.columns):
        return True
    states = set(frame["state"].astype(str).str.upper())
    if not states or states != {str(state_code).upper()}:
        return True
    amounts = pd.to_numeric(frame["amountPaidOnBuildingClaim"], errors="coerce")
    if not np.isfinite(amounts).any():
        return True
    return False


def download_nfip_claims_state(
    state_code: str,
    output_path: Path | str,
    *,
    start_date: str = "1978-01-01",
    end_date: str = DEFAULT_NFIP_END_DATE,
    page_size: int = OPENFEMA_PAGE_SIZE,
    force_refresh: bool = False,
) -> Path:
    """Download one state-level NFIP claims extract as a compact CSV.GZ file."""
    state_code = str(state_code).upper()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = _nfip_chunk_dir(output_path)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fema] downloading NFIP claims for {state_code}", flush=True)
    yearly_frames: list[pd.DataFrame] = []
    for year, year_start, year_end in _year_bounds(start_date, end_date):
        chunk_path = _nfip_chunk_path(output_path, state_code=state_code, year=year)
        cached = (
            _load_nfip_chunk_if_valid(chunk_path)
            if chunk_path.exists() and not force_refresh
            else None
        )
        if cached is not None:
            print(f"[fema] {state_code} {year}: reusing cached yearly chunk", flush=True)
            yearly_frames.append(cached)
            continue
        print(
            f"[fema] {state_code} {year}: downloading yearly chunk ({year_start} to {year_end})",
            flush=True,
        )
        frame = _download_nfip_claims_window(
            state_code,
            start_date=year_start,
            end_date=year_end,
            page_size=page_size,
        )
        _write_csv_gz_atomic(frame, chunk_path)
        print(f"[fema] {state_code} {year}: wrote {len(frame)} rows to {chunk_path}", flush=True)
        yearly_frames.append(frame)
    frame = pd.concat(yearly_frames, ignore_index=True)
    if frame.empty:
        raise ValueError(f"No OpenFEMA NFIP claims were returned for state={state_code!r}.")
    frame = frame.sort_values("dateOfLoss").reset_index(drop=True)
    _write_csv_gz_atomic(frame, output_path)
    print(f"[fema] wrote {len(frame)} rows to {output_path}", flush=True)
    return output_path


def read_nfip_claims_csv(path: Path | str) -> pd.DataFrame:
    """Read one compact NFIP claims CSV.GZ file."""
    return pd.read_csv(path, parse_dates=["dateOfLoss"])


def load_cpi_2025_base(path: Path | str) -> pd.Series:
    """Load the fixed annual CPI-U deflator table with 2025 equal to 100."""
    frame = pd.read_csv(path)
    if "year" not in frame.columns or "cpi_2025_base" not in frame.columns:
        raise ValueError("CPI table must contain 'year' and 'cpi_2025_base' columns.")
    return pd.Series(
        frame["cpi_2025_base"].astype(float).to_numpy(),
        index=frame["year"].astype(int).to_numpy(),
        dtype=float,
    )


def _deflate_to_2025_usd(
    amounts: pd.Series, *, dates: pd.Series, cpi_2025: pd.Series
) -> pd.Series:
    """Convert nominal claim payouts to 2025 USD using a fixed annual CPI table."""
    years = dates.dt.year.astype(int)
    deflator = years.map(cpi_2025)
    return amounts.astype(float) * (100.0 / deflator.astype(float))


def prepare_nfip_claim_series(
    path: Path | str,
    *,
    state_code: str,
    cpi_table_path: Path | str,
) -> dict[str, PreparedSeries]:
    """Prepare zero-filled and positive-only NFIP daily payout series for one state."""
    state_code = str(state_code).upper()
    claims = read_nfip_claims_csv(path)
    if claims.empty:
        raise ValueError("NFIP claims file did not contain any rows.")
    claims = claims.rename(columns={col: col.strip() for col in claims.columns})
    if "state" not in claims.columns:
        raise ValueError("NFIP claims file must contain a 'state' column.")
    claims = claims[claims["state"].astype(str).str.upper() == state_code].copy()
    if claims.empty:
        raise ValueError(f"NFIP claims file did not contain any {state_code} rows.")
    claims["amountPaidOnBuildingClaim"] = pd.to_numeric(
        claims["amountPaidOnBuildingClaim"], errors="coerce"
    )
    claims = claims[
        claims["dateOfLoss"].notna()
        & np.isfinite(claims["amountPaidOnBuildingClaim"])
        & (claims["amountPaidOnBuildingClaim"] >= 0)
    ].copy()
    cpi_2025 = load_cpi_2025_base(cpi_table_path)
    claims["amount_real_2025"] = _deflate_to_2025_usd(
        claims["amountPaidOnBuildingClaim"],
        dates=claims["dateOfLoss"],
        cpi_2025=cpi_2025,
    )
    claims = claims[np.isfinite(claims["amount_real_2025"])].copy()
    daily = (
        claims.groupby(claims["dateOfLoss"].dt.normalize())["amount_real_2025"].sum().sort_index()
    )
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    zero_filled = daily.reindex(full_index, fill_value=0.0).astype(float)
    positive_only = zero_filled[zero_filled > 0].astype(float)
    state_label = (
        "Florida" if state_code == "FL" else "Texas" if state_code == "TX" else state_code
    )
    display = PreparedSeries(
        name=f"{state_label} daily NFIP building payouts",
        value_name="building_payout_2025usd",
        series=zero_filled,
        annual_maxima=zero_filled.groupby(zero_filled.index.year).max(),
        metadata={
            "provider": "fema",
            "state_code": state_code,
            "state_name": state_label,
            "unit": "2025 USD",
            "series_role": "display",
            "series_basis": "calendar_day",
        },
    )
    evi = PreparedSeries(
        name=f"{state_label} positive-day NFIP building payouts",
        value_name="building_payout_2025usd",
        series=positive_only,
        annual_maxima=positive_only.groupby(positive_only.index.year).max(),
        metadata={
            "provider": "fema",
            "state_code": state_code,
            "state_name": state_label,
            "unit": "2025 USD",
            "series_role": "evi",
            "series_basis": "claim_active_day",
        },
    )
    ei = PreparedSeries(
        name=f"{state_label} daily NFIP building payout waves",
        value_name="building_payout_2025usd",
        series=zero_filled,
        annual_maxima=zero_filled.groupby(zero_filled.index.year).max(),
        metadata={
            "provider": "fema",
            "state_code": state_code,
            "state_name": state_label,
            "unit": "2025 USD",
            "series_role": "ei",
            "series_basis": "calendar_day",
        },
    )
    return {"display": display, "evi": evi, "ei": ei}


__all__ = [
    "OPENFEMA_NFIP_CLAIMS_ENDPOINT",
    "download_nfip_claims_state",
    "load_cpi_2025_base",
    "nfip_claims_needs_refresh",
    "prepare_nfip_claim_series",
    "read_nfip_claims_csv",
]
