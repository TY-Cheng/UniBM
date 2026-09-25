"""Rebuild SPY/QQQ pilot inputs from frozen Massive connector responses.

Run with uv run python scripts/data_prep/finance_snapshot.py
No credentials or network are used. To refresh, repeat the exact call_api and
query_data requests preserved beside their responses in data/raw/pilots/finance.
The original full-history requests were entitlement-clipped by the provider;
separate old-history probes returned NOT_ENTITLED. Never describe this as a
full-inception sample. The first close is a lag and is not a zero-loss observation.
"""

import csv
import hashlib
import io
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/pilots/finance"
OUT = ROOT / "data/processed/pilots"
CUTOFF = "2025-12-31"
RETRIEVED_ON = "2026-09-25"
CALENDAR_SOURCES = [
    "https://www.nyse.com/markets/hours-calendars",
    "https://ir.theice.com/press/news-details/2021/NYSE-Group-Announces-2022-2023-and-2024-Holiday-and-Early-Closings-Calendar/default.aspx",
    "https://s2.q4cdn.com/154085107/files/doc_news/archive/266fafb2-a85c-433a-8350-0760d895b49c.pdf",
    "https://www.nyse.com/publicdocs/nyse/markets/american-options/rule-interpretations/2025/National_Day_of_Mourning_20250102.pdf",
]


def response_text(name):
    payload = json.loads((RAW / f"{name}_response.json").read_text())
    response = payload["response"]
    assert not response.get("isError"), name
    text = response.get("structuredContent", {}).get("result")
    if text is None:
        text = "\n".join(x["text"] for x in response["content"] if x["type"] == "text")
    assert "truncated" not in text.lower() and "next page" not in text.lower(), name
    return text


def records(name):
    text = response_text(name)
    if text.startswith("Warning [EMPTY]"):
        return []
    assert not text.startswith("Warning"), (name, text)
    csv.field_size_limit(10_000_000)
    rows = list(csv.DictReader(io.StringIO(text)))
    if name.endswith("bars_export"):
        assert len(rows) == 1 and "raw_json" in rows[0], name
        return json.loads(rows[0]["raw_json"])
    return rows


def current_share_cash(amount, ex_date, splits, as_of):
    """Cash per ex-date share converted with every later effective split."""
    return float(amount) * math.prod(
        float(event["split_from"]) / float(event["split_to"])
        for event in splits
        if ex_date < event["execution_date"] <= as_of
    )


def log_return(previous_close, close, dividend):
    return np.log((close + dividend) / previous_close)


def sessions(start, end):
    # ponytail: calendar is bounded to 2016-2025; use an exchange calendar for expansion.
    assert pd.Timestamp(start).year >= 2016 and pd.Timestamp(end).year <= 2025
    holidays = AbstractHolidayCalendar(
        rules=[
            Holiday("New Year", month=1, day=1, observance=sunday_to_monday),
            USMartinLutherKingJr,
            USPresidentsDay,
            GoodFriday,
            USMemorialDay,
            Holiday(
                "Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday
            ),
            Holiday("Independence", month=7, day=4, observance=nearest_workday),
            USLaborDay,
            USThanksgivingDay,
            Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        ]
    ).holidays(start, end)
    closures = pd.DatetimeIndex(["2018-12-05", "2025-01-09"])
    return pd.bdate_range(start, end).difference(holidays.union(closures))


def self_check():
    # A $100 share becomes two $49 shares plus $1 cash per new share: no loss.
    split = [{"execution_date": "2025-12-30", "split_from": 1, "split_to": 2}]
    assert current_share_cash(1, "2025-12-30", split, RETRIEVED_ON) == 1
    assert math.isclose(log_return(100 / 2, 49, 1), 0, abs_tol=1e-14)
    # A further 2:1 split AFTER the analysis cutoff halves all three terms.
    split.append({"execution_date": "2026-03-01", "split_from": 1, "split_to": 2})
    cash = current_share_cash(1, "2025-12-30", split, RETRIEVED_ON)
    assert cash == 0.5
    assert math.isclose(log_return(25, 24.5, cash), 0, abs_tol=1e-14)
    assert current_share_cash(2, "2025-12-29", split, RETRIEVED_ON) == 0.5
    # Hand checks: $100 -> $90 plus $5 cash is a 5% loss; a gain stays zero.
    assert math.isclose(-log_return(100, 90, 5), -math.log(0.95), abs_tol=1e-14)
    assert max(-log_return(100, 101, 0), 0) == 0
    assert pd.Timestamp("2021-12-31") in sessions("2021-12-30", "2021-12-31")
    assert pd.Timestamp("2022-06-20") not in sessions("2022-06-17", "2022-06-21")
    assert pd.Timestamp("2025-01-09") not in sessions("2025-01-08", "2025-01-10")


def build(ticker):
    key = ticker.lower()
    bars = pd.DataFrame(records(f"{key}_bars_export"))
    assert len(bars) == 2329 and set(bars["ticker"]) == {ticker}
    bars["date"] = (
        pd.to_datetime(bars.t, unit="ms", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    assert bars.date.is_unique and bars.date.is_monotonic_increasing
    assert bars.date.max() == pd.Timestamp(CUTOFF)
    native_close_checks = []
    for raw_path in sorted(RAW.glob(f"{key}_dailycheck_*_response.json")):
        rows = records(raw_path.name.removesuffix("_response.json"))
        assert len(rows) == 1 and rows[0]["symbol"] == ticker
        item = rows[0]
        close = bars.loc[bars.date == pd.Timestamp(item["from"]), "c"].item()
        assert math.isclose(close, float(item["close"]), rel_tol=0, abs_tol=1e-10)
        native_close_checks.append({"date": item["from"], "close": close, "matched": True})
    assert np.isfinite(bars[["o", "h", "l", "c", "v"]].to_numpy()).all()
    assert (bars[["o", "h", "l", "c", "v"]] > 0).all().all()
    assert ((bars.c >= bars.l) & (bars.c <= bars.h)).all()
    expected = sessions(bars.date.min(), bars.date.max())
    observed = pd.DatetimeIndex(bars.date)
    missing = expected.difference(observed).strftime("%Y-%m-%d").tolist()
    extra = observed.difference(expected).strftime("%Y-%m-%d").tolist()
    assert not missing and not extra, (ticker, missing, extra)
    splits = records(f"{key}_splits")
    dividends = records(f"{key}_dividends")
    assert len({r["id"] for r in dividends}) == len(dividends)
    assert len({r["id"] for r in splits}) == len(splits)
    by_date = {}
    relevant = []
    for item in dividends:
        ex_date = item["ex_dividend_date"]
        if not bars.date.min().strftime("%Y-%m-%d") <= ex_date <= CUTOFF:
            continue
        assert item["currency"] == "USD" and float(item["cash_amount"]) >= 0
        amount = current_share_cash(item["cash_amount"], ex_date, splits, RETRIEVED_ON)
        # Independently reconcile provider current-share dividends to split history.
        assert math.isclose(
            amount, float(item["split_adjusted_cash_amount"]), rel_tol=1e-6, abs_tol=1e-8
        ), item["id"]
        date = pd.Timestamp(ex_date)
        assert date in observed, (ticker, ex_date)
        by_date[date] = by_date.get(date, 0.0) + amount
        relevant.append(item)
    bars["dividend"] = bars.date.map(by_date).fillna(0.0)
    bars["return"] = log_return(bars.c.shift(), bars.c, bars.dividend)
    bars["value"] = (-bars["return"]).clip(lower=0.0)
    result = bars.iloc[1:][["date", "value", "return", "c", "dividend"]].rename(
        columns={"c": "close"}
    )
    assert len(result) == len(bars) - 1 and np.isfinite(result.iloc[:, 1:]).all().all()
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"{key}.csv"
    result.to_csv(output, index=False, date_format="%Y-%m-%d", float_format="%.17g")
    raw_paths = sorted(RAW.glob(f"{key}_*response.json")) + [RAW / "endpoint_documentation.json"]
    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_paths
    }
    cap_text = response_text(f"{key}_cap_probe")
    assert "NOT_ENTITLED" in cap_text
    extreme_rows = result.loc[result["return"].abs() > 0.10]
    metadata = {
        "ticker": ticker,
        "retrieved_on": RETRIEVED_ON,
        "provider": "Massive.com via installed Massive MCP connector",
        "source_host": "https://api.massive.com",
        "rights": {
            "access": "Connected account retrieved this data; old price history returned NOT_ENTITLED.",
            "redistribution": "Not established by connector retrieval. Raw data kept locally; no publication or redistribution performed.",
            "account_tier": "Not inspected; approximately ten-year rolling price-history cap inferred, not a verified plan name.",
        },
        "requests": {
            p.name: json.loads(p.read_text())["query"]
            for p in raw_paths
            if p.name != "endpoint_documentation.json"
        },
        "coverage": {
            "requested_start": "1993-01-01" if ticker == "SPY" else "1999-01-01",
            "requested_end": CUTOFF,
            "close_start": str(bars.date.min().date()),
            "close_end": str(bars.date.max().date()),
            "return_start": str(result.date.min().date()),
            "return_end": str(result.date.max().date()),
            "closes": len(bars),
            "returns": len(result),
            "limitation": "The full requested range was silently clamped. A separate request ending 2016-09-26 returned NOT_ENTITLED; earlier history is not available through this account.",
            "historical_probe_response": cap_text.strip(),
            "first_close_usage": "Lag only; omitted from CSV because no entitled prior close exists.",
            "dividend_records_retrieved": len(dividends),
            "dividend_records_used": len(relevant),
            "split_events_retrieved_through_fetch_date": len(splits),
        },
        "definition": {
            "return": "log((C_split_adjusted_t + D_same_basis_t) / C_split_adjusted_previous_trading_day)",
            "value": "max(-return, 0)",
            "close": "Daily aggregate close with adjusted=true (splits only).",
            "dividend": "Sum of USD cash dividends on ex-dividend date, times product(split_from/split_to) for all later splits effective through retrieval date; reconciled to provider split_adjusted_cash_amount.",
            "clock": "Observed US equity trading sessions, America/New_York dates; no weekend/holiday padding. Early closes remain sessions.",
            "zeros": "All nonnegative log-return sessions retained as zero left-tail loss.",
            "units": "Natural-log return; dimensionless. close and dividend in USD per current-basis share.",
            "point_in_time": "Historical data snapshot at retrieval; revised prices/actions possible. This is not point-in-time vintage data or an executable net-of-cost return.",
        },
        "quality_checks": {
            "self_check": "Passed split, post-cutoff split, dividend-neutral loss, hand-computed 5% loss, gain-zero and calendar checks.",
            "duplicate_dates": 0,
            "nonfinite_prices_or_returns": 0,
            "nonpositive_ohlc_or_volume": 0,
            "missing_expected_sessions": missing,
            "unexpected_sessions": extra,
            "expected_sessions": len(expected),
            "calendar": "Bounded 2016-2025 NYSE holiday-rule audit using existing pandas; no new dependency. Includes Juneteenth from 2022, 2018-12-05 and 2025-01-09 closures; Saturday New Year has no Friday closure.",
            "calendar_sources": CALENDAR_SOURCES,
            "dividends_without_price": 0,
            "dividend_basis_mismatches": 0,
            "zero_loss_count": int((result.value == 0).sum()),
            "positive_loss_count": int((result.value > 0).sum()),
            "max_loss": float(result.value.max()),
            "absolute_log_return_above_10pct": json.loads(
                extreme_rows.to_json(orient="records", date_format="iso")
            ),
            "native_daily_close_checks": native_close_checks,
            "anomaly_note": "All >10% absolute log-return sessions and their prior closes matched the provider's separate daily-summary endpoint. This is within-provider consistency, not independent vendor validation. No winsorization or replacement.",
        },
        "reproduce": "uv run python scripts/data_prep/finance_snapshot.py",
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "raw_sha256": source_hashes,
        "processed_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / f"{key}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "ticker": ticker,
                "coverage": metadata["coverage"],
                "zero_loss_count": metadata["quality_checks"]["zero_loss_count"],
                "max_loss": metadata["quality_checks"]["max_loss"],
                "calendar_missing": missing,
                "calendar_extra": extra,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    self_check()
    for symbol in ["SPY", "QQQ"]:
        build(symbol)
