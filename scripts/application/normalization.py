"""Reproducible, past-only scale adjustment for documented real-data cases."""

from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from data_prep.ghcn import PreparedSeries


def ewma_scale(values, warmup, decay, squared=False):
    """Initialize on warmup observations; scale at t only incorporates indices <t."""
    x = np.asarray(values, dtype=float)
    if len(x) <= warmup or not np.isfinite(x).all():
        raise ValueError("Insufficient or nonfinite scale inputs")
    z = x * x if squared else x
    seed = float(z[:warmup].mean())
    if seed <= 0:
        raise ValueError("Nonpositive EWMA initialization; no arbitrary epsilon is added")
    scale = np.full(len(x), np.nan)
    scale[warmup] = seed
    if len(x) > warmup + 1:
        scale[warmup + 1 :] = lfilter([1 - decay], [1, -decay], z[warmup:-1], zi=[decay * seed])[0]
    if squared:
        scale = np.sqrt(scale)
    if not np.isfinite(scale[warmup:]).all() or (scale[warmup:] <= 0).any():
        raise ValueError("Invalid resolved EWMA scale; preserve evidence instead of clipping")
    return scale


def longest_valid_frame(frame, calendar):
    """Select the longest complete run on the supplied calendar frequency."""
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Source dates must be unique and sorted")
    if calendar:
        frame = frame.reindex(
            pd.date_range(
                frame.index.min(),
                frame.index.max(),
                freq=calendar if isinstance(calendar, str) else "D",
            )
        )
    valid = np.isfinite(frame.value) & (frame.value >= 0)
    if "return" in frame:
        valid &= np.isfinite(frame["return"])
    groups = (~valid).cumsum()
    counts = frame.loc[valid].groupby(groups[valid]).size()
    if counts.empty:
        raise ValueError("No valid input run")
    # A length tie is resolved in favor of the latest run, independent of fitted results.
    winner = counts[counts == counts.max()].index[-1]
    selected = frame.loc[valid & (groups == winner)].copy()
    return selected, {
        "all_rows": len(frame),
        "valid_rows": int(valid.sum()),
        "missing_or_invalid_rows": int((~valid).sum()),
        "longest_run_rows": len(selected),
        "eligible_runs": len(counts),
        "selection": "longest contiguous valid run; latest on length tie",
    }


def climate_frame(source: Path, key: str):
    """Build full-calendar QC data; Phoenix uses the existing retrospective index formula."""
    from data_prep.ghcn import read_ghcn_station_csv, _extract_ghcn_element

    station = "USW00012918" if key == "houston" else "USW00023183"
    records = read_ghcn_station_csv(source)
    records = records.loc[records.date <= "2025-12-31"]
    rain = _extract_ghcn_element(records, "PRCP", scale=10.0)
    index = pd.date_range(rain.index.min(), "2025-12-31", freq="D")
    frame = pd.DataFrame({"prcp_mm": rain.reindex(index)})
    frame["value"] = frame.prcp_mm
    detail = {"formula": "quality-controlled daily PRCP / 10 in mm; true zeros retained"}
    if key == "phoenix":
        frame["tmax_c"] = _extract_ghcn_element(records, "TMAX", scale=10.0).reindex(index)
        frame["prcp_roll_mm"] = frame.prcp_mm.rolling(30, min_periods=30).sum()
        # Select by input availability, before looking at the severity estimates.
        frame["value"] = frame.prcp_roll_mm.where(np.isfinite(frame.tmax_c))
        ready, readiness = longest_valid_frame(frame, calendar="D")
        doy = ready.index.dayofyear
        for column in ("tmax_c", "prcp_roll_mm"):
            mean = ready.groupby(doy)[column].transform("mean")
            sd = ready.groupby(doy)[column].transform("std")
            if not np.isfinite(sd).all() or (sd <= 0).any():
                raise ValueError("Phoenix day-of-year baseline has an undefined scale")
            ready[column + "_mean"] = mean
            ready[column + "_sd"] = sd
        hot = (ready.tmax_c - ready.tmax_c_mean) / ready.tmax_c_sd
        dry = (ready.prcp_roll_mm_mean - ready.prcp_roll_mm) / ready.prcp_roll_mm_sd
        frame["value"] = np.nan
        frame.loc[ready.index, "value"] = hot.clip(lower=0) + dry.clip(lower=0)
        detail = {
            "formula": "max((TMAX - mean_doy(TMAX))/sd_doy(TMAX), 0) + max((mean_doy(P30) - P30)/sd_doy(P30), 0)",
            "rolling_days": 30,
            "rolling_min_periods": 30,
            "baseline": "retained full-year contiguous run; dayofyear means and sample SD, same grouping as original case",
            "baseline_start": str(ready.index[0]),
            "baseline_end": str(ready.index[-1]),
            "input_readiness_audit": readiness,
            "retrospective": "full-period climatology includes future observations relative to an earlier date; not a real-time index",
            "dependence": "overlapping 30-day precipitation sums induce persistence in the derived index",
        }
        for column in ("tmax_c_mean", "tmax_c_sd", "prcp_roll_mm_mean", "prcp_roll_mm_sd"):
            frame[column] = ready[column].reindex(frame.index)
    return frame, {
        "station_id": station,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        **detail,
    }


def prepare_normalized_climate(source: Path, key: str) -> PreparedSeries:
    """Normalize the longest full-calendar run; retain zeros and omit 30 warmup days."""
    frame, detail = climate_frame(source, key)
    raw, audit = longest_valid_frame(frame, "D")
    scale = ewma_scale(raw.value.to_numpy(), 30, 2 ** (-1 / 180))
    series = (raw.value / scale).iloc[30:]
    metadata = {
        **detail,
        **audit,
        "normalization": "past-only EWMA level",
        "warmup_days": 30,
        "half_life_days": 180,
        "normalization_formula": "Y_t=X_t/m_t; m_t=rho*m_(t-1)+(1-rho)*X_(t-1)",
        "maxima_period": "annual",
        "months": list(range(1, 13)),
        "ci_scope": "conditional on preprocessing and selected window",
    }
    return PreparedSeries(key, "normalized", series, series.resample("YE").max(), metadata)
