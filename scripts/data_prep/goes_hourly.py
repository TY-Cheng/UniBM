"""Build UTC hourly maxima from quality-screened, uncompressed minute inputs."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/pilots"


def aggregate(flux, unresolved):
    """Keep >=57 valid minutes and reject any unresolved saturation in the hour."""
    x = np.asarray(flux).reshape(-1, 60)
    good = np.isfinite(x)
    counts = good.sum(axis=1)
    maxima = np.where(good, x, -np.inf).max(axis=1)
    maxima[counts == 0] = np.nan
    sat = np.asarray(unresolved).reshape(-1, 60).sum(axis=1)
    eligible = (counts >= 57) & (sat == 0)
    return maxima, counts, sat, eligible


def selfcheck():
    x = np.arange(240, dtype=float).reshape(4, 60) + 1
    x[0, :3] = np.nan
    x[1, :4] = np.nan
    x[3] = np.nan
    sat = np.zeros(240, bool)
    sat[120] = True
    maxima, counts, saturation, eligible = aggregate(x.ravel(), sat)
    assert eligible.tolist() == [True, False, False, False]
    assert counts.tolist() == [57, 56, 60, 0]
    assert maxima[0] == 60 and np.isnan(maxima[3]) and saturation[2] == 1


if __name__ == "__main__":
    selfcheck()
    path = DATA / "goes_minutes.npz"
    minute = json.loads((DATA / "goes_minutes.json").read_text())
    with np.load(path) as z:
        maxima, counts, saturation, eligible = aggregate(z["flux"], z["unresolved_saturation"])
    times = pd.date_range(minute["calendar_start"], periods=len(maxima), freq="h")
    frame = pd.DataFrame(
        {
            "date": times,
            "value": np.where(eligible, maxima, np.nan),
            "raw_hourly_max": maxima,
            "valid_minutes": counts,
            "unresolved_saturation_minutes": saturation,
            "eligible": eligible,
        }
    )
    dest = DATA / "goes_hourly.csv"
    frame.to_csv(dest, index=False, float_format="%.10g")
    edges = np.diff(np.r_[False, eligible, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    order = np.argsort(ends - starts)[::-1][:20]
    segments = [
        {
            "start": str(times[starts[i]]),
            "end": str(times[ends[i] - 1]),
            "hours": int(ends[i] - starts[i]),
            "days": float((ends[i] - starts[i]) / 24),
        }
        for i in order
    ]
    meta = {
        "dataset": "NOAA science-quality GOES XRS-B hourly maximum of 1-minute mean irradiance",
        "units": "W/m2",
        "frequency": "1h",
        "calendar_start": str(times[0]),
        "calendar_end": str(times[-1]),
        "calendar_hours": len(times),
        "eligible_hours": int(eligible.sum()),
        "unresolved_saturation_hours": int((saturation > 0).sum()),
        "saturation_hour_dates": [str(t) for t in times[saturation > 0]],
        "hour_rule": "UTC hour [h,h+1h), >=57/60 valid minutes; reject any unresolved saturation; no interpolation, no compression, no daily coverage filter",
        "minute_quality": minute["quality"],
        "priority_intervals": minute["priority_intervals"],
        "source_files": minute["source_files"],
        "longest_segments": segments,
        "minute_input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "aggregator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "caveats": [
            "Missing minutes can hide the actual hourly peak even at 95% completeness.",
            "Archive span is not the fitted continuous-run span.",
            "Solar cycles, residual intersatellite calibration, and legacy measurement floors remain.",
        ],
    }
    (DATA / "goes_hourly.json").write_text(json.dumps(meta, indent=2))
    print(
        json.dumps(
            {k: v for k, v in meta.items() if k not in ("source_files", "priority_intervals")},
            indent=2,
        )
    )
