#!/usr/bin/env python3
"""NOAA science XRS-B minute snapshot; run with uv run --with netCDF4 python scripts/data_prep/goes_snapshot.py."""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import urllib.request
import numpy as np
import pandas as pd
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/goes"
OUT = ROOT / "data/processed/inputs"
START, STOP = pd.Timestamp("1995-01-03"), pd.Timestamp("2026-01-01")
LEGACY = "https://data.ngdc.noaa.gov/instruments/solar-space-observing/particle-detectors/sem/goes/access/science/xrs/"
MODERN = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/"
# Expanded NOAA legacy Readme Appendix A, Table 11; blanks carry forward in time.
SCHEDULE = [
    ("1995-01-03", 7, 8),
    ("1995-03-01", 8, 7),
    ("1998-07-27", 8, 10),
    ("2003-04-08 15:00", 10, 12),
    ("2003-05-15 15:00", 12, 10),
    ("2006-06-28", 12, 11),
    ("2007-01-01", 10, 11),
    ("2007-04-12", 11, 10),
    ("2008-02-10 16:30", 10, 0),
    ("2009-12-01", 14, 0),
    ("2010-09-01", 14, 15),
    ("2010-10-28", 15, 0),
    ("2011-09-01", 15, 14),
    ("2012-10-23 16:00", 14, 15),
    ("2012-11-19 16:31", 15, 14),
    ("2015-01-26 16:01", 15, 13),
    ("2015-05-21 18:00", 14, 13),
    ("2015-06-09 16:25", 15, 13),
    ("2016-05-03 13:00", 13, 14),
    ("2016-05-12 17:30", 14, 13),
    ("2016-05-16 17:00", 14, 15),
    ("2016-06-09 17:30", 15, 13),
    ("2017-02-07", 16, 15),
    ("2018-06-01", 16, 17),
    ("2023-01-10", 16, 18),
    ("2025-04-07", 18, 19),
]
DOCS = [LEGACY + "GOES_1-15_XRS_Science-Quality_Data_Readme.pdf"] + [
    MODERN + "goes16/l2/docs/" + n
    for n in ("GOES-R_XRS_L2_Data_Users_Guide.pdf", "GOES-R_XRS_L2_Data_ReadMe.pdf")
]


def download(url):
    p = RAW / url.rsplit("/", 1)[-1]
    cached = p.exists()
    begun = time.perf_counter()
    if not cached:
        temp = p.with_suffix(p.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temp.replace(p)
    digest = hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
    return {
        "url": url,
        "path": str(p.relative_to(ROOT)),
        "bytes": p.stat().st_size,
        "sha256": digest,
        "cached": cached,
        "download_seconds": time.perf_counter() - begun,
    }


def scalar(value):
    return (
        value.tolist()
        if isinstance(value, np.ndarray)
        else value.item()
        if isinstance(value, np.generic)
        else value
    )


def attrs(variable):
    return {k: scalar(v) for k, v in variable.__dict__.items()}


def bit_condition(variable, values, meaning):
    names = variable.flag_meanings.split()
    if meaning not in names:
        return np.zeros(values.shape, dtype=bool)
    i = names.index(meaning)
    return (values & variable.flag_masks[i]) == variable.flag_values[i]


def max_gap(valid):
    points = np.r_[-1, np.flatnonzero(valid), len(valid)]
    return int(np.diff(points).max() - 1)


def self_test():
    assert max_gap(np.ones(1440, dtype=bool)) == 0
    assert max_gap(np.zeros(1440, dtype=bool)) == 1440
    assert max_gap(np.array([0, 0, 1, 0, 1, 0, 0, 0], dtype=bool)) == 3
    f = np.array([0, 1, 2, 4, 5, 255], dtype=np.uint16)

    class Flags:
        flag_meanings = "good_data eclipse bad_data interpolated"
        flag_masks = np.array([3, 1, 2, 4])
        flag_values = np.array([0, 1, 2, 4])

    assert bit_condition(Flags, f, "interpolated").tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    # Flux interpolation is rejected even though the good_data mask ignores that bit.
    assert ((f == 0) & bit_condition(Flags, f, "good_data")).sum() == 1
    ranks = np.array([0, 1, 255])
    proposed = np.array([1, 0, 1])
    assert (proposed < ranks).tolist() == [False, True, True]
    print("self_test: passed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    begun = time.perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(START, STOP - pd.Timedelta(days=1), freq="D")
    n = len(dates) * 1440
    primary = np.zeros(n, dtype=np.uint8)
    secondary = np.zeros(n, dtype=np.uint8)
    intervals = []
    for i, (stamp, p, s) in enumerate(SCHEDULE):
        end = pd.Timestamp(SCHEDULE[i + 1][0]) if i + 1 < len(SCHEDULE) else STOP
        a, b = (
            int((pd.Timestamp(stamp) - START).total_seconds() / 60),
            int((end - START).total_seconds() / 60),
        )
        # NOAA caveat 6.2 recommends GOES-13 only for gap filling.
        actual = (s, p) if p == 13 else (p, s)
        primary[a:b], secondary[a:b] = actual
        intervals.append(
            {
                "start": str(pd.Timestamp(stamp)),
                "end_exclusive": str(end),
                "noaa_primary": p,
                "noaa_secondary": s,
                "selection_priority": list(actual),
            }
        )
    needed = set()
    for year in range(START.year, STOP.year):
        a = max(0, int((pd.Timestamp(year, 1, 1) - START).total_seconds() / 60))
        b = min(n, int((pd.Timestamp(year + 1, 1, 1) - START).total_seconds() / 60))
        needed.update(
            (int(sat), year) for sat in np.unique(np.r_[primary[a:b], secondary[a:b]]) if sat >= 8
        )
    urls = []
    listings = []
    missing = []
    for sat in sorted({s for s, y in needed}):
        base = (
            LEGACY + f"goes{sat:02}/" if sat < 16 else MODERN + f"goes{sat}/l2/data/"
        ) + "xrsf-l2-avg1m_science/"
        html = (
            (RAW / f"index_g{sat:02}.html").read_text()
            if args.offline
            else urllib.request.urlopen(base, timeout=60).read().decode()
        )
        (RAW / f"index_g{sat:02}.html").write_text(html)
        listings.append(base)
        files = re.findall(r'href="([^\"]+\.nc)"', html)
        for s, y in sorted(needed):
            if s != sat:
                continue
            options = [f for f in files if f"_y{y}_v" in f]
            if options:
                urls.append(
                    base
                    + max(
                        options,
                        key=lambda f: tuple(map(int, re.search(r"_v([\d-]+)", f)[1].split("-"))),
                    )
                )
            else:
                missing.append(
                    {
                        "satellite": sat,
                        "year": y,
                        "reason": "no annual science file in official listing",
                    }
                )
    all_urls = DOCS + urls
    if args.offline:
        assert all((RAW / u.rsplit("/", 1)[-1]).exists() for u in all_urls), (
            "Offline source file missing"
        )
    (RAW / "requested_urls.json").write_text(json.dumps(all_urls, indent=2))
    print(f"download: {len(urls)} annual files; missing satellite-years: {missing}", flush=True)
    inventory = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(download, u): u for u in all_urls}
        for future in as_completed(jobs):
            row = future.result()
            inventory.append(row)
            print(
                f"download {len(inventory)}/{len(all_urls)} {Path(row['path']).name} {row['bytes']} bytes {row['download_seconds']:.1f}s cached={row['cached']}",
                flush=True,
            )
    (RAW / "inventory.json").write_text(json.dumps(inventory, indent=2))
    flux = np.full(n, np.nan, dtype=np.float32)
    rank = np.full(n, 255, dtype=np.uint8)
    sat_selected = np.zeros(n, dtype=np.uint8)
    electron = np.zeros(n, dtype=np.uint16)
    saturation = np.zeros(n, dtype=bool)
    metadata = []
    for url in urls:
        path = RAW / url.rsplit("/", 1)[-1]
        sat = int(re.search(r"_g(\d+)_", path.name)[1])
        with Dataset(path) as d:
            tv = d["time"]
            origin = pd.Timestamp(tv.units.split("since ")[1].replace(" UTC", ""))
            minute = (np.asarray(tv[:]) + ((origin - START).total_seconds())) / 60
            assert np.isfinite(minute).all() and np.all(np.abs(minute - np.rint(minute)) < 1e-6)
            index = np.rint(minute).astype(np.int64)
            assert np.all(np.diff(index) > 0), path
            in_range = (index >= 0) & (index < n)
            idx = index[in_range]
            fvar = d["xrsb_flux"]
            v = np.ma.filled(fvar[:], np.nan)[in_range]
            flagname = "xrsb_flag" if "xrsb_flag" in d.variables else "xrsb_flags"
            flagvar = d[flagname]
            flags = np.ma.filled(flagvar[:].astype(np.uint16), 65535)[in_range]
            candidate_rank = np.where(
                primary[idx] == sat, 0, np.where(secondary[idx] == sat, 1, 255)
            ).astype(np.uint8)
            scheduled = candidate_rank < 255
            valid = np.isfinite(v) & (v > 0) & (flags == 0)
            satmask = (~np.ma.getmaskarray(flagvar[:])[in_range]) & bit_condition(
                flagvar, flags, "saturation"
            )
            exname = flagname + "_excluded"
            if exname in d.variables:
                ex = d[exname]
                ev = ex[:]
                exvalid = ~np.ma.getmaskarray(ev)[in_range]
                ev = np.ma.filled(ev, 0).astype(np.uint16)[in_range]
                satmask |= exvalid & bit_condition(ex, ev, "saturated")
            valid &= ~satmask
            saturation[idx[scheduled & satmask]] = True
            ef = np.zeros(len(idx), dtype=np.uint16)
            if "electron_correction_flag" in d.variables:
                ev = d["electron_correction_flag"]
                ef = np.ma.filled(ev[:], 65535).astype(np.uint16)[in_range]
                valid &= bit_condition(ev, ef, "e_correction_valid")
            take = scheduled & valid & (candidate_rank < rank[idx])
            outidx = idx[take]
            flux[outidx] = v[take]
            rank[outidx] = candidate_rank[take]
            sat_selected[outidx] = sat
            electron[outidx] = ef[take]
            metadata.append(
                {
                    "file": path.name,
                    "global": attrs(d),
                    "actual_timestamp_start_utc": str(START + pd.Timedelta(minutes=int(index[0]))),
                    "actual_timestamp_end_utc": str(START + pd.Timedelta(minutes=int(index[-1]))),
                    "variables": {
                        key: attrs(d[key])
                        for key in [
                            "time",
                            "xrsb_flux",
                            flagname,
                            "xrsb_num",
                            "electron_correction_flag",
                            exname,
                        ]
                        if key in d.variables
                    },
                    "scheduled_rows": int(scheduled.sum()),
                    "valid_scheduled_rows": int((scheduled & valid).sum()),
                }
            )
        print(f"processed {path.name}", flush=True)
    unresolved = saturation & ~np.isfinite(flux)
    minute_path = OUT / "goes_minutes.npz"
    np.savez_compressed(
        minute_path,
        flux=flux,
        satellite=sat_selected,
        rank=rank,
        unresolved_saturation=unresolved,
    )
    good = np.isfinite(flux)
    edges = np.diff(np.r_[False, good, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = ends - starts
    order = np.argsort(lengths)[::-1][:20]
    segments = [
        {
            "start": str(START + pd.Timedelta(minutes=int(starts[i]))),
            "end": str(START + pd.Timedelta(minutes=int(ends[i] - 1))),
            "minutes": int(lengths[i]),
            "days": float(lengths[i] / 1440),
        }
        for i in order
    ]
    report = {
        "dataset": "NOAA science-quality XRS-B 0.1-0.8 nm, 1-minute mean irradiance",
        "units": "W/m2",
        "frequency": "1min",
        "calendar_start": str(START),
        "calendar_end": str(STOP - pd.Timedelta(minutes=1)),
        "calendar_minutes": int(n),
        "valid_minutes": int(good.sum()),
        "invalid_minutes": int((~good).sum()),
        "eligible_runs": int(len(starts)),
        "longest_segments": segments,
        "priority_intervals": intervals,
        "source_files": inventory,
        "quality": "finite positive, unmasked flux; all flux quality flags zero; valid electron correction for GOES-R; no saturation; primary then secondary; no interpolation or time compression",
        "daily_coverage_threshold_applied": False,
        "source_netcdf_metadata": metadata,
        "sha256": hashlib.file_digest(minute_path.open("rb"), "sha256").hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - begun,
    }
    (OUT / "goes_minutes.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("source_files", "source_netcdf_metadata", "priority_intervals")
            },
            indent=2,
        ),
        flush=True,
    )
    return


if __name__ == "__main__":
    main()
