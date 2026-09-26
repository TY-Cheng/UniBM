"""Rebuild static case figures without report export or network access.

Run ``PYTHONPATH=scripts uv run python -m application.docs_cases``.
Finance and GOES require the local provider snapshots documented on their pages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from application.fit import build_application_bundle
from application.inputs import build_application_inputs
from application.normalization import ewma_scale
from application.specs import (
    APPLICATIONS,
    ApplicationBundle,
    ApplicationPreparedInputs,
    ApplicationSpec,
)
from config import resolve_repo_dirs
from data_prep.ghcn import PreparedSeries
from shared.runtime import initialize_numerical_worker, status
from unibm.evi import estimate_evi_quantile

ROOT = Path(__file__).resolve().parents[2]
EXTRA_KEYS = ("goes", "spy", "qqq")


def extra_input_available(key: str, root: Path) -> bool:
    """Skip absent snapshots, but reject an incomplete CSV/provenance pair."""
    name = "goes_hourly" if key == "goes" else key
    base = root / "data/processed/inputs" / name
    present = [base.with_suffix(ext).is_file() for ext in (".csv", ".json")]
    if any(present) and not all(present):
        raise FileNotFoundError(f"Incomplete prepared input: {base} requires CSV and JSON")
    return all(present)


def segmented_scale(values, warmup=720, decay=2 ** (-1 / 4320)):
    """Restart the lagged level EWMA after each gap; keep warmup positions missing."""
    x = np.asarray(values, dtype=float)
    edges = np.diff(np.r_[False, np.isfinite(x), False].astype(int))
    scale = np.full(len(x), np.nan)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start > warmup:
            scale[start:stop] = ewma_scale(x[start:stop], warmup, decay)
    return scale


def load_extra_input(key: str, root: Path):
    """Check local prepared-input provenance before reconstructing normalization."""
    name = "goes_hourly" if key == "goes" else key
    source = root / "data/processed/inputs" / f"{name}.csv"
    metadata = json.loads(source.with_suffix(".json").read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = metadata["output_sha256" if key == "goes" else "processed_sha256"]
    if digest != expected:
        raise ValueError(f"Prepared input hash differs from its provider record: {source}")
    frame = pd.read_csv(source, parse_dates=["date"]).set_index("date")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Input dates must be sorted and unique")
    if frame.index.max() > pd.Timestamp("2025-12-31 23:59:59"):
        raise ValueError("These documented snapshots end on 2025-12-31")
    if key == "goes":
        expected_dates = pd.date_range(frame.index.min(), frame.index.max(), freq="h", name="date")
        if not frame.index.equals(expected_dates):
            raise ValueError("GOES input must retain every calendar hour, including missing hours")
        eligible = (frame.valid_minutes >= 57) & (frame.unresolved_saturation_minutes == 0)
        if not np.array_equal(np.isfinite(frame.value), eligible):
            raise ValueError("GOES values disagree with the hourly quality mask")
        if (frame.value.dropna() <= 0).any():
            raise ValueError("GOES irradiance must be positive")
        scale = segmented_scale(frame.value.to_numpy())
        normalization = {
            "kind": "lagged EWMA level",
            "warmup_observations": 720,
            "half_life_observations": 4320,
            "reset_after_gap": True,
        }
    else:
        from data_prep.finance_snapshot import sessions

        if not frame.index.equals(sessions(frame.index.min(), frame.index.max())):
            raise ValueError("Finance input is missing expected trading sessions")
        if not np.isfinite(frame[["value", "return"]]).all().all():
            raise ValueError("Nonfinite finance observation")
        if not np.allclose(frame.value, (-frame["return"]).clip(lower=0), rtol=1e-12, atol=1e-15):
            raise ValueError("Finance values must retain gains as zero loss")
        scale = ewma_scale(frame["return"].to_numpy(), 252, 0.94, squared=True)
        normalization = {
            "kind": "lagged EWMA root mean square of signed returns",
            "warmup_observations": 252,
            "decay": 0.94,
        }
    frame["scale"] = scale
    frame["normalized"] = frame.value / scale
    if key != "goes":
        frame = frame.iloc[252:]
    provenance = {
        "source_sha256": digest,
        "normalization": normalization,
        "input": str(source.relative_to(root)),
        "ci_scope": "not reported"
        if key == "goes"
        else "conditional on preprocessing; EWMA not re-estimated in bootstrap",
    }
    return frame, provenance


def extra_bundle(key: str, root: Path):
    """Reuse application FGLS for finance, and explicit point-only OLS for gapped GOES."""
    frame, metadata = load_extra_input(key, root)
    series = frame.normalized
    label = "GOES XRS-B" if key == "goes" else key.upper() + " left-tail loss"
    p = PreparedSeries(key, "normalized", series, series.resample("YE").max(), metadata)
    spec = ApplicationSpec(
        key=key,
        provider="NOAA" if key == "goes" else "Massive",
        label=label + " · EWMA normalized",
        figure_stem=key + "_normalized",
        raw_key=key,
        ylabel="irradiance / past EWMA level"
        if key == "goes"
        else "log loss / past EWMA volatility",
        time_series_title="Hourly history · gaps and warmups retained",
        scaling_title="Complete-window quantile scaling · OLS"
        if key == "goes"
        else "Sliding block-maxima quantile scaling",
        scaling_ylabel="log block-maximum quantile",
        observations_per_year=8766 if key == "goes" else 252,
        design_life_level_basis="calendar_hour" if key == "goes" else "trading_day",
        design_life_level_yscale="log",
        target_stability_title="Summary stability across block sizes",
        formal_ei=key != "goes",
        ei_allow_zeros=True,
    )
    inputs = ApplicationPreparedInputs(p, p, p)
    if key != "goes":
        return build_application_bundle(spec, inputs)
    fit = estimate_evi_quantile(series.to_numpy(), regression="OLS", quantile=0.5, sliding=True)
    valid = np.isfinite(series.to_numpy())
    edges = np.diff(np.r_[False, valid, False].astype(int))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    counts = [np.maximum(lengths - b + 1, 0).sum() for b in fit.curve.block_sizes]
    np.testing.assert_array_equal(counts, fit.curve.counts)
    return ApplicationBundle(spec, inputs, fit, None, None, None, None, None)


def build_documented_cases(root=ROOT, *, keys=None, available=False):
    """Render selected cases; absent optional inputs are reported, never downloaded."""
    from application.outputs import write_application_web_figure

    root = Path(root)
    keys = list(keys) if keys is not None else [s.key for s in APPLICATIONS] + list(EXTRA_KEYS)
    known = {s.key: s for s in APPLICATIONS}
    web = root / "docs/assets/cases"
    web.mkdir(parents=True, exist_ok=True)
    for key in keys:
        if key not in known and key not in EXTRA_KEYS:
            raise ValueError(f"Unknown documented case: {key}")
        if key in EXTRA_KEYS and available and not extra_input_available(key, root):
            status("cases", f"{key}: local input unavailable; keeping frozen docs assets")
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if key in known:
                spec = known[key]
                prepared = build_application_inputs(resolve_repo_dirs(root), specs=(spec,))
                bundle = build_application_bundle(spec, prepared[key])
            else:
                bundle = extra_bundle(key, root)
            write_application_web_figure(bundle, web)
        for warning in caught:
            status("cases", f"{key}: {warning.message}")
        status("cases", f"{key}: figure and JSON written; xi={bundle.evi_fit.slope:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", nargs="+")
    parser.add_argument("--available", action="store_true")
    args = parser.parse_args()
    initialize_numerical_worker()
    build_documented_cases(keys=args.keys, available=args.available)
