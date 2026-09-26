"""Protect the statistical clock, lagged scale, and optional-input boundary."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from application.docs_cases import build_documented_cases, load_extra_input, segmented_scale
from application.normalization import ewma_scale, longest_valid_frame
from data_prep.finance_snapshot import self_check as finance_self_check
from data_prep.goes_hourly import aggregate
from shared.runtime import bootstrap_thread_cap
from unibm.evi import block_maxima


def test_lagged_ewma_preserves_zeros_and_does_not_use_future_returns():
    returns = np.array([-1.0, 2.0, -3.0, 0.0, 10.0, -6.0])
    scale = ewma_scale(returns, 3, 0.94, squared=True)
    variance = np.mean(returns[:3] ** 2)
    for t in range(3, len(returns)):
        assert scale[t] ** 2 == pytest.approx(variance)
        variance = 0.94 * variance + 0.06 * returns[t] ** 2
    altered = returns.copy()
    altered[4:] = 999
    np.testing.assert_allclose(ewma_scale(altered, 3, 0.94, True)[:5], scale[:5])
    losses = np.maximum(-returns[3:], 0)
    np.testing.assert_array_equal(losses / scale[3:] == 0, losses == 0)
    with pytest.raises(ValueError, match="Nonpositive"):
        ewma_scale(np.zeros(8), 3, 0.5)


def test_gaps_restart_warmup_and_windows_never_cross_them():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0])
    scale = segmented_scale(x, 3, 0.5)
    np.testing.assert_allclose(scale[3:5], [7 / 3, 31 / 6])
    assert np.isnan(scale[5:9]).all()
    np.testing.assert_allclose(scale[9:], [3.0, 4.0])
    altered = x.copy()
    altered[:5] *= 100
    np.testing.assert_allclose(segmented_scale(altered, 3, 0.5)[6:], scale[6:])
    y = x / scale
    for b in (2, 3, 4):
        expected = [
            max(y[i : i + b]) for i in range(len(y) - b + 1) if np.isfinite(y[i : i + b]).all()
        ]
        np.testing.assert_allclose(block_maxima(y, b), expected)
    frame = pd.DataFrame(
        {"value": [1.0, 2.0, np.nan, 3.0, 4.0]}, index=pd.date_range("2000-01-01", periods=5)
    )
    selected, _ = longest_valid_frame(frame, "D")
    assert selected.index[0] == pd.Timestamp("2000-01-04")


def test_hourly_quality_and_finance_adjustments():
    x = np.arange(180, dtype=float) + 1
    x[:3] = np.nan
    x[60:64] = np.nan
    saturated = np.zeros(180, bool)
    saturated[120] = True
    maxima, counts, _, eligible = aggregate(x, saturated)
    assert eligible.tolist() == [True, False, False]
    assert counts.tolist() == [57, 56, 60]
    assert maxima[0] == 60
    finance_self_check()


def test_absent_optional_inputs_keep_assets_but_corrupt_inputs_fail():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        web = root / "docs/assets/cases"
        web.mkdir(parents=True)
        asset = web / "goes_normalized.png"
        asset.write_bytes(b"frozen")
        parent_cap = bootstrap_thread_cap()
        build_documented_cases(root, keys=["goes"], available=True)
        assert bootstrap_thread_cap() == parent_cap
        assert asset.read_bytes() == b"frozen"
        local = root / "data/processed/inputs"
        local.mkdir(parents=True)
        (local / "goes_hourly.csv").write_text("not a valid input")
        with pytest.raises(FileNotFoundError, match="Incomplete"):
            build_documented_cases(root, keys=["goes"], available=True)
        (local / "goes_hourly.json").write_text(json.dumps({"output_sha256": "wrong"}))
        with pytest.raises(ValueError, match="hash"):
            load_extra_input("goes", root)
