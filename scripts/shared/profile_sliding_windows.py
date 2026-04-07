"""Profile guarded sliding-window helpers against the pre-refactor baselines."""
# ruff: noqa: E402

from __future__ import annotations

if __package__ in {None, ""}:
    import importlib.util
    from pathlib import Path

    _helper_path = Path(__file__).resolve().parents[1] / "shared" / "import_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("_shared_import_bootstrap", _helper_path)
    if _spec is None or _spec.loader is None:  # pragma: no cover - import bootstrap failure
        raise ImportError(f"Could not load import bootstrap helper from {_helper_path}.")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    _module.ensure_scripts_on_path_from_entry(__file__)

from dataclasses import dataclass
from time import perf_counter
import tracemalloc

import numpy as np
import pandas as pd

from config import resolve_repo_dirs
from unibm.bootstrap import _sliding_block_maxima
from unibm.core import block_maxima
from unibm.extremal_index import _rolling_window_minima
from shared.runtime import status


PROFILE_REPEATS = 3
PROFILE_RANDOM_STATE = 20260407


@dataclass(frozen=True)
class SlidingWindowScenario:
    """One input family for the sliding-window profiling harness."""

    name: str
    n_obs: int
    window: int
    missing_rate: float = 0.0


def _baseline_sliding_window_extreme_valid(
    vec: np.ndarray,
    window: int,
    *,
    reducer: str,
) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if window < 2 or arr.size < window:
        return np.asarray([], dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(arr, window)
    valid = np.all(np.isfinite(windows), axis=1)
    if reducer == "max":
        return windows.max(axis=1)[valid]
    return windows.min(axis=1)[valid]


def _baseline_circular_sliding_window_maximum(
    vec: np.ndarray,
    window: int,
) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if window < 2 or arr.size < window:
        return np.asarray([], dtype=float)
    wrapped = np.concatenate([arr, arr[: window - 1]])
    windows = np.lib.stride_tricks.sliding_window_view(wrapped, window)[: arr.size]
    return windows.max(axis=1)


def _make_series(scenario: SlidingWindowScenario) -> np.ndarray:
    rng = np.random.default_rng(PROFILE_RANDOM_STATE + scenario.n_obs + scenario.window)
    values = rng.standard_normal(scenario.n_obs).astype(float)
    if scenario.missing_rate > 0:
        n_missing = max(1, int(round(scenario.n_obs * scenario.missing_rate)))
        missing_idx = rng.choice(scenario.n_obs, size=n_missing, replace=False)
        values[missing_idx] = np.nan
    return values


def _measure_callable(func, *args) -> tuple[np.ndarray, float, int]:
    runtimes: list[float] = []
    peaks: list[int] = []
    result: np.ndarray | None = None
    for _ in range(PROFILE_REPEATS):
        tracemalloc.start()
        start = perf_counter()
        result = np.asarray(func(*args), dtype=float)
        runtimes.append(perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
        peaks.append(int(peak))
        tracemalloc.stop()
    return (
        result if result is not None else np.asarray([], dtype=float),
        float(np.median(runtimes)),
        int(max(peaks)),
    )


def _profile_operation(
    scenario: SlidingWindowScenario,
    *,
    operation: str,
    current,
    baseline,
) -> list[dict[str, object]]:
    values = _make_series(scenario)
    current_result, current_time, current_peak = _measure_callable(
        current, values, scenario.window
    )
    baseline_result, baseline_time, baseline_peak = _measure_callable(
        baseline, values, scenario.window
    )
    np.testing.assert_allclose(current_result, baseline_result, equal_nan=True)
    finite_share = float(np.mean(np.isfinite(values))) if values.size else 0.0
    speedup = baseline_time / current_time if current_time > 0 else np.inf
    peak_ratio = baseline_peak / current_peak if current_peak > 0 else np.inf
    return [
        {
            "operation": operation,
            "scenario": scenario.name,
            "implementation": "current",
            "n_obs": scenario.n_obs,
            "window": scenario.window,
            "finite_share": finite_share,
            "runtime_seconds": current_time,
            "peak_bytes": current_peak,
            "speedup_vs_baseline": speedup,
            "peak_ratio_vs_baseline": peak_ratio,
        },
        {
            "operation": operation,
            "scenario": scenario.name,
            "implementation": "baseline",
            "n_obs": scenario.n_obs,
            "window": scenario.window,
            "finite_share": finite_share,
            "runtime_seconds": baseline_time,
            "peak_bytes": baseline_peak,
            "speedup_vs_baseline": 1.0,
            "peak_ratio_vs_baseline": 1.0,
        },
    ]


def build_sliding_window_profile() -> pd.DataFrame:
    """Benchmark current guarded window ops against the original baselines."""
    scenarios = [
        SlidingWindowScenario(name="daily_finite", n_obs=365 * 40, window=30, missing_rate=0.0),
        SlidingWindowScenario(name="daily_missing", n_obs=365 * 40, window=30, missing_rate=0.05),
        SlidingWindowScenario(name="stress_finite", n_obs=250_000, window=128, missing_rate=0.0),
        SlidingWindowScenario(name="stress_missing", n_obs=250_000, window=128, missing_rate=0.01),
    ]
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        status("window_profile", f"profiling block maxima for {scenario.name}")
        rows.extend(
            _profile_operation(
                scenario,
                operation="core.block_maxima",
                current=lambda values, window: block_maxima(values, window, sliding=True),
                baseline=lambda values, window: _baseline_sliding_window_extreme_valid(
                    values,
                    window,
                    reducer="max",
                ),
            )
        )
        status("window_profile", f"profiling bootstrap sliding maxima for {scenario.name}")
        rows.extend(
            _profile_operation(
                scenario,
                operation="bootstrap._sliding_block_maxima",
                current=_sliding_block_maxima,
                baseline=_baseline_circular_sliding_window_maximum,
            )
        )
        status("window_profile", f"profiling EI rolling minima for {scenario.name}")
        rows.extend(
            _profile_operation(
                scenario,
                operation="extremal_index._rolling_window_minima",
                current=lambda values, window: _rolling_window_minima(
                    values, window, sliding=True
                ),
                baseline=lambda values, window: _baseline_sliding_window_extreme_valid(
                    values,
                    window,
                    reducer="min",
                ),
            )
        )
    frame = pd.DataFrame(rows)
    frame["peak_megabytes"] = frame["peak_bytes"] / (1024.0 * 1024.0)
    return frame.sort_values(["operation", "scenario", "implementation"]).reset_index(drop=True)


def main() -> None:
    dirs = resolve_repo_dirs()
    out_path = dirs["DIR_OUT"] / "sliding_window_profile.csv"
    profile = build_sliding_window_profile()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(out_path, index=False)
    status("window_profile", f"wrote profiling summary to {out_path}")
    print(profile.to_string(index=False))


if __name__ == "__main__":
    main()
