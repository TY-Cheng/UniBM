"""Application screening helpers for manuscript-ready univariate series."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from unibm.ei import estimate_k_gaps, estimate_pooled_bm_ei, prepare_ei_bundle
from unibm.evi import block_maxima, estimate_evi_quantile
from shared.runtime import resolve_int_env


DEFAULT_SCREENING_BOOTSTRAP_REPS = 40


@dataclass(frozen=True)
class ScreeningReview:
    """Screening summary for one candidate real-data application series."""

    name: str
    n_obs: int
    n_years: float
    start: str
    end: str
    daily_positive_share: float
    maxima_positive_share: float
    seasonality_strength: float
    xi_hat: float
    xi_lower: float
    xi_upper: float
    plateau_bounds: tuple[int, int]
    plateau_points: int
    supports_frechet_working_model: bool
    recommended: bool

    def to_record(self) -> dict[str, object]:
        """Return a flat record suitable for CSV/JSON export."""
        return asdict(self)


@dataclass(frozen=True)
class EiScreeningReview:
    """Screening summary for one candidate formal-EI application series."""

    name: str
    analysis_type: str
    n_obs: int
    n_years: float
    start: str
    end: str
    daily_positive_share: float
    daily_zero_share: float
    theta_hat_bb: float
    theta_lo_bb: float
    theta_hi_bb: float
    theta_hat_k_gaps: float
    theta_lo_k_gaps: float
    theta_hi_k_gaps: float
    stable_level_lo: int
    stable_level_hi: int
    recommended: bool

    def to_record(self) -> dict[str, object]:
        """Return a flat record suitable for CSV/JSON export."""
        return asdict(self)


def _seasonality_strength(series: pd.Series) -> float:
    """Summarize how concentrated the series mean is across calendar months."""
    if not isinstance(series.index, pd.DatetimeIndex):
        return np.nan
    monthly = series.groupby(series.index.month).mean()
    overall = float(series.mean())
    if not np.isfinite(overall) or abs(overall) < 1e-8:
        return np.nan
    return float(monthly.std(ddof=0) / abs(overall))


def screen_extreme_series(
    series: pd.Series,
    *,
    name: str,
    min_years: int = 20,
    quantile: float = 0.5,
    min_plateau_points: int = 5,
    min_xi_lower: float = -0.25,
    min_maxima_positive_share: float = 0.95,
    bootstrap_reps: int | None = None,
) -> ScreeningReview:
    """Screen a candidate series for inclusion as a block-maxima application."""
    series = series.dropna()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("A DatetimeIndex is required for dataset screening.")
    n_years = (series.index.max() - series.index.min()).days / 365.25
    if bootstrap_reps is None:
        bootstrap_reps = resolve_int_env(
            "UNIBM_SCREENING_BOOTSTRAP_REPS",
            default=DEFAULT_SCREENING_BOOTSTRAP_REPS,
            minimum=0,
        )
    fit = estimate_evi_quantile(
        series.values,
        quantile=quantile,
        sliding=True,
        bootstrap_reps=int(bootstrap_reps),
    )
    daily_positive_share = float(np.mean(np.asarray(series.values) > 0))
    plateau_points = int(np.sum(fit.plateau_mask))
    smallest_plateau = fit.plateau_bounds[0]
    # We explicitly check whether block maxima stay overwhelmingly positive on
    # the selected plateau, because a heavily zero-inflated series can look
    # usable day-to-day but still collapse after aggregation.
    plateau_maxima = block_maxima(series.values, block_size=smallest_plateau, sliding=True)
    maxima_positive_share = (
        float(np.mean(np.asarray(plateau_maxima) > 0)) if plateau_maxima.size else float("nan")
    )
    supports_frechet_working_model = bool(fit.confidence_interval[0] > 0)
    recommended = bool(
        (n_years >= min_years)
        and (plateau_points >= min_plateau_points)
        and (fit.confidence_interval[0] >= min_xi_lower)
        and (not np.isnan(maxima_positive_share))
        and (maxima_positive_share >= min_maxima_positive_share)
    )
    return ScreeningReview(
        name=name,
        n_obs=int(series.size),
        n_years=float(n_years),
        start=str(series.index.min().date()),
        end=str(series.index.max().date()),
        daily_positive_share=daily_positive_share,
        maxima_positive_share=maxima_positive_share,
        seasonality_strength=_seasonality_strength(series),
        xi_hat=float(fit.slope),
        xi_lower=float(fit.confidence_interval[0]),
        xi_upper=float(fit.confidence_interval[1]),
        plateau_bounds=fit.plateau_bounds,
        plateau_points=plateau_points,
        supports_frechet_working_model=supports_frechet_working_model,
        recommended=recommended,
    )


def screening_dataframe(reviews: Iterable[ScreeningReview]) -> pd.DataFrame:
    """Convert screening outputs into a stable table."""
    frame = pd.DataFrame(review.to_record() for review in reviews)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["recommended", "supports_frechet_working_model", "n_years", "plateau_points"],
        ascending=[False, False, False, False],
    )


def screen_extremal_index_series(
    series: pd.Series,
    *,
    name: str,
    min_years: int = 20,
    allow_zeros: bool = False,
) -> EiScreeningReview:
    """Screen a candidate series for application-side EI estimation."""
    series = series.dropna()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("A DatetimeIndex is required for EI dataset screening.")
    n_years = (series.index.max() - series.index.min()).days / 365.25
    bundle = prepare_ei_bundle(series.values, allow_zeros=allow_zeros)
    bb_fit = estimate_pooled_bm_ei(bundle, base_path="bb", sliding=True, regression="OLS")
    kg_fit = estimate_k_gaps(bundle)
    values = np.asarray(series.values, dtype=float)
    daily_positive_share = float(np.mean(values > 0))
    daily_zero_share = float(np.mean(values == 0))
    stable_window = bb_fit.stable_window
    stable_level_lo = -1 if stable_window is None else int(stable_window.lo)
    stable_level_hi = -1 if stable_window is None else int(stable_window.hi)
    recommended = bool(
        (n_years >= min_years)
        and np.isfinite(bb_fit.theta_hat)
        and np.isfinite(kg_fit.theta_hat)
        and (stable_level_lo > 0)
        and (stable_level_hi >= stable_level_lo)
    )
    return EiScreeningReview(
        name=name,
        analysis_type="ei",
        n_obs=int(series.size),
        n_years=float(n_years),
        start=str(series.index.min().date()),
        end=str(series.index.max().date()),
        daily_positive_share=daily_positive_share,
        daily_zero_share=daily_zero_share,
        theta_hat_bb=float(bb_fit.theta_hat),
        theta_lo_bb=float(bb_fit.confidence_interval[0]),
        theta_hi_bb=float(bb_fit.confidence_interval[1]),
        theta_hat_k_gaps=float(kg_fit.theta_hat),
        theta_lo_k_gaps=float(kg_fit.confidence_interval[0]),
        theta_hi_k_gaps=float(kg_fit.confidence_interval[1]),
        stable_level_lo=stable_level_lo,
        stable_level_hi=stable_level_hi,
        recommended=recommended,
    )
