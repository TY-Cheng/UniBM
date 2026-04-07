"""Application workflow dataclasses and static registry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_prep.ghcn import PreparedSeries
from unibm.extremal_index import EiPreparedBundle, ExtremalIndexEstimate
from unibm.models import ScalingFit


APPLICATION_RANDOM_STATE = 7
APPLICATION_EI_BOOTSTRAP_REPS = 120
RETURN_LEVEL_HORIZONS = np.asarray([1.0, 10.0, 25.0, 50.0], dtype=float)
APPLICATION_EVI_METHOD_IDS = ("sliding_median_fgls",)
APPLICATION_EI_METHOD_IDS = (
    "bb_sliding_fgls",
    "northrop_sliding_fgls",
    "k_gaps",
    "ferro_segers",
)


@dataclass(frozen=True)
class ApplicationPreparedInputs:
    """Role-specific prepared series used by one manuscript application."""

    display: PreparedSeries
    evi: PreparedSeries
    ei: PreparedSeries


@dataclass(frozen=True)
class ApplicationSpec:
    """Static configuration for one application-facing case study."""

    key: str
    provider: str
    label: str
    figure_stem: str
    raw_key: str
    ylabel: str
    time_series_title: str
    scaling_title: str
    scaling_ylabel: str
    quantile: float = 0.5
    observations_per_year: float | None = None
    return_level_basis: str = "calendar_year"
    return_level_label: str = "return period (years)"
    return_level_yscale: str = "linear"
    target_stability_title: str | None = None
    secondary_case: bool = False
    formal_ei: bool = True
    ei_allow_zeros: bool = False


@dataclass(frozen=True)
class ApplicationBundle:
    """Prepared series and fitted results for one application."""

    spec: ApplicationSpec
    prepared: ApplicationPreparedInputs
    evi_fit: ScalingFit
    ei_bundle: EiPreparedBundle | None
    ei_bb_sliding_fgls: ExtremalIndexEstimate | None
    ei_northrop_sliding_fgls: ExtremalIndexEstimate | None
    ei_k_gaps: ExtremalIndexEstimate | None
    ei_ferro_segers: ExtremalIndexEstimate | None

    @property
    def ei_primary(self) -> ExtremalIndexEstimate:
        """Return the headline BB-sliding-FGLS estimate."""
        if self.ei_bb_sliding_fgls is None:
            raise ValueError(f"{self.spec.label} does not participate in formal EI analysis.")
        return self.ei_bb_sliding_fgls

    @property
    def ei_comparator(self) -> ExtremalIndexEstimate:
        """Return the legacy K-gaps comparator for compatibility."""
        if self.ei_k_gaps is None:
            raise ValueError(f"{self.spec.label} does not participate in formal EI analysis.")
        return self.ei_k_gaps

    @property
    def ei_method_map(self) -> dict[str, ExtremalIndexEstimate]:
        """Return the application-side EI comparison set keyed by method id."""
        if not self.spec.formal_ei:
            return {}
        assert self.ei_bb_sliding_fgls is not None
        assert self.ei_northrop_sliding_fgls is not None
        assert self.ei_k_gaps is not None
        assert self.ei_ferro_segers is not None
        return {
            "bb_sliding_fgls": self.ei_bb_sliding_fgls,
            "northrop_sliding_fgls": self.ei_northrop_sliding_fgls,
            "k_gaps": self.ei_k_gaps,
            "ferro_segers": self.ei_ferro_segers,
        }


APPLICATIONS = (
    ApplicationSpec(
        key="houston_hobby_precipitation",
        provider="ghcn",
        label="Houston precipitation",
        figure_stem="houston_precipitation",
        raw_key="USW00012918.csv.gz",
        ylabel="precipitation (mm)",
        time_series_title="Houston wet-season daily precipitation and annual maxima",
        scaling_title="Houston sliding-block quantile scaling",
        scaling_ylabel="log median block maximum",
        observations_per_year=183.0,
        target_stability_title="Houston target stability across block sizes",
        formal_ei=False,
    ),
    ApplicationSpec(
        key="phoenix_hot_dry_severity",
        provider="ghcn",
        label="Phoenix hot-dry severity",
        figure_stem="phoenix_hotdry",
        raw_key="USW00023183.csv.gz",
        ylabel="hot-dry severity",
        time_series_title="Phoenix warm-season hot-dry severity and annual maxima",
        scaling_title="Phoenix sliding-block quantile scaling",
        scaling_ylabel="log median block maximum",
        observations_per_year=214.0,
        secondary_case=True,
        target_stability_title="Phoenix target stability across block sizes",
        formal_ei=False,
    ),
    ApplicationSpec(
        key="tx_streamflow",
        provider="usgs",
        label="Texas streamflow",
        figure_stem="tx_streamflow",
        raw_key="TX",
        ylabel="discharge (cfs)",
        time_series_title="Texas daily discharge and annual maxima",
        scaling_title="Texas streamflow sliding-block quantile scaling",
        scaling_ylabel="log median block maximum",
        observations_per_year=365.25,
        return_level_yscale="log",
        target_stability_title="Texas streamflow target stability across block sizes",
    ),
    ApplicationSpec(
        key="fl_streamflow",
        provider="usgs",
        label="Florida streamflow",
        figure_stem="fl_streamflow",
        raw_key="FL",
        ylabel="discharge (cfs)",
        time_series_title="Florida daily discharge and annual maxima",
        scaling_title="Florida streamflow sliding-block quantile scaling",
        scaling_ylabel="log median block maximum",
        observations_per_year=365.25,
        return_level_yscale="log",
        target_stability_title="Florida streamflow target stability across block sizes",
    ),
    ApplicationSpec(
        key="tx_nfip_claims",
        provider="fema",
        label="Texas NFIP claims",
        figure_stem="tx_nfip_claims",
        raw_key="TX",
        ylabel="building payouts (2025 USD)",
        time_series_title="Texas NFIP daily building payouts and annual maxima",
        scaling_title="Texas NFIP active-day sliding-block quantile scaling",
        scaling_ylabel="log median block maximum (positive payout days)",
        return_level_basis="claim_active_day",
        return_level_label="claim-active-day return period (years)",
        return_level_yscale="log",
        target_stability_title="Texas NFIP target stability across block sizes",
        ei_allow_zeros=True,
    ),
    ApplicationSpec(
        key="fl_nfip_claims",
        provider="fema",
        label="Florida NFIP claims",
        figure_stem="fl_nfip_claims",
        raw_key="FL",
        ylabel="building payouts (2025 USD)",
        time_series_title="Florida NFIP daily building payouts and annual maxima",
        scaling_title="Florida NFIP active-day sliding-block quantile scaling",
        scaling_ylabel="log median block maximum (positive payout days)",
        return_level_basis="claim_active_day",
        return_level_label="claim-active-day return period (years)",
        return_level_yscale="log",
        target_stability_title="Florida NFIP target stability across block sizes",
        ei_allow_zeros=True,
    ),
)


def spec_by_key() -> dict[str, ApplicationSpec]:
    """Return the application registry keyed by application id."""
    return {spec.key: spec for spec in APPLICATIONS}


__all__ = [
    "APPLICATION_EI_BOOTSTRAP_REPS",
    "APPLICATION_EI_METHOD_IDS",
    "APPLICATION_EVI_METHOD_IDS",
    "APPLICATION_RANDOM_STATE",
    "APPLICATIONS",
    "ApplicationBundle",
    "ApplicationPreparedInputs",
    "ApplicationSpec",
    "RETURN_LEVEL_HORIZONS",
    "spec_by_key",
]
