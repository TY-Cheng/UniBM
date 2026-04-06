"""Dataset-specific preparation helpers for manuscript applications."""

from .fema import (
    download_nfip_claims_state,
    load_cpi_2025_base,
    prepare_nfip_claim_series,
    read_nfip_claims_csv,
)
from .ghcn import (
    PreparedSeries,
    materialize_derived_series,
    prepare_hot_dry_series,
    prepare_precipitation_series,
    read_ghcn_station_csv,
)
from .usgs import (
    download_usgs_daily_discharge,
    prepare_usgs_streamflow_series,
    read_usgs_daily_discharge_csv,
)

__all__ = [
    "PreparedSeries",
    "download_nfip_claims_state",
    "download_usgs_daily_discharge",
    "load_cpi_2025_base",
    "materialize_derived_series",
    "prepare_nfip_claim_series",
    "prepare_hot_dry_series",
    "prepare_precipitation_series",
    "prepare_usgs_streamflow_series",
    "read_nfip_claims_csv",
    "read_ghcn_station_csv",
    "read_usgs_daily_discharge_csv",
]
