"""Dataset-specific preparation helpers for manuscript applications."""

from .ghcn import (
    PreparedSeries,
    materialize_derived_series,
    prepare_hot_dry_series,
    prepare_precipitation_series,
    read_ghcn_station_csv,
)

__all__ = [
    "PreparedSeries",
    "materialize_derived_series",
    "prepare_hot_dry_series",
    "prepare_precipitation_series",
    "read_ghcn_station_csv",
]
