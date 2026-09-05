"""Public UniBM package facade.

The package exposes grouped public namespaces under ``unibm.evi`` and
``unibm.ei`` together with the standalone helper module ``unibm.cdf``.
Repo-local benchmark, application,
and repository orchestration code lives outside the package under
``scripts/``.
"""

from typing import TYPE_CHECKING

from .__about__ import __version__

if TYPE_CHECKING:
    from . import ei, evi
    from .evi.design import estimate_design_life_level
    from .evi.estimation import estimate_evi_quantile

__all__ = ["__version__", "ei", "evi", "estimate_design_life_level", "estimate_evi_quantile"]


def __getattr__(name: str):
    """Lazily expose grouped subpackages and headline estimators."""
    import importlib

    if name in {"ei", "evi"}:
        value = importlib.import_module(f"{__name__}.{name}")
    elif name == "estimate_evi_quantile":
        value = importlib.import_module(f"{__name__}.evi.estimation").estimate_evi_quantile
    elif name == "estimate_design_life_level":
        value = importlib.import_module(f"{__name__}.evi.design").estimate_design_life_level
    else:
        raise AttributeError(f"module 'unibm' has no attribute {name!r}")
    globals()[name] = value
    return value
