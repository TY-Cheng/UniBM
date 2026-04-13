"""Public UniBM package facade.

The reusable statistical package exposes grouped public namespaces under
``unibm.evi`` and ``unibm.ei`` together with the standalone helper module
``unibm.cdf``. Repo-local benchmark, application,
and notebook orchestration code lives outside the package under
``scripts/``.
"""

from .__about__ import __version__
from .evi import estimate_design_life_level, estimate_evi_quantile

__all__ = ["__version__", "ei", "evi", "estimate_design_life_level", "estimate_evi_quantile"]


def __getattr__(name: str):
    """Lazily expose the canonical grouped subpackages."""
    if name in {"ei", "evi"}:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module 'unibm' has no attribute {name!r}")
