"""Public UniBM package facade.

The package exposes the core EVI, pooled EI, and design-life workflows here.
Grouped public namespaces ``unibm.evi`` and ``unibm.ei`` contain the full
estimator and helper APIs; ``unibm.cdf`` provides empirical ranks.
Repository benchmark, application, and orchestration code lives under
``scripts/``, outside the installable package.
"""

from typing import TYPE_CHECKING

from .__about__ import __version__

if TYPE_CHECKING:
    from . import ei, evi
    from .ei.bm import estimate_pooled_bm_ei
    from .ei.bootstrap import bootstrap_bm_ei_path
    from .ei.preparation import prepare_ei_bundle
    from .evi.design import estimate_design_life_level, estimate_design_life_level_interval
    from .evi.estimation import estimate_evi_quantile

__all__ = [
    "__version__",
    "ei",
    "evi",
    "estimate_evi_quantile",
    "prepare_ei_bundle",
    "bootstrap_bm_ei_path",
    "estimate_pooled_bm_ei",
    "estimate_design_life_level",
    "estimate_design_life_level_interval",
]


def __getattr__(name: str):
    """Import a public namespace or estimator on first access and cache it.

    Unknown names raise ``AttributeError``, as for an ordinary module attribute.
    Deferring these imports keeps ``import unibm`` lightweight.
    """
    import importlib

    if name in {"ei", "evi"}:
        value = importlib.import_module(f"{__name__}.{name}")
    elif name == "estimate_evi_quantile":
        value = importlib.import_module(f"{__name__}.evi.estimation").estimate_evi_quantile
    elif name in {"estimate_design_life_level", "estimate_design_life_level_interval"}:
        value = getattr(importlib.import_module(f"{__name__}.evi.design"), name)
    elif name == "prepare_ei_bundle":
        value = importlib.import_module(f"{__name__}.ei.preparation").prepare_ei_bundle
    elif name == "bootstrap_bm_ei_path":
        value = importlib.import_module(f"{__name__}.ei.bootstrap").bootstrap_bm_ei_path
    elif name == "estimate_pooled_bm_ei":
        value = importlib.import_module(f"{__name__}.ei.bm").estimate_pooled_bm_ei
    else:
        raise AttributeError(f"module 'unibm' has no attribute {name!r}")
    globals()[name] = value
    return value
