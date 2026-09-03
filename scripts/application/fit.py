"""Fitting and process-parallel orchestration for application bundles."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import pandas as pd

from unibm.ei import (
    EiPreparedBundle,
    ExtremalIndexEstimate,
    bootstrap_bm_ei_path,
    estimate_ferro_segers,
    estimate_k_gaps,
    estimate_pooled_bm_ei,
    prepare_ei_bundle,
)
from unibm.evi import estimate_evi_quantile
from application.specs import (
    APPLICATION_EI_BOOTSTRAP_REPS,
    APPLICATION_RANDOM_STATE,
    APPLICATIONS,
    ApplicationBundle,
    ApplicationPreparedInputs,
    ApplicationSpec,
)
from shared.runtime import resolve_int_env, status


def _application_worker_count(n_tasks: int) -> int:
    """Resolve the application worker count from the environment."""
    requested = os.environ.get("UNIBM_APPLICATION_WORKERS")
    if requested is None:
        workers = resolve_int_env(
            "UNIBM_BENCHMARK_WORKERS",
            default=max(min((os.cpu_count() or 1) - 1, n_tasks), 1),
            minimum=1,
        )
    else:
        workers = resolve_int_env("UNIBM_APPLICATION_WORKERS", default=1, minimum=1)
    return int(min(max(workers, 1), max(n_tasks, 1)))


def _build_application_bundle_worker(
    task: tuple[ApplicationSpec, ApplicationPreparedInputs],
) -> ApplicationBundle:
    """Worker wrapper so application bundles can be built in subprocesses."""
    spec, inputs = task
    return build_application_bundle(spec, inputs)


def fit_application_ei_estimates(
    series: pd.Series,
    *,
    allow_zeros: bool,
    label: str | None = None,
    status_prefix: str = "application",
) -> tuple[EiPreparedBundle, dict[str, ExtremalIndexEstimate]]:
    """Fit the four default application EI estimators for one prepared series."""
    if label is not None:
        status(status_prefix, f"preparing EI bundle for {label}")
    ei_bundle = prepare_ei_bundle(series.values, allow_zeros=allow_zeros)
    if label is not None:
        status(status_prefix, f"bootstrapping EI covariance for {label}")
    bootstrap_results = {
        base_path: bootstrap_bm_ei_path(
            ei_bundle.values,
            base_path=base_path,
            sliding=True,
            block_sizes=ei_bundle.block_sizes,
            reps=APPLICATION_EI_BOOTSTRAP_REPS,
            random_state=APPLICATION_RANDOM_STATE,
            allow_zeros=allow_zeros,
        )
        for base_path in ("bb", "northrop")
    }
    if label is not None:
        status(status_prefix, f"fitting BB-sliding-FGLS EI for {label}")
    bb_sliding_fgls = estimate_pooled_bm_ei(
        ei_bundle,
        base_path="bb",
        sliding=True,
        regression="FGLS",
        bootstrap_result=bootstrap_results["bb"],
    )
    if label is not None:
        status(status_prefix, f"fitting Northrop-sliding-FGLS EI for {label}")
    northrop_sliding_fgls = estimate_pooled_bm_ei(
        ei_bundle,
        base_path="northrop",
        sliding=True,
        regression="FGLS",
        bootstrap_result=bootstrap_results["northrop"],
    )
    if label is not None:
        status(status_prefix, f"fitting K-gaps EI comparator for {label}")
    k_gaps = estimate_k_gaps(ei_bundle)
    if label is not None:
        status(status_prefix, f"fitting Ferro-Segers EI comparator for {label}")
    ferro_segers = estimate_ferro_segers(ei_bundle)
    return ei_bundle, {
        "bb_sliding_fgls": bb_sliding_fgls,
        "northrop_sliding_fgls": northrop_sliding_fgls,
        "k_gaps": k_gaps,
        "ferro_segers": ferro_segers,
    }


def build_application_bundle(
    spec: ApplicationSpec,
    inputs: ApplicationPreparedInputs,
) -> ApplicationBundle:
    """Fit the primary EVI and EI application estimators for one case."""
    status("application", f"fitting EVI for {spec.label}")
    evi_fit = estimate_evi_quantile(
        inputs.evi.series.values,
        quantile=spec.quantile,
        sliding=True,
        bootstrap_reps=120,
        random_state=APPLICATION_RANDOM_STATE,
    )
    ei_bundle: EiPreparedBundle | None = None
    ei_estimates: dict[str, ExtremalIndexEstimate] | None = None
    if spec.formal_ei:
        allow_zeros = spec.ei_allow_zeros
        ei_bundle, ei_estimates = fit_application_ei_estimates(
            inputs.ei.series,
            allow_zeros=allow_zeros,
            label=spec.label,
            status_prefix="application",
        )
    return ApplicationBundle(
        spec=spec,
        prepared=inputs,
        evi_fit=evi_fit,
        ei_bundle=ei_bundle,
        ei_bb_sliding_fgls=(None if ei_estimates is None else ei_estimates["bb_sliding_fgls"]),
        ei_northrop_sliding_fgls=(
            None if ei_estimates is None else ei_estimates["northrop_sliding_fgls"]
        ),
        ei_k_gaps=None if ei_estimates is None else ei_estimates["k_gaps"],
        ei_ferro_segers=None if ei_estimates is None else ei_estimates["ferro_segers"],
    )


def build_application_bundles_from_inputs(
    inputs: dict[str, ApplicationPreparedInputs],
    *,
    specs: tuple[ApplicationSpec, ...] = APPLICATIONS,
) -> list[ApplicationBundle]:
    """Build every configured application bundle from prepared inputs."""
    tasks = [(spec, inputs[spec.key]) for spec in specs]
    workers = _application_worker_count(len(tasks))
    if workers <= 1:
        status("application", f"building {len(tasks)} application bundles sequentially")
        return [build_application_bundle(spec, inputs[spec.key]) for spec in specs]
    status(
        "application", f"building {len(tasks)} application bundles with {workers} worker processes"
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_build_application_bundle_worker, tasks, chunksize=1))


def build_application_bundles(
    dirs: dict[str, Path],
    *,
    raw_paths: dict[str, Path] | None = None,
) -> list[ApplicationBundle]:
    """Compatibility wrapper that prepares inputs before fitting bundles."""
    from application.inputs import build_application_inputs

    inputs = build_application_inputs(dirs, raw_paths=raw_paths)
    return build_application_bundles_from_inputs(inputs)


__all__ = [
    "build_application_bundle",
    "build_application_bundles",
    "build_application_bundles_from_inputs",
    "fit_application_ei_estimates",
]
