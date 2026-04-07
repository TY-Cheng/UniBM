"""Fitting and process-parallel orchestration for application bundles."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np
import pandas as pd

from unibm.bootstrap import draw_circular_block_bootstrap_samples
from unibm.core import estimate_evi_quantile
from unibm.extremal_index import (
    EiPreparedBundle,
    bootstrap_bm_ei_path_draws,
    estimate_k_gaps,
    estimate_pooled_bm_ei,
    extract_stable_path_window,
    prepare_ei_bundle,
)
from .application_specs import (
    APPLICATION_EI_BOOTSTRAP_REPS,
    APPLICATION_RANDOM_STATE,
    APPLICATIONS,
    ApplicationBundle,
    ApplicationPreparedInputs,
    ApplicationSpec,
)
from .workflow_runtime import resolve_int_env, status


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


def _materialize_ei_bootstrap_result(
    series: pd.Series,
    *,
    ei_bundle: EiPreparedBundle,
    allow_zeros: bool,
    reps: int = APPLICATION_EI_BOOTSTRAP_REPS,
    random_state: int = APPLICATION_RANDOM_STATE,
) -> dict[str, np.ndarray | None]:
    """Build the pooled BB-sliding FGLS covariance summary for one real series."""
    sample_bank = draw_circular_block_bootstrap_samples(
        series.to_numpy(dtype=float),
        reps=reps,
        random_state=random_state,
    )
    full_draws = bootstrap_bm_ei_path_draws(
        sample_bank.samples,
        block_sizes=ei_bundle.block_sizes,
        allow_zeros=allow_zeros,
    )
    selected_levels, _ = extract_stable_path_window(ei_bundle.paths[("bb", True)])
    full_levels = np.asarray(ei_bundle.block_sizes, dtype=int)
    selected_idx = [int(np.flatnonzero(full_levels == level)[0]) for level in selected_levels]
    selected_draws = full_draws[("bb", True)][:, selected_idx]
    valid_draws = selected_draws[np.all(np.isfinite(selected_draws), axis=1)]
    covariance = None
    if valid_draws.shape[0] >= 2:
        covariance = np.atleast_2d(np.cov(valid_draws, rowvar=False))
    return {
        "block_sizes": selected_levels,
        "samples": valid_draws,
        "covariance": covariance,
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
    allow_zeros = spec.ei_allow_zeros
    status("application", f"preparing EI bundle for {spec.label}")
    ei_bundle = prepare_ei_bundle(inputs.ei.series.values, allow_zeros=allow_zeros)
    status("application", f"bootstrapping EI covariance for {spec.label}")
    bootstrap_result = _materialize_ei_bootstrap_result(
        inputs.ei.series,
        ei_bundle=ei_bundle,
        allow_zeros=allow_zeros,
        reps=APPLICATION_EI_BOOTSTRAP_REPS,
        random_state=APPLICATION_RANDOM_STATE,
    )
    status("application", f"fitting BB-sliding-FGLS EI for {spec.label}")
    ei_primary = estimate_pooled_bm_ei(
        ei_bundle,
        base_path="bb",
        sliding=True,
        regression="FGLS",
        bootstrap_result=bootstrap_result,
    )
    status("application", f"fitting K-gaps EI comparator for {spec.label}")
    ei_comparator = estimate_k_gaps(ei_bundle)
    return ApplicationBundle(
        spec=spec,
        prepared=inputs,
        evi_fit=evi_fit,
        ei_bundle=ei_bundle,
        ei_primary=ei_primary,
        ei_comparator=ei_comparator,
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
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_build_application_bundle_worker, tasks, chunksize=1))


def build_application_bundles(
    dirs: dict[str, Path],
    *,
    raw_paths: dict[str, Path] | None = None,
) -> list[ApplicationBundle]:
    """Compatibility wrapper that prepares inputs before fitting bundles."""
    from .application_inputs import build_application_inputs

    inputs = build_application_inputs(dirs, raw_paths=raw_paths)
    return build_application_bundles_from_inputs(inputs)


__all__ = [
    "build_application_bundle",
    "build_application_bundles",
    "build_application_bundles_from_inputs",
]
