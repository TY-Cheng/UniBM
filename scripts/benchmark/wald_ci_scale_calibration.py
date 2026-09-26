"""Refit six fixed-shrinkage Wald methods on the existing simulation banks.

This isolated experiment never updates package defaults, canonical benchmark
CSVs, applications or external reports. Run as a module with PYTHONPATH=src:scripts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd
import scipy

from benchmark.design import (
    BENCHMARK_MASTER_SEED,
    _series_cache_file,
    default_ei_simulation_configs,
    default_evi_simulation_configs,
    scenario_random_state,
)
from shared.runtime import initialize_numerical_worker, status
from unibm._bootstrap_precision import ADAPTIVE_REPS, MCSE_TOLERANCE
from unibm.ei import bootstrap_bm_ei_path, estimate_pooled_bm_ei, prepare_ei_bundle
from unibm.evi import estimate_evi_quantile
from unibm.evi._accelerator import kernels


ROOT = Path(__file__).resolve().parents[2]
SHRINKAGE = {"evi": 0.73, "ei": 0.37}
EI_PATHS = tuple((path, sliding) for path in ("bb", "northrop") for sliding in (False, True))


def run_scenario(task):
    """Use production estimators and seeds, retaining every failed fit as a row."""
    branch, cfg, reps = task
    started = time.perf_counter()
    seed = scenario_random_state(cfg, master_seed=BENCHMARK_MASTER_SEED)
    source = _series_cache_file(ROOT / "out/benchmark/cache", cfg, random_state=seed)
    with np.load(source) as archive:
        bank = archive["series"][:reps]
    if bank.shape != (reps, cfg.n_obs) or not np.isfinite(bank).all():
        raise ValueError(f"Incomplete simulation bank: {source}")
    paths = (("median", False), ("median", True)) if branch == "evi" else EI_PATHS
    rows = []
    for rep, values in enumerate(bank):
        bootstrap_seed = rep if branch == "evi" else seed + 10_000 * rep
        bundle = (
            prepare_ei_bundle(values, allow_zeros=False, path_keys=EI_PATHS)
            if branch == "ei"
            else None
        )
        for path, sliding in paths:
            scheme = "sliding" if sliding else "disjoint"
            row = dict(
                branch=branch,
                method=f"{path}_{scheme}_fgls",
                scheme=scheme,
                scenario=cfg.scenario,
                family=cfg.family,
                rep=rep,
                n_obs=cfg.n_obs,
                xi_true=cfg.xi_true,
                theta_true=cfg.theta_true,
                truth=cfg.xi_true if branch == "evi" else cfg.theta_true,
                scenario_seed=seed,
                bootstrap_seed=bootstrap_seed,
                shrinkage=SHRINKAGE[branch],
                estimate=np.nan,
                ci_lo=np.nan,
                ci_hi=np.nan,
                standard_error=np.nan,
                z_standard_error=np.nan,
                window_lo=np.nan,
                window_hi=np.nan,
                bootstrap_reps=np.nan,
                precision_met=False,
                mcse_max_ratio=np.nan,
                flat_window=False,
                boundary_active=False,
                valid=False,
                error="",
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                try:
                    if branch == "evi":
                        fit = estimate_evi_quantile(
                            values,
                            regression="FGLS",
                            sliding=sliding,
                            quantile=0.5,
                            covariance_shrinkage=SHRINKAGE[branch],
                            bootstrap_reps="adaptive",
                            random_state=bootstrap_seed,
                            n_threads=1,
                        )
                        row.update(
                            estimate=fit.slope,
                            standard_error=fit.standard_error,
                            window_lo=int(fit.curve.positive_block_sizes[fit.plateau.start]),
                            window_hi=int(fit.curve.positive_block_sizes[fit.plateau.stop - 1]),
                            flat_window=bool(np.ptp(fit.plateau.y) == 0),
                        )
                    else:
                        bootstrap = bootstrap_bm_ei_path(
                            values,
                            allow_zeros=False,
                            base_path=path,
                            sliding=sliding,
                            block_sizes=bundle.block_sizes,
                            reps="adaptive",
                            covariance_shrinkage=SHRINKAGE[branch],
                            random_state=bootstrap_seed,
                            n_threads=1,
                        )
                        fit = estimate_pooled_bm_ei(
                            bundle,
                            base_path=path,
                            sliding=sliding,
                            regression="FGLS",
                            covariance_shrinkage=SHRINKAGE[branch],
                            bootstrap_result=bootstrap,
                        )
                        row.update(
                            estimate=fit.theta_hat,
                            standard_error=fit.standard_error,
                            z_standard_error=fit.z_standard_error,
                            window_lo=fit.stable_window.lo,
                            window_hi=fit.stable_window.hi,
                            boundary_active=fit.ci_variant.endswith("_boundary"),
                        )
                    row.update(
                        ci_lo=fit.confidence_interval[0],
                        ci_hi=fit.confidence_interval[1],
                        bootstrap_reps=fit.bootstrap_reps_used,
                        precision_met=fit.bootstrap_precision_met,
                        mcse_max_ratio=fit.bootstrap_mcse_max_ratio,
                    )
                    row["valid"] = bool(
                        np.isfinite([row["estimate"], row["ci_lo"], row["ci_hi"]]).all()
                        and row["ci_lo"] <= row["ci_hi"]
                        and (branch != "ei" or 0 < row["estimate"] <= 1)
                    )
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    row["error"] = str(exc)
                row["warning"] = " | ".join(sorted({str(w.message) for w in caught}))
            rows.append(row)
    result = pd.DataFrame(rows)
    provenance = dict(
        branch=branch,
        scenario=cfg.scenario,
        source=str(source.relative_to(ROOT)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        series_sha256=hashlib.sha256(np.ascontiguousarray(bank).tobytes()).hexdigest(),
        elapsed_s=time.perf_counter() - started,
    )
    return result, provenance


def main():
    """Retain per-scenario evidence without reusing earlier fitted results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "out/research/wald_ci_scale_calibration"
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit-reps", type=int, default=100)
    parser.add_argument("--limit-scenarios", type=int, default=84)
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if not any(out.is_relative_to(ROOT / name) for name in ("out", ".cache")):
        parser.error("Outputs must stay under this repository's out/ or .cache/.")
    if out.exists() and any(out.iterdir()):
        parser.error("Choose an empty output directory; earlier experiments are preserved.")
    if (
        not 1 <= args.limit_scenarios <= 84
        or not 5 <= args.limit_reps <= 100
        or args.limit_reps % 5
        or args.workers < 1
    ):
        parser.error(
            "Require 1..84 scenarios, 5..100 reps divisible by five, and positive workers."
        )
    configs = {"evi": default_evi_simulation_configs(), "ei": default_ei_simulation_configs()}
    indices = np.linspace(0, 83, args.limit_scenarios, dtype=int)
    tasks = [(branch, configs[branch][i], args.limit_reps) for branch in configs for i in indices]
    # Existing production banks contain 100 records even for prefix cost pilots.
    for _, cfg, _ in tasks:
        path = _series_cache_file(
            ROOT / "out/benchmark/cache", cfg, random_state=scenario_random_state(cfg)
        )
        if not path.is_file():
            parser.error(f"Missing retained simulation bank: {path}")
    (out / "trials").mkdir(parents=True)
    sources = list((ROOT / "src/unibm").rglob("*.py")) + list((ROOT / "src/unibm").rglob("*.pyx"))
    sources += [
        Path(__file__),
        ROOT / "scripts/benchmark/design.py",
        ROOT / "scripts/shared/runtime.py",
    ]
    if kernels is not None:
        sources.append(Path(kernels.__file__).resolve())
    manifest = dict(
        status="running",
        started_utc=datetime.now(timezone.utc).isoformat(),
        master_seed=BENCHMARK_MASTER_SEED,
        n_obs=365,
        monte_carlo_reps=args.limit_reps,
        scenarios_per_branch=args.limit_scenarios,
        shrinkage=SHRINKAGE,
        adaptive_reps=ADAPTIVE_REPS,
        mcse_tolerance=MCSE_TOLERANCE,
        workers=args.workers,
        native_extensions=kernels is not None,
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        configs={b: [asdict(configs[b][i]) for i in indices] for b in configs},
        new_simulations=0,
        scenarios=[],
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.perf_counter()
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize_numerical_worker,
    ) as pool:
        futures = {pool.submit(run_scenario, task): task for task in tasks}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                detail, provenance = future.result()
            except Exception as exc:
                manifest.update(status="failed", error=str(exc))
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
                raise
            name = f"{provenance['branch']}__{provenance['scenario']}.csv.gz"
            detail.to_csv(out / "trials" / name, index=False)
            provenance["trials_file"] = name
            provenance["trials_sha256"] = hashlib.sha256(
                (out / "trials" / name).read_bytes()
            ).hexdigest()
            manifest["scenarios"].append(provenance)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            status(
                "wald_scale",
                f"{count}/{len(tasks)}: {provenance['branch']} {provenance['scenario']}, {detail.valid.sum()}/{len(detail)} valid",
            )
    manifest.update(
        status="fits_completed",
        fit_elapsed_s=time.perf_counter() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not args.fit_only:
        from benchmark.wald_ci_scale_report import analyze

        analyze(out)
    print(f"Fits: {out}; {manifest['fit_elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
