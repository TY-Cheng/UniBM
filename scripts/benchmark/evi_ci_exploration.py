"""Paired median-EVI covariance/CI experiment; never changes estimator defaults.

Run from the repository with ``PYTHONPATH=scripts:src python -m
benchmark.evi_ci_exploration``. All 49 candidates within a block scheme share
the observed window and final adaptive bootstrap draws. Formulas and limits
are recorded in the generated report, alongside every unsuccessful fit.
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
import shlex
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy

from benchmark.design import (
    BENCHMARK_MASTER_SEED,
    default_evi_simulation_configs,
    load_or_simulate_series_bank,
    scenario_random_state,
)
from shared.runtime import status
from unibm._bootstrap_precision import ADAPTIVE_REPS, MCSE_TOLERANCE, adaptive_covariance
from unibm._validation import regularize_covariance
from unibm.evi import generate_block_sizes
from unibm.evi._accelerator import kernels
from unibm.evi._regression import Z_CRIT_95
from unibm.evi.blocks import block_summary_curve
from unibm.evi.bootstrap import build_block_summary_bootstrap_backbone, _summary_evaluator
from unibm.evi.selection import select_penultimate_window


DELTAS = (0.0, 0.15, 0.37, 0.55, 0.75, 1.0)
COVARIANCES = tuple(f"diagonal_{x:g}" for x in DELTAS) + (
    "schafer_strimmer",
    "ledoit_wolf",
    "oas",
    "ols",
)
INTERVALS = ("model_wald", "propagated_wald", "bias_normal", "basic", "aligned_percentile")
CANDIDATES = tuple(
    (cov, ci)
    for cov in COVARIANCES
    for ci in INTERVALS
    if not (cov == "ols" and ci == "model_wald")
)
BASELINE = "diagonal_0.37/model_wald"
METHOD_INDEX = np.array([COVARIANCES.index(cov) for cov, _ in CANDIDATES])
CI_INDEX = np.array([INTERVALS.index(ci) for _, ci in CANDIDATES])


def automatic_shrinkages(rows: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Return SS correlation shrinkage, LW and original OAS intensities.

    SS follows Schaefer--Strimmer (2005), Eq. 10 and Appendix A. LW follows
    Chen et al. (2010), Eq. 13; OAS uses their Eq. 23 including the 2/p terms.
    LW intensity uses the centered ML covariance; all candidates then shrink
    the same unbiased covariance to avoid changing its overall normalization.
    OAS is Gaussian-derived; its assumptions need not hold for these draws.
    """
    n, p = rows.shape
    centered = rows - rows.mean(axis=0)
    sd = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z = np.divide(centered, sd, out=np.zeros_like(centered), where=sd > 0)
    products = z.T @ z
    variance = n / (n - 1) ** 3 * np.maximum((z * z).T @ (z * z) - products * products / n, 0.0)
    correlation = products / (n - 1)
    off_diagonal = ~np.eye(p, dtype=bool)
    ss_denominator = np.sum(correlation[off_diagonal] ** 2)
    ss = np.sum(variance[off_diagonal]) / ss_denominator if ss_denominator > 0 else 1.0
    ml = covariance * ((n - 1) / n)
    trace = np.trace(ml)
    square_trace = np.sum(ml * ml)
    distance = max(float(square_trace - trace * trace / p), 0.0)
    fourth = np.mean(np.sum(centered * centered, axis=1) ** 2)
    lw = (fourth - square_trace) / (n * distance) if distance > 0 else 1.0
    oas = (
        ((1 - 2 / p) * square_trace + trace * trace) / ((n + 1 - 2 / p) * distance)
        if distance > 0
        else 1.0
    )
    return np.clip([ss, lw, oas], 0.0, 1.0)


def candidate_intervals(rows, covariance, x, observed_path, center_path):
    """Evaluate all candidates with one matrix projection of the paired draws.

    For slope row a of A=(X'WX)^-1 X'W, propagated variance is a S a'.
    Bootstrap errors are a(y* - y_center), where y_center uses the unresampled
    segment-maxima bank. Basic reflects their quantiles around observed xi;
    aligned percentile adds them. Bias-normal subtracts their mean.
    Weights are fixed across bootstrap rows, but re-estimated when this whole
    function is called on a delete-group subset for Monte Carlo diagnostics.
    """
    X = np.column_stack((np.ones_like(x), x))
    maps = np.full((len(x), len(COVARIANCES)), np.nan)
    model_se = np.full(len(COVARIANCES), np.nan)
    errors = [""] * len(COVARIANCES)
    shrinkages = np.r_[DELTAS, automatic_shrinkages(rows, covariance), np.nan]
    try:
        ridge = (
            regularize_covariance(covariance, covariance_shrinkage=0.0, context="CI study")
            - covariance
        )
        targets = np.repeat(np.diag(np.diag(covariance))[None], 9, axis=0)
        targets[7:] = np.eye(len(x)) * np.trace(covariance) / len(x)
        coefficients = shrinkages[:9, None, None]
        regularized = (1 - coefficients) * covariance + coefficients * targets + ridge
        weights = np.linalg.pinv(regularized)
        normal_inverse = np.linalg.pinv(X.T @ weights @ X)
        maps[:, :9] = (normal_inverse @ X.T @ weights)[:, 1, :].T
        model_se[:9] = np.sqrt(np.maximum(normal_inverse[:, 1, 1], 0.0))
    except (ValueError, np.linalg.LinAlgError) as exc:
        errors[:9] = [str(exc)] * 9
    maps[:, 9] = np.linalg.pinv(X)[1]
    observed = observed_path @ maps
    center = center_path @ maps
    draws = rows @ maps
    boot_errors = draws - center
    propagated_se = np.sqrt(np.maximum(np.einsum("ij,ik,kj->j", maps, covariance, maps), 0))
    q_lo, q_hi = np.quantile(boot_errors, [0.025, 0.975], axis=0, method="linear")
    bias = boot_errors.mean(axis=0)
    endpoints = np.stack(
        [
            np.stack((observed - Z_CRIT_95 * model_se, observed + Z_CRIT_95 * model_se), axis=1),
            np.stack(
                (observed - Z_CRIT_95 * propagated_se, observed + Z_CRIT_95 * propagated_se),
                axis=1,
            ),
            np.stack(
                (
                    observed - bias - Z_CRIT_95 * propagated_se,
                    observed - bias + Z_CRIT_95 * propagated_se,
                ),
                axis=1,
            ),
            np.stack((observed - q_hi, observed - q_lo), axis=1),
            np.stack((observed + q_lo, observed + q_hi), axis=1),
        ],
        axis=1,
    )[METHOD_INDEX, CI_INDEX]
    scales = propagated_se[METHOD_INDEX].copy()
    scales[CI_INDEX == 0] = model_se[METHOD_INDEX[CI_INDEX == 0]]
    return {
        "xi": observed[METHOD_INDEX],
        "endpoints": endpoints,
        "scale": scales,
        "bootstrap_se": propagated_se[METHOD_INDEX],
        "model_se": model_se[METHOD_INDEX],
        "shrinkage": shrinkages[METHOD_INDEX],
        "center_xi": center[METHOD_INDEX],
        "bootstrap_bias": bias[METHOD_INDEX],
        "draws": draws,
        "maps": maps,
        "errors": [errors[index] for index in METHOD_INDEX],
    }


def evaluate_series(values, *, sliding, seed):
    """Use the unchanged selector and sampler, monitoring all 49 candidates."""
    started = time.perf_counter()
    grid = generate_block_sizes(len(values))
    curve = block_summary_curve(values, grid, target="quantile", sliding=sliding)
    window = select_penultimate_window(curve.log_block_sizes, curve.log_values)
    selected = slice(window.start, window.stop)
    backbone = build_block_summary_bootstrap_backbone(
        values,
        curve.positive_block_sizes,
        sliding=sliding,
        reps=2,
        random_state=seed,
    )
    if backbone is None:
        raise ValueError("Automatic superblock length permits fewer than two segments.")
    center_path = np.log(
        [
            np.quantile(backbone.maxima_by_block[int(b)], 0.5, method="median_unbiased")
            for b in curve.positive_block_sizes[selected]
        ]
    )

    def evaluate(covariance, samples):
        """Monitor observed xi and the actual two endpoints for each candidate."""
        result = candidate_intervals(
            samples[:, selected], covariance[selected, selected], window.x, window.y, center_path
        )
        targets = np.column_stack((result["xi"], result["endpoints"]))
        return targets.ravel(), np.repeat(result["scale"], 3)

    segments = backbone.segment_draws.shape[1]
    with _summary_evaluator(backbone, target="quantile", quantile=0.5, n_threads=1) as summaries:

        def draw(count, rng):
            """Append paired segment draws; the diagnostic RNG remains separate."""
            indices = rng.integers(0, segments, size=(count, segments))
            return np.log(summaries(indices))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            bootstrap = adaptive_covariance(draw, evaluate, random_state=seed)
    covariance = bootstrap["covariance"][selected, selected]
    result = candidate_intervals(
        bootstrap["samples"][:, selected], covariance, window.x, window.y, center_path
    )
    mcse = np.asarray(bootstrap["bootstrap_mcse"]).reshape(-1, 3)
    scales = result["scale"][:, None]
    ratios = np.max(
        np.divide(
            mcse, scales, out=np.full_like(mcse, np.inf), where=np.isfinite(scales) & (scales > 0)
        ),
        axis=1,
    )
    unique = np.array([len(np.unique(col)) for col in result["draws"].T])
    detail = pd.DataFrame(
        {
            "covariance": [c for c, _ in CANDIDATES],
            "ci": [c for _, c in CANDIDATES],
            "method": [f"{c}/{i}" for c, i in CANDIDATES],
            "xi_hat": result["xi"],
            "ci_lo": result["endpoints"][:, 0],
            "ci_hi": result["endpoints"][:, 1],
            "bootstrap_se": result["bootstrap_se"],
            "model_se": result["model_se"],
            "shrinkage": result["shrinkage"],
            "center_xi": result["center_xi"],
            "bootstrap_bias": result["bootstrap_bias"],
            "precision_ratio": ratios,
            "precision_met": np.isfinite(ratios) & (ratios <= MCSE_TOLERANCE),
            "xi_mcse": mcse[:, 0],
            "ci_lo_mcse": mcse[:, 1],
            "ci_hi_mcse": mcse[:, 2],
            "error": result["errors"],
            "unique_bootstrap_slopes": unique[METHOD_INDEX],
            "bootstrap_reps": bootstrap["bootstrap_reps_used"],
            "all_precision_met": bootstrap["bootstrap_precision_met"],
            "flat_window": np.ptp(window.y) == 0,
            "near_zero_slope": np.abs(result["xi"]) < 1e-10,
            "window_lo": int(curve.positive_block_sizes[window.start]),
            "window_hi": int(curve.positive_block_sizes[window.stop - 1]),
            "window_points": len(window.x),
            "superblock_length": backbone.super_block_size,
            "segments": segments,
            "covariance_rank": np.linalg.matrix_rank(covariance),
            "warning": " | ".join(sorted({str(item.message) for item in caught})),
            "elapsed_s": time.perf_counter() - started,
        }
    )
    return detail


def run_scenario(task):
    """Return one scenario's paired results, retaining statistical failures."""
    cfg, root, limit_reps = task
    scenario_seed = scenario_random_state(cfg, master_seed=BENCHMARK_MASTER_SEED)
    bank = load_or_simulate_series_bank(
        cfg, random_state=scenario_seed, cache_dir=root / "out/benchmark/cache"
    )
    pieces = []
    for rep, values in enumerate(bank[:limit_reps]):
        for sliding in (False, True):
            try:
                detail = evaluate_series(values, sliding=sliding, seed=rep)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                detail = pd.DataFrame(
                    {
                        "covariance": [c for c, _ in CANDIDATES],
                        "ci": [c for _, c in CANDIDATES],
                        "method": [f"{c}/{i}" for c, i in CANDIDATES],
                        "error": str(exc),
                        "xi_hat": np.nan,
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                        "precision_met": False,
                    }
                )
            detail = detail.assign(
                scenario=cfg.scenario,
                family=cfg.family,
                xi_true=cfg.xi_true,
                theta_true=cfg.theta_true,
                n_obs=cfg.n_obs,
                rep=rep,
                scheme="sliding" if sliding else "disjoint",
                scenario_seed=scenario_seed,
                bootstrap_seed=rep,
            )
            valid = np.isfinite(detail[["xi_hat", "ci_lo", "ci_hi"]]).all(axis=1) & (
                detail.ci_lo <= detail.ci_hi
            )
            detail["valid"] = valid
            detail["covered"] = np.where(
                valid, (detail.ci_lo <= cfg.xi_true) & (cfg.xi_true <= detail.ci_hi), np.nan
            )
            detail["width"] = np.where(valid, detail.ci_hi - detail.ci_lo, np.nan)
            detail["lower_penalty"] = np.where(
                valid, 40 * np.maximum(detail.ci_lo - cfg.xi_true, 0), np.nan
            )
            detail["upper_penalty"] = np.where(
                valid, 40 * np.maximum(cfg.xi_true - detail.ci_hi, 0), np.nan
            )
            detail["score"] = detail.width + detail.lower_penalty + detail.upper_penalty
            detail["bias"] = detail.xi_hat - cfg.xi_true
            pieces.append(detail)
    result = pd.concat(pieces, ignore_index=True)
    result.attrs["series_sha256"] = hashlib.sha256(
        np.ascontiguousarray(bank[:limit_reps]).tobytes()
    ).hexdigest()
    return result


def main():
    """Run scenarios in independent processes and retain a source-bound manifest."""
    from benchmark.evi_ci_exploration_report import write_report

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / "out/benchmark/evi_ci_exploration")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--limit-reps",
        type=int,
        default=100,
        help="Use a prefix of each existing 100-series bank for a cost pilot.",
    )
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit_reps <= 100 or args.workers < 1:
        parser.error("limit-reps must be in 1..100 and workers must be positive.")
    out_dir = args.out.resolve()
    if not any(out_dir.is_relative_to(root / directory) for directory in ("out", ".cache")):
        parser.error("Experiment outputs must stay inside this repository's out/ or .cache/.")
    if args.report_only:
        manifest = json.loads((out_dir / "manifest.json").read_text())
        detail = pd.concat(
            [pd.read_csv(p) for p in sorted((out_dir / "trials").glob("*.csv.gz"))],
            ignore_index=True,
        )
        write_report(detail, out_dir, manifest, COVARIANCES, INTERVALS, BASELINE)
        return
    if (out_dir / "manifest.json").exists():
        parser.error(
            "This run already has a manifest; use --report-only or a new output directory."
        )
    (out_dir / "trials").mkdir(parents=True, exist_ok=True)
    configs = default_evi_simulation_configs(n_obs=365, reps=100)
    grid = generate_block_sizes(365)
    source_paths = sorted(
        set(root.glob("src/unibm/**/*.py")) | set((root / "scripts/benchmark").glob("*.py"))
    )
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "command": " ".join(
            f"{name}={shlex.quote(os.environ[name])}"
            for name in (
                "UNIBM_NO_EXTENSIONS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "OMP_NUM_THREADS",
                "PYTHONPATH",
            )
            if name in os.environ
        )
        + " "
        + shlex.join([sys.executable, "-m", "benchmark.evi_ci_exploration", *sys.argv[1:]]),
        "master_seed": BENCHMARK_MASTER_SEED,
        "monte_carlo_reps": args.limit_reps,
        "scenarios": len(configs),
        "configs": [asdict(c) for c in configs],
        "workers": args.workers,
        "native_extensions": kernels is not None,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "block_grid": grid.tolist(),
        "superblock_length": max(2 * int(grid[-1]), int(np.sqrt(365))),
        "segments": 365 // max(2 * int(grid[-1]), int(np.sqrt(365))),
        "adaptive_reps": list(ADAPTIVE_REPS),
        "mcse_tolerance": MCSE_TOLERANCE,
        "candidate_count_per_scheme": len(CANDIDATES),
        "baseline": BASELINE,
        "bootstrap_center": "unresampled matching segment-maxima bank",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "elapsed_s": 0.0,
        "series_sha256": {},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    pieces = []
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(run_scenario, (cfg, root, args.limit_reps)): cfg for cfg in configs
            }
            for future in as_completed(futures):
                cfg = futures[future]
                frame = future.result()
                manifest["series_sha256"][cfg.scenario] = frame.attrs["series_sha256"]
                frame.to_csv(out_dir / "trials" / f"{cfg.scenario}.csv.gz", index=False)
                pieces.append(frame)
                if len(pieces) == 1 or len(pieces) % 7 == 0:
                    status(
                        "evi_ci",
                        f"completed {len(pieces)}/{len(configs)} scenarios; {time.perf_counter() - started:.1f}s",
                    )
        detail = pd.concat(pieces, ignore_index=True)
        assert len(detail) == len(configs) * args.limit_reps * 2 * len(CANDIDATES)
        assert not detail.duplicated(["scenario", "rep", "scheme", "method"]).any()
        manifest.update(
            status="completed",
            elapsed_s=time.perf_counter() - started,
            rows=len(detail),
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        summary = write_report(detail, out_dir, manifest, COVARIANCES, INTERVALS, BASELINE)
        status("evi_ci", f"report: {out_dir / 'report.html'}")
        print(
            summary.groupby("scheme", observed=True)
            .head(5)[["scheme", "method", "score", "covered", "precision_met_rate"]]
            .to_string(index=False),
            flush=True,
        )
    except Exception:
        manifest.update(status="failed", elapsed_s=time.perf_counter() - started)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
