"""Compare prime/100 shrinkages on canonical applications without report export.

Run with PYTHONPATH=src:scripts python -m application.shrinkage_sensitivity.
Each path shares its observed window and bootstrap draws across all deltas.
No real-data coverage or interval score can be computed without a known truth.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd
import scipy

from application.diagnostics import application_observations_per_year
from application.inputs import build_application_inputs
from application.specs import (
    APPLICATIONS,
    APPLICATION_RANDOM_STATE,
    ApplicationBundle,
    DESIGN_LIFE_LEVEL_HORIZONS,
)
from config import resolve_repo_dirs
from shared.runtime import initialize_numerical_worker, status
from unibm._bootstrap_precision import ADAPTIVE_REPS, MCSE_TOLERANCE, adaptive_covariance
from unibm._bootstrap_sampling import default_circular_bootstrap_block_size
from unibm.ei import EI_DEFAULT_COVARIANCE_SHRINKAGE, estimate_pooled_bm_ei, prepare_ei_bundle
from unibm.ei._stats import _log_scale_theta_interval
from unibm.ei.bm import _fit_pooled_z_model
from unibm.ei.bootstrap import _ei_path_sampler
from unibm.ei.selection import extract_stable_path_window
from unibm.evi import (
    DEFAULT_COVARIANCE_SHRINKAGE,
    estimate_design_life_level,
    estimate_design_life_level_interval,
    estimate_evi_quantile,
)
from unibm.evi._regression import Z_CRIT_95, _fit_linear_model
from unibm.evi._accelerator import kernels
from unibm.evi.bootstrap import _adaptive_block_summary_bootstrap


DELTAS = tuple(p / 100 for p in range(2, 101) if all(p % d for d in range(2, int(p**0.5) + 1)))
ROOT = Path(__file__).resolve().parents[2]


def array_hash(values):
    """Hash canonical little-endian values, including their shape."""
    arr = np.asarray(values, dtype="<f8")
    return hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()


def evi_candidates(covariance, window, rate):
    """Monitor xi and four design-life levels and their actual Wald endpoints."""
    design = np.column_stack((np.ones(4), np.log(np.ceil(DESIGN_LIFE_LEVEL_HORIZONS * rate))))
    models, targets, scales = [], [], []
    for delta in DELTAS:
        model = _fit_linear_model(window.x, window.y, covariance, delta)
        xi, se = model["slope"], model["standard_error"]
        log_level = design @ np.array([model["intercept"], xi])
        log_se = np.sqrt(
            np.maximum(np.einsum("ij,jk,ik->i", design, model["cov_beta"], design), 0)
        )
        levels = np.exp(log_level[:, None] + Z_CRIT_95 * log_se[:, None] * [0, -1, 1])
        targets.append(np.r_[xi, xi - Z_CRIT_95 * se, xi + Z_CRIT_95 * se, levels.ravel()])
        scales.append(np.r_[np.repeat(se, 3), np.repeat(levels[:, 0] * log_se, 3)])
        models.append(model)
    return models, np.asarray(targets), np.asarray(scales)


def ei_candidates(covariance, z_values):
    """Monitor theta endpoints and unconstrained z, including the theta=1 boundary."""
    models, targets, scales = [], [], []
    for delta in DELTAS:
        model = _fit_pooled_z_model(z_values, covariance=covariance, covariance_shrinkage=delta)
        z, se = model["intercept"], model["standard_error"]
        raw_z = model["unconstrained_intercept"]
        theta = np.exp(-z)
        targets.append(
            [
                theta,
                *_log_scale_theta_interval(z, se),
                raw_z,
                raw_z - Z_CRIT_95 * se,
                raw_z + Z_CRIT_95 * se,
            ]
        )
        scales.append([theta * se] * 3 + [se] * 3)
        models.append(model)
    return models, np.asarray(targets), np.asarray(scales)


def precision_ratios(bootstrap, scales, columns=slice(None)):
    """Keep per-delta MCSE flags separate from the joint stopping decision."""
    mcse = np.asarray(bootstrap["bootstrap_mcse"]).reshape(scales.shape)
    ratios = np.divide(mcse, scales, out=np.full_like(mcse, np.inf), where=scales > 0)
    return np.max(ratios[:, columns], axis=1)


def run_case(task):
    """Refit all deltas, keeping one sampling budget per application and path."""
    spec, inputs, out = task
    started = time.perf_counter()
    status("shrinkage", f"starting {spec.label}")
    values = inputs.evi.series.to_numpy(dtype=float)
    initial = estimate_evi_quantile(values, regression="OLS", quantile=spec.quantile)
    bundle = ApplicationBundle(spec, inputs, initial, None, None, None, None, None)
    rate = application_observations_per_year(bundle)
    window, curve = initial.plateau, initial.curve
    selected = slice(window.start, window.stop)

    def evaluate_evi(cov, _rows):
        _, targets, scales = evi_candidates(cov[selected, selected], window, rate)
        return targets.ravel(), scales.ravel()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        boot = _adaptive_block_summary_bootstrap(
            values,
            curve.positive_block_sizes,
            target="quantile",
            quantile=spec.quantile,
            sliding=True,
            super_block_size=None,
            random_state=APPLICATION_RANDOM_STATE,
            evaluate=evaluate_evi,
            n_threads=1,
        )
    if boot["covariance"] is None:
        raise ValueError(f"{spec.key}: unavailable EVI bootstrap covariance")
    models, targets, scales = evi_candidates(boot["covariance"][selected, selected], window, rate)
    ratios = precision_ratios(boot, scales)
    parameter_ratios = precision_ratios(boot, scales, slice(0, 3))
    rows, designs, audit = [], [], []

    def retain_bootstrap(method, bootstrap, series, grid, levels, path, length, caught):
        """Retain small numeric evidence; no large resampled time series or pickle."""
        name = f"{spec.key}__{method}.npz"
        np.savez_compressed(
            out / name,
            samples=bootstrap["samples"],
            covariance=bootstrap["covariance"],
            grid=grid,
            levels=levels,
            observed_path=path,
            mcse=bootstrap["bootstrap_mcse"],
        )
        info = dict(
            application=spec.key,
            label=spec.label,
            method=method,
            n_obs=len(series),
            start=str(series.index.min()),
            end=str(series.index.max()),
            series_sha256=array_hash(series.to_numpy()),
            dates_sha256=hashlib.sha256(series.index.asi8.tobytes()).hexdigest(),
            covariance_sha256=array_hash(bootstrap["covariance"]),
            grid_min=int(grid[0]),
            grid_max=int(grid[-1]),
            grid_points=len(grid),
            window_min=int(levels[0]),
            window_max=int(levels[-1]),
            window_points=len(levels),
            resampling_length=int(length),
            bootstrap_reps=bootstrap["bootstrap_reps_used"],
            all_precision_met=bootstrap["bootstrap_precision_met"],
            warning=" | ".join(sorted({str(w.message) for w in caught})),
            artifact=name,
        )
        audit.append(info)
        return info

    info = retain_bootstrap(
        "evi",
        boot,
        inputs.evi.series,
        curve.positive_block_sizes,
        curve.positive_block_sizes[selected],
        curve.log_values,
        boot["super_block_size"],
        caught,
    )
    for i, delta in enumerate(DELTAS):
        fit = estimate_evi_quantile(
            values,
            regression="FGLS",
            quantile=spec.quantile,
            curve=curve,
            plateau=window,
            bootstrap_result=boot,
            covariance_shrinkage=delta,
        )
        np.testing.assert_allclose([fit.slope, *fit.confidence_interval], targets[i, :3])
        rows.append(
            dict(
                **info,
                delta=delta,
                estimate=fit.slope,
                ci_lo=fit.confidence_interval[0],
                ci_hi=fit.confidence_interval[1],
                precision_ratio=ratios[i],
                precision_met=bool(ratios[i] <= MCSE_TOLERANCE),
                parameter_precision_ratio=parameter_ratios[i],
                parameter_precision_met=bool(parameter_ratios[i] <= MCSE_TOLERANCE),
                condition_number=fit.covariance_condition_number_regularized,
            )
        )
        levels = estimate_design_life_level(
            fit, DESIGN_LIFE_LEVEL_HORIZONS, observations_per_year=rate
        )
        lo, hi = estimate_design_life_level_interval(
            fit, DESIGN_LIFE_LEVEL_HORIZONS, observations_per_year=rate
        )
        np.testing.assert_allclose(np.column_stack((levels, lo, hi)), targets[i, 3:].reshape(4, 3))
        for j, years in enumerate(DESIGN_LIFE_LEVEL_HORIZONS):
            design_ratio = precision_ratios(boot, scales, slice(3 + 3 * j, 6 + 3 * j))[i]
            designs.append(
                dict(
                    application=spec.key,
                    label=spec.label,
                    delta=delta,
                    years=years,
                    tau=spec.quantile,
                    observations_per_year=rate,
                    observation_basis=spec.design_life_level_basis,
                    estimate=levels[j],
                    ci_lo=lo[j],
                    ci_hi=hi[j],
                    precision_ratio=design_ratio,
                    precision_met=bool(design_ratio <= MCSE_TOLERANCE),
                )
            )

    if spec.formal_ei:
        prepared = prepare_ei_bundle(
            inputs.ei.series.to_numpy(),
            allow_zeros=spec.ei_allow_zeros,
            path_keys=(("bb", True), ("northrop", True)),
        )
        for base in ("bb", "northrop"):
            path = prepared.paths[base, True]
            levels, z = extract_stable_path_window(path)
            mask = np.isin(prepared.block_sizes, levels)
            indices = np.ix_(mask, mask)

            def evaluate_ei(cov, _rows):
                _, targets, scales = ei_candidates(cov[indices], z)
                return targets.ravel(), scales.ravel()

            length = default_circular_bootstrap_block_size(len(prepared.values))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                with _ei_path_sampler(
                    prepared.values,
                    prepared.block_sizes,
                    base_path=base,
                    sliding=True,
                    length=length,
                    n_threads=1,
                ) as draw:
                    boot = adaptive_covariance(
                        draw, evaluate_ei, random_state=APPLICATION_RANDOM_STATE
                    )
            boot.update(
                block_sizes=prepared.block_sizes,
                base_path=base,
                sliding=True,
                bootstrap_block_length=length,
                bootstrap_block_length_policy="default",
            )
            _, targets, scales = ei_candidates(boot["covariance"][indices], z)
            ratios = precision_ratios(boot, scales)
            info = retain_bootstrap(
                base,
                boot,
                inputs.ei.series,
                prepared.block_sizes,
                levels,
                path.z_path,
                length,
                caught,
            )
            for i, delta in enumerate(DELTAS):
                fit = estimate_pooled_bm_ei(
                    prepared,
                    base_path=base,
                    sliding=True,
                    regression="FGLS",
                    bootstrap_result=boot,
                    covariance_shrinkage=delta,
                )
                np.testing.assert_allclose(
                    [fit.theta_hat, *fit.confidence_interval], targets[i, :3]
                )
                rows.append(
                    dict(
                        **info,
                        delta=delta,
                        estimate=fit.theta_hat,
                        ci_lo=fit.confidence_interval[0],
                        ci_hi=fit.confidence_interval[1],
                        precision_ratio=ratios[i],
                        precision_met=bool(ratios[i] <= MCSE_TOLERANCE),
                        parameter_precision_ratio=ratios[i],
                        parameter_precision_met=bool(ratios[i] <= MCSE_TOLERANCE),
                        condition_number=fit.covariance_condition_number_regularized,
                    )
                )
    elapsed = time.perf_counter() - started
    status("shrinkage", f"finished {spec.label} in {elapsed:.1f}s")
    return rows, designs, audit, elapsed


def build_report(detail, design, out):
    """Plot all prime deltas with uncertainty and an explicit default reference."""
    import matplotlib.pyplot as plt

    summary, sections = [], []
    colors = {"evi": "#1565c0", "bb": "#a62929", "northrop": "#248050"}
    names = {
        "evi": "EVI median sliding FGLS",
        "bb": "BB sliding FGLS",
        "northrop": "Northrop sliding FGLS",
    }
    for key, case in detail.groupby("application", sort=False):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for method, group in case.groupby("method", sort=False):
            group = group.sort_values("delta")
            default_delta = (
                DEFAULT_COVARIANCE_SHRINKAGE
                if method == "evi"
                else EI_DEFAULT_COVARIANCE_SHRINKAGE
            )
            baseline = group.loc[group.delta == default_delta].iloc[0]
            ax = axes[0, 0 if method == "evi" else 1]
            ax.plot(group.delta, group.estimate, ".-", color=colors[method], label=names[method])
            ax.fill_between(
                group.delta, group.ci_lo, group.ci_hi, color=colors[method], alpha=0.15
            )
            failed = group.loc[~group.parameter_precision_met]
            if not failed.empty:
                ax.scatter(
                    failed.delta,
                    failed.estimate,
                    marker="x",
                    color="black",
                    s=28,
                    zorder=4,
                    label="MC precision unmet",
                )
            axes[1, 1].plot(
                group.delta,
                group.condition_number,
                ".-",
                color=colors[method],
                label=names[method],
            )
            width = group.ci_hi - group.ci_lo
            summary.append(
                dict(
                    application=key,
                    method=method,
                    default_delta=default_delta,
                    default_estimate=baseline.estimate,
                    estimate_min=group.estimate.min(),
                    estimate_max=group.estimate.max(),
                    default_ci_width=baseline.ci_hi - baseline.ci_lo,
                    ci_width_min=width.min(),
                    ci_width_max=width.max(),
                    precise_candidates=int(group.precision_met.sum()),
                    parameter_precise_candidates=int(group.parameter_precision_met.sum()),
                    bootstrap_reps=int(baseline.bootstrap_reps),
                )
            )
        levels = design[(design.application == key) & (design.years == 10)].sort_values("delta")
        axes[1, 0].plot(levels.delta, levels.estimate, ".-", color=colors["evi"])
        axes[1, 0].fill_between(
            levels.delta, levels.ci_lo, levels.ci_hi, color=colors["evi"], alpha=0.15
        )
        failed_levels = levels.loc[~levels.precision_met]
        if not failed_levels.empty:
            axes[1, 0].scatter(
                failed_levels.delta,
                failed_levels.estimate,
                marker="x",
                color="black",
                s=28,
                zorder=4,
                label="MC precision unmet",
            )
        axes[1, 0].set_yscale("log")
        axes[1, 1].set_yscale("log")
        titles = [
            "EVI estimate and nominal 95% CI",
            "EI estimate and nominal 95% CI",
            "10-year design-life median and 95% CI",
            "Regularized covariance condition number",
        ]
        for i, (ax, title) in enumerate(zip(axes.flat, titles)):
            ax.set_title(title, fontsize=11)
            ax.set_xlabel(r"Shrinkage $\delta$")
            if i != 3:
                delta = EI_DEFAULT_COVARIANCE_SHRINKAGE if i == 1 else DEFAULT_COVARIANCE_SHRINKAGE
                ax.axvline(delta, color="#555", linestyle="--", lw=1)
            ax.grid(alpha=0.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        axes[0, 0].set_ylabel(r"$\xi$")
        axes[0, 1].set_ylabel(r"$\theta$")
        spec = next(s for s in APPLICATIONS if s.key == key)
        axes[1, 0].set_ylabel(spec.ylabel)
        if not spec.formal_ei:
            axes[0, 1].text(
                0.5,
                0.5,
                "Not part of formal EI analysis",
                ha="center",
                transform=axes[0, 1].transAxes,
            )
        fig.suptitle(
            f"{spec.label}: shrinkage sensitivity\n25 prime/100 values; dashed lines = defaults "
            f"(EVI {DEFAULT_COVARIANCE_SHRINKAGE}, EI {EI_DEFAULT_COVARIANCE_SHRINKAGE})",
            fontsize=14,
        )
        fig.savefig(out / f"{key}.png", dpi=160)
        fig.savefig(out / f"{key}.pdf")
        plt.close(fig)
        audit = case.drop_duplicates("method")[
            [
                "method",
                "n_obs",
                "window_min",
                "window_max",
                "resampling_length",
                "bootstrap_reps",
                "all_precision_met",
            ]
        ]
        table = case[
            [
                "method",
                "delta",
                "estimate",
                "ci_lo",
                "ci_hi",
                "parameter_precision_ratio",
                "parameter_precision_met",
                "precision_met",
            ]
        ]
        sections.append(
            f'<h2>{escape(spec.label)}</h2><img src="{key}.png" alt="Shrinkage sensitivity">'
            + audit.to_html(index=False)
            + "<details><summary>All 25 candidates</summary>"
            + table.to_html(index=False, float_format=lambda x: f"{x:.6g}")
            + "</details>"
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Application shrinkage sensitivity</title>'
        "<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#182536}"
        "img{width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}"
        "details{margin:20px 0}h2{margin-top:55px}</style><h1>Application shrinkage sensitivity</h1>"
        "<p>25 primes / 100: " + ", ".join(f"{d:.2f}" for d in DELTAS) + ".</p>"
        f"<p>{detail.application.nunique()} EVI median-sliding-FGLS cases; "
        f"{detail[detail.method != 'evi'].application.nunique()} also include BB/Northrop sliding FGLS. "
        "Observed windows, input clocks and paired bootstrap draws are fixed across deltas. "
        "EVI uses model-based Wald; EI uses model-based Wald on z, back-transformed to theta. "
        "No interval scale correction is applied. Defaults: "
        f"EVI {DEFAULT_COVARIANCE_SHRINKAGE}, EI {EI_DEFAULT_COVARIANCE_SHRINKAGE}.</p>"
        "<p>S<sub>δ</sub> = (1−δ)S + δ diag(S) + ridge I. The same regularized covariance "
        "determines FGLS weights and the model-based coefficient covariance. No residual-variance "
        "multiplier is applied.</p>"
        "<p>Default fits use the final shared R, which can exceed a run monitoring only the "
        "default delta; small differences from standard application results can therefore arise.</p>"
        "<p>Adaptive R = 128, 256, 512, 768, 1024 jointly monitors all 25 candidates per path. "
        "EVI monitors xi plus design-life levels at 1, 10, 25, 50 years and their endpoints; "
        "EI monitors theta and unconstrained z with endpoints. Precision means MCSE / statistical SE "
        "&le; 0.10, not coverage. Results at the cap remain visible when precision fails.</p>"
        "<p>Black crosses refer to the target shown in that panel (EI also checks unconstrained z). "
        "The all_precision_met flag requires every delta and horizon to pass. Long-horizon "
        "design-life endpoints can fail this joint check even when xi and its CI pass.</p>"
        "<p>Real-data truth is unknown: narrower CIs or stable estimates do not demonstrate lower "
        "Winkler score, correct coverage or greater accuracy. Use this report to identify sensitivity; "
        "choose defaults jointly with simulation evidence. Intervals condition on the selected window. "
        "Design-life CIs concern fitted median maxima, not future-realization prediction intervals. "
        "NFIP retains the canonical active-claim-day EVI / full-calendar EI distinction.</p>"
        '<p>Downloads: <a href="estimates.csv">all estimates</a> · '
        '<a href="design_life.csv">all design-life horizons</a> · '
        '<a href="summary.csv">ranges</a> · <a href="manifest.json">provenance</a></p>'
        + "".join(sections)
        + "</html>",
        encoding="utf-8",
    )


def main():
    """Write an isolated local report; never invoke the application exporter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--case", choices=[s.key for s in APPLICATIONS])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "out/research/application_shrinkage_sensitivity"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    specs = tuple(s for s in APPLICATIONS if args.case is None or s.key == args.case)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    inputs = build_application_inputs(resolve_repo_dirs(ROOT), specs=specs)
    source_paths = sorted((ROOT / "src/unibm").rglob("*.py")) + sorted(
        (ROOT / "scripts").rglob("*.py")
    )
    source_paths += sorted((ROOT / "src/unibm").rglob("*.pyx"))
    if kernels is not None:
        source_paths.append(Path(kernels.__file__).resolve())
    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
    }
    started = time.perf_counter()
    tasks = [(spec, inputs[spec.key], out) for spec in specs]
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(tasks)), initializer=initialize_numerical_worker
    ) as pool:
        results = list(pool.map(run_case, tasks))
    detail = pd.DataFrame([r for result in results for r in result[0]])
    design = pd.DataFrame([r for result in results for r in result[1]])
    detail.to_csv(out / "estimates.csv", index=False)
    design.to_csv(out / "design_life.csv", index=False)
    build_report(detail, design, out)
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        deltas=DELTAS,
        evi_default_shrinkage=DEFAULT_COVARIANCE_SHRINKAGE,
        ei_default_shrinkage=EI_DEFAULT_COVARIANCE_SHRINKAGE,
        random_state=APPLICATION_RANDOM_STATE,
        adaptive_reps=ADAPTIVE_REPS,
        precision_tolerance=MCSE_TOLERANCE,
        workers=min(args.workers, len(tasks)),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        native_kernels=kernels is not None,
        elapsed_s=time.perf_counter() - started,
        sources=source_hashes,
        cases=[r for result in results for r in result[2]],
        case_seconds={s.key: r[3] for s, r in zip(specs, results)},
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    status("shrinkage", f"report: {out / 'report.html'}")


if __name__ == "__main__":
    main()
