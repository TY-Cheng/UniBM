"""Cross-validate EVI/EI Wald interval scales without changing fitted estimators.

EVI scales offsets on xi; EI compares clipped theta offsets with z=-log(theta)
Wald scaling. Objectives and comparisons always use the original parameter scale.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from benchmark.evi_ci_scale_calibration import FOLD_SEED, assign_folds, optimal_scale


KEYS = ["branch", "method", "scale"]
Z_GRID_SIZE = 513


def endpoints(frame, scale, c):
    """Scale only interval offsets; the estimator and covariance stay fixed."""
    point = frame.estimate.to_numpy()
    if scale == "z":
        z, half = -np.log(point), 1.96 * frame.z_standard_error.to_numpy()
        return np.exp(-z - c * half), np.exp(-np.maximum(0, z - c * half))
    lo = point + c * (frame.ci_lo.to_numpy() - point)
    hi = point + c * (frame.ci_hi.to_numpy() - point)
    if scale == "theta":
        lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    return lo, hi


def scores(frame, scale, c):
    """Return Winkler score, width and coverage; failed fits remain visible."""
    lo, hi = endpoints(frame, scale, c)
    truth = frame.truth.to_numpy()
    valid = frame.valid.to_numpy(bool) & np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
    width = np.where(valid, hi - lo, np.nan)
    score = width + 40 * (np.maximum(lo - truth, 0) + np.maximum(truth - hi, 0))
    covered = valid & (lo <= truth) & (truth <= hi)
    return lo, hi, score, width, covered


def objective_grid(frame, scale, candidates, weights):
    """Evaluate bounded-memory vector batches, not a Python loop over records."""
    output = np.empty(len(candidates))
    truth = frame.truth.to_numpy()
    for start in range(0, len(candidates), 128):
        c = candidates[start : start + 128, None]
        lo, hi = endpoints(frame, scale, c)
        loss = hi - lo + 40 * (np.maximum(lo - truth, 0) + np.maximum(truth - hi, 0))
        output[start : start + 128] = loss @ weights
    return output


def _best_candidate(candidates, values, *, status):
    """Reject an unattained c=0 optimum; break actual ties toward c=1."""
    best = np.min(values)
    options = np.flatnonzero(values == best)
    positive = options[candidates[options] > 0]
    if not len(positive):
        return np.nan, "zero_boundary"
    chosen = positive[np.argmin(np.abs(candidates[positive] - 1))]
    return float(candidates[chosen]), status


def theta_scale(frame, weights):
    """Find the global piecewise-linear minimum including all clipping knots.

    Clipping can make the slope decrease: neither convexity nor an early exit
    based on the derivative at zero is valid here.
    """
    point, truth = frame.estimate.to_numpy(), frame.truth.to_numpy()
    a = np.maximum(point - frame.ci_lo.to_numpy(), 0)
    b = np.maximum(frame.ci_hi.to_numpy() - point, 0)
    knots, jumps = [], []
    for length, distance, mask, jump in (
        (a, point - truth, (a > 0) & (truth < point), 40 * a * weights),
        (b, truth - point, (b > 0) & (truth > point), 40 * b * weights),
        (a, point, a > 0, -a * weights),
        (b, 1 - point, b > 0, -b * weights),
    ):
        knots.extend((distance[mask] / length[mask]).tolist())
        jumps.extend(jump[mask].tolist())
    if not knots:
        return 1.0, "flat_minimum", {}
    order = np.argsort(knots)
    knots = np.asarray(knots)[order]
    jumps = np.asarray(jumps, dtype=np.longdouble)[order]
    knots, starts = np.unique(knots, return_index=True)
    jumps = np.add.reduceat(jumps, starts)
    initial = np.sum(
        weights * (np.where(truth < point, -39 * a, a) + np.where(truth > point, -39 * b, b)),
        dtype=np.longdouble,
    )
    slopes = initial + np.r_[np.longdouble(0), np.cumsum(jumps[:-1])]
    loss_zero = np.sum(40 * weights * np.abs(point - truth), dtype=np.longdouble)
    values = loss_zero + np.cumsum(slopes * np.diff(np.r_[0, knots]))
    candidates = np.r_[0, knots, 1.0]
    values = np.r_[loss_zero, values, objective_grid(frame, "theta", np.array([1.0]), weights)]
    c, status = _best_candidate(candidates, values, status="piecewise_linear_minimum")
    direct_fallback = False
    if np.isfinite(c):
        actual = objective_grid(frame, "theta", np.array([c]), weights)[0]
        if not np.isclose(actual, values.min(), rtol=1e-10, atol=1e-10):
            # Some platforms have no extra longdouble precision. Large knots
            # can amplify sweep roundoff; directly evaluate this exceptional case.
            values = objective_grid(frame, "theta", candidates, weights)
            c, status = _best_candidate(candidates, values, status="piecewise_linear_minimum")
            direct_fallback = True
    return c, status, {"search_points": len(candidates), "direct_fallback": direct_fallback}


def z_scale(frame, weights, grid_size=Z_GRID_SIZE):
    """Numerically search every observed kink plus logarithmically spaced points.

    Beyond the last truth crossing, all terms with nonzero SE cover their true
    values and only increase in width. This gives a training-data-only upper
    bound. Local refinements do not constitute a proof of global optimality;
    a second grid density is checked in the report for every fitted coefficient.
    """
    point, truth = frame.estimate.to_numpy(), frame.truth.to_numpy()
    h = 1.96 * frame.z_standard_error.to_numpy()
    if not (
        (point > 0) & (point <= 1) & (truth > 0) & (truth <= 1) & np.isfinite(h) & (h >= 0)
    ).all():
        raise ValueError("Z calibration requires theta and truth in (0,1], finite nonnegative SE.")
    active = h > 0
    if not active.any():
        return 1.0, "flat_minimum", {"search_points": 1, "upper_bound": 1.0}
    z = -np.log(point[active])
    crossings = np.abs(np.log(truth[active]) + z) / h[active]
    upper = float(np.max(crossings))
    if upper == 0:
        return np.nan, "zero_boundary", {"search_points": 1, "upper_bound": 0.0}
    kinks = np.r_[crossings, z / h[active]]
    kinks = kinks[(kinks > 0) & (kinks <= upper)]
    lower = min(float(kinks.min()), min(1.0, upper)) / 100
    candidates = np.unique(np.r_[0, 1.0, kinks, np.geomspace(lower, upper, grid_size)])
    values = objective_grid(frame, "z", candidates, weights)
    minima = np.flatnonzero((values[1:-1] < values[:-2]) & (values[1:-1] <= values[2:])) + 1
    refinements = []
    for i in minima:
        left, right = candidates[i - 1], candidates[i + 1]
        if right == left:
            continue
        # Normalize to [0,1] so tolerances do not depend on the magnitude of c.
        result = minimize_scalar(
            lambda t: objective_grid(frame, "z", np.array([left + t * (right - left)]), weights)[
                0
            ],
            bounds=(0, 1),
            method="bounded",
            options={"xatol": 1e-11},
        )
        refinements.append((left + result.x * (right - left), result.fun))
    if refinements:
        candidates = np.r_[candidates, np.asarray(refinements)[:, 0]]
        values = np.r_[values, np.asarray(refinements)[:, 1]]
    c, status = _best_candidate(candidates, values, status="numerical_minimum")
    return c, status, {"search_points": len(candidates), "upper_bound": upper}


def fit_scale(train, scale):
    """Learn a single coefficient across training scenarios, with equal weights."""
    valid = train.loc[train.valid].copy()
    if valid.empty:
        return np.nan, "no_valid_training_rows", {}
    counts = valid.groupby("scenario").rep.transform("count")
    weights = (1 / counts / valid.scenario.nunique()).to_numpy()
    if scale == "xi":
        c, status = optimal_scale(
            valid.ci_lo - valid.estimate,
            valid.ci_hi - valid.estimate,
            valid.truth - valid.estimate,
            weights,
        )
        return c, status, {}
    if scale == "theta":
        return theta_scale(valid, weights)
    c, status, diagnostics = z_scale(valid, weights)
    fine, fine_status, check = z_scale(valid, weights, grid_size=2 * Z_GRID_SIZE - 1)
    if np.isfinite(c) != np.isfinite(fine):
        raise ArithmeticError("Z calibration boundary status changes with grid density.")
    if np.isfinite(c):
        losses = objective_grid(valid, "z", np.array([c, fine]), weights)
        diagnostics["density_score_gap"] = float(abs(losses[0] - losses[1]))
        if abs(losses[0] - losses[1]) > 1e-7 * max(1, abs(losses[1])):
            raise ArithmeticError("Z calibration objective is unstable to grid density.")
        if losses[1] < losses[0]:
            c, status = fine, fine_status
    diagnostics["fine_search_points"] = check["search_points"]
    return c, status, diagnostics


def cross_validate(frame, replicates):
    """Each series is held out once; failures keep coverage at zero, score NA."""
    frame = frame.copy()
    frame["fold"] = assign_folds(replicates)[frame.rep.to_numpy(int)]
    details, fits = [], []
    for (branch, method), group in frame.groupby(["branch", "method"], sort=True):
        for scale in ("xi",) if branch == "evi" else ("z", "theta"):
            group = group.copy()
            group["scale"] = scale
            group["c_cv"] = np.nan
            group["calibration_status"] = "unfitted"
            for fold in [*range(5), -1]:
                train = group if fold == -1 else group.loc[group.fold != fold]
                c, status, diagnostics = fit_scale(train, scale)
                train_score = scores(train, scale, c)[2]
                raw_score = scores(train, scale, 1.0)[2]

                def balanced(values):
                    return (
                        pd.Series(values, index=train.index).groupby(train.scenario).mean().mean()
                    )

                fits.append(
                    dict(
                        branch=branch,
                        method=method,
                        scale=scale,
                        held_out_fold=fold,
                        c=c,
                        status=status,
                        training_rows=len(train),
                        valid_training_rows=int(train.valid.sum()),
                        training_scenarios=train.loc[train.valid, "scenario"].nunique(),
                        training_score=balanced(train_score),
                        training_raw_score=balanced(raw_score),
                        **diagnostics,
                    )
                )
                if fold >= 0:
                    mask = group.fold == fold
                    group.loc[mask, "c_cv"] = c
                    group.loc[mask, "calibration_status"] = status
            lo, hi, raw_score, raw_width, raw_covered = scores(group, scale, 1.0)
            valid = group.valid.to_numpy(bool)
            if not np.allclose(
                np.c_[lo, hi][valid], group.loc[valid, ["ci_lo", "ci_hi"]], atol=1e-12, rtol=1e-10
            ):
                raise ArithmeticError("c=1 does not reproduce the production CI.")
            lo, hi, score, width, covered = scores(group, scale, group.c_cv.to_numpy())
            group = group.assign(
                scaled_lo=lo,
                scaled_hi=hi,
                score=score,
                width=width,
                covered=covered,
                raw_score=raw_score,
                raw_width=raw_width,
                raw_covered=raw_covered,
                paired_delta=score - raw_score,
            )
            details.append(group)
            print(f"Calibrated {branch}/{method}/{scale}", flush=True)
    return pd.concat(details, ignore_index=True), pd.DataFrame(fits)


def scenario_summary(detail):
    """Separate scored-fit means from coverage over all original attempts."""
    detail = detail.assign(paired_raw_score=detail.raw_score.where(detail.score.notna()))
    result = detail.groupby(
        KEYS + ["scenario", "family", "xi_true", "theta_true"], as_index=False
    ).agg(
        raw_score=("raw_score", "mean"),
        paired_raw_score=("paired_raw_score", "mean"),
        score=("score", "mean"),
        paired_delta=("paired_delta", "mean"),
        raw_coverage=("raw_covered", "mean"),
        coverage=("covered", "mean"),
        raw_width=("raw_width", "mean"),
        width=("width", "mean"),
        attempts=("rep", "size"),
        valid_fits=("valid", "sum"),
        scored_fits=("score", "count"),
        bootstrap_precision_met=("precision_met", "mean"),
        mean_R=("bootstrap_reps", "mean"),
    )
    result["failed_fits"] = result.attempts - result.valid_fits
    result["calibration_unavailable"] = result.valid_fits - result.scored_fits
    result["improvement_percent"] = -100 * result.paired_delta / result.paired_raw_score
    return result


def write_report(detail, fits, out, manifest):
    """Show each scenario, including regressions and parameter-boundary cases."""
    cases = scenario_summary(detail)
    metrics = [
        "raw_score",
        "paired_raw_score",
        "score",
        "paired_delta",
        "raw_coverage",
        "coverage",
        "raw_width",
        "width",
    ]
    summary = cases.groupby(KEYS, as_index=False)[metrics].mean()
    counts = cases.groupby(KEYS, as_index=False).agg(
        better_scenarios=("paired_delta", lambda x: int((x < -1e-10).sum())),
        worse_scenarios=("paired_delta", lambda x: int((x > 1e-10).sum())),
        scored_scenarios=("score", "count"),
        failed_fits=("failed_fits", "sum"),
        calibration_unavailable=("calibration_unavailable", "sum"),
    )
    summary = summary.merge(counts, on=KEYS)
    summary = summary.merge(
        fits[fits.held_out_fold >= 0]
        .groupby(KEYS, as_index=False)
        .agg(c_min=("c", "min"), c_max=("c", "max")),
        on=KEYS,
    )
    summary = summary.merge(
        fits[fits.held_out_fold == -1][KEYS + ["c"]].rename(columns={"c": "c_all_data"}), on=KEYS
    )
    summary["improvement_percent"] = -100 * summary.paired_delta / summary.paired_raw_score
    summary.to_csv(out / "summary.csv", index=False)
    cases.to_csv(out / "by_scenario.csv", index=False)
    fits.to_csv(out / "calibration_fits.csv", index=False)
    detail.to_csv(out / "held_out_trials.csv.gz", index=False)
    for field in ("family", "xi_true", "theta_true"):
        cases.groupby(KEYS + [field], as_index=False)[metrics].mean().to_csv(
            out / f"by_{field}.csv", index=False
        )
    pd.DataFrame(
        {
            "rep": np.arange(manifest["monte_carlo_reps"]),
            "fold": assign_folds(manifest["monte_carlo_reps"]),
        }
    ).to_csv(out / "fold_assignment.csv", index=False)
    sections = []
    for (branch, method, scale), group in cases.groupby(KEYS, sort=True):
        families = sorted(group.family.unique())
        fig, axes = plt.subplots(
            1,
            len(families),
            figsize=(5.2 * len(families), 4.8),
            layout="constrained",
            squeeze=False,
        )
        limit = max(1, float(group.improvement_percent.abs().max()))
        for ax, family in zip(axes.flat, families, strict=True):
            grid = (
                group[group.family == family]
                .pivot(index="theta_true", columns="xi_true", values="improvement_percent")
                .sort_index()
            )
            im = ax.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
            ax.set(title=family.replace("_", " "), xlabel="True EVI xi", ylabel="True EI theta")
            ax.set_xticks(range(len(grid.columns)), [f"{x:g}" for x in grid.columns])
            ax.set_yticks(range(len(grid.index)), [f"{x:g}" for x in grid.index])
            for i, j in np.ndindex(grid.shape):
                value = grid.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:+.1f}" if np.isfinite(value) else "NA",
                    ha="center",
                    va="center",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.5),
                )
        fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            label="Held-out Winkler score improvement (%)",
            shrink=0.8,
        )
        fig.suptitle(f"{branch.upper()} {method} | scale on {scale} | positive = lower score")
        name = f"{branch}_{method}_{scale}"
        fig.savefig(out / f"{name}.png", dpi=150)
        fig.savefig(out / f"{name}.pdf")
        plt.close(fig)
        table = group[
            [
                "family",
                "xi_true",
                "theta_true",
                *metrics,
                "improvement_percent",
                "failed_fits",
                "calibration_unavailable",
            ]
        ]
        sections.append(
            f"<h2>{escape(branch.upper() + ' ' + method + ' / ' + scale)}</h2><img src='{name}.png' alt='Scenario score changes'><details><summary>All scenarios: scores, coverage, widths and failures</summary>"
            + table.to_html(index=False, float_format=lambda x: f"{x:.6g}")
            + "</details>"
        )
    evi = summary[summary.branch == "evi"].set_index("method")
    ei_z = summary[(summary.branch == "ei") & (summary.scale == "z")]
    ei_theta = summary[(summary.branch == "ei") & (summary.scale == "theta")]
    interpretation = f"""## Reading these results

EVI median-sliding mean score improves by {evi.loc["median_sliding_fgls", "improvement_percent"]:.3f}%;
median-disjoint improves by {evi.loc["median_disjoint_fgls", "improvement_percent"]:.2f}%.
These are descriptive held-out changes, not evidence of statistical significance.
Coverage falls after scaling for both methods; score alone selected c.

EI z-scale calibration learns fold-specific c between {ei_z.c_min.min():.2f} and {ei_z.c_max.max():.2f}.
Its mean interval widths are {ei_z.width.min():.3f}–{ei_z.width.max():.3f}, with coverage
{ei_z.coverage.min():.2%}–{ei_z.coverage.max():.2%}. Each method still worsens in
{ei_z.worse_scenarios.min()}–{ei_z.worse_scenarios.max()} of 84 scenarios. Direct theta scaling gives widths
{ei_theta.width.min():.3f}–{ei_theta.width.max():.3f}, approaching the entire parameter range.
For reference, the uninformative interval [0,1] has width 1, coverage 100%,
and Winkler score exactly 1 for every true theta in [0,1]. This reference is
not a fitted or calibrated method. Large average gains here accompany very
wide intervals and substantial differences across scenarios; they do not
justify promoting these coefficients to package defaults or real-data use.
"""
    methods = (
        interpretation
        + """

# Fixed-shrinkage Wald CI scale cross-validation

Six estimators: EVI median sliding/disjoint (delta=0.73), EI BB/Northrop
sliding/disjoint (delta=0.37). Each branch retains its own 84-scenario grid,
N=365 and M=100; smaller CLI runs are explicitly labeled pilots in the manifest.
Raw simulation banks are reused. Production fits and adaptive bootstrap are
recomputed, with the existing 128/256/512/768/1024 stopping rule and seeds.
This is a different sampling budget from the older joint 49-candidate CI study.

One five-fold split, seed 20260925, grouped by replicate index across all
scenarios and methods. Each training split uses 80 replicates per scenario;
20 are held out. One c>0 is learned per method and scale, across all training
scenarios with equal weights. No true-parameter-specific c is selected.
All-data coefficients are retained separately and never used in held-out scores.

EVI scales original CI offsets about xi_hat. EI compares (a) scaling theta CI
offsets about theta_hat and clipping to [0,1], and (b) multiplying the z-scale
Wald half-width by c, clipping the lower z endpoint to zero and exponentiating
with endpoints reversed. Point estimates, covariance, window and bootstrap
draws remain fixed during calibration. c=1 is checked against every valid
production interval. The objective is original-scale mean 95% Winkler score;
coverage is reported but is not the objective or a gate.

EVI uses its convex piecewise-linear optimizer. EI theta calibration evaluates
the full piecewise-linear objective at all truth/clipping knots (not assumed
convex). EI z calibration numerically evaluates all truth/clipping knots plus
logarithmic grids of 513 and 1025 points, refines local minima with SciPy and
checks training-score agreement to 1e-7*max(1,score). This numerical check is
not a proof of a global minimum. A training-only upper search bound follows
from the last finite truth crossing; beyond it, widths only increase. c=0 is
not admissible: an optimum attained only there is reported as unavailable.

Unusable original fits remain failed, contribute zero coverage, and have NA
score/width. Raw means use valid original fits; scaled means and paired deltas
use available paired fits. Improvement percentages divide by paired_raw_score,
the original-score mean on those same paired fits. Training uses
valid fits with equal scenario weights. Calibration failures are disclosed
separately. Unmet original adaptive precision is retained. It does not certify
the scaled endpoints or account for estimation error in c.

Scores pool held-out predictions, once per series. Training folds overlap;
no iid MCSE or formal confidence interval for this CV procedure is claimed.
This is exploratory reuse of an already inspected simulation design, not new
independent-seed validation, real-data calibration, repeated CV, or proof of
post-selection coverage. Improvements are not guaranteed in every scenario.
EVI and EI scores have different targets/scales and must not be pooled.
No package default, application result, documentation result or JoH file changes.
"""
    )
    (out / "report.md").write_text(methods, encoding="utf-8")
    html = (
        """<!doctype html><html lang='en'><meta charset='utf-8'><title>Wald CI scale CV</title>
<style>body{font:15px/1.6 system-ui;max-width:1550px;margin:30px auto;padding:0 20px;color:#172330}img{max-width:100%}table{border-collapse:collapse;font-size:12px}th,td{padding:6px;border:1px solid #ddd;white-space:nowrap}details,.scroll{overflow:auto}pre{white-space:pre-wrap}h2{margin-top:45px}</style>
<h1>EVI / EI: fixed shrinkage + Wald CI scale calibration</h1>
<p>EVI delta=0.73 · EI delta=0.37 · five-fold CV · c&gt;0 · score on the original parameter scale.
Six estimators, ten calibration variants. Positive improvement means lower held-out mean Winkler score.
Coverage is a fraction (0–1); score improvements do not imply 95% coverage.</p>
<p>Each branch has its own 84 scenarios, N=365, M=100. Every scenario table includes worsened results.
No new simulated series. No automatic default change.</p>
<p><a href='summary.csv'>Summary CSV</a> · <a href='by_scenario.csv'>All scenarios CSV</a> · <a href='calibration_fits.csv'>Coefficients</a> · <a href='held_out_trials.csv.gz'>Held-out trials</a> · <a href='manifest.json'>Provenance</a></p>
"""
        + "<div class='scroll'>"
        + summary.to_html(index=False, float_format=lambda x: f"{x:.6g}")
        + "</div>"
        + "<pre>"
        + escape(interpretation)
        + "</pre>"
        + "".join(sections)
        + "<details><summary>Method and limitations</summary><pre>"
        + escape(methods)
        + "</pre></details></html>"
    )
    (out / "report.html").write_text(html, encoding="utf-8")
    return summary


def analyze(out):
    """Verify immutable fit inputs, compute held-out results and record provenance."""
    out = Path(out).resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["status"] != "fits_completed":
        raise ValueError("Analysis requires completed fits and has not been run already.")
    pieces = []
    for item in manifest["scenarios"]:
        path = out / "trials" / item["trials_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["trials_sha256"]:
            raise ValueError(f"Changed trial file: {path}")
        pieces.append(pd.read_csv(path, float_precision="round_trip"))
    frame = pd.concat(pieces, ignore_index=True)
    if (
        len(frame) != manifest["scenarios_per_branch"] * manifest["monte_carlo_reps"] * 6
        or frame.duplicated(["branch", "method", "scenario", "rep"]).any()
    ):
        raise ValueError("Expected uniquely paired complete estimator rows.")
    started = time.perf_counter()
    detail, fits = cross_validate(frame, manifest["monte_carlo_reps"])
    summary = write_report(detail, fits, out, manifest)
    root = Path(__file__).resolve().parents[2]
    manifest.update(
        status="completed",
        cv_elapsed_s=time.perf_counter() - started,
        fold_seed=FOLD_SEED,
        folds=5,
        rows=len(detail),
        completed_utc=datetime.now(timezone.utc).isoformat(),
        calibration_status_counts=fits.status.value_counts().to_dict(),
        analysis_source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("evi_ci_scale_calibration.py"))
        },
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False), flush=True)
    print(f"Report: {out / 'report.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    analyze(parser.parse_args().out)
