"""Cross-fit a scalar CI correction using retained EVI exploration trials only.

No time series are simulated or resampled. A fold contains the same outer
replicate indices in every scenario and method. Interval scaling is anchored
at the original point estimate, including for asymmetric intervals.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import io
import json
from pathlib import Path
import shlex
import sys
import time

import numpy as np
import pandas as pd

from benchmark.evi_ci_exploration_report import balanced_summary
import matplotlib.pyplot as plt


FOLDS = 5
FOLD_SEED = 20260925
KEYS = ["scheme", "method"]
COLUMNS = [
    "scenario",
    "rep",
    "scheme",
    "method",
    "family",
    "xi_true",
    "theta_true",
    "n_obs",
    "xi_hat",
    "ci_lo",
    "ci_hi",
    "valid",
    "flat_window",
    "near_zero_slope",
    "precision_met",
    "bootstrap_reps",
]


def optimal_scale(lo, hi, target, weights):
    """Minimize weighted 95% Winkler loss over c>0 without a search grid.

    Inputs are CI endpoints and truth relative to the point estimate. The
    derivative increases by 40*w*abs(endpoint) at each positive truth/endpoint
    breakpoint. Its first zero crossing therefore identifies the minimum.
    A flat minimum uses the minimizer closest to 1. If only c=0 minimizes the
    loss, no positive minimizer exists: return NaN instead of imposing a floor.
    """
    lo, hi, target, weights = (np.asarray(x, dtype=float) for x in (lo, hi, target, weights))
    if not (
        lo.ndim == 1
        and lo.size
        and lo.shape == hi.shape == target.shape == weights.shape
        and np.isfinite([lo, hi, target, weights]).all()
        and np.all(lo <= hi)
        and np.all(weights > 0)
    ):
        raise ValueError("Expected finite ordered offsets and positive matching weights.")
    initial = weights * (
        hi
        - lo
        + 40
        * (
            np.where(target < 0, lo, np.where(target == 0, np.maximum(lo, 0), 0))
            + np.where(target > 0, -hi, np.where(target == 0, np.maximum(-hi, 0), 0))
        )
    )
    gradient = initial.sum(dtype=np.longdouble)
    if gradient > 0:
        return np.nan, "zero_boundary"
    offsets = np.r_[lo, hi]
    knots = np.divide(
        np.tile(target, 2), offsets, out=np.full(len(offsets), np.nan), where=offsets != 0
    )
    positive = knots > 0
    if not np.isfinite(knots[positive]).all():
        return np.nan, "nonfinite_breakpoint"
    if not positive.any():
        return (1.0, "flat_minimum") if gradient == 0 else (np.nan, "no_finite_minimum")
    order = np.argsort(knots[positive])
    knots = knots[positive][order]
    jumps = (40 * np.abs(offsets[positive]) * np.tile(weights, 2)[positive])[order]
    knots, starts = np.unique(knots, return_index=True)
    jumps = np.add.reduceat(jumps.astype(np.longdouble), starts)
    if gradient == 0:
        return min(1.0, float(knots[0])), "flat_minimum"
    after = gradient + np.cumsum(jumps, dtype=np.longdouble)
    reached = np.flatnonzero(after >= 0)
    if not len(reached):
        return np.nan, "no_finite_minimum"
    index = int(reached[0])
    lower = float(knots[index])
    if after[index] == 0:
        upper = float(knots[index + 1]) if index + 1 < len(knots) else np.inf
        return float(np.clip(1.0, lower, upper)), "flat_minimum"
    return lower, "minimum"


def assign_folds(replicates, seed=FOLD_SEED):
    """Keep shared replicate seeds together, with balanced deterministic folds."""
    if replicates < FOLDS or replicates % FOLDS:
        raise ValueError("The retained replicate count must be a positive multiple of five.")
    result = np.empty(replicates, dtype=int)
    result[np.random.default_rng(seed).permutation(replicates)] = np.arange(replicates) % FOLDS
    return result


def interval_metrics(frame, scale):
    """Evaluate scaled endpoints with the same 95% score as the source study."""
    estimate = frame.xi_hat.to_numpy()
    truth = frame.xi_true.to_numpy()
    lo = estimate + scale * (frame.ci_lo.to_numpy() - estimate)
    hi = estimate + scale * (frame.ci_hi.to_numpy() - estimate)
    width = hi - lo
    lower = 40 * np.maximum(lo - truth, 0)
    upper = 40 * np.maximum(truth - hi, 0)
    return pd.DataFrame(
        {
            "scaled_lo": lo,
            "scaled_hi": hi,
            "score": width + lower + upper,
            "width": width,
            "covered": np.where(np.isfinite(lo + hi), (lo <= truth) & (truth <= hi), np.nan),
            "lower_penalty": lower,
            "upper_penalty": upper,
            "bias": estimate - truth,
        },
        index=frame.index,
    )


def cross_fit(frame, replicates):
    """Learn each c on four folds; use it only on the excluded fifth fold."""
    frame = frame.copy()
    frame["fold"] = assign_folds(replicates)[frame.rep.to_numpy(int)]
    frame["c_cv"] = np.nan
    frame["fit_status"] = "unfitted"
    fits = []
    for (scheme, method), group in frame.groupby(KEYS, sort=True, observed=True):
        for fold in [*range(FOLDS), -1]:
            train = group if fold == -1 else group.loc[group.fold != fold]
            counts = train.groupby("scenario", observed=True).rep.transform("count")
            weights = (1 / counts / train.scenario.nunique()).to_numpy()
            c, fit_status = optimal_scale(
                train.ci_lo - train.xi_hat,
                train.ci_hi - train.xi_hat,
                train.xi_true - train.xi_hat,
                weights,
            )
            metrics = interval_metrics(train, c)
            raw = interval_metrics(train, 1.0)
            fits.append(
                {
                    "scheme": scheme,
                    "method": method,
                    "held_out_fold": fold,
                    "fit_kind": "all_data" if fold == -1 else "training_folds",
                    "c": c,
                    "status": fit_status,
                    "training_rows": len(train),
                    "training_score": float(np.dot(weights, metrics.score.to_numpy())),
                    "training_raw_score": float(np.dot(weights, raw.score.to_numpy())),
                }
            )
            if fold != -1:
                index = group.index[group.fold == fold]
                frame.loc[index, "c_cv"] = c
                frame.loc[index, "fit_status"] = fit_status
    raw = interval_metrics(frame, 1.0)
    scaled = interval_metrics(frame, frame.c_cv.to_numpy())
    frame[raw.columns] = scaled
    frame["raw_score"] = raw.score
    frame["raw_width"] = raw.width
    frame["raw_covered"] = raw.covered
    frame["paired_delta"] = frame.score - raw.score
    frame["relative_lo"] = frame.ci_lo - frame.xi_hat
    frame["relative_hi"] = frame.ci_hi - frame.xi_hat
    error = frame.xi_true - frame.xi_hat
    frame["direction_unreachable"] = (
        ((error > 0) & (frame.relative_hi <= 0))
        | ((error < 0) & (frame.relative_lo >= 0))
        | ((error == 0) & ((frame.relative_lo > 0) | (frame.relative_hi < 0)))
    )
    return frame, pd.DataFrame(fits), raw


def summarize(frame, replicates, *, strata=()):
    """Reuse scenario-balanced means, omitting inappropriate iid CV error bars.

    Calibration training folds overlap. The source report's replication-based
    MCSE does not account for that fitted-c dependence and must not be reused
    as an uncertainty estimate for this cross-validation procedure.
    """
    result = balanced_summary(frame, replicates=replicates, strata=strata)
    return result.drop(columns=[c for c in result if c.endswith("_mcse")])


def write_report(detail, fits, raw, source_manifest, out_dir, manifest):
    """Save paired held-out results and separate, non-validated all-data fits."""
    replicates = source_manifest["monte_carlo_reps"]
    summary = summarize(detail, replicates)
    original = detail.copy()
    original[raw.columns] = raw
    original["paired_delta"] = 0.0
    raw_summary = summarize(original, replicates)[KEYS + ["score", "covered", "width"]]
    summary = summary.merge(
        raw_summary.rename(
            columns={
                "score": "raw_score",
                "covered": "raw_covered",
                "width": "raw_width",
            }
        ),
        on=KEYS,
        validate="one_to_one",
    )
    c_folds = fits.loc[fits.held_out_fold >= 0].groupby(KEYS).c.agg(c_min="min", c_max="max")
    summary = summary.merge(c_folds, on=KEYS, validate="one_to_one")
    summary = summary.merge(
        fits.loc[fits.held_out_fold == -1, KEYS + ["c"]].rename(columns={"c": "c_all_data"}),
        on=KEYS,
        validate="one_to_one",
    )
    diagnostics = detail.groupby(KEYS).agg(
        valid_rate=("score", lambda x: np.isfinite(x).mean()),
        original_precision_met_rate=("precision_met", "mean"),
        unreachable_rate=("direction_unreachable", "mean"),
    )
    summary = summary.merge(diagnostics, on=KEYS, validate="one_to_one")
    summary["improvement_percent"] = -100 * summary.paired_delta / summary.raw_score
    summary["rank"] = summary.score.where(summary.valid_rate == 1).groupby(summary.scheme).rank()
    summary = summary.sort_values(["scheme", "rank", "method"])
    tables = {"summary": summary}
    for field in ("fold", "xi_true", "theta_true", "family", "scenario", "flat_window"):
        table = summarize(detail, replicates, strata=(field,))
        base = summarize(original, replicates, strata=(field,))
        table = table.merge(
            base[KEYS + [field, "score", "covered", "width"]].rename(
                columns={
                    "score": "raw_score",
                    "covered": "raw_covered",
                    "width": "raw_width",
                }
            ),
            on=[*KEYS, field],
            validate="one_to_one",
        )
        tables[f"by_{field}"] = table
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    fits.to_csv(out_dir / "calibration_fits.csv", index=False)
    pd.DataFrame({"rep": np.arange(replicates), "fold": assign_folds(replicates)}).to_csv(
        out_dir / "fold_assignment.csv",
        index=False,
    )
    detail.to_csv(out_dir / "held_out_trials.csv.gz", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, scheme in zip(axes, ("disjoint", "sliding"), strict=True):
        group = summary.loc[summary.scheme == scheme]
        for ci, rows in group.groupby(group.method.str.split("/").str[1]):
            ax.scatter(rows.raw_score, rows.score, label=ci.replace("_", " "), s=40, alpha=0.8)
        limits = [
            min(group.raw_score.min(), group.score.min()) * 0.85,
            max(group.raw_score.max(), group.score.max()) * 1.15,
        ]
        ax.plot(limits, limits, "--", color="gray", lw=1)
        ax.set(
            xscale="log",
            yscale="log",
            xlim=limits,
            ylim=limits,
            title=scheme,
            xlabel="Original mean Winkler score",
            ylabel="Held-out scaled mean Winkler score",
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("Median EVI: five-fold CI scale calibration (below diagonal is better)")
    fig.savefig(out_dir / "comparison.png", dpi=160)
    fig.savefig(out_dir / "comparison.pdf")
    plt.close(fig)

    text = f"""# Median EVI: cross-fitted CI scale calibration

## Material Passport

- Origin Skill: ARS-Codex experiment-agent
- Origin Mode: run
- Origin Date: {manifest["started_utc"]}
- Verification Status: ANALYZED (exploratory cross-validation on retained simulations)
- Version Label: evi_ci_scale_calibration_v1

## Design and calibration

Reuse the completed N=365, 84-scenario, M=100 exploration: 98 combinations in total.
Each complete simulated series belongs to one of five balanced folds, with fold seed
{FOLD_SEED}. Replicate indices share the same fold across all scenarios, schemes and methods.
For every combination and held-out fold, learn one constant c>0 on the other 80 replicates
per scenario. Minimize the equal-scenario mean raw 95% Winkler score. Evaluate on the 20
held-out replicates per scenario. Every original series is evaluated exactly once out of fold.
No new simulation, bootstrap, window selection or covariance fit is performed.

For original point estimate x and endpoints [lo,hi], use [x+c*(lo-x), x+c*(hi-x)].
This scales offsets about x; for asymmetric intervals it changes both width and midpoint.
It is not midpoint-preserving width scaling. The original point estimate itself is unchanged.
Coverage is descriptive, not a calibration target or selection gate.

With a=lo-x, b=hi-x and t=true_xi-x, the loss is
S(c)=c*(b-a)+40*max(c*a-t,0)+40*max(t-c*b,0).
The convex piecewise-linear objective is minimized at a derivative breakpoint, with no
arbitrary c grid or maximum. Flat minima choose the c closest to 1. A zero-boundary infimum
has no admissible positive minimizer and is reported as unavailable, never replaced by an
arbitrary epsilon. Failed fits remain visible; only complete candidates receive ranks.

## Outputs and interpretation

summary.csv reports held-out score, coverage and width, paired against c=1 for the SAME
combination on the SAME series. paired_delta<0 favors scaling. c_min/c_max describe the five
training-fold fits. c_all_data is fitted separately on all 100 replicates per scenario and
is an experimental candidate only; it is NEVER used in held-out scoring.
calibration_fits.csv includes training scores; these are not validation performance.
by_fold.csv reports fold-specific paired results. Folds overlap in their training sets:
neither their spread nor the source report's replicate-clustered MCSE is a confidence interval for
the complete fitted-c procedure. No such MCSE or simultaneous 98-method inference is claimed.

This is one five-fold partition, not repeated cross-validation or a new independent-seed
experiment. The dataset was already inspected during earlier exploration; evaluating all
98 candidates does not remove that broader exploratory selection. Selecting the best row
using these same held-out scores is still exploratory, not an unbiased assessment of that
selection policy. It does not establish transfer to other N, data generators or real data.

All original grid, selector, covariance, CI and adaptive R<=1024 settings are inherited.
Unmet original bootstrap precision is retained; no precision claim is made for scaled
endpoints because only endpoints, not delete-group draws, were retained in this postprocess.
Flat windows and directionally unreachable truths remain in the score. No default changes.

## Reproduce and provenance

{manifest["command"]}

Source manifest and all 84 trial files are SHA-256 bound in manifest.json. CSV floating
values are read with round-trip precision, important for endpoints equal to the anchor.
The original report is at ../evi_ci_exploration/report.html. Source simulation master seed:
{source_manifest["master_seed"]}; source bootstrap seed: original replicate index.

The scalar-error idea is motivated by Buecher-Staud, https://arxiv.org/html/2409.05529v2#S6.
Our c>0 score objective and cross-validation differ from their coverage-based enlargement;
their empirical coefficients and theoretical guarantees are not imported.
"""
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    display = summary[
        KEYS
        + [
            "raw_score",
            "score",
            "paired_delta",
            "improvement_percent",
            "raw_covered",
            "covered",
            "raw_width",
            "width",
            "c_min",
            "c_max",
            "c_all_data",
            "valid_rate",
        ]
    ]
    table = display.to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0)
    sections = "".join(
        f'<details><summary>{escape(name)}</summary><div class="scroll">'
        + tables[name].to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0)
        + "</div></details>"
        for name in ("by_fold", "by_theta_true", "by_xi_true", "by_family", "by_flat_window")
    )
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>EVI CI scale calibration</title><style>
body{{font:16px/1.65 system-ui;max-width:1450px;margin:30px auto;padding:0 24px;color:#172330}}
table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}}
th{{position:sticky;top:0;background:#edf3f8}}td:nth-child(2){{text-align:left}}
.scroll{{max-height:650px;overflow:auto}}img{{max-width:100%}}details{{margin:22px 0}}summary{{cursor:pointer;font-weight:bold}}
pre{{white-space:pre-wrap;background:#f4f6f8;padding:20px}}input{{padding:9px;width:360px}}</style>
<h1>Median EVI：CI 尺度优化的 5-fold cross-validation</h1>
<p>N=365 · 84 场景 · 每场景现有 100 条模拟 · 全部 98 组合 · 未新增模拟或 bootstrap</p>
<p>每个组合用 80 条／场景学习 c，在其余 20 条上评价；轮换五折。点估计不变，允许 c&gt;0 收窄或放大。
主表 score 全部来自留出 fold；paired_delta&lt;0 表示优于同一组合的原始 c=1。</p>
<p>c_min/c_max 为五折系数范围。c_all_data 为全数据拟合的实验候选，未参与主表计分。
这是事后探索，不是通用校准保证；未更改包的默认 CI。Coverage 以 0–1 表示。</p>
<p><a href="summary.csv">配对总表 CSV</a> · <a href="calibration_fits.csv">系数与训练成绩</a> ·
<a href="fold_assignment.csv">fold 分配</a> · <a href="held_out_trials.csv.gz">逐样本留出结果</a> ·
<a href="manifest.json">运行记录</a> · <a href="comparison.pdf">图 PDF</a> ·
<a href="../evi_ci_exploration/report.html">原始探索报告</a></p>
<img src="comparison.png" alt="各组合原始与留出尺度优化 score；对角线以下表示改善">
<h2>全部 98 个组合</h2><input aria-label="筛选组合" placeholder="筛选：如 sliding、basic、oas"
oninput="document.querySelectorAll('#results tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase()))">
<div id="results" class="scroll">{table}</div>{sections}
<details><summary>方法、限制与复现</summary><pre>{escape(text)}</pre></details></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")
    return summary


def main():
    """Validate retained inputs, cross-fit without resampling and save provenance."""
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "out/benchmark/evi_ci_exploration")
    parser.add_argument(
        "--out", type=Path, default=root / "out/benchmark/evi_ci_scale_calibration"
    )
    args = parser.parse_args()
    out_dir = args.out.resolve()
    if not any(out_dir.is_relative_to(root / folder) for folder in ("out", ".cache")):
        parser.error("Output must remain within this repository's out/ or .cache/.")
    if out_dir.exists() and any(out_dir.iterdir()):
        parser.error("Use an empty output directory to preserve earlier results.")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    source_manifest_path = args.source / "manifest.json"
    manifest_bytes = source_manifest_path.read_bytes()
    source = json.loads(manifest_bytes)
    if source["status"] != "completed":
        parser.error("The source exploration must be complete.")
    replicas = source["monte_carlo_reps"]
    assign_folds(replicas)
    frames, hashes = [], {}
    for path in sorted((args.source / "trials").glob("*.csv.gz")):
        data = path.read_bytes()
        hashes[path.name] = hashlib.sha256(data).hexdigest()
        frames.append(
            pd.read_csv(
                io.BytesIO(data), compression="gzip", usecols=COLUMNS, float_precision="round_trip"
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    if (
        len(frame) != source["rows"]
        or len(hashes) != source["scenarios"]
        or frame.scenario.nunique() != source["scenarios"]
        or frame.duplicated(["scenario", "rep", *KEYS]).any()
        or not frame.valid.all()
        or not np.isfinite(frame[["xi_hat", "xi_true", "ci_lo", "ci_hi"]]).all().all()
        or not (frame.ci_lo <= frame.ci_hi).all()
        or not frame.n_obs.eq(365).all()
        or set(frame.rep) != set(range(replicas))
        or not frame.groupby(["scenario", *KEYS]).size().eq(replicas).all()
        or not frame.groupby(["scenario", "rep", "scheme"])
        .size()
        .eq(source["candidate_count_per_scheme"])
        .all()
    ):
        parser.error("Expected complete, finite, uniquely paired N=365 exploration trials.")
    for name in ("scenario", "scheme", "method", "family"):
        frame[name] = frame[name].astype("category")
    detail, fits, raw = cross_fit(frame, replicas)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_utc": started_utc,
        "status": "running",
        "command": "PYTHONPATH=scripts:src "
        + shlex.join(
            [
                sys.executable,
                "-m",
                "benchmark.evi_ci_scale_calibration",
                *sys.argv[1:],
            ]
        ),
        "source_directory": str(args.source.resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_trials_sha256": hashes,
        "rows": len(frame),
        "fold_seed": FOLD_SEED,
        "folds": FOLDS,
        "scale_domain": "c>0",
        "objective": "scenario-balanced mean Winkler",
        "new_simulations": 0,
        "new_bootstrap_draws": 0,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("evi_ci_exploration_report.py"))
        },
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    summary = write_report(detail, fits, raw, source, out_dir, manifest)
    manifest.update(
        status="completed",
        elapsed_s=time.perf_counter() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        calibration_status_counts=fits.status.value_counts().to_dict(),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        summary.groupby("scheme", observed=True)
        .head(5)[
            KEYS
            + [
                "raw_score",
                "score",
                "paired_delta",
                "covered",
                "c_min",
                "c_max",
            ]
        ]
        .to_string(index=False)
    )
    print(f"Report: {out_dir / 'report.html'}; {manifest['elapsed_s']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
