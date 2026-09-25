"""Scenario-balanced summaries for the paired covariance/CI exploration."""

from __future__ import annotations

from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


METRICS = ("score", "width", "covered", "lower_penalty", "upper_penalty", "bias", "paired_delta")


def balanced_summary(frame, *, strata=(), replicates):
    """Average scenario means and cluster MCSE by shared bootstrap-seed index.

    Repeated scenario means get equal weight. Replicate indices share bootstrap
    RNG streams across scenarios, so uncertainty uses replicate-clustered
    influence contributions, not an iid standard error over all pooled rows.
    Missing values describe the stated valid-case cohort, never zero losses.
    """
    keys = ["scheme", "method", *strata]
    output = []
    for identity, group in frame.groupby(keys, observed=True, dropna=False, sort=True):
        codes, scenarios = pd.factorize(group.scenario, sort=True)
        values = np.full((replicates, len(scenarios), len(METRICS)), np.nan)
        values[group.rep.to_numpy(int), codes] = group[list(METRICS)].to_numpy(float)
        counts = np.isfinite(values).sum(axis=0)
        scenario_means = np.divide(
            np.nansum(values, axis=0),
            counts,
            out=np.full_like(counts, np.nan, dtype=float),
            where=counts > 0,
        )
        n_scenarios = np.isfinite(scenario_means).sum(axis=0)
        means = np.divide(
            np.nansum(scenario_means, axis=0),
            n_scenarios,
            out=np.full(len(METRICS), np.nan),
            where=n_scenarios > 0,
        )
        influence = np.divide(
            values - scenario_means, counts, out=np.zeros_like(values), where=counts > 0
        )
        cluster = np.nansum(influence, axis=1)
        mcse = np.full(len(METRICS), np.nan)
        if replicates > 1:
            np.divide(
                np.sqrt(replicates / (replicates - 1) * np.sum(cluster**2, axis=0)),
                n_scenarios,
                out=mcse,
                where=n_scenarios > 0,
            )
        row = dict(zip(keys, identity, strict=True))
        row.update(zip(METRICS, means, strict=True))
        row.update({f"{name}_mcse": value for name, value in zip(METRICS, mcse, strict=True)})
        row.update(
            n_scenarios=int(n_scenarios[0]), n_valid=int(np.isfinite(values[:, :, 0]).sum())
        )
        output.append(row)
    return pd.DataFrame(output)


def score_matrices(summary, out_dir, covariances, intervals):
    """Draw complete candidate matrices; gray cells are unavailable combinations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    positive = summary.loc[summary.score > 0, "score"]
    score_norm = (
        LogNorm(vmin=float(positive.min()), vmax=float(positive.max())) if len(positive) else None
    )
    for column, scheme in enumerate(("disjoint", "sliding")):
        subset = summary.loc[summary.scheme == scheme].copy()
        subset[["covariance", "ci"]] = subset.method.str.split("/", expand=True)
        for row, metric in enumerate(("score", "covered")):
            matrix = (
                subset.pivot(index="covariance", columns="ci", values=metric)
                .reindex(
                    index=covariances,
                    columns=intervals,
                )
                .to_numpy(float)
            )
            ax = axes[row, column]
            cmap = plt.get_cmap("viridis_r" if metric == "score" else "Blues").copy()
            cmap.set_bad("#e8e8e8")
            image = ax.imshow(
                matrix,
                cmap=cmap,
                aspect="auto",
                norm=score_norm if row == 0 else None,
                **({} if row == 0 else {"vmin": 0, "vmax": 1}),
            )
            ax.set_xticks(
                range(len(intervals)),
                [s.replace("_", " ") for s in intervals],
                rotation=25,
                ha="right",
            )
            ax.set_yticks(range(len(covariances)), covariances)
            ax.set_title(
                f"{scheme}: {'mean Winkler score (lower is better)' if row == 0 else 'coverage (nominal 95%)'}"
            )
            for i, j in np.ndindex(matrix.shape):
                value = matrix[i, j]
                label = (
                    "NA"
                    if not np.isfinite(value)
                    else (f"{value:.3g}" if row == 0 else f"{value:.1%}")
                )
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1},
                )
            fig.colorbar(image, ax=ax, shrink=0.65)
    fig.suptitle("Median EVI: paired covariance / confidence-interval exploration", fontsize=17)
    fig.savefig(out_dir / "comparison.png", dpi=160)
    fig.savefig(out_dir / "comparison.pdf")
    plt.close(fig)


def write_report(detail, out_dir: Path, manifest, covariances, intervals, baseline):
    """Keep own-valid and common-valid evidence, failure rates and paired deltas."""
    keys = ["scenario", "rep", "scheme"]
    detail = detail.copy()
    reference = detail.loc[detail.method == baseline, keys + ["score"]].rename(
        columns={"score": "baseline_score"}
    )
    detail = detail.merge(reference, on=keys, how="left", validate="many_to_one")
    detail["paired_delta"] = detail.score - detail.baseline_score
    detail["common_valid"] = detail.groupby(keys, observed=True).valid.transform("all")
    common = detail.copy()
    common.loc[~common.common_valid, list(METRICS)] = np.nan
    replicas = manifest["monte_carlo_reps"]
    summary = balanced_summary(common, replicates=replicas)
    diagnostics = (
        detail.groupby(["scheme", "method"], observed=True)
        .agg(
            valid_rate=("valid", "mean"),
            precision_met_rate=("precision_met", "mean"),
            mean_R=("bootstrap_reps", "mean"),
            cap_rate=("bootstrap_reps", lambda v: np.mean(v == 1024)),
            flat_window_rate=("flat_window", "mean"),
            near_zero_rate=("near_zero_slope", "mean"),
            mean_shrinkage=("shrinkage", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(diagnostics, on=["scheme", "method"], validate="one_to_one")
    summary["rank"] = summary.groupby("scheme").score.rank(method="min")
    summary = summary.sort_values(["scheme", "score", "method"])
    tables = {"summary": summary, "own_valid": balanced_summary(detail, replicates=replicas)}
    for field in ("xi_true", "theta_true", "family", "scenario", "flat_window"):
        tables[f"by_{field}"] = balanced_summary(common, strata=(field,), replicates=replicas)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    # Each covariance has several CIs but only one observed point estimate.
    # Deduplicate before estimating its between-simulation variance.
    calibration = common.drop_duplicates([*keys, "covariance"]).copy()
    calibration["model_variance"] = calibration.model_se**2
    calibration["propagated_variance"] = calibration.bootstrap_se**2
    calibration["squared_error"] = calibration.bias**2
    variance_columns = ["xi_hat", "bias", "model_variance", "propagated_variance", "squared_error"]
    calibration.loc[~calibration.common_valid, variance_columns] = np.nan
    calibration = (
        calibration.groupby(
            ["scenario", "family", "xi_true", "theta_true", "scheme", "covariance"],
            observed=True,
        )
        .agg(
            n_valid=("xi_hat", "count"),
            empirical_variance=("xi_hat", "var"),
            mean_bias=("bias", "mean"),
            mean_squared_error=("squared_error", "mean"),
            mean_model_variance=("model_variance", "mean"),
            mean_propagated_variance=("propagated_variance", "mean"),
        )
        .reset_index()
    )
    calibration["rmse"] = np.sqrt(calibration.mean_squared_error)
    denominator = calibration.empirical_variance.where(calibration.empirical_variance > 0)
    calibration["model_variance_ratio"] = calibration.mean_model_variance / denominator
    calibration["propagated_variance_ratio"] = calibration.mean_propagated_variance / denominator
    calibration.to_csv(out_dir / "variance_calibration.csv", index=False)
    score_matrices(summary, out_dir, covariances, intervals)
    sample_diagnostics = detail.drop_duplicates(keys)
    sample_diagnostics[
        [
            *keys,
            "common_valid",
            "all_precision_met",
            "bootstrap_reps",
            "flat_window",
            "window_lo",
            "window_hi",
            "window_points",
            "segments",
            "elapsed_s",
            "warning",
        ]
    ].to_csv(
        out_dir / "sample_diagnostics.csv",
        index=False,
    )
    n_common = int(sample_diagnostics.common_valid.sum())
    n_total = len(sample_diagnostics)
    text = f"""# Median EVI covariance / CI exploration

## Material Passport

- Origin Skill: ARS-Codex experiment-agent
- Origin Mode: run
- Origin Date: {manifest["started_utc"]}
- Verification Status: ANALYZED (independent validation deferred)
- Version Label: evi_ci_exploration_v1

## Frozen design

N=365; {manifest["scenarios"]} scenarios; M={replicas} per scenario; median sliding/disjoint;
49 candidates per scheme (nine covariance configurations × five CIs, plus four OLS CIs).
Master seed {manifest["master_seed"]}; scenario seeds use the existing stable scenario hash;
bootstrap seed is the replicate index. See manifest.json and per-scenario trials for exact inputs.

b_min=max(5,ceil(N^(1/3))); b_max=min(floor(N^(1-1/e)),floor(N/17));
the untrimmed default grid and existing selector are reused; L=max(2B,floor(sqrt(N))).
At N=365 the grid is {manifest["block_grid"]}, L={manifest["superblock_length"]} with
{manifest["segments"]} complete segments. Selection is frozen within each observed sample.

Adaptive checkpoints: 128,256,512,768,1024. Two partitions into eight delete-groups monitor
each candidate's xi and actual CI endpoints with MCSE/statistical-SE <=0.10. All candidates
within each sample/scheme share the final R. Unmet precision is retained, not discarded.
Weights and automatic shrinkages are recomputed in each delete-group diagnostic; they remain
fixed across the bootstrap replicates defining a candidate interval. This is an internal
Monte Carlo diagnostic, not an anytime-valid guarantee or a coverage guarantee.

## Covariance and interval definitions

Fixed diagonal shrinkages: 0,0.15,0.37,0.55,0.75,1. Schaefer-Strimmer estimates correlation
shrinkage and preserves the sample variances. Ledoit-Wolf and OAS shrink toward mu I.
OAS retains the original 2/p correction; it is Gaussian-derived. All candidates use the same
unbiased sample covariance S and existing numerical ridge max(mean(diag(S))*1e-8,1e-12).
LW intensity is computed on centered ML covariance, then applied to S for common normalization.

Let a be the slope row of A=(X'WX)^-1 X'W, xi=a y_obs, and d*=a(y* - y_center).
The center is the unresampled segment-maxima bank matching the bootstrap construction:
circular within segments for sliding; complete within-segment blocks for disjoint.
Path quantiles use median_unbiased; CI error quantiles use linear interpolation.

- model_wald: xi ± 1.96 sqrt([(X'WX)^-1]_slope,slope).
- propagated_wald: xi ± 1.96 sqrt(a S a').
- bias_normal: xi - mean(d*) ± 1.96 sd(d*).
- basic: [xi-q_.975(d*), xi-q_.025(d*)].
- aligned_percentile: [xi+q_.025(d*), xi+q_.975(d*)].
- OLS sets W=I and omits model_wald; its other four intervals use the same window/draws.

For fixed a, sd(d*)^2 equals a S a' exactly up to floating-point arithmetic. Basic and
aligned percentile coincide when the error quantiles are exactly symmetric. The unshrunk
model/propagated Wald formulas coincide without a ridge and with exact inverse identities;
the retained ridge can make them differ. Such equivalences are not independent evidence.

## Reading the evidence

Lower mean Winkler score is better. Scenario means have equal weight; raw scores are not
rescaled by xi. Coverage is reported separately, with width and both miss penalties.
The primary matrix uses samples on which all 49 candidates in that scheme return finite,
ordered intervals: {n_common}/{n_total} sample/scheme pairs. own_valid.csv retains each method's
own valid cases; valid_rate always uses all attempted samples. Missing scores are never zero.
Candidate-specific missingness limits interpretation of a complete-case ranking.

variance_calibration.csv compares mean estimated model/propagated variances with the
empirical variance of xi_hat across simulations, separately for each scenario and covariance.
Ratios near one indicate variance agreement; bias and RMSE are reported separately because
variance calibration alone does not establish coverage. The M=100 variance estimates are noisy.

paired_delta is candidate score minus the study reference diagonal_0.37/model_wald on the same samples;
negative favors the candidate. MCSE is clustered by replicate index because bootstrap seeds
are shared across scenarios. Reported Monte Carlo uncertainties are descriptive and are not
adjusted simultaneous intervals for the 98 comparisons. No method is promoted automatically.

Flat observed windows and abs(xi_hat)<1e-10 remain in the analysis. Exact flat-window strata
are in by_flat_window.csv; near_zero_rate is reported for every candidate. Repeated bootstrap
slopes and zero SE can make percentile precision diagnostics unreliable; raw unique-slope
counts and the precision flags are retained. No claim of quantile diagnostic calibration is made.
The intervals hold the selected window and weights fixed within bootstrap draws, so do not
explicitly propagate selection uncertainty. The observed-sample Monte Carlo coverage still
evaluates the complete procedure, including repeated window selection between samples.

## Reproduce

```sh
{manifest["command"]}
```

Runtime: {manifest["elapsed_s"]:.2f} s; workers: {manifest["workers"]}; inner numerical threads: 1;
native extensions: {manifest["native_extensions"]}. Source hashes and versions are in manifest.json.
An independent-seed validation and N=4096 study were deferred by the user.

## Sources

- Schaefer-Strimmer (2005), Eq. 10 and Appendix A: https://strimmerlab.github.io/publications/journals/shrinkcov2005.pdf
- Chen et al. (2010), Eqs. 13 and 23: https://arxiv.org/abs/0907.4698
- Buecher-Staud, circular-bootstrap centering: https://arxiv.org/html/2409.05529v2#S5
- Efron-Narasimhan, internal endpoint MCSE: https://pmc.ncbi.nlm.nih.gov/articles/PMC7958418/

These references motivate components; they do not establish validity of the present selected,
multiscale FGLS pipeline or guarantee superior interval scores.
"""
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    columns = [
        "scheme",
        "method",
        "score",
        "score_mcse",
        "paired_delta",
        "paired_delta_mcse",
        "covered",
        "width",
        "valid_rate",
        "precision_met_rate",
        "near_zero_rate",
    ]
    table_html = summary[columns].to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0)
    sections = []
    for name in ("by_xi_true", "by_theta_true", "by_family", "by_flat_window"):
        keep = [
            c
            for c in tables[name].columns
            if not c.endswith("_mcse") or c in ("score_mcse", "paired_delta_mcse")
        ]
        sections.append(
            f'<details><summary>{escape(name)}</summary><div class="scroll">'
            + tables[name][keep].to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0)
            + "</div></details>"
        )
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>EVI covariance / CI exploration</title>
<style>body{{font:16px/1.6 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#172330}}h1,h2{{line-height:1.25}}
table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{padding:7px 9px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}}
th{{position:sticky;top:0;background:#edf3f8}}td:nth-child(2){{text-align:left}}.scroll{{max-height:650px;overflow:auto}}img{{max-width:100%}}
details{{margin:22px 0}}summary{{cursor:pointer;font-weight:bold}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:20px}}input{{padding:9px;width:350px}}</style>
<h1>Median EVI：covariance 与 CI 全量探索</h1>
<p>N=365 · 84 场景 · 每场景 {replicas} 次模拟 · sliding/disjoint 各 49 个组合 · adaptive R≤1024</p>
<p>这是探索结果。主表使用同一种 scheme 下全部候选均成功的共同样本：{n_common}/{n_total}。
较低 score 更好；paired_delta&lt;0 表示优于本实验的 shrinkage 0.37 + Wald 对照。未达标的 bootstrap 精度结果保留。</p>
<p><a href="summary.csv">总表 CSV</a> · <a href="own_valid.csv">各方法自身有效样本</a> · <a href="by_scenario.csv">完整场景表</a> ·
<a href="sample_diagnostics.csv">选窗与精度诊断</a> · <a href="variance_calibration.csv">方差校准与 bias / RMSE</a> · <a href="manifest.json">运行记录</a> · <a href="comparison.pdf">图 PDF</a></p>
<img src="comparison.png" alt="两种 block scheme 的 score 和 coverage 全组合矩阵">
<h2>全部 98 个组合</h2><p>coverage、有效率、精度达标率以 0–1 表示。MCSE 为模拟误差，不是参数的 CI。</p>
<input placeholder="筛选：如 sliding、basic、oas" oninput="document.querySelectorAll('#results tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase()))">
<div id="results" class="scroll">{table_html}</div>{"".join(sections)}
<details><summary>方法、限制与复现命令</summary><pre>{escape(text)}</pre></details></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")
    return summary
