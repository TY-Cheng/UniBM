"""Check paired sensitivity against public estimators and their actual samplers."""

from dataclasses import replace

import numpy as np
import pandas as pd

from application.shrinkage_sensitivity import DELTAS, run_case
from application.specs import APPLICATIONS, ApplicationPreparedInputs
from data_prep.ghcn import PreparedSeries
from unibm.ei import bootstrap_bm_ei_path, estimate_pooled_bm_ei, prepare_ei_bundle
from unibm.evi import (
    estimate_design_life_level,
    estimate_design_life_level_interval,
    estimate_evi_quantile,
)


def test_prime_grid_and_shared_bootstrap_match_public_api(tmp_path):
    assert DELTAS == tuple(
        p / 100
        for p in (
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
            73,
            79,
            83,
            89,
            97,
        )
    )
    values = np.exp(np.random.default_rng(29).normal(size=512))
    series = pd.Series(values, index=pd.date_range("2000-01-01", periods=len(values)))
    prepared = PreparedSeries("synthetic", "value", series, series.resample("YE").max(), {})
    inputs = ApplicationPreparedInputs(prepared, prepared, prepared)
    spec = replace(
        next(s for s in APPLICATIONS if s.key == "tx_streamflow"), observations_per_year=365.25
    )
    rows, designs, audit, _ = run_case((spec, inputs, tmp_path))
    detail, design = pd.DataFrame(rows), pd.DataFrame(designs)
    assert len(detail) == 75 and len(design) == 100 and len(audit) == 3
    assert detail.groupby("method").covariance_sha256.nunique().eq(1).all()
    assert detail.groupby("method").window_min.nunique().eq(1).all()
    assert detail.groupby("method").window_max.nunique().eq(1).all()
    assert detail.groupby("method").bootstrap_reps.nunique().eq(1).all()
    bundle = prepare_ei_bundle(
        values, allow_zeros=False, path_keys=(("bb", True), ("northrop", True))
    )
    for method, group in detail.groupby("method"):
        reps = int(group.bootstrap_reps.iloc[0])
        if method != "evi":
            boot = bootstrap_bm_ei_path(
                values,
                base_path=method,
                sliding=True,
                block_sizes=bundle.block_sizes,
                allow_zeros=False,
                reps=reps,
                random_state=7,
                n_threads=1,
            )
        for delta in (0.02, 0.37, 0.97):
            row = group[group.delta == delta].iloc[0]
            if method == "evi":
                fit = estimate_evi_quantile(
                    values,
                    regression="FGLS",
                    bootstrap_reps=reps,
                    random_state=7,
                    covariance_shrinkage=delta,
                    n_threads=1,
                )
                point = fit.slope
                horizons = design[design.delta == delta].years.to_numpy()
                levels = estimate_design_life_level(fit, horizons, observations_per_year=365.25)
                lo, hi = estimate_design_life_level_interval(
                    fit, horizons, observations_per_year=365.25
                )
                np.testing.assert_allclose(
                    design[design.delta == delta][["estimate", "ci_lo", "ci_hi"]],
                    np.column_stack((levels, lo, hi)),
                )
            else:
                fit = estimate_pooled_bm_ei(
                    bundle,
                    base_path=method,
                    sliding=True,
                    regression="FGLS",
                    bootstrap_result=boot,
                    covariance_shrinkage=delta,
                )
                point = fit.theta_hat
            np.testing.assert_allclose(
                [row.estimate, row.ci_lo, row.ci_hi],
                [point, *fit.confidence_interval],
                rtol=1e-9,
                atol=1e-10,
            )
