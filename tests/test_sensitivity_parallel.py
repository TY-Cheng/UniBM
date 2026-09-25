import pandas as pd
import pytest

from benchmark.design import default_ei_simulation_configs, default_evi_simulation_configs
from benchmark.ei_report import build_ei_shrinkage_sensitivity_summary
from benchmark.evi_report import build_evi_shrinkage_sensitivity_summary


@pytest.mark.parametrize(
    ("build", "make_configs"),
    [
        (build_evi_shrinkage_sensitivity_summary, default_evi_simulation_configs),
        (build_ei_shrinkage_sensitivity_summary, default_ei_simulation_configs),
    ],
)
def test_sensitivity_parallel_matches_serial_exactly(tmp_path, build, make_configs):
    """Scheduling must not alter scenario seeds, adaptive stopping, or aggregation."""
    configs = make_configs(xi_values=(0.5,), theta_values=(0.5,), n_obs=256, reps=1)
    serial, _ = build(tmp_path, configs=configs, max_workers=1, force=True)
    parallel, _ = build(tmp_path, configs=configs, max_workers=2, force=True)
    pd.testing.assert_frame_equal(parallel, serial, check_exact=True)
