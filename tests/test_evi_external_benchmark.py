from __future__ import annotations
# ruff: noqa: E402

import unittest

try:
    from . import _path_setup as test_paths
except ImportError:  # pragma: no cover
    import _path_setup as test_paths

test_paths.ensure_repo_import_paths()

from benchmark.design import default_evi_simulation_configs
from benchmark.evi_external import evaluate_external_config, run_external_benchmark


class ExternalEviBenchmarkTests(unittest.TestCase):
    def test_evaluate_external_config_rejects_bootstrap_ci_mode(self) -> None:
        cfg = default_evi_simulation_configs(
            xi_values=(0.10,),
            theta_values=(0.50,),
            families=("frechet_max_ar",),
            reps=1,
            n_obs=64,
        )[0]

        with self.assertRaisesRegex(NotImplementedError, "ci_method='bootstrap'"):
            evaluate_external_config(cfg, ci_method="bootstrap", random_state=0, cache_dir=None)

    def test_run_external_benchmark_rejects_bootstrap_ci_mode(self) -> None:
        configs = default_evi_simulation_configs(
            xi_values=(0.10,),
            theta_values=(0.50,),
            families=("frechet_max_ar",),
            reps=1,
            n_obs=64,
        )

        with self.assertRaisesRegex(NotImplementedError, "ci_method='bootstrap'"):
            run_external_benchmark(
                random_state=0,
                configs=configs,
                ci_method="bootstrap",
                cache_dir=None,
                max_workers=1,
            )


if __name__ == "__main__":
    unittest.main()
