"""Repository-local workflow modules for EVI/EI benchmark and application entrypoints.

The workflow package is intentionally small:

- ``evi_benchmark.py`` is the runnable EVI benchmark computation entrypoint;
- ``evi_benchmark_external.py`` adds published xi baselines and mixed comparisons;
- ``evi_report.py`` aggregates EVI benchmark outputs and writes manuscript artifacts;
- ``ei_benchmark.py`` is the runnable EI benchmark computation entrypoint;
- ``ei_benchmark_eval.py`` evaluates EI methods per replicate;
- ``ei_report.py`` aggregates EI benchmark outputs and writes manuscript artifacts;
- ``benchmark_design.py`` defines the simulation and method grids;
- ``benchmark_common.py`` provides shared scoring and table helpers;
- ``application.py`` is the runnable application entrypoint;
- ``application_screening.py`` contains application-side screening helpers.
"""

__all__ = [
    "application",
    "application_screening",
    "benchmark_common",
    "benchmark_design",
    "ei_benchmark",
    "ei_benchmark_eval",
    "ei_report",
    "evi_benchmark",
    "evi_benchmark_external",
    "evi_report",
]
