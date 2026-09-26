import pytest

from clean_generated import clean_generated


def test_cleanup_removes_only_known_outputs_and_preserves_cache_and_notes(tmp_path, monkeypatch):
    code = tmp_path / "code"
    report = tmp_path / "deliverables"
    monkeypatch.setenv("UNIBM_REPORT_DIR", str(report))
    monkeypatch.delenv("UNIBM_BENCHMARK_N_OBS", raising=False)
    generated = [
        code / "out/benchmark/evi_summary.csv",
        code / "out/applications/application_summary.json",
        code / "out/applications/report.html",
        code / "out/applications/cases/goes_normalized.json",
        code / "out/applications/cases/houston_precipitation.png",
        report / "Figure/benchmark_evi_summary.pdf",
        report / "Figure/application_composite_tx_streamflow.pdf",
        report / "Table/application_summary.tex",
        report / "report_subset_manifest.json",
    ]
    retained = [
        code / "out/benchmark/cache/series/sample.npz",
        code / "out/benchmark/evi_summary_n730.csv",
        code / "out/benchmark/report/figures/historical.pdf",
        code / "out/research/notes.md",
        code / "out/archive/completed.zip",
        code / "out/applications/cases/manual.png",
        code / "docs/assets/cases/goes_normalized.json",
        code / "out/applications/manual.csv",
        report / "Figure/manual.pdf",
        report / "Table/manual.tex",
        report / "article.tex",
    ]
    for path in generated + retained:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sentinel")
    assert set(clean_generated(code)) == set(generated)
    assert all(not path.exists() for path in generated)
    assert all(path.read_text() == "sentinel" for path in retained)
    assert clean_generated(code) == []


def test_cleanup_rejects_linked_output_directory_before_deleting(tmp_path, monkeypatch):
    code = tmp_path / "code"
    external = tmp_path / "external"
    external.mkdir()
    outside = external / "evi_summary.csv"
    outside.write_text("unrelated")
    (code / "out").mkdir(parents=True)
    (code / "out/benchmark").symlink_to(external, target_is_directory=True)
    monkeypatch.setenv("UNIBM_REPORT_DIR", "")
    with pytest.raises(ValueError, match="symlink"):
        clean_generated(code)
    assert outside.read_text() == "unrelated"


@pytest.mark.parametrize("linked_component", ["out", "out/reports"])
def test_default_report_destination_never_follows_links(tmp_path, monkeypatch, linked_component):
    code = tmp_path / "code"
    external = tmp_path / "external"
    external.mkdir()
    link = code / linked_component
    link.parent.mkdir(parents=True)
    link.symlink_to(external, target_is_directory=True)
    outside = (code / "out/reports/Figure/benchmark_evi_summary.pdf").resolve()
    outside.parent.mkdir(parents=True)
    outside.write_text("unrelated")
    monkeypatch.setenv("UNIBM_REPORT_DIR", "")
    with pytest.raises(ValueError, match="symlink"):
        clean_generated(code)
    assert outside.read_text() == "unrelated"


def test_cleanup_respects_each_benchmark_sample_size(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIBM_REPORT_DIR", "")
    monkeypatch.setenv("UNIBM_BENCHMARK_N_OBS", "730")
    benchmark = tmp_path / "out/benchmark"
    benchmark.mkdir(parents=True)
    for name in (
        "evi_summary.csv",
        "evi_summary_n730.csv",
        "ei_summary.csv",
        "ei_summary_n730.csv",
    ):
        (benchmark / name).write_text("sentinel")
    assert set(clean_generated(tmp_path)) == {
        benchmark / "evi_summary_n730.csv",
        benchmark / "ei_summary.csv",
    }
    assert (benchmark / "evi_summary.csv").read_text() == "sentinel"
    assert (benchmark / "ei_summary_n730.csv").read_text() == "sentinel"
