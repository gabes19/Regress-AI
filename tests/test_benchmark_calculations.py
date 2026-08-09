from benchmarks.benchmark_cpu_gpu import aggregate, describe_speed, update_readme_table


def test_speedup_reports_faster_and_slower_without_negative_speedup():
    speedup, improvement, label = describe_speed(10, 5)
    assert speedup == 2
    assert improvement == 50
    assert label == "50.0% faster"

    speedup, improvement, label = describe_speed(5, 10)
    assert speedup == .5
    assert improvement == -100
    assert label == "100.0% slower"


def test_benchmark_aggregation_uses_medians_and_excludes_cold_from_warm_speedup():
    profile = ("Test", 100, 2, 500, 2)
    records = [
        {"cold": True, "cpu_seconds": 100, "gpu_worker_seconds": 50,
         "gpu_end_to_end_seconds": 20, "gpu_cost_usd": "0.001", "parity_passed": True},
        {"cold": False, "cpu_seconds": 10, "gpu_worker_seconds": 2,
         "gpu_end_to_end_seconds": 4, "gpu_cost_usd": "0.001", "parity_passed": True},
        {"cold": False, "cpu_seconds": 12, "gpu_worker_seconds": 3,
         "gpu_end_to_end_seconds": 6, "gpu_cost_usd": "0.002", "parity_passed": True},
    ]
    result = aggregate(profile, records)
    assert result["cpu_median_seconds"] == 11
    assert result["gpu_warm_e2e_median_seconds"] == 5
    assert result["gpu_cold_e2e_seconds"] == 20
    assert result["end_to_end_speedup"] == 2.2


def test_readme_table_renderer_replaces_only_marked_section(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text(
        "Before\n<!-- BENCHMARK_RESULTS_START -->\nold\n<!-- BENCHMARK_RESULTS_END -->\nAfter",
        encoding="utf-8",
    )
    monkeypatch.setattr("benchmarks.benchmark_cpu_gpu.PROJECT_ROOT", tmp_path)
    item = aggregate(("Test", 100, 2, 500, 2), [
        {"cold": False, "cpu_seconds": 10, "gpu_worker_seconds": 2,
         "gpu_end_to_end_seconds": 4, "gpu_cost_usd": "0.001", "parity_passed": True},
        {"cold": False, "cpu_seconds": 12, "gpu_worker_seconds": 3,
         "gpu_end_to_end_seconds": 6, "gpu_cost_usd": "0.002", "parity_passed": True},
    ])
    update_readme_table([item])
    rendered = readme.read_text(encoding="utf-8")
    assert rendered.startswith("Before") and rendered.endswith("After")
    assert "| Test |" in rendered
    assert "2.20×" in rendered
