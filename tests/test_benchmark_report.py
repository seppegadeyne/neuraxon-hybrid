"""Tests for the published benchmark report."""

from __future__ import annotations

from pathlib import Path

REPORT_PATH = Path("benchmarks/results/analysis/benchmark_report.md")


def test_benchmark_report_publishes_required_sections_and_assets() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")

    required_headings = [
        "# Neuraxon Agent Benchmark Report",
        "## 1. Summary",
        "## 2. Benchmark setup",
        "## 3. Results",
        "## 4. Per scenario type",
        "## 5. Interpretation",
        "## 6. Statistical comparison",
        "## 7. What this does and does not prove",
        "## 8. Verdict",
        "## 9. Artifacts",
    ]
    for heading in required_headings:
        assert heading in report

    required_plot_links = [
        "plots/accuracy_by_agent.png",
        "plots/confidence_distribution.png",
        "plots/neuromodulator_trends.png",
        "plots/learning_curve.png",
        "plots/activity_energy_trends.png",
    ]
    for plot_link in required_plot_links:
        assert plot_link in report


def test_benchmark_report_contains_concise_proven_not_proven_summary() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert "### Proven / not proven summary" in report
    assert "Proven:" in report
    assert "Not proven:" in report
    assert "raw Neuraxon network generalization" in report
    assert "semantic bridge performance" in report
    assert "explicit temporal context adapter performance" in report
    assert "simple baseline performance" in report
    assert "criticality/dynamics instrumentation evidence" in report


def test_benchmark_report_states_go_and_baseline_comparison() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert "GO for the semantic adapter" in report
    assert "NO-GO for raw Neuraxon generalization claims" in report
    assert "pass_temporal_context_bridge_evidence" in report
    assert "semantic_policy_coverage=100%" in report
    assert "100.00%" in report
    assert "15.71%" in report
    assert "28.57%" in report
    assert "700" in report
    assert "significantly better" in report

    comparison_headers = ["Agent", "Runs", "Correct", "Accuracy", "Avg. confidence"]
    for header in comparison_headers:
        assert header in report


def test_benchmark_report_keeps_memory_and_visual_perception_out_of_scope() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert "memory persistence" in report.lower()
    assert "visual" in report.lower()
    assert "not proven" in report.lower()
    assert "generalization" in report.lower()
    assert "holdout/noisy" in report.lower()
    assert "benchmarks/results/holdout_noisy_generalization.json" in report


def test_benchmark_report_states_policy_ablation_result() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert "policy-ablation" in report.lower()
    assert "raw-network" in report.lower() or "raw network" in report.lower()
    assert "semantic-bridge" in report.lower() or "semantic bridge" in report.lower()
    assert "policy-covered observations" in report.lower()


def test_benchmark_report_states_expanded_temporal_dataset_and_separation() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert "108" in report
    assert "counterfactual" in report.lower()
    assert "noise/perturbation" in report.lower()
    assert "last-observation-only" in report.lower()
    assert "sequence-majority" in report.lower()
    assert "semantic-policy-only" in report.lower()
    assert "semantic-policy success" in report.lower()
    assert "temporal_context_bridge" in report
    assert "explicit temporal context adapter" in report.lower()
    assert "raw neuraxon network dynamics" in report.lower()


def test_benchmark_report_interprets_criticality_and_dynamics_metrics() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert "criticality" in report.lower()
    assert "dynamics_metrics.csv" in report
    assert "criticality_summary.csv" in report
    assert "dead/saturated/random-like" in report.lower()
    assert "near useful dynamic regime" in report.lower()
    assert "neutral-state occupancy" in report.lower()
    assert "transition entropy" in report.lower()
