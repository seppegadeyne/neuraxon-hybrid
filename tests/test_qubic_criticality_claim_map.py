from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "benchmarks/results/qubic_nia_vol8_criticality_claim_map.json"
MD_PATH = ROOT / "benchmarks/results/qubic_nia_vol8_criticality_claim_map.md"
COMPARISON_JSON_PATH = ROOT / "benchmarks/results/cunxon_comparison.json"
COMPARISON_MD_PATH = ROOT / "benchmarks/results/cunxon_comparison.md"
AVALANCHE_MATRIX_JSON_PATH = (
    ROOT / "benchmarks/results/cunxon_avalanche_window_intervention_matrix.json"
)
AVALANCHE_MATRIX_MD_PATH = (
    ROOT / "benchmarks/results/cunxon_avalanche_window_intervention_matrix.md"
)
AVALANCHE_TASK_CORRELATION_JSON_PATH = (
    ROOT / "benchmarks/results/cunxon_avalanche_intervention_task_correlation.json"
)
AVALANCHE_TASK_CORRELATION_MD_PATH = (
    ROOT / "benchmarks/results/cunxon_avalanche_intervention_task_correlation.md"
)


def test_qubic_nia_vol8_claim_map_records_claims_evidence_and_next_probe() -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    markdown = MD_PATH.read_text(encoding="utf-8")
    comparison_data = json.loads(COMPARISON_JSON_PATH.read_text(encoding="utf-8"))
    comparison_markdown = COMPARISON_MD_PATH.read_text(encoding="utf-8")

    assert data["source"]["url"].endswith(
        "brain-criticality-branching-ratio-neural-artificial-networks-neuraxon-nia-vol-8"
    )
    assert data["source"]["published"] == "2026-05-13"
    assert data["hypothesis_for_this_slice"] == "branching_ratio_as_measured_regime_gate"
    assert len(data["claims"]) >= 6

    claim_ids = {claim["id"] for claim in data["claims"]}
    assert {
        "branching-ratio-regimes",
        "slightly-subcritical-reverberating",
        "artificial-edge-of-chaos",
        "neuraxon-real-time-invariant",
        "self-organized-criticality",
        "functional-generalization-claim",
    }.issubset(claim_ids)

    evidence_ids = {item["id"] for item in data["evidence_map"]}
    assert {
        "hybrid-criticality-summary",
        "cunxon-vram-resident",
        "cunxon-resident-action",
        "cunxon-aigarth-augmented-train",
        "cunxon-branching-regime-scan",
    }.issubset(evidence_ids)

    assert data["current_evidence_boundary"].startswith("The article is a hypothesis source")
    assert data["recommended_next_probe"]["id"] == "higher_resolution_task_coupled_regime_probe"
    assert data["recommended_next_probe"]["github_issue"].endswith("/issues/85")
    assert data["recommended_next_probe"]["acceptance_criteria"]
    assert any("stress_holdout" in question for question in data["open_questions"])

    assert "# Qubic NIA Vol. 8 criticality claim map" in markdown
    assert "sigma ≈ 1" in markdown
    assert "branching_ratio_mean=1.194208" in markdown
    assert "VRAM-resident active-state ratio mean≈1.017778" in markdown
    assert "stress-holdout mean=0.333333" in markdown
    assert "mean branching proxy=0.997701" in markdown
    assert "0/5 seeds beat the best constant stress baseline" in markdown
    assert "not intelligence evidence" in markdown
    assert "cunxon_branching_regime_scan" in markdown
    assert "cunxon_avalanche_intervention_task_correlation" in markdown
    assert "cunxon_avalanche_window_intervention_matrix" in comparison_markdown
    assert "\\n" not in markdown

    assert comparison_data["qubic_nia_vol8_criticality_claim_map"]["recommended_next_probe"] == (
        "higher_resolution_task_coupled_regime_probe"
    )
    scan_summary = comparison_data["cunxon_branching_regime_scan"]
    assert scan_summary["mean_branching_activity_ratio_proxy"] == 0.997701
    assert scan_summary["mean_stress_holdout"] == 0.333333
    assert scan_summary["beats_stress_baseline_count"] == 0
    assert "Qubic NIA Vol. 8 criticality claim map" in comparison_markdown
    assert "branching ratio is a necessary diagnostic" in comparison_markdown
    assert "cuNxon branching-ratio regime scan" in comparison_markdown
    assert "mean branching proxy=0.997701" in comparison_markdown


def test_cunxon_avalanche_window_intervention_matrix_records_estimator_sensitivity() -> None:
    data = json.loads(AVALANCHE_MATRIX_JSON_PATH.read_text(encoding="utf-8"))
    markdown = AVALANCHE_MATRIX_MD_PATH.read_text(encoding="utf-8")
    comparison_data = json.loads(COMPARISON_JSON_PATH.read_text(encoding="utf-8"))
    comparison_markdown = COMPARISON_MD_PATH.read_text(encoding="utf-8")

    assert data["hypothesis_for_this_slice"] == "avalanche_window_estimator_sensitivity"
    assert data["source_claim_ids"] == [
        "branching-ratio-regimes",
        "self-organized-criticality",
        "functional-generalization-claim",
    ]
    assert data["config_count"] >= 3
    assert data["sample_count"] >= 30
    assert data["max_mode_accuracy_delta_vs_constant_baseline"] > 0.0
    assert data["configurations_with_all_modes_beating_baseline"] == []
    assert data["branching_ratio_estimate_range"][1] > data["branching_ratio_estimate_range"][0]
    assert data["verdict"].startswith("Window-length/sample-interval changes moved")
    assert data["recommended_next_probe"]["github_issue"].endswith("/issues/85")
    assert "not intelligence evidence" in data["evidence_boundary"]

    config_ids = {config["id"] for config in data["configurations"]}
    assert {"short-dense", "baseline-equivalent", "long-sparse"}.issubset(config_ids)
    for config in data["configurations"]:
        assert config["artifact_json"].endswith(".json")
        assert config["sample_count"] == 12
        assert config["all_modes_beat_best_constant_baseline"] is False
        assert set(config["accuracy_by_mode"]) == {"infer", "train"}

    assert "# cuNxon avalanche-window intervention matrix" in markdown
    assert "short-dense" in markdown
    assert "baseline-equivalent" in markdown
    assert "long-sparse" in markdown
    assert "Window-length/sample-interval changes moved" in markdown
    assert "not intelligence evidence" in markdown
    assert "\\n" not in markdown

    matrix_summary = comparison_data["cunxon_avalanche_window_intervention_matrix"]
    assert matrix_summary["hypothesis"] == "avalanche_window_estimator_sensitivity"
    assert matrix_summary["sample_count"] == data["sample_count"]
    assert matrix_summary["configurations_with_all_modes_beating_baseline"] == []
    assert "cuNxon avalanche-window intervention matrix" in comparison_markdown
    assert "estimator sensitivity" in comparison_markdown


def test_cunxon_avalanche_intervention_task_correlation_records_split_quality() -> None:
    data = json.loads(AVALANCHE_TASK_CORRELATION_JSON_PATH.read_text(encoding="utf-8"))
    markdown = AVALANCHE_TASK_CORRELATION_MD_PATH.read_text(encoding="utf-8")
    comparison_data = json.loads(COMPARISON_JSON_PATH.read_text(encoding="utf-8"))
    comparison_markdown = COMPARISON_MD_PATH.read_text(encoding="utf-8")
    claim_data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    claim_markdown = MD_PATH.read_text(encoding="utf-8")

    assert data["hypothesis_for_this_slice"] == "avalanche_intervention_task_correlation"
    assert data["source_claim_ids"] == [
        "branching-ratio-regimes",
        "self-organized-criticality",
        "functional-generalization-claim",
    ]
    assert data["sample_count"] >= 60
    assert data["config_count"] >= 2
    assert "holdout" in data["split_accuracy"]
    assert "stress_holdout" in data["split_accuracy"]
    assert "counterfactual_control" in data["split_accuracy"]
    assert data["configurations_beating_stress_baseline"] == []
    assert "not intelligence evidence" in data["evidence_boundary"]

    for config in data["configurations"]:
        assert "stress_holdout" in config["accuracy_by_split"]
        assert "stress_holdout" in config["best_constant_baseline_by_split"]
        assert config["beats_best_constant_baseline_by_split"]["stress_holdout"] is False

    assert "# cuNxon avalanche intervention/task correlation" in markdown
    assert "stress_holdout" in markdown
    assert "counterfactual_control" in markdown
    assert "constant baselines" in markdown
    assert "not intelligence evidence" in markdown
    assert "\\n" not in markdown

    summary = comparison_data["cunxon_avalanche_intervention_task_correlation"]
    assert summary["hypothesis"] == "avalanche_intervention_task_correlation"
    assert summary["configurations_beating_stress_baseline"] == []
    assert "cuNxon avalanche intervention/task correlation" in comparison_markdown
    assert "stress_holdout" in comparison_markdown

    assert claim_data["recommended_next_probe"]["id"] == (
        "higher_resolution_task_coupled_regime_probe"
    )
    assert "cunxon_avalanche_intervention_task_correlation" in claim_markdown
