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
AVALANCHE_SEED_REPLICATION_JSON_PATH = (
    ROOT / "benchmarks/results/cunxon_avalanche_intervention_seed_replication.json"
)
AVALANCHE_SEED_REPLICATION_MD_PATH = (
    ROOT / "benchmarks/results/cunxon_avalanche_intervention_seed_replication.md"
)
CONTROLLED_REGIME_CALIBRATION_JSON_PATH = (
    ROOT / "benchmarks/results/cunxon_controlled_regime_calibration.json"
)
CONTROLLED_REGIME_CALIBRATION_MD_PATH = (
    ROOT / "benchmarks/results/cunxon_controlled_regime_calibration.md"
)
CRITICALITY_DECODER_SEPARATION_JSON_PATH = (
    ROOT / "benchmarks/results/cunxon_criticality_decoder_separation.json"
)
CRITICALITY_DECODER_SEPARATION_MD_PATH = (
    ROOT / "benchmarks/results/cunxon_criticality_decoder_separation.md"
)
STRESS_GEOMETRY_AUDIT_JSON_PATH = ROOT / "benchmarks/results/cunxon_stress_geometry_audit.json"
STRESS_GEOMETRY_AUDIT_MD_PATH = ROOT / "benchmarks/results/cunxon_stress_geometry_audit.md"
STRESS_AMPLITUDE_LADDER_JSON_PATH = (
    ROOT
    / "benchmarks/results/cunxon_aigarth_action_target_contract_stress_amplitude_ladder_probe.json"
)
STRESS_AMPLITUDE_LADDER_MD_PATH = (
    ROOT
    / "benchmarks/results/cunxon_aigarth_action_target_contract_stress_amplitude_ladder_probe.md"
)
STRESS_OBJECTIVE_JSON_PATH = (
    ROOT / "benchmarks/results/cunxon_aigarth_action_target_contract_stress_objective_probe.json"
)
STRESS_OBJECTIVE_MD_PATH = (
    ROOT / "benchmarks/results/cunxon_aigarth_action_target_contract_stress_objective_probe.md"
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
    assert data["recommended_next_probe"]["id"] == "stress_objective_decoder_geometry_followup"
    assert data["recommended_next_probe"]["status"] == "open"
    assert data["recommended_next_probe"]["github_issue"].endswith("/issues/89")
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
    assert "cunxon_controlled_regime_calibration" in markdown
    assert "cunxon_avalanche_window_intervention_matrix" in comparison_markdown
    assert "\\n" not in markdown

    assert comparison_data["qubic_nia_vol8_criticality_claim_map"]["recommended_next_probe"] == (
        "stress_objective_decoder_geometry_followup"
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
        "stress_objective_decoder_geometry_followup"
    )
    assert "cunxon_avalanche_intervention_task_correlation" in claim_markdown


def test_cunxon_avalanche_intervention_seed_replication_records_brittleness() -> None:
    data = json.loads(AVALANCHE_SEED_REPLICATION_JSON_PATH.read_text(encoding="utf-8"))
    markdown = AVALANCHE_SEED_REPLICATION_MD_PATH.read_text(encoding="utf-8")
    comparison_data = json.loads(COMPARISON_JSON_PATH.read_text(encoding="utf-8"))
    comparison_markdown = COMPARISON_MD_PATH.read_text(encoding="utf-8")
    claim_markdown = MD_PATH.read_text(encoding="utf-8")

    assert data["hypothesis_for_this_slice"] == "avalanche_intervention_task_correlation"
    assert data["seed_offsets"] == [129, 130, 131, 132]
    assert data["sample_count"] >= 300
    assert data["config_count"] == 2
    assert set(data["split_accuracy"]) >= {
        "holdout",
        "hard_holdout",
        "stress_holdout",
        "counterfactual_control",
        "permuted_control",
    }
    assert data["configurations_beating_stress_baseline"] == []
    assert data["split_accuracy"]["stress_holdout"] <= data["best_constant_baseline_by_split"][
        "stress_holdout"
    ]
    assert "not intelligence evidence" in data["evidence_boundary"]

    assert "# cuNxon avalanche intervention/task correlation" in markdown
    assert "129" in markdown
    assert "stress_holdout" in markdown
    assert "permuted_control" in markdown
    assert "not intelligence evidence" in markdown
    assert "\\n" not in markdown

    summary = comparison_data["cunxon_avalanche_intervention_seed_replication"]
    assert summary["hypothesis"] == "avalanche_intervention_seed_replication"
    assert summary["seed_offsets"] == [129, 130, 131, 132]
    assert summary["configurations_beating_stress_baseline"] == []
    assert summary["stress_holdout_accuracy"] == data["split_accuracy"]["stress_holdout"]
    assert "cuNxon avalanche intervention seed replication" in comparison_markdown
    assert "seed-replication" in comparison_markdown
    assert "cunxon_avalanche_intervention_seed_replication" in claim_markdown


def test_cunxon_controlled_regime_calibration_records_estimator_boundaries() -> None:
    data = json.loads(CONTROLLED_REGIME_CALIBRATION_JSON_PATH.read_text(encoding="utf-8"))
    markdown = CONTROLLED_REGIME_CALIBRATION_MD_PATH.read_text(encoding="utf-8")
    comparison_data = json.loads(COMPARISON_JSON_PATH.read_text(encoding="utf-8"))
    comparison_markdown = COMPARISON_MD_PATH.read_text(encoding="utf-8")
    claim_markdown = MD_PATH.read_text(encoding="utf-8")

    assert data["hypothesis_for_this_slice"] == "controlled_regime_calibration"
    assert data["source_issue"].endswith("/issues/86")
    assert data["config_count"] >= 3
    assert data["sample_count"] >= 100
    assert data["regime_drive_scales"] == {
        "low-drive": 0.25,
        "medium-drive": 1.0,
        "high-drive": 2.0,
    }
    assert set(data["split_accuracy"]) >= {
        "holdout",
        "hard_holdout",
        "stress_holdout",
        "counterfactual_control",
        "permuted_control",
    }
    assert data["configurations_beating_stress_baseline"] == []
    assert data["stress_holdout_accuracy"] <= data["best_constant_baseline_by_split"][
        "stress_holdout"
    ]
    assert "not intelligence evidence" in data["evidence_boundary"]
    assert "estimator calibration" in data["verdict"]

    config_ids = {config["id"] for config in data["configurations"]}
    assert {"low-drive", "medium-drive", "high-drive"}.issubset(config_ids)
    for config in data["configurations"]:
        assert "drive_scale" in config
        assert "mean_transition_entropy_bits" in config
        assert "action_distribution" in config
        assert "stress_holdout" in config["accuracy_by_split"]
        assert config["beats_best_constant_baseline_by_split"]["stress_holdout"] is False

    assert "# cuNxon controlled-regime criticality calibration" in markdown
    assert "low-drive" in markdown
    assert "medium-drive" in markdown
    assert "high-drive" in markdown
    assert "stress_holdout" in markdown
    assert "not intelligence evidence" in markdown
    assert "\\n" not in markdown

    summary = comparison_data["cunxon_controlled_regime_calibration"]
    assert summary["hypothesis"] == "controlled_regime_calibration"
    assert summary["configurations_beating_stress_baseline"] == []
    assert summary["stress_holdout_accuracy"] == data["stress_holdout_accuracy"]
    assert "cuNxon controlled-regime criticality calibration" in comparison_markdown
    assert "controlled_regime_calibration" in claim_markdown


def test_cunxon_criticality_decoder_separation_explains_stress_bottleneck() -> None:
    data = json.loads(CRITICALITY_DECODER_SEPARATION_JSON_PATH.read_text(encoding="utf-8"))
    markdown = CRITICALITY_DECODER_SEPARATION_MD_PATH.read_text(encoding="utf-8")
    comparison_data = json.loads(COMPARISON_JSON_PATH.read_text(encoding="utf-8"))
    comparison_markdown = COMPARISON_MD_PATH.read_text(encoding="utf-8")
    claim_data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    claim_markdown = MD_PATH.read_text(encoding="utf-8")

    assert data["hypothesis_for_this_slice"] == "criticality_decoder_separation"
    assert data["source_artifact"] == "benchmarks/results/cunxon_controlled_regime_calibration.json"
    assert data["sample_count"] == 288
    assert data["stress_holdout"]["sample_count"] == 72
    assert data["stress_holdout"]["accuracy"] == 0.333333
    assert data["stress_holdout"]["best_constant_baseline"] == 0.333333
    assert data["stress_holdout"]["query_collapse_rate"] >= 0.9
    assert data["stress_holdout"]["execute_retry_accuracy"] < 0.1
    assert data["stress_holdout"]["query_accuracy"] > 0.9
    assert data["diagnosis"] == "decoder_action_collapse_on_low_margin_execute_retry_stress_cases"
    assert data["estimator_movement_status"] == "present_but_not_task_sufficient"
    assert "not intelligence evidence" in data["evidence_boundary"]

    assert "# cuNxon criticality/decoder separation" in markdown
    assert "query_collapse_rate=0.902778" in markdown
    assert "execute/retry stress accuracy=0.041667" in markdown
    assert "estimator movement is present" in markdown
    assert "decoder/action collapse" in markdown
    assert "not intelligence evidence" in markdown
    assert "\\n" not in markdown

    summary = comparison_data["cunxon_criticality_decoder_separation"]
    assert summary["hypothesis"] == "criticality_decoder_separation"
    assert summary["diagnosis"] == data["diagnosis"]
    assert summary["stress_query_collapse_rate"] == data["stress_holdout"]["query_collapse_rate"]
    assert "cuNxon criticality/decoder separation" in comparison_markdown
    assert "stress query collapse" in comparison_markdown

    assert any(
        item["id"] == "cunxon-criticality-decoder-separation"
        for item in claim_data["evidence_map"]
    )
    assert claim_data["recommended_next_probe"]["status"] == "open"
    assert "cunxon_criticality_decoder_separation" in claim_markdown


def test_cunxon_stress_injection_upper_bound_keeps_stress_baseline_boundary() -> None:
    artifact_path = ROOT / (
        "benchmarks/results/"
        "cunxon_aigarth_action_target_contract_stress_injection_probe.json"
    )
    markdown_path = ROOT / (
        "benchmarks/results/"
        "cunxon_aigarth_action_target_contract_stress_injection_probe.md"
    )
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    comparison_data = json.loads(COMPARISON_JSON_PATH.read_text(encoding="utf-8"))
    comparison_markdown = COMPARISON_MD_PATH.read_text(encoding="utf-8")
    claim_data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    claim_markdown = MD_PATH.read_text(encoding="utf-8")

    assert data["fitness_variant"] == "target_contract_stress_injection"
    assert data["seed_count"] == 5
    assert data["accuracy_summary_by_split"]["stress_train"]["mean"] == 0.3333333333333333
    assert data["accuracy_summary_by_split"]["stress_holdout"]["mean"] == 0.3333333333333333
    assert data["seeds_beating_baseline_by_split"]["stress_train"] == 0
    assert data["seeds_beating_baseline_by_split"]["stress_holdout"] == 0
    assert data["unexpected_action_count"] == 0

    assert "stress-injection audit" in markdown
    assert "leaks stress-like labels" in markdown
    assert "not intelligence evidence" in markdown
    assert "\\n" not in markdown

    summary = comparison_data["cunxon_aigarth_action_target_contract_stress_injection_probe"]
    assert summary["hypothesis"] == "stress_label_injection_upper_bound"
    assert summary["stress_train_mean"] == 0.333333
    assert summary["stress_holdout_mean"] == 0.333333
    assert "stress-injection upper-bound audit" in comparison_markdown
    assert "negative upper-bound/debugging evidence" in comparison_markdown

    assert any(
        item["id"] == "cunxon-stress-injection-upper-bound"
        for item in claim_data["evidence_map"]
    )
    assert claim_data["recommended_next_probe"]["id"] == (
        "stress_objective_decoder_geometry_followup"
    )
    assert claim_data["recommended_next_probe"]["status"] == "open"
    assert "stress-injection upper-bound diagnostic" in claim_markdown


def test_cunxon_stress_geometry_audit_identifies_low_margin_query_collapse() -> None:
    data = json.loads(STRESS_GEOMETRY_AUDIT_JSON_PATH.read_text(encoding="utf-8"))
    markdown = STRESS_GEOMETRY_AUDIT_MD_PATH.read_text(encoding="utf-8")
    comparison_data = json.loads(COMPARISON_JSON_PATH.read_text(encoding="utf-8"))
    comparison_markdown = COMPARISON_MD_PATH.read_text(encoding="utf-8")
    claim_data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    claim_markdown = MD_PATH.read_text(encoding="utf-8")

    assert data["hypothesis_for_this_slice"] == "stress_stimulus_geometry_separability"
    assert data["source_artifact"].endswith(
        "cunxon_aigarth_action_target_contract_stress_injection_probe.json"
    )
    assert data["sample_count"] == 180
    assert data["stress_train_query_collapse_rate"] == 1.0
    assert data["stress_holdout_query_collapse_rate"] == 1.0
    assert data["stress_execute_retry_accuracy"] == 0.0
    assert data["augmented_train_execute_retry_accuracy"] == 1.0
    assert data["mean_abs_sum_by_split"]["stress_train"] < data["mean_abs_sum_by_split"][
        "augmented_train"
    ]
    assert data["recommended_next_probe"]["id"] == "stress_amplitude_ladder_probe"
    assert data["recommended_next_probe"]["github_issue"].endswith("/issues/87")
    assert "not intelligence evidence" in data["evidence_boundary"]

    assert "# cuNxon stress stimulus geometry audit" in markdown
    assert "stress_train query-collapse rate=1.000000" in markdown
    assert "stress_holdout query-collapse rate=1.000000" in markdown
    assert "not intelligence evidence" in markdown
    assert "\\n" not in markdown

    summary = comparison_data["cunxon_stress_geometry_audit"]
    assert summary["hypothesis"] == "stress_stimulus_geometry_separability"
    assert summary["stress_execute_retry_accuracy"] == 0.0
    assert summary["recommended_next_probe"] == "stress_amplitude_ladder_probe"
    assert "cuNxon stress stimulus geometry audit" in comparison_markdown

    assert any(item["id"] == "cunxon-stress-geometry-audit" for item in claim_data["evidence_map"])
    assert claim_data["recommended_next_probe"]["id"] == (
        "stress_objective_decoder_geometry_followup"
    )
    assert "cunxon_stress_geometry_audit" in claim_markdown


def test_cunxon_stress_amplitude_ladder_identifies_drive_threshold_but_not_generalization() -> None:
    data = json.loads(STRESS_AMPLITUDE_LADDER_JSON_PATH.read_text(encoding="utf-8"))
    markdown = STRESS_AMPLITUDE_LADDER_MD_PATH.read_text(encoding="utf-8")
    comparison_data = json.loads(COMPARISON_JSON_PATH.read_text(encoding="utf-8"))
    comparison_markdown = COMPARISON_MD_PATH.read_text(encoding="utf-8")
    claim_data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    claim_markdown = MD_PATH.read_text(encoding="utf-8")

    assert data["status"] == "aigarth target-contract stress amplitude-ladder completed"
    assert data["seed_offsets"] == [142, 143, 144]
    assert data["amplitude_factors"] == [1.0, 1.5, 2.0, 3.0]
    assert data["original_stress_holdout_accuracy_mean"] == 1 / 3
    assert data["original_stress_holdout_query_collapse_rate"] == 1.0
    assert data["best_scaled_stress_holdout_amplitude_factor"] == 3.0
    assert data["best_scaled_stress_holdout_accuracy_mean"] > 0.8
    assert "not intelligence evidence" in data["evidence_boundary"]

    scaled_holdout = {
        summary["amplitude_factor"]: summary
        for summary in data["split_summaries"]
        if summary["split"].startswith("stress_holdout_scaled_")
    }
    assert scaled_holdout[1.0]["query_collapse_rate"] == 1.0
    assert scaled_holdout[3.0]["execute_retry_accuracy"] > 0.8
    assert scaled_holdout[3.0]["seeds_beating_best_baseline"] == 3

    assert "# cuNxon Aigarth target-contract stress amplitude-ladder" in markdown
    assert "Best scaled stress_holdout accuracy: `0.833333` at `3.0x`" in markdown
    assert "not intelligence evidence" in markdown
    assert "\\n" not in markdown

    summary = comparison_data["cunxon_stress_amplitude_ladder_probe"]
    assert summary["hypothesis"] == "stress_amplitude_ladder_separability"
    assert summary["best_scaled_stress_holdout_amplitude_factor"] == 3.0
    assert summary["original_stress_holdout_accuracy_mean"] == 1 / 3
    assert "cuNxon Aigarth target-contract stress amplitude-ladder" in comparison_markdown

    assert any(
        item["id"] == "cunxon-stress-amplitude-ladder"
        for item in claim_data["evidence_map"]
    )
    assert claim_data["recommended_next_probe"]["id"] == (
        "stress_objective_decoder_geometry_followup"
    )
    assert "cunxon_aigarth_action_target_contract_stress_amplitude_ladder_probe" in claim_markdown


def test_cunxon_stress_objective_preserves_scaled_separability_but_original_stress_fails() -> None:
    data = json.loads(STRESS_OBJECTIVE_JSON_PATH.read_text(encoding="utf-8"))
    markdown = STRESS_OBJECTIVE_MD_PATH.read_text(encoding="utf-8")
    comparison_data = json.loads(COMPARISON_JSON_PATH.read_text(encoding="utf-8"))
    comparison_markdown = COMPARISON_MD_PATH.read_text(encoding="utf-8")
    claim_data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    claim_markdown = MD_PATH.read_text(encoding="utf-8")

    assert data["status"] == "aigarth target-contract stress objective completed"
    assert data["seed_offsets"] == [147, 148, 149]
    assert data["fitness_variant"] == "target_contract_stress_margin_weighted"
    assert data["amplitude_factor"] == 3.0
    assert data["original_stress_holdout_accuracy_mean"] == 1 / 3
    assert data["original_stress_holdout_query_collapse_rate"] == 1.0
    assert data["original_stress_holdout_execute_retry_accuracy"] == 0.0
    assert data["scaled_stress_holdout_accuracy_mean"] > 0.88
    assert data["scaled_stress_holdout_execute_retry_accuracy"] == 1.0
    assert data["counterfactual_control_accuracy_mean"] < 1 / 3
    assert data["permuted_control_accuracy_mean"] == 0.0
    assert "not intelligence evidence" in data["evidence_boundary"]

    assert "# cuNxon Aigarth target-contract stress objective" in markdown
    assert "Original stress_holdout accuracy: `0.333333`" in markdown
    assert "Scaled stress_holdout accuracy: `0.888889`" in markdown
    assert "not intelligence evidence" in markdown
    assert "\\n" not in markdown

    summary = comparison_data["cunxon_stress_objective_probe"]
    assert summary["hypothesis"] == "target_aligned_stress_objective"
    assert summary["original_stress_holdout_accuracy_mean"] == 1 / 3
    assert summary["scaled_stress_holdout_accuracy_mean"] > 0.88
    assert "cuNxon Aigarth target-contract stress objective" in comparison_markdown

    assert any(item["id"] == "cunxon-stress-objective" for item in claim_data["evidence_map"])
    assert claim_data["recommended_next_probe"]["id"] == (
        "stress_objective_decoder_geometry_followup"
    )
    assert claim_data["recommended_next_probe"]["status"] == "open"
    assert "cunxon_aigarth_action_target_contract_stress_objective_probe" in claim_markdown
