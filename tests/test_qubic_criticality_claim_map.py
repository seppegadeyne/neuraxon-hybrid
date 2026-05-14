from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "benchmarks/results/qubic_nia_vol8_criticality_claim_map.json"
MD_PATH = ROOT / "benchmarks/results/qubic_nia_vol8_criticality_claim_map.md"
COMPARISON_JSON_PATH = ROOT / "benchmarks/results/cunxon_comparison.json"
COMPARISON_MD_PATH = ROOT / "benchmarks/results/cunxon_comparison.md"


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
    assert data["recommended_next_probe"]["id"] == "cunxon-branching-ratio-regime-scan"
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
    assert "cunxon-branching-ratio-regime-scan" in markdown
    assert "\\n" not in markdown

    assert comparison_data["qubic_nia_vol8_criticality_claim_map"]["recommended_next_probe"] == (
        "cunxon-branching-ratio-regime-scan"
    )
    scan_summary = comparison_data["cunxon_branching_regime_scan"]
    assert scan_summary["mean_branching_activity_ratio_proxy"] == 0.997701
    assert scan_summary["mean_stress_holdout"] == 0.333333
    assert scan_summary["beats_stress_baseline_count"] == 0
    assert "Qubic NIA Vol. 8 criticality claim map" in comparison_markdown
    assert "branching ratio is a necessary diagnostic" in comparison_markdown
    assert "cuNxon branching-ratio regime scan" in comparison_markdown
    assert "mean branching proxy=0.997701" in comparison_markdown
