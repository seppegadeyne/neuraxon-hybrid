"""Tests for cuNxon GPU smoke helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuraxon_agent.cunxon_smoke import (
    CunxonActionProbeResult,
    CunxonActionProbeTrial,
    CunxonLongHorizonResult,
    CunxonLongHorizonSample,
    CunxonSensitivityProbeResult,
    CunxonSensitivityProbeSample,
    CunxonSmokeResult,
    classify_cunxon_status,
    render_action_probe_markdown_report,
    render_long_horizon_markdown_report,
    render_markdown_report,
    render_sensitivity_probe_markdown_report,
    validate_trinary_readout,
    write_action_probe_artifacts,
    write_long_horizon_artifacts,
    write_sensitivity_probe_artifacts,
    write_smoke_artifacts,
)


def test_validate_trinary_readout_accepts_only_minus_zero_plus_one() -> None:
    validate_trinary_readout([-1, 0, 1, 1, -1])

    with pytest.raises(ValueError, match="non-trinary"):
        validate_trinary_readout([-1, 0, 2])

    with pytest.raises(ValueError, match="empty"):
        validate_trinary_readout([])


def test_classify_cunxon_status_keeps_evidence_boundary_explicit() -> None:
    assert (
        classify_cunxon_status(
            configure_ok=False,
            build_ok=False,
            ctest_ok=False,
            python_ok=False,
        )
        == "unusable"
    )
    assert (
        classify_cunxon_status(
            configure_ok=True,
            build_ok=True,
            ctest_ok=False,
            python_ok=False,
        )
        == "build-only"
    )
    assert (
        classify_cunxon_status(
            configure_ok=True,
            build_ok=True,
            ctest_ok=True,
            python_ok=True,
        )
        == "smoke-test viable"
    )


def test_smoke_artifacts_include_gpu_metrics_and_non_intelligence_boundary(tmp_path: Path) -> None:
    result = CunxonSmokeResult(
        status="smoke-test viable",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        steps=24,
        readout=[-1, 0, 1, 0],
        energy=1.25,
        elapsed_ms=12.5,
        notes=["ctypes smoke completed"],
    )

    markdown = render_markdown_report(result)
    assert "NVIDIA GeForce RTX 5090" in markdown
    assert "{-1, 0, +1}" in markdown
    assert "does not prove intelligence" in markdown
    assert "smoke-test viable" in markdown

    json_path = tmp_path / "cunxon.json"
    markdown_path = tmp_path / "cunxon.md"
    write_smoke_artifacts(result, json_path=json_path, markdown_path=markdown_path)

    assert '"status": "smoke-test viable"' in json_path.read_text(encoding="utf-8")
    assert "No broad Neuraxon intelligence claim" in markdown_path.read_text(encoding="utf-8")


def test_long_horizon_report_frames_neuraxon_as_time_dependent_learning_process(
    tmp_path: Path,
) -> None:
    result = CunxonLongHorizonResult(
        status="long-horizon probe viable",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        total_steps=1024,
        sample_interval=256,
        samples=[
            CunxonLongHorizonSample(step=256, readout=[0, 0, 0], energy=1.0, elapsed_ms=2.5),
            CunxonLongHorizonSample(step=512, readout=[0, 1, 0], energy=2.0, elapsed_ms=5.0),
            CunxonLongHorizonSample(step=1024, readout=[0, 1, -1], energy=4.0, elapsed_ms=10.0),
        ],
        readout_change_count=2,
        unique_readouts=3,
        energy_delta=3.0,
        notes=["fake long horizon"],
    )

    markdown = render_long_horizon_markdown_report(result)
    assert "Long-horizon learning caveat" in markdown
    assert "short smoke tests are insufficient" in markdown
    assert "brain-like" in markdown
    assert "does not prove intelligence" in markdown
    assert "1024" in markdown
    assert "readout changes: 2" in markdown

    json_path = tmp_path / "long.json"
    markdown_path = tmp_path / "long.md"
    write_long_horizon_artifacts(result, json_path=json_path, markdown_path=markdown_path)

    assert '"total_steps": 1024' in json_path.read_text(encoding="utf-8")
    assert "short smoke tests are insufficient" in markdown_path.read_text(encoding="utf-8")


def test_action_probe_report_scores_decoded_readout_against_expected_actions(
    tmp_path: Path,
) -> None:
    result = CunxonActionProbeResult(
        status="task-coupled action probe viable",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        trial_steps=32,
        trials=[
            CunxonActionProbeTrial(
                name="execute-positive-drive",
                input_vector=[1.0, 0.25, 0.0],
                expected_action="execute",
                readout=[1, 0, 0],
                decoded_action="PROCEED",
                normalized_action="execute",
                confidence=0.3333,
                outcome="success",
                energy=12.0,
                elapsed_ms=1.0,
            ),
            CunxonActionProbeTrial(
                name="retry-negative-drive",
                input_vector=[-1.0, -0.25, 0.0],
                expected_action="retry",
                readout=[0, 0, 0],
                decoded_action="PAUSE",
                normalized_action="query",
                confidence=1.0,
                outcome="failure",
                energy=13.0,
                elapsed_ms=2.0,
            ),
        ],
        success_count=1,
        accuracy=0.5,
        notes=["fake task-coupled probe"],
    )

    markdown = render_action_probe_markdown_report(result)
    assert "task-coupled action probe" in markdown
    assert "decision-quality" in markdown
    assert "Accuracy: 0.500000" in markdown
    assert "execute-positive-drive" in markdown
    assert "does not prove intelligence" in markdown

    json_path = tmp_path / "action_probe.json"
    markdown_path = tmp_path / "action_probe.md"
    write_action_probe_artifacts(result, json_path=json_path, markdown_path=markdown_path)

    assert '"accuracy": 0.5' in json_path.read_text(encoding="utf-8")
    assert "decision-quality" in markdown_path.read_text(encoding="utf-8")


def test_sensitivity_probe_report_compares_infer_and_train_modes(tmp_path: Path) -> None:
    result = CunxonSensitivityProbeResult(
        status="sensitivity probe viable",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        steps=32,
        samples=[
            CunxonSensitivityProbeSample(
                mode="infer",
                seed_offset=79,
                stimulus="positive-drive",
                input_vector=[1.0, 0.25, 0.0],
                readout=[0, 0, -1],
                decoded_action="RETRY",
                normalized_action="retry",
                confidence=0.3333,
                energy=12.0,
                elapsed_ms=1.0,
            ),
            CunxonSensitivityProbeSample(
                mode="train",
                seed_offset=79,
                stimulus="positive-drive",
                input_vector=[1.0, 0.25, 0.0],
                readout=[1, 0, 0],
                decoded_action="PROCEED",
                normalized_action="execute",
                confidence=0.3333,
                energy=14.0,
                elapsed_ms=2.0,
            ),
        ],
        unique_readouts_by_mode={"infer": 1, "train": 1},
        action_distribution_by_mode={
            "infer": {"retry": 1},
            "train": {"execute": 1},
        },
        action_change_count_by_stimulus={"positive-drive": 1},
        notes=["fake sensitivity probe"],
    )

    markdown = render_sensitivity_probe_markdown_report(result)
    assert "infer-vs-train sensitivity probe" in markdown
    assert "paper-canonical" in markdown
    assert "positive-drive" in markdown
    assert "action changes by stimulus" in markdown
    assert "does not prove intelligence" in markdown

    json_path = tmp_path / "sensitivity.json"
    markdown_path = tmp_path / "sensitivity.md"
    write_sensitivity_probe_artifacts(result, json_path=json_path, markdown_path=markdown_path)

    data = json_path.read_text(encoding="utf-8")
    assert '"status": "sensitivity probe viable"' in data
    assert '"mode": "train"' in data
    assert "infer-vs-train sensitivity probe" in markdown_path.read_text(encoding="utf-8")


def test_tracked_cunxon_investigation_report_preserves_live_smoke_and_boundary() -> None:
    markdown = Path("benchmarks/results/cunxon_smoke.md").read_text(encoding="utf-8")
    data = Path("benchmarks/results/cunxon_smoke.json").read_text(encoding="utf-8")

    assert "Status: `smoke-test viable`" in markdown
    assert "minimal one-sphere ctypes smoke completed" in markdown
    assert "inter-sphere Python demo remains separate" in markdown
    assert "does not prove intelligence" in markdown
    assert "RTX 5090" in markdown
    assert "Compute capability: 12.0" in markdown
    assert '"status": "smoke-test viable"' in data
    assert '"readout": [' in data


def test_tracked_cunxon_comparison_report_separates_gpu_smoke_from_decision_quality() -> None:
    markdown = Path("benchmarks/results/cunxon_comparison.md").read_text(encoding="utf-8")
    data = Path("benchmarks/results/cunxon_comparison.json").read_text(encoding="utf-8")

    assert "cuNxon raw CUDA smoke" in markdown
    assert "cuNxon long-horizon raw dynamics" in markdown
    assert "cuNxon task-coupled action probe" in markdown
    assert "cuNxon infer-vs-train sensitivity probe" in markdown
    assert "short smoke tests are insufficient" in markdown
    assert "no decision-quality score measured" in markdown
    assert "decision-quality measured but not useful" in markdown
    assert "train-mode flat" in markdown
    assert "raw_network" in markdown
    assert "0.145833" in markdown
    assert "random" in markdown
    assert "always_execute" in markdown
    assert "does not prove intelligence" in markdown
    assert '"verdict": "long-horizon runtime viable, not benchmark-integrated"' in data
    assert '"verdict": "task-coupled but not above baselines"' in data
    assert '"verdict": "sensitivity diagnostic train-mode flat"' in data
    assert '"decision_quality_measured": false' in data
    assert '"decision_quality_measured": true' in data


def test_tracked_cunxon_sensitivity_probe_report_records_train_mode_flatness() -> None:
    markdown = Path("benchmarks/results/cunxon_sensitivity_probe.md").read_text(encoding="utf-8")
    data = Path("benchmarks/results/cunxon_sensitivity_probe.json").read_text(encoding="utf-8")

    assert "infer-vs-train sensitivity probe" in markdown
    assert "paper-canonical continuous-learning mode" in markdown
    assert "train | 1 | query=9" in markdown
    assert "infer | 3" in markdown
    assert "does not prove intelligence" in markdown
    assert '"status": "sensitivity probe viable"' in data
    assert '"sample_count": 18' in data
    assert '"train": {' in data

def test_tracked_cunxon_action_probe_report_preserves_flat_readout_caveat() -> None:
    markdown = Path("benchmarks/results/cunxon_action_probe.md").read_text(encoding="utf-8")
    data = Path("benchmarks/results/cunxon_action_probe.json").read_text(encoding="utf-8")

    assert "task-coupled action probe" in markdown
    assert "Accuracy:" in markdown
    assert "does not prove intelligence" in markdown
    assert "flat or baseline-level" in markdown
    assert '"status": "task-coupled action probe viable"' in data
    assert '"accuracy":' in data


def test_tracked_cunxon_long_horizon_report_preserves_learning_caveat() -> None:
    markdown = Path("benchmarks/results/cunxon_long_horizon.md").read_text(encoding="utf-8")
    data = Path("benchmarks/results/cunxon_long_horizon.json").read_text(encoding="utf-8")

    assert "Long-horizon learning caveat" in markdown
    assert "short smoke tests are insufficient" in markdown
    assert "brain-like" in markdown
    assert "readout changes" in markdown
    assert "does not prove intelligence" in markdown
    assert '"status": "long-horizon probe viable"' in data
    assert '"total_steps":' in data
