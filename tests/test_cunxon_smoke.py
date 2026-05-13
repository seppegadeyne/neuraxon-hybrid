"""Tests for cuNxon GPU smoke helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuraxon_agent.cunxon_smoke import (
    CunxonLongHorizonResult,
    CunxonLongHorizonSample,
    CunxonSmokeResult,
    classify_cunxon_status,
    render_long_horizon_markdown_report,
    render_markdown_report,
    validate_trinary_readout,
    write_long_horizon_artifacts,
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
    assert "short smoke tests are insufficient" in markdown
    assert "no decision-quality score measured" in markdown
    assert "raw_network" in markdown
    assert "0.145833" in markdown
    assert "random" in markdown
    assert "always_execute" in markdown
    assert "does not prove intelligence" in markdown
    assert '"verdict": "long-horizon runtime viable, not benchmark-integrated"' in data
    assert '"decision_quality_measured": false' in data


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
