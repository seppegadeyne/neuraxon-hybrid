"""Tests for neuraxon-agent CLI."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from neuraxon_agent.cli import main
from neuraxon_agent.cunxon_smoke import (
    CunxonActionProbeResult,
    CunxonActionProbeTrial,
    CunxonLongHorizonResult,
    CunxonLongHorizonSample,
    CunxonLongSweepProbeResult,
    CunxonLongSweepSample,
    CunxonMultisphereActionCase,
    CunxonMultisphereActionProbeResult,
    CunxonPatternRecallSample,
    CunxonSensitivityProbeResult,
    CunxonSensitivityProbeSample,
    CunxonSmokeResult,
    CunxonSnapshotObservation,
    CunxonSnapshotPatternProbeResult,
)


def test_cli_help() -> None:
    assert main(["--help"]) == 0


def test_cli_think() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = Path(tmpdir) / "obs.json"
        out = Path(tmpdir) / "act.json"
        inp.write_text(json.dumps({"observation": {"type": "prompt", "content": "hi"}}))
        rc = main(["think", "-i", str(inp), "-o", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert "action" in data
        assert "confidence" in data


def test_cli_modulate() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = Path(tmpdir) / "out.json"
        out = Path(tmpdir) / "res.json"
        inp.write_text(json.dumps({"outcome": "success"}))
        rc = main(["modulate", "-i", str(inp), "-o", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert "state" in data


def test_cli_evolve() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "evo.json"
        rc = main(["evolve", "-g", "1", "-e", "1", "--seed", "42", "-o", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert "summary" in data


def test_cli_cunxon_smoke_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_smoke(**_: object) -> CunxonSmokeResult:
        return CunxonSmokeResult(
            status="smoke-test viable",
            upstream_commit="upstream",
            cunxon_commit="cunxon",
            library_path="/tmp/libcunxon.so",
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            steps=3,
            readout=[-1, 0, 1],
            energy=0.5,
            elapsed_ms=1.25,
            notes=["fake smoke"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_smoke", fake_smoke)
    json_path = tmp_path / "cunxon.json"
    markdown_path = tmp_path / "cunxon.md"

    rc = main(
        [
            "cunxon-smoke",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    assert '"status": "smoke-test viable"' in json_path.read_text(encoding="utf-8")
    assert "No broad Neuraxon intelligence claim" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_long_horizon_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**_: object) -> CunxonLongHorizonResult:
        return CunxonLongHorizonResult(
            status="long-horizon probe viable",
            upstream_commit="upstream",
            cunxon_commit="cunxon",
            library_path="/tmp/libcunxon.so",
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            total_steps=1024,
            sample_interval=256,
            samples=[
                CunxonLongHorizonSample(
                    step=256,
                    readout=[0, 0, 0],
                    energy=1.0,
                    elapsed_ms=1.0,
                )
            ],
            readout_change_count=0,
            unique_readouts=1,
            energy_delta=0.0,
            notes=["fake long horizon"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_long_horizon_probe", fake_probe)
    json_path = tmp_path / "long.json"
    markdown_path = tmp_path / "long.md"

    rc = main(
        [
            "cunxon-long-horizon",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--steps",
            "1024",
            "--sample-interval",
            "256",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    assert '"status": "long-horizon probe viable"' in json_path.read_text(encoding="utf-8")
    assert "short smoke tests are insufficient" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_long_sweep_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**_: object) -> CunxonLongSweepProbeResult:
        return CunxonLongSweepProbeResult(
            status="long-sweep probe viable",
            upstream_commit="upstream",
            cunxon_commit="cunxon",
            library_path="/tmp/libcunxon.so",
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            step_horizons=[32, 2048],
            seed_offsets=[79],
            samples=[
                CunxonLongSweepSample(
                    mode="train_rewarded",
                    steps=2048,
                    seed_offset=79,
                    stimulus="query-neutral-drive",
                    input_vector=[0.0, 0.0, 0.0],
                    expected_action="query",
                    readout=[0, 0, 0],
                    decoded_action="PAUSE",
                    normalized_action="query",
                    confidence=1.0,
                    outcome="success",
                    energy=2.0,
                    elapsed_ms=1.0,
                )
            ],
            accuracy_by_mode_and_steps={"train_rewarded": {"2048": 1.0}},
            unique_readouts_by_mode_and_steps={"train_rewarded": {"2048": 1}},
            action_distribution_by_mode_and_steps={"train_rewarded": {"2048": {"query": 1}}},
            baseline_accuracy={"always_query": 1.0},
            notes=["fake long sweep"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_long_sweep_probe", fake_probe)
    json_path = tmp_path / "long_sweep.json"
    markdown_path = tmp_path / "long_sweep.md"

    rc = main(
        [
            "cunxon-long-sweep",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--step-horizons",
            "32,2048",
            "--seed-offsets",
            "79",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    assert '"status": "long-sweep probe viable"' in json_path.read_text(encoding="utf-8")
    assert "longer horizons" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_action_probe_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**_: object) -> CunxonActionProbeResult:
        return CunxonActionProbeResult(
            status="task-coupled action probe viable",
            upstream_commit="upstream",
            cunxon_commit="cunxon",
            library_path="/tmp/libcunxon.so",
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            trial_steps=8,
            trials=[
                CunxonActionProbeTrial(
                    name="query-neutral-drive",
                    input_vector=[0.0, 0.0, 0.0],
                    expected_action="query",
                    readout=[0, 0, 0],
                    decoded_action="PAUSE",
                    normalized_action="query",
                    confidence=1.0,
                    outcome="success",
                    energy=2.0,
                    elapsed_ms=1.0,
                )
            ],
            success_count=1,
            accuracy=1.0,
            notes=["fake action probe"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_action_probe", fake_probe)
    json_path = tmp_path / "action_probe.json"
    markdown_path = tmp_path / "action_probe.md"

    rc = main(
        [
            "cunxon-action-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--trial-steps",
            "8",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    assert '"status": "task-coupled action probe viable"' in json_path.read_text(
        encoding="utf-8"
    )
    assert "decision-quality" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_sensitivity_probe_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**_: object) -> CunxonSensitivityProbeResult:
        return CunxonSensitivityProbeResult(
            status="sensitivity probe viable",
            upstream_commit="upstream",
            cunxon_commit="cunxon",
            library_path="/tmp/libcunxon.so",
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            steps=8,
            samples=[
                CunxonSensitivityProbeSample(
                    mode="train",
                    seed_offset=79,
                    stimulus="positive-drive",
                    input_vector=[1.0, 0.25, 0.0],
                    readout=[1, 0, 0],
                    decoded_action="PROCEED",
                    normalized_action="execute",
                    confidence=0.3333,
                    energy=2.0,
                    elapsed_ms=1.0,
                )
            ],
            unique_readouts_by_mode={"train": 1},
            action_distribution_by_mode={"train": {"execute": 1}},
            action_change_count_by_stimulus={"positive-drive": 0},
            notes=["fake sensitivity probe"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_sensitivity_probe", fake_probe)
    json_path = tmp_path / "sensitivity.json"
    markdown_path = tmp_path / "sensitivity.md"

    rc = main(
        [
            "cunxon-sensitivity-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--steps",
            "8",
            "--seed-offsets",
            "79,80",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    assert '"status": "sensitivity probe viable"' in json_path.read_text(encoding="utf-8")
    assert "infer-vs-train sensitivity probe" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_snapshot_pattern_probe_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**_: object) -> CunxonSnapshotPatternProbeResult:
        return CunxonSnapshotPatternProbeResult(
            status="snapshot-pattern probe viable",
            upstream_commit="upstream",
            cunxon_commit="cunxon",
            library_path="/tmp/libcunxon.so",
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            present_steps=4,
            settle_steps=3,
            pattern_count_after_store=2,
            pattern_count_after_clear=0,
            snapshots=[
                CunxonSnapshotObservation(
                    phase="after-finalize",
                    n_neurons=8,
                    active_state_count=0,
                    neutral_state_count=8,
                    mean_abs_membrane=0.0,
                    mean_abs_complement=0.0,
                    mean_abs_stilde=0.0,
                    mean_firing_rate=0.0,
                    mean_astrocyte=0.0,
                    energy=0.0,
                )
            ],
            recalls=[
                CunxonPatternRecallSample(
                    pattern_name="alpha",
                    mask_fraction=0.5,
                    readout=[1, 0, -1],
                    active_state_count=2,
                    signed_sum=0,
                )
            ],
            recall_hamming_distance=0,
            notes=["fake snapshot pattern"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_snapshot_pattern_probe", fake_probe)
    json_path = tmp_path / "snapshot_pattern.json"
    markdown_path = tmp_path / "snapshot_pattern.md"

    rc = main(
        [
            "cunxon-snapshot-pattern-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--present-steps",
            "4",
            "--settle-steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    assert '"status": "snapshot-pattern probe viable"' in json_path.read_text(
        encoding="utf-8"
    )
    assert "hidden-state/snapshot" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_multisphere_action_probe_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**_: object) -> CunxonMultisphereActionProbeResult:
        return CunxonMultisphereActionProbeResult(
            status="multi-sphere action probe viable",
            upstream_commit="upstream",
            cunxon_commit="cunxon",
            library_path="/tmp/libcunxon.so",
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            train_steps=4,
            eval_steps=3,
            sphere_count=3,
            cases=[
                CunxonMultisphereActionCase(
                    name="execute-holdout",
                    split="holdout",
                    input_vector=[1.0, 0.2, 0.0, 0.0],
                    expected_action="execute",
                    sensory_readout=[1, 0, 0, 0],
                    association_readout=[0, 1, 0, 0],
                    motor_readout=[1, 0, 0],
                    decoded_action="PROCEED",
                    normalized_action="execute",
                    confidence=0.3333,
                    outcome="success",
                    baseline_actions={"always_execute": "execute"},
                    energy=2.0,
                )
            ],
            accuracy_by_split={"holdout": 1.0, "overall": 1.0},
            baseline_accuracy_by_split={"always_execute": {"holdout": 1.0, "overall": 1.0}},
            notes=["fake multi-sphere action"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_multisphere_action_probe", fake_probe)
    json_path = tmp_path / "multi.json"
    markdown_path = tmp_path / "multi.md"

    rc = main(
        [
            "cunxon-multisphere-action-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--train-steps",
            "4",
            "--eval-steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    assert '"status": "multi-sphere action probe viable"' in json_path.read_text(
        encoding="utf-8"
    )
    assert "multi-sphere/action adapter" in markdown_path.read_text(encoding="utf-8")


def test_cli_no_command() -> None:
    assert main([]) == 2
