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
    CunxonAigarthActionCase,
    CunxonAigarthActionHardHoldoutResult,
    CunxonAigarthActionProbeResult,
    CunxonAigarthActionSeedRun,
    CunxonAigarthActionSeedSweepResult,
    CunxonAigarthReadoutProbeResult,
    CunxonAigarthReadoutRun,
    CunxonAigarthStressAmplitudeLadderResult,
    CunxonAigarthStressAmplitudeSplitSummary,
    CunxonAigarthStressObjectiveResult,
    CunxonAvalancheInterventionTaskConfigSummary,
    CunxonAvalancheInterventionTaskCorrelationResult,
    CunxonAvalancheWindowProbeResult,
    CunxonAvalancheWindowSample,
    CunxonBranchingRegimeRun,
    CunxonBranchingRegimeScanResult,
    CunxonControlledRegimeCalibrationConfigSummary,
    CunxonControlledRegimeCalibrationResult,
    CunxonInputProxyTargetCase,
    CunxonInputProxyTargetProbeResult,
    CunxonInterfaceReadoutSample,
    CunxonInterfaceRelaySample,
    CunxonInterfaceSemanticsProbeResult,
    CunxonLongHorizonResult,
    CunxonLongHorizonSample,
    CunxonLongSweepProbeResult,
    CunxonLongSweepSample,
    CunxonMultisphereActionCase,
    CunxonMultisphereActionProbeResult,
    CunxonPatternRecallSample,
    CunxonResidentActionCase,
    CunxonResidentActionProbeResult,
    CunxonSensitivityProbeResult,
    CunxonSensitivityProbeSample,
    CunxonSmokeResult,
    CunxonSnapshotObservation,
    CunxonSnapshotPatternProbeResult,
    CunxonVramResidentResult,
    CunxonVramResidentSample,
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


def test_cli_cunxon_vram_resident_writes_json_markdown_and_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**kwargs: object) -> CunxonVramResidentResult:
        callback = kwargs.get("artifact_callback")
        result = CunxonVramResidentResult(
            status="completed",
            hypothesis=str(kwargs["hypothesis"]),
            active_issue=str(kwargs["active_issue"]),
            pid=9876,
            command=str(kwargs["command"]),
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            started_at="2026-05-13T20:00:00Z",
            updated_at="2026-05-13T20:01:00Z",
            expected_end_at="2026-05-13T20:02:00Z",
            next_poll_after="2026-05-13T20:01:30Z",
            max_runtime_seconds=120,
            sample_interval_seconds=30,
            steps_per_sample=1024,
            total_steps=1024,
            samples=[
                CunxonVramResidentSample(
                    sample_index=1,
                    step=1024,
                    elapsed_seconds=1.0,
                    readout=[0, 0, 0],
                    active_state_count=0,
                    neutral_state_count=15,
                    energy=1.0,
                    energy_delta=1.0,
                )
            ],
            stop_condition="stop after 120 seconds or on cuNxon/API/resource error",
            notes=["fake resident run"],
        )
        if callable(callback):
            callback(result)
        return result

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_vram_resident_probe", fake_probe)
    json_path = tmp_path / "resident.json"
    markdown_path = tmp_path / "resident.md"
    state_path = tmp_path / "resident-state.json"

    rc = main(
        [
            "cunxon-vram-resident",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--hypothesis",
            "wall-clock VRAM residence changes dynamics",
            "--max-runtime-seconds",
            "120",
            "--sample-interval-seconds",
            "30",
            "--steps-per-sample",
            "1024",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
            "--state-output",
            str(state_path),
        ]
    )

    assert rc == 0
    assert '"status": "completed"' in json_path.read_text(encoding="utf-8")
    assert "VRAM-resident" in markdown_path.read_text(encoding="utf-8")
    assert '"pid": 9876' in state_path.read_text(encoding="utf-8")


def test_cli_cunxon_resident_action_probe_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**kwargs: object) -> CunxonResidentActionProbeResult:
        return CunxonResidentActionProbeResult(
            status="resident action probe viable",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            train_epochs=int(kwargs["train_epochs"]),
            train_steps_per_case=int(kwargs["train_steps_per_case"]),
            eval_steps=int(kwargs["eval_steps"]),
            sphere_count=3,
            cases=[
                CunxonResidentActionCase(
                    epoch=1,
                    name="query-holdout-low-drive",
                    split="holdout",
                    input_vector=[0.05, 0.0, -0.05, 0.0],
                    expected_action="query",
                    sensory_readout=[0, 0, 0, 0],
                    association_readout=[0, 0, 0, 0],
                    motor_readout=[0, 0, 0],
                    decoded_action="query",
                    normalized_action="query",
                    confidence=0.0,
                    outcome="success",
                    baseline_actions={"always_query": "query"},
                    energy=1.0,
                    motor_active_state_count=0,
                    elapsed_ms=1.0,
                )
            ],
            accuracy_by_epoch={"1": 1.0},
            accuracy_by_split={"holdout": 1.0, "overall": 1.0},
            baseline_accuracy_by_split={"always_query": {"holdout": 1.0, "overall": 1.0}},
            unique_motor_readouts=1,
            notes=["fake resident action probe"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_resident_action_probe", fake_probe)
    json_path = tmp_path / "resident-action.json"
    markdown_path = tmp_path / "resident-action.md"

    rc = main(
        [
            "cunxon-resident-action-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--train-epochs",
            "2",
            "--train-steps-per-case",
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
    assert '"status": "resident action probe viable"' in json_path.read_text(encoding="utf-8")
    assert "resident task-coupled action probe" in markdown_path.read_text(encoding="utf-8")


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
    assert '"status": "task-coupled action probe viable"' in json_path.read_text(encoding="utf-8")
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
    assert '"status": "snapshot-pattern probe viable"' in json_path.read_text(encoding="utf-8")
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
    assert '"status": "multi-sphere action probe viable"' in json_path.read_text(encoding="utf-8")
    assert "multi-sphere/action adapter" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_interface_semantics_probe_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**_: object) -> CunxonInterfaceSemanticsProbeResult:
        return CunxonInterfaceSemanticsProbeResult(
            status="interface semantics probe viable",
            upstream_commit="upstream",
            cunxon_commit="cunxon",
            library_path="/tmp/libcunxon.so",
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            steps=3,
            readout_samples=[
                CunxonInterfaceReadoutSample(
                    mapping="relative-input-readout",
                    port_ids=[0, 1, 2],
                    neuron_class="input",
                    readout=[1, -1, 1],
                    snapshot_slice=[1, -1, 1],
                    matches_snapshot_slice=True,
                    active_state_count=3,
                    signed_sum=1,
                )
            ],
            relay_samples=[
                CunxonInterfaceRelaySample(
                    mapping="input-neuron-relay",
                    source_port_ids=[0, 1, 2, 3],
                    source_neuron_class="input",
                    source_relay_readout=[1, -1, 0, 1],
                    downstream_input_readout=[0, -1, 0, 0],
                    downstream_active_state_count=1,
                    downstream_energy=4.0,
                )
            ],
            notes=["fake interface semantics"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_interface_semantics_probe", fake_probe)
    json_path = tmp_path / "interface.json"
    markdown_path = tmp_path / "interface.md"

    rc = main(
        [
            "cunxon-interface-semantics-probe",
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
    assert '"status": "interface semantics probe viable"' in json_path.read_text(encoding="utf-8")
    assert "absolute neuron indices" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_input_proxy_target_probe_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**_: object) -> CunxonInputProxyTargetProbeResult:
        return CunxonInputProxyTargetProbeResult(
            status="input-proxy target probe viable",
            upstream_commit="upstream",
            cunxon_commit="cunxon",
            library_path="/tmp/libcunxon.so",
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            train_epochs=2,
            train_steps_per_case=4,
            eval_steps=4,
            target_proxy_port_ids=[3, 4, 5],
            motor_readout_port_ids=[11, 12, 13],
            cases=[
                CunxonInputProxyTargetCase(
                    name="execute-train",
                    split="train",
                    input_vector=[1.0, 0.25, 0.0],
                    expected_action="execute",
                    target_proxy_readout=[1, -1, -1],
                    motor_readout=[0, 0, 0],
                    decoded_action="query",
                    normalized_action="query",
                    confidence=0.0,
                    outcome="failure",
                    target_proxy_alignment=1.0,
                    baseline_actions={"always_execute": "execute", "always_query": "query"},
                    energy=4.0,
                )
            ],
            accuracy_by_split={"overall": 0.0, "train": 0.0},
            target_proxy_alignment_by_split={"overall": 1.0, "train": 1.0},
            baseline_accuracy_by_split={
                "always_execute": {"overall": 1.0, "train": 1.0},
                "always_query": {"overall": 0.0, "train": 0.0},
            },
            notes=["fake input proxy target"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_input_proxy_target_probe", fake_probe)
    json_path = tmp_path / "input-proxy.json"
    markdown_path = tmp_path / "input-proxy.md"

    rc = main(
        [
            "cunxon-input-proxy-target-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--train-epochs",
            "2",
            "--train-steps-per-case",
            "4",
            "--eval-steps",
            "4",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    assert '"status": "input-proxy target probe viable"' in json_path.read_text(encoding="utf-8")
    assert "input-port proxy target probe" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_aigarth_readout_probe_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**kwargs: object) -> CunxonAigarthReadoutProbeResult:
        return CunxonAigarthReadoutProbeResult(
            status="aigarth readout semantics probe viable",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            generations=int(str(kwargs["generations"])),
            population_size=int(str(kwargs["population_size"])),
            eval_steps=int(str(kwargs["eval_steps"])),
            runs=[
                CunxonAigarthReadoutRun(
                    mapping="absolute-output-readout",
                    readout_ids=[36, 37, 38, 39, 40, 41, 42, 43],
                    neuron_class="absolute output block",
                    baseline_margin=0.0,
                    generation_margins=[0.0],
                    final_margin=0.0,
                    improvement=0.0,
                    positive_mean=0.0,
                    negative_mean=0.0,
                    positive_readout=[0, 0, 0, 0, 0, 0, 0, 0],
                    negative_readout=[0, 0, 0, 0, 0, 0, 0, 0],
                )
            ],
            notes=["fake Aigarth readout probe"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_aigarth_readout_probe", fake_probe)
    json_path = tmp_path / "aigarth.json"
    markdown_path = tmp_path / "aigarth.md"

    rc = main(
        [
            "cunxon-aigarth-readout-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--eval-steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    assert '"status": "aigarth readout semantics probe viable"' in json_path.read_text(
        encoding="utf-8"
    )
    assert "Aigarth readout semantics" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_aigarth_action_probe_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**kwargs: object) -> CunxonAigarthActionProbeResult:
        return CunxonAigarthActionProbeResult(
            status="aigarth action probe viable",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            generations=int(str(kwargs["generations"])),
            population_size=int(str(kwargs["population_size"])),
            eval_steps=int(str(kwargs["eval_steps"])),
            readout_ids=[35, 36, 37],
            generation_train_scores=[0.333333],
            cases=[
                CunxonAigarthActionCase(
                    name="query-holdout-low-drive",
                    split="holdout",
                    input_vector=[0.05, 0.0, -0.05],
                    expected_action="query",
                    target_readout=[0, 0, 0],
                    readout=[0, 0, 0],
                    decoded_action="query",
                    normalized_action="query",
                    confidence=0.0,
                    outcome="success",
                    target_alignment=1.0,
                    baseline_actions={"always_query": "query"},
                    energy=1.0,
                )
            ],
            accuracy_by_split={"holdout": 1.0, "overall": 1.0},
            target_alignment_by_split={"holdout": 1.0, "overall": 1.0},
            baseline_accuracy_by_split={"always_query": {"holdout": 1.0, "overall": 1.0}},
            unique_readouts=1,
            action_distribution={"query": 1},
            notes=["fake Aigarth action probe"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_aigarth_action_probe", fake_probe)
    json_path = tmp_path / "aigarth-action.json"
    markdown_path = tmp_path / "aigarth-action.md"

    rc = main(
        [
            "cunxon-aigarth-action-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--eval-steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    assert '"status": "aigarth action probe viable"' in json_path.read_text(encoding="utf-8")
    assert "Aigarth holdout action probe" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_aigarth_action_seed_sweep_probe_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**kwargs: object) -> CunxonAigarthActionSeedSweepResult:
        seed_offsets = list(kwargs["seed_offsets"])
        return CunxonAigarthActionSeedSweepResult(
            status="aigarth action seed sweep completed",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            generations=int(str(kwargs["generations"])),
            population_size=int(str(kwargs["population_size"])),
            eval_steps=int(str(kwargs["eval_steps"])),
            readout_ids=[35, 36, 37],
            seed_offsets=seed_offsets,
            runs=[
                CunxonAigarthActionSeedRun(
                    seed_offset=seed_offsets[0],
                    generation_train_scores=[0.333333],
                    accuracy_by_split={"holdout": 0.333333, "overall": 0.333333},
                    target_alignment_by_split={"holdout": 0.333333, "overall": 0.333333},
                    baseline_accuracy_by_split={
                        "always_query": {"holdout": 0.333333, "overall": 0.333333}
                    },
                    unique_readouts=1,
                    action_distribution={"query": 6},
                    cases=[],
                )
            ],
            accuracy_summary_by_split={
                "holdout": {"mean": 0.333333, "min": 0.333333, "max": 0.333333},
                "overall": {"mean": 0.333333, "min": 0.333333, "max": 0.333333},
            },
            aggregate_action_distribution={"query": 6},
            seeds_beating_baseline_by_split={"holdout": 0, "overall": 0},
            notes=["fake seed sweep"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_aigarth_action_seed_sweep_probe", fake_probe)
    json_path = tmp_path / "aigarth-action-seed-sweep.json"
    markdown_path = tmp_path / "aigarth-action-seed-sweep.md"

    rc = main(
        [
            "cunxon-aigarth-action-seed-sweep-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--eval-steps",
            "3",
            "--seed-offsets",
            "82,83",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    assert '"status": "aigarth action seed sweep completed"' in json_path.read_text(
        encoding="utf-8"
    )
    assert "Aigarth action seed sweep" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_aigarth_action_hard_holdout_probe_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**kwargs: object) -> CunxonAigarthActionHardHoldoutResult:
        raw_seed_offsets = kwargs["seed_offsets"]
        assert isinstance(raw_seed_offsets, list)
        seed_offsets = raw_seed_offsets
        return CunxonAigarthActionHardHoldoutResult(
            status="aigarth hard-holdout audit completed",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            generations=int(str(kwargs["generations"])),
            population_size=int(str(kwargs["population_size"])),
            eval_steps=int(str(kwargs["eval_steps"])),
            readout_ids=[35, 36, 37],
            seed_offsets=seed_offsets,
            strict_expected_actions=["execute", "query", "retry"],
            runs=[
                CunxonAigarthActionSeedRun(
                    seed_offset=seed_offsets[0],
                    generation_train_scores=[0.333333],
                    accuracy_by_split={"hard_holdout": 0.333333, "overall": 0.333333},
                    target_alignment_by_split={"hard_holdout": 0.333333, "overall": 0.333333},
                    baseline_accuracy_by_split={
                        "always_query": {"hard_holdout": 0.333333, "overall": 0.333333}
                    },
                    unique_readouts=1,
                    action_distribution={"query": 6},
                    cases=[],
                )
            ],
            accuracy_summary_by_split={
                "hard_holdout": {"mean": 0.333333, "min": 0.333333, "max": 0.333333},
                "overall": {"mean": 0.333333, "min": 0.333333, "max": 0.333333},
            },
            aggregate_action_distribution={"query": 6},
            seeds_beating_baseline_by_split={"hard_holdout": 0, "overall": 0},
            unexpected_action_count=0,
            unexpected_action_rate=0.0,
            leakage_control_accuracy_mean=0.0,
            train_to_hard_holdout_gap_mean=0.0,
            notes=["fake hard holdout audit"],
        )

    monkeypatch.setattr(
        "neuraxon_agent.cli.run_ctypes_aigarth_action_hard_holdout_probe", fake_probe
    )
    json_path = tmp_path / "aigarth-action-hard-holdout.json"
    markdown_path = tmp_path / "aigarth-action-hard-holdout.md"

    rc = main(
        [
            "cunxon-aigarth-action-hard-holdout-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--seed-offsets",
            "87,88",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--eval-steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    assert '"status": "aigarth hard-holdout audit completed"' in json_path.read_text(
        encoding="utf-8"
    )
    assert "Aigarth action hard-holdout audit" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_aigarth_action_strict_label_probe_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**kwargs: object) -> CunxonAigarthActionHardHoldoutResult:
        raw_seed_offsets = kwargs["seed_offsets"]
        assert isinstance(raw_seed_offsets, list)
        seed_offsets = raw_seed_offsets
        return CunxonAigarthActionHardHoldoutResult(
            status="aigarth strict-label action audit completed",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            generations=int(str(kwargs["generations"])),
            population_size=int(str(kwargs["population_size"])),
            eval_steps=int(str(kwargs["eval_steps"])),
            readout_ids=[35, 36, 37],
            seed_offsets=seed_offsets,
            strict_expected_actions=["execute", "query", "retry"],
            runs=[
                CunxonAigarthActionSeedRun(
                    seed_offset=seed_offsets[0],
                    generation_train_scores=[0.333333],
                    accuracy_by_split={"hard_holdout": 0.333333, "overall": 0.333333},
                    target_alignment_by_split={"hard_holdout": 0.333333, "overall": 0.333333},
                    baseline_accuracy_by_split={
                        "always_query": {"hard_holdout": 0.333333, "overall": 0.333333}
                    },
                    unique_readouts=1,
                    action_distribution={"query": 6},
                    cases=[],
                )
            ],
            accuracy_summary_by_split={
                "hard_holdout": {"mean": 0.333333, "min": 0.333333, "max": 0.333333},
                "overall": {"mean": 0.333333, "min": 0.333333, "max": 0.333333},
            },
            aggregate_action_distribution={"query": 6},
            seeds_beating_baseline_by_split={"hard_holdout": 0, "overall": 0},
            unexpected_action_count=0,
            unexpected_action_rate=0.0,
            leakage_control_accuracy_mean=0.0,
            train_to_hard_holdout_gap_mean=0.0,
            fitness_variant=str(kwargs["fitness_variant"]),
            notes=["fake strict-label audit"],
        )

    monkeypatch.setattr(
        "neuraxon_agent.cli.run_ctypes_aigarth_action_strict_label_probe", fake_probe
    )
    json_path = tmp_path / "aigarth-action-strict-label.json"
    markdown_path = tmp_path / "aigarth-action-strict-label.md"

    rc = main(
        [
            "cunxon-aigarth-action-strict-label-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--seed-offsets",
            "92,93",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--eval-steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    data = json_path.read_text(encoding="utf-8")
    assert '"status": "aigarth strict-label action audit completed"' in data
    assert '"fitness_variant": "strict_label_margin"' in data
    assert "Aigarth strict-label action audit" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_aigarth_action_contract_penalty_probe_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**kwargs: object) -> CunxonAigarthActionHardHoldoutResult:
        raw_seed_offsets = kwargs["seed_offsets"]
        assert isinstance(raw_seed_offsets, list)
        assert kwargs["fitness_variant"] == "strict_label_heavy_penalty"
        return CunxonAigarthActionHardHoldoutResult(
            status="aigarth contract-penalty action audit completed",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            generations=int(str(kwargs["generations"])),
            population_size=int(str(kwargs["population_size"])),
            eval_steps=int(str(kwargs["eval_steps"])),
            readout_ids=[35, 36, 37],
            seed_offsets=raw_seed_offsets,
            strict_expected_actions=["execute", "query", "retry"],
            runs=[
                CunxonAigarthActionSeedRun(
                    seed_offset=97,
                    generation_train_scores=[0.0, 0.5],
                    accuracy_by_split={"hard_holdout": 0.333333, "overall": 0.333333},
                    target_alignment_by_split={"hard_holdout": 0.333333, "overall": 0.333333},
                    baseline_accuracy_by_split={
                        "always_query": {"hard_holdout": 0.333333, "overall": 0.333333}
                    },
                    unique_readouts=1,
                    action_distribution={"query": 6},
                    cases=[],
                )
            ],
            accuracy_summary_by_split={
                "hard_holdout": {"mean": 0.333333, "min": 0.333333, "max": 0.333333},
                "overall": {"mean": 0.333333, "min": 0.333333, "max": 0.333333},
            },
            aggregate_action_distribution={"query": 6},
            seeds_beating_baseline_by_split={"hard_holdout": 0, "overall": 0},
            unexpected_action_count=0,
            unexpected_action_rate=0.0,
            leakage_control_accuracy_mean=0.0,
            train_to_hard_holdout_gap_mean=0.0,
            fitness_variant=str(kwargs["fitness_variant"]),
            notes=["fake contract-penalty audit"],
        )

    monkeypatch.setattr(
        "neuraxon_agent.cli.run_ctypes_aigarth_action_contract_penalty_probe", fake_probe
    )
    json_path = tmp_path / "aigarth-action-contract-penalty.json"
    markdown_path = tmp_path / "aigarth-action-contract-penalty.md"

    rc = main(
        [
            "cunxon-aigarth-action-contract-penalty-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--seed-offsets",
            "97,98",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--eval-steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    data = json_path.read_text(encoding="utf-8")
    assert '"status": "aigarth contract-penalty action audit completed"' in data
    assert '"fitness_variant": "strict_label_heavy_penalty"' in data
    assert "Aigarth contract-penalty action audit" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_aigarth_action_target_contract_probe_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(**kwargs: object) -> CunxonAigarthActionHardHoldoutResult:
        raw_seed_offsets = kwargs["seed_offsets"]
        assert isinstance(raw_seed_offsets, list)
        assert kwargs["fitness_variant"] == "target_contract_margin"
        return CunxonAigarthActionHardHoldoutResult(
            status="aigarth target-contract action audit completed",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            generations=int(str(kwargs["generations"])),
            population_size=int(str(kwargs["population_size"])),
            eval_steps=int(str(kwargs["eval_steps"])),
            readout_ids=[35, 36, 37],
            seed_offsets=raw_seed_offsets,
            strict_expected_actions=["execute", "query", "retry"],
            runs=[
                CunxonAigarthActionSeedRun(
                    seed_offset=102,
                    generation_train_scores=[0.0, 0.5],
                    accuracy_by_split={"hard_holdout": 0.666667, "overall": 0.666667},
                    target_alignment_by_split={"hard_holdout": 0.666667, "overall": 0.666667},
                    baseline_accuracy_by_split={
                        "always_query": {"hard_holdout": 0.333333, "overall": 0.333333}
                    },
                    unique_readouts=2,
                    action_distribution={"execute": 2, "query": 2, "retry": 2},
                    cases=[],
                )
            ],
            accuracy_summary_by_split={
                "hard_holdout": {"mean": 0.666667, "min": 0.666667, "max": 0.666667},
                "overall": {"mean": 0.666667, "min": 0.666667, "max": 0.666667},
            },
            aggregate_action_distribution={"execute": 2, "query": 2, "retry": 2},
            seeds_beating_baseline_by_split={"hard_holdout": 1, "overall": 1},
            unexpected_action_count=0,
            unexpected_action_rate=0.0,
            leakage_control_accuracy_mean=0.0,
            train_to_hard_holdout_gap_mean=0.333333,
            fitness_variant=str(kwargs["fitness_variant"]),
            notes=["fake target-contract audit"],
        )

    monkeypatch.setattr(
        "neuraxon_agent.cli.run_ctypes_aigarth_action_target_contract_probe", fake_probe
    )
    json_path = tmp_path / "aigarth-action-target-contract.json"
    markdown_path = tmp_path / "aigarth-action-target-contract.md"

    rc = main(
        [
            "cunxon-aigarth-action-target-contract-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--seed-offsets",
            "102,103",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--eval-steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    data = json_path.read_text(encoding="utf-8")
    assert '"status": "aigarth target-contract action audit completed"' in data
    assert '"fitness_variant": "target_contract_margin"' in data
    assert "Aigarth target-contract action audit" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_aigarth_action_target_contract_stress_probe_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_probe(**kwargs: object) -> CunxonAigarthActionHardHoldoutResult:
        raw_seed_offsets = kwargs["seed_offsets"]
        assert isinstance(raw_seed_offsets, list)
        assert raw_seed_offsets == [107, 108]
        assert kwargs["fitness_variant"] == "target_contract_margin"
        return CunxonAigarthActionHardHoldoutResult(
            status="aigarth target-contract stress audit completed",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            generations=int(str(kwargs["generations"])),
            population_size=int(str(kwargs["population_size"])),
            eval_steps=int(str(kwargs["eval_steps"])),
            readout_ids=[35, 36, 37],
            seed_offsets=raw_seed_offsets,
            strict_expected_actions=["execute", "query", "retry"],
            runs=[
                CunxonAigarthActionSeedRun(
                    seed_offset=107,
                    generation_train_scores=[0.0, 0.5],
                    accuracy_by_split={
                        "stress_holdout": 0.25,
                        "counterfactual_control": 0.0,
                        "overall": 0.5,
                    },
                    target_alignment_by_split={"stress_holdout": 0.333333},
                    baseline_accuracy_by_split={
                        "always_query": {"stress_holdout": 0.333333, "overall": 0.333333}
                    },
                    unique_readouts=2,
                    action_distribution={"execute": 2, "query": 2, "retry": 2},
                    cases=[],
                )
            ],
            accuracy_summary_by_split={
                "stress_holdout": {"mean": 0.25, "min": 0.25, "max": 0.25},
                "counterfactual_control": {"mean": 0.0, "min": 0.0, "max": 0.0},
                "overall": {"mean": 0.5, "min": 0.5, "max": 0.5},
            },
            aggregate_action_distribution={"execute": 2, "query": 2, "retry": 2},
            seeds_beating_baseline_by_split={
                "stress_holdout": 0,
                "counterfactual_control": 0,
                "overall": 1,
            },
            unexpected_action_count=0,
            unexpected_action_rate=0.0,
            leakage_control_accuracy_mean=0.0,
            train_to_hard_holdout_gap_mean=0.0,
            fitness_variant=str(kwargs["fitness_variant"]),
            notes=["fake target-contract stress audit"],
        )

    monkeypatch.setattr(
        "neuraxon_agent.cli.run_ctypes_aigarth_action_target_contract_stress_probe",
        fake_probe,
    )
    json_path = tmp_path / "aigarth-action-target-contract-stress.json"
    markdown_path = tmp_path / "aigarth-action-target-contract-stress.md"

    rc = main(
        [
            "cunxon-aigarth-action-target-contract-stress-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--seed-offsets",
            "107,108",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--eval-steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    data = json_path.read_text(encoding="utf-8")
    assert '"status": "aigarth target-contract stress audit completed"' in data
    assert '"fitness_variant": "target_contract_margin"' in data
    assert "Aigarth target-contract stress audit" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_aigarth_action_target_contract_augmented_train_probe_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_probe(**kwargs: object) -> CunxonAigarthActionHardHoldoutResult:
        raw_seed_offsets = kwargs["seed_offsets"]
        assert isinstance(raw_seed_offsets, list)
        assert raw_seed_offsets == [112, 113]
        assert kwargs["fitness_variant"] == "target_contract_augmented_train"
        return CunxonAigarthActionHardHoldoutResult(
            status="aigarth target-contract augmented-train audit completed",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            generations=int(str(kwargs["generations"])),
            population_size=int(str(kwargs["population_size"])),
            eval_steps=int(str(kwargs["eval_steps"])),
            readout_ids=[35, 36, 37],
            seed_offsets=raw_seed_offsets,
            strict_expected_actions=["execute", "query", "retry"],
            runs=[
                CunxonAigarthActionSeedRun(
                    seed_offset=112,
                    generation_train_scores=[0.0, 0.5],
                    accuracy_by_split={
                        "augmented_train": 1.0,
                        "stress_holdout": 0.25,
                        "overall": 0.5,
                    },
                    target_alignment_by_split={"augmented_train": 0.9},
                    baseline_accuracy_by_split={
                        "always_query": {"augmented_train": 0.333333, "overall": 0.333333}
                    },
                    unique_readouts=2,
                    action_distribution={"execute": 2, "query": 2, "retry": 2},
                    cases=[],
                )
            ],
            accuracy_summary_by_split={
                "augmented_train": {"mean": 1.0, "min": 1.0, "max": 1.0},
                "stress_holdout": {"mean": 0.25, "min": 0.25, "max": 0.25},
                "overall": {"mean": 0.5, "min": 0.5, "max": 0.5},
            },
            aggregate_action_distribution={"execute": 2, "query": 2, "retry": 2},
            seeds_beating_baseline_by_split={
                "augmented_train": 1,
                "stress_holdout": 0,
                "overall": 1,
            },
            unexpected_action_count=0,
            unexpected_action_rate=0.0,
            leakage_control_accuracy_mean=0.0,
            train_to_hard_holdout_gap_mean=0.0,
            fitness_variant=str(kwargs["fitness_variant"]),
            notes=["fake target-contract augmented-train audit"],
        )

    monkeypatch.setattr(
        "neuraxon_agent.cli.run_ctypes_aigarth_action_target_contract_augmented_train_probe",
        fake_probe,
    )
    json_path = tmp_path / "aigarth-action-target-contract-augmented-train.json"
    markdown_path = tmp_path / "aigarth-action-target-contract-augmented-train.md"

    rc = main(
        [
            "cunxon-aigarth-action-target-contract-augmented-train-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--seed-offsets",
            "112,113",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--eval-steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    data = json_path.read_text(encoding="utf-8")
    assert '"status": "aigarth target-contract augmented-train audit completed"' in data
    assert '"fitness_variant": "target_contract_augmented_train"' in data
    assert "Aigarth target-contract augmented-train audit" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_cli_cunxon_aigarth_action_target_contract_stress_amplitude_ladder_probe_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_probe(**kwargs: object) -> CunxonAigarthStressAmplitudeLadderResult:
        raw_seed_offsets = kwargs["seed_offsets"]
        raw_amplitudes = kwargs["amplitude_factors"]
        assert isinstance(raw_seed_offsets, list)
        assert isinstance(raw_amplitudes, list)
        assert raw_seed_offsets == [142, 143]
        assert raw_amplitudes == [1.0, 2.0]
        return CunxonAigarthStressAmplitudeLadderResult(
            status="aigarth target-contract stress amplitude-ladder completed",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            generations=int(str(kwargs["generations"])),
            population_size=int(str(kwargs["population_size"])),
            eval_steps=int(str(kwargs["eval_steps"])),
            seed_offsets=raw_seed_offsets,
            amplitude_factors=raw_amplitudes,
            split_summaries=[
                CunxonAigarthStressAmplitudeSplitSummary(
                    split="stress_holdout_scaled_2_0x",
                    amplitude_factor=2.0,
                    sample_count=12,
                    accuracy_mean=0.5,
                    best_constant_baseline_mean=0.333333,
                    seeds_beating_best_baseline=1,
                    query_collapse_rate=0.5,
                    execute_retry_accuracy=0.25,
                    action_distribution={"execute": 3, "query": 6, "retry": 3},
                )
            ],
            original_stress_holdout_accuracy_mean=0.333333,
            original_stress_holdout_query_collapse_rate=1.0,
            best_scaled_stress_holdout_accuracy_mean=0.5,
            best_scaled_stress_holdout_amplitude_factor=2.0,
            aggregate_action_distribution={"execute": 3, "query": 6, "retry": 3},
            evidence_boundary=(
                "This is a label-injected separability upper-bound, not intelligence evidence."
            ),
            recommended_next_probe={"id": "target_aligned_stress_objective_followup"},
            notes=["fake stress amplitude ladder"],
        )

    monkeypatch.setattr(
        "neuraxon_agent.cli.run_ctypes_aigarth_action_target_contract_stress_amplitude_ladder_probe",
        fake_probe,
    )
    json_path = tmp_path / "aigarth-action-target-contract-stress-amplitude-ladder.json"
    markdown_path = tmp_path / "aigarth-action-target-contract-stress-amplitude-ladder.md"

    rc = main(
        [
            "cunxon-aigarth-action-target-contract-stress-amplitude-ladder-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--seed-offsets",
            "142,143",
            "--amplitude-factors",
            "1.0,2.0",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--eval-steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["status"] == "aigarth target-contract stress amplitude-ladder completed"
    assert data["amplitude_factors"] == [1.0, 2.0]
    assert "stress amplitude-ladder" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_stress_objective_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_probe(**kwargs: object) -> CunxonAigarthStressObjectiveResult:
        raw_seed_offsets = kwargs["seed_offsets"]
        assert isinstance(raw_seed_offsets, list)
        assert raw_seed_offsets == [147, 148]
        assert kwargs["fitness_variant"] == "target_contract_stress_margin_weighted"
        assert kwargs["amplitude_factor"] == 3.0
        return CunxonAigarthStressObjectiveResult(
            status="aigarth target-contract stress objective completed",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            generations=int(str(kwargs["generations"])),
            population_size=int(str(kwargs["population_size"])),
            eval_steps=int(str(kwargs["eval_steps"])),
            seed_offsets=raw_seed_offsets,
            amplitude_factor=float(str(kwargs["amplitude_factor"])),
            fitness_variant=str(kwargs["fitness_variant"]),
            split_summaries=[
                CunxonAigarthStressAmplitudeSplitSummary(
                    split="stress_holdout_scaled_3_0x",
                    amplitude_factor=3.0,
                    sample_count=12,
                    accuracy_mean=0.833333,
                    best_constant_baseline_mean=0.333333,
                    seeds_beating_best_baseline=2,
                    query_collapse_rate=0.333333,
                    execute_retry_accuracy=0.75,
                    action_distribution={"execute": 4, "query": 4, "retry": 4},
                )
            ],
            original_stress_holdout_accuracy_mean=0.333333,
            original_stress_holdout_query_collapse_rate=1.0,
            original_stress_holdout_execute_retry_accuracy=0.0,
            scaled_stress_holdout_accuracy_mean=0.833333,
            scaled_stress_holdout_query_collapse_rate=0.333333,
            scaled_stress_holdout_execute_retry_accuracy=0.75,
            counterfactual_control_accuracy_mean=0.333333,
            permuted_control_accuracy_mean=0.333333,
            aggregate_action_distribution={"execute": 4, "query": 4, "retry": 4},
            evidence_boundary=(
                "This target-aligned stress objective is diagnostic, not intelligence evidence."
            ),
            recommended_next_probe={"id": "stress_objective_decoder_geometry_followup"},
            notes=["fake stress objective"],
        )

    monkeypatch.setattr(
        "neuraxon_agent.cli.run_ctypes_aigarth_action_target_contract_stress_objective_probe",
        fake_probe,
    )
    json_path = tmp_path / "aigarth-action-target-contract-stress-objective.json"
    markdown_path = tmp_path / "aigarth-action-target-contract-stress-objective.md"

    rc = main(
        [
            "cunxon-aigarth-action-target-contract-stress-objective-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--seed-offsets",
            "147,148",
            "--amplitude-factor",
            "3.0",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--eval-steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["status"] == "aigarth target-contract stress objective completed"
    assert data["fitness_variant"] == "target_contract_stress_margin_weighted"
    assert "stress objective" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_supervised_low_margin_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_probe(**kwargs: object) -> CunxonAigarthStressObjectiveResult:
        raw_seed_offsets = kwargs["seed_offsets"]
        assert isinstance(raw_seed_offsets, list)
        assert raw_seed_offsets == [150, 151]
        assert kwargs["fitness_variant"] == "target_contract_supervised_low_margin"
        return CunxonAigarthStressObjectiveResult(
            status="aigarth target-contract supervised low-margin objective completed",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            generations=int(str(kwargs["generations"])),
            population_size=int(str(kwargs["population_size"])),
            eval_steps=int(str(kwargs["eval_steps"])),
            seed_offsets=raw_seed_offsets,
            amplitude_factor=1.0,
            fitness_variant=str(kwargs["fitness_variant"]),
            split_summaries=[
                CunxonAigarthStressAmplitudeSplitSummary(
                    split="supervised_low_margin_train",
                    amplitude_factor=1.0,
                    sample_count=12,
                    accuracy_mean=1.0,
                    best_constant_baseline_mean=0.333333,
                    seeds_beating_best_baseline=2,
                    query_collapse_rate=0.0,
                    execute_retry_accuracy=1.0,
                    action_distribution={"execute": 4, "query": 4, "retry": 4},
                ),
                CunxonAigarthStressAmplitudeSplitSummary(
                    split="stress_holdout",
                    amplitude_factor=1.0,
                    sample_count=12,
                    accuracy_mean=0.333333,
                    best_constant_baseline_mean=0.333333,
                    seeds_beating_best_baseline=0,
                    query_collapse_rate=1.0,
                    execute_retry_accuracy=0.0,
                    action_distribution={"query": 12},
                ),
            ],
            original_stress_holdout_accuracy_mean=0.333333,
            original_stress_holdout_query_collapse_rate=1.0,
            original_stress_holdout_execute_retry_accuracy=0.0,
            scaled_stress_holdout_accuracy_mean=1.0,
            scaled_stress_holdout_query_collapse_rate=0.0,
            scaled_stress_holdout_execute_retry_accuracy=1.0,
            counterfactual_control_accuracy_mean=0.333333,
            permuted_control_accuracy_mean=0.333333,
            aggregate_action_distribution={"execute": 4, "query": 16, "retry": 4},
            evidence_boundary=(
                "This supervised low-margin diagnostic is not intelligence evidence."
            ),
            recommended_next_probe={"id": "low_margin_target_objective_decision"},
            notes=["fake supervised low-margin objective"],
        )

    monkeypatch.setattr(
        "neuraxon_agent.cli.run_ctypes_aigarth_action_target_contract_supervised_low_margin_probe",
        fake_probe,
    )
    json_path = tmp_path / "supervised-low-margin.json"
    markdown_path = tmp_path / "supervised-low-margin.md"

    rc = main(
        [
            "cunxon-aigarth-action-target-contract-supervised-low-margin-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--seed-offsets",
            "150,151",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--eval-steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["status"] == "aigarth target-contract supervised low-margin objective completed"
    assert data["fitness_variant"] == "target_contract_supervised_low_margin"
    assert "supervised low-margin objective" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_branching_regime_scan_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_probe(**kwargs: object) -> CunxonBranchingRegimeScanResult:
        raw_seed_offsets = kwargs["seed_offsets"]
        assert isinstance(raw_seed_offsets, list)
        assert raw_seed_offsets == [117, 118]
        assert kwargs["source_probe"] == "target_contract_augmented_train"
        return CunxonBranchingRegimeScanResult(
            status="branching-regime scan completed",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            source_probe=str(kwargs["source_probe"]),
            generations=int(str(kwargs["generations"])),
            population_size=int(str(kwargs["population_size"])),
            eval_steps=int(str(kwargs["eval_steps"])),
            seed_offsets=raw_seed_offsets,
            runs=[
                CunxonBranchingRegimeRun(
                    seed_offset=117,
                    active_state_sequence=[1, 1, 2],
                    branching_activity_ratio_proxy=1.25,
                    neutral_occupancy=0.555556,
                    transition_entropy_bits=0.918296,
                    energy_slope=0.5,
                    readout_diversity=2,
                    regime_label="reverberating/near-critical proxy",
                    accuracy_by_split={"holdout": 1.0, "stress_holdout": 0.333333},
                    best_constant_baseline_by_split={
                        "holdout": 0.333333,
                        "stress_holdout": 0.333333,
                    },
                    beats_best_baseline_by_split={"holdout": True, "stress_holdout": False},
                    action_distribution={"execute": 1, "query": 1, "retry": 1},
                )
            ],
            regime_bucket_summary={
                "reverberating/near-critical proxy": {
                    "seed_count": 1,
                    "mean_branching_activity_ratio_proxy": 1.25,
                    "mean_holdout_accuracy": 1.0,
                    "mean_stress_holdout_accuracy": 0.333333,
                    "beats_stress_baseline_count": 0,
                }
            },
            correlation_summary={"stress_holdout_accuracy_vs_branching_proxy": 0.0},
            verdict="Near-critical proxy did not beat stress_holdout baselines.",
            notes=["fake branching regime scan"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_branching_regime_scan_probe", fake_probe)
    json_path = tmp_path / "branching-regime-scan.json"
    markdown_path = tmp_path / "branching-regime-scan.md"

    rc = main(
        [
            "cunxon-branching-regime-scan",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--seed-offsets",
            "117,118",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--eval-steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    data = json_path.read_text(encoding="utf-8")
    assert '"status": "branching-regime scan completed"' in data
    assert '"branching_activity_ratio_proxy": 1.25' in data
    assert "cuNxon branching-ratio regime scan" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_avalanche_window_probe_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_probe(**kwargs: object) -> CunxonAvalancheWindowProbeResult:
        assert kwargs["modes"] == ["infer", "train"]
        assert kwargs["seed_offsets"] == [122, 123]
        return CunxonAvalancheWindowProbeResult(
            status="avalanche-window probe completed",
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            modes=list(kwargs["modes"]),
            seed_offsets=list(kwargs["seed_offsets"]),
            steps=int(str(kwargs["steps"])),
            sample_interval=int(str(kwargs["sample_interval"])),
            samples=[
                CunxonAvalancheWindowSample(
                    mode="infer",
                    seed_offset=122,
                    stimulus="execute-positive-drive",
                    input_vector=[1.0, 0.25, 0.0],
                    expected_action="execute",
                    active_state_sequence=[1, 2],
                    activation_event_sequence=[1, 1],
                    deactivation_event_sequence=[0, 0],
                    branching_ratio_estimate=1.0,
                    active_count_ratio_mean=2.0,
                    neutral_occupancy=0.8,
                    transition_entropy_bits=1.0,
                    avalanche_event_count=2,
                    mean_avalanche_size=1.0,
                    max_avalanche_size=1,
                    final_readout=[1, 0, 0],
                    normalized_action="execute",
                    outcome="success",
                    energy_delta=1.0,
                    elapsed_ms=1.0,
                )
            ],
            accuracy_by_mode={"infer": 1.0},
            baseline_accuracy={"always_execute": 1.0, "always_query": 0.0, "always_retry": 0.0},
            correlation_summary={"accuracy_vs_branching_ratio_estimate": 0.0},
            verdict="Snapshot-level metrics are diagnostic only.",
            notes=["fake avalanche window probe"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_avalanche_window_probe", fake_probe)
    json_path = tmp_path / "avalanche-window.json"
    markdown_path = tmp_path / "avalanche-window.md"

    rc = main(
        [
            "cunxon-avalanche-window-probe",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--modes",
            "infer,train",
            "--seed-offsets",
            "122,123",
            "--steps",
            "32",
            "--sample-interval",
            "8",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    data = json_path.read_text(encoding="utf-8")
    assert '"status": "avalanche-window probe completed"' in data
    assert '"branching_ratio_estimate": 1.0' in data
    assert "cuNxon avalanche-window snapshot probe" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_avalanche_intervention_task_correlation_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_probe(**kwargs: object) -> CunxonAvalancheInterventionTaskCorrelationResult:
        assert kwargs["seed_offsets"] == [127, 128]
        assert kwargs["modes"] == ["infer", "train"]
        return CunxonAvalancheInterventionTaskCorrelationResult(
            status="avalanche intervention/task correlation completed",
            hypothesis_for_this_slice="avalanche_intervention_task_correlation",
            source_claim_ids=["branching-ratio-regimes", "functional-generalization-claim"],
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            seed_offsets=list(kwargs["seed_offsets"]),
            modes=list(kwargs["modes"]),
            configurations=[
                CunxonAvalancheInterventionTaskConfigSummary(
                    id="short-dense-heldout-stress",
                    steps=128,
                    sample_interval=8,
                    sample_count=1,
                    mean_branching_ratio_estimate=0.5,
                    branching_ratio_estimate_range=[0.5, 0.5],
                    mean_neutral_occupancy=0.8,
                    accuracy_by_split={"stress_holdout": 0.0},
                    best_constant_baseline_by_split={"stress_holdout": 1.0},
                    beats_best_constant_baseline_by_split={"stress_holdout": False},
                )
            ],
            samples=[],
            split_accuracy={"stress_holdout": 0.0},
            best_constant_baseline_by_split={"stress_holdout": 1.0},
            configurations_beating_stress_baseline=[],
            correlation_summary={"accuracy_vs_branching_ratio_estimate": 0.0},
            verdict="No config beat stress baselines.",
            evidence_boundary=(
                "Task-coupled criticality instrumentation is not intelligence evidence."
            ),
            notes=["fake task-correlation probe"],
        )

    monkeypatch.setattr(
        "neuraxon_agent.cli.run_ctypes_avalanche_intervention_task_correlation_probe",
        fake_probe,
    )
    json_path = tmp_path / "avalanche-task-correlation.json"
    markdown_path = tmp_path / "avalanche-task-correlation.md"

    rc = main(
        [
            "cunxon-avalanche-intervention-task-correlation",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--seed-offsets",
            "127,128",
            "--modes",
            "infer,train",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    data = json_path.read_text(encoding="utf-8")
    assert '"status": "avalanche intervention/task correlation completed"' in data
    assert '"hypothesis_for_this_slice": "avalanche_intervention_task_correlation"' in data
    assert "stress_holdout" in markdown_path.read_text(encoding="utf-8")


def test_cli_cunxon_controlled_regime_calibration_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_probe(**kwargs: object) -> CunxonControlledRegimeCalibrationResult:
        assert kwargs["seed_offsets"] == [133, 134]
        assert kwargs["modes"] == ["infer", "train"]
        return CunxonControlledRegimeCalibrationResult(
            status="controlled-regime calibration completed",
            hypothesis_for_this_slice="controlled_regime_calibration",
            source_issue="https://github.com/sisutuulenisa/neuraxon-hybrid/issues/86",
            source_claim_ids=["branching-ratio-regimes", "functional-generalization-claim"],
            upstream_commit=str(kwargs["upstream_commit"]),
            cunxon_commit=str(kwargs["cunxon_commit"]),
            library_path=str(kwargs["library_path"]),
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            seed_offsets=list(kwargs["seed_offsets"]),
            modes=list(kwargs["modes"]),
            steps=64,
            sample_interval=8,
            regime_drive_scales={"low-drive": 0.25, "medium-drive": 1.0, "high-drive": 2.0},
            configurations=[
                CunxonControlledRegimeCalibrationConfigSummary(
                    id="low-drive",
                    drive_scale=0.25,
                    steps=64,
                    sample_interval=8,
                    sample_count=1,
                    mean_branching_ratio_estimate=0.1,
                    branching_ratio_estimate_range=[0.1, 0.1],
                    mean_active_count_ratio=0.5,
                    mean_neutral_occupancy=0.9,
                    mean_transition_entropy_bits=0.0,
                    action_distribution={"query": 1},
                    accuracy_by_split={"stress_holdout": 0.0},
                    best_constant_baseline_by_split={"stress_holdout": 1.0},
                    beats_best_constant_baseline_by_split={"stress_holdout": False},
                )
            ],
            samples=[],
            split_accuracy={"stress_holdout": 0.0},
            best_constant_baseline_by_split={"stress_holdout": 1.0},
            stress_holdout_accuracy=0.0,
            configurations_beating_stress_baseline=[],
            correlation_summary={"accuracy_vs_branching_ratio_estimate": 0.0},
            verdict="Controlled-regime estimator calibration did not beat stress baselines.",
            evidence_boundary="Controlled-regime calibration is not intelligence evidence.",
            notes=["fake controlled calibration"],
        )

    monkeypatch.setattr(
        "neuraxon_agent.cli.run_ctypes_controlled_regime_calibration_probe",
        fake_probe,
    )
    json_path = tmp_path / "controlled-regime-calibration.json"
    markdown_path = tmp_path / "controlled-regime-calibration.md"

    rc = main(
        [
            "cunxon-controlled-regime-calibration",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--seed-offsets",
            "133,134",
            "--modes",
            "infer,train",
            "--steps",
            "64",
            "--sample-interval",
            "8",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    data = json_path.read_text(encoding="utf-8")
    assert '"status": "controlled-regime calibration completed"' in data
    assert '"hypothesis_for_this_slice": "controlled_regime_calibration"' in data
    assert "cuNxon controlled-regime criticality calibration" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_cli_cunxon_aigarth_action_remap_audit_writes_artifacts(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(
            {
                "status": "aigarth strict-label action audit completed",
                "fitness_variant": "strict_label_margin",
                "runs": [
                    {
                        "seed_offset": 1,
                        "cases": [
                            {
                                "name": "execute-assertive",
                                "split": "holdout",
                                "expected_action": "execute",
                                "readout": [1, 1, 0],
                                "normalized_action": "assertive",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    json_path = tmp_path / "remap.json"
    markdown_path = tmp_path / "remap.md"

    rc = main(
        [
            "cunxon-aigarth-action-remap-audit",
            "--source-artifacts",
            str(source_path),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    assert '"status": "aigarth action remap audit completed"' in json_path.read_text(
        encoding="utf-8"
    )
    assert "post-hoc diagnostic" in markdown_path.read_text(encoding="utf-8")


def test_cli_no_command() -> None:
    assert main([]) == 2
