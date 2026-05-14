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
    CunxonAigarthReadoutProbeResult,
    CunxonAigarthReadoutRun,
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
    assert '"status": "resident action probe viable"' in json_path.read_text(
        encoding="utf-8"
    )
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
    assert '"status": "interface semantics probe viable"' in json_path.read_text(
        encoding="utf-8"
    )
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
    assert '"status": "input-proxy target probe viable"' in json_path.read_text(
        encoding="utf-8"
    )
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
            generations=int(kwargs["generations"]),
            population_size=int(kwargs["population_size"]),
            eval_steps=int(kwargs["eval_steps"]),
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


def test_cli_no_command() -> None:
    assert main([]) == 2
