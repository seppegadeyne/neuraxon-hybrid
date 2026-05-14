"""Tests for cuNxon GPU smoke helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

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
    CunxonExternalDriveWindowProbeResult,
    CunxonExternalDriveWindowSample,
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
    CunxonSupervisedMotorCase,
    CunxonSupervisedMotorProbeResult,
    CunxonVramResidentResult,
    CunxonVramResidentSample,
    classify_cunxon_status,
    render_action_probe_markdown_report,
    render_aigarth_action_hard_holdout_markdown_report,
    render_aigarth_action_markdown_report,
    render_aigarth_action_seed_sweep_markdown_report,
    render_aigarth_action_strict_label_markdown_report,
    render_aigarth_readout_markdown_report,
    render_external_drive_window_markdown_report,
    render_input_proxy_target_markdown_report,
    render_interface_semantics_markdown_report,
    render_long_horizon_markdown_report,
    render_long_sweep_markdown_report,
    render_markdown_report,
    render_multisphere_action_markdown_report,
    render_resident_action_markdown_report,
    render_sensitivity_probe_markdown_report,
    render_snapshot_pattern_markdown_report,
    render_supervised_motor_markdown_report,
    render_vram_resident_markdown_report,
    validate_trinary_readout,
    write_action_probe_artifacts,
    write_aigarth_action_artifacts,
    write_aigarth_action_hard_holdout_artifacts,
    write_aigarth_action_seed_sweep_artifacts,
    write_aigarth_action_strict_label_artifacts,
    write_aigarth_readout_artifacts,
    write_external_drive_window_artifacts,
    write_input_proxy_target_artifacts,
    write_interface_semantics_artifacts,
    write_long_horizon_artifacts,
    write_long_sweep_artifacts,
    write_multisphere_action_artifacts,
    write_resident_action_artifacts,
    write_sensitivity_probe_artifacts,
    write_smoke_artifacts,
    write_snapshot_pattern_artifacts,
    write_supervised_motor_artifacts,
    write_vram_resident_artifacts,
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


def test_vram_resident_report_writes_progress_and_durable_state(tmp_path: Path) -> None:
    result = CunxonVramResidentResult(
        status="running",
        hypothesis="Keeping the same cuNxon process in VRAM may expose wall-clock drift.",
        active_issue="https://github.com/sisutuulenisa/neuraxon-hybrid/issues/79",
        pid=12345,
        command="neuraxon-agent cunxon-vram-resident --max-runtime-seconds 7200",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        started_at="2026-05-13T20:00:00Z",
        updated_at="2026-05-13T20:15:00Z",
        expected_end_at="2026-05-13T22:00:00Z",
        next_poll_after="2026-05-13T20:30:00Z",
        max_runtime_seconds=7200,
        sample_interval_seconds=900,
        steps_per_sample=262144,
        total_steps=262144,
        samples=[
            CunxonVramResidentSample(
                sample_index=1,
                step=262144,
                elapsed_seconds=42.0,
                readout=[0, 0, 0],
                active_state_count=2,
                neutral_state_count=13,
                energy=123.5,
                energy_delta=123.5,
                gpu_memory_used_mb=2800,
                gpu_utilization_percent=12,
                gpu_temperature_c=44,
            )
        ],
        stop_condition="stop after 7200 seconds or on cuNxon/API/resource error",
        notes=["runtime/dynamics evidence only"],
    )

    markdown = render_vram_resident_markdown_report(result)
    assert "VRAM-resident" in markdown
    assert "same cuNxon process/network resident" in markdown
    assert "does not prove intelligence" in markdown
    assert "Next poll after: 2026-05-13T20:30:00Z" in markdown

    json_path = tmp_path / "resident.json"
    markdown_path = tmp_path / "resident.md"
    state_path = tmp_path / "resident-state.json"
    write_vram_resident_artifacts(
        result,
        json_path=json_path,
        markdown_path=markdown_path,
        state_path=state_path,
    )

    data = json_path.read_text(encoding="utf-8")
    state = state_path.read_text(encoding="utf-8")
    assert '"sample_count": 1' in data
    assert '"pid": 12345' in state
    assert "resident.json" in state
    assert "runtime/dynamics evidence only" in markdown_path.read_text(encoding="utf-8")


def test_resident_action_report_keeps_task_scoring_and_baselines_explicit(
    tmp_path: Path,
) -> None:
    result = CunxonResidentActionProbeResult(
        status="resident action probe viable",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        train_epochs=2,
        train_steps_per_case=4,
        eval_steps=3,
        sphere_count=3,
        cases=[
            CunxonResidentActionCase(
                epoch=1,
                name="execute-train",
                split="train",
                input_vector=[1.0, 0.25, 0.0, 0.1],
                expected_action="execute",
                sensory_readout=[1, 0, 0, 0],
                association_readout=[0, 0, 0, 0],
                motor_readout=[0, 0, 0],
                decoded_action="query",
                normalized_action="query",
                confidence=0.0,
                outcome="failure",
                baseline_actions={"always_execute": "execute", "always_query": "query"},
                energy=12.5,
                motor_active_state_count=0,
                elapsed_ms=1.25,
            ),
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
                baseline_actions={"always_execute": "execute", "always_query": "query"},
                energy=20.0,
                motor_active_state_count=0,
                elapsed_ms=2.5,
            ),
        ],
        accuracy_by_epoch={"1": 0.5},
        accuracy_by_split={"holdout": 1.0, "overall": 0.5, "train": 0.0},
        baseline_accuracy_by_split={
            "always_execute": {"holdout": 0.0, "overall": 0.5, "train": 1.0},
            "always_query": {"holdout": 1.0, "overall": 0.5, "train": 0.0},
        },
        unique_motor_readouts=1,
        notes=["same cuNxon network remains resident across task epochs"],
    )

    markdown = render_resident_action_markdown_report(result)
    assert "resident task-coupled action probe" in markdown
    assert "same cuNxon network/context resident" in markdown
    assert "Trivial baselines" in markdown
    assert "does not prove intelligence" in markdown
    assert "Unique motor readouts: 1" in markdown

    json_path = tmp_path / "resident-action.json"
    markdown_path = tmp_path / "resident-action.md"
    write_resident_action_artifacts(result, json_path=json_path, markdown_path=markdown_path)

    assert '"case_count": 2' in json_path.read_text(encoding="utf-8")
    assert "holdout accuracy must beat trivial baselines" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_aigarth_readout_report_separates_relative_demo_from_absolute_output(
    tmp_path: Path,
) -> None:
    result = CunxonAigarthReadoutProbeResult(
        status="aigarth readout semantics probe viable",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        generations=2,
        population_size=4,
        eval_steps=5,
        runs=[
            CunxonAigarthReadoutRun(
                mapping="relative-demo-readout",
                readout_ids=[0, 1, 2, 3, 4, 5, 6, 7],
                neuron_class="input/hidden alias, not output block",
                baseline_margin=0.5,
                generation_margins=[0.75, 1.0],
                final_margin=1.0,
                improvement=0.5,
                positive_mean=0.5,
                negative_mean=-0.5,
                positive_readout=[1, 1, 0, 0, 0, 0, 0, 0],
                negative_readout=[-1, -1, 0, 0, 0, 0, 0, 0],
            ),
            CunxonAigarthReadoutRun(
                mapping="absolute-output-readout",
                readout_ids=[36, 37, 38, 39, 40, 41, 42, 43],
                neuron_class="absolute output block",
                baseline_margin=0.0,
                generation_margins=[0.0, 0.0],
                final_margin=0.0,
                improvement=0.0,
                positive_mean=0.0,
                negative_mean=0.0,
                positive_readout=[0, 0, 0, 0, 0, 0, 0, 0],
                negative_readout=[0, 0, 0, 0, 0, 0, 0, 0],
            ),
        ],
        notes=["example_aigarth.cu configures readout ids 0..7"],
    )

    markdown = render_aigarth_readout_markdown_report(result)
    assert "Aigarth readout semantics" in markdown
    assert "relative-demo-readout" in markdown
    assert "absolute-output-readout" in markdown
    assert "input/hidden alias" in markdown
    assert "does not prove intelligence" in markdown

    json_path = tmp_path / "aigarth.json"
    markdown_path = tmp_path / "aigarth.md"
    write_aigarth_readout_artifacts(result, json_path=json_path, markdown_path=markdown_path)

    assert '"mapping": "absolute-output-readout"' in json_path.read_text(encoding="utf-8")
    assert "Aigarth readout semantics" in markdown_path.read_text(encoding="utf-8")


def test_aigarth_action_report_scores_holdout_against_trivial_baselines(
    tmp_path: Path,
) -> None:
    result = CunxonAigarthActionProbeResult(
        status="aigarth action probe viable",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        generations=3,
        population_size=6,
        eval_steps=5,
        readout_ids=[35, 36, 37],
        generation_train_scores=[0.333333, 0.666667, 1.0],
        cases=[
            CunxonAigarthActionCase(
                name="execute-train",
                split="train",
                input_vector=[1.0, 0.25, 0.0],
                expected_action="execute",
                target_readout=[1, 0, 0],
                readout=[1, 0, 0],
                decoded_action="proceed",
                normalized_action="execute",
                confidence=1.0,
                outcome="success",
                target_alignment=1.0,
                baseline_actions={"always_execute": "execute", "always_query": "query"},
                energy=12.5,
            ),
            CunxonAigarthActionCase(
                name="retry-holdout-noisy",
                split="holdout",
                input_vector=[-0.8, -0.2, 0.1],
                expected_action="retry",
                target_readout=[-1, 0, 0],
                readout=[0, 0, 0],
                decoded_action="query",
                normalized_action="query",
                confidence=0.0,
                outcome="failure",
                target_alignment=0.666667,
                baseline_actions={"always_execute": "execute", "always_query": "query"},
                energy=20.0,
            ),
        ],
        accuracy_by_split={"holdout": 0.0, "overall": 0.5, "train": 1.0},
        target_alignment_by_split={"holdout": 0.666667, "overall": 0.8333335, "train": 1.0},
        baseline_accuracy_by_split={
            "always_execute": {"holdout": 0.0, "overall": 0.5, "train": 1.0},
            "always_query": {"holdout": 0.0, "overall": 0.0, "train": 0.0},
        },
        unique_readouts=2,
        action_distribution={"execute": 1, "query": 1},
        notes=["fitness callback uses train cases only; holdout is evaluated after evolution"],
    )

    markdown = render_aigarth_action_markdown_report(result)
    assert "Aigarth holdout action probe" in markdown
    assert "train cases only" in markdown
    assert "Accuracy by split" in markdown
    assert "Trivial baselines" in markdown
    assert "Unique readouts: 2" in markdown
    assert "does not prove intelligence" in markdown

    json_path = tmp_path / "aigarth-action.json"
    markdown_path = tmp_path / "aigarth-action.md"
    write_aigarth_action_artifacts(result, json_path=json_path, markdown_path=markdown_path)

    data = json_path.read_text(encoding="utf-8")
    assert '"case_count": 2' in data
    assert '"generation_train_scores": [' in data
    assert "holdout accuracy must beat trivial baselines" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_aigarth_action_seed_sweep_report_records_repeatability_and_coverage(
    tmp_path: Path,
) -> None:
    shared_case = CunxonAigarthActionCase(
        name="execute-holdout-noisy",
        split="holdout",
        input_vector=[0.8, 0.2, 0.1],
        expected_action="execute",
        target_readout=[1, 0, 0],
        readout=[1, 0, 0],
        decoded_action="proceed",
        normalized_action="execute",
        confidence=1.0,
        outcome="success",
        target_alignment=1.0,
        baseline_actions={"always_execute": "execute", "always_query": "query"},
        energy=4.0,
    )
    result = CunxonAigarthActionSeedSweepResult(
        status="aigarth action seed sweep completed",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        generations=3,
        population_size=6,
        eval_steps=5,
        readout_ids=[35, 36, 37],
        seed_offsets=[82, 83],
        runs=[
            CunxonAigarthActionSeedRun(
                seed_offset=82,
                generation_train_scores=[0.5, 0.75],
                accuracy_by_split={"holdout": 0.666667, "overall": 0.666667, "train": 0.666667},
                target_alignment_by_split={"holdout": 0.5, "overall": 0.5, "train": 0.5},
                baseline_accuracy_by_split={
                    "always_query": {"holdout": 0.333333, "overall": 0.333333}
                },
                unique_readouts=4,
                action_distribution={"execute": 2, "query": 4},
                cases=[shared_case],
            ),
            CunxonAigarthActionSeedRun(
                seed_offset=83,
                generation_train_scores=[0.5, 0.5],
                accuracy_by_split={"holdout": 0.333333, "overall": 0.333333, "train": 0.333333},
                target_alignment_by_split={"holdout": 0.25, "overall": 0.25, "train": 0.25},
                baseline_accuracy_by_split={
                    "always_query": {"holdout": 0.333333, "overall": 0.333333}
                },
                unique_readouts=1,
                action_distribution={"query": 6},
                cases=[shared_case],
            ),
        ],
        accuracy_summary_by_split={
            "holdout": {"mean": 0.5, "min": 0.333333, "max": 0.666667},
            "overall": {"mean": 0.5, "min": 0.333333, "max": 0.666667},
            "train": {"mean": 0.5, "min": 0.333333, "max": 0.666667},
        },
        aggregate_action_distribution={"assertive": 1, "execute": 2, "query": 10},
        seeds_beating_baseline_by_split={"holdout": 1, "overall": 1, "train": 1},
        notes=[
            "fresh cuNxon network/context per seed",
            "repeatability audit, not intelligence evidence",
        ],
    )

    markdown = render_aigarth_action_seed_sweep_markdown_report(result)
    assert "Aigarth action seed sweep" in markdown
    assert "Seed repeatability" in markdown
    assert "1/2" in markdown
    assert "partial action coverage" in markdown
    assert "unexpected normalized labels" in markdown
    assert "not intelligence evidence" in markdown

    json_path = tmp_path / "aigarth-action-seed-sweep.json"
    markdown_path = tmp_path / "aigarth-action-seed-sweep.md"
    write_aigarth_action_seed_sweep_artifacts(
        result,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    data = json_path.read_text(encoding="utf-8")
    assert '"seed_count": 2' in data
    assert '"seed_offsets": [' in data
    assert "fresh cuNxon network/context per seed" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_aigarth_action_hard_holdout_report_records_leakage_and_strict_labels(
    tmp_path: Path,
) -> None:
    shared_case = CunxonAigarthActionCase(
        name="execute-hard-low-positive",
        split="hard_holdout",
        input_vector=[0.35, 0.05, -0.1],
        expected_action="execute",
        target_readout=[1, 0, 0],
        readout=[-1, -1, -1],
        decoded_action="ESCALATE",
        normalized_action="assertive",
        confidence=1.0,
        outcome="failure",
        target_alignment=0.0,
        baseline_actions={"always_execute": "execute", "always_query": "query"},
        energy=4.0,
    )
    result = CunxonAigarthActionHardHoldoutResult(
        status="aigarth hard-holdout audit completed",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        generations=3,
        population_size=6,
        eval_steps=5,
        readout_ids=[35, 36, 37],
        seed_offsets=[87, 88],
        strict_expected_actions=["execute", "query", "retry"],
        runs=[
            CunxonAigarthActionSeedRun(
                seed_offset=87,
                generation_train_scores=[0.5, 0.75],
                accuracy_by_split={
                    "hard_holdout": 0.333333,
                    "holdout": 0.666667,
                    "overall": 0.5,
                    "permuted_control": 0.0,
                    "train": 0.666667,
                },
                target_alignment_by_split={"hard_holdout": 0.25, "overall": 0.5},
                baseline_accuracy_by_split={
                    "always_query": {"hard_holdout": 0.333333, "overall": 0.333333}
                },
                unique_readouts=2,
                action_distribution={"assertive": 1, "query": 5},
                cases=[shared_case],
            )
        ],
        accuracy_summary_by_split={
            "hard_holdout": {"mean": 0.333333, "min": 0.333333, "max": 0.333333},
            "holdout": {"mean": 0.666667, "min": 0.666667, "max": 0.666667},
            "overall": {"mean": 0.5, "min": 0.5, "max": 0.5},
            "permuted_control": {"mean": 0.0, "min": 0.0, "max": 0.0},
            "train": {"mean": 0.666667, "min": 0.666667, "max": 0.666667},
        },
        aggregate_action_distribution={"assertive": 1, "query": 5},
        seeds_beating_baseline_by_split={
            "hard_holdout": 0,
            "holdout": 1,
            "overall": 1,
            "permuted_control": 0,
            "train": 1,
        },
        unexpected_action_count=1,
        unexpected_action_rate=1 / 6,
        leakage_control_accuracy_mean=0.0,
        train_to_hard_holdout_gap_mean=0.333334,
        notes=["permuted-control labels are never optimized"],
    )

    markdown = render_aigarth_action_hard_holdout_markdown_report(result)
    assert "Aigarth action hard-holdout audit" in markdown
    assert "hard_holdout | 0.333333" in markdown
    assert "permuted-control leakage/oracle check" in markdown
    assert "Unexpected action rate: 0.166667" in markdown
    assert "not intelligence evidence" in markdown

    json_path = tmp_path / "aigarth-action-hard-holdout.json"
    markdown_path = tmp_path / "aigarth-action-hard-holdout.md"
    write_aigarth_action_hard_holdout_artifacts(
        result,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    data = json_path.read_text(encoding="utf-8")
    assert '"unexpected_action_count": 1' in data
    assert '"leakage_control_accuracy_mean": 0.0' in data
    assert "permuted-control labels are never optimized" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_aigarth_action_strict_label_report_records_penalty_and_contract(
    tmp_path: Path,
) -> None:
    shared_case = CunxonAigarthActionCase(
        name="retry-hard-low-negative",
        split="hard_holdout",
        input_vector=[-0.35, -0.05, 0.1],
        expected_action="retry",
        target_readout=[-1, 1, 0],
        readout=[-1, -1, -1],
        decoded_action="ESCALATE",
        normalized_action="assertive",
        confidence=1.0,
        outcome="failure",
        target_alignment=0.333333,
        baseline_actions={"always_execute": "execute", "always_query": "query"},
        energy=5.0,
    )
    result = CunxonAigarthActionHardHoldoutResult(
        status="aigarth strict-label action audit completed",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        generations=3,
        population_size=6,
        eval_steps=5,
        readout_ids=[35, 36, 37],
        seed_offsets=[92, 93],
        strict_expected_actions=["execute", "query", "retry"],
        runs=[
            CunxonAigarthActionSeedRun(
                seed_offset=92,
                generation_train_scores=[0.25, 0.5],
                accuracy_by_split={
                    "hard_holdout": 0.333333,
                    "holdout": 0.666667,
                    "overall": 0.5,
                    "permuted_control": 0.0,
                    "train": 1.0,
                },
                target_alignment_by_split={"hard_holdout": 0.333333, "overall": 0.5},
                baseline_accuracy_by_split={
                    "always_query": {"hard_holdout": 0.333333, "overall": 0.333333}
                },
                unique_readouts=2,
                action_distribution={"assertive": 1, "execute": 2, "query": 3},
                cases=[shared_case],
            )
        ],
        accuracy_summary_by_split={
            "hard_holdout": {"mean": 0.333333, "min": 0.333333, "max": 0.333333},
            "holdout": {"mean": 0.666667, "min": 0.666667, "max": 0.666667},
            "overall": {"mean": 0.5, "min": 0.5, "max": 0.5},
            "permuted_control": {"mean": 0.0, "min": 0.0, "max": 0.0},
            "train": {"mean": 1.0, "min": 1.0, "max": 1.0},
        },
        aggregate_action_distribution={"assertive": 1, "execute": 2, "query": 3},
        seeds_beating_baseline_by_split={
            "hard_holdout": 0,
            "holdout": 1,
            "overall": 1,
            "permuted_control": 0,
            "train": 1,
        },
        unexpected_action_count=1,
        unexpected_action_rate=1 / 6,
        leakage_control_accuracy_mean=0.0,
        train_to_hard_holdout_gap_mean=0.666667,
        fitness_variant="strict_label_margin",
        notes=["strict-label fitness penalizes out-of-contract normalized labels"],
    )

    markdown = render_aigarth_action_strict_label_markdown_report(result)
    assert "Aigarth strict-label action audit" in markdown
    assert "Fitness variant: `strict_label_margin`" in markdown
    assert "penalizes out-of-contract normalized labels" in markdown
    assert "Unexpected action rate: 0.166667" in markdown
    assert "not intelligence evidence" in markdown

    json_path = tmp_path / "aigarth-action-strict-label.json"
    markdown_path = tmp_path / "aigarth-action-strict-label.md"
    write_aigarth_action_strict_label_artifacts(
        result,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    data = json_path.read_text(encoding="utf-8")
    assert '"fitness_variant": "strict_label_margin"' in data
    assert '"unexpected_action_rate": 0.16666666666666666' in data
    assert "strict-label fitness penalizes" in markdown_path.read_text(encoding="utf-8")


def test_input_proxy_target_report_separates_supported_input_drive_from_decision_quality(
    tmp_path: Path,
) -> None:
    result = CunxonInputProxyTargetProbeResult(
        status="input-proxy target probe viable",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        train_epochs=3,
        train_steps_per_case=8,
        eval_steps=8,
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
                energy=12.5,
            ),
            CunxonInputProxyTargetCase(
                name="query-holdout",
                split="holdout",
                input_vector=[0.0, 0.0, 0.0],
                expected_action="query",
                target_proxy_readout=[0, 0, 0],
                motor_readout=[0, 0, 0],
                decoded_action="query",
                normalized_action="query",
                confidence=0.0,
                outcome="success",
                target_proxy_alignment=0.0,
                baseline_actions={"always_execute": "execute", "always_query": "query"},
                energy=18.0,
            ),
        ],
        accuracy_by_split={"holdout": 1.0, "overall": 0.5, "train": 0.0},
        target_proxy_alignment_by_split={"holdout": 0.0, "overall": 0.5, "train": 1.0},
        baseline_accuracy_by_split={
            "always_execute": {"holdout": 0.0, "overall": 0.5, "train": 1.0},
            "always_query": {"holdout": 1.0, "overall": 0.5, "train": 0.0},
        },
        notes=["target proxy inputs are input-class neurons"],
    )

    markdown = render_input_proxy_target_markdown_report(result)
    assert "input-port proxy target probe" in markdown
    assert "supported input-class external drive" in markdown
    assert "not a desired-output/error-channel claim" in markdown
    assert "does not prove intelligence" in markdown
    assert "holdout" in markdown

    json_path = tmp_path / "input_proxy.json"
    markdown_path = tmp_path / "input_proxy.md"
    write_input_proxy_target_artifacts(result, json_path=json_path, markdown_path=markdown_path)

    assert '"status": "input-proxy target probe viable"' in json_path.read_text(
        encoding="utf-8"
    )
    assert '"case_count": 2' in json_path.read_text(encoding="utf-8")
    assert "supported input-class external drive" in markdown_path.read_text(encoding="utf-8")


def test_external_drive_window_report_compares_input_hidden_and_output_ports(
    tmp_path: Path,
) -> None:
    result = CunxonExternalDriveWindowProbeResult(
        status="external-drive window probe viable",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        steps=5,
        input_vector=[1.0, -1.0, 0.5, -0.5],
        samples=[
            CunxonExternalDriveWindowSample(
                mode="infer",
                driven_port_class="input",
                sensory_port_ids=[0, 1, 2, 3],
                readout_port_ids=[0, 1, 2, 3],
                readout=[1, -1, 1, -1],
                snapshot_slice=[1, -1, 1, -1],
                matches_snapshot_slice=True,
                active_state_count=4,
                signed_sum=0,
                energy=4.0,
            ),
            CunxonExternalDriveWindowSample(
                mode="train",
                driven_port_class="output",
                sensory_port_ids=[8, 9, 10, 11],
                readout_port_ids=[8, 9, 10, 11],
                readout=[0, 0, 0, 0],
                snapshot_slice=[0, 0, 0, 0],
                matches_snapshot_slice=True,
                active_state_count=0,
                signed_sum=0,
                energy=1.0,
            ),
        ],
        notes=["fake external-drive window probe"],
    )

    markdown = render_external_drive_window_markdown_report(result)
    assert "external-drive window probe" in markdown
    assert "input-class direct drive" in markdown
    assert "hidden/output sensory-id windows" in markdown
    assert "does not prove intelligence" in markdown
    assert "not a desired-output/error-channel" in markdown

    json_path = tmp_path / "external_drive.json"
    markdown_path = tmp_path / "external_drive.md"
    write_external_drive_window_artifacts(
        result,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    data = json_path.read_text(encoding="utf-8")
    assert '"status": "external-drive window probe viable"' in data
    assert '"sample_count": 2' in data
    assert "input-class direct drive" in markdown_path.read_text(encoding="utf-8")


def test_long_sweep_report_scores_longer_modes_horizons_and_baselines(tmp_path: Path) -> None:
    result = CunxonLongSweepProbeResult(
        status="long-sweep probe viable",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
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
                stimulus="execute-positive-drive",
                input_vector=[1.0, 0.25, 0.0],
                expected_action="execute",
                readout=[1, 0, 0],
                decoded_action="PROCEED",
                normalized_action="execute",
                confidence=0.3333,
                outcome="success",
                energy=42.0,
                elapsed_ms=12.0,
            ),
            CunxonLongSweepSample(
                mode="train_rewarded",
                steps=2048,
                seed_offset=79,
                stimulus="retry-negative-drive",
                input_vector=[-1.0, -0.25, 0.0],
                expected_action="retry",
                readout=[0, 0, 0],
                decoded_action="PAUSE",
                normalized_action="query",
                confidence=1.0,
                outcome="failure",
                energy=43.0,
                elapsed_ms=13.0,
            ),
        ],
        accuracy_by_mode_and_steps={"train_rewarded": {"2048": 0.5}},
        unique_readouts_by_mode_and_steps={"train_rewarded": {"2048": 2}},
        action_distribution_by_mode_and_steps={
            "train_rewarded": {"2048": {"execute": 1, "query": 1}}
        },
        baseline_accuracy={"always_execute": 0.5, "always_query": 0.0, "always_retry": 0.5},
        notes=["fake long sweep"],
    )

    markdown = render_long_sweep_markdown_report(result)
    assert "long sweep" in markdown
    assert "train_rewarded" in markdown
    assert "2048" in markdown
    assert "always_query" in markdown
    assert "does not prove intelligence" in markdown

    json_path = tmp_path / "long_sweep.json"
    markdown_path = tmp_path / "long_sweep.md"
    write_long_sweep_artifacts(result, json_path=json_path, markdown_path=markdown_path)

    data = json_path.read_text(encoding="utf-8")
    assert '"status": "long-sweep probe viable"' in data
    assert '"step_horizons": [' in data
    assert "longer horizons" in markdown_path.read_text(encoding="utf-8")


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


def test_snapshot_pattern_report_records_hidden_state_and_pattern_memory(tmp_path: Path) -> None:
    result = CunxonSnapshotPatternProbeResult(
        status="snapshot-pattern probe viable",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        present_steps=30,
        settle_steps=20,
        pattern_count_after_store=2,
        pattern_count_after_clear=0,
        snapshots=[
            CunxonSnapshotObservation(
                phase="after-finalize",
                n_neurons=48,
                active_state_count=0,
                neutral_state_count=48,
                mean_abs_membrane=0.0,
                mean_abs_complement=0.0,
                mean_abs_stilde=0.0,
                mean_firing_rate=0.0,
                mean_astrocyte=0.0,
                energy=0.0,
            ),
            CunxonSnapshotObservation(
                phase="after-pattern-store",
                n_neurons=48,
                active_state_count=7,
                neutral_state_count=41,
                mean_abs_membrane=0.12,
                mean_abs_complement=0.04,
                mean_abs_stilde=0.13,
                mean_firing_rate=0.02,
                mean_astrocyte=0.01,
                energy=8.0,
            ),
        ],
        recalls=[
            CunxonPatternRecallSample(
                pattern_name="alpha",
                mask_fraction=0.5,
                readout=[1, 0, -1, 0, 0, 1, 0, -1],
                active_state_count=4,
                signed_sum=0,
            ),
            CunxonPatternRecallSample(
                pattern_name="beta",
                mask_fraction=0.5,
                readout=[0, -1, 0, 1, 1, 0, -1, 0],
                active_state_count=4,
                signed_sum=0,
            ),
        ],
        recall_hamming_distance=6,
        notes=["fake snapshot/pattern probe"],
    )

    markdown = render_snapshot_pattern_markdown_report(result)
    assert "hidden-state/snapshot" in markdown
    assert "pattern store/recall" in markdown
    assert "Hamming distance: 6" in markdown
    assert "does not prove intelligence" in markdown

    json_path = tmp_path / "snapshot_pattern.json"
    markdown_path = tmp_path / "snapshot_pattern.md"
    write_snapshot_pattern_artifacts(result, json_path=json_path, markdown_path=markdown_path)

    data = json_path.read_text(encoding="utf-8")
    assert '"pattern_count_after_store": 2' in data
    assert "pattern store/recall" in markdown_path.read_text(encoding="utf-8")


def test_multisphere_action_report_compares_holdout_against_trivial_baselines(
    tmp_path: Path,
) -> None:
    result = CunxonMultisphereActionProbeResult(
        status="multi-sphere action probe viable",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        train_steps=12,
        eval_steps=8,
        sphere_count=3,
        cases=[
            CunxonMultisphereActionCase(
                name="execute-train",
                split="train",
                input_vector=[1.0, 0.3, 0.1, 0.0],
                expected_action="execute",
                sensory_readout=[1, 0, 0, 0],
                association_readout=[0, 1, 0, 0],
                motor_readout=[1, 0, 0],
                decoded_action="PROCEED",
                normalized_action="execute",
                confidence=0.3333,
                outcome="success",
                baseline_actions={"always_execute": "execute", "always_query": "query"},
                energy=2.0,
            ),
            CunxonMultisphereActionCase(
                name="retry-holdout-noisy",
                split="holdout",
                input_vector=[-0.8, -0.2, 0.6, 0.1],
                expected_action="retry",
                sensory_readout=[0, -1, 0, 0],
                association_readout=[0, 0, -1, 0],
                motor_readout=[0, 0, 0],
                decoded_action="PAUSE",
                normalized_action="query",
                confidence=1.0,
                outcome="failure",
                baseline_actions={"always_execute": "execute", "always_query": "query"},
                energy=3.0,
            ),
        ],
        accuracy_by_split={"train": 1.0, "holdout": 0.0, "overall": 0.5},
        baseline_accuracy_by_split={
            "always_execute": {"train": 1.0, "holdout": 0.0, "overall": 0.5},
            "always_query": {"train": 0.0, "holdout": 0.0, "overall": 0.0},
        },
        notes=["fake multi-sphere adapter"],
    )

    markdown = render_multisphere_action_markdown_report(result)
    assert "multi-sphere/action adapter" in markdown
    assert "holdout" in markdown
    assert "always_execute" in markdown
    assert "does not prove intelligence" in markdown

    json_path = tmp_path / "multi.json"
    markdown_path = tmp_path / "multi.md"
    write_multisphere_action_artifacts(result, json_path=json_path, markdown_path=markdown_path)

    assert '"sphere_count": 3' in json_path.read_text(encoding="utf-8")
    assert "trivial baselines" in markdown_path.read_text(encoding="utf-8")


def test_interface_semantics_report_records_absolute_port_mapping(tmp_path: Path) -> None:
    result = CunxonInterfaceSemanticsProbeResult(
        status="interface semantics probe viable",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
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
            ),
            CunxonInterfaceReadoutSample(
                mapping="absolute-output-readout",
                port_ids=[8, 9, 10],
                neuron_class="output",
                readout=[0, 0, 0],
                snapshot_slice=[0, 0, 0],
                matches_snapshot_slice=True,
                active_state_count=0,
                signed_sum=0,
            ),
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
            ),
            CunxonInterfaceRelaySample(
                mapping="output-neuron-relay",
                source_port_ids=[16, 17, 18, 19],
                source_neuron_class="output",
                source_relay_readout=[0, 0, 0, 0],
                downstream_input_readout=[0, 0, 0, 0],
                downstream_active_state_count=0,
                downstream_energy=1.0,
            ),
        ],
        notes=["fake interface semantics probe"],
    )

    markdown = render_interface_semantics_markdown_report(result)
    assert "absolute neuron indices" in markdown
    assert "relative-input-readout" in markdown
    assert "input-neuron-relay" in markdown
    assert "does not prove intelligence" in markdown
    assert "previous multi-sphere probes" in markdown

    json_path = tmp_path / "interface.json"
    markdown_path = tmp_path / "interface.md"
    write_interface_semantics_artifacts(result, json_path=json_path, markdown_path=markdown_path)

    data = json_path.read_text(encoding="utf-8")
    assert '"status": "interface semantics probe viable"' in data
    assert '"matches_snapshot_slice": true' in data
    assert "absolute neuron indices" in markdown_path.read_text(encoding="utf-8")


def test_supervised_motor_report_scores_target_alignment_and_holdout_baselines(
    tmp_path: Path,
) -> None:
    result = CunxonSupervisedMotorProbeResult(
        status="supervised motor-target probe viable",
        upstream_commit="bd2242fabad08cb73dab2c4170d11fa941030e8c",
        cunxon_commit="b4f6db85f7aff04ddb4e1078d523d514a278521b",
        library_path="/tmp/libcunxon.so",
        device_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        train_epochs=3,
        train_steps_per_case=8,
        eval_steps=5,
        target_port_ids=[8, 9, 10],
        cases=[
            CunxonSupervisedMotorCase(
                name="execute-train",
                split="train",
                input_vector=[1.0, 0.25, 0.0],
                expected_action="execute",
                target_readout=[1, 0, 0],
                teacher_readout=[1, 0, 0],
                eval_readout=[1, 0, 0],
                decoded_action="PROCEED",
                normalized_action="execute",
                confidence=0.3333,
                outcome="success",
                target_alignment=1.0,
                baseline_actions={"always_execute": "execute", "always_query": "query"},
                energy=4.0,
            ),
            CunxonSupervisedMotorCase(
                name="retry-holdout-noisy",
                split="holdout",
                input_vector=[-0.8, -0.2, 0.1],
                expected_action="retry",
                target_readout=[-1, 0, 0],
                teacher_readout=[-1, 0, 0],
                eval_readout=[0, 0, 0],
                decoded_action="PAUSE",
                normalized_action="query",
                confidence=1.0,
                outcome="failure",
                target_alignment=0.6667,
                baseline_actions={"always_execute": "execute", "always_query": "query"},
                energy=5.0,
            ),
        ],
        accuracy_by_split={"holdout": 0.0, "overall": 0.5, "train": 1.0},
        target_alignment_by_split={"holdout": 0.6667, "overall": 0.83335, "train": 1.0},
        baseline_accuracy_by_split={
            "always_execute": {"holdout": 0.0, "overall": 0.5, "train": 1.0},
            "always_query": {"holdout": 0.0, "overall": 0.0, "train": 0.0},
        },
        notes=["fake supervised motor-target probe"],
    )

    markdown = render_supervised_motor_markdown_report(result)
    assert "supervised motor-target" in markdown
    assert "absolute output-neuron target ports" in markdown
    assert "Target alignment" in markdown
    assert "Trivial baselines" in markdown
    assert "does not prove intelligence" in markdown

    json_path = tmp_path / "supervised.json"
    markdown_path = tmp_path / "supervised.md"
    write_supervised_motor_artifacts(result, json_path=json_path, markdown_path=markdown_path)

    data = json_path.read_text(encoding="utf-8")
    assert '"status": "supervised motor-target probe viable"' in data
    assert '"target_port_ids": [' in data
    assert "absolute output-neuron target ports" in markdown_path.read_text(encoding="utf-8")


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
    assert "cuNxon source semantics audit" in markdown
    assert "cuNxon long-horizon raw dynamics" in markdown
    assert "cuNxon long sweep action diagnostic" in markdown
    assert "cuNxon task-coupled action probe" in markdown
    assert "cuNxon infer-vs-train sensitivity probe" in markdown
    assert "cuNxon snapshot/pattern probe" in markdown
    assert "cuNxon multi-sphere/action adapter" in markdown
    assert "cuNxon supervised motor-target adapter" in markdown
    assert "cuNxon resident task-coupled action probe" in markdown
    assert "cuNxon Aigarth readout semantics probe" in markdown
    assert "cuNxon Aigarth holdout/action probe" in markdown
    assert "cuNxon Aigarth action seed sweep" in markdown
    assert "holdout mean=0.533333" in markdown
    assert "3/5 seeds beat the best constant baseline on holdout" in markdown
    assert "unexpected normalized labels: assertive" in markdown
    assert "absolute output readout margin improved 0.625000→1.500000" in markdown
    assert "holdout=0.666667; overall=0.666667; beats constant-action baselines" in markdown
    assert "12 epochs / 72 scored cases" in markdown
    assert "4h run completed with 16 samples/4194304 steps" in markdown
    assert "resident readout stayed `[0, 0, 0]`" in markdown
    assert "no decision-quality score measured" in markdown
    assert "train-mode flat" in markdown
    assert "flat recall" in markdown
    assert "baseline-level holdout accuracy" in markdown
    assert "no desired-output API surface" in markdown
    assert "raw_network" in markdown
    assert "0.145833" in markdown
    assert "random" in markdown
    assert "always_execute" in markdown
    assert "do not prove intelligence" in markdown
    assert '"status": "completed 4h run"' in data
    assert '"sample_count": 16' in data
    assert '"total_steps": 4194304' in data
    assert '"unique_readouts": [' in data
    assert (
        '"verdict": "input-proxy route confirms target drive but motor readout remains '
        'baseline-level"'
        in data
    )
    assert '"target_proxy_alignment_by_split"' in data
    assert "cuNxon input-proxy target route" in markdown
    assert "cuNxon external-drive port-window probe" in markdown
    assert "target proxy train alignment=1.000000" in markdown
    assert "output-window drive remains non-teacher-forcing" in markdown
    assert '"verdict": "long-horizon runtime viable, not benchmark-integrated"' in data
    assert (
        '"verdict": "input-class direct drive observable; hidden/output windows are not '
        'desired-output channels"'
        in data
    )
    assert '"verdict": "longer action sweep remains baseline-level"' in data
    assert '"verdict": "task-coupled but not above baselines"' in data
    assert (
        '"verdict": "resident task-coupled action loop remains flat query and does not beat '
        'trivial baselines"'
        in data
    )
    assert (
        '"verdict": "Aigarth can improve absolute output margins on the two-pattern fitness '
        'task, but this is not holdout/action evidence"'
        in data
    )
    assert (
        '"verdict": "Aigarth action probe beats constant-action baselines on this small '
        'train/holdout set, but remains a toy evidence slice"'
        in data
    )
    assert '"cunxon_aigarth_action_probe"' in data
    assert '"cunxon_aigarth_action_seed_sweep"' in data
    assert '"cunxon_aigarth_action_hard_holdout_probe"' in data
    assert '"cunxon_aigarth_action_strict_label_probe"' in data
    assert "cuNxon Aigarth action hard-holdout audit" in markdown
    assert "cuNxon Aigarth strict-label action audit" in markdown
    assert "hard-holdout mean=0.500000" in markdown
    assert "strict-label fitness" in markdown
    assert "permuted-control mean=0.000000" in markdown
    assert '"mean_holdout": 0.5333333333333333' in data
    assert '"mean_hard_holdout": 0.5' in data
    assert '"fitness_variant": "strict_label_margin"' in data
    assert '"leakage_control_accuracy_mean": 0.0' in data
    assert '"holdout": 3' in data
    assert '"unexpected_actions": [' in data
    assert '"holdout": 0.6666666666666666' in data
    assert '"unique_readouts": 4' in data
    assert '"absolute_output_improvement": 0.875' in data
    assert '"relative_demo_improvement": 0.5' in data
    assert '"case_count": 72' in data
    assert '"unique_motor_readouts": 1' in data
    assert '"verdict": "sensitivity diagnostic train-mode flat"' in data
    assert (
        '"verdict": "snapshot channels viable; pattern recall callable but flat zero in this setup"'
        in data
    )
    assert (
        '"verdict": "three-sphere adapter runs but is flat query and does not beat '
        'trivial baselines"'
        in data
    )
    assert (
        '"verdict": "supervised motor-target adapter remains flat query and does not beat '
        'trivial baselines"'
        in data
    )
    assert (
        '"verdict": "source-level semantics explain why output-port teacher forcing is unsupported"'
        in data
    )
    assert '"decision_quality_measured": false' in data
    assert '"decision_quality_measured": true' in data


def test_tracked_cunxon_aigarth_action_seed_sweep_records_repeatability() -> None:
    markdown = Path("benchmarks/results/cunxon_aigarth_action_seed_sweep.md").read_text(
        encoding="utf-8"
    )
    data = Path("benchmarks/results/cunxon_aigarth_action_seed_sweep.json").read_text(
        encoding="utf-8"
    )

    assert "Aigarth action seed sweep" in markdown
    assert "Seed repeatability" in markdown
    assert "holdout | 0.533333" in markdown
    assert "3/5" in markdown
    assert "unexpected normalized labels: assertive" in markdown
    assert "not intelligence evidence" in markdown
    assert '"seed_count": 5' in data
    assert '"seed_offsets": [' in data
    assert '"mean": 0.5333333333333333' in data
    assert '"aggregate_action_distribution"' in data


def test_tracked_cunxon_aigarth_action_probe_records_holdout_action_result() -> None:
    markdown = Path("benchmarks/results/cunxon_aigarth_action_probe.md").read_text(
        encoding="utf-8"
    )
    data = Path("benchmarks/results/cunxon_aigarth_action_probe.json").read_text(
        encoding="utf-8"
    )

    assert "Aigarth holdout action probe" in markdown
    assert "train cases only" in markdown
    assert "Accuracy by split" in markdown
    assert "Trivial baselines" in markdown
    assert "Unique readouts: 4" in markdown
    assert "does not prove intelligence" in markdown
    assert '"status": "aigarth action probe viable"' in data
    assert '"case_count": 6' in data
    assert '"generation_train_scores": [' in data
    assert '"holdout": 0.6666666666666666' in data
    assert '"action_distribution": {' in data


def test_tracked_cunxon_aigarth_action_strict_label_probe_records_contract_result() -> None:
    markdown = Path("benchmarks/results/cunxon_aigarth_action_strict_label_probe.md").read_text(
        encoding="utf-8"
    )
    data = Path("benchmarks/results/cunxon_aigarth_action_strict_label_probe.json").read_text(
        encoding="utf-8"
    )

    assert "Aigarth strict-label action audit" in markdown
    assert "Fitness variant: `strict_label_margin`" in markdown
    assert "Unexpected action rate" in markdown
    assert "hard_holdout" in markdown
    assert "not intelligence evidence" in markdown
    assert '"status": "aigarth strict-label action audit completed"' in data
    assert '"fitness_variant": "strict_label_margin"' in data
    assert '"strict_expected_actions": [' in data


def test_tracked_cunxon_resident_action_probe_records_baseline_level_loop() -> None:
    markdown = Path("benchmarks/results/cunxon_resident_action_probe.md").read_text(
        encoding="utf-8"
    )
    data = Path("benchmarks/results/cunxon_resident_action_probe.json").read_text(
        encoding="utf-8"
    )

    assert "resident task-coupled action probe" in markdown
    assert "same cuNxon network/context resident" in markdown
    assert "Train epochs: 12" in markdown
    assert "Unique motor readouts: 1" in markdown
    assert "Trivial baselines" in markdown
    assert "does not prove intelligence" in markdown
    assert '"case_count": 72' in data
    assert '"unique_motor_readouts": 1' in data
    assert '"overall": 0.3333333333333333' in data
    assert '"motor_readout": [' in data


def test_tracked_cunxon_source_semantics_audit_records_output_target_boundary() -> None:
    markdown = Path("benchmarks/results/cunxon_source_semantics_audit.md").read_text(
        encoding="utf-8"
    )
    data = Path("benchmarks/results/cunxon_source_semantics_audit.json").read_text(
        encoding="utf-8"
    )

    assert "cuNxon source semantics audit" in markdown
    assert "no desired-output API surface" in markdown
    assert "External inputs are scattered only onto `sensory_input_ids`" in markdown
    assert "C++ example uses absolute output-neuron port ids" in markdown
    assert "Python example still uses relative output-port ids" in markdown
    assert "overall accuracy              : 48.2%" in markdown
    assert "does not prove intelligence" in markdown
    assert '"status": "source semantics audited"' in data
    assert '"overall_accuracy": 0.4825' in data
    assert '"desired_output_api_present": false' in data


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

def test_tracked_cunxon_long_sweep_report_records_baseline_level_long_horizons() -> None:
    markdown = Path("benchmarks/results/cunxon_long_sweep.md").read_text(encoding="utf-8")
    data = Path("benchmarks/results/cunxon_long_sweep.json").read_text(encoding="utf-8")

    assert "long sweep action diagnostic" in markdown
    assert "Step horizons: 32, 512, 4096, 32768" in markdown
    assert "Samples: 108" in markdown
    assert "| train_rewarded | 32768 | 0.333333 | 1 | query=9 |" in markdown
    assert "does not prove intelligence" in markdown
    assert '"sample_count": 108' in data
    assert '"32768": 0.3333333333333333' in data


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
