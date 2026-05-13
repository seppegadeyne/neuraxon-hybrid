"""cuNxon GPU smoke helpers and report rendering.

This module intentionally keeps the cuNxon path as an optional investigation
surface. Loading the CUDA library is explicit and runtime-only so normal CPU
benchmarks/tests keep working without NVIDIA/CUDA dependencies.
"""

from __future__ import annotations

import ctypes as C
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from neuraxon_agent.action import ActionDecoder
from neuraxon_agent.action_contract import normalize_benchmark_action

CUNXON_OK = 0
CUNXON_PROP_DEVICE_NAME = 1
CUNXON_PROP_COMPUTE_CAPABILITY = 2
CUNXON_SPHERE_SENSORY = 0
CUNXON_SPHERE_ASSOCIATION = 1
CUNXON_SPHERE_MOTOR = 2
CUNXON_LINK_FEEDFORWARD = 0
CUNXON_BAND_GAMMA = 5
CUNXON_TOPO_DENSE = 0


@dataclass(frozen=True)
class CunxonSmokeResult:
    """Result of a minimal cuNxon ctypes smoke run or recorded investigation."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    steps: int
    readout: list[int]
    energy: float
    elapsed_ms: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonLongHorizonSample:
    """One sample from a continuous cuNxon long-horizon probe."""

    step: int
    readout: list[int]
    energy: float
    elapsed_ms: float


@dataclass(frozen=True)
class CunxonLongHorizonResult:
    """Result of a longer continuous cuNxon probe on one network instance."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    total_steps: int
    sample_interval: int
    samples: list[CunxonLongHorizonSample]
    readout_change_count: int
    unique_readouts: int
    energy_delta: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonVramResidentSample:
    """One progress sample from a wall-clock/VRAM-resident cuNxon run."""

    sample_index: int
    step: int
    elapsed_seconds: float
    readout: list[int]
    active_state_count: int
    neutral_state_count: int
    energy: float
    energy_delta: float
    gpu_memory_used_mb: int | None = None
    gpu_utilization_percent: int | None = None
    gpu_temperature_c: int | None = None


@dataclass(frozen=True)
class CunxonVramResidentResult:
    """Current/final state of one long wall-clock cuNxon process kept in VRAM."""

    status: str
    hypothesis: str
    active_issue: str
    pid: int
    command: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    started_at: str
    updated_at: str
    expected_end_at: str
    next_poll_after: str
    max_runtime_seconds: int
    sample_interval_seconds: int
    steps_per_sample: int
    total_steps: int
    samples: list[CunxonVramResidentSample]
    stop_condition: str
    notes: list[str] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        """Return the number of progress samples recorded so far."""
        return len(self.samples)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["sample_count"] = self.sample_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_state_dict(
        self,
        *,
        json_path: str | Path,
        markdown_path: str | Path,
        state_path: str | Path,
    ) -> dict[str, object]:
        """Return the durable anti-duplicate state file payload."""
        return {
            "status": self.status,
            "pid": self.pid,
            "command": self.command,
            "active_issue": self.active_issue,
            "hypothesis": self.hypothesis,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "expected_end_at": self.expected_end_at,
            "next_poll_after": self.next_poll_after,
            "max_runtime_seconds": self.max_runtime_seconds,
            "sample_interval_seconds": self.sample_interval_seconds,
            "steps_per_sample": self.steps_per_sample,
            "total_steps": self.total_steps,
            "sample_count": self.sample_count,
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
            "state_path": str(state_path),
            "library_path": self.library_path,
            "gpu": {
                "device_name": self.device_name,
                "compute_capability": self.compute_capability,
            },
            "stop_condition": self.stop_condition,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CunxonLongSweepSample:
    """One long-horizon/mode/seed/stimulus sample decoded through the action contract."""

    mode: str
    steps: int
    seed_offset: int
    stimulus: str
    input_vector: list[float]
    expected_action: str
    readout: list[int]
    decoded_action: str
    normalized_action: str
    confidence: float
    outcome: str
    energy: float
    elapsed_ms: float


@dataclass(frozen=True)
class CunxonLongSweepProbeResult:
    """Longer cuNxon horizon sweep across modes, seeds, and action stimuli."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    step_horizons: list[int]
    seed_offsets: list[int]
    samples: list[CunxonLongSweepSample]
    accuracy_by_mode_and_steps: dict[str, dict[str, float]]
    unique_readouts_by_mode_and_steps: dict[str, dict[str, int]]
    action_distribution_by_mode_and_steps: dict[str, dict[str, dict[str, int]]]
    baseline_accuracy: dict[str, float]
    notes: list[str] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        """Return the number of scored long-sweep samples."""
        return len(self.samples)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["sample_count"] = self.sample_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonActionProbeTrial:
    """One task-coupled cuNxon readout decoded through the existing action contract."""

    name: str
    input_vector: list[float]
    expected_action: str
    readout: list[int]
    decoded_action: str
    normalized_action: str
    confidence: float
    outcome: str
    energy: float
    elapsed_ms: float


@dataclass(frozen=True)
class CunxonActionProbeResult:
    """Task-coupled cuNxon probe result scored against expected benchmark actions."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    trial_steps: int
    trials: list[CunxonActionProbeTrial]
    success_count: int
    accuracy: float
    notes: list[str] = field(default_factory=list)

    @property
    def trial_count(self) -> int:
        """Return the number of scored task trials."""
        return len(self.trials)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["trial_count"] = self.trial_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonSensitivityProbeSample:
    """One cuNxon stimulus readout from either frozen infer or plastic train mode."""

    mode: str
    seed_offset: int
    stimulus: str
    input_vector: list[float]
    readout: list[int]
    decoded_action: str
    normalized_action: str
    confidence: float
    energy: float
    elapsed_ms: float


@dataclass(frozen=True)
class CunxonSensitivityProbeResult:
    """Infer-vs-train cuNxon sensitivity probe result."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    steps: int
    samples: list[CunxonSensitivityProbeSample]
    unique_readouts_by_mode: dict[str, int]
    action_distribution_by_mode: dict[str, dict[str, int]]
    action_change_count_by_stimulus: dict[str, int]
    notes: list[str] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        """Return the number of stimulus/mode/seed samples."""
        return len(self.samples)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["sample_count"] = self.sample_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonSnapshotObservation:
    """Aggregated hidden-state channels captured via cunxonSphereSnapshot."""

    phase: str
    n_neurons: int
    active_state_count: int
    neutral_state_count: int
    mean_abs_membrane: float
    mean_abs_complement: float
    mean_abs_stilde: float
    mean_firing_rate: float
    mean_astrocyte: float
    energy: float


@dataclass(frozen=True)
class CunxonPatternRecallSample:
    """One host-side cuNxon pattern recall sample."""

    pattern_name: str
    mask_fraction: float
    readout: list[int]
    active_state_count: int
    signed_sum: int


@dataclass(frozen=True)
class CunxonSnapshotPatternProbeResult:
    """Result of inspecting cuNxon snapshot and pattern-memory APIs."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    present_steps: int
    settle_steps: int
    pattern_count_after_store: int
    pattern_count_after_clear: int
    snapshots: list[CunxonSnapshotObservation]
    recalls: list[CunxonPatternRecallSample]
    recall_hamming_distance: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonMultisphereActionCase:
    """One train/holdout case from a cuNxon multi-sphere action adapter probe."""

    name: str
    split: str
    input_vector: list[float]
    expected_action: str
    sensory_readout: list[int]
    association_readout: list[int]
    motor_readout: list[int]
    decoded_action: str
    normalized_action: str
    confidence: float
    outcome: str
    baseline_actions: dict[str, str]
    energy: float


@dataclass(frozen=True)
class CunxonMultisphereActionProbeResult:
    """Richer cuNxon sensory→association→motor action probe result."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    train_steps: int
    eval_steps: int
    sphere_count: int
    cases: list[CunxonMultisphereActionCase]
    accuracy_by_split: dict[str, float]
    baseline_accuracy_by_split: dict[str, dict[str, float]]
    notes: list[str] = field(default_factory=list)

    @property
    def case_count(self) -> int:
        """Return the number of scored train/holdout cases."""
        return len(self.cases)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["case_count"] = self.case_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonInterfaceReadoutSample:
    """One same-sphere readout-port mapping sample."""

    mapping: str
    port_ids: list[int]
    neuron_class: str
    readout: list[int]
    snapshot_slice: list[int]
    matches_snapshot_slice: bool
    active_state_count: int
    signed_sum: int


@dataclass(frozen=True)
class CunxonInterfaceRelaySample:
    """One source-port relay mapping sample into a downstream sphere."""

    mapping: str
    source_port_ids: list[int]
    source_neuron_class: str
    source_relay_readout: list[int]
    downstream_input_readout: list[int]
    downstream_active_state_count: int
    downstream_energy: float


@dataclass(frozen=True)
class CunxonInterfaceSemanticsProbeResult:
    """Probe result for cuNxon absolute-vs-relative interface/readout semantics."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    steps: int
    readout_samples: list[CunxonInterfaceReadoutSample]
    relay_samples: list[CunxonInterfaceRelaySample]
    notes: list[str] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        """Return the number of interface samples checked."""
        return len(self.readout_samples) + len(self.relay_samples)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["sample_count"] = self.sample_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonSupervisedMotorCase:
    """One train/holdout case from a teacher-forced motor-target probe."""

    name: str
    split: str
    input_vector: list[float]
    expected_action: str
    target_readout: list[int]
    teacher_readout: list[int]
    eval_readout: list[int]
    decoded_action: str
    normalized_action: str
    confidence: float
    outcome: str
    target_alignment: float
    baseline_actions: dict[str, str]
    energy: float


@dataclass(frozen=True)
class CunxonSupervisedMotorProbeResult:
    """Probe result for explicit supervised motor-target semantics."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    train_epochs: int
    train_steps_per_case: int
    eval_steps: int
    target_port_ids: list[int]
    cases: list[CunxonSupervisedMotorCase]
    accuracy_by_split: dict[str, float]
    target_alignment_by_split: dict[str, float]
    baseline_accuracy_by_split: dict[str, dict[str, float]]
    notes: list[str] = field(default_factory=list)

    @property
    def case_count(self) -> int:
        """Return the number of scored supervised motor cases."""
        return len(self.cases)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["case_count"] = self.case_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class CunxonError(RuntimeError):
    """Raised when the cuNxon C API returns a non-OK status."""


def validate_trinary_readout(readout: Sequence[int]) -> None:
    """Validate that a cuNxon readout is non-empty and trinary {-1, 0, +1}."""
    if len(readout) == 0:
        raise ValueError("cuNxon readout is empty")
    invalid_values = sorted({value for value in readout if value not in {-1, 0, 1}})
    if invalid_values:
        raise ValueError(f"cuNxon readout contains non-trinary values: {invalid_values}")


def classify_cunxon_status(
    *, configure_ok: bool, build_ok: bool, ctest_ok: bool, python_ok: bool
) -> str:
    """Classify cuNxon readiness without overclaiming GPU acceleration evidence."""
    if not configure_ok or not build_ok:
        return "unusable"
    if ctest_ok and python_ok:
        return "smoke-test viable"
    return "build-only"


def render_markdown_report(result: CunxonSmokeResult) -> str:
    """Render a compact human-readable cuNxon smoke report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    readout = ", ".join(_format_trinary(value) for value in result.readout)
    return "\n".join(
        [
            "# cuNxon GPU smoke report",
            "",
            f"Status: `{result.status}`",
            "",
            "## Source",
            "",
            f"- Upstream repo commit: `{result.upstream_commit}`",
            f"- cuNxon commit: `{result.cunxon_commit}`",
            f"- Library: `{result.library_path}`",
            "",
            "## GPU/runtime",
            "",
            f"- Device: {result.device_name}",
            f"- Compute capability: {result.compute_capability}",
            f"- Steps: {result.steps}",
            f"- Elapsed: {result.elapsed_ms:.3f} ms",
            f"- Energy accumulator: {result.energy:.6g}",
            f"- Trinary readout ({'{-1, 0, +1}'}): [{readout}]",
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "No broad Neuraxon intelligence claim: this smoke test only proves that the "
            "cuNxon CUDA library can be loaded and can emit a valid trinary readout. GPU "
            "execution does not prove intelligence, generalization, or usefulness by itself.",
            "",
        ]
    )


def write_smoke_artifacts(
    result: CunxonSmokeResult, *, json_path: str | Path, markdown_path: str | Path
) -> tuple[Path, Path]:
    """Write JSON and Markdown smoke artifacts and return their paths."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown_report(result), encoding="utf-8")
    return json_output, markdown_output


def render_long_horizon_markdown_report(result: CunxonLongHorizonResult) -> str:
    """Render a long-horizon cuNxon probe report with a learning-time caveat."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    sample_rows = [
        "| Step | Readout | Energy | Elapsed ms |",
        "| ---: | --- | ---: | ---: |",
    ]
    for sample in result.samples:
        readout = ", ".join(_format_trinary(value) for value in sample.readout)
        sample_rows.append(
            f"| {sample.step} | [{readout}] | {sample.energy:.6g} | {sample.elapsed_ms:.3f} |"
        )
    return "\n".join(
        [
            "# cuNxon long-horizon learning probe",
            "",
            f"Status: `{result.status}`",
            "",
            "## Source",
            "",
            f"- Upstream repo commit: `{result.upstream_commit}`",
            f"- cuNxon commit: `{result.cunxon_commit}`",
            f"- Library: `{result.library_path}`",
            "",
            "## GPU/runtime",
            "",
            f"- Device: {result.device_name}",
            f"- Compute capability: {result.compute_capability}",
            f"- Total steps: {result.total_steps}",
            f"- Sample interval: {result.sample_interval}",
            f"- Samples: {len(result.samples)}",
            f"- Unique readouts: {result.unique_readouts}",
            f"- readout changes: {result.readout_change_count}",
            f"- Energy delta: {result.energy_delta:.6g}",
            "",
            "## Long-horizon learning caveat",
            "",
            "Neuraxon should be treated as a brain-like, time-dependent learning system: "
            "short smoke tests are insufficient for judging whether it learns. This probe "
            "keeps one cuNxon network instance active for a longer step horizon and samples "
            "readout/energy over time. It is still only runtime/dynamics evidence, not a "
            "decision-quality benchmark.",
            "",
            "## Samples",
            "",
            *sample_rows,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "This long-horizon probe does not prove intelligence, generalization, or useful "
            "learning by itself. It only checks whether a continuous cuNxon run produces "
            "valid trinary readouts and observable dynamics over a longer active window.",
            "",
        ]
    )


def write_long_horizon_artifacts(
    result: CunxonLongHorizonResult, *, json_path: str | Path, markdown_path: str | Path
) -> tuple[Path, Path]:
    """Write JSON and Markdown long-horizon artifacts and return their paths."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(render_long_horizon_markdown_report(result), encoding="utf-8")
    return json_output, markdown_output


def render_vram_resident_markdown_report(result: CunxonVramResidentResult) -> str:
    """Render a progress/final report for a VRAM-resident cuNxon run."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    sample_rows = [
        (
            "| Sample | Step | Elapsed s | Readout | Active | Neutral | Energy | "
            "Energy delta | GPU MiB | GPU util % | Temp C |"
        ),
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sample in result.samples:
        readout = ", ".join(_format_trinary(value) for value in sample.readout)
        sample_rows.append(
            "| "
            f"{sample.sample_index} | {sample.step} | {sample.elapsed_seconds:.1f} | "
            f"[{readout}] | {sample.active_state_count} | {sample.neutral_state_count} | "
            f"{sample.energy:.6g} | {sample.energy_delta:.6g} | "
            f"{_format_optional_int(sample.gpu_memory_used_mb)} | "
            f"{_format_optional_int(sample.gpu_utilization_percent)} | "
            f"{_format_optional_int(sample.gpu_temperature_c)} |"
        )
    return "\n".join(
        [
            "# cuNxon VRAM-resident dynamics run",
            "",
            f"Status: `{result.status}`",
            "",
            "## Hypothesis",
            "",
            result.hypothesis,
            "",
            "## Durable state",
            "",
            f"- Active issue: {result.active_issue}",
            f"- PID: `{result.pid}`",
            f"- Command: `{result.command}`",
            f"- Started at: {result.started_at}",
            f"- Updated at: {result.updated_at}",
            f"- Expected end: {result.expected_end_at}",
            f"- Next poll after: {result.next_poll_after}",
            f"- Stop condition: {result.stop_condition}",
            "",
            "## Source",
            "",
            f"- Upstream repo commit: `{result.upstream_commit}`",
            f"- cuNxon commit: `{result.cunxon_commit}`",
            f"- Library: `{result.library_path}`",
            "",
            "## GPU/runtime",
            "",
            f"- Device: {result.device_name}",
            f"- Compute capability: {result.compute_capability}",
            f"- Max runtime seconds: {result.max_runtime_seconds}",
            f"- Sample interval seconds: {result.sample_interval_seconds}",
            f"- Steps per sample: {result.steps_per_sample}",
            f"- Total steps so far: {result.total_steps}",
            f"- Samples: {result.sample_count}",
            "",
            "## Why this run exists",
            "",
            "Earlier smoke, long-horizon, action, sensitivity, snapshot/pattern, multi-sphere, "
            "long-sweep, supervised-motor and source-semantics probes were runtime-viable but "
            "flat or baseline-level. This run tests a different hypothesis: whether keeping "
            "the same cuNxon process/network resident in VRAM across wall-clock time exposes "
            "hidden-state, energy, occupancy, readout or resource drift that short process-local "
            "runs reset away.",
            "",
            "## Samples",
            "",
            *sample_rows,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "This is runtime/dynamics evidence only. A VRAM-resident process, changing energy, "
            "or changing occupancy does not prove intelligence, task learning, useful recall, "
            "or generalization unless a later task-coupled benchmark beats trivial baselines.",
            "",
        ]
    )


def write_vram_resident_artifacts(
    result: CunxonVramResidentResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    state_path: str | Path,
) -> tuple[Path, Path, Path]:
    """Write progress JSON/Markdown plus the durable anti-duplicate state file."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    state_output = Path(state_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    state_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(render_vram_resident_markdown_report(result), encoding="utf-8")
    state_output.write_text(
        json.dumps(
            result.to_state_dict(
                json_path=json_output,
                markdown_path=markdown_output,
                state_path=state_output,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return json_output, markdown_output, state_output


def read_active_vram_resident_state(state_path: str | Path) -> dict[str, object] | None:
    """Return active VRAM-run state when the recorded PID is still alive."""
    path = Path(state_path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    status = raw.get("status")
    pid_raw = raw.get("pid")
    if status not in {"starting", "running"} or not isinstance(pid_raw, int):
        return None
    if not _pid_is_running(pid_raw):
        return None
    return raw


def ensure_no_active_vram_resident_run(state_path: str | Path) -> None:
    """Raise if the durable state file points at a still-running VRAM resident probe."""
    state = read_active_vram_resident_state(state_path)
    if state is None:
        return
    raise RuntimeError(
        "active cuNxon VRAM-resident run already exists: "
        f"pid={state.get('pid')} state_path={state_path}"
    )


def render_long_sweep_markdown_report(result: CunxonLongSweepProbeResult) -> str:
    """Render a long-horizon cuNxon action sweep report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    summary_rows = [
        "| Mode | Steps | Accuracy | Unique readouts | Action distribution |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for mode, by_steps in sorted(result.accuracy_by_mode_and_steps.items()):
        for steps_text, accuracy in sorted(by_steps.items(), key=lambda item: int(item[0])):
            unique = result.unique_readouts_by_mode_and_steps.get(mode, {}).get(steps_text, 0)
            distribution = result.action_distribution_by_mode_and_steps.get(mode, {}).get(
                steps_text, {}
            )
            distribution_text = ", ".join(
                f"{action}={count}" for action, count in sorted(distribution.items())
            ) or "none"
            summary_rows.append(
                f"| {mode} | {steps_text} | {accuracy:.6f} | {unique} | {distribution_text} |"
            )
    baseline_rows = ["| Baseline | Accuracy |", "| --- | ---: |"]
    for baseline, accuracy in sorted(result.baseline_accuracy.items()):
        baseline_rows.append(f"| {baseline} | {accuracy:.6f} |")
    sample_rows = [
        "| Mode | Steps | Seed | Stimulus | Expected | Readout | Decoded | Outcome | Energy |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for sample in result.samples:
        readout = ", ".join(_format_trinary(value) for value in sample.readout)
        sample_rows.append(
            "| "
            f"{sample.mode} | {sample.steps} | {sample.seed_offset} | {sample.stimulus} | "
            f"{sample.expected_action} | [{readout}] | "
            f"{sample.decoded_action} ({sample.normalized_action}, {sample.confidence:.4f}) | "
            f"{sample.outcome} | {sample.energy:.6g} |"
        )
    horizons = ", ".join(str(step) for step in result.step_horizons)
    seeds = ", ".join(str(seed) for seed in result.seed_offsets)
    return "\n".join(
        [
            "# cuNxon long sweep action diagnostic",
            "",
            f"Status: `{result.status}`",
            f"Samples: {result.sample_count}",
            "",
            "## Source",
            "",
            f"- Upstream repo commit: `{result.upstream_commit}`",
            f"- cuNxon commit: `{result.cunxon_commit}`",
            f"- Library: `{result.library_path}`",
            "",
            "## GPU/runtime",
            "",
            f"- Device: {result.device_name}",
            f"- Compute capability: {result.compute_capability}",
            f"- Step horizons: {horizons}",
            f"- Seed offsets: {seeds}",
            "",
            "## Why this probe exists",
            "",
            "Earlier cuNxon probes showed flat or baseline-level action readouts. This long sweep "
            "keeps the task-coupled action contract but tests longer horizons, multiple seeds, "
            "frozen inference, paper-canonical train mode, and train mode with simple "
            "reward/neuromodulator injection before treating the backend as useful.",
            "",
            "## Long sweep summary",
            "",
            *summary_rows,
            "",
            "## Trivial baselines",
            "",
            *baseline_rows,
            "",
            "## Samples",
            "",
            *sample_rows,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "Longer horizons, reward injection, readout diversity, or one successful sample "
            "does not prove intelligence, generalization, or useful learning. "
            "This diagnostic becomes positive evidence only if it beats trivial baselines "
            "across seeds and horizons.",
            "",
        ]
    )


def write_long_sweep_artifacts(
    result: CunxonLongSweepProbeResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown long-sweep artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(render_long_sweep_markdown_report(result), encoding="utf-8")
    return json_output, markdown_output


def render_action_probe_markdown_report(result: CunxonActionProbeResult) -> str:
    """Render a task-coupled cuNxon action probe report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    trial_rows = [
        "| Trial | Input | Readout | Decoded | Normalized | Expected | Outcome | Energy |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for trial in result.trials:
        input_vector = ", ".join(f"{value:.3g}" for value in trial.input_vector)
        readout = ", ".join(_format_trinary(value) for value in trial.readout)
        trial_rows.append(
            "| "
            f"{trial.name} | [{input_vector}] | [{readout}] | {trial.decoded_action} "
            f"({trial.confidence:.4f}) | {trial.normalized_action} | "
            f"{trial.expected_action} | {trial.outcome} | {trial.energy:.6g} |"
        )
    return "\n".join(
        [
            "# cuNxon task-coupled action probe",
            "",
            f"Status: `{result.status}`",
            f"Accuracy: {result.accuracy:.6f} ({result.success_count}/{result.trial_count})",
            "",
            "## Source",
            "",
            f"- Upstream repo commit: `{result.upstream_commit}`",
            f"- cuNxon commit: `{result.cunxon_commit}`",
            f"- Library: `{result.library_path}`",
            "",
            "## GPU/runtime",
            "",
            f"- Device: {result.device_name}",
            f"- Compute capability: {result.compute_capability}",
            f"- Trial steps: {result.trial_steps}",
            f"- Trials: {result.trial_count}",
            "",
            "## Decision-quality boundary",
            "",
            "This probe is task-coupled: each input drive has an expected benchmark action "
            "and the cuNxon readout is decoded through the existing Neuraxon-Hybrid "
            "ActionDecoder/action-contract mapping. That makes it decision-quality evidence, "
            "but flat or baseline-level results still do not prove intelligence, "
            "generalization, or useful learning. A task-coupled GPU probe does not prove "
            "intelligence unless it beats simple baselines and survives holdout tests.",
            "",
            "## Trials",
            "",
            *trial_rows,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "A GPU-backed action probe only becomes interesting if it beats simple baselines "
            "and survives richer temporal/generalization tests. A flat or baseline-level "
            "readout should be treated as a negative/diagnostic result, not as evidence "
            "for Neuraxon intelligence.",
            "",
        ]
    )


def write_action_probe_artifacts(
    result: CunxonActionProbeResult, *, json_path: str | Path, markdown_path: str | Path
) -> tuple[Path, Path]:
    """Write JSON and Markdown task-coupled action probe artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(render_action_probe_markdown_report(result), encoding="utf-8")
    return json_output, markdown_output


def render_sensitivity_probe_markdown_report(result: CunxonSensitivityProbeResult) -> str:
    """Render an infer-vs-train sensitivity report for cuNxon readouts."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    sample_rows = [
        "| Mode | Seed | Stimulus | Input | Readout | Decoded | Normalized | Energy |",
        "| --- | ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for sample in result.samples:
        input_vector = ", ".join(f"{value:.3g}" for value in sample.input_vector)
        readout = ", ".join(_format_trinary(value) for value in sample.readout)
        sample_rows.append(
            "| "
            f"{sample.mode} | {sample.seed_offset} | {sample.stimulus} | [{input_vector}] | "
            f"[{readout}] | {sample.decoded_action} ({sample.confidence:.4f}) | "
            f"{sample.normalized_action} | {sample.energy:.6g} |"
        )
    mode_rows = ["| Mode | Unique readouts | Action distribution |", "| --- | ---: | --- |"]
    for mode, unique_count in sorted(result.unique_readouts_by_mode.items()):
        distribution = result.action_distribution_by_mode.get(mode, {})
        distribution_text = ", ".join(
            f"{action}={count}" for action, count in sorted(distribution.items())
        ) or "none"
        mode_rows.append(f"| {mode} | {unique_count} | {distribution_text} |")
    stimulus_rows = ["| Stimulus | action changes by stimulus |", "| --- | ---: |"]
    for stimulus, change_count in sorted(result.action_change_count_by_stimulus.items()):
        stimulus_rows.append(f"| {stimulus} | {change_count} |")
    return "\n".join(
        [
            "# cuNxon infer-vs-train sensitivity probe",
            "",
            f"Status: `{result.status}`",
            f"Samples: {result.sample_count}",
            "",
            "## Source",
            "",
            f"- Upstream repo commit: `{result.upstream_commit}`",
            f"- cuNxon commit: `{result.cunxon_commit}`",
            f"- Library: `{result.library_path}`",
            "",
            "## GPU/runtime",
            "",
            f"- Device: {result.device_name}",
            f"- Compute capability: {result.compute_capability}",
            f"- Steps per sample: {result.steps}",
            "",
            "## Why this probe exists",
            "",
            "The previous cuNxon action probe used frozen `cunxonNetworkStepInfer`. Upstream "
            "documents `cunxonNetworkStepTrain` as the paper-canonical continuous-learning "
            "mode, so this diagnostic compares frozen infer readouts with plastic train "
            "readouts on the same stimuli/seeds before claiming a richer adapter works.",
            "",
            "## Mode summary",
            "",
            *mode_rows,
            "",
            "## Stimulus sensitivity",
            "",
            *stimulus_rows,
            "",
            "## Samples",
            "",
            *sample_rows,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "This is a sensitivity diagnostic, not a benchmark win. Input sensitivity, train-mode "
            "diversity, or action changes do not prove intelligence by themselves; this probe "
            "does not prove intelligence and only shows whether cuNxon exposes enough non-flat "
            "signal to justify a richer task adapter.",
            "",
        ]
    )


def write_sensitivity_probe_artifacts(
    result: CunxonSensitivityProbeResult, *, json_path: str | Path, markdown_path: str | Path
) -> tuple[Path, Path]:
    """Write JSON and Markdown infer-vs-train sensitivity artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(render_sensitivity_probe_markdown_report(result), encoding="utf-8")
    return json_output, markdown_output


def render_snapshot_pattern_markdown_report(result: CunxonSnapshotPatternProbeResult) -> str:
    """Render a cuNxon hidden-state snapshot and pattern-memory probe report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    snapshot_rows = [
        (
            "| Phase | Neurons | Active | Neutral | mean abs U | mean abs h | "
            "mean abs s_tilde | mean firing | mean astro | Energy |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for snapshot in result.snapshots:
        snapshot_rows.append(
            "| "
            f"{snapshot.phase} | {snapshot.n_neurons} | {snapshot.active_state_count} | "
            f"{snapshot.neutral_state_count} | {snapshot.mean_abs_membrane:.6g} | "
            f"{snapshot.mean_abs_complement:.6g} | {snapshot.mean_abs_stilde:.6g} | "
            f"{snapshot.mean_firing_rate:.6g} | {snapshot.mean_astrocyte:.6g} | "
            f"{snapshot.energy:.6g} |"
        )
    recall_rows = [
        "| Pattern | Mask fraction | Readout | Active | Signed sum |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for recall in result.recalls:
        readout = ", ".join(_format_trinary(value) for value in recall.readout)
        recall_rows.append(
            f"| {recall.pattern_name} | {recall.mask_fraction:.3g} | [{readout}] | "
            f"{recall.active_state_count} | {recall.signed_sum} |"
        )
    return "\n".join(
        [
            "# cuNxon hidden-state/snapshot + pattern store/recall probe",
            "",
            f"Status: `{result.status}`",
            "",
            "## Source",
            "",
            f"- Upstream repo commit: `{result.upstream_commit}`",
            f"- cuNxon commit: `{result.cunxon_commit}`",
            f"- Library: `{result.library_path}`",
            "",
            "## GPU/runtime",
            "",
            f"- Device: {result.device_name}",
            f"- Compute capability: {result.compute_capability}",
            f"- Pattern present steps: {result.present_steps}",
            f"- Recall settle steps: {result.settle_steps}",
            f"- Pattern count after store: {result.pattern_count_after_store}",
            f"- Pattern count after clear: {result.pattern_count_after_clear}",
            f"- Recall Hamming distance: {result.recall_hamming_distance}",
            "",
            "## Why this probe exists",
            "",
            "The one-sphere action and sensitivity probes mostly exposed flat `query` "
            "readouts. This diagnostic inspects richer cuNxon surfaces before building "
            "another policy: full sphere hidden-state/snapshot channels and the host-side "
            "pattern store/recall API.",
            "",
            "## Snapshot observations",
            "",
            *snapshot_rows,
            "",
            "## Pattern recall samples",
            "",
            *recall_rows,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "Snapshot activity or pattern recall shape does not prove intelligence, "
            "generalization, or useful task learning. It only tells us whether cuNxon "
            "exposes hidden-state and pattern-memory signals that are worth testing in a "
            "richer adapter.",
            "",
        ]
    )


def write_snapshot_pattern_artifacts(
    result: CunxonSnapshotPatternProbeResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown snapshot/pattern artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(render_snapshot_pattern_markdown_report(result), encoding="utf-8")
    return json_output, markdown_output


def render_multisphere_action_markdown_report(result: CunxonMultisphereActionProbeResult) -> str:
    """Render a richer multi-sphere cuNxon action adapter report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    accuracy_rows = ["| Split | cuNxon adapter accuracy |", "| --- | ---: |"]
    for split, accuracy in sorted(result.accuracy_by_split.items()):
        accuracy_rows.append(f"| {split} | {accuracy:.6f} |")
    baseline_rows = ["| Baseline | Split | Accuracy |", "| --- | --- | ---: |"]
    for baseline, by_split in sorted(result.baseline_accuracy_by_split.items()):
        for split, accuracy in sorted(by_split.items()):
            baseline_rows.append(f"| {baseline} | {split} | {accuracy:.6f} |")
    case_rows = [
        "| Case | Split | Expected | Motor readout | Decoded | Outcome | Baselines | Energy |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for case in result.cases:
        motor = ", ".join(_format_trinary(value) for value in case.motor_readout)
        baselines = ", ".join(
            f"{name}={action}" for name, action in sorted(case.baseline_actions.items())
        )
        case_rows.append(
            "| "
            f"{case.name} | {case.split} | {case.expected_action} | [{motor}] | "
            f"{case.decoded_action} ({case.normalized_action}, {case.confidence:.4f}) | "
            f"{case.outcome} | {baselines} | {case.energy:.6g} |"
        )
    return "\n".join(
        [
            "# cuNxon multi-sphere/action adapter probe",
            "",
            f"Status: `{result.status}`",
            "",
            "## Source",
            "",
            f"- Upstream repo commit: `{result.upstream_commit}`",
            f"- cuNxon commit: `{result.cunxon_commit}`",
            f"- Library: `{result.library_path}`",
            "",
            "## GPU/runtime",
            "",
            f"- Device: {result.device_name}",
            f"- Compute capability: {result.compute_capability}",
            f"- Sphere count: {result.sphere_count}",
            f"- Train steps per training case: {result.train_steps}",
            f"- Eval steps per case: {result.eval_steps}",
            f"- Cases: {result.case_count}",
            "",
            "## Adapter and holdout results",
            "",
            *accuracy_rows,
            "",
            "## Trivial baselines",
            "",
            *baseline_rows,
            "",
            "## Cases",
            "",
            *case_rows,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "This multi-sphere/action adapter is useful only if it beats trivial baselines "
            "on holdout cases. Runtime viability, inter-sphere routing, or a single positive "
            "case does not prove intelligence, generalization, or robust learning.",
            "",
        ]
    )


def write_multisphere_action_artifacts(
    result: CunxonMultisphereActionProbeResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown multi-sphere action artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(render_multisphere_action_markdown_report(result), encoding="utf-8")
    return json_output, markdown_output


def render_interface_semantics_markdown_report(
    result: CunxonInterfaceSemanticsProbeResult,
) -> str:
    """Render a cuNxon interface/readout-port semantics report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    readout_rows = [
        (
            "| Mapping | Port IDs | Neuron class | Readout | Snapshot slice | "
            "Matches snapshot | Active | Signed sum |"
        ),
        "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for sample in result.readout_samples:
        readout = ", ".join(_format_trinary(value) for value in sample.readout)
        snapshot = ", ".join(_format_trinary(value) for value in sample.snapshot_slice)
        readout_rows.append(
            "| "
            f"{sample.mapping} | {sample.port_ids} | {sample.neuron_class} | "
            f"[{readout}] | [{snapshot}] | {sample.matches_snapshot_slice} | "
            f"{sample.active_state_count} | {sample.signed_sum} |"
        )
    relay_rows = [
        (
            "| Mapping | Source port IDs | Source class | Source readout | "
            "Downstream input readout | Downstream active | Downstream energy |"
        ),
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for relay_sample in result.relay_samples:
        source = ", ".join(
            _format_trinary(value) for value in relay_sample.source_relay_readout
        )
        downstream = ", ".join(
            _format_trinary(value) for value in relay_sample.downstream_input_readout
        )
        relay_rows.append(
            "| "
            f"{relay_sample.mapping} | {relay_sample.source_port_ids} | "
            f"{relay_sample.source_neuron_class} | "
            f"[{source}] | [{downstream}] | {relay_sample.downstream_active_state_count} | "
            f"{relay_sample.downstream_energy:.6g} |"
        )
    return "\n".join(
        [
            "# cuNxon interface semantics probe",
            "",
            f"Status: `{result.status}`",
            "",
            "## Source",
            "",
            f"- Upstream repo commit: `{result.upstream_commit}`",
            f"- cuNxon commit: `{result.cunxon_commit}`",
            f"- Library: `{result.library_path}`",
            "",
            "## GPU/runtime",
            "",
            f"- Device: {result.device_name}",
            f"- Compute capability: {result.compute_capability}",
            f"- Steps: {result.steps}",
            f"- Samples: {result.sample_count}",
            "",
            "## Why this probe exists",
            "",
            "previous multi-sphere probes stayed flat, so this slice checks whether cuNxon "
            "interfaces use absolute neuron indices rather than relative output-block "
            "indices before building another supervised motor-target adapter.",
            "",
            "## Same-sphere readout-port samples",
            "",
            *readout_rows,
            "",
            "## Relay samples",
            "",
            *relay_rows,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "This probe only checks cuNxon C API interface/readout-port semantics. It does "
            "not prove intelligence, task learning, action quality, or generalization. A "
            "supervised motor-target adapter would still need holdout cases and trivial "
            "baselines before any usefulness claim.",
            "",
        ]
    )


def write_interface_semantics_artifacts(
    result: CunxonInterfaceSemanticsProbeResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown interface semantics artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_interface_semantics_markdown_report(result), encoding="utf-8"
    )
    return json_output, markdown_output


def render_supervised_motor_markdown_report(result: CunxonSupervisedMotorProbeResult) -> str:
    """Render a supervised motor-target semantics probe report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    accuracy_rows = ["| Split | Accuracy | Target alignment |", "| --- | ---: | ---: |"]
    for split, accuracy in sorted(result.accuracy_by_split.items()):
        target_alignment = result.target_alignment_by_split.get(split, 0.0)
        accuracy_rows.append(f"| {split} | {accuracy:.6f} | {target_alignment:.6f} |")
    baseline_rows = ["| Baseline | Split | Accuracy |", "| --- | --- | ---: |"]
    for baseline, by_split in sorted(result.baseline_accuracy_by_split.items()):
        for split, accuracy in sorted(by_split.items()):
            baseline_rows.append(f"| {baseline} | {split} | {accuracy:.6f} |")
    case_rows = [
        (
            "| Case | Split | Expected | Target | Teacher readout | Eval readout | "
            "Decoded | Outcome | Target alignment | Energy |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for case in result.cases:
        target = ", ".join(_format_trinary(value) for value in case.target_readout)
        teacher = ", ".join(_format_trinary(value) for value in case.teacher_readout)
        evaluation = ", ".join(_format_trinary(value) for value in case.eval_readout)
        case_rows.append(
            "| "
            f"{case.name} | {case.split} | {case.expected_action} | [{target}] | "
            f"[{teacher}] | [{evaluation}] | "
            f"{case.decoded_action} ({case.normalized_action}, {case.confidence:.4f}) | "
            f"{case.outcome} | {case.target_alignment:.6f} | {case.energy:.6g} |"
        )
    target_ports = ", ".join(str(port) for port in result.target_port_ids)
    return "\n".join(
        [
            "# cuNxon supervised motor-target probe",
            "",
            f"Status: `{result.status}`",
            "",
            "## Source",
            "",
            f"- Upstream repo commit: `{result.upstream_commit}`",
            f"- cuNxon commit: `{result.cunxon_commit}`",
            f"- Library: `{result.library_path}`",
            "",
            "## GPU/runtime",
            "",
            f"- Device: {result.device_name}",
            f"- Compute capability: {result.compute_capability}",
            f"- Train epochs: {result.train_epochs}",
            f"- Train steps per case: {result.train_steps_per_case}",
            f"- Eval steps per case: {result.eval_steps}",
            f"- Cases: {result.case_count}",
            f"- Absolute output-neuron target ports: [{target_ports}]",
            "",
            "## Why this probe exists",
            "",
            "The interface-semantics probe supports absolute neuron indices for output "
            "ports. This follow-up tests whether teacher-forcing those absolute "
            "output-neuron target ports can create a motor readout that beats trivial "
            "constant-action baselines on holdout cases.",
            "",
            "## Accuracy and Target alignment",
            "",
            *accuracy_rows,
            "",
            "## Trivial baselines",
            "",
            *baseline_rows,
            "",
            "## Cases",
            "",
            *case_rows,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "This supervised motor-target adapter is still diagnostic. Teacher-forcing "
            "absolute output-neuron ports, a positive target-alignment score, or a single "
            "train-case success does not prove intelligence, generalization, or useful "
            "learning unless holdout accuracy beats trivial baselines.",
            "",
        ]
    )


def write_supervised_motor_artifacts(
    result: CunxonSupervisedMotorProbeResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown supervised motor-target artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(render_supervised_motor_markdown_report(result), encoding="utf-8")
    return json_output, markdown_output


def run_ctypes_interface_semantics_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    steps: int = 3,
    device_id: int = 0,
) -> CunxonInterfaceSemanticsProbeResult:
    """Check cuNxon interface/readout port mapping with absolute neuron indices."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    lib_path = Path(library_path)
    lib = _load_library(lib_path)
    ctx = C.c_void_p()
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xC0FFEE2026, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        readout_samples = [
            _run_interface_readout_sample(
                lib=lib,
                ctx=ctx,
                mapping="relative-input-readout",
                port_ids=[0, 1, 2, 3],
                neuron_class="input",
                steps=steps,
            ),
            _run_interface_readout_sample(
                lib=lib,
                ctx=ctx,
                mapping="absolute-output-readout",
                port_ids=[8, 9, 10, 11],
                neuron_class="output",
                steps=steps,
            ),
        ]
        relay_samples = [
            _run_interface_relay_sample(
                lib=lib,
                ctx=ctx,
                mapping="input-neuron-relay",
                source_port_ids=[0, 1, 2, 3],
                source_neuron_class="input",
                steps=steps,
            ),
            _run_interface_relay_sample(
                lib=lib,
                ctx=ctx,
                mapping="output-neuron-relay",
                source_port_ids=[8, 9, 10, 11],
                source_neuron_class="output",
                steps=steps,
            ),
        ]
        return CunxonInterfaceSemanticsProbeResult(
            status="interface semantics probe viable",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            steps=steps,
            readout_samples=readout_samples,
            relay_samples=relay_samples,
            notes=[
                "same-sphere readouts were compared against full-sphere snapshot slices",
                (
                    "relay samples use identical source inputs but alternate "
                    "input-vs-output source ports"
                ),
                (
                    "absolute output ports are input_count + hidden_count + output offset "
                    "for this 4/4/4 probe"
                ),
            ],
        )
    finally:
        if ctx.value:
            lib.cunxonDestroyContext(ctx)


def run_ctypes_supervised_motor_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    train_epochs: int = 8,
    train_steps_per_case: int = 16,
    eval_steps: int = 16,
    device_id: int = 0,
) -> CunxonSupervisedMotorProbeResult:
    """Test teacher-forced absolute output-neuron motor targets against holdouts."""
    if train_epochs <= 0:
        raise ValueError("train_epochs must be positive")
    if train_steps_per_case <= 0:
        raise ValueError("train_steps_per_case must be positive")
    if eval_steps <= 0:
        raise ValueError("eval_steps must be positive")

    lib_path = Path(library_path)
    lib = _load_library(lib_path)
    ctx = C.c_void_p()
    net = C.c_void_p()
    decoder = ActionDecoder(num_output_neurons=3)
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xC0FFEE2026, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        _check(
            lib,
            lib.cunxonNetworkCreate(ctx, C.byref(net), b"neuraxon_hybrid_cunxon_supervised_motor"),
        )

        params = _NetworkParameters()
        _check(lib, lib.cunxonGetDefaultParameters(C.byref(params)))
        params.num_input_neurons = 3
        params.num_hidden_neurons = 5
        params.num_output_neurons = 3
        params.random_seed_offset = 179
        params.synapse_death_prob = 0.0
        params.synapse_formation_prob = 0.0

        sphere_id = C.c_int(-1)
        _check(
            lib,
            lib.cunxonNetworkAddSphere(
                net, b"SUPERVISED_MOTOR", CUNXON_SPHERE_MOTOR, C.byref(params), C.byref(sphere_id)
            ),
        )
        readout_base = params.num_input_neurons + params.num_hidden_neurons
        target_port_ids = [readout_base, readout_base + 1, readout_base + 2]
        sensory_and_target_ids = (C.c_int * 6)(0, 1, 2, *target_port_ids)
        readout_ids = (C.c_int * 3)(*target_port_ids)
        _check(
            lib,
            lib.cunxonNetworkSetSphereInterface(
                net,
                sphere_id.value,
                sensory_and_target_ids,
                6,
                None,
                0,
                None,
                0,
                readout_ids,
                3,
            ),
        )
        _check(lib, lib.cunxonNetworkFinalize(net))

        teacher_readouts: dict[str, list[int]] = {}
        for _epoch in range(train_epochs):
            for name, split, input_vector, expected_action in _default_supervised_motor_specs():
                if split != "train":
                    continue
                target = _target_readout_for_action(expected_action)
                ext_inputs = _pack_supervised_motor_inputs(input_vector, target)
                for _ in range(train_steps_per_case):
                    _check(lib, lib.cunxonNetworkStepTrain(net, ext_inputs, C.c_float(1.0)))
                    _inject_expected_action_modulator(lib, net, expected_action)
                _check(lib, lib.cunxonContextSync(ctx))
                teacher_readouts[name] = _capture_readout(lib, net, sphere_id.value)

        cases: list[CunxonSupervisedMotorCase] = []
        for index, (name, split, input_vector, expected_action) in enumerate(
            _default_supervised_motor_specs()
        ):
            target = _target_readout_for_action(expected_action)
            ext_inputs = _pack_supervised_motor_inputs(input_vector, (0.0, 0.0, 0.0))
            for _ in range(eval_steps):
                _check(lib, lib.cunxonNetworkStepInfer(net, ext_inputs, C.c_float(1.0)))
            _check(lib, lib.cunxonContextSync(ctx))
            eval_readout = _capture_readout(lib, net, sphere_id.value)
            decoded = decoder.decode(eval_readout)
            normalized_action = normalize_benchmark_action(decoded.actie_type)
            cases.append(
                CunxonSupervisedMotorCase(
                    name=name,
                    split=split,
                    input_vector=list(input_vector),
                    expected_action=expected_action,
                    target_readout=list(target),
                    teacher_readout=teacher_readouts.get(name, []),
                    eval_readout=eval_readout,
                    decoded_action=decoded.actie_type,
                    normalized_action=normalized_action,
                    confidence=decoded.confidence,
                    outcome="success" if normalized_action == expected_action else "failure",
                    target_alignment=_target_alignment(eval_readout, target),
                    baseline_actions=_baseline_actions_for_case(index),
                    energy=_capture_energy(lib, net),
                )
            )
        return CunxonSupervisedMotorProbeResult(
            status="supervised motor-target probe viable",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            train_epochs=train_epochs,
            train_steps_per_case=train_steps_per_case,
            eval_steps=eval_steps,
            target_port_ids=target_port_ids,
            cases=cases,
            accuracy_by_split=_supervised_accuracy_by_split(cases),
            target_alignment_by_split=_supervised_target_alignment_by_split(cases),
            baseline_accuracy_by_split=_supervised_baseline_accuracy_by_split(cases),
            notes=[
                (
                    "single motor sphere with sensory inputs plus teacher-forced "
                    "absolute output-neuron target ports"
                ),
                (
                    "training uses StepTrain and expected-action neuromodulator pulses; "
                    "evaluation uses StepInfer without target drive"
                ),
                (
                    "holdout accuracy must beat trivial baselines before this becomes "
                    "useful adapter evidence"
                ),
            ],
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)
        if ctx.value:
            lib.cunxonDestroyContext(ctx)


def run_ctypes_smoke(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    steps: int = 16,
    device_id: int = 0,
) -> CunxonSmokeResult:
    """Run a minimal one-sphere cuNxon smoke test through the C API.

    The smoke intentionally avoids inter-sphere links. The upstream Python demo
    currently uses output indices relative to the output block, while the C++
    demo and API expect absolute neuron indices. This one-sphere path verifies
    library loading, context/device queries, stepping and trinary readout without
    depending on that demo-specific port-index issue.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")

    lib_path = Path(library_path)
    lib = _load_library(lib_path)
    ctx = C.c_void_p()
    net = C.c_void_p()
    start = time.perf_counter()
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xC0FFEE2026, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        _check(lib, lib.cunxonNetworkCreate(ctx, C.byref(net), b"neuraxon_hybrid_cunxon_smoke"))

        params = _NetworkParameters()
        _check(lib, lib.cunxonGetDefaultParameters(C.byref(params)))
        params.num_input_neurons = 3
        params.num_hidden_neurons = 5
        params.num_output_neurons = 3
        params.random_seed_offset = 79
        params.synapse_death_prob = 0.0
        params.synapse_formation_prob = 0.0

        sphere_id = C.c_int(-1)
        _check(
            lib,
            lib.cunxonNetworkAddSphere(
                net, b"SMOKE", CUNXON_SPHERE_SENSORY, C.byref(params), C.byref(sphere_id)
            ),
        )
        sensory_ids = (C.c_int * 3)(0, 1, 2)
        readout_base = params.num_input_neurons + params.num_hidden_neurons
        readout_ids = (C.c_int * 3)(readout_base, readout_base + 1, readout_base + 2)
        _check(
            lib,
            lib.cunxonNetworkSetSphereInterface(
                net,
                sphere_id.value,
                sensory_ids,
                3,
                None,
                0,
                None,
                0,
                readout_ids,
                3,
            ),
        )
        _check(lib, lib.cunxonNetworkFinalize(net))

        input_buffer = (C.c_float * 3)(0.9, -0.25, 0.5)
        input_pointer = C.cast(input_buffer, C.POINTER(C.c_float))
        ext_inputs = (C.POINTER(C.c_float) * 1)(input_pointer)
        for _ in range(steps):
            _check(lib, lib.cunxonNetworkStepInfer(net, ext_inputs, C.c_float(1.0)))
        _check(lib, lib.cunxonContextSync(ctx))

        readout = _capture_readout(lib, net, sphere_id.value)
        energy = _capture_energy(lib, net)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return CunxonSmokeResult(
            status="smoke-test viable",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            steps=steps,
            readout=readout,
            energy=energy,
            elapsed_ms=elapsed_ms,
            notes=[
                "minimal one-sphere ctypes smoke completed",
                "inter-sphere Python demo remains separate from this smoke path",
            ],
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)
        if ctx.value:
            lib.cunxonDestroyContext(ctx)


def run_ctypes_long_horizon_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    total_steps: int = 4096,
    sample_interval: int = 512,
    device_id: int = 0,
) -> CunxonLongHorizonResult:
    """Run one cuNxon network continuously and sample dynamics over a longer horizon."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")

    lib_path = Path(library_path)
    lib = _load_library(lib_path)
    ctx = C.c_void_p()
    net = C.c_void_p()
    start = time.perf_counter()
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xC0FFEE2026, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        _check(
            lib,
            lib.cunxonNetworkCreate(ctx, C.byref(net), b"neuraxon_hybrid_cunxon_long_horizon"),
        )

        params = _NetworkParameters()
        _check(lib, lib.cunxonGetDefaultParameters(C.byref(params)))
        params.num_input_neurons = 3
        params.num_hidden_neurons = 5
        params.num_output_neurons = 3
        params.random_seed_offset = 79
        params.synapse_death_prob = 0.0
        params.synapse_formation_prob = 0.0

        sphere_id = C.c_int(-1)
        _check(
            lib,
            lib.cunxonNetworkAddSphere(
                net, b"LONG", CUNXON_SPHERE_SENSORY, C.byref(params), C.byref(sphere_id)
            ),
        )
        sensory_ids = (C.c_int * 3)(0, 1, 2)
        readout_base = params.num_input_neurons + params.num_hidden_neurons
        readout_ids = (C.c_int * 3)(readout_base, readout_base + 1, readout_base + 2)
        _check(
            lib,
            lib.cunxonNetworkSetSphereInterface(
                net,
                sphere_id.value,
                sensory_ids,
                3,
                None,
                0,
                None,
                0,
                readout_ids,
                3,
            ),
        )
        _check(lib, lib.cunxonNetworkFinalize(net))

        input_buffer = (C.c_float * 3)(0.9, -0.25, 0.5)
        input_pointer = C.cast(input_buffer, C.POINTER(C.c_float))
        ext_inputs = (C.POINTER(C.c_float) * 1)(input_pointer)
        samples: list[CunxonLongHorizonSample] = []
        previous_readout: list[int] | None = None
        readout_change_count = 0
        unique_readouts: set[tuple[int, ...]] = set()
        first_energy: float | None = None
        last_energy = 0.0
        for step in range(1, total_steps + 1):
            _check(lib, lib.cunxonNetworkStepInfer(net, ext_inputs, C.c_float(1.0)))
            if step % sample_interval != 0 and step != total_steps:
                continue
            _check(lib, lib.cunxonContextSync(ctx))
            readout = _capture_readout(lib, net, sphere_id.value)
            energy = _capture_energy(lib, net)
            if previous_readout is not None and readout != previous_readout:
                readout_change_count += 1
            previous_readout = readout
            unique_readouts.add(tuple(readout))
            if first_energy is None:
                first_energy = energy
            last_energy = energy
            samples.append(
                CunxonLongHorizonSample(
                    step=step,
                    readout=readout,
                    energy=energy,
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                )
            )

        return CunxonLongHorizonResult(
            status="long-horizon probe viable",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            total_steps=total_steps,
            sample_interval=sample_interval,
            samples=samples,
            readout_change_count=readout_change_count,
            unique_readouts=len(unique_readouts),
            energy_delta=last_energy - (first_energy if first_energy is not None else last_energy),
            notes=[
                "continuous one-sphere long-horizon probe completed",
                "short smoke tests are insufficient for learning claims",
                "inter-sphere Python demo remains separate from this probe",
            ],
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)
        if ctx.value:
            lib.cunxonDestroyContext(ctx)


def run_ctypes_vram_resident_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    hypothesis: str,
    active_issue: str,
    command: str,
    max_runtime_seconds: int = 14_400,
    sample_interval_seconds: int = 900,
    steps_per_sample: int = 262_144,
    device_id: int = 0,
    artifact_callback: Callable[[CunxonVramResidentResult], None] | None = None,
) -> CunxonVramResidentResult:
    """Keep one cuNxon network/process resident and stream dynamics/resource samples."""
    if max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    if steps_per_sample <= 0:
        raise ValueError("steps_per_sample must be positive")

    lib_path = Path(library_path)
    lib = _load_library(lib_path)
    ctx = C.c_void_p()
    net = C.c_void_p()
    started_at_dt = datetime.now(timezone.utc)
    end_at_dt = started_at_dt.timestamp() + max_runtime_seconds
    start_monotonic = time.monotonic()
    pid = os.getpid()
    samples: list[CunxonVramResidentSample] = []
    total_steps = 0
    last_energy = 0.0
    stop_condition = f"stop after {max_runtime_seconds} seconds or on cuNxon/API/resource error"
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xC0FFEE2026, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        _check(
            lib,
            lib.cunxonNetworkCreate(ctx, C.byref(net), b"neuraxon_hybrid_cunxon_vram_resident"),
        )

        params = _NetworkParameters()
        _check(lib, lib.cunxonGetDefaultParameters(C.byref(params)))
        params.num_input_neurons = 3
        params.num_hidden_neurons = 9
        params.num_output_neurons = 3
        params.random_seed_offset = 279
        params.synapse_death_prob = 0.0
        params.synapse_formation_prob = 0.0

        sphere_id = C.c_int(-1)
        _check(
            lib,
            lib.cunxonNetworkAddSphere(
                net, b"VRAM_RESIDENT", CUNXON_SPHERE_SENSORY, C.byref(params), C.byref(sphere_id)
            ),
        )
        sensory_ids = (C.c_int * 3)(0, 1, 2)
        readout_base = params.num_input_neurons + params.num_hidden_neurons
        readout_ids = (C.c_int * 3)(readout_base, readout_base + 1, readout_base + 2)
        _check(
            lib,
            lib.cunxonNetworkSetSphereInterface(
                net,
                sphere_id.value,
                sensory_ids,
                3,
                None,
                0,
                None,
                0,
                readout_ids,
                3,
            ),
        )
        _check(lib, lib.cunxonNetworkFinalize(net))

        input_buffer = (C.c_float * 3)(0.9, -0.25, 0.5)
        input_pointer = C.cast(input_buffer, C.POINTER(C.c_float))
        ext_inputs = (C.POINTER(C.c_float) * 1)(input_pointer)

        initial_result = _make_vram_resident_result(
            status="running",
            hypothesis=hypothesis,
            active_issue=active_issue,
            pid=pid,
            command=command,
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            started_at=started_at_dt,
            expected_end_timestamp=end_at_dt,
            sample_interval_seconds=sample_interval_seconds,
            steps_per_sample=steps_per_sample,
            max_runtime_seconds=max_runtime_seconds,
            total_steps=total_steps,
            samples=samples,
            stop_condition=stop_condition,
        )
        if artifact_callback is not None:
            artifact_callback(initial_result)

        sample_index = 0
        while time.monotonic() - start_monotonic < max_runtime_seconds:
            sample_start = time.monotonic()
            for _ in range(steps_per_sample):
                _check(lib, lib.cunxonNetworkStepInfer(net, ext_inputs, C.c_float(1.0)))
            total_steps += steps_per_sample
            _check(lib, lib.cunxonContextSync(ctx))
            sample_index += 1
            readout = _capture_readout(lib, net, sphere_id.value)
            snapshot_states = _capture_snapshot_states(lib, net, sphere_id.value)
            energy = _capture_energy(lib, net)
            gpu_sample = _query_nvidia_smi_sample(device_id)
            samples.append(
                CunxonVramResidentSample(
                    sample_index=sample_index,
                    step=total_steps,
                    elapsed_seconds=time.monotonic() - start_monotonic,
                    readout=readout,
                    active_state_count=sum(1 for value in snapshot_states if value != 0),
                    neutral_state_count=sum(1 for value in snapshot_states if value == 0),
                    energy=energy,
                    energy_delta=energy - last_energy,
                    gpu_memory_used_mb=gpu_sample.get("memory_used_mb"),
                    gpu_utilization_percent=gpu_sample.get("utilization_percent"),
                    gpu_temperature_c=gpu_sample.get("temperature_c"),
                )
            )
            last_energy = energy
            progress_result = _make_vram_resident_result(
                status="running",
                hypothesis=hypothesis,
                active_issue=active_issue,
                pid=pid,
                command=command,
                upstream_commit=upstream_commit,
                cunxon_commit=cunxon_commit,
                library_path=str(lib_path),
                device_name=device_name,
                compute_capability=compute_capability,
                started_at=started_at_dt,
                expected_end_timestamp=end_at_dt,
                sample_interval_seconds=sample_interval_seconds,
                steps_per_sample=steps_per_sample,
                max_runtime_seconds=max_runtime_seconds,
                total_steps=total_steps,
                samples=list(samples),
                stop_condition=stop_condition,
            )
            if artifact_callback is not None:
                artifact_callback(progress_result)
            sleep_seconds = sample_interval_seconds - (time.monotonic() - sample_start)
            remaining_seconds = max_runtime_seconds - (time.monotonic() - start_monotonic)
            if sleep_seconds > 0 and remaining_seconds > 0:
                time.sleep(min(sleep_seconds, remaining_seconds))

        return _make_vram_resident_result(
            status="completed",
            hypothesis=hypothesis,
            active_issue=active_issue,
            pid=pid,
            command=command,
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            started_at=started_at_dt,
            expected_end_timestamp=end_at_dt,
            sample_interval_seconds=sample_interval_seconds,
            steps_per_sample=steps_per_sample,
            max_runtime_seconds=max_runtime_seconds,
            total_steps=total_steps,
            samples=samples,
            stop_condition=stop_condition,
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)
        if ctx.value:
            lib.cunxonDestroyContext(ctx)


def run_ctypes_long_sweep_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    step_horizons: Sequence[int] = (32, 512, 4096),
    seed_offsets: Sequence[int] = (79, 80, 81),
    modes: Sequence[str] = ("infer", "train", "train_rewarded"),
    device_id: int = 0,
) -> CunxonLongSweepProbeResult:
    """Sweep longer cuNxon horizons/modes/seeds against the action contract."""
    if not step_horizons:
        raise ValueError("step_horizons must not be empty")
    normalized_horizons = [int(steps) for steps in step_horizons]
    if any(steps <= 0 for steps in normalized_horizons):
        raise ValueError("step_horizons must contain positive integers")
    if not seed_offsets:
        raise ValueError("seed_offsets must not be empty")
    normalized_seeds = [int(seed) for seed in seed_offsets]
    allowed_modes = {"infer", "train", "train_rewarded"}
    normalized_modes = [str(mode) for mode in modes]
    invalid_modes = sorted(set(normalized_modes) - allowed_modes)
    if invalid_modes:
        raise ValueError(f"unsupported long-sweep modes: {invalid_modes}")

    lib_path = Path(library_path)
    lib = _load_library(lib_path)
    ctx = C.c_void_p()
    decoder = ActionDecoder(num_output_neurons=3)
    start = time.perf_counter()
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xC0FFEE2026, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        samples: list[CunxonLongSweepSample] = []
        for mode in normalized_modes:
            for steps in normalized_horizons:
                for seed_offset in normalized_seeds:
                    for stimulus, input_vector, expected_action in _default_action_probe_specs():
                        samples.append(
                            _run_long_sweep_sample(
                                lib=lib,
                                ctx=ctx,
                                mode=mode,
                                steps=steps,
                                seed_offset=seed_offset,
                                stimulus=stimulus,
                                input_vector=input_vector,
                                expected_action=expected_action,
                                decoder=decoder,
                                start=start,
                            )
                        )
        return CunxonLongSweepProbeResult(
            status="long-sweep probe viable",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            step_horizons=normalized_horizons,
            seed_offsets=normalized_seeds,
            samples=samples,
            accuracy_by_mode_and_steps=_long_sweep_accuracy_by_mode_and_steps(samples),
            unique_readouts_by_mode_and_steps=_long_sweep_unique_by_mode_and_steps(samples),
            action_distribution_by_mode_and_steps=_long_sweep_actions_by_mode_and_steps(samples),
            baseline_accuracy=_long_sweep_baseline_accuracy(samples),
            notes=[
                "fresh one-sphere network per mode/steps/seed/stimulus sample",
                "infer is frozen; train is plastic; train_rewarded adds simple "
                "neuromodulator feedback",
                "longer horizons are diagnostic evidence, not benchmark success by themselves",
            ],
        )
    finally:
        if ctx.value:
            lib.cunxonDestroyContext(ctx)


def run_ctypes_action_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    trial_steps: int = 32,
    device_id: int = 0,
) -> CunxonActionProbeResult:
    """Run a tiny task-coupled cuNxon action probe through the action contract."""
    if trial_steps <= 0:
        raise ValueError("trial_steps must be positive")

    lib_path = Path(library_path)
    lib = _load_library(lib_path)
    ctx = C.c_void_p()
    net = C.c_void_p()
    decoder = ActionDecoder(num_output_neurons=3)
    start = time.perf_counter()
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xC0FFEE2026, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        _check(lib, lib.cunxonNetworkCreate(ctx, C.byref(net), b"neuraxon_hybrid_cunxon_action"))

        params = _NetworkParameters()
        _check(lib, lib.cunxonGetDefaultParameters(C.byref(params)))
        params.num_input_neurons = 3
        params.num_hidden_neurons = 5
        params.num_output_neurons = 3
        params.random_seed_offset = 79
        params.synapse_death_prob = 0.0
        params.synapse_formation_prob = 0.0

        sphere_id = C.c_int(-1)
        _check(
            lib,
            lib.cunxonNetworkAddSphere(
                net, b"ACTION", CUNXON_SPHERE_SENSORY, C.byref(params), C.byref(sphere_id)
            ),
        )
        sensory_ids = (C.c_int * 3)(0, 1, 2)
        readout_base = params.num_input_neurons + params.num_hidden_neurons
        readout_ids = (C.c_int * 3)(readout_base, readout_base + 1, readout_base + 2)
        _check(
            lib,
            lib.cunxonNetworkSetSphereInterface(
                net,
                sphere_id.value,
                sensory_ids,
                3,
                None,
                0,
                None,
                0,
                readout_ids,
                3,
            ),
        )
        _check(lib, lib.cunxonNetworkFinalize(net))

        trials: list[CunxonActionProbeTrial] = []
        for name, input_vector, expected_action in _default_action_probe_specs():
            input_buffer = (C.c_float * 3)(*input_vector)
            input_pointer = C.cast(input_buffer, C.POINTER(C.c_float))
            ext_inputs = (C.POINTER(C.c_float) * 1)(input_pointer)
            for _ in range(trial_steps):
                _check(lib, lib.cunxonNetworkStepInfer(net, ext_inputs, C.c_float(1.0)))
            _check(lib, lib.cunxonContextSync(ctx))
            readout = _capture_readout(lib, net, sphere_id.value)
            energy = _capture_energy(lib, net)
            decoded = decoder.decode(readout)
            normalized_action = normalize_benchmark_action(decoded.actie_type)
            outcome = "success" if normalized_action == expected_action else "failure"
            trials.append(
                CunxonActionProbeTrial(
                    name=name,
                    input_vector=list(input_vector),
                    expected_action=expected_action,
                    readout=readout,
                    decoded_action=decoded.actie_type,
                    normalized_action=normalized_action,
                    confidence=decoded.confidence,
                    outcome=outcome,
                    energy=energy,
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                )
            )

        success_count = sum(1 for trial in trials if trial.outcome == "success")
        accuracy = success_count / len(trials) if trials else 0.0
        return CunxonActionProbeResult(
            status="task-coupled action probe viable",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            trial_steps=trial_steps,
            trials=trials,
            success_count=success_count,
            accuracy=accuracy,
            notes=[
                "sequential one-sphere task-coupled probe completed",
                "readouts are decoded with the existing ActionDecoder/action-contract mapping",
                "flat or baseline-level accuracy is negative evidence, not intelligence evidence",
            ],
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)
        if ctx.value:
            lib.cunxonDestroyContext(ctx)


def run_ctypes_sensitivity_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    steps: int = 32,
    seed_offsets: Sequence[int] = (79, 80, 81),
    device_id: int = 0,
) -> CunxonSensitivityProbeResult:
    """Compare frozen infer and plastic train cuNxon responses to fixed stimuli."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not seed_offsets:
        raise ValueError("seed_offsets must not be empty")

    lib_path = Path(library_path)
    lib = _load_library(lib_path)
    ctx = C.c_void_p()
    decoder = ActionDecoder(num_output_neurons=3)
    start = time.perf_counter()
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xC0FFEE2026, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        samples: list[CunxonSensitivityProbeSample] = []
        for mode in ("infer", "train"):
            for seed_offset in seed_offsets:
                for stimulus, input_vector, _expected_action in _default_action_probe_specs():
                    samples.append(
                        _run_sensitivity_sample(
                            lib=lib,
                            ctx=ctx,
                            mode=mode,
                            seed_offset=int(seed_offset),
                            stimulus=stimulus,
                            input_vector=input_vector,
                            steps=steps,
                            decoder=decoder,
                            start=start,
                        )
                    )
        return CunxonSensitivityProbeResult(
            status="sensitivity probe viable",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            steps=steps,
            samples=samples,
            unique_readouts_by_mode=_count_unique_readouts_by_mode(samples),
            action_distribution_by_mode=_count_actions_by_mode(samples),
            action_change_count_by_stimulus=_count_action_changes_by_stimulus(samples),
            notes=[
                "fresh one-sphere network per mode/seed/stimulus sample",
                "infer uses frozen cunxonNetworkStepInfer; train uses paper-canonical StepTrain",
                "sensitivity/diversity is diagnostic evidence, not benchmark success",
            ],
        )
    finally:
        if ctx.value:
            lib.cunxonDestroyContext(ctx)


def run_ctypes_snapshot_pattern_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    present_steps: int = 30,
    settle_steps: int = 20,
    mask_fraction: float = 0.5,
    device_id: int = 0,
) -> CunxonSnapshotPatternProbeResult:
    """Inspect cuNxon full-sphere snapshot and host-side pattern APIs."""
    if present_steps <= 0:
        raise ValueError("present_steps must be positive")
    if settle_steps <= 0:
        raise ValueError("settle_steps must be positive")
    if not 0.0 <= mask_fraction <= 1.0:
        raise ValueError("mask_fraction must be between 0 and 1")

    lib_path = Path(library_path)
    lib = _load_library(lib_path)
    ctx = C.c_void_p()
    net = C.c_void_p()
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xC0FFEE2026, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        _check(
            lib,
            lib.cunxonNetworkCreate(ctx, C.byref(net), b"neuraxon_hybrid_cunxon_snapshot_pattern"),
        )
        params = _NetworkParameters()
        _check(lib, lib.cunxonGetDefaultParameters(C.byref(params)))
        params.num_input_neurons = 8
        params.num_hidden_neurons = 32
        params.num_output_neurons = 8
        params.random_seed_offset = 141
        params.synapse_death_prob = 0.0
        params.synapse_formation_prob = 0.0

        sphere_id = C.c_int(-1)
        _check(
            lib,
            lib.cunxonNetworkAddSphere(
                net, b"PATTERN", CUNXON_SPHERE_SENSORY, C.byref(params), C.byref(sphere_id)
            ),
        )
        sensory_ids = (C.c_int * 8)(*range(8))
        readout_base = params.num_input_neurons + params.num_hidden_neurons
        readout_ids = (C.c_int * 8)(*(readout_base + i for i in range(8)))
        _check(
            lib,
            lib.cunxonNetworkSetSphereInterface(
                net,
                sphere_id.value,
                sensory_ids,
                8,
                None,
                0,
                None,
                0,
                readout_ids,
                8,
            ),
        )
        _check(lib, lib.cunxonNetworkFinalize(net))

        snapshots = [_capture_snapshot_observation(lib, net, sphere_id.value, "after-finalize")]
        patterns = [
            ("alpha", (0.8, 0.8, 0.8, 0.8, -0.8, -0.8, -0.8, -0.8)),
            ("beta", (-0.8, -0.8, -0.8, -0.8, 0.8, 0.8, 0.8, 0.8)),
        ]
        for pattern_name, pattern_values in patterns:
            pattern_buffer = (C.c_float * 8)(*pattern_values)
            _check(
                lib,
                lib.cunxonNetworkStorePattern(
                    net,
                    sphere_id.value,
                    pattern_name.encode("utf-8"),
                    pattern_buffer,
                    8,
                    present_steps,
                    C.c_float(1.0),
                ),
            )
        _check(lib, lib.cunxonContextSync(ctx))
        pattern_count_after_store = _capture_pattern_count(lib, net)
        snapshots.append(
            _capture_snapshot_observation(lib, net, sphere_id.value, "after-pattern-store")
        )

        recalls = [
            _recall_pattern(
                lib,
                net,
                sphere_id.value,
                pattern_name,
                pattern_len=8,
                mask_fraction=mask_fraction,
                settle_steps=settle_steps,
            )
            for pattern_name, _pattern_values in patterns
        ]
        _check(lib, lib.cunxonContextSync(ctx))
        snapshots.append(_capture_snapshot_observation(lib, net, sphere_id.value, "after-recall"))
        recall_distance = (
            _hamming_distance(recalls[0].readout, recalls[1].readout) if len(recalls) >= 2 else 0
        )
        _check(lib, lib.cunxonNetworkClearPatterns(net))
        pattern_count_after_clear = _capture_pattern_count(lib, net)
        return CunxonSnapshotPatternProbeResult(
            status="snapshot-pattern probe viable",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            present_steps=present_steps,
            settle_steps=settle_steps,
            pattern_count_after_store=pattern_count_after_store,
            pattern_count_after_clear=pattern_count_after_clear,
            snapshots=snapshots,
            recalls=recalls,
            recall_hamming_distance=recall_distance,
            notes=[
                "cunxonSphereSnapshot exposed full-neuron state channels",
                "pattern store/list/recall/clear APIs are callable from ctypes",
                "pattern recall shape/signal is diagnostic evidence, not decision-quality evidence",
            ],
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)
        if ctx.value:
            lib.cunxonDestroyContext(ctx)


def run_ctypes_multisphere_action_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    train_steps: int = 24,
    eval_steps: int = 16,
    device_id: int = 0,
) -> CunxonMultisphereActionProbeResult:
    """Run a three-sphere sensory→association→motor action probe against baselines."""
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if eval_steps <= 0:
        raise ValueError("eval_steps must be positive")

    lib_path = Path(library_path)
    lib = _load_library(lib_path)
    ctx = C.c_void_p()
    net = C.c_void_p()
    decoder = ActionDecoder(num_output_neurons=3)
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xC0FFEE2026, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        _check(
            lib,
            lib.cunxonNetworkCreate(
                ctx,
                C.byref(net),
                b"neuraxon_hybrid_cunxon_multisphere_action",
            ),
        )
        sensory_id, association_id, motor_id = _build_multisphere_action_topology(lib, net)
        _check(lib, lib.cunxonNetworkFinalize(net))

        specs = _default_multisphere_action_specs()
        for _name, split, input_vector, expected_action in specs:
            if split != "train":
                continue
            for _ in range(train_steps):
                ext_inputs = _pack_three_sphere_inputs(input_vector)
                _check(lib, lib.cunxonNetworkStepTrain(net, ext_inputs, C.c_float(1.0)))
            _inject_expected_action_modulator(lib, net, expected_action)
        _check(lib, lib.cunxonContextSync(ctx))

        cases: list[CunxonMultisphereActionCase] = []
        for case_index, (name, split, input_vector, expected_action) in enumerate(specs):
            for _ in range(eval_steps):
                ext_inputs = _pack_three_sphere_inputs(input_vector)
                _check(lib, lib.cunxonNetworkStepInfer(net, ext_inputs, C.c_float(1.0)))
            _check(lib, lib.cunxonContextSync(ctx))
            sensory_readout = _capture_readout(lib, net, sensory_id)
            association_readout = _capture_readout(lib, net, association_id)
            motor_readout = _capture_readout(lib, net, motor_id)
            decoded = decoder.decode(motor_readout)
            normalized_action = normalize_benchmark_action(decoded.actie_type)
            baseline_actions = _baseline_actions_for_case(case_index)
            cases.append(
                CunxonMultisphereActionCase(
                    name=name,
                    split=split,
                    input_vector=list(input_vector),
                    expected_action=expected_action,
                    sensory_readout=sensory_readout,
                    association_readout=association_readout,
                    motor_readout=motor_readout,
                    decoded_action=decoded.actie_type,
                    normalized_action=normalized_action,
                    confidence=decoded.confidence,
                    outcome="success" if normalized_action == expected_action else "failure",
                    baseline_actions=baseline_actions,
                    energy=_capture_energy(lib, net),
                )
            )

        return CunxonMultisphereActionProbeResult(
            status="multi-sphere action probe viable",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            train_steps=train_steps,
            eval_steps=eval_steps,
            sphere_count=3,
            cases=cases,
            accuracy_by_split=_accuracy_by_split(cases),
            baseline_accuracy_by_split=_baseline_accuracy_by_split(cases),
            notes=[
                "three-sphere sensory-to-association-to-motor topology completed",
                "motor readout is decoded through the existing ActionDecoder/action contract",
                "holdout and trivial-baseline comparison is required before any adapter claim",
            ],
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)
        if ctx.value:
            lib.cunxonDestroyContext(ctx)


def _capture_snapshot_observation(
    lib: C.CDLL,
    net: C.c_void_p,
    sphere_id: int,
    phase: str,
) -> CunxonSnapshotObservation:
    n_neurons = C.c_int(0)
    _check(
        lib,
        lib.cunxonSphereSnapshot(
            net, sphere_id, None, None, None, None, None, None, C.byref(n_neurons)
        ),
    )
    if n_neurons.value <= 0:
        raise CunxonError("cuNxon snapshot returned an empty neuron count")
    size = n_neurons.value
    u = (C.c_float * size)()
    h = (C.c_float * size)()
    stilde = (C.c_float * size)()
    states = (C.c_int8 * size)()
    firing_rate = (C.c_float * size)()
    astrocyte = (C.c_float * size)()
    _check(
        lib,
        lib.cunxonSphereSnapshot(
            net,
            sphere_id,
            u,
            h,
            stilde,
            states,
            firing_rate,
            astrocyte,
            C.byref(n_neurons),
        ),
    )
    state_values = [int(states[index]) for index in range(size)]
    validate_trinary_readout(state_values)
    return CunxonSnapshotObservation(
        phase=phase,
        n_neurons=size,
        active_state_count=sum(1 for value in state_values if value != 0),
        neutral_state_count=sum(1 for value in state_values if value == 0),
        mean_abs_membrane=_mean_abs(u, size),
        mean_abs_complement=_mean_abs(h, size),
        mean_abs_stilde=_mean_abs(stilde, size),
        mean_firing_rate=_mean(firing_rate, size),
        mean_astrocyte=_mean(astrocyte, size),
        energy=_capture_energy(lib, net),
    )


def _capture_pattern_count(lib: C.CDLL, net: C.c_void_p) -> int:
    count = C.c_int(0)
    _check(lib, lib.cunxonNetworkListPatterns(net, None, 0, C.byref(count)))
    return int(count.value)


def _recall_pattern(
    lib: C.CDLL,
    net: C.c_void_p,
    sphere_id: int,
    pattern_name: str,
    *,
    pattern_len: int,
    mask_fraction: float,
    settle_steps: int,
) -> CunxonPatternRecallSample:
    n_readout = C.c_int(pattern_len)
    readout_buffer = (C.c_int8 * pattern_len)()
    _check(
        lib,
        lib.cunxonNetworkRecallPattern(
            net,
            sphere_id,
            pattern_name.encode("utf-8"),
            pattern_len,
            C.c_float(mask_fraction),
            settle_steps,
            C.c_float(1.0),
            readout_buffer,
            C.byref(n_readout),
        ),
    )
    readout = [int(readout_buffer[index]) for index in range(n_readout.value)]
    validate_trinary_readout(readout)
    return CunxonPatternRecallSample(
        pattern_name=pattern_name,
        mask_fraction=mask_fraction,
        readout=readout,
        active_state_count=sum(1 for value in readout if value != 0),
        signed_sum=sum(readout),
    )


def _hamming_distance(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(1 for left_value, right_value in zip(left, right) if left_value != right_value)


def _mean(values: Any, size: int) -> float:
    if size <= 0:
        return 0.0
    return float(sum(float(values[index]) for index in range(size)) / size)


def _mean_abs(values: Any, size: int) -> float:
    if size <= 0:
        return 0.0
    return float(sum(abs(float(values[index])) for index in range(size)) / size)


def _build_multisphere_action_topology(lib: C.CDLL, net: C.c_void_p) -> tuple[int, int, int]:
    sensory_params = _NetworkParameters()
    association_params = _NetworkParameters()
    motor_params = _NetworkParameters()
    sensory_id = _add_sphere(lib, net, b"SENSORY", CUNXON_SPHERE_SENSORY, sensory_params)
    association_id = _add_sphere(
        lib, net, b"ASSOCIATION", CUNXON_SPHERE_ASSOCIATION, association_params
    )
    motor_id = _add_sphere(lib, net, b"MOTOR", CUNXON_SPHERE_MOTOR, motor_params)

    four_ports = (C.c_int * 4)(0, 1, 2, 3)
    sensory_readout = _output_ids(sensory_params, 4)
    association_readout = _output_ids(association_params, 4)
    motor_readout = _output_ids(motor_params, 3)
    _check(
        lib,
        lib.cunxonNetworkSetSphereInterface(
            net,
            sensory_id,
            four_ports,
            4,
            None,
            0,
            four_ports,
            4,
            sensory_readout,
            4,
        ),
    )
    _check(
        lib,
        lib.cunxonNetworkSetSphereInterface(
            net,
            association_id,
            None,
            0,
            four_ports,
            4,
            four_ports,
            4,
            association_readout,
            4,
        ),
    )
    motor_relay = (C.c_int * 4)(0, 1, 2, 3)
    _check(
        lib,
        lib.cunxonNetworkSetSphereInterface(
            net,
            motor_id,
            None,
            0,
            motor_relay,
            4,
            None,
            0,
            motor_readout,
            3,
        ),
    )
    link_params = _default_link_params()
    link_id = C.c_int(-1)
    _check(
        lib,
        lib.cunxonNetworkAddLink(
            net, sensory_id, association_id, C.byref(link_params), C.byref(link_id)
        ),
    )
    _check(
        lib,
        lib.cunxonNetworkAddLink(
            net,
            association_id,
            motor_id,
            C.byref(link_params),
            C.byref(link_id),
        ),
    )
    return sensory_id, association_id, motor_id



def _add_sphere(
    lib: C.CDLL,
    net: C.c_void_p,
    name: bytes,
    kind: int,
    params: _NetworkParameters,
) -> int:
    _check(lib, lib.cunxonGetDefaultParameters(C.byref(params)))
    # Defaults are loaded here so every sphere has a fully initialized parameter struct.
    if name == b"SENSORY":
        params.num_input_neurons = 4
        params.num_hidden_neurons = 12
        params.num_output_neurons = 4
        params.random_seed_offset = 210
    elif name == b"ASSOCIATION":
        params.num_input_neurons = 4
        params.num_hidden_neurons = 12
        params.num_output_neurons = 4
        params.random_seed_offset = 211
    else:
        params.num_input_neurons = 4
        params.num_hidden_neurons = 10
        params.num_output_neurons = 3
        params.random_seed_offset = 212
    params.synapse_death_prob = 0.0
    params.synapse_formation_prob = 0.0
    sphere_id = C.c_int(-1)
    _check(lib, lib.cunxonNetworkAddSphere(net, name, kind, C.byref(params), C.byref(sphere_id)))
    return int(sphere_id.value)


def _output_ids(params: _NetworkParameters, count: int) -> Any:
    base = params.num_input_neurons + params.num_hidden_neurons
    return (C.c_int * count)(*(base + index for index in range(count)))


def _default_link_params() -> _LinkParameters:
    params = _LinkParameters()
    params.kind = CUNXON_LINK_FEEDFORWARD
    params.coherence_band = CUNXON_BAND_GAMMA
    params.gain = 1.0
    params.delay_steps = 1
    params.transmission_threshold = 0.0
    params.coherence_strength = 0.5
    params.topology = CUNXON_TOPO_DENSE
    params.sparse_prob = 0.3
    params.allow_negative_weights = 1
    params.plasticity_rate = 1e-3
    params.weight_decay = 1e-5
    params.weight_clip = 1.0
    params.normalize_rows = 0
    params.bias = 0.0
    return params


def _pack_three_sphere_inputs(
    input_vector: Sequence[float],
) -> Any:
    input_buffer = (C.c_float * len(input_vector))(*input_vector)
    input_pointer = C.cast(input_buffer, C.POINTER(C.c_float))
    null_pointer = C.POINTER(C.c_float)()
    ext_inputs = (C.POINTER(C.c_float) * 3)(input_pointer, null_pointer, null_pointer)
    ext_inputs._input_buffer = input_buffer  # type: ignore[attr-defined]
    return ext_inputs


def _inject_expected_action_modulator(lib: C.CDLL, net: C.c_void_p, expected_action: str) -> None:
    if expected_action == "execute":
        _check(lib, lib.cunxonNetworkInjectNeuromodulator(net, 0, C.c_float(0.5)))
    elif expected_action == "retry":
        _check(lib, lib.cunxonNetworkInjectNeuromodulator(net, 3, C.c_float(0.35)))
    else:
        _check(lib, lib.cunxonNetworkInjectNeuromodulator(net, 1, C.c_float(0.2)))


def _default_multisphere_action_specs() -> list[
    tuple[str, str, tuple[float, float, float, float], str]
]:
    return [
        ("execute-train", "train", (1.0, 0.25, 0.0, 0.1), "execute"),
        ("retry-train", "train", (-1.0, -0.25, 0.5, -0.1), "retry"),
        ("query-train", "train", (0.0, 0.0, 0.0, 0.0), "query"),
        ("execute-holdout-noisy", "holdout", (0.8, 0.2, 0.1, 0.05), "execute"),
        ("retry-holdout-noisy", "holdout", (-0.8, -0.2, 0.6, -0.05), "retry"),
        ("query-holdout-low-drive", "holdout", (0.05, 0.0, -0.05, 0.0), "query"),
    ]


def _baseline_actions_for_case(_case_index: int) -> dict[str, str]:
    return {
        "always_execute": "execute",
        "always_retry": "retry",
        "always_query": "query",
    }


def _accuracy_by_split(cases: Sequence[CunxonMultisphereActionCase]) -> dict[str, float]:
    by_split: dict[str, list[CunxonMultisphereActionCase]] = {}
    for case in cases:
        by_split.setdefault(case.split, []).append(case)
    by_split["overall"] = list(cases)
    return {
        split: _case_accuracy(split_cases)
        for split, split_cases in sorted(by_split.items())
        if split_cases
    }


def _baseline_accuracy_by_split(
    cases: Sequence[CunxonMultisphereActionCase],
) -> dict[str, dict[str, float]]:
    baseline_names = sorted({name for case in cases for name in case.baseline_actions})
    result: dict[str, dict[str, float]] = {}
    for baseline_name in baseline_names:
        pseudo_cases = [
            case
            for case in cases
            if baseline_name in case.baseline_actions
        ]
        split_scores: dict[str, float] = {}
        for split in sorted({case.split for case in pseudo_cases} | {"overall"}):
            if split == "overall":
                split_cases = pseudo_cases
            else:
                split_cases = [case for case in pseudo_cases if case.split == split]
            if not split_cases:
                continue
            successes = sum(
                1
                for case in split_cases
                if case.baseline_actions[baseline_name] == case.expected_action
            )
            split_scores[split] = successes / len(split_cases)
        result[baseline_name] = split_scores
    return result


def _case_accuracy(cases: Sequence[CunxonMultisphereActionCase]) -> float:
    if not cases:
        return 0.0
    successes = sum(1 for case in cases if case.normalized_action == case.expected_action)
    return successes / len(cases)


def _run_sensitivity_sample(
    *,
    lib: C.CDLL,
    ctx: C.c_void_p,
    mode: str,
    seed_offset: int,
    stimulus: str,
    input_vector: tuple[float, float, float],
    steps: int,
    decoder: ActionDecoder,
    start: float,
) -> CunxonSensitivityProbeSample:
    net = C.c_void_p()
    try:
        name = f"neuraxon_hybrid_cunxon_sensitivity_{mode}_{seed_offset}_{stimulus}"
        _check(lib, lib.cunxonNetworkCreate(ctx, C.byref(net), name.encode("utf-8")))
        params = _NetworkParameters()
        _check(lib, lib.cunxonGetDefaultParameters(C.byref(params)))
        params.num_input_neurons = 3
        params.num_hidden_neurons = 5
        params.num_output_neurons = 3
        params.random_seed_offset = seed_offset
        params.synapse_death_prob = 0.0
        params.synapse_formation_prob = 0.0

        sphere_id = C.c_int(-1)
        _check(
            lib,
            lib.cunxonNetworkAddSphere(
                net, b"SENS", CUNXON_SPHERE_SENSORY, C.byref(params), C.byref(sphere_id)
            ),
        )
        sensory_ids = (C.c_int * 3)(0, 1, 2)
        readout_base = params.num_input_neurons + params.num_hidden_neurons
        readout_ids = (C.c_int * 3)(readout_base, readout_base + 1, readout_base + 2)
        _check(
            lib,
            lib.cunxonNetworkSetSphereInterface(
                net,
                sphere_id.value,
                sensory_ids,
                3,
                None,
                0,
                None,
                0,
                readout_ids,
                3,
            ),
        )
        _check(lib, lib.cunxonNetworkFinalize(net))

        input_buffer = (C.c_float * 3)(*input_vector)
        input_pointer = C.cast(input_buffer, C.POINTER(C.c_float))
        ext_inputs = (C.POINTER(C.c_float) * 1)(input_pointer)
        step_function = (
            lib.cunxonNetworkStepTrain if mode == "train" else lib.cunxonNetworkStepInfer
        )
        for _ in range(steps):
            _check(lib, step_function(net, ext_inputs, C.c_float(1.0)))
        _check(lib, lib.cunxonContextSync(ctx))
        readout = _capture_readout(lib, net, sphere_id.value)
        energy = _capture_energy(lib, net)
        decoded = decoder.decode(readout)
        normalized_action = normalize_benchmark_action(decoded.actie_type)
        return CunxonSensitivityProbeSample(
            mode=mode,
            seed_offset=seed_offset,
            stimulus=stimulus,
            input_vector=list(input_vector),
            readout=readout,
            decoded_action=decoded.actie_type,
            normalized_action=normalized_action,
            confidence=decoded.confidence,
            energy=energy,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)


def _run_long_sweep_sample(
    *,
    lib: C.CDLL,
    ctx: C.c_void_p,
    mode: str,
    steps: int,
    seed_offset: int,
    stimulus: str,
    input_vector: tuple[float, float, float],
    expected_action: str,
    decoder: ActionDecoder,
    start: float,
) -> CunxonLongSweepSample:
    net = C.c_void_p()
    try:
        name = f"neuraxon_hybrid_cunxon_long_sweep_{mode}_{steps}_{seed_offset}_{stimulus}"
        _check(lib, lib.cunxonNetworkCreate(ctx, C.byref(net), name.encode("utf-8")))
        params = _NetworkParameters()
        _check(lib, lib.cunxonGetDefaultParameters(C.byref(params)))
        params.num_input_neurons = 3
        params.num_hidden_neurons = 5
        params.num_output_neurons = 3
        params.random_seed_offset = seed_offset
        params.synapse_death_prob = 0.0
        params.synapse_formation_prob = 0.0

        sphere_id = C.c_int(-1)
        _check(
            lib,
            lib.cunxonNetworkAddSphere(
                net, b"SWEEP", CUNXON_SPHERE_SENSORY, C.byref(params), C.byref(sphere_id)
            ),
        )
        sensory_ids = (C.c_int * 3)(0, 1, 2)
        readout_base = params.num_input_neurons + params.num_hidden_neurons
        readout_ids = (C.c_int * 3)(readout_base, readout_base + 1, readout_base + 2)
        _check(
            lib,
            lib.cunxonNetworkSetSphereInterface(
                net,
                sphere_id.value,
                sensory_ids,
                3,
                None,
                0,
                None,
                0,
                readout_ids,
                3,
            ),
        )
        _check(lib, lib.cunxonNetworkFinalize(net))

        input_buffer = (C.c_float * 3)(*input_vector)
        input_pointer = C.cast(input_buffer, C.POINTER(C.c_float))
        ext_inputs = (C.POINTER(C.c_float) * 1)(input_pointer)
        for _ in range(steps):
            if mode == "infer":
                _check(lib, lib.cunxonNetworkStepInfer(net, ext_inputs, C.c_float(1.0)))
            else:
                _check(lib, lib.cunxonNetworkStepTrain(net, ext_inputs, C.c_float(1.0)))
                if mode == "train_rewarded":
                    _inject_expected_action_modulator(lib, net, expected_action)
        _check(lib, lib.cunxonContextSync(ctx))
        readout = _capture_readout(lib, net, sphere_id.value)
        energy = _capture_energy(lib, net)
        decoded = decoder.decode(readout)
        normalized_action = normalize_benchmark_action(decoded.actie_type)
        return CunxonLongSweepSample(
            mode=mode,
            steps=steps,
            seed_offset=seed_offset,
            stimulus=stimulus,
            input_vector=list(input_vector),
            expected_action=expected_action,
            readout=readout,
            decoded_action=decoded.actie_type,
            normalized_action=normalized_action,
            confidence=decoded.confidence,
            outcome="success" if normalized_action == expected_action else "failure",
            energy=energy,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)


def _long_sweep_accuracy_by_mode_and_steps(
    samples: Sequence[CunxonLongSweepSample],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[CunxonLongSweepSample]]] = {}
    for sample in samples:
        grouped.setdefault(sample.mode, {}).setdefault(str(sample.steps), []).append(sample)
    return {
        mode: {
            steps: sum(1 for sample in step_samples if sample.outcome == "success")
            / len(step_samples)
            for steps, step_samples in sorted(by_steps.items(), key=lambda item: int(item[0]))
        }
        for mode, by_steps in sorted(grouped.items())
    }


def _long_sweep_unique_by_mode_and_steps(
    samples: Sequence[CunxonLongSweepSample],
) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, set[tuple[int, ...]]]] = {}
    for sample in samples:
        grouped.setdefault(sample.mode, {}).setdefault(str(sample.steps), set()).add(
            tuple(sample.readout)
        )
    return {
        mode: {
            steps: len(readouts)
            for steps, readouts in sorted(by_steps.items(), key=lambda item: int(item[0]))
        }
        for mode, by_steps in sorted(grouped.items())
    }


def _long_sweep_actions_by_mode_and_steps(
    samples: Sequence[CunxonLongSweepSample],
) -> dict[str, dict[str, dict[str, int]]]:
    grouped: dict[str, dict[str, dict[str, int]]] = {}
    for sample in samples:
        actions = grouped.setdefault(sample.mode, {}).setdefault(str(sample.steps), {})
        actions[sample.normalized_action] = actions.get(sample.normalized_action, 0) + 1
    return {
        mode: {
            steps: dict(sorted(actions.items()))
            for steps, actions in sorted(by_steps.items(), key=lambda item: int(item[0]))
        }
        for mode, by_steps in sorted(grouped.items())
    }


def _long_sweep_baseline_accuracy(samples: Sequence[CunxonLongSweepSample]) -> dict[str, float]:
    baselines = {
        "always_execute": "execute",
        "always_query": "query",
        "always_retry": "retry",
    }
    if not samples:
        return {name: 0.0 for name in baselines}
    return {
        name: sum(1 for sample in samples if action == sample.expected_action) / len(samples)
        for name, action in sorted(baselines.items())
    }


def _count_unique_readouts_by_mode(
    samples: Sequence[CunxonSensitivityProbeSample],
) -> dict[str, int]:
    readouts: dict[str, set[tuple[int, ...]]] = {}
    for sample in samples:
        readouts.setdefault(sample.mode, set()).add(tuple(sample.readout))
    return {mode: len(values) for mode, values in readouts.items()}


def _count_actions_by_mode(
    samples: Sequence[CunxonSensitivityProbeSample],
) -> dict[str, dict[str, int]]:
    distributions: dict[str, dict[str, int]] = {}
    for sample in samples:
        mode_distribution = distributions.setdefault(sample.mode, {})
        previous_count = mode_distribution.get(sample.normalized_action, 0)
        mode_distribution[sample.normalized_action] = previous_count + 1
    return distributions


def _count_action_changes_by_stimulus(
    samples: Sequence[CunxonSensitivityProbeSample],
) -> dict[str, int]:
    by_stimulus_seed: dict[tuple[str, int], dict[str, str]] = {}
    for sample in samples:
        actions = by_stimulus_seed.setdefault((sample.stimulus, sample.seed_offset), {})
        actions[sample.mode] = sample.normalized_action
    change_counts: dict[str, int] = {
        stimulus: 0 for stimulus, _, _ in _default_action_probe_specs()
    }
    for (stimulus, _seed_offset), actions in by_stimulus_seed.items():
        if actions.get("infer") != actions.get("train"):
            change_counts[stimulus] = change_counts.get(stimulus, 0) + 1
    return change_counts


def _default_action_probe_specs() -> list[tuple[str, tuple[float, float, float], str]]:
    """Return a tiny fixed action-contract task suite for cuNxon probes."""
    return [
        ("execute-positive-drive", (1.0, 0.25, 0.0), "execute"),
        ("retry-negative-drive", (-1.0, -0.25, 0.0), "retry"),
        ("query-neutral-drive", (0.0, 0.0, 0.0), "query"),
    ]


def _default_supervised_motor_specs() -> list[
    tuple[str, str, tuple[float, float, float], str]
]:
    """Return train/holdout cases for absolute-output motor-target testing."""
    return [
        ("execute-train", "train", (1.0, 0.25, 0.0), "execute"),
        ("retry-train", "train", (-1.0, -0.25, 0.0), "retry"),
        ("query-train", "train", (0.0, 0.0, 0.0), "query"),
        ("execute-holdout-noisy", "holdout", (0.8, 0.2, 0.1), "execute"),
        ("retry-holdout-noisy", "holdout", (-0.8, -0.2, 0.1), "retry"),
        ("query-holdout-low-drive", "holdout", (0.05, 0.0, -0.05), "query"),
    ]


def _target_readout_for_action(action: str) -> tuple[int, int, int]:
    if action == "execute":
        return (1, 0, 0)
    if action == "retry":
        return (-1, 0, 0)
    return (0, 0, 0)


def _target_alignment(readout: Sequence[int], target: Sequence[int]) -> float:
    if not target:
        return 0.0
    matches = sum(1 for actual, expected in zip(readout, target) if actual == expected)
    return matches / len(target)


def _pack_supervised_motor_inputs(
    sensory_vector: Sequence[float], target_vector: Sequence[float | int]
) -> Any:
    combined = [float(value) for value in sensory_vector] + [
        float(value) for value in target_vector
    ]
    input_buffer = (C.c_float * len(combined))(*combined)
    input_pointer = C.cast(input_buffer, C.POINTER(C.c_float))
    ext_inputs = (C.POINTER(C.c_float) * 1)(input_pointer)
    ext_inputs._input_buffer = input_buffer  # type: ignore[attr-defined]
    return ext_inputs


def _supervised_accuracy_by_split(
    cases: Sequence[CunxonSupervisedMotorCase],
) -> dict[str, float]:
    by_split: dict[str, list[CunxonSupervisedMotorCase]] = {}
    for case in cases:
        by_split.setdefault(case.split, []).append(case)
    by_split["overall"] = list(cases)
    return {
        split: sum(1 for case in split_cases if case.outcome == "success") / len(split_cases)
        for split, split_cases in sorted(by_split.items())
        if split_cases
    }


def _supervised_target_alignment_by_split(
    cases: Sequence[CunxonSupervisedMotorCase],
) -> dict[str, float]:
    by_split: dict[str, list[CunxonSupervisedMotorCase]] = {}
    for case in cases:
        by_split.setdefault(case.split, []).append(case)
    by_split["overall"] = list(cases)
    return {
        split: sum(case.target_alignment for case in split_cases) / len(split_cases)
        for split, split_cases in sorted(by_split.items())
        if split_cases
    }


def _supervised_baseline_accuracy_by_split(
    cases: Sequence[CunxonSupervisedMotorCase],
) -> dict[str, dict[str, float]]:
    baseline_names = sorted({name for case in cases for name in case.baseline_actions})
    result: dict[str, dict[str, float]] = {}
    for baseline_name in baseline_names:
        split_scores: dict[str, float] = {}
        for split in sorted({case.split for case in cases} | {"overall"}):
            split_cases = (
                cases if split == "overall" else [case for case in cases if case.split == split]
            )
            if not split_cases:
                continue
            split_scores[split] = sum(
                1
                for case in split_cases
                if case.baseline_actions[baseline_name] == case.expected_action
            ) / len(split_cases)
        result[baseline_name] = split_scores
    return result


def _run_interface_readout_sample(
    *,
    lib: C.CDLL,
    ctx: C.c_void_p,
    mapping: str,
    port_ids: Sequence[int],
    neuron_class: str,
    steps: int,
) -> CunxonInterfaceReadoutSample:
    net = C.c_void_p()
    try:
        name = f"neuraxon_hybrid_cunxon_interface_{mapping}"
        _check(lib, lib.cunxonNetworkCreate(ctx, C.byref(net), name.encode("utf-8")))
        sphere_id = _add_interface_probe_sphere(
            lib, net, sphere_name=b"IFACE", readout_ids=port_ids
        )
        _check(lib, lib.cunxonNetworkFinalize(net))
        input_buffer = (C.c_float * 4)(1.0, -1.0, 0.25, 0.75)
        input_pointer = C.cast(input_buffer, C.POINTER(C.c_float))
        ext_inputs = (C.POINTER(C.c_float) * 1)(input_pointer)
        for _ in range(steps):
            _check(lib, lib.cunxonNetworkStepInfer(net, ext_inputs, C.c_float(1.0)))
        _check(lib, lib.cunxonContextSync(ctx))
        readout = _capture_readout(lib, net, sphere_id)
        states = _capture_snapshot_states(lib, net, sphere_id)
        snapshot_slice = [states[index] for index in port_ids]
        return CunxonInterfaceReadoutSample(
            mapping=mapping,
            port_ids=list(port_ids),
            neuron_class=neuron_class,
            readout=readout,
            snapshot_slice=snapshot_slice,
            matches_snapshot_slice=readout == snapshot_slice,
            active_state_count=sum(1 for value in readout if value != 0),
            signed_sum=sum(readout),
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)


def _run_interface_relay_sample(
    *,
    lib: C.CDLL,
    ctx: C.c_void_p,
    mapping: str,
    source_port_ids: Sequence[int],
    source_neuron_class: str,
    steps: int,
) -> CunxonInterfaceRelaySample:
    net = C.c_void_p()
    try:
        name = f"neuraxon_hybrid_cunxon_interface_relay_{mapping}"
        _check(lib, lib.cunxonNetworkCreate(ctx, C.byref(net), name.encode("utf-8")))
        source_id = _add_interface_probe_sphere(
            lib,
            net,
            sphere_name=b"SRC",
            readout_ids=source_port_ids,
            relay_output_ids=source_port_ids,
        )
        downstream_id = _add_interface_probe_sphere(
            lib,
            net,
            sphere_name=b"DST",
            sphere_type=CUNXON_SPHERE_ASSOCIATION,
            readout_ids=[0, 1, 2, 3],
            sensory_ids=None,
            relay_input_ids=[0, 1, 2, 3],
            seed_offset=80,
        )
        link_params = _default_link_params()
        link_id = C.c_int(-1)
        _check(
            lib,
            lib.cunxonNetworkAddLink(
                net, source_id, downstream_id, C.byref(link_params), C.byref(link_id)
            ),
        )
        _check(lib, lib.cunxonNetworkFinalize(net))
        source_input = (C.c_float * 4)(1.0, -1.0, 0.25, 0.75)
        source_pointer = C.cast(source_input, C.POINTER(C.c_float))
        null_pointer = C.POINTER(C.c_float)()
        ext_inputs = (C.POINTER(C.c_float) * 2)(source_pointer, null_pointer)
        for _ in range(steps):
            _check(lib, lib.cunxonNetworkStepInfer(net, ext_inputs, C.c_float(1.0)))
        _check(lib, lib.cunxonContextSync(ctx))
        source_readout = _capture_readout(lib, net, source_id)
        downstream_readout = _capture_readout(lib, net, downstream_id)
        return CunxonInterfaceRelaySample(
            mapping=mapping,
            source_port_ids=list(source_port_ids),
            source_neuron_class=source_neuron_class,
            source_relay_readout=source_readout,
            downstream_input_readout=downstream_readout,
            downstream_active_state_count=sum(1 for value in downstream_readout if value != 0),
            downstream_energy=_capture_energy(lib, net),
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)


def _add_interface_probe_sphere(
    lib: C.CDLL,
    net: C.c_void_p,
    *,
    sphere_name: bytes,
    readout_ids: Sequence[int],
    seed_offset: int = 79,
    sphere_type: int = CUNXON_SPHERE_SENSORY,
    sensory_ids: Sequence[int] | None = (0, 1, 2, 3),
    relay_input_ids: Sequence[int] | None = None,
    relay_output_ids: Sequence[int] | None = None,
) -> int:
    params = _NetworkParameters()
    _check(lib, lib.cunxonGetDefaultParameters(C.byref(params)))
    params.num_input_neurons = 4
    params.num_hidden_neurons = 4
    params.num_output_neurons = 4
    params.random_seed_offset = seed_offset
    params.synapse_death_prob = 0.0
    params.synapse_formation_prob = 0.0
    sphere_id = C.c_int(-1)
    _check(
        lib,
        lib.cunxonNetworkAddSphere(
            net, sphere_name, sphere_type, C.byref(params), C.byref(sphere_id)
        ),
    )
    sensory_ids_array = _optional_int_array(sensory_ids)
    relay_input_ids_array = _optional_int_array(relay_input_ids)
    relay_output_ids_array = _optional_int_array(relay_output_ids)
    readout_ids_array = (C.c_int * len(readout_ids))(*readout_ids)
    _check(
        lib,
        lib.cunxonNetworkSetSphereInterface(
            net,
            sphere_id.value,
            sensory_ids_array,
            len(sensory_ids) if sensory_ids is not None else 0,
            relay_input_ids_array,
            len(relay_input_ids) if relay_input_ids is not None else 0,
            relay_output_ids_array,
            len(relay_output_ids) if relay_output_ids is not None else 0,
            readout_ids_array,
            len(readout_ids),
        ),
    )
    return sphere_id.value


def _optional_int_array(values: Sequence[int] | None) -> Any:
    if values is None:
        return None
    return (C.c_int * len(values))(*values)


def _capture_snapshot_states(lib: C.CDLL, net: C.c_void_p, sphere_id: int) -> list[int]:
    n_neurons = C.c_int(0)
    _check(
        lib,
        lib.cunxonSphereSnapshot(
            net, sphere_id, None, None, None, None, None, None, C.byref(n_neurons)
        ),
    )
    if n_neurons.value <= 0:
        raise CunxonError("cuNxon snapshot returned an empty neuron count")
    size = n_neurons.value
    states = (C.c_int8 * size)()
    _check(
        lib,
        lib.cunxonSphereSnapshot(
            net, sphere_id, None, None, None, states, None, None, C.byref(n_neurons)
        ),
    )
    state_values = [int(states[index]) for index in range(size)]
    validate_trinary_readout(state_values)
    return state_values


def _capture_readout(lib: C.CDLL, net: C.c_void_p, sphere_id: int) -> list[int]:
    readout_size = C.c_int(0)
    _check(lib, lib.cunxonSphereGetReadout(net, sphere_id, None, C.byref(readout_size)))
    if readout_size.value <= 0:
        raise CunxonError("cuNxon returned an empty readout size")
    readout_buffer = (C.c_int8 * readout_size.value)()
    _check(
        lib,
        lib.cunxonSphereGetReadout(net, sphere_id, readout_buffer, C.byref(readout_size)),
    )
    readout = [int(readout_buffer[i]) for i in range(readout_size.value)]
    validate_trinary_readout(readout)
    return readout


def _capture_energy(lib: C.CDLL, net: C.c_void_p) -> float:
    energy = C.c_double(0.0)
    _check(lib, lib.cunxonNetworkGetEnergy(net, C.byref(energy)))
    return float(energy.value)


def _make_vram_resident_result(
    *,
    status: str,
    hypothesis: str,
    active_issue: str,
    pid: int,
    command: str,
    upstream_commit: str,
    cunxon_commit: str,
    library_path: str,
    device_name: str,
    compute_capability: str,
    started_at: datetime,
    expected_end_timestamp: float,
    max_runtime_seconds: int,
    sample_interval_seconds: int,
    steps_per_sample: int,
    total_steps: int,
    samples: list[CunxonVramResidentSample],
    stop_condition: str,
) -> CunxonVramResidentResult:
    now = datetime.now(timezone.utc)
    next_poll_timestamp = min(now.timestamp() + sample_interval_seconds, expected_end_timestamp)
    expected_end_at = _format_utc_datetime(
        datetime.fromtimestamp(expected_end_timestamp, timezone.utc)
    )
    next_poll_after = _format_utc_datetime(
        datetime.fromtimestamp(next_poll_timestamp, timezone.utc)
    )
    return CunxonVramResidentResult(
        status=status,
        hypothesis=hypothesis,
        active_issue=active_issue,
        pid=pid,
        command=command,
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        library_path=library_path,
        device_name=device_name,
        compute_capability=compute_capability,
        started_at=_format_utc_datetime(started_at),
        updated_at=_format_utc_datetime(now),
        expected_end_at=expected_end_at,
        next_poll_after=next_poll_after,
        max_runtime_seconds=max_runtime_seconds,
        sample_interval_seconds=sample_interval_seconds,
        steps_per_sample=steps_per_sample,
        total_steps=total_steps,
        samples=samples,
        stop_condition=stop_condition,
        notes=[
            "same cuNxon process keeps one network/context resident between samples",
            "samples capture readout, full-state occupancy, energy and nvidia-smi resource data",
            "runtime/dynamics evidence only; not a task-learning or intelligence claim",
        ],
    )


def _query_nvidia_smi_sample(device_id: int) -> dict[str, int | None]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "memory_used_mb": None,
            "utilization_percent": None,
            "temperature_c": None,
        }
    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    return {
        "memory_used_mb": _parse_optional_int(parts[0] if len(parts) > 0 else ""),
        "utilization_percent": _parse_optional_int(parts[1] if len(parts) > 1 else ""),
        "temperature_c": _parse_optional_int(parts[2] if len(parts) > 2 else ""),
    }


def _parse_optional_int(raw: str) -> int | None:
    try:
        return int(raw)
    except ValueError:
        return None


def _format_optional_int(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _format_utc_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _format_trinary(value: int) -> str:
    return "+1" if value == 1 else str(value)


def _load_library(path: Path) -> C.CDLL:
    lib = C.CDLL(str(path))
    lib.cunxonGetStatusString.argtypes = [C.c_int]
    lib.cunxonGetStatusString.restype = C.c_char_p
    lib.cunxonGetLastError.argtypes = []
    lib.cunxonGetLastError.restype = C.c_char_p
    lib.cunxonCreateContext.argtypes = [C.POINTER(C.c_void_p), C.c_int, C.c_uint64, C.c_uint32]
    lib.cunxonCreateContext.restype = C.c_int
    lib.cunxonDestroyContext.argtypes = [C.c_void_p]
    lib.cunxonDestroyContext.restype = C.c_int
    lib.cunxonContextSync.argtypes = [C.c_void_p]
    lib.cunxonContextSync.restype = C.c_int
    lib.cunxonContextGetProperty.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_size_t]
    lib.cunxonContextGetProperty.restype = C.c_int
    lib.cunxonNetworkCreate.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.c_char_p]
    lib.cunxonNetworkCreate.restype = C.c_int
    lib.cunxonNetworkDestroy.argtypes = [C.c_void_p]
    lib.cunxonNetworkDestroy.restype = C.c_int
    lib.cunxonGetDefaultParameters.argtypes = [C.POINTER(_NetworkParameters)]
    lib.cunxonGetDefaultParameters.restype = C.c_int
    lib.cunxonNetworkAddSphere.argtypes = [
        C.c_void_p,
        C.c_char_p,
        C.c_int,
        C.POINTER(_NetworkParameters),
        C.POINTER(C.c_int),
    ]
    lib.cunxonNetworkAddSphere.restype = C.c_int
    lib.cunxonNetworkSetSphereInterface.argtypes = [
        C.c_void_p,
        C.c_int,
        C.POINTER(C.c_int),
        C.c_int,
        C.POINTER(C.c_int),
        C.c_int,
        C.POINTER(C.c_int),
        C.c_int,
        C.POINTER(C.c_int),
        C.c_int,
    ]
    lib.cunxonNetworkSetSphereInterface.restype = C.c_int
    lib.cunxonNetworkAddLink.argtypes = [
        C.c_void_p,
        C.c_int,
        C.c_int,
        C.POINTER(_LinkParameters),
        C.POINTER(C.c_int),
    ]
    lib.cunxonNetworkAddLink.restype = C.c_int
    lib.cunxonNetworkFinalize.argtypes = [C.c_void_p]
    lib.cunxonNetworkFinalize.restype = C.c_int
    lib.cunxonNetworkStepInfer.argtypes = [C.c_void_p, C.POINTER(C.POINTER(C.c_float)), C.c_float]
    lib.cunxonNetworkStepInfer.restype = C.c_int
    lib.cunxonNetworkStepTrain.argtypes = [C.c_void_p, C.POINTER(C.POINTER(C.c_float)), C.c_float]
    lib.cunxonNetworkStepTrain.restype = C.c_int
    lib.cunxonNetworkInjectNeuromodulator.argtypes = [C.c_void_p, C.c_int, C.c_float]
    lib.cunxonNetworkInjectNeuromodulator.restype = C.c_int
    lib.cunxonSphereGetReadout.argtypes = [
        C.c_void_p,
        C.c_int,
        C.POINTER(C.c_int8),
        C.POINTER(C.c_int),
    ]
    lib.cunxonSphereGetReadout.restype = C.c_int
    lib.cunxonSphereSnapshot.argtypes = [
        C.c_void_p,
        C.c_int,
        C.POINTER(C.c_float),
        C.POINTER(C.c_float),
        C.POINTER(C.c_float),
        C.POINTER(C.c_int8),
        C.POINTER(C.c_float),
        C.POINTER(C.c_float),
        C.POINTER(C.c_int),
    ]
    lib.cunxonSphereSnapshot.restype = C.c_int
    lib.cunxonNetworkGetEnergy.argtypes = [C.c_void_p, C.POINTER(C.c_double)]
    lib.cunxonNetworkGetEnergy.restype = C.c_int
    lib.cunxonNetworkStorePattern.argtypes = [
        C.c_void_p,
        C.c_int,
        C.c_char_p,
        C.POINTER(C.c_float),
        C.c_int,
        C.c_int,
        C.c_float,
    ]
    lib.cunxonNetworkStorePattern.restype = C.c_int
    lib.cunxonNetworkRecallPattern.argtypes = [
        C.c_void_p,
        C.c_int,
        C.c_char_p,
        C.c_int,
        C.c_float,
        C.c_int,
        C.c_float,
        C.POINTER(C.c_int8),
        C.POINTER(C.c_int),
    ]
    lib.cunxonNetworkRecallPattern.restype = C.c_int
    lib.cunxonNetworkListPatterns.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_int)]
    lib.cunxonNetworkListPatterns.restype = C.c_int
    lib.cunxonNetworkClearPatterns.argtypes = [C.c_void_p]
    lib.cunxonNetworkClearPatterns.restype = C.c_int
    return lib


def _check(lib: C.CDLL, status: int) -> None:
    if status == CUNXON_OK:
        return
    status_raw = lib.cunxonGetStatusString(status)
    status_name = status_raw.decode("utf-8", errors="replace") if status_raw else str(status)
    error_raw = lib.cunxonGetLastError()
    error = error_raw.decode("utf-8", errors="replace") if error_raw else ""
    detail = f" ({error})" if error else ""
    raise CunxonError(f"cuNxon error: {status_name}{detail}")


def _query_device_name(lib: C.CDLL, ctx: C.c_void_p) -> str:
    buffer = C.create_string_buffer(256)
    _check(
        lib,
        lib.cunxonContextGetProperty(ctx, CUNXON_PROP_DEVICE_NAME, buffer, C.sizeof(buffer)),
    )
    return buffer.value.decode("utf-8", errors="replace")


def _query_compute_capability(lib: C.CDLL, ctx: C.c_void_p) -> str:
    cc = C.c_int(0)
    _check(
        lib,
        lib.cunxonContextGetProperty(
            ctx, CUNXON_PROP_COMPUTE_CAPABILITY, C.byref(cc), C.sizeof(cc)
        ),
    )
    return f"{cc.value // 10}.{cc.value % 10}"


class _LinkParameters(C.Structure):
    _fields_ = [
        ("kind", C.c_int),
        ("coherence_band", C.c_int),
        ("gain", C.c_float),
        ("delay_steps", C.c_int),
        ("transmission_threshold", C.c_float),
        ("coherence_strength", C.c_float),
        ("topology", C.c_int),
        ("sparse_prob", C.c_float),
        ("allow_negative_weights", C.c_int),
        ("plasticity_rate", C.c_float),
        ("weight_decay", C.c_float),
        ("weight_clip", C.c_float),
        ("normalize_rows", C.c_int),
        ("bias", C.c_float),
    ]


class _NetworkParameters(C.Structure):
    _fields_ = [
        ("num_input_neurons", C.c_int),
        ("num_hidden_neurons", C.c_int),
        ("num_output_neurons", C.c_int),
        ("num_dendritic_branches", C.c_int),
        ("dendritic_spike_threshold", C.c_float),
        ("dendritic_supralinear_gamma", C.c_float),
        ("ws_k", C.c_int),
        ("ws_beta", C.c_float),
        ("membrane_time_constant", C.c_float),
        ("firing_threshold_excitatory", C.c_float),
        ("firing_threshold_inhibitory", C.c_float),
        ("adaptation_tau", C.c_float),
        ("autoreceptor_tau", C.c_float),
        ("spontaneous_firing_rate", C.c_float),
        ("neuron_health_decay", C.c_float),
        ("target_firing_rate", C.c_float),
        ("homeostatic_rate", C.c_float),
        ("firing_rate_alpha", C.c_float),
        ("threshold_mod_k", C.c_float),
        ("msth_ultrafast_tau", C.c_float),
        ("msth_ultrafast_ceiling", C.c_float),
        ("msth_fast_tau", C.c_float),
        ("msth_fast_gain", C.c_float),
        ("msth_medium_tau", C.c_float),
        ("msth_medium_gain", C.c_float),
        ("msth_slow_tau", C.c_float),
        ("msth_slow_gain", C.c_float),
        ("dsn_kernel_size", C.c_int),
        ("dsn_enabled", C.c_int),
        ("dsn_bias", C.c_float),
        ("dsn_learn_enabled", C.c_int),
        ("dsn_learn_lr", C.c_float),
        ("dsn_target_sensitivity", C.c_float),
        ("dsn_target_bias", C.c_float),
        ("dsn_kernel_clip", C.c_float),
        ("ctsn_rho", C.c_float),
        ("ctsn_enabled", C.c_int),
        ("ctsn_phi_gain", C.c_float),
        ("ctsn_phi_bias", C.c_float),
        ("ctsn_learn_enabled", C.c_int),
        ("ctsn_learn_lr", C.c_float),
        ("ctsn_phi_gain_clip", C.c_float),
        ("ctsn_phi_bias_clip", C.c_float),
        ("tau_fast", C.c_float),
        ("tau_slow", C.c_float),
        ("tau_meta", C.c_float),
        ("tau_stdp", C.c_float),
        ("w_fast_init_min", C.c_float),
        ("w_fast_init_max", C.c_float),
        ("w_slow_init_min", C.c_float),
        ("w_slow_init_max", C.c_float),
        ("w_meta_init_min", C.c_float),
        ("w_meta_init_max", C.c_float),
        ("chrono_alpha_f", C.c_float),
        ("chrono_alpha_s", C.c_float),
        ("chrono_lambda_f", C.c_float),
        ("chrono_lambda_s", C.c_float),
        ("chrono_enabled", C.c_int),
        ("chrono_trace_clip", C.c_float),
        ("chrono_gate_norm", C.c_float),
        ("chrono_raw_clip", C.c_float),
        ("chrono_omega_min", C.c_float),
        ("chrono_omega_max", C.c_float),
        ("chrono_omega_smoothing", C.c_float),
        ("agmp_lambda_e", C.c_float),
        ("agmp_lambda_a", C.c_float),
        ("agmp_eta", C.c_float),
        ("agmp_enabled", C.c_int),
        ("learning_rate", C.c_float),
        ("stdp_window", C.c_float),
        ("associative_alpha", C.c_float),
        ("synapse_integrity_threshold", C.c_float),
        ("synapse_formation_prob", C.c_float),
        ("synapse_death_prob", C.c_float),
        ("neuron_death_threshold", C.c_float),
        ("dopamine_baseline", C.c_float),
        ("serotonin_baseline", C.c_float),
        ("acetylcholine_baseline", C.c_float),
        ("norepinephrine_baseline", C.c_float),
        ("tau_tonic", C.c_float),
        ("tau_phasic", C.c_float),
        ("neuromod_release_rate", C.c_float),
        ("receptor_concentration_cap", C.c_float),
        ("osc_freq", C.c_float * 6),
        ("osc_amplitude", C.c_float),
        ("osc_pac_strength", C.c_float),
        ("random_seed_offset", C.c_uint64),
    ]
