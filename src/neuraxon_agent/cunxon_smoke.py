"""cuNxon GPU smoke helpers and report rendering.

This module intentionally keeps the cuNxon path as an optional investigation
surface. Loading the CUDA library is explicit and runtime-only so normal CPU
benchmarks/tests keep working without NVIDIA/CUDA dependencies.
"""

from __future__ import annotations

import ctypes as C
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

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
_CUNXON_FITNESS_FN = C.CFUNCTYPE(C.c_float, C.c_void_p, C.c_void_p)


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
class CunxonAigarthReadoutRun:
    """One Aigarth evolution run for a specific readout-port mapping."""

    mapping: str
    readout_ids: list[int]
    neuron_class: str
    baseline_margin: float
    generation_margins: list[float]
    final_margin: float
    improvement: float
    positive_mean: float
    negative_mean: float
    positive_readout: list[int]
    negative_readout: list[int]


@dataclass(frozen=True)
class CunxonAigarthReadoutProbeResult:
    """Aigarth evolution result contrasting demo-relative vs absolute-output readouts."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    generations: int
    population_size: int
    eval_steps: int
    runs: list[CunxonAigarthReadoutRun]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonAigarthActionCase:
    """One train/holdout case from an Aigarth-evolved action readout probe."""

    name: str
    split: str
    input_vector: list[float]
    expected_action: str
    target_readout: list[int]
    readout: list[int]
    decoded_action: str
    normalized_action: str
    confidence: float
    outcome: str
    target_alignment: float
    baseline_actions: dict[str, str]
    energy: float


@dataclass(frozen=True)
class CunxonAigarthActionProbeResult:
    """Aigarth evolution result scored through the production action decoder."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    generations: int
    population_size: int
    eval_steps: int
    readout_ids: list[int]
    generation_train_scores: list[float]
    cases: list[CunxonAigarthActionCase]
    accuracy_by_split: dict[str, float]
    target_alignment_by_split: dict[str, float]
    baseline_accuracy_by_split: dict[str, dict[str, float]]
    unique_readouts: int
    action_distribution: dict[str, int]
    seed_offset: int = 82
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
class CunxonAigarthActionSeedRun:
    """One seed from an Aigarth action repeatability sweep."""

    seed_offset: int
    generation_train_scores: list[float]
    accuracy_by_split: dict[str, float]
    target_alignment_by_split: dict[str, float]
    baseline_accuracy_by_split: dict[str, dict[str, float]]
    unique_readouts: int
    action_distribution: dict[str, int]
    cases: list[CunxonAigarthActionCase]


@dataclass(frozen=True)
class CunxonAigarthActionSeedSweepResult:
    """Multi-seed repeatability audit for Aigarth action evidence."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    generations: int
    population_size: int
    eval_steps: int
    readout_ids: list[int]
    seed_offsets: list[int]
    runs: list[CunxonAigarthActionSeedRun]
    accuracy_summary_by_split: dict[str, dict[str, float]]
    aggregate_action_distribution: dict[str, int]
    seeds_beating_baseline_by_split: dict[str, int]
    notes: list[str] = field(default_factory=list)

    @property
    def seed_count(self) -> int:
        """Return the number of seed offsets evaluated."""
        return len(self.runs)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["seed_count"] = self.seed_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonAigarthActionHardHoldoutResult:
    """Hard-holdout/leakage audit for Aigarth action evidence."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    generations: int
    population_size: int
    eval_steps: int
    readout_ids: list[int]
    seed_offsets: list[int]
    strict_expected_actions: list[str]
    runs: list[CunxonAigarthActionSeedRun]
    accuracy_summary_by_split: dict[str, dict[str, float]]
    aggregate_action_distribution: dict[str, int]
    seeds_beating_baseline_by_split: dict[str, int]
    unexpected_action_count: int
    unexpected_action_rate: float
    leakage_control_accuracy_mean: float
    train_to_hard_holdout_gap_mean: float
    fitness_variant: str = "success_plus_alignment"
    notes: list[str] = field(default_factory=list)

    @property
    def seed_count(self) -> int:
        """Return the number of seed offsets evaluated."""
        return len(self.runs)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["seed_count"] = self.seed_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonAigarthStressAmplitudeSplitSummary:
    """Aggregate separability metrics for one stress amplitude split."""

    split: str
    amplitude_factor: float
    sample_count: int
    accuracy_mean: float
    best_constant_baseline_mean: float
    seeds_beating_best_baseline: int
    query_collapse_rate: float
    execute_retry_accuracy: float
    action_distribution: dict[str, int]


@dataclass(frozen=True)
class CunxonAigarthStressAmplitudeLadderResult:
    """Bounded stress-vector amplitude-ladder diagnostic."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    generations: int
    population_size: int
    eval_steps: int
    seed_offsets: list[int]
    amplitude_factors: list[float]
    split_summaries: list[CunxonAigarthStressAmplitudeSplitSummary]
    original_stress_holdout_accuracy_mean: float
    original_stress_holdout_query_collapse_rate: float
    best_scaled_stress_holdout_accuracy_mean: float
    best_scaled_stress_holdout_amplitude_factor: float | None
    aggregate_action_distribution: dict[str, int]
    evidence_boundary: str
    recommended_next_probe: dict[str, object]
    notes: list[str] = field(default_factory=list)

    @property
    def seed_count(self) -> int:
        """Return the number of seed offsets evaluated."""
        return len(self.seed_offsets)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["seed_count"] = self.seed_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonAigarthStressObjectiveResult:
    """Target-aligned objective-shaping diagnostic after the stress amplitude ladder."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    generations: int
    population_size: int
    eval_steps: int
    seed_offsets: list[int]
    amplitude_factor: float
    fitness_variant: str
    split_summaries: list[CunxonAigarthStressAmplitudeSplitSummary]
    original_stress_holdout_accuracy_mean: float
    original_stress_holdout_query_collapse_rate: float
    original_stress_holdout_execute_retry_accuracy: float
    scaled_stress_holdout_accuracy_mean: float
    scaled_stress_holdout_query_collapse_rate: float
    scaled_stress_holdout_execute_retry_accuracy: float
    counterfactual_control_accuracy_mean: float
    permuted_control_accuracy_mean: float
    aggregate_action_distribution: dict[str, int]
    evidence_boundary: str
    recommended_next_probe: dict[str, object]
    notes: list[str] = field(default_factory=list)

    @property
    def seed_count(self) -> int:
        """Return the number of seed offsets evaluated."""
        return len(self.seed_offsets)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["seed_count"] = self.seed_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonBranchingRegimeRun:
    """Per-seed activity-regime proxy metrics coupled to action quality."""

    seed_offset: int
    active_state_sequence: list[int]
    branching_activity_ratio_proxy: float
    neutral_occupancy: float
    transition_entropy_bits: float
    energy_slope: float
    readout_diversity: int
    regime_label: str
    accuracy_by_split: dict[str, float]
    best_constant_baseline_by_split: dict[str, float]
    beats_best_baseline_by_split: dict[str, bool]
    action_distribution: dict[str, int]
    gpu_memory_used_mb: int | None = None
    gpu_utilization_percent: int | None = None
    gpu_temperature_c: int | None = None


@dataclass(frozen=True)
class CunxonBranchingRegimeScanResult:
    """Small cuNxon branching/activity-regime scan with task-coupled baselines."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    source_probe: str
    generations: int
    population_size: int
    eval_steps: int
    seed_offsets: list[int]
    runs: list[CunxonBranchingRegimeRun]
    regime_bucket_summary: dict[str, dict[str, float | int]]
    correlation_summary: dict[str, float]
    verdict: str
    notes: list[str] = field(default_factory=list)

    @property
    def run_count(self) -> int:
        """Return the number of per-seed runs in the scan."""
        return len(self.runs)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["run_count"] = self.run_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonAvalancheWindowSample:
    """One full-sphere snapshot window for a cuNxon stimulus/action case."""

    mode: str
    seed_offset: int
    stimulus: str
    input_vector: list[float]
    expected_action: str
    active_state_sequence: list[int]
    activation_event_sequence: list[int]
    deactivation_event_sequence: list[int]
    branching_ratio_estimate: float
    active_count_ratio_mean: float
    neutral_occupancy: float
    transition_entropy_bits: float
    avalanche_event_count: int
    mean_avalanche_size: float
    max_avalanche_size: int
    final_readout: list[int]
    normalized_action: str
    outcome: str
    energy_delta: float
    elapsed_ms: float
    split: str = "direct"
    gpu_memory_used_mb: int | None = None
    gpu_utilization_percent: int | None = None
    gpu_temperature_c: int | None = None


@dataclass(frozen=True)
class CunxonAvalancheWindowProbeResult:
    """Snapshot-level avalanche/branching diagnostic coupled to action baselines."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    modes: list[str]
    seed_offsets: list[int]
    steps: int
    sample_interval: int
    samples: list[CunxonAvalancheWindowSample]
    accuracy_by_mode: dict[str, float]
    baseline_accuracy: dict[str, float]
    correlation_summary: dict[str, float]
    verdict: str
    notes: list[str] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        """Return the number of mode/seed/stimulus windows in the probe."""
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
class CunxonAvalancheInterventionTaskConfigSummary:
    """Per-intervention summary coupling avalanche metrics to split task quality."""

    id: str
    steps: int
    sample_interval: int
    sample_count: int
    mean_branching_ratio_estimate: float
    branching_ratio_estimate_range: list[float]
    mean_neutral_occupancy: float
    accuracy_by_split: dict[str, float]
    best_constant_baseline_by_split: dict[str, float]
    beats_best_constant_baseline_by_split: dict[str, bool]


@dataclass(frozen=True)
class CunxonAvalancheInterventionTaskCorrelationResult:
    """Task-coupled avalanche intervention/correlation matrix."""

    status: str
    hypothesis_for_this_slice: str
    source_claim_ids: list[str]
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    seed_offsets: list[int]
    modes: list[str]
    configurations: list[CunxonAvalancheInterventionTaskConfigSummary]
    samples: list[CunxonAvalancheWindowSample]
    split_accuracy: dict[str, float]
    best_constant_baseline_by_split: dict[str, float]
    configurations_beating_stress_baseline: list[str]
    correlation_summary: dict[str, float]
    verdict: str
    evidence_boundary: str
    notes: list[str] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        """Return the total number of live windows sampled."""
        return len(self.samples)

    @property
    def config_count(self) -> int:
        """Return the number of intervention configurations."""
        return len(self.configurations)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["sample_count"] = self.sample_count
        data["config_count"] = self.config_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonControlledRegimeCalibrationConfigSummary:
    """Per-drive-regime calibration summary for the snapshot estimator."""

    id: str
    drive_scale: float
    steps: int
    sample_interval: int
    sample_count: int
    mean_branching_ratio_estimate: float
    branching_ratio_estimate_range: list[float]
    mean_active_count_ratio: float
    mean_neutral_occupancy: float
    mean_transition_entropy_bits: float
    action_distribution: dict[str, int]
    accuracy_by_split: dict[str, float]
    best_constant_baseline_by_split: dict[str, float]
    beats_best_constant_baseline_by_split: dict[str, bool]


@dataclass(frozen=True)
class CunxonControlledRegimeCalibrationResult:
    """Controlled low/medium/high drive calibration of the cuNxon criticality estimator."""

    status: str
    hypothesis_for_this_slice: str
    source_issue: str
    source_claim_ids: list[str]
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    seed_offsets: list[int]
    modes: list[str]
    steps: int
    sample_interval: int
    regime_drive_scales: dict[str, float]
    configurations: list[CunxonControlledRegimeCalibrationConfigSummary]
    samples: list[CunxonAvalancheWindowSample]
    split_accuracy: dict[str, float]
    best_constant_baseline_by_split: dict[str, float]
    stress_holdout_accuracy: float
    configurations_beating_stress_baseline: list[str]
    correlation_summary: dict[str, float]
    verdict: str
    evidence_boundary: str
    notes: list[str] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        """Return the total number of calibration windows sampled."""
        return len(self.samples)

    @property
    def config_count(self) -> int:
        """Return the number of controlled regimes."""
        return len(self.configurations)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["sample_count"] = self.sample_count
        data["config_count"] = self.config_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CunxonAigarthActionRemapCase:
    """One Aigarth action case replayed through a strict motor-lane remap."""

    source_path: str
    seed_offset: int
    name: str
    split: str
    expected_action: str
    readout: list[int]
    original_normalized_action: str
    remapped_action: str
    original_outcome: str
    remapped_outcome: str


@dataclass(frozen=True)
class CunxonAigarthActionRemapSourceSummary:
    """Per-source summary for a post-hoc Aigarth action remap audit."""

    source_path: str
    status: str
    fitness_variant: str
    case_count: int
    original_accuracy_by_split: dict[str, float]
    remapped_accuracy_by_split: dict[str, float]
    original_action_distribution: dict[str, int]
    remapped_action_distribution: dict[str, int]
    original_unexpected_action_count: int
    remapped_unexpected_action_count: int


@dataclass(frozen=True)
class CunxonAigarthActionRemapAuditResult:
    """Post-hoc decoder/remap audit for existing Aigarth action artifacts."""

    status: str
    source_paths: list[str]
    remap_strategy: str
    strict_expected_actions: list[str]
    source_summaries: list[CunxonAigarthActionRemapSourceSummary]
    cases: list[CunxonAigarthActionRemapCase]
    original_accuracy_by_split: dict[str, float]
    remapped_accuracy_by_split: dict[str, float]
    original_action_distribution: dict[str, int]
    remapped_action_distribution: dict[str, int]
    original_unexpected_action_count: int
    remapped_unexpected_action_count: int
    remap_accuracy_delta_by_split: dict[str, float]
    notes: list[str] = field(default_factory=list)

    @property
    def source_count(self) -> int:
        """Return the number of source artifacts replayed."""
        return len(self.source_paths)

    @property
    def case_count(self) -> int:
        """Return the number of Aigarth action cases replayed."""
        return len(self.cases)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        data = asdict(self)
        data["source_count"] = self.source_count
        data["case_count"] = self.case_count
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this result as stable JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_artifact_paths(
        cls,
        artifact_paths: Sequence[str | Path],
    ) -> "CunxonAigarthActionRemapAuditResult":
        """Build a post-hoc remap audit from tracked Aigarth action JSON artifacts."""
        if not artifact_paths:
            raise ValueError("artifact_paths must contain at least one path")
        strict_actions = ["execute", "query", "retry"]
        strict_set = set(strict_actions)
        source_summaries: list[CunxonAigarthActionRemapSourceSummary] = []
        all_cases: list[CunxonAigarthActionRemapCase] = []
        normalized_paths = [str(Path(path)) for path in artifact_paths]
        for artifact_path in artifact_paths:
            path = Path(artifact_path)
            data = json.loads(path.read_text(encoding="utf-8"))
            status = str(data.get("status", "unknown"))
            fitness_variant = str(data.get("fitness_variant", "unknown"))
            source_cases: list[CunxonAigarthActionRemapCase] = []
            for run in data.get("runs", []):
                if not isinstance(run, dict):
                    continue
                seed_offset = int(run.get("seed_offset", -1))
                for raw_case in run.get("cases", []):
                    if not isinstance(raw_case, dict):
                        continue
                    readout = [int(value) for value in raw_case.get("readout", [])]
                    original_action = str(raw_case.get("normalized_action", "unknown"))
                    expected_action = str(raw_case.get("expected_action", "unknown"))
                    remapped_action = remap_aigarth_action_readout(readout)
                    replayed = CunxonAigarthActionRemapCase(
                        source_path=str(path),
                        seed_offset=seed_offset,
                        name=str(raw_case.get("name", "unknown")),
                        split=str(raw_case.get("split", "unknown")),
                        expected_action=expected_action,
                        readout=readout,
                        original_normalized_action=original_action,
                        remapped_action=remapped_action,
                        original_outcome=(
                            "success" if original_action == expected_action else "failure"
                        ),
                        remapped_outcome=(
                            "success" if remapped_action == expected_action else "failure"
                        ),
                    )
                    source_cases.append(replayed)
                    all_cases.append(replayed)
            source_summaries.append(
                CunxonAigarthActionRemapSourceSummary(
                    source_path=str(path),
                    status=status,
                    fitness_variant=fitness_variant,
                    case_count=len(source_cases),
                    original_accuracy_by_split=_remap_case_accuracy_by_split(
                        source_cases, remapped=False
                    ),
                    remapped_accuracy_by_split=_remap_case_accuracy_by_split(
                        source_cases, remapped=True
                    ),
                    original_action_distribution=_remap_case_action_distribution(
                        source_cases, remapped=False
                    ),
                    remapped_action_distribution=_remap_case_action_distribution(
                        source_cases, remapped=True
                    ),
                    original_unexpected_action_count=sum(
                        1
                        for case in source_cases
                        if case.original_normalized_action not in strict_set
                    ),
                    remapped_unexpected_action_count=sum(
                        1 for case in source_cases if case.remapped_action not in strict_set
                    ),
                )
            )

        original_accuracy = _remap_case_accuracy_by_split(all_cases, remapped=False)
        remapped_accuracy = _remap_case_accuracy_by_split(all_cases, remapped=True)
        split_keys = set(original_accuracy) | set(remapped_accuracy)
        return cls(
            status="aigarth action remap audit completed",
            source_paths=normalized_paths,
            remap_strategy="signed-first-lane",
            strict_expected_actions=strict_actions,
            source_summaries=source_summaries,
            cases=all_cases,
            original_accuracy_by_split=original_accuracy,
            remapped_accuracy_by_split=remapped_accuracy,
            original_action_distribution=_remap_case_action_distribution(all_cases, remapped=False),
            remapped_action_distribution=_remap_case_action_distribution(all_cases, remapped=True),
            original_unexpected_action_count=sum(
                1 for case in all_cases if case.original_normalized_action not in strict_set
            ),
            remapped_unexpected_action_count=sum(
                1 for case in all_cases if case.remapped_action not in strict_set
            ),
            remap_accuracy_delta_by_split={
                split: remapped_accuracy.get(split, 0.0) - original_accuracy.get(split, 0.0)
                for split in sorted(split_keys)
            },
            notes=[
                "post-hoc diagnostic over existing live cuNxon Aigarth artifacts",
                "signed-first-lane remap follows the project target_readout contract: "
                "+ first lane=execute, - first lane=retry, neutral=query",
                "does not create new cuNxon learning evidence; it only isolates decoder "
                "vocabulary effects",
            ],
        )


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


@dataclass(frozen=True)
class CunxonInputProxyTargetCase:
    """One train/holdout case from an input-neuron proxy-target probe."""

    name: str
    split: str
    input_vector: list[float]
    expected_action: str
    target_proxy_readout: list[int]
    motor_readout: list[int]
    decoded_action: str
    normalized_action: str
    confidence: float
    outcome: str
    target_proxy_alignment: float
    baseline_actions: dict[str, str]
    energy: float


@dataclass(frozen=True)
class CunxonExternalDriveWindowSample:
    """One sample comparing sensory-id drive against a neuron-class window."""

    mode: str
    driven_port_class: str
    sensory_port_ids: list[int]
    readout_port_ids: list[int]
    readout: list[int]
    snapshot_slice: list[int]
    matches_snapshot_slice: bool
    active_state_count: int
    signed_sum: int
    energy: float


@dataclass(frozen=True)
class CunxonExternalDriveWindowProbeResult:
    """Probe result for input/hidden/output external-drive window semantics."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    steps: int
    input_vector: list[float]
    samples: list[CunxonExternalDriveWindowSample]
    notes: list[str] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        """Return the number of port-window samples."""
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
class CunxonInputProxyTargetProbeResult:
    """Probe result for supported input-class target-proxy semantics."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    train_epochs: int
    train_steps_per_case: int
    eval_steps: int
    target_proxy_port_ids: list[int]
    motor_readout_port_ids: list[int]
    cases: list[CunxonInputProxyTargetCase]
    accuracy_by_split: dict[str, float]
    target_proxy_alignment_by_split: dict[str, float]
    baseline_accuracy_by_split: dict[str, dict[str, float]]
    notes: list[str] = field(default_factory=list)

    @property
    def case_count(self) -> int:
        """Return the number of scored input-proxy target cases."""
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
class CunxonResidentActionCase:
    """One eval case from a task-coupled resident cuNxon action probe."""

    epoch: int
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
    motor_active_state_count: int
    elapsed_ms: float


@dataclass(frozen=True)
class CunxonResidentActionProbeResult:
    """Task-coupled resident cuNxon probe result scored across epochs."""

    status: str
    upstream_commit: str
    cunxon_commit: str
    library_path: str
    device_name: str
    compute_capability: str
    train_epochs: int
    train_steps_per_case: int
    eval_steps: int
    sphere_count: int
    cases: list[CunxonResidentActionCase]
    accuracy_by_epoch: dict[str, float]
    accuracy_by_split: dict[str, float]
    baseline_accuracy_by_split: dict[str, dict[str, float]]
    unique_motor_readouts: int
    notes: list[str] = field(default_factory=list)

    @property
    def case_count(self) -> int:
        """Return the number of scored resident action cases."""
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


def render_aigarth_readout_markdown_report(result: CunxonAigarthReadoutProbeResult) -> str:
    """Render an Aigarth readout mapping semantics report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    run_rows = [
        (
            "| Mapping | Ports | Neuron class | Baseline margin | Final margin | "
            "Improvement | Pos mean | Neg mean | Pos readout | Neg readout |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    trajectory_lines: list[str] = []
    for run in result.runs:
        ports = ", ".join(str(port) for port in run.readout_ids)
        pos = ", ".join(_format_trinary(value) for value in run.positive_readout)
        neg = ", ".join(_format_trinary(value) for value in run.negative_readout)
        run_rows.append(
            "| "
            f"{run.mapping} | [{ports}] | {run.neuron_class} | "
            f"{run.baseline_margin:.6f} | {run.final_margin:.6f} | "
            f"{run.improvement:.6f} | {run.positive_mean:.6f} | "
            f"{run.negative_mean:.6f} | [{pos}] | [{neg}] |"
        )
        margins = ", ".join(f"{margin:.6f}" for margin in run.generation_margins)
        trajectory_lines.append(f"- `{run.mapping}` generation margins: {margins or 'none'}")
    return "\n".join(
        [
            "# cuNxon Aigarth readout semantics probe",
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
            f"- Generations: {result.generations}",
            f"- Population size: {result.population_size}",
            f"- Eval steps per class: {result.eval_steps}",
            "",
            "## Why this probe exists",
            "",
            "Upstream `example_aigarth.cu` is the public supervised/evolutionary cuNxon example, "
            "but it configures readout ids `0..7` for a 4-input/32-hidden/8-output sphere. "
            "Those ids are relative to the sphere start and therefore alias input/hidden neurons, "
            "not the absolute output block `36..43`. This probe contrasts the demo-relative "
            "mapping with the absolute output mapping before treating Aigarth as a sanctioned "
            "motor/readout route for Neuraxon-Hybrid.",
            "",
            "## Mapping results",
            "",
            *run_rows,
            "",
            "## Evolution trajectories",
            "",
            *(trajectory_lines or ["- None"]),
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "Aigarth margin movement is useful interface evidence, but it does not prove "
            "intelligence, task learning, or a useful motor adapter unless absolute output "
            "readouts improve against baselines and survive holdout/generalization tests. "
            "Relative demo-readout improvement alone is confounded by input/hidden aliasing.",
            "",
        ]
    )


def write_aigarth_readout_artifacts(
    result: CunxonAigarthReadoutProbeResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown Aigarth readout semantics artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(render_aigarth_readout_markdown_report(result), encoding="utf-8")
    return json_output, markdown_output


def render_aigarth_action_markdown_report(result: CunxonAigarthActionProbeResult) -> str:
    """Render an Aigarth-evolved holdout/action probe report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    split_rows = ["| Split | Accuracy | Target alignment |", "| --- | ---: | ---: |"]
    for split, accuracy in sorted(result.accuracy_by_split.items()):
        alignment = result.target_alignment_by_split.get(split, 0.0)
        split_rows.append(f"| {split} | {accuracy:.6f} | {alignment:.6f} |")
    baseline_rows = ["| Baseline | Split | Accuracy |", "| --- | --- | ---: |"]
    for baseline, by_split in sorted(result.baseline_accuracy_by_split.items()):
        for split, accuracy in sorted(by_split.items()):
            baseline_rows.append(f"| {baseline} | {split} | {accuracy:.6f} |")
    generation_scores = ", ".join(f"{score:.6f}" for score in result.generation_train_scores)
    action_distribution = ", ".join(
        f"{action}={count}" for action, count in sorted(result.action_distribution.items())
    )
    case_rows = [
        "| Case | Split | Expected | Target | Readout | Decoded | Outcome | Alignment | Energy |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for case in result.cases:
        target = ", ".join(_format_trinary(value) for value in case.target_readout)
        readout = ", ".join(_format_trinary(value) for value in case.readout)
        case_rows.append(
            "| "
            f"{case.name} | {case.split} | {case.expected_action} | [{target}] | "
            f"[{readout}] | {case.decoded_action} ({case.normalized_action}, "
            f"{case.confidence:.4f}) | {case.outcome} | {case.target_alignment:.6f} | "
            f"{case.energy:.6g} |"
        )
    return "\n".join(
        [
            "# cuNxon Aigarth holdout action probe",
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
            f"- Generations: {result.generations}",
            f"- Population size: {result.population_size}",
            f"- Eval steps per case: {result.eval_steps}",
            f"- Readout ids: {', '.join(str(port) for port in result.readout_ids)}",
            f"- Cases: {result.case_count}",
            f"- Unique readouts: {result.unique_readouts}",
            f"- Action distribution: {action_distribution or 'none'}",
            "",
            "## Why this probe exists",
            "",
            "The earlier Aigarth readout-semantics probe showed that the public "
            "`cunxonNetworkAigarthStep` evolutionary surface can improve an absolute-output "
            "two-pattern margin. This probe makes that route task-coupled: the fitness "
            "callback uses train cases only, then the evolved network is evaluated on both "
            "train and holdout cases through the existing `ActionDecoder` action contract.",
            "",
            "## Fitness trajectory",
            "",
            f"- Train-only generation scores: {generation_scores or 'none'}",
            "",
            "## Accuracy by split",
            "",
            *split_rows,
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
            "This is Aigarth/evolutionary action-readout evidence, not an intelligence claim. "
            "A train fitness improvement, callable GPU evolution loop, or isolated train success "
            "does not prove intelligence, useful learning, or generalization. The holdout "
            "accuracy must beat trivial baselines before this route can be treated as useful "
            "adapter evidence.",
            "",
        ]
    )


def write_aigarth_action_artifacts(
    result: CunxonAigarthActionProbeResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown Aigarth action probe artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(render_aigarth_action_markdown_report(result), encoding="utf-8")
    return json_output, markdown_output


def render_aigarth_action_seed_sweep_markdown_report(
    result: CunxonAigarthActionSeedSweepResult,
) -> str:
    """Render a multi-seed Aigarth action repeatability report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    summary_rows = [
        "| Split | Mean accuracy | Min | Max | Seeds > best constant baseline |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, summary in sorted(result.accuracy_summary_by_split.items()):
        beating = result.seeds_beating_baseline_by_split.get(split, 0)
        summary_rows.append(
            f"| {split} | {summary.get('mean', 0.0):.6f} | "
            f"{summary.get('min', 0.0):.6f} | {summary.get('max', 0.0):.6f} | "
            f"{beating}/{result.seed_count} |"
        )
    run_rows = [
        "| Seed | Train | Holdout | Overall | Unique readouts | Actions | Final train score |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for run in result.runs:
        actions = ", ".join(
            f"{action}={count}" for action, count in sorted(run.action_distribution.items())
        )
        final_score = run.generation_train_scores[-1] if run.generation_train_scores else 0.0
        run_rows.append(
            f"| {run.seed_offset} | {run.accuracy_by_split.get('train', 0.0):.6f} | "
            f"{run.accuracy_by_split.get('holdout', 0.0):.6f} | "
            f"{run.accuracy_by_split.get('overall', 0.0):.6f} | "
            f"{run.unique_readouts} | {actions or 'none'} | {final_score:.6f} |"
        )
    aggregate_actions = ", ".join(
        f"{action}={count}"
        for action, count in sorted(result.aggregate_action_distribution.items())
    )
    expected_actions = {"execute", "query", "retry"}
    missing_actions = sorted(expected_actions - set(result.aggregate_action_distribution))
    unexpected_actions = sorted(set(result.aggregate_action_distribution) - expected_actions)
    coverage_bits = []
    if missing_actions:
        coverage_bits.append("partial action coverage: missing " + ", ".join(missing_actions))
    else:
        coverage_bits.append("all three action labels appeared at least once")
    if unexpected_actions:
        coverage_bits.append("unexpected normalized labels: " + ", ".join(unexpected_actions))
    coverage_note = "; ".join(coverage_bits)
    return "\n".join(
        [
            "# cuNxon Aigarth action seed sweep",
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
            f"- Generations per seed: {result.generations}",
            f"- Population size: {result.population_size}",
            f"- Eval steps per case: {result.eval_steps}",
            f"- Readout ids: {', '.join(str(port) for port in result.readout_ids)}",
            f"- Seed offsets: {', '.join(str(seed) for seed in result.seed_offsets)}",
            f"- Aggregate action distribution: {aggregate_actions or 'none'}",
            f"- Coverage note: {coverage_note}",
            "",
            "## Why this probe exists",
            "",
            "The single-seed Aigarth action probe was the first cuNxon action lane that "
            "beat constant-action baselines, but it still failed the retry class and used "
            "one engineered seed. This sweep reruns the same train-only Aigarth fitness "
            "protocol with a fresh cuNxon network/context per seed to test repeatability "
            "before treating the result as stronger adapter evidence.",
            "",
            "## Seed repeatability",
            "",
            *summary_rows,
            "",
            "## Per-seed runs",
            "",
            *run_rows,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "This is a repeatability audit, not intelligence evidence. A seed sweep can "
            "show whether a small Aigarth/evolutionary adapter result is stable or brittle, "
            "but it does not by itself prove broad generalization, useful learning, or "
            "production readiness. Partial action coverage remains a caveat until all "
            "expected action labels are learned and verified on harder holdouts.",
            "",
        ]
    )


def write_aigarth_action_seed_sweep_artifacts(
    result: CunxonAigarthActionSeedSweepResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown Aigarth action seed-sweep artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_aigarth_action_seed_sweep_markdown_report(result), encoding="utf-8"
    )
    return json_output, markdown_output


def render_aigarth_action_hard_holdout_markdown_report(
    result: CunxonAigarthActionHardHoldoutResult,
) -> str:
    """Render a hard-holdout and leakage/oracle audit for Aigarth action evidence."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    summary_rows = [
        "| Split | Mean accuracy | Min | Max | Seeds > best constant baseline |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, summary in sorted(result.accuracy_summary_by_split.items()):
        beating = result.seeds_beating_baseline_by_split.get(split, 0)
        summary_rows.append(
            f"| {split} | {summary.get('mean', 0.0):.6f} | "
            f"{summary.get('min', 0.0):.6f} | {summary.get('max', 0.0):.6f} | "
            f"{beating}/{result.seed_count} |"
        )
    run_rows = [
        "| Seed | Train | Holdout | Hard holdout | Permuted control | Overall | Actions |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in result.runs:
        actions = ", ".join(
            f"{action}={count}" for action, count in sorted(run.action_distribution.items())
        )
        run_rows.append(
            f"| {run.seed_offset} | {run.accuracy_by_split.get('train', 0.0):.6f} | "
            f"{run.accuracy_by_split.get('holdout', 0.0):.6f} | "
            f"{run.accuracy_by_split.get('hard_holdout', 0.0):.6f} | "
            f"{run.accuracy_by_split.get('permuted_control', 0.0):.6f} | "
            f"{run.accuracy_by_split.get('overall', 0.0):.6f} | {actions or 'none'} |"
        )
    aggregate_actions = ", ".join(
        f"{action}={count}"
        for action, count in sorted(result.aggregate_action_distribution.items())
    )
    expected = ", ".join(result.strict_expected_actions)
    unexpected_actions = sorted(
        set(result.aggregate_action_distribution) - set(result.strict_expected_actions)
    )
    unexpected_note = ", ".join(unexpected_actions) if unexpected_actions else "none"
    return "\n".join(
        [
            "# cuNxon Aigarth action hard-holdout audit",
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
            f"- Generations per seed: {result.generations}",
            f"- Population size: {result.population_size}",
            f"- Eval steps per case: {result.eval_steps}",
            f"- Readout ids: {', '.join(str(port) for port in result.readout_ids)}",
            f"- Seed offsets: {', '.join(str(seed) for seed in result.seed_offsets)}",
            f"- Fitness variant: `{result.fitness_variant}`",
            f"- Strict expected actions: {expected}",
            f"- Aggregate action distribution: {aggregate_actions or 'none'}",
            f"- Unexpected action count: {result.unexpected_action_count}",
            f"- Unexpected action rate: {result.unexpected_action_rate:.6f}",
            f"- Unexpected normalized labels: {unexpected_note}",
            f"- Train→hard-holdout gap mean: {result.train_to_hard_holdout_gap_mean:.6f}",
            "",
            "## Why this probe exists",
            "",
            "The Aigarth seed sweep partially repeated a tiny baseline-beating action signal, "
            "but it also produced unstable class coverage and unexpected normalized labels. "
            "This audit keeps the train-only Aigarth fitness route but expands evaluation with "
            "harder/noisier holdouts and a permuted-control leakage/oracle check before treating "
            "the route as stronger adapter evidence.",
            "",
            "## Accuracy by split",
            "",
            *summary_rows,
            "",
            "## Per-seed runs",
            "",
            *run_rows,
            "",
            "## Permuted-control leakage/oracle check",
            "",
            "The `permuted_control` split reuses train-like inputs with rotated expected "
            "labels. It is not optimized by the fitness callback. High accuracy here would "
            "be suspicious for label/oracle leakage; low accuracy is a sanity check, not "
            "positive intelligence evidence.",
            f"Mean permuted-control accuracy: {result.leakage_control_accuracy_mean:.6f}.",
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "This is a stress audit of a tiny Aigarth/evolutionary adapter route, not "
            "intelligence evidence. Hard-holdout accuracy, strict action-label coverage, "
            "and leakage controls must remain separate from claims about broad "
            "generalization, useful learning, or production readiness.",
            "",
        ]
    )


def render_aigarth_action_strict_label_markdown_report(
    result: CunxonAigarthActionHardHoldoutResult,
) -> str:
    """Render a strict-label/margin Aigarth action audit report."""
    return (
        render_aigarth_action_hard_holdout_markdown_report(result)
        .replace(
            "# cuNxon Aigarth action hard-holdout audit",
            "# cuNxon Aigarth strict-label action audit",
            1,
        )
        .replace(
            "The Aigarth seed sweep partially repeated a tiny baseline-beating action signal, "
            "but it also produced unstable class coverage and unexpected normalized labels. "
            "This audit keeps the train-only Aigarth fitness route but expands evaluation with "
            "harder/noisier holdouts and a permuted-control leakage/oracle check before treating "
            "the route as stronger adapter evidence.",
            "The hard-holdout audit left one out-of-contract normalized label and a large "
            "train-to-hard-holdout gap. This audit changes the train-only Aigarth fitness "
            "hypothesis: `strict_label_margin` rewards expected execute/query/retry outcomes "
            "and penalizes out-of-contract normalized labels such as assertive, while keeping "
            "holdout, hard-holdout, and permuted-control labels outside the fitness callback.",
            1,
        )
        .replace(
            "This is a stress audit of a tiny Aigarth/evolutionary adapter route, not ",
            "This is a strict-label stress audit of a tiny Aigarth/evolutionary adapter "
            "route, not ",
            1,
        )
    )


def render_aigarth_action_contract_penalty_markdown_report(
    result: CunxonAigarthActionHardHoldoutResult,
) -> str:
    """Render a heavier contract-penalty Aigarth action audit report."""
    return (
        render_aigarth_action_hard_holdout_markdown_report(result)
        .replace(
            "# cuNxon Aigarth action hard-holdout audit",
            "# cuNxon Aigarth contract-penalty action audit",
            1,
        )
        .replace(
            "The Aigarth seed sweep partially repeated a tiny baseline-beating action signal, "
            "but it also produced unstable class coverage and unexpected normalized labels. "
            "This audit keeps the train-only Aigarth fitness route but expands evaluation with "
            "harder/noisier holdouts and a permuted-control leakage/oracle check before treating "
            "the route as stronger adapter evidence.",
            "The strict-label audit improved hard-holdout accuracy but still emitted "
            "out-of-contract normalized labels. This audit keeps the same train-only Aigarth "
            "fitness route and holdout/control split, but applies a heavier unexpected-label "
            "penalty before increasing task difficulty.",
            1,
        )
        .replace(
            "This is a stress audit of a tiny Aigarth/evolutionary adapter route, not ",
            "This is a heavier contract-penalty stress audit of a tiny "
            "Aigarth/evolutionary adapter route, not ",
            1,
        )
    )


def render_aigarth_action_target_contract_markdown_report(
    result: CunxonAigarthActionHardHoldoutResult,
) -> str:
    """Render a target-contract Aigarth action audit report."""
    return (
        render_aigarth_action_hard_holdout_markdown_report(result)
        .replace(
            "# cuNxon Aigarth action hard-holdout audit",
            "# cuNxon Aigarth target-contract action audit",
            1,
        )
        .replace(
            "The Aigarth seed sweep partially repeated a tiny baseline-beating action signal, "
            "but it also produced unstable class coverage and unexpected normalized labels. "
            "This audit keeps the train-only Aigarth fitness route but expands evaluation with "
            "harder/noisier holdouts and a permuted-control leakage/oracle check before treating "
            "the route as stronger adapter evidence.",
            "The remap audit showed that a signed-first-lane decoder removes out-of-contract "
            "labels but does not improve accuracy when applied only post hoc. This audit "
            "moves that decoder contract into the train-only Aigarth fitness itself: "
            "`target_contract_margin` decodes with the signed-first-lane project contract "
            "and rewards target-readout margin while keeping holdout, hard-holdout, and "
            "permuted-control labels outside the fitness callback.",
            1,
        )
        .replace(
            "This is a stress audit of a tiny Aigarth/evolutionary adapter route, not ",
            "This is a target-contract stress audit of a tiny Aigarth/evolutionary adapter "
            "route, not ",
            1,
        )
    )


def render_aigarth_action_target_contract_stress_markdown_report(
    result: CunxonAigarthActionHardHoldoutResult,
) -> str:
    """Render a target-contract Aigarth action stress audit report."""
    return (
        render_aigarth_action_target_contract_markdown_report(result)
        .replace(
            "# cuNxon Aigarth target-contract action audit",
            "# cuNxon Aigarth target-contract stress audit",
            1,
        )
        .replace(
            "The remap audit showed that a signed-first-lane decoder removes out-of-contract "
            "labels but does not improve accuracy when applied only post hoc. This audit "
            "moves that decoder contract into the train-only Aigarth fitness itself: "
            "`target_contract_margin` decodes with the signed-first-lane project contract "
            "and rewards target-readout margin while keeping holdout, hard-holdout, and "
            "permuted-control labels outside the fitness callback.",
            "The target-contract audit is the strongest cuNxon action lane so far, but the "
            "task is tiny and one seed stayed baseline-level on hard-holdout. This audit "
            "keeps the same train-only `target_contract_margin` objective and adds "
            "harder/noisier and counterfactual splits (`stress_holdout` and "
            "`counterfactual_control`) before treating the route as stronger adapter evidence.",
            1,
        )
        .replace(
            "This is a target-contract stress audit of a tiny Aigarth/evolutionary adapter "
            "route, not ",
            "This is a harder target-contract stress audit of a tiny Aigarth/evolutionary "
            "adapter route, not ",
            1,
        )
    )


def render_aigarth_action_target_contract_augmented_train_markdown_report(
    result: CunxonAigarthActionHardHoldoutResult,
) -> str:
    """Render an augmented-train target-contract Aigarth action audit report."""
    return (
        render_aigarth_action_target_contract_stress_markdown_report(result)
        .replace(
            "# cuNxon Aigarth target-contract stress audit",
            "# cuNxon Aigarth target-contract augmented-train audit",
            1,
        )
        .replace(
            "The target-contract audit is the strongest cuNxon action lane so far, but the "
            "task is tiny and one seed stayed baseline-level on hard-holdout. This audit "
            "keeps the same train-only `target_contract_margin` objective and adds "
            "harder/noisier and counterfactual splits (`stress_holdout` and "
            "`counterfactual_control`) before treating the route as stronger adapter evidence.",
            "The target-contract stress audit exposed baseline-level low-margin stress "
            "holdout behavior. This follow-up keeps the signed-first-lane contract but "
            "tests a train-only objective with additional low-margin training cases: "
            "`target_contract_augmented_train` optimizes `train` plus `augmented_train` "
            "cases, while `stress_holdout`, hard holdout, and control labels stay outside "
            "the fitness callback.",
            1,
        )
        .replace(
            "This is a harder target-contract stress audit of a tiny Aigarth/evolutionary "
            "adapter route, not ",
            "This is an augmented-train target-contract stress audit of a tiny "
            "Aigarth/evolutionary adapter route, not ",
            1,
        )
    )


def render_aigarth_action_target_contract_stress_injection_markdown_report(
    result: CunxonAigarthActionHardHoldoutResult,
) -> str:
    """Render a target-contract stress-injection upper-bound audit report."""
    return (
        render_aigarth_action_target_contract_augmented_train_markdown_report(result)
        .replace(
            "# cuNxon Aigarth target-contract augmented-train audit",
            "# cuNxon Aigarth target-contract stress-injection audit",
            1,
        )
        .replace(
            "The target-contract stress audit exposed baseline-level low-margin stress "
            "holdout behavior. This follow-up keeps the signed-first-lane contract but "
            "tests a train-only objective with additional low-margin training cases: "
            "`target_contract_augmented_train` optimizes `train` plus `augmented_train` "
            "cases, while `stress_holdout`, hard holdout, and control labels stay outside "
            "the fitness callback.",
            "The criticality/decoder separation result points to low-margin execute/retry "
            "stress query-collapse. This diagnostic changes the question from "
            "generalization to an upper-bound: `target_contract_stress_injection` "
            "deliberately includes duplicated low-margin `stress_train` cases in the "
            "Aigarth fitness callback, while reporting the original `stress_holdout` and "
            "controls separately. This leaks stress-like labels into optimization and is "
            "therefore adapter-capability/debugging evidence only, not generalization "
            "evidence.",
            1,
        )
        .replace(
            "This is an augmented-train target-contract stress audit of a tiny "
            "Aigarth/evolutionary adapter route, not ",
            "This is a stress-injection upper-bound diagnostic for a tiny "
            "Aigarth/evolutionary adapter route, not ",
            1,
        )
    )


def _format_amplitude_factor(factor: float) -> str:
    return f"{factor:.1f}x"


def render_aigarth_action_target_contract_stress_amplitude_ladder_markdown_report(
    result: CunxonAigarthStressAmplitudeLadderResult,
) -> str:
    """Render a bounded stress-vector amplitude-ladder diagnostic report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    ladder = ", ".join(_format_amplitude_factor(factor) for factor in result.amplitude_factors)
    distribution = ", ".join(
        f"{action}={count}"
        for action, count in sorted(result.aggregate_action_distribution.items())
    )
    split_rows = [
        (
            "| Split | Amplitude | Samples | Accuracy | Best constant baseline | "
            "Seeds > baseline | Query collapse | Execute/retry accuracy | Actions |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in result.split_summaries:
        actions = ", ".join(
            f"{action}={count}" for action, count in sorted(summary.action_distribution.items())
        )
        split_rows.append(
            "| "
            f"`{summary.split}` | {_format_amplitude_factor(summary.amplitude_factor)} | "
            f"{summary.sample_count} | {summary.accuracy_mean:.6f} | "
            f"{summary.best_constant_baseline_mean:.6f} | "
            f"{summary.seeds_beating_best_baseline} | "
            f"{summary.query_collapse_rate:.6f} | "
            f"{summary.execute_retry_accuracy:.6f} | {actions} |"
        )
    best_amplitude = (
        _format_amplitude_factor(result.best_scaled_stress_holdout_amplitude_factor)
        if result.best_scaled_stress_holdout_amplitude_factor is not None
        else "n/a"
    )
    return "\n".join(
        [
            "# cuNxon Aigarth target-contract stress amplitude-ladder",
            "",
            f"Status: `{result.status}`",
            "",
            "## Hypothesis",
            "",
            "The stress geometry audit showed that low-margin execute/retry stress vectors "
            "collapsed to `query`. This bounded stress amplitude-ladder scales those "
            "vectors to test whether stronger stimulus drive reduces query collapse under "
            "the same signed-first-lane target-contract route.",
            "",
            "## Runtime",
            "",
            f"- Device: `{result.device_name}` / compute capability `{result.compute_capability}`",
            f"- Library: `{result.library_path}`",
            f"- Seeds: `{result.seed_offsets}`",
            f"- Generations/population/eval steps: `{result.generations}` / "
            f"`{result.population_size}` / `{result.eval_steps}`",
            f"- Amplitude ladder: {ladder}",
            "",
            "## Core result",
            "",
            "- Original stress_holdout accuracy: "
            f"`{result.original_stress_holdout_accuracy_mean:.6f}`",
            f"- Original stress_holdout query-collapse rate: "
            f"`{result.original_stress_holdout_query_collapse_rate:.6f}`",
            f"- Best scaled stress_holdout accuracy: "
            f"`{result.best_scaled_stress_holdout_accuracy_mean:.6f}` at `{best_amplitude}`",
            f"- Aggregate actions: {distribution}",
            "",
            "## Amplitude split summaries",
            "",
            *split_rows,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            result.evidence_boundary,
            "",
            "This is a label-injected separability upper-bound over scaled stress vectors, "
            "not intelligence evidence and not a generalization claim. Stress/control "
            "quality must beat constant baselines before any useful-computation claim.",
            "",
            "## Recommended next probe",
            "",
            f"- Probe id: `{result.recommended_next_probe.get('id', 'unknown')}`",
            "",
        ]
    )


def write_aigarth_action_hard_holdout_artifacts(
    result: CunxonAigarthActionHardHoldoutResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown Aigarth hard-holdout audit artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_aigarth_action_hard_holdout_markdown_report(result), encoding="utf-8"
    )
    return json_output, markdown_output


def write_aigarth_action_strict_label_artifacts(
    result: CunxonAigarthActionHardHoldoutResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown Aigarth strict-label audit artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_aigarth_action_strict_label_markdown_report(result), encoding="utf-8"
    )
    return json_output, markdown_output


def write_aigarth_action_contract_penalty_artifacts(
    result: CunxonAigarthActionHardHoldoutResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown Aigarth contract-penalty audit artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_aigarth_action_contract_penalty_markdown_report(result), encoding="utf-8"
    )
    return json_output, markdown_output


def write_aigarth_action_target_contract_artifacts(
    result: CunxonAigarthActionHardHoldoutResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown Aigarth target-contract audit artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_aigarth_action_target_contract_markdown_report(result), encoding="utf-8"
    )
    return json_output, markdown_output


def write_aigarth_action_target_contract_stress_artifacts(
    result: CunxonAigarthActionHardHoldoutResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown Aigarth target-contract stress audit artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_aigarth_action_target_contract_stress_markdown_report(result), encoding="utf-8"
    )
    return json_output, markdown_output


def write_aigarth_action_target_contract_augmented_train_artifacts(
    result: CunxonAigarthActionHardHoldoutResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown Aigarth target-contract augmented-train artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_aigarth_action_target_contract_augmented_train_markdown_report(result),
        encoding="utf-8",
    )
    return json_output, markdown_output


def write_aigarth_action_target_contract_stress_injection_artifacts(
    result: CunxonAigarthActionHardHoldoutResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON/Markdown Aigarth target-contract stress-injection artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_aigarth_action_target_contract_stress_injection_markdown_report(result),
        encoding="utf-8",
    )
    return json_output, markdown_output


def write_aigarth_action_target_contract_stress_amplitude_ladder_artifacts(
    result: CunxonAigarthStressAmplitudeLadderResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON/Markdown Aigarth stress amplitude-ladder artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_aigarth_action_target_contract_stress_amplitude_ladder_markdown_report(result),
        encoding="utf-8",
    )
    return json_output, markdown_output


def render_aigarth_action_target_contract_stress_objective_markdown_report(
    result: CunxonAigarthStressObjectiveResult,
) -> str:
    """Render the target-aligned stress objective diagnostic report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    distribution = ", ".join(
        f"{action}={count}"
        for action, count in sorted(result.aggregate_action_distribution.items())
    )
    split_rows = [
        (
            "| Split | Amplitude | Samples | Accuracy | Best constant baseline | "
            "Seeds > baseline | Query collapse | Execute/retry accuracy | Actions |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in result.split_summaries:
        actions = ", ".join(
            f"{action}={count}" for action, count in sorted(summary.action_distribution.items())
        )
        split_rows.append(
            "| "
            f"`{summary.split}` | {_format_amplitude_factor(summary.amplitude_factor)} | "
            f"{summary.sample_count} | {summary.accuracy_mean:.6f} | "
            f"{summary.best_constant_baseline_mean:.6f} | "
            f"{summary.seeds_beating_best_baseline} | "
            f"{summary.query_collapse_rate:.6f} | "
            f"{summary.execute_retry_accuracy:.6f} | {actions} |"
        )
    return "\n".join(
        [
            "# cuNxon Aigarth target-contract stress objective",
            "",
            f"Status: `{result.status}`",
            "",
            "## Hypothesis",
            "",
            "After the stress amplitude-ladder showed that high-amplitude stress vectors "
            "are separable while original low-margin stress_holdout stays collapsed, this "
            "probe tests one target-aligned, margin-weighted objective. The goal is to "
            "preserve the scaled separability signal while checking whether original "
            "stress/control quality improves without upgrading label-injected training "
            "cases to generalization evidence.",
            "",
            "## Runtime",
            "",
            f"- Device: `{result.device_name}` / compute capability `{result.compute_capability}`",
            f"- Library: `{result.library_path}`",
            f"- Seeds: `{result.seed_offsets}`",
            f"- Generations/population/eval steps: `{result.generations}` / "
            f"`{result.population_size}` / `{result.eval_steps}`",
            f"- Fitness variant: `{result.fitness_variant}`",
            f"- Stress amplitude factor: `{_format_amplitude_factor(result.amplitude_factor)}`",
            "",
            "## Core result",
            "",
            "- Original stress_holdout accuracy: "
            f"`{result.original_stress_holdout_accuracy_mean:.6f}`",
            "- Original stress_holdout query-collapse rate: "
            f"`{result.original_stress_holdout_query_collapse_rate:.6f}`",
            "- Original stress_holdout execute/retry accuracy: "
            f"`{result.original_stress_holdout_execute_retry_accuracy:.6f}`",
            "- Scaled stress_holdout accuracy: "
            f"`{result.scaled_stress_holdout_accuracy_mean:.6f}`",
            "- Scaled stress_holdout query-collapse rate: "
            f"`{result.scaled_stress_holdout_query_collapse_rate:.6f}`",
            "- Scaled stress_holdout execute/retry accuracy: "
            f"`{result.scaled_stress_holdout_execute_retry_accuracy:.6f}`",
            "- Counterfactual control accuracy: "
            f"`{result.counterfactual_control_accuracy_mean:.6f}`",
            "- Permuted control accuracy: "
            f"`{result.permuted_control_accuracy_mean:.6f}`",
            f"- Aggregate actions: {distribution}",
            "",
            "## Split summaries",
            "",
            *split_rows,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            result.evidence_boundary,
            "",
            "This target-aligned stress objective is an objective-shaping diagnostic, not "
            "intelligence evidence. If original stress/control quality stays at constant "
            "baselines, the next question is decoder/readout geometry rather than a longer "
            "run of the same objective.",
            "",
            "## Recommended next probe",
            "",
            f"- Probe id: `{result.recommended_next_probe.get('id', 'unknown')}`",
            "",
        ]
    )


def write_aigarth_action_target_contract_stress_objective_artifacts(
    result: CunxonAigarthStressObjectiveResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON/Markdown Aigarth target-contract stress objective artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_aigarth_action_target_contract_stress_objective_markdown_report(result),
        encoding="utf-8",
    )
    return json_output, markdown_output


def render_aigarth_action_target_contract_supervised_low_margin_markdown_report(
    result: CunxonAigarthStressObjectiveResult,
) -> str:
    """Render the supervised low-margin target-objective diagnostic report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    distribution = ", ".join(
        f"{action}={count}"
        for action, count in sorted(result.aggregate_action_distribution.items())
    )
    split_rows = [
        (
            "| Split | Samples | Accuracy | Best constant baseline | "
            "Seeds > baseline | Query collapse | Execute/retry accuracy | Actions |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in result.split_summaries:
        actions = ", ".join(
            f"{action}={count}" for action, count in sorted(summary.action_distribution.items())
        )
        split_rows.append(
            "| "
            f"`{summary.split}` | {summary.sample_count} | {summary.accuracy_mean:.6f} | "
            f"{summary.best_constant_baseline_mean:.6f} | "
            f"{summary.seeds_beating_best_baseline} | "
            f"{summary.query_collapse_rate:.6f} | "
            f"{summary.execute_retry_accuracy:.6f} | {actions} |"
        )
    return "\n".join(
        [
            "# cuNxon Aigarth target-contract supervised low-margin objective",
            "",
            f"Status: `{result.status}`",
            "",
            "## Hypothesis",
            "",
            "The low-margin readout geometry probe showed that original execute/retry "
            "stress lanes sit on the wrong side of the observed query boundary. This "
            "diagnostic tests a supervised low-margin objective: train cases include "
            "normalized low-margin target examples, while original stress_holdout and controls "
            "stay outside the fitness callback.",
            "",
            "## Runtime",
            "",
            f"- Device: `{result.device_name}` / compute capability `{result.compute_capability}`",
            f"- Library: `{result.library_path}`",
            f"- Seeds: `{result.seed_offsets}`",
            f"- Generations/population/eval steps: `{result.generations}` / "
            f"`{result.population_size}` / `{result.eval_steps}`",
            f"- Fitness variant: `{result.fitness_variant}`",
            "",
            "## Core result",
            "",
            "- Original stress_holdout accuracy: "
            f"`{result.original_stress_holdout_accuracy_mean:.6f}`",
            "- Original stress_holdout query-collapse rate: "
            f"`{result.original_stress_holdout_query_collapse_rate:.6f}`",
            "- Original stress_holdout execute/retry accuracy: "
            f"`{result.original_stress_holdout_execute_retry_accuracy:.6f}`",
            "- Counterfactual control accuracy: "
            f"`{result.counterfactual_control_accuracy_mean:.6f}`",
            "- Permuted control accuracy: "
            f"`{result.permuted_control_accuracy_mean:.6f}`",
            f"- Aggregate actions: {distribution}",
            "",
            "## Split summaries",
            "",
            *split_rows,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            result.evidence_boundary,
            "",
            "This supervised low-margin objective is an adapter/objective diagnostic, "
            "not intelligence evidence and not generalization evidence. Original stress/control "
            "quality must beat constant baselines across fresh seeds before any useful-computation "
            "claim changes.",
            "",
            "## Recommended next probe",
            "",
            f"- Probe id: `{result.recommended_next_probe.get('id', 'unknown')}`",
            "",
        ]
    )


def write_aigarth_action_target_contract_supervised_low_margin_artifacts(
    result: CunxonAigarthStressObjectiveResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON/Markdown supervised low-margin target objective artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_aigarth_action_target_contract_supervised_low_margin_markdown_report(result),
        encoding="utf-8",
    )
    return json_output, markdown_output


def render_avalanche_window_markdown_report(
    result: CunxonAvalancheWindowProbeResult,
) -> str:
    """Render a snapshot-level avalanche/branching diagnostic report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    sample_rows = [
        (
            "| Mode | Seed | Stimulus | Branching-ratio estimate | Active-ratio mean | "
            "Neutral occupancy | Avalanche events | Mean/max avalanche | Final action | Outcome |"
        ),
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for sample in result.samples:
        sample_rows.append(
            f"| {sample.mode} | {sample.seed_offset} | {sample.stimulus} | "
            f"{sample.branching_ratio_estimate:.6f} | {sample.active_count_ratio_mean:.6f} | "
            f"{sample.neutral_occupancy:.6f} | {sample.avalanche_event_count} | "
            f"{sample.mean_avalanche_size:.6f}/{sample.max_avalanche_size} | "
            f"{sample.normalized_action} | {sample.outcome} |"
        )
    accuracy_lines = [
        f"- {mode}: {value:.6f}" for mode, value in sorted(result.accuracy_by_mode.items())
    ] or ["- none"]
    baseline_lines = [
        f"- {name}: {value:.6f}" for name, value in sorted(result.baseline_accuracy.items())
    ] or ["- none"]
    correlation_lines = [
        f"- {name}: {value:.6f}" for name, value in sorted(result.correlation_summary.items())
    ] or ["- none"]
    return "\n".join(
        [
            "# cuNxon avalanche-window snapshot probe",
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
            f"- Modes: {', '.join(result.modes)}",
            f"- Seed offsets: {', '.join(str(seed) for seed in result.seed_offsets)}",
            f"- Steps per window: {result.steps}",
            f"- Snapshot sample interval: {result.sample_interval}",
            "",
            "## Why this probe exists",
            "",
            "Qubic NIA Vol. 8 frames branching ratio as an operational criticality "
            "indicator. This probe uses full-sphere snapshot windows rather than only "
            "final action readouts, then compares the branching-ratio estimate with "
            "action-contract accuracy and trivial baselines.",
            "",
            "## Snapshot window metrics",
            "",
            *sample_rows,
            "",
            "## Accuracy by mode",
            "",
            *accuracy_lines,
            "",
            "## Trivial baselines",
            "",
            *baseline_lines,
            "",
            "## Correlation summary",
            "",
            *correlation_lines,
            "",
            "## Verdict",
            "",
            result.verdict,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "This is a full-sphere snapshot-level branching/avalanche diagnostic. A "
            "branching-ratio estimate or visible avalanche activity is not intelligence "
            "evidence unless task quality beats baselines on held-out/control cases.",
            "",
        ]
    )


def write_avalanche_window_artifacts(
    result: CunxonAvalancheWindowProbeResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown avalanche-window probe artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(render_avalanche_window_markdown_report(result), encoding="utf-8")
    return json_output, markdown_output


def render_avalanche_intervention_task_correlation_markdown_report(
    result: CunxonAvalancheInterventionTaskCorrelationResult,
) -> str:
    """Render the task-coupled avalanche intervention/correlation report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    config_rows = [
        (
            "| Config | Steps | Interval | Samples | Mean branching | Branching range | "
            "Mean neutral | Splits beating constants |"
        ),
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for config in result.configurations:
        beating = [
            split
            for split, did_beat in sorted(config.beats_best_constant_baseline_by_split.items())
            if did_beat
        ]
        config_rows.append(
            f"| {config.id} | {config.steps} | {config.sample_interval} | "
            f"{config.sample_count} | {config.mean_branching_ratio_estimate:.6f} | "
            f"{config.branching_ratio_estimate_range[0]:.6f}.."
            f"{config.branching_ratio_estimate_range[1]:.6f} | "
            f"{config.mean_neutral_occupancy:.6f} | {', '.join(beating) or 'none'} |"
        )
    split_rows = [
        "| Split | Accuracy | Best constant baseline | Beats baseline? |",
        "| --- | ---: | ---: | --- |",
    ]
    for split, accuracy in sorted(result.split_accuracy.items()):
        baseline = result.best_constant_baseline_by_split.get(split, 0.0)
        split_rows.append(f"| {split} | {accuracy:.6f} | {baseline:.6f} | {accuracy > baseline} |")
    sample_rows = [
        (
            "| Elapsed | Mode | Seed | Split | Stimulus | Branching | Neutral | "
            "Action | Expected | Outcome |"
        ),
        "| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for sample in result.samples[:60]:
        sample_rows.append(
            f"| {sample.elapsed_ms:.0f}ms | {sample.mode} | {sample.seed_offset} | "
            f"{sample.split} | {sample.stimulus} | {sample.branching_ratio_estimate:.6f} | "
            f"{sample.neutral_occupancy:.6f} | {sample.normalized_action} | "
            f"{sample.expected_action} | {sample.outcome} |"
        )
    if len(result.samples) > 60:
        remaining = len(result.samples) - 60
        sample_rows.append(
            f"| ... | ... | ... | ... | ... | ... | ... | ... | ... | {remaining} more samples |"
        )
    correlation_lines = [
        f"- {name}: {value:.6f}" for name, value in sorted(result.correlation_summary.items())
    ] or ["- none"]
    return "\n".join(
        [
            "# cuNxon avalanche intervention/task correlation",
            "",
            f"Status: `{result.status}`",
            f"Hypothesis: `{result.hypothesis_for_this_slice}`",
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
            f"- Modes: {', '.join(result.modes)}",
            f"- Seed offsets: {', '.join(str(seed) for seed in result.seed_offsets)}",
            f"- Samples: {result.sample_count}",
            "",
            "## Why this probe exists",
            "",
            "This report follows Qubic NIA Vol. 8's branching-ratio/criticality claims by "
            "checking whether bounded avalanche/regime movements are coupled to held-out, "
            "stress and control action quality. The relevant comparator is constant baselines; "
            "criticality metrics alone are not intelligence evidence.",
            "",
            "## Intervention configurations",
            "",
            *config_rows,
            "",
            "## Split accuracy versus constant baselines",
            "",
            *split_rows,
            "",
            "## Correlation summary",
            "",
            *correlation_lines,
            "",
            "## Sample excerpt",
            "",
            *sample_rows,
            "",
            "## Verdict",
            "",
            result.verdict,
            "",
            "## Evidence boundary",
            "",
            result.evidence_boundary,
            "",
            "## Notes",
            "",
            notes,
            "",
        ]
    )


def write_avalanche_intervention_task_correlation_artifacts(
    result: CunxonAvalancheInterventionTaskCorrelationResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown task-coupled avalanche intervention artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_avalanche_intervention_task_correlation_markdown_report(result),
        encoding="utf-8",
    )
    return json_output, markdown_output


def render_controlled_regime_calibration_markdown_report(
    result: CunxonControlledRegimeCalibrationResult,
) -> str:
    """Render the controlled-regime criticality calibration report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    config_rows = [
        (
            "| Regime | Drive scale | Samples | Mean branching | Branching range | "
            "Mean active ratio | Mean neutral | Entropy | Actions | Splits beating constants |"
        ),
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for config in result.configurations:
        actions = ", ".join(
            f"{action}={count}" for action, count in sorted(config.action_distribution.items())
        )
        beating = [
            split
            for split, did_beat in sorted(config.beats_best_constant_baseline_by_split.items())
            if did_beat
        ]
        config_rows.append(
            f"| {config.id} | {config.drive_scale:.2f} | {config.sample_count} | "
            f"{config.mean_branching_ratio_estimate:.6f} | "
            f"{config.branching_ratio_estimate_range[0]:.6f}.."
            f"{config.branching_ratio_estimate_range[1]:.6f} | "
            f"{config.mean_active_count_ratio:.6f} | {config.mean_neutral_occupancy:.6f} | "
            f"{config.mean_transition_entropy_bits:.6f} | {actions or 'none'} | "
            f"{', '.join(beating) or 'none'} |"
        )
    split_rows = [
        "| Split | Accuracy | Best constant baseline | Beats baseline? |",
        "| --- | ---: | ---: | --- |",
    ]
    for split, accuracy in sorted(result.split_accuracy.items()):
        baseline = result.best_constant_baseline_by_split.get(split, 0.0)
        split_rows.append(f"| {split} | {accuracy:.6f} | {baseline:.6f} | {accuracy > baseline} |")
    correlation_lines = [
        f"- {name}: {value:.6f}" for name, value in sorted(result.correlation_summary.items())
    ] or ["- none"]
    sample_rows = [
        (
            "| Mode | Seed | Split | Stimulus | Input vector | Branching | Neutral | "
            "Action | Expected | Outcome |"
        ),
        "| --- | ---: | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for sample in result.samples[:60]:
        sample_rows.append(
            f"| {sample.mode} | {sample.seed_offset} | {sample.split} | {sample.stimulus} | "
            f"{sample.input_vector} | {sample.branching_ratio_estimate:.6f} | "
            f"{sample.neutral_occupancy:.6f} | {sample.normalized_action} | "
            f"{sample.expected_action} | {sample.outcome} |"
        )
    if len(result.samples) > 60:
        remaining = len(result.samples) - 60
        sample_rows.append(
            f"| ... | ... | ... | ... | ... | ... | ... | ... | ... | {remaining} more samples |"
        )
    return "\n".join(
        [
            "# cuNxon controlled-regime criticality calibration",
            "",
            f"Status: `{result.status}`",
            f"Hypothesis: `{result.hypothesis_for_this_slice}`",
            f"Source issue: {result.source_issue}",
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
            f"- Modes: {', '.join(result.modes)}",
            f"- Seed offsets: {', '.join(str(seed) for seed in result.seed_offsets)}",
            f"- Steps / sample interval: {result.steps} / {result.sample_interval}",
            f"- Samples: {result.sample_count}",
            "",
            "## Why this probe exists",
            "",
            "This report follows Qubic NIA Vol. 8 by calibrating the snapshot branching/"
            "avalanche estimator under controlled low, medium and high input-drive regimes. "
            "The goal is estimator calibration, not an intelligence claim; stress/control "
            "quality must beat constant baselines before runtime criticality is treated as "
            "useful computation evidence.",
            "",
            "## Controlled regimes",
            "",
            *config_rows,
            "",
            "## Split accuracy versus constant baselines",
            "",
            *split_rows,
            "",
            "## Correlation summary",
            "",
            *correlation_lines,
            "",
            "## Sample excerpt",
            "",
            *sample_rows,
            "",
            "## Verdict",
            "",
            result.verdict,
            "",
            "## Evidence boundary",
            "",
            result.evidence_boundary,
            "",
            "## Notes",
            "",
            notes,
            "",
        ]
    )


def write_controlled_regime_calibration_artifacts(
    result: CunxonControlledRegimeCalibrationResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown controlled-regime calibration artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_controlled_regime_calibration_markdown_report(result),
        encoding="utf-8",
    )
    return json_output, markdown_output


def render_branching_regime_scan_markdown_report(
    result: CunxonBranchingRegimeScanResult,
) -> str:
    """Render the cuNxon branching/activity-regime scan report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    run_rows = [
        (
            "| Seed | Regime | Branching/activity-ratio proxy | Neutral occupancy | "
            "Transition entropy | Energy slope | Readout diversity | Holdout | "
            "Stress holdout | Beats stress baseline | GPU mem/util/temp |"
        ),
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for run in result.runs:
        gpu = (
            f"{_format_optional_int(run.gpu_memory_used_mb)} MB / "
            f"{_format_optional_int(run.gpu_utilization_percent)}% / "
            f"{_format_optional_int(run.gpu_temperature_c)} C"
        )
        run_rows.append(
            f"| {run.seed_offset} | {run.regime_label} | "
            f"{run.branching_activity_ratio_proxy:.6f} | {run.neutral_occupancy:.6f} | "
            f"{run.transition_entropy_bits:.6f} | {run.energy_slope:.6f} | "
            f"{run.readout_diversity} | {run.accuracy_by_split.get('holdout', 0.0):.6f} | "
            f"{run.accuracy_by_split.get('stress_holdout', 0.0):.6f} | "
            f"{run.beats_best_baseline_by_split.get('stress_holdout', False)} | {gpu} |"
        )
    bucket_rows = [
        (
            "| Regime bucket | Seeds | Mean branching proxy | Mean holdout | "
            "Mean stress_holdout | Stress seeds > best constant baseline |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, summary in sorted(result.regime_bucket_summary.items()):
        bucket_rows.append(
            f"| {label} | {summary.get('seed_count', 0)} | "
            f"{float(summary.get('mean_branching_activity_ratio_proxy', 0.0)):.6f} | "
            f"{float(summary.get('mean_holdout_accuracy', 0.0)):.6f} | "
            f"{float(summary.get('mean_stress_holdout_accuracy', 0.0)):.6f} | "
            f"{int(summary.get('beats_stress_baseline_count', 0))} |"
        )
    correlation_lines = [
        f"- {name}: {value:.6f}" for name, value in sorted(result.correlation_summary.items())
    ] or ["- none"]
    return "\n".join(
        [
            "# cuNxon branching-ratio regime scan",
            "",
            f"Status: `{result.status}`",
            "",
            "## Source",
            "",
            f"- Upstream repo commit: `{result.upstream_commit}`",
            f"- cuNxon commit: `{result.cunxon_commit}`",
            f"- Library: `{result.library_path}`",
            f"- Source probe: `{result.source_probe}`",
            "",
            "## GPU/runtime",
            "",
            f"- Device: {result.device_name}",
            f"- Compute capability: {result.compute_capability}",
            f"- Generations per seed: {result.generations}",
            f"- Population size: {result.population_size}",
            f"- Eval steps per case: {result.eval_steps}",
            f"- Seed offsets: {', '.join(str(seed) for seed in result.seed_offsets)}",
            "",
            "## Why this probe exists",
            "",
            "Qubic NIA Vol. 8 makes branching ratio / criticality a concrete hypothesis. "
            "This scan treats the branching/activity-ratio proxy as instrumentation only: "
            "it must be read beside holdout, stress_holdout, controls, and best constant "
            "baseline scores before any adapter-quality conclusion is considered.",
            "",
            "## Per-seed regime metrics and task quality",
            "",
            *run_rows,
            "",
            "## Regime buckets",
            "",
            *bucket_rows,
            "",
            "## Correlation summary",
            "",
            *correlation_lines,
            "",
            "## Verdict",
            "",
            result.verdict,
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "The branching/activity-ratio proxy is a coarse readout/action-sample proxy, not a "
            "neuroscience-grade branching-ratio estimator. A near-critical-looking proxy is "
            "not intelligence evidence by itself; useful computation requires held-out task "
            "quality above best constant baseline and leakage/control checks.",
            "",
        ]
    )


def write_branching_regime_scan_artifacts(
    result: CunxonBranchingRegimeScanResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown branching-regime scan artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_branching_regime_scan_markdown_report(result), encoding="utf-8"
    )
    return json_output, markdown_output


def render_aigarth_action_remap_audit_markdown_report(
    result: CunxonAigarthActionRemapAuditResult,
) -> str:
    """Render a post-hoc Aigarth action decoder/remap audit report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    source_rows = [
        (
            "| Source | Fitness variant | Cases | Original overall | Remapped overall | "
            "Original unexpected | Remapped unexpected |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in result.source_summaries:
        source_rows.append(
            "| "
            f"{summary.source_path} | {summary.fitness_variant} | {summary.case_count} | "
            f"{summary.original_accuracy_by_split.get('overall', 0.0):.6f} | "
            f"{summary.remapped_accuracy_by_split.get('overall', 0.0):.6f} | "
            f"{summary.original_unexpected_action_count} | "
            f"{summary.remapped_unexpected_action_count} |"
        )
    split_rows = [
        "| Split | Original accuracy | Remapped accuracy | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    split_names = sorted(
        set(result.original_accuracy_by_split) | set(result.remapped_accuracy_by_split)
    )
    for split in split_names:
        split_rows.append(
            "| "
            f"{split} | {result.original_accuracy_by_split.get(split, 0.0):.6f} | "
            f"{result.remapped_accuracy_by_split.get(split, 0.0):.6f} | "
            f"{result.remap_accuracy_delta_by_split.get(split, 0.0):+.6f} |"
        )
    original_distribution = ", ".join(
        f"{action}={count}" for action, count in result.original_action_distribution.items()
    )
    remapped_distribution = ", ".join(
        f"{action}={count}" for action, count in result.remapped_action_distribution.items()
    )
    return "\n".join(
        [
            "# cuNxon Aigarth action remap audit",
            "",
            f"Status: `{result.status}`",
            "",
            "## Scope",
            "",
            "This is a post-hoc diagnostic over existing live cuNxon Aigarth action "
            "artifacts. It does not start a new GPU evolution run and does not create "
            "new cuNxon learning evidence. Its purpose is to isolate whether the "
            "out-of-contract `assertive` labels came from the generic `ActionDecoder` "
            "vocabulary rather than the evolved readout itself.",
            "",
            "## Remap strategy",
            "",
            f"- Strategy: `{result.remap_strategy}`",
            "- `readout[0] > 0` -> `execute`",
            "- `readout[0] < 0` -> `retry`",
            "- `readout[0] == 0` -> `query`",
            "- Expected actions remain `execute`, `query`, `retry`.",
            "",
            "## Source summary",
            "",
            *source_rows,
            "",
            "## Aggregate action distribution",
            "",
            f"- Original: {original_distribution or 'none'}",
            f"- Remapped: {remapped_distribution or 'none'}",
            f"- Original unexpected action count: {result.original_unexpected_action_count}",
            f"- Remapped unexpected action count: {result.remapped_unexpected_action_count}",
            "",
            "## Accuracy replay",
            "",
            *split_rows,
            "",
            "## Interpretation boundary",
            "",
            "Eliminating out-of-contract labels by remapping is not automatically an "
            "adapter improvement. If accuracy falls on train/holdout/hard-holdout splits, "
            "the generic decoder was not the only bottleneck. This remains decoder-contract "
            "diagnostics, not intelligence, broad generalization, or useful-learning evidence.",
            "",
            "## Notes",
            "",
            notes,
            "",
        ]
    )


def write_aigarth_action_remap_audit_artifacts(
    result: CunxonAigarthActionRemapAuditResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown Aigarth action remap audit artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_aigarth_action_remap_audit_markdown_report(result), encoding="utf-8"
    )
    return json_output, markdown_output


def render_resident_action_markdown_report(result: CunxonResidentActionProbeResult) -> str:
    """Render a task-coupled resident cuNxon action probe report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    epoch_rows = ["| Epoch | Accuracy |", "| ---: | ---: |"]
    for epoch, accuracy in sorted(result.accuracy_by_epoch.items(), key=lambda item: int(item[0])):
        epoch_rows.append(f"| {epoch} | {accuracy:.6f} |")
    split_rows = ["| Split | Accuracy |", "| --- | ---: |"]
    for split, accuracy in sorted(result.accuracy_by_split.items()):
        split_rows.append(f"| {split} | {accuracy:.6f} |")
    baseline_rows = ["| Baseline | Split | Accuracy |", "| --- | --- | ---: |"]
    for baseline, by_split in sorted(result.baseline_accuracy_by_split.items()):
        for split, accuracy in sorted(by_split.items()):
            baseline_rows.append(f"| {baseline} | {split} | {accuracy:.6f} |")
    case_rows = [
        (
            "| Epoch | Case | Split | Expected | Motor readout | Decoded | Outcome | "
            "Motor active | Energy | Elapsed ms |"
        ),
        "| ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for case in result.cases:
        motor = ", ".join(_format_trinary(value) for value in case.motor_readout)
        case_rows.append(
            "| "
            f"{case.epoch} | {case.name} | {case.split} | {case.expected_action} | "
            f"[{motor}] | {case.decoded_action} ({case.normalized_action}, "
            f"{case.confidence:.4f}) | {case.outcome} | "
            f"{case.motor_active_state_count} | {case.energy:.6g} | {case.elapsed_ms:.3f} |"
        )
    return "\n".join(
        [
            "# cuNxon resident task-coupled action probe",
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
            f"- Train epochs: {result.train_epochs}",
            f"- Train steps per case: {result.train_steps_per_case}",
            f"- Eval steps per case: {result.eval_steps}",
            f"- Cases: {result.case_count}",
            f"- Unique motor readouts: {result.unique_motor_readouts}",
            "",
            "## Why this probe exists",
            "",
            "The previous four-hour VRAM-resident run kept one small infer-only network alive "
            "but did not include task-coupled scoring. This probe keeps the same cuNxon "
            "network/context resident across repeated train/eval task epochs, decodes motor "
            "readout through the existing action contract, and compares every result with "
            "trivial constant-action baselines.",
            "",
            "## Accuracy by resident epoch",
            "",
            *epoch_rows,
            "",
            "## Accuracy by split",
            "",
            *split_rows,
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
            "This is task-coupled runtime/adapter evidence, not an intelligence claim. A "
            "same-process resident run, action decoding, or one positive case does not prove "
            "intelligence, useful learning, or generalization. The holdout accuracy must beat "
            "trivial baselines before this route can be treated as useful adapter evidence.",
            "",
        ]
    )


def write_resident_action_artifacts(
    result: CunxonResidentActionProbeResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown resident action probe artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(render_resident_action_markdown_report(result), encoding="utf-8")
    return json_output, markdown_output


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
            distribution_text = (
                ", ".join(f"{action}={count}" for action, count in sorted(distribution.items()))
                or "none"
            )
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
        distribution_text = (
            ", ".join(f"{action}={count}" for action, count in sorted(distribution.items()))
            or "none"
        )
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
        source = ", ".join(_format_trinary(value) for value in relay_sample.source_relay_readout)
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
    markdown_output.write_text(render_interface_semantics_markdown_report(result), encoding="utf-8")
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


def render_input_proxy_target_markdown_report(result: CunxonInputProxyTargetProbeResult) -> str:
    """Render an input-class target-proxy semantics probe report."""
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    accuracy_rows = [
        "| Split | Accuracy | Mean target-proxy alignment |",
        "| --- | ---: | ---: |",
    ]
    for split, accuracy in result.accuracy_by_split.items():
        alignment = result.target_proxy_alignment_by_split.get(split, 0.0)
        accuracy_rows.append(f"| {split} | {accuracy:.6f} | {alignment:.6f} |")
    baseline_rows = ["| Baseline | Split | Accuracy |", "| --- | --- | ---: |"]
    for baseline_name, split_scores in sorted(result.baseline_accuracy_by_split.items()):
        for split, score in split_scores.items():
            baseline_rows.append(f"| {baseline_name} | {split} | {score:.6f} |")
    case_rows = [
        (
            "| Case | Split | Expected | Target proxy | Motor readout | Decoded | Outcome | "
            "Proxy alignment | Energy |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for case in result.cases:
        target_proxy = ", ".join(_format_trinary(value) for value in case.target_proxy_readout)
        motor = ", ".join(_format_trinary(value) for value in case.motor_readout)
        case_rows.append(
            "| "
            f"{case.name} | {case.split} | {case.expected_action} | [{target_proxy}] | "
            f"[{motor}] | {case.decoded_action} ({case.normalized_action}, "
            f"{case.confidence:.4f}) | {case.outcome} | "
            f"{case.target_proxy_alignment:.6f} | {case.energy:.6g} |"
        )
    target_ports = ", ".join(str(port) for port in result.target_proxy_port_ids)
    motor_ports = ", ".join(str(port) for port in result.motor_readout_port_ids)
    return "\n".join(
        [
            "# cuNxon input-port proxy target probe",
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
            f"- Target-proxy input ports: [{target_ports}]",
            f"- Motor readout ports: [{motor_ports}]",
            "",
            "## Why this probe exists",
            "",
            "The source-semantics audit found that output-neuron teacher forcing is not "
            "supported by the public cuNxon step-input path. This probe moves the target "
            "drive to supported input-class external drive ports, then separately reads "
            "the motor output ports without target drive during evaluation.",
            "",
            "## Accuracy and proxy alignment",
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
            "This is an interface/adapter diagnostic and not a desired-output/error-channel "
            "claim. Seeing a target value on input-class proxy neurons only proves that "
            "the supported external-drive route is observable; useful learning would still "
            "require motor-output accuracy to beat trivial baselines on holdout cases. This "
            "probe does not prove intelligence, generalization, or useful learning by itself.",
            "",
        ]
    )


def write_input_proxy_target_artifacts(
    result: CunxonInputProxyTargetProbeResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown input-proxy target artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(render_input_proxy_target_markdown_report(result), encoding="utf-8")
    return json_output, markdown_output


def render_external_drive_window_markdown_report(
    result: CunxonExternalDriveWindowProbeResult,
) -> str:
    """Render an external-drive port-window semantics report."""
    input_vector = ", ".join(f"{value:.3g}" for value in result.input_vector)
    notes = "\n".join(f"- {note}" for note in result.notes) or "- None"
    rows = [
        (
            "| Mode | Driven class | Sensory IDs | Readout IDs | Readout | "
            "Snapshot slice | Matches snapshot | Active | Signed sum | Energy |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for sample in result.samples:
        readout = ", ".join(_format_trinary(value) for value in sample.readout)
        snapshot = ", ".join(_format_trinary(value) for value in sample.snapshot_slice)
        rows.append(
            "| "
            f"{sample.mode} | {sample.driven_port_class} | {sample.sensory_port_ids} | "
            f"{sample.readout_port_ids} | [{readout}] | [{snapshot}] | "
            f"{sample.matches_snapshot_slice} | {sample.active_state_count} | "
            f"{sample.signed_sum} | {sample.energy:.6g} |"
        )
    return "\n".join(
        [
            "# cuNxon external-drive window probe",
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
            f"- Steps per sample: {result.steps}",
            f"- Input vector: [{input_vector}]",
            f"- Samples: {result.sample_count}",
            "",
            "## Why this probe exists",
            "",
            "The source-semantics audit and input-proxy target probe suggest that the public "
            "step-input path is an input-class direct drive, not a desired-output/error-channel. "
            "This controlled ctypes probe drives identical values through input, hidden, and "
            "output sensory-id windows in both `StepInfer` and `StepTrain`, then compares the "
            "configured readout to the full-sphere snapshot slice.",
            "",
            "## Port-window samples",
            "",
            *rows,
            "",
            "## Interpretation boundary",
            "",
            "input-class direct drive is useful for observing external stimulation, but "
            "hidden/output sensory-id windows are not a supported desired-output/error-channel "
            "unless they produce target-free motor accuracy above trivial baselines in a later "
            "task-coupled probe.",
            "",
            "## Notes",
            "",
            notes,
            "",
            "## Evidence boundary",
            "",
            "This probe only checks port-window drive and readout semantics. It does not "
            "prove intelligence, task learning, useful recall, action quality, or "
            "generalization.",
            "",
        ]
    )


def write_external_drive_window_artifacts(
    result: CunxonExternalDriveWindowProbeResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown external-drive window artifacts."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_external_drive_window_markdown_report(result), encoding="utf-8"
    )
    return json_output, markdown_output


def run_ctypes_external_drive_window_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    steps: int = 5,
    device_id: int = 0,
) -> CunxonExternalDriveWindowProbeResult:
    """Compare input/hidden/output sensory-id windows with identical external drive."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    lib_path = Path(library_path)
    lib = _load_library(lib_path)
    ctx = C.c_void_p()
    input_vector = [1.0, -1.0, 0.5, -0.5]
    windows = [
        ("input", [0, 1, 2, 3]),
        ("hidden", [4, 5, 6, 7]),
        ("output", [8, 9, 10, 11]),
    ]
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xC0FFEE2026, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        samples = [
            _run_external_drive_window_sample(
                lib=lib,
                ctx=ctx,
                mode=mode,
                driven_port_class=driven_class,
                sensory_port_ids=port_ids,
                readout_port_ids=port_ids,
                input_vector=input_vector,
                steps=steps,
            )
            for mode in ("infer", "train")
            for driven_class, port_ids in windows
        ]
        return CunxonExternalDriveWindowProbeResult(
            status="external-drive window probe viable",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            steps=steps,
            input_vector=input_vector,
            samples=samples,
            notes=[
                (
                    "single 4/4/4 sphere per sample with identical external vector "
                    "and changed sensory ids"
                ),
                (
                    "readouts are compared against full-sphere snapshot slices "
                    "to validate port mapping"
                ),
                "runtime/interface evidence only; not a desired-output or decision-quality claim",
            ],
        )
    finally:
        if ctx.value:
            lib.cunxonDestroyContext(ctx)


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


def run_ctypes_aigarth_readout_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    generations: int = 12,
    population_size: int = 24,
    eval_steps: int = 25,
    device_id: int = 0,
) -> CunxonAigarthReadoutProbeResult:
    """Contrast upstream Aigarth demo-relative readouts with absolute output readouts."""
    if generations <= 0:
        raise ValueError("generations must be positive")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if eval_steps <= 0:
        raise ValueError("eval_steps must be positive")

    lib_path = Path(library_path)
    lib = _load_library(lib_path)
    ctx = C.c_void_p()
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xA164A27E, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        runs = [
            _run_aigarth_readout_mapping(
                lib=lib,
                ctx=ctx,
                mapping="relative-demo-readout",
                readout_ids=list(range(8)),
                neuron_class="input/hidden alias, not output block",
                generations=generations,
                population_size=population_size,
                eval_steps=eval_steps,
                seed_offset=1,
            ),
            _run_aigarth_readout_mapping(
                lib=lib,
                ctx=ctx,
                mapping="absolute-output-readout",
                readout_ids=list(range(36, 44)),
                neuron_class="absolute output block",
                generations=generations,
                population_size=population_size,
                eval_steps=eval_steps,
                seed_offset=1,
            ),
        ]
        return CunxonAigarthReadoutProbeResult(
            status="aigarth readout semantics probe viable",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            generations=generations,
            population_size=population_size,
            eval_steps=eval_steps,
            runs=runs,
            notes=[
                (
                    "contrasts upstream example_aigarth.cu readout ids 0..7 with "
                    "absolute output ids 36..43"
                ),
                "Aigarth is evaluated as interface semantics, not as an intelligence claim",
                (
                    "absolute output readout must improve on its own before using this "
                    "route as a motor adapter"
                ),
            ],
        )
    finally:
        if ctx.value:
            lib.cunxonDestroyContext(ctx)


def run_ctypes_aigarth_action_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    generations: int = 16,
    population_size: int = 32,
    eval_steps: int = 24,
    seed_offset: int = 82,
    evaluation_specs: Sequence[tuple[str, str, tuple[float, float, float], str]] | None = None,
    fitness_variant: str = "success_plus_alignment",
    device_id: int = 0,
) -> CunxonAigarthActionProbeResult:
    """Evolve a cuNxon readout with Aigarth and score train/holdout actions."""
    if generations <= 0:
        raise ValueError("generations must be positive")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if eval_steps <= 0:
        raise ValueError("eval_steps must be positive")

    lib_path = Path(library_path)
    lib = _load_library(lib_path)
    ctx = C.c_void_p()
    net = C.c_void_p()
    decoder = ActionDecoder(num_output_neurons=3)
    callback_errors: list[Exception] = []
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xA164A27E, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        _check(
            lib,
            lib.cunxonNetworkCreate(ctx, C.byref(net), b"neuraxon_hybrid_cunxon_aigarth_action"),
        )
        params = _NetworkParameters()
        _check(lib, lib.cunxonGetDefaultParameters(C.byref(params)))
        params.num_input_neurons = 3
        params.num_hidden_neurons = 32
        params.num_output_neurons = 3
        params.random_seed_offset = seed_offset
        params.synapse_death_prob = 0.0
        params.synapse_formation_prob = 0.0

        sphere_id = C.c_int(-1)
        _check(
            lib,
            lib.cunxonNetworkAddSphere(
                net, b"AIGARTH_ACTION", CUNXON_SPHERE_SENSORY, C.byref(params), C.byref(sphere_id)
            ),
        )
        sensory_ids = (C.c_int * 3)(0, 1, 2)
        readout_start = params.num_input_neurons + params.num_hidden_neurons
        readout_ids = [readout_start, readout_start + 1, readout_start + 2]
        readout_ids_array = (C.c_int * len(readout_ids))(*readout_ids)
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
                readout_ids_array,
                len(readout_ids),
            ),
        )
        _check(lib, lib.cunxonNetworkFinalize(net))

        specs = (
            list(evaluation_specs)
            if evaluation_specs is not None
            else _default_supervised_motor_specs()
        )
        if fitness_variant == "target_contract_augmented_train":
            train_splits = {"train", "augmented_train"}
            train_specs = [spec for spec in specs if spec[1] in train_splits]
        elif fitness_variant == "target_contract_stress_injection":
            train_splits = {"train", "augmented_train", "stress_train"}
            train_specs = [spec for spec in specs if spec[1] in train_splits]
        elif fitness_variant == "target_contract_stress_amplitude_ladder":
            train_specs = [
                spec
                for spec in specs
                if spec[1] in {"train", "augmented_train"}
                or spec[1].startswith("stress_train_scaled_")
            ]
        elif fitness_variant == "target_contract_stress_margin_weighted":
            train_specs = [
                spec
                for spec in specs
                if spec[1] in {"train", "augmented_train"}
                or spec[1].startswith("stress_train_scaled_")
            ]
        elif fitness_variant == "target_contract_supervised_low_margin":
            train_specs = [
                spec
                for spec in specs
                if spec[1] in {"train", "augmented_train", "supervised_low_margin_train"}
            ]
        else:
            train_splits = {"train"}
            train_specs = [spec for spec in specs if spec[1] in train_splits]
        if not train_specs:
            raise ValueError("evaluation_specs must include at least one train case")

        def train_score(candidate_net: C.c_void_p) -> float:
            decoder_strategy = (
                "target_contract"
                if fitness_variant
                in {
                    "target_contract_margin",
                    "target_contract_augmented_train",
                    "target_contract_stress_injection",
                    "target_contract_stress_amplitude_ladder",
                    "target_contract_stress_margin_weighted",
                    "target_contract_supervised_low_margin",
                }
                else "action_decoder"
            )
            cases = _evaluate_aigarth_action_cases(
                lib=lib,
                net=candidate_net,
                sphere_id=sphere_id.value,
                specs=train_specs,
                eval_steps=eval_steps,
                decoder=decoder,
                decoder_strategy=decoder_strategy,
            )
            if not cases:
                return 0.0
            success_score = sum(1.0 for case in cases if case.outcome == "success") / len(cases)
            alignment_score = sum(case.target_alignment for case in cases) / len(cases)
            if fitness_variant == "success_plus_alignment":
                return float(success_score + 0.25 * alignment_score)
            if fitness_variant in {
                "target_contract_margin",
                "target_contract_augmented_train",
                "target_contract_stress_injection",
                "target_contract_stress_amplitude_ladder",
                "target_contract_supervised_low_margin",
            }:
                margin_score = sum(_target_contract_margin(case) for case in cases) / len(cases)
                return float(success_score + 0.25 * alignment_score + 0.25 * margin_score)
            if fitness_variant == "target_contract_stress_margin_weighted":
                weighted_total = 0.0
                weight_sum = 0.0
                for case in cases:
                    weight = 2.0 if case.split.startswith("stress_train_scaled_") else 1.0
                    case_success = 1.0 if case.outcome == "success" else 0.0
                    case_score = (
                        case_success
                        + 0.25 * case.target_alignment
                        + 0.5 * _target_contract_margin(case)
                    )
                    weighted_total += weight * case_score
                    weight_sum += weight
                return float(weighted_total / weight_sum if weight_sum else 0.0)
            if fitness_variant in {"strict_label_margin", "strict_label_heavy_penalty"}:
                strict_actions = {"execute", "query", "retry"}
                unexpected_rate = sum(
                    1.0 for case in cases if case.normalized_action not in strict_actions
                ) / len(cases)
                penalty_weight = 3.0 if fitness_variant == "strict_label_heavy_penalty" else 1.0
                strict_label_score = (
                    success_score + 0.25 * alignment_score - penalty_weight * unexpected_rate
                )
                return float(max(0.0, strict_label_score))
            raise ValueError(f"Unsupported Aigarth action fitness variant: {fitness_variant}")

        def fitness(candidate_net: C.c_void_p, _user_data: C.c_void_p) -> float:
            try:
                return train_score(candidate_net)
            except Exception as exc:  # pragma: no cover - ctypes callback safety net
                callback_errors.append(exc)
                return 0.0

        callback = _CUNXON_FITNESS_FN(fitness)
        generation_train_scores: list[float] = []
        for generation_index in range(generations):
            fraction = generation_index / max(1, generations - 1)
            mutation_fast = 0.18 * (1.0 - 0.6 * fraction)
            mutation_slow = 0.08 * (1.0 - 0.6 * fraction)
            mutation_meta = 0.04 * (1.0 - 0.6 * fraction)
            _check(
                lib,
                lib.cunxonNetworkAigarthConfig(
                    net,
                    population_size,
                    C.c_float(mutation_fast),
                    C.c_float(mutation_slow),
                    C.c_float(mutation_meta),
                ),
            )
            _check(lib, lib.cunxonNetworkAigarthStep(net, callback, None))
            if callback_errors:
                raise callback_errors[0]
            generation_train_scores.append(train_score(net))

        cases = _evaluate_aigarth_action_cases(
            lib=lib,
            net=net,
            sphere_id=sphere_id.value,
            specs=specs,
            eval_steps=eval_steps,
            decoder=decoder,
            decoder_strategy=(
                "target_contract"
                if fitness_variant
                in {
                    "target_contract_margin",
                    "target_contract_augmented_train",
                    "target_contract_stress_injection",
                    "target_contract_stress_amplitude_ladder",
                    "target_contract_stress_margin_weighted",
                    "target_contract_supervised_low_margin",
                }
                else "action_decoder"
            ),
        )
        return CunxonAigarthActionProbeResult(
            status="aigarth action probe viable",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            generations=generations,
            population_size=population_size,
            eval_steps=eval_steps,
            readout_ids=readout_ids,
            generation_train_scores=generation_train_scores,
            cases=cases,
            accuracy_by_split=_aigarth_action_accuracy_by_split(cases),
            target_alignment_by_split=_aigarth_action_target_alignment_by_split(cases),
            baseline_accuracy_by_split=_aigarth_action_baseline_accuracy_by_split(cases),
            unique_readouts=len({tuple(case.readout) for case in cases}),
            action_distribution=_aigarth_action_distribution(cases),
            seed_offset=seed_offset,
            notes=[
                "Aigarth fitness callback uses train cases only; holdout labels are not optimized",
                "final train and holdout readouts are decoded through the existing ActionDecoder",
                "holdout accuracy must beat trivial baselines before any adapter claim",
            ],
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)
        if ctx.value:
            lib.cunxonDestroyContext(ctx)


def run_ctypes_aigarth_action_seed_sweep_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    seed_offsets: Sequence[int] = (82, 83, 84, 85, 86),
    generations: int = 16,
    population_size: int = 32,
    eval_steps: int = 24,
    evaluation_specs: Sequence[tuple[str, str, tuple[float, float, float], str]] | None = None,
    device_id: int = 0,
) -> CunxonAigarthActionSeedSweepResult:
    """Run the Aigarth action probe across fresh cuNxon seeds."""
    if not seed_offsets:
        raise ValueError("seed_offsets must contain at least one value")

    probe_results = [
        run_ctypes_aigarth_action_probe(
            library_path=library_path,
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            generations=generations,
            population_size=population_size,
            eval_steps=eval_steps,
            seed_offset=seed_offset,
            evaluation_specs=evaluation_specs,
            device_id=device_id,
        )
        for seed_offset in seed_offsets
    ]
    runs = [
        CunxonAigarthActionSeedRun(
            seed_offset=result.seed_offset,
            generation_train_scores=result.generation_train_scores,
            accuracy_by_split=result.accuracy_by_split,
            target_alignment_by_split=result.target_alignment_by_split,
            baseline_accuracy_by_split=result.baseline_accuracy_by_split,
            unique_readouts=result.unique_readouts,
            action_distribution=result.action_distribution,
            cases=result.cases,
        )
        for result in probe_results
    ]
    return CunxonAigarthActionSeedSweepResult(
        status="aigarth action seed sweep completed",
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        library_path=str(Path(library_path)),
        device_name=probe_results[0].device_name,
        compute_capability=probe_results[0].compute_capability,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        readout_ids=probe_results[0].readout_ids,
        seed_offsets=list(seed_offsets),
        runs=runs,
        accuracy_summary_by_split=_seed_sweep_accuracy_summary(runs),
        aggregate_action_distribution=_seed_sweep_action_distribution(runs),
        seeds_beating_baseline_by_split=_seed_sweep_beating_baseline_counts(runs),
        notes=[
            "fresh cuNxon network/context per seed",
            "fitness callback still uses train cases only; holdout labels are never optimized",
            "repeatability audit, not intelligence evidence",
        ],
    )


def run_ctypes_aigarth_action_hard_holdout_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    seed_offsets: Sequence[int] = (87, 88, 89, 90, 91),
    generations: int = 16,
    population_size: int = 32,
    eval_steps: int = 24,
    device_id: int = 0,
) -> CunxonAigarthActionHardHoldoutResult:
    """Stress the Aigarth action route with hard holdouts and leakage controls."""
    if not seed_offsets:
        raise ValueError("seed_offsets must contain at least one value")
    specs = _aigarth_action_hard_holdout_specs()
    probe_results = [
        run_ctypes_aigarth_action_probe(
            library_path=library_path,
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            generations=generations,
            population_size=population_size,
            eval_steps=eval_steps,
            seed_offset=seed_offset,
            evaluation_specs=specs,
            device_id=device_id,
        )
        for seed_offset in seed_offsets
    ]
    runs = [
        CunxonAigarthActionSeedRun(
            seed_offset=result.seed_offset,
            generation_train_scores=result.generation_train_scores,
            accuracy_by_split=result.accuracy_by_split,
            target_alignment_by_split=result.target_alignment_by_split,
            baseline_accuracy_by_split=result.baseline_accuracy_by_split,
            unique_readouts=result.unique_readouts,
            action_distribution=result.action_distribution,
            cases=result.cases,
        )
        for result in probe_results
    ]
    aggregate_distribution = _seed_sweep_action_distribution(runs)
    strict_expected_actions = ["execute", "query", "retry"]
    unexpected_action_count = sum(
        count
        for action, count in aggregate_distribution.items()
        if action not in strict_expected_actions
    )
    total_actions = sum(aggregate_distribution.values())
    accuracy_summary = _seed_sweep_accuracy_summary(runs)
    train_values = [run.accuracy_by_split.get("train", 0.0) for run in runs]
    hard_values = [run.accuracy_by_split.get("hard_holdout", 0.0) for run in runs]
    gaps = [train - hard for train, hard in zip(train_values, hard_values, strict=False)]
    return CunxonAigarthActionHardHoldoutResult(
        status="aigarth hard-holdout audit completed",
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        library_path=str(Path(library_path)),
        device_name=probe_results[0].device_name,
        compute_capability=probe_results[0].compute_capability,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        readout_ids=probe_results[0].readout_ids,
        seed_offsets=list(seed_offsets),
        strict_expected_actions=strict_expected_actions,
        runs=runs,
        accuracy_summary_by_split=accuracy_summary,
        aggregate_action_distribution=aggregate_distribution,
        seeds_beating_baseline_by_split=_seed_sweep_beating_baseline_counts(runs),
        unexpected_action_count=unexpected_action_count,
        unexpected_action_rate=unexpected_action_count / total_actions if total_actions else 0.0,
        leakage_control_accuracy_mean=accuracy_summary.get("permuted_control", {}).get("mean", 0.0),
        train_to_hard_holdout_gap_mean=sum(gaps) / len(gaps) if gaps else 0.0,
        notes=[
            "fresh cuNxon network/context per seed",
            "fitness callback still uses train cases only; hard holdout and "
            "permuted-control labels are never optimized",
            "strict expected actions are execute/query/retry; assertive/explore/cautious "
            "remain out-of-contract caveats",
            "hard-holdout audit, not intelligence evidence",
        ],
    )


def run_ctypes_aigarth_action_strict_label_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    seed_offsets: Sequence[int] = (92, 93, 94, 95, 96),
    generations: int = 16,
    population_size: int = 32,
    eval_steps: int = 24,
    fitness_variant: str = "strict_label_margin",
    device_id: int = 0,
) -> CunxonAigarthActionHardHoldoutResult:
    """Run a strict-label Aigarth action fitness audit across fresh cuNxon seeds."""
    if not seed_offsets:
        raise ValueError("seed_offsets must contain at least one value")
    if fitness_variant not in {
        "strict_label_margin",
        "strict_label_heavy_penalty",
        "target_contract_margin",
    }:
        raise ValueError(
            "strict-label audit only supports fitness_variant='strict_label_margin', "
            "'strict_label_heavy_penalty', or 'target_contract_margin'"
        )
    specs = _aigarth_action_hard_holdout_specs()
    probe_results = [
        run_ctypes_aigarth_action_probe(
            library_path=library_path,
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            generations=generations,
            population_size=population_size,
            eval_steps=eval_steps,
            seed_offset=seed_offset,
            evaluation_specs=specs,
            fitness_variant=fitness_variant,
            device_id=device_id,
        )
        for seed_offset in seed_offsets
    ]
    runs = [
        CunxonAigarthActionSeedRun(
            seed_offset=result.seed_offset,
            generation_train_scores=result.generation_train_scores,
            accuracy_by_split=result.accuracy_by_split,
            target_alignment_by_split=result.target_alignment_by_split,
            baseline_accuracy_by_split=result.baseline_accuracy_by_split,
            unique_readouts=result.unique_readouts,
            action_distribution=result.action_distribution,
            cases=result.cases,
        )
        for result in probe_results
    ]
    aggregate_distribution = _seed_sweep_action_distribution(runs)
    strict_expected_actions = ["execute", "query", "retry"]
    unexpected_action_count = sum(
        count
        for action, count in aggregate_distribution.items()
        if action not in strict_expected_actions
    )
    total_actions = sum(aggregate_distribution.values())
    accuracy_summary = _seed_sweep_accuracy_summary(runs)
    train_values = [run.accuracy_by_split.get("train", 0.0) for run in runs]
    hard_values = [run.accuracy_by_split.get("hard_holdout", 0.0) for run in runs]
    gaps = [train - hard for train, hard in zip(train_values, hard_values, strict=False)]
    heavy_penalty = fitness_variant == "strict_label_heavy_penalty"
    target_contract = fitness_variant == "target_contract_margin"
    return CunxonAigarthActionHardHoldoutResult(
        status=(
            "aigarth contract-penalty action audit completed"
            if heavy_penalty
            else (
                "aigarth target-contract action audit completed"
                if target_contract
                else "aigarth strict-label action audit completed"
            )
        ),
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        library_path=str(Path(library_path)),
        device_name=probe_results[0].device_name,
        compute_capability=probe_results[0].compute_capability,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        readout_ids=probe_results[0].readout_ids,
        seed_offsets=list(seed_offsets),
        strict_expected_actions=strict_expected_actions,
        runs=runs,
        accuracy_summary_by_split=accuracy_summary,
        aggregate_action_distribution=aggregate_distribution,
        seeds_beating_baseline_by_split=_seed_sweep_beating_baseline_counts(runs),
        unexpected_action_count=unexpected_action_count,
        unexpected_action_rate=unexpected_action_count / total_actions if total_actions else 0.0,
        leakage_control_accuracy_mean=accuracy_summary.get("permuted_control", {}).get("mean", 0.0),
        train_to_hard_holdout_gap_mean=sum(gaps) / len(gaps) if gaps else 0.0,
        fitness_variant=fitness_variant,
        notes=[
            "fresh cuNxon network/context per seed",
            (
                "heavy contract penalty subtracts three times the unexpected-label rate"
                if heavy_penalty
                else (
                    "target-contract fitness decodes with the signed-first-lane project contract"
                    if target_contract
                    else "strict-label fitness penalizes out-of-contract normalized labels"
                )
            ),
            "fitness callback uses train cases only; holdout, hard-holdout and "
            "permuted-control labels are never optimized",
            (
                "contract-penalty audit, not intelligence evidence"
                if heavy_penalty
                else (
                    "target-contract audit, not intelligence evidence"
                    if target_contract
                    else "strict-label audit, not intelligence evidence"
                )
            ),
        ],
    )


def run_ctypes_aigarth_action_contract_penalty_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    seed_offsets: Sequence[int] = (97, 98, 99, 100, 101),
    generations: int = 16,
    population_size: int = 32,
    eval_steps: int = 24,
    fitness_variant: str = "strict_label_heavy_penalty",
    device_id: int = 0,
) -> CunxonAigarthActionHardHoldoutResult:
    """Run a heavier unexpected-label penalty audit across fresh cuNxon seeds."""
    if fitness_variant != "strict_label_heavy_penalty":
        raise ValueError(
            "contract-penalty audit only supports fitness_variant='strict_label_heavy_penalty'"
        )
    return run_ctypes_aigarth_action_strict_label_probe(
        library_path=library_path,
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        seed_offsets=seed_offsets,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        fitness_variant=fitness_variant,
        device_id=device_id,
    )


def run_ctypes_aigarth_action_target_contract_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    seed_offsets: Sequence[int] = (102, 103, 104, 105, 106),
    generations: int = 16,
    population_size: int = 32,
    eval_steps: int = 24,
    fitness_variant: str = "target_contract_margin",
    device_id: int = 0,
) -> CunxonAigarthActionHardHoldoutResult:
    """Run a target-contract/margin Aigarth action audit across fresh cuNxon seeds."""
    if fitness_variant != "target_contract_margin":
        raise ValueError(
            "target-contract audit only supports fitness_variant='target_contract_margin'"
        )
    return run_ctypes_aigarth_action_strict_label_probe(
        library_path=library_path,
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        seed_offsets=seed_offsets,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        fitness_variant=fitness_variant,
        device_id=device_id,
    )


def run_ctypes_aigarth_action_target_contract_stress_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    seed_offsets: Sequence[int] = (107, 108, 109, 110, 111),
    generations: int = 16,
    population_size: int = 32,
    eval_steps: int = 24,
    fitness_variant: str = "target_contract_margin",
    device_id: int = 0,
) -> CunxonAigarthActionHardHoldoutResult:
    """Run a target-contract audit with harder/noisier and counterfactual splits."""
    if fitness_variant != "target_contract_margin":
        raise ValueError(
            "target-contract stress audit only supports fitness_variant='target_contract_margin'"
        )
    if not seed_offsets:
        raise ValueError("seed_offsets must contain at least one value")
    specs = _aigarth_action_target_contract_stress_specs()
    probe_results = [
        run_ctypes_aigarth_action_probe(
            library_path=library_path,
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            generations=generations,
            population_size=population_size,
            eval_steps=eval_steps,
            seed_offset=seed_offset,
            evaluation_specs=specs,
            fitness_variant=fitness_variant,
            device_id=device_id,
        )
        for seed_offset in seed_offsets
    ]
    runs = [
        CunxonAigarthActionSeedRun(
            seed_offset=result.seed_offset,
            generation_train_scores=result.generation_train_scores,
            accuracy_by_split=result.accuracy_by_split,
            target_alignment_by_split=result.target_alignment_by_split,
            baseline_accuracy_by_split=result.baseline_accuracy_by_split,
            unique_readouts=result.unique_readouts,
            action_distribution=result.action_distribution,
            cases=result.cases,
        )
        for result in probe_results
    ]
    aggregate_distribution = _seed_sweep_action_distribution(runs)
    strict_expected_actions = ["execute", "query", "retry"]
    unexpected_action_count = sum(
        count
        for action, count in aggregate_distribution.items()
        if action not in strict_expected_actions
    )
    total_actions = sum(aggregate_distribution.values())
    accuracy_summary = _seed_sweep_accuracy_summary(runs)
    train_values = [run.accuracy_by_split.get("train", 0.0) for run in runs]
    hard_values = [run.accuracy_by_split.get("hard_holdout", 0.0) for run in runs]
    gaps = [train - hard for train, hard in zip(train_values, hard_values, strict=False)]
    return CunxonAigarthActionHardHoldoutResult(
        status="aigarth target-contract stress audit completed",
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        library_path=str(Path(library_path)),
        device_name=probe_results[0].device_name,
        compute_capability=probe_results[0].compute_capability,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        readout_ids=probe_results[0].readout_ids,
        seed_offsets=list(seed_offsets),
        strict_expected_actions=strict_expected_actions,
        runs=runs,
        accuracy_summary_by_split=accuracy_summary,
        aggregate_action_distribution=aggregate_distribution,
        seeds_beating_baseline_by_split=_seed_sweep_beating_baseline_counts(runs),
        unexpected_action_count=unexpected_action_count,
        unexpected_action_rate=unexpected_action_count / total_actions if total_actions else 0.0,
        leakage_control_accuracy_mean=accuracy_summary.get("permuted_control", {}).get("mean", 0.0),
        train_to_hard_holdout_gap_mean=sum(gaps) / len(gaps) if gaps else 0.0,
        fitness_variant=fitness_variant,
        notes=[
            "fresh cuNxon network/context per seed",
            "target-contract fitness decodes with the signed-first-lane project contract",
            "adds stress_holdout low-margin cases and counterfactual_control rotated-label cases",
            (
                "fitness callback uses train cases only; all holdout and control labels "
                "are never optimized"
            ),
            "target-contract stress audit, not intelligence evidence",
        ],
    )


def _summarize_aigarth_action_runs(
    *,
    probe_results: Sequence[CunxonAigarthActionProbeResult],
    seed_offsets: Sequence[int],
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    generations: int,
    population_size: int,
    eval_steps: int,
    fitness_variant: str,
    status: str,
    notes: Sequence[str],
) -> CunxonAigarthActionHardHoldoutResult:
    """Summarize multi-seed Aigarth action probes with strict-contract diagnostics."""
    runs = [
        CunxonAigarthActionSeedRun(
            seed_offset=result.seed_offset,
            generation_train_scores=result.generation_train_scores,
            accuracy_by_split=result.accuracy_by_split,
            target_alignment_by_split=result.target_alignment_by_split,
            baseline_accuracy_by_split=result.baseline_accuracy_by_split,
            unique_readouts=result.unique_readouts,
            action_distribution=result.action_distribution,
            cases=result.cases,
        )
        for result in probe_results
    ]
    aggregate_distribution = _seed_sweep_action_distribution(runs)
    strict_expected_actions = ["execute", "query", "retry"]
    unexpected_action_count = sum(
        count
        for action, count in aggregate_distribution.items()
        if action not in strict_expected_actions
    )
    total_actions = sum(aggregate_distribution.values())
    accuracy_summary = _seed_sweep_accuracy_summary(runs)
    train_values = [
        max(
            run.accuracy_by_split.get("train", 0.0),
            run.accuracy_by_split.get("augmented_train", 0.0),
        )
        for run in runs
    ]
    hard_values = [run.accuracy_by_split.get("hard_holdout", 0.0) for run in runs]
    gaps = [train - hard for train, hard in zip(train_values, hard_values, strict=False)]
    return CunxonAigarthActionHardHoldoutResult(
        status=status,
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        library_path=str(Path(library_path)),
        device_name=probe_results[0].device_name,
        compute_capability=probe_results[0].compute_capability,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        readout_ids=probe_results[0].readout_ids,
        seed_offsets=list(seed_offsets),
        strict_expected_actions=strict_expected_actions,
        runs=runs,
        accuracy_summary_by_split=accuracy_summary,
        aggregate_action_distribution=aggregate_distribution,
        seeds_beating_baseline_by_split=_seed_sweep_beating_baseline_counts(runs),
        unexpected_action_count=unexpected_action_count,
        unexpected_action_rate=unexpected_action_count / total_actions if total_actions else 0.0,
        leakage_control_accuracy_mean=accuracy_summary.get("permuted_control", {}).get("mean", 0.0),
        train_to_hard_holdout_gap_mean=sum(gaps) / len(gaps) if gaps else 0.0,
        fitness_variant=fitness_variant,
        notes=list(notes),
    )


def run_ctypes_aigarth_action_target_contract_augmented_train_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    seed_offsets: Sequence[int] = (112, 113, 114, 115, 116),
    generations: int = 16,
    population_size: int = 32,
    eval_steps: int = 24,
    fitness_variant: str = "target_contract_augmented_train",
    device_id: int = 0,
) -> CunxonAigarthActionHardHoldoutResult:
    """Run target-contract Aigarth with low-margin augmented training cases."""
    if fitness_variant != "target_contract_augmented_train":
        raise ValueError(
            "target-contract augmented-train audit only supports "
            "fitness_variant='target_contract_augmented_train'"
        )
    if not seed_offsets:
        raise ValueError("seed_offsets must contain at least one value")
    specs = _aigarth_action_target_contract_augmented_train_specs()
    probe_results = [
        run_ctypes_aigarth_action_probe(
            library_path=library_path,
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            generations=generations,
            population_size=population_size,
            eval_steps=eval_steps,
            seed_offset=seed_offset,
            evaluation_specs=specs,
            fitness_variant=fitness_variant,
            device_id=device_id,
        )
        for seed_offset in seed_offsets
    ]
    return _summarize_aigarth_action_runs(
        probe_results=probe_results,
        seed_offsets=seed_offsets,
        library_path=library_path,
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        fitness_variant=fitness_variant,
        status="aigarth target-contract augmented-train audit completed",
        notes=[
            "fresh cuNxon network/context per seed",
            "target-contract fitness decodes with the signed-first-lane project contract",
            "fitness callback uses train plus augmented_train low-margin cases only",
            "stress_holdout, hard holdout, and control labels are never optimized",
            "target-contract augmented-train stress audit, not intelligence evidence",
        ],
    )


def run_ctypes_aigarth_action_target_contract_stress_injection_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    seed_offsets: Sequence[int] = (137, 138, 139, 140, 141),
    generations: int = 16,
    population_size: int = 32,
    eval_steps: int = 24,
    fitness_variant: str = "target_contract_stress_injection",
    device_id: int = 0,
) -> CunxonAigarthActionHardHoldoutResult:
    """Run a stress-label-injection upper-bound audit for low-margin stress cases."""
    if fitness_variant != "target_contract_stress_injection":
        raise ValueError(
            "target-contract stress-injection audit only supports "
            "fitness_variant='target_contract_stress_injection'"
        )
    if not seed_offsets:
        raise ValueError("seed_offsets must contain at least one value")
    specs = _aigarth_action_target_contract_stress_injection_specs()
    probe_results = [
        run_ctypes_aigarth_action_probe(
            library_path=library_path,
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            generations=generations,
            population_size=population_size,
            eval_steps=eval_steps,
            seed_offset=seed_offset,
            evaluation_specs=specs,
            fitness_variant=fitness_variant,
            device_id=device_id,
        )
        for seed_offset in seed_offsets
    ]
    return _summarize_aigarth_action_runs(
        probe_results=probe_results,
        seed_offsets=seed_offsets,
        library_path=library_path,
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        fitness_variant=fitness_variant,
        status="aigarth target-contract stress-injection audit completed",
        notes=[
            "fresh cuNxon network/context per seed",
            "target-contract fitness decodes with the signed-first-lane project contract",
            "fitness callback deliberately includes duplicated stress_train low-margin cases",
            "stress_holdout is the original stress split, but stress-like labels are optimized",
            "upper-bound/debugging diagnostic only; not generalization or intelligence evidence",
        ],
    )


def _query_collapse_rate(cases: Sequence[CunxonAigarthActionCase]) -> float:
    return (
        sum(1 for case in cases if case.normalized_action == "query") / len(cases) if cases else 0.0
    )


def _execute_retry_accuracy(cases: Sequence[CunxonAigarthActionCase]) -> float:
    execute_retry_cases = [case for case in cases if case.expected_action in {"execute", "retry"}]
    if not execute_retry_cases:
        return 0.0
    return sum(1 for case in execute_retry_cases if case.outcome == "success") / len(
        execute_retry_cases
    )


def _best_constant_baseline_for_split(run: CunxonAigarthActionSeedRun, split: str) -> float:
    scores = [
        by_split[split] for by_split in run.baseline_accuracy_by_split.values() if split in by_split
    ]
    return max(scores) if scores else 0.0


def _stress_amplitude_split_summaries(
    runs: Sequence[CunxonAigarthActionSeedRun],
    *,
    amplitude_factors: Sequence[float],
) -> list[CunxonAigarthStressAmplitudeSplitSummary]:
    summaries: list[CunxonAigarthStressAmplitudeSplitSummary] = []
    for factor in amplitude_factors:
        suffix = _amplitude_split_suffix(factor)
        for split in (f"stress_train_scaled_{suffix}", f"stress_holdout_scaled_{suffix}"):
            split_cases = [case for run in runs for case in run.cases if case.split == split]
            accuracies = [
                run.accuracy_by_split[split] for run in runs if split in run.accuracy_by_split
            ]
            baselines = [_best_constant_baseline_for_split(run, split) for run in runs]
            action_distribution = _aigarth_action_distribution(split_cases)
            summaries.append(
                CunxonAigarthStressAmplitudeSplitSummary(
                    split=split,
                    amplitude_factor=float(factor),
                    sample_count=len(split_cases),
                    accuracy_mean=sum(accuracies) / len(accuracies) if accuracies else 0.0,
                    best_constant_baseline_mean=sum(baselines) / len(baselines)
                    if baselines
                    else 0.0,
                    seeds_beating_best_baseline=sum(
                        1
                        for run in runs
                        if split in run.accuracy_by_split
                        and run.accuracy_by_split[split]
                        > _best_constant_baseline_for_split(run, split)
                    ),
                    query_collapse_rate=_query_collapse_rate(split_cases),
                    execute_retry_accuracy=_execute_retry_accuracy(split_cases),
                    action_distribution=action_distribution,
                )
            )
    return summaries


def run_ctypes_aigarth_action_target_contract_stress_amplitude_ladder_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    seed_offsets: Sequence[int] = (142, 143, 144, 145, 146),
    amplitude_factors: Sequence[float] = (1.0, 1.5, 2.0, 3.0),
    generations: int = 16,
    population_size: int = 32,
    eval_steps: int = 24,
    fitness_variant: str = "target_contract_stress_amplitude_ladder",
    device_id: int = 0,
) -> CunxonAigarthStressAmplitudeLadderResult:
    """Run a bounded amplitude ladder over low-margin stress vectors."""
    if fitness_variant != "target_contract_stress_amplitude_ladder":
        raise ValueError(
            "stress amplitude-ladder audit only supports "
            "fitness_variant='target_contract_stress_amplitude_ladder'"
        )
    if not seed_offsets:
        raise ValueError("seed_offsets must contain at least one value")
    if not amplitude_factors:
        raise ValueError("amplitude_factors must contain at least one value")
    specs = _aigarth_action_target_contract_stress_amplitude_ladder_specs(amplitude_factors)
    probe_results = [
        run_ctypes_aigarth_action_probe(
            library_path=library_path,
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            generations=generations,
            population_size=population_size,
            eval_steps=eval_steps,
            seed_offset=seed_offset,
            evaluation_specs=specs,
            fitness_variant=fitness_variant,
            device_id=device_id,
        )
        for seed_offset in seed_offsets
    ]
    source_summary = _summarize_aigarth_action_runs(
        probe_results=probe_results,
        seed_offsets=seed_offsets,
        library_path=library_path,
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        fitness_variant=fitness_variant,
        status="aigarth target-contract stress amplitude-ladder completed",
        notes=[],
    )
    split_summaries = _stress_amplitude_split_summaries(
        source_summary.runs,
        amplitude_factors=amplitude_factors,
    )
    original_stress_cases = [
        case for run in source_summary.runs for case in run.cases if case.split == "stress_holdout"
    ]
    scaled_holdout_summaries = [
        summary for summary in split_summaries if summary.split.startswith("stress_holdout_scaled_")
    ]
    best_scaled = max(
        scaled_holdout_summaries,
        key=lambda summary: summary.accuracy_mean,
        default=None,
    )
    return CunxonAigarthStressAmplitudeLadderResult(
        status="aigarth target-contract stress amplitude-ladder completed",
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        library_path=str(Path(library_path)),
        device_name=probe_results[0].device_name,
        compute_capability=probe_results[0].compute_capability,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        seed_offsets=list(seed_offsets),
        amplitude_factors=[float(factor) for factor in amplitude_factors],
        split_summaries=split_summaries,
        original_stress_holdout_accuracy_mean=source_summary.accuracy_summary_by_split.get(
            "stress_holdout", {}
        ).get("mean", 0.0),
        original_stress_holdout_query_collapse_rate=_query_collapse_rate(original_stress_cases),
        best_scaled_stress_holdout_accuracy_mean=best_scaled.accuracy_mean if best_scaled else 0.0,
        best_scaled_stress_holdout_amplitude_factor=best_scaled.amplitude_factor
        if best_scaled
        else None,
        aggregate_action_distribution=source_summary.aggregate_action_distribution,
        evidence_boundary=(
            "This is a label-injected separability upper-bound over scaled low-margin stress "
            "vectors. Scaled stress_train cases are inside the fitness callback, so any "
            "improvement is diagnostic adapter-capability evidence, not intelligence "
            "evidence and not generalization evidence unless stress/control holdouts beat "
            "constant baselines."
        ),
        recommended_next_probe={
            "id": "target_aligned_stress_objective_followup",
            "github_issue": "https://github.com/sisutuulenisa/neuraxon-hybrid/issues/88",
        },
        notes=[
            "fresh cuNxon network/context per seed",
            "target-contract fitness includes train, augmented_train, and scaled "
            "stress_train cases",
            "original stress_holdout and scaled stress_holdout splits are reported separately",
            "bounded amplitude ladder; not a long-sweep or intelligence claim",
        ],
    )


def _split_summary_by_name(
    summaries: Iterable[CunxonAigarthStressAmplitudeSplitSummary], split: str
) -> CunxonAigarthStressAmplitudeSplitSummary | None:
    for summary in summaries:
        if summary.split == split:
            return summary
    return None


def _aigarth_source_split_summary(
    source_summary: CunxonAigarthActionHardHoldoutResult,
    split: str,
    *,
    amplitude_factor: float = 1.0,
) -> CunxonAigarthStressAmplitudeSplitSummary:
    cases = [case for run in source_summary.runs for case in run.cases if case.split == split]
    return CunxonAigarthStressAmplitudeSplitSummary(
        split=split,
        amplitude_factor=amplitude_factor,
        sample_count=len(cases),
        accuracy_mean=source_summary.accuracy_summary_by_split.get(split, {}).get("mean", 0.0),
        best_constant_baseline_mean=sum(
            _best_constant_baseline_for_split(run, split) for run in source_summary.runs
        )
        / len(source_summary.runs),
        seeds_beating_best_baseline=source_summary.seeds_beating_baseline_by_split.get(
            split, 0
        ),
        query_collapse_rate=_query_collapse_rate(cases),
        execute_retry_accuracy=_execute_retry_accuracy(cases),
        action_distribution=_aigarth_action_distribution(cases),
    )



def run_ctypes_aigarth_action_target_contract_stress_objective_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    seed_offsets: Sequence[int] = (147, 148, 149),
    amplitude_factor: float = 3.0,
    generations: int = 16,
    population_size: int = 32,
    eval_steps: int = 24,
    fitness_variant: str = "target_contract_stress_margin_weighted",
    device_id: int = 0,
) -> CunxonAigarthStressObjectiveResult:
    """Run one target-aligned objective-shaping follow-up after the amplitude ladder."""
    if fitness_variant != "target_contract_stress_margin_weighted":
        raise ValueError(
            "stress objective audit only supports "
            "fitness_variant='target_contract_stress_margin_weighted'"
        )
    if not seed_offsets:
        raise ValueError("seed_offsets must contain at least one value")
    if amplitude_factor <= 0.0:
        raise ValueError("amplitude_factor must be positive")
    specs = _aigarth_action_target_contract_stress_objective_specs(amplitude_factor)
    probe_results = [
        run_ctypes_aigarth_action_probe(
            library_path=library_path,
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            generations=generations,
            population_size=population_size,
            eval_steps=eval_steps,
            seed_offset=seed_offset,
            evaluation_specs=specs,
            fitness_variant=fitness_variant,
            device_id=device_id,
        )
        for seed_offset in seed_offsets
    ]
    source_summary = _summarize_aigarth_action_runs(
        probe_results=probe_results,
        seed_offsets=seed_offsets,
        library_path=library_path,
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        fitness_variant=fitness_variant,
        status="aigarth target-contract stress objective completed",
        notes=[],
    )
    amplitude_summaries = _stress_amplitude_split_summaries(
        source_summary.runs,
        amplitude_factors=[amplitude_factor],
    )
    original_stress_cases = [
        case for run in source_summary.runs for case in run.cases if case.split == "stress_holdout"
    ]
    suffix = _amplitude_split_suffix(amplitude_factor)
    scaled_split = f"stress_holdout_scaled_{suffix}"
    scaled_stress_cases = [
        case for run in source_summary.runs for case in run.cases if case.split == scaled_split
    ]
    original_summary = CunxonAigarthStressAmplitudeSplitSummary(
        split="stress_holdout",
        amplitude_factor=1.0,
        sample_count=len(original_stress_cases),
        accuracy_mean=source_summary.accuracy_summary_by_split.get(
            "stress_holdout", {}
        ).get("mean", 0.0),
        best_constant_baseline_mean=sum(
            _best_constant_baseline_for_split(run, "stress_holdout") for run in source_summary.runs
        )
        / len(source_summary.runs),
        seeds_beating_best_baseline=source_summary.seeds_beating_baseline_by_split.get(
            "stress_holdout", 0
        ),
        query_collapse_rate=_query_collapse_rate(original_stress_cases),
        execute_retry_accuracy=_execute_retry_accuracy(original_stress_cases),
        action_distribution=_aigarth_action_distribution(original_stress_cases),
    )
    scaled_summary = _split_summary_by_name(amplitude_summaries, scaled_split)
    split_summaries = [original_summary, *amplitude_summaries]
    return CunxonAigarthStressObjectiveResult(
        status="aigarth target-contract stress objective completed",
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        library_path=str(Path(library_path)),
        device_name=probe_results[0].device_name,
        compute_capability=probe_results[0].compute_capability,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        seed_offsets=list(seed_offsets),
        amplitude_factor=float(amplitude_factor),
        fitness_variant=fitness_variant,
        split_summaries=split_summaries,
        original_stress_holdout_accuracy_mean=original_summary.accuracy_mean,
        original_stress_holdout_query_collapse_rate=original_summary.query_collapse_rate,
        original_stress_holdout_execute_retry_accuracy=original_summary.execute_retry_accuracy,
        scaled_stress_holdout_accuracy_mean=scaled_summary.accuracy_mean if scaled_summary else 0.0,
        scaled_stress_holdout_query_collapse_rate=scaled_summary.query_collapse_rate
        if scaled_summary
        else _query_collapse_rate(scaled_stress_cases),
        scaled_stress_holdout_execute_retry_accuracy=scaled_summary.execute_retry_accuracy
        if scaled_summary
        else _execute_retry_accuracy(scaled_stress_cases),
        counterfactual_control_accuracy_mean=source_summary.accuracy_summary_by_split.get(
            "counterfactual_control", {}
        ).get("mean", 0.0),
        permuted_control_accuracy_mean=source_summary.accuracy_summary_by_split.get(
            "permuted_control", {}
        ).get("mean", 0.0),
        aggregate_action_distribution=source_summary.aggregate_action_distribution,
        evidence_boundary=(
            "This is one label-injected target-aligned objective-shaping diagnostic. Scaled "
            "stress_train cases are optimized with extra margin weight, while original "
            "stress_holdout, controls, and scaled holdouts are reported separately. It is "
            "not intelligence evidence and not generalization evidence unless original "
            "stress/control splits beat constant baselines."
        ),
        recommended_next_probe={
            "id": "stress_objective_decoder_geometry_followup",
            "condition": "only if original stress/control quality remains baseline-level",
        },
        notes=[
            "fresh cuNxon network/context per seed",
            "fitness callback weights scaled stress_train margin more strongly than the ladder",
            "original stress_holdout and controls are never optimized",
            "objective-shaping diagnostic only; not a long-sweep or intelligence claim",
        ],
    )


def run_ctypes_aigarth_action_target_contract_supervised_low_margin_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    seed_offsets: Sequence[int] = (150, 151, 152),
    generations: int = 16,
    population_size: int = 32,
    eval_steps: int = 24,
    fitness_variant: str = "target_contract_supervised_low_margin",
    device_id: int = 0,
) -> CunxonAigarthStressObjectiveResult:
    """Run a supervised low-margin objective diagnostic after stress query-collapse."""
    if fitness_variant != "target_contract_supervised_low_margin":
        raise ValueError(
            "supervised low-margin audit only supports "
            "fitness_variant='target_contract_supervised_low_margin'"
        )
    if not seed_offsets:
        raise ValueError("seed_offsets must contain at least one value")
    specs = _aigarth_action_target_contract_supervised_low_margin_specs()
    probe_results = [
        run_ctypes_aigarth_action_probe(
            library_path=library_path,
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            generations=generations,
            population_size=population_size,
            eval_steps=eval_steps,
            seed_offset=seed_offset,
            evaluation_specs=specs,
            fitness_variant=fitness_variant,
            device_id=device_id,
        )
        for seed_offset in seed_offsets
    ]
    source_summary = _summarize_aigarth_action_runs(
        probe_results=probe_results,
        seed_offsets=seed_offsets,
        library_path=library_path,
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        fitness_variant=fitness_variant,
        status="aigarth target-contract supervised low-margin objective completed",
        notes=[],
    )
    split_summaries = [
        _aigarth_source_split_summary(source_summary, split)
        for split in (
            "train",
            "augmented_train",
            "supervised_low_margin_train",
            "holdout",
            "hard_holdout",
            "stress_holdout",
            "counterfactual_control",
            "permuted_control",
        )
    ]
    stress_summary = _split_summary_by_name(split_summaries, "stress_holdout")
    supervised_train_summary = _split_summary_by_name(
        split_summaries, "supervised_low_margin_train"
    )
    return CunxonAigarthStressObjectiveResult(
        status="aigarth target-contract supervised low-margin objective completed",
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        library_path=str(Path(library_path)),
        device_name=probe_results[0].device_name,
        compute_capability=probe_results[0].compute_capability,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        seed_offsets=list(seed_offsets),
        amplitude_factor=1.0,
        fitness_variant=fitness_variant,
        split_summaries=split_summaries,
        original_stress_holdout_accuracy_mean=stress_summary.accuracy_mean
        if stress_summary
        else 0.0,
        original_stress_holdout_query_collapse_rate=stress_summary.query_collapse_rate
        if stress_summary
        else 0.0,
        original_stress_holdout_execute_retry_accuracy=stress_summary.execute_retry_accuracy
        if stress_summary
        else 0.0,
        scaled_stress_holdout_accuracy_mean=supervised_train_summary.accuracy_mean
        if supervised_train_summary
        else 0.0,
        scaled_stress_holdout_query_collapse_rate=supervised_train_summary.query_collapse_rate
        if supervised_train_summary
        else 0.0,
        scaled_stress_holdout_execute_retry_accuracy=supervised_train_summary.execute_retry_accuracy
        if supervised_train_summary
        else 0.0,
        counterfactual_control_accuracy_mean=source_summary.accuracy_summary_by_split.get(
            "counterfactual_control", {}
        ).get("mean", 0.0),
        permuted_control_accuracy_mean=source_summary.accuracy_summary_by_split.get(
            "permuted_control", {}
        ).get("mean", 0.0),
        aggregate_action_distribution=source_summary.aggregate_action_distribution,
        evidence_boundary=(
            "This is one supervised low-margin target-objective diagnostic. Normalized "
            "supervised_low_margin_train examples are optimized, while original stress_holdout "
            "and controls are reported separately. It is not intelligence evidence and not "
            "generalization evidence unless original stress/control splits beat constant baselines."
        ),
        recommended_next_probe={
            "id": "low_margin_target_objective_decision",
            "condition": (
                "if original stress_holdout remains baseline-level, inspect "
                "decoder/readout geometry before another objective sweep"
            ),
        },
        notes=[
            "fresh cuNxon network/context per seed",
            (
                "fitness callback sees train, augmented_train and normalized "
                "supervised_low_margin_train only"
            ),
            "original stress_holdout and controls are never optimized",
            "objective-shaping diagnostic only; not a long-sweep or intelligence claim",
        ],
    )


def run_ctypes_branching_regime_scan_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    seed_offsets: Sequence[int] = (117, 118, 119, 120, 121),
    generations: int = 12,
    population_size: int = 24,
    eval_steps: int = 24,
    source_probe: str = "target_contract_augmented_train",
    device_id: int = 0,
) -> CunxonBranchingRegimeScanResult:
    """Run a bounded branching/activity-regime scan coupled to Aigarth action scores."""
    if source_probe != "target_contract_augmented_train":
        raise ValueError(
            "branching-regime scan currently supports "
            "source_probe='target_contract_augmented_train'"
        )
    if not seed_offsets:
        raise ValueError("seed_offsets must contain at least one value")
    source_result = run_ctypes_aigarth_action_target_contract_augmented_train_probe(
        library_path=library_path,
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        seed_offsets=seed_offsets,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        fitness_variant="target_contract_augmented_train",
        device_id=device_id,
    )
    gpu_sample = _query_nvidia_smi_sample(device_id)
    runs = [
        _branching_regime_run_from_aigarth_seed_run(
            run,
            gpu_memory_used_mb=gpu_sample["memory_used_mb"],
            gpu_utilization_percent=gpu_sample["utilization_percent"],
            gpu_temperature_c=gpu_sample["temperature_c"],
        )
        for run in source_result.runs
    ]
    bucket_summary = _branching_regime_bucket_summary(runs)
    correlation_summary = _branching_regime_correlation_summary(runs)
    return CunxonBranchingRegimeScanResult(
        status="branching-regime scan completed",
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        library_path=str(Path(library_path)),
        device_name=source_result.device_name,
        compute_capability=source_result.compute_capability,
        source_probe=source_probe,
        generations=generations,
        population_size=population_size,
        eval_steps=eval_steps,
        seed_offsets=list(seed_offsets),
        runs=runs,
        regime_bucket_summary=bucket_summary,
        correlation_summary=correlation_summary,
        verdict=_branching_regime_verdict(runs),
        notes=[
            "branching/activity ratio is a proxy over final action-case readout samples",
            "source action probe uses train plus augmented_train cases inside fitness only",
            (
                "holdout, stress_holdout, counterfactual_control and permuted_control "
                "labels are never optimized"
            ),
            "regime metrics are diagnostics only and must be interpreted beside task baselines",
        ],
    )


def run_ctypes_avalanche_window_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    modes: Sequence[str] = ("infer", "train"),
    seed_offsets: Sequence[int] = (122, 123, 124),
    steps: int = 256,
    sample_interval: int = 16,
    device_id: int = 0,
    action_specs: Sequence[tuple[str, str, tuple[float, float, float], str]] | None = None,
) -> CunxonAvalancheWindowProbeResult:
    """Run full-sphere snapshot windows to estimate avalanche/branching dynamics."""
    if not modes:
        raise ValueError("modes must contain at least one mode")
    invalid_modes = [mode for mode in modes if mode not in {"infer", "train"}]
    if invalid_modes:
        raise ValueError(f"unsupported avalanche-window modes: {', '.join(invalid_modes)}")
    if not seed_offsets:
        raise ValueError("seed_offsets must contain at least one value")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    if sample_interval > steps:
        raise ValueError("sample_interval must be <= steps")

    lib_path = Path(library_path)
    lib = _load_library(lib_path)
    ctx = C.c_void_p()
    decoder = ActionDecoder(num_output_neurons=3)
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xB4A9C0DE, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        gpu_sample = _query_nvidia_smi_sample(device_id)
        specs = action_specs or [
            (stimulus, "direct", input_vector, expected_action)
            for stimulus, input_vector, expected_action in _default_action_probe_specs()
        ]
        samples = [
            _run_avalanche_window_sample(
                lib=lib,
                ctx=ctx,
                mode=mode,
                seed_offset=seed_offset,
                stimulus=stimulus,
                split=split,
                input_vector=input_vector,
                expected_action=expected_action,
                steps=steps,
                sample_interval=sample_interval,
                decoder=decoder,
                gpu_memory_used_mb=gpu_sample["memory_used_mb"],
                gpu_utilization_percent=gpu_sample["utilization_percent"],
                gpu_temperature_c=gpu_sample["temperature_c"],
            )
            for mode in modes
            for seed_offset in seed_offsets
            for stimulus, split, input_vector, expected_action in specs
        ]
        return CunxonAvalancheWindowProbeResult(
            status="avalanche-window probe completed",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            modes=list(modes),
            seed_offsets=list(seed_offsets),
            steps=steps,
            sample_interval=sample_interval,
            samples=samples,
            accuracy_by_mode=_avalanche_accuracy_by_mode(samples),
            baseline_accuracy=_avalanche_baseline_accuracy(samples),
            correlation_summary=_avalanche_correlation_summary(samples),
            verdict=_avalanche_window_verdict(samples),
            notes=[
                "captures full-sphere state snapshots at bounded step intervals",
                "branching-ratio estimate uses new activations divided by previous active states",
                "action score uses the existing project action contract and trivial baselines",
                "snapshot criticality diagnostics are not intelligence evidence by themselves",
            ],
        )
    finally:
        if ctx.value:
            lib.cunxonDestroyContext(ctx)


def run_ctypes_avalanche_intervention_task_correlation_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    modes: Sequence[str] = ("infer", "train"),
    seed_offsets: Sequence[int] = (127, 128),
    device_id: int = 0,
) -> CunxonAvalancheInterventionTaskCorrelationResult:
    """Run bounded avalanche interventions over held-out/stress/control cases."""
    configs = [
        ("short-dense-heldout-stress", 128, 8),
        ("baseline-equivalent-heldout-stress", 256, 16),
    ]
    specs = _avalanche_intervention_task_specs()
    probe_results = [
        (
            config_id,
            run_ctypes_avalanche_window_probe(
                library_path=library_path,
                upstream_commit=upstream_commit,
                cunxon_commit=cunxon_commit,
                modes=modes,
                seed_offsets=seed_offsets,
                steps=steps,
                sample_interval=sample_interval,
                device_id=device_id,
                action_specs=specs,
            ),
        )
        for config_id, steps, sample_interval in configs
    ]
    samples = [sample for _, probe in probe_results for sample in probe.samples]
    config_summaries = [
        _avalanche_intervention_config_summary(
            config_id,
            probe.samples,
            probe.steps,
            probe.sample_interval,
        )
        for config_id, probe in probe_results
    ]
    split_accuracy = _avalanche_accuracy_by_split(samples)
    best_constant_baseline_by_split = _avalanche_best_constant_baseline_by_split(samples)
    stress_beaters = [
        config.id
        for config in config_summaries
        if config.beats_best_constant_baseline_by_split.get("stress_holdout", False)
    ]
    return CunxonAvalancheInterventionTaskCorrelationResult(
        status="avalanche intervention/task correlation completed",
        hypothesis_for_this_slice="avalanche_intervention_task_correlation",
        source_claim_ids=[
            "branching-ratio-regimes",
            "self-organized-criticality",
            "functional-generalization-claim",
        ],
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        library_path=str(library_path),
        device_name=probe_results[0][1].device_name,
        compute_capability=probe_results[0][1].compute_capability,
        seed_offsets=list(seed_offsets),
        modes=list(modes),
        configurations=config_summaries,
        samples=samples,
        split_accuracy=split_accuracy,
        best_constant_baseline_by_split=best_constant_baseline_by_split,
        configurations_beating_stress_baseline=stress_beaters,
        correlation_summary=_avalanche_correlation_summary(samples),
        verdict=_avalanche_intervention_task_correlation_verdict(config_summaries, samples),
        evidence_boundary=(
            "This is task-coupled regime instrumentation from bounded cuNxon snapshot windows; "
            "avalanche movement or branching-ratio estimates are not intelligence evidence unless "
            "held-out/stress task quality beats constant baselines under controls."
        ),
        notes=[
            "uses fresh seeds after the estimator-sensitivity matrix",
            "scores train, holdout, stress_holdout, counterfactual_control and "
            "permuted_control splits",
            "constant-action baselines are computed per split before interpreting regime movement",
        ],
    )


def run_ctypes_controlled_regime_calibration_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    modes: Sequence[str] = ("infer", "train"),
    seed_offsets: Sequence[int] = (133, 134),
    steps: int = 128,
    sample_interval: int = 8,
    device_id: int = 0,
) -> CunxonControlledRegimeCalibrationResult:
    """Calibrate avalanche/criticality estimator under controlled input-drive regimes."""
    regime_drive_scales = {"low-drive": 0.25, "medium-drive": 1.0, "high-drive": 2.0}
    base_specs = _avalanche_intervention_task_specs()
    probe_results = [
        (
            regime_id,
            drive_scale,
            run_ctypes_avalanche_window_probe(
                library_path=library_path,
                upstream_commit=upstream_commit,
                cunxon_commit=cunxon_commit,
                modes=modes,
                seed_offsets=seed_offsets,
                steps=steps,
                sample_interval=sample_interval,
                device_id=device_id,
                action_specs=_scaled_avalanche_specs(base_specs, regime_id, drive_scale),
            ),
        )
        for regime_id, drive_scale in regime_drive_scales.items()
    ]
    samples = [sample for _, _, probe in probe_results for sample in probe.samples]
    config_summaries = [
        _controlled_regime_calibration_config_summary(
            regime_id,
            drive_scale,
            probe.samples,
            probe.steps,
            probe.sample_interval,
        )
        for regime_id, drive_scale, probe in probe_results
    ]
    split_accuracy = _avalanche_accuracy_by_split(samples)
    best_constant_baseline_by_split = _avalanche_best_constant_baseline_by_split(samples)
    stress_beaters = [
        config.id
        for config in config_summaries
        if config.beats_best_constant_baseline_by_split.get("stress_holdout", False)
    ]
    return CunxonControlledRegimeCalibrationResult(
        status="controlled-regime calibration completed",
        hypothesis_for_this_slice="controlled_regime_calibration",
        source_issue="https://github.com/sisutuulenisa/neuraxon-hybrid/issues/86",
        source_claim_ids=[
            "branching-ratio-regimes",
            "self-organized-criticality",
            "functional-generalization-claim",
        ],
        upstream_commit=upstream_commit,
        cunxon_commit=cunxon_commit,
        library_path=str(library_path),
        device_name=probe_results[0][2].device_name,
        compute_capability=probe_results[0][2].compute_capability,
        seed_offsets=list(seed_offsets),
        modes=list(modes),
        steps=steps,
        sample_interval=sample_interval,
        regime_drive_scales=regime_drive_scales,
        configurations=config_summaries,
        samples=samples,
        split_accuracy=split_accuracy,
        best_constant_baseline_by_split=best_constant_baseline_by_split,
        stress_holdout_accuracy=split_accuracy.get("stress_holdout", 0.0),
        configurations_beating_stress_baseline=stress_beaters,
        correlation_summary=_avalanche_correlation_summary(samples),
        verdict=_controlled_regime_calibration_verdict(config_summaries, samples),
        evidence_boundary=(
            "This is controlled-regime estimator calibration for Qubic NIA Vol. 8 style "
            "criticality diagnostics; drive-dependent branching or occupancy movement is not "
            "intelligence evidence unless stress/control task quality beats constant baselines."
        ),
        notes=[
            "uses low/medium/high input-drive scales over the same held-out/stress/control cases",
            "stress_holdout and control labels remain evaluation-only; no fitness callback is used",
            "reports estimator movement, action distributions and split baselines separately",
        ],
    )


def _run_avalanche_window_sample(
    *,
    lib: C.CDLL,
    ctx: C.c_void_p,
    mode: str,
    seed_offset: int,
    stimulus: str,
    split: str,
    input_vector: tuple[float, float, float],
    expected_action: str,
    steps: int,
    sample_interval: int,
    decoder: ActionDecoder,
    gpu_memory_used_mb: int | None,
    gpu_utilization_percent: int | None,
    gpu_temperature_c: int | None,
) -> CunxonAvalancheWindowSample:
    net = C.c_void_p()
    start = time.perf_counter()
    try:
        name = f"neuraxon_hybrid_cunxon_avalanche_{mode}_{seed_offset}_{stimulus}"
        _check(lib, lib.cunxonNetworkCreate(ctx, C.byref(net), name.encode("utf-8")))
        params = _NetworkParameters()
        _check(lib, lib.cunxonGetDefaultParameters(C.byref(params)))
        params.num_input_neurons = 3
        params.num_hidden_neurons = 32
        params.num_output_neurons = 3
        params.random_seed_offset = seed_offset
        params.synapse_death_prob = 0.0
        params.synapse_formation_prob = 0.0

        sphere_id = C.c_int(-1)
        _check(
            lib,
            lib.cunxonNetworkAddSphere(
                net, b"AVALANCHE_WINDOW", CUNXON_SPHERE_SENSORY, C.byref(params), C.byref(sphere_id)
            ),
        )
        sensory_ids = (C.c_int * 3)(0, 1, 2)
        readout_start = params.num_input_neurons + params.num_hidden_neurons
        readout_ids = (C.c_int * 3)(readout_start, readout_start + 1, readout_start + 2)
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

        input_buffer = (C.c_float * len(input_vector))(*input_vector)
        input_pointer = C.cast(input_buffer, C.POINTER(C.c_float))
        ext_inputs = (C.POINTER(C.c_float) * 1)(input_pointer)
        previous_states = _capture_snapshot_states(lib, net, sphere_id.value)
        initial_energy = _capture_energy(lib, net)
        active_state_sequence: list[int] = []
        activation_event_sequence: list[int] = []
        deactivation_event_sequence: list[int] = []
        all_state_values: list[int] = []
        sampled_readouts: set[tuple[int, ...]] = set()
        step_fn = lib.cunxonNetworkStepTrain if mode == "train" else lib.cunxonNetworkStepInfer
        for step_index in range(1, steps + 1):
            _check(lib, step_fn(net, ext_inputs, C.c_float(1.0)))
            if mode == "train":
                _inject_expected_action_modulator(lib, net, expected_action)
            if step_index % sample_interval == 0 or step_index == steps:
                _check(lib, lib.cunxonContextSync(ctx))
                states = _capture_snapshot_states(lib, net, sphere_id.value)
                readout = _capture_readout(lib, net, sphere_id.value)
                sampled_readouts.add(tuple(readout))
                active_state_sequence.append(sum(1 for value in states if value != 0))
                transitions = zip(previous_states, states, strict=False)
                activation_event_sequence.append(
                    sum(1 for previous, current in transitions if previous == 0 and current != 0)
                )
                transitions = zip(previous_states, states, strict=False)
                deactivation_event_sequence.append(
                    sum(1 for previous, current in transitions if previous != 0 and current == 0)
                )
                all_state_values.extend(states)
                previous_states = states
        final_readout = _capture_readout(lib, net, sphere_id.value)
        decoded = decoder.decode(final_readout)
        normalized_action = normalize_benchmark_action(decoded.actie_type)
        energy_delta = _capture_energy(lib, net) - initial_energy
        branching_ratio = _activation_branching_ratio(
            activation_event_sequence, active_state_sequence
        )
        return CunxonAvalancheWindowSample(
            mode=mode,
            seed_offset=seed_offset,
            stimulus=stimulus,
            input_vector=list(input_vector),
            expected_action=expected_action,
            active_state_sequence=active_state_sequence,
            activation_event_sequence=activation_event_sequence,
            deactivation_event_sequence=deactivation_event_sequence,
            branching_ratio_estimate=branching_ratio,
            active_count_ratio_mean=_branching_activity_ratio_proxy(active_state_sequence),
            neutral_occupancy=(
                sum(1 for value in all_state_values if value == 0) / len(all_state_values)
                if all_state_values
                else 1.0
            ),
            transition_entropy_bits=_transition_entropy_bits(active_state_sequence),
            avalanche_event_count=sum(1 for value in activation_event_sequence if value > 0),
            mean_avalanche_size=_mean_sequence(activation_event_sequence),
            max_avalanche_size=max(activation_event_sequence) if activation_event_sequence else 0,
            final_readout=final_readout,
            normalized_action=normalized_action,
            outcome="success" if normalized_action == expected_action else "failure",
            energy_delta=energy_delta,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            split=split,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_utilization_percent=gpu_utilization_percent,
            gpu_temperature_c=gpu_temperature_c,
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)


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


def run_ctypes_input_proxy_target_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    train_epochs: int = 8,
    train_steps_per_case: int = 16,
    eval_steps: int = 16,
    device_id: int = 0,
) -> CunxonInputProxyTargetProbeResult:
    """Test supported input-class target proxies against motor readouts and baselines."""
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
            lib.cunxonNetworkCreate(
                ctx, C.byref(net), b"neuraxon_hybrid_cunxon_input_proxy_target"
            ),
        )

        params = _NetworkParameters()
        _check(lib, lib.cunxonGetDefaultParameters(C.byref(params)))
        params.num_input_neurons = 6
        params.num_hidden_neurons = 5
        params.num_output_neurons = 3
        params.random_seed_offset = 223
        params.synapse_death_prob = 0.0
        params.synapse_formation_prob = 0.0

        sphere_id = C.c_int(-1)
        _check(
            lib,
            lib.cunxonNetworkAddSphere(
                net,
                b"INPUT_PROXY_TARGET",
                CUNXON_SPHERE_MOTOR,
                C.byref(params),
                C.byref(sphere_id),
            ),
        )
        target_proxy_port_ids = [3, 4, 5]
        motor_readout_base = params.num_input_neurons + params.num_hidden_neurons
        motor_readout_port_ids = [
            motor_readout_base,
            motor_readout_base + 1,
            motor_readout_base + 2,
        ]
        sensory_and_proxy_ids = (C.c_int * 6)(0, 1, 2, *target_proxy_port_ids)
        motor_readout_ids = (C.c_int * 3)(*motor_readout_port_ids)
        _check(
            lib,
            lib.cunxonNetworkSetSphereInterface(
                net,
                sphere_id.value,
                sensory_and_proxy_ids,
                6,
                None,
                0,
                None,
                0,
                motor_readout_ids,
                3,
            ),
        )
        _check(lib, lib.cunxonNetworkFinalize(net))

        train_proxy_readouts: dict[str, list[int]] = {}
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
                states = _capture_snapshot_states(lib, net, sphere_id.value)
                train_proxy_readouts[name] = [states[index] for index in target_proxy_port_ids]

        cases: list[CunxonInputProxyTargetCase] = []
        for index, (name, split, input_vector, expected_action) in enumerate(
            _default_supervised_motor_specs()
        ):
            target = _target_readout_for_action(expected_action)
            ext_inputs = _pack_supervised_motor_inputs(input_vector, (0.0, 0.0, 0.0))
            for _ in range(eval_steps):
                _check(lib, lib.cunxonNetworkStepInfer(net, ext_inputs, C.c_float(1.0)))
            _check(lib, lib.cunxonContextSync(ctx))
            motor_readout = _capture_readout(lib, net, sphere_id.value)
            decoded = decoder.decode(motor_readout)
            normalized_action = normalize_benchmark_action(decoded.actie_type)
            proxy_readout = train_proxy_readouts.get(name, [])
            cases.append(
                CunxonInputProxyTargetCase(
                    name=name,
                    split=split,
                    input_vector=list(input_vector),
                    expected_action=expected_action,
                    target_proxy_readout=proxy_readout,
                    motor_readout=motor_readout,
                    decoded_action=decoded.actie_type,
                    normalized_action=normalized_action,
                    confidence=decoded.confidence,
                    outcome="success" if normalized_action == expected_action else "failure",
                    target_proxy_alignment=(
                        _target_alignment(proxy_readout, target) if proxy_readout else 0.0
                    ),
                    baseline_actions=_baseline_actions_for_case(index),
                    energy=_capture_energy(lib, net),
                )
            )
        return CunxonInputProxyTargetProbeResult(
            status="input-proxy target probe viable",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            train_epochs=train_epochs,
            train_steps_per_case=train_steps_per_case,
            eval_steps=eval_steps,
            target_proxy_port_ids=target_proxy_port_ids,
            motor_readout_port_ids=motor_readout_port_ids,
            cases=cases,
            accuracy_by_split=_input_proxy_accuracy_by_split(cases),
            target_proxy_alignment_by_split=_input_proxy_target_alignment_by_split(cases),
            baseline_accuracy_by_split=_input_proxy_baseline_accuracy_by_split(cases),
            notes=[
                "single sphere with sensory inputs plus input-class target proxy ports",
                (
                    "training uses StepTrain and expected-action neuromodulator pulses; "
                    "evaluation uses StepInfer without target-proxy drive"
                ),
                (
                    "proxy alignment only tests observable supported input drive; motor "
                    "holdout accuracy must beat baselines before any adapter claim"
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


def run_ctypes_resident_action_probe(
    *,
    library_path: str | Path,
    upstream_commit: str,
    cunxon_commit: str,
    train_epochs: int = 6,
    train_steps_per_case: int = 64,
    eval_steps: int = 32,
    device_id: int = 0,
) -> CunxonResidentActionProbeResult:
    """Keep one task-coupled cuNxon network resident across train/eval epochs."""
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
    start = time.perf_counter()
    try:
        _check(lib, lib.cunxonCreateContext(C.byref(ctx), device_id, 0xC0FFEE2026, 0))
        device_name = _query_device_name(lib, ctx)
        compute_capability = _query_compute_capability(lib, ctx)
        _check(
            lib,
            lib.cunxonNetworkCreate(
                ctx,
                C.byref(net),
                b"neuraxon_hybrid_cunxon_resident_action",
            ),
        )
        sensory_id, association_id, motor_id = _build_multisphere_action_topology(lib, net)
        _check(lib, lib.cunxonNetworkFinalize(net))

        specs = _default_multisphere_action_specs()
        cases: list[CunxonResidentActionCase] = []
        for epoch in range(1, train_epochs + 1):
            for _name, split, input_vector, expected_action in specs:
                if split != "train":
                    continue
                for _ in range(train_steps_per_case):
                    ext_inputs = _pack_three_sphere_inputs(input_vector)
                    _check(lib, lib.cunxonNetworkStepTrain(net, ext_inputs, C.c_float(1.0)))
                    _inject_expected_action_modulator(lib, net, expected_action)
            _check(lib, lib.cunxonContextSync(ctx))

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
                motor_states = _capture_snapshot_states(lib, net, motor_id)
                cases.append(
                    CunxonResidentActionCase(
                        epoch=epoch,
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
                        outcome=("success" if normalized_action == expected_action else "failure"),
                        baseline_actions=_baseline_actions_for_case(case_index),
                        energy=_capture_energy(lib, net),
                        motor_active_state_count=sum(1 for value in motor_states if value != 0),
                        elapsed_ms=(time.perf_counter() - start) * 1000.0,
                    )
                )

        return CunxonResidentActionProbeResult(
            status="resident action probe viable",
            upstream_commit=upstream_commit,
            cunxon_commit=cunxon_commit,
            library_path=str(lib_path),
            device_name=device_name,
            compute_capability=compute_capability,
            train_epochs=train_epochs,
            train_steps_per_case=train_steps_per_case,
            eval_steps=eval_steps,
            sphere_count=3,
            cases=cases,
            accuracy_by_epoch=_resident_accuracy_by_epoch(cases),
            accuracy_by_split=_resident_accuracy_by_split(cases),
            baseline_accuracy_by_split=_resident_baseline_accuracy_by_split(cases),
            unique_motor_readouts=len({tuple(case.motor_readout) for case in cases}),
            notes=[
                "same three-sphere cuNxon network/context remains resident across all epochs",
                (
                    "training uses StepTrain plus expected-action neuromodulator pulses; "
                    "evaluation uses StepInfer without target labels"
                ),
                "holdout accuracy must beat trivial baselines before any adapter claim",
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


def _run_aigarth_readout_mapping(
    *,
    lib: C.CDLL,
    ctx: C.c_void_p,
    mapping: str,
    readout_ids: list[int],
    neuron_class: str,
    generations: int,
    population_size: int,
    eval_steps: int,
    seed_offset: int,
) -> CunxonAigarthReadoutRun:
    net = C.c_void_p()
    callback_errors: list[Exception] = []
    try:
        name = f"neuraxon_hybrid_cunxon_aigarth_{mapping}"
        _check(lib, lib.cunxonNetworkCreate(ctx, C.byref(net), name.encode("utf-8")))
        params = _NetworkParameters()
        _check(lib, lib.cunxonGetDefaultParameters(C.byref(params)))
        params.num_input_neurons = 4
        params.num_hidden_neurons = 32
        params.num_output_neurons = 8
        params.random_seed_offset = seed_offset

        sphere_id = C.c_int(-1)
        _check(
            lib,
            lib.cunxonNetworkAddSphere(
                net, b"AIGARTH_READOUT", CUNXON_SPHERE_SENSORY, C.byref(params), C.byref(sphere_id)
            ),
        )
        sensory_ids = (C.c_int * 4)(0, 1, 2, 3)
        readout_ids_array = (C.c_int * len(readout_ids))(*readout_ids)
        _check(
            lib,
            lib.cunxonNetworkSetSphereInterface(
                net,
                sphere_id.value,
                sensory_ids,
                4,
                None,
                0,
                None,
                0,
                readout_ids_array,
                len(readout_ids),
            ),
        )
        _check(lib, lib.cunxonNetworkFinalize(net))

        baseline_margin, _baseline_pos, _baseline_neg, _, _ = _aigarth_margin_and_readouts(
            lib=lib,
            net=net,
            sphere_id=sphere_id.value,
            n_inputs=params.num_input_neurons,
            eval_steps=eval_steps,
        )

        def fitness(candidate_net: C.c_void_p, _user_data: C.c_void_p) -> float:
            try:
                margin, _, _, _, _ = _aigarth_margin_and_readouts(
                    lib=lib,
                    net=candidate_net,
                    sphere_id=sphere_id.value,
                    n_inputs=params.num_input_neurons,
                    eval_steps=eval_steps,
                )
                return float(margin)
            except Exception as exc:  # pragma: no cover - ctypes callback safety net
                callback_errors.append(exc)
                return 0.0

        callback = _CUNXON_FITNESS_FN(fitness)
        generation_margins: list[float] = []
        for generation_index in range(generations):
            fraction = generation_index / max(1, generations - 1)
            mutation_fast = 0.15 * (1.0 - 0.6 * fraction)
            mutation_slow = 0.07 * (1.0 - 0.6 * fraction)
            mutation_meta = 0.03 * (1.0 - 0.6 * fraction)
            _check(
                lib,
                lib.cunxonNetworkAigarthConfig(
                    net,
                    population_size,
                    C.c_float(mutation_fast),
                    C.c_float(mutation_slow),
                    C.c_float(mutation_meta),
                ),
            )
            _check(lib, lib.cunxonNetworkAigarthStep(net, callback, None))
            if callback_errors:
                raise callback_errors[0]
            margin, _, _, _, _ = _aigarth_margin_and_readouts(
                lib=lib,
                net=net,
                sphere_id=sphere_id.value,
                n_inputs=params.num_input_neurons,
                eval_steps=eval_steps,
            )
            generation_margins.append(margin)

        final_margin, positive_mean, negative_mean, positive_readout, negative_readout = (
            _aigarth_margin_and_readouts(
                lib=lib,
                net=net,
                sphere_id=sphere_id.value,
                n_inputs=params.num_input_neurons,
                eval_steps=eval_steps,
            )
        )
        return CunxonAigarthReadoutRun(
            mapping=mapping,
            readout_ids=readout_ids,
            neuron_class=neuron_class,
            baseline_margin=baseline_margin,
            generation_margins=generation_margins,
            final_margin=final_margin,
            improvement=final_margin - baseline_margin,
            positive_mean=positive_mean,
            negative_mean=negative_mean,
            positive_readout=positive_readout,
            negative_readout=negative_readout,
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)


def _aigarth_margin_and_readouts(
    *,
    lib: C.CDLL,
    net: C.c_void_p,
    sphere_id: int,
    n_inputs: int,
    eval_steps: int,
) -> tuple[float, float, float, list[int], list[int]]:
    positive_mean, positive_readout = _aigarth_mean_for_input(
        lib=lib,
        net=net,
        sphere_id=sphere_id,
        n_inputs=n_inputs,
        input_value=0.85,
        eval_steps=eval_steps,
    )
    negative_mean, negative_readout = _aigarth_mean_for_input(
        lib=lib,
        net=net,
        sphere_id=sphere_id,
        n_inputs=n_inputs,
        input_value=-0.85,
        eval_steps=eval_steps,
    )
    return (
        positive_mean - negative_mean,
        positive_mean,
        negative_mean,
        positive_readout,
        negative_readout,
    )


def _aigarth_mean_for_input(
    *,
    lib: C.CDLL,
    net: C.c_void_p,
    sphere_id: int,
    n_inputs: int,
    input_value: float,
    eval_steps: int,
) -> tuple[float, list[int]]:
    _check(lib, lib.cunxonNetworkReset(net))
    input_buffer = (C.c_float * n_inputs)(*[input_value] * n_inputs)
    input_pointer = C.cast(input_buffer, C.POINTER(C.c_float))
    ext_inputs = (C.POINTER(C.c_float) * 1)(input_pointer)
    for _ in range(eval_steps):
        _check(lib, lib.cunxonNetworkStepInfer(net, ext_inputs, C.c_float(1.0)))
    readout = _capture_readout(lib, net, sphere_id)
    return sum(readout) / len(readout), readout


def _evaluate_aigarth_action_cases(
    *,
    lib: C.CDLL,
    net: C.c_void_p,
    sphere_id: int,
    specs: Sequence[tuple[str, str, tuple[float, float, float], str]],
    eval_steps: int,
    decoder: ActionDecoder,
    decoder_strategy: str = "action_decoder",
) -> list[CunxonAigarthActionCase]:
    cases: list[CunxonAigarthActionCase] = []
    for index, (name, split, input_vector, expected_action) in enumerate(specs):
        _check(lib, lib.cunxonNetworkReset(net))
        input_buffer = (C.c_float * len(input_vector))(*input_vector)
        input_pointer = C.cast(input_buffer, C.POINTER(C.c_float))
        ext_inputs = (C.POINTER(C.c_float) * 1)(input_pointer)
        for _ in range(eval_steps):
            _check(lib, lib.cunxonNetworkStepInfer(net, ext_inputs, C.c_float(1.0)))
        readout = _capture_readout(lib, net, sphere_id)
        if decoder_strategy == "target_contract":
            normalized_action = remap_aigarth_action_readout(readout)
            decoded_action = normalized_action
            confidence = 1.0 if readout and readout[0] != 0 else 0.0
        else:
            decoded = decoder.decode(readout)
            decoded_action = decoded.actie_type
            normalized_action = normalize_benchmark_action(decoded.actie_type)
            confidence = decoded.confidence
        target = _target_readout_for_action(expected_action)
        cases.append(
            CunxonAigarthActionCase(
                name=name,
                split=split,
                input_vector=list(input_vector),
                expected_action=expected_action,
                target_readout=list(target),
                readout=readout,
                decoded_action=decoded_action,
                normalized_action=normalized_action,
                confidence=confidence,
                outcome="success" if normalized_action == expected_action else "failure",
                target_alignment=_target_alignment(readout, target),
                baseline_actions=_baseline_actions_for_case(index),
                energy=_capture_energy(lib, net),
            )
        )
    return cases


def _aigarth_action_accuracy_by_split(
    cases: Sequence[CunxonAigarthActionCase],
) -> dict[str, float]:
    by_split: dict[str, list[CunxonAigarthActionCase]] = {}
    for case in cases:
        by_split.setdefault(case.split, []).append(case)
    by_split["overall"] = list(cases)
    return {
        split: sum(1 for case in split_cases if case.outcome == "success") / len(split_cases)
        for split, split_cases in sorted(by_split.items())
        if split_cases
    }


def _aigarth_action_target_alignment_by_split(
    cases: Sequence[CunxonAigarthActionCase],
) -> dict[str, float]:
    by_split: dict[str, list[CunxonAigarthActionCase]] = {}
    for case in cases:
        by_split.setdefault(case.split, []).append(case)
    by_split["overall"] = list(cases)
    return {
        split: sum(case.target_alignment for case in split_cases) / len(split_cases)
        for split, split_cases in sorted(by_split.items())
        if split_cases
    }


def _aigarth_action_baseline_accuracy_by_split(
    cases: Sequence[CunxonAigarthActionCase],
) -> dict[str, dict[str, float]]:
    baseline_names = sorted({name for case in cases for name in case.baseline_actions})
    result: dict[str, dict[str, float]] = {}
    for baseline_name in baseline_names:
        split_scores: dict[str, float] = {}
        for split in sorted({case.split for case in cases} | {"overall"}):
            split_cases = (
                list(cases)
                if split == "overall"
                else [case for case in cases if case.split == split]
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


def _aigarth_action_distribution(
    cases: Sequence[CunxonAigarthActionCase],
) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for case in cases:
        distribution[case.normalized_action] = distribution.get(case.normalized_action, 0) + 1
    return dict(sorted(distribution.items()))


def remap_aigarth_action_readout(readout: Sequence[int]) -> str:
    """Map a three-lane cuNxon action readout through the target-readout contract."""
    first_lane = readout[0] if readout else 0
    if first_lane > 0:
        return "execute"
    if first_lane < 0:
        return "retry"
    return "query"


def _remap_case_accuracy_by_split(
    cases: Sequence[CunxonAigarthActionRemapCase],
    *,
    remapped: bool,
) -> dict[str, float]:
    by_split: dict[str, list[CunxonAigarthActionRemapCase]] = {}
    for case in cases:
        by_split.setdefault(case.split, []).append(case)
    by_split["overall"] = list(cases)
    return {
        split: sum(
            1
            for case in split_cases
            if (
                case.remapped_outcome == "success"
                if remapped
                else case.original_outcome == "success"
            )
        )
        / len(split_cases)
        for split, split_cases in sorted(by_split.items())
        if split_cases
    }


def _remap_case_action_distribution(
    cases: Sequence[CunxonAigarthActionRemapCase],
    *,
    remapped: bool,
) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for case in cases:
        action = case.remapped_action if remapped else case.original_normalized_action
        distribution[action] = distribution.get(action, 0) + 1
    return dict(sorted(distribution.items()))


def _seed_sweep_accuracy_summary(
    runs: Sequence[CunxonAigarthActionSeedRun],
) -> dict[str, dict[str, float]]:
    splits = sorted({split for run in runs for split in run.accuracy_by_split})
    summary: dict[str, dict[str, float]] = {}
    for split in splits:
        values = [run.accuracy_by_split[split] for run in runs if split in run.accuracy_by_split]
        if values:
            summary[split] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
    return summary


def _seed_sweep_action_distribution(
    runs: Sequence[CunxonAigarthActionSeedRun],
) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for run in runs:
        for action, count in run.action_distribution.items():
            distribution[action] = distribution.get(action, 0) + count
    return dict(sorted(distribution.items()))


def _seed_sweep_beating_baseline_counts(
    runs: Sequence[CunxonAigarthActionSeedRun],
) -> dict[str, int]:
    splits = sorted({split for run in runs for split in run.accuracy_by_split})
    counts: dict[str, int] = {}
    for split in splits:
        count = 0
        for run in runs:
            if split not in run.accuracy_by_split:
                continue
            baseline_scores = [
                by_split[split]
                for by_split in run.baseline_accuracy_by_split.values()
                if split in by_split
            ]
            best_baseline = max(baseline_scores) if baseline_scores else 0.0
            if run.accuracy_by_split[split] > best_baseline:
                count += 1
        counts[split] = count
    return counts


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
        pseudo_cases = [case for case in cases if baseline_name in case.baseline_actions]
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


def _resident_accuracy_by_epoch(
    cases: Sequence[CunxonResidentActionCase],
) -> dict[str, float]:
    by_epoch: dict[int, list[CunxonResidentActionCase]] = {}
    for case in cases:
        by_epoch.setdefault(case.epoch, []).append(case)
    return {
        str(epoch): _resident_case_accuracy(epoch_cases)
        for epoch, epoch_cases in sorted(by_epoch.items())
        if epoch_cases
    }


def _resident_accuracy_by_split(
    cases: Sequence[CunxonResidentActionCase],
) -> dict[str, float]:
    by_split: dict[str, list[CunxonResidentActionCase]] = {}
    for case in cases:
        by_split.setdefault(case.split, []).append(case)
    by_split["overall"] = list(cases)
    return {
        split: _resident_case_accuracy(split_cases)
        for split, split_cases in sorted(by_split.items())
        if split_cases
    }


def _resident_baseline_accuracy_by_split(
    cases: Sequence[CunxonResidentActionCase],
) -> dict[str, dict[str, float]]:
    baseline_names = sorted({name for case in cases for name in case.baseline_actions})
    result: dict[str, dict[str, float]] = {}
    for baseline_name in baseline_names:
        split_scores: dict[str, float] = {}
        for split in sorted({case.split for case in cases} | {"overall"}):
            split_cases = (
                list(cases)
                if split == "overall"
                else [case for case in cases if case.split == split]
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


def _resident_case_accuracy(cases: Sequence[CunxonResidentActionCase]) -> float:
    if not cases:
        return 0.0
    successes = sum(1 for case in cases if case.outcome == "success")
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


def _branching_regime_run_from_aigarth_seed_run(
    run: CunxonAigarthActionSeedRun,
    *,
    gpu_memory_used_mb: int | None,
    gpu_utilization_percent: int | None,
    gpu_temperature_c: int | None,
) -> CunxonBranchingRegimeRun:
    """Derive activity-regime proxy metrics from one scored Aigarth action seed."""
    active_sequence = [sum(1 for value in case.readout if value != 0) for case in run.cases]
    if not active_sequence:
        active_sequence = [0]
    readout_values = [value for case in run.cases for value in case.readout]
    neutral_occupancy = (
        sum(1 for value in readout_values if value == 0) / len(readout_values)
        if readout_values
        else 1.0
    )
    energies = [case.energy for case in run.cases]
    energy_slope = (
        (energies[-1] - energies[0]) / max(1, len(energies) - 1) if len(energies) > 1 else 0.0
    )
    best_baselines = _best_constant_baseline_by_split(run.baseline_accuracy_by_split)
    return CunxonBranchingRegimeRun(
        seed_offset=run.seed_offset,
        active_state_sequence=active_sequence,
        branching_activity_ratio_proxy=_branching_activity_ratio_proxy(active_sequence),
        neutral_occupancy=neutral_occupancy,
        transition_entropy_bits=_transition_entropy_bits(active_sequence),
        energy_slope=energy_slope,
        readout_diversity=run.unique_readouts,
        regime_label=_classify_branching_regime(active_sequence, neutral_occupancy),
        accuracy_by_split=run.accuracy_by_split,
        best_constant_baseline_by_split=best_baselines,
        beats_best_baseline_by_split={
            split: accuracy > best_baselines.get(split, 0.0)
            for split, accuracy in run.accuracy_by_split.items()
        },
        action_distribution=run.action_distribution,
        gpu_memory_used_mb=gpu_memory_used_mb,
        gpu_utilization_percent=gpu_utilization_percent,
        gpu_temperature_c=gpu_temperature_c,
    )


def _branching_activity_ratio_proxy(active_sequence: Sequence[int]) -> float:
    """Estimate a coarse branching/activity ratio from sampled active-count transitions."""
    if len(active_sequence) < 2:
        return 0.0
    ratios: list[float] = []
    for previous, current in zip(active_sequence, active_sequence[1:], strict=False):
        if previous == 0:
            ratios.append(1.0 if current == 0 else float(current))
        else:
            ratios.append(current / previous)
    return sum(ratios) / len(ratios) if ratios else 0.0


def _transition_entropy_bits(active_sequence: Sequence[int]) -> float:
    if not active_sequence:
        return 0.0
    counts: dict[int, int] = {}
    for value in active_sequence:
        counts[value] = counts.get(value, 0) + 1
    total = len(active_sequence)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _classify_branching_regime(active_sequence: Sequence[int], neutral_occupancy: float) -> str:
    ratio = _branching_activity_ratio_proxy(active_sequence)
    mean_active = sum(active_sequence) / len(active_sequence) if active_sequence else 0.0
    if mean_active <= 0.1 or ratio < 0.75:
        return "dead/subcritical proxy"
    if ratio > 1.25 or neutral_occupancy < 0.2:
        return "runaway/saturated proxy"
    return "reverberating/near-critical proxy"


def _best_constant_baseline_by_split(
    baseline_accuracy_by_split: dict[str, dict[str, float]],
) -> dict[str, float]:
    splits = sorted({split for values in baseline_accuracy_by_split.values() for split in values})
    return {
        split: max(values.get(split, 0.0) for values in baseline_accuracy_by_split.values())
        for split in splits
    }


def _branching_regime_bucket_summary(
    runs: Sequence[CunxonBranchingRegimeRun],
) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[CunxonBranchingRegimeRun]] = {}
    for run in runs:
        buckets.setdefault(run.regime_label, []).append(run)
    summary: dict[str, dict[str, float | int]] = {}
    for label, bucket_runs in buckets.items():
        summary[label] = {
            "seed_count": len(bucket_runs),
            "mean_branching_activity_ratio_proxy": _mean_sequence(
                run.branching_activity_ratio_proxy for run in bucket_runs
            ),
            "mean_holdout_accuracy": _mean_sequence(
                run.accuracy_by_split.get("holdout", 0.0) for run in bucket_runs
            ),
            "mean_stress_holdout_accuracy": _mean_sequence(
                run.accuracy_by_split.get("stress_holdout", 0.0) for run in bucket_runs
            ),
            "beats_stress_baseline_count": sum(
                1
                for run in bucket_runs
                if run.beats_best_baseline_by_split.get("stress_holdout", False)
            ),
        }
    return dict(sorted(summary.items()))


def _branching_regime_correlation_summary(
    runs: Sequence[CunxonBranchingRegimeRun],
) -> dict[str, float]:
    ratios = [run.branching_activity_ratio_proxy for run in runs]
    return {
        "holdout_accuracy_vs_branching_proxy": _pearson_correlation(
            ratios, [run.accuracy_by_split.get("holdout", 0.0) for run in runs]
        ),
        "stress_holdout_accuracy_vs_branching_proxy": _pearson_correlation(
            ratios, [run.accuracy_by_split.get("stress_holdout", 0.0) for run in runs]
        ),
    }


def _branching_regime_verdict(runs: Sequence[CunxonBranchingRegimeRun]) -> str:
    if not runs:
        return "No runs were captured; no branching-regime or action-quality claim is supported."
    near_runs = [run for run in runs if run.regime_label == "reverberating/near-critical proxy"]
    stress_beats = sum(
        1 for run in near_runs if run.beats_best_baseline_by_split.get("stress_holdout", False)
    )
    if near_runs and stress_beats == len(near_runs):
        return (
            "All near-critical proxy runs beat the best constant baseline on stress_holdout, "
            "but this remains a tiny proxy scan rather than intelligence evidence."
        )
    if near_runs:
        return (
            "Near-critical-looking proxy activity did not consistently beat the best constant "
            "baseline on stress_holdout; branching ratio remains diagnostic, not "
            "sufficient evidence."
        )
    return (
        "This scan did not observe a near-critical proxy bucket; regime instrumentation is useful, "
        "but no criticality/action-quality conclusion is supported."
    )


def _activation_branching_ratio(
    activation_events: Sequence[int], active_state_sequence: Sequence[int]
) -> float:
    """Estimate descendant activations per previously active state across sampled windows."""
    if not activation_events:
        return 0.0
    denominators = [max(1, active) for active in active_state_sequence]
    paired = list(zip(activation_events, denominators, strict=False))
    if not paired:
        return 0.0
    return sum(events / previous_active for events, previous_active in paired) / len(paired)


def _avalanche_accuracy_by_mode(
    samples: Sequence[CunxonAvalancheWindowSample],
) -> dict[str, float]:
    by_mode: dict[str, list[CunxonAvalancheWindowSample]] = {}
    for sample in samples:
        by_mode.setdefault(sample.mode, []).append(sample)
    return {
        mode: sum(1 for sample in mode_samples if sample.outcome == "success") / len(mode_samples)
        for mode, mode_samples in sorted(by_mode.items())
        if mode_samples
    }


def _avalanche_baseline_accuracy(
    samples: Sequence[CunxonAvalancheWindowSample],
) -> dict[str, float]:
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


def _avalanche_correlation_summary(
    samples: Sequence[CunxonAvalancheWindowSample],
) -> dict[str, float]:
    ratios = [sample.branching_ratio_estimate for sample in samples]
    active_ratios = [sample.active_count_ratio_mean for sample in samples]
    outcomes = [1.0 if sample.outcome == "success" else 0.0 for sample in samples]
    neutral = [sample.neutral_occupancy for sample in samples]
    return {
        "accuracy_vs_branching_ratio_estimate": _pearson_correlation(ratios, outcomes),
        "accuracy_vs_active_count_ratio_mean": _pearson_correlation(active_ratios, outcomes),
        "accuracy_vs_neutral_occupancy": _pearson_correlation(neutral, outcomes),
    }


def _avalanche_window_verdict(samples: Sequence[CunxonAvalancheWindowSample]) -> str:
    if not samples:
        return (
            "No snapshot windows were captured; no avalanche, regime, or action-quality claim "
            "is supported."
        )
    accuracy_by_mode = _avalanche_accuracy_by_mode(samples)
    baselines = _avalanche_baseline_accuracy(samples)
    best_baseline = max(baselines.values()) if baselines else 0.0
    beats = [value > best_baseline for value in accuracy_by_mode.values()]
    mean_branching = _mean_sequence(sample.branching_ratio_estimate for sample in samples)
    if beats and all(beats):
        return (
            "Snapshot-level avalanche metrics were captured and every mode beat the best "
            "constant baseline, but this remains a tiny diagnostic probe rather than "
            "intelligence evidence."
        )
    return (
        f"Snapshot-level avalanche metrics were captured (mean branching-ratio estimate "
        f"{mean_branching:.6f}), but action quality did not beat the best constant baseline "
        "for every mode; criticality remains instrumentation, not sufficient evidence."
    )


def _avalanche_accuracy_by_split(
    samples: Sequence[CunxonAvalancheWindowSample],
) -> dict[str, float]:
    by_split: dict[str, list[CunxonAvalancheWindowSample]] = {}
    for sample in samples:
        by_split.setdefault(sample.split, []).append(sample)
    return {
        split: sum(1 for sample in split_samples if sample.outcome == "success")
        / len(split_samples)
        for split, split_samples in sorted(by_split.items())
        if split_samples
    }


def _avalanche_best_constant_baseline_by_split(
    samples: Sequence[CunxonAvalancheWindowSample],
) -> dict[str, float]:
    baselines = {
        "always_execute": "execute",
        "always_query": "query",
        "always_retry": "retry",
    }
    by_split: dict[str, list[CunxonAvalancheWindowSample]] = {}
    for sample in samples:
        by_split.setdefault(sample.split, []).append(sample)
    best_by_split: dict[str, float] = {}
    for split, split_samples in sorted(by_split.items()):
        best_by_split[split] = max(
            sum(1 for sample in split_samples if action == sample.expected_action)
            / len(split_samples)
            for action in baselines.values()
        )
    return best_by_split


def _avalanche_intervention_config_summary(
    config_id: str,
    samples: Sequence[CunxonAvalancheWindowSample],
    steps: int,
    sample_interval: int,
) -> CunxonAvalancheInterventionTaskConfigSummary:
    ratios = [sample.branching_ratio_estimate for sample in samples]
    split_accuracy = _avalanche_accuracy_by_split(samples)
    baselines = _avalanche_best_constant_baseline_by_split(samples)
    return CunxonAvalancheInterventionTaskConfigSummary(
        id=config_id,
        steps=steps,
        sample_interval=sample_interval,
        sample_count=len(samples),
        mean_branching_ratio_estimate=_mean_sequence(ratios),
        branching_ratio_estimate_range=[
            min(ratios) if ratios else 0.0,
            max(ratios) if ratios else 0.0,
        ],
        mean_neutral_occupancy=_mean_sequence(sample.neutral_occupancy for sample in samples),
        accuracy_by_split=split_accuracy,
        best_constant_baseline_by_split=baselines,
        beats_best_constant_baseline_by_split={
            split: accuracy > baselines.get(split, 0.0)
            for split, accuracy in sorted(split_accuracy.items())
        },
    )


def _scaled_avalanche_specs(
    specs: Sequence[tuple[str, str, tuple[float, float, float], str]],
    regime_id: str,
    drive_scale: float,
) -> list[tuple[str, str, tuple[float, float, float], str]]:
    """Return action specs with input vectors scaled for a controlled drive regime."""
    scaled_specs: list[tuple[str, str, tuple[float, float, float], str]] = []
    for stimulus, split, input_vector, expected_action in specs:
        scaled_vector = (
            round(input_vector[0] * drive_scale, 6),
            round(input_vector[1] * drive_scale, 6),
            round(input_vector[2] * drive_scale, 6),
        )
        scaled_specs.append((f"{regime_id}-{stimulus}", split, scaled_vector, expected_action))
    return scaled_specs


def _avalanche_sample_action_distribution(
    samples: Sequence[CunxonAvalancheWindowSample],
) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for sample in samples:
        distribution[sample.normalized_action] = distribution.get(sample.normalized_action, 0) + 1
    return dict(sorted(distribution.items()))


def _controlled_regime_calibration_config_summary(
    config_id: str,
    drive_scale: float,
    samples: Sequence[CunxonAvalancheWindowSample],
    steps: int,
    sample_interval: int,
) -> CunxonControlledRegimeCalibrationConfigSummary:
    ratios = [sample.branching_ratio_estimate for sample in samples]
    split_accuracy = _avalanche_accuracy_by_split(samples)
    baselines = _avalanche_best_constant_baseline_by_split(samples)
    return CunxonControlledRegimeCalibrationConfigSummary(
        id=config_id,
        drive_scale=drive_scale,
        steps=steps,
        sample_interval=sample_interval,
        sample_count=len(samples),
        mean_branching_ratio_estimate=_mean_sequence(ratios),
        branching_ratio_estimate_range=[
            min(ratios) if ratios else 0.0,
            max(ratios) if ratios else 0.0,
        ],
        mean_active_count_ratio=_mean_sequence(
            sample.active_count_ratio_mean for sample in samples
        ),
        mean_neutral_occupancy=_mean_sequence(sample.neutral_occupancy for sample in samples),
        mean_transition_entropy_bits=_mean_sequence(
            sample.transition_entropy_bits for sample in samples
        ),
        action_distribution=_avalanche_sample_action_distribution(samples),
        accuracy_by_split=split_accuracy,
        best_constant_baseline_by_split=baselines,
        beats_best_constant_baseline_by_split={
            split: accuracy > baselines.get(split, 0.0)
            for split, accuracy in sorted(split_accuracy.items())
        },
    )


def _controlled_regime_calibration_verdict(
    configs: Sequence[CunxonControlledRegimeCalibrationConfigSummary],
    samples: Sequence[CunxonAvalancheWindowSample],
) -> str:
    if not samples:
        return "No controlled-regime samples were captured; no claim is supported."
    stress_beaters = [
        config.id
        for config in configs
        if config.beats_best_constant_baseline_by_split.get("stress_holdout", False)
    ]
    branching_range = [
        min(sample.branching_ratio_estimate for sample in samples),
        max(sample.branching_ratio_estimate for sample in samples),
    ]
    if stress_beaters:
        return (
            "Controlled-regime estimator calibration found stress_holdout beaters "
            f"({', '.join(stress_beaters)}), but this remains bounded toy evidence and "
            "needs independent controls before any broader claim."
        )
    return (
        "Controlled-regime estimator calibration moved/recorded branching and occupancy "
        f"metrics across drive settings (branching range {branching_range[0]:.6f}.."
        f"{branching_range[1]:.6f}), but no drive regime beat the best constant baseline "
        "on stress_holdout; criticality remains diagnostic instrumentation, not "
        "intelligence evidence."
    )


def _avalanche_intervention_task_correlation_verdict(
    configs: Sequence[CunxonAvalancheInterventionTaskConfigSummary],
    samples: Sequence[CunxonAvalancheWindowSample],
) -> str:
    if not samples:
        return "No task-coupled avalanche windows were captured; no claim is supported."
    stress_beaters = [
        config.id
        for config in configs
        if config.beats_best_constant_baseline_by_split.get("stress_holdout", False)
    ]
    mean_branching = _mean_sequence(sample.branching_ratio_estimate for sample in samples)
    if stress_beaters:
        return (
            "Some configurations beat constant baselines on stress_holdout "
            f"({', '.join(stress_beaters)}), but this remains bounded toy evidence and needs "
            "independent controls before any broader claim."
        )
    return (
        "Bounded task-coupled avalanche interventions moved/recorded regime metrics "
        "(mean branching "
        f"estimate {mean_branching:.6f}), but no configuration beat the best constant baseline on "
        "stress_holdout; criticality remains diagnostic instrumentation, not intelligence evidence."
    )


def _mean_sequence(values: Sequence[float] | Any) -> float:
    values_list = list(values)
    return sum(values_list) / len(values_list) if values_list else 0.0


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=False))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    return numerator / denominator if denominator else 0.0


def _default_action_probe_specs() -> list[tuple[str, tuple[float, float, float], str]]:
    """Return a tiny fixed action-contract task suite for cuNxon probes."""
    return [
        ("execute-positive-drive", (1.0, 0.25, 0.0), "execute"),
        ("retry-negative-drive", (-1.0, -0.25, 0.0), "retry"),
        ("query-neutral-drive", (0.0, 0.0, 0.0), "query"),
    ]


def _avalanche_intervention_task_specs() -> list[tuple[str, str, tuple[float, float, float], str]]:
    """Return held-out/stress/control cases for task-coupled avalanche probes."""
    return _aigarth_action_target_contract_stress_specs()


def _default_supervised_motor_specs() -> list[tuple[str, str, tuple[float, float, float], str]]:
    """Return train/holdout cases for absolute-output motor-target testing."""
    return [
        ("execute-train", "train", (1.0, 0.25, 0.0), "execute"),
        ("retry-train", "train", (-1.0, -0.25, 0.0), "retry"),
        ("query-train", "train", (0.0, 0.0, 0.0), "query"),
        ("execute-holdout-noisy", "holdout", (0.8, 0.2, 0.1), "execute"),
        ("retry-holdout-noisy", "holdout", (-0.8, -0.2, 0.1), "retry"),
        ("query-holdout-low-drive", "holdout", (0.05, 0.0, -0.05), "query"),
    ]


def _aigarth_action_hard_holdout_specs() -> list[tuple[str, str, tuple[float, float, float], str]]:
    """Return train/holdout/hard/control cases for Aigarth action audits."""
    return [
        *_default_supervised_motor_specs(),
        ("execute-hard-low-positive", "hard_holdout", (0.35, 0.05, -0.1), "execute"),
        ("execute-hard-conflict", "hard_holdout", (0.6, -0.4, 0.2), "execute"),
        ("retry-hard-low-negative", "hard_holdout", (-0.35, -0.05, 0.2), "retry"),
        ("retry-hard-conflict", "hard_holdout", (-0.6, 0.4, -0.2), "retry"),
        ("query-hard-balanced", "hard_holdout", (0.1, -0.1, 0.05), "query"),
        ("query-hard-weak-drive", "hard_holdout", (-0.05, 0.05, 0.1), "query"),
        ("execute-train-permuted-as-retry", "permuted_control", (1.0, 0.25, 0.0), "retry"),
        ("retry-train-permuted-as-query", "permuted_control", (-1.0, -0.25, 0.0), "query"),
        ("query-train-permuted-as-execute", "permuted_control", (0.0, 0.0, 0.0), "execute"),
    ]


def _aigarth_action_target_contract_stress_specs() -> list[
    tuple[str, str, tuple[float, float, float], str]
]:
    """Return stricter target-contract stress cases beyond the standard hard holdout."""
    return [
        *_aigarth_action_hard_holdout_specs(),
        ("execute-stress-low-margin", "stress_holdout", (0.18, -0.12, 0.08), "execute"),
        ("execute-stress-near-neutral", "stress_holdout", (0.12, 0.08, -0.12), "execute"),
        ("retry-stress-low-margin", "stress_holdout", (-0.18, 0.12, -0.08), "retry"),
        ("retry-stress-near-neutral", "stress_holdout", (-0.12, -0.08, 0.12), "retry"),
        ("query-stress-positive-negative", "stress_holdout", (0.18, -0.18, 0.0), "query"),
        ("query-stress-negative-positive", "stress_holdout", (-0.18, 0.18, 0.0), "query"),
        (
            "execute-holdout-counterfactual-query",
            "counterfactual_control",
            (0.8, 0.2, 0.1),
            "query",
        ),
        (
            "retry-holdout-counterfactual-execute",
            "counterfactual_control",
            (-0.8, -0.2, 0.1),
            "execute",
        ),
        (
            "query-holdout-counterfactual-retry",
            "counterfactual_control",
            (0.05, 0.0, -0.05),
            "retry",
        ),
    ]


def _aigarth_action_target_contract_augmented_train_specs() -> list[
    tuple[str, str, tuple[float, float, float], str]
]:
    """Return target-contract stress cases plus low-margin train-only augmentations."""
    return [
        *_aigarth_action_target_contract_stress_specs(),
        (
            "execute-augmented-low-margin-train",
            "augmented_train",
            (0.28, 0.06, -0.02),
            "execute",
        ),
        (
            "execute-augmented-conflict-train",
            "augmented_train",
            (0.42, -0.18, 0.12),
            "execute",
        ),
        (
            "retry-augmented-low-margin-train",
            "augmented_train",
            (-0.28, -0.06, 0.02),
            "retry",
        ),
        (
            "retry-augmented-conflict-train",
            "augmented_train",
            (-0.42, 0.18, -0.12),
            "retry",
        ),
        (
            "query-augmented-balanced-train",
            "augmented_train",
            (0.12, -0.12, 0.02),
            "query",
        ),
        (
            "query-augmented-weak-train",
            "augmented_train",
            (-0.08, 0.08, -0.02),
            "query",
        ),
    ]


def _aigarth_action_target_contract_stress_injection_specs() -> list[
    tuple[str, str, tuple[float, float, float], str]
]:
    """Return stress audit specs plus duplicated stress_train upper-bound cases."""
    stress_train_specs = [
        (f"{name}-train-injected", "stress_train", vector, expected)
        for name, split, vector, expected in _aigarth_action_target_contract_stress_specs()
        if split == "stress_holdout"
    ]
    return [
        *_aigarth_action_target_contract_augmented_train_specs(),
        *stress_train_specs,
    ]


def _supervised_low_margin_vector(
    vector: tuple[float, float, float], expected_action: str
) -> tuple[float, float, float]:
    """Normalize stress-like vectors just past the target-contract query boundary."""
    if expected_action == "execute":
        return (max(abs(vector[0]), 0.30), vector[1], vector[2])
    if expected_action == "retry":
        return (-max(abs(vector[0]), 0.30), vector[1], vector[2])
    return vector


def _aigarth_action_target_contract_supervised_low_margin_specs() -> list[
    tuple[str, str, tuple[float, float, float], str]
]:
    """Return specs with normalized supervised low-margin train-only examples."""
    supervised_specs = [
        (
            f"{name}-supervised-low-margin-train",
            "supervised_low_margin_train",
            _supervised_low_margin_vector(vector, expected),
            expected,
        )
        for name, split, vector, expected in _aigarth_action_target_contract_stress_specs()
        if split == "stress_holdout"
    ]
    return [
        *_aigarth_action_target_contract_augmented_train_specs(),
        *supervised_specs,
    ]


def _amplitude_split_suffix(factor: float) -> str:
    return f"{factor:.1f}".replace(".", "_") + "x"


def _scale_stress_vector(
    vector: tuple[float, float, float], factor: float
) -> tuple[float, float, float]:
    return (float(vector[0] * factor), float(vector[1] * factor), float(vector[2] * factor))


def _aigarth_action_target_contract_stress_amplitude_ladder_specs(
    amplitude_factors: Sequence[float],
) -> list[tuple[str, str, tuple[float, float, float], str]]:
    """Return specs with scaled stress train/evaluation splits for an amplitude ladder."""
    if not amplitude_factors:
        raise ValueError("amplitude_factors must contain at least one value")
    if any(factor <= 0.0 for factor in amplitude_factors):
        raise ValueError("amplitude_factors must be positive")
    base_specs = _aigarth_action_target_contract_augmented_train_specs()
    stress_specs = [
        (name, vector, expected)
        for name, split, vector, expected in _aigarth_action_target_contract_stress_specs()
        if split == "stress_holdout"
    ]
    scaled_specs: list[tuple[str, str, tuple[float, float, float], str]] = []
    for factor in amplitude_factors:
        suffix = _amplitude_split_suffix(factor)
        for name, vector, expected in stress_specs:
            scaled_vector = _scale_stress_vector(vector, factor)
            scaled_specs.append(
                (
                    f"{name}-scaled-{suffix}-train",
                    f"stress_train_scaled_{suffix}",
                    scaled_vector,
                    expected,
                )
            )
            scaled_specs.append(
                (
                    f"{name}-scaled-{suffix}-holdout",
                    f"stress_holdout_scaled_{suffix}",
                    scaled_vector,
                    expected,
                )
            )
    return [*base_specs, *scaled_specs]


def _aigarth_action_target_contract_stress_objective_specs(
    amplitude_factor: float,
) -> list[tuple[str, str, tuple[float, float, float], str]]:
    """Return specs for one target-aligned stress objective-shaping run."""
    return _aigarth_action_target_contract_stress_amplitude_ladder_specs([amplitude_factor])


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


def _target_contract_margin(case: CunxonAigarthActionCase) -> float:
    """Score how strongly the first readout lane supports the expected action."""
    first_lane = case.readout[0] if case.readout else 0
    if case.expected_action == "execute":
        return max(0.0, float(first_lane))
    if case.expected_action == "retry":
        return max(0.0, float(-first_lane))
    return 1.0 if first_lane == 0 else 0.0


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


def _input_proxy_accuracy_by_split(
    cases: Sequence[CunxonInputProxyTargetCase],
) -> dict[str, float]:
    by_split: dict[str, list[CunxonInputProxyTargetCase]] = {}
    for case in cases:
        by_split.setdefault(case.split, []).append(case)
    by_split["overall"] = list(cases)
    return {
        split: sum(1 for case in split_cases if case.outcome == "success") / len(split_cases)
        for split, split_cases in sorted(by_split.items())
        if split_cases
    }


def _input_proxy_target_alignment_by_split(
    cases: Sequence[CunxonInputProxyTargetCase],
) -> dict[str, float]:
    by_split: dict[str, list[CunxonInputProxyTargetCase]] = {}
    for case in cases:
        by_split.setdefault(case.split, []).append(case)
    by_split["overall"] = list(cases)
    return {
        split: sum(case.target_proxy_alignment for case in split_cases) / len(split_cases)
        for split, split_cases in sorted(by_split.items())
        if split_cases
    }


def _input_proxy_baseline_accuracy_by_split(
    cases: Sequence[CunxonInputProxyTargetCase],
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


def _run_external_drive_window_sample(
    *,
    lib: C.CDLL,
    ctx: C.c_void_p,
    mode: str,
    driven_port_class: str,
    sensory_port_ids: Sequence[int],
    readout_port_ids: Sequence[int],
    input_vector: Sequence[float],
    steps: int,
) -> CunxonExternalDriveWindowSample:
    net = C.c_void_p()
    try:
        name = f"neuraxon_hybrid_cunxon_external_drive_{mode}_{driven_port_class}"
        _check(lib, lib.cunxonNetworkCreate(ctx, C.byref(net), name.encode("utf-8")))
        sphere_id = _add_interface_probe_sphere(
            lib,
            net,
            sphere_name=b"EXTDRV",
            sensory_ids=sensory_port_ids,
            readout_ids=readout_port_ids,
        )
        _check(lib, lib.cunxonNetworkFinalize(net))
        input_buffer = (C.c_float * len(input_vector))(*input_vector)
        input_pointer = C.cast(input_buffer, C.POINTER(C.c_float))
        ext_inputs = (C.POINTER(C.c_float) * 1)(input_pointer)
        step_function = (
            lib.cunxonNetworkStepTrain if mode == "train" else lib.cunxonNetworkStepInfer
        )
        for _ in range(steps):
            _check(lib, step_function(net, ext_inputs, C.c_float(1.0)))
        _check(lib, lib.cunxonContextSync(ctx))
        readout = _capture_readout(lib, net, sphere_id)
        states = _capture_snapshot_states(lib, net, sphere_id)
        snapshot_slice = [states[index] for index in readout_port_ids]
        return CunxonExternalDriveWindowSample(
            mode=mode,
            driven_port_class=driven_port_class,
            sensory_port_ids=list(sensory_port_ids),
            readout_port_ids=list(readout_port_ids),
            readout=readout,
            snapshot_slice=snapshot_slice,
            matches_snapshot_slice=readout == snapshot_slice,
            active_state_count=sum(1 for value in readout if value != 0),
            signed_sum=sum(readout),
            energy=_capture_energy(lib, net),
        )
    finally:
        if net.value:
            lib.cunxonNetworkDestroy(net)


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
    lib.cunxonNetworkReset.argtypes = [C.c_void_p]
    lib.cunxonNetworkReset.restype = C.c_int
    lib.cunxonNetworkAigarthConfig.argtypes = [C.c_void_p, C.c_int, C.c_float, C.c_float, C.c_float]
    lib.cunxonNetworkAigarthConfig.restype = C.c_int
    lib.cunxonNetworkAigarthStep.argtypes = [C.c_void_p, _CUNXON_FITNESS_FN, C.c_void_p]
    lib.cunxonNetworkAigarthStep.restype = C.c_int
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
