"""JSON CLI interface for neuraxon-agent, compatible with Hermes tool-calling patterns."""
from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, cast

from neuraxon_agent.cunxon_smoke import (
    CunxonAigarthActionRemapAuditResult,
    CunxonVramResidentResult,
    ensure_no_active_vram_resident_run,
    run_ctypes_action_probe,
    run_ctypes_aigarth_action_contract_penalty_probe,
    run_ctypes_aigarth_action_hard_holdout_probe,
    run_ctypes_aigarth_action_probe,
    run_ctypes_aigarth_action_seed_sweep_probe,
    run_ctypes_aigarth_action_strict_label_probe,
    run_ctypes_aigarth_action_target_contract_augmented_train_probe,
    run_ctypes_aigarth_action_target_contract_probe,
    run_ctypes_aigarth_action_target_contract_stress_probe,
    run_ctypes_aigarth_readout_probe,
    run_ctypes_avalanche_intervention_task_correlation_probe,
    run_ctypes_avalanche_window_probe,
    run_ctypes_branching_regime_scan_probe,
    run_ctypes_controlled_regime_calibration_probe,
    run_ctypes_external_drive_window_probe,
    run_ctypes_input_proxy_target_probe,
    run_ctypes_interface_semantics_probe,
    run_ctypes_long_horizon_probe,
    run_ctypes_long_sweep_probe,
    run_ctypes_multisphere_action_probe,
    run_ctypes_resident_action_probe,
    run_ctypes_sensitivity_probe,
    run_ctypes_smoke,
    run_ctypes_snapshot_pattern_probe,
    run_ctypes_supervised_motor_probe,
    run_ctypes_vram_resident_probe,
    write_action_probe_artifacts,
    write_aigarth_action_artifacts,
    write_aigarth_action_contract_penalty_artifacts,
    write_aigarth_action_hard_holdout_artifacts,
    write_aigarth_action_remap_audit_artifacts,
    write_aigarth_action_seed_sweep_artifacts,
    write_aigarth_action_strict_label_artifacts,
    write_aigarth_action_target_contract_artifacts,
    write_aigarth_action_target_contract_augmented_train_artifacts,
    write_aigarth_action_target_contract_stress_artifacts,
    write_aigarth_readout_artifacts,
    write_avalanche_intervention_task_correlation_artifacts,
    write_avalanche_window_artifacts,
    write_branching_regime_scan_artifacts,
    write_controlled_regime_calibration_artifacts,
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
from neuraxon_agent.evolution import AgentEvolution, EvolutionConfig
from neuraxon_agent.tissue import AgentTissue, TissueState
from neuraxon_agent.vendor.neuraxon2 import NetworkParameters


def _load_json(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _save_json(path: str, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def _parse_seed_offsets(raw: str) -> list[int]:
    offsets = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not offsets:
        raise ValueError("--seed-offsets must contain at least one integer")
    return offsets


def _parse_step_horizons(raw: str) -> list[int]:
    horizons = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not horizons:
        raise ValueError("--step-horizons must contain at least one integer")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("--step-horizons must contain only positive integers")
    return horizons


def _parse_modes(raw: str) -> list[str]:
    modes = [part.strip() for part in raw.split(",") if part.strip()]
    if not modes:
        raise ValueError("--modes must contain at least one mode")
    return modes


def _encode_state(tissue: AgentTissue) -> str:
    """Serialize tissue state to base64-encoded JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        tissue.save(tmp)
        raw = Path(tmp).read_text()
        return base64.b64encode(raw.encode()).decode()
    finally:
        Path(tmp).unlink(missing_ok=True)


def _decode_state(b64: str, params: NetworkParameters | None = None) -> AgentTissue:
    """Deserialize tissue state from base64-encoded JSON."""
    raw = base64.b64decode(b64).decode()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        tissue = AgentTissue.load(tmp)
        return tissue
    finally:
        Path(tmp).unlink(missing_ok=True)


def _tissue_state_to_dict(state: TissueState) -> dict[str, Any]:
    return {
        "energy": state.energy,
        "activity": state.activity,
        "step_count": state.step_count,
        "dopamine": state.dopamine,
        "serotonin": state.serotonin,
        "acetylcholine": state.acetylcholine,
        "norepinephrine": state.norepinephrine,
        "num_neurons": state.num_neurons,
        "num_synapses": state.num_synapses,
    }


def cmd_think(args: argparse.Namespace) -> int:
    try:
        data = _load_json(args.input)
        params = NetworkParameters(**data.get("params", {}))
        tissue = AgentTissue(params)
        if "tissue_state" in data and data["tissue_state"]:
            tissue = _decode_state(data["tissue_state"], params)
        tissue.observe(data.get("observation", {}))
        action = tissue.think(steps=args.steps)
        result = {
            "action": action.actie_type.upper(),
            "confidence": action.confidence,
            "raw_output": action.raw_output,
            "tissue_state": _encode_state(tissue),
            "state": _tissue_state_to_dict(tissue.state),
        }
        _save_json(args.output, result)
        return 0
    except Exception as e:
        _save_json(args.output, {"error": str(e), "action": "ERROR", "confidence": 0.0})
        return 1


def cmd_modulate(args: argparse.Namespace) -> int:
    try:
        data = _load_json(args.input)
        params = NetworkParameters(**data.get("params", {}))
        tissue = AgentTissue(params)
        if "tissue_state" in data and data["tissue_state"]:
            tissue = _decode_state(data["tissue_state"], params)
        tissue.modulate(data.get("outcome", "partial"))
        result = {
            "tissue_state": _encode_state(tissue),
            "state": _tissue_state_to_dict(tissue.state),
        }
        _save_json(args.output, result)
        return 0
    except Exception as e:
        _save_json(args.output, {"error": str(e)})
        return 1


def cmd_evolve(args: argparse.Namespace) -> int:
    try:
        taskset = _load_json(args.taskset) if args.taskset else {}
        config = EvolutionConfig(
            seasons=args.generations,
            episodes_per_season=args.episodes or 10,
            seed=args.seed,
            task_scenarios=taskset.get("scenarios", []),
        )
        evo = AgentEvolution(config=config)
        summary = evo.evolve()
        _save_json(args.output, {"summary": summary})
        return 0
    except Exception as e:
        _save_json(args.output, {"error": str(e)})
        return 1


def cmd_save(args: argparse.Namespace) -> int:
    try:
        data = _load_json(args.input)
        params = NetworkParameters(**data.get("params", {}))
        tissue = AgentTissue(params)
        if "tissue_state" in data and data["tissue_state"]:
            tissue = _decode_state(data["tissue_state"], params)
        tissue.save(args.path)
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


def cmd_load(args: argparse.Namespace) -> int:
    try:
        tissue = AgentTissue.load(args.path)
        result = {
            "tissue_state": _encode_state(tissue),
            "state": _tissue_state_to_dict(tissue.state),
        }
        _save_json(args.output, result)
        return 0
    except Exception as e:
        _save_json(args.output, {"error": str(e)})
        return 1


def cmd_cunxon_smoke(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_smoke(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            steps=args.steps,
            device_id=args.device,
        )
        write_smoke_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon GPU smoke report\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "No broad Neuraxon intelligence claim: failed smoke tests do not support "
            "any GPU-backed decision-quality claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_long_horizon(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_long_horizon_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            total_steps=args.steps,
            sample_interval=args.sample_interval,
            device_id=args.device,
        )
        write_long_horizon_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon long-horizon learning probe\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Long-horizon learning caveat: short smoke tests are insufficient for "
            "judging a brain-like Neuraxon system, but a failed probe also does not "
            "support any GPU-backed learning claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_vram_resident(args: argparse.Namespace) -> int:
    try:
        ensure_no_active_vram_resident_run(args.state_output)
        command = "neuraxon-agent " + " ".join(sys.argv[1:])

        def write_progress(result: CunxonVramResidentResult) -> None:
            write_vram_resident_artifacts(
                result,
                json_path=args.json_output,
                markdown_path=args.markdown_output,
                state_path=args.state_output,
            )

        result = run_ctypes_vram_resident_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            hypothesis=args.hypothesis,
            active_issue=args.active_issue,
            command=command,
            max_runtime_seconds=args.max_runtime_seconds,
            sample_interval_seconds=args.sample_interval_seconds,
            steps_per_sample=args.steps_per_sample,
            device_id=args.device,
            artifact_callback=write_progress,
        )
        write_vram_resident_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
            state_path=args.state_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon VRAM-resident dynamics run\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "This failed or blocked VRAM-resident run does not support any GPU-backed "
            "learning or intelligence claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_long_sweep(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_long_sweep_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            step_horizons=_parse_step_horizons(args.step_horizons),
            seed_offsets=_parse_seed_offsets(args.seed_offsets),
            modes=_parse_modes(args.modes),
            device_id=args.device,
        )
        write_long_sweep_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon long sweep action diagnostic\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed long-sweep diagnostic does not support any "
            "GPU-backed learning or decision-quality claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_action_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_action_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            trial_steps=args.trial_steps,
            device_id=args.device,
        )
        write_action_probe_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon task-coupled action probe\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Decision-quality boundary: a failed task-coupled probe does not support "
            "any GPU-backed action-quality or learning claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_sensitivity_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_sensitivity_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            steps=args.steps,
            seed_offsets=_parse_seed_offsets(args.seed_offsets),
            device_id=args.device,
        )
        write_sensitivity_probe_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon infer-vs-train sensitivity probe\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed sensitivity probe does not support any "
            "GPU-backed learning or decision-quality claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_snapshot_pattern_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_snapshot_pattern_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            present_steps=args.present_steps,
            settle_steps=args.settle_steps,
            mask_fraction=args.mask_fraction,
            device_id=args.device,
        )
        write_snapshot_pattern_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon hidden-state/snapshot + pattern store/recall probe\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed snapshot/pattern probe does not support any "
            "GPU-backed hidden-state, recall, learning, or decision-quality claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_multisphere_action_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_multisphere_action_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            train_steps=args.train_steps,
            eval_steps=args.eval_steps,
            device_id=args.device,
        )
        write_multisphere_action_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon multi-sphere/action adapter probe\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed multi-sphere/action probe does not support any "
            "GPU-backed action-quality, holdout, or learning claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_aigarth_readout_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_aigarth_readout_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            generations=args.generations,
            population_size=args.population_size,
            eval_steps=args.eval_steps,
            device_id=args.device,
        )
        write_aigarth_readout_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon Aigarth readout semantics probe\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed Aigarth/readout probe does not support any "
            "GPU-backed learning, output-readout, or action-quality claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_aigarth_action_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_aigarth_action_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            generations=args.generations,
            population_size=args.population_size,
            eval_steps=args.eval_steps,
            device_id=args.device,
        )
        write_aigarth_action_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon Aigarth holdout action probe\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed Aigarth/action probe does not support any "
            "GPU-backed learning, holdout, or action-quality claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_aigarth_action_seed_sweep_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_aigarth_action_seed_sweep_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            seed_offsets=_parse_seed_offsets(args.seed_offsets),
            generations=args.generations,
            population_size=args.population_size,
            eval_steps=args.eval_steps,
            device_id=args.device,
        )
        write_aigarth_action_seed_sweep_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon Aigarth action seed sweep\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed Aigarth/action seed sweep does not support any "
            "GPU-backed repeatability, holdout, or action-quality claim.\n",
            encoding="utf-8",
        )
        return 1



def cmd_cunxon_aigarth_action_hard_holdout_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_aigarth_action_hard_holdout_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            seed_offsets=_parse_seed_offsets(args.seed_offsets),
            generations=args.generations,
            population_size=args.population_size,
            eval_steps=args.eval_steps,
            device_id=args.device,
        )
        write_aigarth_action_hard_holdout_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon Aigarth action hard-holdout audit\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed Aigarth/action hard-holdout audit does not "
            "support any GPU-backed repeatability, holdout, leakage, or action-quality claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_aigarth_action_strict_label_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_aigarth_action_strict_label_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            seed_offsets=_parse_seed_offsets(args.seed_offsets),
            generations=args.generations,
            population_size=args.population_size,
            eval_steps=args.eval_steps,
            fitness_variant="strict_label_margin",
            device_id=args.device,
        )
        write_aigarth_action_strict_label_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon Aigarth strict-label action audit\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed Aigarth/action strict-label audit does not "
            "support any GPU-backed label-contract, holdout, or action-quality claim.\n",
            encoding="utf-8",
        )
        return 1



def cmd_cunxon_aigarth_action_contract_penalty_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_aigarth_action_contract_penalty_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            seed_offsets=_parse_seed_offsets(args.seed_offsets),
            generations=args.generations,
            population_size=args.population_size,
            eval_steps=args.eval_steps,
            fitness_variant="strict_label_heavy_penalty",
            device_id=args.device,
        )
        write_aigarth_action_contract_penalty_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon Aigarth contract-penalty action audit\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed Aigarth/action contract-penalty audit does not "
            "support any GPU-backed label-contract, holdout, or action-quality claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_aigarth_action_target_contract_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_aigarth_action_target_contract_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            seed_offsets=_parse_seed_offsets(args.seed_offsets),
            generations=args.generations,
            population_size=args.population_size,
            eval_steps=args.eval_steps,
            fitness_variant="target_contract_margin",
            device_id=args.device,
        )
        write_aigarth_action_target_contract_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon Aigarth target-contract action audit\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed Aigarth/action target-contract audit does not "
            "support any GPU-backed label-contract, holdout, or action-quality claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_aigarth_action_target_contract_stress_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_aigarth_action_target_contract_stress_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            seed_offsets=_parse_seed_offsets(args.seed_offsets),
            generations=args.generations,
            population_size=args.population_size,
            eval_steps=args.eval_steps,
            fitness_variant="target_contract_margin",
            device_id=args.device,
        )
        write_aigarth_action_target_contract_stress_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon Aigarth target-contract stress audit\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed Aigarth/action target-contract stress audit does not "
            "support any GPU-backed label-contract, holdout, or action-quality claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_aigarth_action_target_contract_augmented_train_probe(
    args: argparse.Namespace,
) -> int:
    try:
        result = run_ctypes_aigarth_action_target_contract_augmented_train_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            seed_offsets=_parse_seed_offsets(args.seed_offsets),
            generations=args.generations,
            population_size=args.population_size,
            eval_steps=args.eval_steps,
            fitness_variant="target_contract_augmented_train",
            device_id=args.device,
        )
        write_aigarth_action_target_contract_augmented_train_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon Aigarth target-contract augmented-train audit\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed Aigarth/action target-contract augmented-train "
            "audit does not support any GPU-backed label-contract, holdout, or "
            "action-quality claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_branching_regime_scan(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_branching_regime_scan_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            seed_offsets=_parse_seed_offsets(args.seed_offsets),
            generations=args.generations,
            population_size=args.population_size,
            eval_steps=args.eval_steps,
            source_probe=args.source_probe,
            device_id=args.device,
        )
        write_branching_regime_scan_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon branching-ratio regime scan\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed branching-regime scan does not support any "
            "GPU-backed criticality, action-quality, or intelligence claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_avalanche_window_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_avalanche_window_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            modes=_parse_modes(args.modes),
            seed_offsets=_parse_seed_offsets(args.seed_offsets),
            steps=args.steps,
            sample_interval=args.sample_interval,
            device_id=args.device,
        )
        write_avalanche_window_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon avalanche-window snapshot probe\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed snapshot/avalanche probe does not support any "
            "GPU-backed criticality, action-quality, or intelligence claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_avalanche_intervention_task_correlation(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_avalanche_intervention_task_correlation_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            modes=_parse_modes(args.modes),
            seed_offsets=_parse_seed_offsets(args.seed_offsets),
            device_id=args.device,
        )
        write_avalanche_intervention_task_correlation_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon avalanche intervention/task correlation\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed task-coupled avalanche probe does not support any "
            "GPU-backed criticality, action-quality, or intelligence claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_controlled_regime_calibration(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_controlled_regime_calibration_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            modes=_parse_modes(args.modes),
            seed_offsets=_parse_seed_offsets(args.seed_offsets),
            steps=args.steps,
            sample_interval=args.sample_interval,
            device_id=args.device,
        )
        write_controlled_regime_calibration_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon controlled-regime criticality calibration\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed controlled-regime calibration does not support any "
            "GPU-backed criticality, action-quality, or intelligence claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_aigarth_action_remap_audit(args: argparse.Namespace) -> int:
    try:
        source_artifacts = [
            part.strip() for part in args.source_artifacts.split(",") if part.strip()
        ]
        result = CunxonAigarthActionRemapAuditResult.from_artifact_paths(source_artifacts)
        write_aigarth_action_remap_audit_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon Aigarth action remap audit\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed remap audit does not support any GPU-backed "
            "decoder-contract, holdout, action-quality, or learning claim.\n",
            encoding="utf-8",
        )
        return 1



def cmd_cunxon_resident_action_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_resident_action_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            train_epochs=args.train_epochs,
            train_steps_per_case=args.train_steps_per_case,
            eval_steps=args.eval_steps,
            device_id=args.device,
        )
        write_resident_action_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon resident task-coupled action probe\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed resident task probe does not support any "
            "GPU-backed action-quality, holdout, or learning claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_interface_semantics_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_interface_semantics_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            steps=args.steps,
            device_id=args.device,
        )
        write_interface_semantics_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon interface semantics probe\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed interface semantics probe does not support any "
            "GPU-backed learning, action-quality, or interface-semantics claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_external_drive_window_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_external_drive_window_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            steps=args.steps,
            device_id=args.device,
        )
        write_external_drive_window_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon external-drive window probe\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed port-window probe does not support any GPU-backed "
            "learning, desired-output, action-quality, or interface-semantics claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_supervised_motor_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_supervised_motor_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            train_epochs=args.train_epochs,
            train_steps_per_case=args.train_steps_per_case,
            eval_steps=args.eval_steps,
            device_id=args.device,
        )
        write_supervised_motor_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon supervised motor-target probe\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed supervised motor-target probe does not support "
            "any GPU-backed learning, holdout, or action-quality claim.\n",
            encoding="utf-8",
        )
        return 1


def cmd_cunxon_input_proxy_target_probe(args: argparse.Namespace) -> int:
    try:
        result = run_ctypes_input_proxy_target_probe(
            library_path=args.library,
            upstream_commit=args.upstream_commit,
            cunxon_commit=args.cunxon_commit,
            train_epochs=args.train_epochs,
            train_steps_per_case=args.train_steps_per_case,
            eval_steps=args.eval_steps,
            device_id=args.device,
        )
        write_input_proxy_target_artifacts(
            result,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        return 0
    except Exception as e:
        _save_json(args.json_output, {"error": str(e), "status": "unusable"})
        Path(args.markdown_output).write_text(
            "# cuNxon input-port proxy target probe\n\n"
            "Status: `unusable`\n\n"
            f"Error: {e}\n\n"
            "Evidence boundary: a failed input-proxy target probe does not support "
            "any GPU-backed learning, holdout, or action-quality claim.\n",
            encoding="utf-8",
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neuraxon-agent", description="Neuraxon Agent CLI")
    sub = parser.add_subparsers(dest="command")

    p_think = sub.add_parser("think", help="Observe and think")
    p_think.add_argument("--input", "-i", required=True, help="Observation JSON file")
    p_think.add_argument("--output", "-o", required=True, help="Action JSON file")
    p_think.add_argument("--steps", type=int, default=10, help="Simulation steps")
    p_think.set_defaults(func=cmd_think)

    p_mod = sub.add_parser("modulate", help="Apply neuromodulator feedback")
    p_mod.add_argument("--input", "-i", required=True, help="Outcome JSON file")
    p_mod.add_argument("--output", "-o", required=True, help="Result JSON file")
    p_mod.set_defaults(func=cmd_modulate)

    p_evo = sub.add_parser("evolve", help="Evolve agent networks")
    p_evo.add_argument("--taskset", "-t", help="Taskset JSON file")
    p_evo.add_argument("--generations", "-g", type=int, default=5, help="Generations")
    p_evo.add_argument("--episodes", "-e", type=int, default=10, help="Episodes per generation")
    p_evo.add_argument("--seed", type=int, default=None, help="Random seed")
    p_evo.add_argument("--output", "-o", required=True, help="Summary JSON file")
    p_evo.set_defaults(func=cmd_evolve)

    p_save = sub.add_parser("save", help="Save tissue state")
    p_save.add_argument("--input", "-i", required=True, help="State JSON file")
    p_save.add_argument("--path", "-p", required=True, help="Output path")
    p_save.set_defaults(func=cmd_save)

    p_load = sub.add_parser("load", help="Load tissue state")
    p_load.add_argument("--path", "-p", required=True, help="Input path")
    p_load.add_argument("--output", "-o", required=True, help="Result JSON file")
    p_load.set_defaults(func=cmd_load)

    p_cunxon = sub.add_parser("cunxon-smoke", help="Run optional cuNxon CUDA ctypes smoke")
    p_cunxon.add_argument("--library", required=True, help="Path to built libcunxon.so")
    p_cunxon.add_argument("--upstream-commit", required=True, help="Upstream Neuraxon commit")
    p_cunxon.add_argument("--cunxon-commit", required=True, help="cuNxon source commit")
    p_cunxon.add_argument("--steps", type=int, default=16, help="Smoke simulation steps")
    p_cunxon.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_smoke.json",
        help="JSON artifact path",
    )
    p_cunxon.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_smoke.md",
        help="Markdown artifact path",
    )
    p_cunxon.set_defaults(func=cmd_cunxon_smoke)

    p_cunxon_long = sub.add_parser(
        "cunxon-long-horizon",
        help="Run optional cuNxon CUDA long-horizon learning probe",
        description=(
            "Run one cuNxon network continuously for a longer horizon. Short smoke "
            "tests are insufficient for judging whether brain-like Neuraxon dynamics learn."
        ),
    )
    p_cunxon_long.add_argument("--library", required=True, help="Path to built libcunxon.so")
    p_cunxon_long.add_argument("--upstream-commit", required=True, help="Upstream Neuraxon commit")
    p_cunxon_long.add_argument("--cunxon-commit", required=True, help="cuNxon source commit")
    p_cunxon_long.add_argument(
        "--steps",
        type=int,
        default=4096,
        help="Continuous simulation steps",
    )
    p_cunxon_long.add_argument(
        "--sample-interval",
        type=int,
        default=512,
        help="Steps between samples",
    )
    p_cunxon_long.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_long.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_long_horizon.json",
        help="JSON artifact path",
    )
    p_cunxon_long.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_long_horizon.md",
        help="Markdown artifact path",
    )
    p_cunxon_long.set_defaults(func=cmd_cunxon_long_horizon)

    p_cunxon_vram = sub.add_parser(
        "cunxon-vram-resident",
        help="Run one cuNxon process/network resident in VRAM with progress artifacts",
        description=(
            "Keep the same cuNxon context/network alive across wall-clock time and write "
            "a durable state file so follow-up cron runs can poll instead of starting duplicates."
        ),
    )
    p_cunxon_vram.add_argument("--library", required=True, help="Path to built libcunxon.so")
    p_cunxon_vram.add_argument("--upstream-commit", required=True, help="Upstream Neuraxon commit")
    p_cunxon_vram.add_argument("--cunxon-commit", required=True, help="cuNxon source commit")
    p_cunxon_vram.add_argument(
        "--hypothesis",
        required=True,
        help="Bounded hypothesis for this resident runtime run",
    )
    p_cunxon_vram.add_argument(
        "--active-issue",
        default="https://github.com/sisutuulenisa/neuraxon-hybrid/issues/79",
        help="Issue URL this long run is attached to",
    )
    p_cunxon_vram.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=14400,
        help="Maximum wall-clock runtime before clean exit",
    )
    p_cunxon_vram.add_argument(
        "--sample-interval-seconds",
        type=int,
        default=900,
        help="Wall-clock seconds between progress samples",
    )
    p_cunxon_vram.add_argument(
        "--steps-per-sample",
        type=int,
        default=262144,
        help="cuNxon infer steps before each progress sample",
    )
    p_cunxon_vram.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_vram.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_vram_resident_run.json",
        help="Progress JSON artifact path",
    )
    p_cunxon_vram.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_vram_resident_run.md",
        help="Progress Markdown artifact path",
    )
    p_cunxon_vram.add_argument(
        "--state-output",
        default="benchmarks/results/cunxon_vram_resident_run_state.json",
        help="Durable anti-duplicate state path",
    )
    p_cunxon_vram.set_defaults(func=cmd_cunxon_vram_resident)

    p_cunxon_long_sweep = sub.add_parser(
        "cunxon-long-sweep",
        help="Sweep longer cuNxon modes/horizons/seeds against the action contract",
        description=(
            "Run fresh cuNxon samples across infer/train/train_rewarded modes, longer step "
            "horizons, and seed offsets. Scores decoded actions against trivial baselines."
        ),
    )
    p_cunxon_long_sweep.add_argument("--library", required=True, help="Path to built libcunxon.so")
    p_cunxon_long_sweep.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_long_sweep.add_argument("--cunxon-commit", required=True, help="cuNxon source commit")
    p_cunxon_long_sweep.add_argument(
        "--step-horizons",
        default="32,512,4096",
        help="Comma-separated simulation-step horizons per sample",
    )
    p_cunxon_long_sweep.add_argument(
        "--seed-offsets",
        default="79,80,81",
        help="Comma-separated cuNxon random_seed_offset values",
    )
    p_cunxon_long_sweep.add_argument(
        "--modes",
        default="infer,train,train_rewarded",
        help="Comma-separated modes: infer, train, train_rewarded",
    )
    p_cunxon_long_sweep.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_long_sweep.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_long_sweep.json",
        help="JSON artifact path",
    )
    p_cunxon_long_sweep.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_long_sweep.md",
        help="Markdown artifact path",
    )
    p_cunxon_long_sweep.set_defaults(func=cmd_cunxon_long_sweep)

    p_cunxon_action = sub.add_parser(
        "cunxon-action-probe",
        help="Run optional cuNxon CUDA task-coupled action probe",
        description=(
            "Run a tiny cuNxon task suite, decode trinary readouts with the existing "
            "ActionDecoder, and score them against expected benchmark actions."
        ),
    )
    p_cunxon_action.add_argument("--library", required=True, help="Path to built libcunxon.so")
    p_cunxon_action.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_action.add_argument("--cunxon-commit", required=True, help="cuNxon source commit")
    p_cunxon_action.add_argument(
        "--trial-steps",
        type=int,
        default=32,
        help="Simulation steps per task trial",
    )
    p_cunxon_action.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_action.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_action_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_action.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_action_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_action.set_defaults(func=cmd_cunxon_action_probe)

    p_cunxon_aigarth = sub.add_parser(
        "cunxon-aigarth-readout-probe",
        help="Contrast cuNxon Aigarth relative-demo and absolute-output readouts",
        description=(
            "Run the public Aigarth-style evolutionary loop twice: once with the upstream "
            "demo's relative readout ids 0..7 and once with absolute output ids 36..43."
        ),
    )
    p_cunxon_aigarth.add_argument("--library", required=True, help="Path to built libcunxon.so")
    p_cunxon_aigarth.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_aigarth.add_argument("--cunxon-commit", required=True, help="cuNxon source commit")
    p_cunxon_aigarth.add_argument(
        "--generations",
        type=int,
        default=12,
        help="Aigarth generations per readout mapping",
    )
    p_cunxon_aigarth.add_argument(
        "--population-size",
        type=int,
        default=24,
        help="Aigarth mutation population size per generation",
    )
    p_cunxon_aigarth.add_argument(
        "--eval-steps",
        type=int,
        default=25,
        help="Inference steps per positive/negative class evaluation",
    )
    p_cunxon_aigarth.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_aigarth.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_aigarth_readout_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_aigarth.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_aigarth_readout_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_aigarth.set_defaults(func=cmd_cunxon_aigarth_readout_probe)

    p_cunxon_aigarth_action = sub.add_parser(
        "cunxon-aigarth-action-probe",
        help="Run Aigarth evolution and score absolute output readouts against action holdouts",
        description=(
            "Evolve one cuNxon sphere through the public Aigarth fitness API using train "
            "cases only, then decode train/holdout readouts through the existing action contract."
        ),
    )
    p_cunxon_aigarth_action.add_argument(
        "--library", required=True, help="Path to built libcunxon.so"
    )
    p_cunxon_aigarth_action.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_aigarth_action.add_argument(
        "--cunxon-commit", required=True, help="cuNxon source commit"
    )
    p_cunxon_aigarth_action.add_argument(
        "--generations",
        type=int,
        default=16,
        help="Aigarth generations using train-only action fitness",
    )
    p_cunxon_aigarth_action.add_argument(
        "--population-size",
        type=int,
        default=32,
        help="Aigarth mutation population size per generation",
    )
    p_cunxon_aigarth_action.add_argument(
        "--eval-steps",
        type=int,
        default=24,
        help="Inference steps per train/holdout case evaluation",
    )
    p_cunxon_aigarth_action.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_aigarth_action.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_aigarth_action_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_aigarth_action.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_aigarth_action_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_aigarth_action.set_defaults(func=cmd_cunxon_aigarth_action_probe)

    p_cunxon_aigarth_action_sweep = sub.add_parser(
        "cunxon-aigarth-action-seed-sweep-probe",
        help="Run Aigarth action probe across fresh cuNxon seed offsets",
        description=(
            "Repeat the train-only Aigarth action probe with a fresh cuNxon "
            "network/context per seed to audit repeatability and action coverage."
        ),
    )
    p_cunxon_aigarth_action_sweep.add_argument(
        "--library", required=True, help="Path to built libcunxon.so"
    )
    p_cunxon_aigarth_action_sweep.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_aigarth_action_sweep.add_argument(
        "--cunxon-commit", required=True, help="cuNxon source commit"
    )
    p_cunxon_aigarth_action_sweep.add_argument(
        "--seed-offsets",
        default="82,83,84,85,86",
        help="Comma-separated cuNxon random_seed_offset values",
    )
    p_cunxon_aigarth_action_sweep.add_argument(
        "--generations",
        type=int,
        default=16,
        help="Aigarth generations per seed using train-only action fitness",
    )
    p_cunxon_aigarth_action_sweep.add_argument(
        "--population-size",
        type=int,
        default=32,
        help="Aigarth mutation population size per generation",
    )
    p_cunxon_aigarth_action_sweep.add_argument(
        "--eval-steps",
        type=int,
        default=24,
        help="Inference steps per train/holdout case evaluation",
    )
    p_cunxon_aigarth_action_sweep.add_argument(
        "--device", type=int, default=0, help="CUDA device id"
    )
    p_cunxon_aigarth_action_sweep.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_aigarth_action_seed_sweep.json",
        help="JSON artifact path",
    )
    p_cunxon_aigarth_action_sweep.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_aigarth_action_seed_sweep.md",
        help="Markdown artifact path",
    )
    p_cunxon_aigarth_action_sweep.set_defaults(
        func=cmd_cunxon_aigarth_action_seed_sweep_probe
    )

    p_cunxon_aigarth_hard = sub.add_parser(
        "cunxon-aigarth-action-hard-holdout-probe",
        help="Audit Aigarth action route with hard holdouts and leakage controls",
        description=(
            "Repeat the train-only Aigarth action probe across fresh cuNxon seeds, "
            "then score normal holdout, harder/noisier holdout, and permuted-control "
            "cases to separate adapter evidence from oracle/leakage concerns."
        ),
    )
    p_cunxon_aigarth_hard.add_argument(
        "--library", required=True, help="Path to built libcunxon.so"
    )
    p_cunxon_aigarth_hard.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_aigarth_hard.add_argument(
        "--cunxon-commit", required=True, help="cuNxon source commit"
    )
    p_cunxon_aigarth_hard.add_argument(
        "--seed-offsets",
        default="87,88,89,90,91",
        help="Comma-separated cuNxon random_seed_offset values",
    )
    p_cunxon_aigarth_hard.add_argument(
        "--generations",
        type=int,
        default=16,
        help="Aigarth generations per seed using train-only action fitness",
    )
    p_cunxon_aigarth_hard.add_argument(
        "--population-size",
        type=int,
        default=32,
        help="Aigarth mutation population size per generation",
    )
    p_cunxon_aigarth_hard.add_argument(
        "--eval-steps",
        type=int,
        default=24,
        help="Inference steps per train/holdout/control case evaluation",
    )
    p_cunxon_aigarth_hard.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_aigarth_hard.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_aigarth_action_hard_holdout_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_aigarth_hard.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_aigarth_action_hard_holdout_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_aigarth_hard.set_defaults(
        func=cmd_cunxon_aigarth_action_hard_holdout_probe
    )

    p_cunxon_aigarth_strict = sub.add_parser(
        "cunxon-aigarth-action-strict-label-probe",
        help="Audit Aigarth action fitness with strict label penalties",
        description=(
            "Repeat the hard-holdout Aigarth action audit with a strict-label fitness "
            "variant that rewards expected execute/query/retry labels and penalizes "
            "out-of-contract normalized labels such as assertive."
        ),
    )
    p_cunxon_aigarth_strict.add_argument(
        "--library", required=True, help="Path to built libcunxon.so"
    )
    p_cunxon_aigarth_strict.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_aigarth_strict.add_argument(
        "--cunxon-commit", required=True, help="cuNxon source commit"
    )
    p_cunxon_aigarth_strict.add_argument(
        "--seed-offsets",
        default="92,93,94,95,96",
        help="Comma-separated cuNxon random_seed_offset values",
    )
    p_cunxon_aigarth_strict.add_argument(
        "--generations",
        type=int,
        default=16,
        help="Aigarth generations per seed using strict-label train-only fitness",
    )
    p_cunxon_aigarth_strict.add_argument(
        "--population-size",
        type=int,
        default=32,
        help="Aigarth mutation population size per generation",
    )
    p_cunxon_aigarth_strict.add_argument(
        "--eval-steps",
        type=int,
        default=24,
        help="Inference steps per train/holdout/control case evaluation",
    )
    p_cunxon_aigarth_strict.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_aigarth_strict.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_aigarth_action_strict_label_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_aigarth_strict.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_aigarth_action_strict_label_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_aigarth_strict.set_defaults(
        func=cmd_cunxon_aigarth_action_strict_label_probe
    )

    p_cunxon_aigarth_contract_penalty = sub.add_parser(
        "cunxon-aigarth-action-contract-penalty-probe",
        help="Audit Aigarth action fitness with a heavier unexpected-label penalty",
        description=(
            "Repeat the strict-label Aigarth action audit with a heavier train-only "
            "penalty for normalized labels outside execute/query/retry before increasing "
            "task difficulty."
        ),
    )
    p_cunxon_aigarth_contract_penalty.add_argument(
        "--library", required=True, help="Path to built libcunxon.so"
    )
    p_cunxon_aigarth_contract_penalty.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_aigarth_contract_penalty.add_argument(
        "--cunxon-commit", required=True, help="cuNxon source commit"
    )
    p_cunxon_aigarth_contract_penalty.add_argument(
        "--seed-offsets",
        default="97,98,99,100,101",
        help="Comma-separated cuNxon random_seed_offset values",
    )
    p_cunxon_aigarth_contract_penalty.add_argument(
        "--generations",
        type=int,
        default=16,
        help="Aigarth generations per seed using heavy contract-penalty train-only fitness",
    )
    p_cunxon_aigarth_contract_penalty.add_argument(
        "--population-size",
        type=int,
        default=32,
        help="Aigarth mutation population size per generation",
    )
    p_cunxon_aigarth_contract_penalty.add_argument(
        "--eval-steps",
        type=int,
        default=24,
        help="Inference steps per train/holdout/control case evaluation",
    )
    p_cunxon_aigarth_contract_penalty.add_argument(
        "--device", type=int, default=0, help="CUDA device id"
    )
    p_cunxon_aigarth_contract_penalty.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_aigarth_action_contract_penalty_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_aigarth_contract_penalty.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_aigarth_action_contract_penalty_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_aigarth_contract_penalty.set_defaults(
        func=cmd_cunxon_aigarth_action_contract_penalty_probe
    )

    p_cunxon_aigarth_target_contract = sub.add_parser(
        "cunxon-aigarth-action-target-contract-probe",
        help="Audit Aigarth action fitness with signed-first-lane target-contract decoding",
        description=(
            "Repeat the hard-holdout Aigarth action audit with the project target-readout "
            "contract inside train-only fitness instead of only post-hoc remapping."
        ),
    )
    p_cunxon_aigarth_target_contract.add_argument(
        "--library", required=True, help="Path to built libcunxon.so"
    )
    p_cunxon_aigarth_target_contract.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_aigarth_target_contract.add_argument(
        "--cunxon-commit", required=True, help="cuNxon source commit"
    )
    p_cunxon_aigarth_target_contract.add_argument(
        "--seed-offsets",
        default="102,103,104,105,106",
        help="Comma-separated cuNxon random_seed_offset values",
    )
    p_cunxon_aigarth_target_contract.add_argument(
        "--generations",
        type=int,
        default=16,
        help="Aigarth generations per seed using target-contract train-only fitness",
    )
    p_cunxon_aigarth_target_contract.add_argument(
        "--population-size",
        type=int,
        default=32,
        help="Aigarth mutation population size per generation",
    )
    p_cunxon_aigarth_target_contract.add_argument(
        "--eval-steps",
        type=int,
        default=24,
        help="Inference steps per train/holdout/control case evaluation",
    )
    p_cunxon_aigarth_target_contract.add_argument(
        "--device", type=int, default=0, help="CUDA device id"
    )
    p_cunxon_aigarth_target_contract.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_aigarth_action_target_contract_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_aigarth_target_contract.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_aigarth_action_target_contract_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_aigarth_target_contract.set_defaults(
        func=cmd_cunxon_aigarth_action_target_contract_probe
    )

    p_cunxon_aigarth_target_contract_stress = sub.add_parser(
        "cunxon-aigarth-action-target-contract-stress-probe",
        help="Stress target-contract Aigarth action fitness with harder/control splits",
        description=(
            "Repeat the target-contract Aigarth action audit with additional low-margin "
            "stress holdouts and counterfactual controls while keeping fitness train-only."
        ),
    )
    p_cunxon_aigarth_target_contract_stress.add_argument(
        "--library", required=True, help="Path to built libcunxon.so"
    )
    p_cunxon_aigarth_target_contract_stress.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_aigarth_target_contract_stress.add_argument(
        "--cunxon-commit", required=True, help="cuNxon source commit"
    )
    p_cunxon_aigarth_target_contract_stress.add_argument(
        "--seed-offsets",
        default="107,108,109,110,111",
        help="Comma-separated cuNxon random_seed_offset values",
    )
    p_cunxon_aigarth_target_contract_stress.add_argument(
        "--generations",
        type=int,
        default=16,
        help="Aigarth generations per seed using target-contract train-only fitness",
    )
    p_cunxon_aigarth_target_contract_stress.add_argument(
        "--population-size",
        type=int,
        default=32,
        help="Aigarth mutation population size per generation",
    )
    p_cunxon_aigarth_target_contract_stress.add_argument(
        "--eval-steps",
        type=int,
        default=24,
        help="Inference steps per train/holdout/control case evaluation",
    )
    p_cunxon_aigarth_target_contract_stress.add_argument(
        "--device", type=int, default=0, help="CUDA device id"
    )
    p_cunxon_aigarth_target_contract_stress.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_aigarth_action_target_contract_stress_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_aigarth_target_contract_stress.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_aigarth_action_target_contract_stress_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_aigarth_target_contract_stress.set_defaults(
        func=cmd_cunxon_aigarth_action_target_contract_stress_probe
    )

    p_cunxon_aigarth_target_contract_augmented = sub.add_parser(
        "cunxon-aigarth-action-target-contract-augmented-train-probe",
        help="Stress target-contract Aigarth fitness after low-margin train augmentation",
        description=(
            "Repeat the target-contract stress audit with additional low-margin "
            "augmented_train cases inside the fitness callback while keeping stress "
            "holdout and controls outside fitness."
        ),
    )
    p_cunxon_aigarth_target_contract_augmented.add_argument(
        "--library", required=True, help="Path to built libcunxon.so"
    )
    p_cunxon_aigarth_target_contract_augmented.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_aigarth_target_contract_augmented.add_argument(
        "--cunxon-commit", required=True, help="cuNxon source commit"
    )
    p_cunxon_aigarth_target_contract_augmented.add_argument(
        "--seed-offsets",
        default="112,113,114,115,116",
        help="Comma-separated cuNxon random_seed_offset values",
    )
    p_cunxon_aigarth_target_contract_augmented.add_argument(
        "--generations",
        type=int,
        default=16,
        help="Aigarth generations per seed using augmented train-only fitness",
    )
    p_cunxon_aigarth_target_contract_augmented.add_argument(
        "--population-size",
        type=int,
        default=32,
        help="Aigarth mutation population size per generation",
    )
    p_cunxon_aigarth_target_contract_augmented.add_argument(
        "--eval-steps",
        type=int,
        default=24,
        help="Inference steps per train/holdout/control case evaluation",
    )
    p_cunxon_aigarth_target_contract_augmented.add_argument(
        "--device", type=int, default=0, help="CUDA device id"
    )
    p_cunxon_aigarth_target_contract_augmented.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_aigarth_action_target_contract_augmented_train_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_aigarth_target_contract_augmented.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_aigarth_action_target_contract_augmented_train_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_aigarth_target_contract_augmented.set_defaults(
        func=cmd_cunxon_aigarth_action_target_contract_augmented_train_probe
    )

    p_cunxon_branching_regime = sub.add_parser(
        "cunxon-branching-regime-scan",
        help="Scan cuNxon branching/activity regimes beside action baselines",
        description=(
            "Run a bounded target-contract augmented-train Aigarth source probe, then "
            "derive branching/activity-ratio proxy metrics and compare regime buckets "
            "with holdout/stress_holdout action quality and constant baselines."
        ),
    )
    p_cunxon_branching_regime.add_argument(
        "--library", required=True, help="Path to built libcunxon.so"
    )
    p_cunxon_branching_regime.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_branching_regime.add_argument(
        "--cunxon-commit", required=True, help="cuNxon source commit"
    )
    p_cunxon_branching_regime.add_argument(
        "--seed-offsets",
        default="117,118,119,120,121",
        help="Comma-separated cuNxon random_seed_offset values",
    )
    p_cunxon_branching_regime.add_argument(
        "--generations",
        type=int,
        default=12,
        help="Aigarth generations per seed for the bounded source probe",
    )
    p_cunxon_branching_regime.add_argument(
        "--population-size",
        type=int,
        default=24,
        help="Aigarth mutation population size per generation",
    )
    p_cunxon_branching_regime.add_argument(
        "--eval-steps",
        type=int,
        default=24,
        help="Inference steps per action case evaluation",
    )
    p_cunxon_branching_regime.add_argument(
        "--source-probe",
        default="target_contract_augmented_train",
        help="Source action probe used before deriving regime metrics",
    )
    p_cunxon_branching_regime.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_branching_regime.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_branching_regime_scan.json",
        help="JSON artifact path",
    )
    p_cunxon_branching_regime.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_branching_regime_scan.md",
        help="Markdown artifact path",
    )
    p_cunxon_branching_regime.set_defaults(func=cmd_cunxon_branching_regime_scan)

    p_cunxon_avalanche = sub.add_parser(
        "cunxon-avalanche-window-probe",
        help="Capture full-sphere snapshot windows for avalanche/branching diagnostics",
        description=(
            "Run bounded infer/train cuNxon snapshot windows for action stimuli, estimate "
            "activation-event branching ratios from full-sphere state transitions, and compare "
            "the diagnostics with action-contract accuracy and trivial baselines."
        ),
    )
    p_cunxon_avalanche.add_argument(
        "--library", required=True, help="Path to built libcunxon.so"
    )
    p_cunxon_avalanche.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_avalanche.add_argument(
        "--cunxon-commit", required=True, help="cuNxon source commit"
    )
    p_cunxon_avalanche.add_argument(
        "--modes",
        default="infer,train",
        help="Comma-separated modes to sample: infer,train",
    )
    p_cunxon_avalanche.add_argument(
        "--seed-offsets",
        default="122,123,124",
        help="Comma-separated cuNxon random_seed_offset values",
    )
    p_cunxon_avalanche.add_argument(
        "--steps",
        type=int,
        default=256,
        help="Steps per stimulus window",
    )
    p_cunxon_avalanche.add_argument(
        "--sample-interval",
        type=int,
        default=16,
        help="Capture full-sphere snapshots every N steps",
    )
    p_cunxon_avalanche.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_avalanche.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_avalanche_window_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_avalanche.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_avalanche_window_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_avalanche.set_defaults(func=cmd_cunxon_avalanche_window_probe)

    p_cunxon_avalanche_correlation = sub.add_parser(
        "cunxon-avalanche-intervention-task-correlation",
        help="Couple cuNxon avalanche interventions to held-out/stress/control task quality",
        description=(
            "Run bounded cuNxon snapshot-window interventions over train, holdout, stress and "
            "control action splits, then compare avalanche/branching metrics with task quality "
            "and constant-action baselines."
        ),
    )
    p_cunxon_avalanche_correlation.add_argument(
        "--library", required=True, help="Path to built libcunxon.so"
    )
    p_cunxon_avalanche_correlation.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_avalanche_correlation.add_argument(
        "--cunxon-commit", required=True, help="cuNxon source commit"
    )
    p_cunxon_avalanche_correlation.add_argument(
        "--modes",
        default="infer,train",
        help="Comma-separated modes to sample: infer,train",
    )
    p_cunxon_avalanche_correlation.add_argument(
        "--seed-offsets",
        default="127,128",
        help="Comma-separated cuNxon random_seed_offset values",
    )
    p_cunxon_avalanche_correlation.add_argument(
        "--device", type=int, default=0, help="CUDA device id"
    )
    p_cunxon_avalanche_correlation.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_avalanche_intervention_task_correlation.json",
        help="JSON artifact path",
    )
    p_cunxon_avalanche_correlation.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_avalanche_intervention_task_correlation.md",
        help="Markdown artifact path",
    )
    p_cunxon_avalanche_correlation.set_defaults(
        func=cmd_cunxon_avalanche_intervention_task_correlation
    )

    p_cunxon_controlled_regime = sub.add_parser(
        "cunxon-controlled-regime-calibration",
        help="Calibrate cuNxon criticality estimator under controlled drive regimes",
        description=(
            "Run low/medium/high input-drive snapshot windows over the same held-out, stress "
            "and control splits to separate estimator calibration from action-quality claims."
        ),
    )
    p_cunxon_controlled_regime.add_argument(
        "--library", required=True, help="Path to built libcunxon.so"
    )
    p_cunxon_controlled_regime.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_controlled_regime.add_argument(
        "--cunxon-commit", required=True, help="cuNxon source commit"
    )
    p_cunxon_controlled_regime.add_argument(
        "--modes",
        default="infer,train",
        help="Comma-separated modes to sample: infer,train",
    )
    p_cunxon_controlled_regime.add_argument(
        "--seed-offsets",
        default="133,134",
        help="Comma-separated cuNxon random_seed_offset values",
    )
    p_cunxon_controlled_regime.add_argument(
        "--steps",
        type=int,
        default=128,
        help="Steps per controlled-regime stimulus window",
    )
    p_cunxon_controlled_regime.add_argument(
        "--sample-interval",
        type=int,
        default=8,
        help="Capture full-sphere snapshots every N steps",
    )
    p_cunxon_controlled_regime.add_argument(
        "--device", type=int, default=0, help="CUDA device id"
    )
    p_cunxon_controlled_regime.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_controlled_regime_calibration.json",
        help="JSON artifact path",
    )
    p_cunxon_controlled_regime.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_controlled_regime_calibration.md",
        help="Markdown artifact path",
    )
    p_cunxon_controlled_regime.set_defaults(func=cmd_cunxon_controlled_regime_calibration)

    p_cunxon_aigarth_remap = sub.add_parser(
        "cunxon-aigarth-action-remap-audit",
        help="Replay Aigarth action artifacts through an explicit strict motor-lane remap",
        description=(
            "Post-hoc decoder-contract diagnostic: replay existing live cuNxon Aigarth "
            "action JSON artifacts through a signed-first-lane remap to separate generic "
            "ActionDecoder vocabulary effects from evolved readout evidence."
        ),
    )
    p_cunxon_aigarth_remap.add_argument(
        "--source-artifacts",
        required=True,
        help="Comma-separated Aigarth action JSON artifacts to replay",
    )
    p_cunxon_aigarth_remap.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_aigarth_action_remap_audit.json",
        help="JSON artifact path",
    )
    p_cunxon_aigarth_remap.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_aigarth_action_remap_audit.md",
        help="Markdown artifact path",
    )
    p_cunxon_aigarth_remap.set_defaults(func=cmd_cunxon_aigarth_action_remap_audit)

    p_cunxon_sensitivity = sub.add_parser(
        "cunxon-sensitivity-probe",
        help="Compare cuNxon frozen infer and plastic train sensitivity",
        description=(
            "Run fresh one-sphere cuNxon samples in StepInfer and StepTrain modes over "
            "fixed stimuli/seeds to check whether train-mode exposes non-flat action signal."
        ),
    )
    p_cunxon_sensitivity.add_argument("--library", required=True, help="Path to built libcunxon.so")
    p_cunxon_sensitivity.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_sensitivity.add_argument("--cunxon-commit", required=True, help="cuNxon source commit")
    p_cunxon_sensitivity.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Simulation steps per mode/seed/stimulus sample",
    )
    p_cunxon_sensitivity.add_argument(
        "--seed-offsets",
        default="79,80,81",
        help="Comma-separated cuNxon random_seed_offset values",
    )
    p_cunxon_sensitivity.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_sensitivity.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_sensitivity_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_sensitivity.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_sensitivity_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_sensitivity.set_defaults(func=cmd_cunxon_sensitivity_probe)

    p_cunxon_snapshot = sub.add_parser(
        "cunxon-snapshot-pattern-probe",
        help="Inspect cuNxon hidden-state snapshots and pattern store/recall APIs",
        description=(
            "Run a cuNxon sphere through snapshot and host-side pattern store/recall APIs "
            "before building more action adapters."
        ),
    )
    p_cunxon_snapshot.add_argument("--library", required=True, help="Path to built libcunxon.so")
    p_cunxon_snapshot.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_snapshot.add_argument("--cunxon-commit", required=True, help="cuNxon source commit")
    p_cunxon_snapshot.add_argument(
        "--present-steps",
        type=int,
        default=30,
        help="Training/presentation steps per stored pattern",
    )
    p_cunxon_snapshot.add_argument(
        "--settle-steps",
        type=int,
        default=20,
        help="Inference settle steps per pattern recall",
    )
    p_cunxon_snapshot.add_argument(
        "--mask-fraction",
        type=float,
        default=0.5,
        help="Fraction of pattern input masked during recall",
    )
    p_cunxon_snapshot.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_snapshot.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_snapshot_pattern_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_snapshot.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_snapshot_pattern_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_snapshot.set_defaults(func=cmd_cunxon_snapshot_pattern_probe)

    p_cunxon_multi = sub.add_parser(
        "cunxon-multisphere-action-probe",
        help="Run cuNxon sensory-association-motor action adapter against holdout/baselines",
        description=(
            "Build a small three-sphere cuNxon topology, decode motor readout through "
            "the existing action contract, and compare train/holdout cases to trivial baselines."
        ),
    )
    p_cunxon_multi.add_argument("--library", required=True, help="Path to built libcunxon.so")
    p_cunxon_multi.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_multi.add_argument("--cunxon-commit", required=True, help="cuNxon source commit")
    p_cunxon_multi.add_argument(
        "--train-steps",
        type=int,
        default=24,
        help="Training steps per train case before evaluation",
    )
    p_cunxon_multi.add_argument(
        "--eval-steps",
        type=int,
        default=16,
        help="Inference steps per train/holdout evaluation case",
    )
    p_cunxon_multi.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_multi.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_multisphere_action_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_multi.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_multisphere_action_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_multi.set_defaults(func=cmd_cunxon_multisphere_action_probe)

    p_cunxon_resident_action = sub.add_parser(
        "cunxon-resident-action-probe",
        help="Probe task-coupled action scoring while one cuNxon network stays resident",
        description=(
            "Keep one three-sphere cuNxon network/context resident across repeated "
            "train/eval task epochs, then score motor readout against holdouts and baselines."
        ),
    )
    p_cunxon_resident_action.add_argument(
        "--library", required=True, help="Path to built libcunxon.so"
    )
    p_cunxon_resident_action.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_resident_action.add_argument(
        "--cunxon-commit", required=True, help="cuNxon source commit"
    )
    p_cunxon_resident_action.add_argument(
        "--train-epochs",
        type=int,
        default=6,
        help="Number of resident passes over train cases",
    )
    p_cunxon_resident_action.add_argument(
        "--train-steps-per-case",
        type=int,
        default=64,
        help="StepTrain calls per train case per epoch",
    )
    p_cunxon_resident_action.add_argument(
        "--eval-steps",
        type=int,
        default=32,
        help="StepInfer calls per train/holdout evaluation case",
    )
    p_cunxon_resident_action.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_resident_action.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_resident_action_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_resident_action.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_resident_action_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_resident_action.set_defaults(func=cmd_cunxon_resident_action_probe)

    p_cunxon_interface = sub.add_parser(
        "cunxon-interface-semantics-probe",
        help="Probe cuNxon readout and relay port mapping semantics",
        description=(
            "Run focused cuNxon same-sphere readout and source-to-downstream relay checks "
            "to validate absolute-vs-relative neuron index semantics before another adapter."
        ),
    )
    p_cunxon_interface.add_argument("--library", required=True, help="Path to built libcunxon.so")
    p_cunxon_interface.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_interface.add_argument("--cunxon-commit", required=True, help="cuNxon source commit")
    p_cunxon_interface.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Inference steps for each interface sample",
    )
    p_cunxon_interface.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_interface.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_interface_semantics_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_interface.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_interface_semantics_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_interface.set_defaults(func=cmd_cunxon_interface_semantics_probe)

    p_cunxon_external_drive = sub.add_parser(
        "cunxon-external-drive-window-probe",
        help="Probe cuNxon input/hidden/output sensory-id drive semantics",
        description=(
            "Drive identical values through input, hidden, and output sensory-id windows "
            "to test whether non-input windows behave like desired-output channels."
        ),
    )
    p_cunxon_external_drive.add_argument(
        "--library", required=True, help="Path to built libcunxon.so"
    )
    p_cunxon_external_drive.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_external_drive.add_argument(
        "--cunxon-commit", required=True, help="cuNxon source commit"
    )
    p_cunxon_external_drive.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Infer/train steps for each port-window sample",
    )
    p_cunxon_external_drive.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_external_drive.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_external_drive_window_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_external_drive.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_external_drive_window_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_external_drive.set_defaults(func=cmd_cunxon_external_drive_window_probe)

    p_cunxon_supervised = sub.add_parser(
        "cunxon-supervised-motor-probe",
        help="Probe teacher-forced cuNxon motor targets against holdout/baselines",
        description=(
            "Train a small cuNxon motor sphere with explicit teacher-forced absolute "
            "output-neuron target ports, then evaluate train/holdout cases without target drive."
        ),
    )
    p_cunxon_supervised.add_argument("--library", required=True, help="Path to built libcunxon.so")
    p_cunxon_supervised.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_supervised.add_argument("--cunxon-commit", required=True, help="cuNxon source commit")
    p_cunxon_supervised.add_argument(
        "--train-epochs",
        type=int,
        default=8,
        help="Number of passes over train cases with target drive",
    )
    p_cunxon_supervised.add_argument(
        "--train-steps-per-case",
        type=int,
        default=16,
        help="StepTrain calls per train case per epoch",
    )
    p_cunxon_supervised.add_argument(
        "--eval-steps",
        type=int,
        default=16,
        help="StepInfer calls per train/holdout evaluation case without target drive",
    )
    p_cunxon_supervised.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_supervised.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_supervised_motor_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_supervised.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_supervised_motor_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_supervised.set_defaults(func=cmd_cunxon_supervised_motor_probe)

    p_cunxon_input_proxy = sub.add_parser(
        "cunxon-input-proxy-target-probe",
        help="Probe supported input-class target proxies against motor readout/baselines",
        description=(
            "Train a small cuNxon sphere with sensory inputs plus input-class target "
            "proxy ports, then evaluate motor readout without target-proxy drive."
        ),
    )
    p_cunxon_input_proxy.add_argument("--library", required=True, help="Path to built libcunxon.so")
    p_cunxon_input_proxy.add_argument(
        "--upstream-commit",
        required=True,
        help="Upstream Neuraxon commit",
    )
    p_cunxon_input_proxy.add_argument("--cunxon-commit", required=True, help="cuNxon source commit")
    p_cunxon_input_proxy.add_argument(
        "--train-epochs",
        type=int,
        default=8,
        help="Number of passes over train cases with target-proxy drive",
    )
    p_cunxon_input_proxy.add_argument(
        "--train-steps-per-case",
        type=int,
        default=16,
        help="StepTrain calls per train case per epoch",
    )
    p_cunxon_input_proxy.add_argument(
        "--eval-steps",
        type=int,
        default=16,
        help="StepInfer calls per train/holdout evaluation case without target-proxy drive",
    )
    p_cunxon_input_proxy.add_argument("--device", type=int, default=0, help="CUDA device id")
    p_cunxon_input_proxy.add_argument(
        "--json-output",
        default="benchmarks/results/cunxon_input_proxy_target_probe.json",
        help="JSON artifact path",
    )
    p_cunxon_input_proxy.add_argument(
        "--markdown-output",
        default="benchmarks/results/cunxon_input_proxy_target_probe.md",
        help="Markdown artifact path",
    )
    p_cunxon_input_proxy.set_defaults(func=cmd_cunxon_input_proxy_target_probe)

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.command is None:
        parser.print_help()
        return 2
    func = getattr(args, "func")
    if not callable(func):
        return 2
    handler = cast(Callable[[argparse.Namespace], int], func)
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
