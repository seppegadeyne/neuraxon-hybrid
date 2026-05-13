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
    run_ctypes_action_probe,
    run_ctypes_interface_semantics_probe,
    run_ctypes_long_horizon_probe,
    run_ctypes_long_sweep_probe,
    run_ctypes_multisphere_action_probe,
    run_ctypes_sensitivity_probe,
    run_ctypes_smoke,
    run_ctypes_snapshot_pattern_probe,
    write_action_probe_artifacts,
    write_interface_semantics_artifacts,
    write_long_horizon_artifacts,
    write_long_sweep_artifacts,
    write_multisphere_action_artifacts,
    write_sensitivity_probe_artifacts,
    write_smoke_artifacts,
    write_snapshot_pattern_artifacts,
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
