"""cuNxon GPU smoke helpers and report rendering.

This module intentionally keeps the cuNxon path as an optional investigation
surface. Loading the CUDA library is explicit and runtime-only so normal CPU
benchmarks/tests keep working without NVIDIA/CUDA dependencies.
"""

from __future__ import annotations

import ctypes as C
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

CUNXON_OK = 0
CUNXON_PROP_DEVICE_NAME = 1
CUNXON_PROP_COMPUTE_CAPABILITY = 2
CUNXON_SPHERE_SENSORY = 0


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

        readout_size = C.c_int(0)
        _check(lib, lib.cunxonSphereGetReadout(net, sphere_id.value, None, C.byref(readout_size)))
        if readout_size.value <= 0:
            raise CunxonError("cuNxon returned an empty readout size")
        readout_buffer = (C.c_int8 * readout_size.value)()
        _check(
            lib,
            lib.cunxonSphereGetReadout(
                net, sphere_id.value, readout_buffer, C.byref(readout_size)
            ),
        )
        readout = [int(readout_buffer[i]) for i in range(readout_size.value)]
        validate_trinary_readout(readout)

        energy = C.c_double(0.0)
        _check(lib, lib.cunxonNetworkGetEnergy(net, C.byref(energy)))
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
            energy=float(energy.value),
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
    lib.cunxonNetworkFinalize.argtypes = [C.c_void_p]
    lib.cunxonNetworkFinalize.restype = C.c_int
    lib.cunxonNetworkStepInfer.argtypes = [C.c_void_p, C.POINTER(C.POINTER(C.c_float)), C.c_float]
    lib.cunxonNetworkStepInfer.restype = C.c_int
    lib.cunxonSphereGetReadout.argtypes = [
        C.c_void_p,
        C.c_int,
        C.POINTER(C.c_int8),
        C.POINTER(C.c_int),
    ]
    lib.cunxonSphereGetReadout.restype = C.c_int
    lib.cunxonNetworkGetEnergy.argtypes = [C.c_void_p, C.POINTER(C.c_double)]
    lib.cunxonNetworkGetEnergy.restype = C.c_int
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
