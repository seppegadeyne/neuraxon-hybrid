# cuNxon GPU smoke report

Status: `smoke-test viable`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Local upstream checkout: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon`
- Built library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## Aorus GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Driver: 595.71.05
- CUDA/nvcc: 13.2.78
- CMake: 4.3.2
- Compute capability: 12.0
- Build arch: `-DCUNXON_CUDA_ARCH=120`

## Upstream build and test result

Commands run from the upstream `cuNxon` folder:

```bash
cmake -S . -B build-hermes-issue79-20260513140153 -DCUNXON_CUDA_ARCH=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build-hermes-issue79-20260513140153 -j$(nproc)
ctest --test-dir build-hermes-issue79-20260513140153 --output-on-failure
```

Results:

- Configure: passed.
- Build: passed; produced `libcunxon.so`, `libcunxon_static.a`, `example_4sphere`, `example_aigarth`, and `test_cunxon`.
- `ctest`: passed; `1/1 Test #1: cunxon_smoke` in 0.20 sec.
- `example_4sphere 20 10 0`: ran on RTX 5090; overall accuracy was 50.0% for this tiny smoke run and ASC readout distribution was `+1=40, 0=27, -1=31` of 98.
- `example_aigarth`: ran; margin improved from +0.8750 to +1.5000 in the demo output.

## Python ctypes findings

The upstream `examples/python_binding.py` demo was tested after installing `numpy` into a temporary venv. It loaded the RTX 5090 device, then failed while adding the `MTR -> ASC` feedback link:

```text
RuntimeError: cunxonNetworkAddLink: INVALID_ARGUMENT ()
```

Likely cause: the upstream Python demo appears to pass relay/readout output indices relative to each output block, while the C++ demo and C API expect absolute neuron indices (`input + hidden + output_offset`). The C++ demo uses absolute indices and runs. This looks like a Python-demo wrapper issue, not a cuNxon C++/CUDA build failure.

## Neuraxon-Hybrid integration status

Added a guarded Neuraxon-Hybrid-side smoke module and CLI surface:

- `src/neuraxon_agent/cunxon_smoke.py`
- `neuraxon-agent cunxon-smoke --library <path-to-libcunxon.so> ...`

The module validates that cuNxon emits a non-empty trinary readout in `{-1, 0, +1}` and writes JSON/Markdown artifacts.

Live Hybrid smoke result:

- Command: `uv run neuraxon-agent cunxon-smoke --library /home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0 --upstream-commit bd2242fabad08cb73dab2c4170d11fa941030e8c --cunxon-commit b4f6db85f7aff04ddb4e1078d523d514a278521b --steps 16`
- Status: passed.
- Device: NVIDIA GeForce RTX 5090.
- Compute capability: 12.0.
- Steps: 16.
- Elapsed: 99.986 ms.
- Energy accumulator: 339.239.
- Trinary readout ({-1, 0, +1}): [0, 0, 0].
- Notes:
  - minimal one-sphere ctypes smoke completed
  - inter-sphere Python demo remains separate from this smoke path

## Evidence boundary

No broad Neuraxon intelligence claim: this report proves that upstream cuNxon can configure, compile, pass its upstream `ctest`, run its C++ demos on Aorus/RTX 5090, and load/step through the guarded Hybrid `ctypes` smoke path with a valid trinary readout. GPU execution does not prove intelligence, generalization, useful policy learning, or superiority over existing Neuraxon-Hybrid baselines.

## Recommended next step

The next step is a separate cuNxon-backed decision adapter or raw-dynamics probe that maps GPU readout to the same Neuraxon-Hybrid action contract, then compares it directly against `raw_network`, `random`, and `always_execute` without treating speed as intelligence evidence.
