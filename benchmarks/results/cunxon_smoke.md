# cuNxon GPU smoke report

Status: `build-only`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Local upstream checkout: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon`
- Built library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so`

## Aorus GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Driver: 595.71.05
- CUDA/nvcc: 13.2.78
- CMake: 4.3.2
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

Runtime note: the first attempt to run this new Neuraxon-Hybrid ctypes smoke command against the built `libcunxon.so` was blocked by the local execution safety layer with `BLOCKED: User denied. Do NOT retry.` The command was not retried. Therefore the current tracked status remains `build-only`, not `smoke-test viable`.

## Evidence boundary

No broad Neuraxon intelligence claim: this report only proves that upstream cuNxon can configure, compile, pass its upstream `ctest`, and run its C++ demos on Aorus/RTX 5090. GPU execution does not prove intelligence, generalization, useful policy learning, or superiority over existing Neuraxon-Hybrid baselines.

## Recommended next step

If/when local execution approval permits, run the guarded Neuraxon-Hybrid smoke command and only then promote the status from `build-only` to `smoke-test viable`:

```bash
neuraxon-agent cunxon-smoke \
  --library /home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so \
  --upstream-commit bd2242fabad08cb73dab2c4170d11fa941030e8c \
  --cunxon-commit b4f6db85f7aff04ddb4e1078d523d514a278521b \
  --steps 24 \
  --json-output benchmarks/results/cunxon_smoke.json \
  --markdown-output benchmarks/results/cunxon_smoke.md
```
