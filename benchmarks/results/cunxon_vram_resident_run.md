# cuNxon VRAM-resident dynamics run

Status: `completed`

## Hypothesis

Short bootstrap: verify that the VRAM-resident runner writes durable progress/state before launching the multi-hour process.

## Durable state

- Active issue: https://github.com/sisutuulenisa/neuraxon-hybrid/issues/79
- PID: `1927338`
- Command: `neuraxon-agent cunxon-vram-resident --library /home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0 --upstream-commit bd2242fabad08cb73dab2c4170d11fa941030e8c --cunxon-commit b4f6db85f7aff04ddb4e1078d523d514a278521b --hypothesis Short bootstrap: verify that the VRAM-resident runner writes durable progress/state before launching the multi-hour process. --max-runtime-seconds 12 --sample-interval-seconds 4 --steps-per-sample 4096 --json-output benchmarks/results/cunxon_vram_resident_run.json --markdown-output benchmarks/results/cunxon_vram_resident_run.md --state-output benchmarks/results/cunxon_vram_resident_run_state.json`
- Started at: 2026-05-13T20:58:37.413163Z
- Updated at: 2026-05-13T20:58:49.413260Z
- Expected end: 2026-05-13T20:58:49.413163Z
- Next poll after: 2026-05-13T20:58:49.413163Z
- Stop condition: stop after 12 seconds or on cuNxon/API/resource error

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Max runtime seconds: 12
- Sample interval seconds: 4
- Steps per sample: 4096
- Total steps so far: 12288
- Samples: 3

## Why this run exists

Earlier smoke, long-horizon, action, sensitivity, snapshot/pattern, multi-sphere, long-sweep, supervised-motor and source-semantics probes were runtime-viable but flat or baseline-level. This run tests a different hypothesis: whether keeping the same cuNxon process/network resident in VRAM across wall-clock time exposes hidden-state, energy, occupancy, readout or resource drift that short process-local runs reset away.

## Samples

| Sample | Step | Elapsed s | Readout | Active | Neutral | Energy | Energy delta | GPU MiB | GPU util % | Temp C |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 0.3 | [0, -1, 0] | 4 | 11 | 154057 | 154057 | 2454 | 25 | 39 |
| 2 | 8192 | 4.3 | [0, -1, 0] | 5 | 10 | 307498 | 153441 | 2454 | 25 | 39 |
| 3 | 12288 | 8.3 | [0, 0, 0] | 5 | 10 | 461187 | 153688 | 2454 | 25 | 40 |

## Notes

- same cuNxon process keeps one network/context resident between samples
- samples capture readout, full-state occupancy, energy and nvidia-smi resource data
- runtime/dynamics evidence only; not a task-learning or intelligence claim

## Evidence boundary

This is runtime/dynamics evidence only. A VRAM-resident process, changing energy, or changing occupancy does not prove intelligence, task learning, useful recall, or generalization unless a later task-coupled benchmark beats trivial baselines.
