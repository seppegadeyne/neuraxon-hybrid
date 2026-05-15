# cuNxon VRAM-resident dynamics run

Status: `completed`

## Hypothesis

Keeping one cuNxon context/network resident in the same Aorus RTX 5090 process for four wall-clock hours may expose hidden-state occupancy, readout, energy, or resource drift that short process-local probes reset away; this is dynamics evidence only, not an intelligence claim.

## Durable state

- Active issue: https://github.com/sisutuulenisa/neuraxon-hybrid/issues/79
- PID: `1929485`
- Command: `neuraxon-agent cunxon-vram-resident --library /home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0 --upstream-commit bd2242fabad08cb73dab2c4170d11fa941030e8c --cunxon-commit b4f6db85f7aff04ddb4e1078d523d514a278521b --hypothesis Keeping one cuNxon context/network resident in the same Aorus RTX 5090 process for four wall-clock hours may expose hidden-state occupancy, readout, energy, or resource drift that short process-local probes reset away; this is dynamics evidence only, not an intelligence claim. --active-issue https://github.com/sisutuulenisa/neuraxon-hybrid/issues/79 --max-runtime-seconds 14400 --sample-interval-seconds 900 --steps-per-sample 262144 --json-output benchmarks/results/cunxon_vram_resident_run.json --markdown-output benchmarks/results/cunxon_vram_resident_run.md --state-output benchmarks/results/cunxon_vram_resident_run_state.json`
- Started at: 2026-05-13T21:01:49.909576Z
- Updated at: 2026-05-14T01:01:49.909667Z
- Expected end: 2026-05-14T01:01:49.909576Z
- Next poll after: 2026-05-14T01:01:49.909576Z
- Stop condition: stop after 14400 seconds or on cuNxon/API/resource error

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Max runtime seconds: 14400
- Sample interval seconds: 900
- Steps per sample: 262144
- Total steps so far: 4194304
- Samples: 16

## Why this run exists

Earlier smoke, long-horizon, action, sensitivity, snapshot/pattern, multi-sphere, long-sweep, supervised-motor and source-semantics probes were runtime-viable but flat or baseline-level. This run tests a different hypothesis: whether keeping the same cuNxon process/network resident in VRAM across wall-clock time exposes hidden-state, energy, occupancy, readout or resource drift that short process-local runs reset away.

## Samples

| Sample | Step | Elapsed s | Readout | Active | Neutral | Energy | Energy delta | GPU MiB | GPU util % | Temp C |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 262144 | 14.6 | [0, 0, 0] | 3 | 12 | 9.83499e+06 | 9.83499e+06 | 3207 | 47 | 43 |
| 2 | 524288 | 912.8 | [0, 0, 0] | 5 | 10 | 1.9648e+07 | 9.81304e+06 | 2924 | 38 | 38 |
| 3 | 786432 | 1812.8 | [0, 0, 0] | 5 | 10 | 2.94877e+07 | 9.83971e+06 | 2924 | 38 | 38 |
| 4 | 1048576 | 2712.8 | [0, 0, 0] | 5 | 10 | 3.93478e+07 | 9.86003e+06 | 2936 | 39 | 38 |
| 5 | 1310720 | 3612.7 | [0, 0, 0] | 5 | 10 | 4.92204e+07 | 9.87263e+06 | 2936 | 37 | 38 |
| 6 | 1572864 | 4512.8 | [0, 0, 0] | 3 | 12 | 5.87863e+07 | 9.56594e+06 | 2936 | 38 | 38 |
| 7 | 1835008 | 5412.8 | [0, 0, 0] | 3 | 12 | 6.65992e+07 | 7.81282e+06 | 2936 | 38 | 38 |
| 8 | 2097152 | 6312.4 | [0, 0, 0] | 3 | 12 | 7.33921e+07 | 6.79299e+06 | 2936 | 35 | 38 |
| 9 | 2359296 | 7212.3 | [0, 0, 0] | 3 | 12 | 8.01148e+07 | 6.72266e+06 | 2936 | 35 | 38 |
| 10 | 2621440 | 8112.3 | [0, 0, 0] | 3 | 12 | 8.68375e+07 | 6.72266e+06 | 2936 | 35 | 38 |
| 11 | 2883584 | 9012.3 | [0, 0, 0] | 3 | 12 | 9.35601e+07 | 6.72266e+06 | 2936 | 34 | 38 |
| 12 | 3145728 | 9910.9 | [0, 0, 0] | 3 | 12 | 1.00283e+08 | 6.72266e+06 | 2936 | 39 | 38 |
| 13 | 3407872 | 10812.3 | [0, 0, 0] | 3 | 12 | 1.07005e+08 | 6.72266e+06 | 2936 | 34 | 38 |
| 14 | 3670016 | 11712.3 | [0, 0, 0] | 3 | 12 | 1.13728e+08 | 6.72266e+06 | 2936 | 34 | 38 |
| 15 | 3932160 | 12612.3 | [0, 0, 0] | 3 | 12 | 1.20451e+08 | 6.72266e+06 | 2936 | 35 | 38 |
| 16 | 4194304 | 13512.3 | [0, 0, 0] | 3 | 12 | 1.27173e+08 | 6.72266e+06 | 2936 | 35 | 38 |

## Notes

- same cuNxon process keeps one network/context resident between samples
- samples capture readout, full-state occupancy, energy and nvidia-smi resource data
- runtime/dynamics evidence only; not a task-learning or intelligence claim

## Evidence boundary

This is runtime/dynamics evidence only. A VRAM-resident process, changing energy, or changing occupancy does not prove intelligence, task learning, useful recall, or generalization unless a later task-coupled benchmark beats trivial baselines.
