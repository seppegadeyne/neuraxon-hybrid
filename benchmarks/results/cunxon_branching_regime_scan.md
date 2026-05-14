# cuNxon branching-ratio regime scan

Status: `branching-regime scan completed`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`
- Source probe: `target_contract_augmented_train`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Generations per seed: 12
- Population size: 24
- Eval steps per case: 24
- Seed offsets: 117, 118, 119, 120, 121

## Why this probe exists

Qubic NIA Vol. 8 makes branching ratio / criticality a concrete hypothesis. This scan treats the branching/activity-ratio proxy as instrumentation only: it must be read beside holdout, stress_holdout, controls, and best constant baseline scores before any adapter-quality conclusion is considered.

## Per-seed regime metrics and task quality

| Seed | Regime | Branching/activity-ratio proxy | Neutral occupancy | Transition entropy | Energy slope | Readout diversity | Holdout | Stress holdout | Beats stress baseline | GPU mem/util/temp |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 117 | reverberating/near-critical proxy | 0.988506 | 0.622222 | 1.286693 | -140.331426 | 5 | 1.000000 | 0.333333 | False | 2962 MB / 37% / 38 C |
| 118 | reverberating/near-critical proxy | 0.988506 | 0.622222 | 1.286693 | -127.364904 | 6 | 1.000000 | 0.333333 | False | 2962 MB / 37% / 38 C |
| 119 | reverberating/near-critical proxy | 0.982759 | 0.666667 | 1.537644 | -141.701693 | 8 | 1.000000 | 0.333333 | False | 2962 MB / 37% / 38 C |
| 120 | reverberating/near-critical proxy | 0.896552 | 0.733333 | 1.689482 | -141.883515 | 7 | 1.000000 | 0.333333 | False | 2962 MB / 37% / 38 C |
| 121 | reverberating/near-critical proxy | 1.132184 | 0.511111 | 1.429473 | -156.013101 | 8 | 1.000000 | 0.333333 | False | 2962 MB / 37% / 38 C |

## Regime buckets

| Regime bucket | Seeds | Mean branching proxy | Mean holdout | Mean stress_holdout | Stress seeds > best constant baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| reverberating/near-critical proxy | 5 | 0.997701 | 1.000000 | 0.333333 | 0 |

## Correlation summary

- holdout_accuracy_vs_branching_proxy: 0.000000
- stress_holdout_accuracy_vs_branching_proxy: 0.000000

## Verdict

Near-critical-looking proxy activity did not consistently beat the best constant baseline on stress_holdout; branching ratio remains diagnostic, not sufficient evidence.

## Notes

- branching/activity ratio is a proxy over final action-case readout samples
- source action probe uses train plus augmented_train cases inside fitness only
- holdout, stress_holdout, counterfactual_control and permuted_control labels are never optimized
- regime metrics are diagnostics only and must be interpreted beside task baselines

## Evidence boundary

The branching/activity-ratio proxy is a coarse readout/action-sample proxy, not a neuroscience-grade branching-ratio estimator. A near-critical-looking proxy is not intelligence evidence by itself; useful computation requires held-out task quality above best constant baseline and leakage/control checks.
