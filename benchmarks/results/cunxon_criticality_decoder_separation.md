# cuNxon criticality/decoder separation

Status: `criticality decoder separation completed`
Hypothesis: `criticality_decoder_separation`
Source artifact: `benchmarks/results/cunxon_controlled_regime_calibration.json`

## Why this report exists

Qubic NIA Vol. 8 made criticality/branching ratio a measurable hypothesis. The controlled-regime calibration showed that estimator movement is present, but stress_holdout remained at the best constant baseline. This follow-up separates estimator movement from decoder/action collapse before adding task complexity.

## Core diagnosis

- Diagnosis: `decoder_action_collapse_on_low_margin_execute_retry_stress_cases`
- Stress accuracy: 0.333333 vs best constant baseline 0.333333
- stress query collapse: query_collapse_rate=0.902778 (65/72)
- execute/retry stress accuracy=0.041667
- query stress accuracy=0.916667
- Estimator movement status: `present_but_not_task_sufficient`; estimator movement is present but not task-sufficient.

## Stress cases by expected action

| Expected action | Samples | Accuracy | Action distribution | Top readouts |
| --- | ---: | ---: | --- | --- |
| execute | 24 | 0.083333 | execute=2, query=21, retry=1 | [0, 0, 0]×21, [0, 0, -1]×1, [1, 0, 0]×1 |
| query | 24 | 0.916667 | assertive=1, execute=1, query=22 | [0, 0, 0]×22, [1, 0, 1]×1, [1, 0, 0]×1 |
| retry | 24 | 0.000000 | assertive=1, execute=1, query=22 | [0, 0, 0]×22, [1, 0, 1]×1, [1, 0, 0]×1 |

## Stress cases by controlled regime

| Regime | Stress accuracy | Mean branching | Neutral occupancy | Query collapse | Actions |
| --- | ---: | ---: | ---: | ---: | --- |
| high-drive | 0.333333 | 0.165974 | 0.836623 | 0.708333 | assertive=2, execute=4, query=17, retry=1 |
| low-drive | 0.333333 | 0.000000 | 1.000000 | 1.000000 | query=24 |
| medium-drive | 0.333333 | 0.000000 | 1.000000 | 1.000000 | query=24 |

## Contrast splits

| Split | Samples | Accuracy | Mean branching | Neutral occupancy | Actions |
| --- | ---: | ---: | ---: | ---: | --- |
| holdout | 36 | 0.416667 | 0.078025 | 0.932931 | assertive=1, execute=5, query=29, retry=1 |
| hard_holdout | 72 | 0.416667 | 0.078364 | 0.930464 | assertive=2, execute=6, query=61, retry=3 |
| counterfactual_control | 36 | 0.250000 | 0.075752 | 0.934119 | assertive=1, execute=4, query=30, retry=1 |
| permuted_control | 36 | 0.277778 | 0.118162 | 0.893914 | assertive=1, execute=7, query=27, retry=1 |

## Interpretation

Controlled low/medium/high drive changes the cuNxon branching, occupancy, entropy and aggregate action distribution, but the stress bottleneck is dominated by decoder/action collapse: 65/72 stress cases decode to query; execute/retry stress cases score only 2/48, while query stress cases score 22/24.

This supports a narrower bottleneck statement: controlled drive can move activity-regime metrics and produce some non-query actions, but the low-margin stress split still collapses mostly to the query action. The failure is therefore better explained as decoder/action collapse on execute/retry stress cases than as complete estimator collapse.

## Evidence boundary

This is post-hoc separation of estimator movement from decoder/action behavior over an existing live RTX 5090 calibration artifact. It uses no stress/control labels for optimization and is not intelligence evidence; useful-computation claims still require stress/control task quality above constant baselines across seeds and settings.
