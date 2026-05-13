# Neuraxon Agent Benchmark Report

## 1. Summary

After the semantic tissue policy, the mock benchmark is fully solved for the existing scenario set. The tissue now uses the structured observation semantics from the benchmark instead of relying only on random low-level network output. As a result, complete tool calls, missing parameters, retryable failures, ambiguous prompts, non-retryable recovery, and success streaks are handled differently.

Result: `neuraxon_tissue` now achieves 700/700 correct runs = 100.00% accuracy. That is significantly better than both random (15.71%) and always-execute (28.57%).

An additional holdout/noisy generalization smoke benchmark still achieves 140/140 correct runs = 100.00%; all 140 final observations are directly solvable by the explicit semantic policy bridge (`semantic_policy_coverage=100%`). This confirms your suspicion: 100% here is primarily oracle/feature coverage, not evidence for emergent Neuraxon dynamics.

The NIA-inspired temporal dynamics probe was therefore expanded from 6 smoke cases to 108 generated scenarios: 6 action archetypes × 3 dataset seeds × 3 sequence lengths × 2 variants. The final action oracle is hidden inside a generic `temporal_decision_probe`; counterfactual pairs share the exact same final observation but require different actions because of the preceding observations, and noise/perturbation variants change irrelevant fields without changing the latent temporal state. On that probe, `neuraxon_tissue` now achieves 108/108 = 100.00% through the explicit temporal context adapter, but the sequence-majority oracle baseline also achieves 108/108.

Issue #70 therefore adds a stricter anti-oracle temporal benchmark: 48 scenarios with a task-family train/test split, identical final probes, masked schema/field names, counterfactual pairs with the same aggregate sequence statistics, and irrelevant perturbation fields. On that benchmark, sequence-majority achieves 0/48 = 0.00% instead of 100%, while `temporal_context_adapter` and the full `semantic_bridge` mode achieve 48/48 = 100.00%. Raw-network-only achieves 7/48 = 14.58%, so the output explicitly separates adapter success from raw-network success.

Important nuance: this does not yet prove general Neuraxon intelligence. It proves that the runtime now has a working semantic decision bridge for the current mock scenarios. The biological/trinary tissue therefore remains instrumentable, but the useful policy in this slice comes from explicit observation semantics.

### Proven / not proven summary

Proven:

- semantic bridge performance: 700/700 correct runs (100.00%) on the current mock benchmark.
- explicit temporal context adapter performance: 108/108 correct runs (100.00%) on the temporal dynamics probe with identical final observations, plus 48/48 (100.00%) on the anti-oracle masked temporal probe.
- simple baseline performance: random achieves 22/140 (15.71%) and always-execute achieves 40/140 (28.57%) on the mock benchmark; on the older temporal probe, sequence-majority still achieves 100.00%, but on the anti-oracle split, sequence-majority drops to 0/48 (0.00%).
- criticality/dynamics instrumentation evidence: `dynamics_metrics.csv` contains 11,000 per-step samples and `criticality_summary.csv` classifies the tissue run as `near_useful_dynamic_regime` with neutral-state occupancy 0.757844, transition entropy 0.332013, and modulation action-change rate 0.000000.

Not proven:

- raw Neuraxon network generalization: policy-ablation `raw_network` achieves 110/700 (15.71%) on the mock benchmark and 7/48 (14.58%) on the anti-oracle temporal benchmark, so there is no claim that the raw continuous-time/trinary dynamics independently learn a useful policy above simple baselines.
- learned policy from feedback: modulation changes internal state, but this regeneration does not show post-feedback behavioral action changes.
- memory persistence or visual perception value: both remain out of scope until raw/adapter separation provides stronger evidence.

## 2. Benchmark setup

- Scenario dataset: `benchmarks/scenarios/mock_agent_scenarios.json`
- Scenario count: 140
- Scenario types: 7
- Tissue seeds: 5
- Tissue runs: 140 × 5 = 700
- Baselines: random and always-execute, 140 runs each
- Metrics: accuracy, confidence, per-scenario breakdown, learning curve, simple two-proportion z-tests
- Holdout/noisy smoke benchmark: 140 deterministic variants, 1 seed, original scenario-type labels replaced by `holdout_<expected_action>`; 100% semantic-policy coverage, so this should not be treated as a real generalization claim
- Temporal dynamics benchmark: 108 NIA-inspired scenarios (6 action archetypes × 3 dataset seeds × 3 sequence lengths × counterfactual/noise-perturbation variants), 1 tissue seed; the final observation contains no action oracle and counterfactual pairs can have the exact same final probe with different expected actions
- Anti-oracle temporal benchmark: 48 scenarios with a task-family train/test split (24 train, 24 test), identical final probes, masked schema/field names, counterfactual pairs with the same aggregate sequence statistics, perturbation/noise fields, and separate scores for full tissue runtime, raw-network-only, semantic-policy-only, temporal-context-adapter, random, always-execute, last-observation-only, and sequence-majority.

## 3. Results

| Agent | Runs | Correct | Accuracy | Avg. confidence |
|---|---:|---:|---:|---:|
| Neuraxon tissue | 700 | 700 | 100.00% | 1.0000 |
| Random baseline | 140 | 22 | 15.71% | 0.1667 |
| Always-execute baseline | 140 | 40 | 28.57% | 1.0000 |

## 4. Per scenario type

| Scenario type | Tissue accuracy | Random accuracy | Always-execute accuracy |
|---|---:|---:|---:|
| simple_tool_call | 100.00% | 25.00% | 100.00% |
| missing_params_tool_call | 100.00% | 20.00% | 0.00% |
| failed_tool_call | 100.00% | 15.00% | 0.00% |
| ambiguous_prompt | 100.00% | 5.00% | 0.00% |
| complex_multi_step | 100.00% | 10.00% | 100.00% |
| error_recovery | 100.00% | 10.00% | 0.00% |
| success_streak | 100.00% | 25.00% | 0.00% |

## 5. Interpretation

### Holdout/noisy generalization smoke

| Agent | Runs | Correct | Accuracy |
|---|---:|---:|---:|
| Neuraxon tissue | 140 | 140 | 100.00% |
| Random baseline | 140 | 25 | 17.86% |
| Always-execute baseline | 140 | 40 | 28.57% |

The holdout/noisy set is not complete proof of generalization. The variants remove a few original scenario-type labels and add irrelevant/noisy fields, but the final observations remain directly solvable through general observation fields such as missing parameters, retryability, ambiguity, risk, and success streaks.

Semantic-policy coverage audit:

| Measure | Value |
|---|---:|
| Final observations | 140 |
| Directly solvable by semantic policy | 140 |
| Coverage | 100.00% |

Decision: `pass_temporal_context_bridge_evidence`. The semantic policy bridge remains above always-execute on this deterministic holdout/noisy set, but 100% coverage means this score primarily proves that the hand-authored observation features are well covered. The temporal probe below is therefore the relevant state-carry-over check.

### NIA temporal dynamics probe

Qubic's NIA articles set a different bar: Vol. 1 emphasizes continuous time and state carry-over, Vol. 2 trinary neutral/subthreshold buffering, Vol. 3 neuromodulation and plasticity windows, Vol. 5 astrocytic/eligibility-gated plasticity, and Vol. 7 emergence around edge-of-chaos dynamics. A one-shot final observation with explicit semantic fields does not test that.

The temporal dynamics benchmark has therefore now been expanded. The final observation is the same generic `temporal_decision_probe` everywhere; the relevant signals exist only in the preceding observations. The dataset contains 108 scenarios with multiple dataset seeds, sequence lengths, counterfactual pairs, and noise/perturbation variants. Result:

| Agent | Runs | Correct | Accuracy |
|---|---:|---:|---:|
| Neuraxon tissue | 108 | 108 | 100.00% |
| Random baseline | 108 | 19 | 17.59% |
| Always-execute baseline | 108 | 18 | 16.67% |
| Last-observation-only baseline | 108 | 0 | 0.00% |
| Semantic-policy-only baseline | 108 | 0 | 0.00% |
| Sequence-majority oracle baseline | 108 | 108 | 100.00% |

Interpretation: last-observation-only and semantic-policy-only fail completely because the final probe contains no semantic action fields. The sequence-majority baseline proves that the expected action is indeed derivable from the preceding sequence. `neuraxon_tissue` now achieves the same perfect score through `temporal_context_bridge`: an explicit temporal context adapter in AgentTissue that summarizes compact prior-observation evidence before the identical final probe is decided. This preserves the separation between semantic-policy success and temporal state carry-over, and it is better than last-observation-only/always-execute, but it is still explicit adapter logic; raw Neuraxon network dynamics remain reported separately through policy ablation.

### Anti-oracle temporal generalization probe (#70)

The new anti-oracle variant makes the previous temporal probe stricter on exactly the failure mode from issue #70: semantic labels, obvious field names, and sequence-majority heuristics must not solve the benchmark trivially. The scenarios use a task-family train/test split, mask schema/field names (`x*`/`z*` instead of `signal`, `risk`, `missing_count`, etc.), share the same final probe (`{"z0": 0, "z1": "probe", "z2": 1}`), and contain counterfactual pairs whose aggregate sequence statistics match while the correct action differs.

| Agent / mode | Runs | Correct | Accuracy |
|---|---:|---:|---:|
| Full tissue runtime (`semantic_bridge`) | 48 | 48 | 100.00% |
| Temporal-context-adapter mode | 48 | 48 | 100.00% |
| Raw-network-only mode | 48 | 7 | 14.58% |
| Semantic-policy-only mode | 48 | 7 | 14.58% |
| Random baseline | 48 | 5 | 10.42% |
| Always-execute baseline | 48 | 8 | 16.67% |
| Last-observation-only baseline | 48 | 0 | 0.00% |
| Sequence-majority baseline | 48 | 0 | 0.00% |

Interpretation: the anti-oracle benchmark satisfies the stricter acceptance criteria for this phase: the sequence-majority baseline no longer trivially reaches 100%, counterfactuals require prior state, output is deterministic for fixed seeds, and the metrics separate adapter success from raw-network success. The 100% score is still adapter-driven (`temporal_context_bridge`), not raw Neuraxon learning.

### Policy-ablation benchmark

Issue #52 adds the explicit ablation that separates semantic-bridge performance from raw-network performance. The same 140 scenarios × 5 seeds were run in three modes:

| Mode | Runs | Correct | Accuracy | Interpretation |
|---|---:|---:|---:|---|
| semantic-bridge enabled | 700 | 700 | 100.00% | Hand-written semantic bridge solves policy-covered observations. |
| raw-network only | 700 | 110 | 15.71% | Low-level decoder path without semantic policy; this is the raw Neuraxon contribution in this slice. |
| semantic-bridge coverage audit | 700 | 700 | 100.00% | Audit mode that makes it clear policy-covered observations must not count as Neuraxon generalization. |

Conclusion: raw-network performance remains reported separately and is below always-execute on the current mock benchmark. Therefore, no Neuraxon generalization claim is made from policy-covered observations; the 100% score is a semantic-bridge result.

### Decision-layer interpretation

The earlier results showed two separate blockers:

1. Action-contract mismatch: fixed in #45.
2. No semantic distinction between scenarios: fixed in this slice with `SemanticTissuePolicy`.

Before #45, all mock observations effectively encoded to the same input pattern. Therefore, the tissue could not know whether an observation was a simple tool call, a missing parameter, a retryable failure, or an ambiguous prompt. The new semantic policy reads the structured observation fields directly and returns benchmark-aligned actions:

- complete tool request -> `execute`
- missing parameters -> `query`
- retryable failed tool call -> `retry`
- ambiguous prompt -> `explore`
- non-retryable/repeated recovery risk -> `cautious`
- success streak/high confidence -> `assertive`

### Criticality and neuromodulator dynamics

Issue #54 adds per-step dynamics instrumentation to the tissue benchmark without changing vendored upstream Neuraxon code. Each raw benchmark result now contains `dynamics_samples` with activity, energy, trinary state distribution, neutral-state occupancy, and neuromodulator levels per simulated step. In addition, each result contains `criticality_metrics` with activity variance, transition entropy, neutral-state occupancy, branching/activity propagation ratio, and average energy.

The analysis writes two new artifacts under `benchmarks/results/analysis/`: `dynamics_metrics.csv` for per-step inspection and `criticality_summary.csv` for run-level interpretation. The summary explicitly classifies the dynamics as `dead`, `saturated`, `random_like`, or `near_useful_dynamic_regime`, so the report can answer whether a run is dead/saturated/random-like or appears to be in a near useful dynamic regime. Modulation is tracked separately through neuromodulator deltas and `modulation_action_change_rate`: in this slice, modulation primarily changes observable state; a behavioral effect is only demonstrated when later decisions change after feedback.

| Metric | Interpretation |
|---|---|
| Activity variance | Low variance indicates a dead/saturated regime; moderate variance indicates useful dynamics. |
| Transition entropy | Measures how many trinary states change across steps; extremely low is frozen/dead, extremely high with unstable propagation is random-like. |
| Neutral-state occupancy | Tracks whether the trinary network mostly buffers in neutral/subthreshold states or is fully active/saturated. |
| Branching ratio | Simple activity-propagation proxy: active neurons at step t+1 compared with step t. |
| Modulation action-change rate | Indicates whether neuromodulator feedback changes later observable decisions or only internal state. |

## 6. Statistical comparison

| Comparison | Tissue accuracy | Baseline accuracy | Difference | p approx | Significant |
|---|---:|---:|---:|---:|---|
| Tissue vs random | 100.00% | 15.71% | +84.29pp | 0.000000 | yes |
| Tissue vs always-execute | 100.00% | 28.57% | +71.43pp | 0.000000 | yes |

## 7. What this does and does not prove

Proven:

- The benchmark pipeline can now score above baseline.
- The action vocabulary is fully covered, including `cautious`.
- The runtime can route the current mock-agent scenarios deterministically and correctly.
- Diagnostics no longer show missing decoder or observed-action coverage.

Not proven:

- No generalization beyond these hand-authored scenario features.
- No learned policy from feedback.
- No memory persistence value.
- No visual or multimodal perception.
- No evidence that the vendor Neuraxon dynamics independently learn a useful policy; the anti-oracle split makes this explicit with raw-network-only at 14.58% versus temporal-context-adapter at 100.00%.

## 8. Verdict

Status: GO for the semantic adapter and the explicit temporal context adapter; NO-GO for raw Neuraxon generalization claims.

The previous NO-GO blocker (not better than random/always-execute) is fixed for the current mock benchmark. The temporal blocker is more strongly fixed for this bounded slice: identical final probes can be decided differently based on preceding observations, the anti-oracle split beats sequence-majority/last-observation-only, and the reporting separates full tissue, raw-network, semantic-policy-only, and temporal-context-adapter modes. The nuance remains sharp: perfect anti-oracle temporal scores come from explicit adapter logic (`temporal_context_bridge`), not from demonstrated raw continuous-time/neuromodulated learning. Memory persistence and visual perception remain out of scope until raw/adapter separation continues to demonstrate useful decisions.

## 9. Artifacts

- Raw tissue benchmark: `benchmarks/results/neuraxon_tissue_raw.json`
- Summary CSV: `benchmarks/results/analysis/benchmark_summary.csv`
- Scenario breakdown CSV: `benchmarks/results/analysis/scenario_type_breakdown.csv`
- Statistical tests CSV: `benchmarks/results/analysis/statistical_tests.csv`
- Dynamics metrics CSV: `benchmarks/results/analysis/dynamics_metrics.csv`
- Criticality summary CSV: `benchmarks/results/analysis/criticality_summary.csv`
- Diagnostic traces: `benchmarks/results/diagnostics/action_mapping_traces.json`
- Diagnostic report: `benchmarks/results/diagnostics/action_mapping_diagnostic_report.md`
- Holdout/noisy generalization: `benchmarks/results/holdout_noisy_generalization.json`
- Policy-ablation benchmark: `benchmarks/results/policy_ablation.json`
- Plots:
  - `benchmarks/results/analysis/plots/accuracy_by_agent.png`
  - `benchmarks/results/analysis/plots/confidence_distribution.png`
  - `benchmarks/results/analysis/plots/neuromodulator_trends.png`
  - `benchmarks/results/analysis/plots/learning_curve.png`
  - `benchmarks/results/analysis/plots/activity_energy_trends.png`
