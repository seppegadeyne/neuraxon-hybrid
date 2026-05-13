# cuNxon comparison against existing Neuraxon-Hybrid evidence

Verdict: `cuNxon interface semantics clarified; decision probes remain flat/baseline-level`

## What changed

The previous cuNxon slices established CUDA runtime viability, long-horizon raw dynamics, a tiny task-coupled action probe, infer-vs-train sensitivity, hidden-state/pattern inspection, a three-sphere action adapter, and a longer task-coupled sweep over 108 live RTX 5090 samples. This follow-up adds a focused interface semantics probe for same-sphere readout ports and source-to-downstream relay ports.

The longer sweep remains baseline-level. Every mode × horizon accuracy is `0.333333`, exactly equal to the constant-action baselines on the balanced toy set. Shorter frozen infer samples show a little readout diversity, but by `4096`/`32768` steps infer also collapses to `query=9`; both `train` and `train_rewarded` are flat `query=9` at every tested horizon.

The new interface probe supports the absolute-neuron-index interpretation used by the C++ path: same-sphere readouts match their full-sphere snapshot slices for both relative input ports `[0..3]` and absolute output ports `[8..11]` in a 4/4/4 sphere. Relay behavior also differs by source port class: relaying input-neuron ports activates the downstream input readout, while relaying output-neuron ports stayed neutral in this setup. This clarifies interface semantics, but it is not a task-learning result.

The earlier snapshot/pattern, multi-sphere and long-sweep findings still stand: snapshot APIs expose hidden-state changes and pattern APIs are callable, but recall stayed flat zero; the three-sphere adapter ran but decoded all train/holdout cases to `query`, showed baseline-level holdout accuracy, and matched trivial baselines; longer one-sphere action sweeps remained baseline-level.

## Comparison lanes

| Lane | Metric | Value | Evidence |
| --- | --- | ---: | --- |
| cuNxon raw CUDA smoke | decision-quality score | no decision-quality score measured | `benchmarks/results/cunxon_smoke.json` |
| cuNxon long-horizon raw dynamics | continuous runtime samples | 65536 steps; 1 unique readout(s); 0 readout changes | `benchmarks/results/cunxon_long_horizon.json` |
| cuNxon long sweep action diagnostic | success_rate vs trivial baselines | 108 samples; all mode/horizon accuracies=0.333333; train/train_rewarded flat query | `benchmarks/results/cunxon_long_sweep.json` |
| cuNxon task-coupled action probe | success_rate | 0.333333 | `benchmarks/results/cunxon_action_probe.json` |
| cuNxon infer-vs-train sensitivity probe | readout/action diversity | infer unique=3; train unique=1; train-mode flat/query=9 | `benchmarks/results/cunxon_sensitivity_probe.json` |
| cuNxon snapshot/pattern probe | hidden-state and recall signal | snapshots active=0,8,2; patterns=2→0; recall hamming=0 | `benchmarks/results/cunxon_snapshot_pattern_probe.json` |
| cuNxon multi-sphere/action adapter | holdout success_rate vs trivial baselines | holdout=0.333333; overall=0.333333; flat query motor readouts | `benchmarks/results/cunxon_multisphere_action_probe.json` |
| cuNxon interface semantics probe | readout/relay port mapping | same-sphere readouts match snapshot slices; input-port relay activates downstream; output-port relay stays neutral in this setup | `benchmarks/results/cunxon_interface_semantics_probe.json` |
| raw_network | success_rate | 0.145833 | `holdout_noisy_generalization anti-oracle temporal mode` |
| random | success_rate | 0.104167 | `holdout_noisy_generalization anti-oracle temporal baseline` |
| always_execute | success_rate | 0.166667 | `holdout_noisy_generalization anti-oracle temporal baseline` |
| semantic_bridge | success_rate | 1.0 | `holdout_noisy_generalization semantic bridge mode` |
| temporal_context_adapter | success_rate | 1.0 | `holdout_noisy_generalization temporal adapter mode` |

## Interpretation

cuNxon remains interesting as a GPU runtime and diagnostics surface, and the interface semantics probe removes one ambiguity from the next adapter design: readout/relay ports should be treated as absolute neuron indices when targeting output neurons. However, this does not rescue the current decision-quality evidence. Longer training horizons and simple neuromodulator/reward injection did not turn the one-sphere action readout into a policy signal; they mostly collapsed to the same neutral `query` output.

The next useful direction is no longer “just run longer”. A plausible next slice is an explicit supervised motor-target adapter that uses absolute output-neuron indices and tests train/holdout cases against constant-action baselines. If that remains flat, the research should move to lower-level CTSN/readout or plasticity semantics rather than adding horizon length.

## Evidence boundary

This comparison deliberately separates runtime viability and interface semantics from decision quality. Snapshot activity, callable pattern APIs, inter-sphere topology construction, longer horizons, reward injection, and absolute-neuron-index port mapping prove useful diagnostic surfaces, but flat recall and baseline-level action accuracy do not prove intelligence, generalization, or useful learning.
