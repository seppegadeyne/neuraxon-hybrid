# cuNxon comparison against existing Neuraxon-Hybrid evidence

Verdict: `supervised cuNxon motor targets still baseline-level`

## What changed

The previous cuNxon slices established CUDA runtime viability, long-horizon raw dynamics, a tiny task-coupled action probe, infer-vs-train sensitivity, hidden-state/pattern inspection, a three-sphere action adapter, a longer task-coupled sweep over 108 live RTX 5090 samples, and an interface-semantics probe for absolute output-neuron ports. This follow-up adds `neuraxon-agent cunxon-supervised-motor-probe`: a teacher-forced motor-target adapter that drives absolute output-neuron target ports during `StepTrain`, then evaluates train/holdout cases without target drive.

The longer sweep remains baseline-level. Every mode × horizon accuracy is `0.333333`, exactly equal to the constant-action baselines on the balanced toy set. Shorter frozen infer samples show a little readout diversity, but by `4096`/`32768` steps infer also collapses to `query=9`; both `train` and `train_rewarded` are flat `query=9` at every tested horizon.

The interface probe supports the absolute-neuron-index interpretation used by the C++ path: same-sphere readouts match their full-sphere snapshot slices for both relative input ports `[0..3]` and absolute output ports `[8..11]` in a 4/4/4 sphere. The supervised motor-target follow-up used that route with target ports `[8, 9, 10]`, but the learned/evaluated motor readout still stayed `[0, 0, 0]` for all 6 train/holdout cases.

The earlier snapshot/pattern, multi-sphere, long-sweep and new supervised-target findings now point in the same direction: snapshot APIs expose hidden-state changes and pattern APIs are callable, but recall stayed flat zero; the three-sphere adapter ran but decoded all train/holdout cases to `query` and showed baseline-level holdout accuracy; the explicit supervised motor-target adapter also decoded all cases to `query`; and both action adapters matched trivial constant-action baselines at 0.333333 accuracy.

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
| cuNxon supervised motor-target adapter | holdout success_rate vs trivial baselines | holdout=0.333333; overall=0.333333; target_alignment=0.777778; eval readouts flat query | `benchmarks/results/cunxon_supervised_motor_probe.json` |
| raw_network | success_rate | 0.145833 | `holdout_noisy_generalization anti-oracle temporal mode` |
| random | success_rate | 0.104167 | `holdout_noisy_generalization anti-oracle temporal baseline` |
| always_execute | success_rate | 0.166667 | `holdout_noisy_generalization anti-oracle temporal baseline` |
| semantic_bridge | success_rate | 1.0 | `holdout_noisy_generalization semantic bridge mode` |
| temporal_context_adapter | success_rate | 1.0 | `holdout_noisy_generalization temporal adapter mode` |

## Interpretation

cuNxon remains interesting as a GPU runtime and diagnostics surface, and the interface semantics probe removed one ambiguity from adapter design: readout ports should be treated as absolute neuron indices when targeting output neurons. However, teacher-forcing those absolute output-neuron target ports did not rescue the decision-quality evidence. The supervised motor-target adapter remained flat `[0, 0, 0]`/`query` across train and holdout cases, with holdout accuracy equal to all constant-action baselines.

The next useful direction is no longer “just run longer” or “just add a target port”. A plausible next slice is lower-level CTSN/readout/plasticity semantics inspection: why do absolute output ports remain neutral even when used as teacher-forced sensory targets during `StepTrain`, and is there any sanctioned C API route for desired-output/error signals beyond neuromodulator reward injection?

## Evidence boundary

This comparison deliberately separates runtime viability and interface semantics from decision quality. Snapshot activity, callable pattern APIs, inter-sphere topology construction, longer horizons, reward injection, absolute-neuron-index port mapping, and explicit teacher-forced output targets prove useful diagnostic surfaces, but flat recall and baseline-level action accuracy do not prove intelligence, generalization, or useful learning.
