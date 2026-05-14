# cuNxon comparison against existing Neuraxon-Hybrid evidence

Verdict: `source semantics audit: output-port teacher forcing unsupported; VRAM-resident 4h run completed as flat runtime/dynamics evidence`

## What changed

The previous cuNxon slices established CUDA runtime viability, long-horizon raw dynamics, a tiny task-coupled action probe, infer-vs-train sensitivity, hidden-state/pattern inspection, a three-sphere action adapter, a longer task-coupled sweep over 108 live RTX 5090 samples, an interface-semantics probe for absolute output-neuron ports, and a supervised motor-target adapter. This follow-up adds `benchmarks/results/cunxon_source_semantics_audit.*`: a source-level C/CUDA audit plus a live upstream 4-sphere demo rerun. It also adds a bounded `cunxon-vram-resident` runner and final four-hour artifact under `benchmarks/results/cunxon_vram_resident_run.*` so cron runs can keep one process/network resident, poll progress, and avoid repeatedly restarting the network.

The source audit found no desired-output API surface in the public cuNxon execution path. `StepTrain` exposes continuous plasticity and neuromodulator reward injection, but not a label/loss/error-vector argument. External step inputs are scattered only through `sensory_input_ids`, and the membrane kernel treats external input as direct pass-through only for input-class neurons; output neurons are readout-capable but are not directly teacher-forced by putting their absolute ids into `sensory_input_ids`.

The live upstream 4-sphere reward-only example still runs on Aorus RTX 5090, but the checked run (`./example_4sphere 3000 400 0`) scored 193/400 = `0.4825` overall accuracy against the example's stated `50.0%` chance reference. This is useful source/runtime evidence, not decision-quality evidence.

The longer sweep remains baseline-level. Every mode × horizon accuracy is `0.333333`, exactly equal to the constant-action baselines on the balanced toy set. Shorter frozen infer samples show a little readout diversity, but by `4096`/`32768` steps infer also collapses to `query=9`; both `train` and `train_rewarded` are flat `query=9` at every tested horizon.

The interface probe supports the absolute-neuron-index interpretation used by the C++ path: same-sphere readouts match their full-sphere snapshot slices for both relative input ports `[0..3]` and absolute output ports `[8..11]` in a 4/4/4 sphere. The supervised motor-target follow-up used that route with target ports `[8, 9, 10]`, but the learned/evaluated motor readout still stayed `[0, 0, 0]` for all 6 train/holdout cases.

The earlier snapshot/pattern, multi-sphere, long-sweep and new supervised-target findings now point in the same direction: snapshot APIs expose hidden-state changes and pattern APIs are callable, but recall stayed flat zero; the three-sphere adapter ran but decoded all train/holdout cases to `query` and showed baseline-level holdout accuracy; the explicit supervised motor-target adapter also decoded all cases to `query`; and both action adapters matched trivial constant-action baselines at 0.333333 accuracy.

The VRAM-resident run completed its bounded four-hour window without duplicating the process or hitting an observed GPU resource problem. It produced 16 samples / 4,194,304 infer steps from one resident cuNxon process on the Aorus RTX 5090. The resident readout stayed `[0, 0, 0]` for every sample; active-state occupancy briefly moved from 3/15 to 5/15 and then returned to 3/15 from sample 6 onward; the last eight energy deltas plateaued near `6.7226615e+06`; throughput stayed around 291 steps/sec between samples. This is useful runtime/dynamics evidence, but it still does not measure task quality.

## Comparison lanes

| Lane | Metric | Value | Evidence |
| --- | --- | ---: | --- |
| cuNxon raw CUDA smoke | decision-quality score | no decision-quality score measured | `benchmarks/results/cunxon_smoke.json` |
| cuNxon VRAM-resident dynamics run | wall-clock resident runtime samples | 4h run completed with 16 samples/4194304 steps; resident readout stayed `[0, 0, 0]`; active occupancy returned to 3/15; no decision-quality score measured | `benchmarks/results/cunxon_vram_resident_run.json` |
| cuNxon source semantics audit | desired-output API surface | no desired-output API surface; output-port teacher forcing through sensory inputs unsupported | `benchmarks/results/cunxon_source_semantics_audit.json` |
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

cuNxon remains interesting as a GPU runtime and diagnostics surface, and the interface semantics probe removed one ambiguity from adapter design: readout ports should be treated as absolute neuron indices when targeting output neurons. The new source audit removes a second ambiguity: output-neuron readout ports are not a public desired-output/teacher-forcing channel. `StepTrain` turns on plasticity, CTSN/DSN updates and neuromodulator-sensitive learning, but source inspection found no API route that supplies desired motor labels directly to output neurons.

The supervised motor-target adapter's flat `[0, 0, 0]`/`query` result is therefore best interpreted as an unsupported adapter route, not as proof that all possible cuNxon learning semantics are exhausted. The completed VRAM-resident run also makes "just keep the same small process alive longer" a weak standalone hypothesis for this exact 3/9/3 infer-only setup: it produced stable resource/runtime evidence and some early occupancy movement, but no readout diversity. The next useful direction is either a lower-level input-port proxy probe, a task-coupled resident run with explicit stimuli/readout scoring, or an upstream-level desired-output/error-channel patch/probe, not another longer run of the same output-port teacher-forcing adapter.

## Evidence boundary

This comparison deliberately separates runtime viability and interface semantics from decision quality. Snapshot activity, callable pattern APIs, inter-sphere topology construction, longer horizons, reward injection, absolute-neuron-index port mapping, explicit but unsupported output-port target attempts, source-level absence of a desired-output API surface, and the completed VRAM-resident process/state artifacts prove useful diagnostic boundaries. The flat recall, below-chance upstream reward-only demo accuracy in this run, baseline-level Hybrid action accuracy, and resident-process dynamics by themselves do not prove intelligence, generalization, or useful learning.
