# cuNxon comparison against existing Neuraxon-Hybrid evidence

Verdict: `richer cuNxon diagnostics still flat/baseline-level`

## What changed

The previous cuNxon slice established CUDA runtime viability, long-horizon dynamics, a tiny task-coupled action probe, and an infer-vs-train sensitivity diagnostic. This follow-up executed the two next research steps: (1) inspect hidden-state/snapshot and pattern-memory APIs, and (2) build a richer sensory→association→motor multi-sphere/action adapter with holdout cases and trivial baselines.

The snapshot/pattern probe is runtime-viable on the RTX 5090: `cunxonSphereSnapshot` exposes full-neuron channels, pattern store/list/recall/clear calls work, and hidden-state activity changes from [0, 8, 2]. But recall remains flat: both stored patterns recalled `[0, 0, 0, 0, 0, 0, 0, 0]`, with Hamming distance `0`.

The multi-sphere/action adapter also runs, but it does not improve decision quality. All six train/holdout cases decode to `query` from motor readout `[0, 0, 0]`. Accuracy is `0.333333` overall and `0.333333` on holdout, equal to the constant-action trivial baselines in this balanced toy set.

## Comparison lanes

| Lane | Metric | Value | Evidence |
| --- | --- | ---: | --- |
| cuNxon raw CUDA smoke | decision-quality score | no decision-quality score measured | `benchmarks/results/cunxon_smoke.json` |
| cuNxon long-horizon raw dynamics | continuous runtime samples | 65536 steps; 1 unique readout(s); 0 readout changes | `benchmarks/results/cunxon_long_horizon.json` |
| cuNxon task-coupled action probe | success_rate | 0.333333 | `benchmarks/results/cunxon_action_probe.json` |
| cuNxon infer-vs-train sensitivity probe | readout/action diversity | infer unique=3; train unique=1; train-mode flat/query=9 | `benchmarks/results/cunxon_sensitivity_probe.json` |
| cuNxon snapshot/pattern probe | hidden-state and recall signal | snapshots active=0,8,2; patterns=2→0; recall hamming=0 | `benchmarks/results/cunxon_snapshot_pattern_probe.json` |
| cuNxon multi-sphere/action adapter | holdout success_rate vs trivial baselines | holdout=0.333333; overall=0.333333; flat query motor readouts | `benchmarks/results/cunxon_multisphere_action_probe.json` |
| raw_network | success_rate | 0.145833 | `holdout_noisy_generalization anti-oracle temporal mode` |
| random | success_rate | 0.104167 | `holdout_noisy_generalization anti-oracle temporal baseline` |
| always_execute | success_rate | 0.166667 | `holdout_noisy_generalization anti-oracle temporal baseline` |
| semantic_bridge | success_rate | 1.0 | `holdout_noisy_generalization semantic bridge mode` |
| temporal_context_adapter | success_rate | 1.0 | `holdout_noisy_generalization temporal adapter mode` |

## Interpretation

cuNxon is still interesting as a GPU runtime/diagnostics surface: we can load it, step it, query hidden state, use host-side pattern memory, and wire a three-sphere topology from Python. That is not the same as useful Neuraxon-Hybrid behavior. The richer probes strengthen the negative conclusion: in these small task-coupled setups, the cuNxon readout remains flat/baseline-level rather than producing task-coupled actions or useful recall.

The next useful direction is not another smoke test. Either inspect/readjust cuNxon interface semantics and output-port mapping at the C/CUDA level, or design a stronger adapter that explicitly drives motor outputs and compares against holdout temporal baselines before claiming integration value.

## Evidence boundary

This comparison deliberately separates runtime viability from decision quality. Snapshot activity, callable pattern APIs, and inter-sphere topology construction prove useful diagnostic surfaces, but flat recall and baseline-level holdout accuracy do not prove intelligence, generalization, or useful learning.
