# Qubic NIA Vol. 8 criticality claim map

Source: Qubic Scientific Team, "Brain Criticality and the Branching Ratio in Neural and Artificial Networks: A Bioinspired Principle in Neuraxon", published 2026-05-13.

## Hypothesis for this slice

NIA Vol. 8 makes branching ratio / criticality a concrete measurement target for Neuraxon-Hybrid: treat sigma ≈ 1 as a regime hypothesis to test against cuNxon artifacts, not as evidence of intelligence by itself.

## Extracted technical claims

| Claim | Article position | Measurable repo hypothesis |
| --- | --- | --- |
| Branching ratio regimes | sigma < 1 means decay, sigma ≈ 1 means reverberating criticality, sigma > 1 means runaway activity. | cuNxon runs should report active-state propagation/regime labels rather than only readout/action scores. |
| Slightly subcritical reverberation | Cortex is framed as often slightly below sigma=1 and task-modulable, not exactly balanced forever. | Neuraxon-Hybrid should test a bounded reverberating band rather than equality to sigma=1. |
| Artificial edge of chaos | Reservoir/recurrent systems are claimed to peak near order-chaos transition or spectral radius near 1. | Regime metrics must be correlated with task scores and baselines; near-critical-looking dynamics alone are not enough. |
| Neuraxon real-time invariant | Branching ratio is presented as an operational scalar for whether a system is alive, silent, or explosive. | cuNxon diagnostics should include a branching/activity-ratio lane beside energy, occupancy, readout diversity, GPU resources, and task accuracy. |
| Self-organized criticality | Neuraxon is claimed to self-organize toward criticality without centralized fine-tuning and to gain robustness. | This needs trajectory evidence across seeds/interventions plus held-out task improvement, not only a final scalar. |
| Functional generalization | The article says internal evaluations/submitted publications show better generalization, perturbation stability, representational richness, and temporal coherence. | In this repo that requires holdout/stress/control splits, trivial baselines, leakage checks, and a strict split between runtime dynamics and decision quality. |

## Evidence map against current artifacts

| Artifact | Observation | Evidence status |
| --- | --- | --- |
| `benchmarks/results/analysis/criticality_summary.csv` | CPU/adapter analysis already has `branching_ratio_mean=1.194208`, neutral occupancy `0.757844`, transition entropy `0.332013`, and modulation action-change rate `0.000000`. | Partially relevant historical evidence: the metric exists, but did not imply useful raw-network behavior. |
| `benchmarks/results/cunxon_vram_resident_run.json` | Four-hour resident cuNxon run completed 16 samples / 4,194,304 infer steps. Readout stayed `[0, 0, 0]`. Coarse active-state sequence was `3,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3`; VRAM-resident active-state ratio mean≈1.017778. | Runtime/dynamics evidence only. The near-1 coarse ratio is not a validated branching estimator and did not produce readout diversity or task evidence. |
| `benchmarks/results/cunxon_resident_action_probe.json` | Twelve resident train/eval epochs and 72 scored cases all decoded to `query`; train/holdout/overall accuracy stayed `0.333333`, equal to constant-action baselines. | Negative task-coupled residency evidence. Keeping the same process alive did not by itself produce useful action dynamics. |
| `benchmarks/results/cunxon_aigarth_action_target_contract_augmented_train_probe.json` | Target-contract augmented-train Aigarth audit reached holdout mean `1.000000` and hard-holdout mean `0.866667`, zero unexpected labels, but stress-holdout mean=0.333333 and counterfactual-control mean `0.066667`. | Constructive but brittle toy/evolution evidence. It does not establish self-organized criticality or broad generalization. |
| `benchmarks/results/cunxon_branching_regime_scan.json` | Five fresh Aorus RTX 5090 seeds (`117..121`) all bucketed as `reverberating/near-critical proxy`; mean branching proxy=0.997701, holdout mean `1.000000`, stress-holdout mean=0.333333, and 0/5 seeds beat the best constant stress baseline. | New negative/diagnostic evidence: the NIA Vol. 8 regime lane is now operationalized, but near-critical-looking proxy activity did not predict stress-holdout quality above baseline. |

## Interpretation

The NIA Vol. 8 article is useful because it turns a broad brain-inspired claim into a measurable lane: branching/activity regime should be tracked directly. However, current Neuraxon-Hybrid evidence does not let sigma ≈ 1 stand in for intelligence. A coarse near-1 active-state ratio in the completed VRAM-resident cuNxon run coexisted with flat `[0, 0, 0]` readouts and no task score. The task-coupled resident action probe stayed exactly at constant-action baselines. The first bounded branching-regime scan now adds task-coupled evidence: five fresh Aigarth source seeds looked near-critical by the coarse readout proxy and kept standard holdout perfect, but the harder low-margin stress split remained exactly at the best constant baseline. The best current cuNxon action evidence still comes from explicit Aigarth objective shaping, not from demonstrated self-organized criticality.

Current boundary: branching ratio is a necessary diagnostic, not a sufficient success criterion. Any future report should say "near-critical dynamics observed" only when the estimator is explicit and should say "useful computation" only when held-out/stress task quality beats trivial baselines. The first scan is evidence against treating near-1 proxy activity as sufficient: stress-holdout stayed baseline-level.

## Proposed next probe

`cunxon-branching-ratio-regime-scan` (tracked as https://github.com/sisutuulenisa/neuraxon-hybrid/issues/84) now has an initial bounded implementation in `benchmarks/results/cunxon_branching_regime_scan.*`. It logs:

- active-state branching/activity ratio with subsampling caveats;
- neutral occupancy, transition entropy, energy slope, readout diversity, and GPU/resource metrics;
- action accuracy on train/holdout/stress/control splits;
- constant-action baselines for every split;
- regime labels such as dead/subcritical, reverberating/near-critical, and runaway/saturated.

Acceptance criteria:

1. Report a bounded branching/activity-ratio estimate per run and classify the observed regime.
2. Include task-coupled action quality and constant-action baselines for every regime bucket.
3. State plainly whether near-critical/reverberating metrics correlate with holdout/stress performance.
4. Keep `holdout`, `stress_holdout`, `counterfactual_control`, and `permuted_control` labels outside any optimization callback.
5. Update `cunxon_comparison.*` and avoid intelligence claims unless held-out task quality beats baselines.

## Conservative verdict

NIA Vol. 8 is now mapped into a concrete cuNxon research lane, and the first bounded branching-regime scan has been run. Existing cuNxon artifacts still support an evidence-first conclusion: runtime diagnostics are viable, Aigarth objective shaping is a promising toy route, stress-holdout remains baseline-level, and branching/criticality metrics are not intelligence evidence until they predict or improve held-out/stress task performance.
