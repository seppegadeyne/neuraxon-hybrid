# cuNxon avalanche-window intervention matrix

Status: `completed`

## Hypothesis

Qubic NIA Vol. 8 suggests branching ratio / criticality should be an operational Neuraxon invariant. This slice tests a narrower measurement question: if the snapshot avalanche estimator is useful, bounded window-length and sample-interval changes should be reported explicitly beside action quality and constant baselines rather than interpreted as intelligence evidence.

## Configurations and results

| Config | Steps | Interval | Samples | Mean branching | Branching range | Active ratio | Neutral occupancy | Infer acc | Train acc | Best baseline | All modes > baseline |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| short-dense | 128 | 8 | 12 | 0.121067 | 0.000000..0.342796 | 0.957002 | 0.868284 | 0.500000 | 0.333333 | 0.333333 | False |
| baseline-equivalent | 256 | 16 | 12 | 0.157171 | 0.000000..0.426655 | 0.985500 | 0.901042 | 0.333333 | 0.333333 | 0.333333 | False |
| long-sparse | 512 | 32 | 12 | 0.177203 | 0.000000..0.483938 | 1.017984 | 0.913377 | 0.333333 | 0.333333 | 0.333333 | False |

## Aggregate result

- Configurations: 3
- Samples: 36
- Mean branching-ratio estimate: 0.151814
- Branching-ratio estimate range: 0.000000..0.483938
- Mean active-count ratio: 0.986828
- Mean neutral occupancy: 0.894234
- Max mode accuracy delta vs best constant baseline: 0.166667
- Configurations with all modes beating baseline: []
- Aggregate normalized action distribution: {'assertive': 3, 'execute': 4, 'query': 29}

## Verdict

Window-length/sample-interval changes moved the snapshot branching estimator and produced one short-dense infer-mode accuracy above the constant baseline, but no configuration had both infer and train modes beat the best constant baseline; this is estimator-sensitivity and negative robustness evidence, not intelligence evidence.

## Evidence boundary

This matrix tests whether the NIA Vol. 8 criticality/branching metric is stable under bounded window-length/sample-interval changes. It is not intelligence evidence: useful computation would require robust held-out/control task quality above baselines, not only movement in a snapshot estimator or one mode-specific bump.

## Recommended next probe

- ID: `cunxon-avalanche-intervention-task-correlation`
- Purpose: Move from estimator sensitivity to causal/intervention evidence by varying runtime parameters or stimuli while requiring regime movement and held-out/stress action quality above baselines.
- GitHub issue: https://github.com/sisutuulenisa/neuraxon-hybrid/issues/85
- Acceptance criteria:
  - Use fresh seeds and at least one held-out or stress action split, not only train/infer action stimuli.
  - Report branching/avalanche movement beside action quality, constant baselines, and leakage/control checks.
  - Treat mode-specific or seed-specific bumps as hypotheses until they repeat across modes/seeds.

## Source artifacts

- `benchmarks/results/cunxon_avalanche_window_short_dense.json` / `benchmarks/results/cunxon_avalanche_window_short_dense.md`
- `benchmarks/results/cunxon_avalanche_window_baseline_equivalent.json` / `benchmarks/results/cunxon_avalanche_window_baseline_equivalent.md`
- `benchmarks/results/cunxon_avalanche_window_long_sparse.json` / `benchmarks/results/cunxon_avalanche_window_long_sparse.md`
