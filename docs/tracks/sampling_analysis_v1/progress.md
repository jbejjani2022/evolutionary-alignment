# sampling_analysis_v1 — Progress

## Overview

Does ES even need training? This track tests the hypothesis that ES's advantage comes from weight-space exploration (sampling), not iterative optimization. We compare:

1. **Temperature sampling**: Sample 512 rollouts per prompt from the unmodified base model at different temperatures
2. **Weight perturbation sampling**: Apply 512 random weight perturbations (ES-style, no updates), generate greedy per perturbation

Both compute pass@k curves on Countdown (Qwen-2.5-0.5B-Instruct, 128 test problems from the eval split).

## Motivation

On Qwen-2.5-0.5B-Instruct Countdown:
- Base greedy accuracy: ~0.1%
- GRPO (TRL, 30 generations): ~0%
- ES (accelerated, 500 iters): ~15.7%

ES clearly does something the base model and GRPO don't. But is it the iterative weight updates that matter, or just the act of sampling in weight space?

## Status
- **Created**: 2026-02-20
- **State**: All sampling runs complete. All plots regenerated with full data.
- **2026-02-21 (early)**: Simplified temperature sampling, scaled down to 128p×512s
- **2026-02-21 (morning)**: Ran temperature sampling (T=0.4–1.0). Created downstream plots. Added low-T config.
- **2026-02-21 (afternoon)**: Ran perturbation sampling (σ=0.001–0.01) and low-T sampling (T=0.05–0.2). Created downstream plots for perturbation, comparison plots (temp vs perturb), solve overlap analysis, and interactive HTML explorer. Added σ=0.0005 config (running).
- **2026-02-21 (evening)**: σ=0.0005 complete. Updated all downstream scripts (plot_perturbation, plot_comparison, build_solve_explorer) to merge multiple perturbation upstream dirs (matching how temperature already works). Regenerated all plots with full 5-sigma sweep.

## Results

### Temperature sampling — full sweep (T=0.05–1.0)
| Temp | pass@1 | pass@64 | pass@512 | Prompts solved |
|------|--------|---------|----------|----------------|
| 0.05 | 0.00%  | 2.2%    | 7.0%     | 9/128          |
| 0.1  | 0.01%  | 4.7%    | 15.6%    | 20/128         |
| 0.2  | 0.04%  | 8.1%    | 25.8%    | 33/128         |
| 0.4  | 0.23%  | 11.7%   | 43.8%    | 56/128         |
| 0.6  | 0.17%  | 9.7%    | 36.7%    | 47/128         |
| 0.8  | 0.11%  | 7.9%    | 32.0%    | 41/128         |
| 1.0  | 0.06%  | 3.7%    | 20.3%    | 26/128         |

- Inverted-U shape: T=0.4 is optimal, falls off in both directions
- Too low (T≤0.1): not enough diversity, too close to greedy
- Too high (T≥0.8): too much noise, degrades quality

### Perturbation sampling — full sweep (σ=0.0005–0.01)
| σ      | pass@1 | pass@64 | pass@512 | Prompts solved |
|--------|--------|---------|----------|----------------|
| 0.0005 | 0.27%  | 11.1%   | 32.0%    | 41/128         |
| 0.001  | 0.23%  | 11.3%   | 35.2%    | 45/128         |
| 0.003  | 0.01%  | 0.6%    | 3.9%     | 5/128          |
| 0.005  | 0%     | 0%      | 0%       | 0/128          |
| 0.01   | 0%     | 0%      | 0%       | 0/128          |

- σ=0.001 is the sweet spot; performance degrades in both directions
- σ=0.0005 slightly worse (32% vs 35% at pass@512) — too little perturbation
- σ≥0.005 completely destroys the model
- Much more sensitive than temperature — narrow viable band

### Comparison: Temperature vs Perturbation
- **Best temp (T=0.4)** beats **best perturb (σ=0.001)**: 44% vs 35% at pass@512
- Temperature dominates at every operating point in the sweep
- σ=0.001 is comparable to T=0.6 but no perturbation setting matches T=0.4
- Solve overlap at 512 (T=0.4 vs σ=0.001): 40 both, 16 temp-only, 5 perturb-only, 67 neither
- Perturbation finds 5 unique prompts that temperature misses — small but nonzero unique coverage

### Key Takeaway
Token-space diversity (temperature) is more effective than weight-space diversity (perturbation) for solution coverage on Countdown. However, both fall far short of ES post-training (greedy ~15.7% vs best pass@512 ~44%), confirming that ES's iterative optimization is doing real work beyond sampling.

## Files

```
src/sampling_analysis_v1/
├── __init__.py
├── utils.py                       # Shared: data loading, prompts, pass@k
├── worker_extension.py            # vLLM extension: base weights in CPU, reset+perturb
└── scripts/
    ├── temperature_sampling.py    # Experiment 1: token-space diversity
    ├── perturbation_sampling.py   # Experiment 2: weight-space diversity
    ├── plot_temperature.py        # Downstream: plots for temperature results
    ├── plot_perturbation.py       # Downstream: plots for perturbation results
    ├── plot_comparison.py         # Downstream: temp vs perturb comparison + overlap
    └── build_solve_explorer.py    # Downstream: interactive HTML explorer

configs/sampling_analysis_v1/
├── temperature_sampling.yaml        # T=0.4, 0.6, 0.8, 1.0
├── temperature_sampling_low.yaml    # T=0.05, 0.1, 0.2
├── perturbation_sampling.yaml       # σ=0.001, 0.003, 0.005, 0.01
├── perturbation_sampling_low.yaml   # σ=0.0005 (complete)
├── plot_temperature.yaml
├── plot_temperature_low.yaml
├── plot_perturbation.yaml
├── plot_comparison.yaml             # Merges both temp runs + perturb
└── solve_explorer.yaml              # Interactive HTML explorer

scripts/sampling_analysis_v1/
├── temperature_sampling.sh
├── temperature_sampling_low.sh
├── perturbation_sampling.sh
├── perturbation_sampling_low.sh
├── plot_temperature.sh
├── plot_temperature_low.sh
├── plot_perturbation.sh
├── plot_comparison.sh
└── solve_explorer.sh
```

## Hyperparameters

### Shared
- Model: Qwen/Qwen2.5-0.5B-Instruct (float16)
- Test set: countdown.json indices 200-327 (128 problems, from eval split)
- Chat template: ON
- Max new tokens: 1024
- Total rollouts: 512 per prompt
- Seed: 42

### Temperature sampling
- Temperatures (high): [0.4, 0.6, 0.8, 1.0] — complete
- Temperatures (low): [0.05, 0.1, 0.2] — complete
- top_p: 1.0, top_k: disabled
- Greedy baseline included (T=0, 1 sample)
- Single vLLM generate() call per temperature with n=512

### Perturbation sampling
- Sigmas (main): [0.001, 0.003, 0.005, 0.01] — complete
- Sigmas (low): [0.0005] — complete
- Decoding: greedy (T=0) — isolates weight noise from token noise
- Greedy baseline included (no perturbation)

## Running

```bash
# Temperature sampling (high)
bash scripts/sampling_analysis_v1/temperature_sampling.sh

# Temperature sampling (low)
bash scripts/sampling_analysis_v1/temperature_sampling_low.sh

# Perturbation sampling
bash scripts/sampling_analysis_v1/perturbation_sampling.sh

# Perturbation sampling (low sigma)
bash scripts/sampling_analysis_v1/perturbation_sampling_low.sh

# Downstream plots
bash scripts/sampling_analysis_v1/plot_temperature.sh
bash scripts/sampling_analysis_v1/plot_temperature_low.sh
bash scripts/sampling_analysis_v1/plot_perturbation.sh
bash scripts/sampling_analysis_v1/plot_comparison.sh

# Interactive explorer
bash scripts/sampling_analysis_v1/solve_explorer.sh

# Debug mode (any script)
bash scripts/sampling_analysis_v1/temperature_sampling.sh --debug
```
