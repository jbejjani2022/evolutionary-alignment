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
- **State**: Temperature sampling (high-T) complete; low-T and perturbation sampling pending
- **2026-02-21**: Simplified temperature sampling (removed unnecessary round-based batching), scaled down to 128 prompts × 512 samples for feasibility (~1.8h vs ~58h)
- **2026-02-21**: Ran temperature sampling (T=0.4, 0.6, 0.8, 1.0). Results: lower T is strictly better (pass@512: 44% at T=0.4, 20% at T=1.0). Greedy = 0%. Created downstream plots. Added low-temperature config (T=0.05, 0.1, 0.2) — not yet run.

## Results So Far

### Temperature sampling (T=0.4, 0.6, 0.8, 1.0)
| Temp | pass@1 | pass@64 | pass@512 | Prompts solved |
|------|--------|---------|----------|----------------|
| 0.4  | 0.23%  | 11.7%   | 43.8%    | 56/128         |
| 0.6  | 0.17%  | 9.7%    | 36.7%    | 47/128         |
| 0.8  | 0.11%  | 7.9%    | 32.0%    | 41/128         |
| 1.0  | 0.06%  | 3.7%    | 20.3%    | 26/128         |

- Lower temperature strictly dominates — concentrated sampling near the mode helps
- Distribution is heavily zero-inflated: most prompts are never solved
- Even best pass@512 (44%) well below ES post-training greedy accuracy (~15.7%)

## Files

```
src/sampling_analysis_v1/
├── __init__.py
├── utils.py                    # Shared: data loading, prompts, pass@k
├── worker_extension.py         # vLLM extension: base weights in CPU, reset+perturb
└── scripts/
    ├── temperature_sampling.py  # Experiment 1: token-space diversity
    ├── perturbation_sampling.py # Experiment 2: weight-space diversity
    └── plot_temperature.py      # Downstream: plots for temperature results

configs/sampling_analysis_v1/
├── temperature_sampling.yaml
├── temperature_sampling_low.yaml  # T=0.05, 0.1, 0.2
├── perturbation_sampling.yaml
└── plot_temperature.yaml          # Downstream config

scripts/sampling_analysis_v1/
├── temperature_sampling.sh
├── temperature_sampling_low.sh
├── perturbation_sampling.sh
└── plot_temperature.sh
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
- Temperatures (low): [0.05, 0.1, 0.2] — pending
- top_p: 1.0, top_k: disabled
- Greedy baseline included (T=0, 1 sample)
- Single vLLM generate() call per temperature with n=512

### Perturbation sampling
- Sigmas: [0.001, 0.003, 0.005, 0.01]
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

# Downstream plots (after temperature runs complete)
bash scripts/sampling_analysis_v1/plot_temperature.sh

# Debug mode (any script)
bash scripts/sampling_analysis_v1/temperature_sampling.sh --debug
```
