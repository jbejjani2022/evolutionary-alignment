# sampling_analysis_v1 — Progress

## Overview

Does ES even need training? This track tests the hypothesis that ES's advantage comes from weight-space exploration (sampling), not iterative optimization. We compare:

1. **Temperature sampling**: Sample 1024 rollouts per prompt from the unmodified base model at different temperatures
2. **Weight perturbation sampling**: Apply 1024 random weight perturbations (ES-style, no updates), generate greedy per perturbation

Both compute pass@k curves on Countdown (Qwen-2.5-0.5B-Instruct, 2000 test problems).

## Motivation

On Qwen-2.5-0.5B-Instruct Countdown:
- Base greedy accuracy: ~0.1%
- GRPO (TRL, 30 generations): ~0%
- ES (accelerated, 500 iters): ~15.7%

ES clearly does something the base model and GRPO don't. But is it the iterative weight updates that matter, or just the act of sampling in weight space?

## Status
- **Created**: 2026-02-20
- **State**: Scripts written, ready to run

## Files

```
src/sampling_analysis_v1/
├── __init__.py
├── utils.py                    # Shared: data loading, prompts, pass@k
├── worker_extension.py         # vLLM extension: base weights in CPU, reset+perturb
└── scripts/
    ├── temperature_sampling.py # Experiment 1: token-space diversity
    └── perturbation_sampling.py # Experiment 2: weight-space diversity

configs/sampling_analysis_v1/
├── temperature_sampling.yaml
└── perturbation_sampling.yaml

scripts/sampling_analysis_v1/
├── temperature_sampling.sh
└── perturbation_sampling.sh
```

## Hyperparameters

### Shared
- Model: Qwen/Qwen2.5-0.5B-Instruct (float16)
- Test set: countdown.json indices 200-2199 (2000 problems)
- Chat template: ON
- Max new tokens: 1024
- Total rollouts: 1024 per prompt
- Seed: 42

### Temperature sampling
- Temperatures: [0.4, 0.6, 0.8, 1.0]
- top_p: 1.0, top_k: disabled
- Greedy baseline included (T=0, 1 sample)

### Perturbation sampling
- Sigmas: [0.001, 0.003, 0.005, 0.01]
- Decoding: greedy (T=0) — isolates weight noise from token noise
- Greedy baseline included (no perturbation)

## Running

```bash
# Temperature sampling
bash scripts/sampling_analysis_v1/temperature_sampling.sh
bash scripts/sampling_analysis_v1/temperature_sampling.sh --debug  # 20 samples

# Perturbation sampling
bash scripts/sampling_analysis_v1/perturbation_sampling.sh
bash scripts/sampling_analysis_v1/perturbation_sampling.sh --debug
```
