# Research Context: Evolutionary Alignment

## Overview

This project investigates **Evolution Strategies (ES)** as an alternative to **Reinforcement Learning (GRPO)** for LLM post-training / fine-tuning. ES optimizes via weight perturbation and population-based fitness evaluation, requiring no gradient access through the model. We compare ES and GRPO across multiple tasks and analysis dimensions to understand when and why they produce different behaviors.

## Core Thesis

**ES is not a "poor man's GRPO" — it explores qualitatively different weight space and exhibits different failure modes.**

While properly-tuned GRPO can match or exceed ES on raw reward metrics, ES exhibits distinct behavior: less reward hacking on safety tasks, more distributed weight updates across layers, and convergence to smoother basins in parameter space. The choice of optimizer is not just about efficiency — it has implications for alignment robustness.

## Key Findings

### 1. Hyperparameter Tuning Matters — GRPO Baselines Were Undertuned

Qiu et al. (2025) claimed ES > RL, but this was partly due to suboptimal RL baselines (they used TinyZero; we used TRL and got substantially stronger GRPO results). After proper tuning:
- GRPO matches or exceeds ES on reward metrics for both Countdown and Conciseness
- GRPO is faster (~5s/iter vs ~7s/iter on Conciseness, ~9s/iter vs ~30s/iter on Countdown)
- The original paper's RL scripts were never released, making reproducibility difficult

### 2. ES Hyperparameter Sensitivity — A Narrow "Safe" Band

Our α×σ sweep on Conciseness (10 configurations, 4 seeds each) reveals:
- Only 3 of 10 (α, σ) configurations produced no hacked or degenerate solutions
- Two of those three were the exact configurations presented by Qiu et al.
- **σ effect**: Increasing σ raises reward but inflates KL and increases hacking; high σ reliably produces reward hacking across all seeds
- **α effect**: Too-large α destabilizes training, producing degenerate gibberish outputs
- **Joint**: A "safe band" exists at small α + small-to-moderate σ; outside it, ES collapses
- Best non-hacking config: α=0.0005, σ=0.003 (reward 0.969±0.010, KL 0.444±0.192)

### 3. On Conciseness: ES ≈ GRPO When Both Are Properly Tuned

When both methods are well-tuned on the Conciseness task:
- Both find similar strategies (Chinese characters for shorter encodings, correct short answers)
- Both can hack (empty strings, special tokens, gibberish) in some seeds/configs
- Seed variance is high for both methods
- ES never hacked at the two hyperparameter settings from the original paper, while GRPO hacked in most seeds — but extending the ES sweep shows ES can hack too
- GRPO achieves slightly better final reward (0.964 vs 0.969 normalized)

### 4. Weight Space Analysis — ES and GRPO Explore Different Regions

Three complementary analyses on Conciseness checkpoints reveal systematic differences:

**Direct weight perturbation:**
- GRPO checkpoints are *more* locally robust to small perturbations (contradicting the hypothesis that ES finds perturbation-robust weights)
- GRPO is especially robust in the unembedding layer
- At larger perturbation scales, ES checkpoints can actually *improve* (perturbation knocks them into better regions), while GRPO degrades rapidly
- Exception: reward-hacked ES runs (high σ) are maximally robust (trivially — they're already at degenerate solutions)

**Linear mode connectivity:**
- Base→ES: smooth, near-linear reward increase along interpolation path (broad, smooth basin)
- Base→GRPO: flat plateau then sharp rise near GRPO endpoint (narrow, sharp basin)
- ES↔ES: relatively flat connectivity (same smooth basin, cross-seed)
- GRPO↔GRPO: flat at maximum reward (shared high-reward basin, even across different β)
- ES↔GRPO: pronounced dip at intermediate α — clear loss barrier between ES and GRPO solutions

**Weight-change cosine similarity:**
- ES and GRPO change weights in orthogonal directions at every layer (cosine sim ≈ 0)
- ES runs are more consistent with each other than GRPO runs at all layers except the unembedding layer
- GRPO makes highly correlated changes to the unembedding layer across runs (~4x more similar than other layers), suggesting overfitting at that layer

**Two solution regimes identified:**
- Non-hacking regime: ES-dominated, close to base model, smooth connected basins
- Hacked regime: reachable by both ES and GRPO, forms a shared high-reward basin

### 5. Safety Alignment (PKU-SafeRLHF) — ES Shines

On helpful-harmless alignment with Alpaca-7B:
- **Data**: Only 250 training prompts (<1% of 30k PKU-SafeRLHF dataset)
- **ES** converges to **context-aware "helpful refusals"**: explains why a request is denied, cites specific laws, offers lawful alternatives
- **GRPO β=0.01**: mode collapse to a single refusal template repeated across 3030/3036 examples — scores well on unified cost model but provides no genuine help
- **GRPO β=0.1**: reward hacking via hallucinated multi-turn transcripts with "Relationship Advice" hack — briefly refuses then hallucinates unrelated content
- **Unified scorer** (same model used for training): GRPO β=0.01 looks better (higher reward, lower cost)
- **GPT-4.1-mini Elo tournament** (held-out judge): ES achieves highest Elo on both helpfulness and harmlessness; GRPO performs worst
- This tension (train-time judge vs held-out judge) is a key finding: GRPO overfits the training signal
- **Sample efficiency**: ES with only 50 prompts (~0.2% of dataset) still outperforms all Beaver Safe RLHF baselines (V1, V2, V3)

### 6. Countdown Task

- ES and GRPO both work on mathematical reasoning (Countdown)
- Our ES accuracy lower than paper (28.9% vs 37.3%), our GRPO accuracy higher (40.7% vs 14.8%)
- Discrepancy likely due to RL framework differences (TRL vs TinyZero)
- Model: Qwen-2.5-1.5B-Instruct, 200 training samples, 500 iterations

## Open Research Questions

1. **Sampling vs. Optimization**: Does ES's behavioral difference come from its sampling distribution or its optimization dynamics?
   - Potential experiment: Get rollouts via ES perturbations, then optimize via backprop
   - Would disentangle exploration (sampling) from optimization (weight update rule)
   - **Partial answer (sampling_analysis_v1)**: On Countdown (Qwen-2.5-0.5B-Instruct), temperature sampling (token-space diversity) actually beats weight perturbation (weight-space diversity) at every operating point (pass@512: 44% vs 35%). Both fall far short of ES post-training (greedy ~15.7%), confirming ES's iterative optimization matters — it's not just weight-space exploration.

2. **Solution Coverage**: Does ES increase the diversity of solutions found?
   - Recent work shows RL doesn't increase coverage over base LLM (can match with higher k in pass@k)
   - Hypothesis: ES might actually change/increase coverage due to parameter-space exploration
   - Relevant papers:
     - https://arxiv.org/abs/2507.14843 (reproduce Figure 1 with ES)
     - https://arxiv.org/abs/2510.15020
     - https://arxiv.org/abs/2505.24864

3. **KL Regularization**: Schulman's KL approximation concerns
   - http://joschu.net/blog/kl-approx.html
   - Need to verify behavior with transformers 4.30.0

4. **Neurosymbolic advantage**: ES may enable joint optimization of LLM weights + discrete system configs that aren't model outputs (see `docs/neurosymbolic_es.md` for discussion). The strong claim: ES enables optimizing over combinatorial config spaces jointly with weights. The weak claim (overstated): ES enables LLM + tool use — GRPO handles this fine.

## Tasks & Benchmarks

| Task | Model | Description | Key Metric | Status |
|------|-------|-------------|------------|--------|
| Conciseness | Qwen-2.5-7B-Instruct | Generate answers matching target length | R = -\|len(gen) - len(target)\| | Complete (α×σ sweep, GRPO sweep, weight analysis) |
| PKU-SafeRLHF | Alpaca-7B | Helpful-harmless alignment | Unified scorer + GPT Elo | Complete (ES + GRPO baselines + Beaver comparison) |
| Countdown | Qwen-2.5-1.5B-Instruct | Mathematical reasoning | Answer accuracy | Complete (basic comparison) |

## ES Algorithm Summary

Each iteration:
1. Sample `population_size` (30) random seeds
2. For each seed: perturb all weights with Gaussian noise (scale σ), generate responses, compute fitness
3. Normalize fitness across population
4. Update weights: θ ← θ + (α / pop_size) × Σ(normalized_fitness × noise)

Key hyperparameters: α (learning rate), σ (noise scale), population_size (30), precision (bfloat16)

Two implementations exist:
- **Baseline**: Accelerate-based, multi-threaded evaluation, single-node
- **Accelerated**: vLLM + Ray + NCCL, multi-GPU with weight broadcasting via WorkerExtension

## Codebase Structure

### Experiment Code (legacy structure, predates track system)
```
conciseness/           # Conciseness task
├── ES/                # ES training scripts
├── GRPO/              # GRPO training (TRL-based)
├── data/              # train.jsonl (2 samples), eval.jsonl (8 samples)
├── conciseness_eval.py          # Optimized batched evaluation
└── conciseness_eval_seeds_sweep.py

countdown/             # Countdown task
├── ES/                # ES training (baseline + accelerated vLLM)
├── GRPO/              # GRPO training (TRL-based)
├── data/              # countdown.json (~4000 examples)
├── countdown_task.py  # Reward functions
└── eval_countdown_vllm.py       # vLLM evaluation

utils/
└── worker_extn.py     # vLLM WorkerExtension for ES perturbations
```

### Track System (new)
```
src/sampling_analysis_v1/    # New track for sampling analysis
configs/sampling_analysis_v1/
scripts/sampling_analysis_v1/
docs/tracks/sampling_analysis_v1/progress.md
```

### Paper
```
paper/                 # Overleaf-synced (gitignored, separate git repo)
├── main.tex
├── sections/          # abstract, intro, main_experiments, es_dynamics, alignment, robustness, conclusion
├── appendix/          # 14+ appendix files
├── figures/           # ~40 figures
├── alignment-figures/ # ~30 alignment figures
└── references.bib
```

### Resources
```
resources/
├── eggroll/           # EGGROLL paper (PDF + markdown conversion)
└── es-fine-tuning-paper/  # Archived previous working repo (alignment experiments, older code)
```

## Hardware & Infrastructure

- **Typical runs**: 4× H100 GPUs, 256GB RAM (SLURM cluster)
- **vLLM**: Ray-distributed inference with NCCL weight synchronization
- **Scoring API**: FastAPI endpoint serving PKU Beaver-7B reward/cost models
- **Logging**: Weights & Biases + TensorBoard
- **Models**: Qwen-2.5 family (0.5B, 1.5B, 3B, 7B Instruct), Alpaca-7B

## Timeline & Deadlines

**Target: ICML 2026**
- Abstract submission: January 23, 2026 AOE (passed)
- Full paper submission: January 28, 2026 AOE (passed)
- Paper submitted via Overleaf

## Related Work & Literature

### Primary Reference
- Qiu et al. (2025): "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning" (arXiv:2509.24372)
  - We agree: ES finds qualitatively different solutions than GRPO
  - We disagree: ES is not strictly "better" — GRPO matches on reward when properly tuned
  - We extend: weight space analysis, hyperparameter sweep, safety alignment with GRPO baselines

### Other Key Papers
- **EGGROLL** (arXiv:2511.16652): Shows ES can work at scale; claims about neurosymbolic systems are overstated (see `docs/neurosymbolic_es.md`)
- **Safe RLHF** (Dai et al. 2023): Beaver models as baselines for alignment
- **PKU-SafeRLHF** (Ji et al. 2024): Dataset and evaluation framework
- Solution coverage papers: arXiv:2507.14843, 2510.15020, 2505.24864

### Feedback from Final Report
- Main actionable feedback: **Improve literature review**
- OpenReview: https://openreview.net/forum?id=1zqmmcjvdN

## Team

### Joey
- Overleaf setup, repo cleanup and documentation
- Weight magnitude analysis
- Fair conciseness GRPO baseline (sweep β, ensure KL done right)
- Countdown experiments

### Itamar
- Fair GRPO baseline on alignment task
- Reruns with reproduced Alpaca 7b
- Storage savings documentation
- Solution coverage analysis (reproduce arXiv:2507.14843 Figure 1)
- Math / Countdown experiments

### Core
- Literature review
- Framing and figure design
- Consult Sham about weight space comparisons
- Evaluate "Sample with ES, optimize with GD" experiment

## Resources

| Resource | Link |
|----------|------|
| Final report (PDF) | https://drive.google.com/file/d/16McClLBCSU7pgTV-NyVssfjJYrDlq8hF/view |
| Report feedback | https://docs.google.com/document/d/1jZwWTQnekaiTZGfUPxrziBi7SCB0xx4mqq3opbr-HH4/edit |
| OpenReview | https://openreview.net/forum?id=1zqmmcjvdN |
| Fresh experiments repo | https://github.com/jbejjani2022/evolutionary-alignment |
| Working repo (final project) | https://github.com/jbejjani2022/es-fine-tuning-paper |
| Old repo (local) | /home/ubuntu/work/evolutionary-alignment/resources/es-fine-tuning-paper |

## Key Message for Paper

> Evolution Strategies is not just a scalable alternative to RL — it explores a qualitatively different solution space. While properly-tuned GRPO can match ES on reward metrics, ES exhibits less reward hacking on safety tasks, better generalization to held-out judges, and distributes weight updates more evenly across layers. On PKU-SafeRLHF, ES with only 250 examples converges to helpful refusals that outperform both Safe RLHF benchmarks and GRPO baselines under external evaluation — even though GRPO looks better under the training-time judge. This makes ES a promising approach for safety-critical alignment where robustness to reward hacking matters more than raw optimization performance.
