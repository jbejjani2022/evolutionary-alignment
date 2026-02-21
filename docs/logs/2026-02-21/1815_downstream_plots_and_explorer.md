# Session Log: Downstream Plots & Solve Explorer

## Summary

Added downstream plotting scripts for perturbation sampling results, temperature-vs-perturbation comparison plots, and an interactive HTML solve overlap explorer.

## Tasks Completed

- **Perturbation downstream plots**: Created `plot_perturbation.py` (pass@k, correct distribution, summary bars) — mirrors the existing temperature plots. Key finding: σ=0.001 is the only useful setting (pass@512=35%), σ≥0.005 solves nothing.
- **Temperature low-temp individual plots**: Created config/sh to reuse `plot_temperature.py` for low-T results (T=0.05, 0.1, 0.2). Confirms T=0.4 is optimal — lower temperatures hurt (pass@512: 26% at T=0.2, 7% at T=0.05).
- **Temperature vs perturbation comparison plots**: Created `plot_comparison.py` with 3 plots:
  - pass@k overlay (all 7 temperatures + 4 sigmas, orange solid vs blue dashed)
  - Best-vs-best bar chart (T=0.4 at 44% vs σ=0.001 at 35%)
  - Coverage vs depth scatter (breadth/depth tradeoff across configs)
  - Solve overlap bar + per-prompt detail strip (40 both, 16 temp-only, 5 perturb-only, 67 neither)
- **Solve overlap explorer HTML**: Interactive page with proportional-height category buttons (left) and scrollable prompt list (right) showing numbers, target, solution, and comparative solve bars.
- **Low-sigma perturbation config**: Created `perturbation_sampling_low.yaml` (σ=0.0005) to test whether going below σ=0.001 helps — run in progress.
- **Comparison script updated**: Now accepts `temperature_upstream_dirs` (list) to merge multiple temperature runs into a unified sweep.
- **Fixed coverage plot**: Original had redundant axes (coverage ≈ pass@512); replaced with coverage (breadth) vs depth (avg correct rate among solved prompts).

## Files Created

- `src/sampling_analysis_v1/scripts/plot_perturbation.py`
- `src/sampling_analysis_v1/scripts/plot_comparison.py`
- `src/sampling_analysis_v1/scripts/build_solve_explorer.py`
- `configs/sampling_analysis_v1/plot_perturbation.yaml`
- `configs/sampling_analysis_v1/plot_temperature_low.yaml`
- `configs/sampling_analysis_v1/plot_comparison.yaml`
- `configs/sampling_analysis_v1/solve_explorer.yaml`
- `configs/sampling_analysis_v1/perturbation_sampling_low.yaml`
- `scripts/sampling_analysis_v1/plot_perturbation.sh`
- `scripts/sampling_analysis_v1/plot_temperature_low.sh`
- `scripts/sampling_analysis_v1/plot_comparison.sh`
- `scripts/sampling_analysis_v1/solve_explorer.sh`
- `scripts/sampling_analysis_v1/perturbation_sampling_low.sh`

## Key Findings

- Temperature sampling dominates perturbation sampling at every operating point
- T=0.4 is optimal (inverted-U shape across full sweep T=0.05–1.0)
- σ=0.001 is competitive with T=0.6 but all other sigmas are useless
- 5 prompts solved only by perturbation (not temperature) — weight-space exploration does find unique solutions, but few
- Both methods leave 67/128 prompts unsolved at 512 samples

## Next Steps

- Check σ=0.0005 perturbation run results (currently running)
- If σ=0.0005 improves, may need even finer sigma sweep
- Consider union-of-methods analysis: does combining temp + perturb improve coverage meaningfully?
